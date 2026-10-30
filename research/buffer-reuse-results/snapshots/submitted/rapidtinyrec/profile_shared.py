"""Separate uninstrumented latency, light scope timing, Python and CUDA traces.

Profiling samples are representatives of the ten frozen prompt lengths, not a
new 500-user latency estimate. No per-scope CUDA synchronizations are added.
"""
import argparse
import contextlib
import cProfile
import functools
import gc
import gzip
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import pstats
import shutil
import subprocess
import time
from unittest.mock import patch

from . import benchmark as b
from .bulk_constraints import BulkPrefixConstrainedLogitsProcessor
from transformers.cache_utils import DynamicCache
from transformers.generation.utils import GenerationMixin

torch = b.torch


class Scopes:
    def __init__(self, mode):
        self.mode = mode
        self.rows = []
        self.stack = []

    @contextlib.contextmanager
    def scope(self, name):
        if self.mode == 'torch':
            with torch.profiler.record_function('section/' + name):
                yield
            return
        row = {'name': name, 'parent': self.stack[-1]['name'] if self.stack else None,
               'children_s': 0.0}
        start_event, end_event = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start = time.perf_counter()
        start_event.record()
        self.stack.append(row)
        try:
            yield
        finally:
            end_event.record()
            elapsed = time.perf_counter() - start
            self.stack.pop()
            row.update(wall_ms=elapsed * 1000,
                       exclusive_wall_ms=(elapsed - row.pop('children_s')) * 1000,
                       events=(start_event, end_event))
            if self.stack:
                self.stack[-1]['children_s'] += elapsed
            self.rows.append(row)

    def data(self):
        result = []
        for row in self.rows:
            events = row['events']
            result.append({**{k: v for k, v in row.items() if k != 'events'},
                           'cuda_stream_elapsed_ms_including_idle': events[0].elapsed_time(events[1])})
        return result

    @contextlib.contextmanager
    def install(self, model):
        first = True

        def scoped(original, name):
            @functools.wraps(original)
            def call(*args, **kw):
                with self.scope(name):
                    return original(*args, **kw)
            return call

        original_forward = model.forward

        @functools.wraps(original_forward)
        def forward(*args, **kw):
            nonlocal first
            name = 'prefill_model' if first else 'decode_model'
            first = False
            with self.scope(name):
                return original_forward(*args, **kw)

        original_softmax = torch.nn.functional.log_softmax

        @functools.wraps(original_softmax)
        def log_softmax(*args, **kw):
            with self.scope('vocabulary_log_softmax'):
                return original_softmax(*args, **kw)

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(model, 'forward', forward))
            for obj, attr, name in [
                (model.model, 'forward', 'transformer'),
                (model.lm_head, 'forward', 'vocabulary_projection'),
                (b.ConstrainedLogitsProcessor, '__call__', 'constraints'),
                (DynamicCache, 'batch_repeat_interleave', 'cache_expand'),
                (DynamicCache, 'reorder_cache', 'cache_reorder'),
                (model, '_get_top_k_continuations', 'beam_topk_continuations'),
                (model, '_get_running_beams_for_next_iteration', 'beam_running_update'),
                (model, '_update_finished_beams', 'beam_finished_update'),
                (model, '_check_early_stop_heuristic', 'beam_early_stop'),
                (model, '_beam_search_has_unfinished_sequences', 'beam_stop_sync'),
                (model, 'prepare_inputs_for_generation', 'prepare_inputs'),
                (model, '_update_model_kwargs_for_generation', 'update_model_kwargs'),
                (torch, 'topk', 'topk')]:
                stack.enter_context(patch.object(obj, attr, scoped(getattr(obj, attr), name)))
            stack.enter_context(patch.object(torch.nn.functional, 'log_softmax', log_softmax))
            yield


def summarize_scopes(rows):
    groups = b.collections.defaultdict(list)
    for row in rows:
        groups[row['name']].append(row)
    return {name: {'calls': len(items), **{key: sum(x[key] for x in items)
             for key in ('wall_ms', 'exclusive_wall_ms', 'cuda_stream_elapsed_ms_including_idle')}}
            for name, items in groups.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release-root', type=Path, required=True)
    ap.add_argument('--reference-results', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, default=Path(os.environ.get('RAPIDTINYREC_OUTPUT_DIR', 'results')))
    ap.add_argument('--dtypes', nargs='+', choices=('bfloat16', 'float32'), default=['float32'])
    ap.add_argument('--constraint-ablation', action='store_true')
    args = ap.parse_args()
    assert len(args.dtypes) == 1, 'Use a fresh process per dtype so timings precede profiler initialization'
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    reference = json.loads((args.reference_results / 'manifest.json').read_text())
    expected = json.loads((b.ROOT / 'expected.json').read_text())
    assert hashlib.sha256(json.dumps(reference['users'], sort_keys=True).encode()).hexdigest() == expected['users_sha256']
    for rel, sha in expected['release_sha256'].items():
        assert b.digest(args.release_root / rel) == sha, rel
    for rel, sha in expected['source_sha256'].items():
        assert b.digest(b.ROOT / rel) == sha, rel
    representatives = {}
    for user in reference['users']:
        representatives.setdefault(user['prompt_tokens'], user)
    users = [u for _, u in sorted(representatives.items())]
    ckpt = args.release_root / 'Office_ckpt'
    tok = b.AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    constraints = b.collections.defaultdict(set)
    for line in (args.release_root / 'Amazon/info/Office_Products_5_2016-10-2018-11.txt').read_text().splitlines():
        ids = tok('### Response:\n' + line.split('\t')[0].strip() + '\n').input_ids + [tok.eos_token_id]
        assert all(u['prompt_token_ids'][-3:] == ids[:3] for u in users)
        for j in range(3, len(ids)):
            constraints[tuple(ids[:j] if j == 3 else ids[3:j])].add(ids[j])
    constraints = {k: list(v) for k, v in constraints.items()}
    report = {'hardware': torch.cuda.get_device_name(0),
              'gpu_info': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader'], text=True),
              'versions': {name: b.importlib.metadata.version(name) for name in ('torch', 'transformers', 'tokenizers')},
              'cohort_sha256': expected['users_sha256'], 'users': users,
              'source_sha256': {p.name: b.digest(p) for p in (Path(__file__), b.ROOT/'benchmark.py', b.ROOT/'shared_prefill.py', b.ROOT/'source/LogitProcessor.py', b.ROOT/'bulk_constraints.py')},
              'protocol': 'One representative per frozen prompt length. Per dtype: warm all ten; three uninstrumented repetitions per representative; one lightly instrumented request per representative; one Python profile at length 91; one CPU/CUDA trace each at lengths 76 and 121. All profiling outputs checked against same-allocation uninstrumented outputs. No per-scope CUDA synchronization. Uninstrumented runs exclude output extraction as in the 500-user benchmark.',
              'scope_timing_note': 'Exclusive wall times partition instrumented request wall time, including host waits. CUDA event elapsed includes CPU/stream idle and is NOT summed GPU kernel work. Torch trace overhead is separate from latency measurements.',
              'precision': {}}
    (out/'generation_source.py').write_text(inspect.getsource(GenerationMixin._beam_search))

    for dtype_name in args.dtypes:
        dtype = getattr(torch, dtype_name)
        model = b.AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=dtype,
            local_files_only=True, attn_implementation='sdpa').to('cuda').eval()
        assert all(p.dtype == dtype for p in model.parameters())
        model.config.pad_token_id = tok.eos_token_id
        model.config.eos_token_id = tok.eos_token_id
        model.config.bos_token_id = tok.bos_token_id
        inputs = {u['cohort_index']: torch.tensor([u['prompt_token_ids']], device='cuda') for u in users}
        masks = {i: torch.ones_like(x) for i, x in inputs.items()}
        references = {}
        reference_scores = {}

        @torch.inference_mode()
        def generate(user, bulk=False):
            cfg = b.GenerationConfig(max_new_tokens=8, length_penalty=0.0, num_beams=50,
                num_return_sequences=50, pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
                bos_token_id=tok.bos_token_id, do_sample=False, top_k=None, top_p=None, use_cache=True)
            processor = BulkPrefixConstrainedLogitsProcessor if bulk else b.ConstrainedLogitsProcessor
            proc = processor(lambda _, ids: constraints.get(tuple(ids), []),
                50, base_model=str(ckpt), eos_token_id=tok.eos_token_id)
            with b.PrefillOnce(model, 50):
                return model.generate(inputs[user['cohort_index']], attention_mask=masks[user['cohort_index']],
                    generation_config=cfg, do_sample=False, return_dict_in_generate=True,
                    output_scores=True, logits_processor=b.LogitsProcessorList([proc]))

        def validate(generated, user, strict=True):
            tokens = generated.sequences[:, user['prompt_tokens']:].cpu().tolist()
            scores = generated.sequences_scores.float().cpu().tolist()
            assert len(tokens) == len({tuple(x) for x in tokens}) == 50
            if user['cohort_index'] in references:
                if strict:
                    assert tokens == references[user['cohort_index']], 'Profiling changed rankings'
            else:
                references[user['cohort_index']] = tokens
                reference_scores[user['cohort_index']] = scores
            return {'all50_rankings_equal': tokens == references[user['cohort_index']],
                    'all50_sequence_scores_equal': scores == reference_scores[user['cohort_index']],
                    'token_ids': tokens, 'sequence_scores': scores}

        for user in users:
            output = generate(user)
            torch.cuda.synchronize()
            validate(output, user)
            del output
        measurements = []
        ablation_measurements = []
        if args.constraint_ablation:
            for user in users:
                output = generate(user, bulk=True)
                torch.cuda.synchronize()
                validate(output, user, strict=False)
                del output
        for repeat in range(3):
            for sample_index, user in enumerate(users if repeat % 2 == 0 else list(reversed(users))):
                order = [False, True] if args.constraint_ablation else [False]
                if (repeat + sample_index) % 2:
                    order = list(reversed(order))
                for bulk in order:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    output = generate(user, bulk=bulk)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    checks = validate(output, user, strict=not bulk)
                    row = {'cohort_index': user['cohort_index'], 'prompt_tokens': user['prompt_tokens'],
                           'repeat': repeat, 'elapsed_s': elapsed, **checks}
                    (ablation_measurements if bulk else measurements).append(row)
                    del output
        payload = {'uninstrumented': measurements, 'uninstrumented_summary': b.distribution([r['elapsed_s'] for r in measurements]),
                   'light_profiles': [], 'torch_profiles': [], 'reference_token_ids': references}
        report['precision'][dtype_name] = payload
        if args.constraint_ablation:
            payload['bulk_prefix_ablation'] = {'measurements': ablation_measurements,
                'summary': b.distribution([r['elapsed_s'] for r in ablation_measurements]),
                'ranking_mismatches': sum(not r['all50_rankings_equal'] for r in ablation_measurements),
                'score_mismatches': sum(not r['all50_sequence_scores_equal'] for r in ablation_measurements),
                'protocol': 'Original and bulk constraint processor alternate order on the same representative request, three repetitions per prompt length, before any profiler use in the process. Keep both full-vocabulary normalizations, all model calls, and HF beam search unchanged.'}
            b.announce({'stage': 'bulk_prefix_ablation', 'dtype': dtype_name,
                **{k: v for k, v in payload['bulk_prefix_ablation'].items() if k != 'measurements'}})
        b.announce({'stage': 'uninstrumented', 'dtype': dtype_name, **payload['uninstrumented_summary']})
        b.save(out/'profile.json', report)
        for user in users:
            scopes = Scopes('light')
            torch.cuda.synchronize()
            with scopes.install(model):
                with scopes.scope('request'):
                    output = generate(user)
                    with scopes.scope('final_sync'):
                        torch.cuda.synchronize()
            torch.cuda.synchronize()  # resolves the final event; outside measured request
            validate(output, user)
            rows = scopes.data()
            payload['light_profiles'].append({'cohort_index': user['cohort_index'],
                'prompt_tokens': user['prompt_tokens'], 'scopes': rows, 'summary': summarize_scopes(rows)})
            del output
        user = representatives[91]
        pyprof = cProfile.Profile()
        torch.cuda.synchronize()
        pyprof.enable()
        output = generate(user)
        torch.cuda.synchronize()
        pyprof.disable()
        validate(output, user)
        del output
        pyprof.dump_stats(str(out / f'python_{dtype_name}.prof'))
        stats = pstats.Stats(pyprof)
        payload['python_profile'] = [{'file': k[0], 'line': k[1], 'function': k[2],
            'primitive_calls': v[0], 'calls': v[1], 'self_s': v[2], 'cumulative_s': v[3]}
            for k, v in sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)]
        buf = io.StringIO()
        pstats.Stats(pyprof, stream=buf).sort_stats('cumulative').print_stats(70)
        (out / f'python_{dtype_name}.txt').write_text(buf.getvalue())
        for length in (76, 121):
            user = representatives[length]
            scopes = Scopes('torch')
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=True, profile_memory=False, with_stack=False) as prof:
                with scopes.install(model), scopes.scope('request'):
                    output = generate(user)
                    with scopes.scope('final_sync'):
                        torch.cuda.synchronize()
            validate(output, user)
            del output
            trace = out / f'trace_{dtype_name}_{length}.json'
            prof.export_chrome_trace(str(trace))
            with trace.open('rb') as src, gzip.open(str(trace) + '.gz', 'wb') as dst:
                shutil.copyfileobj(src, dst)
            trace.unlink()
            averages = []
            for event in prof.key_averages():
                averages.append({'name': event.key, 'count': event.count,
                    'cpu_total_us': event.cpu_time_total, 'self_cpu_us': event.self_cpu_time_total,
                    'device_total_us': event.device_time_total, 'self_device_us': event.self_device_time_total})
            payload['torch_profiles'].append({'prompt_tokens': length, 'trace': trace.name + '.gz',
                'events': averages, 'cuda_capture_has_device_time': any(e['self_device_us'] > 0 for e in averages)})
            (out / f'torch_{dtype_name}_{length}.txt').write_text(prof.key_averages().table(sort_by='self_device_time_total', row_limit=50))
            b.announce({'stage': 'torch_profile', 'dtype': dtype_name, 'prompt_tokens': length,
                        'cuda_events': payload['torch_profiles'][-1]['cuda_capture_has_device_time']})
            b.save(out/'profile.json', report)
            del prof
            gc.collect()
        del model, inputs, masks
        gc.collect()
        torch.cuda.empty_cache()
        b.save(out/'profile.json', report)
    report['complete'] = True
    b.save(out/'profile.json', report)
    b.announce({'stage': 'finished', 'complete': True})


if __name__ == '__main__':
    main()
