"""Profile replay only after uninstrumented requests; never use trace latency as a benchmark."""
import contextlib
import functools
import gzip
import shutil
from unittest.mock import patch

from transformers.cache_utils import DynamicCache

from . import benchmark as b
from . import decode_replay as replay
from .audit_decode_pilot import bits, flat
from .gpu_constraints import GpuSidConstrainedLogitsProcessor
from .profile_shared import Scopes, summarize_scopes

torch = b.torch


@contextlib.contextmanager
def instrument(scopes, model, compiler):
    def scoped(original, name):
        @functools.wraps(original)
        def call(*args, **kwargs):
            with scopes.scope(name):
                return original(*args, **kwargs)
        return call

    original = compiler.compiled
    @functools.wraps(original)
    def forward(**kwargs):
        name = 'compiled_prefill' if kwargs['input_ids'].shape[1] > 1 else 'compiled_decode'
        with scopes.scope(name):
            return original(**kwargs)

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(compiler, 'compiled', forward))
        for obj, attr, name in [
            (replay, 'signature', 'replay_signature'),
            (replay.ReplayBuffers, 'load', 'replay_input_staging'),
            (replay.ReplayOutput, 'fresh', 'replay_output_container'),
            (torch.cuda.CUDAGraph, 'replay', 'graph_replay'),
            (DynamicCache, 'batch_repeat_interleave', 'cache_expand'),
            (DynamicCache, 'reorder_cache', 'cache_reorder'),
            (GpuSidConstrainedLogitsProcessor, '__call__', 'constraints'),
            (model, 'prepare_inputs_for_generation', 'prepare_inputs'),
            (model, '_update_model_kwargs_for_generation', 'update_model_kwargs'),
            (model, '_get_top_k_continuations', 'beam_topk_continuations'),
            (model, '_get_running_beams_for_next_iteration', 'beam_running_update'),
            (model, '_update_finished_beams', 'beam_finished_update'),
            (model, '_check_early_stop_heuristic', 'beam_early_stop'),
            (model, '_beam_search_has_unfinished_sequences', 'beam_stop_sync')]:
            stack.enter_context(patch.object(obj, attr, scoped(getattr(obj, attr), name)))
        yield


def profile(model, compiler, pool, users, generate, unpack, reference, out):
    assert compiler.phase == 'complete' and pool.frozen
    representatives = {}
    for user in users:
        representatives.setdefault(user['prompt_tokens'], user)
    assert len(representatives) == 2
    report = {'complete': False, 'light_profiles': [], 'torch_profiles': [],
              'before_replay': pool.report(), 'source_sha256': {
                  name: b.digest(b.ROOT / name) for name in ('profile_decode_replay.py', 'profile_shared.py')},
              'note': 'Uninstrumented pilot timings were saved before profiling. Light CUDA-event spans include CPU/stream idle; use trace kernel activities for GPU work. Scopes are nested, not additive. All instrumented outputs are checked against uninstrumented FP32 bits.'}

    def validate(output, user, variant):
        raw = unpack(output, user)
        expected = reference[(user['cohort_index'], variant)]
        assert raw['token_ids'] == expected['token_ids'], 'Profiled ranks changed'
        assert all(bits(flat(raw, field)) == bits(flat(expected, field))
                   for field in ('sequence_logp', 'transition_logp')), 'Profiled score bits changed'
        return raw

    for repeat in range(3):
        for length, user in sorted(representatives.items()):
            for variant in (('compiled_gqa', 'replay') if repeat % 2 == 0 else ('replay', 'compiled_gqa')):
                scopes = Scopes('light')
                torch.cuda.synchronize()
                with instrument(scopes, model, compiler), scopes.scope('request'):
                    output = generate(user, variant == 'replay')
                    with scopes.scope('final_sync'):
                        torch.cuda.synchronize()
                torch.cuda.synchronize()
                raw = validate(output, user, variant)
                del output
                rows = scopes.data()
                report['light_profiles'].append({'repeat': repeat, 'prompt_tokens': length,
                    'cohort_index': user['cohort_index'], 'variant': variant,
                    'scopes': rows, 'summary': summarize_scopes(rows), 'output': raw})
    b.save(out / 'replay_profile.json', report)
    for length, user in sorted(representatives.items()):
        for variant in ('compiled_gqa', 'replay'):
            scopes = Scopes('torch')
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True, profile_memory=False, with_stack=False) as prof:
                with instrument(scopes, model, compiler), scopes.scope('request'):
                    output = generate(user, variant == 'replay')
                    with scopes.scope('final_sync'):
                        torch.cuda.synchronize()
            raw = validate(output, user, variant)
            del output
            trace = out / f'replay_trace_{variant}_{length}.json'
            prof.export_chrome_trace(str(trace))
            with trace.open('rb') as src, gzip.open(str(trace) + '.gz', 'wb') as dst:
                shutil.copyfileobj(src, dst)
            trace.unlink()
            events = [{'name': e.key, 'input_shapes': e.input_shapes, 'count': e.count,
                       'cpu_total_us': e.cpu_time_total, 'self_cpu_us': e.self_cpu_time_total,
                       'device_total_us': e.device_time_total, 'self_device_us': e.self_device_time_total}
                      for e in prof.key_averages(group_by_input_shape=True)]
            report['torch_profiles'].append({'prompt_tokens': length, 'cohort_index': user['cohort_index'],
                'variant': variant, 'trace': trace.name + '.gz', 'events': events, 'output': raw})
            b.save(out / 'replay_profile.json', report)
            b.announce({'stage': 'replay_trace_finished', 'variant': variant, 'prompt_tokens': length})
    assert report['before_replay']['events'] == pool.report()['events'], 'Unexpected capture during profiling'
    assert len(compiler.events) == compiler.frozen_event_count
    report.update(complete=True, after_replay=pool.report(), measurement_compilations=0)
    b.save(out / 'replay_profile.json', report)
