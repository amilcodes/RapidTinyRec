"""Profile eager/compiled forward phases without changing compiled module internals.

All uninstrumented comparisons precede light timing and CUPTI initialization.
CUDA event spans include stream idle time; only trace activities are GPU work.
"""
import argparse
import contextlib
import functools
import gc
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from unittest.mock import patch

from . import benchmark as b
from .compiled_forward import CompiledForward
from .gpu_constraints import GpuSidConstraintTables, GpuSidConstrainedLogitsProcessor
from .profile_shared import Scopes, summarize_scopes
from transformers.cache_utils import DynamicCache

torch = b.torch
VARIANTS = ('eager', 'compiled', 'compiled_prefill', 'compiled_decode')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release-root', type=Path, required=True)
    ap.add_argument('--reference-results', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, default=Path(os.environ.get('RAPIDTINYREC_OUTPUT_DIR', 'results')))
    ap.add_argument('--mask-ablation', action='store_true', help='Compare explicit compiled masks with the validated unpadded causal fast path')
    args = ap.parse_args()
    variants = ('eager', 'compiled', 'eager_gqa', 'compiled_gqa') if args.mask_ablation else VARIANTS
    profile_variants = ('compiled', 'compiled_gqa') if args.mask_ablation else ('eager', 'compiled')
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    if args.mask_ablation:
        import transformers.masking_utils as mask_source
        import transformers.integrations.sdpa_attention as sdpa_source
        import transformers.models.qwen2.modeling_qwen2 as qwen_source
        source_dir = out/'installed_transformers'
        source_dir.mkdir(exist_ok=True)
        for module in (mask_source, sdpa_source, qwen_source):
            shutil.copy2(inspect.getfile(module), source_dir/Path(inspect.getfile(module)).name)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference = json.loads((args.reference_results/'manifest.json').read_text())
    expected = json.loads((b.ROOT/'expected.json').read_text())
    assert reference['experiment'] == 'compile_forward' and reference['dtype'] == 'float32'
    assert hashlib.sha256(json.dumps(reference['users'], sort_keys=True).encode()).hexdigest() == expected['users_sha256']
    for rel, sha in expected['release_sha256'].items():
        assert b.digest(args.release_root/rel) == sha, rel
    for rel, sha in expected['source_sha256'].items():
        assert b.digest(b.ROOT/rel) == sha, rel
    representatives = {}
    for user in reference['users']:
        representatives.setdefault(user['prompt_tokens'], user)
    users = [u for _,u in sorted(representatives.items())]
    assert len(users) == 10
    ckpt = args.release_root/'Office_ckpt'
    tok = b.AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    constraints = b.collections.defaultdict(set)
    for line in (args.release_root/'Amazon/info/Office_Products_5_2016-10-2018-11.txt').read_text().splitlines():
        ids = tok('### Response:\n'+line.split('\t')[0].strip()+'\n').input_ids+[tok.eos_token_id]
        assert all(u['prompt_token_ids'][-3:] == ids[:3] for u in users)
        for j in range(3,len(ids)):
            constraints[tuple(ids[:j] if j == 3 else ids[3:j])].add(ids[j])
    constraints = {k:list(v) for k,v in constraints.items()}
    tables = GpuSidConstraintTables(constraints, users[0]['prompt_token_ids'][-3:], 198, tok.eos_token_id, 'cuda')
    model = b.AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float32,
        local_files_only=True, attn_implementation='sdpa').to('cuda').eval()
    assert all(x.dtype == torch.float32 for x in model.parameters())
    model.config.pad_token_id = tok.eos_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.config.bos_token_id = tok.bos_token_id
    inputs = {u['cohort_index']:torch.tensor([u['prompt_token_ids']],device='cuda') for u in users}
    masks = {i:torch.ones_like(x) for i,x in inputs.items()}
    if args.mask_ablation:
        assert model.config.model_type == 'qwen2' and not model.model.has_sliding_layers
        assert set(model.config.layer_types) == {'full_attention'}
        assert all(bool(mask.all().item()) for mask in masks.values())
    torch.cuda.synchronize()

    def gpu_state():
        fields = 'name,uuid,driver_version,memory.total,ecc.mode.current,clocks.current.graphics,clocks.current.memory,power.limit,power.draw,pstate,temperature.gpu'
        result = subprocess.run(['nvidia-smi','--query-gpu='+fields,'--format=csv'],text=True,capture_output=True)
        return {'fields':fields,'output':result.stdout,'stderr':result.stderr,'returncode':result.returncode}

    report = {'complete':False, 'dtype':'float32','tf32':False, 'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
        'hostname':subprocess.check_output(['hostname'],text=True).strip(), 'gpu_start':gpu_state(),
        'versions':{name:b.importlib.metadata.version(name) for name in ('torch','transformers','tokenizers')},
        'cohort_sha256':expected['users_sha256'], 'reference_manifest_sha256':reference['manifest_sha256'],
        'users':users, 'variants':list(variants), 'mask_ablation':args.mask_ablation, 'uninstrumented':[], 'light_profiles':[], 'torch_profiles':[],
        'historical_checks':[], 'source_sha256':{name:b.digest(b.ROOT/name) for name in
        ('profile_compiled.py','compiled_forward.py','profile_shared.py','benchmark.py','shared_prefill.py','gpu_constraints.py')},
        'profile_variants':list(profile_variants),
        'protocol':'First frozen user of each of ten prompt lengths. Four variants. Warm every shape; require stable compiler; then eight counterbalanced uninstrumented repetitions per variant and shape before any profiler use. Three light profiles per profile_variant and shape. Two CPU/CUDA traces per profile_variant at lengths 76 and 121. Output extraction excluded from uninstrumented timing. No per-scope CUDA synchronization. Preserve every timed sample.',
        'scope_note':'Representative requests and repeated trials; not a new 500-user p95 estimate. Light CUDA events include stream/CPU idle. Detailed traces add overhead; count only actual GPU activities.'}
    compiler_dir = out/'compilation'
    compiler_dir.mkdir(exist_ok=True)
    compiler = CompiledForward(model, compiler_dir, b.announce)
    current_scopes = None
    original_reorder = DynamicCache.reorder_cache

    def reorder(cache, *a, **kw):
        with current_scopes.scope('cache_reorder') if current_scopes else contextlib.nullcontext():
            return original_reorder(cache,*a,**kw)

    class Processor:
        def __init__(self, scope):
            self.proc = GpuSidConstrainedLogitsProcessor(tables)
            self.scope = scope
        def __call__(self, input_ids, scores):
            with self.scope.scope('constraints') if self.scope else contextlib.nullcontext():
                return self.proc(input_ids,scores)

    @torch.inference_mode()
    def generate(user, variant, scopes=None):
        nonlocal current_scopes
        current_scopes = scopes
        call_count = 0
        expansion_scope = None
        @functools.wraps(compiler.eager)
        def forward(*a, **kw):
            nonlocal call_count
            phase = 'prefill_model' if call_count == 0 else 'decode_'+str(call_count)
            use_compiled = variant in ('compiled', 'compiled_gqa') or (variant == 'compiled_prefill' and call_count == 0) or (variant == 'compiled_decode' and call_count > 0)
            if variant.endswith('_gqa'):
                # This experiment admits only a complete unpadded history with
                # empty cache, followed by single-token cached decoding. For
                # these two shapes SDPA's is_causal flag is exactly sufficient.
                # The 2D all-ones mask still reaches HF generation/position IDs;
                # replace it only at the model boundary after shared prefill.
                ids = kw.get('input_ids', a[0] if a else None)
                cache = kw.get('past_key_values')
                if call_count == 0:
                    assert ids.shape == (1, user['prompt_tokens'])
                    assert cache is None or cache.get_seq_length() == 0
                else:
                    assert ids.shape == (50, 1)
                    assert cache.get_seq_length() == user['prompt_tokens']+call_count-1
                kw['attention_mask'] = {'full_attention':None}
            call_count += 1
            with scopes.scope(phase) if scopes else contextlib.nullcontext():
                return (compiler.compiled if use_compiled else compiler.eager)(*a,**kw)
        def on_prefill(stage, cache):
            nonlocal expansion_scope
            if scopes:
                if stage == 'before_expansion':
                    expansion_scope = scopes.scope('cache_expand')
                    expansion_scope.__enter__()
                else:
                    expansion_scope.__exit__(None,None,None)
        cfg = b.GenerationConfig(max_new_tokens=8,length_penalty=0.0,num_beams=50,num_return_sequences=50,
            pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id,bos_token_id=tok.bos_token_id,
            do_sample=False,top_k=None,top_p=None,use_cache=True)
        try:
            with patch.object(model,'forward',forward), b.PrefillOnce(model,50,on_prefill=on_prefill if scopes else None):
                output = model.generate(inputs[user['cohort_index']],attention_mask=masks[user['cohort_index']],
                    generation_config=cfg,do_sample=False,return_dict_in_generate=True,output_scores=True,
                    logits_processor=b.LogitsProcessorList([Processor(scopes)]))
            assert call_count == 5, call_count
            return output
        finally:
            current_scopes = None

    references = {}
    def extract(output,user):
        return {'token_ids':output.sequences[:,user['prompt_tokens']:].cpu().tolist(),
            'sequence_logp':output.sequences_scores.float().cpu().tolist(),
            'transition_logp':model.compute_transition_scores(output.sequences,output.scores,
                output.beam_indices,normalize_logits=False).float().cpu().tolist()}
    def validate(output,user,variant):
        raw = extract(output,user)
        key = (user['cohort_index'],variant)
        assert len(raw['token_ids']) == len({tuple(x) for x in raw['token_ids']}) == 50
        if key not in references:
            references[key] = raw
        assert raw == references[key], 'Profiling/repetition changed outputs'
        return raw

    with patch.object(DynamicCache,'reorder_cache',reorder):
        for user in users:
            for variant in variants:
                before = len(compiler.events)
                start = time.perf_counter()
                output = generate(user,variant)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-start
                validate(output,user,variant)
                del output
                compiler.add_warmup({'phase':'initial','variant':variant,'prompt_tokens':user['prompt_tokens'],
                    'elapsed_s':elapsed,'new_backend_graphs':len(compiler.events)-before})
        for sweep in range(3):
            before = len(compiler.events)
            for user in users:
                for variant in (('compiled','compiled_gqa') if args.mask_ablation else ('compiled',)):
                    output = generate(user,variant)
                    torch.cuda.synchronize()
                    validate(output,user,variant)
                    del output
            if len(compiler.events) == before:
                break
        else:
            raise RuntimeError('Compiler did not stabilize')
        compiler.freeze()
        b.announce({'stage':'compiler_ready','seconds':compiler.setup_wall_s,'attempts':len(compiler.events)})
        for user in users:
            old = json.loads((args.reference_results/'users'/f"{user['cohort_index']:04d}.json").read_text())
            for variant,old_name in [('eager','baseline'),('compiled','shared')]:
                raw = references[(user['cohort_index'],variant)]
                report['historical_checks'].append({'cohort_index':user['cohort_index'],'variant':variant,
                    'comparison':b.compare(old['runs'][old_name],raw)})
        for repeat in range(8):
            for index,user in enumerate(users if repeat%2 == 0 else list(reversed(users))):
                offset = (repeat+user['prompt_tokens'])%4
                order = list(variants[offset:]+variants[:offset])
                if repeat >= 4:
                    order.reverse()
                for variant in order:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    output = generate(user,variant)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter()-start
                    raw = validate(output,user,variant)
                    report['uninstrumented'].append({'repeat':repeat,'cohort_index':user['cohort_index'],
                        'prompt_tokens':user['prompt_tokens'],'variant':variant,'order':order,'elapsed_s':elapsed,**raw})
                    del output
            b.save(out/'profile.json',report)
            b.announce({'stage':'uninstrumented_repeat','repeat':repeat})
        report['gpu_after_uninstrumented'] = gpu_state()

        for repeat in range(3):
            for index,user in enumerate(users):
                for variant in (profile_variants if (repeat+index)%2 == 0 else profile_variants[::-1]):
                    scopes = Scopes('light')
                    torch.cuda.synchronize()
                    with scopes.scope('request'):
                        output = generate(user,variant,scopes)
                        with scopes.scope('final_sync'):
                            torch.cuda.synchronize()
                    torch.cuda.synchronize()
                    validate(output,user,variant)
                    rows = scopes.data()
                    report['light_profiles'].append({'repeat':repeat,'cohort_index':user['cohort_index'],
                        'prompt_tokens':user['prompt_tokens'],'variant':variant,'scopes':rows,'summary':summarize_scopes(rows)})
                    del output
        b.save(out/'profile.json',report)
        for repeat in range(2):
            for length in (76,121):
                user = representatives[length]
                for variant in (profile_variants if repeat == 0 else profile_variants[::-1]):
                    scopes = Scopes('torch')
                    torch.cuda.synchronize()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                            record_shapes=True,profile_memory=False,with_stack=False) as prof:
                        with scopes.scope('request'):
                            output = generate(user,variant,scopes)
                            with scopes.scope('final_sync'):
                                torch.cuda.synchronize()
                    validate(output,user,variant)
                    del output
                    trace = out/f'trace_{variant}_{length}_{repeat}.json'
                    prof.export_chrome_trace(str(trace))
                    with trace.open('rb') as src,gzip.open(str(trace)+'.gz','wb') as dst:
                        shutil.copyfileobj(src,dst)
                    trace.unlink()
                    averages = [{'name':e.key,'input_shapes':e.input_shapes,'count':e.count,
                        'cpu_total_us':e.cpu_time_total,'self_cpu_us':e.self_cpu_time_total,
                        'device_total_us':e.device_time_total,'self_device_us':e.self_device_time_total}
                        for e in prof.key_averages(group_by_input_shape=True)]
                    report['torch_profiles'].append({'repeat':repeat,'prompt_tokens':length,'variant':variant,
                        'trace':trace.name+'.gz','events':averages})
                    b.save(out/'profile.json',report)
                    b.announce({'stage':'trace_finished','variant':variant,'length':length,'repeat':repeat})
                    del prof
                    gc.collect()
        compiler.finish()
    report['complete'] = True
    report['gpu_end'] = gpu_state()
    report['reference_outputs'] = [{'cohort_index':i,'variant':v,**raw} for (i,v),raw in references.items()]
    b.save(out/'profile.json',report)
    b.announce({'stage':'finished','complete':True})


if __name__ == '__main__':
    main()
