"""Three-way float32/A6000 benchmark, with compiled GQA mask specialization."""
import argparse
import collections
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from unittest.mock import patch

from . import benchmark as b
from .compiled_forward import CompiledForward, SETTINGS
from .gqa_benchmark_protocol import VARIANTS, build_orders
from .gpu_constraints import GpuSidConstraintTables, GpuSidConstrainedLogitsProcessor
from .validate_gpu_constraints import validate as validate_constraints

torch = b.torch


def gpu_state():
    fields = 'name,uuid,driver_version,memory.total,ecc.mode.current,clocks.current.graphics,clocks.current.memory,power.limit,power.draw,pstate,temperature.gpu'
    return {'fields':fields,'output':subprocess.check_output(['nvidia-smi','--query-gpu='+fields,'--format=csv'],text=True)}


def summarize(rows, complete=False):
    return {'complete':complete,'completed_users':len(rows),
        'latency':{v:b.distribution([r['runs'][v]['elapsed_s'] for r in rows]) for v in VARIANTS},
        'gqa_vs_compiled':{
            'matching_rank_lists':sum(r['comparisons']['gqa_vs_compiled']['ranked_tokens_equal'] for r in rows),
            'matching_sequence_score_vectors':sum(r['comparisons']['gqa_vs_compiled']['sequence_scores_equal_in_rank_order'] for r in rows),
            'matching_transition_score_arrays':sum(r['comparisons']['gqa_vs_compiled']['transition_scores_equal_in_rank_order'] for r in rows)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release-root', type=Path, required=True)
    ap.add_argument('--reference-results', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(42)
    torch.set_num_threads(4)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert torch.cuda.get_device_name(0) == 'NVIDIA RTX A6000', 'A6000 required; no hardware substitution'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    expected = json.loads((b.ROOT/'expected.json').read_text())
    reference = json.loads((args.reference_results/'manifest.json').read_text())
    assert reference['experiment'] == 'compile_forward' and reference['dtype'] == 'float32'
    old_hash = reference['manifest_sha256']
    assert hashlib.sha256(json.dumps({k:v for k,v in reference.items() if k!='manifest_sha256'},sort_keys=True).encode()).hexdigest() == old_hash
    users = reference['users']
    assert len(users) == len({u['user_id'] for u in users}) == 500
    assert [u['cohort_index'] for u in users] == list(range(500))
    assert hashlib.sha256(json.dumps(users,sort_keys=True).encode()).hexdigest() == expected['users_sha256']
    for rel,sha in expected['release_sha256'].items():
        assert b.digest(args.release_root/rel) == sha, rel
    for rel,sha in expected['source_sha256'].items():
        assert b.digest(b.ROOT/rel) == sha, rel
    orders, balance = build_orders(users)
    out = args.output_root/'gqa_500_users'
    out.mkdir(parents=True,exist_ok=True)
    requests = out/'users'
    requests.mkdir(exist_ok=True)
    assert not list(requests.glob('*.json')), 'Never overwrite or combine measured sessions'
    ckpt = args.release_root/'Office_ckpt'
    tok = b.AutoTokenizer.from_pretrained(ckpt,local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    assert tok.encode('\n',add_special_tokens=False) == [198]
    constraints = collections.defaultdict(set)
    for line in (args.release_root/'Amazon/info/Office_Products_5_2016-10-2018-11.txt').read_text().splitlines():
        ids = tok('### Response:\n'+line.split('\t')[0].strip()+'\n').input_ids+[tok.eos_token_id]
        assert all(u['prompt_token_ids'][-3:] == ids[:3] for u in users)
        for j in range(3,len(ids)):
            constraints[tuple(ids[:j] if j==3 else ids[3:j])].add(ids[j])
    constraints = {k:list(v) for k,v in constraints.items()}
    tables = GpuSidConstraintTables(constraints,users[0]['prompt_token_ids'][-3:],198,tok.eos_token_id,'cuda')
    check = validate_constraints(tables,constraints,users[0]['prompt_token_ids'][-3:],tok.eos_token_id,len(tok),'cuda')
    b.save(out/'constraint_validation.json',{'metadata':tables.metadata,'validation':check})
    model = b.AutoModelForCausalLM.from_pretrained(ckpt,torch_dtype=torch.float32,local_files_only=True,attn_implementation='sdpa').to('cuda').eval()
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert model.config.model_type == 'qwen2' and not model.model.has_sliding_layers
    assert set(model.config.layer_types) == {'full_attention'}
    model.config.pad_token_id = tok.eos_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.config.bos_token_id = tok.bos_token_id
    # These are unpadded single requests. All-ones masks are constructed here;
    # generation extends them with ones. No padding-content sync in timing.
    inputs = {u['cohort_index']:torch.tensor([u['prompt_token_ids']],device='cuda') for u in users}
    masks = {i:torch.ones_like(x) for i,x in inputs.items()}
    torch.cuda.synchronize()
    provenance = ['benchmark_gqa.py','gqa_benchmark_protocol.py','compiled_forward.py','benchmark.py','shared_prefill.py','gpu_constraints.py','validate_gpu_constraints.py','expected.json','source/data.py','source/LogitProcessor.py']
    manifest = {'experiment':'gqa_three_way','dtype':'float32','tf32':False,'num_beams':50,'num_return_sequences':50,
        'request_batch':1,'max_new_tokens':8,'length_penalty':0.0,'threads':4,'seed':42,'users':users,
        'variant_definitions':{'eager':'Shared prefill, GPU constraints, eager forward',
            'compiled':'Shared prefill, GPU constraints, original compiled forward with generated mask',
            'compiled_gqa':'Same compiled forward with the validated unpadded causal mask specialization'},
        'cohort_sha256':expected['users_sha256'],'reference_manifest_sha256':old_hash,
        'reference_results':str(args.reference_results),'source_sha256':{n:b.digest(b.ROOT/n) for n in provenance},
        'orders':{str(i):list(v) for i,v in orders.items()},'order_balance':balance,
        'hostname':subprocess.check_output(['hostname'],text=True).strip(),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
        'gpu_start':gpu_state(),'cuda_runtime':torch.version.cuda,'versions':{n:b.importlib.metadata.version(n) for n in ['torch','transformers','tokenizers']},
        'compilation_settings':SETTINGS,
        'protocol':'One synchronized, unprofiled request per variant per frozen user. Three-variant ordering balanced by history length and globally. All five forwards and full vocabulary scoring retained. All compilation/warmup/input preparation/output extraction/persistence excluded. Generation setup, GPU constraints, dense cache expansion and reordering included. Fresh compiler caches per run; both compiled variants share this process/cache. Warm every shape and forbid measurement-time compilation. Preserve all samples, mismatches and outliers.',
        'p95_method':'Linear interpolation at sorted index (N-1)*0.95.'}
    manifest['manifest_sha256'] = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    b.save(out/'manifest.json',manifest)
    b.announce({'stage':'model_ready','hardware':torch.cuda.get_device_name(0),'users':500,'order_balance':balance['global']})
    compiler_dir = out/'compilation'
    compiler_dir.mkdir(exist_ok=True)
    compiler = CompiledForward(model,compiler_dir,b.announce)

    @torch.inference_mode()
    def generate(user,variant):
        calls = 0
        @functools.wraps(compiler.eager)
        def forward(*a,**kw):
            nonlocal calls
            if variant == 'compiled_gqa':
                ids = kw.get('input_ids',a[0] if a else None)
                cache = kw.get('past_key_values')
                if calls == 0:
                    assert ids.shape == (1,user['prompt_tokens'])
                    assert cache is None or cache.get_seq_length() == 0
                else:
                    assert ids.shape == (50,1)
                    assert cache.get_seq_length() == user['prompt_tokens']+calls-1
                kw['attention_mask'] = {'full_attention':None}
            calls += 1
            return (compiler.eager if variant == 'eager' else compiler.compiled)(*a,**kw)
        cfg = b.GenerationConfig(max_new_tokens=8,length_penalty=0.0,num_beams=50,num_return_sequences=50,
            pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id,bos_token_id=tok.bos_token_id,
            do_sample=False,top_k=None,top_p=None,use_cache=True)
        with patch.object(model,'forward',forward),b.PrefillOnce(model,50):
            output = model.generate(inputs[user['cohort_index']],attention_mask=masks[user['cohort_index']],
                generation_config=cfg,do_sample=False,return_dict_in_generate=True,output_scores=True,
                logits_processor=b.LogitsProcessorList([GpuSidConstrainedLogitsProcessor(tables)]))
        assert calls == 5
        return output

    def unpack(output,user):
        raw = {'token_ids':output.sequences[:,user['prompt_tokens']:].cpu().tolist(),
            'sequence_logp':output.sequences_scores.float().cpu().tolist(),
            'transition_logp':model.compute_transition_scores(output.sequences,output.scores,output.beam_indices,normalize_logits=False).float().cpu().tolist()}
        assert len(raw['token_ids']) == len({tuple(s) for s in raw['token_ids']}) == 50
        assert all(len(s)==5 and s[-2:]==[198,tok.eos_token_id] for s in raw['token_ids'])
        assert all(math.isfinite(x) for x in raw['sequence_logp'])
        assert all(len(row)==5 and all(math.isfinite(x) for x in row) for row in raw['transition_logp'])
        return raw

    warm_outputs = {}
    def warm(user,variant,phase):
        before = len(compiler.events)
        timing_before = sum(torch._dynamo.utils.compilation_time_metrics.get('_compile.compile_inner',[]))
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = generate(user,variant)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        raw = unpack(output,user)
        del output
        key = (user['cohort_index'],variant)
        if key in warm_outputs:
            assert raw == warm_outputs[key], 'Warmup output instability'
        warm_outputs[key] = raw
        for event in compiler.events[before:]:
            event['variant'] = variant
        compiler.add_warmup({'phase':phase,'variant':variant,'prompt_tokens':user['prompt_tokens'],
            'elapsed_s':elapsed,'new_backend_graphs':len(compiler.events)-before,
            'outer_compiler_seconds':sum(torch._dynamo.utils.compilation_time_metrics.get('_compile.compile_inner',[]))-timing_before})
        b.announce({'stage':'warmup','phase':phase,'variant':variant,'length':user['prompt_tokens'],'excluded_seconds':elapsed})

    representatives = {}
    for user in users:
        representatives.setdefault(user['prompt_tokens'],user)
    for length,user in sorted(representatives.items()):
        for variant in VARIANTS:
            warm(user,variant,'initial')
    for sweep in range(3):
        before = len(compiler.events)
        for length,user in sorted(representatives.items()):
            for variant in ('compiled','compiled_gqa'):
                warm(user,variant,'stability')
        if len(compiler.events)==before:
            break
    else:
        raise RuntimeError('Compiler did not stabilize')
    compiler.freeze()
    b.save(out/'compile_timing.json',{
        'setup_and_warmup_wall_s':compiler.setup_wall_s,
        'wrapper_creation_s':compiler.wrapper_creation_s,
        'per_variant':{v:{
            'warmup_request_seconds':sum(w['elapsed_s'] for w in compiler.warmups if w['variant']==v),
            'compiler_frontend_backend_seconds':sum(w['outer_compiler_seconds'] for w in compiler.warmups if w['variant']==v),
            'backend_seconds':sum(e['backend_compile_s'] for e in compiler.events if e.get('variant')==v),
            'first_request_including_compile_s':next(w['elapsed_s'] for w in compiler.warmups if w['variant']==v)} for v in VARIANTS},
        'note':'These timings overlap; do not add nested rows. Fresh per-run caches, shared by both compiled variants. Original compilation warms first; the GQA compiler cost is incremental after original compilation, not an independent cold-start measurement. Setup includes all three variants and stability sweeps. Excludes model loading, input preparation, and Slurm queue time. No compile time is part of inference latency.'})
    b.save(out/'warmup_outputs.json',[{'cohort_index':i,'variant':v,**raw} for (i,v),raw in warm_outputs.items()])
    b.announce({'stage':'compiler_ready','setup_and_warmup_seconds':compiler.setup_wall_s})
    rows = []
    for user in users:
        i = user['cohort_index']
        runs = {}
        for variant in orders[i]:
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = generate(user,variant)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            raw = unpack(output,user)
            del output
            if (i,variant) in warm_outputs:
                assert raw == warm_outputs[(i,variant)], 'Measured output changed after warmup'
            runs[variant] = {'elapsed_s':elapsed,**raw}
        historical = json.loads((args.reference_results/'users'/f'{i:04d}.json').read_text())
        assert historical['manifest_sha256']==old_hash and historical['user_id']==user['user_id']
        row = {**{k:user[k] for k in ['cohort_index','source_row_index','user_id','prompt_tokens','target_sid']},
            'order':list(orders[i]),'manifest_sha256':manifest['manifest_sha256'],'completed_at':b.utc_now(),'runs':runs,
            'comparisons':{'gqa_vs_compiled':b.compare(runs['compiled'],runs['compiled_gqa']),
                'gqa_vs_eager':b.compare(runs['eager'],runs['compiled_gqa']),
                'compiled_vs_eager':b.compare(runs['eager'],runs['compiled'])},
            'historical':{'eager':b.compare(historical['runs']['baseline'],runs['eager']),
                'compiled':b.compare(historical['runs']['shared'],runs['compiled'])},
            'cuda_allocated_bytes':torch.cuda.memory_allocated(),'cuda_reserved_bytes':torch.cuda.memory_reserved()}
        b.save(requests/f'{i:04d}.json',row)
        rows.append(row)
        if (i+1)%25 == 0 or i==0:
            summary = summarize(rows)
            b.save(out/'summary.json',summary)
            b.announce({'stage':'progress',**summary})
    compiler.finish()
    # Preserve source for the actual generated forward modules, excluding binaries.
    generated = out/'generated_inductor'
    generated.mkdir(exist_ok=True)
    index = []
    for path in Path(os.environ['TORCHINDUCTOR_CACHE_DIR']).rglob('*.py'):
        data = path.read_bytes()
        if b'extern_kernels.mm' in data:
            shutil.copy2(path,generated/path.name)
            index.append({'name':path.name,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
    b.save(generated/'index.json',index)
    result = {**summarize(rows,True),'manifest_sha256':manifest['manifest_sha256'],'gpu_end':gpu_state()}
    b.save(out/'summary.json',result)
    b.announce({'stage':'finished',**result})


if __name__ == '__main__':
    main()
