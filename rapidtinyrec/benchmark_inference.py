"""Sequential folded inference benchmark; FP32 default, explicit experimental BF16."""
import argparse
import collections
import contextlib
import functools
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

from . import benchmark as b
from .compiled_forward import CompiledForward
from .prefix_prefill import PrefillPrefix
from .folded_attention import fold_decode_attention
from .gpu_constraints import GpuSidConstraintTables,GpuSidConstrainedLogitsProcessor

from .inference_precision import PRECISIONS, model_dtype, precision_policy, policy_state, validate_runtime, validate_output

torch=b.torch


def select_users(reference_users, full_cohort):
    assert len(reference_users)==500
    lengths=sorted({u['prompt_tokens'] for u in reference_users})
    if full_cohort:
        assert len(lengths)==10
        return reference_users,[next(u for u in reference_users if u['prompt_tokens']==n) for n in lengths]
    users=[u for length in (lengths[0],lengths[-1])
           for u in [x for x in reference_users if x['prompt_tokens']==length][:2]]
    assert len(users)==4
    return users,users


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release-root',type=Path,required=True)
    ap.add_argument('--reference-results',type=Path,required=True)
    ap.add_argument('--output-root',type=Path,required=True)
    ap.add_argument('--full-cohort',action='store_true')
    ap.add_argument('--precision', choices=PRECISIONS, default='fp32')
    ap.add_argument('--repeats', type=int, default=2, help='Pilot repetitions in alternating forward/reverse user order')
    ap.add_argument('--save-full-scores', action='store_true', help='Lossless gzip FP32 scores for independent acceptance audit; outside latency timer')
    args=ap.parse_args()
    if args.repeats < 1 or args.repeats > 10:
        ap.error('--repeats must be between 1 and 10')
    if args.output_root.exists():
        ap.error('--output-root must be a new directory')
    validate_runtime(torch, b.importlib.metadata.version('transformers'))
    before = policy_state(torch)
    try:
        with precision_policy(torch, args.precision) as active_policy:
            with fold_decode_attention(True):
                run(args, active_policy)
    finally:
        if args.output_root.exists():
            b.save(args.output_root/'policy-restoration.json', {'before':before, 'after':policy_state(torch), 'restored':before==policy_state(torch)})


def run(args, active_policy):
    variant='folded_prefix_replay'
    dtype=model_dtype(torch,args.precision)
    cache_paths={k:Path(os.environ[k]).resolve() for k in ('TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR')}
    def artifact_inventory():
        return {k:{'path':str(p),'files':sum(x.is_file() for x in p.rglob('*')) if p.exists() else 0}
                for k,p in cache_paths.items()}
    artifacts_before=artifact_inventory()
    assert not sum(v['files'] for v in artifacts_before.values()), 'Use fresh precision-specific compiler caches'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    assert torch.cuda.get_device_name(0)=='NVIDIA RTX A6000'
    torch.manual_seed(42);torch.set_num_threads(4)
    expected=json.loads((b.ROOT/'expected.json').read_text())
    reference=json.loads((args.reference_results/'manifest.json').read_text())
    assert reference['experiment']=='gqa_three_way' and reference['dtype']=='float32'
    content={k:v for k,v in reference.items() if k!='manifest_sha256'}
    assert hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()==reference['manifest_sha256']
    assert hashlib.sha256(json.dumps(reference['users'],sort_keys=True).encode()).hexdigest()==expected['users_sha256']
    for name,want in expected['release_sha256'].items():assert b.digest(args.release_root/name)==want,name
    users,warm_users=select_users(reference['users'],args.full_cohort)
    out=args.output_root
    out.mkdir(parents=True,exist_ok=False)
    (out/'users').mkdir()
    tok=b.AutoTokenizer.from_pretrained(args.release_root/'Office_ckpt',local_files_only=True)
    tok.pad_token=tok.eos_token;tok.padding_side='left'
    constraints=collections.defaultdict(set)
    for line in (args.release_root/'Amazon/info/Office_Products_5_2016-10-2018-11.txt').read_text().splitlines():
        ids=tok('### Response:\n'+line.split('\t')[0].strip()+'\n').input_ids+[tok.eos_token_id]
        assert all(u['prompt_token_ids'][-3:]==ids[:3] for u in users)
        for j in range(3,len(ids)):constraints[tuple(ids[:j] if j==3 else ids[3:j])].add(ids[j])
    tables=GpuSidConstraintTables({k:list(v) for k,v in constraints.items()},users[0]['prompt_token_ids'][-3:],198,tok.eos_token_id,'cuda')
    model=b.AutoModelForCausalLM.from_pretrained(args.release_root/'Office_ckpt',torch_dtype=dtype,
        local_files_only=True,attn_implementation='sdpa').to('cuda').eval()
    assert model.config.model_type=='qwen2' and not model.model.has_sliding_layers
    assert set(model.config.layer_types)=={'full_attention'}
    model.config.pad_token_id=tok.eos_token_id;model.config.eos_token_id=tok.eos_token_id;model.config.bos_token_id=tok.bos_token_id
    assert all(p.dtype==dtype for p in model.parameters())
    inputs={u['cohort_index']:torch.tensor([u['prompt_token_ids']],device='cuda') for u in users}
    masks={i:torch.ones_like(x) for i,x in inputs.items()}
    manifest={'experiment':'folded_inference_v1',
        'precision':args.precision, 'experimental':args.precision!='fp32', 'precision_policy':active_policy,
        'variant':variant, 'repeats':1 if args.full_cohort else args.repeats, 'full_cohort':args.full_cohort,
        'save_full_scores':args.save_full_scores,
        'attention_layout':'folded_query_heads' if variant=='folded_prefix_replay' else 'native_gqa',
        'compiler_artifacts':{'reused':False,'before':artifacts_before},
        'users':users,'dtype':str(dtype).removeprefix('torch.'),'tf32':active_policy['matmul_tf32'],'hardware':torch.cuda.get_device_name(0),
        'torch_git':torch.version.git_version,'cuda_version':torch.version.cuda,
        'cohort_sha256':expected['users_sha256'],'reference_manifest_sha256':reference['manifest_sha256'],
        'versions':{n:b.importlib.metadata.version(n) for n in ['torch','transformers','tokenizers']},
        'sha256':{n:b.digest(b.ROOT/n) for n in ['benchmark_inference.py','inference_precision.py','folded_attention.py','shared_prefix.py','prefix_replay.py','prefix_prefill.py','input_arena.py','decode_replay.py',
            'compiled_forward.py','shared_prefill.py','gpu_constraints.py','expected.json','direct_reorder.py']},
        'protocol':'Sequential folded inference, shared physical prefix, full-vocabulary double normalization, beam50, all five FP32 scores. Explicit precision selected before model construction/capture. Setup excluded; per-request staging/generation/reorder included. Pilot is an acceptance sample, not a new population quality or full500 latency claim.'}
    b.save(out/'manifest.json',manifest)
    compiler=CompiledForward(model,out,b.announce)
    from .shared_prefix import PrefixArena
    from .prefix_replay import PrefixReplayPool, ReorderSuffix
    maximum=max(u['prompt_tokens'] for u in users)
    arena=PrefixArena(maximum)
    pool=PrefixReplayPool(compiler.compiled,max_graphs=40 if args.full_cohort else 8,arena=arena)
    direct=ReorderSuffix(pool)

    @torch.inference_mode()
    def generate(user,replay,direct_reorder=False):
        calls=0
        @functools.wraps(compiler.eager)
        def forward(*a,**kw):
            nonlocal calls
            assert not a
            step=calls;calls+=1
            ids,cache=kw['input_ids'],kw['past_key_values']
            assert ids.shape==((1,user['prompt_tokens']) if step==0 else (50,1))
            assert cache.get_seq_length()==(0 if step==0 else user['prompt_tokens']+step-1)
            kw['attention_mask']={'full_attention':None}
            result=pool(**kw) if replay and step else compiler.compiled(**kw)
            if result.logits.dtype != dtype or any(x.dtype != dtype for layer in result.past_key_values.layers for x in (layer.keys,layer.values)):
                raise RuntimeError('Model logits/KV precision changed')
            return result
        cfg=b.GenerationConfig(max_new_tokens=8,length_penalty=0.,num_beams=50,num_return_sequences=50,
            pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id,bos_token_id=tok.bos_token_id,
            do_sample=False,top_k=None,top_p=None,use_cache=True)
        with patch.object(model,'forward',forward),(PrefillPrefix(model,50,cache_expander=arena.start_request) if replay else b.PrefillOnce(model,50)), \
                (direct.install(model) if direct_reorder else contextlib.nullcontext()):
            output=model.generate(inputs[user['cohort_index']],attention_mask=masks[user['cohort_index']],
                generation_config=cfg,return_dict_in_generate=True,output_scores=True,
                logits_processor=b.LogitsProcessorList([GpuSidConstrainedLogitsProcessor(tables)]))
        assert calls==5
        return output

    def unpack(output,user):
        validate_output(torch,output,args.precision,[entry[2].logits.untyped_storage().data_ptr() for entry in pool.entries.values()])
        return {'beam_indices':output.beam_indices.cpu().tolist(), 'score_dtypes':[str(s.dtype) for s in output.scores], 'token_ids':output.sequences[:,user['prompt_tokens']:].cpu().tolist(),
            'sequence_logp':output.sequences_scores.float().cpu().tolist(),
            'transition_logp':model.compute_transition_scores(output.sequences,output.scores,output.beam_indices,normalize_logits=False).float().cpu().tolist()}

    def memory():
        return {'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved(),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}

    def save_scores(output, name):
        if not args.save_full_scores:
            return
        target=out/'full_scores';target.mkdir(exist_ok=True)
        metadata=[]
        for i,score in enumerate(output.scores):
            data=score.detach().contiguous().cpu().numpy().tobytes()
            filename=f'{name}_{i}.f32.gz'
            with (target/filename).open('xb') as stream:
                with gzip.GzipFile(fileobj=stream,mode='wb',mtime=0) as compressed:
                    compressed.write(data)
            metadata.append({'file':filename,'shape':list(score.shape),'dtype':'float32','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        b.save(target/f'{name}.json',metadata)

    def tensor_digest(tensor):
        return hashlib.sha256(tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

    def immutable_inventory():
        tensors={f'parameter:{n}':p for n,p in model.named_parameters()}
        tensors.update({f'input:{i}':t for i,t in inputs.items()})
        tensors.update({f'mask:{i}':t for i,t in masks.items()})
        def visit(name,value):
            if isinstance(value,torch.Tensor):tensors[name]=value
            elif isinstance(value,dict):
                for k,v in value.items():visit(f'{name}.{k}',v)
        for name,value in vars(tables).items():visit('table:'+name,value)
        return {n:{'sha256':tensor_digest(t),'shape':list(t.shape),'dtype':str(t.dtype)} for n,t in tensors.items()}

    immutable_before=immutable_inventory()
    b.save(out/'immutable-before.json',immutable_before)
    parameter_versions={name:(id(p),p._version,str(p.dtype),p.data_ptr()) for name,p in model.named_parameters()}
    input_copies={i:t.clone() for i,t in inputs.items()}
    setup_started=time.perf_counter()
    warm_outputs={}
    (out/'warmup_outputs').mkdir()
    for user in warm_users:
        for mode in ('dense_forward',variant):
            torch.cuda.synchronize();started=time.perf_counter()
            output=generate(user,mode!='dense_forward',mode!='dense_forward')
            torch.cuda.synchronize();elapsed=time.perf_counter()-started
            raw=unpack(output,user);del output
            key=user['cohort_index']
            b.save(out/'warmup_outputs'/f'{key}_{mode}.json',raw)
            if mode=='dense_forward':warm_outputs[key]=raw
            # Cross-variant exactness is audited after preserving all pilot outputs.
            # Never turn an FP32 mismatch into a tolerance-based pass.
            compiler.add_warmup({'variant':mode,'elapsed_s':elapsed,'prompt_tokens':user['prompt_tokens']})
    for user in reversed(warm_users):
        output=generate(user,True,True)
        raw=unpack(output,user);save_scores(output,f'warm_{user["cohort_index"]}');del output
        key=user['cohort_index']
        b.save(out/'warmup_outputs'/f'{key}_repeat.json',raw)
        previous=json.loads((out/'warmup_outputs'/f'{key}_{variant}.json').read_text())
        assert raw==previous, 'Mixed-length candidate instability'
        warm_outputs[key]=raw
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated()<40*1024**3
    assert torch.cuda.memory_reserved()<46*1024**3
    pool.freeze();compiler.freeze()
    b.save(out/'setup.json',{'combined_setup_and_warmup_s':time.perf_counter()-setup_started,
        'compiler_artifacts_after':artifact_inventory(),
        'replay':pool.report(),'direct_reorder':direct.report(),
        'warmup_cohort_indices':[u['cohort_index'] for u in warm_users],'memory':memory(),
        'note':'Isolated process residency. Setup includes nested compiler and capture timings; do not add. Peak measurement counters reset after this record.'})
    torch.cuda.reset_peak_memory_stats()
    rows=[]
    trials=[(0,u) for u in users] if args.full_cohort else [
        (repeat,u) for repeat in range(args.repeats) for u in (users if repeat%2==0 else list(reversed(users)))]
    retained=[]
    for repeat,user in trials:
        torch.cuda.synchronize();started=time.perf_counter()
        output=generate(user,True,True)
        torch.cuda.synchronize();elapsed=time.perf_counter()-started
        raw=unpack(output,user)
        save_scores(output,f'{repeat}_{user["cohort_index"]:04d}')
        if args.save_full_scores:
            for old,copies in retained:
                if any(not torch.equal(x.view(torch.int32),y.view(torch.int32)) for x,y in zip(old,copies)):
                    raise RuntimeError('Previously returned full scores changed')
            retained.append((output.scores,tuple(s.clone() for s in output.scores)))
        del output
        row={'repeat':repeat,'cohort_index':user['cohort_index'],'prompt_tokens':user['prompt_tokens'],
             'variant':variant,'run':{'elapsed_s':elapsed,**raw}}
        rows.append(row)
        b.save(out/'users'/f"{repeat}_{user['cohort_index']:04d}.json",row)
        if user['cohort_index'] in warm_outputs:
            assert raw==warm_outputs[user['cohort_index']], 'Replay drifted'
        b.announce({'stage':'measured','variant':variant,'repeat':repeat,'cohort_index':user['cohort_index']})
    b.save(out/'requests.json',rows)
    compiler.finish()
    assert parameter_versions=={name:(id(p),p._version,str(p.dtype),p.data_ptr()) for name,p in model.named_parameters()}, 'Model parameters changed'
    assert all(torch.equal(inputs[i],t) for i,t in input_copies.items()), 'Caller inputs changed'
    immutable_after=immutable_inventory()
    b.save(out/'immutable-after.json',immutable_after)
    assert immutable_before==immutable_after, 'Parameter/input/mask/table values changed'
    assert policy_state(torch)==active_policy, 'Numerical policy drifted'
    summary={'complete':True,'precision':args.precision,'experimental':args.precision!='fp32','parameters_and_inputs_unchanged':True,'retained_score_tensors_checked':sum(len(x) for x,_ in retained),'measured_requests':len(rows),'latency':b.distribution([r['run']['elapsed_s'] for r in rows]),
        'replay':pool.report(),'memory':memory(),'direct_reorder':direct.report()}
    b.save(out/'summary.json',summary);b.announce(summary)


if __name__=='__main__':main()
