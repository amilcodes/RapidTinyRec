"""Isolated quality-track ABBA phase: accepted folded FP32 versus TF32 matmul."""
import argparse
import collections
import contextlib
import functools
import hashlib
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

from . import benchmark as b
from .compiled_forward import CompiledForward
from .precision_policy import PHASES, precision_policy, validate_runtime, policy_state, validate_state
from .precision_dispatch import BlasLogs
from .prefix_prefill import PrefillPrefix
from .folded_attention import fold_decode_attention
from .decode_replay import DecodeReplayPool
from .gpu_constraints import GpuSidConstraintTables,GpuSidConstrainedLogitsProcessor

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
    ap.add_argument('--phase',type=int,choices=range(4),required=True)
    ap.add_argument('--reuse-compiler-artifacts',action='store_true',
        help='Reuse this variant\'s earlier on-disk compiler artifacts; GPU process and graphs are still fresh')
    args=ap.parse_args()
    variant=PHASES[args.phase]
    validate_runtime(torch, b.importlib.metadata.version('transformers'))
    with precision_policy(torch, variant) as numerical_policy, fold_decode_attention(True):
        half=0 if args.phase<2 else 1
        assert not args.reuse_compiler_artifacts or args.phase in (2,3)
        cache_paths={k:Path(os.environ[k]).resolve() for k in ('TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR')}
        def artifact_inventory():
            return {k:{'path':str(p),'files':sum(x.is_file() for x in p.rglob('*')) if p.exists() else 0}
                    for k,p in cache_paths.items()}
        artifacts_before=artifact_inventory()
        assert (sum(v['files'] for v in artifacts_before.values())>0)==args.reuse_compiler_artifacts
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        assert torch.cuda.get_device_name(0)=='NVIDIA RTX A6000'
        from torch._inductor import config as inductor_config
        compiler_policy={k:getattr(inductor_config,k) for k in ('max_autotune','max_autotune_gemm','max_autotune_pointwise')}
        assert not any(compiler_policy.values()),'No algorithm sweep'
        torch.manual_seed(42);torch.set_num_threads(4)
        expected=json.loads((b.ROOT/'expected.json').read_text())
        reference=json.loads((args.reference_results/'manifest.json').read_text())
        assert reference['experiment']=='gqa_three_way' and reference['dtype']=='float32'
        content={k:v for k,v in reference.items() if k!='manifest_sha256'}
        assert hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()==reference['manifest_sha256']
        assert hashlib.sha256(json.dumps(reference['users'],sort_keys=True).encode()).hexdigest()==expected['users_sha256']
        for name,want in expected['release_sha256'].items():assert b.digest(args.release_root/name)==want,name
        users,warm_users=select_users(reference['users'],args.full_cohort)
        out=args.output_root/f'phase_{args.phase}'
        out.mkdir(parents=True,exist_ok=True)
        assert not (out/'manifest.json').exists(),'Use a new output directory'
        (out/'users').mkdir()
        tok=b.AutoTokenizer.from_pretrained(args.release_root/'Office_ckpt',local_files_only=True)
        tok.pad_token=tok.eos_token;tok.padding_side='left'
        constraints=collections.defaultdict(set)
        for line in (args.release_root/'Amazon/info/Office_Products_5_2016-10-2018-11.txt').read_text().splitlines():
            ids=tok('### Response:\n'+line.split('\t')[0].strip()+'\n').input_ids+[tok.eos_token_id]
            assert all(u['prompt_token_ids'][-3:]==ids[:3] for u in users)
            for j in range(3,len(ids)):constraints[tuple(ids[:j] if j==3 else ids[3:j])].add(ids[j])
        tables=GpuSidConstraintTables({k:list(v) for k,v in constraints.items()},users[0]['prompt_token_ids'][-3:],198,tok.eos_token_id,'cuda')
        model_started=time.perf_counter()
        model=b.AutoModelForCausalLM.from_pretrained(args.release_root/'Office_ckpt',torch_dtype=torch.float32,
            local_files_only=True,attn_implementation='sdpa').to('cuda').eval()
        torch.cuda.synchronize()
        model_load_s=time.perf_counter()-model_started
        assert model.config.model_type=='qwen2' and not model.model.has_sliding_layers
        assert set(model.config.layer_types)=={'full_attention'}
        model.config.pad_token_id=tok.eos_token_id;model.config.eos_token_id=tok.eos_token_id;model.config.bos_token_id=tok.bos_token_id
        assert all(p.dtype==torch.float32 for p in model.parameters())
        inputs={u['cohort_index']:torch.tensor([u['prompt_token_ids']],device='cuda') for u in users}
        masks={i:torch.ones_like(x) for i,x in inputs.items()}
        manifest={'experiment':'precision_500' if args.full_cohort else 'precision_pilot',
            'phase':args.phase,'variant':variant,'half':half,
            'attention_layout':'folded_query_heads',
            'compiler_artifacts':{'reused':args.reuse_compiler_artifacts,'before':artifacts_before},
            'users':users,'dtype':'float32','tf32':variant=='folded_tf32','precision_policy':numerical_policy,'compiler_policy':compiler_policy,
            'torch_git':torch.version.git_version,'cuda_version':torch.version.cuda,'job_id':os.environ['SLURM_JOB_ID'],
            'model_load_s':model_load_s,'model_parameter_logical_bytes':sum(x.numel()*x.element_size() for x in model.parameters()),
            'model_parameter_unique_bytes':sum({x.untyped_storage().data_ptr():x.untyped_storage().nbytes() for x in model.parameters()}.values()),'hardware':torch.cuda.get_device_name(0),
            'cohort_sha256':expected['users_sha256'],'reference_manifest_sha256':reference['manifest_sha256'],
            'versions':{n:b.importlib.metadata.version(n) for n in ['torch','transformers','tokenizers']},
            'sha256':{n:b.digest(b.ROOT/n) for n in ['benchmark_precision.py','precision_policy.py','precision_dispatch.py','precision_quality_adapter.py','quality_frontier.py','quality_resources.json','folded_attention.py','shared_prefix.py','prefix_replay.py','prefix_prefill.py','input_arena.py','decode_replay.py',
                'compiled_forward.py','shared_prefill.py','gpu_constraints.py','expected.json','direct_reorder.py']},
            'protocol':'Fresh folded-FP32/TF32/TF32/FP32 ABBA. FP32 storage and outputs; candidate changes only CUDA matmul allow_tf32 before compilation/capture, cudnn remains off. Original beam50, five full-vocabulary score steps and termination. Numerical fidelity is reported separately from paired SID quality. Original shape/attention/cache/reorder unchanged. Timed generation includes staging/wrappers/HF clones; input preparation, output extraction/audits and files excluded. Setup/capture/profiling excluded; nested intervals are not additive. No quality equivalence or automatic baseline promotion.'}
        b.save(out/'manifest.json',manifest)
        compiler=CompiledForward(model,out,b.announce)
        from .input_arena import SharedInputArena
        from .direct_reorder import ReorderIntoReplay
        from .shared_prefix import PrefixArena
        from .prefix_replay import PrefixReplayPool, ReorderSuffix
        prefix_mode=True
        maximum=max(u['prompt_tokens'] for u in users)
        arena=PrefixArena(maximum) if prefix_mode else SharedInputArena(maximum+3)
        pool_class=PrefixReplayPool if prefix_mode else DecodeReplayPool
        pool=pool_class(compiler.compiled,max_graphs=40 if args.full_cohort else 8,arena=arena)
        direct=ReorderSuffix(pool) if prefix_mode else ReorderIntoReplay(pool)

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
                assert result.logits.dtype==torch.float32,'Model logits must remain FP32'
                return result
            cfg=b.GenerationConfig(max_new_tokens=8,length_penalty=0.,num_beams=50,num_return_sequences=50,
                pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id,bos_token_id=tok.bos_token_id,
                do_sample=False,top_k=None,top_p=None,use_cache=True)
            with patch.object(model,'forward',forward),(PrefillPrefix(model,50,cache_expander=arena.start_request) if prefix_mode and replay else b.PrefillOnce(model,50)), \
                    (direct.install(model) if direct_reorder else contextlib.nullcontext()):
                output=model.generate(inputs[user['cohort_index']],attention_mask=masks[user['cohort_index']],
                    generation_config=cfg,return_dict_in_generate=True,output_scores=True,
                    logits_processor=b.LogitsProcessorList([GpuSidConstrainedLogitsProcessor(tables)]))
            assert calls==5
            return output

        def unpack(output,user):
            assert output.sequences_scores.dtype==torch.float32 and all(s.dtype==torch.float32 for s in output.scores)
            saved_ptrs=[s.untyped_storage().data_ptr() for s in output.scores]
            graph_ptrs={entry[2].logits.untyped_storage().data_ptr() for entry in pool.entries.values()}
            assert len(saved_ptrs)==len(set(saved_ptrs))==5 and not (set(saved_ptrs)&graph_ptrs)
            return {'graph_logit_storage_ptrs':sorted(graph_ptrs),'saved_score_storage_ptrs':saved_ptrs,'saved_score_dtypes':[str(s.dtype) for s in output.scores],
                'beam_indices':output.beam_indices.cpu().tolist(),'token_ids':output.sequences[:,user['prompt_tokens']:].cpu().tolist(),
                'sequence_logp':output.sequences_scores.float().cpu().tolist(),
                'transition_logp':model.compute_transition_scores(output.sequences,output.scores,output.beam_indices,normalize_logits=False).float().cpu().tolist()}

        def memory():
            return {'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved(),
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}

        setup_started=time.perf_counter()
        logs=BlasLogs(out)
        capture_logs=logs.begin('capture')
        # Native API logging covers actual compiled warmup/capture, never timing.
        # A logger failure is recorded and blocks the independent dispatch gate.
        warm_outputs={};retained={}
        (out/'warmup_outputs').mkdir()
        for user in warm_users:
            for mode in ('dense_forward',variant):
                torch.cuda.synchronize();started=time.perf_counter()
                output=generate(user,mode!='dense_forward',mode!='dense_forward')
                torch.cuda.synchronize();elapsed=time.perf_counter()-started
                raw=unpack(output,user)
                if mode==variant and user['prompt_tokens'] not in retained:retained[user['prompt_tokens']]=(user,output)
                del output
                key=user['cohort_index']
                b.save(out/'warmup_outputs'/f'{key}_{mode}.json',raw)
                if mode=='dense_forward':warm_outputs[key]=raw
                # Cross-policy fidelity is informational; within-policy stability stays strict.
                compiler.add_warmup({'variant':mode,'elapsed_s':elapsed,'prompt_tokens':user['prompt_tokens']})
        for user in reversed(warm_users):
            output=generate(user,True,True)
            raw=unpack(output,user);del output
            key=user['cohort_index']
            b.save(out/'warmup_outputs'/f'{key}_repeat.json',raw)
            # Preserve instability for the independent audit instead of losing later samples.
            warm_outputs[key]=raw
        for user,output in retained.values():
            b.save(out/'warmup_outputs'/f"{user['cohort_index']}_retained.json",unpack(output,user))
        del user,output
        retained.clear()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated()<40*1024**3
        assert torch.cuda.memory_reserved()<46*1024**3
        pool.freeze();compiler.freeze()
        logs.end()
        validate_state(policy_state(torch),variant)
        b.save(out/'setup.json',{'combined_setup_and_warmup_s':time.perf_counter()-setup_started,
            'compiler_artifacts_after':artifact_inventory(),'precision_policy':policy_state(torch),
            'replay':pool.report(),'direct_reorder':direct.report(),
            'warmup_cohort_indices':[u['cohort_index'] for u in warm_users],'memory':memory(),
            'note':'Isolated process residency. Setup includes nested compiler and capture timings; do not add. Peak measurement counters reset after this record.'})
        torch.cuda.reset_peak_memory_stats()
        rows=[]
        trials=[(0,u) for i,u in enumerate(users) if i%2==half] if args.full_cohort else [
            (repeat,u) for repeat in range(half*3,half*3+3) for u in users]
        for repeat,user in trials:
            torch.cuda.synchronize();started=time.perf_counter()
            output=generate(user,True,True)
            torch.cuda.synchronize();elapsed=time.perf_counter()-started
            raw=unpack(output,user);del output
            row={'repeat':repeat,'cohort_index':user['cohort_index'],'prompt_tokens':user['prompt_tokens'],
                 'variant':variant,'run':{'elapsed_s':elapsed,**raw}}
            rows.append(row)
            b.save(out/'users'/f"{repeat}_{user['cohort_index']:04d}.json",row)
            b.announce({'stage':'measured','variant':variant,'repeat':repeat,'cohort_index':user['cohort_index']})
        b.save(out/'requests.json',rows)
        compiler.finish()
        validate_state(policy_state(torch),variant)
        summary={'complete':True,'precision_policy':policy_state(torch),'measured_requests':len(rows),'latency':b.distribution([r['run']['elapsed_s'] for r in rows]),
            'replay':pool.report(),'memory':memory(),'direct_reorder':direct.report()}
        b.save(out/'summary.json',summary);b.announce(summary)
        # Freeze the timing counters above before diagnostic replay/profiling.
        # One actual request per length, preserving profile outputs for equality.
        profiles=[]
        for length in sorted({u['prompt_tokens'] for u in users}):
            if length not in (min(u['prompt_tokens'] for u in users),max(u['prompt_tokens'] for u in users)):
                continue
            user=next(u for u in warm_users if u['prompt_tokens']==length)
            trace=f'profile-{length}.json'
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                with torch.profiler.record_function('frozen_precision_generation'):
                    output=generate(user,True,True)
                    torch.cuda.synchronize()
            prof.export_chrome_trace(str(out/trace))
            raw=unpack(output,user);del output
            name=f'profile-output-{length}.json';b.save(out/name,raw)
            profiles.append(dict(prompt_tokens=length,cohort_index=user['cohort_index'],trace=trace,output=name))
        # Preserve generated wrappers and kernels (not binary compiler caches).
        generated=out/'generated-code';generated.mkdir()
        code=[]
        for cache_name,cache_path in cache_paths.items():
            for src in sorted(cache_path.rglob('*.py')):
                data=src.read_bytes();sha=hashlib.sha256(data).hexdigest();dest=generated/(sha+'.py')
                if not dest.exists():dest.write_bytes(data)
                code.append(dict(cache=cache_name,relative=str(src.relative_to(cache_path)),file=str(dest.relative_to(out)),sha256=sha))
        b.save(out/'dispatch.json',dict(capture_logs=capture_logs,profiles=profiles,generated_code=code,
            precision_policy=policy_state(torch),replay_after_profiles=pool.report(),direct_reorder_after_profiles=direct.report(),
            timing_boundary='API logs disabled before freeze/measurement. Profiles/code extraction after primary timing.'))
        compiler.finish()



if __name__=='__main__':main()
