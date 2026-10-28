"""Isolated fixed-prompt-cache TF32 ABBA pilot; original HF scoring."""
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
from .precision_policy import precision_policy, validate_runtime, policy_state, validate_state
from .precision_log_staging import capture_session
from .precision_provenance import verify_release
from .prompt_cache_protocol import PHASES, PROTOCOL, load_protocol, phase_record
from .prompt_cache import FixedPromptCache, PromptIdentity
from .prompt_cache_generation import PromptCacheGenerator
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
    ap.add_argument('--phase',type=int,choices=range(4),required=True)
    ap.add_argument('--reuse-compiler-artifacts',action='store_true')
    ap.add_argument('--api-log-local-root',type=Path,required=True)
    args=ap.parse_args()
    args.full_cohort=False
    protocol=load_protocol(b.ROOT.parent/PROTOCOL)
    assert Path(os.environ['RAPIDTINYREC_LOG_STAGE_WRAPPER']).resolve()==args.api_log_local_root.resolve()
    variant=PHASES[args.phase]
    validate_runtime(torch, b.importlib.metadata.version('transformers'))
    with precision_policy(torch, 'folded_tf32') as numerical_policy, fold_decode_attention(True):
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
        release_verification=verify_release(args.release_root,expected['release_sha256'],b.digest)
        quality_pins=json.loads((b.ROOT/'quality_resources.json').read_text())
        verify_release(args.release_root,quality_pins,b.digest)
        users,warm_users=select_users(reference['users'],args.full_cohort)
        phase_protocol=phase_record(protocol,args.phase,args.reuse_compiler_artifacts,users) if protocol is not None else None
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
        model_dtype=torch.float32
        model_started=time.perf_counter()
        model=b.AutoModelForCausalLM.from_pretrained(args.release_root/'Office_ckpt',torch_dtype=model_dtype,
            local_files_only=True,attn_implementation='sdpa').to('cuda').eval()
        torch.cuda.synchronize()
        model_load_s=time.perf_counter()-model_started
        assert model.config.model_type=='qwen2' and not model.model.has_sliding_layers
        assert set(model.config.layer_types)=={'full_attention'}
        model.config.pad_token_id=tok.eos_token_id;model.config.eos_token_id=tok.eos_token_id;model.config.bos_token_id=tok.bos_token_id
        assert all(p.dtype==model_dtype for p in model.parameters())
        inputs={u['cohort_index']:torch.tensor([u['prompt_token_ids']],device='cuda') for u in users}
        masks={i:torch.ones_like(x) for i,x in inputs.items()}
        manifest={'experiment':'fixed_prompt_cache_tf32_pilot',
            'phase':args.phase,'variant':variant,'half':half,
            'release_verification':release_verification,'phase_protocol':phase_protocol,
            'api_log_staging':args.api_log_local_root is not None,
            'attention_layout':'folded_query_heads',
            'compiler_artifacts':{'reused':args.reuse_compiler_artifacts,'before':artifacts_before},
            'users':users,'dtype':str(model_dtype).removeprefix('torch.'),'tf32':True,'precision_policy':numerical_policy,'compiler_policy':compiler_policy,
            'torch_git':torch.version.git_version,'cuda_version':torch.version.cuda,'job_id':os.environ['SLURM_JOB_ID'],
            'model_load_s':model_load_s,'model_parameter_logical_bytes':sum(x.numel()*x.element_size() for x in model.parameters()),
            'model_parameter_unique_bytes':sum({x.untyped_storage().data_ptr():x.untyped_storage().nbytes() for x in model.parameters()}.values()),'hardware':torch.cuda.get_device_name(0),
            'cohort_sha256':expected['users_sha256'],'reference_manifest_sha256':reference['manifest_sha256'],
            'versions':{n:b.importlib.metadata.version(n) for n in ['torch','transformers','tokenizers']},
            'sha256':{n:b.digest(b.ROOT/n) for n in ['benchmark_prompt_cache.py','prompt_cache_protocol.py','prompt_cache.py','prompt_cache_generation.py','precision_policy.py','precision_provenance.py','precision_dispatch.py','precision_log_staging.py','precision_quality_adapter.py','quality_frontier.py','quality_resources.json','folded_attention.py','shared_prefix.py','prefix_replay.py','prefix_prefill.py','input_arena.py','decode_replay.py',
                'compiled_forward.py','shared_prefill.py','gpu_constraints.py','expected.json','direct_reorder.py']},
            'protocol':protocol['policy']}
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

        candidate = variant == 'fixed69_tf32'
        prompt_cache = None
        cached_generators = {}
        prepared_cached = {}
        prefix_snapshots = []
        generation_cfg=b.GenerationConfig(max_new_tokens=8,length_penalty=0.,num_beams=50,num_return_sequences=50,
            pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id,bos_token_id=tok.bos_token_id,
            do_sample=False,top_k=None,top_p=None,use_cache=True)

        def save_prefix_snapshot(stage):
            if prompt_cache is None:
                return
            # Untimed audit-only CPU copies. Preserve actual tensors so an
            # allocated auditor can independently compare bit patterns.
            tensors=tuple(tuple(x.cpu() for x in pair) for pair in prompt_cache.snapshot())
            name=f'prompt-cache-{stage}.pt'
            torch.save(tensors,out/name)
            prefix_snapshots.append(dict(stage=stage,file=name,sha256=b.digest(out/name),
                                         metadata=prompt_cache.report()))
            b.save(out/'prompt-cache-snapshots.json',prefix_snapshots)

        @torch.inference_mode()
        def generate(user,replay,direct_reorder=False):
            if candidate:
                assert direct_reorder == replay
                return cached_generators[replay].generate(prepared_cached[replay][user['cohort_index']],
                    b.LogitsProcessorList([GpuSidConstrainedLogitsProcessor(tables)]))
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
                assert result.logits.dtype==model_dtype,'Unexpected model logit dtype'
                assert all(x.dtype==model_dtype for layer in result.past_key_values.layers for x in (layer.keys,layer.values)), 'Unexpected KV dtype'
                return result
            with patch.object(model,'forward',forward),(PrefillPrefix(model,50,cache_expander=arena.start_request) if prefix_mode and replay else b.PrefillOnce(model,50)), \
                    (direct.install(model) if direct_reorder else contextlib.nullcontext()):
                output=model.generate(inputs[user['cohort_index']],attention_mask=masks[user['cohort_index']],
                    generation_config=generation_cfg,return_dict_in_generate=True,output_scores=True,
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
        with capture_session(out,args.api_log_local_root) as capture_logs:
            initial_prefix_build_s = None
            if candidate:
                token_pins={k:v for k,v in expected['release_sha256'].items()
                            if k.startswith('Office_ckpt/') and k not in
                            ('Office_ckpt/model.safetensors','Office_ckpt/config.json','Office_ckpt/generation_config.json')}
                model_pins={k:expected['release_sha256'][k] for k in
                            ('Office_ckpt/model.safetensors','Office_ckpt/config.json')}
                manifest_hash=lambda pins:hashlib.sha256(json.dumps(pins,sort_keys=True).encode()).hexdigest()
                identity=PromptIdentity(manifest_hash(model_pins),manifest_hash(token_pins),'folded_tf32')
                prefix_ids=inputs[users[0]['cohort_index']][:,:69]
                torch.cuda.synchronize();started=time.perf_counter()
                prompt_cache=FixedPromptCache(model,prefix_ids,identity,forward=compiler.compiled)
                torch.cuda.synchronize();initial_prefix_build_s=time.perf_counter()-started
                assert prompt_cache.report()['prefix_token_sha256']==protocol['prefix_token_sha256']
                save_prefix_snapshot('initial')
                for replay in (False,True):
                    generator=PromptCacheGenerator(prompt_cache,generation_cfg,arena if replay else None,
                        prefill_forward=compiler.compiled,decode_forward=pool if replay else compiler.compiled,
                        reorder=direct if replay else None)
                    cached_generators[replay]=generator
                    prepared_cached[replay]={u['cohort_index']:generator.prepare(inputs[u['cohort_index']]) for u in users}
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
            save_prefix_snapshot('after-warmup')
            pool.freeze();compiler.freeze()
        validate_state(policy_state(torch),'folded_tf32')
        b.save(out/'setup.json',{'combined_setup_and_warmup_s':time.perf_counter()-setup_started,
            'compiler_artifacts_after':artifact_inventory(),'precision_policy':policy_state(torch),
            'initial_prefix_build_including_compile_s':initial_prefix_build_s,
            'prompt_cache':prompt_cache.report() if candidate else None,
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
        validate_state(policy_state(torch),'folded_tf32')
        summary={'complete':True,'precision_policy':policy_state(torch),'measured_requests':len(rows),'latency':b.distribution([r['run']['elapsed_s'] for r in rows]),
            'replay':pool.report(),'memory':memory(),'direct_reorder':direct.report(),
            'prompt_cache':prompt_cache.report() if candidate else None}
        b.save(out/'summary.json',summary);b.announce(summary)
        # Snapshot allocations are audit overhead, not generation peak memory.
        save_prefix_snapshot('after-measurement')
        # Freeze the timing counters above before diagnostic replay/profiling.
        # One actual request per length, preserving profile outputs for equality.
        profiles=[]
        for length in sorted({u['prompt_tokens'] for u in users}):
            if length not in (min(u['prompt_tokens'] for u in users),max(u['prompt_tokens'] for u in users)):
                continue
            user=next(u for u in warm_users if u['prompt_tokens']==length)
            trace=f'profile-{length}.json'
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                with torch.profiler.record_function('frozen_prompt_cache_generation'):
                    output=generate(user,True,True)
                    torch.cuda.synchronize()
            prof.export_chrome_trace(str(out/trace))
            raw=unpack(output,user);del output
            name=f'profile-output-{length}.json';b.save(out/name,raw)
            profiles.append(dict(prompt_tokens=length,cohort_index=user['cohort_index'],trace=trace,output=name))
        save_prefix_snapshot('after-profiles')
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
