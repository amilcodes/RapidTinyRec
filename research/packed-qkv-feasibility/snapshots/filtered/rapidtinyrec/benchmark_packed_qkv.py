"""Distinct QKV phase producer. Called only by a future reviewed allocated runner.

No standalone CLI is enabled until allocated audit/source/dispatch/runner gates
are implemented. Original BF16 generate closure retains per-call config timing.
"""
import collections
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from . import benchmark as b
from .bf16_policy import precision_policy, validate_runtime, policy_state, validate_state
from .folded_attention import fold_decode_attention
from .precision_provenance import verify_release
from .precision_log_staging import capture_session
from .compiled_forward import CompiledForward
from .shared_prefix import PrefixArena
from .prefix_replay import PrefixReplayPool, ReorderSuffix
from .gpu_constraints import GpuSidConstraintTables
from .qkv_generation import make_generation
from .qkv_phase_state import PhaseState, ARMS
from .qkv_inventory_schema import pinned_config
from .qkv_measurement import schedule, unpack, check_retained
from .qkv_views import observe_views, validate_coverage
from .qkv_storage import collect_inventory, live_replay_ptrs
from .qkv_capture_pool import BoundPrefixReplayPool

torch=b.torch


def produce_phase(args, allocation, work_deadline):
    """Allocation caller must already validate source/config/scheduler identity."""
    assert allocation['submission']=='allocated_observation'
    assert allocation['job_id']==os.environ['SLURM_JOB_ID']
    assert work_deadline['job_id']==allocation['job_id'] and work_deadline['run_id']==allocation['run_id']
    assert time.monotonic()<work_deadline['deadline_monotonic']
    assert args.phase in range(4) and args.reuse_compiler_artifacts==(args.phase>=2)
    protocol_bytes=(b.ROOT.parent/'protocol.json').read_bytes();protocol=json.loads(protocol_bytes)
    assert protocol['schema']=='packed-qkv-bf16-pilot-v1'
    assert Path(os.environ['RAPIDTINYREC_LOG_STAGE_WRAPPER']).resolve()==args.api_log_local_root.resolve()
    assert torch.cuda.is_available() and torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='NVIDIA RTX A6000'
    validate_runtime(torch,b.importlib.metadata.version('transformers'))
    out=args.output_root/f'phase_{args.phase}';out.mkdir(parents=True,exist_ok=False)
    (out/'users').mkdir();(out/'warmup_outputs').mkdir()
    cache_paths={k:Path(os.environ[k]).resolve() for k in ('TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR')}
    def artifacts():
        return {k:dict(path=str(p),files=sum(x.is_file() for x in p.rglob('*')) if p.exists() else 0) for k,p in cache_paths.items()}
    initial_artifacts=artifacts();assert (sum(x['files'] for x in initial_artifacts.values())>0)==args.reuse_compiler_artifacts
    # The setup observer resolves both roots before compilation can create them.
    for path in cache_paths.values():path.mkdir(parents=True,exist_ok=True)
    expected=json.loads((b.ROOT/'expected.json').read_text());reference=json.loads((args.reference_results/'manifest.json').read_text())
    assert reference['experiment']=='gqa_three_way' and reference['dtype']=='float32'
    assert hashlib.sha256(json.dumps({k:v for k,v in reference.items() if k!='manifest_sha256'},sort_keys=True).encode()).hexdigest()==reference['manifest_sha256']
    assert hashlib.sha256(json.dumps(reference['users'],sort_keys=True).encode()).hexdigest()==expected['users_sha256']
    users=[next(u for u in reference['users'] if u['cohort_index']==i) for i in (1,2,6,10)]
    trials=list(schedule(users,args.phase//2))
    release_verification=verify_release(args.release_root,expected['release_sha256'],b.digest)
    quality_verification=verify_release(args.release_root,json.loads((b.ROOT/'quality_resources.json').read_text()),b.digest)
    config=pinned_config(b.ROOT.parent/'release-config.json',expected['release_sha256']['Office_ckpt/config.json'])
    from torch._inductor import config as inductor_config
    compiler_policy={k:getattr(inductor_config,k) for k in ('max_autotune','max_autotune_gemm','max_autotune_pointwise')};assert not any(compiler_policy.values())
    torch.manual_seed(42);torch.set_num_threads(4)
    with precision_policy(torch,'folded_bf16') as numerical_policy, fold_decode_attention(True):
        tok=b.AutoTokenizer.from_pretrained(args.release_root/'Office_ckpt',local_files_only=True)
        tok.pad_token=tok.eos_token;tok.padding_side='left';constraints=collections.defaultdict(set)
        for line in (args.release_root/'Amazon/info/Office_Products_5_2016-10-2018-11.txt').read_text().splitlines():
            ids=tok('### Response:\n'+line.split('\t')[0].strip()+'\n').input_ids+[tok.eos_token_id]
            assert all(u['prompt_token_ids'][-3:]==ids[:3] for u in users)
            for j in range(3,len(ids)):constraints[tuple(ids[:j] if j==3 else ids[3:j])].add(ids[j])
        root=users[0]['prompt_token_ids'][-3:]
        tables=GpuSidConstraintTables(constraints,root,198,tok.eos_token_id,'cuda')
        started=time.perf_counter()
        model=b.AutoModelForCausalLM.from_pretrained(args.release_root/'Office_ckpt',torch_dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa').to('cuda').eval()
        torch.cuda.synchronize();load_s=time.perf_counter()-started
        assert model.config.model_type=='qwen2' and not model.model.has_sliding_layers and set(model.config.layer_types)=={'full_attention'}
        model.config.pad_token_id=tok.eos_token_id;model.config.eos_token_id=tok.eos_token_id;model.config.bos_token_id=tok.bos_token_id
        inputs={u['cohort_index']:torch.tensor([u['prompt_token_ids']],device='cuda') for u in users};masks={i:torch.ones_like(x) for i,x in inputs.items()}
        phase=PhaseState(args.phase,model,inputs,masks,tables,config,users,constraints,root)
        with phase.install():
            if phase.packing is not None:assert phase.packing['extra_bytes']==protocol['packed_layout']['extra_parameter_copy_bytes']==176275456
            b.save(out/'immutable-original.json',phase.before_packing);b.save(out/'immutable-installed.json',phase.installed)
            b.save(out/'packing.json',dict(setup=phase.setup,proof=phase.packing))
            manifest=dict(schema='packed-qkv-phase-v1',phase=args.phase,variant=ARMS[args.phase],half=args.phase//2,
                users=users,allocation=allocation,work_deadline=work_deadline,job_id=allocation['job_id'],run_id=allocation['run_id'],
                protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(),release_verification=release_verification,quality_release_verification=quality_verification,
                cohort_sha256=expected['users_sha256'],reference_manifest_sha256=reference['manifest_sha256'],
                api_log_staging=True,hardware=torch.cuda.get_device_name(0),model_load_s=load_s,dtype='bfloat16',attention_layout='folded_query_heads',
                compiler_policy=compiler_policy,precision_policy=numerical_policy,compiler_artifacts=dict(reused=args.reuse_compiler_artifacts,before=initial_artifacts),
                versions={n:b.importlib.metadata.version(n) for n in ('torch','transformers','tokenizers')},torch_git=torch.version.git_version,cuda_version=torch.version.cuda,
                source_sha256={str(p.relative_to(b.ROOT.parent)):b.digest(p) for p in b.ROOT.glob('*.py')},timing_contract=protocol['timing'])
            b.save(out/'manifest.json',manifest)
            # Untimed actual eager requests expose packed view layouts before any
            # compiler construction. Hooks are removed before compiled warmup.
            eager=SimpleNamespace(eager=model.forward,compiled=model.forward)
            eager_arena=PrefixArena(max(u['prompt_tokens'] for u in users))
            eager_pool=PrefixReplayPool(model.forward,max_graphs=8,arena=eager_arena)
            eager_generate=make_generation(model,eager,eager_arena,eager_pool,ReorderSuffix(eager_pool),inputs,masks,tables,tok)
            with observe_views(model) as views:
                for length in (76,121):
                    user=next(u for u in users if u['prompt_tokens']==length)
                    output=eager_generate(user,False,False);torch.cuda.synchronize()
                    b.save(out/'warmup_outputs'/f"{user['cohort_index']}_eager_layout.json",unpack(model,output,user,supplied_storage_ptrs={r['storage_ptr'] for r in phase.installed.values()}));del output
            if phase.arm=='packed_qkv_bf16':validate_coverage(views,config,(76,121))
            else:assert not views
            b.save(out/'packed-eager-views.json',dict(records=views,eager_only=True,compiled_layout_verified=False))
            del eager_generate,eager_pool,eager_arena,eager
            compiler=CompiledForward(model,out,b.announce);arena=PrefixArena(max(u['prompt_tokens'] for u in users));pool=BoundPrefixReplayPool(compiler.compiled,max_graphs=8,arena=arena,cache_roots=cache_paths);direct=ReorderSuffix(pool)
            generate=make_generation(model,compiler,arena,pool,direct,inputs,masks,tables,tok)
            engine=SimpleNamespace(pool=pool,arena=arena,arm=phase.arm,inputs=inputs,masks=masks,tables=tables)
            def raw(output,user):
                return unpack(model,output,user,{entry[2].logits.untyped_storage().data_ptr() for entry in pool.entries.values()},live_replay_ptrs(pool,arena)|{r['storage_ptr'] for r in phase.installed.values()})
            def mem():
                return dict(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            setup_started=time.perf_counter();held=[]
            with pool.observe_setup(out), capture_session(out,args.api_log_local_root) as capture_logs:
                for user in users:
                    for mode in ('direct','replay'):
                        torch.cuda.synchronize();started=time.perf_counter();output=generate(user,mode=='replay',mode=='replay');torch.cuda.synchronize()
                        compiler.add_warmup(dict(variant=mode,elapsed_s=time.perf_counter()-started,prompt_tokens=user['prompt_tokens']))
                        b.save(out/'warmup_outputs'/f"{user['cohort_index']}_{mode}.json",raw(output,user))
                        if mode=='replay':held.append((user,output))
                        del output
                for user in reversed(users):
                    output=generate(user,True,True);torch.cuda.synchronize();b.save(out/'warmup_outputs'/f"{user['cohort_index']}_reverse.json",raw(output,user));del output
                check_retained([x for _,x in held])
                for user,output in held:b.save(out/'warmup_outputs'/f"{user['cohort_index']}_retained.json",raw(output,user))
                del output;held.clear();pool.freeze();compiler.freeze()
            b.save(out/'setup.json',dict(combined_setup_and_warmup_s=time.perf_counter()-setup_started,compiler_artifacts_after=artifacts(),replay=pool.report(),direct_reorder=direct.report(),memory=mem(),precision_policy=policy_state(torch)))
            b.save(out/'immutable-after-setup.json',phase.checkpoint('after_setup'))
            b.save(out/'storage-before-measurement.json',collect_inventory(engine,model,'before-measurement'));b.save(out/'capture-bindings-before-measurement.json',pool.capture_report())
            torch.cuda.reset_peak_memory_stats();rows=[]
            for repeat,user in trials:
                torch.cuda.synchronize();started=time.perf_counter();output=generate(user,True,True);torch.cuda.synchronize();elapsed=time.perf_counter()-started
                row=dict(repeat=repeat,cohort_index=user['cohort_index'],prompt_tokens=user['prompt_tokens'],variant=phase.arm,run=dict(elapsed_s=elapsed,**raw(output,user)))
                del output;rows.append(row);b.save(out/'users'/f"{repeat}_{user['cohort_index']:04d}.json",row)
            b.save(out/'requests.json',rows);compiler.finish();validate_state(policy_state(torch),'folded_bf16')
            b.save(out/'summary.json',dict(complete=True,measured_requests=len(rows),replay=pool.report(),direct_reorder=direct.report(),memory=mem(),precision_policy=policy_state(torch)))
            b.save(out/'immutable-after-measurement.json',phase.checkpoint('after_measurement'))
            b.save(out/'storage-after-measurement.json',collect_inventory(engine,model,'after-measurement'));b.save(out/'capture-bindings-after-measurement.json',pool.capture_report())
            profiles=[]
            for length in (76,121):
                user=next(u for u in users if u['prompt_tokens']==length);trace=f'profile-{length}.json'
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    with torch.profiler.record_function('frozen_packed_qkv_generation'):
                        output=generate(user,True,True);torch.cuda.synchronize()
                prof.export_chrome_trace(str(out/trace));name=f'profile-output-{length}.json';b.save(out/name,raw(output,user));del output
                profiles.append(dict(prompt_tokens=length,cohort_index=user['cohort_index'],trace=trace,output=name))
            code=[];generated=out/'generated-code';generated.mkdir()
            for name,path in cache_paths.items():
                for src in sorted(path.rglob('*.py')):
                    data=src.read_bytes();digest=hashlib.sha256(data).hexdigest();dest=generated/(digest+'.py')
                    if not dest.exists():dest.write_bytes(data)
                    code.append(dict(cache=name,relative=str(src.relative_to(path)),file=str(dest.relative_to(out)),sha256=digest))
            b.save(out/'dispatch.json',dict(capture_logs=capture_logs,profiles=profiles,generated_code=code,precision_policy=policy_state(torch),replay_after_profiles=pool.report(),direct_reorder_after_profiles=direct.report()))
            b.save(out/'immutable-after-profiles.json',phase.checkpoint('after_profiles'))
            b.save(out/'storage-after-profiles.json',collect_inventory(engine,model,'after-profiles'));b.save(out/'capture-bindings-after-profiles.json',pool.capture_report());compiler.finish();validate_state(policy_state(torch),'folded_bf16')
        assert phase.complete and time.monotonic()<work_deadline['deadline_monotonic']
        b.save(out/'phase-closure.json',dict(complete=True,original_identity_restored=True,ended_monotonic=time.monotonic()))

if __name__=='__main__':
    raise SystemExit('Not submission ready: allocated runner, audit and dispatch/view gates remain required.')
