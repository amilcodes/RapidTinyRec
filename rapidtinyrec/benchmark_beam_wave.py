"""Distinct allocated serial-pair versus two-request wave TF32 pilot phase."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import time
from .beam_wave_protocol import PROTOCOL,PROTOCOL_SHA256,PHASES,load_protocol,phase_schedule


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('release-root','reference-results','output-root','api-log-local-root'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--phase',type=int,choices=range(4),required=True)
    ap.add_argument('--reuse-compiler-artifacts',action='store_true')
    args=ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('OSCAR_RUN_DIR'):
        raise RuntimeError('Beam-wave benchmark requires its actual allocated job')
    from .beam_wave_allocation import observe_allocation
    run_root=Path(os.environ['OSCAR_RUN_DIR']).resolve()
    code=Path(__file__).resolve().parents[1]
    allocation=observe_allocation(run_root,code,args.output_root)
    protocol=load_protocol(code/PROTOCOL)
    work=json.loads((args.output_root/'wave-work-start.json').read_text())
    if (work['schema']!='two-request-beam-wave-work-v1' or work['job_id']!=allocation['job_id']
            or work['run_id']!=allocation['run_id'] or work['protocol_sha256']!=PROTOCOL_SHA256
            or work['deadline_monotonic']-work['started_monotonic']!=1620
            or not work['started_monotonic']<=time.monotonic()<work['deadline_monotonic']):
        raise ValueError('Invalid original wave work deadline')
    if Path(os.environ.get('RAPIDTINYREC_LOG_STAGE_WRAPPER','')).resolve()!=args.api_log_local_root.resolve():
        raise ValueError('Allocated log preservation wrapper required')
    if args.reuse_compiler_artifacts!=(args.phase>=2):raise ValueError('Fresh ABBA disk-reuse contract differs')
    # All heavyweight imports follow actual allocation/source/deadline checks.
    from . import benchmark as b
    from .compiled_forward import CompiledForward
    from .precision_policy import precision_policy,validate_runtime,policy_state,validate_state
    from .precision_log_staging import capture_session
    from .precision_provenance import verify_release
    from .folded_attention import fold_decode_attention
    from .gpu_constraints import GpuSidConstraintTables
    from .beam_wave_generation import PairGeneration
    from .beam_wave_measurement import measure_pair
    from .beam_wave_outputs import unpack_requests
    from .beam_wave_storage import collect_inventory
    torch=b.torch
    validate_runtime(torch,b.importlib.metadata.version('transformers'))
    with precision_policy(torch,'folded_tf32') as numerical_policy,fold_decode_attention(True):
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        assert torch.cuda.get_device_name(0)=='NVIDIA RTX A6000'
        from torch._inductor import config as inductor_config
        compiler_policy={k:getattr(inductor_config,k) for k in ('max_autotune','max_autotune_gemm','max_autotune_pointwise')}
        assert not any(compiler_policy.values())
        torch.manual_seed(42);torch.set_num_threads(4)
        caches={k:Path(os.environ[k]).resolve() for k in ('TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR')}
        def inventory():return {k:dict(path=str(p),files=sum(x.is_file() for x in p.rglob('*'))) for k,p in caches.items()}
        before=inventory();assert (sum(x['files'] for x in before.values())>0)==args.reuse_compiler_artifacts
        expected=json.loads((b.ROOT/'expected.json').read_text())
        reference=json.loads((args.reference_results/'manifest.json').read_text())
        assert reference['experiment']=='gqa_three_way' and reference['dtype']=='float32'
        content={k:v for k,v in reference.items() if k!='manifest_sha256'}
        assert hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()==reference['manifest_sha256']
        assert hashlib.sha256(json.dumps(reference['users'],sort_keys=True).encode()).hexdigest()==expected['users_sha256']==protocol['release_users_sha256']
        release=verify_release(args.release_root,expected['release_sha256'],b.digest)
        quality_release=verify_release(args.release_root,json.loads((b.ROOT/'quality_resources.json').read_text()),b.digest)
        by_user={u['cohort_index']:u for u in reference['users']}
        users=[by_user[i] for i in (1,2,6,10)]
        assert [u['prompt_tokens'] for u in users]==[76,76,121,121]
        out=args.output_root/f'phase_{args.phase}';out.mkdir(exist_ok=False)
        for directory in ('trials','warmup_outputs'): (out/directory).mkdir()
        tok=b.AutoTokenizer.from_pretrained(args.release_root/'Office_ckpt',local_files_only=True)
        tok.pad_token=tok.eos_token;tok.padding_side='left';constraints=collections.defaultdict(set)
        for line in (args.release_root/'Amazon/info/Office_Products_5_2016-10-2018-11.txt').read_text().splitlines():
            ids=tok('### Response:\n'+line.split('\t')[0].strip()+'\n').input_ids+[tok.eos_token_id]
            assert all(u['prompt_token_ids'][-3:]==ids[:3] for u in users)
            for j in range(3,len(ids)):constraints[tuple(ids[:j] if j==3 else ids[3:j])].add(ids[j])
        tables=GpuSidConstraintTables({k:list(v) for k,v in constraints.items()},users[0]['prompt_token_ids'][-3:],198,tok.eos_token_id,'cuda')
        started=time.perf_counter()
        model=b.AutoModelForCausalLM.from_pretrained(args.release_root/'Office_ckpt',torch_dtype=torch.float32,
            local_files_only=True,attn_implementation='sdpa').to('cuda').eval()
        torch.cuda.synchronize();load_s=time.perf_counter()-started
        assert model.config.model_type=='qwen2' and not model.model.has_sliding_layers
        assert set(model.config.layer_types)=={'full_attention'}
        model.config.pad_token_id=tok.eos_token_id;model.config.eos_token_id=tok.eos_token_id;model.config.bos_token_id=tok.bos_token_id
        variant=PHASES[args.phase]
        cfg=b.GenerationConfig(max_new_tokens=8,length_penalty=0.,num_beams=50,num_return_sequences=50,
            pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id,bos_token_id=tok.bos_token_id,
            do_sample=False,top_k=None,top_p=None,use_cache=True)
        compiler=CompiledForward(model,out,b.announce);engine=PairGeneration(model,compiler,cfg,tables,variant,121)
        inputs={u['cohort_index']:torch.tensor([u['prompt_token_ids']],device='cuda') for u in users}
        groups=([pair['cohort_indices'] for pair in protocol['pairs']] if engine.wave else [[u['cohort_index']] for u in users])
        for group in groups:
            engine.prepare_inputs(group,[inputs[i] for i in group])
            if engine.wave:engine.prepare_inputs(group[::-1],[inputs[i] for i in group[::-1]])
        manifest=dict(experiment=protocol['experiment'],schema='two-request-beam-wave-phase-v1',phase=args.phase,
            phase_protocol=protocol['phases'][args.phase],protocol_sha256=PROTOCOL_SHA256,variant=variant,half=args.phase//2,
            users=users,cohort_sha256=expected['users_sha256'],reference_manifest_sha256=reference['manifest_sha256'],
            release_verification=release,quality_release_verification=quality_release,job_id=allocation['job_id'],run_id=allocation['run_id'],
            allocation=allocation,work_deadline=work,api_log_staging=True,attention_layout='folded_query_heads',dtype='float32',tf32=True,
            precision_policy=numerical_policy,compiler_policy=compiler_policy,compiler_artifacts=dict(reused=args.reuse_compiler_artifacts,before=before),
            versions={n:b.importlib.metadata.version(n) for n in ('torch','transformers','tokenizers')},torch_git=torch.version.git_version,
            cuda_version=torch.version.cuda,hardware=torch.cuda.get_device_name(0),model_load_s=load_s,vocab_size=model.config.vocab_size,
            kv_layout=dict(layers=model.config.num_hidden_layers,kv_heads=model.config.num_key_value_heads,
                           head_dim=model.config.hidden_size//model.config.num_attention_heads),
            model_parameter_logical_bytes=sum(x.numel()*x.element_size() for x in model.parameters()),
            model_parameter_unique_bytes=sum({x.untyped_storage().data_ptr():x.untyped_storage().nbytes() for x in model.parameters()}.values()),
            source_sha256={p.name:b.digest(p) for p in b.ROOT.glob('*.py')},timing_contract=protocol['timing'])
        b.save(out/'manifest.json',manifest)
        def unpack(output,group):
            return unpack_requests(model,output,group,by_user[group[0]]['prompt_tokens'],engine.graph_logit_storage_ptrs())
        def memory():return dict(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        setup=time.perf_counter()
        with capture_session(out,args.api_log_local_root) as capture_logs:
            retained=[]
            for group in groups:
                for mode in ('direct','replay'):
                    torch.cuda.synchronize();start=time.perf_counter();output=engine.generate(group,replay=mode=='replay')
                    torch.cuda.synchronize();elapsed=time.perf_counter()-start
                    b.save(out/'warmup_outputs'/('_'.join(map(str,group))+'_'+mode+'.json'),dict(users=group,requests=unpack(output,group)))
                    if mode=='replay':retained.append((group,output))
                    compiler.add_warmup(dict(variant=mode,elapsed_s=elapsed,prompt_tokens=by_user[group[0]]['prompt_tokens']))
                    del output
            for group in reversed(groups):
                ordered=group[::-1] if engine.wave else group
                output=engine.generate(ordered)
                b.save(out/'warmup_outputs'/('_'.join(map(str,group))+'_reverse.json'),dict(users=ordered,requests=unpack(output,ordered)))
                del output
            for group,output in retained:
                b.save(out/'warmup_outputs'/('_'.join(map(str,group))+'_retained.json'),dict(users=group,requests=unpack(output,group)))
            del output;retained.clear();torch.cuda.synchronize()
            assert torch.cuda.memory_allocated()<40*1024**3 and torch.cuda.memory_reserved()<46*1024**3
            engine.pool.freeze();compiler.freeze()
        validate_state(policy_state(torch),'folded_tf32')
        b.save(out/'setup.json',dict(combined_setup_and_warmup_s=time.perf_counter()-setup,compiler_artifacts_after=inventory(),
            precision_policy=policy_state(torch),replay=engine.pool.report(),direct_reorder=engine.reorder.report(),memory=memory()))
        b.save(out/'storage-before-measurement.json',collect_inventory(engine,model,'before-measurement'))
        torch.cuda.reset_peak_memory_stats();trials=[]
        for spec in phase_schedule(args.phase):
            trial=measure_pair(spec,engine.generate,torch.cuda.synchronize,time.perf_counter,unpack)
            trials.append(trial);b.save(out/'trials'/f"{spec['repeat']}_{spec['pair_index']}.json",trial)
            b.announce(dict(stage='measured_pair',phase=args.phase,repeat=spec['repeat'],pair_index=spec['pair_index']))
        b.save(out/'trials.json',trials);compiler.finish();validate_state(policy_state(torch),'folded_tf32')
        b.save(out/'summary.json',dict(complete=True,measured_pairs=6,measured_requests=12,precision_policy=policy_state(torch),
            replay=engine.pool.report(),direct_reorder=engine.reorder.report(),memory=memory()))
        b.save(out/'storage-after-measurement.json',collect_inventory(engine,model,'after-measurement'))
        profiles=[]
        for pair in protocol['pairs']:
            group=pair['cohort_indices'];length=pair['prompt_tokens'];outputs=[]
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                with torch.profiler.record_function('frozen_beam_wave_generation'):
                    if engine.wave:outputs.append((group,engine.generate(group)))
                    else:
                        for user in group:
                            outputs.append(([user],engine.generate([user])));torch.cuda.synchronize()
                    torch.cuda.synchronize()
            trace=f'profile-{length}.json';prof.export_chrome_trace(str(out/trace))
            raw=[]
            for g,output in outputs:raw.extend(unpack(output,g))
            outputs.clear();del output
            name=f'profile-output-{length}.json';b.save(out/name,dict(users=group,requests=raw))
            profiles.append(dict(prompt_tokens=length,cohort_indices=group,trace=trace,output=name))
        b.save(out/'storage-after-profiles.json',collect_inventory(engine,model,'after-profiles'))
        generated=out/'generated-code';generated.mkdir();code_records=[]
        for cache_name,cache_path in caches.items():
            for src in sorted(cache_path.rglob('*.py')):
                data=src.read_bytes();sha=hashlib.sha256(data).hexdigest();dest=generated/(sha+'.py')
                if not dest.exists():dest.write_bytes(data)
                code_records.append(dict(cache=cache_name,relative=str(src.relative_to(cache_path)),file=str(dest.relative_to(out)),sha256=sha))
        b.save(out/'dispatch.json',dict(capture_logs=capture_logs,profiles=profiles,generated_code=code_records,
            precision_policy=policy_state(torch),replay_after_profiles=engine.pool.report(),direct_reorder_after_profiles=engine.reorder.report()))
        compiler.finish()


if __name__=='__main__':main()
