"""Independent auditor for the distinct uncached/fixed69 TF32 pilot.

Raw phases retain their declared identities. Live source/allocation identity,
global and audit-stage deadlines, and log recovery are checked before numerical
evaluation. This does not relabel BF16, serving or full500 phases.
"""
import json
import math
import os
import argparse
import time
import traceback
from pathlib import Path
from .audit_decode_pilot import read,digest,json_digest,compare,latency
from .audit_prefix_profile import exact
from .audit_precision import validate_run,validate_output_storage,validate_log_staging,TORCH_COMMIT
from .audit_precision_dispatch import dispatch_evidence
from .precision_policy import validate_state
from .prompt_cache_protocol import PROTOCOL,PROTOCOL_SHA256,load_protocol,phase_record,validate_request_inventory,PHASES
from .audit_prompt_cache_storage import audit_snapshots,validate_request_storage
from .precision_quality_adapter import quality_inputs
from .quality_frontier import evaluate
VARIANTS=('uncached_tf32','fixed69_tf32')


def validate_phase_header(manifest,phase,record,protocol):
    assert manifest['experiment']=='fixed_prompt_cache_tf32_pilot'
    assert manifest['phase']==phase and manifest['half']==phase//2 and manifest['variant']==PHASES[phase]
    assert str(manifest['job_id'])==str(record['job_id'])
    assert manifest['dtype']=='float32' and manifest['tf32'] is True
    validate_state(manifest['precision_policy'],'folded_tf32')
    assert manifest['phase_protocol']==phase_record(protocol,phase,manifest['compiler_artifacts']['reused'],manifest['users'])


def inspect_phases(root,historical,snapshot,run_record,*,setup_diagnostic=False,staged_pilot=False,full500_quality=False):
    assert not setup_diagnostic and not staged_pilot and not full500_quality
    assert os.environ.get('SLURM_JOB_ID'),'Allocated prompt-cache audit only'
    assert {p.name for p in root.glob('phase_*') if p.is_dir()}=={f'phase_{i}' for i in range(4)}
    record=read(run_record)
    assert record['resources']['gpus']==1 and record['resources']['gpu_type']=='nvidia_rtx_a6000'
    assert record['resources']['cpus']==4 and record['resources']['memory_gb']==32
    assert record['resources']['partition']=='gpu' and record['resources']['qos']=='norm-gpu'
    assert record['resources']['time']=='00:30:00'
    for name,sha in record['sha256'].items():assert digest(snapshot.parent/name)==sha,name
    assert len(record['sha256'])==len(record['files'])
    assert record['project']=='rapidtinyrec-prompt-cache-pilot' and record['profile']=='fixed-prompt-cache-pilot'
    expected=read(snapshot/'expected.json')
    assert expected==read(Path(__file__).with_name('expected.json'))
    reference=read(historical/'manifest.json')
    assert reference['experiment']=='gqa_three_way' and reference['dtype']=='float32'
    assert json_digest({k:v for k,v in reference.items() if k!='manifest_sha256'})==reference['manifest_sha256']
    assert len(reference['users'])==500 and json_digest(reference['users'])==expected['users_sha256']
    all_lengths=sorted({u['prompt_tokens'] for u in reference['users']})
    phases=[]; joined={}; historical_pairs={v:[] for v in VARIANTS}; seen={}; artifact_paths={}
    full=False; repeated=[]; ancestry_stable=True
    schedule=list(enumerate(PHASES))
    for phase,variant in schedule:
        path=root/f'phase_{phase}'
        m,s,w,c=(read(path/n) for n in ('manifest.json','summary.json','setup.json','compilation.json'))
        validate_log_staging(path,m)
        rows=read(path/'requests.json'); half=0 if phase<2 else 1
        validate_phase_header(m,phase,record,load_protocol(snapshot.parent/PROTOCOL))
        assert m['phase']==phase and m['half']==half and m['variant']==variant
        assert m['attention_layout']=='folded_query_heads'
        artifacts=m['compiler_artifacts']
        assert artifacts['reused']==(phase in (2,3))
        assert set(artifacts['before'])==set(w['compiler_artifacts_after'])=={'TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR'}
        paths={k:v['path'] for k,v in artifacts['before'].items()}
        assert paths=={k:v['path'] for k,v in w['compiler_artifacts_after'].items()}
        if phase<2:
            assert not any(v['files'] for v in artifacts['before'].values())
            artifact_paths[variant]=paths
        else:
            assert paths==artifact_paths[variant]
            assert sum(v['files'] for v in artifacts['before'].values())>0
        assert sum(v['files'] for v in w['compiler_artifacts_after'].values())>0
        assert s['complete'] and m['hardware']=='NVIDIA RTX A6000' and m['dtype']=='float32' and m['tf32'] is True
        assert m['compiler_policy']==dict(max_autotune=False,max_autotune_gemm=False,max_autotune_pointwise=False)
        assert m['torch_git']==TORCH_COMMIT and m['cuda_version']=='12.6'
        assert str(m['job_id'])==str(record['job_id'])
        assert m['model_load_s']>0 and 0<m['model_parameter_unique_bytes']<=m['model_parameter_logical_bytes']
        for policy in (m['precision_policy'],w['precision_policy'],s['precision_policy']):validate_state(policy,'folded_tf32')
        protocol=load_protocol(snapshot.parent/PROTOCOL)
        assert m['phase_protocol']==phase_record(protocol,phase,m['compiler_artifacts']['reused'],m['users'])
        assert m['versions']=={'torch':'2.14.0+cu126','transformers':'4.55.4','tokenizers':'0.21.4'}
        assert m['cohort_sha256']==expected['users_sha256'] and m['reference_manifest_sha256']==reference['manifest_sha256']
        names={'benchmark_prompt_cache.py','prompt_cache_protocol.py','prompt_cache.py','prompt_cache_generation.py','precision_policy.py','precision_dispatch.py','precision_log_staging.py','precision_quality_adapter.py','quality_frontier.py','quality_resources.json','folded_attention.py','shared_prefix.py','prefix_replay.py','prefix_prefill.py','input_arena.py','decode_replay.py','compiled_forward.py',
               'shared_prefill.py','gpu_constraints.py','expected.json','direct_reorder.py'}
        if 'release_verification' in m:
            from .precision_provenance import validate_release
            names.add('precision_provenance.py')
            validate_release(m['release_verification'],expected['release_sha256'])
        assert set(m['sha256'])==names
        for name,sha in m['sha256'].items():assert digest(snapshot/name)==sha,name
        users=reference['users'] if full else [u for n in (all_lengths[0],all_lengths[-1])
            for u in [x for x in reference['users'] if x['prompt_tokens']==n][:2]]
        assert m['users']==users
        lengths=all_lengths if full else [all_lengths[0],all_lengths[-1]]
        warm=[next(u['cohort_index'] for u in users if u['prompt_tokens']==n) for n in lengths] if full else [u['cohort_index'] for u in users]
        assert w['warmup_cohort_indices']==warm
        trials=[(0,u) for i,u in enumerate(users) if i%2==half] if full else [(r,u) for r in range(half*3,half*3+3) for u in users]
        assert [(r['repeat'],r['cohort_index']) for r in rows]==[(i,u['cohort_index']) for i,u in trials]
        validate_request_inventory(protocol,phase,rows)
        assert len(rows)==s['measured_requests']==12
        assert len(list((path/'users').glob('*.json')))==len(rows)
        assert c['phase']=='complete' and c['measurement_compilations']==0
        assert c['events'] and all(e.get('success') or e.get('expected_restart') for e in c['events'])
        assert not c['compiler_counters'].get('graph_break') and not c['compiler_counters'].get('unimplemented')
        before,after=w['replay'],s['replay']; graphs=4*len(lengths)
        assert before['graphs']==after['graphs']==graphs and before['frozen'] and after['frozen']
        assert before['events']==after['events'] and len(after['events'])==graphs
        assert sorted(e['cache_tokens'] for e in after['events'])==sorted(n+j for n in lengths for j in range(4))
        # Pinned Qwen2: 28 layers, two KV heads, width128, beam50, FP32.
        bytes_per_token=28*2*50*2*128*4; metadata_bytes=101*8
        prefix_mode=True
        logical=sum((bytes_per_token//50*n+bytes_per_token*j if prefix_mode else bytes_per_token*(n+j))+metadata_bytes for n in lengths for j in range(4))
        physical=(bytes_per_token//50*max(lengths)+bytes_per_token*3+graphs*metadata_bytes) if prefix_mode else bytes_per_token*(max(lengths)+3)+graphs*metadata_bytes
        assert before['input_buffer_bytes']==after['input_buffer_bytes']==logical
        assert before['unique_input_storage_bytes']==after['unique_input_storage_bytes']==physical
        assert before['shared_input_arena']==after['shared_input_arena']==True
        assert after['replays']-before['replays']==4*len(rows)
        for e in after['events']:
            n,j=next((n,j) for n in lengths for j in range(4) if n+j==e['cache_tokens'])
            want=(bytes_per_token//50*n+bytes_per_token*j if prefix_mode else bytes_per_token*(n+j))+metadata_bytes
            assert e['batch']==50 and e['input_buffer_bytes']==want
            assert e['capture_and_warmup_s']>0
        capture=sum(e['capture_and_warmup_s'] for e in after['events'])
        assert before['capture_and_warmup_s']==after['capture_and_warmup_s']==capture
        assert w['combined_setup_and_warmup_s']>=capture>0
        first,last=w['direct_reorder'],s['direct_reorder']
        for k,n in [('direct_calls',4),('fallback_calls',1)]:assert last[k]-first[k]==n*len(rows)
        if prefix_mode:
            assert last['prefix_bytes_reordered']==first['prefix_bytes_reordered']==0
            assert last['suffix_bytes_written']-first['suffix_bytes_written']==bytes_per_token*6*len(rows)
            for r in (before,after):
                arena=r['physical_prefix']
                assert arena['physical_prefix_batch']==1 and arena['suffix_batch']==50
                assert arena['max_prefix_tokens']==max(lengths) and arena['max_suffix_tokens']==3
                assert arena['prefix_capacity_bytes']==bytes_per_token//50*max(lengths)
                assert arena['suffix_capacity_bytes']==bytes_per_token*3
                assert r['output_suffix_bytes']==bytes_per_token*10*len(lengths)
            a,z=before['physical_prefix'],after['physical_prefix']
            assert z['requests_loaded']-a['requests_loaded']==len(rows)
            assert z['prefix_bytes_copied']-a['prefix_bytes_copied']==sum(bytes_per_token//50*r['prompt_tokens'] for r in rows)
        else:
            assert last['kv_tensors_written']-first['kv_tensors_written']==224*len(rows)
            assert last['kv_bytes_written']-first['kv_bytes_written']==sum(bytes_per_token*(r['prompt_tokens']*4+6) for r in rows)
        for mem in (w['memory'],s['memory']):
            assert 0<physical<=mem['allocated_bytes']<=mem['reserved_bytes']<46*1024**3
            assert mem['allocated_bytes']<=mem['peak_allocated_bytes']<=mem['peak_reserved_bytes']
            assert mem['reserved_bytes']<=mem['peak_reserved_bytes']
        for row,(_,user) in zip(rows,trials):
            assert row['variant']==variant and row['prompt_tokens']==user['prompt_tokens']
            assert row==read(path/'users'/f"{row['repeat']}_{row['cohort_index']:04d}.json")
            run=row['run'];validate_run(run);validate_output_storage(run)
            assert len(run['graph_logit_storage_ptrs'])==graphs
            key=(row['repeat'],row['cohort_index'])
            pair=joined.setdefault(key,{'repeat':row['repeat'],'cohort_index':row['cohort_index'],'prompt_tokens':row['prompt_tokens'],'runs':{}})
            assert variant not in pair['runs'];pair['runs'][variant]=run
            repeat_key=(row['cohort_index'],variant)
            if repeat_key in seen:
                old=seen[repeat_key]
                repeated.append((old,run))
                ancestry_stable &= old['beam_indices']==run['beam_indices']
            seen[repeat_key]=run
            old=read(historical/'users'/f"{row['cohort_index']:04d}.json")
            assert old['cohort_index']==row['cohort_index'] and old['manifest_sha256']==reference['manifest_sha256']
            historical_pairs[variant].append((old['runs']['compiled_gqa'],run))
        repeat_check=None
        warm_comparisons=[]; warm_historical=[]; warm_repeat=[]; profile_pairs=[]
        for index in warm:
            baseline=read(path/'warmup_outputs'/f'{index}_dense_forward.json')
            candidate=read(path/'warmup_outputs'/f'{index}_{variant}.json')
            repeat=read(path/'warmup_outputs'/f'{index}_repeat.json')
            warm_repeat.append((candidate,repeat))
            ancestry_stable &= candidate['beam_indices']==repeat['beam_indices']==baseline['beam_indices']
            for x in (baseline,candidate,repeat):validate_run(dict(elapsed_s=1,**x));validate_output_storage(x)
            for row in rows:
                if row['cohort_index']==index:
                    repeated.append((candidate,row['run']))
                    ancestry_stable &= candidate['beam_indices']==row['run']['beam_indices']
            warm_comparisons.append((baseline,candidate))
            old=read(historical/'users'/f'{index:04d}.json')['runs']['compiled_gqa']
            warm_historical.extend((old,x) for x in (baseline,candidate,repeat))
        for n in lengths:
            index=next(u['cohort_index'] for u in users if u['prompt_tokens']==n)
            retained=read(path/'warmup_outputs'/f'{index}_retained.json')
            previous=read(path/'warmup_outputs'/f'{index}_{variant}.json')
            validate_output_storage(retained);validate_run(dict(elapsed_s=1,**retained))
            warm_repeat.append((previous,retained))
            ancestry_stable &= previous['beam_indices']==retained['beam_indices']
        dispatch_raw=read(path/'dispatch.json')
        validate_state(dispatch_raw['precision_policy'],'folded_tf32')
        post=dispatch_raw['replay_after_profiles']
        assert post['replays']-after['replays']==8+(4*len(rows) if full500_quality else 0) and post['events']==after['events'] and post['frozen']
        assert sorted(t['prompt_tokens'] for t in dispatch_raw['profiles'])==[min(lengths),max(lengths)]
        for t in dispatch_raw['profiles']:
            assert t['cohort_index']==next(u['cohort_index'] for u in users if u['prompt_tokens']==t['prompt_tokens'])
            observed=read(path/t['output']);validate_run(dict(elapsed_s=1,**observed));validate_output_storage(observed)
            previous=read(path/'warmup_outputs'/f"{t['cohort_index']}_{variant}.json")
            profile_pairs.append((previous,observed));ancestry_stable &= previous['beam_indices']==observed['beam_indices']
        assert dispatch_raw['generated_code']
        for code in dispatch_raw['generated_code']:assert digest(path/code['file'])==code['sha256']
        dispatch=dispatch_evidence(path,dispatch_raw,True)
        timing=latency([{'runs':{variant:r['run']}} for r in rows],(variant,))[variant]
        assert timing['n']==s['latency']['n']
        for stat in ('median','p95'):assert math.isclose(timing[stat+'_ms'],s['latency'][stat+'_s']*1000,rel_tol=1e-12)
        prefix_audit=None
        if phase in (1,2):
            config_path=snapshot.parent/'release-config.json'
            assert digest(config_path)==expected['release_sha256']['Office_ckpt/config.json']
            prefix_audit=audit_snapshots(path,config=read(config_path),protocol=protocol,release_sha256=expected['release_sha256'])
            prefix_audit['request_storage']=validate_request_storage(
                read(path/'prompt-cache-request-storage.json'),prefix_audit,
                warm_lengths=[u['prompt_tokens'] for u in users])
            snapshots=read(path/'prompt-cache-snapshots.json')
            assert w['prompt_cache']==snapshots[1]['metadata'] and s['prompt_cache']==snapshots[2]['metadata']
            assert 0<w['initial_prefix_build_including_compile_s']<=w['combined_setup_and_warmup_s']
            for row in rows:
                for ptr in row['run']['saved_score_storage_ptrs']+row['run']['graph_logit_storage_ptrs']:
                    assert all(not (a<=ptr<b) for a,b in prefix_audit['storage_ranges'])
        else:
            assert w['prompt_cache'] is None and s['prompt_cache'] is None
            assert w['initial_prefix_build_including_compile_s']==0
            assert not list(path.glob('prompt-cache-*.pt')) and not (path/'prompt-cache-snapshots.json').exists()
        phases.append({'prefix_snapshot_audit':prefix_audit,'phase':phase,'variant':variant,'latency_ms':timing,'memory':s['memory'],
            'compiler_artifacts_reused':artifacts['reused'],
            'setup_memory':w['memory'],'unique_input_storage_bytes':physical,'logical_input_bytes':logical,
            'physical_prefix':after.get('physical_prefix'),'direct_reorder':last,
            'model_load_s':m['model_load_s'],'model_parameter_logical_bytes':m['model_parameter_logical_bytes'],'model_parameter_unique_bytes':m['model_parameter_unique_bytes'],
            'dispatch':dispatch,'repeat_comparison':compare(warm_repeat),'profile_comparison':compare(profile_pairs),'warmup_comparison':compare(warm_comparisons),'warmup_historical_comparison':compare(warm_historical),
            'initial_prefix_build_including_compile_s':w['initial_prefix_build_including_compile_s'],
            'setup_s':w['combined_setup_and_warmup_s'],'backend_compile_s':c['backend_compile_total_s'],
            'capture_and_warmup_s':capture,'measurement_compilations':0,'measurement_captures':0})
    return dict(record=record,expected=expected,users=users,lengths=lengths,full=full,
        phases=phases,joined=joined,repeated=repeated,ancestry_stable=ancestry_stable,
        historical_pairs=historical_pairs)

def evaluate_phases(e,root,snapshot,release,*,wrapper,allocation_identity):
    """Evaluate validated saved phases on the current allocation.

    Callers must validate provenance and deadlines before inspection. Recovery
    callers supply their own live CPU identity; the saved run/job remain the
    original generation identity. This helper does not certify recovery.
    """
    job=os.environ.get('SLURM_JOB_ID')
    assert job and job.isdecimal(),'Allocated quality audit only'
    fields=allocation_identity['scheduler_fields']
    assert fields['JobId']==job and fields['JobState']=='RUNNING','Current audit allocation required'
    rows=[e['joined'][k] for k in sorted(e['joined'])]
    assert len(rows)==24 and all(set(r['runs'])==set(VARIANTS) for r in rows)
    protocol=load_protocol(snapshot.parent/PROTOCOL)
    assert [u['cohort_index'] for u in e['users']]==protocol['cohort_indices']
    prior=protocol['baseline'];prior_root=Path(prior['remote_dir'])
    baseline_audit_path=Path(prior['recovered_audit_remote_path'])
    assert digest(baseline_audit_path)==prior['recovered_audit_sha256']
    baseline_audit=read(baseline_audit_path)
    assert baseline_audit['recovery_audit_passed'] is True
    assert baseline_audit['original_run_id']==prior['run_id'] and baseline_audit['original_job_id']==prior['job_id']
    raw_manifest=read(snapshot.parent/'baseline-raw-manifest.json')
    assert digest(snapshot.parent/'baseline-raw-manifest.json')==prior['raw_manifest_sha256']
    hashes={x['path']:x['sha256'] for x in raw_manifest['files']}
    old={}
    for phase in (1,2):
        name=f'outputs/phase_{phase}/requests.json';p=prior_root/name
        assert digest(p)==hashes[name]
        for r in read(p):
            assert r['variant']=='folded_tf32' and r['cohort_index'] not in old
            old[r['cohort_index']]=r['run']
    assert len(old)==500
    controls=[(old[r['cohort_index']],r['runs']['uncached_tf32']) for r in rows]
    for phase in e['phases']:
        if phase['variant']=='uncached_tf32':
            path=root/f"phase_{phase['phase']}"/'warmup_outputs'
            for u in e['users']:
                for mode in ('dense_forward','uncached_tf32','repeat'):
                    controls.append((old[u['cohort_index']],read(path/f"{u['cohort_index']}_{mode}.json")))
    control=compare(controls)
    control_ancestry=all(a['beam_indices']==b['beam_indices'] for a,b in controls)
    # Explicit adapter mapping to evaluator control/candidate; raw phase identities remain unchanged.
    quality_rows=[dict(r,runs={'folded_fp32':r['runs']['uncached_tf32'],'folded_tf32':r['runs']['fixed69_tf32']}) for r in rows]
    quality=evaluate(**quality_inputs(release,e['users'],quality_rows))
    timing=latency(rows,VARIANTS)
    bylength={str(n):latency([r for r in rows if r['prompt_tokens']==n],VARIANTS) for n in e['lengths']}
    byhalf={str(h):latency([r for r in rows if r['repeat']//3==h],VARIANTS) for h in (0,1)}
    bycross={f'{n}:{h}':latency([r for r in rows if r['prompt_tokens']==n and r['repeat']//3==h],VARIANTS) for n in e['lengths'] for h in (0,1)}
    repeated=compare(e['repeated'])
    stable=exact(repeated) and e['ancestry_stable'] and all(exact(p[k]) for p in e['phases'] for k in ('warmup_comparison','repeat_comparison','profile_comparison'))
    dispatch=all(p['dispatch']['verified'] for p in e['phases'])
    validity=quality['token_repeat_stability'] and all(all(v==0 for v in counts.values()) for counts in quality['output_validity'].values())
    speed=all(t['fixed69_tf32'][k]<=.95*t['uncached_tf32'][k] for t in [timing,*bylength.values(),*byhalf.values()] for k in ('median_ms','p95_ms')) and all(t['fixed69_tf32'][k]<=t['uncached_tf32'][k] for t in bycross.values() for k in ('median_ms','p95_ms'))
    result=dict(schema='fixed-prompt-cache-pilot-audit-v1',integrity_audit_passed=True,run_id=e['record']['run_id'],job_id=e['record']['job_id'],
        unique_users=4,paired_requests=24,quality=quality,wrapper=wrapper,allocation_identity=allocation_identity,control_reference='audited TF32 full500',control_comparison=control,
        control_exact=exact(control) and control_ancestry,stability_exact=stable,dispatch_verified=dispatch,validity_passed=validity,
        comparison=compare([(r['runs']['uncached_tf32'],r['runs']['fixed69_tf32']) for r in rows]),repeated_comparison=repeated,
        latency_ms=timing,latency_by_length_ms=bylength,latency_by_half_ms=byhalf,latency_by_length_half_ms=bycross,
        performance_gate=dict(passed=speed,minimum_incremental_reduction_percent=5),phases=e['phases'],
        protocol_sha256=PROTOCOL_SHA256,promotion=False,noninferiority_established=False,
        note='Four users only: cannot establish population quality parity. Historical FP32 comparisons in phase details are informational; primary exact control is saved TF32.')
    result['execution_audit_passed']=all(result[k] for k in ('integrity_audit_passed','control_exact','stability_exact','dispatch_verified','validity_passed'))
    result['eligible_for_full500']=result['execution_audit_passed'] and speed
    return result


def audit(root,historical,snapshot,run_record,release):
    assert os.environ.get('SLURM_JOB_ID'),'Allocated prompt-cache audit only'
    from .prompt_cache_runner import observe,validate_audit_start
    from .audit_precision_setup import validate_wrapper
    record=read(run_record)
    live=observe(Path(os.environ['OSCAR_RUN_DIR']).resolve(),snapshot.parent.resolve(),root.resolve())
    for key in ('run_id','job_id','files','sha256','resources'):assert record[key]==live[key],key
    budget=validate_audit_start(root,record,time.monotonic())
    wrapper=validate_wrapper(root,record,phases=(0,1,2,3),child_timeout_s=1500)
    phases=inspect_phases(root,historical,snapshot,run_record)
    result=evaluate_phases(phases,root,snapshot,release,wrapper=wrapper,allocation_identity=live['allocation_identity'])
    assert time.monotonic()<budget['effective_deadline_monotonic'],'Audit exceeded original budget'
    result['deadline_validation']=budget
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','historical','snapshot','run-record','release','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists(),'Preserve existing audit evidence'
    try:result=audit(args.root,args.historical,args.snapshot,args.run_record,args.release)
    except Exception as exc:
        result=dict(schema='fixed-prompt-cache-pilot-audit-v1',integrity_audit_passed=False,
            execution_audit_passed=False,error=repr(exc),traceback=traceback.format_exc(),
            promotion=False,noninferiority_established=False)
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result.get(k) for k in ('execution_audit_passed','eligible_for_full500','error')}))
    if not result['execution_audit_passed']:raise SystemExit(1)


if __name__=='__main__':main()
