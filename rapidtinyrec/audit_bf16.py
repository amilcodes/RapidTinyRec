"""Independent BF16-vs-TF32 pilot audit, preserving original raw evidence."""
import argparse
import json
import math
import os
from pathlib import Path
import traceback
from .audit_decode_pilot import read,digest,json_digest,compare,latency
from .audit_prefix_profile import exact
from .audit_precision import validate_run,validate_output_storage,validate_log_staging,TORCH_COMMIT
from .audit_bf16_dispatch import dispatch_evidence
from .bf16_policy import validate_state
from .bf16_protocol import PROTOCOL,PROTOCOL_SHA256,load_protocol,phase_record
from .precision_quality_adapter import quality_inputs
from .quality_frontier import evaluate
VARIANTS=('folded_tf32','folded_bf16')
def inspect_phases(root,historical,snapshot,run_record,*,setup_diagnostic=False,staged_pilot=False,full500_quality=False):
    assert not setup_diagnostic and not staged_pilot and not full500_quality
    record=read(run_record)
    assert record['resources']['gpus']==1 and record['resources']['gpu_type']=='nvidia_rtx_a6000'
    assert record['resources']['cpus']==4 and record['resources']['memory_gb']==32
    assert record['resources']['partition']=='gpu' and record['resources']['qos']=='norm-gpu'
    assert record['resources']['time']=='00:30:00'
    for name,sha in record['sha256'].items():assert digest(snapshot.parent/name)==sha,name
    assert len(record['sha256'])==len(record['files'])
    expected=read(snapshot/'expected.json')
    assert expected==read(Path(__file__).with_name('expected.json'))
    reference=read(historical/'manifest.json')
    assert reference['experiment']=='gqa_three_way' and reference['dtype']=='float32'
    assert json_digest({k:v for k,v in reference.items() if k!='manifest_sha256'})==reference['manifest_sha256']
    assert len(reference['users'])==500 and json_digest(reference['users'])==expected['users_sha256']
    all_lengths=sorted({u['prompt_tokens'] for u in reference['users']})
    phases=[]; joined={}; historical_pairs={v:[] for v in VARIANTS}; seen={}; artifact_paths={}
    full=None; repeated=[]; ancestry_stable=True
    schedule=[(1,'folded_tf32'),(2,'folded_tf32')] if setup_diagnostic else list(enumerate(('folded_tf32','folded_bf16','folded_bf16','folded_tf32')))
    for phase,variant in schedule:
        path=root/f'phase_{phase}'
        m,s,w,c=(read(path/n) for n in ('manifest.json','summary.json','setup.json','compilation.json'))
        validate_log_staging(path,m)
        rows=read(path/'requests.json'); half=0 if phase<2 else 1
        assert m['experiment']=='bf16_vs_tf32_pilot'
        if full is None:full=m['experiment'] in ('precision_500','precision_full500_quality')
        if setup_diagnostic:assert not full,'Setup diagnostic is four-user only'
        assert (m['experiment'] in ('precision_500','precision_full500_quality'))==full
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
        assert s['complete'] and m['hardware']=='NVIDIA RTX A6000' and m['dtype']==('bfloat16' if phase in (1,2) else 'float32') and m['tf32'] is True
        assert m['compiler_policy']==dict(max_autotune=False,max_autotune_gemm=False,max_autotune_pointwise=False)
        assert m['torch_git']==TORCH_COMMIT and m['cuda_version']=='12.6'
        assert str(m['job_id'])==str(record['job_id'])
        assert m['model_load_s']>0 and 0<m['model_parameter_unique_bytes']<=m['model_parameter_logical_bytes']
        for policy in (m['precision_policy'],w['precision_policy'],s['precision_policy']):validate_state(policy,variant)
        protocol=load_protocol(snapshot.parent/PROTOCOL)
        assert m['phase_protocol']==phase_record(protocol,phase,m['compiler_artifacts']['reused'],m['users'])
        assert m['versions']=={'torch':'2.14.0+cu126','transformers':'4.55.4','tokenizers':'0.21.4'}
        assert m['cohort_sha256']==expected['users_sha256'] and m['reference_manifest_sha256']==reference['manifest_sha256']
        names={'benchmark_bf16.py','bf16_policy.py','bf16_protocol.py','precision_dispatch.py','precision_log_staging.py','precision_quality_adapter.py','quality_frontier.py','quality_resources.json','folded_attention.py','shared_prefix.py','prefix_replay.py','prefix_prefill.py','input_arena.py','decode_replay.py','compiled_forward.py',
               'shared_prefill.py','gpu_constraints.py','expected.json','direct_reorder.py'}
        if 'release_verification' in m:
            from .precision_provenance import validate_release
            names.add('precision_provenance.py')
            validate_release(m['release_verification'],expected['release_sha256'])
        if full500_quality:names.add('precision_full500.py')
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
        assert len(rows)==s['measured_requests']==(250 if full else 12)
        assert len(list((path/'users').glob('*.json')))==len(rows)
        assert c['phase']=='complete' and c['measurement_compilations']==0
        assert c['events'] and all(e.get('success') or e.get('expected_restart') for e in c['events'])
        assert not c['compiler_counters'].get('graph_break') and not c['compiler_counters'].get('unimplemented')
        before,after=w['replay'],s['replay']; graphs=4*len(lengths)
        assert before['graphs']==after['graphs']==graphs and before['frozen'] and after['frozen']
        assert before['events']==after['events'] and len(after['events'])==graphs
        assert sorted(e['cache_tokens'] for e in after['events'])==sorted(n+j for n in lengths for j in range(4))
        # Pinned Qwen2: 28 layers, two KV heads, width128, beam50, FP32.
        bytes_per_token=28*2*50*2*128*(2 if variant=='folded_bf16' else 4); metadata_bytes=101*8
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
        if full500_quality:
            from .precision_full500 import load_protocol, PROTOCOL, validate_phase, validate_repeats
            protocol=load_protocol(snapshot.parent/PROTOCOL)
            validate_phase(m,protocol,expected['release_sha256'])
            repeats=read(path/'stability-requests.json')
            repeat_check=validate_repeats(rows,repeats,protocol['phases'][phase])
            saved=read(path/'stability-summary.json')
            assert set(saved)==set(repeat_check)|{'replay','direct_reorder','memory'}
            assert all(saved[k]==v for k,v in repeat_check.items())
            assert sorted(p.name for p in (path/'stability-users').glob('*.json'))==sorted(f"{r['cohort_index']:04d}.json" for r in repeats)
            by_index={r['cohort_index']:r for r in rows}
            for r in repeats:
                assert r==read(path/'stability-users'/f"{r['cohort_index']:04d}.json")
                assert len(r['run']['graph_logit_storage_ptrs'])==graphs
                repeated.append((by_index[r['cohort_index']]['run'],r['run']))
            ancestry_stable &= repeat_check['ancestry_stable']
            replay=saved['replay']; reorder=saved['direct_reorder']
            assert replay['frozen'] and replay['graphs']==graphs
            assert replay['events']==after['events'] and replay['capture_and_warmup_s']==after['capture_and_warmup_s']
            assert replay['replays']-after['replays']==4*len(repeats)
            assert replay['physical_prefix']['requests_loaded']-after['physical_prefix']['requests_loaded']==len(repeats)
            assert replay['physical_prefix']['prefix_bytes_copied']-after['physical_prefix']['prefix_bytes_copied']==sum(bytes_per_token//50*r['prompt_tokens'] for r in repeats)
            assert reorder['direct_calls']-last['direct_calls']==4*len(repeats)
            assert reorder['fallback_calls']-last['fallback_calls']==len(repeats)
            assert reorder['suffix_bytes_written']-last['suffix_bytes_written']==bytes_per_token*6*len(repeats)
            assert reorder['prefix_bytes_reordered']==last['prefix_bytes_reordered']==0
            mem=saved['memory']
            assert 0<physical<=mem['allocated_bytes']<=mem['reserved_bytes']<46*1024**3
            assert mem['allocated_bytes']<=mem['peak_allocated_bytes']<=mem['peak_reserved_bytes']
            assert mem['reserved_bytes']<=mem['peak_reserved_bytes']
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
        validate_state(dispatch_raw['precision_policy'],variant)
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
        dispatch=dispatch_evidence(path,dispatch_raw,phase in (1,2))
        timing=latency([{'runs':{variant:r['run']}} for r in rows],(variant,))[variant]
        assert timing['n']==s['latency']['n']
        for stat in ('median','p95'):assert math.isclose(timing[stat+'_ms'],s['latency'][stat+'_s']*1000,rel_tol=1e-12)
        phases.append({'phase':phase,'variant':variant,'latency_ms':timing,'memory':s['memory'],
            'compiler_artifacts_reused':artifacts['reused'],
            'setup_memory':w['memory'],'unique_input_storage_bytes':physical,'logical_input_bytes':logical,
            'physical_prefix':after.get('physical_prefix'),'direct_reorder':last,
            'model_load_s':m['model_load_s'],'model_parameter_logical_bytes':m['model_parameter_logical_bytes'],'model_parameter_unique_bytes':m['model_parameter_unique_bytes'],
            'dispatch':dispatch,'repeat_comparison':compare(warm_repeat),'profile_comparison':compare(profile_pairs),'warmup_comparison':compare(warm_comparisons),'warmup_historical_comparison':compare(warm_historical),
            'setup_s':w['combined_setup_and_warmup_s'],'backend_compile_s':c['backend_compile_total_s'],
            'capture_and_warmup_s':capture,'measurement_compilations':0,'measurement_captures':0})
        if full500_quality:phases[-1]['untimed_repeat_check']=repeat_check
    return dict(record=record,expected=expected,users=users,lengths=lengths,full=full,
        phases=phases,joined=joined,repeated=repeated,ancestry_stable=ancestry_stable,
        historical_pairs=historical_pairs)


def audit(root,historical,snapshot,run_record,release):
    assert os.environ.get('SLURM_JOB_ID'),'Allocated quality audit only'
    import time
    from .bf16_runner import observe
    from .audit_precision_setup import validate_wrapper
    record=read(run_record)
    live=observe(Path(os.environ['OSCAR_RUN_DIR']).resolve(),snapshot.parent.resolve(),root.resolve())
    for k in ('run_id','job_id','files','sha256','resources'):assert record[k]==live[k],k
    start=read(root/'bf16-work-start.json')
    assert start['job_id']==record['job_id'] and start['protocol_sha256']==PROTOCOL_SHA256
    assert start['started_monotonic']<=time.monotonic()<start['deadline_monotonic']
    wrapper=validate_wrapper(root,record,phases=(0,1,2,3),child_timeout_s=1500)
    e=inspect_phases(root,historical,snapshot,run_record)
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
    controls=[(old[r['cohort_index']],r['runs']['folded_tf32']) for r in rows]
    for phase in e['phases']:
        if phase['variant']=='folded_tf32':
            path=root/f"phase_{phase['phase']}"/'warmup_outputs'
            for u in e['users']:
                for mode in ('dense_forward','folded_tf32','repeat'):
                    controls.append((old[u['cohort_index']],read(path/f"{u['cohort_index']}_{mode}.json")))
    control=compare(controls)
    control_ancestry=all(a['beam_indices']==b['beam_indices'] for a,b in controls)
    # Explicit adapter mapping to evaluator control/candidate; raw phase identities remain unchanged.
    quality_rows=[dict(r,runs={'folded_fp32':r['runs']['folded_tf32'],'folded_tf32':r['runs']['folded_bf16']}) for r in rows]
    quality=evaluate(**quality_inputs(release,e['users'],quality_rows))
    timing=latency(rows,VARIANTS)
    bylength={str(n):latency([r for r in rows if r['prompt_tokens']==n],VARIANTS) for n in e['lengths']}
    byhalf={str(h):latency([r for r in rows if r['repeat']//3==h],VARIANTS) for h in (0,1)}
    bycross={f'{n}:{h}':latency([r for r in rows if r['prompt_tokens']==n and r['repeat']//3==h],VARIANTS) for n in e['lengths'] for h in (0,1)}
    repeated=compare(e['repeated'])
    stable=exact(repeated) and e['ancestry_stable'] and all(exact(p[k]) for p in e['phases'] for k in ('warmup_comparison','repeat_comparison','profile_comparison'))
    dispatch=all(p['dispatch']['verified'] for p in e['phases'])
    validity=quality['token_repeat_stability'] and all(all(v==0 for v in counts.values()) for counts in quality['output_validity'].values())
    speed=all(t['folded_bf16'][k]<=.9*t['folded_tf32'][k] for t in [timing,*bylength.values(),*byhalf.values()] for k in ('median_ms','p95_ms')) and all(t['folded_bf16'][k]<=t['folded_tf32'][k] for t in bycross.values() for k in ('median_ms','p95_ms'))
    result=dict(schema='bf16-vs-tf32-pilot-audit-v1',integrity_audit_passed=True,run_id=e['record']['run_id'],job_id=e['record']['job_id'],
        unique_users=4,paired_requests=24,quality=quality,wrapper=wrapper,allocation_identity=live['allocation_identity'],control_reference='audited TF32 full500',control_comparison=control,
        control_exact=exact(control) and control_ancestry,stability_exact=stable,dispatch_verified=dispatch,validity_passed=validity,
        comparison=compare([(r['runs']['folded_tf32'],r['runs']['folded_bf16']) for r in rows]),repeated_comparison=repeated,
        latency_ms=timing,latency_by_length_ms=bylength,latency_by_half_ms=byhalf,latency_by_length_half_ms=bycross,
        performance_gate=dict(passed=speed,minimum_incremental_reduction_percent=10),phases=e['phases'],
        protocol_sha256=PROTOCOL_SHA256,promotion=False,noninferiority_established=False,
        note='Four users only: cannot establish population quality parity. Historical FP32 comparisons in phase details are informational; primary exact control is saved TF32.')
    result['execution_audit_passed']=all(result[k] for k in ('integrity_audit_passed','control_exact','stability_exact','dispatch_verified','validity_passed'))
    result['eligible_for_full500']=result['execution_audit_passed'] and speed
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('root','historical','snapshot','run-record','release','output'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();assert not a.output.exists()
    try:result=audit(a.root,a.historical,a.snapshot,a.run_record,a.release)
    except Exception as exc:result=dict(integrity_audit_passed=False,execution_audit_passed=False,error=repr(exc),traceback=traceback.format_exc(),promotion=False,noninferiority_established=False)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result.get(k) for k in ('execution_audit_passed','eligible_for_full500','error')}))
    if not result['execution_audit_passed']:raise SystemExit(1)

if __name__=='__main__':main()
