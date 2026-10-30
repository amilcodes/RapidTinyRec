"""Audit the precision quality track without changing the exact-score baseline audit."""
import argparse
import json
import math
import os
import traceback
import struct
from pathlib import Path
from .audit_decode_pilot import read, digest, json_digest, compare, latency, bits, flat
from .audit_prefix_profile import exact

from .audit_precision_dispatch import dispatch_evidence
from .precision_quality_adapter import quality_inputs
from .quality_frontier import evaluate

VARIANTS=('folded_fp32','folded_tf32')
TORCH_COMMIT='08187d9e0fba026dc8217405802ab5381dc88d90'

def validate_run(run):
    # Retain invalid/duplicate recommendation rows for the quality/validity report.
    assert len(run['token_ids'])==len(run['sequence_logp'])==len(run['transition_logp'])==50
    assert all(isinstance(t,list) and all(type(x) is int and x>=0 for x in t) for t in run['token_ids'])
    assert all(len(t)==5 for t in run['transition_logp'])
    assert math.isfinite(run['elapsed_s']) and run['elapsed_s']>0
    assert all(math.isfinite(x) for f in ('sequence_logp','transition_logp') for x in flat(run,f))
    for score,transitions in zip(run['sequence_logp'],run['transition_logp']):
        total=0.
        for value in transitions:total=struct.unpack('!f',struct.pack('!f',total+value))[0]
        assert bits([score])==bits([total]),'Original five-step FP32 score sum changed'


def check_policy(state, candidate):
    assert state == dict(matmul_tf32=candidate,cudnn_tf32=False,float32_matmul_precision='high' if candidate else 'highest',blas_preference='_BlasBackend.Cublas')

def validate_output_storage(run):
    ptrs=run['saved_score_storage_ptrs']
    assert len(ptrs)==len(set(ptrs))==5 and all(type(p) is int and p>0 for p in ptrs)
    graph_ptrs=run['graph_logit_storage_ptrs']
    assert isinstance(graph_ptrs,list) and all(type(p) is int and p>0 for p in graph_ptrs)
    assert len(graph_ptrs)==len(set(graph_ptrs)) and not (set(ptrs)&set(graph_ptrs))
    assert run['saved_score_dtypes']==['torch.float32']*5
    ancestry=run['beam_indices']
    assert len(ancestry)==50 and all(len(row)==5 and all(type(x) is int and -1<=x<50 for x in row) for row in ancestry)

def fidelity_gate(result):
    controls=[result['historical_comparisons']['folded_fp32']]
    controls += [p['warmup_historical_comparison'] for p in result['phases'] if p['variant']=='folded_fp32']
    stable=[result['repeated_comparison']]
    stable += [p[k] for p in result['phases'] for k in ('warmup_comparison','repeat_comparison','profile_comparison')]
    return dict(control_exact=all(exact(c) for c in controls), stability_exact=all(exact(c) for c in stable) and result['ancestry_stable'],
                candidate_bit_exact=exact(result['comparison']),
                note='Candidate/control fidelity is reported, never required for this quality track. Control historical equality and within-policy repeat/graph stability remain strict.')

def performance_gate(result):
    strata=[result['latency_ms'],*result['latency_by_length_ms'].values(),*result['latency_by_half_ms'].values()]
    useful=all(t['folded_tf32'][s]<=.90*t['folded_fp32'][s] for t in strata for s in ('median_ms','p95_ms'))
    no_cross_regression=all(t['folded_tf32'][s]<=t['folded_fp32'][s] for t in result['latency_by_length_half_ms'].values() for s in ('median_ms','p95_ms'))
    return dict(minimum_reduction_percent=10,passed=useful and no_cross_regression,
        note='Predeclared minimum experimental effect for exploratory full-quality review, not an SLO, significance or production promotion.')

def validate_log_staging(path, manifest):
    """Staging never relaxes dispatch; verify copied bytes and measurement exclusion."""
    if not manifest.get('api_log_staging', False):
        return
    record=read(path/'log-staging.json')
    assert record['schema']=='precision-staged-capture-v1'
    for key in ('setup_succeeded','logs_disabled','published','measurement_allowed'):
        assert record[key] is True,key
    assert record['setup_error'] is None and 'preservation_error' not in record
    publication=record['publication']
    assert publication['complete'] is True
    assert publication['filesystem']['filesystem'] in ('tmpfs','ext4','xfs')
    rows=publication['files']
    assert len(rows)==2 and {r['path'] for r in rows}=={'capture-cublas.log','capture-cublaslt.log'}
    for row in rows:
        source=path/row['path']
        assert source.stat().st_size==row['bytes']>0
        assert digest(source)==row['sha256']
    assert 0<=record['copy_s']<=record['session_wall_s']


def inspect_phases(root,historical,snapshot,run_record,*,setup_diagnostic=False,staged_pilot=False,full500_quality=False):
    assert sum(map(bool,(setup_diagnostic,staged_pilot,full500_quality)))<=1
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
    schedule=[(1,'folded_tf32'),(2,'folded_tf32')] if setup_diagnostic else list(enumerate(('folded_fp32','folded_tf32','folded_tf32','folded_fp32')))
    for phase,variant in schedule:
        path=root/f'phase_{phase}'
        m,s,w,c=(read(path/n) for n in ('manifest.json','summary.json','setup.json','compilation.json'))
        validate_log_staging(path,m)
        rows=read(path/'requests.json'); half=0 if phase<2 else 1
        assert m['experiment'] in (('precision_full500_quality',) if full500_quality else ('precision_setup_feasibility',) if setup_diagnostic else (('precision_paired_staged_pilot',) if staged_pilot else ('precision_500','precision_pilot')))
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
        assert s['complete'] and m['hardware']=='NVIDIA RTX A6000' and m['dtype']=='float32' and m['tf32']==(phase in (1,2))
        assert m['compiler_policy']==dict(max_autotune=False,max_autotune_gemm=False,max_autotune_pointwise=False)
        assert m['torch_git']==TORCH_COMMIT and m['cuda_version']=='12.6'
        assert str(m['job_id'])==str(record['job_id'])
        assert m['model_load_s']>0 and 0<m['model_parameter_unique_bytes']<=m['model_parameter_logical_bytes']
        for policy in (m['precision_policy'],w['precision_policy'],s['precision_policy']):check_policy(policy,phase in (1,2))
        assert m['versions']=={'torch':'2.14.0+cu126','transformers':'4.55.4','tokenizers':'0.21.4'}
        assert m['cohort_sha256']==expected['users_sha256'] and m['reference_manifest_sha256']==reference['manifest_sha256']
        names={'benchmark_precision.py','precision_policy.py','precision_dispatch.py','precision_log_staging.py','precision_quality_adapter.py','quality_frontier.py','quality_resources.json','folded_attention.py','shared_prefix.py','prefix_replay.py','prefix_prefill.py','input_arena.py','decode_replay.py','compiled_forward.py',
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
        check_policy(dispatch_raw['precision_policy'],phase in (1,2))
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


def audit(root,historical,snapshot,run_record,release,*,staged_pilot=False,full500_quality=False):
    evidence=inspect_phases(root,historical,snapshot,run_record,staged_pilot=staged_pilot,full500_quality=full500_quality)
    record,expected,users,lengths,full=(evidence[k] for k in ('record','expected','users','lengths','full'))
    phases,joined,repeated,ancestry_stable,historical_pairs=(evidence[k] for k in
        ('phases','joined','repeated','ancestry_stable','historical_pairs'))
    assert [p['phase'] for p in phases]==[0,1,2,3], 'Full quality audit requires complete ABBA'
    paired=[joined[k] for k in sorted(joined)]
    assert len(paired)==(500 if full else 24) and all(set(r['runs'])==set(VARIANTS) for r in paired)
    timing=latency(paired,VARIANTS)
    if full and not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Full500 bootstrap only on allocated compute; preserve raw data and audit there')
    quality_input=quality_inputs(release,users,paired)
    quality=evaluate(**quality_input)
    result={'integrity_audit_passed':True,'quality':quality,'quality_input_sha256':json_digest(quality_input),
        'repeated_comparison':compare(repeated),'ancestry_stable':ancestry_stable,'unique_users':500 if full else 4,'paired_requests':len(paired),
        'cohort_sha256':expected['users_sha256'],'comparison':compare([(r['runs'][VARIANTS[0]],r['runs'][VARIANTS[1]]) for r in paired]),
        'historical_comparisons':{v:compare(p) for v,p in historical_pairs.items()},'latency_ms':timing,
        'reductions_percent':{s:100*(1-timing[VARIANTS[1]][s]/timing[VARIANTS[0]][s]) for s in ('median_ms','p95_ms')},
        'latency_by_length_ms':{str(n):latency([r for r in paired if r['prompt_tokens']==n],VARIANTS) for n in lengths},
        'latency_by_half_ms':{str(h):latency([r for r in paired if (r['cohort_index']%2 if full else r['repeat']//3)==h],VARIANTS) for h in (0,1)},
        'latency_by_length_half_ms':{f'{n}:{h}':latency([r for r in paired if r['prompt_tokens']==n and (r['cohort_index']%2 if full else r['repeat']//3)==h],VARIANTS) for n in lengths for h in (0,1)},
        'phases':phases,'snapshot_files_verified':len(record['files']),'audit_script_sha256':digest(Path(__file__)),
        'note':'Memory is isolated per fresh process; latency is paired by user/repeat across ABBA phases, not interleaved requests. One allocation does not establish run-to-run variance. Setup timings are nested; do not add. Integrity passing alone does not imply bit equality.'}

    result.update(fidelity_gate(result))
    result['dispatch_verified']=all(p['dispatch']['verified'] for p in phases)
    result['performance_gate']=performance_gate(result)
    result['validity_passed']=quality['token_repeat_stability'] and all(all(v==0 for v in counts.values()) for counts in quality['output_validity'].values())
    result['eligible_for_full_quality_review']=all(result[k] for k in ('control_exact','stability_exact','dispatch_verified','validity_passed')) and result['performance_gate']['passed']
    result['promotion']=False
    result['noninferiority_established']=False
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('results',type=Path);ap.add_argument('--historical',type=Path,required=True)
    ap.add_argument('--snapshot',type=Path,required=True);ap.add_argument('--run-record',type=Path,required=True)
    ap.add_argument('--release-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    assert not a.output.exists(),'Never overwrite audit evidence'
    try:
        result=audit(a.results,a.historical,a.snapshot,a.run_record,a.release_root)
    except Exception as exc:
        result=dict(integrity_audit_passed=False,error=repr(exc),traceback=traceback.format_exc(),promotion=False)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result.get(k) for k in ('integrity_audit_passed','control_exact','stability_exact','candidate_bit_exact','dispatch_verified','eligible_for_full_quality_review','error')},indent=2))
    if not result.get('eligible_for_full_quality_review'):raise SystemExit(1)

if __name__=='__main__':main()
