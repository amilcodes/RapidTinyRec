"""Independent standard-library audit of immutable native FP32 projection evidence."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import struct
import sys

BACKENDS = ('default', 'cublas', 'cublaslt')
ROLES = ('gate', 'up', 'down')
CASES = [(u,l) for u in (1,2,6,10) for l in (0,13,27)]
COMMIT = '2b3ec34829036a65cd9d1398ea72a0167dc37470'


def sha(data): return hashlib.sha256(data).hexdigest()


def compare_bytes(left, right):
    assert len(left)==len(right) and len(left)%4==0
    n=len(left)//4
    if left==right:
        nonfinite=sum(not math.isfinite(v[0]) for v in struct.iter_unpack('<f',left))
        return dict(values=n,matching_bits=n,max_abs_finite=0.,nonfinite_left=nonfinite,nonfinite_right=nonfinite)
    matched=0; maximum=0.; nonfinite_a=nonfinite_b=0
    for i,(a,b) in enumerate(zip(struct.iter_unpack('<f',left),struct.iter_unpack('<f',right))):
        a,b=a[0],b[0]
        matched += left[4*i:4*i+4]==right[4*i:4*i+4]
        if math.isfinite(a) and math.isfinite(b): maximum=max(maximum,abs(a-b))
        nonfinite_a += not math.isfinite(a);nonfinite_b += not math.isfinite(b)
    return dict(values=n,matching_bits=matched,max_abs_finite=maximum,nonfinite_left=nonfinite_a,nonfinite_right=nonfinite_b)


def check_layout(ref, shape):
    assert ref['shape']==list(shape) and ref['stride']==[shape[1],1]
    assert ref['dtype']=='torch.float32' and ref['storage_offset']==0


def read_tensor(root, entries, ref, shape, cache):
    check_layout(ref,shape)
    digest=ref['sha256']
    assert re.fullmatch('[0-9a-f]{64}',digest)
    if digest not in cache:
        entry=entries[digest]
        assert entry['file']==f'tensors/{digest}.f32.gz'
        data=(root/entry['file']).read_bytes()
        assert len(data)==entry['compressed_bytes'] and sha(data)==entry['compressed_sha256']
        decoded=gzip.decompress(data)
        assert len(decoded)==entry['bytes'] and sha(decoded)==digest
        cache[digest]=decoded
    value=cache[digest]
    assert len(value)==math.prod(shape)*4
    return value


def row_reverse(data, rows, width):
    step=width*4
    return b''.join(data[i*step:(i+1)*step] for i in reversed(range(rows)))


def trace_evidence(path):
    data=json.loads(path.read_text());events=data['traceEvents']
    kernels=[dict(name=e['name'],grid=e.get('args',{}).get('grid'),block=e.get('args',{}).get('block'),dur_us=e.get('dur'))
             for e in events if e.get('cat')=='kernel']
    roles={e.get('name') for e in events if e.get('name','').startswith('fp32_probe_')}
    assert roles=={'fp32_probe_'+r for r in ROLES} and len(kernels)>=3
    return dict(kernels=kernels,trace_sha256=sha(path.read_bytes()))


def audit_report(root, d, result):
    assert d['schema']==1 and d['complete'] is True
    assert d['torch']=='2.14.0+cu126' and d['cuda']=='12.6' and d['torch_git_version']==COMMIT
    assert d['hardware']=='NVIDIA RTX A6000' and d['transformers']=='4.55.4'
    precision=d['runtime_precision']
    assert precision['tf32'] is False and precision['cudnn_tf32'] is False
    assert precision['tunable_enabled'] is False and precision['tuning_enabled'] is False
    assert precision['matmul_precision']=='highest'
    assert precision['initial_backend'].endswith('.Cublas') and d['restored_backend']==precision['initial_backend']
    assert [(c['cohort_index'],c['layer']) for c in d['cases']]==CASES
    result.update(comparisons=[], timing_cases=[], dispatch=[], input_output_storage=[],model_parameter_logical_bytes=d['model_parameter_logical_bytes'],model_parameter_unique_bytes=d['model_parameter_unique_bytes'])
    entries=d['tensors'];used=set();weight_hashes={};cache={};counts={}
    def tensor(ref,shape):
        used.add(ref['sha256'])
        return read_tensor(root,entries,ref,shape,cache)
    def comparison(category,a,b,**identity):
        # Cached by raw byte content hashes, while preserving every observation/count.
        key=(sha(a),sha(b))
        if key not in counts:counts[key]=compare_bytes(a,b)
        row=dict(category=category,**identity,**counts[key]);result['comparisons'].append(row)
        return row
    for case in d['cases']:
        u,l=case['cohort_index'],case['layer']
        assert case['prompt_tokens'] in (76,121)
        assert case['graph_count']==9 and case['timed_graph_replays']==2304
        assert case['measured_captures']==case['measured_compilations']==0
        assert set(case['variants'])==set(BACKENDS)
        ptrs=[p for v in case['variants'].values() for p in v['output_ptrs'].values()]
        assert len(ptrs)==len(set(ptrs))==9
        assert len(set(case['staging_ptrs']))==3 and not set(ptrs)&set(case['staging_ptrs'])
        expected_in=(50*1536*2+50*8960)*4
        expected_out=(50*8960*2+50*1536)*4*3
        assert case['input_logical_bytes']==expected_in and case['output_logical_bytes']==expected_out
        assert case['unique_input_output_bytes']==expected_in+expected_out
        result['input_output_storage'].append(dict(cohort_index=u,layer=l,unique_bytes=expected_in+expected_out))
        for role in ROLES:
            width,outer=(8960,1536) if role=='down' else (1536,8960)
            xs,ys=(50,width),(50,outer)
            assert case['weight_shapes'][role]==[outer,width]
            wh=case['weight_sha256'][role]
            assert re.fullmatch('[0-9a-f]{64}',wh)
            assert weight_hashes.setdefault((l,role),wh)==wh
            original=tensor(case['inputs'][role],xs)
            changed=row_reverse(original,50,width)
            assert original!=changed, 'Changed-input probe must actually change values'
            states={}
            for backend in BACKENDS:
                variant=case['variants'][backend]
                expected_enum={'default':'Cublas','cublas':'Cublas','cublaslt':'Cublaslt'}[backend]
                assert variant['requested_backend']==backend
                assert variant['selected_backend'].endswith('.'+expected_enum)
                assert set(variant['states'])=={'original','changed','restored'}
                for state,detail in variant['states'].items():
                    assert tensor(detail['inputs'][role],xs)==(changed if state=='changed' else original)
                    refs=detail['roles'][role]
                    assert set(refs)=={'direct','direct_repeat','graph','graph_repeat','retained_after_other_roles'}
                    values={k:tensor(v,ys) for k,v in refs.items()}
                    for k in refs:
                        if k!='direct':comparison(k,values['direct'],values[k],cohort_index=u,layer=l,role=role,backend=backend,state=state)
                    states[backend,state]=values['direct']
                comparison('restored_input',states[backend,'original'],states[backend,'restored'],cohort_index=u,layer=l,role=role,backend=backend)
                comparison('after_timing',states[backend,'original'],tensor(variant['after_timing'][role],ys),cohort_index=u,layer=l,role=role,backend=backend)
                samples=variant['projection_us'][role]
                assert len(samples)==8 and all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in samples)
                summary=dict(median_us=statistics.median(samples),p95_us=statistics.quantiles(samples,n=100,method='inclusive')[94],min_us=min(samples),max_us=max(samples))
                assert summary==variant['timing_summary'][role]
                result['timing_cases'].append(dict(cohort_index=u,layer=l,role=role,backend=backend,samples_us=samples,**summary))
            for state in ('original','changed','restored'):
                for backend in ('cublas','cublaslt'):
                    comparison('versus_default',states['default',state],states[backend,state],cohort_index=u,layer=l,role=role,backend=backend,state=state)
        for backend,v in case['variants'].items():
            dispatch=dict(cohort_index=u,layer=l,backend=backend,**trace_evidence(root/v['trace']))
            logs=v['logs'];dispatch['logs_available']=logs['available'];dispatch['verified']=False
            if logs['available']:
                a=(root/logs['cublas']).read_text() if (root/logs['cublas']).exists() else ''
                b=(root/logs['cublaslt']).read_text() if (root/logs['cublaslt']).exists() else ''
                dispatch.update(cublas_sgemm_mentions=len(re.findall(r'cublasSgemm',a)),
                                cublaslt_matmul_mentions=len(re.findall(r'cublasLtMatmul\b',b)),
                                cublas_log_sha256=sha(a.encode()),cublaslt_log_sha256=sha(b.encode()))
                dispatch['verified']=(dispatch['cublaslt_matmul_mentions']>=3 and dispatch['cublas_sgemm_mentions']==0) if backend=='cublaslt' else dispatch['cublas_sgemm_mentions']>=3
            else:dispatch['limitation']=logs.get('error','Unavailable')
            result['dispatch'].append(dispatch)
        cache.clear()
    assert used==set(entries), 'Missing or unreferenced raw tensor objects'
    for category in sorted({r['category'] for r in result['comparisons']}):
        rows=[r for r in result['comparisons'] if r['category']==category]
        result.setdefault('counts_by_category',{})[category]=dict(observations=len(rows),values=sum(r['values'] for r in rows),matching_bits=sum(r['matching_bits'] for r in rows),max_abs_finite=max(r['max_abs_finite'] for r in rows))
    def exact(rows):return all(r['values']==r['matching_bits'] and r['nonfinite_left']==r['nonfinite_right']==0 for r in rows)
    result['control_exact']=exact(r for r in result['comparisons'] if r.get('backend')!='cublaslt')
    result['candidate_exact']=exact(result['comparisons'])
    result['stability_exact']=exact(r for r in result['comparisons'] if r['category']!='versus_default')
    result['dispatch_verified']=all(r['verified'] for r in result['dispatch'])
    table={(r['cohort_index'],r['layer'],r['role'],r['backend']):r for r in result['timing_cases']}
    result['candidate_faster_median_cases']=sum(table[u,l,r,'cublaslt']['median_us']<table[u,l,r,'default']['median_us'] for u,l in CASES for r in ROLES)
    result['candidate_faster_p95_cases']=sum(table[u,l,r,'cublaslt']['p95_us']<table[u,l,r,'default']['p95_us'] for u,l in CASES for r in ROLES)
    result['projection_cases']=36
    result['raw_tensor_objects']=len(entries)
    result['integrity_audit_passed']=True
    result['end_to_end_promotion']=False
    result['note']='Fixed-input microtimings only. No generation/ranking-quality claim or full500 gate. Repeated observations are not independent users. Inspect useful benefit before a separately scoped generation pilot.'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path)
    ap.add_argument('--snapshot',type=Path,required=True)
    ap.add_argument('--run-record',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    result=dict(integrity_audit_passed=False,candidate_exact=False,errors=[])
    try:
        record=json.loads(args.run_record.read_text());d=json.loads((args.root/'diagnosis.json').read_text())
        assert record['job_id']==d['job_id']
        for name,expected in record['sha256'].items():assert sha((args.snapshot/name).read_bytes())==expected,name
        expected=json.loads((args.snapshot/'rapidtinyrec/expected.json').read_text())
        assert d['release_sha256']==expected['release_sha256'] and d['cohort_sha256']==expected['users_sha256']
        for name,digest in d['source_sha256'].items():assert record['sha256'][name]==digest,name
        assert set(d['source_sha256'])=={n for n in record['sha256'] if n.startswith('rapidtinyrec/')}
        result['snapshot_files_verified']=len(record['sha256'])
        result['audit_source_sha256']=sha(Path(__file__).read_bytes())
        audit_report(args.root,d,result)
    except Exception as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result.get(k) for k in ['integrity_audit_passed','control_exact','candidate_exact','stability_exact','dispatch_verified','candidate_faster_median_cases','candidate_faster_p95_cases','errors']},indent=2))
    if not result['integrity_audit_passed'] or not result['candidate_exact']:sys.exit(1)


if __name__=='__main__':main()
