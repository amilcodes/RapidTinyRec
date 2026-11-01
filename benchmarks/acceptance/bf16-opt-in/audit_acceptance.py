"""Allocated independent raw consumer. Never run against old jobs or locally."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def read(p):
    return json.loads(Path(p).read_text())


def dense(root, name):
    meta=read(root/'full_scores'/f'{name}.json')
    assert type(meta) is list and len(meta)==5
    result=[]
    for step,record in enumerate(meta):
        assert set(record)=={'file','shape','dtype','bytes','sha256'}
        assert record['file']==f'{name}_{step}.f32.gz'
        assert record['shape']==[50,152265] and record['dtype']=='float32'
        assert type(record['bytes']) is int and record['bytes']==50*152265*4
        with gzip.open(root/'full_scores'/record['file'],'rb') as stream:
            raw=stream.read(record['bytes']+1)
        assert len(raw)==record['bytes']
        assert hashlib.sha256(raw).hexdigest()==record['sha256']
        values=np.frombuffer(raw,dtype='<f4').reshape(50,152265)
        assert not np.isnan(values).any() and not np.isposinf(values).any()
        assert np.isfinite(values).any(axis=1).all()
        result.append(raw)
    return result


def audit(out, expected_path):
    expected=read(expected_path);ids=[u['cohort_index'] for u in expected['users']]
    report={'schema':'bf16-opt-in-acceptance-audit-v1','passed':False,'modes':{}}
    for mode in ('fp32','bf16-experimental'):
        root=Path(out)/mode;manifest=read(root/'manifest.json');summary=read(root/'summary.json')
        assert manifest['precision']==mode and manifest['experimental']==(mode!='fp32')
        assert manifest['users']==expected['users'] and manifest['repeats']==2 and manifest['save_full_scores'] is True
        assert manifest['dtype']==('float32' if mode=='fp32' else 'bfloat16')
        assert manifest['torch_git']=='08187d9e0fba026dc8217405802ab5381dc88d90' and manifest['cuda_version']=='12.6'
        assert manifest['versions']=={'torch':'2.14.0+cu126','transformers':'4.55.4','tokenizers':'0.21.4'}
        assert set(manifest['sha256'])=={'benchmark_inference.py','inference_precision.py','folded_attention.py','shared_prefix.py','prefix_replay.py','prefix_prefill.py','input_arena.py','decode_replay.py','compiled_forward.py','shared_prefill.py','gpu_constraints.py','expected.json','direct_reorder.py'}
        for name,digest in manifest['sha256'].items():
            assert hashlib.sha256((Path(__file__).parent/'rapidtinyrec'/name).read_bytes()).hexdigest()==digest
        active=manifest['precision_policy']
        assert active['matmul_tf32']==(mode!='fp32') and active['cudnn_tf32'] is False
        assert active['float32_matmul_precision']==('highest' if mode=='fp32' else 'high')
        assert active['blas_preference']=='_BlasBackend.Cublas'
        if mode!='fp32':
            assert active['bf16_reduced_precision_reduction'] is False and active['bf16_split_k'] is True
            assert active['math_sdpa_reduced_precision_reduction'] is False
        assert manifest['tf32']==(mode!='fp32') and manifest['hardware']=='NVIDIA RTX A6000'
        assert summary['complete'] is True and summary['measured_requests']==8
        assert summary['retained_score_tensors_checked']==40
        assert read(root/'immutable-before.json')==read(root/'immutable-after.json')
        immutable=read(root/'immutable-before.json')
        assert sum(k.startswith('table:') for k in immutable)==12
        assert sum(k.startswith('parameter:') for k in immutable)>100
        restored=read(root/'policy-restoration.json')
        assert restored['before']==restored['after'] and restored['restored'] is True
        compiled=read(root/'compilation.json')
        assert compiled['phase']=='complete' and compiled['measurement_compilations']==0
        assert summary['replay']['frozen'] is True
        setup=read(root/'setup.json')
        assert summary['replay']['graphs']==setup['replay']['graphs']==8
        assert summary['replay']['events']==setup['replay']['events']
        rows=read(root/'requests.json')
        assert [(r['repeat'],r['cohort_index']) for r in rows]==[(n,i) for n in range(2) for i in (ids if n==0 else ids[::-1])]
        for i in ids:
            warm=dense(root,f'warm_{i}')
            previous=None
            for n in range(2):
                row=read(root/'users'/f'{n}_{i:04d}.json');value=row['run']
                assert row==next(r for r in rows if r['repeat']==n and r['cohort_index']==i)
                for key,want in expected['outputs'][mode][str(i)].items():assert value[key]==want,(mode,i,key)
                assert value['score_dtypes']==['torch.float32']*5
                assert type(value['elapsed_s']) is float and 0<value['elapsed_s']<10
                scores=dense(root,f'{n}_{i:04d}')
                assert scores==warm,(mode,i,'full-score bits')
                stable={k:v for k,v in value.items() if k!='elapsed_s'}
                assert previous is None or stable==previous
                previous=stable
        expected_dense={f'{name}_{step}.f32.gz' for name in [*(f'warm_{i}' for i in ids),*(f'{n}_{i:04d}' for n in range(2) for i in ids)] for step in range(5)}
        assert {p.name for p in (root/'full_scores').glob('*.gz')}==expected_dense
        elapsed=[r['run']['elapsed_s']*1000 for r in rows]
        report['modes'][mode]={'requests':8,'full_score_comparisons':40,'historical_outputs_exact':True,'repeat_reverse_and_retention':True,'median_ms':float(np.median(elapsed)),'p95_ms':float(np.percentile(elapsed,95)),'setup_seconds':read(root/'setup.json')['combined_setup_and_warmup_s'],'note':'Four-user acceptance sample; not a new full500 latency or quality claim.'}
    report['passed']=True
    return report


if __name__=='__main__':
    import sys
    result=audit(Path(sys.argv[1]),Path(sys.argv[2]))
    with (Path(sys.argv[1])/'acceptance-audit.json').open('x') as f:json.dump(result,f,indent=2)
