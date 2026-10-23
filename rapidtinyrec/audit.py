"""Independently recompute latency and FP32 output equality from raw requests."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import struct


def bits(values):
    return [struct.pack('!f', x) for x in values]


def audit(root):
    manifest=json.loads((root/'manifest.json').read_text())
    stored=json.loads((root/'summary.json').read_text())
    expected=json.loads(Path(__file__).with_name('expected.json').read_text())
    payload={k:v for k,v in manifest.items() if k!='manifest_sha256'}
    assert hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==manifest['manifest_sha256']
    assert hashlib.sha256(json.dumps(manifest['users'],sort_keys=True).encode()).hexdigest()==expected['users_sha256']==manifest['cohort_sha256']
    rows=[json.loads(p.read_text()) for p in sorted((root/'users').glob('*.json'))]
    assert stored['complete'] and len(rows)==len(manifest['users'])==500
    variants=tuple(rows[0]['runs'])
    for i,(row,user) in enumerate(zip(rows,manifest['users'])):
        assert row['cohort_index']==user['cohort_index']==i
        assert row['user_id']==user['user_id']
        assert row['prompt_tokens']==user['prompt_tokens']
        assert row['manifest_sha256']==manifest['manifest_sha256']
        assert set(row['runs'])==set(row['order'])==set(variants)
        for run in row['runs'].values():
            assert len(run['token_ids'])==len(run['sequence_logp'])==len(run['transition_logp'])==50
            assert len({tuple(t) for t in run['token_ids']})==50
            assert all(len(t)==5 and t[-2:]==[198,151645] for t in run['token_ids'])
            assert all(len(t)==5 for t in run['transition_logp'])
            assert math.isfinite(run['elapsed_s']) and run['elapsed_s']>0
            assert all(math.isfinite(x) for x in run['sequence_logp'])
            assert all(math.isfinite(x) for t in run['transition_logp'] for x in t)
    report={'audit_passed':True,'users':500,'cohort_sha256':manifest['cohort_sha256'],
            'latency_ms':{},'comparisons':{}}
    for v in variants:
        values=[r['runs'][v]['elapsed_s'] for r in rows]
        median,p95=statistics.median(values),statistics.quantiles(values,n=100,method='inclusive')[94]
        prior=stored.get('latency',stored)[v]
        assert math.isclose(median,prior['median_s'],rel_tol=1e-12)
        assert math.isclose(p95,prior['p95_s'],rel_tol=1e-12)
        report['latency_ms'][v]={'median':1000*median,'p95':1000*p95}
    for a,b in itertools.combinations(variants,2):
        comparison={'matching_rank_lists':0,'matching_rank_positions':0,
                    'matching_sequence_values_in_rank_order':0,'matching_transition_values_in_rank_order':0,
                    'max_sequence_difference_common':0.,'max_transition_difference_common':0.}
        for row in rows:
            x,y=row['runs'][a],row['runs'][b]
            comparison['matching_rank_lists']+=x['token_ids']==y['token_ids']
            comparison['matching_rank_positions']+=sum(i==j for i,j in zip(x['token_ids'],y['token_ids']))
            for field,label in [('sequence_logp','sequence'),('transition_logp','transition')]:
                vx,vy=x[field],y[field]
                if label=='transition':vx,vy=([v for t in array for v in t] for array in (vx,vy))
                comparison['matching_'+label+'_values_in_rank_order']+=sum(i==j for i,j in zip(bits(vx),bits(vy)))
                mx,my=({tuple(t):s for t,s in zip(r['token_ids'],r[field])} for r in (x,y))
                differences=[]
                for token in mx.keys() & my.keys():
                    sx,sy=([mx[token]],[my[token]]) if label=='sequence' else (mx[token],my[token])
                    differences.extend(abs(i-j) for i,j in zip(sx,sy))
                key='max_'+label+'_difference_common'
                comparison[key]=max(comparison[key],max(differences,default=0.))
        report['comparisons'][a+'_vs_'+b]=comparison
    compilation=root/'compilation.json'
    if not compilation.exists():compilation=root/'compilation/compilation.json'
    if compilation.exists():
        data=json.loads(compilation.read_text())
        assert data['phase']=='complete' and data['measurement_compilations']==0
        assert not data['compiler_counters'].get('graph_break')
        assert all(e.get('success') or e.get('expected_restart') for e in data['events'])
        report['compilation']={k:data[k] for k in ['setup_wall_s','backend_compile_total_s','measurement_compilations']}
    elif manifest.get('experiment','shared_prefill') in ('compile_forward','gqa_three_way','combined_suffix'):
        raise AssertionError('Missing compilation report')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=audit(args.results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
