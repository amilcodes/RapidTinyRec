"""Audit packed decode MLP against folded-query controls and historical FP32 bits."""
import argparse
import json
import math
from pathlib import Path
from .audit_decode_pilot import read, digest, json_digest, validate_run, compare, latency, bits, flat
from .audit_prefix_profile import exact

VARIANTS=('folded_prefix_replay','packed_mlp_replay')

def audit(root,historical,snapshot,run_record):
    record=read(run_record)
    assert record['resources']['gpus']==1 and record['resources']['gpu_type']=='nvidia_rtx_a6000'
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
    full=None
    for phase,variant in enumerate(('folded_prefix_replay','packed_mlp_replay','packed_mlp_replay','folded_prefix_replay')):
        path=root/f'phase_{phase}'
        m,s,w,c=(read(path/n) for n in ('manifest.json','summary.json','setup.json','compilation.json'))
        rows=read(path/'requests.json'); half=0 if phase<2 else 1
        assert m['experiment'] in ('packed_mlp_500','packed_mlp_pilot')
        if full is None:full=m['experiment']=='packed_mlp_500'
        assert (m['experiment']=='packed_mlp_500')==full
        assert m['phase']==phase and m['half']==half and m['variant']==variant
        assert m['attention_layout']=='folded_query_heads'
        packing=m['packed_mlp']
        assert packing==w['packed_mlp']==s['packed_mlp']
        assert w['model_loading_s']>0
        assert 0<=w['packing_s']<w['combined_setup_and_warmup_s']
        if phase in (1,2):
            assert w['packing_s']>0 and packing['weight_bits_preserved_at_install']
            assert packing['unique_weight_bytes']==28*2*8960*1536*4
            assert packing['persistent_extra_weight_bytes']==0 and len(packing['layers'])==28
            for layer in packing['layers']:
                assert layer=={'hidden_size':1536,'intermediate_size':8960,'unique_weight_bytes':2*8960*1536*4,
                    'original_gate_up_bytes':2*8960*1536*4,'persistent_extra_weight_bytes':0,'disjoint_contiguous_views':True}
        else:assert packing is None and w['packing_s']==0
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
        assert s['complete'] and m['hardware']=='NVIDIA RTX A6000' and m['dtype']=='float32' and m['tf32'] is False
        assert m['versions']=={'torch':'2.14.0+cu126','transformers':'4.55.4','tokenizers':'0.21.4'}
        assert m['cohort_sha256']==expected['users_sha256'] and m['reference_manifest_sha256']==reference['manifest_sha256']
        names={'benchmark_packed_mlp.py','packed_mlp.py','folded_attention.py','shared_prefix.py','prefix_replay.py','prefix_prefill.py','input_arena.py','decode_replay.py','compiled_forward.py',
               'shared_prefill.py','gpu_constraints.py','expected.json','direct_reorder.py'}
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
            run=row['run'];validate_run(run)
            key=(row['repeat'],row['cohort_index'])
            pair=joined.setdefault(key,{'repeat':row['repeat'],'cohort_index':row['cohort_index'],'prompt_tokens':row['prompt_tokens'],'runs':{}})
            assert variant not in pair['runs'];pair['runs'][variant]=run
            repeat_key=(row['cohort_index'],variant)
            if repeat_key in seen:
                old=seen[repeat_key]
                assert old['token_ids']==run['token_ids']
                assert all(bits(flat(old,f))==bits(flat(run,f)) for f in ('sequence_logp','transition_logp'))
            seen[repeat_key]=run
            old=read(historical/'users'/f"{row['cohort_index']:04d}.json")
            assert old['cohort_index']==row['cohort_index'] and old['manifest_sha256']==reference['manifest_sha256']
            historical_pairs[variant].append((old['runs']['compiled_gqa'],run))
        warm_comparisons=[]; warm_historical=[]
        for index in warm:
            baseline=read(path/'warmup_outputs'/f'{index}_dense_forward.json')
            candidate=read(path/'warmup_outputs'/f'{index}_{variant}.json')
            repeat=read(path/'warmup_outputs'/f'{index}_repeat.json')
            assert candidate['token_ids']==repeat['token_ids']
            assert all(bits(flat(candidate,f))==bits(flat(repeat,f)) for f in ('sequence_logp','transition_logp'))
            warm_comparisons.append((baseline,candidate))
            old=read(historical/'users'/f'{index:04d}.json')['runs']['compiled_gqa']
            warm_historical.extend((old,x) for x in (baseline,candidate,repeat))
        timing=latency([{'runs':{variant:r['run']}} for r in rows],(variant,))[variant]
        assert timing['n']==s['latency']['n']
        for stat in ('median','p95'):assert math.isclose(timing[stat+'_ms'],s['latency'][stat+'_s']*1000,rel_tol=1e-12)
        phases.append({'phase':phase,'variant':variant,'latency_ms':timing,'memory':s['memory'],
            'compiler_artifacts_reused':artifacts['reused'],
            'setup_memory':w['memory'],'unique_input_storage_bytes':physical,'logical_input_bytes':logical,
            'physical_prefix':after.get('physical_prefix'),'direct_reorder':last,
            'warmup_comparison':compare(warm_comparisons),'warmup_historical_comparison':compare(warm_historical),
            'model_loading_s':w['model_loading_s'],'packing_s':w['packing_s'],'packed_mlp':packing,
            'setup_s':w['combined_setup_and_warmup_s'],'backend_compile_s':c['backend_compile_total_s'],
            'capture_and_warmup_s':capture,'measurement_compilations':0,'measurement_captures':0})
    paired=[joined[k] for k in sorted(joined)]
    assert len(paired)==(500 if full else 24) and all(set(r['runs'])==set(VARIANTS) for r in paired)
    timing=latency(paired,VARIANTS)
    return {'integrity_audit_passed':True,'unique_users':500 if full else 4,'paired_requests':len(paired),
        'cohort_sha256':expected['users_sha256'],'comparison':compare([(r['runs'][VARIANTS[0]],r['runs'][VARIANTS[1]]) for r in paired]),
        'historical_comparisons':{v:compare(p) for v,p in historical_pairs.items()},'latency_ms':timing,
        'reductions_percent':{s:100*(1-timing[VARIANTS[1]][s]/timing[VARIANTS[0]][s]) for s in ('median_ms','p95_ms')},
        'latency_by_length_ms':{str(n):latency([r for r in paired if r['prompt_tokens']==n],VARIANTS) for n in lengths},
        'latency_by_half_ms':{str(h):latency([r for r in paired if (r['cohort_index']%2 if full else r['repeat']//3)==h],VARIANTS) for h in (0,1)},
        'phases':phases,'snapshot_files_verified':len(record['files']),'audit_script_sha256':digest(Path(__file__)),
        'note':'Memory is isolated per fresh process; latency is paired by user/repeat across ABBA phases, not interleaved requests. One allocation does not establish run-to-run variance. Setup timings are nested; do not add. Integrity passing alone does not imply bit equality.'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('results',type=Path);ap.add_argument('--historical',type=Path,required=True)
    ap.add_argument('--snapshot',type=Path,required=True);ap.add_argument('--run-record',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    result=audit(a.results,a.historical,a.snapshot,a.run_record)
    comparisons=[result['comparison'],*result['historical_comparisons'].values()]
    comparisons.extend(p[k] for p in result['phases'] for k in ('warmup_comparison','warmup_historical_comparison'))
    result['exactness_passed']=all(exact(c) for c in comparisons)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
    if not result['exactness_passed']:raise SystemExit('FP32 equality failed; all comparisons preserved')

if __name__=='__main__':main()
