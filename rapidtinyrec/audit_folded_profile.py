"""Audit folded-query profiling and attribute diagnostic matrix multiplication shapes."""
import argparse
import collections
import gzip
import json
import math
from pathlib import Path
import statistics

from .audit_decode_pilot import read,digest,json_digest,validate_run,compare
from .audit_replay_profile import union_us


def kernel_category(name):
    name=name.lower()
    # Fused kernel names can mention attention without executing SDPA itself.
    if name.startswith('triton_'):
        if '_cat' in name:return 'fused_cat_layout_rope'
        if 'softmax' in name:return 'softmax'
        if 'pow' in name or 'rsqrt' in name:return 'normalization'
        if 'silu' in name:return 'silu_elementwise'
        return 'other_triton'
    if any(x in name for x in ('fmha','attention_kernel','efficient_attention')):return 'attention'
    if any(x in name for x in ('gemm','gemv','cublas')):return 'matrix_multiplication'
    if 'softmax' in name:return 'softmax'
    if 'index' in name:return 'indexing'
    if any(x in name for x in ('copy','catarray','catkernel')):return 'copy_or_cat'
    return 'other'


def analyze_events(events):
    scopes=[e for e in events if e.get('cat')=='user_annotation' and e['name'].startswith('section/')]
    cpu={e.get('args',{}).get('External id'):e for e in events if e.get('cat')=='cpu_op'}
    launches={e['args']['correlation']:e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver')
              and e.get('ph')=='X' and 'correlation' in e.get('args',{})}
    activities=[e for e in events if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')]
    groups=collections.defaultdict(lambda:{'count':0,'gpu_ms':0.})
    ops=collections.defaultdict(lambda:{'count':0,'gpu_ms':0.})
    stage_events=collections.defaultdict(list)
    for e in activities:
        a=e.get('args',{});anchor=launches.get(a.get('correlation')) or cpu.get(a.get('External id'))
        parents=[s for s in scopes if anchor and s['tid']==anchor['tid'] and s['ts']<=anchor['ts']<s['ts']+s['dur']]
        stage=min(parents,key=lambda s:s['dur'])['name'][8:] if parents else 'unassigned'
        if stage=='request':stage='generation_other'
        category=kernel_category(e['name']) if e['cat']=='kernel' else e['cat']
        row=groups[(stage,category,e['name'])];row['count']+=1;row['gpu_ms']+=e['dur']/1000
        stage_events[stage].append(e)
        op=cpu.get(a.get('External id')) or (cpu.get(anchor.get('args',{}).get('External id')) if anchor else None)
        if op:
            key=(stage,op['name'],json.dumps(op.get('args',{}).get('Input Dims',[])))
            row=ops[key];row['count']+=1;row['gpu_ms']+=e['dur']/1000
    kernels=[e for e in activities if e['cat']=='kernel']
    request=[s for s in scopes if s['name']=='section/request']
    assert len(request)==1
    return {'request_instrumented_wall_ms':request[0]['dur']/1000,'activities':len(activities),
        'kernel_count':len(kernels),'kernel_sum_ms':sum(e['dur'] for e in kernels)/1000,
        'kernel_union_ms':union_us(kernels)/1000,'activity_union_ms':union_us(activities)/1000,
        'stages':{name:{'activity_sum_ms':sum(e['dur'] for e in rows)/1000,
                       'activity_union_ms':union_us(rows)/1000,'activities':len(rows)} for name,rows in stage_events.items()},
        'kernels':[{'stage':s,'category':c,'name':n,**v} for (s,c,n),v in sorted(groups.items(),key=lambda x:-x[1]['gpu_ms'])],
        'operations':[{'stage':s,'name':n,'input_dims':json.loads(d),**v} for (s,n,d),v in sorted(ops.items(),key=lambda x:-x[1]['gpu_ms'])],
        'runtime_calls':dict(collections.Counter(e['name'] for e in launches.values()))}


def exact(result):
    return (result['matching_rank_positions']==result['rank_positions'] and
            result['matching_sequence_values']==result['sequence_values'] and
            result['matching_transition_values']==result['transition_values'])


def matmul_role(name, dims):
    """Classify only recognized linear shapes; identical Q/O or K/V stay grouped."""
    offset = 1 if name == 'aten::addmm' else 0
    if name not in ('aten::mm', 'aten::addmm') or len(dims) < offset + 2:
        return None
    left, right = dims[offset:offset + 2]
    if len(left) != 2 or len(right) != 2 or left[1] != right[0]:
        return None
    roles = {(1536, 8960): 'mlp_gate_or_up', (8960, 1536): 'mlp_down',
             (1536, 1536): 'query_or_output_projection',
             (1536, 256): 'key_or_value_projection',
             (1536, 152265): 'full_vocabulary_projection'}
    return roles.get((left[1], right[1]), 'other_linear')


def matmul_breakdown(trace):
    """Sum correlated GPU activity, not CPU time or an extra latency budget.

    Replay graph nodes may lack operation shapes. Diagnostic calls supply those
    correlations; unknown operations remain in the full trace for inspection.
    Activity counts are not asserted to equal matmul calls or memory transfers.
    """
    groups = collections.defaultdict(lambda: {'gpu_ms': 0., 'gpu_activity_count': 0, 'shapes': []})
    for op in trace['operations']:
        role = matmul_role(op['name'], op['input_dims'])
        if role is None:
            continue
        group = groups[(op['stage'], role)]
        group['gpu_ms'] += op['gpu_ms']
        group['gpu_activity_count'] += op['count']
        group['shapes'].append({'operation': op['name'], 'input_dims': op['input_dims'],
                               'gpu_activity_count': op['count'], 'gpu_ms': op['gpu_ms']})
    return [{'stage': stage, 'role': role, **values}
            for (stage, role), values in sorted(groups.items(), key=lambda row: -row[1]['gpu_ms'])]


def audit(root,historical,snapshot,run_record):
    record=read(run_record)
    assert record['resources']['gpus']==1 and record['resources']['gpu_type']=='nvidia_rtx_a6000'
    for name,sha in record['sha256'].items():assert digest(snapshot.parent/name)==sha,name
    expected=read(snapshot/'expected.json');assert expected==read(Path(__file__).with_name('expected.json'))
    ref=read(historical/'manifest.json')
    assert ref['experiment']=='gqa_three_way' and ref['dtype']=='float32'
    assert json_digest({k:v for k,v in ref.items() if k!='manifest_sha256'})==ref['manifest_sha256']
    assert len(ref['users'])==500 and json_digest(ref['users'])==expected['users_sha256']
    manifest,setup,compiler,profile=(read(root/n) for n in ('manifest.json','setup.json','compilation.json','prefix_profile.json'))
    assert manifest['experiment']=='folded_query_profile' and manifest['variant']=='folded_prefix_replay'
    assert manifest['attention_layout']=='folded_query_heads' and manifest['phase']==1
    assert manifest['dtype']=='float32' and manifest['tf32'] is False and manifest['hardware']=='NVIDIA RTX A6000'
    assert manifest['versions']=={'torch':'2.14.0+cu126','transformers':'4.55.4','tokenizers':'0.21.4'}
    assert manifest['users']==ref['users'] and manifest['reference_manifest_sha256']==ref['manifest_sha256']
    assert manifest['cohort_sha256']==expected['users_sha256']
    assert not manifest['compiler_artifacts']['reused'] and not any(v['files'] for v in manifest['compiler_artifacts']['before'].values())
    for name,sha in {**manifest['sha256'],**profile['source_sha256']}.items():assert digest(snapshot/name)==sha,name
    assert profile['complete'] and profile['measurement_compilations']==profile['captures_during_profiling']==0
    assert compiler['phase']=='complete' and compiler['measurement_compilations']==0
    assert compiler['events'] and all(e.get('success') or e.get('expected_restart') for e in compiler['events'])
    assert not compiler['compiler_counters'].get('graph_break') and not compiler['compiler_counters'].get('unimplemented')
    before,after=profile['before_replay'],profile['after_replay']
    assert before==setup['replay']
    assert before['frozen'] and after['frozen'] and before['graphs']==after['graphs']==40
    assert before['events']==after['events'] and len(after['events'])==40
    assert after['replays']-before['replays']==4*(60+30+3)
    lengths=sorted({u['prompt_tokens'] for u in ref['users']})
    assert len(lengths)==10
    users=[next(u for u in ref['users'] if u['prompt_tokens']==n) for n in lengths]
    assert setup['warmup_cohort_indices']==[u['cohort_index'] for u in users]
    assert sorted(e['cache_tokens'] for e in after['events'])==sorted(n+j for n in lengths for j in range(4))
    bytes_token=28*2*50*2*128*4
    logical=sum(bytes_token//50*n+bytes_token*j+808 for n in lengths for j in range(4))
    physical=bytes_token//50*max(lengths)+bytes_token*3+40*808
    for value in (before,after):
        assert value['input_buffer_bytes']==logical and value['unique_input_storage_bytes']==physical
        assert value['physical_prefix']['physical_prefix_batch']==1 and value['physical_prefix']['suffix_batch']==50
        assert value['physical_prefix']['prefix_capacity_bytes']==bytes_token//50*max(lengths)
        assert value['physical_prefix']['suffix_capacity_bytes']==bytes_token*3
    historical_outputs={}
    for u in users:
        old=read(historical/'users'/f'{u["cohort_index"]:04d}.json')
        assert old['cohort_index']==u['cohort_index'] and old['manifest_sha256']==ref['manifest_sha256']
        historical_outputs[u['cohort_index']]=old['runs']['compiled_gqa']
    comparisons={};traces=[];latencies=collections.defaultdict(list);light=collections.defaultdict(list)
    diagnostic=read(root/'diagnostic_warmup.json')
    assert [(r['cohort_index'],r['prompt_tokens']) for r in diagnostic['rows']]==[(u['cohort_index'],u['prompt_tokens']) for u in users]
    groups={k:profile[k] for k in ('uninstrumented','light_profiles','torch_profiles')}
    groups['diagnostic_warmup']=diagnostic['rows']
    for group,rows in groups.items():
        pairs=[];seen=set()
        for row in rows:
            u=next(u for u in users if u['cohort_index']==row['cohort_index'])
            assert row['prompt_tokens']==u['prompt_tokens']
            validate_run({**row['output'],'elapsed_s':row.get('elapsed_s',1.)})
            pairs.append((historical_outputs[u['cohort_index']],row['output']))
            key=(u['cohort_index'],row.get('repeat',0),row.get('mode',''))
            assert key not in seen;seen.add(key)
            if group in ('uninstrumented','light_profiles'):
                assert row['repeat'] in range(6 if group=='uninstrumented' else 3)
            if group=='uninstrumented':latencies[u['prompt_tokens']].append(row['elapsed_s']*1000)
            if group=='light_profiles':
                scopes=row['scopes'];counts=collections.Counter(s['name'] for s in scopes)
                assert counts['compiled_prefill']==counts['prefix_copy']==counts['final_unused_cache_reorder']==1
                assert counts['graph_decode']==counts['replay_input_staging']==counts['suffix_reorder']==4
                # HF normalizes once and the pinned constraint processor again.
                assert counts['vocabulary_log_softmax']==10
                request=next(s for s in scopes if s['name']=='request')
                assert math.isclose(sum(s['exclusive_wall_ms'] for s in scopes),request['wall_ms'],abs_tol=1e-6)
                totals=collections.defaultdict(float)
                for s in scopes:totals[s['name']]+=s['exclusive_wall_ms']
                light[u['prompt_tokens']].append(totals)
            if group=='torch_profiles':
                assert row['mode'] in ('replay','diagnostic') and u in (users[0],users[5],users[-1])
                name=f'prefix_trace_{row["mode"]}_{u["prompt_tokens"]}.json.gz';assert row['trace']==name
                with gzip.open(root/name,'rt') as stream:trace=analyze_events(json.load(stream)['traceEvents'])
                assert trace['kernel_count']>0 and trace['kernel_sum_ms']>0
                wanted='graph_decode' if row['mode']=='replay' else 'diagnostic_decode'
                assert wanted in trace['stages'] and 'compiled_prefill' in trace['stages']
                traces.append({'prompt_tokens':u['prompt_tokens'],'mode':row['mode'],'sha256':digest(root/name),**trace})
        assert len(rows)=={'uninstrumented':60,'light_profiles':30,'torch_profiles':6,'diagnostic_warmup':10}[group]
        comparisons[group]=compare(pairs)
    warm=[]
    for u in users:
        for name in ('dense_forward','folded_prefix_replay','repeat'):
            raw=read(root/'warmup_outputs'/f'{u["cohort_index"]}_{name}.json')
            validate_run({**raw,'elapsed_s':1.});warm.append((historical_outputs[u['cohort_index']],raw))
    comparisons['warmup']=compare(warm)
    for name,value in read(root/'inductor_sources.json').items():assert digest(root/'inductor_sources'/name)==value['sha256']
    return {'integrity_audit_passed':True,'exactness_passed':all(exact(v) for v in comparisons.values()),
        'historical_comparisons':comparisons,'latency_ms_by_length':{n:{'n':len(v),'median':statistics.median(v),
            'p95':statistics.quantiles(v,n=100,method='inclusive')[94]} for n,v in latencies.items()},
        'light_exclusive_host_ms_mean':{n:{k:statistics.mean(r.get(k,0) for r in rows) for k in set().union(*rows)} for n,rows in light.items()},
        'traces':traces,'matmul_attribution':[{'prompt_tokens':t['prompt_tokens'],'mode':t['mode'],
            'groups':matmul_breakdown(t)} for t in traces],'setup_s':setup['combined_setup_and_warmup_s'],
        'backend_compile_s_nested':compiler['backend_compile_total_s'],
        'graph_preparation_s_nested':before['capture_and_warmup_s'],'diagnostic_warmup_s_nested':diagnostic['elapsed_s'],
        'measurement_compilations':0,'measurement_captures':0,'uninstrumented_memory':profile['uninstrumented_memory'],
        'audit_source_sha256':digest(Path(__file__)),
        'note':'Representatives are not the 500-user distribution. Trace wall time is instrumented; activity sums can overlap. Fused cat/layout/RoPE kernels include operations beyond dense KV copies. Diagnostic traces are attribution only, not comparable serving latency.'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('results',type=Path);ap.add_argument('--historical',type=Path,required=True)
    ap.add_argument('--snapshot',type=Path,required=True);ap.add_argument('--run-record',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    result=audit(a.results,a.historical,a.snapshot,a.run_record)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('traces','light_exclusive_host_ms_mean')},indent=2))
    if not result['exactness_passed']:raise SystemExit('FP32 exactness failed; evidence preserved')


if __name__=='__main__':main()
