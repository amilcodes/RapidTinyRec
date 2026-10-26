"""Torch-free validation of beam graph evidence and lifetime/capture counters."""
import gzip
import hashlib
import json
import math
import struct

RUNNING = '_get_running_beams_for_next_iteration'
FINISHED = '_update_finished_beams'
METHODS = (RUNNING, FINISHED)
SCENARIOS = ('equal','kth_ties','signed_zero','near_ties','infinities','finished_flags','penalty')
HASHES = {
    '_gather_beams':'4e29903be152b0c59a229a00b8b333466b19103330f242f52c25ccdb596e87e3',
    RUNNING:'97511a4b165780149cfebe20ed583f79e411b6dda676e2430fae57442d7e8a07',
    FINISHED:'b2bd70364b53bdd154e4f54b22d0774846ac53f6720ccea319c8d77d38700d74',
    '_beam_search':'432fafa552a0b380f14340750057593112be902c03ae7eb4f29328d378a4d0c5',
}


def exact_comparison(c):
    assert c['values'] > 0 and 0 <= c['matching_bits'] <= c['values']
    assert math.isfinite(c['max_abs_finite']) and c['max_abs_finite'] >= 0
    return (c['matching_bits'] == c['values'] and c['max_abs_finite'] == 0
            and c['finite_mask_equal'] and c['no_nan'] and c['left_sha256'] == c['right_sha256'])


def output_spec(method, prompt):
    shapes = [[1,50,prompt+8],[1,50],[1,50,8]]
    dtypes = ['torch.int64','torch.float32','torch.int32']
    if method == FINISHED:
        shapes.append([1,50]); dtypes.append('torch.bool')
    return list(zip(shapes,dtypes))


def input_spec(method, prompt):
    common = dict(topk_log_probs=([1,100],'torch.float32'),
        topk_running_sequences=([1,100,prompt+8],'torch.int64'),
        topk_running_beam_indices=([1,100,8],'torch.int32'),
        next_token_hits_stopping_criteria=([1,100],'torch.bool'))
    if method == FINISHED:
        common.update(sequences=([1,50,prompt+8],'torch.int64'),
            beam_scores=([1,50],'torch.float32'), beam_indices=([1,50,8],'torch.int32'),
            is_early_stop_heuristic_unsatisfied=([1,1],'torch.bool'),
            is_sent_finished=([1,50],'torch.bool'), top_num_beam_mask=([100],'torch.bool'))
    return common


SIZES = {'torch.float32':4,'torch.int32':4,'torch.int64':8,'torch.bool':1}


def storage_bytes(spec):
    return sum(math.prod(shape)*SIZES[dtype] for shape,dtype in spec)


def audit_report(r, lengths, penalty=0., early=False, *, synthetic_probe=False):
    expected = {(m,n,s) for m in METHODS for n in lengths for s in range(5)}
    assert r['graphs'] == len(expected) and r['frozen'] and r['disjoint_step_storage']
    assert not r['active_request'] and r['aborted_requests'] == 0
    assert r['helper_source_sha256'] == HASHES
    assert len(r['events']) == len(expected)
    assert {(e['method'],e['prompt_tokens'],e['step']) for e in r['events']} == expected
    for e in r['events']:
        method,n,step=e['method'],e['prompt_tokens'],e['step']
        spec=input_spec(method,n)
        tensors={x[0]:(x[2],x[4]) for x in e['signature'] if x[1]=='tensor'}
        assert tensors==spec
        for name,kind,*value in e['signature']:
            if kind=='tensor':
                shape,stride,dtype,device=value
                assert device=='cuda:0'
                expected_stride=[math.prod(shape[i+1:]) for i in range(len(shape))]
                # inputs() starts finished state from [:, :50].clone() of a
                # 100-candidate tensor. clone preserves its singleton batch
                # stride. Later recurrent state and generation use canonical
                # strides. Validate this exact source-derived layout, not an
                # arbitrary singleton-stride exemption.
                if synthetic_probe and method==FINISHED and step==0 and name in ('sequences','beam_indices'):
                    expected_stride[0] *= 2
                assert stride==expected_stride
        scalars={x[0]:x[2] for x in e['signature'] if x[1]!='tensor'}
        want=dict(num_beams=50)
        if method==FINISHED:want.update(cur_len=n+step,decoder_prompt_len=n,length_penalty=penalty,early_stopping=early)
        assert scalars==want
        assert e['input_bytes']==storage_bytes(spec.values())
        assert e['output_bytes']==storage_bytes(output_spec(method,n))
        assert e['capture_and_warmup_s']>0
    assert r['input_logical_bytes']==sum(e['input_bytes'] for e in r['events'])
    assert r['output_logical_bytes']==sum(e['output_bytes'] for e in r['events'])
    assert r['unique_input_output_storage_bytes']==r['input_logical_bytes']+r['output_logical_bytes']
    assert r['capture_and_warmup_s']==sum(e['capture_and_warmup_s'] for e in r['events'])


def flattened(value):
    if isinstance(value,list):
        return [x for v in value for x in flattened(v)]
    return [value]


def raw_comparison(c,left,right):
    a,b=flattened(left),flattened(right)
    assert len(a)==len(b)==c['values']
    fmt={'torch.float32':'f','torch.int64':'q','torch.int32':'i','torch.bool':'?'}[c['dtype']]
    ab=[struct.pack('<'+fmt,x) for x in a];bb=[struct.pack('<'+fmt,x) for x in b]
    assert hashlib.sha256(b''.join(ab)).hexdigest()==c['left_sha256']
    assert hashlib.sha256(b''.join(bb)).hexdigest()==c['right_sha256']
    assert sum(x==y for x,y in zip(ab,bb))==c['matching_bits']
    assert all(math.isfinite(x)==math.isfinite(y) for x,y in zip(a,b))==c['finite_mask_equal']
    assert all(not math.isnan(x) and not math.isnan(y) for x,y in zip(a,b))==c['no_nan']
    delta=max((abs(x-y) for x,y in zip(a,b) if math.isfinite(x) and math.isfinite(y)),default=0.)
    assert delta==c['max_abs_finite']


def audit_probe(probe,raw_path):
    assert probe['complete'] and probe['scenarios']==list(SCENARIOS)
    assert raw_path.name==probe['raw_outputs_file']
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest()==probe['raw_outputs_sha256']
    key=lambda r:(r['scenario'],r['request'],r['step'],r['method'])
    expected={(s,r,t,m) for s in SCENARIOS for r in range(4) for t in range(5) for m in METHODS}
    rows={key(r):r for r in probe['graph_direct']}
    assert len(probe['graph_direct'])==len(rows)==len(expected) and set(rows)==expected
    seen=set()
    with gzip.open(raw_path,'rt') as f:
        for line in f:
            raw=json.loads(line)
            if 'retained' in raw:continue
            k=key(raw);assert k in rows and k not in seen;seen.add(k)
            assert raw['comparisons']==rows[k]['comparisons']
            assert raw['direct_repeat_comparisons']==rows[k]['direct_repeat_comparisons']
            for c,a,b in zip(raw['comparisons'],raw['direct'],raw['graph']):raw_comparison(c,a,b)
            assert len(raw['direct'])==len(raw['graph'])==len(raw['comparisons'])
            assert len(raw['direct_repeat'])==len(raw['direct_repeat_comparisons'])==len(raw['comparisons'])
            for c,a,b in zip(raw['direct_repeat_comparisons'],raw['direct'],raw['direct_repeat']):raw_comparison(c,a,b)
    assert seen==expected
    totals={};passed=True
    for field in ('graph_direct','retained_after_five_steps'):
        assert len(probe[field])==len(expected) and {key(r) for r in probe[field]}==expected
        for r in probe[field]:
            n=(76,121,121,76)[r['request']];assert r['prompt_tokens']==n
            assert len(r['comparisons'])==(3 if r['method']==RUNNING else 4)
            for c,(shape,dtype) in zip(r['comparisons'],output_spec(r['method'],n)):
                assert c['shape']==shape and c['dtype']==dtype and c['values']==math.prod(shape)
                passed &= exact_comparison(c)
            if field=='graph_direct':
                assert len(r['direct_repeat_comparisons'])==len(r['comparisons'])
                for c,original in zip(r['direct_repeat_comparisons'],r['comparisons']):
                    assert (c['shape'],c['dtype'],c['values'],c['left_sha256'])==(original['shape'],original['dtype'],original['values'],original['left_sha256'])
                    passed &= exact_comparison(c)
            if field=='retained_after_five_steps':
                passed &= r['independent_step_outputs']
                assert [c['left_sha256'] for c in r['comparisons']]==[c['right_sha256'] for c in rows[key(r)]['comparisons']]
        cs=[c for r in probe[field] for c in r['comparisons']]
        totals[field]=dict(values=sum(c['values'] for c in cs),matching_bits=sum(c['matching_bits'] for c in cs),max_abs_finite=max(c['max_abs_finite'] for c in cs))
    cs=[c for r in probe['graph_direct'] for c in r['direct_repeat_comparisons']]
    totals['direct_repeated']=dict(values=sum(c['values'] for c in cs),matching_bits=sum(c['matching_bits'] for c in cs),max_abs_finite=max(c['max_abs_finite'] for c in cs))
    assert [r['scenario'] for r in probe['reports']]==list(SCENARIOS)
    for item in probe['reports']:
        r=item['replay'];s=item['scenario']
        audit_report(r,[76,121],1.2 if s=='penalty' else 0.,s=='finished_flags',synthetic_probe=True)
        assert r['requests']==4 and r['replays']==40
        assert r['staged_bytes']==2*r['input_logical_bytes']
    return dict(exact=passed,**totals)


def audit_beam_phase(setup,summary,ownership,rows,candidate):
    before,after=setup['beam_replay'],summary['beam_replay']
    if not candidate:
        assert before is after is None and ownership==[]
        assert setup['score_ownership_checks']==summary['score_ownership_checks']==0
        return None
    lengths=sorted({r['prompt_tokens'] for r in rows})
    for r in (before,after):audit_report(r,lengths)
    assert before['events']==after['events'],'Measured beam capture'
    requests=len(rows)
    assert after['replays']-before['replays']==10*requests
    assert after['requests']-before['requests']==requests
    staged=sum(e['input_bytes'] for row in rows for e in before['events'] if e['prompt_tokens']==row['prompt_tokens'])
    assert after['staged_bytes']-before['staged_bytes']==staged
    assert summary['score_ownership_checks']-setup['score_ownership_checks']==requests
    assert len(ownership)==summary['score_ownership_checks']
    assert setup['score_ownership_checks']==2*len(setup['warmup_cohort_indices'])
    for row in ownership:
        assert row['steps']==row['distinct_saved_score_storages']==5
        assert row['helper_calls']==10 and row['distinct_helper_output_storages']==35
        assert row['saved_score_shapes']==[[50,152265]]*5
        assert row['saved_outputs_independent_of_graphs'] and row['lease_open_during_extraction']
        assert len(row['final_output_comparisons'])==3
        specs=[([50,row['prompt_tokens']+5],'torch.int64'),([50],'torch.float32'),([50,5],'torch.int32')]
        for c,(shape,dtype) in zip(row['final_output_comparisons'],specs):
            assert c['values']==math.prod(shape) and c['shape']==shape and c['dtype']==dtype
    assert setup['combined_setup_and_warmup_s']>=after['capture_and_warmup_s']
    cs=[c for row in ownership for c in row['final_output_comparisons']]
    return dict(after,final_outputs_exact=all(exact_comparison(c) for c in cs),
        final_output_comparison=dict(values=sum(c['values'] for c in cs),matching_bits=sum(c['matching_bits'] for c in cs),max_abs_finite=max(c['max_abs_finite'] for c in cs)))
