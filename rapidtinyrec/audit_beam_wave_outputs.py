"""Independent metadata/numerical-shape checks for saved pair output records.

This small checker does not load a model, bootstrap users, certify SID validity,
or replace the allocated source/dispatch/deadline/quality audit.
"""
import math
import struct
from .beam_wave_protocol import validate_timing,validate_phase_rows


def fp32(value):
    if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('Finite score required')
    try:return struct.pack('!f',value)
    except (OverflowError,struct.error) as exc:raise ValueError('Score outside FP32') from exc


def validate_request(row,user,slot,batch_requests,vocab_size):
    if type(vocab_size) is not int or vocab_size<1:raise ValueError('Explicit full vocabulary size required')
    if (row['cohort_index']!=user or row['request_slot']!=slot or row['original_batch_requests']!=batch_requests
            or row['original_beam_row_offset']!=slot*50):
        raise ValueError('Original request/beam identity differs')
    fields=('token_ids','sequence_logp','transition_logp','beam_indices','global_beam_indices')
    if any(len(row[k])!=50 for k in fields):raise ValueError('Original50ranks required')
    for tokens,score,steps,local,glob in zip(*(row[k] for k in fields)):
        if len(tokens)!=5 or any(type(x) is not int or x<0 for x in tokens):raise ValueError('Five token transitions required')
        if len(steps)!=5 or len(local)!=5 or len(glob)!=5:raise ValueError('Five score/ancestry transitions required')
        total=0.
        for step in steps:total=struct.unpack('!f',fp32(total+struct.unpack('!f',fp32(step))[0]))[0]
        if fp32(total)!=fp32(score):raise ValueError('Original five-step FP32 score sum differs')
        for a,b in zip(local,glob):
            if type(a) is not int or type(b) is not int or not -1<=a<50:
                raise ValueError('Invalid local ancestry')
            if b!=(-1 if a==-1 else a+slot*50):raise ValueError('Global ancestry crosses or was relabeled across requests')
    ptrs=row['saved_score_storage_ptrs'];graph=row['graph_logit_storage_ptrs'];extents=row['score_row_extents']
    if (len(ptrs)!=5 or len(set(ptrs))!=5 or any(type(p) is not int or p<=0 for p in ptrs)
            or len(set(graph))!=len(graph) or any(type(p) is not int or p<=0 for p in graph) or set(ptrs)&set(graph)):
        raise ValueError('HF score storage must be independently owned and separate from graph logits')
    if row['saved_score_dtypes']!=['torch.float32']*5 or len(extents)!=5:raise ValueError('Five FP32 score storages required')
    span=50*vocab_size*4
    for pointer,extent in zip(ptrs,extents):
        if extent['storage_ptr']!=pointer:raise ValueError('Score extent pointer differs')
        begin,end,size=[extent[k] for k in ('byte_start','byte_end','storage_bytes')]
        if (any(type(x) is not int for x in (begin,end,size)) or not 0<=begin<end<=size
                or begin%4 or end-begin!=span or size<batch_requests*span):
            raise ValueError('Full-vocabulary score row extent invalid')


def validate_trial(trial,vocab_size):
    if trial['schema']!='two-request-beam-wave-trial-v1':raise ValueError('Distinct pair trial required')
    timing=validate_timing(trial['timing']);requests=trial['requests'];wave=timing['variant']=='beam_wave_tf32'
    validate_request_group(requests,timing['cohort_indices'],wave,vocab_size)
    return timing


def validate_request_group(requests,users,wave,vocab_size):
    if len(requests)!=2 or len(users)!=2 or len(set(users))!=2:
        raise ValueError('Exactly two distinct saved user outputs required')
    for slot,(row,user) in enumerate(zip(requests,users)):
        validate_request(row,user,slot if wave else 0,2 if wave else 1,vocab_size)
    first,second=requests
    if wave:
        if first['saved_score_storage_ptrs']!=second['saved_score_storage_ptrs']:
            raise ValueError('Wave users must identify the same native HF score tensor')
        for a,b in zip(first['score_row_extents'],second['score_row_extents']):
            if a['byte_end']!=b['byte_start'] or a['storage_bytes']!=b['storage_bytes']:
                raise ValueError('Wave score request rows overlap or have a gap')
    elif set(first['saved_score_storage_ptrs'])&set(second['saved_score_storage_ptrs']):
        raise ValueError('Retained serial request outputs alias each other')
    return True


def validate_trials(phase,trials,vocab_size):
    timings=[validate_trial(t,vocab_size) for t in trials]
    validate_phase_rows(phase,timings)
    return dict(schema='two-request-beam-wave-output-check-v1',phase=phase,pairs=len(trials),requests=2*len(trials),
                original_score_shape_and_sum=True,request_ancestry_mapping=True,supplied_score_storage_ownership=True,
                sid_validity_checked=False,dispatch_checked=False,allocation_checked=False,full_audit_passed=None)


def compare_user_outputs(left,right):
    """Compare user-owned values after independently validating global ancestry.

    Request reversal changes global row offsets. Local ancestry comparison is
    meaningful only after validate_request has proven the explicit offset map.
    """
    if left['cohort_index']!=right['cohort_index']:raise ValueError('Cannot compare different users')
    scores=lambda row:[fp32(x) for x in row['sequence_logp']]+[fp32(x) for steps in row['transition_logp'] for x in steps]
    return dict(tokens_exact=left['token_ids']==right['token_ids'],
                scores_bit_exact=scores(left)==scores(right),local_ancestry_exact=left['beam_indices']==right['beam_indices'])
