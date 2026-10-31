"""New intervention evidence builder. No CUDA CLI until allocation review.

Comparisons and device-to-host hashes are outside any future measured/captured
region. This module does not establish runtime beam lineage or model gains.
"""
import hashlib
import copy
import torch
from owned_blocks import owned_block_log_softmax,_metadata
from audit_live import inspect_live


def two_site(x, metadata_sink=None):
    first = owned_block_log_softmax(x, requests=2, vocabulary=x.shape[1],
                                   site='first', step=0, metadata_sink=metadata_sink)
    second = owned_block_log_softmax(first, requests=2, vocabulary=x.shape[1],
                                    site='second', step=0, metadata_sink=metadata_sink)
    return first, second


def _hash(x):
    return hashlib.sha256(x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def _comparison(a, b):
    if a.shape != b.shape or a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError('Comparison geometry/type differs')
    diff = int((a.view(torch.int32) != b.view(torch.int32)).sum().item())
    return dict(exact=diff == 0, differing_elements=diff, left_sha256=_hash(a), right_sha256=_hash(b))


def _shifted_copy(x, offset):
    storage = torch.empty(x.numel() + offset, dtype=torch.float32, device=x.device)
    view = storage[offset:].view_as(x)
    view.copy_(x)
    if storage.data_ptr() % 16 != 0 or view.data_ptr() % 16 != (offset * 4) % 16:
        raise ValueError('Declared source alignment not realized')
    return view


def compare_case(x, *, seed, offset, candidate=two_site, mode="eager"):
    """Two-mode comparison orchestration, all failures retained in records.

The fresh serial controls use native50-row operations with no intervention
copy inside either normalization. Control input staging is explicit setup.
"""
    if mode not in ('eager','cuda_graph'):raise ValueError('Unknown execution mode')
    if (type(seed) is not int or type(offset) is not int or offset not in range(4)
            or type(x) is not torch.Tensor or x.dtype != torch.float32
            or tuple(x.shape[:1]) != (100,) or x.ndim != 2 or not x.is_contiguous()
            or x.requires_grad or x.storage_offset() != offset
            or x.untyped_storage().data_ptr() % 16 != 0
            or x.data_ptr() % 16 != offset * 4):
        raise ValueError('Explicit shifted100-row FP32 case required')
    if not bool(torch.isfinite(x).all()): raise ValueError('Finite seeded logits required')
    source_saved = x.clone()
    serial_inputs = [torch.empty_like(x[:50]), torch.empty_like(x[50:])]
    for i,t in enumerate(serial_inputs):
        t.copy_(x[i*50:(i+1)*50])
        if t.storage_offset() != 0 or t.data_ptr() % 16: raise ValueError('Serial control alignment differs')
    def serial():
        a=[torch.nn.functional.log_softmax(t,dim=-1) for t in serial_inputs]
        b=[torch.nn.functional.log_softmax(t,dim=-1) for t in a]
        return torch.cat(a),torch.cat(b)
    reference=serial();control_repeat=serial()
    records=[];comparisons={}
    for i,site in enumerate(('first','second')):
        comparisons['control_repeat_'+site]=_comparison(reference[i],control_repeat[i])
    results={};snapshots={}
    reverse=_shifted_copy(torch.cat((x[50:],x[:50])),offset)
    reverse_saved=reverse.clone()
    for name,source in [('forward',x),('repeat',x),('reverse',reverse),('restored',x)]:
        observed=[]
        pair=candidate(source,observed.append)
        if not isinstance(pair,tuple) or len(pair)!=2:raise ValueError('Two sites required')
        if len(observed)!=4:raise ValueError('Both sites and both request ownership records required')
        records.append(dict(order=name,observations=observed))
        results[name]=pair;snapshots[name]=tuple(t.clone() for t in pair)
        for i,site in enumerate(('first','second')):
            aligned=torch.cat((pair[i][50:],pair[i][:50])) if name=='reverse' else pair[i]
            comparisons[name+'_'+site]=_comparison(reference[i],aligned)
    for name,pair in results.items():
        for i,site in enumerate(('first','second')):
            comparisons[name+'_retained_'+site]=_comparison(snapshots[name][i],pair[i])
    comparisons['source_untouched']=_comparison(source_saved,x)
    comparisons['reverse_source_untouched']=_comparison(reverse_saved,reverse)
    live={'source':x,'reverse':reverse,'source_saved':source_saved,'reverse_saved':reverse_saved}
    live.update({'serial_'+str(i):t for i,t in enumerate(serial_inputs)})
    for label,pair in [('reference',reference),('control_repeat',control_repeat)]:
        live.update({label+'_'+site:t for site,t in zip(('first','second'),pair)})
    for label,table in [('result',results),('snapshot',snapshots)]:
        for order,pair in table.items():live.update({label+'_'+order+'_'+site:t for site,t in zip(('first','second'),pair)})
    graph=None
    if mode=='cuda_graph':
        live.update(graph_input=candidate.static_input,graph_first=candidate.static_outputs[0],graph_second=candidate.static_outputs[1])
        graph=dict(bindings=copy.deepcopy(candidate.bindings),capture_observations=copy.deepcopy(candidate.observations),receipts=copy.deepcopy(candidate.receipts))
    spans=[(t.untyped_storage().data_ptr(),t.untyped_storage().nbytes()) for t in live.values()]
    disjoint=all(a+n<=b or b+m<=a for i,(a,n) in enumerate(spans) for b,m in spans[i+1:])
    identity=dict(source_sha256=_hash(source_saved),reverse_sha256=_hash(reverse_saved),
                  source_blocks=[_hash(source_saved[:50]),_hash(source_saved[50:])],
                  serial_blocks=[_hash(t) for t in serial_inputs],
                  reverse_blocks=[_hash(reverse_saved[:50]),_hash(reverse_saved[50:])])
    live_audit=inspect_live(live,width=x.shape[1],mode=mode)
    if live_audit['comparisons']!=comparisons:raise ValueError('Independent live comparisons differ from producer')
    finite=live_audit['all_finite']
    return dict(seed=seed,offset=offset,vocabulary=x.shape[1],mode=mode,
                live_numerical_audit=live_audit,input_identity=identity,live_tensors={k:_metadata(v) for k,v in live.items()},graph=graph,
                comparisons=comparisons,metadata=records,live_outputs_disjoint=disjoint,
                finite_outputs=finite,passed=all(c['exact'] for c in comparisons.values()) and disjoint and finite,
                model_executed=False,production_latency_claim=False)

if __name__=='__main__':
    raise SystemExit('Disabled: CUDA capture, independent audit, allocation/closure and fresh readiness pending')
