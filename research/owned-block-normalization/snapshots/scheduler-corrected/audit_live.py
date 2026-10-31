"""Independent in-allocation comparison of retained live tensors.

No producer imports, normalization, seeded generation or graph replay. This
checker runs before owners are released, outside any future latency timer.
"""
import hashlib
import torch


def inspect_live(live, *, width, mode):
    orders=('forward','repeat','reverse','restored');sites=('first','second')
    names={'source','reverse','source_saved','reverse_saved','serial_0','serial_1'}
    names|={p+'_'+s for p in ('reference','control_repeat') for s in sites}
    names|={p+'_'+o+'_'+s for p in ('result','snapshot') for o in orders for s in sites}
    if mode=='cuda_graph':names|={'graph_input','graph_first','graph_second'}
    elif mode!='eager':raise ValueError('Unknown live audit mode')
    if set(live)!=names:raise ValueError('Independent live audit owner inventory differs')
    devices=set();finite={};metadata={};hashes={}
    def digest(t):
        return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    for name,t in live.items():
        rows=50 if name.startswith('serial_') else 100
        if (type(t) is not torch.Tensor or t.dtype!=torch.float32 or t.requires_grad or
                tuple(t.shape)!=(rows,width) or not t.is_contiguous()):
            raise ValueError('Independent live audit tensor contract differs')
        devices.add(str(t.device));storage=t.untyped_storage()
        metadata[name]=dict(shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),device=str(t.device),
                            storage_base=storage.data_ptr(),storage_bytes=storage.nbytes(),pointer=t.data_ptr(),
                            offset=t.storage_offset(),pointer_mod16=t.data_ptr()%16)
        finite[name]=dict(elements=t.numel(),nonfinite_elements=int(torch.count_nonzero(~torch.isfinite(t)).item()))
        hashes[name]=digest(t)
    if len(devices)!=1:raise ValueError('Mixed live audit devices')
    def compare(left,right,reverse=False):
        a=live[left];b=live[right]
        if reverse:b=torch.cat((b[50:],b[:50]),dim=0)
        count=int(torch.count_nonzero(a.view(torch.int32).ne(b.view(torch.int32))).item())
        return dict(exact=count==0,differing_elements=count,left_sha256=hashes[left],
                    right_sha256=digest(b) if reverse else hashes[right])
    comparisons={}
    for site in sites:
        comparisons['control_repeat_'+site]=compare('reference_'+site,'control_repeat_'+site)
        for order in orders:
            comparisons[order+'_'+site]=compare('reference_'+site,'result_'+order+'_'+site,order=='reverse')
            comparisons[order+'_retained_'+site]=compare('snapshot_'+order+'_'+site,'result_'+order+'_'+site)
    comparisons['source_untouched']=compare('source_saved','source')
    comparisons['reverse_source_untouched']=compare('reverse_saved','reverse')
    return dict(schema='request-block-live-numerical-audit-v1',comparisons=comparisons,
                finite_counts=finite,owners=metadata,owner_sha256=hashes,
                all_finite=all(x['nonfinite_elements']==0 for x in finite.values()),
                all_comparisons_exact=all(x['exact'] for x in comparisons.values()),
                output_recomputed=False,execution_certified=False)
