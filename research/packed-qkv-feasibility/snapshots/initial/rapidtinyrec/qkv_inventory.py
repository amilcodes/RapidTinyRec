"""Untimed producer snapshots and bitwise packed-weight construction evidence."""
import hashlib
import torch
from .immutable_inventory_v2 import tensor_items


def digest(t):
    # Contiguous copies are bounded per first-dimension row chunk. All expected
    # inventory tensors are contiguous; noncontiguous packed slices also work.
    h = hashlib.sha256()
    if t.ndim == 0:
        h.update(t.detach().reshape(1).view(torch.uint8).cpu().numpy().tobytes())
    else:
        rows = max(1, 1048576 // max(1, t[0].numel())) if t.shape[0] else 1
        for i in range(0, t.shape[0], rows):
            x = t.detach()[i:i+rows].contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
            h.update(memoryview(x))
    return h.hexdigest()


def tensors(model, inputs, masks, tables):
    result = {'model.parameter.'+n: t for n, t in model.named_parameters(remove_duplicate=False)}
    result.update({'model.buffer.'+n: t for n, t in model.named_buffers(remove_duplicate=False)})
    if set(inputs) != set(masks):
        raise ValueError('Input/mask coverage differs')
    result.update(tensor_items({i:(inputs[i], masks[i]) for i in inputs}, 'input'))
    result.update(tensor_items(vars(tables), 'table'))
    return result


def snapshot(model, inputs, masks, tables):
    return {n: dict(shape=list(t.shape), dtype=str(t.dtype), stride=list(t.stride()),
            logical_bytes=t.numel()*t.element_size(), sha256=digest(t),
            storage_offset=t.storage_offset(), storage_bytes=t.untyped_storage().nbytes(),
            storage_ptr=t.untyped_storage().data_ptr(), object_id=id(t), device=str(t.device))
            for n, t in sorted(tensors(model, inputs, masks, tables).items())}


def packing_evidence(model):
    """Compare every packed slice with the still-owned original parameter bytes."""
    original_ptrs = {t.untyped_storage().data_ptr() for t in model.parameters()}
    used = set()
    evidence = []
    for name, attention in model.named_modules():
        if not hasattr(attention, '_research_packed_qkv'):
            continue
        packed = attention._research_packed_qkv
        parts = [attention.q_proj, attention.k_proj, attention.v_proj]
        widths = [p.out_features for p in parts]
        if tuple(widths) != tuple(packed.widths):
            raise ValueError('Packed width/order differs')
        row = dict(module=name, widths=widths, extra_bytes=0, slices=[])
        for kind in ('weight', 'bias'):
            t = getattr(packed, kind)
            ptr = t.untyped_storage().data_ptr()
            if ptr in original_ptrs or ptr in used or not t.is_contiguous() or t.dtype != torch.bfloat16:
                raise ValueError('Packed storage aliases or has wrong layout/dtype')
            used.add(ptr)
            required_shape = (sum(widths), parts[0].in_features) if kind == 'weight' else (sum(widths),)
            if tuple(t.shape) != required_shape or t.storage_offset() != 0 or t.untyped_storage().nbytes() != t.numel()*t.element_size():
                raise ValueError('Packed storage extent differs')
            row['extra_bytes'] += t.numel()*t.element_size()
            offset = 0
            for source, width in zip(parts, widths):
                original = getattr(source, kind)
                section = t[offset:offset+width]
                a, b = digest(original), digest(section)
                if a != b or original.dtype != t.dtype or original.shape != section.shape:
                    raise ValueError('Packed bytes differ from original')
                row['slices'].append(dict(kind=kind, offset=offset, width=width,
                                          original_sha256=a, packed_sha256=b))
                offset += width
        evidence.append(row)
    if len(evidence) != model.config.num_hidden_layers:
        raise ValueError('Missing packed attention layer')
    return dict(layers=evidence, extra_bytes=sum(x['extra_bytes'] for x in evidence))
