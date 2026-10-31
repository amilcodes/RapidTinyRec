"""Unintegrated research primitive. Caller must independently prove beam lineage.

All allocation, copies, checks and concatenation belong inside the caller's
generation timer. The optional metadata sink is for untimed setup only.
"""
import torch


def _metadata(t):
    storage = t.untyped_storage()
    return dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype),
                device=str(t.device), storage_base=storage.data_ptr(),
                storage_bytes=storage.nbytes(), pointer=t.data_ptr(),
                offset=t.storage_offset(), pointer_mod16=t.data_ptr() % 16)


def _disjoint(a, b):
    return (a['storage_base'] + a['storage_bytes'] <= b['storage_base'] or
            b['storage_base'] + b['storage_bytes'] <= a['storage_base'])


def owned_block_log_softmax(scores, *, requests, vocabulary, site, step,
                           metadata_sink=None):
    """Normalize consecutive 50-row request blocks with fresh aligned inputs.

This helper neither establishes row lineage nor certifies CUDA exactness.
It deliberately has no generation entry point or default reduced vocabulary.
"""
    if type(scores) is not torch.Tensor:
        raise TypeError('Base Tensor required')
    if (type(requests) is not int or requests not in (1, 2) or
            type(vocabulary) is not int or vocabulary <= 0 or
            type(step) is not int or not 0 <= step < 5 or
            site not in ('first', 'second')):
        raise ValueError('Explicit request geometry/site/step required')
    if (scores.dtype != torch.float32 or scores.device.type not in ('cpu', 'cuda') or
            scores.requires_grad or scores.layout != torch.strided or
            scores.ndim != 2 or tuple(scores.shape) != (requests * 50, vocabulary) or
            not scores.is_contiguous()):
        raise ValueError('Contiguous inference FP32 request blocks required')
    if metadata_sink is not None and not callable(metadata_sink):
        raise TypeError('Metadata sink must be callable')
    source = _metadata(scores)
    outputs = []
    for index in range(requests):
        view = scores[index * 50:(index + 1) * 50]
        owned = torch.empty((50, vocabulary), dtype=torch.float32, device=scores.device)
        own_meta = _metadata(owned)
        if (own_meta['offset'] != 0 or own_meta['stride'] != [vocabulary, 1] or
                own_meta['pointer_mod16'] != 0 or not _disjoint(source, own_meta) or
                any(not _disjoint(_metadata(previous), own_meta) for previous in outputs)):
            raise ValueError('Fresh aligned input ownership failed')
        owned.copy_(view)
        normalized = torch.nn.functional.log_softmax(owned, dim=-1)
        normalized_meta = _metadata(normalized)
        if (not _disjoint(own_meta, normalized_meta) or
                not _disjoint(source, normalized_meta) or
                any(not _disjoint(_metadata(previous), normalized_meta) for previous in outputs)):
            raise ValueError('Native output ownership failed')
        if metadata_sink is not None:
            metadata_sink(dict(site=site, step=step, request_position=index,
                               source_slice=_metadata(view), owned_input=own_meta,
                               normalized_output=normalized_meta))
        outputs.append(normalized)
        del owned, view, normalized
    return torch.cat(outputs, dim=0)
