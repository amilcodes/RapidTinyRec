import copy

import math

import unittest

def inventory(wave=True):
    pointer = 100

    def tensor(shape, dtype='torch.float32', owner=None, size=None):
        nonlocal pointer
        pointer += 1
        bytes_ = 8 if dtype == 'torch.int64' else 4
        strides = []
        n = 1
        for dim in reversed(shape):
            strides.insert(0, n)
            n *= dim
        return dict(shape=list(shape), stride=strides, dtype=dtype, device='cuda:0', storage_ptr=owner or pointer, storage_bytes=size or n * bytes_, byte_start=0, logical_bytes=n * bytes_, element_bytes=bytes_, contiguous=True)
    physical = 2 if wave else 1
    batch = 100 if wave else 50
    prefix = [[tensor([physical * 2 * 121 * 4]) for _ in range(2)]]
    suffix = [[tensor([batch * 2 * 3 * 4]) for _ in range(2)]]
    model = {'weight': tensor([4, 4])}
    graphs = []
    for length in (76, 121):
        for step in range(4):
            p = [[tensor([physical, 2, length, 4], owner=r['storage_ptr'], size=r['storage_bytes']) for r in prefix[0]]]
            s = [[tensor([batch, 2, step, 4], owner=r['storage_ptr'], size=r['storage_bytes']) for r in suffix[0]]]
            graphs.append(dict(index=len(graphs), cache_tokens=length + step, prefixes=p, suffixes=s, output_prefixes=copy.deepcopy(p), output_suffixes=[[tensor([batch, 2, step + 1, 4]) for _ in range(2)]], logits=tensor([batch, 1, 7]), tokens={k: tensor([1] if k == 'cache_position' else [batch, 1], dtype='torch.int64') for k in ('input_ids', 'position_ids', 'cache_position')}))
    return dict(schema='two-request-beam-wave-storage-v1', stage='before-measurement', variant='beam_wave_tf32' if wave else 'serial_pair_tf32', model_parameters=model, prefix_capacity=prefix, suffix_capacity=suffix, graphs=graphs)
