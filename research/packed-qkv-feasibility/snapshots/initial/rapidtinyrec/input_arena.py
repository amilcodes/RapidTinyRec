"""Reuse KV input capacity across mutually exclusive exact-length CUDA graphs.

One model, one sequential stream, no concurrent requests. Views have the same
contiguous layout as independent input clones. Outputs and graph private pools
remain independent. This does not share history storage across beam rows.
"""
import torch


class SharedInputArena:
    def __init__(self, max_cache_tokens):
        if max_cache_tokens < 1:
            raise ValueError('Positive capacity required')
        self.max_cache_tokens = max_cache_tokens
        self.layout = None
        self.storage = None

    def views(self, pairs):
        layout = tuple(tuple((x.shape[0], x.shape[1], x.shape[3], x.dtype, x.device)
                             for x in pair) for pair in pairs)
        if any(x.ndim != 4 or not x.is_contiguous() or not 0 < x.shape[2] <= self.max_cache_tokens
               for pair in pairs for x in pair):
            raise ValueError('Arena requires contiguous KV inputs within fixed capacity')
        if self.layout is None:
            self.layout = layout
            self.storage = tuple(tuple(torch.empty(x.shape[0]*x.shape[1]*self.max_cache_tokens*x.shape[3],
                                                  dtype=x.dtype, device=x.device) for x in pair) for pair in pairs)
        if layout != self.layout:
            raise ValueError('Arena batch, heads, layers, dtype or device changed')
        # Slicing a [B,H,max_length,D] allocation would retain max_length strides.
        # Slice a flat allocation first, then reshape to the exact input layout.
        return tuple(tuple(flat[:x.numel()].view(x.shape) for flat,x in zip(store,pair))
                     for store,pair in zip(self.storage,pairs))

    def check_output(self, output):
        pointers = {x.untyped_storage().data_ptr() for pair in self.storage for x in pair}
        if any(x.untyped_storage().data_ptr() in pointers
               for x in [output.logits]+[v for pair in output.pairs for v in pair]):
            raise ValueError('Captured output aliases reusable input storage')
