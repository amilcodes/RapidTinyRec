"""Experimental beam gather directly into the next replay graph's KV inputs.

Use only around one sequential generate() on the model bound to the pool.
Every returned cache is already reordered; no lazy cache semantics are exposed
to Hugging Face. The final unused cache, and shapes not captured during warmup,
follow ordinary reordering. Captured model outputs remain untouched.
"""
import contextlib
from unittest.mock import patch

import torch

from .decode_replay import cache_from_pairs, cache_pairs


def cache_spec(pairs):
    # Sources can have a different layout: index_select writes the destination
    # layout. The replay forward still validates its full shape/stride signature.
    return tuple(tuple((tuple(x.shape), x.dtype, x.device) for x in pair) for pair in pairs)


class ReorderIntoReplay:
    def __init__(self, pool):
        self.pool = pool
        self.targets = {}
        self.indexed_entries = -1
        self.direct_calls = 0
        self.fallback_calls = 0
        self.kv_tensors_written = 0
        self.kv_bytes_written = 0

    def refresh(self):
        if len(self.pool.entries) == self.indexed_entries:
            return
        targets = {}
        for buffers, _, _ in self.pool.entries.values():
            key = cache_spec(buffers.pairs)
            if key in targets:
                raise ValueError('Multiple replay buffers for the same KV shape')
            targets[key] = buffers.pairs
        self.targets = targets
        self.indexed_entries = len(self.pool.entries)

    @torch.inference_mode()
    def reorder(self, cache, beam_idx):
        self.refresh()
        sources = cache_pairs(cache)
        targets = self.targets.get(cache_spec(sources))
        if targets is None:
            self.fallback_calls += 1
            cache.reorder_cache(beam_idx)
            return cache
        flat_sources = [x for pair in sources for x in pair]
        flat_targets = [x for pair in targets for x in pair]
        # The pinned HF beam loop stores its ancestry in int32; index_select
        # accepts both supported integer widths without a conversion kernel.
        if beam_idx.ndim != 1 or beam_idx.dtype not in (torch.int32, torch.int64) or any(
                x.shape[0] != beam_idx.numel() or x.device != beam_idx.device for x in flat_targets):
            raise ValueError('Expected one on-device parent index per beam')
        # Out-of-place gather is essential for duplicated parents. Reject any
        # overlapping storage, even views with different starting addresses.
        source_storage = {x.untyped_storage().data_ptr() for x in flat_sources}
        if any(x.untyped_storage().data_ptr() in source_storage for x in flat_targets):
            raise ValueError('Direct reorder source aliases a replay destination')
        for source, target in zip(flat_sources, flat_targets):
            torch.index_select(source, 0, beam_idx, out=target)
        self.direct_calls += 1
        self.kv_tensors_written += len(flat_targets)
        self.kv_bytes_written += sum(x.numel() * x.element_size() for x in flat_targets)
        return cache_from_pairs(targets)

    @contextlib.contextmanager
    def install(self, model):
        if hasattr(model, '_reorder_cache'):
            raise ValueError('Model already supplies a custom cache reorder')
        # The pinned HF beam loop installs this returned cache in model_kwargs.
        with patch.object(model, '_reorder_cache', self.reorder, create=True):
            yield

    def report(self):
        return {'direct_calls': self.direct_calls, 'fallback_calls': self.fallback_calls,
                'kv_tensors_written': self.kv_tensors_written, 'kv_bytes_written': self.kv_bytes_written}
