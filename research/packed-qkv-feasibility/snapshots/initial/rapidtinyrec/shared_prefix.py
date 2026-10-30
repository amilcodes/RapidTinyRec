"""One physical history per request; only generated suffixes have beam rows.

Attention still receives an exact-length contiguous dense tensor, assembled
temporarily inside each layer. This is a cache-storage experiment, not a new
attention kernel. Sequential inference on one model/stream only.
"""
import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

from .decode_replay import cache_pairs


class PrefixLayer(DynamicLayer):
    def __init__(self, prefix, suffix):
        super().__init__()
        self.prefix_keys, self.prefix_values = prefix
        self.keys, self.values = suffix

    def get_seq_length(self, cache_position=None):
        return self.prefix_keys.shape[-2] + self.keys.shape[-2]

    def update(self, key_states, value_states, cache_kwargs=None):
        # Never retain a dense history as cache output. The attention operation
        # and reduction shapes stay identical to the dense-cache control.
        self.keys = torch.cat((self.keys, key_states), dim=-2)
        self.values = torch.cat((self.values, value_states), dim=-2)
        batch = key_states.shape[0]
        return (torch.cat((self.prefix_keys.expand(batch, -1, -1, -1), self.keys), dim=-2),
                torch.cat((self.prefix_values.expand(batch, -1, -1, -1), self.values), dim=-2))

    def crop(self, max_length):
        raise NotImplementedError('Prefix cache supports beam decoding only')

    def batch_repeat_interleave(self, repeats):
        raise NotImplementedError('Prefix cache already has explicit beam suffixes')


class PrefixCache(DynamicCache):
    def __init__(self, prefixes, suffixes):
        super().__init__()
        self.layers = [PrefixLayer(p, s) for p, s in zip(prefixes, suffixes)]
        if not self.layers or len(prefixes) != len(suffixes):
            raise ValueError('Expected equal nonempty prefix/suffix layer lists')


def split_pairs(cache):
    if type(cache) is not PrefixCache or any(type(l) is not PrefixLayer for l in cache.layers):
        raise ValueError('Expected a populated PrefixCache')
    return (tuple((l.prefix_keys, l.prefix_values) for l in cache.layers),
            tuple((l.keys, l.values) for l in cache.layers))


class PrefixArena:
    """Stable single-row prefix and beam suffix capacity shared by all graphs."""
    def __init__(self, max_prefix_tokens, max_suffix_tokens=3):
        if max_prefix_tokens < 1 or max_suffix_tokens < 1:
            raise ValueError('Positive prefix and suffix capacities required')
        self.max_prefix_tokens = max_prefix_tokens
        self.max_suffix_tokens = max_suffix_tokens
        self.storage = None
        self.layout = None
        self.prefix_views = {}
        self.suffix_views = {}
        self.requests_loaded = 0
        self.prefix_bytes_copied = 0

    def start_request(self, cache, beams):
        incoming = cache_pairs(cache)
        length = cache.get_seq_length()
        layout = tuple(tuple((x.shape[1], x.shape[3], x.dtype, x.device) for x in p) for p in incoming)
        if not 0 < length <= self.max_prefix_tokens or any(
                x.ndim != 4 or x.shape[0] != 1 or x.shape[2] != length for p in incoming for x in p):
            raise ValueError('Expected one complete history within prefix capacity')
        if self.storage is None:
            self.layout, self.beams = layout, beams
            self.prefix_storage = tuple(tuple(torch.empty(x.shape[1]*self.max_prefix_tokens*x.shape[3],
                dtype=x.dtype, device=x.device) for x in p) for p in incoming)
            self.suffix_storage = tuple(tuple(torch.empty(beams*x.shape[1]*self.max_suffix_tokens*x.shape[3],
                dtype=x.dtype, device=x.device) for x in p) for p in incoming)
            self.storage = self.prefix_storage + self.suffix_storage
        if layout != self.layout or beams != self.beams:
            raise ValueError('Prefix arena layout or beam count changed')
        prefixes, suffixes = self.views(length, 0)
        for src, dst in zip(incoming, prefixes):
            for x, y in zip(src, dst):
                y.copy_(x)
        self.requests_loaded += 1
        self.prefix_bytes_copied += sum(x.numel()*x.element_size() for p in prefixes for x in p)
        return PrefixCache(prefixes, suffixes)

    def views(self, prefix_length, suffix_length):
        if self.storage is None or not 0 < prefix_length <= self.max_prefix_tokens or not 0 <= suffix_length <= self.max_suffix_tokens:
            raise ValueError('Cache shape outside arena capacity')
        if prefix_length not in self.prefix_views:
            self.prefix_views[prefix_length] = tuple(tuple(flat[:h*prefix_length*d].view(1,h,prefix_length,d)
                for flat,(h,d,_,_) in zip(store,layout)) for store,layout in zip(self.prefix_storage,self.layout))
        if suffix_length not in self.suffix_views:
            self.suffix_views[suffix_length] = tuple(tuple(flat[:self.beams*h*suffix_length*d].view(self.beams,h,suffix_length,d)
                for flat,(h,d,_,_) in zip(store,layout)) for store,layout in zip(self.suffix_storage,self.layout))
        return self.prefix_views[prefix_length], self.suffix_views[suffix_length]

    def check_output(self, output):
        pointers = {x.untyped_storage().data_ptr() for p in self.storage for x in p}
        if any(x.untyped_storage().data_ptr() in pointers for x in [output.logits]+[v for p in output.pairs for v in p]):
            raise ValueError('Graph suffix output aliases shared inputs')
        # Prefixes intentionally remain the same single-row tensors.
        if any(x.shape[0] != 1 or x.untyped_storage().data_ptr() not in pointers for p in output.prefixes for x in p):
            raise ValueError('Graph copied or expanded the persistent prefix')

    def report(self):
        size = lambda pairs: sum(x.untyped_storage().nbytes() for p in pairs for x in p)
        return {'requests_loaded': self.requests_loaded, 'prefix_bytes_copied': self.prefix_bytes_copied,
                'prefix_capacity_bytes': size(self.prefix_storage), 'suffix_capacity_bytes': size(self.suffix_storage),
                'max_prefix_tokens': self.max_prefix_tokens, 'max_suffix_tokens': self.max_suffix_tokens,
                'physical_prefix_batch': 1, 'suffix_batch': self.beams,
                'attention_materialization': 'Temporary dense exact-length K/V per layer; unchanged SDPA'}
