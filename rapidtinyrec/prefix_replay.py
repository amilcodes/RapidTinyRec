"""CUDA replay and direct suffix reordering for the physical prefix cache."""
import contextlib
from unittest.mock import patch

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from .decode_replay import DecodeReplayPool, ReplayBuffers, TENSOR_KEYS, FIXED_KEYS, tensor_spec
from .shared_prefix import PrefixCache, split_pairs


def signature(kwargs):
    allowed = set(TENSOR_KEYS) | FIXED_KEYS | {'past_key_values', 'attention_mask'}
    if set(kwargs)-allowed or kwargs.get('inputs_embeds') is not None:
        raise ValueError('Unsupported prefix replay arguments')
    mask = kwargs.get('attention_mask')
    if not isinstance(mask, dict) or set(mask) != {'full_attention'} or mask['full_attention'] is not None:
        raise ValueError('Expected validated native GQA mask specialization')
    if not kwargs.get('use_cache') or not kwargs.get('return_dict') or kwargs.get('logits_to_keep') != 1 or kwargs.get('output_attentions') or kwargs.get('output_hidden_states'):
        raise ValueError('Expected cached single-token inference without auxiliary outputs')
    ids = kwargs['input_ids']
    if ids.ndim != 2 or ids.shape[1] != 1 or kwargs['position_ids'].shape != ids.shape or kwargs['cache_position'].shape != (1,):
        raise ValueError('Unsupported token or position shape')
    prefixes, suffixes = split_pairs(kwargs['past_key_values'])
    n, s = prefixes[0][0].shape[2], suffixes[0][0].shape[2]
    for prefix, suffix in zip(prefixes, suffixes):
        for p, t in zip(prefix, suffix):
            if p.shape != (1,t.shape[1],n,t.shape[3]) or t.shape != (ids.shape[0],p.shape[1],s,p.shape[3]) or not p.is_contiguous() or not t.is_contiguous():
                raise ValueError('Inconsistent prefix/suffix cache layout')
    return (tuple((k,tensor_spec(kwargs[k])) for k in TENSOR_KEYS),
            tuple(tuple(tensor_spec(x) for x in p) for p in prefixes+suffixes),
            tuple((k,kwargs[k]) for k in sorted(FIXED_KEYS) if k in kwargs))


class PrefixBuffers(ReplayBuffers):
    def __init__(self, kwargs, arena):
        self.signature = signature(kwargs)
        self.tensors = {k:kwargs[k].detach().clone() for k in TENSOR_KEYS}
        prefix, suffix = split_pairs(kwargs['past_key_values'])
        self.prefixes, self.pairs = arena.views(prefix[0][0].shape[2], suffix[0][0].shape[2])
        self.fixed = {k:kwargs[k] for k in FIXED_KEYS if k in kwargs}
        self.load(kwargs)

    def kwargs(self):
        return {**self.tensors, **self.fixed, 'attention_mask':{'full_attention':None},
                'past_key_values':PrefixCache(self.prefixes,self.pairs)}

    def load(self, kwargs):
        if signature(kwargs) != self.signature:
            raise ValueError('Prefix replay input signature changed')
        prefix, suffix = split_pairs(kwargs['past_key_values'])
        if any(x is not y for p,q in zip(prefix,self.prefixes) for x,y in zip(p,q)):
            raise ValueError('Request prefix is not in the shared arena')
        for k in TENSOR_KEYS:
            self.tensors[k].copy_(kwargs[k])
        for source, target in zip(suffix,self.pairs):
            for x,y in zip(source,target):
                if x is not y:
                    y.copy_(x)

    def inputs(self):
        return list(self.tensors.values())+[x for p in self.prefixes+self.pairs for x in p]


class PrefixOutput:
    def __init__(self, output):
        self.logits = output.logits
        self.prefixes, self.pairs = split_pairs(output.past_key_values)

    def fresh(self):
        return CausalLMOutputWithPast(logits=self.logits,past_key_values=PrefixCache(self.prefixes,self.pairs))

    def cache_length(self):
        return self.prefixes[0][0].shape[2]+self.pairs[0][0].shape[2]


class PrefixReplayPool(DecodeReplayPool):
    signature = staticmethod(signature)
    buffer_class = PrefixBuffers
    output_class = PrefixOutput

    def report(self):
        result = super().report()
        result['physical_prefix'] = self.arena.report()
        result['output_suffix_bytes'] = sum(x.numel()*x.element_size() for _,_,out in self.entries.values() for p in out.pairs for x in p)
        result['scope'] = 'Physical prefix shared across beams and graphs; suffixes alone are reordered. Dense attention K/V remains temporary per layer. Sequential inference only.'
        return result


class ReorderSuffix:
    def __init__(self, pool):
        self.pool = pool
        self.direct_calls = self.final_calls = self.suffix_bytes_written = 0

    @torch.inference_mode()
    def reorder(self, cache, beam_idx):
        prefixes, sources = split_pairs(cache)
        length = sources[0][0].shape[2]
        if beam_idx.ndim != 1 or beam_idx.dtype not in (torch.int32,torch.int64) or any(
                x.shape[0] != beam_idx.numel() or x.device != beam_idx.device for p in sources for x in p):
            raise ValueError('Expected one parent index per beam on the cache device')
        if length > self.pool.arena.max_suffix_tokens:
            # HF still reorders the final unused cache. Preserve its semantics.
            self.final_calls += 1
            cache.reorder_cache(beam_idx)
            return cache
        _, targets = self.pool.arena.views(prefixes[0][0].shape[2], length)
        if length:
            pointers = {x.untyped_storage().data_ptr() for p in sources for x in p}
            if any(x.untyped_storage().data_ptr() in pointers for p in targets for x in p):
                raise ValueError('Suffix gather source aliases its destination')
            for source,target in zip(sources,targets):
                for x,y in zip(source,target):
                    torch.index_select(x,0,beam_idx,out=y)
            self.suffix_bytes_written += sum(x.numel()*x.element_size() for p in targets for x in p)
        self.direct_calls += 1
        return PrefixCache(prefixes,targets)

    @contextlib.contextmanager
    def install(self, model):
        if hasattr(model,'_reorder_cache'):
            raise ValueError('Model already supplies a custom cache reorder')
        with patch.object(model,'_reorder_cache',self.reorder,create=True):
            yield

    def report(self):
        return {'direct_calls':self.direct_calls,'fallback_calls':self.final_calls,
                'suffix_bytes_written':self.suffix_bytes_written,'prefix_bytes_reordered':0}
