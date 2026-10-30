"""CUDA replay contracts for two independent request-major beam blocks.

Only the generic graph capture/scheduling machinery is shared with serial
replay. KV mapping, input ownership and fresh output caches are wave-specific.
"""
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from .decode_replay import DecodeReplayPool,ReplayBuffers,TENSOR_KEYS,FIXED_KEYS,tensor_spec
from .beam_wave_arena import split_wave,borrowed_wave


def signature(kwargs):
    allowed=set(TENSOR_KEYS)|FIXED_KEYS|{'past_key_values','attention_mask'}
    if set(kwargs)-allowed or kwargs.get('inputs_embeds') is not None:
        raise ValueError('Unsupported wave replay arguments')
    mask=kwargs.get('attention_mask')
    if not isinstance(mask,dict) or set(mask)!={'full_attention'} or mask['full_attention'] is not None:
        raise ValueError('Wave replay requires validated native-GQA mask specialization')
    if (not kwargs.get('use_cache') or not kwargs.get('return_dict') or kwargs.get('logits_to_keep')!=1
            or kwargs.get('output_attentions') or kwargs.get('output_hidden_states')):
        raise ValueError('Cached last-token inference without auxiliary outputs required')
    cache=kwargs['past_key_values'];prefix,suffix=split_wave(cache)
    ids=kwargs['input_ids'];rows=2*cache.layout.beams
    if ids.shape!=(rows,1) or kwargs['position_ids'].shape!=ids.shape or kwargs['cache_position'].shape!=(1,):
        raise ValueError('Wave token/position shapes must retain both beam blocks')
    if any(kwargs[k].device!=ids.device or kwargs[k].dtype!=torch.int64 for k in TENSOR_KEYS):
        raise ValueError('Wave token/position dtype or device changed')
    n=prefix[0][0].shape[2];s=suffix[0][0].shape[2]
    for p_pair,s_pair in zip(prefix,suffix):
        if p_pair[0].shape!=p_pair[1].shape or s_pair[0].shape!=s_pair[1].shape:
            raise ValueError('Wave key/value shapes differ')
        for p,t in zip(p_pair,s_pair):
            if (p.ndim!=4 or t.ndim!=4 or n<1 or p.shape!=(2,t.shape[1],n,t.shape[3])
                    or t.shape!=(rows,p.shape[1],s,p.shape[3]) or not p.is_contiguous() or not t.is_contiguous()
                    or p.dtype!=t.dtype or p.device!=ids.device or t.device!=ids.device):
                raise ValueError('Inconsistent wave prefix/suffix layout')
    return (cache.layout,tuple((k,tensor_spec(kwargs[k])) for k in TENSOR_KEYS),
            tuple(tuple(tensor_spec(x) for x in pair) for pair in prefix+suffix),
            tuple((k,kwargs[k]) for k in sorted(FIXED_KEYS) if k in kwargs))


class WaveBuffers(ReplayBuffers):
    def __init__(self,kwargs,arena):
        self.signature=signature(kwargs)
        self.arena=arena;self.layout=arena.layout
        prefix,suffix=arena.validate_prefix(kwargs['past_key_values'])
        self.prefixes,self.pairs=arena.views(prefix[0][0].shape[2],suffix[0][0].shape[2])
        self.tensors={k:kwargs[k].detach().clone() for k in TENSOR_KEYS}
        self.fixed={k:kwargs[k] for k in FIXED_KEYS if k in kwargs}
        self.load(kwargs)

    def kwargs(self):
        return {**self.tensors,**self.fixed,'attention_mask':{'full_attention':None},
                'past_key_values':borrowed_wave(self.prefixes,self.pairs,self.layout)}

    def load(self,kwargs):
        if signature(kwargs)!=self.signature:raise ValueError('Wave replay signature changed')
        _,suffix=self.arena.validate_prefix(kwargs['past_key_values'])
        # Reject shifted overlapping views before copying anything. Identity is
        # the only permitted alias: direct reorder may have filled this target.
        for srcs,dsts in zip(suffix,self.pairs):
            for src,dst in zip(srcs,dsts):
                if src is not dst and src.numel() and src.untyped_storage().data_ptr()==dst.untyped_storage().data_ptr():
                    raise ValueError('Wave staging source overlaps destination')
        for k in TENSOR_KEYS:self.tensors[k].copy_(kwargs[k])
        for srcs,dsts in zip(suffix,self.pairs):
            for src,dst in zip(srcs,dsts):
                if src is not dst:dst.copy_(src)

    def inputs(self):
        return list(self.tensors.values())+[x for pair in self.prefixes+self.pairs for x in pair]


class WaveOutput:
    def __init__(self,output):
        self.logits=output.logits
        self.layout=output.past_key_values.layout
        self.prefixes,self.pairs=split_wave(output.past_key_values)
        if self.logits.ndim!=3 or self.logits.shape[:2]!=(2*self.layout.beams,1):
            raise ValueError('Wave graph logits must retain request-major last-token rows')

    def fresh(self):
        return CausalLMOutputWithPast(logits=self.logits,
            past_key_values=borrowed_wave(self.prefixes,self.pairs,self.layout))

    def cache_length(self):
        return self.prefixes[0][0].shape[2]+self.pairs[0][0].shape[2]


class WaveReplayPool(DecodeReplayPool):
    signature=staticmethod(signature)
    buffer_class=WaveBuffers
    output_class=WaveOutput

    def report(self):
        result=super().report()
        result['physical_prefix']=self.arena.report()
        result['output_suffix_bytes']=sum(x.numel()*x.element_size() for _,_,out in self.entries.values()
                                          for pair in out.pairs for x in pair)
        result['scope']='Two request-major beam blocks; two physical prefixes, independent mutable suffix rows. Sequential wave use only; staging/reorder/HF score clones remain timed.'
        return result
