"""Stable two-request KV inputs and request-isolated direct suffix gathers.

Sequential waves only. Starting a new wave invalidates borrowed cache views.
Graph outputs must own suffix/logit storage; prefix views remain borrowed.
"""
import torch
from transformers.cache_utils import DynamicCache
from .beam_wave import WaveLayout, WavePrefixCache, WavePrefixLayer
from .decode_replay import cache_pairs


def split_wave(cache):
    if type(cache) is not WavePrefixCache or not cache.layers or any(
            type(layer) is not WavePrefixLayer or layer.layout != cache.layout for layer in cache.layers):
        raise ValueError('Expected a populated two-request wave cache')
    return (tuple((x.prefix_keys,x.prefix_values) for x in cache.layers),
            tuple((x.keys,x.values) for x in cache.layers))


def borrowed_wave(prefixes,suffixes,layout):
    if not prefixes or len(prefixes)!=len(suffixes):
        raise ValueError('Equal nonempty prefix/suffix layers required')
    # Explicit constructor for graph-owned/arena-owned tensors, no hidden clone.
    result=WavePrefixCache.__new__(WavePrefixCache)
    DynamicCache.__init__(result)
    result.layout=layout
    result.layers=[WavePrefixLayer(p,s,layout) for p,s in zip(prefixes,suffixes)]
    return result


class WaveArena:
    def __init__(self,max_prefix_tokens,max_suffix_tokens=3,beams=50):
        if any(type(x) is not int or x<1 for x in (max_prefix_tokens,max_suffix_tokens)):
            raise ValueError('Positive integer capacities required')
        self.layout=WaveLayout(beams)
        self.max_prefix_tokens=max_prefix_tokens;self.max_suffix_tokens=max_suffix_tokens
        self.storage=None;self.prefix_views={};self.suffix_views={}
        self.waves_loaded=0;self.prefix_bytes_copied=0

    def start_wave(self,prefill):
        incoming=cache_pairs(prefill);length=prefill.get_seq_length()
        if not 0<length<=self.max_prefix_tokens or any(
                x.ndim!=4 or x.shape[0]!=2 or x.shape[2]!=length for pair in incoming for x in pair):
            raise ValueError('Two complete equal-length histories within capacity required')
        if any(k.shape!=v.shape or k.dtype!=v.dtype or k.device!=v.device for k,v in incoming):
            raise ValueError('Inconsistent key/value layout')
        spec=tuple(tuple((x.shape[1],x.shape[3],x.dtype,x.device) for x in pair) for pair in incoming)
        if self.storage is not None and spec!=self.spec:
            raise ValueError('Wave arena layer layout changed')
        if self.storage is None:
            self.spec=spec
            self.prefix_storage=tuple(tuple(x.new_empty(2*x.shape[1]*self.max_prefix_tokens*x.shape[3])
                for x in pair) for pair in incoming)
            self.suffix_storage=tuple(tuple(x.new_empty(2*self.layout.beams*x.shape[1]*self.max_suffix_tokens*x.shape[3])
                for x in pair) for pair in incoming)
            self.storage=self.prefix_storage+self.suffix_storage
        prefix,suffix=self.views(length,0)
        for sources,targets in zip(incoming,prefix):
            for src,dst in zip(sources,targets):dst.copy_(src)
        self.waves_loaded+=1
        self.prefix_bytes_copied+=sum(x.numel()*x.element_size() for pair in prefix for x in pair)
        return borrowed_wave(prefix,suffix,self.layout)

    def views(self,prefix_length,suffix_length):
        if (self.storage is None or type(prefix_length) is not int or type(suffix_length) is not int
                or not 0<prefix_length<=self.max_prefix_tokens or not 0<=suffix_length<=self.max_suffix_tokens):
            raise ValueError('Wave shape outside arena capacity')
        if prefix_length not in self.prefix_views:
            self.prefix_views[prefix_length]=tuple(tuple(flat[:2*h*prefix_length*d].view(2,h,prefix_length,d)
                for flat,(h,d,_,_) in zip(store,spec)) for store,spec in zip(self.prefix_storage,self.spec))
        if suffix_length not in self.suffix_views:
            rows=2*self.layout.beams
            self.suffix_views[suffix_length]=tuple(tuple(flat[:rows*h*suffix_length*d].view(rows,h,suffix_length,d)
                for flat,(h,d,_,_) in zip(store,spec)) for store,spec in zip(self.suffix_storage,self.spec))
        return self.prefix_views[prefix_length],self.suffix_views[suffix_length]

    def validate_prefix(self,cache):
        prefix,suffix=split_wave(cache)
        if cache.layout!=self.layout:raise ValueError('Wave beam layout changed')
        expected,_=self.views(prefix[0][0].shape[2],0)
        if len(prefix)!=len(expected) or any(x is not y for a,b in zip(prefix,expected) for x,y in zip(a,b)):
            raise ValueError('Wave prefix does not belong to this arena')
        length=suffix[0][0].shape[2]
        for pair,spec in zip(suffix,self.spec):
            for x,(h,d,dtype,device) in zip(pair,spec):
                if x.shape!=(2*self.layout.beams,h,length,d) or x.dtype!=dtype or x.device!=device:
                    raise ValueError('Invalid wave suffix layout')
        return prefix,suffix

    def check_output(self,output):
        cache=borrowed_wave(output.prefixes,output.pairs,self.layout)
        self.validate_prefix(cache)
        inputs={x.untyped_storage().data_ptr() for pair in self.storage for x in pair}
        outputs=[output.logits]+[x for pair in output.pairs for x in pair]
        if any(x.untyped_storage().data_ptr() in inputs for x in outputs):
            raise ValueError('Wave output aliases arena inputs')
        pointers=[x.untyped_storage().data_ptr() for x in outputs]
        if len(set(pointers))!=len(pointers):raise ValueError('Wave outputs share mutable storage')

    def report(self):
        size=lambda pairs:sum(x.untyped_storage().nbytes() for pair in pairs for x in pair)
        return dict(waves_loaded=self.waves_loaded,requests_loaded=2*self.waves_loaded,
                    prefix_bytes_copied=self.prefix_bytes_copied,
                    prefix_capacity_bytes=size(self.prefix_storage),suffix_capacity_bytes=size(self.suffix_storage),
                    physical_prefix_batch=2,suffix_batch=2*self.layout.beams,
                    max_prefix_tokens=self.max_prefix_tokens,max_suffix_tokens=self.max_suffix_tokens)


class WaveReorder:
    def __init__(self,arena):
        self.arena=arena;self.direct_calls=0;self.final_calls=0;self.suffix_bytes_written=0

    @torch.inference_mode()
    def reorder(self,cache,parents):
        prefix,source=self.arena.validate_prefix(cache)
        self.arena.layout.validate_parents(parents)
        if any(x.device!=parents.device for pair in source for x in pair):
            raise ValueError('Parent device differs from suffix')
        length=source[0][0].shape[2]
        if length>self.arena.max_suffix_tokens:
            cache.reorder_cache(parents);self.final_calls+=1
            return cache
        _,target=self.arena.views(prefix[0][0].shape[2],length)
        if length:
            pointers={x.untyped_storage().data_ptr() for pair in source for x in pair}
            if any(x.untyped_storage().data_ptr() in pointers for pair in target for x in pair):
                raise ValueError('Wave gather source aliases destination')
            for a,b in zip(source,target):
                for x,y in zip(a,b):torch.index_select(x,0,parents,out=y)
            self.suffix_bytes_written+=sum(x.numel()*x.element_size() for pair in target for x in pair)
        self.direct_calls+=1
        return borrowed_wave(prefix,target,self.arena.layout)
