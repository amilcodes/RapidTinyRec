"""Experimental CUDA graph replay for exact-length, single-token GQA decoding.

One graph owns each input shape. Input copies and fresh cache containers remain
inside request timing. This is not a physically shared-prefix cache. A pool is
bound to one inference-only model and one sequential CUDA stream; callers must
consume outputs before replaying the same shape again.
"""
import time

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.modeling_outputs import CausalLMOutputWithPast


TENSOR_KEYS = ('input_ids', 'position_ids', 'cache_position')
FIXED_KEYS = {'use_cache', 'return_dict', 'logits_to_keep',
              'output_attentions', 'output_hidden_states', 'inputs_embeds'}


def cache_pairs(cache):
    if type(cache) is not DynamicCache or cache.get_seq_length() <= 0:
        raise ValueError('Replay requires a populated DynamicCache')
    pairs=[]
    for layer in cache.layers:
        if type(layer) is not DynamicLayer or layer.keys is None or layer.values is None:
            raise ValueError('Only populated full-attention DynamicLayers are supported')
        pairs.append((layer.keys,layer.values))
    return tuple(pairs)


def cache_from_pairs(pairs):
    # This constructor path fills the default empty layer instead of appending
    # behind it, and references each tensor without a copy or concatenation.
    return DynamicCache.from_legacy_cache(pairs)


def tensor_spec(tensor):
    return (tuple(tensor.shape),tuple(tensor.stride()),tensor.dtype,tensor.device)


def signature(kwargs):
    allowed=set(TENSOR_KEYS)|FIXED_KEYS|{'past_key_values','attention_mask'}
    if set(kwargs)-allowed:
        raise ValueError(f'Unsupported replay arguments: {sorted(set(kwargs)-allowed)}')
    # Generation supplies this keyword even on token-ID decode steps. Preserve
    # the explicit None for the compiled call; embedding inputs need a different
    # tensor staging contract and remain unsupported.
    if kwargs.get('inputs_embeds') is not None:
        raise ValueError('Replay requires token IDs, not embedding inputs')
    mask=kwargs.get('attention_mask')
    if not isinstance(mask,dict) or set(mask)!={'full_attention'} or mask['full_attention'] is not None:
        raise ValueError('Replay requires the validated native-GQA mask specialization')
    if not kwargs.get('use_cache') or not kwargs.get('return_dict') or kwargs.get('logits_to_keep')!=1:
        raise ValueError('Expected cached last-token logits with a structured output')
    if kwargs.get('output_attentions') or kwargs.get('output_hidden_states'):
        raise ValueError('Auxiliary outputs are unsupported')
    ids=kwargs['input_ids']
    if ids.ndim!=2 or ids.shape[1]!=1:
        raise ValueError('Only one-token cached decoding is supported')
    if kwargs['position_ids'].shape!=ids.shape or kwargs['cache_position'].shape!=(1,):
        raise ValueError('Unsupported position shapes')
    pairs=cache_pairs(kwargs['past_key_values'])
    length=kwargs['past_key_values'].get_seq_length()
    for pair in pairs:
        for value in pair:
            if value.ndim!=4 or value.shape[0]!=ids.shape[0] or value.shape[-2]!=length:
                raise ValueError('Cache lengths or batch dimensions differ')
    return (tuple((key,tensor_spec(kwargs[key])) for key in TENSOR_KEYS),
            tuple(tuple(tensor_spec(x) for x in pair) for pair in pairs),
            tuple((key,kwargs[key]) for key in sorted(FIXED_KEYS) if key in kwargs))


class ReplayBuffers:
    """Stable tensors are independent of any cache containers a forward mutates."""
    def __init__(self,kwargs,arena=None):
        self.signature=signature(kwargs)
        self.tensors={k:kwargs[k].detach().clone() for k in TENSOR_KEYS}
        incoming=cache_pairs(kwargs['past_key_values'])
        self.pairs=(arena.views(incoming) if arena is not None else
                    tuple(tuple(x.detach().clone() for x in pair) for pair in incoming))
        self.fixed={k:kwargs[k] for k in FIXED_KEYS if k in kwargs}
        if arena is not None:
            self.load(kwargs)

    def kwargs(self):
        return {**self.tensors,**self.fixed,'attention_mask':{'full_attention':None},
                'past_key_values':cache_from_pairs(self.pairs)}

    def load(self,kwargs):
        if signature(kwargs)!=self.signature:
            raise ValueError('Replay input shape, layout or settings changed')
        for k in TENSOR_KEYS:
            self.tensors[k].copy_(kwargs[k])
        # Keep these costs visible in the first pilot. A later ablation can
        # batch copies or fuse reordering with input staging if they dominate.
        for destination,source in zip(self.pairs,cache_pairs(kwargs['past_key_values'])):
            for target,value in zip(destination,source):
                # Direct beam reordering can already have filled this exact
                # tensor. Identity is deliberately stricter than data_ptr().
                if target is not value:
                    target.copy_(value)

    def num_bytes(self):
        tensors=list(self.tensors.values())+[x for p in self.pairs for x in p]
        return sum(x.numel()*x.element_size() for x in tensors)


class ReplayOutput:
    """Borrow graph tensors through disposable HF cache containers."""
    def __init__(self,output):
        self.logits=output.logits
        self.pairs=cache_pairs(output.past_key_values)

    def fresh(self):
        return CausalLMOutputWithPast(logits=self.logits,past_key_values=cache_from_pairs(self.pairs))


class DecodeReplayPool:
    def __init__(self,forward,max_graphs=40,arena=None):
        self.forward=forward
        self.arena=arena
        self.max_graphs=max_graphs
        self.entries={}
        self.events=[]
        self.frozen=False
        self.replays=0

    @torch.inference_mode()
    def prepare(self,kwargs):
        if self.frozen:
            raise RuntimeError('Graph capture after freeze is forbidden')
        key=signature(kwargs)
        if key in self.entries:
            return
        if len(self.entries)>=self.max_graphs:
            raise RuntimeError('Graph count exceeds the explicit memory budget')
        device=kwargs['input_ids'].device
        if device.type!='cuda':
            raise ValueError('CUDA graph capture requires CUDA inputs')
        torch.cuda.synchronize(device)
        started=time.perf_counter()
        before=torch.cuda.memory_allocated(device)
        buffers=ReplayBuffers(kwargs,self.arena)
        stream=torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                warm=self.forward(**buffers.kwargs())
                del warm
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            output=self.forward(**buffers.kwargs())
        torch.cuda.synchronize(device)
        captured=ReplayOutput(output)
        if self.arena is not None:
            self.arena.check_output(captured)
        if captured.pairs[0][0].shape[-2]!=kwargs['past_key_values'].get_seq_length()+1:
            raise AssertionError('Capture did not advance the cache by exactly one token')
        self.entries[key]=(buffers,graph,captured)
        self.events.append({'cache_tokens':kwargs['past_key_values'].get_seq_length(),
            'batch':kwargs['input_ids'].shape[0],'capture_and_warmup_s':time.perf_counter()-started,
            'input_buffer_bytes':buffers.num_bytes(),
            'allocated_delta_bytes':torch.cuda.memory_allocated(device)-before})

    def freeze(self):
        if not self.entries:
            raise RuntimeError('No graphs captured')
        self.frozen=True

    @torch.inference_mode()
    def __call__(self,**kwargs):
        key=signature(kwargs)
        if key not in self.entries:
            if self.frozen:
                raise RuntimeError('Uncaptured decode shape during measurement')
            self.prepare(kwargs)
        buffers,graph,output=self.entries[key]
        buffers.load(kwargs)
        graph.replay()
        self.replays+=1
        return output.fresh()

    def report(self):
        tensors=[x for buffers,_,_ in self.entries.values()
                 for x in list(buffers.tensors.values())+[v for pair in buffers.pairs for v in pair]]
        storage={(x.device,x.untyped_storage().data_ptr()):x.untyped_storage().nbytes() for x in tensors}
        return {'graphs':len(self.entries),'frozen':self.frozen,'replays':self.replays,
            'capture_and_warmup_s':sum(e['capture_and_warmup_s'] for e in self.events),
            'input_buffer_bytes':sum(e['input_buffer_bytes'] for e in self.events),
            'unique_input_storage_bytes':sum(storage.values()),
            'shared_input_arena':self.arena is not None,
            'events':self.events,
            'scope':'Experimental model-forward replay only. Input staging, cache containers, beam search and constraints remain timed. Independent graph output pools; optional shared KV input storage. Sequential use only.'}
