"""Two independent requests per native HF beam wave; CPU preparation only.

Each request owns a block of beams. Prefix KV retains two physical prompt rows,
while generated suffix KV has two beam blocks. Attention still materializes
ordinary dense KV, so this is not a new attention/scoring algorithm. Parent
validation and prefill checks belong inside any future generation timer.
"""
from dataclasses import dataclass
import functools

import torch
from transformers.cache_utils import DynamicCache,DynamicLayer
from .decode_replay import cache_pairs


@dataclass(frozen=True)
class WaveLayout:
    beams: int = 50
    requests: int = 2

    def __post_init__(self):
        if type(self.beams) is not int or self.beams<1 or type(self.requests) is not int or self.requests!=2:
            raise ValueError('Exactly two requests and positive beams required')

    def validate_parents(self,parents):
        if (parents.ndim!=1 or parents.numel()!=self.requests*self.beams
                or parents.dtype not in (torch.int32,torch.int64)):
            raise ValueError('Expected one global parent per request-major beam row')
        owners=torch.arange(self.requests,device=parents.device).repeat_interleave(self.beams)
        if not bool(((parents>=owners*self.beams)&(parents<(owners+1)*self.beams)).all()):
            raise ValueError('Parent crosses a request boundary')
        return parents

    def global_parents(self,local):
        if local.shape!=(self.requests,self.beams) or local.dtype not in (torch.int32,torch.int64):
            raise ValueError('Expected [two requests, beams] local parent indices')
        if not bool(((local>=0)&(local<self.beams)).all()):
            raise ValueError('Local parent outside request beam range')
        offsets=torch.arange(self.requests,device=local.device)[:,None]*self.beams
        return (local+offsets).reshape(-1)


class WavePrefixLayer(DynamicLayer):
    def __init__(self,prefix,suffix,layout):
        super().__init__()
        self.prefix_keys,self.prefix_values=prefix
        self.keys,self.values=suffix
        self.layout=layout

    def get_seq_length(self,cache_position=None):
        return self.prefix_keys.shape[-2]+self.keys.shape[-2]

    def update(self,key_states,value_states,cache_kwargs=None):
        if key_states.shape!=value_states.shape or key_states.shape[0]!=2*self.layout.beams:
            raise ValueError('Decode rows must retain both request beam blocks')
        self.keys=torch.cat((self.keys,key_states),dim=-2)
        self.values=torch.cat((self.values,value_states),dim=-2)
        return tuple(torch.cat((prefix.repeat_interleave(self.layout.beams,dim=0),suffix),dim=-2)
                     for prefix,suffix in ((self.prefix_keys,self.keys),(self.prefix_values,self.values)))

    def _reorder_validated(self,parents):
        self.keys=self.keys.index_select(0,parents)
        self.values=self.values.index_select(0,parents)

    def reorder_cache(self,beam_idx):
        self._reorder_validated(self.layout.validate_parents(beam_idx))

    def crop(self,max_length):raise NotImplementedError('Beam waves do not crop cache')
    def batch_repeat_interleave(self,repeats):raise NotImplementedError('Wave beam rows already expanded')


class WavePrefixCache(DynamicCache):
    def __init__(self,prefill,layout):
        super().__init__()
        self.layout=layout
        pairs=cache_pairs(prefill);length=prefill.get_seq_length()
        if not pairs or length<1 or any(x.ndim!=4 or x.shape[0]!=2 or x.shape[2]!=length
                                        for pair in pairs for x in pair):
            raise ValueError('Two equal-length physical prompt rows required')
        self.layers=[]
        for pair in pairs:
            if pair[0].shape!=pair[1].shape or pair[0].dtype!=pair[1].dtype or pair[0].device!=pair[1].device:
                raise ValueError('Inconsistent key/value layout')
            prefix=tuple(x.clone() for x in pair)
            suffix=tuple(x.new_empty((2*layout.beams,x.shape[1],0,x.shape[3])) for x in pair)
            self.layers.append(WavePrefixLayer(prefix,suffix,layout))

    def reorder_cache(self,beam_idx):
        # Validate once before mutating any layer; no partial cross-request write.
        self.layout.validate_parents(beam_idx)
        for layer in self.layers:layer._reorder_validated(beam_idx)

    def batch_repeat_interleave(self,repeats):raise NotImplementedError('Wave beam rows already expanded')


class WavePrefill:
    """Prefill two physical histories after HF expands each into its own beams.

    Patch only one model instance for the duration of a sequential wave. Native
    HF retains batch_size=2 and num_beams=50; no request becomes one100-beam search.
    """
    def __init__(self,model,beams=50,arena=None):
        self.model=model;self.layout=WaveLayout(beams);self.used=False;self.arena=arena
        if arena is not None and arena.layout!=self.layout:
            raise ValueError('Wave prefill and arena beam layouts differ')

    def __enter__(self):
        if getattr(self.model,'_beam_wave_active',False):raise ValueError('Model already owns an active wave')
        if self.model.training:raise ValueError('Eval mode required for beam waves')
        self.used=False
        self.model._beam_wave_active=True
        self.original=self.model.forward
        @functools.wraps(self.original)
        def forward(*args,**kwargs):
            if self.used:return self.original(*args,**kwargs)
            if args:raise ValueError('Wave prefill expects keyword inputs')
            ids=kwargs.get('input_ids');beams=self.layout.beams
            if ids is None or ids.ndim!=2 or ids.shape[0]!=2*beams:
                raise ValueError('Exactly two request-major beam blocks required')
            grouped=ids.reshape(2,beams,-1)
            if not torch.equal(grouped,grouped[:,:1].expand_as(grouped)):
                raise ValueError('Initial beam rows differ within a request')
            cache=kwargs.get('past_key_values')
            if not isinstance(cache,DynamicCache) or cache.get_seq_length()!=0:
                raise ValueError('Wave prefill requires an empty DynamicCache')
            if not kwargs.get('use_cache') or not kwargs.get('return_dict'):
                raise ValueError('Cached dictionary output required')
            if kwargs.get('output_attentions') or kwargs.get('output_hidden_states'):
                raise ValueError('Auxiliary outputs unsupported')
            mask=kwargs.get('attention_mask')
            if isinstance(mask,torch.Tensor) and (mask.ndim!=2 or mask.shape!=ids.shape or not bool((mask==1).all())):
                raise ValueError('First wave supports equal-length unpadded requests only')
            reduced=dict(kwargs);selection=torch.tensor([0,beams],device=ids.device)
            for name in ('input_ids','attention_mask','position_ids','token_type_ids'):
                value=reduced.get(name)
                if isinstance(value,torch.Tensor):
                    if value.ndim<2 or value.shape[0] not in (1,2*beams):raise ValueError('Unsupported wave input shape')
                    if value.shape[0]==2*beams:reduced[name]=value.index_select(0,selection)
            self.used=True
            output=self.original(**reduced)
            output.past_key_values=(self.arena.start_wave(output.past_key_values) if self.arena is not None
                                    else WavePrefixCache(output.past_key_values,self.layout))
            output.logits=output.logits.repeat_interleave(beams,dim=0)
            return output
        self.model.forward=forward
        return self

    def __exit__(self,*exc):
        self.model.forward=self.original
        self.model._beam_wave_active=False
