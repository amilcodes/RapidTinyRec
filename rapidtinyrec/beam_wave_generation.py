"""Paired pilot engine adapters; source preparation, not a submission entry point.

The serial arm retains the audited physical-prefix replay implementation.
The wave arm uses native two-request HF beam search and wave-owned KV mapping.
Input assembly is explicit preparation; wrappers, checks, prefill copies, parent
validation, staging and original HF clones remain inside generate().
"""
import contextlib
import functools
from unittest.mock import patch
import torch
from transformers import LogitsProcessorList
from .gpu_constraints import GpuSidConstrainedLogitsProcessor
from .shared_prefill import PrefillOnce
from .shared_prefix import PrefixArena
from .prefix_prefill import PrefillPrefix
from .prefix_replay import PrefixReplayPool,ReorderSuffix
from .beam_wave import WavePrefill
from .beam_wave_arena import WaveArena,WaveReorder
from .beam_wave_replay import WaveReplayPool


class PairGeneration:
    def __init__(self,model,compiler,config,tables,variant,max_prompt_tokens):
        if variant not in ('serial_pair_tf32','beam_wave_tf32'):raise ValueError('Distinct pair variant required')
        if model.training or model.config.model_type!='qwen2':raise ValueError('Eval Qwen2 required')
        if any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('Both pair variants retain FP32 weights')
        if (config.num_beams!=50 or config.num_return_sequences!=50 or config.length_penalty!=0
                or config.max_new_tokens!=8 or config.do_sample or not config.use_cache):
            raise ValueError('Original beam50 generation policy required')
        self.model=model;self.compiler=compiler;self.config=config;self.tables=tables;self.variant=variant
        self.wave=variant=='beam_wave_tf32';self.inputs={};self.active=False
        self.arena=WaveArena(max_prompt_tokens) if self.wave else PrefixArena(max_prompt_tokens)
        self.pool=(WaveReplayPool if self.wave else PrefixReplayPool)(compiler.compiled,max_graphs=8,arena=self.arena)
        self.reorder=WaveReorder(self.arena) if self.wave else ReorderSuffix(self.pool)

    def prepare_inputs(self,users,token_ids):
        if self.active or self.pool.frozen:raise ValueError('Input preparation after freeze or during generation')
        expected=2 if self.wave else 1
        if len(users)!=expected or len(set(users))!=expected or len(token_ids)!=expected:
            raise ValueError('Wrong request count for pair engine')
        if any(x.ndim!=2 or x.shape[0]!=1 or x.shape[1]!=token_ids[0].shape[1]
               or x.dtype!=torch.int64 or x.device!=token_ids[0].device for x in token_ids):
            raise ValueError('Equal-length unpadded token-ID rows required')
        if not 0<token_ids[0].shape[1]<=self.arena.max_prefix_tokens:raise ValueError('Prompt outside arena capacity')
        key=tuple(users)
        if key in self.inputs:raise ValueError('Prepared input identity already exists')
        ids=torch.cat(token_ids,dim=0)
        self.inputs[key]=(ids,torch.ones_like(ids))

    @torch.inference_mode()
    def generate(self,users,replay=True):
        if self.active:raise ValueError('Pair engine already active')
        key=tuple(users)
        if key not in self.inputs:raise ValueError('Pair input order was not prepared')
        ids,mask=self.inputs[key];count=ids.shape[0];length=ids.shape[1];calls=0
        @functools.wraps(self.compiler.eager)
        def forward(*args,**kwargs):
            nonlocal calls
            if args:raise ValueError('Keyword-only generation expected')
            step=calls;calls+=1
            shape=(count,length) if step==0 else (count*50,1)
            if kwargs['input_ids'].shape!=shape or kwargs['past_key_values'].get_seq_length()!=(0 if step==0 else length+step-1):
                raise ValueError('Generation forward deviated from five-step pair schedule')
            # Exactly the existing unpadded full-attention GQA specialization.
            kwargs['attention_mask']={'full_attention':None}
            result=self.pool(**kwargs) if replay and step else self.compiler.compiled(**kwargs)
            if result.logits.dtype!=torch.float32:raise ValueError('Pair logits must remain FP32')
            return result
        if self.wave:
            prefill=WavePrefill(self.model,arena=self.arena if replay else None)
        else:
            prefill=(PrefillPrefix(self.model,50,cache_expander=self.arena.start_request) if replay
                     else PrefillOnce(self.model,50))
        self.active=True
        try:
            with patch.object(self.model,'forward',forward),prefill, \
                    (self.reorder.install(self.model) if replay else contextlib.nullcontext()):
                result=self.model.generate(ids,attention_mask=mask,generation_config=self.config,
                    return_dict_in_generate=True,output_scores=True,
                    logits_processor=LogitsProcessorList([GpuSidConstrainedLogitsProcessor(self.tables)]))
            if calls!=5 or len(result.scores)!=5:raise ValueError('Original five score transitions required')
            return result
        finally:
            self.active=False

    def graph_logit_storage_ptrs(self):
        return sorted({out.logits.untyped_storage().data_ptr() for _,_,out in self.pool.entries.values()})
