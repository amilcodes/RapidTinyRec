"""Combine the two forced-suffix forwards without changing HF beam scoring.

Specialized to one unpadded MiniOneRec request, 50 beams, three SID tokens,
newline 198, and EOS 151645. The final query pair is [SID3, newline].
The two attention queries use separate KV slices, preserving offset causality
and native GQA. Dense/pointwise work is batched across both positions.
"""
import contextlib
import functools
from unittest.mock import patch

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .shared_prefill import PrefillOnce


def suffix_split_gqa(module, query, key, value, attention_mask, dropout=0.0,
                     scaling=None, is_causal=None, **kwargs):
    assert query.shape[2] == 2 and attention_mask is None
    assert dropout == 0.0 and kwargs.get('sliding_window') is None
    assert not kwargs.get('output_attentions', False)
    # K/V already include both new tokens. SID3 cannot see the last (newline)
    # entry, whereas newline sees the full cache. q_len=1 and is_causal=False
    # are deliberate: all keys in each supplied slice are visible.
    first = torch.nn.functional.scaled_dot_product_attention(
        query[:, :, :1, :].contiguous(),
        key[:, :, :-1, :].contiguous(), value[:, :, :-1, :].contiguous(),
        dropout_p=0.0, scale=scaling, is_causal=False, enable_gqa=True)
    second = torch.nn.functional.scaled_dot_product_attention(
        query[:, :, 1:, :].contiguous(), key.contiguous(), value.contiguous(),
        dropout_p=0.0, scale=scaling, is_causal=False, enable_gqa=True)
    return torch.cat((first, second), dim=2).transpose(1, 2).contiguous(), None


def register_suffix_attention():
    ALL_ATTENTION_FUNCTIONS.register('suffix_split_gqa', suffix_split_gqa)


def validate_attention():
    """Compare with an explicit offset-causal float64 oracle on the GPU."""
    generator = torch.Generator(device='cuda').manual_seed(8729)
    q = torch.randn(3,12,2,8,device='cuda',dtype=torch.float64,generator=generator)
    k = torch.randn(3,2,11,8,device='cuda',dtype=torch.float64,generator=generator)
    v = torch.randn(3,2,11,8,device='cuda',dtype=torch.float64,generator=generator)
    scale = 8**-0.5
    got,_ = suffix_split_gqa(None,q,k,v,None,scaling=scale)
    kk,vv = k.repeat_interleave(6,dim=1),v.repeat_interleave(6,dim=1)
    scores = q @ kk.transpose(-1,-2)*scale
    scores[:,:,0,-1] = -torch.inf
    want = (scores.softmax(-1) @ vv).transpose(1,2).contiguous()
    delta = (got-want).abs().max().item()
    assert delta < 1e-12, delta
    changed_k,changed_v = k.clone(),v.clone()
    changed_k[:,:,-1,:] += 5
    changed_v[:,:,-1,:] += 7
    changed,_ = suffix_split_gqa(None,q,changed_k,changed_v,None,scaling=scale)
    assert torch.equal(got[:,0],changed[:,0]), 'Future newline leaked into SID3'
    assert not torch.equal(got[:,1],changed[:,1]), 'Newline query ignored its own key/value'
    return {'explicit_offset_causal_oracle_max_abs':delta,
            'future_token_isolation_exact':True,'newline_self_attention_present':True,
            'dtype':'float64','device':'cuda','query_heads':12,'kv_heads':2}


class SuffixGeneration:
    """Model-instance context; not safe for concurrent use of this model."""
    def __init__(self,model,compiler,prompt_tokens,combine):
        self.model,self.compiler,self.prompt_tokens,self.combine = model,compiler,prompt_tokens,combine
        self.generation_calls = self.model_forwards = self.suffix_reorders = 0
        self.pending_eos = self.newline_input = None
        self.reorder_indices = None
        self.stack = contextlib.ExitStack()

    def __enter__(self):
        assert not hasattr(self.model,'_reorder_cache')
        assert self.model.config._attn_implementation == 'sdpa'
        @functools.wraps(self.compiler.eager)
        def forward(*args,**kwargs):
            assert not args
            step = self.generation_calls
            self.generation_calls += 1
            ids,cache = kwargs['input_ids'],kwargs['past_key_values']
            if step == 0:
                assert ids.shape == (1,self.prompt_tokens) and cache.get_seq_length() == 0
            else:
                assert ids.shape == (50,1)
                expected = self.prompt_tokens + step - 1
                if self.combine and step == 4:
                    expected += 1
                assert cache.get_seq_length() == expected
            kwargs['attention_mask'] = {'full_attention':None}
            if self.combine and step == 4:
                assert self.model_forwards == 4 and self.suffix_reorders == 1
                assert self.pending_eos is not None
                self.newline_input = ids
                logits,self.pending_eos = self.pending_eos,None
                return CausalLMOutputWithPast(logits=logits,past_key_values=cache)
            self.model_forwards += 1
            if not (self.combine and step == 3):
                return self.compiler.compiled(**kwargs)
            assert kwargs['logits_to_keep'] == 1
            assert kwargs['position_ids'].shape == (50,1)
            assert kwargs['cache_position'].shape == (1,)
            assert kwargs.get('token_type_ids') is None and kwargs.get('inputs_embeds') is None
            kwargs['input_ids'] = torch.cat((ids,torch.full_like(ids,198)),dim=1)
            kwargs['position_ids'] = torch.cat((kwargs['position_ids'],kwargs['position_ids']+1),dim=1)
            kwargs['cache_position'] = torch.cat((kwargs['cache_position'],kwargs['cache_position']+1))
            kwargs['logits_to_keep'] = 2
            previous_attention = self.model.config._attn_implementation
            self.model.config._attn_implementation = 'suffix_split_gqa'
            try:
                output = self.compiler.compiled(**kwargs)
            finally:
                # This is a property with a setter but no deleter; mock.patch
                # attempts delattr on exit when the attribute is not local.
                self.model.config._attn_implementation = previous_attention
            assert output.logits.shape[1] == 2
            self.pending_eos = output.logits[:,1:2,:]
            output.logits = output.logits[:,:1,:]
            return output

        def reorder(cache,beam_idx):
            cache.reorder_cache(beam_idx)
            if self.pending_eos is not None:
                self.pending_eos = self.pending_eos.index_select(0,beam_idx)
                self.reorder_indices = beam_idx
                self.suffix_reorders += 1
            return cache

        self.stack.enter_context(patch.object(self.model,'forward',forward))
        self.stack.enter_context(patch.object(self.model,'_reorder_cache',reorder,create=True))
        self.stack.enter_context(PrefillOnce(self.model,50))
        return self

    def __exit__(self,*exc):
        return self.stack.__exit__(*exc)

    def audit(self):
        """Run only after timing; these device-to-host checks may synchronize."""
        assert self.generation_calls == 5
        assert self.model_forwards == (4 if self.combine else 5)
        if self.combine:
            assert self.pending_eos is None and self.suffix_reorders == 1
            assert bool((self.newline_input==198).all())
            permutation = self.reorder_indices.cpu().tolist()
            assert sorted(permutation) == list(range(50)), 'Expected one forced newline per SID'
        else:
            assert self.suffix_reorders == 0
            permutation = None
        return {'generation_calls':self.generation_calls,'model_forwards':self.model_forwards,
                'suffix_reorders':self.suffix_reorders,'newline_beam_permutation':permutation}
