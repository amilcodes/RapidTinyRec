"""Bind fixed prompt prefill to the existing HF five-step beam generation.

Preparation owns the exact input IDs used by generate. No caller-supplied token
batch can be silently substituted for the prepared cached suffix. The caller
supplies the compiled prefill forward and decode replay callable after warmup;
this adapter neither compiles lazily nor retries failures through an eager path.
"""
import contextlib
import copy
from dataclasses import dataclass
import functools
from unittest.mock import patch

import torch

from .prefix_prefill import PrefillPrefix


@dataclass(frozen=True)
class PreparedGeneration:
    owner: object
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    suffix: object


class PromptCacheGenerator:
    def __init__(self, cache, generation_config, arena, *, prefill_forward,
                 decode_forward, reorder=None):
        cfg = generation_config
        if (cfg.num_beams != 50 or cfg.num_return_sequences != 50
                or cfg.num_beam_groups != 1 or cfg.do_sample
                or cfg.length_penalty != 0. or cfg.max_new_tokens != 8
                or not cfg.use_cache):
            raise ValueError('Original deterministic beam50 generation config required')
        self.cache, self.model, self.arena = cache, cache.model, arena
        self.config = copy.deepcopy(cfg)
        self.prefill_forward, self.decode_forward = prefill_forward, decode_forward
        self.reorder = reorder
        self._owner = object()
        self.active = False

    @torch.inference_mode()
    def prepare(self, input_ids):
        # All token equality checks, input staging and mask allocation occur
        # before timing. Prepared objects are immutable-by-contract to callers.
        owned = input_ids.clone()
        suffix = self.cache.prepare(owned)
        return PreparedGeneration(self._owner, owned, torch.ones_like(owned), suffix)

    @torch.inference_mode()
    def generate(self, prepared, logits_processor):
        if not isinstance(prepared, PreparedGeneration) or prepared.owner is not self._owner:
            raise ValueError('Prepared generation belongs to another generator')
        if self.active:
            raise ValueError('Concurrent or recursive generation is unsupported')
        self.active = True
        calls = 0
        original = self.model.forward

        @functools.wraps(original)
        def forward(*args, **kw):
            nonlocal calls
            if args:
                raise ValueError('Keyword-only generation inputs required')
            step = calls
            calls += 1
            length = prepared.suffix.total_tokens
            expected_shape = (1, length) if step == 0 else (50, 1)
            expected_cache = 0 if step == 0 else length+step-1
            if (kw['input_ids'].shape != expected_shape
                    or kw['past_key_values'].get_seq_length() != expected_cache):
                raise ValueError('Unexpected prefill/decode generation boundary')
            if step == 0:
                # FixedPromptCache supplies its explicit non-square mask to the
                # compiled suffix forward. It must never pass through the
                # unmasked single-token decode specialization below.
                return self.cache.prefill(prepared.suffix, forward=self.prefill_forward)
            kw['attention_mask'] = {'full_attention': None}
            return self.decode_forward(**kw)

        try:
            with patch.object(self.model, 'forward', forward), PrefillPrefix(
                    self.model, 50, cache_expander=self.arena.start_request), (
                    self.reorder.install(self.model) if self.reorder is not None
                    else contextlib.nullcontext()):
                output = self.model.generate(prepared.input_ids,
                    attention_mask=prepared.attention_mask,
                    generation_config=self.config, return_dict_in_generate=True,
                    output_scores=True, logits_processor=logits_processor)
            if calls != 5 or len(output.scores) != 5:
                raise ValueError('Expected the original five scored transitions')
            return output
        finally:
            self.active = False
