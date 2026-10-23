"""Prefill one request once before the existing HF beam-search loop branches.

Tested with Qwen2ForCausalLM, Transformers 4.55.4, and DynamicCache. This reuses
the computed prefix, but materializes a dense cache row for each beam. It is
not a paged, physically shared cache. The context patches one model instance;
do not use that instance concurrently while this context is active.
"""
import functools

import torch
from transformers.cache_utils import DynamicCache


class PrefillOnce:
    def __init__(self, model, num_beams, on_prefill=None):
        self.model = model
        self.num_beams = num_beams
        self.on_prefill = on_prefill
        self.used = False

    def __enter__(self):
        self.original = self.model.forward

        # HF inspects this signature to enable last-token-only vocabulary
        # projection. Losing it would change both memory use and the benchmark.
        @functools.wraps(self.original)
        def forward(*args, **kwargs):
            if self.used:
                return self.original(*args, **kwargs)
            if args:
                raise ValueError('PrefillOnce expects generate() keyword inputs')
            ids = kwargs.get('input_ids')
            if ids is None or ids.ndim != 2 or ids.shape[0] != self.num_beams:
                raise ValueError('PrefillOnce supports exactly one request')
            cache = kwargs.get('past_key_values')
            if not isinstance(cache, DynamicCache) or cache.get_seq_length() != 0:
                raise ValueError('PrefillOnce requires an empty DynamicCache')
            if not kwargs.get('use_cache') or not kwargs.get('return_dict'):
                raise ValueError('use_cache=True and return_dict=True required')
            if kwargs.get('output_attentions') or kwargs.get('output_hidden_states'):
                raise ValueError('Auxiliary attention/hidden-state outputs unsupported')

            # generate() just repeated this one request into identical beam rows.
            # Slice batch-shaped inputs only; cache_position is a shared 1-D axis.
            single = dict(kwargs)
            for name in ('input_ids', 'attention_mask', 'position_ids', 'token_type_ids'):
                value = single.get(name)
                if isinstance(value, torch.Tensor):
                    if value.ndim < 2 or value.shape[0] not in (1, self.num_beams):
                        raise ValueError(f'Unsupported {name} shape: {value.shape}')
                    single[name] = value[:1]
            self.used = True
            output = self.original(**single)
            if self.on_prefill:
                self.on_prefill('before_expansion', output.past_key_values)
            output.past_key_values.batch_repeat_interleave(self.num_beams)
            output.logits = output.logits.repeat_interleave(self.num_beams, dim=0)
            if self.on_prefill:
                self.on_prefill('after_expansion', output.past_key_values)
            return output

        self.model.forward = forward
        return self

    def __exit__(self, *exc):
        self.model.forward = self.original
