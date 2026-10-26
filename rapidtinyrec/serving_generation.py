"""Lane-private generation using the accepted folded generation body verbatim.

Imports are lazy so source/protocol tests need no Torch installation. The global
folded attention patch must already span the entire server lifetime.
"""
import contextlib
import functools
from unittest.mock import patch


def make_generation(model, compiler, arena, pool, direct, tok, tables, inputs, masks):
    from . import benchmark as b
    from .prefix_prefill import PrefillPrefix
    from .gpu_constraints import GpuSidConstrainedLogitsProcessor
    torch = b.torch
    prefix_mode = True

    @torch.inference_mode()
    def generate(user,replay,direct_reorder=False):
        calls=0
        @functools.wraps(compiler.eager)
        def forward(*a,**kw):
            nonlocal calls
            assert not a
            step=calls;calls+=1
            ids,cache=kw['input_ids'],kw['past_key_values']
            assert ids.shape==((1,user['prompt_tokens']) if step==0 else (50,1))
            assert cache.get_seq_length()==(0 if step==0 else user['prompt_tokens']+step-1)
            kw['attention_mask']={'full_attention':None}
            return pool(**kw) if replay and step else compiler.compiled(**kw)
        cfg=b.GenerationConfig(max_new_tokens=8,length_penalty=0.,num_beams=50,num_return_sequences=50,
            pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id,bos_token_id=tok.bos_token_id,
            do_sample=False,top_k=None,top_p=None,use_cache=True)
        with patch.object(model,'forward',forward),(PrefillPrefix(model,50,cache_expander=arena.start_request) if prefix_mode and replay else b.PrefillOnce(model,50)), \
                (direct.install(model) if direct_reorder else contextlib.nullcontext()):
            output=model.generate(inputs[user['cohort_index']],attention_mask=masks[user['cohort_index']],
                generation_config=cfg,return_dict_in_generate=True,output_scores=True,
                logits_processor=b.LogitsProcessorList([GpuSidConstrainedLogitsProcessor(tables)]))
        assert calls==5
        return output
    return generate
