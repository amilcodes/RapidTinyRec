"""Pack single-token GQA query heads into rows for native SDPA.

This changes layout and dispatch shapes, not the mathematical attention rows.
GPU rounding equality is a benchmark requirement, not an assumption. Prefill,
masked/causal attention, dropout, and non-GQA calls keep their original path.
The context patches process-global F.sdpa; use only in an isolated worker.
"""
import contextlib
from unittest.mock import patch
import torch
import torch.nn.functional as F

_SDPA=F.scaled_dot_product_attention


def folded_sdpa(query,key,value,attn_mask=None,dropout_p=0.0,is_causal=False,*,scale=None,enable_gqa=False):
    if (enable_gqa and query.ndim==key.ndim==value.ndim==4 and query.shape[-2]==1
            and attn_mask is None and dropout_p==0.0 and not is_causal
            and query.shape[-3]>key.shape[-3]):
        batch,heads,_,width=query.shape
        kv_heads=key.shape[-3]
        if heads%kv_heads or key.shape!=value.shape or key.shape[0]!=batch or key.shape[-1]!=width:
            raise ValueError('Unsupported grouped decode layout')
        # Row g*ratio+r maps to KV head g in both layouts. RoPE was already
        # applied. Attention rows do not interact when no mask/causality is used.
        packed=query.reshape(batch,kv_heads,heads//kv_heads,width)
        output=_SDPA(packed,key,value,attn_mask=None,dropout_p=0.0,is_causal=False,
                     scale=scale,enable_gqa=False)
        return output.reshape(batch,heads,1,width)
    return _SDPA(query,key,value,attn_mask=attn_mask,dropout_p=dropout_p,
                 is_causal=is_causal,scale=scale,enable_gqa=enable_gqa)


@contextlib.contextmanager
def fold_decode_attention(enabled=True):
    if enabled:
        with patch.object(F,'scaled_dot_product_attention',folded_sdpa):yield
    else:
        yield
