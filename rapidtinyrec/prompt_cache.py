"""Fixed cross-request prompt KV reuse for inference-only Qwen2.

This is a CPU-tested building block, not a measured CUDA optimization. The
caller verifies checkpoint/tokenizer digests before construction. Preparation
validates tokens outside request timing. Request execution includes cloning the
cached KV into an independently mutable DynamicCache; these copies must remain
inside any eventual benchmark. Existing decode replay and scoring are unchanged.
"""
from dataclasses import dataclass
import hashlib
import json
import re

import torch

from .decode_replay import cache_from_pairs, cache_pairs


def _config_key(model):
    return hashlib.sha256(json.dumps(model.config.to_dict(), sort_keys=True,
                                    default=str).encode()).hexdigest()


def _policy_key():
    return (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            torch.get_float32_matmul_precision())


def _parameter_key(model):
    # Normal loaded Parameters carry mutation counters even when forwards use
    # inference_mode. Reject inference-created parameters without counters.
    try:
        return tuple((id(p), p._version, p.dtype, p.device) for p in model.parameters())
    except RuntimeError as exc:
        raise ValueError('Version-tracked frozen parameters required') from exc


@dataclass(frozen=True)
class PromptIdentity:
    checkpoint_manifest_sha256: str
    tokenizer_manifest_sha256: str
    precision: str

    def __post_init__(self):
        for value in (self.checkpoint_manifest_sha256, self.tokenizer_manifest_sha256):
            if not re.fullmatch('[0-9a-f]{64}', value):
                raise ValueError('Verified resource-manifest SHA256 digests required')
        if not self.precision:
            raise ValueError('Explicit precision identity required')


@dataclass(frozen=True)
class PreparedPrompt:
    owner: object
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    cache_position: torch.Tensor
    attention_mask: torch.Tensor
    total_tokens: int


class FixedPromptCache:
    """One model, one numerical policy, fixed prefix, sequential requests only.

    Model weights must remain frozen for this object's lifetime. Model/resource
    identity is explicit; neither an object ID nor the descriptive precision
    label is a substitute for verifying the resource manifests. No padded,
    sliding-window, dynamic-RoPE, training, or embedding-input paths are supported.
    """
    @torch.inference_mode()
    def __init__(self, model, prefix_ids, identity, *, forward=None):
        if not isinstance(identity, PromptIdentity):
            raise ValueError('PromptIdentity required')
        if model.training or model.config.model_type != 'qwen2':
            raise ValueError('An eval-mode Qwen2 model is required')
        if (getattr(model.model, 'has_sliding_layers', False)
                or set(model.config.layer_types) != {'full_attention'}):
            raise ValueError('Only full attention is supported')
        rope = model.config.rope_scaling
        if rope and rope.get('rope_type', rope.get('type', 'default')) != 'default':
            raise ValueError('Only static default RoPE is supported')
        if model.config._attn_implementation != 'sdpa':
            raise ValueError('Only the reviewed SDPA path is supported')
        params = tuple(model.parameters())
        self.dtype, self.device = params[0].dtype, params[0].device
        if self.dtype not in (torch.float32, torch.bfloat16) or any(
                p.dtype != self.dtype or p.device != self.device for p in params):
            raise ValueError('Uniform FP32 or BF16 parameters required')
        if (prefix_ids.ndim != 2 or prefix_ids.shape[0] != 1
                or prefix_ids.shape[1] < 1 or prefix_ids.dtype != torch.long
                or prefix_ids.device != self.device):
            raise ValueError('Expected one nonempty integer prefix on the model device')
        self.model, self.identity = model, identity
        self._forward = model.forward if forward is None else forward
        self._prefix_ids = prefix_ids.detach().clone()
        self.prefix_tokens = prefix_ids.shape[1]
        self._config, self._policy = _config_key(model), _policy_key()
        self._parameters = _parameter_key(model)
        self._owner = object()
        position = torch.arange(self.prefix_tokens, device=self.device)
        output = self._forward(input_ids=self._prefix_ids,
            attention_mask={'full_attention': None}, position_ids=position[None],
            cache_position=position, past_key_values=cache_from_pairs(()),
            use_cache=True, return_dict=True, logits_to_keep=1)
        self._pairs = tuple(tuple(x.detach().clone() for x in pair)
                            for pair in cache_pairs(output.past_key_values))
        if len(self._pairs) != model.config.num_hidden_layers or any(
                x.shape[0] != 1 or x.shape[-2] != self.prefix_tokens
                or x.dtype != self.dtype or x.device != self.device
                for pair in self._pairs for x in pair):
            raise ValueError('Unexpected populated prefix layout')
        self.requests = 0

    def _check_owner(self):
        if (self.model.training or _config_key(self.model) != self._config
                or _policy_key() != self._policy
                or _parameter_key(self.model) != self._parameters):
            raise ValueError('Model/config/precision changed; rebuild the prompt cache')

    @torch.inference_mode()
    def prepare(self, prompt_ids):
        """Validate a single unpadded prompt; staging is outside generation time."""
        self._check_owner()
        if (prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1
                or prompt_ids.dtype != torch.long or prompt_ids.device != self.device
                or prompt_ids.shape[1] <= self.prefix_tokens
                or prompt_ids.shape[1] > self.model.config.max_position_embeddings):
            raise ValueError('One unpadded prompt with a nonempty suffix is required')
        if not torch.equal(prompt_ids[:, :self.prefix_tokens], self._prefix_ids):
            raise ValueError('Prompt does not have the fixed cached prefix')
        total = prompt_ids.shape[1]
        positions = torch.arange(self.prefix_tokens, total, device=self.device)
        keys = torch.arange(total, device=self.device)
        # Row i is absolute query P+i. All P cached keys and suffix keys <=i
        # are visible. Upper-left non-square is_causal=True would be wrong.
        allowed = keys[None, :] <= positions[:, None]
        mask = torch.zeros((1, 1, total-self.prefix_tokens, total),
                           dtype=self.dtype, device=self.device)
        mask.masked_fill_(~allowed[None, None], torch.finfo(self.dtype).min)
        return PreparedPrompt(self._owner, prompt_ids[:, self.prefix_tokens:].clone(),
                              positions[None].clone(), positions, mask, total)

    @torch.inference_mode()
    def prefill(self, prepared, *, forward=None):
        """Warm-request prefill; include all checks and KV clones in timing."""
        self._check_owner()
        if not isinstance(prepared, PreparedPrompt) or prepared.owner is not self._owner:
            raise ValueError('Prepared prompt belongs to another cache')
        request_cache = cache_from_pairs(tuple(tuple(x.clone() for x in pair)
                                              for pair in self._pairs))
        output = (self._forward if forward is None else forward)(
            input_ids=prepared.input_ids,
            attention_mask={'full_attention': prepared.attention_mask},
            position_ids=prepared.position_ids, cache_position=prepared.cache_position,
            past_key_values=request_cache, use_cache=True, return_dict=True,
            logits_to_keep=1)
        if output.past_key_values.get_seq_length() != prepared.total_tokens:
            raise ValueError('Suffix prefill returned the wrong cache length')
        self.requests += 1
        return output

    def snapshot(self):
        """Untimed audit copies; never expose persistent KV to request mutation."""
        return tuple(tuple(x.clone() for x in pair) for pair in self._pairs)

    def report(self):
        return dict(prefix_tokens=self.prefix_tokens, requests=self.requests,
                    cached_kv_bytes=sum(x.numel()*x.element_size()
                                        for pair in self._pairs for x in pair),
                    request_kv_copy_bytes=sum(x.numel()*x.element_size()
                                             for pair in self._pairs for x in pair),
                    dtype=str(self.dtype), model_config_sha256=self._config,
                    identity=vars(self.identity), policy=self._policy,
                    prefix_token_sha256=hashlib.sha256(json.dumps(
                        self._prefix_ids.cpu().tolist()).encode()).hexdigest())
