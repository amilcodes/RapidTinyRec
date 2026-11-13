"""Add Qwen2 (QKV bias, no QK-norm) to FlashRec 0.1.2 (sohu-mptc/FlashRec @ 1089682).

usage: python flashrec_qwen2.py FLASHREC_REPO

Qwen2 differs from Qwen3 in exactly two places that FlashRec's model touches:
  * q/k/v projections carry a bias  -> fused qkv_proj gets the concatenated bias
  * there is no per-head q_norm / k_norm -> skipped, and the fused Triton
    QK-norm + RoPE + FP8 KV-store kernel gets HAS_NORM=False (RoPE + store only)
Everything else (RMSNorm, SwiGLU, RoPE theta, GQA, tied embeddings) is shared.
"""
import sys
from pathlib import Path

root = Path(sys.argv[1]) / 'python' / 'flashrec'


def sub(path, old, new, count=1):
    p = root / path
    s = p.read_text()
    n = s.count(old)
    if n != count:
        raise SystemExit(f'{path}: expected {count} match(es), found {n}:\n{old}')
    p.write_text(s.replace(old, new))


# --- model ---------------------------------------------------------------------
sub('models/qwen3.py', '    tie_word_embeddings: bool\n',
    '    tie_word_embeddings: bool\n'
    '    attention_bias: bool = False  # Qwen2: q/k/v projections have a bias\n'
    '    qk_norm: bool = True  # Qwen2: no per-head q_norm / k_norm\n')
sub('models/qwen3.py',
    '            cfg.hidden_size, (self.num_qo + 2 * self.num_kv) * cfg.head_dim\n        )',
    '            cfg.hidden_size, (self.num_qo + 2 * self.num_kv) * cfg.head_dim,\n'
    '            bias=cfg.attention_bias,\n        )')
sub('models/qwen3.py',
    '        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)\n'
    '        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)\n',
    '        self.qk_norm = bool(cfg.qk_norm)\n'
    '        if self.qk_norm:\n'
    '            self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)\n'
    '            self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)\n'
    '        else:\n'
    '            self.q_norm = self.k_norm = None\n')
sub('models/qwen3.py',
    '                self.q_norm.weight,\n'
    '                self.k_norm.weight,\n'
    '                self.q_norm.eps,\n',
    '                self.q_norm.weight if self.qk_norm else None,\n'
    '                self.k_norm.weight if self.qk_norm else None,\n'
    '                self.q_norm.eps if self.qk_norm else 0.0,\n')
sub('models/qwen3.py',
    '        if not stored:\n'
    '            q = self.q_norm(q)\n'
    '            k = self.k_norm(k)\n',
    '        if not stored:\n'
    '            if self.qk_norm:\n'
    '                q = self.q_norm(q)\n'
    '                k = self.k_norm(k)\n'
    '            else:  # Qwen2: hand RoPE the same fresh contiguous [t, H, D] tensors q_norm/k_norm would\n'
    '                q = q.contiguous()\n'
    '                k = k.contiguous()\n')

# --- fused Triton kernel: HAS_NORM constexpr ---------------------------------------
K = 'kernel/qk_rope_kv.py'
sub(K, '        BLOCK: tl.constexpr,\n        HALF: tl.constexpr,\n    ):',
    '        BLOCK: tl.constexpr,\n        HALF: tl.constexpr,\n        HAS_NORM: tl.constexpr,\n    ):')
sub(K,
    '        qw1 = tl.load(qw_ptr + lo, mask=mask_lo, other=0.0).to(tl.float32)\n'
    '        qw2 = tl.load(qw_ptr + hi, mask=mask_hi, other=0.0).to(tl.float32)\n'
    '        kw1 = tl.load(kw_ptr + lo, mask=mask_lo, other=0.0).to(tl.float32)\n'
    '        kw2 = tl.load(kw_ptr + hi, mask=mask_hi, other=0.0).to(tl.float32)\n',
    '        if HAS_NORM:\n'
    '            qw1 = tl.load(qw_ptr + lo, mask=mask_lo, other=0.0).to(tl.float32)\n'
    '            qw2 = tl.load(qw_ptr + hi, mask=mask_hi, other=0.0).to(tl.float32)\n'
    '            kw1 = tl.load(kw_ptr + lo, mask=mask_lo, other=0.0).to(tl.float32)\n'
    '            kw2 = tl.load(kw_ptr + hi, mask=mask_hi, other=0.0).to(tl.float32)\n'
    '        else:\n'
    '            qw1 = tl.full((HALF,), 1.0, tl.float32)\n'
    '            qw2 = qw1\n'
    '            kw1 = qw1\n'
    '            kw2 = qw1\n')
# rstd: identity without norm (two occurrences: q loop and k loop)
sub(K, '            rstd = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)\n',
    '            if HAS_NORM:\n'
    '                rstd = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)\n'
    '            else:\n'
    '                rstd = 1.0\n', count=2)
sub(K,
    '    qw = q_weight.reshape(-1).to(device=q.device, dtype=q.dtype).contiguous()\n'
    '    kw = k_weight.reshape(-1).to(device=q.device, dtype=q.dtype).contiguous()\n',
    '    has_norm = q_weight is not None and k_weight is not None\n'
    '    if not has_norm:  # Qwen2: RoPE + FP8 KV store only (weights unused)\n'
    '        q_weight = k_weight = torch.ones(d, device=q.device, dtype=q.dtype)\n'
    '    qw = q_weight.reshape(-1).to(device=q.device, dtype=q.dtype).contiguous()\n'
    '    kw = k_weight.reshape(-1).to(device=q.device, dtype=q.dtype).contiguous()\n')
sub(K, '            HALF=max(d // 2, 16),\n            num_warps=4,',
    '            HALF=max(d // 2, 16),\n            HAS_NORM=has_norm,\n            num_warps=4,')

# --- config + weights ------------------------------------------------------------------
W = 'models/weight.py'
sub(W, '        tie_word_embeddings=bool(cfg.get("tie_word_embeddings", True)),\n    )',
    '        tie_word_embeddings=bool(cfg.get("tie_word_embeddings", True)),\n'
    '        attention_bias=_is_qwen2(cfg) or bool(cfg.get("attention_bias", False)),\n'
    '        qk_norm=not _is_qwen2(cfg),\n    )\n\n\n'
    'def _is_qwen2(cfg) -> bool:\n'
    '    archs = cfg.get("architectures") or []\n'
    '    return cfg.get("model_type") == "qwen2" or "Qwen2ForCausalLM" in archs')
# collect q/k/v biases
sub(W, '        # tail like self_attn.q_proj.weight\n',
    '        if tail.endswith(".bias") and tail.rsplit(".", 2)[-2] in ("q_proj", "k_proj", "v_proj"):\n'
    '            attn_prefix, _, which = (prefix + tail[: -len(".bias")]).rpartition(".")\n'
    '            pending_qkv_bias.setdefault(attn_prefix + ".", {})[which] = t.to(device=device)\n'
    '            continue\n'
    '        # tail like self_attn.q_proj.weight\n')
sub(W, '    pending_qkv: Dict[str, Dict[str, torch.Tensor]] = {}\n',
    '    pending_qkv: Dict[str, Dict[str, torch.Tensor]] = {}\n'
    '    pending_qkv_bias: Dict[str, Dict[str, torch.Tensor]] = {}\n')
sub(W,
    '        linears[dest].load(\n'
    '            w,\n'
    '            quantize_fp8=quantize_fp8 and w.dtype != torch.float8_e4m3fn,\n'
    '            weight_scale=scale,\n'
    '        )\n\n'
    '    for mlp_prefix',
    '        bparts = pending_qkv_bias.get(attn_prefix)\n'
    '        bias = (\n'
    '            torch.cat([bparts["q_proj"], bparts["k_proj"], bparts["v_proj"]]).to(dtype=compute_dtype)\n'
    '            if bparts else None\n'
    '        )\n'
    '        linears[dest].load(\n'
    '            w,\n'
    '            bias=bias,\n'
    '            quantize_fp8=quantize_fp8 and w.dtype != torch.float8_e4m3fn,\n'
    '            weight_scale=scale,\n'
    '        )\n\n'
    '    for mlp_prefix')
print('patched', root)
