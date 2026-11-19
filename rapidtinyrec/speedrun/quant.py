"""Weight-only / W8A8 quantization hooks for Qwen2Weights.

Each scheme replaces the four per-layer projection weights (and optionally the
LM head) with callable modules and installs a dispatching `W.linear`.
The token embedding used for lookups always stays in the model dtype.

Schemes
  gl_w8      GemLite A16W8 int8 per-channel weights, bf16/fp16 activations
  gl_w8_head gl_w8 plus an int8 LM-head copy (rows padded to a multiple of 64)
  gl_w4      GemLite A16W4 (HQQ-style asymmetric int4, group 128, RTN scales)
  gl_a8w8    GemLite dynamic int8 activations x int8 weights (batched regime)
  ao_w8      torchao Int8WeightOnly (needs torch.compile to be fast)
  ao_w4      torchao Int4WeightOnly group 128
"""
import torch
import torch.nn.functional as F

PROJ = ('wqkv', 'wo', 'wgu', 'wd')


def _dispatch(x, w, b=None):
    if isinstance(w, torch.Tensor):
        return F.linear(x, w, b)
    y = w(x)
    return y if b is None else y + b


def _quant_rtn_int4(w, group=128):
    """Asymmetric RTN int4 per group: returns (W_q uint8 [N,K], scales, zeros) in HQQ layout."""
    N, K = w.shape
    wg = w.float().reshape(N, K // group, group)
    mn, mx = wg.amin(-1, keepdim=True), wg.amax(-1, keepdim=True)
    scale = ((mx - mn) / 15).clamp(min=1e-6)
    zero = -mn / scale
    q = (wg / scale + zero).round().clamp(0, 15).to(torch.uint8)
    return q.reshape(N, K), scale.reshape(N, K // group), zero.reshape(N, K // group)


def _gemlite(W, scheme):
    import gemlite  # noqa: F401
    from gemlite.helper import A16W8_INT8, A8W8_int8_dynamic, A16W4_HQQ_INT
    dt = W.dtype
    if scheme in ('gl_w8', 'gl_w8_head'):
        mk = lambda w, b=None: A16W8_INT8(device=w.device, dtype=dt).from_weights(w, bias=b)
    elif scheme == 'gl_a8w8':
        mk = lambda w, b=None: A8W8_int8_dynamic(device=w.device, dtype=dt).from_weights(w, bias=b)
    elif scheme == 'gl_w4':
        def mk(w, b=None):
            q, s, z = _quant_rtn_int4(w)
            return A16W4_HQQ_INT(device=w.device, dtype=dt).from_weights(
                q, s.to(dt), z.to(dt), bias=b)
    else:
        raise ValueError(scheme)
    for layer in W.layers:
        layer['wqkv'] = mk(layer['wqkv'], layer['bqkv']); layer['bqkv'] = None
        for k in ('wo', 'wgu', 'wd'):
            layer[k] = mk(layer[k])
    if scheme == 'gl_w8_head':
        q = A16W8_INT8(device=W.lm_head.device, dtype=dt).from_weights(W.lm_head)
        W.lm_head = q
    W.linear = _dispatch
    torch.cuda.empty_cache()


class _PaddedHead(torch.nn.Module):
    def __init__(self, q, V):
        super().__init__()
        self.q, self.V = q, V
        self.shape = (V,)

    def forward(self, x):
        return self.q(x)[..., :self.V]


def _torchao(W, scheme):
    from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig
    cfg = Int8WeightOnlyConfig() if scheme == 'ao_w8' else Int4WeightOnlyConfig(group_size=128)

    def mk(w, b=None):
        lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=b is not None, device=w.device, dtype=w.dtype)
        with torch.no_grad():
            lin.weight.copy_(w)
            if b is not None:
                lin.bias.copy_(b)
        seq = torch.nn.Sequential(lin)
        quantize_(seq, cfg)
        return seq[0]
    for layer in W.layers:
        layer['wqkv'] = mk(layer['wqkv'], layer['bqkv']); layer['bqkv'] = None
        for k in ('wo', 'wgu', 'wd'):
            layer[k] = mk(layer[k])
    W.linear = _dispatch
    torch.cuda.empty_cache()


class FP8Linear(torch.nn.Module):
    """W8A8 FP8 (e4m3) with per-output-channel weight scales and per-token dynamic activation
    scales via torch._scaled_mm (Ada/Hopper). Rows are padded to a multiple of 16."""

    def __init__(self, w, b=None, rowwise=True):
        super().__init__()
        ws = w.float().abs().amax(1, keepdim=True).clamp(min=1e-12) / 448.0
        self.w8 = (w.float() / ws).to(torch.float8_e4m3fn).contiguous()   # [N,K]
        self.ws = ws.t().contiguous()                                      # [1,N]
        self.b = b
        self.rowwise = rowwise
        self.shape = w.shape

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        M = x2.shape[0]
        pad = (-M) % 16
        if pad:
            x2 = F.pad(x2, (0, 0, 0, pad))
        xf = x2.float()
        if self.rowwise:
            xs = xf.abs().amax(1, keepdim=True).clamp(min=1e-12) / 448.0
            y = torch._scaled_mm((xf / xs).to(torch.float8_e4m3fn), self.w8.t(), scale_a=xs,
                                 scale_b=self.ws, out_dtype=x.dtype)
        else:
            xs = xf.abs().amax().clamp(min=1e-12) / 448.0
            y = torch._scaled_mm((xf / xs).to(torch.float8_e4m3fn), self.w8.t(), scale_a=xs,
                                 scale_b=self.ws.amax(), out_dtype=x.dtype)
        y = y[:M]
        if self.b is not None:
            y = y + self.b
        return y.reshape(*shp[:-1], -1)


def _fp8(W, scheme):
    rowwise = scheme != 'fp8_tensor'
    for layer in W.layers:
        layer['wqkv'] = FP8Linear(layer['wqkv'], layer['bqkv'], rowwise); layer['bqkv'] = None
        for k in ('wo', 'wgu', 'wd'):
            layer[k] = FP8Linear(layer[k], None, rowwise)
    W.linear = _dispatch
    torch.cuda.empty_cache()


def get(scheme):
    if scheme.startswith('fp8'):
        return lambda W: _fp8(W, scheme)
    if scheme.startswith('gl_'):
        return lambda W: _gemlite(W, scheme)
    if scheme.startswith('ao_'):
        return lambda W: _torchao(W, scheme)
    raise ValueError(scheme)


# ---------------------------------------------------------------------------------------------
# Static-scale FP8 (W8A8, e4m3) with SmoothQuant-style smoothing, for Hopper/Ada.
# Per-input-channel smoothing factors s move activation outliers into the weights and are folded
# exactly into the preceding operator (ln1 for qkv, ln2 for gate/up, the v rows of wqkv for o_proj,
# the up rows of wgu for down). Activations then use one static per-tensor scale per projection,
# so quantizing x is a pure elementwise op that torch.compile fuses into the preceding kernel.

class FP8Static(torch.nn.Module):
    """Tensor-wise static FP8: one scale for the weight, one calibrated scale for the activation.
    FP8 keeps exponent bits, so per-tensor weight scaling loses little; both scales are 0-dim
    device tensors created once, so a call is quantize (elementwise, fusable) + one cuBLASLt GEMM."""

    def __init__(self, w, a_scale, b=None, out_dtype=torch.float16):
        super().__init__()
        wsc = w.float().abs().amax().clamp(min=1e-12) / 448.0
        self.w8t = (w.float() / wsc).to(torch.float8_e4m3fn).t()             # [K,N] column-major view
        self.sw = wsc.reshape(()).to(torch.float32)
        self.sa = torch.tensor(float(a_scale), device=w.device, dtype=torch.float32)
        self.inv_a = 1.0 / float(a_scale)
        self.b = None if b is None else b.to(out_dtype)
        self.out_dtype = out_dtype

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        M = x2.shape[0]
        pad = (-M) % 16
        if pad:
            x2 = F.pad(x2, (0, 0, 0, pad))
        x8 = (x2.float() * self.inv_a).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        y = torch._scaled_mm(x8, self.w8t, scale_a=self.sa, scale_b=self.sw, bias=self.b, out_dtype=self.out_dtype)
        return y[:M].reshape(*shp[:-1], -1)


@torch.no_grad()
def calibrate_absmax(W, run_fn):
    """Per-input-channel |x| max for every projection, recorded while run_fn() runs the engine eagerly."""
    stats = {}
    orig = W.linear

    def rec(x, w, b=None):
        if isinstance(w, torch.Tensor) and w.dim() == 2 and w.shape[0] != W.embed.shape[0]:
            m = x.reshape(-1, x.shape[-1]).abs().amax(0).float()
            k = id(w)
            stats[k] = torch.maximum(stats[k], m) if k in stats else m
        return orig(x, w, b)

    W.linear = rec
    try:
        run_fn()
    finally:
        W.linear = orig
    return stats


@torch.no_grad()
def apply_fp8_smooth(W, stats, alpha=0.5, margin=1.0):
    """Fold smoothing into the weights/norms, then replace projections with FP8Static modules."""
    H, KV, hd, D, I = W.H, W.KV, W.hd, W.D, W.I
    rep = H // KV
    for layer in W.layers:
        wq, wg, wo, wd = layer['wqkv'], layer['wgu'], layer['wo'], layer['wd']
        ax = {n: stats[id(t)] for n, t in (('qkv', wq), ('gu', wg), ('o', wo), ('d', wd))}

        def smooth(ax_, w):
            aw = w.float().abs().amax(0).clamp(min=1e-5)
            return (ax_.clamp(min=1e-5) ** alpha / aw ** (1 - alpha)).clamp(1e-3, 1e3)

        # qkv: x = rms(h) * ln1 -> fold 1/s into ln1
        s = smooth(ax['qkv'], wq)
        layer['ln1'] = (layer['ln1'].float() / s).to(layer['ln1'].dtype)
        wq_s = wq.float() * s[None]
        # o_proj input = attention output; channel (head h, dim d) comes from v head h // rep:
        # one factor per (kv head, dim), shared by the heads of a group, folded into the v rows of wqkv
        so = smooth(ax['o'], wo).view(H, hd).view(KV, rep, hd).amax(1)          # [KV, hd]
        so_full = so[:, None, :].expand(KV, rep, hd).reshape(H * hd)
        v0 = (H + KV) * hd
        wq_s[v0:] = wq_s[v0:] / so.reshape(-1)[:, None]
        bq = layer['bqkv'].float().clone()
        bq[v0:] = bq[v0:] / so.reshape(-1)
        wo_s = wo.float() * so_full[None]
        # gate/up: x = rms(h) * ln2 -> fold into ln2
        sg = smooth(ax['gu'], wg)
        layer['ln2'] = (layer['ln2'].float() / sg).to(layer['ln2'].dtype)
        wg_s = wg.float() * sg[None]
        # down input = silu(gate) * up -> fold into the up rows of wgu
        sd = smooth(ax['d'], wd)
        wg_s[I:] = wg_s[I:] / sd[:, None]
        wd_s = wd.float() * sd[None]
        # static activation scales after smoothing
        a_qkv = (ax['qkv'] / s).amax().item() * margin / 448.0
        a_o = (ax['o'] / so_full).amax().item() * margin / 448.0
        a_gu = (ax['gu'] / sg).amax().item() * margin / 448.0
        a_d = (ax['d'] / sd).amax().item() * margin / 448.0
        od = W.dtype
        layer['wqkv'] = FP8Static(wq_s, a_qkv, bq, od); layer['bqkv'] = None
        layer['wo'] = FP8Static(wo_s, a_o, None, od)
        layer['wgu'] = FP8Static(wg_s, a_gu, None, od)
        layer['wd'] = FP8Static(wd_s, a_d, None, od)
    W.linear = _dispatch
    torch.cuda.empty_cache()
