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
