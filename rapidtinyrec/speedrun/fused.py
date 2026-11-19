"""Fused-layer forward for batch-1 tree forwards: 5 GEMM launches + RoPE + attention per layer.

Same math as `Forward`, with
  - an FP32 residual stream, updated in place by the O and down GEMMs (accumulate epilogues, split-K
    atomics for the down projection), so residual adds never launch their own kernels;
  - both RMSNorms folded into the following GEMM (gain folded into the weight columns offline, the
    per-row 1/rms accumulated inside the GEMM's K loop);
  - SiLU(gate)*up in the gate/up GEMM's epilogue; bias in the QKV epilogue;
  - optional INT8 weights (symmetric, per output channel), dequantized on the accumulator.
Kernel: rapidtinyrec/speedrun/wgemm.py. Rows beyond 256 are processed in 256-row chunks (identical
numerics; used only by offline quality runs).
"""
import torch

from .model import Forward, rope
from .wgemm import wgemm, quant_int8

# (BN, BK, SPLIT, num_warps, num_stages) per GEMM; overridden by tuned configs when given
DEFAULT_CFG = {'qkv': (32, 128, 1, 4, 4), 'o': (32, 128, 2, 4, 4), 'gu': (64, 64, 1, 8, 3), 'down': (32, 128, 4, 4, 4)}
CHUNK = 256


class FusedForward(Forward):
    def __init__(self, W, int8=False, cfg=None):
        super().__init__(W, attn='triton')
        self.int8 = int8
        self.cfg = dict(DEFAULT_CFG, **(cfg or {}))
        self.F = []
        for layer in W.layers:
            ln1, ln2 = layer['ln1'].float(), layer['ln2'].float()
            ws = {'qkv': layer['wqkv'].float() * ln1[None, :], 'o': layer['wo'].float(),
                  'gu': layer['wgu'].float() * ln2[None, :], 'down': layer['wd'].float()}
            f = {'bqkv': layer['bqkv'].contiguous()}
            for k, w in ws.items():
                if int8:
                    f[k], f[k + '_s'] = quant_int8(w)
                else:
                    f[k], f[k + '_s'] = w.to(W.dtype).contiguous(), None
            self.F.append(f)
        self.lidx = {id(layer): i for i, layer in enumerate(W.layers)}
        self._rope = torch.compile(self._rope_split, dynamic=False, fullgraph=True)

    def weight_bytes(self):
        return sum(t.numel() * t.element_size() for f in self.F for t in f.values() if t is not None)

    def _gemm(self, x, name, f, **kw):
        """x [M,K]; chunks of CHUNK rows (out/accum views follow the same chunks)."""
        M = x.shape[0]
        out = kw.pop('out', None)
        if M <= CHUNK:
            return wgemm(x, f[name], scale=f[name + '_s'], cfg=self.cfg[name], out=out, **kw)
        ys = []
        for s in range(0, M, CHUNK):
            ys.append(wgemm(x[s:s + CHUNK], f[name], scale=f[name + '_s'], cfg=self.cfg[name],
                            out=None if out is None else out[s:s + CHUNK], **kw))
        return out if out is not None else torch.cat(ys, 0)

    def _rope_split(self, y, pos, cos_t, sin_t):
        W = self.W
        q, k, v = y.split([W.H * W.hd, W.KV * W.hd, W.KV * W.hd], -1)
        shp = y.shape[:-1]
        q = q.reshape(*shp, W.H, W.hd)
        k = k.reshape(*shp, W.KV, W.hd)
        v = v.reshape(*shp, W.KV, W.hd)
        cos = cos_t[pos].unsqueeze(-2)
        sin = sin_t[pos].unsqueeze(-2)
        return rope(q, cos, sin), rope(k, cos, sin), v

    def qkv(self, x, layer, pos):
        W = self.W
        f = self.F[self.lidx[id(layer)]]
        shp = x.shape[:-1]
        y = self._gemm(x.reshape(-1, W.D), 'qkv', f, bias=f['bqkv'], norm_eps=W.eps)
        return self._rope(y.view(*shp, -1), pos, W.cos, W.sin)

    def post(self, x, o, layer):
        W = self.W
        assert x.dtype == torch.float32 and x.is_contiguous()
        f = self.F[self.lidx[id(layer)]]
        x2 = x.view(-1, W.D)
        o2 = o.reshape(-1, W.H * W.hd)
        if not o2.is_contiguous():
            o2 = o2.contiguous()
        self._gemm(o2, 'o', f, out=x2, accum=True)
        act = self._gemm(x2, 'gu', f, norm_eps=W.eps, swiglu=True)
        self._gemm(act, 'down', f, out=x2, accum=True)
        return x

    def final(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.W.eps)
        return self.W.norm * xf.to(self.W.dtype)

    def _prefill_tree_fast(self, x, xn, *args, **kw):
        return super()._prefill_tree_fast(x.float(), xn.float(), *args, **kw)

    def prefill(self, x, pos, kcache, vcache):
        return super().prefill(x.float().contiguous(), pos, kcache, vcache)

    def step(self, x, *args, **kw):
        return super().step(x.float().contiguous(), *args, **kw)


def adopt(eng, int8=False, cfg=None):
    """Swap an engine's Forward for the fused one, keeping its attention/tree settings."""
    import torch._dynamo.config as dc
    dc.cache_size_limit = 1024
    if hasattr(dc, 'recompile_limit'):
        dc.recompile_limit = 1024
    dc.automatic_dynamic_shapes = False
    old = eng.fwd
    new = FusedForward(eng.W, int8=int8, cfg=cfg)
    new.tree_fast = old.tree_fast
    new.store_fn = None
    new.layer_fn = None
    eng.fwd = new
    return new
