"""Weight-streaming skinny GEMM for batch-1 tree forwards (M = 16..256 rows).

y[M,N] = epilogue( prologue(x)[M,K] @ W[N,K]^T )  with W in FP16 or INT8 (per-output-channel scale).

At batch 1 every layer GEMM is bound by reading W once; cuBLAS reaches 23-63% of H100 HBM bandwidth at
our shapes (profile: qkv 7.3 us for 6.3 MB, o 6.2 us, gate-up 26 us for 55 MB, down 18 us). This kernel
keeps all M rows in one tile so each weight byte is read exactly once, splits K across programs when N
is small, and fuses the layer's elementwise work:
  - RMSNorm prologue: the norm's gain is folded into W offline; the per-row 1/rms is accumulated over
    the K loop from the same x tiles and applied to the accumulator at the end (exact algebra).
  - epilogues: bias; SiLU(gate) * up (gate rows [0,N), up rows [N,2N) of W); accumulate into y
    (y += x W^T: the residual stream is updated in place, with atomics under split-K).
x may be FP32 (residual stream) or FP16; the dot runs in FP16 with FP32 accumulation.
INT8 weights halve the bytes; dequantization is a per-column scale on the accumulator.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _wgemm_kernel(x_ptr, w_ptr, s_ptr, b_ptr, y_ptr,
                  M, N, K, sxm, swn, sym, eps,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr,
                  NORM: tl.constexpr, SCALE: tl.constexpr, BIAS: tl.constexpr, ACCUM: tl.constexpr,
                  SWIGLU: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rm = tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    k_per = tl.cdiv(K, SPLIT * BK) * BK
    k0 = pid_k * k_per
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    acc2 = tl.zeros((BM, BN), dtype=tl.float32)
    ss = tl.zeros((BM,), dtype=tl.float32)
    mmask = rm < M
    nmask = rn < N
    for kk in range(k0, k0 + k_per, BK):
        kidx = kk + rk
        kmask = kidx < K
        xa = tl.load(x_ptr + rm[:, None] * sxm + kidx[None, :], mask=mmask[:, None] & kmask[None, :], other=0.0)
        if NORM:
            xf = xa.to(tl.float32)
            ss += tl.sum(xf * xf, 1)
        xh = xa.to(tl.float16)
        wt = tl.load(w_ptr + rn[None, :] * swn + kidx[:, None], mask=kmask[:, None] & nmask[None, :], other=0)
        acc = tl.dot(xh, wt.to(tl.float16), acc)
        if SWIGLU:
            wu = tl.load(w_ptr + (rn[None, :] + N) * swn + kidx[:, None], mask=kmask[:, None] & nmask[None, :], other=0)
            acc2 = tl.dot(xh, wu.to(tl.float16), acc2)
    if NORM:
        inv = tl.rsqrt(ss / K + eps)
        acc = acc * inv[:, None]
        if SWIGLU:
            acc2 = acc2 * inv[:, None]
    if SCALE:
        acc = acc * tl.load(s_ptr + rn, mask=nmask, other=0.0)[None, :]
        if SWIGLU:
            acc2 = acc2 * tl.load(s_ptr + N + rn, mask=nmask, other=0.0)[None, :]
    if SWIGLU:
        acc = acc / (1.0 + tl.exp(-acc)) * acc2
    if BIAS:
        if pid_k == 0:
            acc += tl.load(b_ptr + rn, mask=nmask, other=0.0)[None, :].to(tl.float32)
    yp = y_ptr + rm[:, None] * sym + rn[None, :]
    omask = mmask[:, None] & nmask[None, :]
    if ACCUM:
        if SPLIT > 1:
            tl.atomic_add(yp, acc, mask=omask, sem='relaxed')
        else:
            tl.store(yp, (tl.load(yp, mask=omask, other=0.0).to(tl.float32) + acc).to(y_ptr.dtype.element_ty), mask=omask)
    else:
        tl.store(yp, acc.to(y_ptr.dtype.element_ty), mask=omask)


def wgemm(x, w, scale=None, bias=None, norm_eps=None, swiglu=False, cfg=None, out=None, accum=False):
    """x [M,K] (fp16 or fp32, unit stride along K), w [N(*2 if swiglu),K] fp16/int8, scale [N(*2)] fp32.
    accum=True: out[M,N] += x @ w^T (out must be given; fp32 for split-K). Else returns a new fp16 y."""
    M, K = x.shape
    N = w.shape[0] // (2 if swiglu else 1)
    BN, BK, SPLIT, nw, ns = cfg or (64, 128, 1, 4, 4)
    assert not (SPLIT > 1 and (norm_eps is not None or swiglu or not accum)), 'split-K only for accumulate GEMMs'
    BM = max(16, triton.next_power_of_2(M))
    assert BM <= 256, 'one M tile only (batch-1 tree forwards)'
    # keep the pipelined tiles inside this GPU's shared memory (fp32 x tiles are large at BM=256)
    lim = _smem(x.device)
    while ns > 2 and ns * (BM * BK * x.element_size() + BK * BN * 2 * (2 if swiglu else 1)) > lim:
        ns -= 1
    while BK > 32 and ns * (BM * BK * x.element_size() + BK * BN * 2 * (2 if swiglu else 1)) > lim:
        BK //= 2
    if out is None:
        assert not accum
        out = torch.empty(M, N, device=x.device, dtype=torch.float16)
    grid = (triton.cdiv(N, BN), SPLIT)
    _wgemm_kernel[grid](x, w, scale if scale is not None else x, bias if bias is not None else x, out,
                        M, N, K, x.stride(0), w.stride(0), out.stride(0), float(norm_eps or 0.0),
                        BM=BM, BN=BN, BK=BK, SPLIT=SPLIT, NORM=norm_eps is not None, SCALE=scale is not None,
                        BIAS=bias is not None, ACCUM=accum, SWIGLU=swiglu, num_warps=nw, num_stages=ns)
    return out


_SMEM = {}


def _smem(dev):
    i = dev.index if dev.index is not None else torch.cuda.current_device()
    if i not in _SMEM:
        p = torch.cuda.get_device_properties(i)
        _SMEM[i] = int(getattr(p, 'shared_memory_per_block_optin', 101376)) - 8192
    return _SMEM[i]


def quant_int8(w):
    """Symmetric per-output-channel INT8: w ~= q * s[:, None]."""
    s = w.float().abs().amax(1).clamp(min=1e-8) / 127.0
    q = torch.round(w.float() / s[:, None]).clamp(-127, 127).to(torch.int8)
    return q.contiguous(), s.contiguous()
