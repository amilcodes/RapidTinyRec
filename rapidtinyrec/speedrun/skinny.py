"""Split-K GEMM for skinny activations (M = 16..256 rows) against wide weights.

y[M, N] = x[M, K] @ w[N, K]^T (+ bias) (+ residual)

At batch 1 the engine's forwards carry 64-216 rows. For the projections whose output is narrow
(qkv N=2048, o N=1536, down N=1536 with K=8960) a standard tiling launches too few thread blocks to
stream the weights at full bandwidth. Splitting K across blocks restores parallelism; partial sums
are accumulated in fp32 with atomics, then one small epilogue adds bias / residual and casts.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _splitk_kernel(x_ptr, w_ptr, acc_ptr, M, N, K, sxm, sxk, swn, swk, K_PER,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    k0 = pid_k * K_PER
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, K_PER, BK):
        rk = k0 + kk + tl.arange(0, BK)
        a = tl.load(x_ptr + rm[:, None] * sxm + rk[None, :] * sxk, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(w_ptr + rn[None, :] * swn + rk[:, None] * swk, mask=(rn[None, :] < N) & (rk[:, None] < K), other=0.0)
        acc += tl.dot(a, b)
    tl.atomic_add(acc_ptr + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N), sem='relaxed')


@triton.jit
def _epilogue(acc_ptr, b_ptr, r_ptr, y_ptr, M, N, HAS_B: tl.constexpr, HAS_R: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < M * N
    v = tl.load(acc_ptr + i, mask=m, other=0.0)
    if HAS_B:
        v += tl.load(b_ptr + i % N, mask=m, other=0.0).to(tl.float32)
    if HAS_R:
        v += tl.load(r_ptr + i, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + i, v.to(y_ptr.dtype.element_ty), mask=m)


def splitk_linear(x, w, bias=None, residual=None, cfg=(16, 128, 128, 4), num_warps=4, num_stages=3, acc=None):
    """x [..., K], w [N, K] -> [..., N]. cfg = (BM, BN, BK, SPLIT)."""
    shp = x.shape
    x2 = x.reshape(-1, shp[-1])
    M, K = x2.shape
    N = w.shape[0]
    BM, BN, BK, SPLIT = cfg
    k_per = triton.cdiv(triton.cdiv(K, SPLIT), BK) * BK
    split = triton.cdiv(K, k_per)
    if acc is None:
        acc = torch.zeros(M, N, device=x.device, dtype=torch.float32)
    else:
        acc.zero_()
    grid = (triton.cdiv(N, BN), triton.cdiv(M, BM), split)
    _splitk_kernel[grid](x2, w, acc, M, N, K, x2.stride(0), x2.stride(1), w.stride(0), w.stride(1), k_per,
                         BM=BM, BN=BN, BK=BK, num_warps=num_warps, num_stages=num_stages)
    y = torch.empty(M, N, device=x.device, dtype=x.dtype)
    r = residual.reshape(M, N) if residual is not None else y
    BLOCK = 1024
    _epilogue[(triton.cdiv(M * N, BLOCK),)](acc, bias if bias is not None else y, r, y, M, N,
                                           HAS_B=bias is not None, HAS_R=residual is not None, BLOCK=BLOCK)
    return y.reshape(*shp[:-1], N)


def bm_for(M):
    return 16 if M <= 16 else 32 if M <= 32 else 64 if M <= 64 else 128


def tune(M, N, K, dtype=torch.float16, device='cuda', reps=50):
    """Return (best_cfg, best_ms, cublas_ms) for one shape."""
    x = torch.randn(M, K, device=device, dtype=dtype)
    w = torch.randn(N, K, device=device, dtype=dtype) * 0.02
    ref = torch.nn.functional.linear(x, w)

    def t(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(reps):
                fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / reps

    cub = t(lambda: torch.nn.functional.linear(x, w))
    best = (None, float('inf'))
    acc = torch.zeros(M, N, device=device, dtype=torch.float32)
    for BM in sorted({bm_for(M), 16, 32, 64} if M <= 64 else {64, 128}):
        for BN in (32, 64, 128):
            for BK in (64, 128, 256):
                for SPLIT in (1, 2, 4, 8, 16):
                    if K // SPLIT < BK:
                        continue
                    for nw in (4, 8):
                        for ns in (2, 3, 4):
                            cfg = (BM, BN, BK, SPLIT)
                            try:
                                y = splitk_linear(x, w, cfg=cfg, num_warps=nw, num_stages=ns, acc=acc)
                                if (y - ref).abs().max().item() > 5e-2 * ref.abs().max().item():
                                    continue
                                ms = t(lambda: splitk_linear(x, w, cfg=cfg, num_warps=nw, num_stages=ns, acc=acc))
                            except Exception:
                                continue
                            if ms < best[1]:
                                best = ((cfg, nw, ns), ms)
    return best[0], best[1], cub
