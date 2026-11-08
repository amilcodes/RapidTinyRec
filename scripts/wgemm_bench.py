"""Benchmark the weight-streaming Triton GEMM (FP16 / INT8 weights) against cuBLAS at the layer shapes of
Qwen2-1.5B, at batch-1 tree-forward row counts. Each timing is a CUDA graph of 28 calls over 28 distinct
weight copies (one per layer, so nothing is served from L2), divided by 28.

usage: wgemm_bench.py OUT.json
"""
import itertools, json, sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rapidtinyrec.speedrun.wgemm import wgemm, quant_int8

dev = torch.device('cuda')
L = 28
SHAPES = {'qkv': (1536, 2048), 'o': (1536, 1536), 'gu': (1536, 17920), 'down': (8960, 1536)}
MS = [16, 64, 144, 224]


def graph_time(fn, reps=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay(); torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / L)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    out = {}
    torch.manual_seed(0)
    for name, (K, N) in SHAPES.items():
        ws = [torch.randn(N, K, device=dev, dtype=torch.float16) * 0.02 for _ in range(L)]
        qs = [quant_int8(w) for w in ws]
        for M in MS:
            x = torch.randn(M, K, device=dev, dtype=torch.float16)
            ref = F.linear(x, ws[0]).float()
            row = {'bytes_fp16_MB': N * K * 2 / 1e6}
            row['cublas_us'] = graph_time(lambda: [F.linear(x, w) for w in ws])
            for kind in ('fp16', 'int8'):
                best = None
                for BN, BK, SPLIT, nw, ns in itertools.product((32, 64, 128), (64, 128, 256), (1, 2, 4, 8), (4, 8), (3, 4)):
                    BM = max(16, 1 << (M - 1).bit_length())
                    if BM * BN > 256 * 64 or (SPLIT > 1 and K // SPLIT < BK) or (N // BN) * SPLIT > 1024:
                        continue
                    if SPLIT > 1 and N // BN >= 132:
                        continue
                    cfg = (BN, BK, SPLIT, nw, ns)
                    try:
                        if kind == 'fp16':
                            y = wgemm(x, ws[0], cfg=cfg)
                            fn = lambda: [wgemm(x, w, cfg=cfg) for w in ws]
                        else:
                            y = wgemm(x, qs[0][0], scale=qs[0][1], cfg=cfg)
                            fn = lambda: [wgemm(x, q, scale=s, cfg=cfg) for q, s in qs]
                        err = (y.float() - ref).abs().max().item()
                        if kind == 'fp16' and err > 0.05 or kind == 'int8' and err > 0.2:
                            continue
                        t = graph_time(fn, reps=10)
                    except Exception as e:  # noqa: BLE001 - resource limits for some configs
                        continue
                    if best is None or t < best[0]:
                        best = (t, cfg, err)
                if best:
                    row[f'{kind}_us'], row[f'{kind}_cfg'], row[f'{kind}_err'] = best
                    row[f'{kind}_TBps'] = (N * K * (2 if kind == 'fp16' else 1)) / best[0] / 1e6
            row['cublas_TBps'] = N * K * 2 / row['cublas_us'] / 1e6
            out[f'{name}/M{M}'] = row
            print(json.dumps({'stage': 'wgemm', 'shape': name, 'M': M, **row}), flush=True)
        del ws, qs
        torch.cuda.empty_cache()
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
