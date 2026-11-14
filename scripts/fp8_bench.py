"""Ceiling check for static-scale FP8 on Hopper: GEMM-only time of torch._scaled_mm (e4m3, static
scales, fp16/bf16 out) vs F.linear in FP16 at the engine's shapes. usage: fp8_bench.py [--out f.json]"""
import argparse, json
import torch
import torch.nn.functional as F

SHAPES = {'qkv': (2048, 1536), 'o': (1536, 1536), 'gate_up': (17920, 1536), 'down': (1536, 8960)}


def timeit(fn, reps=100):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); g.replay(); e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out'); a = ap.parse_args()
    dev = 'cuda'
    res = {'gpu': torch.cuda.get_device_name(0), 'rows': []}
    for M in (16, 64, 144, 224, 704):
        tot = {'fp16': 0.0, 'fp8_tensor': 0.0, 'fp8_row_bf16': 0.0}
        for name, (N, K) in SHAPES.items():
            x = torch.randn(M, K, device=dev, dtype=torch.float16)
            w = torch.randn(N, K, device=dev, dtype=torch.float16) * 0.02
            x8 = x.to(torch.float8_e4m3fn); w8 = w.to(torch.float8_e4m3fn)
            one = torch.ones((), device=dev)
            sa = torch.ones(M, 1, device=dev); sb = torch.ones(1, N, device=dev)
            t16 = timeit(lambda: F.linear(x, w))
            t8 = timeit(lambda: torch._scaled_mm(x8, w8.t(), scale_a=one, scale_b=one, out_dtype=torch.float16))
            try:
                t8r = timeit(lambda: torch._scaled_mm(x8, w8.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))
            except Exception as ex:
                t8r = float('nan')
            row = {'M': M, 'op': name, 'fp16_ms': t16, 'fp8_tensor_ms': t8, 'fp8_row_bf16_ms': t8r,
                   'fp16_GBps': N * K * 2 / t16 / 1e6, 'fp8_GBps': N * K / t8 / 1e6}
            res['rows'].append(row); print(json.dumps(row), flush=True)
            tot['fp16'] += t16; tot['fp8_tensor'] += t8; tot['fp8_row_bf16'] += t8r
        print(json.dumps({'M': M, 'per_forward_28_layers_ms': {k: 28 * v for k, v in tot.items()}}), flush=True)
        res.setdefault('per_forward', []).append({'M': M, **{k: 28 * v for k, v in tot.items()}})
    if a.out:
        open(a.out, 'w').write(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
