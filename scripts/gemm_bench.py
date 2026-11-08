"""cuBLAS vs split-K Triton for the engine's skinny GEMM shapes; prints achieved weight bandwidth.

usage: gemm_bench.py [--ms 16,64,140,216] [--out res.json]
"""
import argparse, json, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rapidtinyrec.speedrun.skinny import tune

SHAPES = {'qkv': (2048, 1536), 'o': (1536, 1536), 'gate_up': (17920, 1536), 'down': (1536, 8960)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ms', default='16,64,140,216'); ap.add_argument('--out')
    a = ap.parse_args()
    gpu = torch.cuda.get_device_name(0)
    res = {'gpu': gpu, 'rows': []}
    for M in (int(x) for x in a.ms.split(',')):
        tot_c, tot_s = 0.0, 0.0
        for name, (N, K) in SHAPES.items():
            cfg, ms, cub = tune(M, N, K)
            gb = N * K * 2 / 1e9
            row = {'M': M, 'op': name, 'N': N, 'K': K, 'cublas_ms': cub, 'splitk_ms': ms, 'cfg': cfg,
                   'cublas_GBps': gb / cub * 1e3, 'splitk_GBps': gb / ms * 1e3 if cfg else None}
            res['rows'].append(row)
            tot_c += cub; tot_s += min(ms, cub)
            print(json.dumps(row), flush=True)
        res.setdefault('per_layer', []).append({'M': M, 'cublas_ms': tot_c, 'best_ms': tot_s, 'x28_cublas': 28 * tot_c, 'x28_best': 28 * tot_s})
        print(json.dumps(res['per_layer'][-1]), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
