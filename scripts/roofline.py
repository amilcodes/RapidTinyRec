"""Forward cost vs tokens per forward (the batch-1 cost model).

Times one full Qwen2 forward (all layers + LM head) for M query tokens that attend to a shared
~100-token prompt, under CUDA-graph replay, for several M and precisions/quant schemes.
"""
import argparse, json, sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rapidtinyrec.speedrun.model import Qwen2Weights, Forward
from rapidtinyrec.speedrun import data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dtype', default='bfloat16')
    ap.add_argument('--quant', default='none')
    ap.add_argument('--ms', default='1,8,16,32,50,64,88,100,128,192,256,384,512,768,1024')
    ap.add_argument('--head', default='full', choices=['full', 'none'])
    a = ap.parse_args()
    dev = torch.device('cuda')
    dt = getattr(torch, a.dtype)
    quant = None
    if a.quant != 'none':
        from rapidtinyrec.speedrun import quant as Q
        quant = Q.get(a.quant)
    W = Qwen2Weights(Path(a.release) / data.CKPT, dt, dev, quant=quant)
    fwd = Forward(W)
    P = 112
    kc = torch.randn(W.L, 1, W.KV, P, W.hd, device=dev, dtype=dt)
    vc = torch.randn_like(kc)
    lengths = torch.tensor([100], device=dev)
    pmask = torch.where(torch.arange(P, device=dev)[None] < 100, 0.0, torch.finfo(torch.float32).min)
    res = []
    for M in [int(x) for x in a.ms.split(',')]:
        kp = torch.zeros(W.L, 1, M, 4, W.KV, W.hd, device=dev, dtype=dt)
        vp = torch.zeros_like(kp)
        x = torch.randn(1, M, 1, W.D, device=dev, dtype=dt) * 0.1
        pos = torch.full((1, M, 1), 100, device=dev)

        def run():
            h = fwd.step(x, pos, kc, vc, pmask, kp, vp, 0, lengths=lengths)
            if a.head == 'full':
                lg = W.linear(h, W.lm_head)
                return torch.logsumexp(lg.float(), -1)
            return h
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(g):
            run()
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        ts = []
        for _ in range(30):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        ts.sort()
        r = {'M': M, 'median_ms': ts[len(ts) // 2], 'min_ms': ts[0], 'weight_gb': W.param_bytes() / 1e9}
        r['eff_gbps'] = r['weight_gb'] / (r['median_ms'] / 1e3)
        res.append(r)
        print(json.dumps(r), flush=True)
        del g, kp, vp
        torch.cuda.empty_cache()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump({'args': vars(a), 'gpu': torch.cuda.get_device_name(0), 'results': res},
              open(Path(a.out) / 'roofline.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
