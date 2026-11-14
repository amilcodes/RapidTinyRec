"""FlashRec (sohu-mptc/FlashRec, a public SID-GR engine: CUDA-graph wide beam, fused trie kernel, restricted
per-codebook LM head, radix prefix cache, cross-request decode waves) on our exact workload, same GPU.

usage: python scripts/flashrec_bench.py --model CKPT --workload DIR --out DIR [--precision bf16|fp16|fp8]
       [--modes quality,latency,throughput] [--batches 16,32,64,128]

Runs in FlashRec's own environment (it does not import rapidtinyrec). The workload comes from
scripts/export_workload.py: the released evaluation's prompt token ids, target SIDs, the 500-user cohort, the
shared prompt prefix, and the SID layout. FlashRec needs scripts/flashrec_qwen2.py applied (MiniOneRec is Qwen2).

FlashRec's objective is beam search over a per-level codebook log-softmax with no suffix term: in our engine,
`--norm codebook --suffix none` beam-50, which is the agreement reference.

Latency boundary: the engine's offline API, `engine.generate(input_ids=..., n=50, max_tokens=3)`, wall time from
the call to the ranked beams on the host (one request at a time, after warmup). The shared prompt prefix is pinned
in FlashRec's radix cache (its counterpart of our template cache), and every other cached prefix is evicted before
each timed request, so a repeated prompt is never served from cache.
"""
import argparse, json, math, statistics, time
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--workload', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--precision', default='bf16', choices=['bf16', 'fp16', 'fp8'])
ap.add_argument('--modes', default='quality,latency,throughput')
ap.add_argument('--beam', type=int, default=50)
ap.add_argument('--batches', default='16,32,64,128')
ap.add_argument('--throughput-users', type=int, default=1024)
ap.add_argument('--repeats', type=int, default=2)
ap.add_argument('--graph-max-bs', type=int, default=0, help='cuda_graph_max_bs (default: 50 x max batch)')
ap.add_argument('--no-radix', action='store_true')
ap.add_argument('--candidates', type=int, default=0,
                help='per-beam candidates kept before the trie mask (FlashRec default 2 x beam); 0 = default')
ap.add_argument('--profile', type=int, default=0, help='CUDA-kernel time per request over N profiled requests')
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
W = json.loads((Path(a.workload) / 'workload.json').read_text())
modes = set(a.modes.split(','))
batches = [int(x) for x in a.batches.split(',')]

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')   # FlashRec logs JIT / graph status
import flashrec
import flashrec.engine.engine as fe
from flashrec import BeamRecConfig, BeamRecEngine

# The layout is passed explicitly (MiniOneRec's <a_*><b_*><c_*> tokens are not OneRec's <s_a_*> + <|sid_begin|>),
# so skip tokenizer-based inference, which would otherwise require the OneRec naming. No boundary tokens.
fe.resolve_sid_layout = lambda config: None
if a.candidates:
    # FlashRec keeps the top 2 x beam tokens of the level's codebook per beam, THEN applies the trie mask
    # (search/expand.py); a valid child ranked below that in the codebook can never be selected. Widening the cut
    # to the full codebook makes its search exact beam search under its own objective.
    import flashrec.core as fcore
    fcore.BeamRequest.beam_candidates = property(lambda self: max(int(a.candidates), int(self.beam_width)))

ids_flat = [t for lvl in W['sid_levels'] for t in lvl]
gmax = a.graph_max_bs or a.beam * max(batches + [16])
caps = sorted({a.beam * k for k in (1, 2, 4, 8, 16, 32, 64, 128) if a.beam * k <= gmax} | {gmax})
cfg = BeamRecConfig(
    model_path=a.model, beam_width=a.beam, max_tokens=3,
    quantization='fp8' if a.precision == 'fp8' else None,
    kv_cache_dtype='fp8_e4m3' if a.precision == 'fp8' else ('float16' if a.precision == 'fp16' else 'bfloat16'),
    compute_dtype='float16' if a.precision == 'fp16' else 'bfloat16',
    sid_token_range=','.join(map(str, ids_flat)),
    sid_codebook_sizes=','.join(map(str, W['codebook_sizes'])),
    sid_vocab_file=str(Path(a.workload) / 'catalog_tokens.json'),
    cuda_graph_max_bs=gmax, cuda_graph_capture_sizes=caps, batch_slots=gmax,
    max_running_requests=max(64, max(batches)), enable_radix=not a.no_radix)
meta = {'flashrec_version': flashrec.__version__, 'flashrec_commit': getattr(flashrec, '__git_commit__', None),
        'args': vars(a), 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'graph_capture_sizes': caps}
s = time.time()
eng = BeamRecEngine(cfg)
eng.ensure_cuda_graph()
meta['setup_s'] = time.time() - s
g = eng.runner.graph
meta['decode_graph_captured'] = g is not None
try:
    meta['decode_graph_detail'] = {k: (sorted(v) if isinstance(v, dict) else v) for k, v in vars(g).items()
                                   if k in ('graphs', 'capture_sizes', 'sizes', 'bs_list')} if g is not None else None
except Exception as exc:
    meta['decode_graph_detail'] = repr(exc)
P, T = W['prompt_ids'], [tuple(t) for t in W['targets']]
if eng.prefix_cache is not None:
    meta['template_pinned'] = eng.fill_prefix_kv(list(W['template_ids']))
print(json.dumps({'stage': 'loaded', **meta}), flush=True)


def evict():
    if eng.prefix_cache is not None:
        eng.prefix_cache.evict(1 << 30)          # unpinned leaves only; the pinned template stays


def ranked(res):
    return [tuple(int(x) for x in s.tokens[-3:]) for s in res.sequences]


def metrics(rk, tgt, ks=(1, 3, 5, 10, 20, 50)):
    r = rk.index(tgt) if tgt in rk else None
    m = {}
    for k in ks:
        hit = r is not None and r < k
        m[f'HR@{k}'] = float(hit); m[f'NDCG@{k}'] = 1 / math.log2(r + 2) if hit else 0.0
    return m


def agg(ms):
    return {k: statistics.mean(m[k] for m in ms) for k in ms[0]}


def pct(v, q):
    v = sorted(v); i = (len(v) - 1) * q; lo, hi = math.floor(i), math.ceil(i)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def dist(v):
    return {'n': len(v), 'median_ms': 1e3 * statistics.median(v), 'p95_ms': 1e3 * pct(v, .95),
            'mean_ms': 1e3 * statistics.mean(v), 'min_ms': 1e3 * min(v), 'max_ms': 1e3 * max(v)}


def req(i):
    return eng.make_request(input_ids=P[i], n=a.beam, max_tokens=3)


res = {'meta': meta}
cohort = W['cohort500']
# warmup: a few requests at batch 1 and a few bursts at each batch size
for i in cohort[:20]:
    eng.generate(input_ids=P[i], n=a.beam, max_tokens=3)
for B in batches:
    eng.generate_many([req(i) for i in range(B)])
evict(); torch.cuda.synchronize()

if a.profile:
    from torch.profiler import profile, ProfilerActivity
    ids = [cohort[k % len(cohort)] for k in range(a.profile)]
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in ids:
            evict()
            eng.generate(input_ids=P[i], n=a.beam, max_tokens=3)
        torch.cuda.synchronize()
    evs = list(prof.key_averages())
    tot = sum(getattr(e, 'self_device_time_total', getattr(e, 'self_cuda_time_total', 0)) for e in evs) / a.profile / 1e3
    cnt = sum(e.count for e in evs) / a.profile
    res['profile'] = {'requests': a.profile, 'gpu_kernel_ms_per_request': tot, 'kernels_per_request': cnt}
    (out / 'profile.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=40))
    print(json.dumps({'stage': 'profile', **res['profile']}), flush=True)
    evict(); torch.cuda.synchronize()

if 'latency' in modes:
    times, recs, per_user = [], [], []
    for rep in range(a.repeats):
        for i in cohort:
            evict(); torch.cuda.synchronize()
            st = time.perf_counter()
            r = eng.generate(input_ids=P[i], n=a.beam, max_tokens=3)
            e = time.perf_counter()
            times.append(e - st)
            if rep == 0:
                rk = ranked(r)
                per_user.append(metrics(rk, T[i]))
                recs.append({'row': i, 'L': len(P[i]), 'ms': 1e3 * (e - st), 'sids': [list(x) for x in rk],
                             'scores': [float(s.beam_score) for s in r.sequences]})
    res['latency'] = {'api_wall': dist(times), 'metrics_500': agg(per_user)}
    json.dump(recs, open(out / 'latency_users.json', 'w'))
    print(json.dumps({'stage': 'latency', **res['latency']}), flush=True)

if 'quality' in modes:
    s = time.time(); per_user, top10, invalid, n_beams = [], [], 0, []
    cat = {tuple(x) for x in json.loads((Path(a.workload) / 'catalog_tokens.json').read_text())}
    for c0 in range(0, len(P), 64):
        evict()
        rr = eng.generate_many([req(i) for i in range(c0, min(c0 + 64, len(P)))])
        for j, r in enumerate(rr):
            rk = ranked(r)
            invalid += sum(x not in cat for x in rk); n_beams.append(len(rk))
            per_user.append(metrics(rk, T[c0 + j])); top10.append([list(x) for x in rk[:10]])
    json.dump({'per_user': per_user, 'top10': top10}, open(out / 'quality_users.json', 'w'))   # bench.py format
    res['quality'] = {'n': len(per_user), 'metrics': agg(per_user), 'invalid_sids': invalid,
                      'beams_min': min(n_beams), 'wall_s': time.time() - s}
    print(json.dumps({'stage': 'quality', **res['quality']}), flush=True)

if 'throughput' in modes:
    tp = {}
    N = a.throughput_users
    order = [i % len(P) for i in range(N)]
    for B in batches:
        evict(); torch.cuda.synchronize()
        s = time.perf_counter()
        for c0 in range(0, N, B):                  # distinct prompts (N < test rows): no full-prompt cache hits
            eng.generate_many([req(i) for i in order[c0:c0 + B]])
        torch.cuda.synchronize()
        el = time.perf_counter() - s
        tp[str(B)] = {'users': N, 'seconds': el, 'users_per_s': N / el}
        print(json.dumps({'stage': 'throughput', 'batch': B, **tp[str(B)]}), flush=True)
    res['throughput'] = tp

json.dump(res, open(out / 'results.json', 'w'), indent=1)
print(json.dumps({'stage': 'done'}), flush=True)
