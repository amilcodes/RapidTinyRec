"""Speedrun benchmark: isolated latency on the frozen 500-user cohort, quality on
the full Office test set, and batched throughput.

Timing boundary (latency): pinned host prompt -> device copy -> one CUDA-graph
replay of the whole request (prefill, 3 constrained beam steps, forced-suffix
scoring, final sort) -> synchronize. Outputs stay on the device inside the
timer; a second timer also includes the device->host copy of the 50 ranked
SIDs and scores. Model load, compilation, graph capture and warmup excluded.
"""
import argparse, json, math, os, statistics, subprocess, sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rapidtinyrec.speedrun.model import Qwen2Weights
from rapidtinyrec.speedrun.beam import SidCatalog, BeamEngine, GraphRunner
from rapidtinyrec.speedrun.exact import ExactEngine, ExactEngine3
from rapidtinyrec.speedrun import data


def pct(v, q):
    v = sorted(v); i = (len(v) - 1) * q; lo, hi = math.floor(i), math.ceil(i)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def dist(v):
    return {'n': len(v), 'median_ms': 1e3 * statistics.median(v), 'p95_ms': 1e3 * pct(v, .95),
            'mean_ms': 1e3 * statistics.mean(v), 'min_ms': 1e3 * min(v), 'max_ms': 1e3 * max(v)}


def bucket(n, q):
    return (n + q - 1) // q * q


def agg_metrics(per_user):
    keys = per_user[0].keys()
    return {k: statistics.mean(m[k] for m in per_user) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ref', help='historical FP32 reference results dir (users/NNNN.json)')
    ap.add_argument('--dtype', default='bfloat16', choices=['float32', 'bfloat16', 'float16'])
    ap.add_argument('--tf32', action='store_true')
    ap.add_argument('--suffix', default='fused', choices=['fused', 'separate', 'none'])
    ap.add_argument('--compile', default='none', choices=['none', 'default', 'max-autotune-no-cudagraphs'])
    ap.add_argument('--quant', default='none')
    ap.add_argument('--pad-quantum', type=int, default=16)
    ap.add_argument('--modes', default='latency,quality,throughput')
    ap.add_argument('--quality-batch', type=int, default=16)
    ap.add_argument('--throughput-batches', default='1,4,8,16,32')
    ap.add_argument('--throughput-users', type=int, default=1024)
    ap.add_argument('--repeats', type=int, default=1, help='timed passes over the cohort')
    ap.add_argument('--no-graph', action='store_true')
    ap.add_argument('--engine', default='beam', choices=['beam', 'exact', 'exact3'])
    ap.add_argument('--n-prefix', type=int, default=128)
    ap.add_argument('--norm', default='full', choices=['full', 'codebook', 'trie'])
    ap.add_argument('--attn', default='torch', choices=['torch', 'triton'])
    ap.add_argument('--fuse-first', action='store_true', help='trie-level prefill fusion')
    ap.add_argument('--n-expand', type=int, default=64)
    ap.add_argument('--profile', action='store_true', help='eager kernel-time breakdown for a few users')
    ap.add_argument('--check-exhaustive', type=int, default=0, help='compare first K cohort users against exhaustive exact scoring')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = a.tf32
    torch.backends.cudnn.allow_tf32 = a.tf32
    dev = torch.device('cuda')
    dt = getattr(torch, a.dtype)
    meta = {'dataset': data.DATASET, 'args': vars(a), 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
            'nvidia_smi': subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem',
                                          '--format=csv,noheader'], capture_output=True, text=True).stdout.strip(),
            'host': os.uname().nodename, 'slurm_job': os.environ.get('SLURM_JOB_ID')}
    print(json.dumps(meta), flush=True)
    tok = data.load_tokenizer(a.release)
    rows = data.test_rows(a.release)
    t0 = time.time()
    all_ids = data.prompts(a.release, tok, rows)
    targets = [data.sid_tokens(tok, r['item_sid']) for r in rows]
    cohort = data.cohort500(rows)
    cat = SidCatalog(tok, Path(a.release) / data.CATALOG, dev)
    quant = None
    if a.quant != 'none':
        from rapidtinyrec.speedrun import quant as Q
        quant = Q.get(a.quant)
    W = Qwen2Weights(Path(a.release) / data.CKPT, dt, dev, quant=quant)
    if a.engine == 'beam':
        eng = BeamEngine(W, cat, suffix_mode=a.suffix, norm=a.norm, fuse_first=a.fuse_first)
    elif a.engine == 'exact':
        eng = ExactEngine(W, cat, n_expand=a.n_expand, suffix_mode=a.suffix, norm=a.norm, fuse_first=a.fuse_first)
    else:
        eng = ExactEngine3(W, cat, n_prefix=a.n_prefix, n_expand=a.n_expand, suffix_mode=a.suffix, norm=a.norm, fuse_first=a.fuse_first)
    if a.attn == 'triton':
        from rapidtinyrec.speedrun import kernels
        chk = kernels.selfcheck(dt)
        meta['attn_selfcheck_maxabs'] = chk
        print(json.dumps({'stage': 'attn_selfcheck', **chk}), flush=True)
        assert chk['pass'], chk
        eng.fwd.attn = 'triton'
    if a.compile != 'none':
        import torch._inductor.config as ic
        import torch._dynamo.config as dc
        ic.coordinate_descent_tuning = True
        dc.cache_size_limit = 1024
        if hasattr(dc, 'recompile_limit'):
            dc.recompile_limit = 1024
        dc.automatic_dynamic_shapes = False
        eng.fwd.qkv = torch.compile(eng.fwd.qkv, mode=a.compile, dynamic=False, fullgraph=True)
        eng.fwd.post = torch.compile(eng.fwd.post, mode=a.compile, dynamic=False, fullgraph=True)
    torch.cuda.synchronize()
    meta.update(setup_load_s=time.time() - t0, catalog=cat.stats, weight_bytes=W.param_bytes(),
                alloc_after_load=torch.cuda.memory_allocated())
    print(json.dumps({'stage': 'loaded', **{k: meta[k] for k in ('setup_load_s', 'catalog', 'weight_bytes')}}), flush=True)
    runners = {}

    def runner(B, P):
        key = (B, P)
        if key not in runners:
            s = time.time()
            runners[key] = GraphRunner(eng, B, P, tok.eos_token_id, use_graph=not a.no_graph)
            meta.setdefault('capture', []).append({'B': B, 'P': P, 's': time.time() - s,
                                                   'alloc': torch.cuda.memory_allocated()})
        return runners[key]

    modes = a.modes.split(',')
    results = {'meta': meta}
    if a.profile:
        from torch.profiler import profile, ProfilerActivity
        P = bucket(len(all_ids[cohort[0]]), a.pad_quantum)
        bufs = eng.alloc(1, P)
        t = torch.full((1, P), tok.eos_token_id, dtype=torch.long, device=dev)
        t[0, :len(all_ids[cohort[0]])] = torch.tensor(all_ids[cohort[0]], device=dev)
        ln = torch.tensor([len(all_ids[cohort[0]])], device=dev)
        with torch.inference_mode():
            for _ in range(3):
                eng.run(t, ln, bufs)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(5):
                    eng.run(t, ln, bufs)
                torch.cuda.synchronize()
        tab = prof.key_averages().table(sort_by='cuda_time_total', row_limit=45)
        (out / 'profile.txt').write_text(tab)
        evs = [e for e in prof.key_averages()]
        tot = sum(getattr(e, 'self_device_time_total', getattr(e, 'self_cuda_time_total', 0)) for e in evs) / 5 / 1e3
        cnt = sum(e.count for e in evs) / 5
        meta['profile_gpu_ms_per_request'] = tot
        meta['profile_kernels_per_request'] = cnt
        print(json.dumps({'stage': 'profile', 'gpu_kernel_ms_per_request': tot, 'kernels_per_request': cnt}), flush=True)
        print(tab, flush=True)
    # ---------------- latency (batch 1, frozen cohort)
    if 'latency' in modes:
        lens = sorted({bucket(len(all_ids[i]), a.pad_quantum) for i in cohort})
        s = time.time()
        for P in lens:
            runner(1, P)
        meta['latency_capture_s'] = time.time() - s
        staged = []
        for i in cohort:
            r = runner(1, bucket(len(all_ids[i]), a.pad_quantum))
            staged.append((i, r, *r.load([all_ids[i]])))
        # untimed warm pass
        for i, r, t, ln in staged[:20]:
            r(t, ln)
        torch.cuda.synchronize()
        times, times_d2h, per_user, recs, recs_cert = [], [], [], [], []
        for rep in range(a.repeats):
            for i, r, t, ln in staged:
                torch.cuda.synchronize()
                st = time.perf_counter()
                o = r(t, ln)
                torch.cuda.synchronize()
                e1 = time.perf_counter()
                sids = o['sids'].cpu(); sc = o['scores'].cpu(); tr = o['trans'].cpu() if 'trans' in o else None
                e2 = time.perf_counter()
                times.append(e1 - st); times_d2h.append(e2 - st)
                if rep == 0:
                    if 'certified' in o:
                        recs_cert.append(o['certified'][0].cpu().tolist())
                    ranked = [tuple(x) for x in sids[0].tolist()]
                    per_user.append(data.rank_metrics(ranked, targets[i]))
                    recs.append({'row': i, 'L': len(all_ids[i]), 'ms': 1e3 * (e1 - st), 'sids': [list(x) for x in ranked],
                                 'scores': sc[0].tolist(), 'trans': tr[0].tolist() if tr is not None else None})
        lat = {'device_only': dist(times), 'with_d2h': dist(times_d2h), 'metrics_500': agg_metrics(per_user),
               'by_length': {}}
        if recs_cert:
            ks = [k for k in eng.ks if k <= eng.N]
            lat['certified_rate'] = {f'top{k}': statistics.mean(c[j] for c in recs_cert) for j, k in enumerate(ks)}
        for L in sorted({r['L'] for r in recs}):
            v = [r['ms'] / 1e3 for r in recs if r['L'] == L]
            lat['by_length'][L] = dist(v)
        if a.ref and data.DATASET == 'office':
            agree = {'rank_positions_equal': 0, 'top50_overlap': 0, 'top1_equal': 0, 'top10_set_equal': 0,
                     'lists_identical': 0, 'max_abs_score_diff': 0.0}
            for k, rec in enumerate(recs):
                ref = json.load(open(Path(a.ref) / 'users' / f'{k:04d}.json'))
                rr = ref['runs']['compiled_gqa']
                rs = [tuple(x[:3]) for x in rr['token_ids']]
                ms = [tuple(x) for x in rec['sids']][:50]
                eq = sum(x == y for x, y in zip(rs, ms))
                agree['rank_positions_equal'] += eq
                agree['lists_identical'] += eq == 50
                agree['top50_overlap'] += len(set(rs) & set(ms))
                agree['top1_equal'] += rs[0] == ms[0]
                agree['top10_set_equal'] += set(rs[:10]) == set(ms[:10])
                refsc = dict(zip(rs, rr['sequence_logp']))
                d = [abs(refsc[s] - v) for s, v in zip(ms, rec['scores']) if s in refsc]
                agree['max_abs_score_diff'] = max(agree['max_abs_score_diff'], max(d))
            lat['agreement_vs_fp32_reference'] = agree
        if a.check_exhaustive:
            ref_eng = ExactEngine(W, cat, n_expand=64, suffix_mode=a.suffix if a.suffix != 'separate' else 'fused', norm=a.norm)
            ref_eng.fwd.attn = eng.fwd.attn
            st = {'n': 0, 'top1_match': 0, 'cert_violations': 0, 'cert_top10_users': 0}
            for k in (1, 5, 10, 20, 50):
                st[f'search_recall@{k}'] = 0.0
            ex_metrics, ex_lists = [], []
            s_ex = time.time()
            for rec in recs[:a.check_exhaustive]:
                i = rec['row']
                P = bucket(len(all_ids[i]), a.pad_quantum)
                t = torch.full((1, P), tok.eos_token_id, dtype=torch.long, device=dev)
                t[0, :len(all_ids[i])] = torch.tensor(all_ids[i], device=dev)
                sc, _ = ref_eng.exhaustive(t, torch.tensor([len(all_ids[i])], device=dev))
                order = sc.argsort(descending=True)[:100].tolist()
                glob = [tuple(ref_eng.sid_tok[j].tolist()) for j in order]
                mine = [tuple(x) for x in rec['sids']]
                st['n'] += 1
                st['top1_match'] += mine[0] == glob[0]
                for k in (1, 5, 10, 20, 50):
                    st[f'search_recall@{k}'] += len(set(mine[:k]) & set(glob[:k])) / k
                if recs_cert and len(recs_cert[st['n'] - 1]) > 2:
                    c10 = recs_cert[st['n'] - 1][2]
                    st['cert_top10_users'] += c10
                    tol = 1e-3 if dt == torch.float32 else 5e-2
                    allsc = dict(zip([tuple(x) for x in ref_eng.sid_tok.tolist()], sc.tolist()))
                    th = allsc[mine[9]]
                    viol = any(g not in mine[:10] and allsc[g] > th + tol for g in glob[:10])
                    st['cert_violations'] += bool(c10) and viol
                ex_metrics.append(data.rank_metrics(glob[:50], targets[i]))
                ex_lists.append({'row': i, 'top100': [list(x) for x in glob], 'scores': sc[order].tolist()})
            for k in (1, 5, 10, 20, 50):
                st[f'search_recall@{k}'] /= st['n']
            st['top1_match'] /= st['n']
            st['exhaustive_metrics'] = agg_metrics(ex_metrics)
            st['engine_metrics_same_users'] = agg_metrics(per_user[:st['n']])
            st['wall_s'] = time.time() - s_ex
            lat['vs_exhaustive'] = st
            json.dump(ex_lists, open(out / 'exhaustive_users.json', 'w'))
            print(json.dumps({'stage': 'exhaustive', **st}), flush=True)
        results['latency'] = lat
        json.dump(recs, open(out / 'latency_users.json', 'w'))
        print(json.dumps({'stage': 'latency', **{k: v for k, v in lat.items() if k != 'by_length'}}), flush=True)
    # ---------------- quality (full test set, batched)
    if 'quality' in modes:
        B = a.quality_batch
        order = sorted(range(len(rows)), key=lambda i: len(all_ids[i]))
        per_user = [None] * len(rows)
        tops = [None] * len(rows)
        certs = []
        s = time.time()
        for k in range(0, len(order), B):
            chunk = order[k:k + B]
            P = bucket(max(len(all_ids[i]) for i in chunk), a.pad_quantum)
            r = runner(B, P)
            idl = [all_ids[i] for i in chunk] + [all_ids[chunk[0]]] * (B - len(chunk))
            t, ln = r.load(idl)
            o = r(t, ln)
            sids = o['sids'].cpu().tolist()
            if 'certified' in o:
                certs.extend(o['certified'].cpu().tolist()[:len(chunk)])
            for j, i in enumerate(chunk):
                ranked = [tuple(x) for x in sids[j]]
                per_user[i] = data.rank_metrics(ranked, targets[i])
                tops[i] = ranked[:10]
        torch.cuda.synchronize()
        q = {'n': len(rows), 'metrics': agg_metrics(per_user), 'wall_s': time.time() - s}
        if certs:
            ks = [k for k in eng.ks if k <= eng.N]
            q['certified_rate'] = {f'top{k}': statistics.mean(c[j] for c in certs) for j, k in enumerate(ks)}
        results['quality'] = q
        json.dump({'per_user': per_user, 'top10': tops}, open(out / 'quality_users.json', 'w'))
        print(json.dumps({'stage': 'quality', **q}), flush=True)
    # ---------------- throughput (batched users, sorted-by-length batches)
    if 'throughput' in modes:
        users = list(range(min(a.throughput_users, len(rows))))
        tp = {}
        for B in [int(x) for x in a.throughput_batches.split(',')]:
            order = sorted(users, key=lambda i: len(all_ids[i]))
            batches = []
            for k in range(0, len(order), B):
                chunk = order[k:k + B]
                if len(chunk) < B:
                    break
                P = bucket(max(len(all_ids[i]) for i in chunk), a.pad_quantum)
                r = runner(B, P)
                batches.append((r, *r.load([all_ids[i] for i in chunk])))
            for r, t, ln in batches[:3]:
                r(t, ln)
            torch.cuda.synchronize()
            lat = []
            s = time.perf_counter()
            for r, t, ln in batches:
                st = time.perf_counter()
                o = r(t, ln)
                torch.cuda.synchronize()
                lat.append(time.perf_counter() - st)
            el = time.perf_counter() - s
            n = len(batches) * B
            tp[B] = {'users': n, 'users_per_s': n / el, 'batch_latency': dist(lat),
                     'peak_alloc_gb': torch.cuda.max_memory_allocated() / 2**30}
            print(json.dumps({'stage': 'throughput', 'B': B, **tp[B]}), flush=True)
        results['throughput'] = tp
    meta['max_alloc_gb'] = torch.cuda.max_memory_allocated() / 2**30
    json.dump(results, open(out / 'results.json', 'w'), indent=1)
    print(json.dumps({'stage': 'done'}), flush=True)


if __name__ == '__main__':
    main()
