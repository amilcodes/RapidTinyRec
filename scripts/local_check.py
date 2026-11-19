"""Local (CPU/MPS, eager) correctness check of the speedrun engine against
historical FP32 reference rankings for the first N cohort users."""
import argparse, json, sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rapidtinyrec.speedrun.model import Qwen2Weights
from rapidtinyrec.speedrun.beam import SidCatalog, BeamEngine, GraphRunner
from rapidtinyrec.speedrun import data

ap = argparse.ArgumentParser()
ap.add_argument('--release', required=True)
ap.add_argument('--ref', required=True)
ap.add_argument('--n', type=int, default=3)
ap.add_argument('--device', default='mps')
ap.add_argument('--dtype', default='float32')
ap.add_argument('--suffix', default='fused')
ap.add_argument('--batch', type=int, default=1)
a = ap.parse_args()
dev = torch.device(a.device)
dt = getattr(torch, a.dtype)
tok = data.load_tokenizer(a.release)
rows = data.test_rows(a.release)
sel = data.cohort500(rows)[:a.n]
ids = data.prompts(a.release, tok, [rows[i] for i in sel])
cat = SidCatalog(tok, Path(a.release) / data.CATALOG, dev)
print(cat.stats)
W = Qwen2Weights(Path(a.release) / 'Office_ckpt', dt, dev)
eng = BeamEngine(W, cat, suffix_mode=a.suffix)
for start in range(0, a.n, a.batch):
    chunk = list(range(start, min(a.n, start + a.batch)))
    P = max(len(ids[i]) for i in chunk)
    P = (P + 15) // 16 * 16
    gr = GraphRunner(eng, len(chunk), P, tok.eos_token_id)
    t, ln = gr.load([ids[i] for i in chunk])
    t0 = time.time()
    with torch.inference_mode():
        out = gr(t.to(dev), ln.to(dev))
    sids = out['sids'].cpu().tolist(); sc = out['scores'].cpu().tolist(); tr = out['trans'].cpu().tolist()
    el = time.time() - t0
    for j, u in enumerate(chunk):
        ref = json.load(open(Path(a.ref) / 'users' / f'{u:04d}.json'))
        assert len(ids[u]) == ref['prompt_tokens'], (len(ids[u]), ref['prompt_tokens'])
        r = ref['runs']['compiled_gqa']
        rs = [tuple(x[:3]) for x in r['token_ids']]
        ms = [tuple(x) for x in sids[j]]
        same = sum(x == y for x, y in zip(rs, ms))
        overlap = len(set(rs) & set(ms))
        ref_sc = dict(zip(rs, r['sequence_logp']))
        d = max(abs(ref_sc[s] - v) for s, v in zip(ms, sc[j]) if s in ref_sc)
        print(f'user {u} L={len(ids[u])} rank_match {same}/50 overlap {overlap}/50 max|dscore| {d:.2e} '
              f'top1 {ms[0]==rs[0]} t={el:.2f}s tr0={[round(x,4) for x in tr[j][0]]} ref_tr0={[round(x,4) for x in r["transition_logp"][0]]}')
