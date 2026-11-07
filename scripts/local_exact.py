"""Local consistency check: beam vs certified exact vs exhaustive (eager)."""
import argparse, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rapidtinyrec.speedrun.model import Qwen2Weights
from rapidtinyrec.speedrun.beam import SidCatalog, BeamEngine, GraphRunner
from rapidtinyrec.speedrun.exact import ExactEngine
from rapidtinyrec.speedrun import data
ap = argparse.ArgumentParser(); ap.add_argument('--release'); ap.add_argument('--n', type=int, default=3)
ap.add_argument('--device', default='mps'); ap.add_argument('--N', type=int, default=64); a = ap.parse_args()
dev = torch.device(a.device)
tok = data.load_tokenizer(a.release); rows = data.test_rows(a.release)
sel = data.cohort500(rows)[:a.n]; ids = data.prompts(a.release, tok, [rows[i] for i in sel])
cat = SidCatalog(tok, Path(a.release) / data.CATALOG, dev)
W = Qwen2Weights(Path(a.release) / 'Office_ckpt', torch.float32, dev)
beam = BeamEngine(W, cat); ex = ExactEngine(W, cat, n_expand=a.N)
with torch.inference_mode():
  for u in range(a.n):
    P = (len(ids[u]) + 15) // 16 * 16
    t = torch.full((1, P), tok.eos_token_id, dtype=torch.long); t[0, :len(ids[u])] = torch.tensor(ids[u]); t = t.to(dev)
    ln = torch.tensor([len(ids[u])], device=dev)
    ob = beam.run(t, ln, beam.alloc(1, P))
    oe = ex.run(t, ln, ex.alloc(1, P))
    sc, u2 = ex.exhaustive(t, ln)
    full = {tuple(s): v for s, v in zip(ex.sid_tok.tolist(), sc.tolist())}
    bs = [tuple(x) for x in ob['sids'][0].tolist()]; bsc = ob['scores'][0].tolist()
    es = [tuple(x) for x in oe['sids'][0].tolist()]; esc = oe['scores'][0].tolist()
    order = sorted(full, key=lambda s: -full[s])
    d_beam = max(abs(full[s] - v) for s, v in zip(bs, bsc))
    d_ex = max(abs(full[s] - v) for s, v in zip(es, esc))
    assert (u2 >= sc - 1e-5).all(), 'bound violated'
    print(f'user {u}: beam-vs-exhaustive max|d|={d_beam:.2e} exact-vs-exhaustive max|d|={d_ex:.2e} '
          f'certified={oe["certified"][0].tolist()} exact top10==global top10: {es[:10]==order[:10]} '
          f'beam top10==global: {bs[:10]==order[:10]} beam top50 recall of global top50: {len(set(bs)&set(order[:50]))}/50 '
          f'global top1 rank in beam: {bs.index(order[0]) if order[0] in bs else None}')
