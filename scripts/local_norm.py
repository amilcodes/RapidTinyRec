import sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rapidtinyrec.speedrun.model import Qwen2Weights
from rapidtinyrec.speedrun.beam import SidCatalog, BeamEngine
from rapidtinyrec.speedrun.exact import ExactEngine
from rapidtinyrec.speedrun import data
rel = sys.argv[1]; dev = torch.device('mps')
tok = data.load_tokenizer(rel); rows = data.test_rows(rel)
sel = data.cohort500(rows)[:2]; ids = data.prompts(rel, tok, [rows[i] for i in sel])
cat = SidCatalog(tok, Path(rel) / data.CATALOG, dev); print(cat.stats)
W = Qwen2Weights(Path(rel) / 'Office_ckpt', torch.float32, dev)
with torch.inference_mode():
  for u in range(2):
    P = (len(ids[u]) + 15) // 16 * 16
    t = torch.full((1, P), tok.eos_token_id, dtype=torch.long); t[0, :len(ids[u])] = torch.tensor(ids[u]); t = t.to(dev)
    ln = torch.tensor([len(ids[u])], device=dev)
    for norm in ('full', 'codebook', 'trie'):
        b = BeamEngine(W, cat, norm=norm, suffix_mode='none'); ob = b.run(t, ln, b.alloc(1, P))
        e = ExactEngine(W, cat, n_expand=64, norm=norm, suffix_mode='none'); oe = e.run(t, ln, e.alloc(1, P))
        bs = [tuple(x) for x in ob['sids'][0].tolist()]; es = [tuple(x) for x in oe['sids'][0].tolist()]
        bsc = dict(zip(bs, ob['scores'][0].tolist())); esc = dict(zip(es, oe['scores'][0].tolist()))
        common = set(bs) & set(es)
        print(f'u{u} {norm:8s} beam top3 {[round(x,3) for x in ob["scores"][0,:3].tolist()]} exact top3 {[round(x,3) for x in oe["scores"][0,:3].tolist()]} '
              f'cert {oe["certified"][0].tolist()} top10 same {bs[:10]==es[:10]} max|d| common {max(abs(bsc[s]-esc[s]) for s in common):.1e} '
              f'exact better-than-beam-10th: {sum(v > ob["scores"][0,9].item()+1e-6 for v in oe["scores"][0,:10].tolist())}')
