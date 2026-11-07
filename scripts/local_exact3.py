import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rapidtinyrec.speedrun.model import Qwen2Weights
from rapidtinyrec.speedrun.beam import SidCatalog, BeamEngine
from rapidtinyrec.speedrun.exact import ExactEngine, ExactEngine3
from rapidtinyrec.speedrun import data
rel = sys.argv[1]; dev = torch.device('mps')
tok = data.load_tokenizer(rel); rows = data.test_rows(rel)
sel = data.cohort500(rows)[:3]; ids = data.prompts(rel, tok, [rows[i] for i in sel])
cat = SidCatalog(tok, Path(rel) / data.CATALOG, dev)
W = Qwen2Weights(Path(rel) / 'Office_ckpt', torch.float32, dev)
with torch.inference_mode():
  for u in range(3):
    P = (len(ids[u]) + 15) // 16 * 16
    t = torch.full((1, P), tok.eos_token_id, dtype=torch.long); t[0, :len(ids[u])] = torch.tensor(ids[u]); t = t.to(dev)
    ln = torch.tensor([len(ids[u])], device=dev)
    for norm, suf in (('full', 'fused'), ('trie', 'none')):
        b = BeamEngine(W, cat, norm=norm, suffix_mode=suf); ob = b.run(t, ln, b.alloc(1, P))
        e = ExactEngine(W, cat, n_expand=128, norm=norm, suffix_mode=suf); oe = e.run(t, ln, e.alloc(1, P))
        e3 = ExactEngine3(W, cat, n_prefix=128, n_expand=32, norm=norm, suffix_mode=suf); o3 = e3.run(t, ln, e3.alloc(1, P))
        L = lambda o: [tuple(x) for x in o['sids'][0].tolist()]
        bs, es, s3 = L(ob), L(oe), L(o3)
        print(f'u{u} {norm}: e3 cert {o3["certified"][0].tolist()} e1(N=128) cert {oe["certified"][0].tolist()} '
              f'e3 top10==e1 top10 {s3[:10]==es[:10]} e3 top10==beam {s3[:10]==bs[:10]} '
              f'top scores e3 {[round(x,3) for x in o3["scores"][0,:3].tolist()]} beam {[round(x,3) for x in ob["scores"][0,:3].tolist()]}')
