import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rapidtinyrec.speedrun.model import Qwen2Weights
from rapidtinyrec.speedrun.beam import SidCatalog, BeamEngine
from rapidtinyrec.speedrun.exact import ExactEngine3
from rapidtinyrec.speedrun import data
rel = sys.argv[1]; dev = torch.device('mps')
tok = data.load_tokenizer(rel); rows = data.test_rows(rel)
sel = data.cohort500(rows)[:3]; ids = data.prompts(rel, tok, [rows[i] for i in sel])
cat = SidCatalog(tok, Path(rel) / data.CATALOG, dev)
W = Qwen2Weights(Path(rel) / data.CKPT, torch.float32, dev)
with torch.inference_mode():
    P = 128
    t = torch.full((3, P), tok.eos_token_id, dtype=torch.long)
    for u in range(3): t[u, :len(ids[u])] = torch.tensor(ids[u])
    t = t.to(dev); ln = torch.tensor([len(x) for x in ids], device=dev)
    for cls, kw in ((BeamEngine, {}), (ExactEngine3, dict(n_prefix=128, n_expand=32))):
        e0 = cls(W, cat, **kw); e1 = cls(W, cat, fuse_first=True, **kw)
        o0 = e0.run(t, ln, e0.alloc(3, P)); o1 = e1.run(t, ln, e1.alloc(3, P))
        for u in range(3):
            a = [tuple(x) for x in o0['sids'][u].tolist()]; b = [tuple(x) for x in o1['sids'][u].tolist()]
            print(cls.__name__, 'user', u, 'ranks equal', sum(x == y for x, y in zip(a, b)), '/', len(a),
                  'max|dscore|', (o0['scores'][u] - o1['scores'][u]).abs().max().item())
