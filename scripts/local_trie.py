"""Local check: ExactTrieEngine (skips single-child prefixes) vs ExactEngine3 trie with a large M."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rapidtinyrec.speedrun.model import Qwen2Weights
from rapidtinyrec.speedrun.beam import SidCatalog
from rapidtinyrec.speedrun.exact import ExactEngine3, ExactTrieEngine
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
    ref = ExactEngine3(W, cat, n_prefix=256, n_expand=200, norm='trie', suffix_mode='none', fuse_first=True, ks=(1, 10, 50, 100))
    o0 = ref.run(t, ln, ref.alloc(3, P))
    for M, N, ks in ((16, 50, (1, 5, 10, 20, 50)), (64, 100, (10, 50, 100))):
        e = ExactTrieEngine(W, cat, n_prefix=M, n_out=N, ks=ks)
        o1 = e.run(t, ln, e.alloc(3, P))
        for u in range(3):
            a = [tuple(x) for x in o0['sids'][u].tolist()][:N]; b = [tuple(x) for x in o1['sids'][u].tolist()]
            print(f'M={M} N={N} user {u}: ranks equal {sum(x == y for x, y in zip(a, b))}/{N}',
                  'max|dscore|', (o0['scores'][u][:N] - o1['scores'][u]).abs().max().item(),
                  'cert', o1['certified'][u].tolist(), 'ref cert', o0['certified'][u].tolist())
