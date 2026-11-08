"""SASRec on the MiniOneRec train split; writes per-user log q(SID | history) for valid and test rows.

Used as a collaborative value term in certified search: score = LLM score + beta * log q(i | history).
The term is per user and per item, so the certified search bound becomes U2 + beta * max over the prefix's
items (exact). Items sharing a SID are merged by logsumexp.

usage: sasrec.py --release REL --train-csv CSV --valid-csv CSV --out DIR
"""
import argparse, ast, csv, json, math, random, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rapidtinyrec.speedrun import data
from rapidtinyrec.speedrun.beam import SidCatalog


class SASRec(nn.Module):
    def __init__(self, n, d=64, L=20, layers=2, heads=2, drop=0.2):
        super().__init__()
        self.item = nn.Embedding(n + 1, d, padding_idx=0)
        self.pos = nn.Embedding(L, d)
        self.L = L
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)

    def forward(self, seq):                     # seq [B,L] item ids (0 = pad, left-padded)
        B, L = seq.shape
        x = self.item(seq) * math.sqrt(self.item.embedding_dim) + self.pos(torch.arange(L, device=seq.device))[None]
        keep = (seq != 0)[..., None].to(x.dtype)
        x = self.drop(x) * keep                    # timeline mask (no key-padding mask: all-masked rows give NaN)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=seq.device), 1)
        h = self.enc(x, mask=causal)
        h = self.norm(h[:, -1])
        return h @ self.item.weight[1:].T        # [B, n] logits over items 1..n


def load(path, L):
    rows = list(csv.DictReader(open(path)))
    seqs, tgt = [], []
    for r in rows:
        h = [int(x) + 1 for x in ast.literal_eval(r['history_item_id'])][-L:]
        seqs.append([0] * (L - len(h)) + h)
        tgt.append(int(r['item_id']) + 1)
    return torch.tensor(seqs), torch.tensor(tgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', required=True); ap.add_argument('--train-csv', required=True)
    ap.add_argument('--valid-csv', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=200); ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--L', type=int, default=20); ap.add_argument('--patience', type=int, default=10)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); random.seed(0)
    dev = torch.device('cuda')
    info = [l.split('\t') for l in open(Path(a.release) / data.CATALOG).read().splitlines() if l.strip()]
    n = max(int(x[2]) for x in info) + 1
    tok = data.load_tokenizer(a.release)
    cat = SidCatalog(tok, Path(a.release) / data.CATALOG, dev)
    uniq = sorted(set(cat.item_sid_tokens))
    sid_index = {s: i for i, s in enumerate(uniq)}
    item2sid = torch.zeros(n, dtype=torch.long)
    for sid, _, iid in info:
        item2sid[int(iid)] = sid_index[data.sid_tokens(tok, sid)]
    Xtr, Ytr = load(a.train_csv, a.L)
    Xva, Yva = load(a.valid_csv, a.L)
    Xte, Yte = load(Path(a.release) / data.TEST_CSV, a.L)
    m = SASRec(n, a.d, a.L).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)

    def evaluate(X, Y):
        m.eval()
        hits, nd = 0.0, 0.0
        with torch.no_grad():
            for s in range(0, len(X), 1024):
                lg = m(X[s:s + 1024].to(dev))
                y = Y[s:s + 1024].to(dev) - 1
                rank = (lg > lg.gather(1, y[:, None])).sum(1)
                hits += (rank < 10).sum().item()
                nd += torch.where(rank < 10, 1 / torch.log2(rank.float() + 2), torch.zeros_like(rank, dtype=torch.float)).sum().item()
        m.train()
        return hits / len(X), nd / len(X)
    best, best_ep, bad = -1, 0, 0
    t0 = time.time()
    for ep in range(a.epochs):
        perm = torch.randperm(len(Xtr))
        for s in range(0, len(perm), 256):
            b = perm[s:s + 256]
            loss = F.cross_entropy(m(Xtr[b].to(dev)), Ytr[b].to(dev) - 1)
            opt.zero_grad(); loss.backward(); opt.step()
        hr, nd = evaluate(Xva, Yva)
        assert hr < 0.99, 'evaluation degenerate (NaN logits?)'
        if nd > best:
            best, best_ep, bad = nd, ep, 0
            torch.save(m.state_dict(), out / 'sasrec.pt')
        else:
            bad += 1
            if bad >= a.patience:
                break
        if ep % 5 == 0:
            print(json.dumps({'stage': 'train', 'epoch': ep, 'loss': loss.item(), 'valid_HR@10': hr, 'valid_NDCG@10': nd}), flush=True)
    m.load_state_dict(torch.load(out / 'sasrec.pt'))
    res = {'best_epoch': best_ep, 'valid': evaluate(Xva, Yva), 'test': evaluate(Xte, Yte), 'train_s': time.time() - t0}
    print(json.dumps({'stage': 'eval', 'sasrec': res}), flush=True)
    # per-user log q over SIDs (logsumexp over items sharing a SID), sid_tok order
    m.eval()
    S = len(uniq)
    for split, X in (('valid', Xva), ('test', Xte)):
        outs = []
        with torch.no_grad():
            for s in range(0, len(X), 1024):
                full = m(X[s:s + 1024].to(dev)).float().log_softmax(-1)        # [b, n]: column j = item id j
                lq = full
                mx = torch.full((lq.shape[0], S), float('-inf'), device=dev)
                idx = item2sid.to(dev)[None].expand(lq.shape[0], -1)
                mx.scatter_reduce_(1, idx, full, 'amax')
                se = torch.zeros(lq.shape[0], S, device=dev).scatter_add_(1, idx, (full - mx.gather(1, idx)).exp())
                outs.append((mx + se.log()).half().cpu())
        torch.save({'logq': torch.cat(outs), 'sids': uniq}, out / f'sasrec_{split}.pt')
    (out / 'sasrec.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({'stage': 'done'}), flush=True)


if __name__ == '__main__':
    main()
