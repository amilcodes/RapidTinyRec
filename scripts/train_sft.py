"""Model branch: short fine-tune of the released MiniOneRec checkpoint with a loss aligned to the decoder.

The decoder ranks items by per-level log-softmax over *restricted* sets (codebook for a, b; valid trie
children for c, "cbtrie"), while MiniOneRec was trained with full-vocabulary next-token cross-entropy.
Losses (on the three SID positions only; no '\\n'/EOS terms, which the decoder does not score):
  full    full-vocabulary CE                          (control: same data, same steps)
  cbtrie  CE of log-softmax over the codebook (a, b) and over the valid children of (a, b) (c)
  trie    CE over valid trie children at every level
Optional KL(reference || model) on the same restricted distributions keeps the model near the release.

Writes a release-like directory OUT/ckpt_{test,valid}/ (Amazon/ symlinks + Office_ckpt with new weights) so
bench.py can evaluate it unchanged.

usage: train_sft.py --release REL --train-release TRAIN --valid-release VALID --out DIR --loss cbtrie
"""
import argparse, json, math, os, random, shutil, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rapidtinyrec.speedrun import data
from rapidtinyrec.speedrun.beam import SidCatalog


def level_logp(hs, head, cat, sids, kind):
    """hs [B,3,D] hidden at the positions predicting a, b, c; sids [B,3] tokens.
    Returns per-level log-prob of the target [B,3] and per-level log-distributions (masked) for KL."""
    lg = F.linear(hs, head).float()                         # [B,3,S] over all SID tokens
    B = hs.shape[0]
    loc = cat.tok2loc[sids]                                  # [B,3] column of each target
    masks = []
    a_idx = torch.searchsorted(cat.first, sids[:, 0])
    t1 = cat.t1[a_idx]; v1 = cat.v1[a_idx]                    # children of a
    col = (t1 == sids[:, 1:2]).float().argmax(1)
    pair = cat.pair_table[a_idx, col]
    t2 = cat.t2[pair]; v2 = cat.v2[pair]
    S = lg.shape[-1]
    for lvl in range(3):
        m = cat.level_mask[lvl][None].expand(B, S).clone()
        if kind == 'trie' or (kind == 'cbtrie' and lvl == 2):
            m = torch.zeros(B, S, dtype=torch.bool, device=hs.device)
            if lvl == 0:
                m[:, cat.tok2loc[cat.first]] = True
            elif lvl == 1:
                m.scatter_(1, cat.tok2loc[t1].masked_fill(~v1, 0), v1)
                m[:, cat.tok2loc[t1[:, 0]]] |= True
            else:
                m.scatter_(1, cat.tok2loc[t2].masked_fill(~v2, 0), v2)
                m[:, cat.tok2loc[t2[:, 0]]] |= True
        masks.append(m)
    M = torch.stack(masks, 1)                                # [B,3,S]
    lp = lg.masked_fill(~M, float('-inf')).log_softmax(-1)
    tgt = lp.gather(-1, loc[..., None])[..., 0]
    return tgt, lp, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', required=True); ap.add_argument('--train-release', required=True)
    ap.add_argument('--valid-release', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--loss', default='cbtrie', choices=['full', 'cbtrie', 'trie'])
    ap.add_argument('--kl', type=float, default=0.0); ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--epochs', type=float, default=1.0); ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--level-w', default='1,1,1'); ap.add_argument('--warmup', type=int, default=50)
    ap.add_argument('--max-steps', type=int, default=0); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); random.seed(a.seed)
    dev = torch.device('cuda')
    from transformers import AutoModelForCausalLM
    tok = data.load_tokenizer(a.release)
    cat = SidCatalog(tok, Path(a.release) / data.CATALOG, dev)
    rows = data.test_rows(a.train_release)
    ids = data.prompts(a.train_release, tok, rows)
    tg = [data.sid_tokens(tok, r['item_sid']) for r in rows]
    keep = [i for i in range(len(rows)) if len(tg[i]) == 3]
    print(json.dumps({'stage': 'train', 'rows': len(rows), 'usable': len(keep)}), flush=True)
    ck = Path(a.release) / data.CKPT
    model = AutoModelForCausalLM.from_pretrained(ck, torch_dtype=torch.float32, local_files_only=True).to(dev)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    ref = None
    if a.kl > 0:
        ref = AutoModelForCausalLM.from_pretrained(ck, torch_dtype=torch.bfloat16, local_files_only=True).to(dev).eval()
        ref.requires_grad_(False)
    sid_ids = cat.sid_ids
    lw = torch.tensor([float(x) for x in a.level_w.split(',')], device=dev)
    steps_per_epoch = len(keep) // a.batch
    total = int(steps_per_epoch * a.epochs) if not a.max_steps else a.max_steps
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    eos = tok.eos_token_id
    order = []
    rng = random.Random(a.seed)          # own RNG: data.prompts() (MiniOneRec's dataset) reseeds `random` with 42
    step = 0
    t0 = time.time()
    log = []
    model.train()
    while step < total:
        if not order:
            order = keep[:]; rng.shuffle(order)
        chunk = [order.pop() for _ in range(min(a.batch, len(order)))]
        seqs = [list(ids[i]) + list(tg[i][:2]) for i in chunk]
        Lm = max(len(s) for s in seqs)
        x = torch.full((len(chunk), Lm), eos, dtype=torch.long)
        att = torch.zeros(len(chunk), Lm, dtype=torch.long)
        for j, s in enumerate(seqs):
            x[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1
        x, att = x.to(dev), att.to(dev)
        plen = torch.tensor([len(ids[i]) for i in chunk], device=dev)
        pos = (plen - 1)[:, None] + torch.arange(3, device=dev)[None]          # positions predicting a, b, c
        sids = torch.tensor([tg[i] for i in chunk], device=dev)
        bi = torch.arange(len(chunk), device=dev)[:, None]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            hs = model.model(input_ids=x, attention_mask=att).last_hidden_state[bi, pos]       # [B,3,D]
            head = model.get_output_embeddings().weight
            if a.loss == 'full':
                lg = F.linear(hs, head).float()
                tgt = lg.log_softmax(-1).gather(-1, sids[..., None])[..., 0]
                lp_m, M = None, None
            else:
                tgt, lp_m, M = level_logp(hs, head[sid_ids], cat, sids, a.loss)
            loss = -(tgt * lw).sum(1).mean()
            kl = torch.zeros((), device=dev)
            if ref is not None:
                with torch.no_grad():
                    hr = ref.model(input_ids=x, attention_mask=att).last_hidden_state[bi, pos]
                    _, lp_r, _ = level_logp(hr, ref.get_output_embeddings().weight[sid_ids], cat, sids,
                                            a.loss if a.loss != 'full' else 'cbtrie')
                if lp_m is None:
                    _, lp_m, M = level_logp(hs, head[sid_ids], cat, sids, 'cbtrie')
                pr = lp_r.exp()
                kl = (pr * (lp_r - lp_m)).masked_fill(~M, 0.0).sum(-1).sum(1).mean()
                loss = loss + a.kl * kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 50 == 0 or step == total:
            rec = {'stage': 'train', 'step': step, 'total': total, 'loss': loss.item(), 'kl': kl.item(),
                   'lr': sched.get_last_lr()[0], 's': time.time() - t0}
            log.append(rec)
            print(json.dumps(rec), flush=True)
    # ---- save as release-like dirs for bench.py
    model.eval()
    for split, rel in (('test', a.release), ('valid', a.valid_release)):
        d = out / f'ckpt_{split}'
        d.mkdir(parents=True, exist_ok=True)
        if not (d / 'Amazon').exists():
            os.symlink(Path(rel).resolve() / 'Amazon', d / 'Amazon')
    cdir = out / 'ckpt_test' / data.CKPT
    model.to(torch.bfloat16).save_pretrained(cdir, safe_serialization=True)
    for f in ck.iterdir():
        if f.suffix in ('.json', '.txt') and not (cdir / f.name).exists() or f.name in ('merges.txt', 'vocab.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json'):
            if f.name not in ('config.json', 'generation_config.json', 'model.safetensors.index.json'):
                shutil.copy(f, cdir / f.name)
    vlink = out / 'ckpt_valid' / data.CKPT
    if not vlink.exists():
        os.symlink(cdir.resolve(), vlink)
    (out / 'train_log.json').write_text(json.dumps({'args': vars(a), 'log': log}, indent=1))
    print(json.dumps({'stage': 'done', 'ckpt': str(cdir)}), flush=True)


if __name__ == '__main__':
    main()
