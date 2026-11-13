"""Isolate where FlashRec's beam scores diverge from the model: re-score its returned SIDs with the HF reference
model (FP32, per-level codebook log-softmax, which is FlashRec's documented objective) under several engine
toggles. Runs in FlashRec's venv.

usage: python scripts/flashrec_diag.py --model CKPT --workload DIR --out OUT.json [--users 8]
"""
import argparse, gc, json, logging, statistics
from pathlib import Path
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True); ap.add_argument('--workload', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--users', type=int, default=8)
a = ap.parse_args()
logging.basicConfig(level=logging.WARNING)
W = json.loads((Path(a.workload) / 'workload.json').read_text())
levels = [torch.tensor(l, device='cuda') for l in W['sid_levels']]
rows = W['cohort500'][:a.users]

from transformers import AutoModelForCausalLM
hf = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).cuda().eval()


@torch.no_grad()
def true_steps(prompt, sid):
    lg = hf(torch.tensor([prompt + list(sid[:2])], device='cuda')).logits[0, -3:].float()
    return [(lg[i, sid[i]] - torch.logsumexp(lg[i, levels[i]], 0)).item() for i in range(3)]


import flashrec.engine.engine as fe
from flashrec import BeamRecConfig, BeamRecEngine
fe.resolve_sid_layout = lambda config: None
ids_flat = [t for l in W['sid_levels'] for t in l]
CFGS = {
    'default': {},
    'no_radix': {'enable_radix': False},
    'no_graph': {'enable_cuda_graph': False},
    'no_fused_expand': {'enable_fused_expand': False, 'enable_graph_expand': False},
    'no_per_codebook': {'enable_per_codebook_lm_head': False},
    'all_off': {'enable_radix': False, 'enable_cuda_graph': False, 'enable_fused_expand': False,
                'enable_graph_expand': False, 'enable_prefill_batch': False, 'enable_decode_pack': False},
}
res = {}
for name, kw in CFGS.items():
    cfg = BeamRecConfig(model_path=a.model, beam_width=50, max_tokens=3, quantization=None, kv_cache_dtype='float16',
                        compute_dtype='float16', sid_token_range=','.join(map(str, ids_flat)),
                        sid_codebook_sizes=','.join(map(str, W['codebook_sizes'])),
                        sid_vocab_file=str(Path(a.workload) / 'catalog_tokens.json'), mem_fraction_static=0.35,
                        cuda_graph_max_bs=800, enable_warmup=False, **kw)
    try:
        eng = BeamRecEngine(cfg)
        if cfg.enable_cuda_graph:
            eng.ensure_cuda_graph()
        errs, top1, out = [], 0, []
        for r in rows:
            P = W['prompt_ids'][r]
            g = eng.generate(input_ids=P, n=50, max_tokens=3)
            seqs = [(tuple(int(x) for x in s.tokens[-3:]), float(s.beam_score)) for s in g.sequences[:5]]
            tr = [(sid, rep, true_steps(P, sid)) for sid, rep in seqs]
            errs += [abs(rep - sum(st)) for _, rep, st in tr]
            out.append([{'sid': sid, 'reported': rep, 'true_steps': st} for sid, rep, st in tr])
        res[name] = {'mean_abs_err': statistics.mean(errs), 'max_abs_err': max(errs), 'users': out}
        print(json.dumps({'cfg': name, 'mean_abs_err': round(res[name]['mean_abs_err'], 4),
                          'max_abs_err': round(res[name]['max_abs_err'], 4)}), flush=True)
        del eng
    except Exception as exc:
        res[name] = {'error': repr(exc)}
        print(json.dumps({'cfg': name, 'error': repr(exc)[:500]}), flush=True)
    gc.collect(); torch.cuda.empty_cache()
json.dump(res, open(a.out, 'w'), indent=0)
