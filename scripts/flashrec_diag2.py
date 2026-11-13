"""Is FlashRec's model forward right on MiniOneRec (Qwen2 patch)? Check the patched config/weights, then compare its
step-by-step beam scores (max_tokens = 1, 2, 3) with the HF reference (FP32, per-level codebook log-softmax).

usage: python scripts/flashrec_diag2.py --model CKPT --workload DIR [--users 6]
"""
import argparse, json, logging
from pathlib import Path
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True); ap.add_argument('--workload', required=True); ap.add_argument('--users', type=int, default=6)
a = ap.parse_args()
logging.basicConfig(level=logging.WARNING)
W = json.loads((Path(a.workload) / 'workload.json').read_text())
levels = [torch.tensor(l, device='cuda') for l in W['sid_levels']]
rows = W['cohort500'][:a.users]
print(json.dumps({'hf_config': {k: v for k, v in json.loads((Path(a.model) / 'config.json').read_text()).items()
                                if k in ('model_type', 'architectures', 'tie_word_embeddings', 'rope_theta', 'num_key_value_heads',
                                         'num_attention_heads', 'hidden_size', 'use_sliding_window', 'rope_scaling')}}))
from transformers import AutoModelForCausalLM
hf = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).cuda().eval()


@torch.no_grad()
def hf_steps(prompt, toks):
    lg = hf(torch.tensor([prompt + list(toks[:-1])], device='cuda')).logits[0, -len(toks):].float()
    return [(lg[i, toks[i]] - torch.logsumexp(lg[i, levels[i]], 0)).item() for i in range(len(toks))]


@torch.no_grad()
def hf_top(prompt, k=5):
    lg = hf(torch.tensor([prompt], device='cuda')).logits[0, -1].float()
    lp = lg[levels[0]] - torch.logsumexp(lg[levels[0]], 0)
    v, i = lp.topk(k)
    return [(int(levels[0][j]), round(float(x), 4)) for x, j in zip(v, i)]


import flashrec.engine.engine as fe
from flashrec import BeamRecConfig, BeamRecEngine
fe.resolve_sid_layout = lambda config: None
cfg = BeamRecConfig(model_path=a.model, beam_width=50, max_tokens=3, quantization=None, kv_cache_dtype='float16',
                    compute_dtype='float16', sid_token_range=','.join(str(t) for l in W['sid_levels'] for t in l),
                    sid_codebook_sizes=','.join(map(str, W['codebook_sizes'])),
                    sid_vocab_file=str(Path(a.workload) / 'catalog_tokens.json'), mem_fraction_static=0.35,
                    enable_cuda_graph=False, enable_radix=False, enable_warmup=False)
eng = BeamRecEngine(cfg)
mc = eng.runner.model_config
at = eng.runner.model.layers[0].self_attn
print(json.dumps({'flashrec_cfg': {k: getattr(mc, k) for k in ('attention_bias', 'qk_norm', 'tie_word_embeddings', 'rope_theta',
                                                                 'num_attention_heads', 'num_key_value_heads', 'head_dim')},
                  'qkv_bias_loaded': None if at.qkv_proj.bias is None else float(at.qkv_proj.bias.float().abs().sum()),
                  'q_norm': at.q_norm is not None,
                  'lm_head_is_embed': eng.runner.model.lm_head_weight is None}))
for r in rows:
    P = W['prompt_ids'][r]
    out = {'row': r, 'hf_top_first': hf_top(P)}
    for T in (1, 2, 3):
        g = eng.generate(input_ids=P, n=5 if T == 1 else 50, max_tokens=T)
        s = g.sequences[0]
        toks = [int(x) for x in s.tokens[-T:]]
        st = hf_steps(P, toks)
        out[f'T{T}'] = {'toks': toks, 'reported': round(float(s.beam_score), 4), 'cum': round(float(s.cum_logprob), 4),
                        'hf_steps': [round(x, 4) for x in st], 'hf_sum': round(sum(st), 4)}
        if T == 1:
            out['T1_top5'] = [([int(x) for x in q.tokens[-1:]], round(float(q.beam_score), 4)) for q in g.sequences[:5]]
    print(json.dumps(out), flush=True)
