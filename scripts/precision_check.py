"""How far does plain half-precision inference of the checkpoint drift from FP32 on the first SID code?
HF model in fp32 / fp16 / bf16; per-level codebook log-softmax of the first code for a few cohort users.

usage: python scripts/precision_check.py WORKLOAD_DIR CKPT [N_USERS]
"""
import json, statistics, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM

wd, ck = Path(sys.argv[1]), sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 32
W = json.loads((wd / 'workload.json').read_text())
lv0 = torch.tensor(W['sid_levels'][0], device='cuda')
rows = W['cohort500'][:n]
out = {}
for dt in (torch.float32, torch.float16, torch.bfloat16):
    m = AutoModelForCausalLM.from_pretrained(ck, torch_dtype=dt).cuda().eval()
    lps = []
    with torch.no_grad():
        for r in rows:
            lg = m(torch.tensor([W['prompt_ids'][r]], device='cuda')).logits[0, -1].float()
            lps.append((lg[lv0] - torch.logsumexp(lg[lv0], 0)).cpu())
    out[str(dt)] = torch.stack(lps)
    del m; torch.cuda.empty_cache()
ref = out['torch.float32']
for k, v in out.items():
    d = (v - ref).abs()
    top = ref.argmax(1)
    print(json.dumps({'cfg': k, 'mean_abs_lp_err_all88': round(float(d.mean()), 4),
                      'mean_abs_lp_err_top_code': round(float(d.gather(1, top[:, None]).mean()), 4),
                      'max_abs_lp_err_top_code': round(float(d.gather(1, top[:, None]).max()), 4),
                      'top1_agree': float((v.argmax(1) == top).float().mean())}), flush=True)
