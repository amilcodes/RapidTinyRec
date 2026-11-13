"""Write a mathematically identical copy of a Qwen2 checkpoint with KV heads repeated (GQA group -> 1 when
KV = num_attention_heads). Used to test whether an engine's attention path mishandles a GQA group size.

usage: python scripts/expand_kv_heads.py SRC_CKPT DST_DIR NEW_KV_HEADS
"""
import json, shutil, sys
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

src, dst, nkv = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
dst.mkdir(parents=True, exist_ok=True)
cfg = json.loads((src / 'config.json').read_text())
kv, H = cfg['num_key_value_heads'], cfg['num_attention_heads']
hd = cfg.get('head_dim') or cfg['hidden_size'] // H
assert nkv % kv == 0 and H % nkv == 0, (kv, nkv, H)
rep = nkv // kv
idx = src / 'model.safetensors.index.json'
files = sorted(set(json.loads(idx.read_text())['weight_map'].values())) if idx.exists() else ['model.safetensors']
out = {}
for f in files:
    with safe_open(str(src / f), 'pt') as h:
        for k in h.keys():
            t = h.get_tensor(k)
            if k.endswith(('self_attn.k_proj.weight', 'self_attn.v_proj.weight')):
                t = t.view(kv, hd, -1).repeat_interleave(rep, 0).reshape(nkv * hd, -1)
            elif k.endswith(('self_attn.k_proj.bias', 'self_attn.v_proj.bias')):
                t = t.view(kv, hd).repeat_interleave(rep, 0).reshape(-1)
            out[k] = t.contiguous()
if cfg.get('tie_word_embeddings') and 'lm_head.weight' in out:
    del out['lm_head.weight']                       # tied: HF re-ties to embed_tokens
save_file(out, str(dst / 'model.safetensors'), metadata={'format': 'pt'})
cfg['num_key_value_heads'] = nkv
(dst / 'config.json').write_text(json.dumps(cfg, indent=1))
for p in src.iterdir():
    if p.suffix in ('.json', '.txt', '.model', '.jinja') and p.name not in ('config.json', 'model.safetensors.index.json'):
        shutil.copy(p, dst / p.name)
print(json.dumps({'kv_heads': [kv, nkv], 'group': [H // kv, H // nkv], 'tensors': len(out), 'dst': str(dst)}))
