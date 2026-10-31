"""Untimed full-score byte hashes and bounded immutable tensor inventory."""
import hashlib,re
import torch
from .beam_wave_outputs import unpack_requests as original_unpack

def tensor_digest(t):
 flat=t.detach().reshape(-1);h=hashlib.sha256()
 for i in range(0,flat.numel(),1048576):
  chunk=flat[i:i+1048576].contiguous().view(torch.uint8).cpu().numpy()
  h.update(memoryview(chunk))
 return dict(shape=list(t.shape),dtype=str(t.dtype),sha256=h.hexdigest())

def unpack_requests(model,output,users,prompt_tokens,graph_logit_storage_ptrs=()):
 if len(users)!=1:raise ValueError('Single request only')
 rows=original_unpack(model,output,users,prompt_tokens,graph_logit_storage_ptrs)
 rows[0]['full_score_hashes']=[tensor_digest(s) for s in output.scores]
 return rows

def validate_retained_ownership(retained):
 seen=set()
 for group,out in retained:
  ptrs=[s.untyped_storage().data_ptr() for s in out.scores]
  if len(ptrs)!=5 or len(set(ptrs))!=5 or set(ptrs)&seen:raise ValueError('Retained score owners overlap')
  seen.update(ptrs)

def immutable_inventory(engine,model):
 tensors={'model.parameter.'+n:t for n,t in model.named_parameters()}
 tensors.update({'model.buffer.'+n:t for n,t in model.named_buffers()})
 for group,pair in engine.inputs.items():
  for i,t in enumerate(pair):tensors[f'input.{group}.{i}']=t
 for n,t in vars(engine.tables).items():
  if isinstance(t,torch.Tensor):tensors['table.'+n]=t
 return {n:tensor_digest(t) for n,t in sorted(tensors.items())}

def validate_immutable_inventory(inv,users):
 if not inv or not any(k.startswith('model.parameter.') for k in inv) or not any(k.startswith('table.') for k in inv):raise ValueError('Missing immutable tensor families')
 for u in users:
  for i in range(2):
   if f"input.{(u['cohort_index'],)}.{i}" not in inv:raise ValueError('Missing prepared input')
 for r in inv.values():
  if not re.fullmatch('[0-9a-f]{64}',r['sha256']) or not isinstance(r['shape'],list):raise ValueError('Malformed immutable hash')
