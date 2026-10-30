"""Complete tensor-container walk for a FUTURE harness; submitted run untouched."""
import torch

def tensor_items(value,prefix,ancestors=()):
 if isinstance(value,torch.Tensor):yield prefix,value;return
 if isinstance(value,(dict,list,tuple)):
  if id(value) in ancestors:raise ValueError('Cyclic tensor container')
  items=sorted(value.items(),key=lambda kv:repr(kv[0])) if isinstance(value,dict) else enumerate(value)
  for key,item in items:yield from tensor_items(item,prefix+'['+repr(key)+']',ancestors+(id(value),))

def immutable_tensors(engine,model):
 tensors={'model.parameter.'+n:t for n,t in model.named_parameters()}
 tensors.update({'model.buffer.'+n:t for n,t in model.named_buffers()})
 for name,value in [('input',engine.inputs),('table',vars(engine.tables))]:
  for key,tensor in tensor_items(value,name):
   if key in tensors:raise ValueError('Duplicate immutable tensor path')
   tensors[key]=tensor
 return tensors
