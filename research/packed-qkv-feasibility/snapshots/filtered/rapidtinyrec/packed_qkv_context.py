"""Separate BF16-only source-pinned prototype. No submitted harness imports this.
Only the three Q/K/V projection expressions change. Rotary, cache, attention
and output projection remain the installed pinned original implementation.
"""
import ast,contextlib,hashlib,inspect,textwrap,types
from pathlib import Path
import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from .packed_qkv import PackedQKV
PIN=Path(__file__).with_name('qkv_forward_pin.txt').read_text().strip()

def packed_forward():
 original=Qwen2Attention.forward;source=inspect.getsource(original)
 if hashlib.sha256(source.encode()).hexdigest()!=PIN:raise ValueError('Pinned attention source differs')
 tree=ast.parse(textwrap.dedent(source));fn=tree.body[0];hits=0;body=[]
 expected={'query_states':'q_proj','key_states':'k_proj','value_states':'v_proj'}
 for node in fn.body:
  if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in expected:
   name=node.targets[0].id;projection=expected[name]
   want=f'self.{projection}(hidden_states).view(hidden_shape).transpose(1, 2)'
   if ast.unparse(node.value)!=want:raise ValueError('Unexpected projection expression')
   if hits==0:body.extend(ast.parse('query_states, key_states, value_states = self._research_packed_qkv(hidden_states)').body)
   node.value=ast.parse(f'{name}.view(hidden_shape).transpose(1, 2)',mode='eval').body;hits+=1
  body.append(node)
 if hits!=3:raise ValueError('Exactly three original projections required')
 fn.body=body;ast.fix_missing_locations(tree);namespace=dict(original.__globals__)
 exec(compile(tree,'<pinned-packed-qkv-prototype>','exec'),namespace)
 return namespace[fn.name]

@contextlib.contextmanager
def packed_qkv_bf16(model):
 if model.training:raise ValueError('Eval-only prototype')
 modules=[m for m in model.modules() if isinstance(m,Qwen2Attention)]
 if not modules:raise ValueError('No Qwen2 attention modules')
 if any(m.training or m.q_proj.weight.dtype!=torch.bfloat16 or hasattr(m,'_research_packed_qkv') or 'forward' in m.__dict__ for m in modules):raise ValueError('Fresh BF16 attention modules required')
 fn=packed_forward();installed=[]
 try:
  for m in modules:
   m._research_packed_qkv=PackedQKV(m.q_proj,m.k_proj,m.v_proj)
   installed.append(m);m.forward=types.MethodType(fn,m)
  yield dict(layers=len(modules),extra_packed_bytes=sum(t.numel()*t.element_size() for m in modules for t in m._research_packed_qkv.buffers()),cuda_validated=False)
 finally:
  for m in reversed(installed):
   if 'forward' in m.__dict__:del m.forward
   del m._research_packed_qkv
