"""Seeded prototype for a distinct BF16 projection hypothesis; not model-integrated.
Packing occurs outside generation; execution retains full Q/K/V output widths.
Splits intentionally share one output owner, with disjoint column ranges.
"""
import torch
from torch import nn
from torch.nn import functional as F
class PackedQKV(nn.Module):
 def __init__(self,q,k,v):
  super().__init__();parts=(q,k,v)
  if any(not isinstance(p,nn.Linear) or p.bias is None for p in parts):raise ValueError('Three biased native linear projections required')
  if len({(p.in_features,p.weight.dtype,p.weight.device) for p in parts})!=1 or k.out_features!=v.out_features or q.out_features%k.out_features:raise ValueError('Incompatible grouped projections')
  if any(p.bias.dtype!=p.weight.dtype or p.bias.device!=p.weight.device for p in parts):raise ValueError('Bias dtype/device differs')
  self.widths=(q.out_features,k.out_features,v.out_features);self.in_features=q.in_features
  self.register_buffer('weight',torch.cat([p.weight.detach() for p in parts]).contiguous(),persistent=False)
  self.register_buffer('bias',torch.cat([p.bias.detach() for p in parts]).contiguous(),persistent=False)
 def forward(self,x):
  if x.shape[-1]!=self.in_features or x.dtype!=self.weight.dtype or x.device!=self.weight.device:raise ValueError('Input contract differs')
  return F.linear(x,self.weight,self.bias).split(self.widths,dim=-1)
