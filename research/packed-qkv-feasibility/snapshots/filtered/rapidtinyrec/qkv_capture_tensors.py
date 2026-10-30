"""CPU metadata snapshots only: no tensor data, copies, .item or GPU work.

Snapshot immediately without retaining tensor references, so observation does
not extend temporary allocation lifetimes inside CUDA graph capture.
"""
import torch
from .qkv_storage import tensor_record


def snapshot_tensors(values):
    if type(values) not in (list,tuple):return {}
    result={}
    for i,value in enumerate(values):
        if isinstance(value,torch.Tensor):
            if type(value) not in (torch.Tensor,torch.nn.Parameter):raise ValueError('Tensor subclasses unsupported in capture metadata')
            result[str(i)]=tensor_record(value)
    return result
