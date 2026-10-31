"""Observe actual eager packed column views; hooks never remain during compile.

This is eager view-layout evidence only. CUDA graph/generated-wrapper layout
and actual native BF16 dispatch need independent allocated validation.
"""
import contextlib
import math
from .packed_qkv import PackedQKV
from .qkv_storage import tensor_record


def validate_views(record, widths, hidden_shape, device):
    if record['input_shape']!=list(hidden_shape) or record['widths']!=list(widths) or len(record['views'])!=3:
        raise ValueError('Packed view scope differs')
    total=sum(widths);leading=list(hidden_shape[:-1]);stride=[math.prod((leading+[total])[i+1:]) for i in range(len(leading)+1)]
    owners=set();offset=0
    for width,row in zip(widths,record['views']):
        if (row['shape']!=leading+[width] or row['stride']!=stride or row['dtype']!='torch.bfloat16'
            or row['device']!=device or row['byte_start']!=offset*2 or row['element_bytes']!=2
            or row['logical_bytes']!=math.prod(leading)*width*2
            or row['storage_bytes']!=math.prod(leading)*total*2 or row['storage_ptr']<=0):
            raise ValueError('Packed column view stride, offset, extent or dtype differs')
        owners.add(row['storage_ptr']);offset+=width
    if len(owners)!=1:raise ValueError('Packed views do not share exactly one owner')
    return dict(column_ranges_disjoint=True,full_width_covered=True,eager_only=True)


@contextlib.contextmanager
def observe_views(model):
    records=[];handles=[]
    def hook(name):
        def record(module,inputs,outputs):
            row=dict(module=name,input_shape=list(inputs[0].shape),widths=list(module.widths),views=[tensor_record(x) for x in outputs])
            validate_views(row,module.widths,inputs[0].shape,str(inputs[0].device));records.append(row)
        return record
    try:
        for name,module in model.named_modules():
            if isinstance(module,PackedQKV):handles.append(module.register_forward_hook(hook(name)))
        yield records
    finally:
        for handle in handles:handle.remove()


def validate_coverage(records, config, lengths, device='cuda:0'):
    widths=[config.hidden_size,config.hidden_size//config.num_attention_heads*config.num_key_value_heads,config.hidden_size//config.num_attention_heads*config.num_key_value_heads]
    expected=[]
    for length in lengths:
        for step in range(5):
            for layer in range(config.num_hidden_layers):
                expected.append((f'model.layers.{layer}.self_attn._research_packed_qkv', [1,length,config.hidden_size] if step==0 else [50,1,config.hidden_size]))
    if len(records)!=len(expected):raise ValueError('Missing layer/step view observation')
    for row,(name,shape) in zip(records,expected):
        if row['module']!=name:raise ValueError('Layer view order differs')
        validate_views(row,widths,shape,device)
    return dict(observations=len(records),layers=config.num_hidden_layers,lengths=list(lengths),eager_only=True,compiled_layout_verified=False)
