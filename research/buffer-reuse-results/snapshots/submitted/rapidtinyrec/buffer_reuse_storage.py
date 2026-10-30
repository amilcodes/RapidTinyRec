"""Independent live tensor inventory for frozen serial/wave replay ownership.

Records supplied model parameters, arena capacity, graph inputs and graph
outputs. It cannot prove completeness of hidden CUDA allocator/stream state.
"""
import math


def tensor_record(tensor):
    return dict(shape=list(tensor.shape),stride=list(tensor.stride()),dtype=str(tensor.dtype),device=str(tensor.device),
        storage_ptr=tensor.untyped_storage().data_ptr(),storage_bytes=tensor.untyped_storage().nbytes(),
        byte_start=tensor.storage_offset()*tensor.element_size(),logical_bytes=tensor.numel()*tensor.element_size(),
        element_bytes=tensor.element_size(),contiguous=tensor.is_contiguous())


def collect_inventory(engine,model,stage):
    if stage not in ('before-measurement','after-measurement','after-profiles') or not engine.pool.frozen:
        raise ValueError('Frozen replay inventory stage required')
    graphs=[]
    for index,(_,entry) in enumerate(sorted(engine.pool.entries.items(),key=lambda pair:repr(pair[0]))):
        buffers,_,output=entry
        graphs.append(dict(index=index,cache_tokens=output.cache_length()-1,
            tokens={k:tensor_record(x) for k,x in buffers.tensors.items()},
            prefixes=[[tensor_record(x) for x in pair] for pair in buffers.prefixes],
            suffixes=[[tensor_record(x) for x in pair] for pair in buffers.pairs],
            output_prefixes=[[tensor_record(x) for x in pair] for pair in output.prefixes],
            output_suffixes=[[tensor_record(x) for x in pair] for pair in output.pairs],logits=tensor_record(output.logits)))
    return dict(schema='released-buffer-reuse-storage-v1',stage=stage,variant=engine.arm,
        model_parameters={name:tensor_record(p) for name,p in model.named_parameters()},
        prefix_capacity=[[tensor_record(x) for x in pair] for pair in engine.arena.prefix_storage],
        suffix_capacity=[[tensor_record(x) for x in pair] for pair in engine.arena.suffix_storage],graphs=graphs,
        scope='Supplied live tensors only; not hidden allocator/stream completeness or numerical proof.')


def _record(r):
    keys=('storage_ptr','storage_bytes','byte_start','logical_bytes','element_bytes')
    if any(type(r[k]) is not int for k in keys) or r['storage_ptr']<=0 or r['storage_bytes']<=0:
        raise ValueError('Invalid live storage record')
    if (r['element_bytes'] not in (4,8) or r['byte_start']<0 or r['logical_bytes']<0
            or r['byte_start']+r['logical_bytes']>r['storage_bytes'] or r['contiguous'] is not True
            or any(type(n) is not int or n<0 for n in r['shape'])
            or math.prod(r['shape'])*r['element_bytes']!=r['logical_bytes']):
        raise ValueError('Invalid contiguous tensor extent')
    if r['dtype'] not in ('torch.float32','torch.int64'):raise ValueError('Unexpected replay dtype')
    return (r['device'],r['storage_ptr'])


def validate_inventory(record,*,layers,kv_heads,head_dim,vocab_size,device='cuda:0'):
    if record['schema']!='released-buffer-reuse-storage-v1' or record['stage'] not in ('before-measurement','after-measurement','after-profiles'):
        raise ValueError('Distinct storage identity required')
    if record['variant'] not in ('released_tf32','reuse_normalized_tf32'):raise ValueError('Invalid storage variant')
    wave=False;batch=100 if wave else 50;physical=2 if wave else 1
    def checked(r,shape=None,dtype='torch.float32'):
        key=_record(r)
        if r['device']!=device or r['dtype']!=dtype or (shape is not None and r['shape']!=list(shape)):
            raise ValueError('Live tensor shape/device/dtype differs')
        return key
    def pairs(rows):
        if len(rows)!=layers or any(len(pair)!=2 for pair in rows):raise ValueError('Missing KV layer')
        return [x for pair in rows for x in pair]
    params=record['model_parameters']
    if not params:raise ValueError('Missing model storage inventory')
    # Model weight tying is permitted. Replay buffers and outputs may not alias weights.
    model_keys={checked(r) for r in params.values()}
    capacity={};seen=set(model_keys)
    for name,rowsize,tokens in [('prefix_capacity',physical,121),('suffix_capacity',batch,3)]:
        values=pairs(record[name]);keys=[]
        for r in values:
            key=checked(r,(rowsize*kv_heads*tokens*head_dim,))
            if key in seen:raise ValueError('Arena capacity aliases another mutable owner or model')
            seen.add(key);keys.append(key)
        capacity[name]=keys
    graphs=record['graphs']
    if len(graphs)!=8 or sorted(g['index'] for g in graphs)!=list(range(8)):
        raise ValueError('Expected eight distinct graphs')
    layouts=[];outputs=[]
    for g in graphs:
        prefix=pairs(g['prefixes']);suffix=pairs(g['suffixes']);out_prefix=pairs(g['output_prefixes']);out_suffix=pairs(g['output_suffixes'])
        length=prefix[0]['shape'][2];suffix_length=suffix[0]['shape'][2]
        if length not in (76,121) or suffix_length not in range(4) or g['cache_tokens']!=length+suffix_length:
            raise ValueError('Unexpected graph cache lengths')
        layouts.append((length,suffix_length))
        for i,(p,s,op,os) in enumerate(zip(prefix,suffix,out_prefix,out_suffix)):
            if checked(p,(physical,kv_heads,length,head_dim))!=capacity['prefix_capacity'][i] or p['byte_start']!=0:
                raise ValueError('Graph prefix is not its assigned arena owner')
            if checked(s,(batch,kv_heads,suffix_length,head_dim))!=capacity['suffix_capacity'][i] or s['byte_start']!=0:
                raise ValueError('Graph suffix input is not its assigned arena owner')
            if op!=p:raise ValueError('Graph output prefix copied, expanded or remapped')
            outputs.append((checked(os,(batch,kv_heads,suffix_length+1,head_dim)),os))
        if set(g['tokens'])!={'input_ids','position_ids','cache_position'}:raise ValueError('Wrong graph token buffers')
        for name,r in g['tokens'].items():
            key=checked(r,(1,) if name=='cache_position' else (batch,1),'torch.int64')
            if key in seen:raise ValueError('Graph token buffers share mutable storage')
            seen.add(key)
        outputs.append((checked(g['logits'],(batch,1,vocab_size)),g['logits']))
    if sorted(layouts)!=sorted((length,s) for length in (76,121) for s in range(4)):
        raise ValueError('Duplicate or missing frozen graph shape')
    for key,r in outputs:
        if key in seen:raise ValueError('Graph outputs alias inputs/model or another graph output')
        seen.add(key)
    return dict(graphs=8,physical_prefix_batch=physical,decode_batch=batch,
                supplied_storage_ownership_verified=True,hidden_allocator_completeness=False)


def validate_stages(records,**layout):
    if [r['stage'] for r in records]!=['before-measurement','after-measurement','after-profiles']:
        raise ValueError('All three ordered frozen storage snapshots required')
    checks=[validate_inventory(r,**layout) for r in records]
    baseline={k:v for k,v in records[0].items() if k!='stage'}
    if any({k:v for k,v in r.items() if k!='stage'}!=baseline for r in records[1:]):
        raise ValueError('Frozen graph/arena storage changed after setup')
    return dict(stable_across_measurement_profiles=True,stages=checks)
