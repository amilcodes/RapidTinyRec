"""Exact cache-copy addresses, masks and value selection; no model arithmetic."""
import ast
import json
import numpy as np
from .qkv_index_expression import IntegerExpressions,scalar_assignments
from .qkv_kernel_reads import embedded_kernel,pointer_reads
from .qkv_launch_domain import verify_flat_store


def verify_copy_partition(source,function,launch,output,*,mode,prefix,suffix,old_pointer,new_pointer=None):
    if mode not in ('append','concat') or type(prefix) is not int or not 1<=prefix<=121 or type(suffix) is not int or suffix not in range(4):raise ValueError('Invalid independent cache partition')
    length=suffix+1 if mode=='append' else prefix+suffix+1
    domain=verify_flat_store(source,function,launch,output,50*256*length)
    x=np.arange(domain['padded_lanes'],dtype=np.int64);inside=x<50*256*length
    row=x//(256*length);head=(x//(128*length))%2;token=(x//128)%length;col=x%128
    if mode=='append':parts=[(old_pointer,inside&(token<suffix),row*256*suffix+head*128*suffix+token*128+col)]
    else:
        if new_pointer is None or new_pointer==old_pointer:raise ValueError('Distinct concat source pointers required')
        parts=[(old_pointer,inside&(token<prefix),head*128*prefix+token*128+col),
               (new_pointer,inside&(token>=prefix),row*256*(suffix+1)+head*128*(suffix+1)+(token-prefix)*128+col)]
    fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==function)
    load_lines=[n.lineno for n in ast.walk(fn) if isinstance(n,ast.Call) and ast.unparse(n.func)=='tl.load']
    if len(load_lines)!=len(set(load_lines)):raise ValueError('Ambiguous same-line load identities')
    assignments=scalar_assignments(source,function)
    expr=IntegerExpressions(assignments,{**launch['runtime_integers'],**launch['constexpr'],'xindex':x,'xoffset':(x//domain['block'])*domain['block']})
    loads={};expected=np.zeros(x.shape,dtype=np.int8);partition=[]
    for label,(pointer,mask,address) in enumerate(parts,1):
        reads=pointer_reads(source,function,pointer)['reads']
        if len(reads)!=1:raise ValueError('Expected exactly one old-cache copy load per input')
        read=reads[0];actual_mask=np.broadcast_to(np.asarray(expr.evaluate(read['mask'])),x.shape)
        actual_address=np.broadcast_to(np.asarray(expr.evaluate(read['offset'])),x.shape)
        if actual_mask.dtype.kind!='b' or not np.array_equal(actual_mask,mask) or not np.array_equal(actual_address[mask],address[mask]):raise ValueError('Cache load mask/address changes row/head/token partition')
        loads[read['line']]=label;expected[mask]=label;partition.append(dict(pointer=pointer,lanes=int(mask.sum())))
    # Provenance evaluation only. Arithmetic is unknown even if it could cancel:
    # old cache lanes must be direct loaded values through lossless BF16/FP32
    # casts and actual where predicates, never arbitrary numeric expressions.
    cache={};active=set()
    def selected(node):
        if isinstance(node,ast.Name) and node.id in assignments:
            if node.id in active:raise ValueError('Cyclic value selection')
            if node.id not in cache:
                active.add(node.id)
                try:cache[node.id]=selected(assignments[node.id])
                finally:active.remove(node.id)
            return cache[node.id]
        if isinstance(node,ast.Call):
            name=ast.unparse(node.func)
            if name=='tl.load':return loads.get(node.lineno,0)
            if isinstance(node.func,ast.Attribute) and node.func.attr=='to' and len(node.args)==1 and not node.keywords and ast.unparse(node.args[0]) in ('tl.float32','tl.bfloat16'):return selected(node.func.value)
            if name=='tl.where' and len(node.args)==3 and not node.keywords:
                mask=np.asarray(expr.evaluate(node.args[0]))
                if mask.dtype.kind!='b':raise ValueError('Value selector is not Boolean')
                return np.where(mask,selected(node.args[1]),selected(node.args[2]))
        return 0
    stores=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and ast.unparse(n.func)=='tl.store' and n.args and any(isinstance(v,ast.Name) and v.id==output for v in ast.walk(n.args[0]))]
    if len(stores)!=1:raise ValueError('Expected one cache output store')
    chosen=np.broadcast_to(np.asarray(selected(stores[0].args[1])),x.shape);checked=expected!=0
    if not np.array_equal(chosen[checked],expected[checked]):raise ValueError('Store value does not select exact old-cache copy on each partition')
    return dict(mode=mode,prefix=prefix,suffix=suffix,partitions=partition,copy_selection_verified=True,
                zero_length_old_suffix_checked=mode=='append' and suffix==0,complete_compiled_view_proof=False)


def validate_cache_copy(source,launches,cache_inputs,cache_returns,attention_layout,graph,proof_cache=None):
    if proof_cache is None:proof_cache={}
    initial=len(proof_cache);p=graph['prefixes'][0][0]['shape'][2];s=graph['suffixes'][0][0]['shape'][2]
    if p not in (76,121) or s not in range(4) or graph['cache_tokens']!=p+s:raise ValueError('Independent cache lengths differ')
    by_line={r['line']:r for r in launches['launches']};proofs=[]
    def prove(k,launch,out,mode,old,new=None):
        key=json.dumps(dict(hash=k['sha256'],function=k['function'],out=out,mode=mode,prefix=p,suffix=s,old=old,new=new,integers=launch['runtime_integers'],config=launch['constexpr'],grid=launch['actual_grid']),sort_keys=True)
        if key not in proof_cache:proof_cache[key]=verify_copy_partition(k['source'],k['function'],launch,out,mode=mode,prefix=p,suffix=s,old_pointer=old,new_pointer=new)
        return proof_cache[key]
    for returned in cache_returns['suffixes']:
        layer,role=returned['layer'],returned['role'];metadata=returned['metadata']
        # The actual output tensor must be the exact returned suffix owner, not
        # merely a source variable with the expected dimensions.
        writers=[]
        for out in attention_layout['kernel_outputs']:
            launch=by_line[out['line']]
            if launch['runtime_tensors'].get(out['pointer'])==metadata:writers.append((out,launch))
        if len(writers)!=1:raise ValueError('Suffix return lacks unique actual kernel writer')
        out,append_launch=writers[0];k=embedded_kernel(source,out['kernel'])
        bound=[u for u in cache_inputs['uses'] if u['line']==out['line'] and all(cache_inputs['inputs'][u['input_index']][a]==b for a,b in [('family','suffixes'),('layer',layer),('role',role)])]
        if len(bound)!=1:raise ValueError('Suffix writer does not read its exact old suffix')
        append=prove(k,append_launch,out['pointer'],'append',bound[0]['parameter'])
        final=attention_layout['attention'][layer]['positions'][role]['writer'];concat_launch=by_line[final['line']];k=embedded_kernel(source,final['kernel'])
        prefix=[u for u in cache_inputs['uses'] if u['line']==final['line'] and all(cache_inputs['inputs'][u['input_index']][a]==b for a,b in [('family','prefixes'),('layer',layer),('role',role)])]
        suffix_args=[name for name,record in concat_launch['runtime_tensors'].items() if record==metadata]
        if len(prefix)!=1 or len(suffix_args)!=1:raise ValueError('Final attention concat does not bind exact layer/role prefix and returned suffix')
        concat=prove(k,concat_launch,final['pointer'],'concat',prefix[0]['parameter'],suffix_args[0])
        proofs.append(dict(layer=layer,role=role,append=append,concat=concat))
    if len(proofs)!=56 or {(x['layer'],x['role']) for x in proofs}!={(i,r) for i in range(28) for r in ('k','v')}:raise ValueError('Incomplete cache copy coverage')
    return dict(copies=proofs,new_unique_copy_evaluations=len(proof_cache)-initial,prefix_suffix_copy_verified=True,complete_compiled_view_proof=False)
