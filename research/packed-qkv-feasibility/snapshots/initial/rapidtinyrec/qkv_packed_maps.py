"""Exact packed column read maps on independently bound new-token domains.

Semantic roles come from actual attention argument positions. Output layouts
come from source allocation lifetime analysis; suffix length comes from the
validated graph storage. Prefix/old-suffix content copying is a separate gate.
"""
import math
import json
import numpy as np
from .qkv_index_expression import IntegerExpressions,scalar_assignments
from .qkv_kernel_reads import embedded_kernel,pointer_reads
from .qkv_semantic_routes import kernel_stores
from .qkv_launch_domain import verify_flat_store


def verify_role_reads(kernel,function,pointer,output_pointer,layout,launch,role,suffix_length):
    if role not in ('q','k','v') or type(suffix_length) is not int or suffix_length not in range(4):raise ValueError('Invalid independent role/cache')
    dims=(50,2,6,128) if role=='q' else (50,2,suffix_length+1,128)
    stride=tuple(math.prod(dims[i+1:]) for i in range(4))
    if tuple(layout['shape'])!=dims or tuple(layout['stride'])!=stride or layout['dtype']!='torch.bfloat16':raise ValueError('Packed consumer layout differs from independent role/suffix')
    store=verify_flat_store(kernel,function,launch,output_pointer,math.prod(dims))
    x=np.arange(store['padded_lanes'],dtype=np.int64);inside=x<math.prod(dims)
    if role=='q':row=x//1536;col=x%1536;new=inside
    else:
        length=suffix_length+1;row=x//(256*length);col=((x//(128*length))%2)*128+x%128
        new=inside&(((x//128)%length)==suffix_length)
    start={'q':0,'k':1536,'v':1792}[role]
    direct=row*2048+start+col
    rotated=row*2048+start+(col//128)*128+(col+64)%128
    inputs={**launch['runtime_integers'],**launch['constexpr'],'xindex':x,'xoffset':(x//store['block'])*store['block']}
    expr=IntegerExpressions(scalar_assignments(kernel,function),inputs)
    reads=pointer_reads(kernel,function,pointer)['reads']
    if not 1<=len(reads)<=(1 if role=='v' else 3):raise ValueError('Unsupported packed read multiplicity')
    dcount=np.zeros(x.shape,dtype=np.int8);rcount=np.zeros(x.shape,dtype=np.int8);proof=[]
    for read in reads:
        mask=np.broadcast_to(np.asarray(expr.evaluate(read['mask'])),x.shape)
        address=np.broadcast_to(np.asarray(expr.evaluate(read['offset'])),x.shape)
        if mask.dtype.kind!='b':raise ValueError('Packed load mask must be Boolean')
        candidates=[]
        if np.array_equal(mask,new) and np.array_equal(address[mask],direct[mask]):candidates.append('direct')
        if role!='v':
            for label,allowed in [('rotation_all',new),('rotation_first',new&(col%128<64)),('rotation_second',new&(col%128>=64))]:
                if np.array_equal(mask,allowed) and np.array_equal(address[mask],rotated[mask]):candidates.append(label)
        if len(candidates)!=1:raise ValueError('Packed read has wrong role columns, cache lane, rotation, or padding mask')
        kind=candidates[0]
        if kind=='direct':dcount+=mask
        else:rcount+=mask
        proof.append(dict(line=read['line'],kind=kind,active_lanes=int(np.count_nonzero(mask))))
    if not np.array_equal(dcount,new.astype(np.int8)) or not np.array_equal(rcount,new.astype(np.int8) if role!='v' else np.zeros_like(rcount)):raise ValueError('Missing/duplicate direct or rotated packed lane coverage')
    return dict(role=role,suffix_length=suffix_length,reads=proof,new_token_lanes=int(np.count_nonzero(new)),store=store,
                exact_packed_columns_and_masks=True,prefix_suffix_copy_verified=False,complete_compiled_view_proof=False)


def validate_packed_maps(source,roles,layout_proof,launches,graph,proof_cache=None):
    if proof_cache is None:proof_cache={}
    initial_count=len(proof_cache)
    prefixes=[p for pair in graph['prefixes'] for p in pair];suffixes=[p for pair in graph['suffixes'] for p in pair]
    if len(prefixes)!=56 or len(suffixes)!=56:raise ValueError('Complete graph KV inputs required')
    prefix=prefixes[0]['shape'][2];suffix=suffixes[0]['shape'][2]
    if prefix not in (76,121) or suffix not in range(4) or graph['cache_tokens']!=prefix+suffix:raise ValueError('Independent graph cache lengths differ')
    if any(p['shape']!=[1,2,prefix,128] for p in prefixes) or any(p['shape']!=[50,2,suffix,128] for p in suffixes):raise ValueError('Inconsistent graph KV layout')
    launch_by_line={x['line']:x for x in launches['launches']};groups={}
    for read in roles['reads']:
        key=(read['layer'],read['role'],read['consumer_line'],read['pointer']);groups.setdefault(key,[]).append(read['load_line'])
    proofs=[]
    for (layer,role,line,pointer),read_lines in groups.items():
        launch=launch_by_line[line];symbol=launch['kernel_symbol'];k=embedded_kernel(source,symbol)
        _,stores=kernel_stores(k['source'],k['function'])
        stores=[s for s in stores if any((pointer,n) in s['dependencies'] for n in read_lines)]
        if len(stores)!=1 or any((pointer,n) not in stores[0]['dependencies'] for n in read_lines):raise ValueError('Packed read does not reach one complete output store')
        store=stores[0];outputs=[o for o in layout_proof['kernel_outputs'] if o['line']==line and o['kernel']==symbol and o['pointer']==store['pointer']]
        if len(outputs)!=1:raise ValueError('Packed consumer output layout not bound')
        actual_reads=pointer_reads(k['source'],k['function'],pointer)['reads']
        if sorted(read_lines)!=sorted(x['line'] for x in actual_reads):raise ValueError('Semantic routes omit packed reads')
        layout=outputs[0]['layout']
        # Reuse only identical source/address domains, never layer/role routing.
        # Every observed route, store and allocation is still checked above.
        key=json.dumps(dict(source_sha256=k['sha256'],function=k['function'],pointer=pointer,output=store['pointer'],
                            shape=layout['shape'],stride=layout['stride'],dtype=layout['dtype'],
                            integers=launch['runtime_integers'],constexpr=launch['constexpr'],grid=launch['actual_grid'],role=role,suffix=suffix),sort_keys=True)
        if key not in proof_cache:proof_cache[key]=verify_role_reads(k['source'],k['function'],pointer,store['pointer'],layout,launch,role,suffix)
        proof=proof_cache[key]
        proofs.append(dict(layer=layer,line=line,**proof))
    if len(proofs)!=84 or {(p['layer'],p['role']) for p in proofs}!={(i,r) for i in range(28) for r in ('q','k','v')}:raise ValueError('Expected all28 Q/K/V packed consumer maps')
    return dict(maps=proofs,new_unique_map_evaluations=len(proof_cache)-initial_count,exact_packed_columns_and_masks=True,prefix_suffix_copy_verified=False,complete_compiled_view_proof=False)
