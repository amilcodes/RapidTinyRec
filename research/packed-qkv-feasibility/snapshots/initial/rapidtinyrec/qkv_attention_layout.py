"""Independent folded-attention output extents and final-store coverage.

Pinned release geometry: beam50, Q12/KV2, head128, 28 layers. Cache length comes
from the independently validated captured graph storage. Packed input address
and intermediate prefix/suffix routing are separate, still mandatory proofs.
"""
import ast
import math
import json
from .qkv_index_expression import IntegerExpressions
from .qkv_wrapper_syntax import call_definition,flat_statements
from .qkv_kernel_reads import embedded_kernel
from .qkv_launch_domain import verify_flat_store

ATTENTION='torch.ops.aten._scaled_dot_product_flash_attention.default'


def validate_attention_layout(source,scalars,launches,cache_tokens,proof_cache=None):
    if proof_cache is None:proof_cache={}
    initial_count=len(proof_cache)
    if type(cache_tokens) is not int or not 1<=cache_tokens<=128:raise ValueError('Unsupported independent graph cache extent')
    expected={'q':(50,2,6,128),'k':(50,2,cache_tokens+1,128),'v':(50,2,cache_tokens+1,128)}
    fn,_=call_definition(ast.parse(source));env=dict(scalars['entry_names']);updates={}
    for u in scalars['scalar_updates']:updates.setdefault(u['line'],[]).append(u)
    layouts={};writers={};serial=0;attention=[];tuples=set();outputs=set();kernels={};kernel_outputs=[]
    launch_by_line={r['line']:r for r in launches['launches']}
    def shape(node):
        if not isinstance(node,ast.Tuple):raise ValueError('Expected explicit shape/stride tuple')
        result=tuple(IntegerExpressions({},env).evaluate(x) for x in node.elts)
        if any(type(x) is not int or x<=0 for x in result):raise ValueError('Invalid independently evaluated layout')
        return result
    def contiguous(dims):return tuple(math.prod(dims[i+1:]) for i in range(len(dims)))
    def kernel(symbol):
        if symbol not in kernels:kernels[symbol]=embedded_kernel(source,symbol)
        return kernels[symbol]
    for statement in flat_statements(fn):
        if isinstance(statement,ast.Delete):
            for t in statement.targets:
                if isinstance(t,ast.Name):layouts.pop(t.id,None);tuples.discard(t.id);outputs.discard(t.id)
        elif isinstance(statement,ast.Assign) and len(statement.targets)==1 and isinstance(statement.targets[0],ast.Name):
            name=statement.targets[0].id;v=statement.value;prior=dict(layouts);layouts.pop(name,None);tuples.discard(name);outputs.discard(name)
            if isinstance(v,ast.Name) and v.id in prior:layouts[name]=prior[v.id]
            elif isinstance(v,ast.Call) and ast.unparse(v.func)=='empty_strided_cuda':
                if len(v.args)!=3:raise ValueError('Unexpected allocation signature')
                dims,steps=shape(v.args[0]),shape(v.args[1]);serial+=1
                layouts[name]=dict(shape=dims,stride=steps,owner=serial,dtype=ast.unparse(v.args[2]))
            elif isinstance(v,ast.Call) and ast.unparse(v.func)=='reinterpret_tensor' and isinstance(v.args[0],ast.Name) and v.args[0].id in prior:
                old=prior[v.args[0].id];dims,steps=shape(v.args[1]),shape(v.args[2])
                if len(v.args)!=4 or ast.literal_eval(v.args[3])!=0 or old['stride']!=contiguous(old['shape']) or steps!=contiguous(dims) or math.prod(dims)!=math.prod(old['shape']):raise ValueError('Unproved output alias extent/offset')
                layouts[name]=dict(shape=dims,stride=steps,owner=old['owner'],dtype=old['dtype'])
            elif isinstance(v,ast.Call) and ast.unparse(v.func)==ATTENTION:
                if len(v.args)!=3 or any(not isinstance(a,ast.Name) for a in v.args):raise ValueError('Unsupported attention positions')
                layer={}
                for role,arg in zip(('q','k','v'),v.args):
                    layout=prior.get(arg.id)
                    if layout is None or layout['dtype']!='torch.bfloat16' or layout['shape']!=expected[role] or layout['stride']!=contiguous(expected[role]):raise ValueError('Attention layout differs from independent folded geometry/cache')
                    write=writers.get(layout['owner'])
                    if write is None:raise ValueError('Attention input has no current generated output store')
                    k=kernel(write['kernel']);launch=launch_by_line[write['line']]
                    key=json.dumps(dict(source=k['sha256'],function=k['function'],pointer=write['pointer'],extent=math.prod(expected[role]),integers=launch['runtime_integers'],config=launch['constexpr'],grid=launch['actual_grid']),sort_keys=True)
                    if key not in proof_cache:proof_cache[key]=verify_flat_store(k['source'],k['function'],launch,write['pointer'],math.prod(expected[role]))
                    proof=proof_cache[key]
                    layer[role]=dict(buffer=arg.id,shape=list(layout['shape']),stride=list(layout['stride']),writer=write,store=proof)
                if len({prior[a.id]['owner'] for a in v.args})!=3:raise ValueError('Attention outputs alias')
                attention.append(dict(line=v.lineno,positions=layer));tuples.add(name)
            elif isinstance(v,ast.Subscript) and isinstance(v.value,ast.Name) and v.value.id in tuples and isinstance(v.slice,ast.Constant) and v.slice.value==0:outputs.add(name)
        for call in (n for n in ast.walk(statement) if isinstance(n,ast.Call)):
            if ast.unparse(call.func)=='assert_tensor_metadata' and len(call.args)==5 and isinstance(call.args[0],ast.Name) and call.args[0].id in outputs:
                dims,steps=shape(call.args[1]),shape(call.args[2])
                if dims!=expected['q'] or steps!=contiguous(dims) or ast.unparse(call.args[3])!='torch.bfloat16' or ast.literal_eval(call.args[4])!=ATTENTION:raise ValueError('Attention output metadata differs')
                serial+=1;layouts[call.args[0].id]=dict(shape=dims,stride=steps,owner=serial,dtype='torch.bfloat16')
            if isinstance(call.func,ast.Attribute) and call.func.attr=='run' and isinstance(call.func.value,ast.Name):
                symbol=call.func.value.id;k=kernel(symbol);f=next(n for n in ast.parse(k['source']).body if isinstance(n,ast.FunctionDef) and n.name==k['function'])
                params=[a.arg for a in f.args.args if a.annotation is None or ast.unparse(a.annotation)!='tl.constexpr']
                for store in (n for n in ast.walk(f) if isinstance(n,ast.Call) and ast.unparse(n.func)=='tl.store'):
                    address=store.args[0];base=address.left if isinstance(address,ast.BinOp) and isinstance(address.op,ast.Add) else address
                    if isinstance(base,ast.Name) and base.id in params:
                        i=params.index(base.id)
                        if i<len(call.args) and isinstance(call.args[i],ast.Name) and call.args[i].id in layouts:
                            writer=dict(line=call.lineno,kernel=symbol,pointer=base.id)
                            writers[layouts[call.args[i].id]['owner']]=writer
                            kernel_outputs.append(dict(**writer,layout=dict(layouts[call.args[i].id]),buffer=call.args[i].id))
            # Any native matrix write invalidates earlier kernel content in the
            # allocation, including aliases that retain its storage identity.
            if ast.unparse(call.func) in ('extern_kernels.mm','extern_kernels.addmm'):
                for kw in call.keywords:
                    if kw.arg=='out' and isinstance(kw.value,ast.Name) and kw.value.id in layouts:writers.pop(layouts[kw.value.id]['owner'],None)
        for u in updates.get(statement.lineno,[]):
            if u['value'] is None:env.pop(u['name'],None)
            else:env[u['name']]=u['value']
    if len(attention)!=28:raise ValueError('Expected all28 folded attention sites')
    return dict(attention=attention,kernel_outputs=kernel_outputs,new_unique_store_evaluations=len(proof_cache)-initial_count,independent_attention_extents_verified=True,final_attention_store_coverage_verified=True,
                intermediate_cache_maps_verified=False,packed_column_maps_verified=False,complete_compiled_view_proof=False)
