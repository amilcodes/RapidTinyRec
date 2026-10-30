"""Extract, never execute, pointer-load address/mask syntax from embedded Triton.

This exposes observed source expressions. It cannot infer a Q/K/V role or launch
layout from tensor sizes alone; those independent bindings remain mandatory.
"""
import ast
import hashlib
from functools import lru_cache
from .qkv_index_expression import scalar_assignments


@lru_cache(maxsize=8)
def _wrapper_assignments(wrapper_source):
    # Exact immutable source key. Keep parsed nodes private; never return them
    # from the public extractor. No tensor/runtime state is cached here.
    return tuple((node.targets[0].id,node.value) for node in ast.parse(wrapper_source).body
                 if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name))


def embedded_kernel(wrapper_source,kernel_symbol):
    matches=[]
    for symbol,call in _wrapper_assignments(wrapper_source):
        if symbol!=kernel_symbol:continue
        if not isinstance(call,ast.Call) or ast.unparse(call.func)!='async_compile.triton' or len(call.args)!=2:raise ValueError('Unsupported embedded kernel definition')
        if not all(isinstance(x,ast.Constant) and isinstance(x.value,str) for x in call.args):raise ValueError('Kernel name/source must be literal')
        if len(call.keywords)!=1 or call.keywords[0].arg!='device_str' or ast.literal_eval(call.keywords[0].value)!='cuda':raise ValueError('Unexpected kernel device/options')
        name,source=(x.value for x in call.args);matches.append((name,source))
    if len(matches)!=1:raise ValueError('Missing/ambiguous embedded kernel')
    name,source=matches[0]
    return dict(function=name,source=source,sha256=hashlib.sha256(source.encode()).hexdigest())


def pointer_reads(kernel_source,function_name,pointer_name):
    tree=ast.parse(kernel_source);functions=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==function_name]
    if len(functions)!=1:raise ValueError('Missing/ambiguous Triton function')
    function=functions[0];parameters=[x.arg for x in function.args.args]
    if pointer_name not in parameters:raise ValueError('Pointer is not an actual kernel parameter')
    assignments=scalar_assignments(kernel_source,function_name)
    reads=[];covered=set()
    for node in ast.walk(function):
        if not isinstance(node,ast.Call) or ast.unparse(node.func)!='tl.load' or not node.args:continue
        address=node.args[0]
        uses=[x for x in ast.walk(address) if isinstance(x,ast.Name) and x.id==pointer_name]
        if not uses:continue
        if len(uses)!=1 or not isinstance(address,ast.BinOp) or not isinstance(address.op,ast.Add) or not isinstance(address.left,ast.Name) or address.left.id!=pointer_name:raise ValueError('Unsupported pointer/address alias or operation')
        keyword={x.arg:x.value for x in node.keywords}
        if len(keyword)!=len(node.keywords) or None in keyword:raise ValueError('Ambiguous load options')
        if len(node.args)>3 or (len(node.args)>=2 and 'mask' in keyword):raise ValueError('Ambiguous load mask')
        mask=node.args[1] if len(node.args)>=2 else keyword.get('mask',ast.Constant(True))
        reads.append(dict(line=node.lineno,offset=ast.unparse(address.right),mask=ast.unparse(mask)))
        covered.add(id(uses[0]))
    all_uses={id(x) for x in ast.walk(function) if isinstance(x,ast.Name) and x.id==pointer_name}
    if not reads or all_uses!=covered:raise ValueError('Pointer escaped declared loads, was aliased, written, or never read')
    return dict(pointer=pointer_name,parameter_index=parameters.index(pointer_name),reads=reads,assignments=assignments,
                role_bound=False,launch_domain_bound=False,complete_compiled_view_proof=False)
