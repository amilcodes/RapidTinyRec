"""Read generated Inductor matrix/layout evidence without executing wrappers.

Packed output extent/stride is checked here. Fused downstream Q/K/V indexing
is a separate proof: this module must not claim compiled split-view correctness.
"""
import ast
from collections import Counter
from .audit_qkv_outputs import require
from .qkv_wrapper_syntax import call_definition


def literal_tuple(node):
    try:return tuple(ast.literal_eval(node))
    except (ValueError,TypeError):return None


def extent_product(nodes):
    coefficient=1;symbols=[]
    def factor(node):
        nonlocal coefficient
        if isinstance(node,ast.Constant) and type(node.value) is int and node.value>0:coefficient*=node.value
        elif isinstance(node,ast.Name):symbols.append(node.id)
        elif isinstance(node,ast.BinOp) and isinstance(node.op,ast.Mult):factor(node.left);factor(node.right)
        else:raise ValueError('Unsupported symbolic allocation extent')
    for node in nodes:factor(node)
    return coefficient,tuple(sorted(symbols))


def contiguous_layout(shape,stride):
    if not isinstance(shape,ast.Tuple) or not isinstance(stride,ast.Tuple) or len(shape.elts)!=len(stride.elts):return False
    return all(extent_product(shape.elts[i+1:])==extent_product([step]) for i,step in enumerate(stride.elts))


def inspect_wrapper(source,packed):
    tree=ast.parse(source)
    function,_=call_definition(tree,required=False)
    if function is None:return None
    nodes=sorted([n for n in ast.walk(function) if isinstance(n,(ast.Assign,ast.Call))],key=lambda n:(n.lineno,n.col_offset))
    layouts={};calls=[];attention_tuples=set();attention_outputs=set()
    for node in nodes:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id;value=node.value
            attention_tuples.discard(name);attention_outputs.discard(name)
            if isinstance(value,ast.Call) and isinstance(value.func,ast.Name) and value.func.id=='empty_strided_cuda':
                require(len(value.args)>=3,'Incomplete CUDA output allocation')
                layouts[name]=dict(shape=value.args[0],stride=value.args[1],dtype=ast.unparse(value.args[2]))
            elif isinstance(value,ast.Name) and value.id in layouts:
                layouts[name]=layouts[value.id].copy()
            elif isinstance(value,ast.Call) and isinstance(value.func,ast.Name) and value.func.id=='reinterpret_tensor' and len(value.args)==4 and isinstance(value.args[0],ast.Name) and value.args[0].id in layouts:
                prior=layouts[value.args[0].id];shape,stride=value.args[1:3]
                try:proven=ast.literal_eval(value.args[3])==0 and contiguous_layout(prior['shape'],prior['stride']) and contiguous_layout(shape,stride) and extent_product(shape.elts)==extent_product(prior['shape'].elts)
                except ValueError:proven=False
                if proven:layouts[name]=dict(shape=shape,stride=stride,dtype=prior['dtype'])
                else:layouts.pop(name,None)  # An unproved view cannot qualify as a native matrix out.

            else:
                layouts.pop(name,None)
                if isinstance(value,ast.Call) and ast.unparse(value.func)=='torch.ops.aten._scaled_dot_product_flash_attention.default':attention_tuples.add(name)
                elif isinstance(value,ast.Subscript) and isinstance(value.value,ast.Name) and value.value.id in attention_tuples and isinstance(value.slice,ast.Constant) and value.slice.value==0:attention_outputs.add(name)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='assert_tensor_metadata' and len(node.args)==5 and isinstance(node.args[0],ast.Name) and node.args[0].id in attention_outputs:
            require(ast.literal_eval(node.args[4])=='torch.ops.aten._scaled_dot_product_flash_attention.default','Attention output metadata source differs')
            layouts[node.args[0].id]=dict(shape=node.args[1],stride=node.args[2],dtype=ast.unparse(node.args[3]))
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute) or not isinstance(node.func.value,ast.Name) or node.func.value.id!='extern_kernels':continue
        op=node.func.attr
        require(op in ('mm','addmm'),'Unexpected native matrix operator')
        require(len(node.args)==(3 if op=='addmm' else 2),'Unexpected matrix operand count')
        options={k.arg:k.value for k in node.keywords}
        require(len(options)==len(node.keywords) and set(options)<=({'out','alpha','beta'} if op=='addmm' else {'out'}),'Unexpected matrix options')
        if op=='addmm':
            require(all(ast.literal_eval(options[k])==1 for k in ('alpha','beta') if k in options),'Projection scaling differs')
        args=node.args[-2:]
        require(len(args)==2,'Missing matrix operands')
        weight=args[1]
        require(isinstance(weight,ast.Call) and isinstance(weight.func,ast.Name) and weight.func.id=='reinterpret_tensor' and len(weight.args)==4,'Unrecognized matrix weight view')
        shape=literal_tuple(weight.args[1]);stride=literal_tuple(weight.args[2])
        require(shape is not None and stride==(1,shape[0]) and ast.literal_eval(weight.args[3])==0,'Weight transpose stride/offset differs')
        roles={('addmm',(1536,1536)):'query',('addmm',(1536,256)):'key_or_value',('addmm',(1536,2048)):'packed_qkv',('mm',(1536,1536)):'attention_output',('mm',(1536,8960)):'gate_or_up',('mm',(8960,1536)):'down',('mm',(1536,152265)):'vocab_head'}
        role=roles.get((op,shape));require(role is not None,'Unexpected matrix geometry/operator')
        out=[k.value for k in node.keywords if k.arg=='out']
        require(len(out)==1 and isinstance(out[0],ast.Name) and out[0].id in layouts,f'Unknown output allocation or alias at line {node.lineno}: {[ast.unparse(x) for x in out]}')
        layout=layouts[out[0].id]
        require(layout['dtype']=='torch.bfloat16','Matrix output is not BF16')
        require(isinstance(layout['shape'],ast.Tuple) and len(layout['shape'].elts)==2 and ast.literal_eval(layout['shape'].elts[1])==shape[1]
                and literal_tuple(layout['stride'])==(shape[1],1),'Output matrix extent/stride differs')
        if role=='packed_qkv':
            activation=args[0]
            require(isinstance(activation,ast.Call) and isinstance(activation.func,ast.Name) and activation.func.id=='reinterpret_tensor' and len(activation.args)==4,'Packed activation view missing')
            require(isinstance(activation.args[1],ast.Tuple) and len(activation.args[1].elts)==2 and ast.literal_eval(activation.args[1].elts[1])==1536
                    and literal_tuple(activation.args[2])==(1536,1) and ast.literal_eval(activation.args[3])==0,'Packed activation stride/extent differs')
            require(ast.dump(activation.args[1].elts[0])==ast.dump(layout['shape'].elts[0]),'Packed projection row extent differs')
        calls.append(dict(line=node.lineno,operator=op,role=role,weight_shape=list(shape),weight_stride=list(stride),output=out[0].id,output_shape=ast.unparse(layout['shape']),output_stride=ast.unparse(layout['stride']),output_dtype=layout['dtype']))
    if not calls:return None
    expected={'packed_qkv':28,'attention_output':28,'gate_or_up':56,'down':28,'vocab_head':1} if packed else {'query':28,'key_or_value':56,'attention_output':28,'gate_or_up':56,'down':28,'vocab_head':1}
    require(dict(Counter(c['role'] for c in calls))==expected,'Incomplete/extra model matrix inventory')
    return dict(matrix_calls=len(calls),roles=expected,calls=calls,packed_output_extent_verified=packed,compiled_split_view_verified=False)
