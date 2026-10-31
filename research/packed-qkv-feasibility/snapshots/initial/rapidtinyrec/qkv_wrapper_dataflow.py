"""Link each packed GEMM allocation generation to actual embedded kernel reads.

Conservative flat-wrapper dataflow only. It rejects escaped pointers, untracked
views and ambiguous control flow. The returned graph does NOT independently
identify Q/K/V semantic consumers or prove launch/mask/index-map correctness.
"""
import ast
from .audit_qkv_outputs import require
from .qkv_generated_matrix import inspect_wrapper
from .qkv_kernel_reads import embedded_kernel,pointer_reads
from .qkv_wrapper_syntax import call_definition,flat_statements


def packed_kernel_dataflow(source):
    matrix=inspect_wrapper(source,True)
    require(matrix is not None,'Complete packed model matrix wrapper required')
    packed={x['line']:x for x in matrix['calls'] if x['role']=='packed_qkv'}
    tree=ast.parse(source);function,_=call_definition(tree)
    live={};owners=[];kernels={}
    def names(node):return {x.id for x in ast.walk(node) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Load)}
    def mentions(node):return names(node)&set(live)
    def kernel(symbol):
        if symbol not in kernels:
            k=embedded_kernel(source,symbol);module=ast.parse(k['source']);defs=[x for x in module.body if isinstance(x,ast.FunctionDef) and x.name==k['function']]
            require(len(defs)==1,'Ambiguous kernel function');k['parameters']=[x.arg for x in defs[0].args.args];kernels[symbol]=k
        return kernels[symbol]
    for statement in flat_statements(function):
        require(not isinstance(statement,(ast.If,ast.For,ast.While,ast.Try,ast.With,ast.Match)),'Control-flow-dependent packed pointer lifetime unsupported')
        calls=[n for n in ast.walk(statement) if isinstance(n,ast.Call)]
        producers=[n for n in calls if n.lineno in packed and isinstance(n.func,ast.Attribute) and ast.unparse(n.func)=='extern_kernels.addmm']
        if producers:
            require(len(producers)==1 and isinstance(statement,ast.Expr) and statement.value is producers[0],'Nested packed producer unsupported')
            node=producers[0];record=packed[node.lineno];out=record['output']
            require(not any(mentions(a) for a in node.args),'Packed projection consumes another packed output')
            owner=dict(layer=len(owners),producer_line=node.lineno,initial_name=out,matrix=record,reads=[])
            # A new GEMM write to an aliased allocation invalidates all aliases
            # of its prior contents, not just the variable used as out=.
            if out in live:
                old=live[out];live={k:v for k,v in live.items() if v!=old}
            owners.append(owner);live[out]=len(owners)-1;continue
        if isinstance(statement,ast.Assign):
            require(not any(mentions(n) for n in calls),'Packed pointer escapes through assigned call/view')
            used=mentions(statement.value)
            if used:
                require(isinstance(statement.value,ast.Name) and len(statement.targets)==1 and isinstance(statement.targets[0],ast.Name),'Only direct packed buffer alias supported')
                live[statement.targets[0].id]=live[statement.value.id]
            else:
                for target in statement.targets:
                    for node in ast.walk(target):
                        if isinstance(node,ast.Name):live.pop(node.id,None)
            continue
        if isinstance(statement,ast.Delete):
            for target in statement.targets:
                require(isinstance(target,ast.Name),'Unsupported packed deletion');live.pop(target.id,None)
            continue
        used=mentions(statement)
        if not used:continue
        require(isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Call),'Packed buffer escapes wrapper expression')
        node=statement.value
        require(isinstance(node.func,ast.Attribute) and node.func.attr=='run' and isinstance(node.func.value,ast.Name),'Packed output used by unrecognized consumer')
        symbol=node.func.value.id;k=kernel(symbol)
        require(len(node.args)<=len(k['parameters']),'Kernel argument count differs')
        require(not any(mentions(keyword.value) for keyword in node.keywords),'Packed pointer in kernel keyword argument')
        for position,arg in enumerate(node.args):
            if not mentions(arg):continue
            require(isinstance(arg,ast.Name) and arg.id in live,'Packed pointer view/offset in launch unsupported')
            ptr=k['parameters'][position];extracted=pointer_reads(k['source'],k['function'],ptr)
            owners[live[arg.id]]['reads'].append(dict(consumer_line=node.lineno,kernel_symbol=symbol,kernel_sha256=k['sha256'],parameter=ptr,parameter_index=position,
                pointer_name=arg.id,load_expressions=extracted['reads'],launch_args=[ast.unparse(a) for a in node.args],launch_keywords={a.arg:ast.unparse(a.value) for a in node.keywords},
                role_bound=False,launch_domain_bound=False))
    require(len(owners)==28 and all(x['reads'] for x in owners),'Every packed projection must have observed generated consumers')
    return dict(schema='packed-qkv-wrapper-dataflow-v1',layers=owners,kernels={name:{k:v for k,v in value.items() if k!='parameters'} for name,value in kernels.items()},
                packed_output_to_kernel_parameter_bound=True,semantic_consumers_verified=False,complete_compiled_view_proof=False)
