"""Conservative source data dependencies ending at actual SDPA Q/K/V positions.

Positions come from the pinned attention operator, never from variable names or
column offsets. This proves routes only, not masks, numerical expressions or
complete writes; independent address/domain and output-bit gates remain needed.
"""
import ast
from .audit_qkv_outputs import require
from .qkv_kernel_reads import embedded_kernel
from .qkv_index_expression import scalar_assignments
from .qkv_wrapper_syntax import call_definition,flat_statements

ATTENTION='torch.ops.aten._scaled_dot_product_flash_attention.default'


def kernel_stores(source,function_name):
    fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==function_name)
    assignments=scalar_assignments(source,function_name);parameters=[a.arg for a in fn.args.args]
    active=set();cache={}
    def pointer(address):
        base=address.left if isinstance(address,ast.BinOp) and isinstance(address.op,ast.Add) else address
        require(isinstance(base,ast.Name) and base.id in parameters,'Unsupported kernel memory base')
        return base.id
    def loads(node):
        if isinstance(node,ast.Name) and node.id in assignments:
            require(node.id not in active,'Cyclic kernel value')
            if node.id in cache:return cache[node.id]
            active.add(node.id)
            try:result=loads(assignments[node.id]);cache[node.id]=result;return result
            finally:active.remove(node.id)
        if isinstance(node,ast.Call) and ast.unparse(node.func)=='tl.load':return {(pointer(node.args[0]),node.lineno)}
        result=set()
        for child in ast.iter_child_nodes(node):result|=loads(child)
        return result
    stores=[]
    for node in ast.walk(fn):
        if isinstance(node,ast.Call):
            name=ast.unparse(node.func)
            require(not name.startswith('tl.atomic_'),'Atomic dataflow unsupported')
            if name=='tl.store':
                require(2<=len(node.args)<=3,'Unexpected store arguments')
                stores.append(dict(pointer=pointer(node.args[0]),line=node.lineno,dependencies=sorted(loads(node.args[1])),
                                   address=ast.unparse(node.args[0]),mask=ast.unparse(node.args[2]) if len(node.args)==3 else next((ast.unparse(k.value) for k in node.keywords if k.arg=='mask'),'True')))
    require(bool(stores),'No kernel stores for semantic routing')
    return parameters,stores


def bind_attention_routes(source,packed_dataflow):
    fn,_=call_definition(ast.parse(source));live={};content={};serial=0;routes={};attention=[];kernels={}
    producers={p['producer_line']:p['layer'] for p in packed_dataflow['layers']}
    wanted={}
    for layer in packed_dataflow['layers']:
        for consumer in layer['reads']:
            for load in consumer['load_expressions']:
                key=(layer['layer'],consumer['consumer_line'],consumer['parameter'],load['line'])
                require(key not in wanted,'Duplicate packed read identity');wanted[key]=dict(layer=layer['layer'],consumer_line=consumer['consumer_line'],pointer=consumer['parameter'],load_line=load['line'])
    def fresh(name,value=None):
        nonlocal serial
        serial+=1;live[name]=serial;content[serial]=value if value is not None else ('taints',set())
    def value(name):return content[live[name]] if name in live else ('taints',set())
    for statement in flat_statements(fn):
        require(not isinstance(statement,(ast.If,ast.For,ast.While,ast.Try,ast.With,ast.Match)),'Ambiguous wrapper control flow')
        if isinstance(statement,ast.Delete):
            for t in statement.targets:
                require(isinstance(t,ast.Name),'Unsupported deletion');live.pop(t.id,None)
            continue
        if isinstance(statement,ast.Assign) and len(statement.targets)==1 and isinstance(statement.targets[0],ast.Name):
            name=statement.targets[0].id;expr=statement.value
            if isinstance(expr,ast.Call) and ast.unparse(expr.func)==ATTENTION:
                require(len(expr.args)==3 and all(isinstance(a,ast.Name) for a in expr.args),'Expected direct Q/K/V attention arguments')
                require(len(expr.keywords)==1 and expr.keywords[0].arg=='scale' and ast.literal_eval(expr.keywords[0].value)==128**-0.5,'Attention scale/options changed')
                roles={}
                for role,arg in zip(('q','k','v'),expr.args):
                    kind,taints=value(arg.id);require(kind=='taints' and bool(taints),'Attention input lacks packed ancestry')
                    layers={x[0] for x in taints};require(len(layers)==1,'Attention input mixes layers')
                    roles[role]=dict(buffer=arg.id,layer=next(iter(layers)),reads=sorted(taints))
                    for key in taints:routes.setdefault(key,set()).add((len(attention),role))
                require(len({x['layer'] for x in roles.values()})==1,'Attention Q/K/V layer mismatch')
                attention.append(dict(line=expr.lineno,positions=roles));fresh(name);continue
            if isinstance(expr,ast.Name) and expr.id in live:live[name]=live[expr.id]
            elif isinstance(expr,ast.Call) and isinstance(expr.func,ast.Name) and expr.func.id=='reinterpret_tensor' and isinstance(expr.args[0],ast.Name) and expr.args[0].id in live:
                require(len(expr.args)==4 and ast.literal_eval(expr.args[3])==0,'Offset view semantic route unsupported');live[name]=live[expr.args[0].id]
            else:fresh(name)
            continue
        if not isinstance(statement,ast.Expr) or not isinstance(statement.value,ast.Call):continue
        call=statement.value;name=ast.unparse(call.func)
        if call.lineno in producers:
            out=next(k.value for k in call.keywords if k.arg=='out');require(isinstance(out,ast.Name),'Packed output unknown')
            if out.id not in live:fresh(out.id)
            content[live[out.id]]=('packed',producers[call.lineno]);continue
        if isinstance(call.func,ast.Attribute) and call.func.attr=='run' and isinstance(call.func.value,ast.Name):
            relevant=[value(a.id) for a in call.args if isinstance(a,ast.Name)]
            if not any(kind=='packed' or bool(v) for kind,v in relevant):continue
            symbol=call.func.value.id
            if symbol not in kernels:
                embedded=embedded_kernel(source,symbol);kernels[symbol]=kernel_stores(embedded['source'],embedded['function'])
            params,stores=kernels[symbol];args={param:arg.id for param,arg in zip(params,call.args) if isinstance(arg,ast.Name)}
            writes={}
            for store in stores:
                out=args.get(store['pointer']);require(out is not None and out in live,'Kernel store output has no supplied allocation')
                taints=set()
                for parameter,line in store['dependencies']:
                    variable=args.get(parameter)
                    if variable is None:continue
                    kind,prior=value(variable)
                    if kind=='packed':
                        key=(prior,call.lineno,parameter,line);require(key in wanted,'Untracked packed pointer load');taints.add(key)
                    else:taints|=prior
                writes.setdefault(live[out],set()).update(taints)
            for owner,taints in writes.items():content[owner]=('taints',taints)
        elif name in ('extern_kernels.mm','extern_kernels.addmm'):
            for k in call.keywords:
                if k.arg=='out' and isinstance(k.value,ast.Name) and k.value.id in live:content[live[k.value.id]]=('taints',set())
    require(len(attention)==28 and [a['positions']['q']['layer'] for a in attention]==list(range(28)),'Expected 28 ordered model attention layers')
    require(set(routes)==set(wanted),'Packed read missing from attention consumer routes')
    bound=[]
    for key in sorted(wanted):
        destinations=routes[key];require(len(destinations)==1,'Packed read reaches multiple semantic roles')
        layer,role=next(iter(destinations));require(layer==key[0],'Packed read reaches wrong attention layer')
        bound.append(dict(**wanted[key],attention_layer=layer,role=role))
    require(all({r['role'] for r in bound if r['layer']==i}=={'q','k','v'} for i in range(28)),'Missing Q/K/V role route')
    return dict(schema='packed-qkv-semantic-routes-v1',attention=attention,reads=bound,
                semantic_attention_positions_bound=True,launch_domain_verified=False,store_mask_coverage_verified=False,complete_compiled_view_proof=False)
