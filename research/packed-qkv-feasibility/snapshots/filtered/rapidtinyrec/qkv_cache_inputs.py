"""Bind independently inventoried graph KV inputs to actual wrapper/run slots."""
import ast
from .qkv_wrapper_syntax import call_definition,flat_statements
from .qkv_kernel_reads import embedded_kernel


def bind_cache_inputs(source,call,graph):
    fn,_=call_definition(ast.parse(source));statements=list(flat_statements(fn));unpack=statements[0]
    if not isinstance(unpack,ast.Assign) or len(unpack.targets)!=1 or not isinstance(unpack.targets[0],(ast.Tuple,ast.List)) or ast.unparse(unpack.value)!='args':raise ValueError('Missing boxed unpack')
    names=unpack.targets[0].elts;count=call['scalar_observation']['boxed_count']
    if len(names)!=count or any(not isinstance(n,ast.Name) for n in names):raise ValueError('Boxed source/runtime count differs')
    boxed=call['boxed_tensors']
    if any(not k.isdecimal() or str(int(k))!=k or not 0<=int(k)<count for k in boxed):raise ValueError('Invalid boxed tensor position')
    live={};inputs=[]
    for family in ('prefixes','suffixes'):
        if len(graph[family])!=28 or any(len(pair)!=2 for pair in graph[family]):raise ValueError('Complete cache family required')
        for layer,pair in enumerate(graph[family]):
            for role,record in zip(('k','v'),pair):
                hits=[int(k) for k,v in boxed.items() if v==record]
                if len(hits)!=1:raise ValueError('Cache input is missing or ambiguous in captured boxed arguments')
                slot=hits[0];name=names[slot].id
                if name in live:raise ValueError('Graph cache owners overlap')
                item=dict(family=family,layer=layer,role=role,slot=slot,name=name,metadata=record)
                live[name]=len(inputs);inputs.append(item)
    observations={r['line']:r for r in call['launch_observations']};uses=[]
    for statement in statements[1:]:
        for node in (x for x in ast.walk(statement) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='run' and isinstance(x.func.value,ast.Name)):
            k=embedded_kernel(source,node.func.value.id);f=next(x for x in ast.parse(k['source']).body if isinstance(x,ast.FunctionDef) and x.name==k['function'])
            ps=[p.arg for p in f.args.args if p.annotation is None or ast.unparse(p.annotation)!='tl.constexpr'];obs=observations[node.lineno]
            for pos,arg in enumerate(node.args):
                if isinstance(arg,ast.Name) and arg.id in live:
                    item=inputs[live[arg.id]];param=ps[pos]
                    if obs['runtime_tensors'].get(param)!=item['metadata']:raise ValueError('Actual runtime cache pointer/layout differs from captured graph input')
                    uses.append(dict(input_index=live[arg.id],line=node.lineno,kernel=node.func.value.id,parameter=param))
                elif any(isinstance(x,ast.Name) and x.id in live for x in ast.walk(arg)):raise ValueError('Unsupported cache input launch view')
        if isinstance(statement,ast.Assign):
            target=statement.targets[0] if len(statement.targets)==1 else None;v=statement.value
            source_name=v.id if isinstance(v,ast.Name) else v.args[0].id if isinstance(v,ast.Call) and ast.unparse(v.func)=='copy_if_misaligned' and len(v.args)==1 and isinstance(v.args[0],ast.Name) else None
            prior=live.get(source_name)
            for t in statement.targets:
                for n in ast.walk(t):
                    if isinstance(n,ast.Name):live.pop(n.id,None)
            if prior is not None:
                if not isinstance(target,ast.Name):raise ValueError('Ambiguous cache alias')
                live[target.id]=prior
        elif isinstance(statement,ast.Delete):
            for n in statement.targets:
                if isinstance(n,ast.Name):live.pop(n.id,None)
    if {u['input_index'] for u in uses}!=set(range(112)):raise ValueError('Cache input lacks an actual observed launch binding')
    return dict(inputs=inputs,uses=uses,actual_graph_cache_inputs_bound=True,prefix_suffix_copy_verified=False,complete_compiled_view_proof=False)


def bind_cache_returns(source,call,graph):
    fn,_=call_definition(ast.parse(source));returns=[n for n in flat_statements(fn) if isinstance(n,ast.Return)]
    if len(returns)!=1 or not isinstance(returns[0].value,ast.Tuple):raise ValueError('Expected one flat generated return')
    values=returns[0].value.elts;actual=call['return_tensors']
    if len(values)!=57 or set(actual)!={str(i) for i in range(57)}:raise ValueError('Expected exactly56 generated suffix outputs plus logits')
    if graph['output_prefixes']!=graph['prefixes']:raise ValueError('Prefix alias identity changed outside generated return')
    results=[];seen=set()
    if len(graph['output_suffixes'])!=28 or any(len(p)!=2 for p in graph['output_suffixes']):raise ValueError('Missing graph output suffix family')
    for layer,pair in enumerate(graph['output_suffixes']):
        for role,record in zip(('k','v'),pair):
            matches=[int(k) for k,v in actual.items() if v==record]
            if len(matches)!=1 or matches[0] in seen:raise ValueError('Returned cache owner missing/ambiguous')
            pos=matches[0];seen.add(pos)
            if not isinstance(values[pos],ast.Name):raise ValueError('Unsupported cache return view')
            results.append(dict(layer=layer,role=role,slot=pos,name=values[pos].id,metadata=record))
    logits=[int(k) for k,v in actual.items() if v==graph['logits']]
    if len(logits)!=1 or logits[0] in seen or seen|set(logits)!=set(range(57)):raise ValueError('Actual logits return does not complete exact output inventory')
    return dict(suffixes=results,logits_slot=logits[0],actual_graph_cache_returns_bound=True,
                prefix_suffix_copy_verified=False,complete_compiled_view_proof=False)
