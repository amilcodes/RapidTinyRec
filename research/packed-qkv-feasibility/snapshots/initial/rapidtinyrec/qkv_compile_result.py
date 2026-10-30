"""Bind observed compile-result metadata to actual loaded kernel and arguments.

This checks the installed runtime's result association. It does not replay or
independently certify the compiler's transformation from source to machine code.
"""
import ast
import copy


def named_map(mapping,names):
    if type(mapping) is not dict:raise ValueError('Compiler argument map must be a dict')
    result={}
    for key,value in mapping.items():
        if type(key) is tuple and len(key)==1:key=key[0]
        if type(key) is int and 0<=key<len(names):key=names[key]
        if type(key) is not str or key not in names or key in result:raise ValueError('Unknown/duplicate compiler argument')
        if type(value) not in (str,int,bool,type(None)):raise ValueError('Nonprimitive compiler argument metadata')
        result[key]=value
    return result


def snapshot_result(result,kernel):
    if vars(result).get('kernel') is not kernel:raise ValueError('Compile result does not own loaded kernel')
    names=vars(kernel).get('arg_names')
    if type(names) is not list or len(names)!=len(set(names)) or any(type(n) is not str for n in names):raise ValueError('Invalid compiled argument names')
    meta=vars(result).get('compile_meta');config=vars(result).get('config')
    if type(meta) is not dict or meta.get('device_type')!='cuda' or meta.get('cc')!=86:raise ValueError('Expected actual A6000 compiler target')
    kwargs=vars(config).get('kwargs');warps=vars(config).get('num_warps');stages=vars(config).get('num_stages')
    if type(kwargs) is not dict or any(type(k) is not str or type(v) is not int for k,v in kwargs.items()) or type(warps) is not int or type(stages) is not int:raise ValueError('Unsupported actual compiler configuration')
    return dict(result_id=id(result),owner_id=id(kernel),arg_names=list(names),kernel_hash=vars(kernel).get('hash'),
                signature=named_map(meta.get('signature'),names),constants=named_map(meta.get('constants'),names),
                config=dict(kwargs),num_warps=warps,num_stages=stages,device_type='cuda',cc=86)


def observe_result_for_autotuner(autotuner,runner):
    results=vars(autotuner).get('compile_results')
    if type(results) is not list:raise ValueError('No actual compile results')
    matched=[r for r in results if id(vars(r).get('kernel'))==runner['metadata']['owner_id']]
    if len(matched)!=1:raise ValueError('Static runner lacks unique autotuner compile result')
    result=matched[0];snapshot=snapshot_result(result,vars(result)['kernel'])
    source=getattr(vars(autotuner).get('fn'),'src',None)
    if type(source) is not str or not source or len(source)>2000000:raise ValueError('Missing bounded actual JIT source')
    snapshot['jit_source']=source
    return snapshot


def function_body(source,name):
    matches=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name]
    if len(matches)!=1:raise ValueError('Ambiguous JIT function source')
    node=copy.deepcopy(matches[0]);node.decorator_list=[]
    return ast.dump(node,include_attributes=False)


def validate_compile_result(kernel,launch):
    runner=launch['static_runner'];m=runner['metadata'];r=launch['compile_result'];loaded=runner['load_identity']['compile_result']
    if {k:v for k,v in r.items() if k!='jit_source'}!=loaded:raise ValueError('Compile result changed between load and capture')
    if function_body(r['jit_source'],kernel['function'])!=function_body(kernel['source'],kernel['function']):raise ValueError('Actual JIT source differs from captured embedded function')
    if r['owner_id']!=m['owner_id'] or r['kernel_hash']!=m['hash'] or r['arg_names']!=m['arg_names'] or r['config']!=launch['constexpr'] or r['num_warps']!=m['num_warps']:raise ValueError('Compile result/kernel/config identity differs')
    names=r['arg_names'];full=m['full_constexprs'];constants=r['constants'];declared=m['declared_constexprs']
    if set(constants)!={names[i] for i in full}:raise ValueError('Compiler constants do not justify every elided argument')
    observed={**launch['runtime_integers'],**launch['constexpr']}
    for name,value in constants.items():
        if type(value) is not int or name not in observed or type(observed[name]) is not int or value!=observed[name]:raise ValueError('Compiler specialization differs from actual scalar/config argument')
    if not set(names)-set(constants)<=set(r['signature']) or not set(r['signature'])<=set(names):raise ValueError('Compiler signature coverage differs')
    if any(type(v) is not str for v in r['signature'].values()):raise ValueError('Compiler signature types must be strings')
    tensor_types={'torch.bfloat16':'*bf16','torch.float32':'*fp32','torch.float64':'*fp64','torch.float16':'*fp16','torch.int64':'*i64','torch.int32':'*i32','torch.bool':'*i1'}
    for name,kind in r['signature'].items():
        if kind=='constexpr':
            if name not in constants:raise ValueError('Unjustified constexpr signature')
        elif name in launch['runtime_tensors']:
            if tensor_types.get(launch['runtime_tensors'][name]['dtype'])!=kind:raise ValueError('Actual tensor dtype differs from compiler signature')
        elif name in observed:
            if kind not in ('i32','i64') or type(observed[name]) is not int or not -(2**(int(kind[1:])-1))<=observed[name]<2**(int(kind[1:])-1):raise ValueError('Actual scalar differs from compiler signature')
        else:raise ValueError('Compiler argument lacks actual type evidence')
    return dict(runtime_compile_result_and_jit_source_verified=True,
                implicit_elided_names=[names[i] for i in full if i not in declared],implicit_elision_verified=True,
                source_to_binary_compilation_verified=False,complete_dispatch_verified=False)
