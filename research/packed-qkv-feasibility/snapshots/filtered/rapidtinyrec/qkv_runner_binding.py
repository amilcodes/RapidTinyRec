"""Observe static Python runner identity and argument forwarding during setup.

No tensor objects are retained, no file reads or hashing occur in capture.
Loaded function/module handles identify an object, not its original cubin bytes.
"""
import ast
import hashlib
import types
from pathlib import Path
from .qkv_capture_tensors import snapshot_tensors


def observe_runner(frame):
    selected=frame.f_back.f_globals.get('runner');owner=frame.f_locals.get('self')
    if not isinstance(selected,types.MethodType) or selected.__self__ is not owner or selected.__func__.__code__ is not frame.f_code:
        raise ValueError('Actual runner differs from launcher global bound method')
    args=frame.f_locals.get('args')
    if type(args) is not tuple:raise ValueError('Unsupported runner argument tuple')
    values={'owner_id':id(owner)}
    for key in ('name','hash','arg_tys','function','module','num_warps','shared','device_agnostic','_has_tensordesc'):
        value=vars(owner).get(key)
        if type(value) not in (str,int,bool):raise ValueError('Missing primitive static runner metadata: '+key)
        values[key]=value
    for key in ('arg_names','declared_constexprs','full_constexprs'):
        value=vars(owner).get(key)
        if type(value) is not list or any(type(v) is not (str if key=='arg_names' else int) for v in value):raise ValueError('Unsupported runner argument metadata')
        values[key]=list(value)
    if values['device_agnostic'] or values['_has_tensordesc'] or type(values['function']) is not int or values['function']<=0 or type(values['module']) is not int or values['module']<=0:
        raise ValueError('Unsupported device-agnostic/TMA/unloaded runner')
    grid=[frame.f_locals.get(k) for k in ('grid_x','grid_y','grid_z')];stream=frame.f_locals.get('stream')
    if any(type(v) is not int or v<=0 for v in grid) or type(stream) is not int:raise ValueError('Invalid actual runner grid/stream')
    return dict(metadata=values,grid=grid,stream=stream,argument_count=len(args),
                integers={str(i):v for i,v in enumerate(args) if type(v) is int},tensors=snapshot_tensors(args),
                code=frame.f_code,source_file=frame.f_code.co_filename,returned=False)


def attest_runner(records,fingerprint):
    if len(records)!=1 or records[0]['returned'] is not True:raise ValueError('One completed actual static runner required')
    r=records[0];result={k:v for k,v in r.items() if k not in ('code','source_file')}
    result.update(source_sha256=hashlib.sha256(Path(r['source_file']).read_bytes()).hexdigest(),code_sha256=fingerprint(r['code']),launcher_global_bound=True)
    return result


def validate_runner_forwarding(kernel,launch):
    r=launch['static_runner'];m=r['metadata']
    fn=next(n for n in ast.parse(kernel['source']).body if isinstance(n,ast.FunctionDef) and n.name==kernel['function'])
    names=[a.arg for a in fn.args.args];declared=[i for i,a in enumerate(fn.args.args) if a.annotation is not None and ast.unparse(a.annotation)=='tl.constexpr']
    full=m['full_constexprs']
    if m['name']!=kernel['function'] or m['arg_names']!=names or m['declared_constexprs']!=declared or len(full)!=len(set(full)) or any(type(i) is not int or not 0<=i<len(names) for i in full) or not set(declared)<=set(full):raise ValueError('Static runner signature differs from embedded source')
    if r['launcher_global_bound'] is not True or r['returned'] is not True or r['grid']!=launch['actual_grid'] or r['stream']!=launch['actual_stream']:raise ValueError('Runner call completion/grid differs')
    forwarded=[n for i,n in enumerate(names) if i not in full]
    if len(forwarded)!=r['argument_count'] or len(m['arg_tys'])!=len(forwarded):raise ValueError('Runner argument count/type signature differs')
    for i,name in enumerate(forwarded):
        key=str(i)
        if name in launch['runtime_integers']:
            if r['integers'].get(key)!=launch['runtime_integers'][name] or key in r['tensors']:raise ValueError('Runner scalar forwarding changed')
        elif name in launch['runtime_tensors']:
            if r['tensors'].get(key)!=launch['runtime_tensors'][name] or key in r['integers']:raise ValueError('Runner tensor forwarding changed')
        else:raise ValueError('Unclassified actual runner argument')
    if set(r['integers'])|set(r['tensors'])!={str(i) for i in range(len(forwarded))}:raise ValueError('Runner argument snapshot coverage differs')
    implicit=[names[i] for i in full if i not in declared]
    return dict(actual_launcher_global_and_forwarded_arguments_bound=True,forwarded_names=forwarded,
                implicit_elided_names=implicit,implicit_elision_verified=not implicit,
                loaded_function=m['function'],loaded_module=m['module'],kernel_hash=m['hash'],
                compiled_binary_identity_verified=False,complete_dispatch_verified=False)
