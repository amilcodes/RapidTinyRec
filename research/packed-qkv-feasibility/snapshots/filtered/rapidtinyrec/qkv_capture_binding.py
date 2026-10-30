"""Observe actual generated Python call entries during untimed graph capture.

The original replay prepare implementation runs unchanged. Profiling is removed
before request measurement; code fingerprints bind executed call bytecode to the
copied generated source. This does not prove fused Q/K/V index expressions.
"""
import ast
import contextlib
import hashlib
import json
import sys
import types
from pathlib import Path
from .qkv_wrapper_syntax import call_definition
from .qkv_launch_observer import LaunchObserver,attest_launches
from .qkv_capture_tensors import snapshot_tensors


def code_fingerprint(code):
    def constant(x):
        if isinstance(x,types.CodeType):return {'code':payload(x)}
        if isinstance(x,tuple):return {'tuple':[constant(v) for v in x]}
        if isinstance(x,frozenset):return {'frozenset':sorted([constant(v) for v in x],key=lambda v:json.dumps(v,sort_keys=True))}
        return {'type':type(x).__name__,'repr':repr(x)}
    def payload(c):
        return dict(bytecode=c.co_code.hex(),constants=[constant(x) for x in c.co_consts],names=c.co_names,
                    variables=c.co_varnames,free=c.co_freevars,cells=c.co_cellvars,args=c.co_argcount,
                    positional_only=c.co_posonlyargcount,keyword_only=c.co_kwonlyargcount,flags=c.co_flags,
                    stack=c.co_stacksize,exception_table=c.co_exceptiontable.hex())
    return hashlib.sha256(json.dumps(payload(code),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def source_call_fingerprint(data,filename):
    # Compile syntax to code objects only; never execute imported/generated code.
    module=compile(data,filename,'exec',dont_inherit=True)
    _,owner=call_definition(ast.parse(data))
    scope=module
    if owner:
        classes=[c for c in module.co_consts if isinstance(c,types.CodeType) and c.co_name==owner]
        if len(classes)!=1:raise ValueError('Ambiguous generated code owner')
        scope=classes[0]
    functions=[c for c in scope.co_consts if isinstance(c,types.CodeType) and c.co_name=='call']
    if len(functions)!=1:raise ValueError('Expected one generated call function')
    return code_fingerprint(functions[0])


@contextlib.contextmanager
def observe_generated_capture(cache_roots,is_capturing,load_observer=None):
    if sys.getprofile() is not None:raise ValueError('Refuse to replace another profiler')
    roots={name:Path(path).resolve(strict=True) for name,path in cache_roots.items()}
    observed=[];active={};launch_observer=LaunchObserver()
    def profile(frame,event,arg):
        if event not in ('call','return'):return
        if frame.f_code.co_name not in ('call','load_kernel','run','launcher'):return
        if load_observer is not None:load_observer.event(frame,event,code_fingerprint)
        launch_observer.event(frame,event,active)
        if event=='return' and id(frame) in active:
            record=active.pop(id(frame))
            record['scalar_observation']['return_integers']={k:v for k,v in frame.f_locals.items() if type(v) is int}
            record['scalar_observation']['returned']=True
            record['return_tensors']=snapshot_tensors(arg)
            return
        if event!='call' or frame.f_code.co_name!='call':return
        name=frame.f_code.co_filename
        # No file IO/hash/compile or tensor data operations during capture.
        # CPU tensor metadata is snapshotted without retaining tensor objects.
        if name.startswith('<') or not any(name.startswith(str(p)+'/') for p in roots.values()):return
        if is_capturing():
            boxed=frame.f_locals.get('args')
            scalar=dict(schema='capture-python-integers-v1',boxed_kind=type(boxed).__name__,
                        boxed_count=len(boxed) if type(boxed) in (list,tuple) else None,
                        entry_integers={str(i):v for i,v in enumerate(boxed) if type(v) is int} if type(boxed) in (list,tuple) else {},
                        return_integers={},returned=False)
            record=dict(filename=name,code=frame.f_code,scalar_observation=scalar,boxed_tensors=snapshot_tensors(boxed))
            observed.append(record);active[id(frame)]=record
    sys.setprofile(profile)
    try:yield observed
    finally:sys.setprofile(None)


def attest_observed(observed,cache_roots):
    roots={name:Path(path).resolve(strict=True) for name,path in cache_roots.items()};result=[]
    for record in observed:
        filename,code=record['filename'],record['code']
        if not record['scalar_observation']['returned']:raise ValueError('Observed wrapper did not return')
        path=Path(filename);resolved=path.resolve(strict=True)
        matches=[name for name,root in roots.items() if resolved.is_relative_to(root)]
        if len(matches)!=1 or path.is_symlink() or path!=resolved:raise ValueError('Ambiguous or noncanonical generated source path')
        data=path.read_bytes();fingerprint=code_fingerprint(code)
        if source_call_fingerprint(data,str(path))!=fingerprint:raise ValueError('Executed generated call differs from saved source')
        result.append(dict(cache=matches[0],relative=str(path.relative_to(roots[matches[0]])),sha256=hashlib.sha256(data).hexdigest(),call_code_sha256=fingerprint,scalar_observation=record['scalar_observation'],boxed_tensors=record['boxed_tensors'],return_tensors=record['return_tensors'],launch_observations=attest_launches(data.decode(),record.get('launch_observations',[]),code_fingerprint)))
    if not result:raise ValueError('No actual generated wrapper observed during capture')
    return result
