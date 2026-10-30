"""Setup-only Python frame observations of actual Inductor run/launcher calls.

Called from the existing temporary profiler. No tensor access or replacement of
kernel dispatch. Unsupported runtime call paths provide no launch proof.
"""
import ast
import hashlib
from pathlib import Path
from .qkv_kernel_reads import embedded_kernel
from .qkv_capture_tensors import snapshot_tensors
from .qkv_runner_binding import observe_runner,attest_runner
from .qkv_compile_result import observe_result_for_autotuner


class LaunchObserver:
    def __init__(self):self.runs={};self.launchers={};self.static_runners={}

    def event(self,frame,event,wrappers):
        if event=='call' and frame.f_code.co_name=='run' and id(frame.f_back) in self.launchers:
            if not frame.f_code.co_filename.endswith('/torch/_inductor/runtime/static_triton_launcher.py'):return
            runner=observe_runner(frame);self.launchers[id(frame.f_back)].setdefault('static_runners',[]).append(runner);self.static_runners[id(frame)]=runner
        elif event=='return' and id(frame) in self.static_runners:
            self.static_runners.pop(id(frame))['returned']=True
        elif event=='call' and frame.f_code.co_name=='run' and id(frame.f_back) in wrappers:
            # Bound Python runtime method, called directly by the generated
            # wrapper. Never infer which object ran from a generated name alone.
            if not frame.f_code.co_filename.endswith('/torch/_inductor/runtime/triton_heuristics.py'):return
            owner=frame.f_locals.get('self');args=frame.f_locals.get('args')
            if owner is None or type(args) is not tuple:raise ValueError('Unsupported runtime run frame')
            record=dict(line=frame.f_back.f_lineno,runtime_file=frame.f_code.co_filename,runtime_code=frame.f_code,
                        kernel_object=owner,globals=frame.f_back.f_globals,argument_count=len(args),stream=frame.f_locals.get('stream'),
                        integers={str(i):v for i,v in enumerate(args) if type(v) is int},tensors=snapshot_tensors(args),launches=[],returned=False)
            wrappers[id(frame.f_back)].setdefault('launch_observations',[]).append(record);self.runs[id(frame)]=record
        elif event=='call' and frame.f_code.co_name=='launcher' and id(frame.f_back) in self.runs:
            run=self.runs[id(frame.f_back)];owner=run['kernel_object']
            choices=getattr(owner,'launchers',None)
            if type(choices) is not list or len(choices)!=1:raise ValueError('Expected one selected runtime launcher')
            selected=getattr(owner,'_cached_launcher',None) or choices[0]
            if getattr(selected,'__code__',None) is not frame.f_code:raise ValueError('Actual launcher differs from selected runtime code')
            config=choices[0].config.kwargs
            if type(config) is not dict or any(type(k) is not str or type(v) is not int for k,v in config.items()):raise ValueError('Unsupported launch config constants')
            launch=dict(code=frame.f_code,config=dict(config),entry_integers={k:v for k,v in frame.f_locals.items() if type(v) is int},returned=False)
            run['launches'].append(launch);self.launchers[id(frame)]=launch
        elif event=='return' and id(frame) in self.launchers:
            launch=self.launchers.pop(id(frame));launch['returned']=True
            launch['grid']=[frame.f_locals.get('grid_'+str(i)) for i in range(3)]
            if any(type(x) is not int or not 1<=x<2**31 for x in launch['grid']):raise ValueError('Missing/invalid actual grid dimensions')
        elif event=='return' and id(frame) in self.runs:
            self.runs.pop(id(frame))['returned']=True


def attest_launches(wrapper_source,observations,fingerprint):
    nodes={n.lineno:n for n in ast.walk(ast.parse(wrapper_source)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='run'}
    result=[]
    for record in observations:
        node=nodes.get(record['line'])
        if node is None or not isinstance(node.func.value,ast.Name):raise ValueError('Observed launch lacks generated call site')
        symbol=node.func.value.id
        if record['globals'].get(symbol) is not record['kernel_object']:raise ValueError('Observed kernel object differs from generated symbol')
        if not record['returned'] or len(record['launches'])!=1 or not record['launches'][0]['returned']:raise ValueError('Expected one completed actual launcher')
        kernel=embedded_kernel(wrapper_source,symbol)
        fn=[n for n in ast.parse(kernel['source']).body if isinstance(n,ast.FunctionDef) and n.name==kernel['function']]
        if len(fn)!=1:raise ValueError('Ambiguous embedded function')
        params=fn[0].args.args;runtime_params=[p.arg for p in params if p.annotation is None or ast.unparse(p.annotation)!='tl.constexpr']
        constants={p.arg for p in params if p.annotation is not None and ast.unparse(p.annotation)=='tl.constexpr'}
        launch=record['launches'][0]
        if len(node.args)!=record['argument_count'] or len(runtime_params)!=len(node.args):raise ValueError('Unsupported runtime/constexpr argument elision')
        if set(launch['config'])!=constants:raise ValueError('Config does not cover exact constexpr parameter set')
        integers={runtime_params[int(i)]:v for i,v in record['integers'].items()}
        for name,value in integers.items():
            if launch['entry_integers'].get(name)!=value:raise ValueError('run/launcher scalar mismatch')
        if type(record['stream']) is not int or launch['entry_integers'].get('stream')!=record['stream']:raise ValueError('Runtime/launcher stream differs')
        runner=attest_runner(launch.get('static_runners',[]),fingerprint)
        compile_result=observe_result_for_autotuner(record['kernel_object'],runner)
        runtime_path=Path(record['runtime_file'])
        result.append(dict(line=node.lineno,kernel_symbol=symbol,kernel_sha256=kernel['sha256'],
                           runtime_source_sha256=hashlib.sha256(runtime_path.read_bytes()).hexdigest(),runtime_code_sha256=fingerprint(record['runtime_code']),
                           launcher_code_sha256=fingerprint(launch['code']),runtime_integers=integers,
                           runtime_tensors={runtime_params[int(i)]:v for i,v in record['tensors'].items()},
                           constexpr=launch['config'],actual_grid=launch['grid'],actual_stream=record['stream'],static_runner=runner,compile_result=compile_result,
                           observed_run_to_launcher=True,grid_formula_verified=False,complete_compiled_view_proof=False))
    return result
