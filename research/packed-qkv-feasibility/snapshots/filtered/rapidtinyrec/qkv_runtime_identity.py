"""Independently read installed runtime source; never execute saved source.

The observer's source path is not trusted. The allocated auditor locates the
runtime through its installed torch package and compares compiled code objects.
Canonical static 1D launcher checking is intentionally narrower than all Triton
launchers. Unsupported launch conventions fail closed for requested proofs.
"""
import __future__
import hashlib
import types
from pathlib import Path
from .qkv_capture_binding import code_fingerprint


def source_method(data,owner,name):
    module=compile(data,'<installed-runtime>','exec',dont_inherit=True)
    classes=[c for c in module.co_consts if isinstance(c,types.CodeType) and c.co_name==owner]
    if len(classes)!=1:raise ValueError('Missing/ambiguous installed runtime class')
    methods=[c for c in classes[0].co_consts if isinstance(c,types.CodeType) and c.co_name==name]
    if len(methods)!=1:raise ValueError('Missing/ambiguous installed runtime method')
    return code_fingerprint(methods[0])


def inspect_runtime_source(data):
    # Independent auditor snapshot, not a producer supplied digest or pathname.
    return dict(schema='qkv-installed-runtime-v1',source_sha256=hashlib.sha256(data).hexdigest(),
                run_code_sha256=source_method(data,'CachingAutotuner','run'),
                generator_code_sha256=source_method(data,'CompileResult','_gen_launcher_code'),
                maker_code_sha256=source_method(data,'StaticTritonCompileResult','make_launcher'))


def observe_installed_runtime(output_root=None):
    import torch
    package=Path(torch.__file__).resolve(strict=True).parent
    path=package/'_inductor/runtime/triton_heuristics.py'
    if path.is_symlink() or not path.is_file():raise ValueError('Invalid installed runtime source')
    data=path.read_bytes();result=inspect_runtime_source(data)
    result.update(path=str(path),torch_version=torch.__version__)
    if output_root is not None:
        folder=Path(output_root)/'runtime-source';folder.mkdir(exist_ok=True)
        if folder.is_symlink():raise ValueError('Runtime artifact directory is a symlink')
        artifact=folder/(result['source_sha256']+'.py')
        if artifact.exists() or artifact.is_symlink():
            if artifact.is_symlink() or artifact.read_bytes()!=data:raise ValueError('Runtime artifact changed')
        else:
            with artifact.open('xb') as stream:stream.write(data)
        result['source_artifact']=str(artifact.relative_to(output_root))
    static_path=package/'_inductor/runtime/static_triton_launcher.py'
    if static_path.is_symlink() or not static_path.is_file():raise ValueError('Invalid installed static runner source')
    static_data=static_path.read_bytes()
    result['static_runner']=dict(source_sha256=hashlib.sha256(static_data).hexdigest(),code_sha256=source_method(static_data,'StaticallyLaunchedTritonKernel','run'),load_code_sha256=source_method(static_data,'StaticallyLaunchedTritonKernel','load_kernel'))
    if output_root is not None:
        artifact=folder/(result['static_runner']['source_sha256']+'.py')
        if artifact.exists() or artifact.is_symlink():
            if artifact.is_symlink() or artifact.read_bytes()!=static_data:raise ValueError('Static runtime artifact changed')
        else:
            with artifact.open('xb') as stream:stream.write(static_data)
        result['static_runner']['source_artifact']=str(artifact.relative_to(output_root))
    return result


def validate_runtime_calls(report,installed):
    if installed['schema']!='qkv-installed-runtime-v1':raise ValueError('Independent installed runtime observation required')
    count=0
    for graph in report['graphs']:
        for call in graph['wrappers']:
            for launch in call['launch_observations']:
                if launch['runtime_source_sha256']!=installed['source_sha256'] or launch['runtime_code_sha256']!=installed['run_code_sha256']:
                    raise ValueError('Captured runtime differs from independently installed source/code')
                runner=launch['static_runner']
                if runner['source_sha256']!=installed['static_runner']['source_sha256'] or runner['code_sha256']!=installed['static_runner']['code_sha256']:
                    raise ValueError('Static runner differs from independently installed source/code')
                count+=1
    if count==0:raise ValueError('No runtime calls to corroborate')
    return dict(runtime_calls=count,installed_source_and_run_code_verified=True,
                launcher_forwarding_verified=False,compiled_binary_identity_verified=False,
                complete_dispatch_verified=False)


def validate_static_pointwise_launcher(record,parameters):
    """Compile-only bytecode comparison to independent restricted forwarding.

    Caller must bind parameters to the embedded kernel and independently prove
    grid/domain. This does not authenticate the runner global or compiled binary.
    """
    if not parameters or len(parameters)!=len(set(parameters)) or any(type(p) is not str or not p.isidentifier() or p in ('stream','grid_0','grid_1','grid_2','runner') for p in parameters):
        raise ValueError('Unsupported launcher parameters')
    if 'xnumel' not in parameters or set(record['constexpr'])!={'XBLOCK'}:raise ValueError('Only canonical pointwise launcher supported')
    block=record['constexpr']['XBLOCK']
    if type(block) is not int or not 1<block<=1024 or block&(block-1):raise ValueError('Invalid launcher block')
    source='def launcher('+', '.join(parameters)+', stream):\n'
    source+=f'    grid_0 = -((xnumel) // -({block}))\n    grid_1 = 1\n    grid_2 = 1\n'
    source+='    runner('+', '.join(['grid_0','grid_1','grid_2','stream',*parameters])+')\n'
    # exec in the pinned runtime inherits its annotations future flag.
    code=compile(source,'<string>','exec',flags=__future__.annotations.compiler_flag,dont_inherit=True)
    fn=next(c for c in code.co_consts if isinstance(c,types.CodeType) and c.co_name=='launcher')
    if code_fingerprint(fn)!=record['launcher_code_sha256']:raise ValueError('Launcher grid/argument forwarding differs from canonical static pointwise path')
    return dict(canonical_launcher_code_verified=True,compiled_binary_identity_verified=False)
