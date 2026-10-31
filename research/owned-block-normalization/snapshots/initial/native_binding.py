"""Pinned native entry and runtime metadata gates; never an execution certificate.

No CUDA work is launched here. Full allocated publication/source-lock binding is
still required; a matching version/dispatch entry is not a kernel execution proof.
"""
import hashlib
from pathlib import Path
import types
import torch

FUNCTIONAL_SHA256='95ff403085bb179477a01df63acfddf59dfac0fb58829e99a9c8b5c2fe98899b'
TORCH_GIT='08187d9e0fba026dc8217405802ab5381dc88d90'
NATIVE_SCHEMA='aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor'
SOURCE_FILES=('owned_blocks.py','compare_intervention.py','graph_intervention.py',
              'produce_cases.py','audit_evidence.py','audit_inputs.py','native_binding.py','audit_live.py','loaded_binding.py','publication.py',
              'allocation_contract.py','allocation.py','bounded_runner.py','final_audit.py','whole_run.py','guarded_whole.py','terminal_metadata.py','allocated_entry.py')


def require(value,message):
    if not value:raise ValueError(message)


def _code_key(code):
    return (code.co_code,tuple(_code_key(c) if isinstance(c,types.CodeType) else c for c in code.co_consts),
            code.co_names,code.co_varnames,code.co_argcount,code.co_posonlyargcount,
            code.co_kwonlyargcount,code.co_flags,code.co_freevars,code.co_cellvars,
            getattr(code,'co_exceptiontable',b''))


def verify_source_manifest(root,expected):
    """Expected hashes must come from the externally reviewed immutable lock."""
    root=Path(root).resolve()
    require(set(expected)==set(SOURCE_FILES),'Exact production source inventory required')
    observed={}
    for name in SOURCE_FILES:
        p=root/name
        require(p.is_file() and not p.is_symlink(),'Source must be a regular non-symlink file')
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        require(type(expected[name]) is str and digest==expected[name],'Reviewed source hash mismatch: '+name)
        observed[name]=digest
    return observed


def verify_native_entry():
    f=torch.nn.functional.log_softmax
    require(type(f) is types.FunctionType,'Native functional entry replaced')
    source_path=Path(torch.nn.functional.__file__).resolve()
    raw=source_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==FUNCTIONAL_SHA256,'Pinned functional source mismatch')
    # Compile the entire module but never execute its imports or definitions.
    module_code=compile(raw,str(source_path),'exec',dont_inherit=True)
    codes=[c for c in module_code.co_consts if isinstance(c,types.CodeType) and c.co_name=='log_softmax']
    require(len(codes)==1 and _code_key(f.__code__)==_code_key(codes[0]),'Loaded functional code mismatch')
    require(f.__globals__ is vars(torch.nn.functional),'Functional globals replaced')
    require(f.__defaults__==(None,3,None) and f.__kwdefaults__ is None,'Functional defaults changed')
    require(f.__globals__['has_torch_function_unary'] is torch._C._has_torch_function_unary,
            'Tensor override predicate replaced')
    descriptor=torch.Tensor.log_softmax
    require(type(descriptor) is types.MethodDescriptorType and
            descriptor is torch._C.TensorBase.log_softmax and
            descriptor.__objclass__ is torch._C.TensorBase and descriptor.__name__=='log_softmax',
            'Tensor normalization method replaced')
    from torch.utils._python_dispatch import _get_current_dispatch_mode
    from torch.overrides import _get_current_function_mode
    require(_get_current_dispatch_mode() is None and _get_current_function_mode() is None,
            'Active tensor override mode forbidden')
    schema=str(torch.ops.aten._log_softmax.default._schema)
    require(schema==NATIVE_SCHEMA,'Native normalization schema mismatch')
    return dict(functional_sha256=FUNCTIONAL_SHA256,loaded_functional_code_matches=True,
                tensor_method_native_descriptor=True,override_modes_absent=True,
                native_schema=schema,execution_certified=False)


def collect_runtime():
    require(torch.cuda.is_available(),'CUDA runtime required; CPU evidence cannot certify allocation')
    device=torch.cuda.current_device()
    prop=torch.cuda.get_device_properties(device)
    return dict(torch_version=str(torch.__version__),torch_git=torch.version.git_version,
                cuda_version=torch.version.cuda,device_count=torch.cuda.device_count(),
                current_device=device,device_name=prop.name,capability=list(torch.cuda.get_device_capability(device)),
                total_memory_bytes=prop.total_memory,num_threads=torch.get_num_threads(),
                cuda_normalization_registered=torch._C._dispatch_has_kernel_for_dispatch_key('aten::_log_softmax','CUDA'),
                capture_active=torch.cuda.is_current_stream_capturing())


def validate_runtime(record):
    expected=dict(torch_version='2.14.0+cu126',torch_git=TORCH_GIT,cuda_version='12.6',
                  device_count=1,current_device=0,device_name='NVIDIA RTX A6000',capability=[8,6],
                  num_threads=4,cuda_normalization_registered=True,capture_active=False)
    require(set(record)==set(expected)|{'total_memory_bytes'},'Exact runtime metadata required')
    for k,v in expected.items():
        require(type(record[k]) is type(v) and record[k]==v,'Runtime mismatch: '+k)
    require(all(type(x) is int for x in record['capability']),'Integer capability required')
    memory=record['total_memory_bytes']
    require(type(memory) is int and 47*2**30<=memory<=49*2**30,'Regular48GiB A6000 memory required')
    return dict(runtime_metadata_passed=True,native_cuda_registration_present=True,
                execution_certified=False,submission_ready=False)


if __name__=='__main__':
    raise SystemExit('Disabled: allocated source/publication/audit/closure/readiness integration pending')
