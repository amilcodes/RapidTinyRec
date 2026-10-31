"""Bind loaded local Python definitions/imports/defaults to reviewed files.

Compile source without executing it. This protects the current plain-function
primitive package; unsupported decorated definitions fail closed.
"""
import ast
import importlib
from pathlib import Path
import types
from native_binding import verify_source_manifest,_code_key


def _value(node,namespace):
    if isinstance(node,ast.Name):return namespace[node.id]
    if isinstance(node,ast.Attribute):return getattr(_value(node.value,namespace),node.attr)
    return ast.literal_eval(node)


def _function(node,code,actual,namespace,label):
    if node.decorator_list or type(actual) is not types.FunctionType:
        raise ValueError('Unsupported/replaced function: '+label)
    if _code_key(code)!=_code_key(actual.__code__) or actual.__globals__ is not namespace or actual.__closure__:
        raise ValueError('Loaded code/globals differ: '+label)
    defaults=tuple(_value(n,namespace) for n in node.args.defaults) or None
    keywords={n.arg:_value(v,namespace) for n,v in zip(node.args.kwonlyargs,node.args.kw_defaults) if v is not None} or None
    if actual.__defaults__!=defaults or actual.__kwdefaults__!=keywords:
        raise ValueError('Loaded defaults differ: '+label)


def verify_loaded(root,expected):
    root=Path(root).resolve();hashes=verify_source_manifest(root,expected);definitions=0;imports=0
    for filename in sorted(hashes):
        name=Path(filename).stem;module=importlib.import_module(name)
        if Path(module.__file__).resolve()!=root/filename:raise ValueError('Loaded module path differs: '+name)
        raw=(root/filename).read_bytes();tree=ast.parse(raw)
        compiled=compile(raw,str(root/filename),'exec',dont_inherit=True)
        codes={c.co_name:c for c in compiled.co_consts if isinstance(c,types.CodeType)}
        namespace=vars(module)
        for node in tree.body:
            if isinstance(node,ast.Import):
                for alias in node.names:
                    binding=alias.asname or alias.name.split('.')[0]
                    original=importlib.import_module(alias.name if alias.asname else alias.name.split('.')[0])
                    if namespace[binding] is not original:raise ValueError('Imported module alias differs: '+name+'.'+binding)
                    imports+=1
            elif isinstance(node,ast.ImportFrom):
                if node.level or node.module is None:raise ValueError('Unsupported relative import')
                original=importlib.import_module(node.module)
                for alias in node.names:
                    if alias.name=='*' or namespace[alias.asname or alias.name] is not getattr(original,alias.name):
                        raise ValueError('Imported object alias differs: '+name+'.'+alias.name)
                    imports+=1
            elif isinstance(node,ast.FunctionDef):
                _function(node,codes[node.name],namespace[node.name],namespace,name+'.'+node.name);definitions+=1
            elif isinstance(node,ast.ClassDef):
                cls=namespace[node.name]
                if node.bases or node.keywords or node.decorator_list or type(cls) is not type:
                    raise ValueError('Unsupported/replaced class: '+name+'.'+node.name)
                methods={c.co_name:c for c in codes[node.name].co_consts if isinstance(c,types.CodeType)}
                for method in node.body:
                    if isinstance(method,ast.FunctionDef):
                        _function(method,methods[method.name],vars(cls)[method.name],namespace,name+'.'+node.name+'.'+method.name);definitions+=1
            elif isinstance(node,ast.Assign):
                # Literal module configuration (pins/inventory/orders) is bound;
                # generated set comprehensions are covered by reviewed module
                # source, not claimed to be reconstructed by this verifier.
                try:value=ast.literal_eval(node.value)
                except (ValueError,TypeError):continue
                for target in node.targets:
                    if isinstance(target,ast.Name) and (type(namespace[target.id]) is not type(value) or namespace[target.id]!=value):
                        raise ValueError('Literal module value differs: '+name+'.'+target.id)
    return dict(source_sha256=hashes,loaded_definitions=definitions,import_aliases=imports,
                loaded_source_bindings_passed=True,execution_certified=False)
