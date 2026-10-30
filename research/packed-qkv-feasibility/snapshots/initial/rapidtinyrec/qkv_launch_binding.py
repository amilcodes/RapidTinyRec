"""Cross-check observed run/launcher integer parameters against wrapper source.

Observed grid/config values are retained; grid formula and masks are NOT proved
by this binding and remain mandatory independent gates.
"""
import ast
from .qkv_kernel_reads import embedded_kernel
from .qkv_runner_binding import validate_runner_forwarding
from .qkv_compile_result import validate_compile_result
from .qkv_wrapper_syntax import call_definition,flat_statements


def bind_launch_observations(source,scalars,observations):
    fn,_=call_definition(ast.parse(source));nodes=[]
    for statement in flat_statements(fn):
        nodes.extend(n for n in ast.walk(statement) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='run' and isinstance(n.func.value,ast.Name))
    if [r['line'] for r in observations]!=[n.lineno for n in nodes]:raise ValueError('Actual launcher coverage/order differs from source')
    scalar_by_line={r['line']:r for r in scalars['launches']};result=[]
    for node,record in zip(nodes,observations):
        symbol=node.func.value.id;k=embedded_kernel(source,symbol)
        if record['kernel_symbol']!=symbol or record['kernel_sha256']!=k['sha256'] or record['observed_run_to_launcher'] is not True:raise ValueError('Launch/source kernel identity differs')
        definitions=[n for n in ast.parse(k['source']).body if isinstance(n,ast.FunctionDef) and n.name==k['function']]
        if len(definitions)!=1:raise ValueError('Ambiguous kernel signature')
        ps=definitions[0].args.args
        runtime=[p.arg for p in ps if p.annotation is None or ast.unparse(p.annotation)!='tl.constexpr']
        constants={p.arg for p in ps if p.annotation is not None and ast.unparse(p.annotation)=='tl.constexpr'}
        if len(runtime)!=len(node.args) or set(record['constexpr'])!=constants:raise ValueError('Runtime/constexpr signature mismatch')
        if not set(record['runtime_integers'])<=set(runtime):raise ValueError('Unknown runtime scalar')
        if any(type(v) is not int or not -(2**31)<=v<2**31 for v in [*record['runtime_integers'].values(),*record['constexpr'].values()]):raise ValueError('Scalar outside exact integer domain')
        for pos,value in scalar_by_line[node.lineno]['scalar_arguments'].items():
            if record['runtime_integers'].get(runtime[int(pos)])!=value:raise ValueError('Wrapper/runtime scalar mismatch')
        if len(record['actual_grid'])!=3 or any(type(v) is not int or not 1<=v<2**31 for v in record['actual_grid']):raise ValueError('Invalid observed grid')
        for key in ('runtime_source_sha256','runtime_code_sha256','launcher_code_sha256'):
            if len(record[key])!=64 or any(c not in '0123456789abcdef' for c in record[key]):raise ValueError('Missing runtime identity digest')
        result.append(dict(record,source_runtime_parameters=runtime,runner_binding=validate_runner_forwarding(k,record),compile_result_binding=validate_compile_result(k,record),grid_formula_verified=False))
    return dict(launches=result,source_to_actual_launcher_parameters_bound=True,launch_domain_verified=False,
                complete_compiled_view_proof=False,limitation='Actual runtime grid/config observations, not independent grid formula/store-mask/index proof.')
