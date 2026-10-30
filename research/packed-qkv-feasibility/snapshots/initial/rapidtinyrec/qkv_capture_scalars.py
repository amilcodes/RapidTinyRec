"""Bind captured Python integers to source-ordered wrapper launch expressions.

No tensor reads, .item(), generated code execution, or inferred runtime values.
This does not prove Triton grid selection, tensor-valued scalar arguments, or
store masks; incomplete launch entries remain explicitly unresolved.
"""
import ast
from .qkv_index_expression import IntegerExpressions
from .qkv_wrapper_syntax import call_definition,flat_statements


def bind_capture_scalars(source,observation):
    if observation['schema']!='capture-python-integers-v1' or observation['returned'] is not True:raise ValueError('Missing completed scalar observation')
    if observation['boxed_kind'] not in ('list','tuple'):raise ValueError('Expected actual boxed wrapper arguments')
    count=observation['boxed_count'];entry=observation['entry_integers'];final=observation['return_integers']
    if type(count) is not int or not 1<=count<=4096:raise ValueError('Invalid boxed argument count')
    if any(not isinstance(k,str) or not k.isdecimal() or str(int(k))!=k or not 0<=int(k)<count or type(v) is not int for k,v in entry.items()):raise ValueError('Invalid boxed integer inventory')
    if any(not isinstance(k,str) or not k.isidentifier() or type(v) is not int for k,v in final.items()):raise ValueError('Invalid final Python integer inventory')
    fn,_=call_definition(ast.parse(source));statements=list(flat_statements(fn))
    if not statements or not isinstance(statements[0],ast.Assign):raise ValueError('Missing initial boxed unpack')
    unpack=statements[0]
    if len(unpack.targets)!=1 or not isinstance(unpack.targets[0],(ast.Tuple,ast.List)) or ast.unparse(unpack.value)!='args':raise ValueError('Unsupported boxed unpack')
    names=unpack.targets[0].elts
    if len(names)!=count or any(not isinstance(x,ast.Name) for x in names) or len({x.id for x in names})!=count:raise ValueError('Boxed source/runtime count differs')
    known={names[int(k)].id:v for k,v in entry.items()};entry_names=dict(known);derived=dict(known);launches=[];updates=[]
    def evaluate(node):
        refs={x.id for x in ast.walk(node) if isinstance(x,ast.Name)}
        if not refs<=known.keys():return None
        try:value=IntegerExpressions({},known).evaluate(node)
        except (ValueError,TypeError):return None
        return value if type(value) is int else None
    for statement in statements[1:]:
        if isinstance(statement,(ast.If,ast.For,ast.While,ast.Try,ast.With,ast.Match,ast.AugAssign,ast.AnnAssign)):raise ValueError('Unsupported scalar control/mutation')
        for node in ast.walk(statement):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='run' and isinstance(node.func.value,ast.Name):
                bound={};unresolved=[]
                for i,arg in enumerate(node.args):
                    value=evaluate(arg)
                    if value is None:unresolved.append(i)
                    else:bound[str(i)]=value
                launches.append(dict(line=node.lineno,kernel=node.func.value.id,scalar_arguments=bound,unresolved_positions=unresolved,
                                     launch_domain_verified=False))
        if isinstance(statement,ast.Assign):
            assigned=[x.id for t in statement.targets for x in ast.walk(t) if isinstance(x,ast.Name)]
            value=evaluate(statement.value)
            if value is None and any(name in known for name in assigned):raise ValueError('Scalar overwritten by unsupported expression')
            for name in assigned:known.pop(name,None)
            if value is not None:
                if len(statement.targets)!=1 or not isinstance(statement.targets[0],ast.Name):raise ValueError('Unsupported scalar assignment target')
                name=assigned[0];known[name]=value;derived[name]=value
                updates.append(dict(line=statement.lineno,name=name,value=value))
        elif isinstance(statement,ast.Delete):
            for target in statement.targets:
                if isinstance(target,ast.Name):
                    known.pop(target.id,None);updates.append(dict(line=statement.lineno,name=target.id,value=None))
    for name,value in known.items():
        if name not in final or final[name]!=value:raise ValueError('Source-derived scalar differs from actual returned local')
    # Entry/derived integers must fit the same exact arithmetic proof domain.
    for value in derived.values():
        if not -(2**31)<=value<2**31:raise ValueError('Scalar outside signed32 index domain')
    return dict(schema='capture-source-scalars-v1',entry_arguments=len(entry),entry_names=entry_names,scalar_updates=updates,derived_integers=derived,launches=launches,
                actual_python_integer_values_bound=True,launch_domain_verified=False,complete_compiled_view_proof=False)
