"""Pinned generated-wrapper entry forms: plain call or Runner.call alias."""
import ast


def call_definition(tree,required=True):
    plain=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='call']
    runners=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Runner']
    methods=[n for c in runners for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='call']
    if not plain and not methods and not required:return None,None
    if len(plain)+len(methods)!=1:raise ValueError('Ambiguous/missing generated entry')
    if plain:return plain[0],None
    if len(runners)!=1:raise ValueError('Ambiguous Runner')
    aliases=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='call' for t in n.targets)]
    owners=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='runner' for t in n.targets)]
    if len(aliases)!=1 or ast.unparse(aliases[0].value)!='runner.call' or len(owners)!=1 or ast.unparse(owners[0].value)!='Runner(partitions=[])':raise ValueError('Unbound Runner.call entry')
    return methods[0],'Runner'


def flat_statements(function):
    for node in function.body:
        if isinstance(node,ast.With):
            if len(node.items)!=1 or node.items[0].optional_vars is not None or ast.unparse(node.items[0].context_expr)!='torch.cuda._DeviceGuard(0)':raise ValueError('Unsupported wrapper context')
            # Exactly one transparent pinned device guard; no branch flattening.
            if any(isinstance(x,ast.With) for x in node.body):raise ValueError('Nested wrapper context unsupported')
            yield from node.body
        else:yield node
