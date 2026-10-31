"""Conservative exact 1D pointwise launch and store-coverage checks.

The caller must bind actual launch observations and independently derive each
output extent/role. This helper does not infer an extent from a producer record.
Unsupported multidimensional/reduction grids fail closed.
"""
import ast
import numpy as np
from .qkv_index_expression import IntegerExpressions,scalar_assignments


def pointwise_domain(source,function,launch):
    functions=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==function]
    if len(functions)!=1:raise ValueError('Ambiguous function')
    fn=functions[0];assignments=scalar_assignments(source,function)
    scalars=launch['runtime_integers'];config=launch['constexpr']
    if 'xnumel' not in scalars or set(config)!={'XBLOCK'}:raise ValueError('Only actual 1D pointwise scalar/block supported')
    n,b=scalars['xnumel'],config['XBLOCK']
    if type(n) is not int or not 1<=n<=2000000 or type(b) is not int or not 1<=b<=1024 or b&(b-1):raise ValueError('Unsupported bounded launch domain')
    expected=[(n+b-1)//b,1,1]
    if launch['actual_grid']!=expected:raise ValueError('Actual grid differs from independent ceiling division')
    canonical={'xoffset':'tl.program_id(0) * XBLOCK','xindex':'xoffset + tl.arange(0, XBLOCK)'}
    for name,value in canonical.items():
        actual=assignments.get(name)
        # Pinned Inductor emits arange(...)[:] as an identity view.
        if name=='xindex' and isinstance(actual,ast.BinOp) and isinstance(actual.op,ast.Add) and isinstance(actual.right,ast.Subscript) and isinstance(actual.right.slice,ast.Slice) and actual.right.slice.lower is None and actual.right.slice.upper is None and actual.right.slice.step is None:
            actual=ast.BinOp(left=actual.left,op=actual.op,right=actual.right.value)
        if actual is None or ast.dump(actual)!=ast.dump(ast.parse(value,mode='eval').body):raise ValueError('Unsupported actual pointwise lane mapping')
    programs=[x for x in ast.walk(fn) if isinstance(x,ast.Call) and ast.unparse(x.func)=='tl.program_id']
    ranges=[x for x in ast.walk(fn) if isinstance(x,ast.Call) and ast.unparse(x.func)=='tl.arange']
    if len(programs)!=1 or len(ranges)!=1:raise ValueError('Additional launch axes unsupported')
    if 'xnumel' in assignments and (not isinstance(assignments['xnumel'],ast.Constant) or type(assignments['xnumel'].value) is not int or assignments['xnumel'].value!=n):raise ValueError('Kernel changes actual extent')
    padded=expected[0]*b
    return dict(active_lanes=n,padded_lanes=padded,block=b,grid=expected,grid_formula_verified=True,complete_compiled_view_proof=False)


def verify_flat_store(source,function,launch,pointer,expected_extent):
    """All active/padding lanes, no sampling. Extent must come from caller schema."""
    domain=pointwise_domain(source,function,launch)
    if type(expected_extent) is not int or expected_extent!=domain['active_lanes']:raise ValueError('Independent output extent differs')
    fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==function)
    stores=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and ast.unparse(n.func)=='tl.store' and n.args and any(isinstance(x,ast.Name) and x.id==pointer for x in ast.walk(n.args[0]))]
    if len(stores)!=1:raise ValueError('Expected one complete output store')
    node=stores[0];address=node.args[0];kw={k.arg:k.value for k in node.keywords}
    if len(node.args) not in (2,3) or len(kw)!=len(node.keywords) or None in kw or len(node.args)==3 and 'mask' in kw:raise ValueError('Ambiguous store signature')
    if not isinstance(address,ast.BinOp) or not isinstance(address.op,ast.Add) or not isinstance(address.left,ast.Name) or address.left.id!=pointer:raise ValueError('Unsupported output address')
    mask=node.args[2] if len(node.args)==3 else kw.get('mask',ast.Constant(True))
    x=np.arange(domain['padded_lanes'],dtype=np.int64)
    expr=IntegerExpressions(scalar_assignments(source,function),{**launch['runtime_integers'],**launch['constexpr'],'xindex':x,'xoffset':(x//domain['block'])*domain['block']})
    active=np.broadcast_to(np.asarray(expr.evaluate(mask)),x.shape)
    offset=np.broadcast_to(np.asarray(expr.evaluate(address.right)),x.shape)
    if active.dtype.kind!='b' or not np.array_equal(active,x<expected_extent):raise ValueError('Store mask omits active lanes or writes padding')
    if not np.array_equal(offset[active],x[active]):raise ValueError('Store does not write independent contiguous output exactly once')
    return dict(domain,pointer=pointer,output_extent=expected_extent,store_mask_coverage_verified=True,complete_compiled_view_proof=False)
