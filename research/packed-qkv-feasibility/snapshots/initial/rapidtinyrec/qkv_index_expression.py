"""Restricted integer AST evaluation for generated Triton address expressions.

No eval/exec/import/kernel execution. This is a building block, not a complete
compiled-view proof: callers must independently bind pointer, output role, launch
domain and masks to the capture-bound wrapper before accepting a model gate.
"""
import ast
import operator
import numpy as np


class IntegerExpressions:
    def __init__(self,assignments,inputs):
        self.assignments=assignments;self.inputs=dict(inputs);self.cache={};self.active=set();self.nodes=0

    def evaluate(self,expression):
        node=ast.parse(expression,mode='eval').body if isinstance(expression,str) else expression
        self.nodes=0
        return self._node(node)

    def _node(self,node):
        self.nodes+=1
        if self.nodes>4096:raise ValueError('Index expression expansion bound exceeded')
        value=self._compute(node)
        array=np.asarray(value)
        if array.dtype.kind not in 'biuO':raise ValueError('Noninteger index value')
        if array.dtype.kind=='O' and any(type(x) not in (int,bool) for x in array.flat):raise ValueError('Noninteger object index')
        # Validate BEFORE converting inputs or applying any parent operation.
        # All operands fit signed32; supported +/-/*, positive // and %, and
        # bitwise operations therefore fit signed64 before this next check.
        # In particular the largest product is 2**62; shifts/powers are banned.
        if np.any(array < -(2**31)) or np.any(array > 2**31-1):raise ValueError('Index intermediate exceeds signed32 proof domain')
        if isinstance(value,np.ndarray):
            return value if array.dtype.kind=='b' else array.astype(np.int64,copy=False)
        if isinstance(value,np.generic):return value.item()
        return value

    def _compute(self,node):
        if isinstance(node,ast.Constant) and type(node.value) in (int,bool):return node.value
        if isinstance(node,ast.Name):
            if node.id in self.inputs:return self.inputs[node.id]
            if node.id in self.cache:return self.cache[node.id]
            if node.id not in self.assignments or node.id in self.active:raise ValueError('Unknown/cyclic index variable: '+node.id)
            self.active.add(node.id)
            try:value=self._node(self.assignments[node.id]);self.cache[node.id]=value;return value
            finally:self.active.remove(node.id)
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd,ast.Invert,ast.Not)):
            value=self._node(node.operand)
            return {ast.USub:operator.neg,ast.UAdd:operator.pos,ast.Invert:operator.invert,ast.Not:np.logical_not}[type(node.op)](value)
        if isinstance(node,ast.BinOp):
            operations={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.FloorDiv:operator.floordiv,ast.Mod:operator.mod,ast.BitAnd:operator.and_,ast.BitOr:operator.or_,ast.BitXor:operator.xor}
            if type(node.op) not in operations:raise ValueError('Unsupported/noninteger address operation')
            left,right=self._node(node.left),self._node(node.right)
            if isinstance(node.op,(ast.FloorDiv,ast.Mod)) and np.any(np.asarray(right)<=0):raise ValueError('Positive integer divisor required')
            return operations[type(node.op)](left,right)
        if isinstance(node,ast.Compare):
            operations={ast.Lt:operator.lt,ast.LtE:operator.le,ast.Eq:operator.eq,ast.NotEq:operator.ne,ast.Gt:operator.gt,ast.GtE:operator.ge}
            left=self._node(node.left);result=True
            for op,right_node in zip(node.ops,node.comparators):
                if type(op) not in operations:raise ValueError('Unsupported mask comparison')
                right=self._node(right_node);result=np.logical_and(result,operations[type(op)](left,right));left=right
            return result
        if isinstance(node,ast.BoolOp) and isinstance(node.op,(ast.And,ast.Or)):
            values=[self._node(x) for x in node.values()];result=values[0]
            for value in values[1:]:result=(np.logical_and if isinstance(node.op,ast.And) else np.logical_or)(result,value)
            return result
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='to' and len(node.args)==1 and not node.keywords and ast.unparse(node.args[0]) in ('tl.int32','tl.int64'):
            value=self._node(node.func.value)
            if np.asarray(value).dtype.kind=='b':raise ValueError('Boolean cast unsupported')
            return value  # Every intermediate already fits signed32 exactly.
        if isinstance(node,ast.Call) and ast.unparse(node.func)=='tl.full' and len(node.args)==3 and not node.keywords:
            if ast.unparse(node.args[0])!='[1]' or ast.unparse(node.args[2]) not in ('tl.int32','tl.int64') or not isinstance(node.args[1],ast.Constant) or type(node.args[1].value) is not int:raise ValueError('Unsupported integer full')
            return self._node(node.args[1])
        if isinstance(node,ast.Call) and ast.unparse(node.func) in ('tl.where','triton_helpers.div_floor_integer') and not node.keywords:
            values=[self._node(x) for x in node.args]
            if ast.unparse(node.func)=='tl.where' and len(values)==3:return np.where(*values)
            if ast.unparse(node.func)=='triton_helpers.div_floor_integer' and len(values)==2 and np.all(np.asarray(values[1])>0):return np.floor_divide(*values)
        raise ValueError('Unsupported generated address syntax: '+type(node).__name__)


def scalar_assignments(kernel_source,function_name):
    tree=ast.parse(kernel_source);functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function_name]
    if len(functions)!=1:raise ValueError('Expected one declared kernel function')
    result={}
    # Only direct SSA assignments. Control-flow-dependent addressing requires a
    # separate validator; silently flattening branches would be unsound.
    for node in functions[0].body:
        if isinstance(node,(ast.If,ast.For,ast.While,ast.Try,ast.With)):raise ValueError('Unsupported kernel address control flow')
        if isinstance(node,ast.Assign):
            if len(node.targets)!=1 or not isinstance(node.targets[0],ast.Name):raise ValueError('Unsupported kernel assignment')
            name=node.targets[0].id
            if name in result:raise ValueError('Non-SSA kernel variable')
            result[name]=node.value
    return result


def validate_flat_column_map(offset,mask,assignments,*,rows,widths,role,rotation=False,head_dim=128,block=256):
    """Exact exhaustive integer address/mask check for a declared flat role map.

    This only supports direct Q/K/V or RoPE half-rotation reads with a flat
    output domain. Launch-domain/role/pointer extraction is NOT supplied here.
    Padding lanes must be masked exactly; no inbounds-only or sample proof.
    """
    if type(rows) is not int or not 1<=rows<=50 or len(widths)!=3 or any(type(w) is not int or w<=0 for w in widths):raise ValueError('Invalid independent layout')
    if role not in ('q','k','v') or type(block) is not int or not 1<=block<=1024:raise ValueError('Invalid role/launch block')
    index=('q','k','v').index(role);width=widths[index];total=sum(widths);start=sum(widths[:index]);count=rows*width
    if count>100000 or total>4096 or type(head_dim) is not int or head_dim<=0 or head_dim%2 or width%head_dim:raise ValueError('Unsupported bounded column domain')
    if rotation and role=='v':raise ValueError('V must not use rotary half permutation')
    padded=((count+block-1)//block)*block;x=np.arange(padded,dtype=np.int64)
    expressions=IntegerExpressions(assignments,{'xindex':x,'xnumel':count,'XBLOCK':block})
    actual=np.asarray(expressions.evaluate(offset));active=np.asarray(expressions.evaluate(mask))
    if actual.dtype.kind not in 'iuO' or active.dtype.kind!='b':raise ValueError('Address/mask types differ')
    actual=np.broadcast_to(actual,x.shape);active=np.broadcast_to(active,x.shape);expected_active=x<count
    if not np.array_equal(active,expected_active):raise ValueError('Incomplete/extra active lanes')
    columns=x%width
    if rotation:columns=(columns//head_dim)*head_dim+(columns+head_dim//2)%head_dim
    expected=(x//width)*total+start+columns
    if not np.array_equal(actual[active],expected[active]):raise ValueError('Compiled column address map differs')
    return dict(role=role,rows=rows,width=width,total_width=total,rotation=rotation,active_lanes=count,padding_lanes=padded-count,
                exact_integer_map=True,complete_compiled_view_proof=False)
