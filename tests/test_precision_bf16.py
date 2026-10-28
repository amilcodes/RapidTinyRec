import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from rapidtinyrec.bf16_policy import validate_state
from rapidtinyrec.bf16_protocol import load_protocol,PROTOCOL,PHASES,phase_record
from rapidtinyrec.bf16_runner import allowance
from rapidtinyrec.audit_bf16_dispatch import api_bf16_gemms,dispatch_evidence
from rapidtinyrec.audit_bf16 import inspect_phases

ROOT=Path(__file__).resolve().parents[1]

class Bf16PolicyTests(unittest.TestCase):
    def test_pilot_auditor_protocol_loader_is_not_shadowed_by_dead_full500_import(self):
        # Regression: a function-local import in an unreachable full500 branch
        # made the earlier pilot load_protocol call raise UnboundLocalError.
        self.assertTrue({'load_protocol','PROTOCOL'}.isdisjoint(inspect_phases.__code__.co_varnames))
        self.assertIn('load_protocol',inspect_phases.__code__.co_names)
        for flag in ('setup_diagnostic','staged_pilot','full500_quality'):
            with self.assertRaises(AssertionError):
                inspect_phases(None,None,None,None,**{flag:True})
    def test_policy_rejects_tf32_or_reduction_changes(self):
        state=dict(matmul_tf32=True,cudnn_tf32=False,float32_matmul_precision='high',blas_preference='_BlasBackend.Cublas',
            bf16_reduced_precision_reduction=False,bf16_split_k=True,math_sdpa_reduced_precision_reduction=False)
        for v in ('folded_tf32','folded_bf16'):validate_state(state,v)
        for k in ('matmul_tf32','cudnn_tf32','bf16_reduced_precision_reduction','bf16_split_k','math_sdpa_reduced_precision_reduction'):
            changed=dict(state);changed[k]=not changed[k]
            with self.assertRaises(AssertionError):validate_state(changed,'folded_bf16')
    def test_unknown_policy_rejected(self):
        with self.assertRaises(AssertionError):validate_state({},'folded_fp32')
    def test_distinct_frozen_protocol(self):
        p=load_protocol(ROOT/PROTOCOL)
        self.assertEqual(tuple(x['variant'] for x in p['phases']),PHASES)
        users=[dict(cohort_index=i) for i in p['cohort_indices']]
        for phase in range(4):phase_record(p,phase,phase>=2,users)
        with self.assertRaises(AssertionError):phase_record(p,2,False,users)
    def test_protocol_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'p.json';path.write_bytes((ROOT/PROTOCOL).read_bytes()+b' ')
            with self.assertRaises(AssertionError):load_protocol(path)
    def test_deadline_budgets(self):
        self.assertEqual(allowance('generation',0,0),1500)
        self.assertEqual(allowance('audit',0,1550),70)
        self.assertEqual(allowance('audit',0,1620),0)
        self.assertEqual(allowance('recovery',0,1619),1)

def api_block(shape=(8960,50,1536),dtype='CUDA_R_16BF(14)',compute='CUBLAS_COMPUTE_32F(68)'):
    return 'I! cuBLAS (v12.3) function cublasStatus_t cublasGemmEx(...) called:\n'+''.join(f'i! {k}: type=int; val={v}\n' for k,v in zip(('m','n','k'),shape))+''.join(f'i! {k}: type=cudaDataType; val={dtype}\n' for k in ('Atype','Btype','Ctype'))+f'i! computeType: type=cublasComputeType_t; val={compute}\n'

class Bf16DispatchTests(unittest.TestCase):
    def test_bf16_native_shape_and_fp32_accumulation(self):
        c=api_bf16_gemms(api_block())[0]
        self.assertTrue(c['bf16_inputs'] and c['bf16_output'] and c['fp32_compute'])
        self.assertEqual(c['shape'],dict(m=8960,n=50,k=1536))
    def test_fp16_or_bad_compute_does_not_pass(self):
        self.assertFalse(api_bf16_gemms(api_block(dtype='CUDA_R_16F(2)'))[0]['bf16_inputs'])
        self.assertFalse(api_bf16_gemms(api_block(compute='CUBLAS_COMPUTE_16F(64)'))[0]['fp32_compute'])
    def test_numeric_types(self):
        self.assertTrue(api_bf16_gemms(api_block(dtype='14',compute='68'))[0]['fp32_compute'])
        self.assertTrue(api_bf16_gemms(api_block(dtype='14',compute='CUDA_R_32F(0)'))[0]['fp32_compute'])
    def test_unrelated_probe_and_unlinked_kernel_cannot_pass(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'blas').write_text(api_block((8,8,8)))
            trace={'traceEvents':[dict(cat='kernel',name='ampere_bf16_gemm',dur=1,args={'correlation':1})]}
            for n in ('a','b'):(p/n).write_text(json.dumps(trace))
            r=dict(capture_logs=dict(available=True,cublas='blas'),profiles=[dict(trace=n) for n in ('a','b')])
            self.assertFalse(dispatch_evidence(p,r,True)['verified'])
    def test_actual_shapes_and_graph_link_required(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'blas').write_text(api_block()+api_block((1536,50,8960)))
            trace={'traceEvents':[dict(cat='cuda_runtime',name='cudaGraphLaunch',args={'correlation':i}) for i in range(4)]+[dict(cat='kernel',name='ampere_bf16_s16816gemm',dur=2,args={'correlation':2})]}
            for n in ('a','b'):(p/n).write_text(json.dumps(trace))
            r=dict(capture_logs=dict(available=True,cublas='blas'),profiles=[dict(trace=n) for n in ('a','b')])
            self.assertTrue(dispatch_evidence(p,r,True)['verified'])
            trace['traceEvents'][-1]['args']['correlation']=99;(p/'b').write_text(json.dumps(trace))
            self.assertFalse(dispatch_evidence(p,r,True)['verified'])

class UnchangedScoringTests(unittest.TestCase):
    def nodes(self,name):
        return list(ast.walk(ast.parse((ROOT/'rapidtinyrec'/name).read_text())))
    def test_unpack_and_timed_loop_unchanged(self):
        old,new=(self.nodes(n) for n in ('benchmark_precision.py','benchmark_bf16.py'))
        get=lambda nodes:next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name=='unpack')
        self.assertEqual(ast.dump(get(old)),ast.dump(get(new)))
        get_loop=lambda nodes:next(n for n in nodes if isinstance(n,ast.For) and isinstance(n.iter,ast.Name) and n.iter.id=='trials')
        self.assertEqual(ast.dump(get_loop(old)),ast.dump(get_loop(new)))
    def test_generate_changes_only_dtype_assertions(self):
        old,new=(self.nodes(n) for n in ('benchmark_precision.py','benchmark_bf16.py'))
        class NoAssertions(ast.NodeTransformer):
            def visit_Assert(self,n):return None
        get=lambda nodes:NoAssertions().visit(copy.deepcopy(next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name=='generate')))
        self.assertEqual(ast.dump(get(old)),ast.dump(get(new)))

if __name__=='__main__':unittest.main()
