import contextlib
import copy
import gzip
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from rapidtinyrec import audit_fp32_backend as audit
from rapidtinyrec.fp32_backend_probe import backend_scope, case_keys, expected_shapes, validate_tensors


class FakeAPI:
    def __init__(self): self.value='previous'; self.calls=[]
    def __call__(self,value=None):
        if value is not None:self.value=value;self.calls.append(value)
        return self.value


class BackendTests(unittest.TestCase):
    def test_backend_restoration_on_success_and_exception(self):
        api=FakeAPI()
        with backend_scope(api,'cublaslt') as active:self.assertEqual(active,'cublaslt')
        self.assertEqual(api.value,'previous')
        with self.assertRaisesRegex(RuntimeError,'probe'):
            with backend_scope(api,'cublas'):raise RuntimeError('probe')
        self.assertEqual(api.value,'previous')

    def test_unsupported_backend_has_no_side_effect(self):
        api=FakeAPI()
        with self.assertRaises(ValueError):
            with backend_scope(api,'autotune'):pass
        self.assertEqual(api.calls,[])

    def test_fixed_scope_and_shapes(self):
        self.assertEqual(case_keys(),[(u,l) for u in (1,2,6,10) for l in (0,13,27)])
        self.assertEqual(expected_shapes('gate'),((50,1536),(8960,1536),(50,8960)))
        self.assertEqual(expected_shapes('down'),((50,8960),(1536,8960),(50,1536)))
        with self.assertRaises(ValueError):expected_shapes('wide')

    def test_tensor_shapes_strides_dtype_and_alias_guards(self):
        class T:
            dtype='torch.float32'
            def __init__(self,shape,ptr):self.shape=shape;self.ptr=ptr;self.layout=(shape[1],1)
            def stride(self):return self.layout
            def untyped_storage(self):return self
            def data_ptr(self):return self.ptr
        x,w,y=[T(s,i) for i,s in enumerate(expected_shapes('gate'))]
        validate_tensors(x,w,y,'gate')
        y.ptr=x.ptr
        with self.assertRaisesRegex(ValueError,'aliases'):validate_tensors(x,w,y,'gate')
        y.ptr=2;x.layout=(1,50)
        with self.assertRaisesRegex(ValueError,'stride'):validate_tensors(x,w,y,'gate')
        x.layout=(1536,1);x.dtype='torch.float16'
        with self.assertRaisesRegex(ValueError,'FP32'):validate_tensors(x,w,y,'gate')


class BitTests(unittest.TestCase):
    def test_signed_zero_and_sub_ulp_values_are_not_relaxed(self):
        a=struct.pack('<3I',0,0x3f800000,0x3f800001)
        b=struct.pack('<3I',0x80000000,0x3f800001,0x3f800001)
        c=audit.compare_bytes(a,b)
        self.assertEqual(c['matching_bits'],1);self.assertEqual(c['values'],3)
        self.assertEqual(c['max_abs_finite'],2**-23)

    def test_nonfinite_cannot_pass_finite_exactness(self):
        x=struct.pack('<2f',float('inf'),float('nan'))
        c=audit.compare_bytes(x,x)
        self.assertEqual(c['matching_bits'],2);self.assertEqual(c['nonfinite_left'],2)

    def test_wrong_length_and_layout_rejected(self):
        with self.assertRaises(AssertionError):audit.compare_bytes(b'0',b'0')
        with self.assertRaises(AssertionError):audit.check_layout(dict(shape=[50,1536],stride=[1,50],dtype='torch.float32',storage_offset=0),(50,1536))


def fixture(root):
    entries={}
    (root/'tensors').mkdir()
    def put(shape,changed=False):
        width=shape[1]
        data=struct.pack('<f',1.)*width+b'\0'*(49*width*4)
        if changed:data=audit.row_reverse(data,50,width)
        digest=audit.sha(data);name=f'tensors/{digest}.f32.gz';z=gzip.compress(data,mtime=0)
        (root/name).write_bytes(z)
        entries[digest]=dict(file=name,bytes=len(data),compressed_bytes=len(z),compressed_sha256=audit.sha(z))
        return dict(shape=list(shape),stride=[width,1],dtype='torch.float32',storage_offset=0,sha256=digest)
    case=dict(cohort_index=1,layer=0,prompt_tokens=76,graph_count=9,timed_graph_replays=2304,measured_captures=0,measured_compilations=0,
              inputs={},weight_shapes={},weight_sha256={},variants={},staging_ptrs=[100,101,102],input_logical_bytes=2406400,output_logical_bytes=11673600,unique_input_output_bytes=14080000)
    # Full-sized shape fixture checks production shape validation; only CASES is narrowed for unit tests.
    for role in audit.ROLES:
        width,outer=(8960,1536) if role=='down' else (1536,8960)
        case['inputs'][role]=put((50,width));case['weight_shapes'][role]=[outer,width];case['weight_sha256'][role]='a'*64
    for bi,backend in enumerate(audit.BACKENDS):
        v=dict(requested_backend=backend,selected_backend='_BlasBackend.'+['Cublas','Cublas','Cublaslt'][bi],states={},output_ptrs={r:10+bi*3+i for i,r in enumerate(audit.ROLES)},projection_us={},timing_summary={},after_timing={})
        for state in ('original','changed','restored'):
            v['states'][state]=dict(inputs={},roles={})
            for role in audit.ROLES:
                width,outer=(8960,1536) if role=='down' else (1536,8960)
                v['states'][state]['inputs'][role]=put((50,width),state=='changed')
                refs={k:put((50,outer),state=='changed') for k in ('direct','direct_repeat','graph','graph_repeat','retained_after_other_roles')}
                v['states'][state]['roles'][role]=refs
                v['after_timing'][role]=put((50,outer))
                v['projection_us'][role]=[10.-bi]*8
                v['timing_summary'][role]=dict(median_us=10.-bi,p95_us=10.-bi,min_us=10.-bi,max_us=10.-bi)
        name=backend+'-trace.json';v['trace']=name
        (root/name).write_text(json.dumps(dict(traceEvents=[dict(name='fp32_probe_'+r,cat='user_annotation') for r in audit.ROLES]+[dict(name='sgemm',cat='kernel',dur=2,args=dict(grid=[70,1,3],block=[128,1,1]))]*3)))
        v['logs']=dict(available=True,cublas=backend+'.blas.log',cublaslt=backend+'.lt.log')
        (root/v['logs']['cublas']).write_text('cublasSgemm\n'*3 if bi<2 else '')
        (root/v['logs']['cublaslt']).write_text('cublasLtMatmul\n'*3 if bi==2 else '')
        case['variants'][backend]=v
    return dict(schema=1,complete=True,torch='2.14.0+cu126',cuda='12.6',torch_git_version=audit.COMMIT,hardware='NVIDIA RTX A6000',transformers='4.55.4',
                runtime_precision=dict(tf32=False,cudnn_tf32=False,tunable_enabled=False,tuning_enabled=False,matmul_precision='highest',initial_backend='_BlasBackend.Cublas'),
                restored_backend='_BlasBackend.Cublas',cases=[case],tensors=entries,model_parameter_logical_bytes=10,model_parameter_unique_bytes=10)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.d=fixture(self.root)
    def tearDown(self):self.tmp.cleanup()
    def check(self):
        result={}
        with patch.object(audit,'CASES',[(1,0)]):audit.audit_report(self.root,self.d,result)
        return result

    def test_complete_fixture_and_no_automatic_promotion(self):
        r=self.check();self.assertTrue(r['candidate_exact']);self.assertTrue(r['dispatch_verified']);self.assertFalse(r['end_to_end_promotion'])
        self.assertEqual(r['candidate_faster_median_cases'],3)

    def test_changed_graph_and_retention_failures_are_retained(self):
        v=self.d['cases'][0]['variants']['cublaslt']
        v['states']['changed']['roles']['gate']['graph']=v['states']['original']['roles']['gate']['graph']
        v['states']['changed']['roles']['gate']['retained_after_other_roles']=v['states']['original']['roles']['gate']['graph']
        r=self.check();self.assertFalse(r['candidate_exact']);self.assertFalse(r['stability_exact'])
        self.assertTrue(any(x['matching_bits']<x['values'] for x in r['comparisons']))

    def test_unexpected_explicit_control_difference_is_not_accepted(self):
        v=self.d['cases'][0]['variants']['cublas']
        v['states']['original']['roles']['gate']['direct']=v['states']['changed']['roles']['gate']['direct']
        r=self.check();self.assertFalse(r['control_exact'])

    def test_corrupt_raw_blob_rejected(self):
        e=next(iter(self.d['tensors'].values()));(self.root/e['file']).write_bytes(b'bad')
        with self.assertRaises(AssertionError):self.check()

    def test_incomplete_cases_wrong_timing_and_alias_rejected(self):
        with self.assertRaises(AssertionError):audit.audit_report(self.root,self.d,{})
        self.d['cases'][0]['variants']['default']['projection_us']['gate'].pop()
        with self.assertRaises(AssertionError):self.check()
        self.d['cases'][0]['variants']['default']['projection_us']['gate'].append(10.)
        self.d['cases'][0]['staging_ptrs'][0]=10
        with self.assertRaises(AssertionError):self.check()

    def test_dispatch_fallback_blocks_verified_backend_change(self):
        v=self.d['cases'][0]['variants']['cublaslt']
        (self.root/v['logs']['cublas']).write_text('cublasSgemm\n')
        r=self.check();self.assertTrue(r['candidate_exact']);self.assertFalse(r['dispatch_verified'])

    def test_failed_cli_preserves_failure_json(self):
        output=self.root/'audit.json'
        p=subprocess.run([sys.executable,'-m','rapidtinyrec.audit_fp32_backend',str(self.root),'--snapshot',str(self.root),'--run-record',str(self.root/'missing.json'),'--output',str(output)],capture_output=True)
        self.assertEqual(p.returncode,1)
        result=json.loads(output.read_text());self.assertFalse(result['integrity_audit_passed']);self.assertTrue(result['errors'])


if __name__=='__main__':unittest.main()
