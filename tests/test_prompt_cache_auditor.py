import os
from pathlib import Path
import unittest
from unittest.mock import patch
from rapidtinyrec.prompt_cache_protocol import PROTOCOL,load_protocol,phase_record
from rapidtinyrec.audit_prompt_cache import validate_phase_header,inspect_phases,evaluate_phases
from rapidtinyrec.audit_prompt_cache_storage import validate_request_storage


class AuditorTests(unittest.TestCase):
    def header(self):
        protocol=load_protocol(Path(__file__).resolve().parents[1]/PROTOCOL)
        users=[dict(cohort_index=i,prompt_tokens=n,prompt_token_ids=protocol['prefix_token_ids']+[1]*(n-69))
               for i,n in zip([1,2,6,10],[76,76,121,121])]
        m=dict(experiment='fixed_prompt_cache_tf32_pilot',phase=1,half=0,variant='fixed69_tf32',
               job_id='10',dtype='float32',tf32=True,users=users,compiler_artifacts=dict(reused=False),
               precision_policy=dict(matmul_tf32=True,cudnn_tf32=False,float32_matmul_precision='high',blas_preference='_BlasBackend.Cublas'),
               phase_protocol=phase_record(protocol,1,False,users))
        return m,1,dict(job_id='10'),protocol

    def test_distinct_header(self):validate_phase_header(*self.header())

    def test_other_experiments_policies_or_job_rejected(self):
        for key,value in (('experiment','bf16_vs_tf32_pilot'),('variant','folded_tf32'),
                          ('dtype','bfloat16'),('job_id','11'),('phase',2),('half',1)):
            args=self.header();args[0][key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):validate_phase_header(*args)
        args=self.header();args[0]['phase_protocol']['schema']='precision-phase-v1'
        with self.assertRaises(AssertionError):validate_phase_header(*args)

    def test_allocation_required_before_raw_audit(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(AssertionError,'Allocated prompt-cache'):
                inspect_phases(None,None,None,None)
            with self.assertRaisesRegex(AssertionError,'Allocated quality'):
                evaluate_phases(None,None,None,None,wrapper=None,allocation_identity={})

    def test_unsupported_scope_rejected(self):
        for key in ('full500_quality','staged_pilot','setup_diagnostic'):
            with self.assertRaises(AssertionError):inspect_phases(None,None,None,None,**{key:True})

    def ownership(self):
        report=dict(shape=[1,2,3,2],layers=1,storage_ranges=[(4096,4144),(8192,8240)])
        warm=[5,5,7,7];lengths=[n for n in warm for _ in range(2)]+warm[::-1]+warm*3+[5,7]
        records=[]
        for i,n in enumerate(lengths):
            record=dict(request_index=i,prompt_tokens=n)
            for mode,t in (('initial',3),('complete',n)):
                record[mode]=[dict(layer=0,kind=k,shape=[1,2,t,2],dtype='torch.float32',
                                  storage_ptr=16384+4096*j,storage_bytes=16*t) for j,k in enumerate(('key','value'))]
            records.append(record)
        return records,report,warm

    def test_sequential_request_address_reuse_allowed(self):
        records,report,warm=self.ownership()
        self.assertTrue(validate_request_storage(records,report,warm_lengths=warm)['persistent_and_request_kv_disjoint'])

    def test_partial_range_alias_and_wrong_inventory_rejected(self):
        for ptr in (4096,4100,16384):
            records,report,warm=self.ownership();records[20]['complete'][1]['storage_ptr']=ptr
            with self.assertRaisesRegex(AssertionError,'aliases'):
                validate_request_storage(records,report,warm_lengths=warm)
        records,report,warm=self.ownership();records.pop()
        with self.assertRaises(AssertionError):validate_request_storage(records,report,warm_lengths=warm)


if __name__=='__main__':unittest.main()
