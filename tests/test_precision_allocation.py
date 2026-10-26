from copy import deepcopy
import hashlib
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rapidtinyrec.precision_allocation import (CONFIG,PROJECT,SSH_TARGET,REMOTE_ROOT,COMMAND,
    scheduler_fields,validate_manifest,validate_scheduler,verify_source,reconcile_record)
from rapidtinyrec.precision_full500 import PROTOCOL,PROFILE,load_protocol
from rapidtinyrec.audit_precision_full500 import audit,validate_live_record,main

ROOT=Path(__file__).resolve().parents[1]

class AllocationIdentityTests(unittest.TestCase):
    def setUp(self):
        self.protocol=load_protocol(ROOT/PROTOCOL)
        self.remote=REMOTE_ROOT+'/'+PROJECT+'/precision-full500-quality-fixture'
        resources={**self.protocol['resources'],'setup':['module purge'],'command':COMMAND}
        self.prepared=dict(run_id='precision-full500-quality-fixture',project=PROJECT,profile=PROFILE,
            ssh_target=SSH_TARGET,remote_dir=self.remote,job_id=None,submission='prepared',
            resources=resources,files=[CONFIG,PROTOCOL,COMMAND[1]],
            sha256={name:'a'*64 for name in (CONFIG,PROTOCOL,COMMAND[1])})
        self.config=dict(project=PROJECT,ssh_target=SSH_TARGET,remote_root=REMOTE_ROOT,profiles={PROFILE:resources})
        self.env=dict(SLURM_JOB_ID='12345',OSCAR_RUN_DIR=self.remote,OSCAR_OUTPUT_DIR=self.remote+'/outputs')
        self.raw=('JobId=12345 JobState=RUNNING Partition=gpu QOS=norm-gpu TimeLimit=00:30:00 '
            'NumCPUs=4 NumNodes=1 Command='+self.remote+'/job.sh WorkDir='+self.remote+
            ' AllocTRES=cpu=4,mem=32G,node=1,gres/gpu=1,gres/gpu:nvidia_rtx_a6000=1 NodeList=gpu-fixture')
        self.fields=scheduler_fields(self.raw)
    def validate(self):return validate_manifest(self.prepared,self.config,self.protocol,'12345\n',self.env,self.fields)
    def records(self):
        uploaded=json.dumps(self.prepared).encode();job=b'12345\n'
        allocated=deepcopy(self.prepared);allocated.update(job_id='12345',submission='allocated_observation',
            allocation_identity=dict(schema='precision-allocated-identity-v1',requires_local_oscar_reconciliation=True,
                uploaded_manifest_sha256=hashlib.sha256(uploaded).hexdigest(),sbatch_job_id_file_sha256=hashlib.sha256(job).hexdigest(),
                scheduler_stdout=self.raw,scheduler_fields=self.fields,scheduler_command=['scontrol','show','job','-o','12345']))
        actual=deepcopy(self.prepared);actual.update(job_id='12345',submission='submitted')
        return allocated,actual,uploaded,job
    def test_manifest_sbatch_and_scheduler_are_bound(self):
        self.assertEqual(self.validate(),'12345')
        result=reconcile_record(*self.records());self.assertTrue(result['matched']);self.assertFalse(result['bootstrap_executed'])
    def test_wrong_env_and_prepared_identity_fail(self):
        for mutate in [lambda:self.env.update(SLURM_JOB_ID='54321'),lambda:self.env.pop('SLURM_JOB_ID'),
                       lambda:self.env.update(OSCAR_OUTPUT_DIR='/other'),lambda:self.prepared.update(job_id='12345'),
                       lambda:self.prepared.update(submission='submitted'),lambda:self.prepared.update(project='other'),
                       lambda:self.prepared['files'].append(CONFIG),lambda:self.prepared['resources'].update(cpus=8)]:
            self.setUp();mutate()
            with self.assertRaises(AssertionError):self.validate()
    def test_scheduler_resources_and_execution_identity_fail_closed(self):
        for k,v in [('JobId','54321'),('JobState','COMPLETED'),('Partition','gpu-he'),('QOS','other'),
                    ('TimeLimit','01:00:00'),('NumCPUs','8'),('NumNodes','2'),('Command','other'),('WorkDir','other'),
                    ('AllocTRES','cpu=4,mem=32G,node=1,gres/gpu=2,gres/gpu:nvidia_rtx_a6000=2')]:
            fields=dict(self.fields,**{k:v})
            with self.subTest(k=k),self.assertRaises(AssertionError):validate_scheduler(fields,'12345',self.remote)
        fields=dict(self.fields,AllocTRES=self.fields['AllocTRES'].replace('32G','32768M'))
        validate_scheduler(fields,'12345',self.remote)
    def test_duplicate_scheduler_fields_and_unknown_memory_fail(self):
        with self.assertRaises(AssertionError):scheduler_fields(self.raw+' JobId=12345')
        fields=dict(self.fields,AllocTRES=self.fields['AllocTRES'].replace('32G','32000000000'))
        with self.assertRaises(AssertionError):validate_scheduler(fields,'12345',self.remote)
        fields=dict(self.fields,AllocTRES=self.fields['AllocTRES']+',cpu=4')
        with self.assertRaises(AssertionError):validate_scheduler(fields,'12345',self.remote)
    def test_config_protocol_or_entry_script_omission_fails(self):
        for name in (CONFIG,PROTOCOL,COMMAND[1]):
            self.setUp();self.prepared['files'].remove(name);del self.prepared['sha256'][name]
            with self.assertRaises(AssertionError):self.validate()
    def test_reconciliation_rejects_changed_actual_record_manifest_job_file(self):
        for key in ('job_id','run_id','profile','sha256','resources'):
            a,r,m,j=self.records();r[key]='changed'
            with self.subTest(key=key),self.assertRaises(AssertionError):reconcile_record(a,r,m,j)
        a,r,m,j=self.records()
        with self.assertRaises(AssertionError):reconcile_record(a,r,m+b' ',j)
        with self.assertRaises(AssertionError):reconcile_record(a,r,m,b'54321\n')
    def test_reconciliation_rejects_changed_scheduler_proof(self):
        a,r,m,j=self.records();a['allocation_identity']['scheduler_fields']=dict(self.fields,NumCPUs='8')
        with self.assertRaises(AssertionError):reconcile_record(a,r,m,j)
    def test_live_record_rejects_different_source_and_job(self):
        a,_,_,_=self.records();validate_live_record(a,deepcopy(a))
        for key in ('job_id','sha256','resources'):
            other=deepcopy(a);other[key]='changed'
            with self.assertRaises(AssertionError):validate_live_record(a,other)
    def test_no_allocation_cannot_call_paired_audit(self):
        with patch.dict(os.environ,{},clear=True),patch('rapidtinyrec.audit_precision_full500.paired_audit') as paired:
            with self.assertRaisesRegex(AssertionError,'allocated compute'):audit(None,None,None,None,None)
            paired.assert_not_called()
    def test_cli_preserves_failed_audit_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'audit.json'
            argv=['audit','allocated','--output',str(output)]
            for key in ('root','historical','snapshot','run-root','release-root'):argv+=['--'+key,temp]
            with patch.dict(os.environ,{},clear=True),patch('sys.argv',argv),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as error:main()
                self.assertEqual(error.exception.code,1)
                original=output.read_bytes();result=json.loads(original)
                self.assertFalse(result['integrity_audit_passed'])
                self.assertIn('allocated compute',result['traceback'])
                with self.assertRaisesRegex(AssertionError,'overwrite'):main()
                self.assertEqual(output.read_bytes(),original)
    def test_local_reconcile_cli_cannot_invoke_resampling(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a,r,m,j=self.records()
            blobs={'allocated-record':json.dumps(a).encode(),'run-record':json.dumps(r).encode(),
                   'uploaded-manifest':m,'job-id-file':j}
            argv=['audit','reconcile','--output',str(root/'result.json')]
            for name,data in blobs.items():
                (root/name).write_bytes(data);argv+=['--'+name,str(root/name)]
            with patch('sys.argv',argv),patch('rapidtinyrec.audit_precision_full500.paired_audit') as paired,contextlib.redirect_stdout(io.StringIO()):
                main();paired.assert_not_called()
            self.assertTrue(json.loads((root/'result.json').read_text())['matched'])

class SourceIdentityTests(unittest.TestCase):
    def test_every_running_dependency_must_be_submitted_and_hash_match(self):
        with tempfile.TemporaryDirectory() as temp:
            code=Path(temp);(code/'rapidtinyrec').mkdir()
            hashes={}
            for p in (ROOT/'rapidtinyrec').glob('*.py'):
                target=code/'rapidtinyrec'/p.name;target.write_bytes(p.read_bytes())
                hashes['rapidtinyrec/'+p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
            record={'sha256':hashes};verify_source(code,record)
            key='rapidtinyrec/precision_allocation.py';original=(code/key).read_bytes();(code/key).write_bytes(original+b' ')
            with self.assertRaises(AssertionError):verify_source(code,record)
            (code/key).write_bytes(original);del hashes[key]
            with self.assertRaises(KeyError):verify_source(code,record)
    def test_traversal_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            code=Path(temp);(code/'file').write_text('x');(code/'link').symlink_to(code/'file')
            for name in ('../file','/file','link'):
                with self.assertRaises(AssertionError):verify_source(code,{'sha256':{name:hashlib.sha256(b'x').hexdigest()}})

if __name__=='__main__':unittest.main()
