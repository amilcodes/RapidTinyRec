import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from rapidtinyrec import serving_allocation as a
from rapidtinyrec import serving_policy as policy
from rapidtinyrec.serving_contract import load_protocol, PROTOCOL, PROTOCOL_SHA256, PROFILE
from rapidtinyrec.serving_initialize import lane_initializer

ROOT=Path(__file__).resolve().parents[1]


class AllocationTests(unittest.TestCase):
    def setUp(self):
        self.protocol=load_protocol(ROOT/PROTOCOL)
        self.remote=a.REMOTE_ROOT+'/'+a.PROJECT+'/exact-serving-fixture'
        resources=dict(self.protocol['resources'],setup=['module purge'],command=a.COMMAND)
        names=[a.CONFIG,PROTOCOL,a.COMMAND[1],'source-provenance.json','rapidtinyrec/expected.json']
        self.prepared=dict(run_id='exact-serving-fixture',project=a.PROJECT,profile=PROFILE,ssh_target=a.SSH_TARGET,
            remote_dir=self.remote,job_id=None,submission='prepared',resources=resources,files=names,
            sha256={name:PROTOCOL_SHA256 if name==PROTOCOL else 'a'*64 for name in names})
        self.config=dict(project=a.PROJECT,ssh_target=a.SSH_TARGET,remote_root=a.REMOTE_ROOT,profiles={PROFILE:resources})
        self.env=dict(SLURM_JOB_ID='12345',OSCAR_RUN_DIR=self.remote,OSCAR_OUTPUT_DIR=self.remote+'/outputs')
        self.raw=('JobId=12345 JobState=RUNNING Partition=gpu QOS=norm-gpu TimeLimit=00:30:00 NumCPUs=4 NumNodes=1 '
            'Command='+self.remote+'/job.sh WorkDir='+self.remote+' AllocTRES=cpu=4,mem=32G,node=1,gres/gpu=1 '
            'TresPerNode=gres/gpu:nvidia_rtx_a6000:1 NodeList=gpu-fixture')
        self.fields=a.scheduler_fields(self.raw)

    def validate(self):return a.validate_manifest(self.prepared,self.config,self.protocol,'12345\n',self.env,self.fields)

    def records(self):
        manifest=json.dumps(self.prepared).encode();job=b'12345\n'
        allocated=copy.deepcopy(self.prepared);allocated.update(submission='allocated_observation',job_id='12345')
        allocated['allocation_identity']=dict(schema='serving-allocated-identity-v1',requires_local_oscar_reconciliation=True,
            numerical_environment={key:None for key in policy.POLICY_ENV},
            uploaded_manifest_sha256=hashlib.sha256(manifest).hexdigest(),sbatch_job_id_file_sha256=hashlib.sha256(job).hexdigest(),
            scheduler_command=['scontrol','show','job','-o','12345'],scheduler_stdout=self.raw,scheduler_fields=self.fields,
            observed_source_sha256=self.prepared['sha256'],config=self.config,protocol_sha256=PROTOCOL_SHA256,started_ns=1,verified_ns=2)
        actual=copy.deepcopy(self.prepared);actual.update(submission='submitted',job_id='12345')
        return allocated,actual,manifest,job,self.protocol

    def test_exact_contract_and_metadata_reconciliation(self):
        self.assertEqual(self.validate(),'12345')
        result=a.reconcile_record(*self.records());self.assertTrue(result['matched'])
        self.assertFalse(result['advancement_ready']);self.assertFalse(result['model_executed'])
        self.assertEqual([p['resident_lanes'] for p in self.protocol['phases']],[1,2,2,1])
        self.assertEqual(self.protocol['total_lane_initializations'],6)
        self.assertFalse(self.protocol['deadline']['budget_validated'])

    def test_changed_protocol_bytes_and_source_inventory_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'protocol.json';path.write_bytes((ROOT/PROTOCOL).read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'protocol changed'):load_protocol(path)
        for mode in ('missing','duplicate','traversal','digest','protocol'):
            self.setUp()
            if mode=='missing':self.prepared['files'].remove(a.CONFIG);del self.prepared['sha256'][a.CONFIG]
            if mode=='duplicate':self.prepared['files'].append(a.CONFIG)
            if mode=='traversal':self.prepared['files'].append('../x');self.prepared['sha256']['../x']='a'*64
            if mode=='digest':self.prepared['sha256'][a.CONFIG]='x'
            if mode=='protocol':self.prepared['sha256'][PROTOCOL]='a'*64
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.validate()

    def test_job_environment_and_resource_types_fail_closed(self):
        for mode in ('job','run','output','prepared','bool','project'):
            self.setUp()
            if mode=='job':self.env['SLURM_JOB_ID']='54321'
            if mode=='run':self.prepared['run_id']='../escape'
            if mode=='output':self.env['OSCAR_OUTPUT_DIR']='/other'
            if mode=='prepared':self.prepared['job_id']='12345'
            if mode=='bool':self.prepared['resources']['gpus']=True
            if mode=='project':self.prepared['project']='precision-full500'
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.validate()

    def test_wrong_scheduler_fields_and_duplicate_resources_rejected(self):
        for key,value in [('JobState','COMPLETED'),('TimeLimit','01:00:00'),('NumCPUs','8'),('Command','other'),
                          ('TresPerNode','gres/gpu:a100:1'),('AllocTRES','cpu=4,mem=32G,node=1,gres/gpu=2'),
                          ('AllocTRES','cpu=4,mem=32G,node=1,gres/gpu=1,cpu=4'),
                          ('AllocTRES','cpu=4,mem=32G,node=1,gres/gpu=1,gres/gpu:a100=1')]:
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                a.validate_scheduler(dict(self.fields,**{key:value}),'12345',self.remote)
        with self.assertRaisesRegex(ValueError,'Duplicate'):a.scheduler_fields(self.raw+' JobId=12345')
        a.validate_scheduler(dict(self.fields,AllocTRES=self.fields['AllocTRES'].replace('32G','32768M')),'12345',self.remote)

    def test_changed_observed_or_terminal_evidence_rejected(self):
        for mode in ('job','hash','scheduler','source','clock'):
            args=list(self.records());identity=args[0]['allocation_identity']
            if mode=='job':args[1]['job_id']='54321'
            if mode=='hash':args[2]+=b' '
            if mode=='scheduler':identity['scheduler_fields']=dict(self.fields,NumCPUs='8')
            if mode=='source':identity['observed_source_sha256']={}
            if mode=='clock':identity['verified_ns']=False
            with self.subTest(mode=mode),self.assertRaises(ValueError):a.reconcile_record(*args)

    def test_observation_uses_live_scheduler_and_preserves_original_submission_files(self):
        with tempfile.TemporaryDirectory() as temp:
            temp=str(Path(temp).resolve())
            run=Path(temp)/a.PROJECT/'exact-serving-fixture';code=run/'code';output=run/'outputs'
            code.mkdir(parents=True);output.mkdir()
            for original in (ROOT/'rapidtinyrec').rglob('*'):
                if not original.is_file() or '__pycache__' in original.parts:continue
                target=code/original.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(original.read_bytes())
            protocol_path=code/PROTOCOL;protocol_path.parent.mkdir(parents=True);protocol_path.write_bytes((ROOT/PROTOCOL).read_bytes())
            config=copy.deepcopy(self.config);config['remote_root']=temp
            (code/a.CONFIG).write_text(json.dumps(config))
            (code/a.COMMAND[1]).parent.mkdir();(code/a.COMMAND[1]).write_text('# fixture only')
            (code/'source-provenance.json').write_text('{}')
            prepared=copy.deepcopy(self.prepared);prepared['remote_dir']=str(run)
            prepared['sha256']={p.relative_to(code).as_posix():a.digest(p) for p in code.rglob('*') if p.is_file()}
            prepared['files']=sorted(prepared['sha256'])
            manifest=json.dumps(prepared).encode();(run/'source-manifest.json').write_bytes(manifest);(run/'job-id').write_bytes(b'12345\n')
            raw=self.raw.replace(self.remote,str(run))
            env=dict(SLURM_JOB_ID='12345',OSCAR_RUN_DIR=str(run),OSCAR_OUTPUT_DIR=str(output))
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(a,'REMOTE_ROOT',temp));stack.enter_context(patch.dict(os.environ,env,clear=True))
                stack.enter_context(patch.object(sys,'dont_write_bytecode',True))
                query=stack.enter_context(patch.object(a.subprocess,'run',return_value=types.SimpleNamespace(stdout=raw)))
                for name,module in list(sys.modules.items()):
                    if name=='rapidtinyrec' or name.startswith('rapidtinyrec.'):
                        origin=getattr(module,'__file__',None)
                        if origin:stack.enter_context(patch.object(module,'__file__',str(code/Path(origin).resolve().relative_to(ROOT))))
                observed=a.observe_allocation(run)
                self.assertEqual(observed['job_id'],'12345');self.assertEqual(observed['allocation_identity']['observed_source_sha256'],prepared['sha256'])
                query.assert_called_once_with(['scontrol','show','job','-o','12345'],check=True,text=True,capture_output=True,timeout=10)
            self.assertEqual((run/'source-manifest.json').read_bytes(),manifest)
            self.assertEqual((run/'job-id').read_bytes(),b'12345\n');self.assertEqual(list(output.iterdir()),[])

    def test_optimized_python_cannot_disable_allocation_guard(self):
        code="from rapidtinyrec.serving_allocation import validate_scheduler; validate_scheduler({},'123','/bad')"
        result=subprocess.run([sys.executable,'-O','-B','-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn('Scheduler allocation/execution identity differs',result.stderr)


class SourceTests(unittest.TestCase):
    def test_recursive_uploaded_package_hashes_aliases_and_bytecode(self):
        with tempfile.TemporaryDirectory() as temp:
            code=Path(temp);hashes={}
            for original in (ROOT/'rapidtinyrec').rglob('*'):
                if not original.is_file() or '__pycache__' in original.parts:continue
                rel=original.relative_to(ROOT);target=code/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(original.read_bytes())
                hashes[rel.as_posix()]=a.digest(target)
            with contextlib.ExitStack() as stack:
                for name,module in list(sys.modules.items()):
                    if name=='rapidtinyrec' or name.startswith('rapidtinyrec.'):
                        origin=getattr(module,'__file__',None)
                        if origin:stack.enter_context(patch.object(module,'__file__',str(code/Path(origin).resolve().relative_to(ROOT))))
                self.assertEqual(a.verify_source(code,{'sha256':hashes}),hashes)
                nested=code/'rapidtinyrec/source/LogitProcessor.py';saved=nested.read_bytes();nested.write_bytes(saved+b' ')
                with self.assertRaisesRegex(ValueError,'bytes differ'):a.verify_source(code,{'sha256':hashes})
                nested.write_bytes(saved)
                extra=code/'rapidtinyrec/unreviewed.py';extra.write_text('# extra')
                with self.assertRaisesRegex(ValueError,'Undeclared'):a.verify_source(code,{'sha256':hashes})
                extra.unlink();bytecode=code/'rapidtinyrec/stale.pyc';bytecode.write_bytes(b'pyc')
                with self.assertRaisesRegex(ValueError,'Bytecode'):a.verify_source(code,{'sha256':hashes})
                bytecode.unlink();link=code/'rapidtinyrec/alias.py';os.link(nested,link)
                with self.assertRaisesRegex(ValueError,'Aliased'):a.verify_source(code,{'sha256':hashes})
                link.unlink();link.symlink_to(nested)
                with self.assertRaisesRegex(ValueError,'Linked'):a.verify_source(code,{'sha256':hashes})

    def test_package_outside_uploaded_tree_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'Running package'):a.verify_source(temp,{'sha256':{}})


class PolicyTests(unittest.TestCase):
    def backend(self):
        ns=types.SimpleNamespace
        return ns(__version__='2.14.0+cu126',version=ns(cuda='12.6',git_version='08187d9e0fba026dc8217405802ab5381dc88d90'),
            cuda=ns(is_available=lambda:True,device_count=lambda:1,get_device_name=lambda i:'NVIDIA RTX A6000'),
            backends=ns(cuda=ns(matmul=ns(allow_tf32=False),preferred_blas_library=lambda:'_BlasBackend.Cublas'),cudnn=ns(allow_tf32=False)),
            get_num_threads=lambda:2,get_float32_matmul_precision=lambda:'highest')

    def test_overrides_rejected_even_when_nominally_disabled(self):
        policy.validate_environment({})
        for name in policy.FORBIDDEN_ENV:
            with self.subTest(name=name),self.assertRaises(ValueError):policy.validate_environment({name:'0'})
        with self.assertRaises(ValueError):policy.validate_environment({'PYTORCH_TUNABLEOP_ENABLED':'1'})

    def test_pinned_runtime_math_and_backend(self):
        backend=self.backend();policy.validate_runtime(backend,'4.55.4');policy.validate_state(policy.policy_state(backend))
        for mode in ('cuda','blas','math','threads'):
            backend=self.backend()
            if mode=='cuda':backend.version.cuda='13.0'
            if mode=='blas':backend.backends.cuda.preferred_blas_library=lambda:'_BlasBackend.Cublaslt'
            if mode=='math':backend.get_float32_matmul_precision=lambda:'high'
            if mode=='threads':backend.get_num_threads=lambda:4
            with self.subTest(mode=mode),self.assertRaises((ValueError,RuntimeError)):
                policy.validate_runtime(backend,'4.55.4');policy.validate_state(policy.policy_state(backend))

    def test_initializer_environment_and_live_allocation_precede_torch(self):
        with patch.dict(os.environ,{'SLURM_JOB_ID':'123','NVIDIA_TF32_OVERRIDE':'0'},clear=True):
            with self.assertRaisesRegex(ValueError,'override'):lane_initializer('/bad','/bad','/bad')
        with patch.dict(os.environ,{'SLURM_JOB_ID':'123'},clear=True):
            with self.assertRaisesRegex(RuntimeError,'Live allocation'):lane_initializer('/bad','/bad','/bad')
        with patch.dict(os.environ,{'SLURM_JOB_ID':'123','OSCAR_RUN_DIR':'/bad'},clear=True),patch.object(a,'observe_allocation',side_effect=ValueError('live proof failed')) as observe:
            with self.assertRaisesRegex(ValueError,'live proof failed'):lane_initializer('/bad','/bad','/bad')
            observe.assert_called_once_with('/bad')
