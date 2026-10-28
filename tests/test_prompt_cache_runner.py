import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from rapidtinyrec.prompt_cache_runner import PROJECT,PROFILE,COMMAND,PROTOCOL,PROTOCOL_SHA256,allowance,validate_manifest,validate_audit_start
from rapidtinyrec.prompt_cache_protocol import load_protocol

ROOT=Path(__file__).resolve().parents[1]


class RunnerTests(unittest.TestCase):
    def manifest(self):
        protocol=load_protocol(ROOT/PROTOCOL)
        remote=f'/oscar/scratch/aagraw73/agent-runs/{PROJECT}/test-run'
        resources=dict(protocol['resources'],setup=['module purge'],command=COMMAND)
        files=[PROTOCOL,COMMAND[1],'oscar.prompt-cache-pilot.json','release-config.json',
               'baseline-raw-manifest.json','rapidtinyrec/audit_prompt_cache.py','rapidtinyrec/prompt_cache_runner.py']
        prepared=dict(submission='prepared',job_id=None,project=PROJECT,profile=PROFILE,run_id='test-run',remote_dir=remote,
                      resources=resources,ssh_target='aagraw73@sshcampus.ccv.brown.edu',files=files,
                      sha256={k:PROTOCOL_SHA256 if k==PROTOCOL else 'a'*64 for k in files})
        config=dict(project=PROJECT,profiles={PROFILE:copy.deepcopy(resources)},ssh_target=prepared['ssh_target'],
                    remote_root='/oscar/scratch/aagraw73/agent-runs')
        env=dict(SLURM_JOB_ID='20',OSCAR_RUN_DIR=remote,OSCAR_OUTPUT_DIR=remote+'/outputs')
        return prepared,'20',remote,protocol,config,env

    def test_uploaded_manifest_and_actual_job_binding(self):validate_manifest(*self.manifest())

    def test_wrong_experiment_source_scope_or_job_rejected(self):
        for key,value in (('profile','bf16-vs-tf32-pilot'),('project','other'),('job_id','20'),
                          ('remote_dir','/other'),('files',[])):
            args=self.manifest();args[0][key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):validate_manifest(*args)
        args=self.manifest();args[5]['SLURM_JOB_ID']='21'
        with self.assertRaises(AssertionError):validate_manifest(*args)

    def test_finite_inclusive_stage_budgets(self):
        self.assertEqual(allowance('generation',100,101),1500)
        self.assertEqual(allowance('audit',100,1650),70)
        self.assertEqual(allowance('audit',100,1720),0)
        for bad in (float('nan'),float('inf'),True,99):
            with self.assertRaises(AssertionError):allowance('audit',100,bad)

    def stages(self,root):
        start=dict(schema='prompt-cache-pilot-work-v1',job_id='20',protocol_sha256=PROTOCOL_SHA256,
                   started_monotonic=100,deadline_monotonic=1720)
        gen=dict(name='generation',started_monotonic=101,ended_monotonic=1200,timeout_s=1500,
                 within_deadline=True,child=dict(completed_normally=True))
        recovery=dict(name='recovery',started_monotonic=1201,ended_monotonic=1210,timeout_s=519,
                      within_deadline=True,child=dict(completed_normally=True))
        audit=dict(name='audit',started_monotonic=1211,timeout_s=120)
        for name,value in (('work-start',start),('generation-exit',gen),('recovery-exit',recovery),('audit-start',audit)):
            (root/f'prompt-cache-{name}.json').write_text(json.dumps(value))

    def test_audit_has_shorter_deadline_than_global_remaining(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.stages(root)
            self.assertEqual(validate_audit_start(root,dict(job_id='20'),1212)['effective_deadline_monotonic'],1331)
            with self.assertRaises(AssertionError):validate_audit_start(root,dict(job_id='20'),1331)

    def test_failed_stage_overlap_and_budget_extension_rejected(self):
        for name,key,value in (('generation-exit','within_deadline',False),('recovery-exit','started_monotonic',1199),
                               ('audit-start','timeout_s',121),('work-start','deadline_monotonic',1800)):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);self.stages(root);p=root/f'prompt-cache-{name}.json'
                entry=json.loads(p.read_text());entry[key]=value;p.write_text(json.dumps(entry))
                with self.assertRaises(AssertionError):validate_audit_start(root,dict(job_id='20'),1212)

    def test_cli_preserves_failure_and_refuses_overwrite_without_local_compute(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'failed.json'
            command=[sys.executable,'-m','rapidtinyrec.audit_prompt_cache']
            for name in ('root','historical','snapshot','run-record','release'):command+=['--'+name,str(root)]
            command+=['--output',str(out)]
            env={k:v for k,v in os.environ.items() if k!='SLURM_JOB_ID'}
            result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,1)
            saved=out.read_bytes();audit=json.loads(saved)
            self.assertFalse(audit['execution_audit_passed']);self.assertIn('Allocated prompt-cache audit only',audit['error'])
            repeated=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
            self.assertNotEqual(repeated.returncode,0);self.assertEqual(out.read_bytes(),saved)


if __name__=='__main__':unittest.main()
