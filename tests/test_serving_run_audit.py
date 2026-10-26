import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rapidtinyrec import serving_run_audit as a
from rapidtinyrec import serving_setup_audit as setup
from rapidtinyrec.serving_contract import PROTOCOL
import test_serving_allocation as allocation_tests
from test_serving_disk_audit import build, read, write
from test_serving_worker_audit import wrap
from test_serving_phase_cli import identity


class RunAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.run=Path(self.tmp.name).resolve();self.out=self.run/'outputs'
        self.roots,self.history,expected=build(self.out);wrap(self.roots)
        original=setup.read_unique
        p=patch.object(setup,'read_unique',side_effect=lambda path:expected if Path(path).name=='expected.json' else original(path))
        p.start();self.addCleanup(p.stop)
        allocation=allocation_tests.AllocationTests();allocation.setUp();self.allocation=allocation
        protocol=copy.deepcopy(allocation.protocol)
        protocol['cohort'].update(users=read(self.roots[0]/'setup/initialization-provenance.json')['users'],
            full_users_sha256=expected['users_sha256'],release_sha256=expected['release_sha256'],
            historical_manifest_file_sha256=hashlib.sha256((self.history/'manifest.json').read_bytes()).hexdigest())
        p=patch.object(a,'load_protocol',return_value=protocol);p.start();self.addCleanup(p.stop)
        p=patch.object(a.time,'perf_counter_ns',return_value=4000000);p.start();self.addCleanup(p.stop)
        code=self.run/'code';write(code/'rapidtinyrec/expected.json',expected)
        write(code/'oscar.serving-throughput.json',allocation.config);write(code/'source-provenance.json',{'synthetic':True})
        for name in ['rapidtinyrec/source/LogitProcessor.py','jobs/serving-throughput.sh','rapidtinyrec/serving_runner.py','rapidtinyrec/serving_run_audit.py','rapidtinyrec/serving_terminal_audit.py']:
            target=code/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text('# synthetic CPU fixture\n')
        target=code/PROTOCOL;target.parent.mkdir(parents=True);target.write_bytes((Path(__file__).resolve().parents[1]/PROTOCOL).read_bytes())
        allocation_tests.write_provenance(code)
        allocation.prepared['files']=sorted(p.relative_to(code).as_posix() for p in code.rglob('*') if p.is_file())
        allocation.prepared['sha256']={name:hashlib.sha256((code/name).read_bytes()).hexdigest() for name in allocation.prepared['files']}
        allocated,_,manifest,job,_=allocation.records();(self.run/'source-manifest.json').write_bytes(manifest);(self.run/'job-id').write_bytes(job)
        allocated['allocation_identity'].update(started_ns=1,verified_ns=1)
        coord=self.out/'coordinator';write(coord/'allocation-observation.json',allocated)
        self.parent=identity(100,10,100);remote=allocation.remote
        self.python='/pinned/python';self.release='/release';self.historical='/historical'
        command=[self.python,'-B','-m','rapidtinyrec.serving_runner','--coordinator','--run-root',remote,'--release-root',self.release,'--historical',self.historical]
        outer=dict(schema='serving-supervisor-start-v1',global_started_ns=1,deadline_ns=1+1620*10**9,
                   supervisor_pid=10,cwd=remote+'/code',command=command)
        write(self.out/'supervision/supervisor-start.json',outer)
        write(self.out/'supervision/coordinator-launch.json',dict(pid=100,process_group=100,identity=self.parent,launched_ns=1,observed_ns=1))
        (self.out/'supervision/coordinator.log').write_text('currently open stdout')
        write(coord/'entry.json',dict(identity=self.parent,process_group=100,session=100,started_ns=1,supervisor_start=outer))
        for i,root in enumerate(self.roots):
            provenance=read(root/'setup/initialization-provenance.json')
            provenance.update(cuda_version='12.6',device_name='NVIDIA RTX A6000',device_count=1,numerical_policy=dict(
                matmul_tf32=False,cudnn_tf32=False,intraop_threads=2,float32_matmul_precision='highest',blas_backend='_BlasBackend.Cublas'))
            write(root/'setup/initialization-provenance.json',provenance)
            observation=copy.deepcopy(allocated);observation['allocation_identity'].update(started_ns=i*1000000+11,verified_ns=i*1000000+12)
            write(root/'allocation-observation.json',observation)
            persisted=read(root/'allocation-persisted.json');persisted['sha256']=hashlib.sha256((root/'allocation-observation.json').read_bytes()).hexdigest();write(root/'allocation-persisted.json',persisted)
            entry=read(root/'worker-entry.json');ticket=entry['launch'];write(self.out/'launch'/f'phase_{i}.json',ticket)
            command=[self.python,'-B','-m','rapidtinyrec.serving_phase_cli','--run-root',remote,'--phase',str(i),'--release-root',self.release,'--historical',self.historical]
            spawn=dict(phase=i,command=command,returncode=None,error=None,pid=200+i,identity=entry['identity'],
                       observed_ns=i*1000000+5,process_group=100,session=100)
            write(coord/f'phase_{i}-spawn.json',spawn)
            write(coord/f'phase_{i}-exit.json',dict(spawn,returncode=0,exit_observed_ns=i*1000000+300050))
            for when in ('before','after'):write(coord/f'phase_{i}-cache-{when}.json',read(root/f'cache-{when}.json'))
            write(coord/f'phase_{i}-binding.json',{'online_passed':False})
            (coord/f'phase_{i}.log').write_text('synthetic phase stdout')

    def call(self):return a.audit_pre_exit(self.run,self.history)

    def test_all_original_bytes_and_parent_records_reconstructed_before_exit(self):
        report=self.call();self.assertTrue(report['evidence_gates_passed']);self.assertFalse(report['advancement_ready'])
        self.assertTrue(report['terminal_reconciliation_pending']);self.assertFalse(report['original_submission_record_reconciled'])
        self.assertIn('run/source-manifest.json',report['input_sha256'])
        self.assertIn('run/code/rapidtinyrec/source/LogitProcessor.py',report['input_sha256'])
        self.assertNotIn('run/outputs/supervision/coordinator.log',report['input_sha256'])
        self.assertFalse((self.out/'coordinator/final-audit.json').exists())

    def test_original_manifest_and_actual_source_byte_drift_rejected(self):
        path=self.run/'code/rapidtinyrec/source/LogitProcessor.py';original=path.read_bytes();path.write_text('changed')
        with self.assertRaisesRegex(ValueError,'source bytes changed'):self.call()
        path.write_bytes(original);path=self.run/'source-manifest.json';path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'Submitted bytes differ'):self.call()

    def test_live_scheduler_observation_is_revalidated(self):
        path=self.out/'coordinator/allocation-observation.json';row=read(path)
        row['allocation_identity']['scheduler_stdout']=row['allocation_identity']['scheduler_stdout'].replace('NumCPUs=4','NumCPUs=8')
        row['allocation_identity']['scheduler_fields']['NumCPUs']='8';write(path,row)
        with self.assertRaisesRegex(ValueError,'Scheduler allocation'):self.call()

    def test_parent_spawn_birth_or_worker_command_drift_rejected(self):
        path=self.out/'coordinator/phase_0-spawn.json';original=read(path);changed=copy.deepcopy(original);changed['identity']['start_ticks']+=1;write(path,changed)
        with self.assertRaisesRegex(ValueError,'spawn/exit'):self.call()
        write(path,original)
        for name in ('spawn','exit'):
            path=self.out/f'coordinator/phase_0-{name}.json';row=read(path);row['command'][1]='-O';write(path,row)
        with self.assertRaisesRegex(ValueError,'worker command'):self.call()

    def test_parent_cache_and_actual_ticket_must_match_raw_worker(self):
        path=self.out/'coordinator/phase_1-cache-after.json';original=read(path);write(path,{})
        with self.assertRaisesRegex(ValueError,'cache inventory'):self.call()
        write(path,original);path=self.out/'launch/phase_1.json';row=read(path);row['launch_token']='a'*32;write(path,row)
        with self.assertRaisesRegex(ValueError,'launch/parent'):self.call()

    def test_no_self_report_or_terminal_record_in_pre_exit_inputs(self):
        write(self.out/'coordinator/final-audit.json',{'evidence_gates_passed':True})
        with self.assertRaisesRegex(ValueError,'pre-exit coordinator'):self.call()

    def test_outer_command_and_global_start_binding_rejected(self):
        path=self.out/'supervision/supervisor-start.json';row=read(path);row['command'][3]='other.runner';write(path,row)
        entry=read(self.out/'coordinator/entry.json');entry['supervisor_start']=row;write(self.out/'coordinator/entry.json',entry)
        with self.assertRaisesRegex(ValueError,'exact serving coordinator'):self.call()

    def test_unexpected_or_aliased_live_stdout_not_excluded(self):
        path=self.out/'supervision/coordinator.log';path.unlink();path.symlink_to(self.out/'coordinator/phase_0.log')
        with self.assertRaisesRegex(ValueError,'live coordinator stdout'):self.call()

    def test_observed_numerical_environment_and_runtime_required(self):
        path=self.out/'coordinator/allocation-observation.json';original=read(path);changed=copy.deepcopy(original)
        changed['allocation_identity']['numerical_environment']['NVIDIA_TF32_OVERRIDE']='0';write(path,changed)
        with self.assertRaisesRegex(ValueError,'override'):self.call()
        write(path,original);path=self.roots[0]/'setup/initialization-provenance.json';row=read(path)
        row['cuda_version']='12.8';write(path,row)
        with self.assertRaisesRegex(ValueError,'runtime/device'):self.call()

    def test_complete_wrapper_cannot_rescue_failed_gpu_overlap(self):
        path=self.roots[2]/'execution/profile/trace.json';trace=read(path)
        for event in trace['traceEvents']:
            lane=((event['args']['correlation']-1)//4%2 if event['cat']!='user_annotation'
                  else int(event['name'].split('/')[1].split('_')[1]))
            if lane==1:event['ts']+=40
        write(path,trace);report=self.call()
        self.assertFalse(report['evidence_gates_passed'])
        self.assertFalse(report['gates']['actual_gpu_overlap'])
        self.assertTrue(report['allocation_binding']['matched'])
        self.assertFalse(report['advancement_ready'])


if __name__=='__main__':unittest.main()
