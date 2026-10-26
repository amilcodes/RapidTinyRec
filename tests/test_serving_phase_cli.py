import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from rapidtinyrec import serving_phase_cli as worker
from rapidtinyrec import serving_process as process
from rapidtinyrec.serving_contract import PROTOCOL, PROTOCOL_SHA256, load_protocol


def identity(pid=20, parent=10, start=200):
    return dict(pid=pid,parent_pid=parent,start_ticks=start,boot_id='a'*8+'-'+ 'b'*4+'-'+ 'c'*4+'-'+ 'd'*4+'-'+ 'e'*12,
                hostname='compute',clock_ticks=100)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();(self.root/'outputs/launch').mkdir(parents=True)
        self.protocol=load_protocol(Path(__file__).resolve().parents[1]/PROTOCOL)
        self.parent=identity(10,1,100);self.child=identity();self.phase=1
        self.cache=self.root/'cache/lanes-2';self.cache.mkdir(parents=True)
        now=time.perf_counter_ns()
        self.launch=dict(schema='serving-phase-launch-v1',run_root=str(self.root),phase=1,
            protocol_sha256=PROTOCOL_SHA256,launch_token='f'*32,parent_identity=self.parent,
            global_started_ns=now-200,launch_started_ns=now-100,deadline_ns=now-200+1620*10**9,
            cache=dict(family='lanes-2',root=str(self.cache),reuse=False),process_group=10,session=10)
        self.ticket=self.root/'outputs/launch/phase_1.json';self.ticket.write_text(json.dumps(self.launch))
        self.out=self.root/'outputs/phase_1';self.events=[];self.failure=None;self.close_failure=False
        self.factory=NS(allocation_observation={'source':'observed'})
        self.server=NS(closed=False,lanes=[NS(thread=NS(is_alive=lambda:False))])
        def initialize(*args):
            self.events.append('factory');self.assertTrue((self.out/'worker-entry.json').exists())
            return self.factory
        @contextmanager
        def service(factory,callback,**kw):
            self.events.append('initialize_models')
            self.assertEqual(json.loads((self.out/'allocation-observation.json').read_text()),factory.allocation_observation)
            self.assertTrue((self.out/'allocation-persisted.json').exists())
            self.assertEqual(kw['lane_count'],2);self.assertEqual(kw['expected_graphs'],8)
            kw['lifecycle'].append({'operation':'initialize'})
            try:yield self.server,{'admission_events':[{'operation':'initialize'}]}
            finally:
                self.events.append('close');kw['lifecycle'].append({'operation':'close','success':not self.close_failure})
                self.server.closed=True
                if self.close_failure:raise RuntimeError('close failure')
        def execute(*args):
            self.events.append('execute');self.assertTrue((self.out/'preparation.json').exists())
            if self.failure:raise self.failure
            return {'complete':True,'gpu_overlap_proven':False,'profile_numerical_passed':False}
        for name,func in [('load_protocol',lambda p:self.protocol),('process_identity',lambda pid=None:self.child if pid is None else self.parent),
                          ('lane_initializer',initialize),('prepared_service',service),('run_prepared_phase',execute),
                          ('setup_auditor',lambda *a:object()),('verify_source',lambda *a:self.events.append('source'))]:
            p=patch.object(worker,name,side_effect=func);p.start();self.addCleanup(p.stop)
        p=patch.object(sys,'dont_write_bytecode',True);p.start();self.addCleanup(p.stop)
        for name in ('getpgrp','getsid'):
            p=patch.object(worker.os,name,return_value=10);p.start();self.addCleanup(p.stop)
        self.env=dict(OSCAR_RUN_DIR=str(self.root),OSCAR_OUTPUT_DIR=str(self.root/'outputs'),SERVING_LAUNCH_TOKEN='f'*32,
            TORCHINDUCTOR_CACHE_DIR=str(self.cache/'inductor'),TRITON_CACHE_DIR=str(self.cache/'triton'),CUDA_CACHE_PATH=str(self.cache/'cuda'))
        p=patch.dict(os.environ,self.env,clear=True);p.start();self.addCleanup(p.stop)

    def call(self):return worker.run_phase(self.root,1,self.root/'release',self.root/'history')

    def test_allocation_saved_before_models_and_close_saved_without_promoting_failed_gates(self):
        result=self.call();self.assertFalse(result['gpu_overlap_proven'])
        self.assertEqual(self.events,['factory','source','initialize_models','execute','close','source'])
        report=json.loads((self.out/'worker-exit.json').read_text())
        self.assertTrue(report['complete']);self.assertFalse(report['advancement_ready'])
        self.assertLess(report['service_closed_ns'],report['finished_ns'])
        self.assertEqual(json.loads((self.out/'lifecycle.json').read_text())[-1]['operation'],'close')
        with self.assertRaises(FileExistsError):self.call()

    def test_measurement_failure_preserves_shutdown_and_original_exception(self):
        self.failure=RuntimeError('raw measurement failure')
        with self.assertRaisesRegex(RuntimeError,'raw measurement failure'):self.call()
        report=json.loads((self.out/'worker-exit.json').read_text())
        self.assertFalse(report['complete']);self.assertFalse(report['execution_complete'])
        self.assertIsNone(report['service_closed_ns'])
        self.assertEqual(json.loads((self.out/'lifecycle.json').read_text())[-1]['operation'],'close')
        self.assertFalse((self.out/'cache-after.json').exists())

    def test_shutdown_failure_never_complete(self):
        self.close_failure=True
        with self.assertRaisesRegex(RuntimeError,'close failure'):self.call()
        report=json.loads((self.out/'worker-exit.json').read_text())
        self.assertFalse(report['complete']);self.assertTrue(report['execution_complete'])

    def test_preflight_failure_keeps_worker_failure_without_model_admission(self):
        with patch.object(worker,'lane_initializer',side_effect=ValueError('resource mismatch')):
            with self.assertRaisesRegex(ValueError,'resource mismatch'):self.call()
        self.assertFalse(self.events)
        self.assertEqual(json.loads((self.out/'lifecycle.json').read_text()),[])
        self.assertFalse(json.loads((self.out/'worker-exit.json').read_text())['complete'])

    def test_expired_or_wrong_parent_ticket_rejected_before_factory(self):
        for field,value in [('deadline_ns',1),('parent_identity',identity(11,1,100)),('launch_token','a'*32)]:
            with self.subTest(field=field):
                bad=copy.deepcopy(self.launch);bad[field]=value;self.ticket.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):self.call()
                self.assertFalse(self.out.exists());self.assertFalse(self.events)

    def test_wrong_cold_cache_and_environment_fail_before_model(self):
        (self.cache/'foreign').write_text('old cache')
        with self.assertRaisesRegex(ValueError,'Cold/reuse'):self.call()
        self.assertFalse(self.events);self.assertTrue((self.out/'cache-before.json').exists())

    def test_worker_module_import_and_help_do_not_import_torch(self):
        code="import sys; import rapidtinyrec.serving_phase_cli; assert 'torch' not in sys.modules"
        subprocess.run([sys.executable,'-B','-c',code],check=True,capture_output=True)

    def test_changed_source_after_factory_never_initializes_models(self):
        with patch.object(worker,'verify_source',side_effect=ValueError('changed source')):
            with self.assertRaisesRegex(ValueError,'changed source'):self.call()
        self.assertEqual(self.events,['factory'])
        self.assertTrue((self.out/'allocation-observation.json').exists())
        self.assertFalse(json.loads((self.out/'worker-exit.json').read_text())['complete'])

    def test_reused_cache_still_requires_correct_family_and_parent_birth(self):
        launch=copy.deepcopy(self.launch);launch['phase']=2;launch['cache']['reuse']=True
        process.validate_launch(launch,self.root,2,self.protocol,self.child,self.parent,'f'*32)
        changed=copy.deepcopy(self.parent);changed['start_ticks']+=1
        with self.assertRaisesRegex(ValueError,'birth identity'):
            process.validate_launch(launch,self.root,2,self.protocol,self.child,changed,'f'*32)
        launch['cache']['family']='lanes-1'
        with self.assertRaisesRegex(ValueError,'cache family'):
            process.validate_launch(launch,self.root,2,self.protocol,self.child,self.parent,'f'*32)

    def test_worker_rejects_escaped_group_before_factory(self):
        with patch.object(worker.os,'getpgrp',return_value=999):
            with self.assertRaisesRegex(ValueError,'inherit its supervised'):self.call()
        self.assertFalse(self.events);self.assertFalse(self.out.exists())


class ProcessTests(unittest.TestCase):
    def test_linux_birth_parser_with_parentheses_and_spaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'20').mkdir();(root/'sys/kernel/random').mkdir(parents=True)
            (root/'sys/kernel/random/boot_id').write_text(identity()['boot_id'])
            (root/'20/stat').write_text('20 (name with ) parens) S 10 '+'0 '*17+'200 '+'0 '*3)
            result=process.process_identity(20,proc=root)
            self.assertEqual((result['pid'],result['parent_pid'],result['start_ticks']),(20,10,200))
            (root/'20/stat').write_text('21 (wrong) S 10')
            with self.assertRaises(ValueError):process.process_identity(20,proc=root)

    def test_cache_inventory_hashes_and_rejects_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();(root/'inductor').mkdir();file=root/'inductor/kernel.py';file.write_text('cache')
            first=process.observe_cache(root);file.write_text('changed')
            self.assertNotEqual(first,process.observe_cache(root))
            os.link(file,root/'alias')
            with self.assertRaisesRegex(ValueError,'Aliased'):process.observe_cache(root)

    def test_cache_policy_rejects_alternate_roots_and_overrides(self):
        root=Path('/run/cache/lanes-1');env=dict(TORCHINDUCTOR_CACHE_DIR=str(root/'inductor'),
            TRITON_CACHE_DIR=str(root/'triton'),CUDA_CACHE_PATH=str(root/'cuda'))
        process.validate_cache_environment(root,env)
        for key,value in [('TRITON_CACHE_DIR','/shared'),('TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE','0')]:
            with self.subTest(key=key),self.assertRaises(ValueError):process.validate_cache_environment(root,dict(env,**{key:value}))


if __name__=='__main__':unittest.main()
