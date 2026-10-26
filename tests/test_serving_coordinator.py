import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from rapidtinyrec import serving_coordinator as c
from rapidtinyrec.serving_contract import PROTOCOL, load_protocol
from rapidtinyrec.serving_process import observe_cache
from test_serving_phase_cli import identity


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.run=Path(self.tmp.name).resolve();(self.run/'outputs/supervision').mkdir(parents=True)
        self.parent=identity(100,10,100);self.protocol=load_protocol(Path(__file__).resolve().parents[1]/PROTOCOL)
        now=time.perf_counter_ns();outer=dict(supervisor_pid=10,global_started_ns=now-1000,deadline_ns=now-1000+1620*10**9)
        (self.run/'outputs/supervision/supervisor-start.json').write_text(json.dumps(outer))
        self.launched=[];self.fail_phase=None;self.corrupt=False;self.escape=False;self.final_called=False
        def ident(pid=None):return self.parent if pid is None else identity(pid,100,pid)
        owner=self
        class Child:
            def __init__(self, command, **kw):
                phase=int(command[command.index('--phase')+1]);owner.launched.append(phase)
                self.pid=200+phase;self.returncode=None;self.phase=phase
                owner.assertFalse(kw['start_new_session']);owner.assertEqual(command[1:4],['-B','-m','rapidtinyrec.serving_phase_cli'])
                launch=json.loads((owner.run/f'outputs/launch/phase_{phase}.json').read_text())
                owner.assertEqual(kw['env']['SERVING_LAUNCH_TOKEN'],launch['launch_token'])
                root=owner.run/f'outputs/phase_{phase}';root.mkdir()
                cache=Path(launch['cache']['root']);before=observe_cache(cache)
                start=time.perf_counter_ns();(cache/'compiled').write_text(str(phase));after=observe_cache(cache)
                entry=dict(phase=phase,identity=ident(self.pid),launch=launch,started_ns=start,process_group=100,session=100)
                end=dict(phase=phase,identity=ident(self.pid),complete=True,execution_complete=True,error=None,
                         service_closed_ns=time.perf_counter_ns(),finished_ns=time.perf_counter_ns())
                for file,value in [('worker-entry.json',entry),('worker-exit.json',end),('cache-before.json',before),('cache-after.json',after)]:
                    (root/file).write_text(json.dumps(value))
                if owner.corrupt and phase==1:(owner.run/'cache/lanes-1/compiled').write_text('changed between phases')
            def wait(self,**kw):self.returncode=7 if self.phase==owner.fail_phase else 0;return self.returncode
            def kill(self):self.returncode=-9
        for name,value in [('process_identity',ident),('load_protocol',lambda p:self.protocol),('observe_allocation',lambda p:{}),('verify_source',lambda *a:None)]:
            p=patch.object(c,name,side_effect=value);p.start();self.addCleanup(p.stop)
        p=patch.object(c.subprocess,'Popen',side_effect=Child);p.start();self.addCleanup(p.stop)
        for name,value in [('getpgrp',lambda:100),('getsid',lambda pid:100),('getpgid',lambda pid:999 if self.escape else 100),('waitid',lambda *a:None)]:
            p=patch.object(c.os,name,side_effect=value);p.start();self.addCleanup(p.stop)
        p=patch.object(sys,'dont_write_bytecode',True);p.start();self.addCleanup(p.stop)
        p=patch.dict(os.environ,dict(OSCAR_RUN_DIR=str(self.run),OSCAR_OUTPUT_DIR=str(self.run/'outputs')),clear=True);p.start();self.addCleanup(p.stop)

    def final(self,run,historical):
        self.final_called=True;self.assertEqual(self.launched,[0,1,2,3])
        return dict(evidence_gates_passed=False,advancement_ready=False)

    def call(self):return c.coordinate(self.run,self.run/'release',self.run/'history',final_auditor=self.final)

    def test_four_fresh_phases_cache_family_continuity_and_failed_gate_preserved(self):
        report=self.call();self.assertFalse(report['evidence_gates_passed']);self.assertTrue(self.final_called)
        for i in range(4):
            root=self.run/'outputs/coordinator'
            parent=json.loads((root/f'phase_{i}-exit.json').read_text())
            self.assertEqual(parent['identity']['pid'],200+i)
            self.assertEqual(parent['returncode'],0)
        result=json.loads((root/'exit.json').read_text());self.assertTrue(result['complete']);self.assertFalse(result['advancement_ready'])
        self.assertEqual(json.loads((root/'phase_2-cache-before.json').read_text()),json.loads((root/'phase_1-cache-after.json').read_text()))
        self.assertEqual(json.loads((root/'phase_3-cache-before.json').read_text()),json.loads((root/'phase_0-cache-after.json').read_text()))

    def test_worker_failure_never_launches_later_phase_or_final_audit(self):
        self.fail_phase=1
        with self.assertRaisesRegex(ValueError,'Worker failed'):self.call()
        self.assertEqual(self.launched,[0,1]);self.assertFalse(self.final_called)
        result=json.loads((self.run/'outputs/coordinator/exit.json').read_text())
        self.assertFalse(result['complete']);self.assertEqual(result['phases_completed'],[0])

    def test_changed_idle_family_cache_blocks_reuse(self):
        self.corrupt=True
        with self.assertRaisesRegex(ValueError,'same-family cache'):self.call()
        self.assertEqual(self.launched,[0,1,2]);self.assertFalse(self.final_called)

    def test_escaped_child_group_stops_schedule(self):
        self.escape=True
        with self.assertRaisesRegex(ValueError,'escaped supervisor'):self.call()
        self.assertEqual(self.launched,[0]);self.assertFalse(self.final_called)
        self.assertTrue((self.run/'outputs/coordinator/phase_0-exit.json').exists())

    def test_final_auditor_required_before_any_preflight(self):
        with patch.object(c,'observe_allocation',side_effect=AssertionError('must not preflight')):
            with self.assertRaisesRegex(ValueError,'on-disk run auditor'):
                c.coordinate(self.run,'release','history',final_auditor=None)
        self.assertFalse(self.launched)

    def test_reconciliation_rejects_birth_clock_and_cache_changes(self):
        self.call();out=self.run/'outputs';root=out/'coordinator';phase=out/'phase_0'
        args=[json.loads((out/'launch/phase_0.json').read_text()),json.loads((root/'phase_0-exit.json').read_text()),
              json.loads((phase/'worker-entry.json').read_text()),json.loads((phase/'worker-exit.json').read_text()),
              json.loads((root/'phase_0-cache-before.json').read_text()),json.loads((root/'phase_0-cache-after.json').read_text()),
              json.loads((phase/'cache-before.json').read_text()),json.loads((phase/'cache-after.json').read_text())]
        for which,mutate in [(1,lambda v:v['identity'].update(start_ticks=999)),
                             (3,lambda v:v.update(service_closed_ns=None)),
                             (7,lambda v:v.clear())]:
            changed=copy.deepcopy(args);mutate(changed[which])
            with self.subTest(which=which),self.assertRaises(ValueError):c.reconcile_worker(*changed)

    def test_final_audit_failure_keeps_completed_phases_but_not_run_completion(self):
        def fail(*args):raise ValueError('raw run evidence mismatch')
        with self.assertRaisesRegex(ValueError,'raw run evidence mismatch'):
            c.coordinate(self.run,'release','history',final_auditor=fail)
        result=json.loads((self.run/'outputs/coordinator/exit.json').read_text())
        self.assertEqual(result['phases_completed'],[0,1,2,3]);self.assertFalse(result['complete'])
        self.assertFalse((self.run/'outputs/coordinator/final-audit.json').exists())


if __name__=='__main__':unittest.main()
