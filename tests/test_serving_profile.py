from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

from rapidtinyrec.serving_lanes import LaneServer,BurstExecutionError
from rapidtinyrec.serving_profile import profile_after_measurements
from test_serving_measurement_audit import fixture as measurements
from test_serving_overlap_audit import fixture as trace
from test_serving_setup_audit import raw


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.output=Path(self.tmp.name)/'profile';self.events=[];self.fail=False;self.drift=False
        def initialize(index):
            identity=dict(owner_thread_id=threading.get_ident(),device='cuda:0',stream_id=index+100,failed=False)
            return SimpleNamespace(context=SimpleNamespace(identity=lambda:dict(identity)))
        def run(state,user):
            if self.fail:raise RuntimeError('runtime failed')
            start=time.perf_counter_ns();r=raw()
            if self.drift:r['beam_indices'][0][0]=1
            end=time.perf_counter_ns()
            return dict(user,generation_started_ns=start,generation_finished_ns=end,generation_elapsed_s=(end-start)/1e9,raw=r)
        self.server=LaneServer(initialize,run,lane_count=2);self.server.freeze();self.addCleanup(self.server.close)
        self.original=self.server.run;events=self.events
        @contextmanager
        def marker(name):
            events.append(('marker',name,threading.get_ident()));yield
        def config(**kwargs):events.append(('config',kwargs));return kwargs
        class Profiler:
            def __init__(self,**kwargs):events.append(('profiler',kwargs))
            def __enter__(self):events.append('enter');return self
            def __exit__(self,*args):events.append('exit')
            def export_chrome_trace(self,path):events.append('export');Path(path).write_text(json.dumps(trace()[0]))
        self.backend=SimpleNamespace(profiler=SimpleNamespace(_ExperimentalConfig=config,profile=Profiler,record_function=marker,ProfilerActivity=SimpleNamespace(CPU='cpu',CUDA='cuda')))
        p,self.users,_=measurements();self.phase=p[1]

    def call(self):return profile_after_measurements(self.server,self.users,self.phase,self.output,backend=self.backend)

    def test_all_thread_config_owner_markers_restored_callback_and_saved_evidence(self):
        result=self.call();self.assertTrue(result['numerical_passed']);self.assertTrue(result['overlap']['gpu_overlap_proven'])
        self.assertFalse(result['performance_sample']);self.assertFalse(result['advancement_ready'])
        self.assertIs(self.server.run,self.original)
        self.assertEqual(self.events[0],('config',dict(profile_all_threads=True,custom_profiler_config='')))
        markers=[v for v in self.events if isinstance(v,tuple) and v[0]=='marker']
        self.assertEqual(len(markers),4);self.assertEqual(len({v[2] for v in markers}),2)
        self.assertNotIn(threading.get_ident(),{v[2] for v in markers})
        self.assertLess(self.events.index('exit'),self.events.index('export'))
        self.assertEqual({p.name for p in self.output.iterdir()},{'profile-start.json','profile-records.json','trace.json','profile-audit.json'})
        with self.assertRaises(FileExistsError):self.call()

    def test_failed_burst_drained_restores_callback_and_keeps_trace(self):
        self.fail=True
        with self.assertRaises(BurstExecutionError):self.call()
        self.assertIs(self.server.run,self.original);self.assertTrue(self.server.failed)
        self.assertTrue((self.output/'trace.json').exists());self.assertTrue((self.output/'profile-failed-records.json').exists())
        self.assertTrue((self.output/'profile-failure.json').exists())

    def test_incomplete_measurement_never_starts_profiler(self):
        self.phase['secondary'].pop()
        with self.assertRaisesRegex(ValueError,'Complete measurements'):self.call()
        self.assertFalse(self.output.exists());self.assertEqual(self.events,[])

    def test_profile_ancestry_drift_is_saved_as_failed(self):
        self.drift=True;r=self.call();self.assertFalse(r['numerical_passed']);self.assertFalse(r['ancestry_stable'])
        self.assertTrue((self.output/'profile-records.json').exists())
