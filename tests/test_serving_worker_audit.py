import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rapidtinyrec import serving_disk_audit as audit
from rapidtinyrec import serving_setup_audit as setup
from rapidtinyrec.serving_contract import PROTOCOL_SHA256
from rapidtinyrec.serving_worker_audit import WORKER_FILES
from test_serving_disk_audit import build, read, write
import test_serving_allocation as allocation_tests
from test_serving_phase_cli import identity


def wrap(roots):
    allocation=allocation_tests.AllocationTests();allocation.setUp()
    parent=identity(100,10,100);last={}
    for phase,root in enumerate(roots):
        offset=phase*1000000;lanes=(1,2,2,1)[phase];family=f'lanes-{lanes}'
        raw=copy.deepcopy(allocation.records()[0])
        raw['allocation_identity'].update(started_ns=offset+11,verified_ns=offset+12)
        write(root/'allocation-observation.json',raw)
        write(root/'allocation-persisted.json',dict(schema='serving-allocation-persisted-v1',published_ns=offset+20,
            sha256=hashlib.sha256((root/'allocation-observation.json').read_bytes()).hexdigest()))
        cache=raw['remote_dir']+'/cache/'+family;reuse=phase in (2,3)
        launch=dict(schema='serving-phase-launch-v1',phase=phase,run_root=raw['remote_dir'],protocol_sha256=PROTOCOL_SHA256,
            parent_identity=parent,launch_token=f'{phase:032x}',global_started_ns=1,deadline_ns=1+1620*10**9,
            launch_started_ns=offset+2,process_group=100,session=100,cache=dict(family=family,root=cache,reuse=reuse))
        child=identity(200+phase,100,200+phase)
        write(root/'worker-entry.json',dict(schema='serving-worker-entry-v1',phase=phase,started_ns=offset+3,
            identity=child,launch=launch,process_group=100,session=100,cache_environment=dict(
                TORCHINDUCTOR_CACHE_DIR=cache+'/inductor',TRITON_CACHE_DIR=cache+'/triton',CUDA_CACHE_PATH=cache+'/cuda')))
        write(root/'worker-exit.json',dict(schema='serving-worker-exit-v1',phase=phase,identity=child,
            started_ns=offset+3,finished_ns=offset+300040,service_closed_ns=offset+300030,complete=True,execution_complete=True,error=None))
        events=copy.deepcopy(read(root/'preparation.json')['admission_events'])
        for lane in range(lanes):
            events.append(dict(operation='close',lane=lane,thread_id=10+lane,started_ns=offset+300010+lane*2,
                finished_ns=offset+300011+lane*2,success=True,error=None))
        write(root/'lifecycle.json',events)
        before=last.get(family,{})
        after={'inductor/compiled.py':dict(bytes=12,sha256=hashlib.sha256(str(phase).encode()).hexdigest())}
        write(root/'cache-before.json',before);write(root/'cache-after.json',after);last[family]=after


class WorkerAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.roots,self.history,expected=build(Path(self.tmp.name));wrap(self.roots)
        original=setup.read_unique
        p=patch.object(setup,'read_unique',side_effect=lambda path:expected if Path(path).name=='expected.json' else original(path))
        p.start();self.addCleanup(p.stop)

    def call(self):return audit.audit_run(self.roots,self.history)

    def test_raw_wrappers_reconstructed_without_full_run_advancement(self):
        result=self.call();self.assertTrue(result['evidence_gates_passed']);self.assertFalse(result['advancement_ready'])
        self.assertEqual(len(result['evidence_sha256']),275)
        self.assertTrue(all(p['worker']['passed'] for p in result['phases']))

    def test_missing_wrapper_or_mixed_bare_phase_cannot_be_ignored(self):
        path=self.roots[0]/'allocation-persisted.json';saved=path.read_bytes();path.unlink()
        with self.assertRaises(ValueError):self.call()
        path.write_bytes(saved)
        for name in WORKER_FILES:(self.roots[0]/name).unlink()
        with self.assertRaisesRegex(ValueError,'Mixed bare-core'):self.call()

    def test_allocation_publication_hash_and_preload_clock_enforced(self):
        path=self.roots[0]/'allocation-persisted.json';original=read(path)
        for key,value in [('sha256','0'*64),('published_ns',1000),('published_ns',True)]:
            write(path,dict(original,**{key:value}))
            with self.subTest(key=key),self.assertRaises(ValueError):self.call()
        write(path,original)

    def test_close_owner_failure_missing_or_early_close_rejected(self):
        path=self.roots[1]/'lifecycle.json';original=read(path)
        for mode in ('owner','failed','early','missing','prefix'):
            rows=copy.deepcopy(original)
            if mode=='owner':rows[-1]['thread_id']=999
            if mode=='failed':rows[-1].update(success=False,error='close failed')
            if mode=='early':rows[-1]['started_ns']=1
            if mode=='missing':rows.pop()
            if mode=='prefix':rows[0]['thread_id']+=1
            write(path,rows)
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.call()
        write(path,original)

    def test_shutdown_deadline_or_group_escape_rejected(self):
        path=self.roots[0]/'worker-exit.json';original=read(path)
        write(path,dict(original,finished_ns=1620*10**9+1))
        with self.assertRaisesRegex(ValueError,'deadline'):self.call()
        write(path,original);path=self.roots[0]/'worker-entry.json';row=read(path);row['session']=999;write(path,row)
        with self.assertRaisesRegex(ValueError,'group/session'):self.call()

    def test_same_family_cache_drift_and_reused_birth_rejected(self):
        path=self.roots[2]/'cache-before.json';original=read(path);changed=copy.deepcopy(original)
        changed['inductor/compiled.py']['sha256']='0'*64;write(path,changed)
        with self.assertRaisesRegex(ValueError,'ABBA process/cache'):self.call()
        write(path,original)
        for name in ('worker-entry.json','worker-exit.json'):
            path=self.roots[2]/name;row=read(path);row['identity']=read(self.roots[1]/name)['identity'];write(path,row)
        with self.assertRaisesRegex(ValueError,'ABBA process/cache'):self.call()

    def test_false_worker_completion_or_changed_birth_is_not_rescued(self):
        path=self.roots[0]/'worker-exit.json';original=read(path)
        for mode in ('failed','birth'):
            row=copy.deepcopy(original)
            if mode=='failed':row['complete']=False
            else:row['identity']['start_ticks']+=1
            write(path,row)
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.call()

    def test_phase_cannot_reset_global_deadline_or_swap_observed_source(self):
        path=self.roots[3]/'worker-entry.json';original=read(path);changed=copy.deepcopy(original)
        changed['launch']['global_started_ns']+=1;changed['launch']['deadline_ns']+=1;write(path,changed)
        with self.assertRaisesRegex(ValueError,'one global deadline'):self.call()
        write(path,original)
        path=self.roots[3]/'allocation-observation.json';row=read(path)
        row['allocation_identity']['observed_source_sha256']['changed.py']='0'*64;write(path,row)
        persisted=self.roots[3]/'allocation-persisted.json';row=read(persisted)
        row['sha256']=hashlib.sha256(path.read_bytes()).hexdigest();write(persisted,row)
        with self.assertRaisesRegex(ValueError,'source changed across'):self.call()


if __name__=='__main__':unittest.main()
