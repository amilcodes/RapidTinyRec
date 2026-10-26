import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from rapidtinyrec import serving_phase as a
from rapidtinyrec.serving_lanes import BurstExecutionError
from test_serving_measurement_audit import fixture


class PhaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.out=Path(self.tmp.name)/'phase';self.events=[];self.fail=False
        phases,self.users,_=fixture();self.source=phases[1];self.count=0;self.records=[]
        self.server=NS(_owner=lambda:None,lanes=[0,1],frozen=True)
        def run(requests,concurrency):
            kind='primary' if self.count<3 else 'secondary';local=self.count%3
            self.events.append(('burst',kind,local,concurrency));self.count+=1
            rows=self.source[kind][local]['records']
            self.assertEqual([q.payload['cohort_index'] for q in requests],[r['result']['cohort_index'] for r in rows])
            if self.fail:
                rows[0]['error']='failed';raise BurstExecutionError(rows)
            self.records=rows;return rows
        self.server.run_burst=run
        self.ownership_count=0;self.changed=False
        def ownership(server):
            self.events.append('ownership');self.ownership_count+=1
            return {'owners':self.ownership_count if self.changed else 1}
        def snapshot(server):self.events.append('snapshot');return {'snapshot':True}
        def audit(before,after,rows):
            self.events.append('audit')
            if self.count<=6:self.assertTrue(list(self.out.glob('*-records.json')))
            return {'passed':True}
        def profile(server,users,phase,path):
            self.events.append('profile');self.assertEqual([len(phase[k]) for k in ('primary','secondary')],[3,3])
            self.assertTrue((self.out/'measurements.json').exists());path.mkdir()
            (path/'profile-records.json').write_text(json.dumps(self.records))
            return dict(numerical_passed=False,overlap={'gpu_overlap_proven':False})
        for name,func in [('live_ownership',ownership),('collect_snapshot',snapshot),('audit_interval',audit),('profile_after_measurements',profile)]:
            p=patch.object(a,name,side_effect=func);p.start();self.addCleanup(p.stop)

    def call(self):return a.run_prepared_phase(self.server,self.users,1,self.out)

    def test_complete_schedule_profiles_last_and_preserves_failed_gates(self):
        result=self.call();self.assertTrue(result['complete']);self.assertFalse(result['advancement_ready'])
        self.assertFalse(result['gpu_overlap_proven']);self.assertFalse(result['profile_numerical_passed'])
        self.assertEqual(self.count,6);self.assertEqual(self.events.count('snapshot'),14)
        self.assertGreater(self.events.index('profile'),max(i for i,e in enumerate(self.events) if isinstance(e,tuple) and e[0]=='burst'))
        self.assertEqual(self.events.count('ownership'),3)
        self.assertEqual(len(list(self.out.glob('*-records.json'))),6)
        self.assertTrue((self.out/'phase-complete.json').exists())
        with self.assertRaises(FileExistsError):self.call()

    def test_failed_request_raw_survives_and_no_later_admission(self):
        self.fail=True
        with self.assertRaises(BurstExecutionError):self.call()
        self.assertEqual(self.count,1);self.assertNotIn('profile',self.events)
        self.assertTrue((self.out/'primary-0-records.json').exists());self.assertTrue((self.out/'phase-failure.json').exists())
        self.assertFalse((self.out/'phase-complete.json').exists())

    def test_audit_failure_keeps_both_snapshots_and_raw(self):
        with patch.object(a,'audit_interval',side_effect=ValueError('counter mismatch')):
            with self.assertRaisesRegex(ValueError,'counter mismatch'):self.call()
        for suffix in ('before','records','after'):self.assertTrue((self.out/f'primary-0-{suffix}.json').exists())
        self.assertFalse((self.out/'phase-complete.json').exists())

    def test_live_owner_change_stops_before_profile(self):
        self.changed=True
        with self.assertRaisesRegex(ValueError,'ownership changed'):self.call()
        self.assertTrue((self.out/'measurements.json').exists());self.assertTrue((self.out/'ownership-measured.json').exists())
        self.assertNotIn('profile',self.events)

    def test_wrong_service_phase_and_cohort_rejected_before_files(self):
        self.server.lanes=[0]
        with self.assertRaisesRegex(ValueError,'ABBA'):self.call()
        self.server.lanes=[0,1];self.users[0]['prompt_tokens']=77
        with self.assertRaisesRegex(ValueError,'cohort'):self.call()
        self.assertFalse(self.out.exists())
