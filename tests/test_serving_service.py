import contextlib
import sys
import threading
import types
import unittest
from unittest.mock import patch

from rapidtinyrec.serving_lanes import LaneServer, Request
from rapidtinyrec.serving_service import prepared_service


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.events = []; self.active_patch = False; self.shared = None
        @contextlib.contextmanager
        def folding(enabled):
            self.assertTrue(enabled); self.active_patch = True; self.events.append('patch')
            try: yield
            finally: self.events.append('unpatch'); self.active_patch = False
        fake = types.ModuleType('rapidtinyrec.folded_attention')
        fake.fold_decode_attention = folding
        self.patch = patch.dict(sys.modules, {'rapidtinyrec.folded_attention': fake})
        self.patch.start(); self.addCleanup(self.patch.stop)

    def initialize(self, index):
        case = self
        self.assertTrue(self.active_patch); self.events.append(('init', index))
        class Lane:
            def __init__(self): self.owner = threading.get_ident(); self.retained = True
            def check(self):
                case.assertEqual(threading.get_ident(), self.owner)
                case.assertTrue(case.active_patch)
            def freeze(self): self.check(); case.events.append(('freeze', index))
            def ownership_snapshot(self, expected_graphs):
                self.check(); case.assertEqual(expected_graphs, 1)
                case.events.append(('snapshot', index))
                other = 0 if case.shared == 'storage' else index
                pool = 1 if case.shared == 'pool' else index+1
                record = dict(identity=dict(owner_thread_id=self.owner, stream_id=index+1,
                                            device='cuda:0', failed=False),
                    graphs=[dict(index=0, primary=[pool, 0], retained=[[pool, 0]])],
                    storage=[dict(name='owner', role='model', device='cuda:0',
                                  address=1000+other*100, storage_bytes=10, logical_bytes=10)])
                return dict(record=record, graph_refs=(object(),), tensor_refs=(object(),))
            def release_retained(self):
                self.check(); self.retained = False; case.events.append(('release', index))
            def run(self, user):
                self.check(); case.assertFalse(self.retained)
                case.events.append(('run', index)); return user
            def close(self): self.check(); case.events.append(('close', index))
        return Lane()

    def audit(self, records):
        self.assertEqual(set(records), {0, 1})
        self.assertFalse(any(isinstance(e, tuple) and e[0] == 'release' for e in self.events))
        self.events.append('audit'); return {'passed': True}

    def test_all_setup_precedes_freeze_audit_precedes_release_patch_outlives_close(self):
        with prepared_service(self.initialize, self.audit, lane_count=2, expected_graphs=1) as (server, report):
            rows = server.run_burst([Request('a', 1), Request('b', 2)], 2)
            self.assertEqual([r['result'] for r in rows], [1, 2])
            self.assertTrue(report['setup_audit']['passed'])
        self.assertEqual(self.events[:7], ['patch', ('init', 0), ('init', 1), ('freeze', 0),
                                          ('freeze', 1), ('snapshot', 0), ('snapshot', 1)])
        self.assertEqual(self.events[7:10], ['audit', ('release', 0), ('release', 1)])
        self.assertEqual(self.events[-3:], [('close', 0), ('close', 1), 'unpatch'])

    def test_pool_or_storage_alias_prevents_setup_audit_and_admission(self):
        for kind in ('pool', 'storage'):
            self.events = []; self.shared = kind
            with self.assertRaises(ValueError):
                with prepared_service(self.initialize, self.audit, lane_count=2, expected_graphs=1):
                    self.fail('Ownership failure must prevent yield')
            self.assertNotIn('audit', self.events)
            self.assertEqual(self.events[-3:], [('close', 0), ('close', 1), 'unpatch'])

    def test_numeric_audit_failure_never_releases_or_yields(self):
        for result in (None, {'passed': False}, {'passed': 1}):
            self.events = []
            with self.assertRaises(ValueError):
                with prepared_service(self.initialize, lambda records: result, lane_count=2, expected_graphs=1):
                    self.fail('Failed audit must prevent yield')
            self.assertFalse(any(isinstance(e, tuple) and e[0] == 'release' for e in self.events))
            self.assertEqual(self.events[-3:], [('close', 0), ('close', 1), 'unpatch'])

    def test_initialization_failure_closes_prior_lane_before_unpatch(self):
        def init(index):
            if index == 1: raise ValueError('allocation failure')
            return self.initialize(index)
        with self.assertRaises(ValueError):
            with prepared_service(init, self.audit, lane_count=2, expected_graphs=1): pass
        self.assertEqual(self.events, ['patch', ('init', 0), ('close', 0), 'unpatch'])

    def test_inspection_failure_blocks_future_admission_but_all_lanes_close(self):
        self.active_patch = True
        with LaneServer(self.initialize, lambda lane, user: lane.run(user),
                        close=lambda lane: self.events.append(('closed', lane.owner))) as server:
            def fail(lane): raise ValueError('inspection')
            with self.assertRaises(ValueError): server.visit(fail)
            with self.assertRaises(RuntimeError): server.freeze()
        self.assertEqual(sum(isinstance(e, tuple) and e[0] == 'closed' for e in self.events), 2)

    def test_one_lane_control_uses_same_gates(self):
        def audit(records):
            self.assertEqual(set(records), {0}); return {'passed': True}
        with prepared_service(self.initialize, audit, lane_count=1, expected_graphs=1) as (server, report):
            rows = server.run_burst([Request('a', 1)], 1)
            self.assertEqual(rows[0]['result'], 1)
            self.assertEqual(set(report['owners']), {0})
        self.assertEqual(self.events[-2:], [('close', 0), 'unpatch'])

    def test_admission_journal_uses_owner_clocks_and_snapshot_survives_close(self):
        journal=[]
        with prepared_service(self.initialize,self.audit,lane_count=2,expected_graphs=1,lifecycle=journal) as (server,report):
            events=report['admission_events']
            self.assertEqual(events[-1]['operation'],'admitted')
            for row in events:
                self.assertTrue(row['success']);self.assertIsNone(row['error'])
                self.assertLessEqual(row['started_ns'],row['finished_ns'])
                if row['lane'] is not None:
                    self.assertEqual(row['thread_id'],report['owners'][row['lane']]['identity']['owner_thread_id'])
            self.assertLess(max(r['finished_ns'] for r in events if r['operation']=='initialize'),
                            min(r['started_ns'] for r in events if r['operation']=='freeze'))
        self.assertEqual([r['operation'] for r in journal[-2:]],['close','close'])
        self.assertEqual(report['admission_events'][-1]['operation'],'admitted')

    def test_failed_setup_journal_preserved_and_admission_absent(self):
        journal=[]
        with self.assertRaises(ValueError):
            with prepared_service(self.initialize,lambda records:{'passed':False},lane_count=2,expected_graphs=1,lifecycle=journal):
                self.fail('Must not admit')
        rows=[r for r in journal if r['operation']=='setup_audit']
        self.assertEqual(len(rows),1);self.assertFalse(rows[0]['success']);self.assertIn('did not pass',rows[0]['error'])
        self.assertFalse(any(r['operation']=='admitted' for r in journal))
        self.assertEqual([r['operation'] for r in journal[-2:]],['close','close'])


if __name__ == '__main__': unittest.main()
