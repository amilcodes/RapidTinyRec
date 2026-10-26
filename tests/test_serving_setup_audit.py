import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from rapidtinyrec import serving_setup_audit as a


def raw():
    return dict(token_ids=[[i, 1, 2, 198, 151645] for i in range(50)],
                sequence_logp=[0.] * 50, sequence_bits=[0] * 50,
                transition_logp=[[0.] * 5 for _ in range(50)],
                transition_bits=[[0] * 5 for _ in range(50)],
                beam_indices=[[0] * 5 for _ in range(50)])


def owner(offset=0):
    rows = []
    def add(name, role):
        rows.append(dict(name=name, role=role, device='cuda:0', address=offset+100*(len(rows)+1), storage_bytes=64, logical_bytes=64))
    for key in (0, 1):
        for name in ('sequences', 'sequence_scores', 'beam_indices', 'score_0', 'score_1', 'score_2', 'score_3', 'score_4'):
            add(f'retained.{key}.{name}', 'retained')
    for i in range(8): add(f'replay.{i}.output.logits', 'graph_logits')
    return dict(identity=dict(device='cuda:0'), storage=rows)


class SetupAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'new'; self.root.mkdir()
        self.history = Path(self.tmp.name)/'history'; (self.history/'users').mkdir(parents=True)
        users = [dict(cohort_index=i, source_row_index=i, user_id=str(i), prompt_tokens=76 if i%2 == 0 else 121, target_sid=str(i)) for i in range(500)]
        self.users = [users[i] for i in (0,2,1,3)]
        manifest = dict(experiment='gqa_three_way', dtype='float32', users=users)
        manifest['manifest_sha256'] = a.json_digest(manifest)
        self.write(self.history/'manifest.json', manifest)
        expected = dict(users_sha256=a.json_digest(users), release_sha256={'checkpoint':'abc'})
        original = a.read_unique
        self.mock = patch.object(a, 'read_unique', side_effect=lambda p: expected if Path(p).name == 'expected.json' else original(p))
        self.mock.start(); self.addCleanup(self.mock.stop)
        self.write(self.root/'initialization-provenance.json', dict(users=self.users, observed_release_sha256=expected['release_sha256'], reference_manifest_sha256=manifest['manifest_sha256'], torch_version='2.14.0+cu126', torch_git='08187d9e0fba026dc8217405802ab5381dc88d90', transformers_version='4.55.4', tf32=False))
        for u in self.users:
            old = raw()
            for key in ('sequence_bits','transition_bits','beam_indices'): del old[key]
            self.write(self.history/'users'/f"{u['cohort_index']:04d}.json",dict(u, manifest_sha256=manifest['manifest_sha256'], runs={'compiled_gqa': old}))
        for lane in (0,1):
            warm = self.root/f'lane_{lane}'/'warmup'; warm.mkdir(parents=True)
            for u in self.users:
                for mode in ('dense','replay','repeat'): self.write(warm/f"{u['cohort_index']}_{mode}.json", raw())
            self.write(warm/'retained_after_mixed.json', {'0': raw(), '1': raw()})
        self.owners = {0: owner(), 1: owner(10000)}

    def write(self, p, data): p.write_text(json.dumps(data))
    def audit(self): return a.audit_setup(self.root,self.history,self.owners)

    def test_one_and_two_lane_complete_setup(self):
        result = self.audit(); self.assertTrue(result['passed'])
        self.assertFalse(result['historical_ancestry_available'])
        self.assertEqual(result['historical_comparison']['requests'], 28)
        self.assertEqual(result['score_ownership']['1']['independent_score_storages'],10)
        self.owners.pop(1); self.assertTrue(self.audit()['passed'])

    def test_valid_score_and_retained_ancestry_drift_preserved(self):
        path = self.root/'lane_0/warmup/0_repeat.json'; r = raw()
        r['transition_logp'][0][0] = -1.; r['sequence_logp'][0] = -1.
        r['transition_bits'][0][0] = a.int_bits(-1.); r['sequence_bits'][0] = a.int_bits(-1.)
        self.write(path,r); result=self.audit(); self.assertFalse(result['passed']); self.assertTrue(result['outputs_valid'])
        self.write(path,raw()); path=self.root/'lane_0/warmup/retained_after_mixed.json'
        r={'0':raw(),'1':raw()};r['0']['beam_indices'][0][0]=1;self.write(path,r)
        result=self.audit();self.assertFalse(result['passed']);self.assertFalse(result['ancestry_stable'])

    def test_malformed_bits_missing_extra_and_aliased_evidence_rejected(self):
        path=self.root/'lane_0/warmup/0_repeat.json';r=raw();r['sequence_bits'][0]=1;self.write(path,r)
        with self.assertRaisesRegex(ValueError,'bits'):self.audit()
        path.unlink()
        with self.assertRaisesRegex(ValueError,'file set'):self.audit()
        path.symlink_to('0_dense.json')
        with self.assertRaisesRegex(ValueError,'file set'):self.audit()
        path.unlink();self.write(path,raw());self.write(path.parent/'extra.json',{})
        with self.assertRaisesRegex(ValueError,'file set'):self.audit()

    def test_score_alias_graph_alias_omitted_and_wrong_device_rejected(self):
        good=copy.deepcopy(self.owners)
        for mutation in ('score','graph','missing','device'):
            self.owners=copy.deepcopy(good);rows=self.owners[0]['storage'];score=rows[3]
            if mutation=='score':rows[4]['address']=score['address']
            if mutation=='graph':rows[-1]['address']=score['address']+32
            if mutation=='missing':rows.pop(3)
            if mutation=='device':score['device']='cpu'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.audit()

    def test_reference_and_initializer_identity_rejected(self):
        path=self.history/'users/0000.json';r=json.loads(path.read_text());r['user_id']='wrong';self.write(path,r)
        with self.assertRaisesRegex(ValueError,'row identity'):self.audit()
        r['user_id']='0';self.write(path,r)
        path=self.root/'initialization-provenance.json';r=json.loads(path.read_text());r['tf32']=True;self.write(path,r)
        with self.assertRaisesRegex(ValueError,'provenance'):self.audit()

    def test_failure_is_saved_and_never_overwritten(self):
        output=self.root/'audit.json';callback=a.setup_auditor(self.root,self.history,output)
        self.owners[0]['storage'].pop()
        with self.assertRaises(ValueError):callback(self.owners)
        saved=output.read_bytes();self.assertTrue(json.loads(saved)['malformed'])
        with self.assertRaises(FileExistsError):callback(self.owners)
        self.assertEqual(saved,output.read_bytes())

    def test_cross_lane_consistent_ancestry_drift_is_rejected(self):
        warm=self.root/'lane_1/warmup'
        for path in warm.iterdir():
            data=json.loads(path.read_text())
            runs=data.values() if path.name=='retained_after_mixed.json' else [data]
            for run in runs:run['beam_indices'][0][0]=1
            self.write(path,data)
        result=self.audit()
        self.assertFalse(result['passed']);self.assertFalse(result['ancestry_stable'])

    def test_duplicate_json_and_bool_ancestry_rejected(self):
        path=self.root/'duplicate.json';path.write_text('{"x":1,"x":2}')
        with self.assertRaisesRegex(ValueError,'Duplicate'):a.read_unique(path)
        r=raw();r['beam_indices'][0][0]=False
        with self.assertRaisesRegex(ValueError,'ancestry'):a.validate_raw(r)
