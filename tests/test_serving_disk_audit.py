import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rapidtinyrec import serving_disk_audit as a
from rapidtinyrec import serving_setup_audit as setup
from rapidtinyrec.serving_invariants import BYTES_PER_TOKEN, PREFIX_BYTES_PER_TOKEN
from test_serving_setup_audit import raw, owner
from test_serving_invariants import fixture as invariant_fixture
from test_serving_measurement_audit import fixture as measurement_fixture
from test_serving_overlap_audit import fixture as overlap_fixture
from test_serving_admission_audit import fixture as admission_fixture


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def read(path):
    return json.loads(path.read_text())


def build(base):
    history = base/'history'
    all_users = [dict(cohort_index=i, source_row_index=i, user_id=str(i), prompt_tokens=76 if i%2 == 0 else 121, target_sid=str(i)) for i in range(500)]
    users = [all_users[i] for i in (0, 2, 1, 3)]
    manifest = dict(experiment='gqa_three_way', dtype='float32', users=all_users)
    manifest['manifest_sha256'] = setup.json_digest(manifest)
    write(history/'manifest.json', manifest)
    expected = dict(users_sha256=setup.json_digest(all_users), release_sha256={'checkpoint': 'abc'})
    for u in users:
        old = raw()
        for key in ('sequence_bits', 'transition_bits', 'beam_indices'):
            del old[key]
        write(history/'users'/f"{u['cohort_index']:04d}.json", dict(u, manifest_sha256=manifest['manifest_sha256'], runs={'compiled_gqa': old}))
    phases, _, _ = measurement_fixture(); roots = []
    for index, phase in enumerate(phases):
        root = base/f'phase_{index}'; roots.append(root)
        execution = root/'execution'; lanes = phase['lane_count']; offset = index*1000000
        state = invariant_fixture()[0]
        state['lanes'] = [copy.deepcopy(state['lanes'][0]) for _ in range(lanes)]
        owners = {}
        for i, lane in enumerate(state['lanes']):
            lane['identity'].update(owner_thread_id=10+i, stream_id=100+i)
            lane['compiler']['events'][0]['index'] = 0
            owners[str(i)] = owner(i*10000)
            owners[str(i)]['identity'] = copy.deepcopy(lane['identity'])
            owners[str(i)]['graphs'] = [dict(index=n, primary=[1+i*100+n, 0], retained=[[1+i*100+n, 0]]) for n in range(8)]
            prefix = root/f'setup/lane_{i}'
            write(prefix/'initialization-stages.json', dict(lane=i,
                allocation_started_ns=offset+30+i*50, allocation_finished_ns=offset+40+i*50,
                warmup_started_ns=offset+50+i*50, warmup_finished_ns=offset+60+i*50,
                allocation_s=10/1e9, warmup_s=10/1e9))
            write(prefix/'compilation.json', dict(phase='measurement', measurement_compilations=0, backend_graphs=1, events=lane['compiler']['events']))
            (prefix/'compiler_times.txt').write_text('synthetic CPU fixture')
            (prefix/'compiled_graphs').mkdir(); (prefix/'compiled_graphs/fx_000.py').write_text('# synthetic CPU fixture')
            for u in users:
                for mode in ('dense', 'replay', 'repeat'):
                    write(prefix/f"warmup/{u['cohort_index']}_{mode}.json", raw())
            write(prefix/'warmup/retained_after_mixed.json', {'0': raw(), '1': raw()})
        write(root/'setup/initialization-provenance.json', dict(users=users, observed_release_sha256=expected['release_sha256'], reference_manifest_sha256=manifest['manifest_sha256'], torch_version='2.14.0+cu126', torch_git='08187d9e0fba026dc8217405802ab5381dc88d90', transformers_version='4.55.4', tf32=False))
        write(root/'setup/resource-verification-stages.json', dict(initializer_started_ns=offset+10, resources_verified_ns=offset+20, elapsed_s=10/1e9))
        write(root/'preparation.json', dict(owners=owners, setup_audit={'passed': True}, admission_events=admission_fixture(lanes, offset)[0]))
        write(root/'setup-audit.json', {'passed': True})
        live = copy.deepcopy(owners)
        for row in live.values():
            row['storage'] = [v for v in row['storage'] if v['role'] != 'retained']
        for tag in ('before', 'measured', 'after'):
            write(execution/f'ownership-{tag}.json', dict(owners=live, graph_ownership={'passed': True}, storage_ownership={'passed': True}))
        write(execution/'phase-start.json', dict(phase=index, users=users, started_ns=offset+1000))
        def snapshot_interval(label, records):
            before = copy.deepcopy(state)
            before.update(started_ns=records[0]['arrival_ns']-20, finished_ns=records[0]['arrival_ns']-19)
            for r in records:
                lane = state['lanes'][r['lane']]
                lane['replay']['replays'] += 4
                lane['replay']['physical_prefix']['requests_loaded'] += 1
                lane['replay']['physical_prefix']['prefix_bytes_copied'] += PREFIX_BYTES_PER_TOKEN*r['result']['prompt_tokens']
                lane['reorder']['direct_calls'] += 4; lane['reorder']['fallback_calls'] += 1
                lane['reorder']['suffix_bytes_written'] += BYTES_PER_TOKEN*6
            after = copy.deepcopy(state)
            end = max(r['completed_ns'] for r in records)
            after.update(started_ns=end+1, finished_ns=end+2)
            write(execution/f'{label}-before.json', before); write(execution/f'{label}-after.json', after)
        for kind in ('primary', 'secondary'):
            for burst in phase[kind]:
                label = f"{kind}-{burst['repeat']}"
                for row in burst['records']:
                    u = users[row['result']['cohort_index']]
                    row['result'].update(cohort_index=u['cohort_index'], prompt_tokens=u['prompt_tokens'])
                    row['request_id'] = f"{label.replace('-', ':')}:{u['cohort_index']}"
                    for key in ('arrival_ns','submitted_ns','started_ns','completed_ns'):
                        row[key] += offset
                    for key in ('generation_started_ns','generation_finished_ns'):
                        row['result'][key] += offset
                write(execution/f'{label}-records.json', burst['records'])
                snapshot_interval(label, burst['records'])
                write(execution/f'{label}-audit.json', {'passed': True})
        write(execution/'measurements.json', phase)
        records = copy.deepcopy(phase['primary'][0]['records'])
        for row in records:
            row['request_id'] = str(row['result']['cohort_index'])
            for key in ('arrival_ns','submitted_ns','started_ns','completed_ns'):
                row[key] += 100000
            for key in ('generation_started_ns','generation_finished_ns'):
                row['result'][key] += 100000
        snapshot_interval('profile', records)
        write(execution/'profile-invariants.json', {'passed': True})
        trace, markers = overlap_fixture()
        for event in trace['traceEvents']:
            if event['cat'] == 'user_annotation':
                pos = int(event['name'].rsplit('_',1)[1]); lane = pos%lanes
                event['name'] = f"serving_profile/lane_{lane}/user_{users[pos]['cohort_index']}"
            else:
                pos = (event['args']['correlation']-1)//4; lane = pos%lanes
            if lanes == 1:
                event['ts'] += (pos-pos//2)*100
                if event['cat'] == 'kernel': event['args']['stream'] = 7
                else: event['tid'] = 20
        markers = {e['name']: pos%lanes for pos,e in enumerate([e for e in trace['traceEvents'] if e['cat']=='user_annotation'])}
        write(execution/'profile/trace.json', trace)
        write(execution/'profile/profile-records.json', records)
        completion = max(r['completed_ns'] for r in phase['secondary' if lanes==2 else 'primary'][-1]['records'])
        write(execution/'profile/profile-start.json', dict(phase=index, started_ns=records[0]['arrival_ns']-10,
            last_measurement_completion_ns=completion, owners=[live[str(i)]['identity'] for i in range(lanes)], expected_markers=markers, performance_sample=False))
        write(execution/'profile/profile-audit.json', {'numerical_passed': True, 'gpu_overlap_proven': True})
        write(execution/'phase-complete.json', dict(schema='serving-phase-completion-v1', complete=True, phase=index,
            started_ns=offset+1000, finished_ns=offset+300000, primary_requests=12, secondary_requests=12 if lanes==2 else 0, profile_requests=4,
            profile_numerical_passed=True, gpu_overlap_proven=True))
    return roots, history, expected


class DiskAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.roots, self.history, expected = build(self.base)
        original = setup.read_unique
        mock = patch.object(setup, 'read_unique', side_effect=lambda p: expected if Path(p).name == 'expected.json' else original(p))
        mock.start(); self.addCleanup(mock.stop)

    def audit(self): return a.audit_run(self.roots, self.history)

    def change_burst(self, index, kind, position, mutation):
        path = self.roots[index]/'execution/measurements.json'; phase = read(path)
        burst = phase[kind][position]; mutation(burst['records'])
        write(path, phase)
        write(path.parent/f"{kind}-{burst['repeat']}-records.json", burst['records'])

    def test_complete_reconstruction_never_authorizes_gpu_advancement(self):
        result = self.audit()
        self.assertTrue(result['evidence_gates_passed']); self.assertTrue(all(result['gates'].values()))
        self.assertFalse(result['advancement_ready']); self.assertFalse(result['online_verdicts_trusted'])
        self.assertEqual(result['measurements']['comparisons']['paired']['requests'], 24)
        self.assertEqual(len(result['phases']), 4)
        self.assertEqual(len(result['evidence_sha256']), 247)
        self.assertEqual(result['phases'][1]['profile_overlap']['graph_kernel_overlap_us'], 32)

    def test_online_pass_flags_cannot_rescue_valid_raw_drift(self):
        def drift(rows): rows[0]['result']['raw']['token_ids'][0][0] = 999
        self.change_burst(1, 'secondary', 0, drift)
        result = self.audit()
        self.assertFalse(result['gates']['measured_numerics']); self.assertFalse(result['evidence_gates_passed'])
        self.assertTrue(result['gates']['actual_gpu_overlap'])

    def test_setup_ancestry_binding_catches_consistent_measurement_drift(self):
        def drift(rows):
            for r in rows: r['result']['raw']['beam_indices'][0][0] = 1
        for index, root in enumerate(self.roots):
            for kind in ('primary','secondary'):
                for pos in range(3 if kind=='primary' or index in (1,2) else 0):
                    self.change_burst(index, kind, pos, drift)
            path=root/'execution/profile/profile-records.json'; data=read(path);drift(data);write(path,data)
        result=self.audit()
        self.assertTrue(result['gates']['measured_numerics']);self.assertTrue(result['gates']['profile_numerics'])
        self.assertFalse(result['gates']['setup_to_execution'])

    def test_saved_overlap_true_cannot_replace_actual_kernel_intersection(self):
        path=self.roots[2]/'execution/profile/trace.json';trace=read(path)
        for e in trace['traceEvents']:
            lane=(e['args']['correlation']-1)//4%2 if e['cat']!='user_annotation' else int(e['name'].split('/')[1].split('_')[1])
            if lane==1:e['ts']+=40
        write(path,trace);result=self.audit()
        self.assertFalse(result['gates']['actual_gpu_overlap']);self.assertTrue(result['gates']['measured_numerics'])

    def test_missing_duplicate_and_linked_files_rejected(self):
        path=self.roots[1]/'execution/primary-0-records.json';saved=path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'Non-regular'):self.audit()
        path.write_text('{"x":1,"x":2}')
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.audit()
        path.unlink();os.link(self.roots[0]/'execution/primary-0-records.json',path)
        with self.assertRaisesRegex(ValueError,'Aliased'):self.audit()
        path.unlink();path.write_bytes(saved)
        path=self.roots[0]/'execution/extra.json';write(path,{})
        with self.assertRaisesRegex(ValueError,'file set'):self.audit()

    def test_unobserved_between_burst_work_and_owner_change_rejected(self):
        path=self.roots[1]/'execution/primary-1-before.json';data=read(path);saved=copy.deepcopy(data)
        data['lanes'][0]['replay']['replays']+=1;write(path,data)
        with self.assertRaisesRegex(ValueError,'Replay delta'):self.audit()
        write(path,saved)
        path=self.roots[1]/'execution/ownership-after.json';data=read(path);data['owners']['0']['identity']['stream_id']+=999;write(path,data)
        with self.assertRaisesRegex(ValueError,'Live ownership'):self.audit()

    def test_setup_order_profile_clock_and_cross_phase_overlap_rejected(self):
        cases=[(self.roots[1]/'setup/lane_1/initialization-stages.json','allocation_started_ns',1000020),
               (self.roots[1]/'execution/profile/profile-start.json','started_ns',1000000),
               (self.roots[0]/'execution/phase-complete.json','finished_ns',2000000)]
        for path,key,value in cases:
            data=read(path);saved=copy.deepcopy(data);data[key]=value;write(path,data)
            with self.subTest(key=key),self.assertRaises(ValueError):self.audit()
            write(path,saved)

    def test_failure_saved_once_and_input_roots_never_written(self):
        path=self.roots[0]/'execution/primary-0-records.json';path.unlink()
        output=self.base/'audit.json'
        with self.assertRaisesRegex(ValueError, 'Non-regular'):a.save_audit(self.roots,self.history,output)
        saved=output.read_bytes();self.assertTrue(read(output)['malformed'])
        with self.assertRaises(FileExistsError):a.save_audit(self.roots,self.history,output)
        self.assertEqual(saved,output.read_bytes())
        with self.assertRaisesRegex(ValueError,'outside'):a.save_audit(self.roots,self.history,self.roots[0]/'audit.json')

    def test_mutation_during_audit_and_summary_disagreement_rejected(self):
        evidence=a.Evidence();path=self.roots[0]/'execution/phase-start.json';evidence.register(path)
        path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'changed'):evidence.finish()
        path=self.roots[0]/'execution/measurements.json';data=read(path);data['primary'][0]['records'][0]['completed_ns']+=1;write(path,data)
        with self.assertRaisesRegex(ValueError,'summary'):self.audit()
