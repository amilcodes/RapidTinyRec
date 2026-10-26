"""Verify the diagnostic archive and independently recompute its saved comparisons."""
import hashlib
import json
from pathlib import Path
import statistics
import tarfile
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'benchmarks/diagnostics/packed-mlp'


class PackedDiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads((ROOT/'summary.json').read_text())
        cls.audit = json.loads((ROOT/'audit.json').read_text())
        with tarfile.open(ROOT/'evidence.tar.gz') as archive:
            members = archive.getmembers()
            assert all(m.isfile() for m in members)
            assert len({m.name for m in members}) == len(members)
            cls.files = {m.name:archive.extractfile(m).read() for m in members}
        cls.data = json.loads(cls.files['raw/outputs/diagnosis/diagnosis.json'])

    def test_integrity_and_source(self):
        source=json.loads((ROOT/'source.json').read_text())
        manifest=json.loads((ROOT/'archive-manifest.json').read_text())
        self.assertEqual(set(manifest),set(self.files))
        for name,meta in manifest.items():
            self.assertEqual(len(self.files[name]),meta['bytes'])
            self.assertEqual(hashlib.sha256(self.files[name]).hexdigest(),meta['sha256'])
        for name,digest in source['evidence_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest)
        record=json.loads(self.files['provenance/run-record.json'])
        self.assertEqual(record['job_id'],'7949846')
        self.assertEqual(len(record['sha256']),47)
        self.assertEqual(record['sha256'],json.loads(self.files['raw/source-manifest.json'])['sha256'])
        for name,digest in record['sha256'].items():
            self.assertEqual(hashlib.sha256(self.files['snapshot/'+name]).hexdigest(),digest)

    def test_comparison_counts_and_graph_controls(self):
        self.assertTrue(self.data['complete'])
        self.assertFalse(self.data['tf32'])
        rows=self.data['cases']
        self.assertEqual({(r['cohort_index'],r['layer']) for r in rows},
                         {(i,l) for i in (1,2,6,10) for l in (0,13,27)})
        graph_values=0
        for variant, expected in self.audit['aggregates'].items():
            for field, aggregate in expected.items():
                values=[r['variants'][variant][field] for r in rows]
                self.assertTrue(all(v['finite'] for v in values))
                self.assertEqual(sum(v['values'] for v in values),aggregate['values'])
                self.assertEqual(sum(v['matching_bits'] for v in values),aggregate['matching_bits'])
                self.assertEqual(max(v['max_abs'] for v in values),aggregate['max_abs'])
                if variant in ('separate','packed_views_separate'):
                    self.assertEqual(aggregate['matching_bits'],aggregate['values'])
                else:
                    self.assertLess(aggregate['matching_bits'],aggregate['values'])
        for r in rows:
            probe=r['identical_value_split_pointwise']
            self.assertEqual(probe['matching_bits'],probe['values'])
            self.assertEqual(probe['max_abs'],0.)
            for detail in r['variants'].values():
                self.assertEqual(len(detail['graph_matches_direct']),2)
                for c in detail['graph_matches_direct']:
                    self.assertEqual(c['matching_bits'],c['values'])
                    self.assertEqual(c['max_abs'],0.)
                    graph_values+=c['values']
        self.assertEqual(graph_values,self.summary['graph_direct_matching_values'])

    def test_timing_dispersion_and_kernel_launches(self):
        for row,summary in zip(self.data['cases'],self.summary['projection_timing_by_case']):
            for variant,detail in row['variants'].items():
                samples=detail['graph_replay_projection_us'];self.assertEqual(len(samples),8)
                self.assertEqual(statistics.median(samples),summary['timing'][variant]['median_us'])
                self.assertEqual(statistics.quantiles(samples,n=100,method='inclusive')[94],summary['timing'][variant]['p95_us'])
                self.assertEqual(min(samples),summary['timing'][variant]['min_us'])
                self.assertEqual(max(samples),summary['timing'][variant]['max_us'])
            self.assertGreater(row['variants']['batched_two']['projection_median_us'],
                               row['variants']['separate']['projection_median_us'])
        for variant,expected in self.summary['first_case_kernel_launches'].items():
            trace=json.loads(self.files['raw/outputs/diagnosis/'+variant+'_trace.json'])
            actual=[{k:e.get(k) for k in ('name','dur','args')} for e in trace['traceEvents']
                    if e.get('cat') in ('kernel','gpu_memset')]
            self.assertEqual(actual,expected)


if __name__ == '__main__':
    unittest.main()
