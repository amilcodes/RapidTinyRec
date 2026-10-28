import copy
import json
from pathlib import Path
import tempfile
import unittest
from rapidtinyrec.beam_wave_protocol import (PROTOCOL,load_protocol,validate_protocol,phase_schedule,
    validate_timing,validate_phase_rows,compare_phases)

ROOT=Path(__file__).resolve().parents[1]


def phases(wave=.12):
    result={}
    for phase in range(4):
        result[phase]=[dict(spec,schema='two-request-beam-wave-timing-v1',arrival_offsets_s=[0,0],
            completion_offsets_s=[.1,.2] if phase in (0,3) else [wave,wave],
            pair_duration_s=.2 if phase in (0,3) else wave) for spec in phase_schedule(phase)]
    return result


class WaveProtocolTests(unittest.TestCase):
    def test_frozen_protocol_and_tamper(self):
        p=load_protocol(ROOT/PROTOCOL);validate_protocol(p)
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'changed.json';out.write_text(json.dumps(p))
            with self.assertRaisesRegex(ValueError,'changed'):load_protocol(out)

    def test_scope_rejects_relabeling_reduced_work_and_extended_deadline(self):
        original=load_protocol(ROOT/PROTOCOL)
        for field,value in [('profile','precision-pilot'),('waves_per_phase',5),('promotion',True)]:
            p=copy.deepcopy(original);p[field]=value
            with self.assertRaises(ValueError):validate_protocol(p)
        p=copy.deepcopy(original);p['budget']['global_work_deadline_s']=1800
        with self.assertRaises(ValueError):validate_protocol(p)

    def test_balanced_order_and_pairing(self):
        a=phase_schedule(0)+phase_schedule(3);b=phase_schedule(1)+phase_schedule(2)
        for left,right in zip(a,b):
            for k in ['repeat','pair_index','cohort_indices','half']:self.assertEqual(left[k],right[k])
        for user in [1,2,6,10]:self.assertEqual(sum(r['cohort_indices'][0]==user for r in a),3)

    def test_rejects_amortized_wave_latency(self):
        r=phases()[1][0];r['completion_offsets_s']=[.06,.06]
        with self.assertRaisesRegex(ValueError,'whole-wave'):validate_timing(r)

    def test_rejects_serial_service_time_mislabeled_response(self):
        r=phases()[0][0];r['completion_offsets_s']=[.1,.1]
        with self.assertRaisesRegex(ValueError,'queue time'):validate_timing(r)

    def test_rejects_nonfinite_empty_negative_and_delayed_arrival(self):
        for value in [float('nan'),float('inf'),0,-1,True]:
            r=phases()[0][0];r['pair_duration_s']=value
            with self.assertRaises(ValueError):validate_timing(r)
        r=phases()[0][0];r['arrival_offsets_s']=[0,.05]
        with self.assertRaises(ValueError):validate_timing(r)

    def test_incomplete_duplicate_reordered_phases_fail_closed(self):
        p=phases()
        with self.assertRaises(ValueError):compare_phases({k:v for k,v in p.items() if k!=2})
        with self.assertRaises(ValueError):validate_phase_rows(0,p[0][:-1])
        p[0][1]=p[0][0]
        with self.assertRaises(ValueError):compare_phases(p)

    def test_reports_pair_throughput_and_real_request_response(self):
        review=compare_phases(phases());overall=review['strata']['overall']
        self.assertAlmostEqual(overall['control']['throughput_requests_per_s'],10)
        self.assertAlmostEqual(overall['candidate']['throughput_requests_per_s'],2/.12)
        self.assertAlmostEqual(overall['candidate']['response']['median_ms'],120)
        self.assertAlmostEqual(overall['control']['first_response']['median_ms'],100)
        self.assertAlmostEqual(overall['control']['second_response']['median_ms'],200)
        self.assertEqual(len(review['strata']),9);self.assertTrue(review['timing_screen_passed'])
        self.assertIsNone(review['raw_audit_passed']);self.assertFalse(review['promotion'])

    def test_throughput_gain_does_not_hide_response_regression(self):
        review=compare_phases(phases(.17))
        self.assertGreater(review['strata']['overall']['throughput_gain_percent'],5)
        self.assertGreater(review['strata']['overall']['response_median_change_percent'],0)
        self.assertFalse(review['timing_screen_passed'])


if __name__=='__main__':unittest.main()
