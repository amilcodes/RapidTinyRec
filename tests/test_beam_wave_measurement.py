import unittest
from rapidtinyrec.beam_wave_protocol import phase_schedule
from rapidtinyrec.beam_wave_measurement import measure_pair


class PairMeasurementTests(unittest.TestCase):
    def run_pair(self,phase,clock_values):
        events=[];held=[];clock_values=iter(clock_values)
        def clock():events.append('clock');return next(clock_values)
        def sync():events.append('sync')
        def generate(users):
            events.append(('generate',users));output={'users':users};held.append(output);return output
        def unpack(output,users):
            events.append(('unpack',users));self.assertEqual(output['users'],users)
            return [dict(cohort_index=u) for u in users]
        return measure_pair(phase_schedule(phase)[0],generate,sync,clock,unpack),events,held

    def test_serial_second_response_and_extraction_boundary(self):
        row,events,held=self.run_pair(0,[10,10.1,10.25])
        self.assertAlmostEqual(row['timing']['completion_offsets_s'][0],.1)
        self.assertAlmostEqual(row['timing']['completion_offsets_s'][1],.25)
        self.assertEqual(events,['sync','clock',('generate',[1]),'sync','clock',('generate',[2]),'sync','clock',('unpack',[1]),('unpack',[2])])
        self.assertEqual(len(held),2)

    def test_wave_is_one_call_both_complete_together(self):
        row,events,held=self.run_pair(1,[5,5.12])
        self.assertEqual(row['timing']['completion_offsets_s'],[row['timing']['pair_duration_s']]*2)
        self.assertEqual(events,['sync','clock',('generate',[1,2]),'sync','clock',('unpack',[1,2])]);self.assertEqual(len(held),1)

    def test_generation_failure_does_not_emit_partial_success(self):
        events=[]
        def fail(users):events.append(users);raise RuntimeError('generation failed')
        with self.assertRaisesRegex(RuntimeError,'generation failed'):
            measure_pair(phase_schedule(0)[0],fail,lambda:None,lambda:0,lambda *a:self.fail('unpack after failed generation'))
        self.assertEqual(events,[[1]])

    def test_wrong_output_order_rejected(self):
        clock=iter([1,2])
        with self.assertRaisesRegex(ValueError,'pair order'):
            measure_pair(phase_schedule(1)[0],lambda users:None,lambda:None,lambda:next(clock),
                lambda out,users:[dict(cohort_index=users[1]),dict(cohort_index=users[0])])

    def test_invalid_scope_rejected_before_generation(self):
        bad=dict(phase_schedule(0)[0],cohort_indices=[1,6])
        with self.assertRaises(ValueError):measure_pair(bad,lambda *a:self.fail(),lambda:self.fail(),lambda:self.fail(),lambda *a:self.fail())

    def test_clock_regression_rejected(self):
        with self.assertRaises(ValueError):self.run_pair(0,[10,10.2,10.1])


if __name__=='__main__':unittest.main()
