"""Numerical audit checks run with the standard library, without a model."""
import copy
import unittest

from rapidtinyrec.audit_decode_pilot import compare, validate_run


def sample():
    return {'elapsed_s': 0.1,
            'token_ids': [[1, 2, i, 198, 151645] for i in range(50)],
            'sequence_logp': [-15.] * 50,
            'transition_logp': [[-1., -2., -3., -4., -5.] for _ in range(50)]}


class PilotAuditTests(unittest.TestCase):
    def test_candidate_alignment_does_not_hide_rank_changes(self):
        left = sample()
        left['sequence_logp'][0] = -16.
        left['transition_logp'][0][0] = -2.
        right = copy.deepcopy(left)
        for field in ('token_ids', 'sequence_logp', 'transition_logp'):
            right[field][0], right[field][1] = right[field][1], right[field][0]
        validate_run(left)
        validate_run(right)
        result = compare([(left, right)])
        self.assertEqual(result['matching_rank_lists'], 0)
        self.assertEqual(result['matching_rank_positions'], 48)
        self.assertEqual(result['matching_sequence_values'], 48)
        self.assertEqual(result['matching_transition_values'], 248)
        self.assertEqual(result['max_sequence_difference_common'], 0.)
        self.assertEqual(result['max_transition_difference_common'], 0.)

    def test_signed_zero_is_not_bit_exact(self):
        left = sample()
        right = copy.deepcopy(left)
        left['sequence_logp'][0], right['sequence_logp'][0] = 0., -0.
        self.assertEqual(left['sequence_logp'], right['sequence_logp'])
        result = compare([(left, right)])
        self.assertEqual(result['matching_sequence_values'], 49)
        self.assertEqual(result['matching_sequence_vectors'], 0)
        self.assertEqual(result['max_sequence_difference_common'], 0.)

    def test_small_drift_is_counted_without_tolerance(self):
        left = sample()
        right = copy.deepcopy(left)
        right['sequence_logp'][0] += 2 ** -20
        result = compare([(left, right)])
        self.assertEqual(result['matching_sequence_values'], 49)
        self.assertEqual(result['max_sequence_difference_common'], 2 ** -20)

    def test_missing_suffix_probability_and_nonfinite_latency_fail(self):
        run = sample()
        validate_run(run)
        run['sequence_logp'][0] = -6.
        with self.assertRaisesRegex(AssertionError, 'Five-token FP32 sum'):
            validate_run(run)
        run = sample()
        run['elapsed_s'] = float('nan')
        with self.assertRaises(AssertionError):
            validate_run(run)


if __name__ == '__main__':
    unittest.main()
