"""Guard matmul role attribution against confusing identical shapes or attention."""
import unittest
from rapidtinyrec.audit_folded_profile import matmul_role, matmul_breakdown


class FoldedProfileAttributionTests(unittest.TestCase):
    def test_roles_preserve_indistinguishable_projections(self):
        self.assertEqual(matmul_role('aten::mm', [[50, 1536], [1536, 8960]]), 'mlp_gate_or_up')
        self.assertEqual(matmul_role('aten::mm', [[50, 8960], [8960, 1536]]), 'mlp_down')
        self.assertEqual(matmul_role('aten::addmm', [[1536], [50, 1536], [1536, 1536], [], []]), 'query_or_output_projection')
        self.assertEqual(matmul_role('aten::addmm', [[256], [50, 1536], [1536, 256], [], []]), 'key_or_value_projection')
        self.assertEqual(matmul_role('aten::mm', [[50, 1536], [1536, 152265]]), 'full_vocabulary_projection')
        self.assertIsNone(matmul_role('aten::bmm', [[50, 6, 128], [50, 128, 121]]))
        self.assertIsNone(matmul_role('aten::mm', [[50, 1536], [8960, 1536]]))

    def test_stage_separation_and_activity_counts(self):
        trace = {'operations': [
            {'stage': 'compiled_prefill', 'name': 'aten::mm', 'input_dims': [[121,1536],[1536,8960]], 'count': 56, 'gpu_ms': 12.},
            {'stage': 'diagnostic_decode', 'name': 'aten::mm', 'input_dims': [[50,1536],[1536,8960]], 'count': 224, 'gpu_ms': 30.},
            {'stage': 'diagnostic_decode', 'name': 'aten::mm', 'input_dims': [[50,8960],[8960,1536]], 'count': 112, 'gpu_ms': 10.},
            {'stage': 'diagnostic_decode', 'name': 'aten::_efficient_attention_forward', 'input_dims': [], 'count': 112, 'gpu_ms': 4.},
        ]}
        result = matmul_breakdown(trace)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]['role'], 'mlp_gate_or_up')
        self.assertEqual(result[0]['gpu_ms'], 30.)
        self.assertEqual(result[0]['gpu_activity_count'], 224)
        self.assertEqual(sum(r['gpu_ms'] for r in result), 52.)


if __name__ == '__main__':
    unittest.main()
