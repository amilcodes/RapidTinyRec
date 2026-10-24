"""Synthetic trace checks: graph attribution, overlap and fused-kernel labels."""
import unittest
from rapidtinyrec.audit_prefix_profile import analyze_events, kernel_category, exact


class PrefixTraceTests(unittest.TestCase):
    def test_graph_activity_attribution_and_overlapping_gpu_time(self):
        events=[
            {'cat':'user_annotation','name':'section/request','tid':1,'ts':0,'dur':10000},
            {'cat':'user_annotation','name':'section/graph_decode','tid':1,'ts':100,'dur':100},
            {'cat':'cuda_runtime','name':'cudaGraphLaunch','ph':'X','tid':1,'ts':110,'dur':10,'args':{'correlation':7}},
            {'cat':'kernel','name':'ampere_sgemm','ts':1000,'dur':2000,'args':{'correlation':7}},
            {'cat':'kernel','name':'fmha_cutlass','ts':2000,'dur':2000,'args':{'correlation':7}},
        ]
        trace=analyze_events(events)
        self.assertEqual(trace['kernel_sum_ms'],4.)
        self.assertEqual(trace['kernel_union_ms'],3.)
        self.assertEqual(trace['stages']['graph_decode']['activity_sum_ms'],4.)
        self.assertEqual({r['category'] for r in trace['kernels']},{'attention','matrix_multiplication'})

    def test_fused_cat_is_not_mislabeled_as_pure_attention(self):
        self.assertEqual(kernel_category('triton_poi_fused__scaled_dot_product_efficient_attention_add_cat_cos_3'),'fused_cat_layout_rope')
        self.assertEqual(kernel_category('fmha_cutlassF_f32_aligned_64x64'),'attention')

    def test_score_bit_mismatch_is_not_exact(self):
        result={'matching_rank_positions':50,'rank_positions':50,'matching_sequence_values':49,
                'sequence_values':50,'matching_transition_values':250,'transition_values':250}
        self.assertFalse(exact(result))


if __name__=='__main__':unittest.main()
