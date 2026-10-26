import copy
import unittest
from unittest.mock import patch
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
from rapidtinyrec.packed_mlp import PackedDecodeMLP


class PackedMLPTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.original = Qwen2MLP(Qwen2Config(hidden_size=8, intermediate_size=13)).eval()
        self.packed = PackedDecodeMLP(copy.deepcopy(self.original))

    def test_decode_math_and_prefill_exact_fallback(self):
        for batch, length in ((1,1),(5,1),(1,3),(2,7)):
            x = torch.randn(batch,length,8)
            with torch.no_grad():
                actual, expected = self.packed(x), self.original(x)
            if length == 1:
                # CPU functional check only; GPU score-bit equality is a separate gate.
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            else:
                self.assertTrue(torch.equal(actual.view(torch.int32), expected.view(torch.int32)))

    def test_weight_views_preserve_bits_without_duplicate_storage(self):
        report = self.packed.ownership()
        self.assertEqual(report['unique_weight_bytes'], 2*13*8*4)
        self.assertEqual(report['persistent_extra_weight_bytes'], 0)
        for name in ('gate_proj','up_proj'):
            self.assertTrue(torch.equal(getattr(self.packed,name).weight.view(torch.int32),
                                        getattr(self.original,name).weight.view(torch.int32)))
        self.assertNotIn('gate_up_weight', self.packed.state_dict())
        with torch.no_grad():
            self.packed.up_proj.weight[0,0] = 7.0
        self.assertEqual(self.packed.gate_up_weight[13,0].item(), 7.0)
        self.assertNotEqual(self.packed.gate_up_weight[0,0].item(), 7.0)

    def test_no_forward_weight_packing_and_projection_shapes(self):
        original_linear = torch.nn.functional.linear
        shapes = []
        def linear(x,w,bias=None):
            shapes.append(tuple(w.shape)); return original_linear(x,w,bias)
        with patch('torch.cat', side_effect=AssertionError('Forward packing')), patch('torch.nn.functional.linear', side_effect=linear):
            self.packed(torch.randn(5,1,8))
        self.assertEqual(shapes, [(26,8),(8,13)])
        shapes.clear()
        with patch('torch.nn.functional.linear', side_effect=linear):
            self.packed(torch.randn(1,3,8))
        self.assertEqual(shapes, [(13,8),(13,8),(8,13)])

    def test_fullgraph_dynamic_dispatch(self):
        fn = torch.compile(self.packed, backend='eager', fullgraph=True, dynamic=True)
        with torch.no_grad():
            for length in (1,3,7,1):
                x=torch.randn(2,length,8)
                torch.testing.assert_close(fn(x),self.packed(x),rtol=0,atol=0)
        self.packed.ownership()

    def test_reject_unsupported_installation(self):
        with self.assertRaises(ValueError): PackedDecodeMLP(self.packed)
        with self.assertRaises(ValueError): PackedDecodeMLP(copy.deepcopy(self.original).train())
        biased=copy.deepcopy(self.original)
        biased.gate_proj=torch.nn.Linear(8,13,bias=True)
        with self.assertRaises(ValueError): PackedDecodeMLP(biased)

if __name__ == '__main__': unittest.main()
