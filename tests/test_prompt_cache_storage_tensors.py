"""Tiny tensor tests only; no release model or bootstrap."""
import unittest
import torch
from rapidtinyrec.audit_prompt_cache_storage import validate_tensors


class TensorSnapshotTests(unittest.TestCase):
    def fixture(self):
        # A transposed dense layout matches possible Qwen KV strides.
        x=torch.arange(12,dtype=torch.float32).reshape(1,3,2,2).transpose(1,2)
        return [((x.clone(),x.clone()),) for _ in range(4)]

    def check(self,snapshots):
        return validate_tensors(snapshots,layers=1,shape=[1,2,3,2])

    def test_noncontiguous_unchanged_bits_pass(self):
        snapshots=self.fixture();self.assertFalse(snapshots[0][0][0].is_contiguous())
        self.assertTrue(self.check(snapshots)['bit_unchanged'])

    def test_signed_zero_bit_mutation_rejected(self):
        snapshots=self.fixture();snapshots[-1][0][0][0,0,0,0]=-0.0
        self.assertTrue(torch.equal(snapshots[0][0][0],snapshots[-1][0][0]))
        with self.assertRaisesRegex(AssertionError,'bits changed'):self.check(snapshots)

    def test_nan_and_wrong_dtype_rejected(self):
        snapshots=self.fixture();snapshots[2][0][0][0,0,0,0]=float('nan')
        with self.assertRaisesRegex(AssertionError,'Nonfinite'):self.check(snapshots)
        snapshots=self.fixture();snapshots[2]=((snapshots[2][0][0].bfloat16(),snapshots[2][0][1]),)
        with self.assertRaises(AssertionError):self.check(snapshots)

    def test_missing_layer_or_stage_rejected(self):
        snapshots=self.fixture()
        with self.assertRaises(AssertionError):self.check(snapshots[:-1])
        snapshots[2]=()
        with self.assertRaises(AssertionError):self.check(snapshots)


if __name__=='__main__':unittest.main()
