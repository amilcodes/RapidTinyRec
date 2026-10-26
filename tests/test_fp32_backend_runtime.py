import gzip
from pathlib import Path
import tempfile
import unittest
import torch
from rapidtinyrec.diagnose_fp32_backend import TensorStore
from rapidtinyrec.fp32_backend_probe import backend_scope


class RuntimeControlTests(unittest.TestCase):
    def test_default_preference_resolves_to_cublas_in_pinned_build(self):
        # The getter resolves Default in Context.cpp; accepting a literal Default
        # enum here would make the GPU diagnostic reject the actual pinned API.
        api=torch.backends.cuda.preferred_blas_library
        previous=api()
        with backend_scope(api,'default') as resolved:self.assertTrue(resolved.endswith('.Cublas'))
        self.assertEqual(api(),previous)

    def test_raw_store_preserves_signed_zero_bits_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as d:
            store=TensorStore(Path(d))
            x=torch.tensor([0, -2147483648, 1065353217],dtype=torch.int32).view(torch.float32)
            ref=store.put(x);again=store.put(x.clone())
            self.assertEqual(ref,again);self.assertEqual(len(store.entries),1)
            raw=gzip.decompress((Path(d)/store.entries[ref['sha256']]['file']).read_bytes())
            self.assertEqual(raw,x.numpy().tobytes())
            with self.assertRaises(AssertionError):store.put(x.half())


if __name__=='__main__':unittest.main()
