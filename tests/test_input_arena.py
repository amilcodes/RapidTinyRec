"""Storage ownership checks; CUDA replay correctness is gated by the GPU pilot."""
from types import SimpleNamespace
import unittest
import torch
from rapidtinyrec.decode_replay import DecodeReplayPool, ReplayBuffers, cache_pairs, signature
from rapidtinyrec.direct_reorder import ReorderIntoReplay
from rapidtinyrec.input_arena import SharedInputArena
from test_decode_replay import example


class ArenaTests(unittest.TestCase):
    @torch.inference_mode()
    def test_exact_views_ownership_and_duplicate_parents_across_lengths(self):
        arena = SharedInputArena(12)
        pool = DecodeReplayPool(None, arena=arena)
        args = {n: example(n) for n in (7,12)}
        for n,kw in args.items():
            buf=ReplayBuffers(kw,arena)
            pool.entries[signature(kw)]=(buf,None,None)
            for source,target in zip(cache_pairs(kw['past_key_values']),buf.pairs):
                for x,y in zip(source,target):
                    self.assertEqual(x.stride(),y.stride())
                    self.assertTrue(y.is_contiguous())
        small,big=[e[0] for e in pool.entries.values()]
        for a,b in zip(small.pairs,big.pairs):
            for x,y in zip(a,b):self.assertEqual(x.data_ptr(),y.data_ptr())
        direct=ReorderIntoReplay(pool)
        parents=torch.tensor([3,3,0,2],dtype=torch.int32)
        for n in (7,12,7,12,7):
            kw=example(n)
            sources=cache_pairs(kw['past_key_values'])
            saved=[[x.clone() for x in p] for p in sources]
            cache=direct.reorder(kw['past_key_values'],parents)
            buf=pool.entries[signature(kw)][0]
            buf.load({**kw,'past_key_values':cache})
            for pair,old,target in zip(sources,saved,buf.pairs):
                for x,y,z in zip(pair,old,target):
                    self.assertTrue(torch.equal(x,y))
                    self.assertTrue(torch.equal(z,y.index_select(0,parents)))
            arena.check_output(SimpleNamespace(logits=torch.zeros(1),pairs=sources))
            with self.assertRaisesRegex(ValueError,'aliases'):
                arena.check_output(SimpleNamespace(logits=torch.zeros(1),pairs=buf.pairs))
        physical=2*2*4*2*12*8*4 + 2*(4+4+1)*8
        self.assertEqual(pool.report()['unique_input_storage_bytes'],physical)
        self.assertGreater(pool.report()['input_buffer_bytes'],physical)

    def test_reject_overflow_layout_and_dtype_changes(self):
        arena=SharedInputArena(12)
        ReplayBuffers(example(),arena)
        with self.assertRaisesRegex(ValueError,'capacity'):ReplayBuffers(example(13),arena)
        pairs=cache_pairs(example()['past_key_values'])
        with self.assertRaisesRegex(ValueError,'changed'):
            arena.views(tuple(tuple(x.double() for x in p) for p in pairs))
        with self.assertRaisesRegex(ValueError,'contiguous'):
            arena.views(tuple(tuple(x.transpose(1,3) for x in p) for p in pairs))
