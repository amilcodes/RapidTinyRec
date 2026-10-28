"""Synthetic CPU tests for two-request stable input storage and parent isolation."""
import unittest
from types import SimpleNamespace
import torch
from rapidtinyrec.decode_replay import cache_from_pairs
from rapidtinyrec.beam_wave_arena import WaveArena,WaveReorder,split_wave


class WaveArenaTests(unittest.TestCase):
    def prefill(self,length=3):
        k=torch.arange(2*2*length*4,dtype=torch.float32).reshape(2,2,length,4)
        return cache_from_pairs(((k,k+100),))

    def ready(self):
        arena=WaveArena(5,3,beams=3);cache=arena.start_wave(self.prefill())
        x=torch.arange(48.).reshape(6,2,1,4);cache.update(x,x+1000,0)
        return arena,cache,x

    def test_stable_views_pair_mapping_and_capacity(self):
        arena=WaveArena(5,3,3);cache=arena.start_wave(self.prefill());p,s=split_wave(cache)
        self.assertIs(p[0][0],arena.views(3,0)[0][0][0])
        self.assertTrue(torch.equal(p[0][0],self.prefill().layers[0].keys))
        pointer=p[0][0].data_ptr();arena.start_wave(self.prefill(5));again=arena.start_wave(self.prefill())
        self.assertEqual(again.layers[0].prefix_keys.data_ptr(),pointer)
        self.assertEqual(arena.report()['requests_loaded'],6)
        self.assertEqual(arena.report()['prefix_capacity_bytes'],2*2*5*4*4*2)
        with self.assertRaises(ValueError):arena.start_wave(self.prefill(6))
        with self.assertRaises(ValueError):arena.views(3,4)

    def test_direct_parent_offsets_and_duplicate_independence(self):
        arena,cache,x=self.ready();reorder=WaveReorder(arena)
        parents=arena.layout.global_parents(torch.tensor([[2,2,0],[1,0,1]]))
        result=reorder.reorder(cache,parents)
        self.assertTrue(torch.equal(result.layers[0].keys,x.index_select(0,parents)))
        result.layers[0].keys[0].fill_(-1)
        self.assertTrue(torch.equal(result.layers[0].keys[1],x[2]))
        self.assertEqual(reorder.direct_calls,1)
        self.assertEqual(reorder.suffix_bytes_written,2*x.numel()*4)

    def test_cross_request_failure_before_write(self):
        arena,cache,_=self.ready();reorder=WaveReorder(arena)
        _,targets=arena.views(3,1)
        for pair in targets:
            for x in pair:x.fill_(123)
        with self.assertRaisesRegex(ValueError,'boundary'):
            reorder.reorder(cache,torch.tensor([0,1,3,3,4,5]))
        self.assertTrue(all(bool((x==123).all()) for pair in targets for x in pair))
        self.assertEqual(reorder.direct_calls,0)

    def test_alias_and_foreign_arena_rejected(self):
        arena,cache,_=self.ready();reorder=WaveReorder(arena);parents=torch.arange(6)
        result=reorder.reorder(cache,parents)
        with self.assertRaisesRegex(ValueError,'aliases'):reorder.reorder(result,parents)
        foreign=WaveArena(5,3,3);foreign.start_wave(self.prefill())
        with self.assertRaisesRegex(ValueError,'belong'):WaveReorder(foreign).reorder(cache,parents)

    def test_final_unused_cache_keeps_native_reorder_semantics(self):
        arena,cache,_=self.ready();x=torch.ones(6,2,1,4)
        for _ in range(3):cache.update(x,x+1,0)
        reorder=WaveReorder(arena);before=cache.layers[0].keys.clone()
        parents=arena.layout.global_parents(torch.tensor([[1,1,0],[2,0,0]]))
        result=reorder.reorder(cache,parents)
        self.assertTrue(torch.equal(result.layers[0].keys,before.index_select(0,parents)))
        self.assertEqual(reorder.final_calls,1);self.assertEqual(reorder.direct_calls,0)

    def test_output_ownership_and_prefix_identity(self):
        arena,cache,_=self.ready();prefix,suffix=split_wave(cache)
        output=SimpleNamespace(prefixes=prefix,pairs=suffix,logits=torch.zeros(6,1,16))
        arena.check_output(output)
        _,output.pairs=arena.views(3,1)
        with self.assertRaisesRegex(ValueError,'aliases'):arena.check_output(output)
        output.pairs=suffix;output.prefixes=tuple(tuple(x.clone() for x in p) for p in prefix)
        with self.assertRaisesRegex(ValueError,'belong'):arena.check_output(output)


if __name__=='__main__':unittest.main()
