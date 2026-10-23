"""CPU ownership/shape checks; these do not establish CUDA performance."""
import unittest

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from rapidtinyrec.decode_replay import (
    DecodeReplayPool, ReplayBuffers, ReplayOutput, cache_from_pairs, cache_pairs)


def example(length=7):
    pairs=tuple((torch.randn(4,2,length,8),torch.randn(4,2,length,8)) for _ in range(2))
    return {'input_ids':torch.ones(4,1,dtype=torch.long),'position_ids':torch.full((4,1),length),
            'cache_position':torch.tensor([length]),'attention_mask':{'full_attention':None},
            'past_key_values':cache_from_pairs(pairs),'use_cache':True,'return_dict':True,'logits_to_keep':1}


class ReplayTests(unittest.TestCase):
    def test_cache_container_preserves_layer_count_and_references(self):
        args=example()
        original=cache_pairs(args['past_key_values'])
        rebuilt=cache_from_pairs(original)
        self.assertEqual(len(rebuilt.layers),2)
        self.assertEqual(rebuilt.get_seq_length(),7)
        self.assertIs(rebuilt.layers[0].keys,original[0][0])

    def test_staging_survives_mutation_and_repeated_loads(self):
        args=example()
        buffers=ReplayBuffers(args)
        pointers=[x.data_ptr() for pair in buffers.pairs for x in pair]
        temporary=buffers.kwargs()['past_key_values']
        for i in range(2):
            temporary.update(torch.randn(4,2,1,8),torch.randn(4,2,1,8),i)
        self.assertEqual(temporary.get_seq_length(),8)
        self.assertEqual(buffers.kwargs()['past_key_values'].get_seq_length(),7)
        for _ in range(3):
            incoming=example()
            incoming['input_ids'].fill_(42)
            buffers.load(incoming)
            self.assertEqual(pointers,[x.data_ptr() for pair in buffers.pairs for x in pair])
            self.assertTrue(torch.equal(buffers.tensors['input_ids'],incoming['input_ids']))
            for x,y in zip(buffers.pairs,cache_pairs(incoming['past_key_values'])):
                for a,b in zip(x,y):
                    self.assertTrue(torch.equal(a,b))
                    self.assertNotEqual(a.data_ptr(),b.data_ptr())

    def test_beam_reorder_cannot_replace_captured_tensor_references(self):
        args=example()
        original=cache_pairs(args['past_key_values'])
        snapshots=tuple(tuple(x.clone() for x in p) for p in original)
        wrapped=ReplayOutput(CausalLMOutputWithPast(logits=torch.randn(4,1,10),past_key_values=args['past_key_values']))
        borrowed=wrapped.fresh()
        borrowed.past_key_values.reorder_cache(torch.tensor([2,2,0,3]))
        for actual,expected in zip(cache_pairs(wrapped.fresh().past_key_values),snapshots):
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(actual,expected)))
        self.assertTrue(torch.equal(borrowed.past_key_values.layers[0].keys,original[0][0][[2,2,0,3]]))

    def test_reject_shape_and_semantics_changes(self):
        buffers=ReplayBuffers(example())
        with self.assertRaises(ValueError):buffers.load(example(length=8))
        args=example();args['attention_mask']=torch.ones(4,8)
        with self.assertRaises(ValueError):ReplayBuffers(args)
        args=example();args['input_ids']=torch.ones(4,2,dtype=torch.long)
        with self.assertRaises(ValueError):ReplayBuffers(args)
        args=example();args['output_attentions']=True
        with self.assertRaises(ValueError):ReplayBuffers(args)

    def test_frozen_pool_never_captures_or_falls_back(self):
        pool=DecodeReplayPool(lambda **kwargs:self.fail('Unexpected forward'))
        pool.frozen=True
        with self.assertRaisesRegex(RuntimeError,'Uncaptured'):pool(**example())
        with self.assertRaisesRegex(RuntimeError,'after freeze'):pool.prepare(example())


if __name__=='__main__':
    unittest.main()
