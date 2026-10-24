"""Diagnostic dispatch retains the graph input ownership contract on CPU."""
import unittest
from types import SimpleNamespace
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from rapidtinyrec.decode_replay import cache_from_pairs
from rapidtinyrec.shared_prefix import PrefixArena
from rapidtinyrec.prefix_replay import PrefixBuffers,PrefixReplayPool,signature
from rapidtinyrec.profile_prefix import diagnostic


class PrefixDiagnosticTests(unittest.TestCase):
    @torch.inference_mode()
    def test_stable_inputs_fresh_containers_and_suffix_outputs(self):
        arena=PrefixArena(7)
        source=((torch.randn(1,2,7,8),torch.randn(1,2,7,8)),)
        cache=arena.start_request(cache_from_pairs(source),4)
        kw={'input_ids':torch.ones(4,1,dtype=torch.long),'position_ids':torch.full((4,1),7),
            'cache_position':torch.tensor([7]),'attention_mask':{'full_attention':None},
            'use_cache':True,'return_dict':True,'logits_to_keep':1,'past_key_values':cache}
        buffers=PrefixBuffers(kw,arena)
        def forward(**args):
            c=args['past_key_values'];self.assertEqual(c.get_seq_length(),7)
            c.update(torch.ones(4,2,1,8),torch.ones(4,2,1,8),0)
            return CausalLMOutputWithPast(logits=torch.zeros(4,1,16),past_key_values=c)
        pool=PrefixReplayPool(forward,arena=arena)
        pool.entries[signature(kw)]=(buffers,None,None)
        with diagnostic(pool):
            first=pool(**kw);second=pool(**kw)
        self.assertEqual(first.past_key_values.get_seq_length(),8)
        self.assertEqual(second.past_key_values.get_seq_length(),8)
        self.assertIsNot(first.past_key_values,second.past_key_values)
        self.assertEqual(cache.get_seq_length(),7)
        self.assertEqual(pool.replays,0)


if __name__=='__main__':unittest.main()
