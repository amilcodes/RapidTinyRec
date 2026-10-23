"""Physical storage, branching, and real HF generation checks on CPU."""
import contextlib
import functools
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import GenerationConfig, LogitsProcessorList, Qwen2Config, Qwen2ForCausalLM
from rapidtinyrec.decode_replay import cache_from_pairs, cache_pairs
from rapidtinyrec.shared_prefix import PrefixArena, PrefixCache, split_pairs
from rapidtinyrec.prefix_replay import PrefixBuffers, PrefixOutput, ReorderSuffix, signature
from rapidtinyrec.shared_prefill import PrefillOnce
from rapidtinyrec.gpu_constraints import GpuSidConstraintTables, GpuSidConstrainedLogitsProcessor
from test_decoding import catalog


class SharedPrefixTests(unittest.TestCase):
    @torch.inference_mode()
    def test_storage_and_duplicate_parent_reorder(self):
        torch.manual_seed(710)
        arena = PrefixArena(12)
        source = tuple(tuple(torch.randn(1,2,7,8) for _ in range(2)) for _ in range(2))
        cache = arena.start_request(cache_from_pairs(source),4)
        prefixes, _ = split_pairs(cache)
        saved = tuple(tuple(x.clone() for x in p) for p in prefixes)
        direct = ReorderSuffix(SimpleNamespace(arena=arena))
        for i in range(2):
            dense = cache.update(torch.randn(4,2,1,8),torch.randn(4,2,1,8),i)
            for d,p in zip(dense,source[i]):
                self.assertTrue(torch.equal(d[:,:,:7],p.expand(4,-1,-1,-1)))
        suffix_saved = tuple(tuple(x.clone() for x in p) for p in split_pairs(cache)[1])
        result = direct.reorder(cache,torch.tensor([2,2,0,3],dtype=torch.int32))
        for p, oldp, suffix, olds in zip(split_pairs(result)[0],saved,split_pairs(result)[1],suffix_saved):
            for x, old in zip(p,oldp): self.assertTrue(torch.equal(x,old)); self.assertEqual(x.shape[0],1)
            for x, old in zip(suffix,olds): self.assertTrue(torch.equal(x,old[[2,2,0,3]]))
        self.assertEqual(result.get_seq_length(),8)
        self.assertEqual(direct.report()['prefix_bytes_reordered'],0)
        with self.assertRaisesRegex(ValueError,'aliases'):
            direct.reorder(result,torch.tensor([2,2,0,3]))
        for length in (12,7,9):
            incoming = tuple(tuple(torch.randn(1,2,length,8) for _ in range(2)) for _ in range(2))
            actual=arena.start_request(cache_from_pairs(incoming),4)
            self.assertTrue(all(torch.equal(a,b) for p,q in zip(split_pairs(actual)[0],incoming) for a,b in zip(p,q)))
        views = [arena.views(n,0)[0][0][0] for n in (7,9,12)]
        self.assertEqual(len({x.untyped_storage().data_ptr() for x in views}),1)
        self.assertTrue(all(x.is_contiguous() for x in views))
        with self.assertRaises(ValueError):arena.views(13,0)
        with self.assertRaises(ValueError):arena.views(7,4)

    @torch.inference_mode()
    def test_real_fp32_generation_across_requests_lengths_and_branches(self):
        torch.set_num_threads(1);torch.manual_seed(1062)
        model=Qwen2ForCausalLM(Qwen2Config(vocab_size=512,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64,
            bos_token_id=1,eos_token_id=511,pad_token_id=511)).float().eval()
        model.config._attn_implementation='sdpa'
        tables=GpuSidConstraintTables(catalog(),(1,2,3),510,511,'cpu')
        cfg=GenerationConfig(max_new_tokens=8,num_beams=50,num_return_sequences=50,length_penalty=0.,
            do_sample=False,use_cache=True,pad_token_id=511,eos_token_id=511)
        original=model.forward
        arena=PrefixArena(11)
        pool=SimpleNamespace(arena=arena,entries={})
        direct=ReorderSuffix(pool)
        for request in range(3):
            for history in ([1,7+request,8,1,2,3],[1,7+request,8,9,10,11,12,13,1,2,3]):
                results=[]
                for candidate in (False,True):
                    @functools.wraps(original)
                    def forward(*a,**kw):
                        self.assertFalse(a);kw['attention_mask']={'full_attention':None}
                        if candidate and kw['past_key_values'].get_seq_length():
                            key=signature(kw)
                            if key not in pool.entries:pool.entries[key]=PrefixBuffers(kw,arena)
                            buffers=pool.entries[key];buffers.load(kw)
                            result=PrefixOutput(original(**buffers.kwargs()))
                            arena.check_output(result)
                            return result.fresh()
                        return original(**kw)
                    x=torch.tensor([history]);before=direct.report()
                    with patch.object(model,'forward',forward), PrefillOnce(model,50,
                            cache_expander=arena.start_request if candidate else None), \
                            (direct.install(model) if candidate else contextlib.nullcontext()):
                        output=model.generate(x,attention_mask=torch.ones_like(x),generation_config=cfg,
                            logits_processor=LogitsProcessorList([GpuSidConstrainedLogitsProcessor(tables)]),
                            return_dict_in_generate=True,output_scores=True)
                    self.assertFalse(hasattr(model,'_reorder_cache'))
                    transitions=model.compute_transition_scores(output.sequences,output.scores,output.beam_indices,normalize_logits=False)
                    self.assertEqual(transitions.shape,(50,5))
                    results.append((output.sequences.clone(),output.sequences_scores.view(torch.int32).clone(),transitions.view(torch.int32).clone()))
                    if candidate:
                        self.assertEqual(direct.direct_calls-before['direct_calls'],4)
                        self.assertEqual(direct.final_calls-before['fallback_calls'],1)
                self.assertTrue(all(torch.equal(a,b) for a,b in zip(*results)),(request,history))
        self.assertEqual(len(pool.entries),8)
        for buffers in pool.entries.values():
            self.assertTrue(all(x.shape[0]==1 for p in buffers.prefixes for x in p))
            self.assertTrue(all(x.shape[2]<=3 for p in buffers.pairs for x in p))

if __name__=='__main__':unittest.main()
