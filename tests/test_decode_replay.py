"""CPU ownership/shape checks; these do not establish CUDA performance."""
import functools
import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import GenerationConfig, LogitsProcessorList, Qwen2Config, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from rapidtinyrec.decode_replay import (
    DecodeReplayPool, ReplayBuffers, ReplayOutput, cache_from_pairs, cache_pairs)
from rapidtinyrec.gpu_constraints import GpuSidConstraintTables, GpuSidConstrainedLogitsProcessor
from rapidtinyrec.shared_prefill import PrefillOnce
from rapidtinyrec.profile_decode_replay import instrument
from test_decoding import catalog


def example(length=7):
    pairs=tuple((torch.randn(4,2,length,8),torch.randn(4,2,length,8)) for _ in range(2))
    return {'input_ids':torch.ones(4,1,dtype=torch.long),'position_ids':torch.full((4,1),length),
            'cache_position':torch.tensor([length]),'attention_mask':{'full_attention':None},
            'past_key_values':cache_from_pairs(pairs),'use_cache':True,'return_dict':True,'logits_to_keep':1}


class ReplayTests(unittest.TestCase):
    def test_profile_hooks_restore_after_exception(self):
        class FakeScopes:
            def __init__(self):self.names=[]
            @contextlib.contextmanager
            def scope(self,name):
                self.names.append(name)
                yield
        scopes=FakeScopes()
        model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1)).eval()
        compiler=SimpleNamespace(compiled=model.forward)
        forward,prepare,load,replay=compiler.compiled,model.prepare_inputs_for_generation,ReplayBuffers.load,torch.cuda.CUDAGraph.replay
        with self.assertRaisesRegex(RuntimeError,'profiling failed'):
            with instrument(scopes,model,compiler),torch.inference_mode():
                output=compiler.compiled(input_ids=torch.tensor([[1,2,3]]),use_cache=True)
                output.past_key_values.reorder_cache(torch.tensor([0]))
                raise RuntimeError('profiling failed')
        self.assertIn('compiled_prefill',scopes.names)
        self.assertIn('cache_reorder',scopes.names)
        self.assertEqual(compiler.compiled,forward)
        self.assertEqual(model.prepare_inputs_for_generation,prepare)
        self.assertIs(ReplayBuffers.load,load)
        self.assertIs(torch.cuda.CUDAGraph.replay,replay)

    @torch.inference_mode()
    def test_generation_kwargs_and_all_five_scores_survive_cpu_staging(self):
        torch.set_num_threads(1)
        torch.manual_seed(625)
        config=Qwen2Config(vocab_size=512,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
            max_position_embeddings=64,bos_token_id=1,eos_token_id=511,pad_token_id=511)
        model=Qwen2ForCausalLM(config).double().eval()
        model.config._attn_implementation='sdpa'
        tables=GpuSidConstraintTables(catalog(),(1,2,3),510,511,'cpu')
        cfg=GenerationConfig(max_new_tokens=8,num_beams=50,num_return_sequences=50,
            length_penalty=0.,do_sample=False,use_cache=True,pad_token_id=511,eos_token_id=511)
        original=model.forward
        for history in ([1,7,8,1,2,3],[1,7,8,9,10,11,12,13,1,2,3]):
            results=[]
            for staged in (False,True):
                calls=[]
                @functools.wraps(original)
                def forward(*args,**kwargs):
                    self.assertFalse(args)
                    calls.append(tuple(kwargs['input_ids'].shape))
                    kwargs['attention_mask']={'full_attention':None}
                    if staged and kwargs['past_key_values'].get_seq_length():
                        # Exercise the actual prepare_inputs_for_generation keys,
                        # including its explicit inputs_embeds=None placeholder.
                        self.assertIn('inputs_embeds',kwargs)
                        self.assertIsNone(kwargs['inputs_embeds'])
                        buffers=ReplayBuffers(kwargs)
                        self.assertIn('inputs_embeds',buffers.kwargs())
                        return ReplayOutput(original(**buffers.kwargs())).fresh()
                    return original(**kwargs)
                x=torch.tensor([history])
                with patch.object(model,'forward',forward),PrefillOnce(model,50):
                    output=model.generate(x,attention_mask=torch.ones_like(x),generation_config=cfg,
                        logits_processor=LogitsProcessorList([GpuSidConstrainedLogitsProcessor(tables)]),
                        return_dict_in_generate=True,output_scores=True)
                self.assertEqual(calls,[(1,len(history))]+[(50,1)]*4)
                transitions=model.compute_transition_scores(output.sequences,output.scores,
                    output.beam_indices,normalize_logits=False)
                self.assertEqual(tuple(transitions.shape),(50,5))
                results.append((output.sequences,output.sequences_scores.float().view(torch.int32),
                    transitions.float().view(torch.int32)))
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(*results)))

    @torch.inference_mode()
    def test_staged_decode_matches_qwen_after_duplicate_parent_selection(self):
        torch.set_num_threads(1)
        torch.manual_seed(914)
        config=Qwen2Config(vocab_size=512,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
            max_position_embeddings=64)
        model=Qwen2ForCausalLM(config).double().eval()
        model.config._attn_implementation='sdpa'
        for length in (7,12):
            buffers=None
            pointers=None
            for request in range(3):
                prefix=model(input_ids=torch.randint(0,512,(1,length)),
                    use_cache=True,return_dict=True,logits_to_keep=1)
                cache=prefix.past_key_values
                cache.batch_repeat_interleave(50)
                # Give rows distinct suffix states before selecting parents,
                # so a permutation/duplication error cannot hide behind copies.
                first=model(input_ids=torch.arange(50).view(50,1),past_key_values=cache,
                    position_ids=torch.full((50,1),length),cache_position=torch.tensor([length]),
                    attention_mask={'full_attention':None},use_cache=True,return_dict=True,logits_to_keep=1)
                first.past_key_values.reorder_cache(torch.arange(49,-1,-1)//2)
                incoming={'input_ids':torch.randint(0,512,(50,1)),
                    'past_key_values':first.past_key_values,'position_ids':torch.full((50,1),length+1),
                    'cache_position':torch.tensor([length+1]),'attention_mask':{'full_attention':None},
                    'use_cache':True,'return_dict':True,'logits_to_keep':1}
                reference=model(**{**incoming,'past_key_values':cache_from_pairs(cache_pairs(incoming['past_key_values']))})
                if buffers is None:
                    buffers=ReplayBuffers(incoming)
                    pointers=[x.data_ptr() for pair in buffers.pairs for x in pair]
                else:
                    buffers.load(incoming)
                actual=model(**buffers.kwargs())
                self.assertTrue(torch.equal(reference.logits,actual.logits),(length,request))
                for expected_pair,actual_pair in zip(cache_pairs(reference.past_key_values),cache_pairs(actual.past_key_values)):
                    self.assertTrue(all(torch.equal(a,b) for a,b in zip(expected_pair,actual_pair)))
                self.assertEqual(pointers,[x.data_ptr() for pair in buffers.pairs for x in pair])
                self.assertEqual(buffers.kwargs()['past_key_values'].get_seq_length(),length+1)

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
        args=example();args['inputs_embeds']=torch.randn(4,1,16)
        with self.assertRaisesRegex(ValueError,'embedding inputs'):ReplayBuffers(args)

    def test_frozen_pool_never_captures_or_falls_back(self):
        pool=DecodeReplayPool(lambda **kwargs:self.fail('Unexpected forward'))
        pool.frozen=True
        with self.assertRaisesRegex(RuntimeError,'Uncaptured'):pool(**example())
        with self.assertRaisesRegex(RuntimeError,'after freeze'):pool.prepare(example())


if __name__=='__main__':
    unittest.main()
