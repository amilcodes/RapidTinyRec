"""Small CPU models exercise beam branching and exact constraint score masks."""
import collections
import contextlib
import functools
import unittest
from unittest.mock import patch

import torch
from transformers import GenerationConfig, LogitsProcessorList, Qwen2Config, Qwen2ForCausalLM

from rapidtinyrec.bulk_constraints import BulkPrefixConstrainedLogitsProcessor
from rapidtinyrec.gpu_constraints import GpuSidConstraintTables, GpuSidConstrainedLogitsProcessor
from rapidtinyrec.shared_prefill import PrefillOnce
from rapidtinyrec.source.LogitProcessor import ConstrainedLogitsProcessor


def catalog():
    constraints=collections.defaultdict(set)
    root=(1,2,3)
    for a in range(40,45):
        for b in range(70,75):
            for c in range(100,104):
                sid=(a,b,c)
                constraints[root].add(a)
                constraints[sid[:1]].add(b)
                constraints[sid[:2]].add(c)
                constraints[sid].add(510)
                constraints[sid+(510,)].add(511)
    return {k:list(v) for k,v in constraints.items()}


class DecodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        config=Qwen2Config(vocab_size=512,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
            max_position_embeddings=64,bos_token_id=1,eos_token_id=511,pad_token_id=511)
        cls.model=Qwen2ForCausalLM(config).double().eval()
        cls.model.config._attn_implementation='sdpa'

    @torch.inference_mode()
    def test_prefill_shapes_rankings_and_cache_expansion(self):
        model=self.model
        constraints=catalog()
        config=GenerationConfig(max_new_tokens=8,num_beams=50,num_return_sequences=50,
            length_penalty=0.,do_sample=False,use_cache=True,pad_token_id=511,eos_token_id=511)
        for history in ([1,7,8,1,2,3],[1,7,8,9,10,11,12,13,1,2,3]):
            with self.subTest(length=len(history)):
                outputs=[]
                for shared in (False,True):
                    shapes=[]
                    cache_shapes=[]
                    original=model.forward
                    @functools.wraps(original)
                    def forward(*args,**kwargs):
                        shapes.append(tuple(kwargs['input_ids'].shape))
                        return original(*args,**kwargs)
                    def capture(phase,cache):
                        cache_shapes.append((phase,cache.layers[0].keys.shape[0],cache.get_seq_length()))
                    processor=ConstrainedLogitsProcessor(lambda batch,ids:constraints.get(tuple(ids),[]),50,'Qwen2',511)
                    with patch.object(model,'forward',forward):
                        with PrefillOnce(model,50,on_prefill=capture) if shared else contextlib.nullcontext():
                            x=torch.tensor([history])
                            output=model.generate(x,attention_mask=torch.ones_like(x),generation_config=config,
                                logits_processor=LogitsProcessorList([processor]),return_dict_in_generate=True,output_scores=True)
                    self.assertEqual(model.forward,original)
                    self.assertEqual(shapes,[(1 if shared else 50,len(history))]+[(50,1)]*4)
                    if shared:
                        self.assertEqual(cache_shapes,[('before_expansion',1,len(history)),('after_expansion',50,len(history))])
                    outputs.append(output)
                self.assertTrue(torch.equal(outputs[0].sequences,outputs[1].sequences))
                self.assertEqual(len({tuple(s) for s in outputs[0].sequences.tolist()}),50)
                torch.testing.assert_close(outputs[0].sequences_scores,outputs[1].sequences_scores,rtol=0,atol=1e-5)

    def test_constraint_masks_include_valid_and_invalid_prefixes(self):
        constraints=catalog()
        tables=GpuSidConstraintTables(constraints,(1,2,3),510,511,'cpu')
        scores=torch.randn(50,512)
        for step in range(6):
            length=3 if step==0 else step
            cases=[k for k in constraints if len(k)==length]+[(0,)*length,(-1,)*length,(511,)*length]
            cases=(cases[-20:]*50)[:50]
            x=torch.tensor(cases)
            processors=[ConstrainedLogitsProcessor(lambda batch,ids:constraints.get(tuple(ids),[]),50,'Qwen2',511),
                BulkPrefixConstrainedLogitsProcessor(lambda batch,ids:constraints.get(tuple(ids),[]),50,'Qwen2',511),
                GpuSidConstrainedLogitsProcessor(tables)]
            values=[]
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',UserWarning)
                for proc in processors:
                    proc.count=step
                    values.append(proc(x,scores))
            for value in values[1:]:
                self.assertTrue(torch.equal(values[0].view(torch.int32),value.view(torch.int32)),step)

    def test_wrapper_restores_forward_after_exception(self):
        original=self.model.forward
        with self.assertRaisesRegex(RuntimeError,'test failure'):
            with PrefillOnce(self.model,50):
                raise RuntimeError('test failure')
        self.assertEqual(self.model.forward,original)


if __name__=='__main__':
    unittest.main()
