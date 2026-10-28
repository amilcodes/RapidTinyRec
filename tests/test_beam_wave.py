"""Tiny CPU wave isolation checks; no release checkpoint or CUDA speed claim."""
import functools
import unittest
from unittest.mock import patch
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM,GenerationConfig,LogitsProcessorList
from transformers.cache_utils import DynamicCache
from rapidtinyrec.beam_wave import WaveLayout,WavePrefixCache,WavePrefill
from rapidtinyrec.decode_replay import cache_from_pairs
from rapidtinyrec.gpu_constraints import GpuSidConstraintTables,GpuSidConstrainedLogitsProcessor
from test_decoding import catalog


class WaveCacheTests(unittest.TestCase):
    def cache(self):
        keys=torch.stack((torch.full((2,3,4),5.),torch.full((2,3,4),99.)))
        values=keys+100
        return WavePrefixCache(cache_from_pairs(((keys,values),)),WaveLayout(3))

    def test_global_parent_offsets_and_cross_request_rejection(self):
        layout=WaveLayout(3);parents=layout.global_parents(torch.tensor([[2,2,0],[1,0,1]]))
        self.assertEqual(parents.tolist(),[2,2,0,4,3,4]);layout.validate_parents(parents)
        for bad in ([0,1,3,3,4,5],[0,1,2,2,4,5],[-1,1,2,3,4,5]):
            with self.assertRaisesRegex(ValueError,'boundary'):layout.validate_parents(torch.tensor(bad))
        with self.assertRaises(ValueError):layout.global_parents(torch.tensor([[0,1,3],[0,1,2]]))

    def test_attention_prefix_mapping_and_duplicate_parent_suffix_gather(self):
        cache=self.cache();layer=cache.layers[0];before=layer.prefix_keys.clone()
        new=torch.arange(6.).reshape(6,1,1,1).expand(6,2,1,4).clone()
        k,v=cache.update(new,new+10,0)
        self.assertTrue(torch.equal(k[:3,:,:3],before[:1].expand(3,-1,-1,-1)))
        self.assertTrue(torch.equal(k[3:,:,:3],before[1:].expand(3,-1,-1,-1)))
        parents=WaveLayout(3).global_parents(torch.tensor([[2,2,0],[1,0,1]]))
        cache.reorder_cache(parents)
        self.assertEqual(layer.keys[:,0,0,0].tolist(),[2,2,0,4,3,4])
        self.assertTrue(torch.equal(layer.prefix_keys,before))
        layer.keys[0].fill_(-999)
        self.assertEqual(layer.keys[1,0,0,0].item(),2)

    def test_invalid_parent_cannot_partially_mutate_cache(self):
        cache=self.cache();x=torch.arange(48.).reshape(6,2,1,4)
        cache.update(x,x+100,0);before=cache.layers[0].keys.clone()
        with self.assertRaises(ValueError):cache.reorder_cache(torch.tensor([0,1,3,3,4,5]))
        self.assertTrue(torch.equal(cache.layers[0].keys,before))

    def test_independent_wave_cache_ownership(self):
        a=self.cache();b=self.cache()
        a.layers[0].prefix_keys.fill_(-1)
        self.assertEqual(b.layers[0].prefix_keys[0,0,0,0].item(),5)
        self.assertNotEqual(a.layers[0].prefix_values.untyped_storage().data_ptr(),b.layers[0].prefix_values.untyped_storage().data_ptr())


class WaveGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.manual_seed(42)
        cls.model=Qwen2ForCausalLM(Qwen2Config(vocab_size=512,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64,
            bos_token_id=1,eos_token_id=511,pad_token_id=511)).float().eval()
        cls.model.config._attn_implementation='sdpa'
        cls.config=GenerationConfig(max_new_tokens=8,num_beams=50,num_return_sequences=50,
            length_penalty=0.,do_sample=False,use_cache=True,pad_token_id=511,eos_token_id=511)
        cls.ids=torch.tensor([[1,7,8,1,2,3],[1,19,27,1,2,3]])

    def generate(self,ids,wave):
        tables=GpuSidConstraintTables(catalog(),(1,2,3),510,511,'cpu')
        shapes=[];original=self.model.forward
        @functools.wraps(original)
        def traced(*a,**kw):
            shapes.append(tuple(kw['input_ids'].shape));return original(*a,**kw)
        import contextlib
        with patch.object(self.model,'forward',traced):
            with WavePrefill(self.model) if wave else contextlib.nullcontext():
                output=self.model.generate(ids,attention_mask=torch.ones_like(ids),generation_config=self.config,
                    logits_processor=LogitsProcessorList([GpuSidConstrainedLogitsProcessor(tables)]),
                    return_dict_in_generate=True,output_scores=True)
        self.assertEqual(self.model.forward,original)
        return output,shapes

    @torch.inference_mode()
    def test_two_request_native_beam_search_and_serial_outputs(self):
        native,_=self.generate(self.ids,False);wave,shapes=self.generate(self.ids,True)
        self.assertEqual(shapes,[(2,6)]+[(100,1)]*4)
        self.assertTrue(torch.equal(wave.sequences,native.sequences))
        torch.testing.assert_close(wave.sequences_scores,native.sequences_scores,rtol=0,atol=2e-5)
        self.assertEqual(len(wave.scores),5)
        for request in range(2):
            serial,_=self.generate(self.ids[request:request+1],False)
            self.assertTrue(torch.equal(wave.sequences[request*50:(request+1)*50],serial.sequences))
            torch.testing.assert_close(wave.sequences_scores[request*50:(request+1)*50],serial.sequences_scores,rtol=0,atol=2e-5)
        self.assertTrue(all(x.dtype==torch.float32 and x.shape==(100,512) for x in wave.scores))
        self.assertEqual(len({x.untyped_storage().data_ptr() for x in wave.scores}),5)

    @torch.inference_mode()
    def test_reversed_request_order_and_retained_outputs(self):
        first,_=self.generate(self.ids,True);saved=[x.clone() for x in first.scores]
        reversed_wave,_=self.generate(self.ids.flip(0),True)
        self.assertTrue(torch.equal(first.sequences[:50],reversed_wave.sequences[50:]))
        self.assertTrue(torch.equal(first.sequences[50:],reversed_wave.sequences[:50]))
        for old,new in zip(first.scores,saved):self.assertTrue(torch.equal(old,new))
        repeated,_=self.generate(self.ids,True)
        self.assertTrue(torch.equal(first.sequences,repeated.sequences))

    def test_context_exception_restores_model_and_excludes_nested_wave(self):
        original=self.model.forward
        with self.assertRaisesRegex(RuntimeError,'sentinel'):
            with WavePrefill(self.model):
                with self.assertRaisesRegex(ValueError,'active wave'):WavePrefill(self.model).__enter__()
                raise RuntimeError('sentinel')
        self.assertEqual(self.model.forward,original);self.assertFalse(self.model._beam_wave_active)

    def test_padded_inputs_and_nonidentical_initial_beams_rejected(self):
        ids=self.ids.repeat_interleave(50,dim=0);mask=torch.ones_like(ids);mask[0,0]=0
        with WavePrefill(self.model):
            with self.assertRaisesRegex(ValueError,'unpadded'):
                self.model(input_ids=ids,attention_mask=mask,past_key_values=DynamicCache(),use_cache=True,return_dict=True)
        ids=ids.clone();ids[1,0]=99
        with WavePrefill(self.model):
            with self.assertRaisesRegex(ValueError,'beam rows differ'):
                self.model(input_ids=ids,attention_mask=torch.ones_like(ids),past_key_values=DynamicCache(),use_cache=True,return_dict=True)


if __name__=='__main__':unittest.main()
