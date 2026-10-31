"""Tiny CPU engine integration; simulated staging is explicitly not CUDA replay."""
import unittest
from types import SimpleNamespace
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM,GenerationConfig
from rapidtinyrec.beam_wave_generation import PairGeneration
from rapidtinyrec.beam_wave_outputs import unpack_requests
from rapidtinyrec.gpu_constraints import GpuSidConstraintTables
from test_decoding import catalog


class CPUStaging:
    def __init__(self,pool):
        self.real=pool;self.frozen=False;self.entries={};self.calls=0
    def __call__(self,**kwargs):
        buffers=self.real.buffer_class(kwargs,self.real.arena)
        output=self.real.output_class(self.real.forward(**buffers.kwargs()))
        self.real.arena.check_output(output);self.calls+=1
        return output.fresh()


class PairGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.manual_seed(42)
        cls.model=Qwen2ForCausalLM(Qwen2Config(vocab_size=512,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64,
            bos_token_id=1,eos_token_id=511,pad_token_id=511)).float().eval()
        cls.model.config._attn_implementation='sdpa'
        cls.config=GenerationConfig(max_new_tokens=8,num_beams=50,num_return_sequences=50,length_penalty=0.,
            do_sample=False,use_cache=True,bos_token_id=1,eos_token_id=511,pad_token_id=511)
        cls.ids=torch.tensor([[1,7,8,1,2,3],[1,19,27,1,2,3]])
        cls.tables=GpuSidConstraintTables(catalog(),(1,2,3),510,511,'cpu')

    def engine(self,wave):
        forward=self.model.forward
        engine=PairGeneration(self.model,SimpleNamespace(eager=forward,compiled=forward),self.config,self.tables,
                              'beam_wave_tf32' if wave else 'serial_pair_tf32',6)
        if wave:
            engine.prepare_inputs([1,2],[self.ids[:1],self.ids[1:]])
            engine.prepare_inputs([2,1],[self.ids[1:],self.ids[:1]])
        else:
            for i in range(2):engine.prepare_inputs([i+1],[self.ids[i:i+1]])
        return engine

    @torch.inference_mode()
    def test_both_engine_arms_preserve_untimed_reference_and_staged_outputs(self):
        original=self.model.forward
        for wave in (False,True):
            engine=self.engine(wave);users=[1,2] if wave else [1]
            dense=engine.generate(users,replay=False)
            engine.pool=CPUStaging(engine.pool)
            staged=engine.generate(users,replay=True)
            self.assertTrue(torch.equal(dense.sequences,staged.sequences))
            torch.testing.assert_close(dense.sequences_scores,staged.sequences_scores,rtol=0,atol=0)
            self.assertEqual(engine.pool.calls,4)
            raw=unpack_requests(self.model,staged,users,6)
            self.assertEqual([r['cohort_index'] for r in raw],users)
            self.assertEqual(self.model.forward,original);self.assertFalse(engine.active)
            self.assertFalse(hasattr(self.model,'_reorder_cache'))

    @torch.inference_mode()
    def test_reversed_staged_wave_keeps_saved_score_outputs(self):
        engine=self.engine(True);engine.pool=CPUStaging(engine.pool)
        first=engine.generate([1,2]);saved=[s.clone() for s in first.scores]
        second=engine.generate([2,1])
        self.assertTrue(torch.equal(first.sequences[:50],second.sequences[50:]))
        for before,after in zip(saved,first.scores):self.assertTrue(torch.equal(before,after))

    def test_wrong_input_count_length_and_frozen_preparation_rejected(self):
        engine=self.engine(True)
        for users,ids in [([3],[self.ids[:1]]),([3,3],[self.ids[:1],self.ids[1:]]),
                          ([3,4],[self.ids[:1],self.ids[1:,:5]])]:
            with self.assertRaises(ValueError):engine.prepare_inputs(users,ids)
        with self.assertRaises(ValueError):engine.generate([3,4])
        engine.pool.frozen=True
        with self.assertRaises(ValueError):engine.prepare_inputs([3,4],[self.ids[:1],self.ids[1:]])

    def test_forward_failure_restores_model_reorder_and_owner(self):
        engine=self.engine(True);original=self.model.forward
        def fail(**kwargs):raise RuntimeError('sentinel')
        engine.compiler.compiled=fail
        with self.assertRaisesRegex(RuntimeError,'sentinel'):engine.generate([1,2])
        self.assertFalse(engine.active);self.assertEqual(self.model.forward,original)
        self.assertFalse(hasattr(self.model,'_reorder_cache'));self.assertFalse(self.model._beam_wave_active)


if __name__=='__main__':unittest.main()
