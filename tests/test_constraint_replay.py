"""CPU contract checks; CUDA capture and mixed-request bits require the GPU pilot."""
import unittest

import torch

from rapidtinyrec.gpu_constraints import GpuSidConstraintTables, GpuSidConstrainedLogitsProcessor
from rapidtinyrec.constraint_replay import ConstraintReplayBuffers, ConstraintReplayPool, constrain


def tables():
    root=(1,2,3); newline=4; eos=5
    sids=[(8,12,16),(8,13,17),(9,14,18)]
    mapping={root:{8,9}}
    for sid in sids:
        for n in (1,2):mapping.setdefault(sid[:n],set()).add(sid[n])
        mapping[sid]={newline};mapping[sid+(newline,)]={eos}
    return GpuSidConstraintTables(mapping,root,newline,eos,'cpu')


class ConstraintReplayTests(unittest.TestCase):
    def test_pinned_hf_beam_search_retains_processor_output_storage(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM, GenerationConfig, LogitsProcessorList
        class RecordingProcessor(GpuSidConstrainedLogitsProcessor):
            def __init__(self, table):
                super().__init__(table); self.outputs=[]; self.saved=[]
            def __call__(self, input_ids, scores):
                result=super().__call__(input_ids,scores)
                self.outputs.append(result);self.saved.append(result.clone())
                return result
        cfg=Qwen2Config(vocab_size=32,hidden_size=8,intermediate_size=16,num_hidden_layers=1,
                       num_attention_heads=2,num_key_value_heads=1,bos_token_id=1,eos_token_id=5,pad_token_id=5)
        torch.manual_seed(42)
        model=Qwen2ForCausalLM(cfg).eval(); processor=RecordingProcessor(tables())
        generation=GenerationConfig(max_new_tokens=5,num_beams=2,num_return_sequences=2,
            eos_token_id=5,pad_token_id=5,bos_token_id=1,length_penalty=0.,do_sample=False)
        with torch.inference_mode():
            output=model.generate(torch.tensor([[1,2,3]]),attention_mask=torch.ones(1,3,dtype=torch.long),
                generation_config=generation,logits_processor=LogitsProcessorList([processor]),
                return_dict_in_generate=True,output_scores=True)
            self.assertEqual(len(output.scores),5)
            for actual,processed,saved in zip(output.scores,processor.outputs,processor.saved):
                self.assertEqual(actual.untyped_storage().data_ptr(),processed.untyped_storage().data_ptr())
                self.assertTrue(torch.equal(actual.view(torch.int32),saved.view(torch.int32)))
            transition=model.compute_transition_scores(output.sequences,output.scores,output.beam_indices,normalize_logits=False)
            self.assertEqual(tuple(transition.shape),(2,5))

    def test_same_operations_all_steps_and_unknown_prefixes(self):
        table=tables(); processor=GpuSidConstrainedLogitsProcessor(table)
        scores=torch.arange(3*32,dtype=torch.float32).reshape(3,32)/17
        # Valid, unknown in-range and out-of-range prefixes across mixed lengths.
        tokens=[[[7,1,2,3],[7,1,2,3],[7,0,0,0]],
                [[7,2,3,8],[7,2,3,9],[7,2,3,-1]],
                [[7,3,8,12],[7,3,9,13],[7,3,8,100]],
                [[7,8,12,16],[7,9,14,17],[7,8,12,-1]],
                [[7,8,12,16,4],[7,9,14,18,4],[7,0,0,0,0]]]
        for step, values in enumerate(tokens):
            ids=torch.tensor(values)
            expected=processor(ids,scores)
            actual=constrain(table,step,ids[:,-3:].contiguous(),scores)
            self.assertTrue(torch.equal(expected.view(torch.int32),actual.view(torch.int32)))

    def test_staging_owns_copies_and_accepts_mixed_prompt_lengths(self):
        scores=torch.arange(64,dtype=torch.float32).reshape(2,32)
        ids=torch.arange(12,dtype=torch.long).reshape(2,6)
        buffers=ConstraintReplayBuffers(ids,scores)
        first_scores=buffers.scores.clone(); first_ids=buffers.ids.clone()
        scores.zero_();ids.zero_()
        self.assertTrue(torch.equal(buffers.scores,first_scores))
        self.assertTrue(torch.equal(buffers.ids,first_ids))
        longer=torch.arange(20,dtype=torch.long).reshape(2,10)
        buffers.load(longer,scores)
        self.assertTrue(torch.equal(buffers.ids,longer[:,-3:]))
        with self.assertRaises(ValueError):buffers.load(longer,scores.double())
        with self.assertRaises(ValueError):buffers.load(longer,scores[:,::2])
        with self.assertRaises(ValueError):buffers.load(longer[:,:2],scores)

    def test_request_output_lifetime_prevents_overlapping_reuse(self):
        pool=ConstraintReplayPool(tables()); lease=pool.processor()
        with self.assertRaises(RuntimeError):pool.processor()
        with self.assertRaises(RuntimeError):pool.freeze()
        lease.close()
        with self.assertRaises(RuntimeError):lease(torch.zeros(2,3,dtype=torch.long),torch.zeros(2,32))
        with self.assertRaises(RuntimeError):lease.close()
        next_lease=pool.processor();self.assertEqual(next_lease.count,0)
        next_lease.close()
        with self.assertRaises(RuntimeError):pool.freeze()


if __name__ == '__main__':
    unittest.main()
