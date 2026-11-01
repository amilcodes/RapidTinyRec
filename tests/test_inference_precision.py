"""New opt-in policy/CLI and native CPU generation checks; CUDA proof is separate."""
import contextlib
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, GenerationConfig, LogitsProcessorList

from rapidtinyrec.inference_precision import model_dtype, precision_policy, policy_state, validate_output
from rapidtinyrec.gpu_constraints import GpuSidConstraintTables, GpuSidConstrainedLogitsProcessor
from rapidtinyrec.shared_prefill import PrefillOnce
from rapidtinyrec.folded_attention import fold_decode_attention


class PrecisionTests(unittest.TestCase):
    def test_policy_default_and_bf16_interrupt_restore(self):
        before=policy_state(torch)
        with precision_policy(torch,environ={}) as state:
            self.assertFalse(state['matmul_tf32'])
            self.assertEqual(model_dtype(torch,'fp32'),torch.float32)
        self.assertEqual(policy_state(torch),before)
        with self.assertRaises(KeyboardInterrupt):
            with precision_policy(torch,'bf16-experimental',environ={}) as state:
                self.assertTrue(state['matmul_tf32'])
                self.assertFalse(state['bf16_reduced_precision_reduction'])
                self.assertFalse(state['math_sdpa_reduced_precision_reduction'])
                raise KeyboardInterrupt()
        self.assertEqual(policy_state(torch),before)

    def test_override_unknown_mode_and_live_policy_drift_rejected(self):
        before=policy_state(torch)
        for mode,env in [('int8',{}),('bf16-experimental',{'NVIDIA_TF32_OVERRIDE':'1'})]:
            with self.assertRaises((ValueError,RuntimeError)):
                with precision_policy(torch,mode,environ=env):self.fail('must reject')
            self.assertEqual(policy_state(torch),before)
        with self.assertRaisesRegex(RuntimeError,'changed while'):
            with precision_policy(torch,environ={}):torch.backends.cudnn.allow_tf32=True
        self.assertEqual(policy_state(torch),before)

    def test_launcher_fresh_cache_and_precision_forwarding(self):
        import tempfile
        from rapidtinyrec import run
        with tempfile.TemporaryDirectory() as cache:
            with patch.object(sys,'argv',['run','folded','--cache-root',cache,'--precision','bf16-experimental']),patch.object(run.subprocess,'call',return_value=0) as call:
                with self.assertRaises(SystemExit):run.main()
                first=call.call_args.kwargs['env']['TORCHINDUCTOR_CACHE_DIR']
                self.assertIn('bf16-experimental',call.call_args.args[0])
            with patch.object(sys,'argv',['run','folded','--cache-root',cache]),patch.object(run.subprocess,'call',return_value=0) as call:
                with self.assertRaises(SystemExit):run.main()
                self.assertNotEqual(first,call.call_args.kwargs['env']['TORCHINDUCTOR_CACHE_DIR'])
                self.assertNotIn('--precision',call.call_args.args[0])

    def test_cli_default_and_existing_output_rejected_without_cuda(self):
        import tempfile
        from rapidtinyrec import benchmark_inference as runner
        with tempfile.TemporaryDirectory() as existing:
            with patch.object(sys,'argv',['runner','--release-root','unused','--reference-results','unused','--output-root',existing]),patch.object(runner,'validate_runtime') as validate:
                with self.assertRaises(SystemExit):runner.main()
                validate.assert_not_called()

    def test_bf16_native_beam_keeps_owned_fp32_scores_and_restores_hooks(self):
        torch.set_num_threads(4);torch.manual_seed(50111)
        config=Qwen2Config(vocab_size=512,hidden_size=96,intermediate_size=160,
            num_hidden_layers=2,num_attention_heads=6,num_key_value_heads=2,
            max_position_embeddings=128,pad_token_id=0,eos_token_id=5,bos_token_id=1)
        grammar={(1,2,3):[100,101,102,103]}
        for a in range(100,104):
            grammar[(a,)]=[110,111,112,113]
            for b in range(110,114):
                grammar[(a,b)]=[120,121,122,123]
                for c in range(120,124):
                    grammar[(a,b,c)]=[4]
                    grammar[(a,b,c,4)]=[5]
        tables=GpuSidConstraintTables(grammar,(1,2,3),4,5,'cpu')
        ids=torch.tensor([[40,41,42,43,1,2,3]])
        mask=torch.ones_like(ids);ids_before=ids.clone()
        for mode in ('fp32','bf16-experimental'):
            model=Qwen2ForCausalLM(config).to(dtype=model_dtype(torch,mode)).eval()
            original=model.forward;params={k:p.clone() for k,p in model.named_parameters()}
            cfg=GenerationConfig(max_new_tokens=8,num_beams=50,num_return_sequences=50,length_penalty=0.,do_sample=False,pad_token_id=0,eos_token_id=5,bos_token_id=1)
            outputs=[]
            with torch.inference_mode(),precision_policy(torch,mode,environ={}),fold_decode_attention(True):
                for _ in range(2):
                    with PrefillOnce(model,50):
                        out=model.generate(ids,attention_mask=mask,generation_config=cfg,logits_processor=LogitsProcessorList([GpuSidConstrainedLogitsProcessor(tables)]),return_dict_in_generate=True,output_scores=True)
                    validate_output(torch,out,mode)
                    outputs.append(out)
                self.assertTrue(torch.equal(outputs[0].sequences,outputs[1].sequences))
                self.assertTrue(torch.equal(outputs[0].beam_indices,outputs[1].beam_indices))
                for a,b in zip(outputs[0].scores,outputs[1].scores):
                    self.assertTrue(torch.equal(a.view(torch.int32),b.view(torch.int32)))
                    self.assertNotEqual(a.data_ptr(),b.data_ptr())
                bad=SimpleNamespace(scores=(out.scores[0],)*5,sequences_scores=out.sequences_scores)
                with self.assertRaisesRegex(RuntimeError,'alias'):validate_output(torch,bad,mode)
            self.assertEqual(model.forward,original)
            self.assertTrue(torch.equal(ids,ids_before))
            self.assertTrue(all(torch.equal(p,params[k]) for k,p in model.named_parameters()))


if __name__=='__main__':unittest.main()
