"""Tiny Qwen2 CPU checks; no release checkpoint, CUDA, or cohort bootstrap."""
import functools
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, GenerationConfig, LogitsProcessorList

from rapidtinyrec.decode_replay import cache_pairs
from rapidtinyrec.prompt_cache import FixedPromptCache, PromptIdentity
from rapidtinyrec.prompt_cache_generation import PromptCacheGenerator
from rapidtinyrec.prefix_prefill import PrefillPrefix
from rapidtinyrec.shared_prefix import PrefixArena
from rapidtinyrec.gpu_constraints import GpuSidConstraintTables, GpuSidConstrainedLogitsProcessor
from test_decoding import catalog


class PromptCacheTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(891)
        self.model = Qwen2ForCausalLM(Qwen2Config(vocab_size=512, hidden_size=32,
            intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=64,
            bos_token_id=1, eos_token_id=511, pad_token_id=511)).float().eval()
        self.model.config._attn_implementation = 'sdpa'
        self.identity = PromptIdentity('a'*64, 'b'*64, 'fp32-cpu-test')
        self.prefix = torch.tensor([[1, 17, 21, 33]])

    def make_cache(self):
        return FixedPromptCache(self.model, self.prefix, self.identity)

    @torch.inference_mode()
    def test_suffix_logits_and_complete_kv_match_full_prefill(self):
        cache = self.make_cache()
        for suffix in ([14], [7, 8, 9], [19, 2, 10, 4, 7, 11]):
            ids = torch.cat((self.prefix, torch.tensor([suffix])), dim=1)
            reference = self.model(input_ids=ids, attention_mask={'full_attention': None},
                use_cache=True, return_dict=True, logits_to_keep=1)
            result = cache.prefill(cache.prepare(ids))
            torch.testing.assert_close(result.logits, reference.logits, atol=2e-7, rtol=2e-5)
            for got, want in zip(cache_pairs(result.past_key_values), cache_pairs(reference.past_key_values)):
                for x, y in zip(got, want):
                    torch.testing.assert_close(x, y, atol=2e-7, rtol=2e-5)

    def test_non_square_mask_absolute_positions_and_staging_ownership(self):
        cache = self.make_cache()
        ids = torch.tensor([[1, 17, 21, 33, 8, 9, 10]])
        prepared = cache.prepare(ids)
        self.assertEqual(prepared.cache_position.tolist(), [4, 5, 6])
        self.assertEqual(prepared.position_ids.tolist(), [[4, 5, 6]])
        self.assertEqual((prepared.attention_mask[0, 0] == 0).tolist(),
            [[True]*5+[False]*2, [True]*6+[False], [True]*7])
        ids[0, -1] = 77
        self.assertEqual(prepared.input_ids.tolist(), [[8, 9, 10]])

    @torch.inference_mode()
    def test_requests_and_reorder_cannot_mutate_persistent_kv(self):
        cache = self.make_cache()
        saved = cache.snapshot()
        first = cache.prefill(cache.prepare(torch.tensor([[1, 17, 21, 33, 10, 12]])))
        old_logits = first.logits.clone()
        first.past_key_values.batch_repeat_interleave(3)
        first.past_key_values.reorder_cache(torch.tensor([2, 2, 0]))
        for pair in cache_pairs(first.past_key_values):
            for x in pair: x.fill_(900)
        other = cache.prefill(cache.prepare(torch.tensor([[1, 17, 21, 33, 51, 9, 7]])))
        repeated = cache.prefill(cache.prepare(torch.tensor([[1, 17, 21, 33, 10, 12]])))
        self.assertTrue(torch.equal(old_logits, repeated.logits))
        for a, b in zip(cache.snapshot(), saved):
            for x, y in zip(a, b): self.assertTrue(torch.equal(x, y))
        ptrs = lambda out: {x.untyped_storage().data_ptr() for p in cache_pairs(out.past_key_values) for x in p}
        self.assertFalse(ptrs(first) & ptrs(other))
        self.assertFalse(ptrs(other) & ptrs(repeated))
        self.assertEqual(cache.report()['cached_kv_bytes'], 2*2*1*2*4*8*4)

    def test_wrong_prefix_empty_suffix_or_foreign_prepared_request_rejected(self):
        cache = self.make_cache()
        for ids in ([[1, 17, 21, 32, 9]], [[1, 17, 21, 33]], [[1, 17, 21, 33, 9]]*2):
            with self.assertRaises(ValueError): cache.prepare(torch.tensor(ids))
        another = self.make_cache()
        prepared = another.prepare(torch.tensor([[1, 17, 21, 33, 8]]))
        with self.assertRaisesRegex(ValueError, 'another cache'): cache.prefill(prepared)

    def test_policy_config_and_weight_changes_invalidate_cache(self):
        cache = self.make_cache()
        prepared = cache.prepare(torch.tensor([[1, 17, 21, 33, 8]]))
        old = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = not old
            with self.assertRaisesRegex(ValueError, 'changed'): cache.prefill(prepared)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old
        self.model.config.rope_theta *= 2
        with self.assertRaisesRegex(ValueError, 'changed'): cache.prefill(prepared)
        self.model.config.rope_theta /= 2
        with torch.no_grad(): next(self.model.parameters()).add_(0.01)
        with self.assertRaisesRegex(ValueError, 'changed'): cache.prefill(prepared)

    def test_training_and_dynamic_rope_and_invalid_identity_rejected(self):
        with self.assertRaises(ValueError): PromptIdentity('bad', 'b'*64, 'fp32')
        self.model.train()
        with self.assertRaisesRegex(ValueError, 'eval'): self.make_cache()
        self.model.eval()
        self.model.config.rope_scaling = {'rope_type': 'dynamic', 'factor': 2.0}
        with self.assertRaisesRegex(ValueError, 'static'): self.make_cache()

    @torch.inference_mode()
    def test_fifty_beam_five_scores_and_existing_arena_across_requests(self):
        cache = self.make_cache()
        original = self.model.forward
        arena = PrefixArena(12)
        tables = GpuSidConstraintTables(catalog(), (1, 2, 3), 510, 511, 'cpu')
        cfg = GenerationConfig(max_new_tokens=8, num_beams=50, num_return_sequences=50,
            length_penalty=0., do_sample=False, use_cache=True, pad_token_id=511, eos_token_id=511)
        for suffix in ([8, 1, 2, 3], [9, 7, 14, 1, 2, 3], [8, 1, 2, 3]):
            ids = torch.cat((self.prefix, torch.tensor([suffix])), dim=1)
            prepared = cache.prepare(ids)
            results = []
            for candidate in (False, True):
                @functools.wraps(original)
                def forward(*args, **kw):
                    if candidate and kw['past_key_values'].get_seq_length() == 0:
                        return cache.prefill(prepared, forward=original)
                    kw['attention_mask'] = {'full_attention': None}
                    return original(*args, **kw)
                with patch.object(self.model, 'forward', forward), PrefillPrefix(
                        self.model, 50, cache_expander=arena.start_request):
                    out = self.model.generate(ids, attention_mask=torch.ones_like(ids),
                        generation_config=cfg, return_dict_in_generate=True, output_scores=True,
                        logits_processor=LogitsProcessorList([GpuSidConstrainedLogitsProcessor(tables)]))
                self.assertEqual(len(out.scores), 5)
                self.assertTrue(all(s.dtype == torch.float32 for s in out.scores))
                self.assertEqual(len({s.untyped_storage().data_ptr() for s in out.scores}), 5)
                results.append((out.sequences.clone(), out.sequences_scores.clone(),
                    self.model.compute_transition_scores(out.sequences, out.scores,
                        out.beam_indices, normalize_logits=False).clone()))
            self.assertTrue(torch.equal(results[0][0], results[1][0]))
            for a, b in zip(results[0][1:], results[1][1:]):
                torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)

    @torch.inference_mode()
    def test_generation_adapter_routes_masked_suffix_and_four_decode_steps(self):
        cache = self.make_cache()
        original = self.model.forward
        arena = PrefixArena(12)
        cfg = GenerationConfig(max_new_tokens=8, num_beams=50, num_return_sequences=50,
            length_penalty=0., do_sample=False, use_cache=True, pad_token_id=511, eos_token_id=511)
        tables = GpuSidConstraintTables(catalog(), (1, 2, 3), 510, 511, 'cpu')
        events = []
        def prefill(**kw):
            self.assertEqual(kw['past_key_values'].get_seq_length(), 4)
            self.assertIsNotNone(kw['attention_mask']['full_attention'])
            self.assertEqual(kw['position_ids'][0, 0].item(), 4)
            events.append('suffix')
            return original(**kw)
        def decode(**kw):
            self.assertEqual(kw['input_ids'].shape, (50, 1))
            self.assertIsNone(kw['attention_mask']['full_attention'])
            events.append('decode')
            return original(**kw)
        generator = PromptCacheGenerator(cache, cfg, arena, prefill_forward=prefill, decode_forward=decode)
        inputs = torch.tensor([[1, 17, 21, 33, 8, 1, 2, 3]])
        request = generator.prepare(inputs)
        inputs[0, 4] = 88  # Prepared generation owns its original request.
        retained = None
        for _ in range(2):
            output = generator.generate(request, LogitsProcessorList([GpuSidConstrainedLogitsProcessor(tables)]))
            self.assertEqual(events[-5:], ['suffix']+['decode']*4)
            if retained is not None:
                self.assertTrue(torch.equal(retained, output.sequences))
            retained = output.sequences.clone()
            self.assertEqual(output.sequences[:, 4].unique().tolist(), [8])
            self.assertEqual(len({s.untyped_storage().data_ptr() for s in output.scores}), 5)
        self.assertEqual(self.model.forward, original)
        self.assertFalse(generator.active)
        other = PromptCacheGenerator(cache, cfg, arena, prefill_forward=prefill, decode_forward=decode)
        with self.assertRaisesRegex(ValueError, 'another generator'):
            other.generate(request, LogitsProcessorList())

    @torch.inference_mode()
    def test_generation_adapter_restores_model_after_forward_failure(self):
        cache = self.make_cache()
        original = self.model.forward
        cfg = GenerationConfig(max_new_tokens=8, num_beams=50, num_return_sequences=50,
            length_penalty=0., do_sample=False, use_cache=True, pad_token_id=511, eos_token_id=511)
        def fail(**kw): raise RuntimeError('simulated compiled failure')
        generator = PromptCacheGenerator(cache, cfg, PrefixArena(12),
            prefill_forward=fail, decode_forward=original)
        request = generator.prepare(torch.tensor([[1, 17, 21, 33, 8, 1, 2, 3]]))
        with self.assertRaisesRegex(RuntimeError, 'compiled failure'):
            generator.generate(request, LogitsProcessorList())
        self.assertEqual(self.model.forward, original)
        self.assertFalse(generator.active)


if __name__ == '__main__':
    unittest.main()
