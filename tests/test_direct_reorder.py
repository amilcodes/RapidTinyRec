"""CPU ownership and real HF loop checks, not CUDA performance evidence."""
import contextlib
import functools
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import GenerationConfig, LogitsProcessorList, Qwen2Config, Qwen2ForCausalLM

from rapidtinyrec.decode_replay import ReplayBuffers, ReplayOutput, cache_from_pairs, cache_pairs, signature
from rapidtinyrec.direct_reorder import ReorderIntoReplay
from rapidtinyrec.gpu_constraints import GpuSidConstraintTables, GpuSidConstrainedLogitsProcessor
from rapidtinyrec.shared_prefill import PrefillOnce
from test_decode_replay import example
from test_decoding import catalog


class DirectReorderTests(unittest.TestCase):
    @torch.inference_mode()
    def test_duplicate_parents_write_stable_buffers_without_extra_kv_copy(self):
        args = example()
        buffers = ReplayBuffers(args)
        sources = cache_pairs(args['past_key_values'])
        original = [[x.clone() for x in pair] for pair in sources]
        direct = ReorderIntoReplay(SimpleNamespace(entries={signature(args): (buffers, None, None)}))
        parents = torch.tensor([2, 2, 0, 3], dtype=torch.int32)
        result = direct.reorder(args['past_key_values'], parents)
        for source, saved, target, layer in zip(sources, original, buffers.pairs, result.layers):
            self.assertIs(layer.keys, target[0])
            self.assertIs(layer.values, target[1])
            for x, old, y in zip(source, saved, target):
                self.assertTrue(torch.equal(x, old))
                self.assertTrue(torch.equal(y, old.index_select(0, parents)))
        # Instrument copy_ to prove that the next load copies only token/position
        # metadata, while the already filled K/V tensors retain their addresses.
        copied = []
        original_copy = torch.Tensor.copy_
        def copy(target, value, *a, **kw):
            copied.append(target)
            return original_copy(target, value, *a, **kw)
        with patch.object(torch.Tensor, 'copy_', copy):
            buffers.load({**args, 'past_key_values': result})
        self.assertEqual(len(copied), 3)
        self.assertEqual(direct.direct_calls, 1)
        with self.assertRaisesRegex(ValueError, 'aliases'):
            direct.reorder(result, parents)

    @torch.inference_mode()
    def test_three_way_real_generation_and_repeated_buffer_reuse(self):
        torch.set_num_threads(1)
        torch.manual_seed(1062)
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=512, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=64, bos_token_id=1, eos_token_id=511, pad_token_id=511)).double().eval()
        model.config._attn_implementation = 'sdpa'
        tables = GpuSidConstraintTables(catalog(), (1,2,3), 510, 511, 'cpu')
        cfg = GenerationConfig(max_new_tokens=8, num_beams=50, num_return_sequences=50,
            length_penalty=0., do_sample=False, use_cache=True, pad_token_id=511, eos_token_id=511)
        original = model.forward
        pool = SimpleNamespace(entries={})
        direct = ReorderIntoReplay(pool)
        for request in range(3):
            for history in ([1,7+request,8,1,2,3], [1,7+request,8,9,10,11,12,13,1,2,3]):
                results = []
                for variant in ('eager', 'staged', 'direct'):
                    calls = []
                    @functools.wraps(original)
                    def forward(*a, **kw):
                        self.assertFalse(a)
                        calls.append(tuple(kw['input_ids'].shape))
                        kw['attention_mask'] = {'full_attention': None}
                        if variant != 'eager' and kw['past_key_values'].get_seq_length():
                            key = signature(kw)
                            if key not in pool.entries:
                                pool.entries[key] = (ReplayBuffers(kw), None, None)
                            buffers = pool.entries[key][0]
                            buffers.load(kw)
                            return ReplayOutput(original(**buffers.kwargs())).fresh()
                        return original(**kw)
                    x = torch.tensor([history])
                    before = direct.report()
                    with patch.object(model, 'forward', forward), PrefillOnce(model, 50), \
                            (direct.install(model) if variant == 'direct' else contextlib.nullcontext()):
                        output = model.generate(x, attention_mask=torch.ones_like(x), generation_config=cfg,
                            logits_processor=LogitsProcessorList([GpuSidConstrainedLogitsProcessor(tables)]),
                            return_dict_in_generate=True, output_scores=True)
                    self.assertFalse(hasattr(model, '_reorder_cache'))
                    self.assertEqual(calls, [(1,len(history))]+[(50,1)]*4)
                    transitions = model.compute_transition_scores(output.sequences, output.scores,
                        output.beam_indices, normalize_logits=False)
                    self.assertEqual(tuple(transitions.shape), (50,5))
                    results.append((output.sequences, output.sequences_scores.float().view(torch.int32),
                                    transitions.float().view(torch.int32)))
                    if variant == 'direct':
                        self.assertEqual(direct.direct_calls-before['direct_calls'], 4)
                        self.assertEqual(direct.fallback_calls-before['fallback_calls'], 1)
                for candidate in results[1:]:
                    self.assertTrue(all(torch.equal(a,b) for a,b in zip(results[0],candidate)))

    def test_hook_restores_after_exception_and_rejects_custom_model(self):
        direct = ReorderIntoReplay(SimpleNamespace(entries={}))
        model = SimpleNamespace()
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            with direct.install(model):
                raise RuntimeError('interrupted')
        self.assertFalse(hasattr(model, '_reorder_cache'))
        model._reorder_cache = lambda *a: None
        with self.assertRaisesRegex(ValueError, 'already supplies'):
            with direct.install(model):
                pass


if __name__ == '__main__':
    unittest.main()
