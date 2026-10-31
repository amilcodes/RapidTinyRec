"""CPU replay contract tests; graph execution still requires allocated CUDA."""
import unittest
from types import SimpleNamespace
import torch
from rapidtinyrec.decode_replay import cache_from_pairs
from rapidtinyrec.beam_wave import WavePrefill
from rapidtinyrec.beam_wave_arena import WaveArena,WaveReorder,borrowed_wave,split_wave
from rapidtinyrec.beam_wave_replay import WaveBuffers,WaveOutput,WaveReplayPool,signature


class WaveReplayTests(unittest.TestCase):
    def ready(self):
        arena=WaveArena(6,3,3)
        k=torch.arange(48,dtype=torch.float32).reshape(2,2,3,4)
        cache=arena.start_wave(cache_from_pairs(((k,k+100),)))
        x=torch.arange(48,dtype=torch.float32).reshape(6,2,1,4)
        cache.update(x,x+1000,0)
        kwargs=dict(input_ids=torch.arange(6).reshape(6,1),position_ids=torch.full((6,1),4),
                    cache_position=torch.tensor([4]),past_key_values=cache,attention_mask={'full_attention':None},
                    use_cache=True,return_dict=True,logits_to_keep=1)
        return arena,kwargs

    def test_buffers_copy_owned_tokens_borrow_prefix_and_stage_suffix(self):
        arena,kw=self.ready();buffers=WaveBuffers(kw,arena)
        incoming=kw['past_key_values'];a=buffers.kwargs();b=buffers.kwargs()
        self.assertIsNot(a['past_key_values'],b['past_key_values'])
        self.assertIs(a['past_key_values'].layers[0].prefix_keys,incoming.layers[0].prefix_keys)
        torch.testing.assert_close(a['past_key_values'].layers[0].keys,incoming.layers[0].keys)
        kw['input_ids'].fill_(99)
        self.assertEqual(buffers.tensors['input_ids'][0,0].item(),0)
        buffers.load(kw);self.assertEqual(buffers.tensors['input_ids'][0,0].item(),99)
        x=torch.zeros(6,2,1,4);a['past_key_values'].update(x,x,0)
        self.assertEqual(b['past_key_values'].get_seq_length(),4)
        self.assertEqual(buffers.kwargs()['past_key_values'].get_seq_length(),4)

    def test_staging_rejects_foreign_and_overlap_before_token_copy(self):
        arena,kw=self.ready();buffers=WaveBuffers(kw,arena)
        _,target=arena.views(3,1);prefix,_=split_wave(kw['past_key_values'])
        alias=tuple(tuple(x.view_as(x) for x in pair) for pair in target)
        bad={**kw,'input_ids':torch.full((6,1),99),'past_key_values':borrowed_wave(prefix,alias,arena.layout)}
        with self.assertRaisesRegex(ValueError,'overlaps'):buffers.load(bad)
        self.assertEqual(buffers.tensors['input_ids'][0,0].item(),0)
        foreign,other=self.ready()
        with self.assertRaisesRegex(ValueError,'belong'):buffers.load(other)
        # Exact identities from direct gather are allowed.
        buffers.load({**kw,'past_key_values':borrowed_wave(prefix,target,arena.layout)})

    def test_signature_rejects_missing_request_and_unsafe_mask(self):
        _,kw=self.ready();signature(kw)
        for changes in ({'input_ids':torch.ones(3,1,dtype=torch.long)},
                        {'position_ids':torch.ones(6,1,dtype=torch.float32)},
                        {'attention_mask':None},{'logits_to_keep':0}):
            with self.assertRaises(ValueError):signature({**kw,**changes})

    def test_output_fresh_containers_preserve_storage_contract(self):
        arena,kw=self.ready();buffers=WaveBuffers(kw,arena);cache=buffers.kwargs()['past_key_values']
        x=torch.zeros(6,2,1,4);cache.update(x,x+1,0)
        out=WaveOutput(SimpleNamespace(logits=torch.zeros(6,1,16),past_key_values=cache))
        arena.check_output(out)
        a=out.fresh();b=out.fresh();self.assertIsNot(a.past_key_values,b.past_key_values)
        a.past_key_values.update(x,x,0)
        self.assertEqual(b.past_key_values.get_seq_length(),5)
        self.assertEqual(out.cache_length(),5)
        self.assertIs(a.logits,b.logits)  # HF must clone graph logits; benchmark audits this.

    def test_capture_rejects_cpu_and_reorder_context_restores(self):
        arena,kw=self.ready();pool=WaveReplayPool(lambda **k:None,arena=arena)
        with self.assertRaisesRegex(ValueError,'CUDA'):pool.prepare(kw)
        model=SimpleNamespace();direct=WaveReorder(arena)
        with self.assertRaisesRegex(RuntimeError,'sentinel'):
            with direct.install(model):
                self.assertEqual(model._reorder_cache,direct.reorder)
                with self.assertRaises(ValueError):
                    with direct.install(model):pass
                raise RuntimeError('sentinel')
        self.assertFalse(hasattr(model,'_reorder_cache'))
        with self.assertRaisesRegex(ValueError,'layouts differ'):WavePrefill(model,50,arena)


if __name__=='__main__':unittest.main()
