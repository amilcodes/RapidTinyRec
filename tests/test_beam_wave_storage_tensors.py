"""Tiny CPU check connecting real tensor inventories to independent validation."""
import unittest
from types import SimpleNamespace
import torch
from rapidtinyrec.beam_wave_arena import WaveArena,borrowed_wave
from rapidtinyrec.beam_wave_replay import WaveBuffers,WaveOutput
from rapidtinyrec.beam_wave_storage import collect_inventory,validate_inventory
from rapidtinyrec.decode_replay import cache_from_pairs


class WaveTensorInventoryTests(unittest.TestCase):
    def test_actual_arena_and_graph_output_extents(self):
        arena=WaveArena(121);entries={}
        for length in (76,121):
            x=torch.zeros(2,2,length,4);arena.start_wave(cache_from_pairs(((x,x+1),)))
            for step in range(4):
                prefixes,suffixes=arena.views(length,step)
                for pair in suffixes:
                    for tensor in pair:tensor.zero_()
                kw=dict(input_ids=torch.zeros(100,1,dtype=torch.long),position_ids=torch.full((100,1),length+step),
                    cache_position=torch.tensor([length+step]),past_key_values=borrowed_wave(prefixes,suffixes,arena.layout),
                    attention_mask={'full_attention':None},use_cache=True,return_dict=True,logits_to_keep=1)
                buffers=WaveBuffers(kw,arena);cache=buffers.kwargs()['past_key_values'];next_k=torch.ones(100,2,1,4)
                cache.update(next_k,next_k+1,0)
                output=WaveOutput(SimpleNamespace(logits=torch.zeros(100,1,7),past_key_values=cache))
                entries[(length,step)]=(buffers,None,output)
        engine=SimpleNamespace(variant='beam_wave_tf32',arena=arena,pool=SimpleNamespace(frozen=True,entries=entries))
        model=torch.nn.Linear(4,4)
        record=collect_inventory(engine,model,'before-measurement')
        result=validate_inventory(record,layers=1,kv_heads=2,head_dim=4,vocab_size=7,device='cpu')
        self.assertTrue(result['supplied_storage_ownership_verified'])
        entries[(76,0)][2].logits=entries[(121,0)][2].logits
        with self.assertRaisesRegex(ValueError,'alias'):
            validate_inventory(collect_inventory(engine,model,'after-measurement'),layers=1,kv_heads=2,head_dim=4,vocab_size=7,device='cpu')


if __name__=='__main__':unittest.main()
