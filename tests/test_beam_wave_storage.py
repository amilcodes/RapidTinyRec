import copy
import math
import unittest
from rapidtinyrec.beam_wave_storage import validate_inventory,validate_stages


def inventory(wave=True):
    pointer=100
    def tensor(shape,dtype='torch.float32',owner=None,size=None):
        nonlocal pointer
        pointer+=1;bytes_=8 if dtype=='torch.int64' else 4
        strides=[];n=1
        for dim in reversed(shape):strides.insert(0,n);n*=dim
        return dict(shape=list(shape),stride=strides,dtype=dtype,device='cuda:0',storage_ptr=owner or pointer,
            storage_bytes=size or n*bytes_,byte_start=0,logical_bytes=n*bytes_,element_bytes=bytes_,contiguous=True)
    physical=2 if wave else 1;batch=100 if wave else 50
    prefix=[[tensor([physical*2*121*4]) for _ in range(2)]]
    suffix=[[tensor([batch*2*3*4]) for _ in range(2)]]
    model={'weight':tensor([4,4])};graphs=[]
    for length in (76,121):
        for step in range(4):
            p=[[tensor([physical,2,length,4],owner=r['storage_ptr'],size=r['storage_bytes']) for r in prefix[0]]]
            s=[[tensor([batch,2,step,4],owner=r['storage_ptr'],size=r['storage_bytes']) for r in suffix[0]]]
            graphs.append(dict(index=len(graphs),cache_tokens=length+step,prefixes=p,suffixes=s,output_prefixes=copy.deepcopy(p),
                output_suffixes=[[tensor([batch,2,step+1,4]) for _ in range(2)]],logits=tensor([batch,1,7]),
                tokens={k:tensor([1] if k=='cache_position' else [batch,1],dtype='torch.int64') for k in ('input_ids','position_ids','cache_position')}))
    return dict(schema='two-request-beam-wave-storage-v1',stage='before-measurement',variant='beam_wave_tf32' if wave else 'serial_pair_tf32',
        model_parameters=model,prefix_capacity=prefix,suffix_capacity=suffix,graphs=graphs)


LAYOUT=dict(layers=1,kv_heads=2,head_dim=4,vocab_size=7)


class WaveStorageTests(unittest.TestCase):
    def test_both_variants_and_empty_suffix_views(self):
        for wave in (True,False):
            result=validate_inventory(inventory(wave),**LAYOUT)
            self.assertTrue(result['supplied_storage_ownership_verified']);self.assertFalse(result['hidden_allocator_completeness'])

    def test_graph_output_cannot_alias_other_output_or_model(self):
        for ptr in (inventory()['model_parameters']['weight']['storage_ptr'],inventory()['graphs'][0]['logits']['storage_ptr']):
            r=inventory();r['graphs'][1]['logits']['storage_ptr']=ptr
            with self.assertRaisesRegex(ValueError,'alias'):validate_inventory(r,**LAYOUT)

    def test_prefix_assignment_and_shape_cannot_cross_requests(self):
        r=inventory();r['graphs'][0]['prefixes'][0][0]['storage_ptr']=r['prefix_capacity'][0][1]['storage_ptr']
        with self.assertRaisesRegex(ValueError,'owner'):validate_inventory(r,**LAYOUT)
        r=inventory();r['graphs'][0]['prefixes'][0][0]['shape'][0]=1
        with self.assertRaises(ValueError):validate_inventory(r,**LAYOUT)

    def test_token_alias_and_out_of_bounds_extent_rejected(self):
        r=inventory();g=r['graphs'][0];g['tokens']['position_ids']['storage_ptr']=g['tokens']['input_ids']['storage_ptr']
        with self.assertRaisesRegex(ValueError,'mutable storage'):validate_inventory(r,**LAYOUT)
        r=inventory();r['graphs'][0]['logits']['storage_bytes']=1
        with self.assertRaises(ValueError):validate_inventory(r,**LAYOUT)

    def test_missing_layer_or_graph_cannot_pass(self):
        r=inventory();r['graphs'].pop()
        with self.assertRaises(ValueError):validate_inventory(r,**LAYOUT)
        r=inventory();r['graphs'][0]['output_suffixes']=[]
        with self.assertRaises(ValueError):validate_inventory(r,**LAYOUT)

    def test_three_stages_preserve_exact_storage_identity(self):
        a=inventory();records=[dict(copy.deepcopy(a),stage=s) for s in ('before-measurement','after-measurement','after-profiles')]
        self.assertTrue(validate_stages(records,**LAYOUT)['stable_across_measurement_profiles'])
        records[2]['graphs'][0]['logits']['storage_ptr']=999999
        with self.assertRaisesRegex(ValueError,'changed'):validate_stages(records,**LAYOUT)


if __name__=='__main__':unittest.main()
