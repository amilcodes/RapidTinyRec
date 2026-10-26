import gc
from types import SimpleNamespace as NS
import unittest
import weakref

from rapidtinyrec.serving_inventory import enumerate_folded_lane
from rapidtinyrec.serving_storage import describe_tensors,audit_lane_storage

class Tensor:
    device='cpu'
    def __init__(self,address):self.address=address
    def untyped_storage(self):return self
    def data_ptr(self):return self.address
    def nbytes(self):return 16
    def numel(self):return 4
    def element_size(self):return 4

class Model:
    def __init__(self,tensor):self.tensor=tensor
    def named_parameters(self,remove_duplicate):return [('weight',self.tensor)]
    def named_buffers(self,remove_duplicate):return []

def fixture(start=1000,make=None):
    counter=start
    def tensor():
        nonlocal counter
        counter+=32
        return Tensor(counter) if make is None else make()
    model=Model(tensor())
    tables=NS(sizes_python=[1,1,1],offsets=tensor(),sizes=tensor(),root_prefix=tensor(),first_tokens=tensor(),
        eos=tensor(),newline=tensor(),valid_sids=tensor(),strides={i:tensor() for i in (1,2,3)},
        tables={i:tensor() for i in (1,2)},metadata={})
    prefixes=((tensor(),tensor()),);suffixes=((tensor(),tensor()),)
    arena=NS(max_prefix_tokens=121,max_suffix_tokens=3,storage=prefixes+suffixes,layout=None,
        prefix_views={76:prefixes},suffix_views={1:suffixes},requests_loaded=1,prefix_bytes_copied=16,
        beams=50,prefix_storage=prefixes,suffix_storage=suffixes)
    buffers=NS(signature=(),tensors={n:tensor() for n in ('input_ids','position_ids','cache_position')},
        prefixes=prefixes,pairs=suffixes,fixed={'use_cache':True,'return_dict':True,'logits_to_keep':1,'inputs_embeds':None})
    output=NS(logits=tensor(),prefixes=prefixes,pairs=((tensor(),tensor()),))
    replay=NS(forward=None,arena=arena,max_graphs=8,entries={'shape':(buffers,object(),output)},
        events=[],frozen=True,replays=0)
    return model,tables,arena,replay,{'user':{'input_ids':tensor()}},{'first':{'sequences':tensor(),'score0':tensor()}}

class FoldedInventoryTests(unittest.TestCase):
    def test_alias_in_any_explicit_owner_is_detected(self):
        for owner in ('model','constraint','prompt','retained','graph_output','arena'):
            a=fixture();b=list(fixture(10000));shared=a[0].tensor
            if owner=='model':b[0].tensor=shared
            if owner=='constraint':b[1].eos=shared
            if owner=='prompt':b[4]['user']['input_ids']=shared
            if owner=='retained':b[5]['first']['score0']=shared
            if owner=='graph_output':b[3].entries['shape'][2].logits=shared
            if owner=='arena':b[2].prefix_views[100]=((shared,shared),)
            left=enumerate_folded_lane(*a);right=enumerate_folded_lane(*b)
            with self.subTest(owner=owner),self.assertRaisesRegex(ValueError,'Cross-lane'):
                audit_lane_storage({0:describe_tensors(left),1:describe_tensors(right)})
    def test_unknown_owner_fields_fail_instead_of_silent_omission(self):
        for owner in ('constraints','arena','replay','inputs','outputs'):
            values=fixture();objects={'constraints':values[1],'arena':values[2],'replay':values[3],
                'inputs':values[3].entries['shape'][0],'outputs':values[3].entries['shape'][2]}
            objects[owner].new_tensor=Tensor(90000)
            with self.subTest(owner=owner),self.assertRaisesRegex(ValueError,'schema changed'):
                enumerate_folded_lane(*values)
    def test_unfrozen_wrong_arena_and_hidden_fixed_tensor_fail(self):
        for change in ('unfrozen','different_arena','fixed_tensor','truncated_aggregate'):
            values=fixture();replay=values[3]
            if change=='unfrozen':replay.frozen=False
            if change=='different_arena':replay.arena=object()
            if change=='fixed_tensor':replay.entries['shape'][0].fixed['inputs_embeds']=Tensor(90000)
            if change=='truncated_aggregate':values[2].storage=(values[2].storage[0][:1],)+values[2].storage[1:]
            with self.subTest(change=change),self.assertRaises(ValueError):enumerate_folded_lane(*values)
    def test_returned_inventory_retains_tensors_through_comparison(self):
        values=fixture();weak=weakref.ref(values[0].tensor);rows=enumerate_folded_lane(*values)
        del values;gc.collect();self.assertIsNotNone(weak())
        self.assertTrue(audit_lane_storage({0:describe_tensors(rows)})['enumerated_storage_disjoint'])
        del rows;gc.collect();self.assertIsNone(weak())
    def test_duplicate_qualified_names_and_missing_prompts_fail(self):
        values=list(fixture());values[4]={}
        with self.assertRaises(ValueError):enumerate_folded_lane(*values)
        values=list(fixture());values[4]={'a.b':{'c':Tensor(80000)},'a':{'b.c':Tensor(90000)}}
        with self.assertRaisesRegex(ValueError,'unique tensor names'):enumerate_folded_lane(*values)
    def test_real_torch_model_buffers_and_tied_names_are_inventoried(self):
        try:import torch
        except ImportError:self.skipTest('Real Torch check runs in decoding CI')
        values=list(fixture(make=lambda:torch.zeros(4)))
        model=torch.nn.Module();model.register_parameter('weight',torch.nn.Parameter(torch.ones(4)))
        model.register_parameter('tied',model.weight);model.register_buffer('buffer',torch.ones(2));values[0]=model
        rows=enumerate_folded_lane(*values);names={n for n,r,t in rows}
        self.assertTrue({'model.parameter.weight','model.parameter.tied','model.buffer.buffer'}<=names)
        self.assertTrue(audit_lane_storage({0:describe_tensors(rows)})['enumerated_storage_disjoint'])

if __name__=='__main__':unittest.main()
