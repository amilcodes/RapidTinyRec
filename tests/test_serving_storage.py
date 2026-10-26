import unittest
from rapidtinyrec.serving_storage import describe_tensors,audit_lane_storage


def row(name,address,size,logical=None,device='cuda:0'):
    return dict(name=name,role='test',device=device,address=address,storage_bytes=size,
                logical_bytes=size if logical is None else logical)


class StorageRangeTests(unittest.TestCase):
    def test_aliases_count_logical_views_but_union_physical_capacity(self):
        rows=[row('base',100,100),row('slice',100,100,20),row('external_view',150,100)]
        result=audit_lane_storage({0:rows})['lanes']['0']
        self.assertEqual(result['logical_bytes'],220)
        self.assertEqual(result['unique_storage_bytes'],150)

    def test_overlapping_distinct_base_addresses_fail(self):
        with self.assertRaisesRegex(ValueError,'Cross-lane'):
            audit_lane_storage({0:[row('a',100,100)],1:[row('b',199,100)]})

    def test_adjacent_ranges_and_different_devices_are_disjoint(self):
        result=audit_lane_storage({0:[row('a',100,100)],1:[row('b',200,100),row('c',100,100,device='cuda:1')]})
        self.assertTrue(result['enumerated_storage_disjoint'])

    def test_bad_counts_null_pointer_and_inconsistent_capacity_fail(self):
        for rows in ([row('a',0,1)],[row('a',10,-1)],[row('a',10,1.5)],
                     [row('a',10,10),row('b',10,20)],[row('a',0,0,1)],[row('a',10,10,device='cuda')]):
            with self.subTest(rows=rows),self.assertRaises(ValueError):audit_lane_storage({0:rows})

    def test_missing_lanes_and_duplicate_names_fail(self):
        for inventories in ({},{False:[row('a',10,10)]},{1:[row('a',10,10)]},{0:[]},{0:[row('a',10,10),row('a',20,10)]}):
            with self.subTest(inventories=inventories),self.assertRaises(ValueError):audit_lane_storage(inventories)

    def test_empty_view_retains_backing_storage_and_expansion_is_allowed(self):
        result=audit_lane_storage({0:[row('empty_view',100,100,0),row('expand',100,100,1000),row('empty_allocation',0,0)]})
        self.assertEqual(result['lanes']['0']['unique_storage_bytes'],100)
        with self.assertRaisesRegex(ValueError,'Cross-lane'):
            audit_lane_storage({0:[row('empty_view',100,100,0)],1:[row('b',100,100)]})

    def test_real_tensor_views_cannot_cross_lane_even_with_disjoint_elements(self):
        try:import torch
        except ImportError:self.skipTest('Real tensor check runs in decoding CI')
        backing=torch.arange(20,dtype=torch.float32)
        left=describe_tensors([('left','prompt',backing[:10])])
        right=describe_tensors([('right','prompt',backing[10:])])
        self.assertEqual(left[0]['address'],right[0]['address'])
        with self.assertRaisesRegex(ValueError,'Cross-lane'):audit_lane_storage({0:left,1:right})
        independent=describe_tensors([('copy','prompt',backing[10:].clone())])
        self.assertTrue(audit_lane_storage({0:left,1:independent})['enumerated_storage_disjoint'])


if __name__=='__main__':unittest.main()
