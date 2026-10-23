"""The full experiment must preserve the cohort and warm every capture shape."""
import collections
import itertools
import unittest

from rapidtinyrec.pilot_decode_replay import execution_order, select_users


class ProtocolTests(unittest.TestCase):
    def test_full_cohort_selection_and_execution_balance(self):
        users=[{'cohort_index':i,'prompt_tokens':76+(i%10)*5} for i in range(500)]
        selected,warm=select_users(users,True)
        self.assertIs(selected,users)
        self.assertEqual([u['cohort_index'] for u in warm],list(range(10)))
        variants=('compiled_gqa','replay','direct_replay')
        orders=[execution_order(variants,0,i,True) for i in range(500)]
        self.assertEqual(set(orders),set(itertools.permutations(variants)))
        self.assertEqual(sorted(collections.Counter(orders).values()),[83,83,83,83,84,84])
        for variant in variants:
            counts=collections.Counter(order.index(variant) for order in orders)
            self.assertLessEqual(max(counts.values())-min(counts.values()),2)
        self.assertEqual(len({u['prompt_tokens']+step for u in warm for step in range(4)}),40)

    def test_pilot_selection_and_all_six_orders(self):
        users=[{'cohort_index':i,'prompt_tokens':76+(i%10)*5} for i in range(500)]
        selected,warm=select_users(users,False)
        self.assertEqual([u['cohort_index'] for u in selected],[0,10,9,19])
        self.assertEqual(selected,warm)
        variants=('compiled_gqa','replay','direct_replay')
        self.assertEqual({execution_order(variants,r,0,False) for r in range(6)},set(itertools.permutations(variants)))


if __name__=='__main__':
    unittest.main()
