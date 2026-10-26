import copy
import unittest
from rapidtinyrec.quality_frontier import evaluate


def fixture():
    catalog=[[i,1000+i,2000+i] for i in range(60)]
    users=[dict(cohort_index=i,prompt_tokens=76 if i==0 else 121,target_token_ids=catalog[i]) for i in range(2)]
    rows=[]
    for i in range(2):
        tokens=[s+[198,99] for s in catalog[:50]]
        for v in ('control','candidate'):
            rows.append(dict(cohort_index=i,variant=v,repeat=0,token_ids=copy.deepcopy(tokens)))
    return users,rows,catalog


def run(users,rows,catalog):return evaluate(users,rows,catalog,99,bootstrap_samples=64,bootstrap_seed=7)


class QualityFrontierTests(unittest.TestCase):
    def test_known_ranks_metrics_and_zero_paired_intervals(self):
        u,r,c=fixture();q=run(u,r,c)
        self.assertEqual(q['overall']['independent_users'],2)
        self.assertEqual(q['overall']['metrics']['control']['1']['hit_rate'],.5)
        self.assertEqual(q['overall']['metrics']['control']['5']['mrr'],.75)
        self.assertAlmostEqual(q['overall']['metrics']['control']['5']['ndcg'],(1+1/__import__('math').log2(3))/2)
        self.assertEqual(q['overall']['paired_user_bootstrap_95pct']['10']['ndcg'],{'lower':0.,'upper':0.})
        self.assertFalse(q['noninferiority_established']);self.assertFalse(q['promotion'])

    def test_repeats_never_inflate_users_and_instability_is_preserved(self):
        u,r,c=fixture();more=copy.deepcopy(r)
        for row in more:row['repeat']=1
        q=run(u,r+more,c);self.assertEqual(q['overall']['independent_users'],2)
        self.assertEqual(q['overall']['metrics'],run(u,r,c)['overall']['metrics'])
        more[1]['token_ids'].reverse()
        q=run(u,r+more,c);self.assertFalse(q['token_repeat_stability'])
        self.assertFalse(q['per_user'][0]['variants']['candidate']['stable_token_ranks_across_repeats'])

    def test_invalid_and_duplicates_keep_original_rank_cost(self):
        u,r,c=fixture();r[1]['token_ids'][0][-1]=98
        r[1]['token_ids'][1]=copy.deepcopy(r[1]['token_ids'][2])
        q=run(u,r,c)
        self.assertIsNone(q['per_user'][0]['variants']['candidate']['target_rank'])
        self.assertEqual(q['output_validity']['candidate']['invalid_outputs'],1)
        self.assertEqual(q['output_validity']['candidate']['duplicate_valid_sids'],1)
        self.assertEqual(q['overall']['paired_mean_changes']['10']['hit_rate'],-.5)

    def test_unchanged_aggregate_does_not_hide_user_regressions(self):
        u,r,c=fixture()
        # Control user0 hits@1 and user1 misses; candidate swaps these outcomes.
        r[1]['token_ids'][0],r[1]['token_ids'][1]=r[1]['token_ids'][1],r[1]['token_ids'][0]
        r[3]['token_ids'][0],r[3]['token_ids'][1]=r[3]['token_ids'][1],r[3]['token_ids'][0]
        q=run(u,r,c);x=q['overall']
        self.assertEqual(x['paired_mean_changes']['1']['hit_rate'],0.)
        self.assertEqual(x['changed_users']['1']['hit_rate'],{'improved':1,'regressed':1,'unchanged':0})
        self.assertLess(x['paired_user_bootstrap_95pct']['1']['hit_rate']['lower'],0)
        self.assertGreater(x['paired_user_bootstrap_95pct']['1']['hit_rate']['upper'],0)

    def test_pairing_missing_and_duplicate_data_rejected(self):
        u,r,c=fixture()
        for rows in [r[:-1],r+[copy.deepcopy(r[0])],r[:2]]:
            with self.assertRaises(ValueError):run(u,rows,c)
        with self.assertRaises(ValueError):run(u+[u[0]],r,c)
        r[1]['repeat']=2
        with self.assertRaises(ValueError):run(u,r,c)

    def test_invalid_schema_and_target_rejected(self):
        u,r,c=fixture();u[0]['target_token_ids']=[9999,9999,9999]
        with self.assertRaises(ValueError):run(u,r,c)
        u,r,c=fixture();r[0]['token_ids'].pop()
        with self.assertRaises(ValueError):run(u,r,c)
        u,r,c=fixture();r[0]['token_ids'][0][0]=True
        with self.assertRaises(ValueError):run(u,r,c)

    def test_input_order_does_not_change_seeded_paired_results(self):
        u,r,c=fixture();r[1]['token_ids'].reverse()
        self.assertEqual(run(u,r,c),run(list(reversed(u)),list(reversed(r)),c))

    def test_malformed_termination_counts_as_invalid_not_silent_removal(self):
        u,r,c=fixture();r[1]['token_ids'][0]=r[1]['token_ids'][0][:3]
        q=run(u,r,c)
        self.assertEqual(q['per_user'][0]['variants']['candidate']['invalid_outputs'],1)
        self.assertIsNone(q['per_user'][0]['variants']['candidate']['target_rank'])

if __name__=='__main__':unittest.main()
