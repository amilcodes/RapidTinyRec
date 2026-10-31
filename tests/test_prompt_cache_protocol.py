import copy
from pathlib import Path
import tempfile
import unittest

from rapidtinyrec.prompt_cache_protocol import (PROTOCOL, PHASES, load_protocol,
    validate_protocol, phase_record, validate_request_inventory, audit_timeout)


class PromptCacheProtocolTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(__file__).resolve().parents[1]/PROTOCOL
        self.protocol = load_protocol(self.path)
        self.users = [dict(cohort_index=i,prompt_tokens=n,
            prompt_token_ids=self.protocol['prefix_token_ids']+[9]*(n-69))
            for i,n in zip([1,2,6,10],[76,76,121,121])]

    def rows(self, phase):
        return [dict(repeat=r,cohort_index=u['cohort_index'],prompt_tokens=u['prompt_tokens'],
                     variant=PHASES[phase]) for r in range(phase//2*3,phase//2*3+3) for u in self.users]

    def test_all_four_distinct_phases_and_request_inventories(self):
        for phase in range(4):
            record=phase_record(self.protocol,phase,phase>=2,self.users)
            self.assertEqual(record['schema'],'fixed-prompt-cache-phase-v1')
            self.assertEqual(record['prefix_tokens'],69 if phase in (1,2) else 0)
            validate_request_inventory(self.protocol,phase,self.rows(phase))

    def test_protocol_byte_change_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'p.json';p.write_bytes(self.path.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'hash'):load_protocol(p)

    def test_relabel_or_phase_order_cannot_pass(self):
        for key,value in [('experiment','bf16_vs_tf32_pilot'),('profile','precision-full500-quality'),
                          ('schema','serving-throughput-v1')]:
            protocol=copy.deepcopy(self.protocol);protocol[key]=value
            with self.assertRaises(ValueError):validate_protocol(protocol)
        protocol=copy.deepcopy(self.protocol);protocol['phases'][2]['variant']='uncached_tf32'
        with self.assertRaisesRegex(ValueError,'ABBA'):validate_protocol(protocol)

    def test_prefix_mismatch_and_cohort_selection_rejected(self):
        self.users[0]['prompt_token_ids'][1]+=1
        with self.assertRaises(ValueError):phase_record(self.protocol,0,False,self.users)
        with self.assertRaises(ValueError):phase_record(self.protocol,1,False,self.users[::-1])

    def test_reuse_and_bad_phase_rejected(self):
        for phase,reused in [(0,True),(2,False),(-1,False),(4,True),(True,False)]:
            with self.assertRaises(ValueError):phase_record(self.protocol,phase,reused,self.users)

    def test_missing_duplicate_wrong_half_or_label_rejected(self):
        rows=self.rows(0)
        for wrong in [rows[:-1],rows[:-1]+[rows[0]],self.rows(3)]:
            with self.assertRaises(ValueError):validate_request_inventory(self.protocol,0,wrong)
        rows[0]['variant']='folded_tf32'
        with self.assertRaises(ValueError):validate_request_inventory(self.protocol,0,rows)

    def test_audit_cannot_extend_work_deadline(self):
        self.assertEqual(audit_timeout(200),120)
        self.assertEqual(audit_timeout(17.25),17.25)
        for remaining in (0,-1):
            with self.assertRaises(TimeoutError):audit_timeout(remaining)
        for remaining in (float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):audit_timeout(remaining)


if __name__=='__main__':unittest.main()
