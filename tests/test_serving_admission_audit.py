import copy
import unittest
from rapidtinyrec.serving_admission_audit import audit_admission


def fixture(lanes=2, offset=0):
    owners={i:{'identity':{'owner_thread_id':10+i}} for i in range(lanes)}
    stages=[dict(allocation_started_ns=offset+30+i*50,warmup_finished_ns=offset+60+i*50) for i in range(lanes)]
    events=[]
    def add(op,lane,start,finish):
        events.append(dict(operation=op,lane=lane,thread_id=999 if lane is None else 10+lane,
                           started_ns=offset+start,finished_ns=offset+finish,success=True,error=None))
    add('patch_installed',None,21,22)
    for i in range(lanes):add('initialize',i,25+i*50,65+i*50)
    for i in range(lanes):add('freeze',i,200+i*2,201+i*2)
    for i in range(lanes):add('snapshot',i,210+i*2,211+i*2)
    add('ownership_audit',None,220,221);add('setup_audit',None,222,223)
    for i in range(lanes):add('release_retained',i,224+i*2,225+i*2)
    add('admitted',None,230,231)
    return events,owners,stages,offset+20,offset+1000


class AdmissionAuditTests(unittest.TestCase):
    def test_one_and_two_lane_setup_precedes_first_freeze(self):
        for lanes in (1,2):
            result=audit_admission(*fixture(lanes))
            self.assertTrue(result['passed']);self.assertEqual(result['lanes'],lanes)
            self.assertLess(result['all_setup_finished_ns'],result['first_freeze_started_ns'])

    def test_early_freeze_missing_duplicate_and_reordered_events_rejected(self):
        for mode in ('early','missing','duplicate','reorder'):
            args=list(fixture());events=args[0]
            if mode=='early':events[3]['started_ns']=100
            if mode=='missing':events.pop()
            if mode=='duplicate':events.append(copy.deepcopy(events[-1]))
            if mode=='reorder':events[2],events[3]=events[3],events[2]
            with self.subTest(mode=mode),self.assertRaises(ValueError):audit_admission(*args)

    def test_owner_controller_failed_stage_and_boolean_clock_rejected(self):
        for mode in ('owner','controller','failed','bool'):
            args=list(fixture());events=args[0]
            if mode=='owner':events[1]['thread_id']=11
            if mode=='controller':events[-1]['thread_id']=10
            if mode=='failed':events[3].update(success=False,error='freeze failed')
            if mode=='bool':events[1]['finished_ns']=True
            with self.subTest(mode=mode),self.assertRaises(ValueError):audit_admission(*args)

    def test_stage_escape_and_resource_or_measurement_boundary_rejected(self):
        for mode in ('warm','resource','phase'):
            args=list(fixture())
            if mode=='warm':args[2][1]['warmup_finished_ns']=120
            if mode=='resource':args[3]=22
            if mode=='phase':args[4]=231
            with self.subTest(mode=mode),self.assertRaises(ValueError):audit_admission(*args)
