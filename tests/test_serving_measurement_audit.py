import copy
import unittest
from rapidtinyrec import serving_measurement_audit as a
from test_serving_setup_audit import raw


def fixture():
    users=[dict(cohort_index=i,prompt_tokens=76 if i<2 else 121) for i in range(4)]
    phases=[]
    for index,lanes in enumerate((1,2,2,1)):
        phase=dict(phase=index,lane_count=lanes,concurrency=lanes,primary=[],secondary=[])
        for kind in ('primary','secondary'):
            if kind=='secondary' and lanes==1:continue
            for local,order in enumerate(a.ORDERS):
                arrival=100000+(local+(3 if kind=='secondary' else 0))*10000
                concurrency=lanes if kind=='primary' else 1;rows=[]
                for pos,u in enumerate(order):
                    started=arrival+10+(pos//concurrency)*1000
                    submitted=arrival+pos+1 if concurrency==2 else started-10
                    rows.append(dict(request_id=str(pos),lane=pos%lanes,arrival_ns=arrival,
                        submitted_ns=submitted,started_ns=started,completed_ns=started+990,error=None,
                        result=dict(users[u],generation_started_ns=started+10,generation_finished_ns=started+810,generation_elapsed_s=800/1e9,raw=raw())))
                phase[kind].append(dict(repeat=(index//2)*3+local,records=rows))
        phases.append(phase)
    return phases,users,{i:raw() for i in range(4)}


class MeasurementTests(unittest.TestCase):
    def test_complete_schedule_counts_metrics_and_no_gpu_claim(self):
        p,u,h=fixture();r=a.audit_measurements(p,u,h)
        self.assertTrue(r['numerical_passed']);self.assertTrue(r['performance_passed'])
        self.assertFalse(r['advancement_ready']);self.assertFalse(r['gpu_overlap_proven'])
        self.assertEqual(r['comparisons']['paired']['requests'],24)
        self.assertEqual(r['comparisons']['historical']['requests'],72)
        self.assertEqual(r['secondary_diagnostic']['requests'],24)
        self.assertEqual(r['primary']['1']['overall']['requests_per_second'],1e6)
        self.assertEqual(r['primary']['2']['overall']['requests_per_second'],2e6)

    def test_generation_outside_response_and_clock_lies_rejected(self):
        for kind in ('finish','elapsed','bool','arrival','serial'):
            p,u,h=fixture();row=p[0]['primary'][0]['records'][1]
            if kind=='finish':row['result']['generation_finished_ns']=row['completed_ns']+1
            if kind=='elapsed':row['result']['generation_elapsed_s']=1.
            if kind=='bool':row['started_ns']=True
            if kind=='arrival':row['arrival_ns']+=1
            if kind=='serial':row['submitted_ns']-=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):a.audit_measurements(p,u,h)

    def test_lane_overlap_and_schedule_corruption_rejected(self):
        for kind in ('lane','user','repeat','secondary','burst','failure'):
            p,u,h=fixture();rows=p[1]['primary'][0]['records']
            if kind=='lane':rows[2]['lane']=1
            if kind=='user':rows[0]['result']['cohort_index']=2
            if kind=='repeat':p[1]['primary'][0]['repeat']=2
            if kind=='secondary':p[1]['secondary'].pop()
            if kind=='burst':p[1]['primary'][1]=copy.deepcopy(p[1]['primary'][0]);p[1]['primary'][1]['repeat']=1
            if kind=='failure':rows[0]['error']='failed'
            with self.subTest(kind=kind),self.assertRaises(ValueError):a.audit_measurements(p,u,h)

    def test_consistent_numeric_drift_remains_failed_not_exception(self):
        p,u,h=fixture()
        for phase in p:
            for kind in ('primary','secondary'):
                for burst in phase[kind]:
                    for row in burst['records']:
                        row['result']['raw']['token_ids'][0][0]=1000
        result=a.audit_measurements(p,u,h)
        self.assertFalse(result['numerical_passed']);self.assertTrue(result['outputs_valid'])
        self.assertTrue(result['performance_passed'])

    def test_secondary_ancestry_drift_cannot_be_hidden(self):
        p,u,h=fixture();p[1]['secondary'][0]['records'][0]['result']['raw']['beam_indices'][0][0]=1
        result=a.audit_measurements(p,u,h)
        self.assertFalse(result['numerical_passed']);self.assertFalse(result['ancestry_stable'])

    def test_slow_candidate_preserved_in_failed_performance_report(self):
        p,u,h=fixture()
        for phase in (p[1],p[2]):
            for burst in phase['primary']:
                start=burst['records'][0]['arrival_ns']
                for row in burst['records']:
                    for key in ('submitted_ns','started_ns','completed_ns'):row[key]=start+3*(row[key]-start)
                    for key in ('generation_started_ns','generation_finished_ns'):row['result'][key]=start+3*(row['result'][key]-start)
                    row['result']['generation_elapsed_s']*=3
        result=a.audit_measurements(p,u,h)
        self.assertTrue(result['numerical_passed']);self.assertFalse(result['performance_passed'])
        self.assertFalse(result['gates']['overall_throughput'])
