import copy
import unittest
from rapidtinyrec.serving_overlap_audit import audit_overlap,union,intersection


def fixture(serial=False):
    events=[];expected={}
    for pos in range(4):
        lane=pos%2;start=(pos//2)*100+(40*lane if serial else 0);name=f'serving_profile/lane_{lane}/user_{pos}'
        expected[name]=lane
        events.append(dict(name=name,cat='user_annotation',ph='X',pid=10,tid=20+lane,ts=start,dur=39))
        for step in range(4):
            correlation=pos*4+step+1
            events.append(dict(name='cudaGraphLaunch',cat='cuda_runtime',ph='X',pid=10,tid=20+lane,ts=start+step*8+1,dur=1,args={'correlation':correlation}))
            events.append(dict(name='decode',cat='kernel',ph='X',pid=0,tid=7+lane,ts=start+step*8+2,dur=4,args={'correlation':correlation,'device':0,'context':1,'stream':7+lane,'graph id':step+1,'graph node id':10+step}))
    return dict(traceEvents=events),expected


class OverlapTests(unittest.TestCase):
    def test_positive_overlap_is_union_intersection(self):
        t,e=fixture();r=audit_overlap(t,e)
        self.assertTrue(r['gpu_overlap_proven']);self.assertEqual(r['graph_kernel_overlap_us'],32)
        self.assertEqual(r['graph_launches'],16)
        self.assertEqual(union([(0,5),(2,7),(9,10)]),[[0,7],[9,10]])
        self.assertEqual(intersection(union([(0,5),(2,7)]),[[1,6]]),5)

    def test_host_overlap_and_unrelated_kernel_cannot_pass(self):
        t,e=fixture(True)
        for event in t['traceEvents']:
            if event['cat']=='user_annotation':event['dur']=80
        t['traceEvents'].append(dict(name='prefill',cat='kernel',ph='X',ts=0,dur=300,args={'correlation':999}))
        r=audit_overlap(t,e);self.assertFalse(r['gpu_overlap_proven']);self.assertEqual(r['graph_kernel_overlap_us'],0)

    def test_missing_duplicate_ambiguous_and_cross_thread_launches_rejected(self):
        for mode in ('missing','duplicate','thread','extra','kernel'):
            t,e=fixture()
            if mode=='missing':t['traceEvents'].pop(0)
            if mode=='duplicate':t['traceEvents'].append(copy.deepcopy(t['traceEvents'][1]))
            if mode=='thread':t['traceEvents'][1]['tid']=123
            if mode=='extra':t['traceEvents'][0]['name']='serving_profile/unknown'
            if mode=='kernel':t['traceEvents'].pop(2)
            with self.subTest(mode=mode),self.assertRaises(ValueError):audit_overlap(t,e)

    def test_stream_device_graph_and_interval_corruption_rejected(self):
        for mode in ('stream','device','node','outside','duplicate','nan'):
            t,e=fixture();k=t['traceEvents'][2]
            if mode=='stream':k['args']['stream']=99
            if mode=='device':k['args']['device']=1
            if mode=='node':k['args']['graph node id']=0
            if mode=='outside':k['dur']=1000
            if mode=='duplicate':t['traceEvents'].append(copy.deepcopy(k))
            if mode=='nan':k['ts']=float('nan')
            with self.subTest(mode=mode),self.assertRaises(ValueError):audit_overlap(t,e)

    def test_shared_stream_cannot_prove_independent_execution(self):
        t,e=fixture()
        for v in t['traceEvents']:
            if v['cat']=='kernel':v['args']['stream']=7
        with self.assertRaisesRegex(ValueError,'Disjoint'):audit_overlap(t,e)
