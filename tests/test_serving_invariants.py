import copy
from types import SimpleNamespace as NS
import unittest
from rapidtinyrec import serving_invariants as a


def fixture():
    byte=a.BYTES_PER_TOKEN;prefix=a.PREFIX_BYTES_PER_TOKEN;metadata=a.METADATA_BYTES
    events=[dict(cache_tokens=n+j,batch=50,input_buffer_bytes=prefix*n+byte*j+metadata,capture_and_warmup_s=1.) for n in (76,121) for j in range(4)]
    replay=dict(graphs=8,frozen=True,shared_input_arena=True,replays=100,events=events,capture_and_warmup_s=8.,input_buffer_bytes=sum(e['input_buffer_bytes'] for e in events),unique_input_storage_bytes=prefix*121+byte*3+8*metadata,output_suffix_bytes=byte*20,physical_prefix=dict(requests_loaded=10,prefix_bytes_copied=1000,prefix_capacity_bytes=prefix*121,suffix_capacity_bytes=byte*3,max_prefix_tokens=121,max_suffix_tokens=3,physical_prefix_batch=1,suffix_batch=50))
    lane=dict(identity=dict(owner_thread_id=1,device='cuda:0',stream_id=3,failed=False),compiler=dict(phase='measurement',events=[{'success':True}],frozen_event_count=1),replay=replay,reorder=dict(direct_calls=100,fallback_calls=25,suffix_bytes_written=1000,prefix_bytes_reordered=0),global_compiler=dict(error_on_recompile=True,counters={},generated_kernel_count=10),policy=dict(matmul_tf32=False,cudnn_tf32=False,intraop_threads=2))
    memory=dict(allocated_bytes=2**30,reserved_bytes=2**31,peak_allocated_bytes=2**30,peak_reserved_bytes=2**31)
    before=dict(schema='serving-runtime-snapshot-v1',started_ns=1,finished_ns=2,lanes=[lane],memory=memory)
    after=copy.deepcopy(before);after.update(started_ns=100,finished_ns=101)
    after['lanes'][0]['replay']['replays']+=4
    arena=after['lanes'][0]['replay']['physical_prefix'];arena['requests_loaded']+=1;arena['prefix_bytes_copied']+=prefix*76
    reorder=after['lanes'][0]['reorder'];reorder['direct_calls']+=4;reorder['fallback_calls']+=1;reorder['suffix_bytes_written']+=byte*6
    rows={0:[dict(lane=0,error=None,arrival_ns=3,started_ns=4,completed_ns=5,result=dict(prompt_tokens=76))]}
    return before,after,rows


class InvariantTests(unittest.TestCase):
    def test_one_and_two_lanes_reconcile_without_summing_device_memory(self):
        x,y,r=fixture();self.assertTrue(a.audit_interval(x,y,r)['passed'])
        for snapshot in (x,y):
            lane=copy.deepcopy(snapshot['lanes'][0]);lane['identity'].update(owner_thread_id=2,stream_id=4);snapshot['lanes'].append(lane)
        r[1]=copy.deepcopy(r[0]);r[1][0]['lane']=1
        result=a.audit_interval(x,y,r);self.assertTrue(result['passed']);self.assertEqual(result['memory_after']['allocated_bytes'],2**30)

    def test_counter_deltas_and_interval_escape_rejected(self):
        for kind in ('replays','staging','reorder','clock','lane'):
            x,y,r=fixture()
            if kind=='replays':y['lanes'][0]['replay']['replays']+=1
            if kind=='staging':y['lanes'][0]['replay']['physical_prefix']['prefix_bytes_copied']+=1
            if kind=='reorder':y['lanes'][0]['reorder']['suffix_bytes_written']+=1
            if kind=='clock':r[0][0]['completed_ns']=101
            if kind=='lane':r[0][0]['lane']=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):a.audit_interval(x,y,r)

    def test_capture_compiler_policy_and_capacity_mutation_rejected(self):
        for kind in ('capture','compiler','policy','prefix','global'):
            x,y,r=fixture();lane=y['lanes'][0]
            if kind=='capture':lane['replay']['events'][0]['allocated_delta_bytes']=2
            if kind=='compiler':lane['compiler']['events'][0]['new']=True
            if kind=='policy':lane['policy']['intraop_threads']=4
            if kind=='prefix':lane['replay']['physical_prefix']['physical_prefix_batch']=50
            if kind=='global':lane['global_compiler']['counters']={'graph_break':{'reason':1}}
            with self.subTest(kind=kind),self.assertRaises(ValueError):a.audit_interval(x,y,r)

    def test_allocator_impossibility_and_alias_owners_rejected(self):
        for kind in ('capacity','peak','shared'):
            x,y,r=fixture()
            if kind=='capacity':y['memory']['allocated_bytes']=1
            if kind=='peak':y['memory']['peak_reserved_bytes']=47*1024**3
            if kind=='shared':
                for snap in (x,y):snap['lanes'].append(copy.deepcopy(snap['lanes'][0]))
                r[1]=[]
            with self.subTest(kind=kind),self.assertRaises(ValueError):a.audit_interval(x,y,r)

    def test_snapshot_deepcopies_state_and_does_not_run_or_reset_allocator(self):
        x,_,_=fixture();state=x['lanes'][0];calls=[]
        backend=NS(_dynamo=NS(config=NS(error_on_recompile=True),utils=NS(counters={})),_inductor=NS(metrics=NS(generated_kernel_count=10)),backends=NS(cuda=NS(matmul=NS(allow_tf32=False)),cudnn=NS(allow_tf32=False)),get_num_threads=lambda:2)
        lane=NS(frozen=True,closed=False,retained={},context=NS(identity=lambda:state['identity'],call=lambda f:f()),compiler=NS(**state['compiler']),pool=NS(report=lambda:state['replay']),direct=NS(report=lambda:state['reorder']))
        for method,key in [('memory_allocated','allocated_bytes'),('memory_reserved','reserved_bytes'),('max_memory_allocated','peak_allocated_bytes'),('max_memory_reserved','peak_reserved_bytes')]:
            def read(device,key=key):calls.append((key,device));return x['memory'][key]
            if not hasattr(backend,'cuda'):backend.cuda=NS()
            setattr(backend.cuda,method,read)
        server=NS(_owner=lambda:None,frozen=True,visit=lambda f:[f(lane)])
        report=a.collect_snapshot(server,backend=backend)
        state['replay']['events'][0]['batch']=999
        self.assertEqual(report['lanes'][0]['replay']['events'][0]['batch'],50)
        self.assertEqual(len(calls),4);self.assertTrue(all(device==0 for _,device in calls))
