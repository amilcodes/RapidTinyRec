import threading
import unittest
from rapidtinyrec.serving_lanes import LaneServer, Request, BurstExecutionError, summarize_burst


class ServingLaneTests(unittest.TestCase):
    def test_serial_setup_and_stable_thread_ownership_with_parallel_execution(self):
        events=[];barrier=threading.Barrier(2)
        def initialize(lane):
            events.append(('init',lane,threading.get_ident()))
            return dict(lane=lane,thread=threading.get_ident(),count=0)
        def run(state,payload):
            self.assertEqual(threading.get_ident(),state['thread'])
            barrier.wait(timeout=2)
            state['count']+=1
            return (state['lane'],state['count'],payload)
        def freeze(state):events.append(('freeze',state['lane'],threading.get_ident()))
        def close(state):events.append(('close',state['lane'],threading.get_ident()))
        with LaneServer(initialize,run,freeze,close) as server:
            server.freeze()
            a=server.run_burst([Request(str(i),i) for i in range(4)],2)
            b=server.run_burst([Request(str(i),i+10) for i in range(4)],2)
            self.assertEqual([r['lane'] for r in a],[0,1,0,1])
            self.assertEqual([r['result'][1] for r in a],[1,1,2,2])
            self.assertEqual([r['result'][1] for r in b],[3,3,4,4])
            self.assertEqual([r['result'][2] for r in a],list(range(4)))
        self.assertEqual([(x[0],x[1]) for x in events],[('init',0),('init',1),('freeze',0),('freeze',1),('close',0),('close',1)])
        for lane in range(2):self.assertEqual(len({e[2] for e in events if e[1]==lane}),1)
        self.assertNotEqual(events[0][2],events[1][2])

    def test_serial_admission_uses_both_resident_lanes_without_overlap(self):
        active=threading.Lock();order=[]
        def run(lane,value):
            self.assertTrue(active.acquire(blocking=False))
            try:order.append((lane,value));return value
            finally:active.release()
        with LaneServer(lambda i:i,run) as server:
            server.freeze();rows=server.run_burst([Request(str(i),i) for i in range(4)],1)
        self.assertEqual(order,[(0,0),(1,1),(0,2),(1,3)])
        self.assertTrue(all(a['completed_ns']<=b['started_ns'] for a,b in zip(rows,rows[1:])))

    def test_parallel_failure_drains_other_lane_and_blocks_further_admission(self):
        barrier=threading.Barrier(2);finished=[]
        def run(lane,value):
            barrier.wait(timeout=2)
            if lane==0:raise ValueError('saved failure')
            finished.append(value);return value
        with LaneServer(lambda i:i,run) as server:
            server.freeze()
            with self.assertRaises(BurstExecutionError) as raised:
                server.run_burst([Request('a',1),Request('b',2)],2)
            self.assertEqual(finished,[2]);self.assertEqual(len(raised.exception.records),2)
            self.assertIn('saved failure',raised.exception.records[0]['error'])
            with self.assertRaises(RuntimeError):server.run_burst([Request('c',3)],1)

    def test_single_resident_lane_is_the_practical_control(self):
        initialized=[]
        def init(i):initialized.append(i);return i
        with LaneServer(init,lambda lane,payload:(lane,payload),lane_count=1) as server:
            server.freeze()
            rows=server.run_burst([Request(str(i),i) for i in range(4)],1)
            self.assertEqual([r['lane'] for r in rows],[0]*4)
            with self.assertRaises(ValueError):server.run_burst([Request('a',0)],2)
        self.assertEqual(initialized,[0])
        with self.assertRaises(ValueError):LaneServer(init,lambda s,p:p,lane_count=3)

    def test_failed_lane_never_runs_queued_requests_again(self):
        calls=[]
        def run(lane,value):
            calls.append((lane,value))
            if lane==0:raise ValueError('lane failed')
            return value
        with LaneServer(lambda i:i,run) as server:
            server.freeze()
            with self.assertRaises(BurstExecutionError) as raised:
                server.run_burst([Request(str(i),i) for i in range(8)],2)
        self.assertEqual([v for lane,v in calls if lane==0],[0])
        self.assertEqual(len(raised.exception.records),8)
        self.assertTrue(all(r['error'] is not None for r in raised.exception.records[2::2]))

    def test_guards_and_idempotent_close(self):
        server=LaneServer(lambda i:i,lambda s,p:p)
        with self.assertRaises(RuntimeError):server.run_burst([Request('a',0)],1)
        server.freeze()
        with self.assertRaises(RuntimeError):server.freeze()
        for requests,concurrency in [([],1),([Request('a',0),Request('a',1)],2),([Request('a',0)],3)]:
            with self.assertRaises(ValueError):server.run_burst(requests,concurrency)
        errors=[]
        def other():
            try:server.run_burst([Request('a',0)],1)
            except RuntimeError as e:errors.append(str(e))
        t=threading.Thread(target=other);t.start();t.join();self.assertEqual(len(errors),1)
        server.close();server.close()
        with self.assertRaises(RuntimeError):server.run_burst([Request('a',0)],1)

    def test_failed_initialization_closes_previous_lane(self):
        closed=[]
        def init(i):
            if i==1:raise ValueError('init')
            return i
        with self.assertRaises(ValueError):LaneServer(init,lambda s,p:p,close=closed.append)
        self.assertEqual(closed,[0])

    def test_clock_metrics_include_queue_and_do_not_sum_parallel_service(self):
        rows=[dict(request_id='a',lane=0,arrival_ns=0,submitted_ns=10,started_ns=1000000,completed_ns=11000000,error=None),
              dict(request_id='b',lane=1,arrival_ns=0,submitted_ns=20,started_ns=2000000,completed_ns=12000000,error=None)]
        stats=summarize_burst(rows)
        self.assertEqual(stats['burst_elapsed_s'],.012)
        self.assertAlmostEqual(stats['burst_requests_per_second'],2/.012)
        self.assertEqual(stats['response']['median_ms'],11.5)
        self.assertEqual(stats['queue']['median_ms'],1.5)
        self.assertEqual(stats['service']['median_ms'],10)
        with self.assertRaises(ValueError):summarize_burst(rows+[rows[0]])
        rows[0]['completed_ns']=0
        with self.assertRaises(ValueError):summarize_burst(rows)

if __name__=='__main__':unittest.main()
