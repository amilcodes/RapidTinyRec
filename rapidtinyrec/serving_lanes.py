"""Bounded two-lane burst admission; numerical work belongs to owned runtimes.

Each lane initializes, freezes, runs and closes on one persistent host thread.
Initialization/freeze/close are serialized across lanes. A runtime must own its
model, CUDA stream, arena and graph outputs; run() must finish its stream and
materialize independent CPU outputs before returning. This module does not
batch model rows, provide HTTP service, or assert CUDA numerical equivalence.
"""
from concurrent.futures import Future
from dataclasses import dataclass
import queue
import statistics
import threading
import time


@dataclass(frozen=True)
class Request:
    request_id: str
    payload: object


class BurstExecutionError(RuntimeError):
    def __init__(self, records):
        super().__init__('Burst failed; all submitted work drained and records preserved')
        self.records = records


class _Lane:
    def __init__(self, index, initialize):
        self.index = index
        self.commands = queue.Queue()
        self.ready = Future()
        self.thread = threading.Thread(target=self._loop, args=(initialize,),
                                       name=f'serving-lane-{index}', daemon=False)
        self.thread.start()
        try:
            self.ready.result()
        except BaseException:
            self.thread.join()
            raise

    def _loop(self, initialize):
        try:
            state = initialize(self.index)
        except BaseException as exc:
            self.ready.set_exception(exc)
            return
        self.ready.set_result(None)
        while True:
            item = self.commands.get()
            if item is None:
                return
            future, action = item
            try:
                future.set_result(action(state))
            except BaseException as exc:
                future.set_exception(exc)

    def call(self, action):
        future = Future()
        self.commands.put((future, action))
        return future

    def stop(self):
        self.commands.put(None)
        self.thread.join()


class LaneServer:
    """One controller; one or two private runtimes; no lane is reentered.

    One resident lane is the practical serial control. Two resident lanes allow
    a secondary serial/concurrent comparison with identical round-robin mapping.
    No timeout cancels active GPU work or reuses its storage prematurely.
    """
    def __init__(self, initialize, run, freeze=lambda state: None,
                 close=lambda state: None, *, lane_count=2):
        if type(lane_count) is not int or lane_count not in (1, 2):
            raise ValueError("Only one or two resident lanes are supported")
        self.controller = threading.get_ident()
        self.run = run
        self.freeze_runtime = freeze
        self.close_runtime = close
        self.lanes = []
        self.closed = False
        self.frozen = False
        self.failed = False
        try:
            for index in range(lane_count):
                self.lanes.append(_Lane(index, initialize))
        except BaseException:
            self.close()
            raise

    def _owner(self):
        if threading.get_ident() != self.controller:
            raise RuntimeError('Only the owning controller may admit work')
        if self.closed:
            raise RuntimeError('Server is closed')
        if self.failed:
            raise RuntimeError('Server failed; close it without further admission')

    def freeze(self):
        self._owner()
        if self.frozen:
            raise RuntimeError('Already frozen')
        try:
            for lane in self.lanes:
                lane.call(self.freeze_runtime).result()
        except BaseException:
            self.failed = True
            raise
        self.frozen = True

    def run_burst(self, requests, concurrency):
        self._owner()
        if not self.frozen:
            raise RuntimeError('Freeze all runtimes before admission')
        if type(concurrency) is not int or concurrency not in (1, 2) or concurrency > len(self.lanes):
            raise ValueError('Concurrency must be one or two and fit resident lanes')
        requests = tuple(requests)
        if not requests or len({r.request_id for r in requests}) != len(requests):
            raise ValueError('A nonempty burst needs unique request IDs')
        arrival = time.perf_counter_ns()
        records = []
        pending = []
        aborted = threading.Event()

        def submit(index, request):
            lane = self.lanes[index % len(self.lanes)]
            submitted = time.perf_counter_ns()
            def execute(state):
                started = time.perf_counter_ns()
                result = None
                error = None
                try:
                    if aborted.is_set():
                        error = "Skipped after another request failed; runtime not reused"
                    else:
                        result = self.run(state, request.payload)
                except BaseException as exc:
                    error = repr(exc)
                    aborted.set()
                completed = time.perf_counter_ns()
                return dict(request_id=request.request_id, lane=lane.index,
                            arrival_ns=arrival, submitted_ns=submitted,
                            started_ns=started, completed_ns=completed,
                            result=result, error=error)
            return lane.call(execute)

        # All requests arrive at once. This is a finite burst, not open-loop
        # saturation throughput. Completion includes the runtime's extraction.
        for index, request in enumerate(requests):
            future = submit(index, request)
            if concurrency == 1:
                record = future.result()
                records.append(record)
                if record['error'] is not None:
                    break
            else:
                pending.append(future)
        records.extend(future.result() for future in pending)
        if any(row['error'] is not None for row in records):
            self.failed = True
            raise BurstExecutionError(records)
        return records

    def close(self):
        if threading.get_ident() != self.controller:
            raise RuntimeError('Only the owning controller may close lanes')
        if self.closed:
            return
        errors = []
        for lane in self.lanes:
            try:
                lane.call(self.close_runtime).result()
            except BaseException as exc:
                errors.append(exc)
            finally:
                lane.stop()
        self.closed = True
        if errors:
            raise errors[0]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def summarize_burst(records):
    if not records or len({r['request_id'] for r in records}) != len(records):
        raise ValueError('Empty or duplicate request records')
    arrivals = {r['arrival_ns'] for r in records}
    if len(arrivals) != 1 or any(r['error'] is not None for r in records):
        raise ValueError('Invalid arrival clock or failed burst')
    for row in records:
        if not (row['arrival_ns'] <= row['submitted_ns'] <= row['started_ns'] < row['completed_ns']):
            raise ValueError('Invalid monotonic timestamps')
    elapsed = (max(r['completed_ns'] for r in records) - next(iter(arrivals))) / 1e9
    def stats(values):
        return dict(median_ms=statistics.median(values),
                    p95_ms=statistics.quantiles(values,n=100,method='inclusive')[94] if len(values)>1 else values[0])
    return dict(requests=len(records),burst_elapsed_s=elapsed,
                burst_requests_per_second=len(records)/elapsed,
                response=stats([(r['completed_ns']-r['arrival_ns'])/1e6 for r in records]),
                queue=stats([(r['started_ns']-r['arrival_ns'])/1e6 for r in records]),
                service=stats([(r['completed_ns']-r['started_ns'])/1e6 for r in records]),
                boundary='Finite simultaneous-arrival burst; response includes queue and runtime output materialization. Not HTTP/open-loop saturation throughput.')
