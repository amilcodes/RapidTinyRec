"""Untimed frozen-runtime snapshots and CPU counter/capacity reconciliation.

Device allocator values describe the whole process/device, never one lane.
Ownership, numerical fidelity, actual overlap and source identity remain separate.
"""
import copy
import math
import time

BYTES_PER_TOKEN = 28*2*50*2*128*4
PREFIX_BYTES_PER_TOKEN = BYTES_PER_TOKEN//50
METADATA_BYTES = 101*8


def snapshot_lane(lane, backend):
    identity=lane.context.identity()
    def collect():
        if not lane.frozen or lane.closed or lane.retained:
            raise ValueError('Snapshots require frozen lanes with retained setup outputs released')
        return copy.deepcopy(dict(identity=identity,
            compiler=dict(phase=lane.compiler.phase,events=lane.compiler.events,
                          frozen_event_count=lane.compiler.frozen_event_count),
            replay=lane.pool.report(),reorder=lane.direct.report(),
            global_compiler=dict(error_on_recompile=backend._dynamo.config.error_on_recompile,
                counters={str(k):dict(v) for k,v in backend._dynamo.utils.counters.items()},
                generated_kernel_count=backend._inductor.metrics.generated_kernel_count),
            policy=dict(matmul_tf32=backend.backends.cuda.matmul.allow_tf32,
                        cudnn_tf32=backend.backends.cudnn.allow_tf32,intraop_threads=backend.get_num_threads())))
    return lane.context.call(collect)


def collect_snapshot(server, *, backend=None):
    """Call on the idle controller between complete bursts; performs no model work."""
    server._owner()
    if not server.frozen:raise ValueError('Snapshot before freeze')
    if backend is None:
        import torch as backend
    started=time.perf_counter_ns()
    lanes=server.visit(lambda lane:snapshot_lane(lane,backend))
    memory=dict(allocated_bytes=backend.cuda.memory_allocated(0),reserved_bytes=backend.cuda.memory_reserved(0),
                peak_allocated_bytes=backend.cuda.max_memory_allocated(0),peak_reserved_bytes=backend.cuda.max_memory_reserved(0))
    return dict(schema='serving-runtime-snapshot-v1',started_ns=started,finished_ns=time.perf_counter_ns(),
                lanes=lanes,memory=memory,memory_scope='Whole process CUDA device 0; do not sum across lanes.')


def integer(value):
    if type(value) is not int or value<0:raise ValueError('Nonnegative integer counter/capacity required')
    return value


def validate_lane(row):
    policy=row['policy']
    if policy!={'matmul_tf32':False,'cudnn_tf32':False,'intraop_threads':2} or any(policy[k] is not False for k in ('matmul_tf32','cudnn_tf32')):
        raise ValueError('Frozen FP32/two-thread policy changed')
    identity=row['identity']
    if identity['failed'] is not False or identity['device']!='cuda:0' or any(integer(identity[k])==0 for k in ('owner_thread_id','stream_id')):
        raise ValueError('Live indexed private owner required')
    compiler=row['compiler'];events=compiler['events']
    if compiler['phase']!='measurement' or not events or integer(compiler['frozen_event_count'])!=len(events) or not events[-1].get('success') or any(not(e.get('success') or e.get('expected_restart')) for e in events):
        raise ValueError('Frozen successful compiler evidence required')
    global_state=row['global_compiler']
    if global_state['error_on_recompile'] is not True or any(global_state['counters'].get(k) for k in ('graph_break','unimplemented')):
        raise ValueError('Global compiler guard or no-break policy failed')
    integer(global_state['generated_kernel_count'])
    replay=row['replay'];arena=replay['physical_prefix'];reorder=row['reorder']
    for k in ('graphs','replays','input_buffer_bytes','unique_input_storage_bytes','output_suffix_bytes'):integer(replay[k])
    for k in ('requests_loaded','prefix_bytes_copied','prefix_capacity_bytes','suffix_capacity_bytes','max_prefix_tokens','max_suffix_tokens','physical_prefix_batch','suffix_batch'):integer(arena[k])
    for k in ('direct_calls','fallback_calls','suffix_bytes_written','prefix_bytes_reordered'):integer(reorder[k])
    if replay['graphs']!=8 or replay['frozen'] is not True or replay['shared_input_arena'] is not True or len(replay['events'])!=8:
        raise ValueError('Eight frozen shared-prefix graphs per lane required')
    if sorted(e['cache_tokens'] for e in replay['events'])!=[n+j for n in (76,121) for j in range(4)]:
        raise ValueError('Graph shape inventory changed')
    for event in replay['events']:
        n,j=next((n,j) for n in (76,121) for j in range(4) if n+j==event['cache_tokens'])
        if event['batch']!=50 or integer(event['input_buffer_bytes'])!=PREFIX_BYTES_PER_TOKEN*n+BYTES_PER_TOKEN*j+METADATA_BYTES:
            raise ValueError('Graph input capacity/shape differs')
        duration=event['capture_and_warmup_s']
        if type(duration) not in (int,float) or not math.isfinite(duration) or duration<=0:raise ValueError('Positive capture duration required')
    expected=dict(input_buffer_bytes=sum(PREFIX_BYTES_PER_TOKEN*n+BYTES_PER_TOKEN*j+METADATA_BYTES for n in (76,121) for j in range(4)),
                  unique_input_storage_bytes=PREFIX_BYTES_PER_TOKEN*121+BYTES_PER_TOKEN*3+8*METADATA_BYTES,
                  output_suffix_bytes=BYTES_PER_TOKEN*20)
    if any(replay[k]!=v for k,v in expected.items()) or replay['capture_and_warmup_s']!=sum(e['capture_and_warmup_s'] for e in replay['events']):
        raise ValueError('Logical/unique input/output capacities differ')
    if any(arena[k]!=v for k,v in dict(prefix_capacity_bytes=PREFIX_BYTES_PER_TOKEN*121,suffix_capacity_bytes=BYTES_PER_TOKEN*3,max_prefix_tokens=121,max_suffix_tokens=3,physical_prefix_batch=1,suffix_batch=50).items()) or reorder['prefix_bytes_reordered']!=0:
        raise ValueError('Physical prefix/suffix contract differs')


def audit_interval(before,after,requests_by_lane):
    """Reconcile raw request rows assigned to each lane over exactly one interval."""
    count=len(before['lanes'])
    if count not in (1,2) or len(after['lanes'])!=count or set(requests_by_lane)!=set(range(count)) or any(type(k) is not int for k in requests_by_lane):
        raise ValueError('Exact lane/request inventory required')
    for snapshot in (before,after):
        if snapshot['schema']!='serving-runtime-snapshot-v1' or integer(snapshot['started_ns'])>integer(snapshot['finished_ns']):raise ValueError('Snapshot clock/schema invalid')
        for row in snapshot['lanes']:validate_lane(row)
        memory=snapshot['memory']
        for k in ('allocated_bytes','reserved_bytes','peak_allocated_bytes','peak_reserved_bytes'):integer(memory[k])
        if not (0<memory['allocated_bytes']<=memory['reserved_bytes']<=memory['peak_reserved_bytes']<46*1024**3 and memory['allocated_bytes']<=memory['peak_allocated_bytes']<=memory['peak_reserved_bytes']):
            raise ValueError('Invalid whole-device allocator bounds')
        if sum(row['replay']['unique_input_storage_bytes'] for row in snapshot['lanes'])>memory['allocated_bytes']:
            raise ValueError('Whole-device memory cannot cover private lane inputs')
        if len({r['identity']['owner_thread_id'] for r in snapshot['lanes']})!=count or len({r['identity']['stream_id'] for r in snapshot['lanes']})!=count:
            raise ValueError('Lane owners/streams are shared')
    if before['finished_ns']>=after['started_ns']:raise ValueError('Snapshot interval ordering invalid')
    if any(after['memory'][k]<before['memory'][k] for k in ('peak_allocated_bytes','peak_reserved_bytes')):
        raise ValueError('Allocator peaks were reset inside the audited interval')
    deltas={}
    for lane,(first,last) in enumerate(zip(before['lanes'],after['lanes'])):
        if any(first[k]!=last[k] for k in ('identity','compiler','global_compiler','policy')):raise ValueError('Frozen owner/compiler/policy changed')
        rows=requests_by_lane[lane]
        for r in rows:
            if r['lane']!=lane or r['error'] is not None or not before['finished_ns']<=r['arrival_ns']<=r['started_ns']<r['completed_ns']<=after['started_ns']:
                raise ValueError('Request is not inside its owner snapshot interval')
        lengths=[r['result']['prompt_tokens'] for r in rows]
        if any(type(n) is not int or n not in (76,121) for n in lengths):raise ValueError('Unplanned request length')
        old,new=copy.deepcopy(first['replay']),copy.deepcopy(last['replay'])
        requests=len(rows)
        if new.pop('replays')-old.pop('replays')!=4*requests:raise ValueError('Replay delta differs')
        a,z=old['physical_prefix'],new['physical_prefix']
        if z.pop('requests_loaded')-a.pop('requests_loaded')!=requests or z.pop('prefix_bytes_copied')-a.pop('prefix_bytes_copied')!=PREFIX_BYTES_PER_TOKEN*sum(lengths):raise ValueError('Input staging delta differs')
        if old!=new:raise ValueError('Graph/capture/capacity state changed')
        wanted=dict(direct_calls=4*requests,fallback_calls=requests,suffix_bytes_written=BYTES_PER_TOKEN*6*requests,prefix_bytes_reordered=0)
        delta={k:last['reorder'][k]-first['reorder'][k] for k in wanted}
        if delta!=wanted:raise ValueError('Original reorder work differs')
        deltas[str(lane)]=dict(requests=requests,replays=4*requests,reorder=delta)
    return dict(schema='serving-invariant-audit-v1',passed=True,lanes=deltas,
                memory_before=before['memory'],memory_after=after['memory'],
                note='Whole-device allocator snapshots and explicit frozen counters only; separate numerical/live ownership/overlap and provenance audits required.')
