"""Read-only graph-kernel attribution and interval-union overlap audit."""
import math


def span(event):
    start, duration = event.get('ts'), event.get('dur')
    if (event.get('ph') != 'X' or any(type(x) not in (int,float) or not math.isfinite(x) for x in (start,duration))
            or start < 0 or duration <= 0):
        raise ValueError('Complete positive finite trace interval required')
    return start, start+duration


def union(intervals):
    result=[]
    for start,end in sorted(intervals):
        if result and start<=result[-1][1]:result[-1][1]=max(end,result[-1][1])
        else:result.append([start,end])
    return result


def intersection(left,right):
    i=j=0;total=0.
    while i<len(left) and j<len(right):
        total+=max(0,min(left[i][1],right[j][1])-max(left[i][0],right[j][0]))
        if left[i][1]<=right[j][1]:i+=1
        else:j+=1
    return total


def audit_overlap(trace, expected):
    """Expected maps exact owner-produced marker names to lane indices.

    Stream IDs are CUPTI trace identities; never compare them to CUDA pointers.
    This is trace evidence only. Profile raw fidelity and live graph/tensor/source
    ownership must pass separately before this can support advancement.
    """
    if len(expected)!=4 or set(expected.values()) not in ({0},{0,1}) or any(type(v) is not int for v in expected.values()):
        raise ValueError('Four frozen owner markers required')
    lanes=set(expected.values())
    if sorted(expected.values())!=([0]*4 if lanes=={0} else [0,0,1,1]):raise ValueError('Frozen round-robin owner counts required')
    events=trace['traceEvents'];markers={};threads={}
    for event in events:
        name=event.get('name','')
        if isinstance(name,str) and name.startswith('serving_profile/'):
            if name not in expected or name in markers or event.get('cat')!='user_annotation':raise ValueError('Unexpected/duplicate profile marker')
            span(event);lane=expected[name];thread=(event['pid'],event['tid'])
            if any(type(v) is not int or v<=0 for v in thread):raise ValueError('Owner trace process/thread required')
            if lane in threads and threads[lane]!=thread:raise ValueError('Lane moved between trace threads')
            threads[lane]=thread;markers[name]=event
    if set(markers)!=set(expected):raise ValueError('Missing owner profile markers')
    if len(set(threads.values()))!=len(lanes) or len({v[0] for v in threads.values()})!=1:raise ValueError('Distinct owner threads in one process required')
    for lane in lanes:
        intervals=sorted(span(markers[name]) for name in expected if expected[name]==lane)
        if any(a[1]>b[0] for a,b in zip(intervals,intervals[1:])):raise ValueError('Owner request markers overlap')
    launches={};counts={name:0 for name in expected}
    for event in events:
        if event.get('name')!='cudaGraphLaunch' or event.get('cat') not in ('cuda_runtime','cuda_driver'):continue
        start,end=span(event)
        owners=[name for name,m in markers.items() if (event.get('pid'),event.get('tid'))==(m['pid'],m['tid']) and span(m)[0]<=start and end<=span(m)[1]]
        if len(owners)!=1:raise ValueError('Unattributed/ambiguous graph launch')
        correlation=event.get('args',{}).get('correlation')
        if type(correlation) is not int or correlation<0 or correlation in launches:raise ValueError('Unique graph launch correlation required')
        launches[correlation]=(owners[0],start);counts[owners[0]]+=1
    if any(v!=4 for v in counts.values()):raise ValueError('Four decode graph launches per request required')
    intervals={lane:[] for lane in lanes};streams={lane:set() for lane in lanes};linked={c:0 for c in launches};seen=set()
    for event in events:
        if event.get('cat')!='kernel':continue
        args=event.get('args',{});correlation=args.get('correlation')
        if correlation not in launches:continue  # Unrelated prefill cannot prove overlap.
        if type(correlation) is not int:raise ValueError('Integer kernel correlation required')
        start,end=span(event);name,launched=launches[correlation];lane=expected[name]
        if start<launched or end>span(markers[name])[1]:raise ValueError('Graph kernel outside owner completion interval')
        stream=tuple(args.get(k) for k in ('device','context','stream'))
        if any(type(v) is not int for v in stream) or stream[0]!=0 or stream[1]<0 or stream[2]<=0:raise ValueError('Explicit single-device CUDA context/stream required')
        if any(type(args.get(k)) is not int or args[k]<=0 for k in ('graph id','graph node id')):raise ValueError('Captured graph/node identity missing')
        identity=(correlation,start,end,stream,event.get('name'),args['graph node id'])
        if identity in seen:raise ValueError('Duplicate graph kernel evidence')
        seen.add(identity);linked[correlation]+=1;streams[lane].add(stream);intervals[lane].append((start,end))
    if any(v==0 for v in linked.values()):raise ValueError('Every graph launch needs linked kernels')
    if any(len(v)!=1 for v in streams.values()):raise ValueError('Each lane requires one stable actual replay stream')
    actual={lane:next(iter(v)) for lane,v in streams.items()}
    if len(set(actual.values()))!=len(lanes) or len({v[:2] for v in actual.values()})!=1:raise ValueError('Disjoint actual streams on one device/context required')
    merged={lane:union(v) for lane,v in intervals.items()}
    overlap=intersection(merged[0],merged[1]) if len(lanes)==2 else 0.
    return dict(schema='serving-overlap-audit-v1',attribution_passed=True,
                gpu_overlap_proven=len(lanes)==2 and overlap>0,graph_kernel_overlap_us=overlap,
                lane_graph_active_us={str(k):sum(b-a for a,b in v) for k,v in merged.items()},
                actual_trace_streams={str(k):list(v) for k,v in actual.items()},
                trace_owner_threads={str(k):list(v) for k,v in threads.items()},
                graph_launches=len(launches),graph_kernel_events=sum(linked.values()),
                note='Instrumented graph-linked kernel union intersection only; no throughput or speed estimate.')
