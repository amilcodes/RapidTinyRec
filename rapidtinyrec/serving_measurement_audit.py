"""Independent CPU audit of frozen finite-burst records, without GPU claims.

Input users/history must first pass the setup identity audit. This core validates
schedule, clocks, raw numerics and performance; actual CUDA trace/ownership and
source/allocation evidence are separately required before advancement.
"""
import math
import statistics

from .audit_decode_pilot import compare
from .audit_prefix_profile import exact
from .serving_setup_audit import validate_raw

ORDERS = ((0,1,2,3), (0,2,3,1), (1,0,3,2))


def stats(values):
    return dict(n=len(values), median_ms=statistics.median(values),
                p95_ms=statistics.quantiles(values,n=100,method='inclusive')[94] if len(values)>1 else values[0])


def audit_burst(records, users, *, lane_count, concurrency):
    if type(lane_count) is not int or type(concurrency) is not int or lane_count not in (1,2) or concurrency not in (1,2) or concurrency>lane_count:
        raise ValueError('Invalid lane/admission configuration')
    if len(records)!=4 or len(users)!=4 or len({r['request_id'] for r in records})!=4:
        raise ValueError('Complete four-request burst with unique IDs required')
    arrival=records[0]['arrival_ns']; previous={}; valid=True; end=0
    for position,(row,user) in enumerate(zip(records,users)):
        if not isinstance(row['request_id'],str) or not row['request_id'] or row['error'] is not None:
            raise ValueError('Failed/missing request cannot be dropped or timed as success')
        if type(row['lane']) is not int or row['lane']!=position%lane_count:
            raise ValueError('Frozen round-robin lane mapping required')
        result=row['result']
        if any(type(result[k]) is not int or result[k]!=user[k] for k in ('cohort_index','prompt_tokens')):
            raise ValueError('Request order/cohort mismatch')
        times=[row[k] for k in ('arrival_ns','submitted_ns','started_ns')]+[result['generation_started_ns'],result['generation_finished_ns'],row['completed_ns']]
        if any(type(x) is not int or x<0 for x in times) or not (times[0]<=times[1]<=times[2]<=times[3]<times[4]<=times[5]):
            raise ValueError('Generation must finish within response clock interval')
        if row['arrival_ns']!=arrival:
            raise ValueError('All requests must share one arrival clock')
        elapsed=result['generation_elapsed_s']
        if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or not math.isclose(elapsed,(times[4]-times[3])/1e9,rel_tol=1e-12,abs_tol=0):
            raise ValueError('Generation elapsed does not match raw clock')
        if row['started_ns']<previous.get(row['lane'],0):
            raise ValueError('Same-lane request intervals overlap')
        if concurrency==1 and row['submitted_ns']<end:
            raise ValueError('Serial admission precedes previous completion')
        if position and row['submitted_ns']<records[position-1]['submitted_ns']:
            raise ValueError('Admission order changed')
        previous[row['lane']]=row['completed_ns'];end=row['completed_ns']
        valid &= validate_raw(result['raw'])
    duration=max(row['completed_ns'] for row in records)-arrival
    overlaps=[max(0,min(a['completed_ns'],b['completed_ns'])-max(a['started_ns'],b['started_ns']))
              for i,a in enumerate(records) for b in records[i+1:] if a['lane']!=b['lane']]
    return dict(requests=4,duration_ns=duration,requests_per_second=4e9/duration,
                outputs_valid=bool(valid),cross_lane_host_overlap_ns=sum(overlaps),
                gpu_overlap_proven=False)


def metrics(bursts):
    rows=[row for burst in bursts for row in burst['records']]
    def values(group):
        return dict(response=stats([(r['completed_ns']-r['arrival_ns'])/1e6 for r in group]),
                    queue=stats([(r['started_ns']-r['arrival_ns'])/1e6 for r in group]),
                    service=stats([(r['completed_ns']-r['started_ns'])/1e6 for r in group]),
                    generation=stats([r['result']['generation_elapsed_s']*1e3 for r in group]))
    return dict(requests=len(rows),bursts=len(bursts),
                requests_per_second=len(rows)*1e9/sum(b['audit']['duration_ns'] for b in bursts),
                overall=values(rows),
                by_length={str(n):values([r for r in rows if r['result']['prompt_tokens']==n]) for n in sorted({r['result']['prompt_tokens'] for r in rows})},
                by_position={str(i):values([b['records'][i] for b in bursts]) for i in range(4)})


def audit_measurements(phases, users, historical):
    """Audit four ABBA phases and separate post-primary serial diagnostics.

    Each phase supplies phase/lane_count/concurrency plus primary and secondary
    lists of {repeat, records}. All raw request records are retained in reports.
    """
    if len(users)!=4 or len({u['cohort_index'] for u in users})!=4 or [u['prompt_tokens'] for u in users]!=[76,76,121,121]:
        raise ValueError('Frozen four-user extreme-length cohort required')
    if set(historical)!={u['cohort_index'] for u in users} or len(phases)!=4:
        raise ValueError('Complete historical user set and four phases required')
    historical_valid=all(validate_raw(raw,historical=True) for raw in historical.values())
    primary={1:[],2:[]}; secondary=[]; seen={}; historical_pairs=[]; stable=[]; joined={}; ancestry=True; valid=historical_valid; audited=[]
    for index,phase in enumerate(phases):
        lanes=(1,2,2,1)[index];half=index//2
        if any(type(phase[k]) is not int or phase[k]!=v for k,v in (('phase',index),('lane_count',lanes),('concurrency',lanes))):
            raise ValueError('Fresh-process ABBA phase identity mismatch')
        if len(phase['primary'])!=3 or len(phase['secondary'])!=(3 if lanes==2 else 0):
            raise ValueError('Primary/secondary burst counts differ from frozen schedule')
        previous_completion=-1
        for kind in ('primary','secondary'):
            for local,burst in enumerate(phase[kind]):
                repeat=half*3+local
                if type(burst['repeat']) is not int or burst['repeat']!=repeat:
                    raise ValueError('Frozen repeat/half order mismatch')
                ordered=[users[i] for i in ORDERS[local]]
                rows=burst['records'];report=audit_burst(rows,ordered,lane_count=lanes,concurrency=lanes if kind=='primary' else 1)
                if rows[0]['arrival_ns']<=previous_completion:
                    raise ValueError('Bursts overlap or diagnostics precede primary measurements')
                previous_completion=max(r['completed_ns'] for r in rows)
                item=dict(phase=index,half=half,repeat=repeat,kind=kind,records=rows,audit=report)
                (primary[lanes] if kind=='primary' else secondary).append(item);audited.append(item)
                valid &= report['outputs_valid']
                for row in rows:
                    key=row['result']['cohort_index'];raw=row['result']['raw']
                    historical_pairs.append((historical[key],raw))
                    if key in seen:
                        stable.append((seen[key],raw));ancestry &= seen[key]['beam_indices']==raw['beam_indices']
                    else:seen[key]=raw
                    if kind=='primary':
                        pair=joined.setdefault((repeat,key),{})
                        if lanes in pair:raise ValueError('Duplicate paired observation')
                        pair[lanes]=raw
    if len(joined)!=24 or any(set(pair)!={1,2} for pair in joined.values()):
        raise ValueError('All twenty-four paired requests required')
    comparisons=dict(historical=compare(historical_pairs),stability=compare(stable),paired=compare([(p[1],p[2]) for p in joined.values()]))
    summary={str(lanes):dict(overall=metrics(bursts),by_half={str(h):metrics([b for b in bursts if b['half']==h]) for h in (0,1)}) for lanes,bursts in primary.items()}
    control,candidate=summary['1'],summary['2'];gates={}
    for name,c,t in [('overall',control['overall'],candidate['overall'])]+[(f'half_{h}',control['by_half'][str(h)],candidate['by_half'][str(h)]) for h in (0,1)]:
        gates[name+'_throughput']=t['requests_per_second']>=1.05*c['requests_per_second']
        gates[name+'_response']=all(t['overall']['response'][s]<=c['overall']['response'][s] for s in ('median_ms','p95_ms'))
    for length in ('76','121'):
        gates['length_'+length+'_response']=all(candidate['overall']['by_length'][length]['response'][s]<=control['overall']['by_length'][length]['response'][s] for s in ('median_ms','p95_ms'))
    return dict(schema='serving-measurement-audit-core-v1',numerical_passed=bool(valid and ancestry and all(exact(c) for c in comparisons.values())),
                performance_passed=all(gates.values()),gates=gates,comparisons=comparisons,ancestry_stable=bool(ancestry),outputs_valid=bool(valid),
                primary=summary,secondary_diagnostic=metrics(secondary),bursts=audited,
                concurrent_host_overlap_observed=all(b['audit']['cross_lane_host_overlap_ns']>0 for b in primary[2]),
                throughput_aggregation='Total completed requests divided by sum of complete burst durations; no gaps or setup included.',
                advancement_ready=False,gpu_overlap_proven=False,
                remaining='Actual CUDA overlap/graph-launch attribution, runtime invariants, memory, setup/source/run/allocation identity and harness binding required.')
