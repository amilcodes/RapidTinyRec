"""Distinct wave identity, pair schedule and finite-pair latency accounting.

Metadata arithmetic only. This does not certify generation, allocation, source,
quality or dispatch; those require the separate allocated raw auditor.
"""
import hashlib
import json
import math
import statistics
from pathlib import Path

PROTOCOL='benchmarks/preparation/quality-frontier/beam-wave-pilot-protocol.json'
PROTOCOL_SHA256='1df324f384dfe5763b031667b94da7df741cca4b90bf27889a3d324c3de02580'
PHASES=('serial_pair_tf32','beam_wave_tf32','beam_wave_tf32','serial_pair_tf32')
PAIRS=((1,2),(6,10))
LENGTHS=(76,121)


def validate_protocol(p):
    if (p['schema']!='two-request-beam-wave-pilot-v1' or p['experiment']!='two_request_beam_wave_tf32_pilot'
            or p['profile']!='two-request-beam-wave-pilot'):
        raise ValueError('Distinct beam-wave identity required')
    if p['phases']!=[dict(phase=i,variant=v,half=i//2,reuse_disk_artifacts=i>=2) for i,v in enumerate(PHASES)]:
        raise ValueError('Four fresh ABBA processes required')
    if p['pairs']!=[dict(pair_index=i,cohort_indices=list(users),prompt_tokens=LENGTHS[i]) for i,users in enumerate(PAIRS)]:
        raise ValueError('Frozen equal-length pair scope differs')
    if [p[k] for k in ('repeats_per_half','waves_per_phase','requests_per_phase','graphs_per_phase')]!=[3,6,12,8]:
        raise ValueError('Frozen pilot workload differs')
    if [p['budget'][k] for k in ('global_work_deadline_s','generation_child_deadline_s','audit_maximum_s','nominal_cleanup_s')]!=[1620,1500,120,180]:
        raise ValueError('Global deadline contract differs')
    if p['promotion'] is not False or p['noninferiority_established'] is not False:
        raise ValueError('Exploratory pilot cannot promote or establish noninferiority')


def load_protocol(path):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PROTOCOL_SHA256:raise ValueError('Frozen wave protocol changed')
    result=json.loads(raw);validate_protocol(result);return result


def phase_schedule(phase):
    if type(phase) is not int or phase not in range(4):raise ValueError('Invalid wave phase')
    return [dict(phase=phase,variant=PHASES[phase],half=phase//2,repeat=repeat,pair_index=i,
                 cohort_indices=list(users if repeat%2==0 else users[::-1]),prompt_tokens=LENGTHS[i])
            for repeat in range((phase//2)*3,(phase//2)*3+3) for i,users in enumerate(PAIRS)]


def validate_timing(row):
    """Offsets are actual completed output times, not amortized kernel times."""
    if row.get('schema')!='two-request-beam-wave-timing-v1':raise ValueError('Wrong timing schema')
    phase=row['phase'];schedule=phase_schedule(phase)
    identity={k:row[k] for k in schedule[0]}
    if identity not in schedule:raise ValueError('Wrong pair identity or serial order')
    if row['arrival_offsets_s']!=[0,0]:raise ValueError('Both requests must be ready at pair origin')
    durations=row['completion_offsets_s'];end=row['pair_duration_s']
    if len(durations)!=2 or any(type(x) not in (int,float) or not math.isfinite(x) or x<=0 for x in [*durations,end]):
        raise ValueError('Finite positive pair and completion durations required')
    if PHASES[phase]=='serial_pair_tf32':
        if not durations[0]<durations[1] or durations[1]!=end:
            raise ValueError('Serial response must include second-request queue time')
    elif durations!=[end,end]:
        raise ValueError('Both wave responses complete at whole-wave completion')
    return row


def validate_phase_rows(phase,rows):
    expected=phase_schedule(phase)
    if len(rows)!=len(expected):raise ValueError('Incomplete wave phase')
    for row,spec in zip(rows,expected):
        validate_timing(row)
        if any(row[k]!=v for k,v in spec.items()):raise ValueError('Duplicate, reordered or wrong-phase pair')


def percentile(values,q):
    values=sorted(values)
    if not values:raise ValueError('Empty percentile')
    x=(len(values)-1)*q;lo=math.floor(x);hi=math.ceil(x)
    return values[lo]+(values[hi]-values[lo])*(x-lo)


def summarize(rows):
    if not rows:raise ValueError('Empty pair stratum')
    for row in rows:validate_timing(row)
    spans=[x['pair_duration_s'] for x in rows]
    response=[v for x in rows for v in x['completion_offsets_s']]
    stats=lambda xs:dict(median_ms=statistics.median(xs)*1000,p95_ms=percentile(xs,.95)*1000)
    return dict(pairs=len(rows),requests=2*len(rows),throughput_requests_per_s=2*len(rows)/sum(spans),
                pair_makespan=stats(spans),response=stats(response),
                first_response=stats([x['completion_offsets_s'][0] for x in rows]),
                second_response=stats([x['completion_offsets_s'][1] for x in rows]))


def compare_phases(phases):
    """Require all four inventories before producing matched descriptive strata."""
    if set(phases)!={0,1,2,3}:raise ValueError('All four wave phases required')
    for phase,rows in phases.items():validate_phase_rows(phase,rows)
    control=phases[0]+phases[3];candidate=phases[1]+phases[2]
    key=lambda r:(r['repeat'],r['pair_index'])
    c={key(r):r for r in control};t={key(r):r for r in candidate}
    if set(c)!=set(t):raise ValueError('Pairing differs')
    groups={'overall':list(c)}
    for length in LENGTHS:groups[f'length_{length}']=[k for k in c if c[k]['prompt_tokens']==length]
    for half in range(2):
        groups[f'half_{half}']=[k for k in c if c[k]['half']==half]
        for length in LENGTHS:groups[f'length_{length}_half_{half}']=[k for k in c if c[k]['half']==half and c[k]['prompt_tokens']==length]
    result={}
    for name,keys in groups.items():
        a=summarize([c[k] for k in keys]);b=summarize([t[k] for k in keys])
        result[name]=dict(control=a,candidate=b,throughput_gain_percent=100*(b['throughput_requests_per_s']/a['throughput_requests_per_s']-1),
            response_median_change_percent=100*(b['response']['median_ms']/a['response']['median_ms']-1),
            response_p95_change_percent=100*(b['response']['p95_ms']/a['response']['p95_ms']-1))
    primary=['overall','length_76','length_121','half_0','half_1']
    perf=(all(result[k]['throughput_gain_percent']>=5 for k in primary)
          and all(v['throughput_gain_percent']>=0 for v in result.values())
          and result['overall']['response_median_change_percent']<=0
          and result['overall']['response_p95_change_percent']<=0)
    return dict(schema='two-request-beam-wave-timing-review-v1',strata=result,
                timing_screen_passed=perf,raw_audit_passed=None,quality_equivalence=False,promotion=False)
