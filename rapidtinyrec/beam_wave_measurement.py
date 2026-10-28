"""Collect a finite two-request trial with output extraction outside timing.

Callbacks allow boundary tests without loading Torch or the release checkpoint.
The caller owns input preparation and source/live-allocation/deadline checks.
"""
from .beam_wave_protocol import phase_schedule,validate_timing


def measure_pair(spec,generate,synchronize,clock,unpack):
    phase=spec['phase']
    if spec not in phase_schedule(phase):raise ValueError('Pair is outside frozen schedule')
    users=spec['cohort_indices']
    outputs=[];completed=[]
    synchronize()
    start=clock()
    if spec['variant']=='serial_pair_tf32':
        for user in users:
            outputs.append(generate([user]))
            synchronize();completed.append(clock()-start)
    else:
        outputs.append(generate(users))
        synchronize();end=clock()-start;completed=[end,end]
    timing=dict(spec,schema='two-request-beam-wave-timing-v1',arrival_offsets_s=[0,0],
                completion_offsets_s=completed,pair_duration_s=completed[-1])
    validate_timing(timing)
    # Retain both serial outputs through second completion. Never reuse a
    # graph-backed borrowed score tensor: unpack must verify HF clone ownership.
    if len(outputs)==2:
        raw=[]
        for output,user in zip(outputs,users):
            single=unpack(output,[user])
            if len(single)!=1:raise ValueError('Serial extraction must return exactly one request')
            raw.append(single[0])
    else:
        raw=unpack(outputs[0],users)
    if len(raw)!=2 or any(r['cohort_index']!=u for r,u in zip(raw,users)):
        raise ValueError('Extracted request outputs differ from timed pair order')
    return dict(schema='two-request-beam-wave-trial-v1',timing=timing,requests=raw)
