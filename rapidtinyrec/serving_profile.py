"""Untimed owner-thread profile after complete primary/secondary measurements."""
import json
from pathlib import Path
import time

from .audit_decode_pilot import compare
from .audit_prefix_profile import exact
from .serving_initialize import save_exclusive
from .serving_lanes import Request
from .serving_measurement_audit import ORDERS, audit_burst
from .serving_overlap_audit import audit_overlap
from .serving_setup_audit import read_unique


def profile_after_measurements(server, users, phase, output, *, backend=None):
    """One stock profiler, all existing owner threads, no timing samples added.

    The final harness must additionally bind these records to the active phase,
    source/allocation and deadlines. Injected backends are for CPU lifecycle tests.
    """
    server._owner()
    if not server.frozen or len(users)!=4 or len({u['cohort_index'] for u in users})!=4:
        raise ValueError('Frozen service and four distinct users required')
    lanes=len(server.lanes);index=phase['phase']
    if type(index) is not int or index not in range(4) or lanes!=(1,2,2,1)[index] or phase['lane_count']!=lanes or phase['concurrency']!=lanes:
        raise ValueError('Frozen phase/admission identity required')
    previous=-1;measured={}
    for kind in ('primary','secondary'):
        if len(phase[kind])!=(3 if kind=='primary' or lanes==2 else 0):raise ValueError('Complete measurements must precede profiling')
        for local,burst in enumerate(phase[kind]):
            if burst['repeat']!=index//2*3+local:raise ValueError('Frozen measurement schedule required')
            audit_burst(burst['records'],[users[i] for i in ORDERS[local]],lane_count=lanes,concurrency=lanes if kind=='primary' else 1)
            if burst['records'][0]['arrival_ns']<=previous:raise ValueError('Measurements overlap or are out of order')
            previous=max(r['completed_ns'] for r in burst['records'])
            for row in burst['records']:measured.setdefault(row['result']['cohort_index'],row['result']['raw'])
    started=time.perf_counter_ns()
    if previous>=started:raise ValueError('Measurement completion must precede profiler initialization')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    identities=server.visit(lambda state:state.context.identity())
    if len({v['owner_thread_id'] for v in identities})!=lanes:raise ValueError('Distinct persistent owners required')
    expected={f"serving_profile/lane_{pos%lanes}/user_{u['cohort_index']}":pos%lanes for pos,u in enumerate(users)}
    save_exclusive(root/'profile-start.json',dict(phase=index,started_ns=started,last_measurement_completion_ns=previous,
                   owners=identities,expected_markers=expected,performance_sample=False))
    if backend is None:
        import torch as backend
    original=server.run;profiler=None;records=None;failure=None
    def marked(state,user):
        identity=state.context.identity()
        matches=[i for i,v in enumerate(identities) if v==identity]
        if len(matches)!=1:raise ValueError('Profile runtime owner changed')
        name=f"serving_profile/lane_{matches[0]}/user_{user['cohort_index']}"
        if name not in expected:raise ValueError('Unplanned profile request')
        with backend.profiler.record_function(name):
            return original(state,user)
    try:
        config=backend.profiler._ExperimentalConfig(profile_all_threads=True,custom_profiler_config='')
        profiler=backend.profiler.profile(activities=[backend.profiler.ProfilerActivity.CPU,backend.profiler.ProfilerActivity.CUDA],experimental_config=config)
        server.run=marked
        with profiler:
            records=server.run_burst([Request(str(u['cohort_index']),u) for u in users],lanes)
        save_exclusive(root/'profile-records.json',records)
    except BaseException as error:
        failure=error
        if hasattr(error,'records'):save_exclusive(root/'profile-failed-records.json',error.records)
        elif records is not None:save_exclusive(root/'profile-failed-records.json',records)
    finally:
        server.run=original
        if profiler is not None:
            try:
                profiler.export_chrome_trace(str(root/'trace.json'))
            except BaseException as error:
                if failure is None:failure=error
                else:failure.add_note(f'Trace export also failed: {error!r}')
    if failure is not None:
        save_exclusive(root/'profile-failure.json',dict(error=repr(failure)))
        raise failure
    try:
        trace=read_unique(root/'trace.json')
        overlap=audit_overlap(trace,expected)
        clocks=audit_burst(records,users,lane_count=lanes,concurrency=lanes)
        pairs=[(measured[r['result']['cohort_index']],r['result']['raw']) for r in records]
        fidelity=compare(pairs);ancestry=all(a['beam_indices']==b['beam_indices'] for a,b in pairs)
        report=dict(overlap=overlap,numerical_comparison=fidelity,ancestry_stable=ancestry,
                    numerical_passed=clocks['outputs_valid'] and exact(fidelity) and ancestry,
                    performance_sample=False,advancement_ready=False)
    except Exception as error:
        save_exclusive(root/'profile-failure.json',dict(error=repr(error)))
        raise
    save_exclusive(root/'profile-audit.json',report)
    return report
