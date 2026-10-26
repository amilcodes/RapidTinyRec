"""Frozen phase orchestration on an already admitted, prepared service.

This is not an allocation/submission CLI. A harness must bind setup/source/live
allocation, cold/reused disk cache provenance and an external hard deadline.
"""
import copy
from pathlib import Path
import time

from .serving_graph_ownership import audit_graph_owners
from .serving_storage import audit_lane_storage
from .serving_initialize import save_exclusive
from .serving_lanes import Request
from .serving_measurement_audit import ORDERS, audit_burst
from .serving_invariants import collect_snapshot, audit_interval
from .serving_profile import profile_after_measurements


def live_ownership(server):
    """Hold simultaneous strong references until both live ownership audits end."""
    snapshots=server.visit(lambda lane:lane.ownership_snapshot(8))
    try:
        records={i:s['record'] for i,s in enumerate(snapshots)}
        graph=audit_graph_owners(records,8)
        storage=audit_lane_storage({i:r['storage'] for i,r in records.items()})
        return copy.deepcopy(dict(owners=records,graph_ownership=graph,storage_ownership=storage))
    finally:
        snapshots=None


def run_prepared_phase(server, users, phase_index, output):
    """Preserve all raw bursts before auditing; profile only after measurements.

    Setup gates must already have admitted this server. A complete run auditor
    must independently re-read these files; this online audit is not that reader.
    """
    server._owner()
    if type(phase_index) is not int or phase_index not in range(4) or len(server.lanes)!=(1,2,2,1)[phase_index] or not server.frozen:
        raise ValueError('Frozen ABBA service identity required')
    if len(users)!=4 or len({u['cohort_index'] for u in users})!=4 or [u['prompt_tokens'] for u in users]!=[76,76,121,121]:
        raise ValueError('Frozen four-user serving cohort required')
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    lanes=len(server.lanes);phase=dict(phase=phase_index,lane_count=lanes,concurrency=lanes,primary=[],secondary=[])
    start=time.perf_counter_ns();save_exclusive(root/'phase-start.json',dict(phase=phase_index,users=users,started_ns=start,advancement_ready=False))
    try:
        ownership_before=live_ownership(server);save_exclusive(root/'ownership-before.json',ownership_before)
        for kind in ('primary','secondary'):
            if kind=='secondary' and lanes==1:continue
            for local,order in enumerate(ORDERS):
                repeat=phase_index//2*3+local;label=f'{kind}-{repeat}'
                ordered=[users[i] for i in order]
                before=collect_snapshot(server);save_exclusive(root/f'{label}-before.json',before)
                records=None;failure=None
                try:
                    records=server.run_burst([Request(f'{kind}:{repeat}:{u["cohort_index"]}',u) for u in ordered],lanes if kind=='primary' else 1)
                except BaseException as error:
                    failure=error
                    if hasattr(error,'records'):records=error.records
                if records is not None:save_exclusive(root/f'{label}-records.json',records)
                if failure is not None:raise failure
                after=collect_snapshot(server);save_exclusive(root/f'{label}-after.json',after)
                # Both snapshots and raw results survive all audit failures.
                clocks=audit_burst(records,ordered,lane_count=lanes,concurrency=lanes if kind=='primary' else 1)
                invariants=audit_interval(before,after,{i:[r for r in records if r['lane']==i] for i in range(lanes)})
                save_exclusive(root/f'{label}-audit.json',dict(clocks=clocks,invariants=invariants))
                phase[kind].append(dict(repeat=repeat,records=records))
        save_exclusive(root/'measurements.json',phase)
        ownership_measured=live_ownership(server);save_exclusive(root/'ownership-measured.json',ownership_measured)
        if ownership_measured!=ownership_before:raise ValueError('Live storage/graph ownership changed during measurements')
        profile_before=collect_snapshot(server);save_exclusive(root/'profile-before.json',profile_before)
        profile=profile_after_measurements(server,users,phase,root/'profile')
        profile_after=collect_snapshot(server);save_exclusive(root/'profile-after.json',profile_after)
        from .serving_setup_audit import read_unique
        profiled_records=read_unique(root/'profile/profile-records.json')
        profile_invariants=audit_interval(profile_before,profile_after,{i:[r for r in profiled_records if r['lane']==i] for i in range(lanes)})
        save_exclusive(root/'profile-invariants.json',profile_invariants)
        ownership_after=live_ownership(server);save_exclusive(root/'ownership-after.json',ownership_after)
        if ownership_after!=ownership_before:raise ValueError('Live storage/graph ownership changed during profiling')
        result=dict(schema='serving-phase-completion-v1',complete=True,phase=phase_index,
                    started_ns=start,finished_ns=time.perf_counter_ns(),primary_requests=12,
                    secondary_requests=12 if lanes==2 else 0,profile_requests=4,
                    profile_numerical_passed=profile['numerical_passed'],
                    gpu_overlap_proven=profile['overlap']['gpu_overlap_proven'],advancement_ready=False)
        save_exclusive(root/'phase-complete.json',result)
        return result
    except BaseException as error:
        save_exclusive(root/'phase-failure.json',dict(phase=phase_index,error=repr(error),finished_ns=time.perf_counter_ns(),complete=False))
        raise
