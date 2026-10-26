"""Serving phase worker. Requires a parent launch ticket; never submits jobs.

The external deadline supervisor, final run reconciliation and reviewed six-lane
budget are still required. This worker alone is not an allocation-ready harness.
"""
import argparse
import os
from pathlib import Path
import sys
import time

from .serving_allocation import require, verify_source
from .serving_contract import PROTOCOL, load_protocol
from .serving_process import (process_identity, validate_launch, check_deadline,
                              observe_cache, validate_cache_environment)
from .serving_setup_audit import read_unique, setup_auditor
from .serving_initialize import lane_initializer, save_exclusive
from .serving_service import prepared_service
from .serving_phase import run_prepared_phase


def run_phase(run_root, phase, release_root, historical):
    started = time.perf_counter_ns()
    require(sys.dont_write_bytecode and sys.flags.optimize == 0,
            'Serving worker requires -B and normal Python assertions')
    require(type(phase) is int and phase in range(4), 'Frozen phase index required')
    run = Path(run_root)
    require(run.is_dir() and run.resolve() == run and str(run) == os.environ.get('OSCAR_RUN_DIR') and
            str(run/'outputs') == os.environ.get('OSCAR_OUTPUT_DIR'), 'Canonical allocated run/output required')
    require(not (run/'outputs').is_symlink(), 'Linked output directory')
    protocol = load_protocol(run/'code'/PROTOCOL)
    ticket = run/'outputs'/'launch'/f'phase_{phase}.json'
    require(ticket.is_file() and not ticket.is_symlink() and ticket.resolve() == ticket and
            ticket.stat().st_nlink == 1, 'Original parent launch ticket required')
    ticket_bytes = ticket.read_bytes(); launch = read_unique(ticket)
    identity = process_identity(); parent = process_identity(identity['parent_pid'])
    spec = validate_launch(launch, run, phase, protocol, identity, parent,
                           os.environ.get('SERVING_LAUNCH_TOKEN'))
    require(launch['launch_started_ns'] <= started, 'Parent launch must precede worker entry')
    cache_env = validate_cache_environment(launch['cache']['root'], os.environ)
    root = run/'outputs'/f'phase_{phase}'; root.mkdir(exist_ok=False)
    journal = []; failure = None; completed = None; closed_ns = None
    save_exclusive(root/'worker-entry.json', dict(schema='serving-worker-entry-v1', phase=phase,
        started_ns=started, identity=identity, launch=launch, cache_environment=cache_env))
    try:
        before = observe_cache(launch['cache']['root'])
        save_exclusive(root/'cache-before.json', before)
        require(bool(before) == spec['reuse_disk_cache'], 'Cold/reuse cache contents differ')
        check_deadline(launch)
        initialize = lane_initializer(release_root, historical, root/'setup')
        # This is written before prepared_service can invoke any model lane.
        observation = initialize.allocation_observation
        save_exclusive(root/'allocation-observation.json', observation)
        verify_source(run/'code', observation)
        check_deadline(launch)
        callback = setup_auditor(root/'setup', historical, root/'setup-audit.json')
        with prepared_service(initialize, callback, lane_count=spec['resident_lanes'],
                              expected_graphs=spec['graphs_per_lane'], lifecycle=journal) as (server, report):
            save_exclusive(root/'preparation.json', report)
            check_deadline(launch)
            completed = run_prepared_phase(server, protocol['cohort']['users'], phase, root/'execution')
        closed_ns = time.perf_counter_ns()
        require(server.closed and all(not lane.thread.is_alive() for lane in server.lanes),
                'Owner threads remain live after shutdown')
        check_deadline(launch)
        save_exclusive(root/'cache-after.json', observe_cache(launch['cache']['root']))
        verify_source(run/'code', observation)
        require(ticket.read_bytes() == ticket_bytes and process_identity() == identity and
                process_identity(identity['parent_pid']) == parent, 'Launch/process identity changed')
        check_deadline(launch)
    except BaseException as error:
        failure = repr(error)
        raise
    finally:
        # Uncatchable kill cannot run finally; the future supervisor must retain
        # parent-observed termination and classify missing shutdown as incomplete.
        save_exclusive(root/'lifecycle.json', journal)
        save_exclusive(root/'worker-exit.json', dict(schema='serving-worker-exit-v1', phase=phase,
            identity=identity, started_ns=started, finished_ns=time.perf_counter_ns(),
            service_closed_ns=closed_ns, execution_complete=completed is not None,
            complete=failure is None, error=failure, advancement_ready=False))
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--phase', type=int, choices=range(4), required=True)
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--historical', type=Path, required=True)
    args = parser.parse_args()
    run_phase(args.run_root, args.phase, args.release_root, args.historical)


if __name__ == '__main__':
    main()
