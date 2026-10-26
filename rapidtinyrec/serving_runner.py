"""Execute serving phases only inside the fixed external deadline supervisor.

This is an allocated-compute entrypoint, not a submission command. Source/live
allocation verification and all phase/audit work run in the supervised child.
A normal exit may preserve failed experimental gates; it never means promotion.
"""
import time
# Start before CLI parsing, runtime imports, filesystem or allocation preflight.
_ENTRY_STARTED_NS = time.perf_counter_ns()

import argparse
import os
from pathlib import Path
import re
import sys


def _entry_guard(run_root, release_root, historical):
    if not sys.dont_write_bytecode or sys.flags.optimize != 0:
        raise ValueError('Serving runner requires -B and nonoptimized Python')
    paths=[Path(p) for p in (run_root,release_root,historical)]
    if any(not p.is_absolute() or '..' in p.parts for p in paths):
        raise ValueError('Absolute canonical run/release/historical paths required')
    run,release,history=paths
    if str(run)!=os.environ.get('OSCAR_RUN_DIR') or str(run/'outputs')!=os.environ.get('OSCAR_OUTPUT_DIR'):
        raise ValueError('Allocated run/output environment required')
    if re.fullmatch('[1-9][0-9]*',os.environ.get('SLURM_JOB_ID','')) is None:
        raise ValueError('Actual allocated job environment required')
    return run,release,history


def allocated_auditor(run, historical):
    """The only production final callback; retain malformed-evidence failures."""
    from .serving_run_audit import audit_pre_exit
    from .serving_initialize import save_exclusive
    try:
        return audit_pre_exit(run,historical)
    except BaseException as error:
        save_exclusive(Path(run)/'outputs/coordinator/audit-failure.json',dict(
            schema='serving-allocated-audit-failure-v1',failed_ns=time.perf_counter_ns(),
            error=repr(error),evidence_gates_passed=False,advancement_ready=False))
        raise


def run_coordinator(run, release, historical):
    from .serving_coordinator import coordinate
    from .serving_initialize import save_exclusive
    try:
        # No caller-supplied auditor can replace the production disk reader.
        coordinate(run,release,historical,final_auditor=allocated_auditor)
    except BaseException as error:
        save_exclusive(run/'outputs/coordinator-failure.json',dict(
            schema='serving-coordinator-failure-v1',failed_ns=time.perf_counter_ns(),
            error=repr(error),advancement_ready=False))
        raise
    # Numerical/performance failures live in final-audit.json. Exit status
    # distinguishes incomplete execution, so a negative result is preserved.
    return 0


def run_supervised(run, release, historical, *, global_started_ns):
    from .serving_supervisor import supervise
    command=[sys.executable,'-B','-m','rapidtinyrec.serving_runner','--coordinator',
             '--run-root',str(run),'--release-root',str(release),'--historical',str(historical)]
    result=supervise(command,run/'code',dict(os.environ),run/'outputs/supervision',global_started_ns=global_started_ns)
    return 0 if result['complete'] is True else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coordinator',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--release-root',type=Path,required=True)
    parser.add_argument('--historical',type=Path,required=True)
    args=parser.parse_args(argv)
    run,release,history=_entry_guard(args.run_root,args.release_root,args.historical)
    if args.coordinator:return run_coordinator(run,release,history)
    return run_supervised(run,release,history,global_started_ns=_ENTRY_STARTED_NS)


if __name__=='__main__':raise SystemExit(main())
