"""Allocated-compute supervisor for native API log preservation.

This does not submit jobs or choose a benchmark. The allocation must reserve time
for termination and copying after the child deadline. SIGKILL/node loss of the
supervisor itself cannot guarantee recovery; its prewritten location record
identifies any local evidence. Only normal child exit can support a full audit.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from .precision_log_staging import LogStage, filesystem_for, save_record


def supervise(command, env, timeout_s, grace_s=10):
    if timeout_s <= 0 or grace_s <= 0:
        raise ValueError('Positive child timeout and termination grace required')
    stop = []
    previous = {}
    def request_stop(signum, frame):
        if not stop: stop.append(signum)
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous[sig] = signal.signal(sig, request_stop)
    process = None
    started = time.monotonic()
    reason = None
    try:
        process = subprocess.Popen(command, env=env, start_new_session=True)
        while process.poll() is None:
            if stop or time.monotonic()-started >= timeout_s:
                reason = 'signal' if stop else 'deadline'
                break
            time.sleep(min(.05, timeout_s))
        if reason:
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: process.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=grace_s)
        else:
            process.wait()
        return {'schema': 'precision-child-exit-v1', 'returncode': process.returncode,
                'reason': reason or 'exited', 'signal': stop[0] if stop else None,
                'elapsed_s': time.monotonic()-started,
                'completed_normally': reason is None and process.returncode == 0}
    finally:
        if process is not None:
            # Terminate only the private child process group; never unrelated jobs.
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            if process.poll() is None: process.wait(timeout=grace_s)
        for sig, handler in previous.items(): signal.signal(sig, handler)


def recover_logs(output_root, local_root, filesystem):
    """Call only after the supervised child has exited; retain evidence on error."""
    output_root = Path(output_root).resolve(strict=True)
    local_root = Path(local_root).resolve(strict=True)
    reports = []
    for location in sorted(output_root.glob('phase_*/log-staging-location.json')):
        item = {'location': str(location), 'recovered': False}
        try:
            if location.is_symlink() or location.parent.is_symlink():
                raise ValueError('Symlink location forbidden')
            data = json.loads(location.read_text())
            if data['schema'] != 'precision-log-location-v1':
                raise ValueError('Unknown location schema')
            root = Path(data['local_directory'])
            if root.is_symlink() or root.resolve().parent != local_root:
                raise ValueError('Log directory outside private wrapper root')
            if Path(data['output_directory']).resolve() != location.parent.resolve():
                raise ValueError('Output location mismatch')
            if data['filesystem'] != filesystem:
                raise ValueError('Filesystem provenance mismatch')
            stage = LogStage.__new__(LogStage)
            stage.root = root.resolve(strict=True); stage.output = location.parent.resolve()
            stage.filesystem = filesystem; stage.published = False
            item['publication'] = stage.publish(loggers_disabled=True, require_complete=False)
            stage.cleanup()
            item['recovered'] = True
        except Exception as error:
            item['error'] = repr(error)
        reports.append(item)
    leftovers = sorted(p.name for p in local_root.iterdir())
    return {'schema': 'precision-log-recovery-v1', 'locations': reports,
            'leftover_local_entries': leftovers,
            'recovery_complete': all(r['recovered'] for r in reports) and not leftovers,
            'note': 'Process-exit recovery preserves available bytes, not buffered bytes lost on forced termination. It does not mark an incomplete experiment successful.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=float, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Allocated compute required')
    if not 0 < args.timeout_seconds <= 1620:
        parser.error('Child deadline must leave at least180seconds in a30minute allocation')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command: parser.error('Explicit child command required')
    fs = filesystem_for(args.local_root, Path('/proc/self/mountinfo').read_text())
    output = args.output_root.resolve(strict=True)
    if not output.is_dir() or (output/'log-wrapper-start.json').exists():
        parser.error('Fresh existing output directory required')
    local = Path(tempfile.mkdtemp(prefix='rapidtinyrec-allocation-', dir=args.local_root)).resolve()
    env = dict(os.environ, RAPIDTINYREC_LOG_STAGE_WRAPPER=str(local))
    # The explicit placeholder avoids appending benchmark-specific flags to shell scripts.
    command = [str(local) if a == '{LOG_LOCAL_ROOT}' else a for a in command]
    save_record(output/'log-wrapper-start.json', {'schema': 'precision-log-wrapper-v1',
        'job_id': os.environ['SLURM_JOB_ID'], 'local_directory': str(local),
        'filesystem': fs, 'command': command, 'child_timeout_s': args.timeout_seconds})
    report = {}
    try:
        report['child'] = supervise(command, env, args.timeout_seconds)
    except BaseException as error:
        report['supervisor_error'] = repr(error)
        raise
    finally:
        report['recovery'] = recover_logs(output, local, fs)
        if report['recovery']['recovery_complete']: local.rmdir()
        save_record(output/'log-wrapper-exit.json', report)
    if (not report['child']['completed_normally'] or not report['recovery']['recovery_complete']
            or not report['recovery']['locations']):
        raise SystemExit(1)


if __name__ == '__main__': main()
