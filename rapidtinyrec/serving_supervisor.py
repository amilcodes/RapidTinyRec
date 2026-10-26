"""External process-group deadline primitive for the serving harness.

No Torch, source hashing, Slurm query or model work belongs in this parent. Put
all such work, phases, auditing and recovery in the supervised coordinator.
The coordinator/harness and independent reconciliation are still separate work.
"""
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from .serving_allocation import require
from .serving_initialize import save_exclusive
from .serving_process import process_identity


def _kill_group(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
        return True
    except ProcessLookupError:
        return False


def _run_group(command, cwd, env, output, *, started_ns, deadline_ns, observe_identity=process_identity):
    """POSIX process-group supervision; short deadlines are for CPU tests only.

    waitid(WNOWAIT) keeps the leader PID reserved until the watchdog has stopped
    and its process group is killed, including descendants after leader exit.
    Thus no kill is directed at a recycled leader PID. This function must be
    the sole process waiter. A watchdog thread also covers blocked parent-side
    process observation or evidence writes after the child has been launched.
    """
    require(hasattr(os, 'WNOWAIT') and hasattr(os, 'waitid'), 'POSIX waitid/WNOWAIT required')
    require(type(started_ns) is int and type(deadline_ns) is int and 0 < started_ns < deadline_ns,
            'Positive absolute monotonic deadline required')
    require(isinstance(command, list) and command and all(isinstance(x, str) for x in command),
            'Explicit argument vector required')
    output = Path(output); output.mkdir(exist_ok=False)
    save_exclusive(output/'supervisor-start.json', dict(schema='serving-supervisor-start-v1',
        global_started_ns=started_ns, deadline_ns=deadline_ns, command=command, cwd=str(cwd),
        supervisor_pid=os.getpid(), timing_scope='Preflight through coordinator exit, including all phases/audit/recovery'))
    child = None; ready = threading.Event(); done = threading.Event()
    timeout = {}; failure = None; exit_observed_ns = None; group_cleanup = False

    def watchdog():
        if done.wait(max(0, (deadline_ns-time.perf_counter_ns())/1e9)):
            return
        timeout['deadline_observed_ns'] = time.perf_counter_ns()
        # Popen publishes its child before setting ready. Even if publication is
        # late, the expired deadline cannot turn into a fresh relative timeout.
        ready.wait()
        if child is not None:
            try:
                timeout['signal_sent'] = _kill_group(child.pid)
                timeout['signal_ns'] = time.perf_counter_ns()
            except BaseException as error:
                timeout['signal_error'] = repr(error)

    watch = threading.Thread(target=watchdog, name='serving-deadline', daemon=True)
    watch.start()
    try:
        if time.perf_counter_ns() >= deadline_ns:
            raise TimeoutError('Global work deadline expired before process launch')
        with (output/'coordinator.log').open('xb') as log:
            launched_ns = time.perf_counter_ns()
            child = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, close_fds=True)
            ready.set()
            identity = observe_identity(child.pid)
            save_exclusive(output/'coordinator-launch.json', dict(pid=child.pid,
                process_group=child.pid, launched_ns=launched_ns,
                observed_ns=time.perf_counter_ns(), identity=identity))
            # Do not poll/wait/reap the leader until all group signals finish.
            # WNOHANG also lets a failed watchdog signal become a saved failure,
            # instead of blocking forever in waitid when termination is denied.
            while os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT | os.WNOHANG) is None:
                if timeout.get('signal_error'):
                    raise RuntimeError('Deadline group termination failed: '+timeout['signal_error'])
                if time.perf_counter_ns() > deadline_ns+5*10**9:
                    raise TimeoutError('Group did not exit after the deadline signal')
                time.sleep(.02)
            exit_observed_ns = time.perf_counter_ns()
    except BaseException as error:
        failure = repr(error)
    finally:
        ready.set(); done.set(); watch.join()
        if child is not None:
            # The unreaped leader still reserves this PID/group ID even when
            # it exited early and left compiler or phase descendants behind.
            try:
                group_cleanup = _kill_group(child.pid)
            except BaseException as error:
                failure = failure or repr(error)
            finally:
                # A group-signal error must not leak the leader's zombie or
                # erase its actual exit status. It still fails supervision.
                try:
                    child.wait(timeout=5)
                except BaseException as error:
                    failure = failure or repr(error)
        finished_ns = time.perf_counter_ns()
        expired = bool(timeout) or (exit_observed_ns or finished_ns) >= deadline_ns
        result = dict(schema='serving-supervisor-exit-v1', global_started_ns=started_ns,
            deadline_ns=deadline_ns, exit_observed_ns=exit_observed_ns, finished_ns=finished_ns,
            child_pid=None if child is None else child.pid,
            returncode=None if child is None else child.returncode,
            deadline_exceeded=expired, watchdog=timeout, group_cleanup_signal_sent=group_cleanup,
            complete=bool(child is not None and child.returncode == 0 and not expired and failure is None),
            error=failure, advancement_ready=False,
            final_record_scope='Minimal supervisor termination bookkeeping after work; not additional model/audit work')
        save_exclusive(output/'supervisor-exit.json', result)
    return result


def supervise(command, cwd, env, output, *, global_started_ns):
    """Production ceiling is fixed; the caller starts it before any preflight.

    The future harness must bind command/source/run and put its coordinator in
    this group. A successful process exit is not a successful serving audit.
    """
    return _run_group(command, cwd, env, output, started_ns=global_started_ns,
                      deadline_ns=global_started_ns+1620*10**9)
