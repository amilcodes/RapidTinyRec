"""Whole-run watchdog for one owned process group, including final audit.

The worker and its compute child MUST remain in this group. No nested sessions
are allowed by the reviewed entry source. SIGKILL of this watchdog itself is
outside Python cleanup guarantees; Slurm remains the allocation-level backstop.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from allocation_contract import COMPUTE_SECONDS,GLOBAL_SECONDS


def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def logs(directory,names):
    return {n:dict(bytes=(Path(directory)/n).stat().st_size,sha256=hashlib.sha256((Path(directory)/n).read_bytes()).hexdigest()) for n in names}


def interrupt(signum,frame):
    raise InterruptedError('Whole-run interrupt '+str(signum))


def checked_budget(value,maximum):
    if type(value) not in (int,float) or not math.isfinite(value) or not 0<value<=maximum:
        raise ValueError('Invalid bounded duration')
    return value


def kill_group(pid):
    try:os.killpg(pid,signal.SIGKILL);return True
    except ProcessLookupError:return False


def run_whole(command,cwd,env,output,*,limit=GLOBAL_SECONDS,started=None):
    # The monotonic deadline begins before output creation, log opening or spawn.
    now=time.monotonic()
    if started is None:started=now
    if type(started) not in (int,float) or not math.isfinite(started) or not 0<=started<=now:raise ValueError('Original start clock invalid')
    deadline=started+checked_budget(limit,GLOBAL_SECONDS)
    if now>=deadline:raise TimeoutError('Original whole-run budget exhausted before spawn')
    output=Path(output);output.mkdir(exist_ok=False)
    child=None;reason='exception';error=None;remaining=False;closed=False;cleanup_done=False
    previous={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    for s in previous:signal.signal(s,interrupt)
    try:
        with (output/'whole.stdout').open('xb') as stdout,(output/'whole.stderr').open('xb') as stderr:
            child_env=dict(env);child_env['OWNED_BLOCK_GLOBAL_STARTED']=str(started)
            child_env['OWNED_BLOCK_GLOBAL_DEADLINE']=str(deadline)
            # Child group is known from its PID; it and all ordinary descendants
            # are covered by this one watchdog, including a terminated worker.
            child=subprocess.Popen(command,cwd=cwd,env=child_env,stdout=stdout,stderr=stderr,start_new_session=True)
            try:
                child.wait(timeout=max(.001,deadline-time.monotonic()));reason='exited'
            except subprocess.TimeoutExpired:reason='global_deadline'
            finally:
                remaining=kill_group(child.pid)
                child.wait(timeout=1);cleanup_done=True
        closed=True
    except BaseException as exc:
        error=repr(exc)
        if isinstance(exc,InterruptedError):reason='interrupted'
    finally:
        # Block further catchable termination while reaping the owned group.
        for s in previous:signal.signal(s,signal.SIG_IGN)
        if child is not None and not cleanup_done:
            try:
                remaining=kill_group(child.pid) or remaining
                child.wait(timeout=1);cleanup_done=True
            except (OSError,subprocess.TimeoutExpired) as exc:
                error=(error or '')+'; group cleanup failed: '+repr(exc)
        for s,handler in previous.items():signal.signal(s,handler)
    ended=time.monotonic()
    # Hashes represent available closed bytes. A killed run cannot certify
    # buffered application output or successful publication.
    files=logs(output,('whole.stdout','whole.stderr'))
    receipt=dict(schema='owned-block-whole-exit-v1',command=list(command),pid=None if child is None else child.pid,
                 returncode=None if child is None else child.returncode,reason=reason,error=error,
                 started_monotonic=started,ended_monotonic=ended,global_deadline=deadline,
                 remaining_group_terminated=remaining,logs=files,
                 completed_normally=closed and cleanup_done and error is None and reason=='exited' and child.returncode==0 and not remaining and ended<=deadline,
                 execution_certified=False,submission_ready=False)
    # Receipt closure itself must also fit the original window.
    receipt['publication_monotonic']=time.monotonic()
    receipt['completed_normally']=receipt['completed_normally'] and receipt['publication_monotonic']<=deadline
    save(output/'whole-exit.json',receipt)
    return receipt


def run_member(command,cwd,env,output,*,limit=COMPUTE_SECONDS):
    """Compute member of the current isolated whole-run group, never setsid.

Only the outer watchdog can certify final group closure. On member timeout or
exception this worker must exit; the watchdog then kills any remaining members.
"""
    began=time.monotonic();checked_budget(limit,COMPUTE_SECONDS)
    started=float(env['OWNED_BLOCK_GLOBAL_STARTED']);glob=float(env['OWNED_BLOCK_GLOBAL_DEADLINE'])
    if not all(math.isfinite(x) for x in (started,glob)) or not 0<=started<=began<glob or glob-started>GLOBAL_SECONDS:
        raise ValueError('Missing/invalid original whole-run clock')
    if os.getsid(0)!=os.getpgrp() or os.getpid()!=os.getpgrp():
        raise ValueError('Compute supervisor must lead the owned isolated group')
    output=Path(output);child=None;reason='exited'
    with (output/'child.stdout').open('xb') as stdout,(output/'child.stderr').open('xb') as stderr:
        child=subprocess.Popen(command,cwd=cwd,env=dict(env),stdout=stdout,stderr=stderr,start_new_session=False)
        try:child.wait(timeout=max(.001,min(began+limit,glob)-time.monotonic()))
        except subprocess.TimeoutExpired:
            reason='compute_deadline';child.kill();child.wait(timeout=1)
        except BaseException:
            child.kill();child.wait(timeout=1);raise
    ended=time.monotonic()
    receipt=dict(schema='owned-block-member-exit-v1',command=list(command),pid=child.pid,group_id=os.getpgrp(),
                 returncode=child.returncode,reason=reason,started_monotonic=began,ended_monotonic=ended,
                 compute_deadline=began+limit,global_deadline=glob,group_closure_deferred_to_outer=True,
                 completed_normally=reason=='exited' and child.returncode==0 and ended<=min(began+limit,glob),
                 logs=logs(output,('child.stdout','child.stderr')),execution_certified=False)
    save(output/'member-exit.json',receipt)
    return receipt
