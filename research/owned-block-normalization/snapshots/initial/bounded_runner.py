"""Bound one fresh child and hash closed logs; outer540s wrapper still required."""
import hashlib
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from allocation_contract import COMPUTE_SECONDS,GLOBAL_SECONDS


def remaining(start,now,compute_started,compute_limit=COMPUTE_SECONDS,global_limit=GLOBAL_SECONDS):
    values=(start,now,compute_started,compute_limit,global_limit)
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in values):raise ValueError('Finite clocks/budgets required')
    if not 0<=start<=compute_started<=now or not 0<compute_limit<=COMPUTE_SECONDS or not 0<global_limit<=GLOBAL_SECONDS:
        raise ValueError('Clock order or maximum budget differs')
    return max(0,min(compute_started+compute_limit,start+global_limit)-now)


def supervise(command,cwd,env,output,started,*,compute_limit=COMPUTE_SECONDS,global_limit=GLOBAL_SECONDS):
    output=Path(output);began=time.monotonic();budget=remaining(started,began,began,compute_limit,global_limit)
    if budget<=0:raise TimeoutError('No original budget remains')
    child=None;timed_out=False;remaining_group_terminated=False
    with (output/'child.stdout').open('xb') as stdout,(output/'child.stderr').open('xb') as stderr:
        child=subprocess.Popen(command,cwd=cwd,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
        try:
            wait=remaining(started,time.monotonic(),began,compute_limit,global_limit)
            if wait<=0:raise subprocess.TimeoutExpired(command,wait)
            child.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            timed_out=True
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=max(.001,min(5,started+global_limit-time.monotonic())))
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait(timeout=1)
        except BaseException:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait(timeout=1);raise
    # A child exit alone does not prove its newly created process group ended.
    # Terminate any remaining descendants before treating logs as closed.
    try:
        os.killpg(child.pid,signal.SIGKILL);remaining_group_terminated=True
    except ProcessLookupError:pass
    ended=time.monotonic()
    files={name:dict(bytes=(output/name).stat().st_size,sha256=hashlib.sha256((output/name).read_bytes()).hexdigest()) for name in ('child.stdout','child.stderr')}
    return dict(schema='owned-block-child-exit-v1',command=list(command),pid=child.pid,returncode=child.returncode,
                reason='deadline' if timed_out else 'exited',started_monotonic=began,ended_monotonic=ended,
                compute_deadline=began+compute_limit,global_deadline=started+global_limit,
                remaining_group_terminated=remaining_group_terminated,
                completed_normally=not timed_out and not remaining_group_terminated and child.returncode==0 and ended<=min(began+compute_limit,started+global_limit),
                within_global_deadline=ended<=started+global_limit,logs=files)
