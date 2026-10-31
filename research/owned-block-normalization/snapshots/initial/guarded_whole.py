"""Independent deadline/parent-liveness guardian through final publication.

The guardian owns worker creation, so there is no parent-to-guardian registration
race. It stays alive through parent log hashing and receipt publication. This
protects against termination of the parent; it is not fault tolerance against
simultaneous loss of the guardian/host. All members must keep the worker group.
"""
import json
import math
import os
from pathlib import Path
import select
import signal
import subprocess
import time
from allocation_contract import GLOBAL_SECONDS
from whole_run import save,logs,kill_group,checked_budget


def send(fd,record):
    data=(json.dumps(record,allow_nan=False,separators=(',',':'))+'\n').encode()
    while data:
        count=os.write(fd,data);data=data[count:]


def receive(fd,deadline):
    data=b''
    while True:
        left=deadline-time.monotonic()
        if left<=0:raise TimeoutError('Original guardian deadline')
        ready,_,_=select.select([fd],[],[],left)
        if not ready:raise TimeoutError('Original guardian deadline')
        byte=os.read(fd,1)
        if not byte:raise EOFError('Parent/guardian pipe closed')
        data+=byte
        if len(data)>65536:raise ValueError('Guardian message too large')
        if byte==b'\n':return json.loads(data)


def guardian(command,cwd,env,output,started,deadline,parent,requests,replies):
    child=None;cleaned=False;reason='guardian_exception';error=None;closed=None;created=False
    try:
        os.setsid()
        output=Path(output);output.mkdir(exist_ok=False);created=True
        child_env=dict(env,OWNED_BLOCK_GLOBAL_STARTED=str(started),OWNED_BLOCK_GLOBAL_DEADLINE=str(deadline))
        with (output/'whole.stdout').open('xb') as stdout,(output/'whole.stderr').open('xb') as stderr:
            # Parent death before spawn is detected first. Death during spawn
            # is detected by the same pipe; the guardian already owns the PID.
            ready,_,_=select.select([requests],[],[],0)
            if ready and os.read(requests,1)==b'':raise EOFError('Parent exited before spawn')
            if time.monotonic()>=deadline:raise TimeoutError('Deadline before spawn')
            child=subprocess.Popen(command,cwd=cwd,env=child_env,stdout=stdout,stderr=stderr,start_new_session=True)
            send(replies,dict(event='worker_started',guardian_pid=os.getpid(),worker_pid=child.pid))
            while child.poll() is None:
                left=deadline-time.monotonic()
                if left<=0:raise TimeoutError('Whole-run deadline')
                ready,_,_=select.select([requests],[],[],min(.02,left))
                if ready:
                    if os.read(requests,1)==b'':raise EOFError('Parent exited during worker')
                    raise ValueError('Parent completion before worker exit')
            remaining=kill_group(child.pid);child.wait(timeout=1);cleaned=True
        closed=dict(event='worker_closed',pid=child.pid,returncode=child.returncode,
                    remaining_group_terminated=remaining,ended_monotonic=time.monotonic())
        send(replies,closed)
        # The parent can now hash closed logs and publish its receipt, while
        # this separate process still enforces the ORIGINAL deadline.
        done=receive(requests,deadline)
        if done!={'event':'publication_closed'}:raise ValueError('Final publication handshake differs')
        if time.monotonic()>=deadline:raise TimeoutError('Publication deadline')
        reason='publication_closed'
        record=dict(schema='owned-block-guardian-exit-v1',guardian_pid=os.getpid(),parent_pid=parent,
                    worker=closed,started_monotonic=started,global_deadline=deadline,
                    closed_monotonic=time.monotonic(),reason=reason,error=None,
                    execution_certified=False,submission_ready=False)
        save(output/'guardian-exit.json',record)
        if time.monotonic()>=deadline:raise TimeoutError('Guardian receipt deadline')
        send(replies,dict(event='guardian_closed',guardian_pid=os.getpid()))
        return 0
    except BaseException as exc:
        error=repr(exc)
        reason='global_deadline' if isinstance(exc,TimeoutError) else 'parent_exited' if isinstance(exc,(EOFError,BrokenPipeError)) else 'guardian_exception'
        if child is not None and not cleaned:
            try:kill_group(child.pid);child.wait(timeout=1);cleaned=True
            except BaseException as cleanup:error+='; cleanup '+repr(cleanup)
        if reason=='global_deadline':
            try:os.kill(parent,signal.SIGKILL)
            except ProcessLookupError:pass
        # Failure publication is best effort and can never become success.
        try:
            if created:save(Path(output)/'guardian-failure.json',dict(reason=reason,error=error,parent_pid=parent,worker_pid=None if child is None else child.pid,global_deadline=deadline,execution_certified=False))
        except BaseException:pass
        return 1
    finally:
        os.close(requests);os.close(replies)


def run_guarded_whole(command,cwd,env,output,*,limit=GLOBAL_SECONDS,started=None,finalize=None):
    now=time.monotonic()
    if started is None:started=now
    if type(started) not in (int,float) or not math.isfinite(started) or not 0<=started<=now:raise ValueError('Original start clock invalid')
    deadline=started+checked_budget(limit,GLOBAL_SECONDS)
    if now>=deadline:raise TimeoutError('No original guardian budget remains')
    # All pipe descriptors are close-on-exec, so worker descendants cannot keep
    # the parent-liveness pipe open after parent death.
    request_read,request_write=os.pipe();reply_read,reply_write=os.pipe();parent=os.getpid()
    guardian_pid=os.fork()
    if guardian_pid==0:
        os.close(request_write);os.close(reply_read)
        status=guardian(command,cwd,env,output,started,deadline,parent,request_read,reply_write)
        os._exit(status)
    os.close(request_read);os.close(reply_write)
    waited=False
    try:
        beginning=receive(reply_read,deadline)
        if beginning.get('event')!='worker_started' or beginning.get('guardian_pid')!=guardian_pid:raise ValueError('Worker start receipt differs')
        closed=receive(reply_read,deadline)
        if closed.get('event')!='worker_closed' or closed['pid']!=beginning['worker_pid']:raise ValueError('Worker closure differs')
        receipt=dict(schema='owned-block-whole-exit-v2',command=list(command),pid=closed['pid'],guardian_pid=guardian_pid,
                     returncode=closed['returncode'],reason='exited',error=None,started_monotonic=started,
                     ended_monotonic=closed['ended_monotonic'],global_deadline=deadline,
                     remaining_group_terminated=closed['remaining_group_terminated'],logs=logs(output,('whole.stdout','whole.stderr')),
                     completed_normally=closed['returncode']==0 and not closed['remaining_group_terminated'] and closed['ended_monotonic']<=deadline,
                     guardian_closure_required=True,execution_certified=False,submission_ready=False)
        # Optional final metadata consumer executes under guardian protection.
        if finalize is not None:receipt['finalization']=finalize(receipt)
        receipt['publication_monotonic']=time.monotonic()
        receipt['completed_normally']=receipt['completed_normally'] and receipt['publication_monotonic']<=deadline
        save(Path(output)/'whole-exit.json',receipt)
        send(request_write,{'event':'publication_closed'})
        ack=receive(reply_read,deadline)
        if ack!={'event':'guardian_closed','guardian_pid':guardian_pid}:raise ValueError('Guardian closure differs')
        _,status=os.waitpid(guardian_pid,0);waited=True
        if status!=0:raise ValueError('Guardian failed after publication')
        return receipt
    finally:
        os.close(request_write);os.close(reply_read)
        if not waited:
            # Closing the only parent write end tells the independent guardian
            # to kill its owned worker group, even on parent exceptions.
            os.waitpid(guardian_pid,0)
