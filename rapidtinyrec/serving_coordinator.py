"""Four fresh serving phases beneath an already active external supervisor.

This callable requires a separately supplied final on-disk auditor. The
production serving_runner binds it to the pre-exit allocated disk reader.
"""
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from .serving_allocation import digest, identical, observe_allocation, require, verify_source
from .serving_contract import PROTOCOL, PROTOCOL_SHA256, load_protocol
from .serving_initialize import save_exclusive
from .serving_process import process_identity, observe_cache, check_deadline
from .serving_setup_audit import read_unique


def reconcile_worker(launch, observed, entry, exit_record, before, after, worker_before, worker_after):
    """Recompute process/clock/cache binding; never accept an online gate flag."""
    identity = observed['identity']; phase = launch['phase']; parent = launch['parent_identity']
    require(identical(entry['launch'], launch) and identical(entry['identity'], identity) and
            identical(exit_record['identity'], identity), 'Observed/worker process identity differs')
    require(identity['parent_pid'] == parent['pid'] and identity['boot_id'] == parent['boot_id'] and
            identity['hostname'] == parent['hostname'] and identity['clock_ticks'] == parent['clock_ticks'] and
            identity['start_ticks'] >= parent['start_ticks'], 'Worker birth/parent binding differs')
    require(observed['pid'] == identity['pid'] and identity['pid'] != parent['pid'] and
            observed['process_group'] == observed['session'] == entry['process_group'] == entry['session'] ==
            launch['process_group'] == launch['session'] == parent['pid'], 'Worker escaped supervised group/session')
    require(type(entry['phase']) is int and type(exit_record['phase']) is int and entry['phase'] == exit_record['phase'] == phase,
            'Worker phase identity differs')
    require(observed['returncode'] == 0 and type(observed['returncode']) is int and
            observed['error'] is None and
            exit_record['complete'] is True and exit_record['execution_complete'] is True and exit_record['error'] is None,
            'Failed or incomplete phase cannot advance the schedule')
    clocks = [launch['launch_started_ns'], entry['started_ns'], exit_record['service_closed_ns'],
              exit_record['finished_ns'], observed['exit_observed_ns'], launch['deadline_ns']]
    require(all(type(v) is int and v > 0 for v in clocks) and
            all(a <= z for a,z in zip(clocks,clocks[1:])) and clocks[-2] < clocks[-1],
            'Worker/parent shutdown clocks escape the global deadline')
    require(launch['launch_started_ns'] <= observed['observed_ns'] <= observed['exit_observed_ns'],
            'Parent process-observation clock differs')
    require(identical(before,worker_before) and identical(after,worker_after) and bool(after),
            'Parent/worker cache inventory differs or no cache remains')
    return dict(phase=phase,process_bound=True,cache_bound=True,advancement_ready=False)


def coordinate(run_root, release_root, historical, *, final_auditor):
    """Run under serving_runner supervision with its mandatory disk auditor."""
    require(callable(final_auditor), 'Separate complete on-disk run auditor required')
    require(sys.dont_write_bytecode and sys.flags.optimize == 0, 'Coordinator requires -B and normal assertions')
    run = Path(run_root)
    require(run.resolve() == run and str(run) == os.environ.get('OSCAR_RUN_DIR') and
            str(run/'outputs') == os.environ.get('OSCAR_OUTPUT_DIR'), 'Canonical allocated coordinator run required')
    outer = read_unique(run/'outputs/supervision/supervisor-start.json')
    parent = process_identity(); group = os.getpgrp(); session = os.getsid(0)
    require(group == session == parent['pid'] and parent['parent_pid'] == outer['supervisor_pid'],
            'Coordinator must be the externally supervised session leader')
    boundary = dict(global_started_ns=outer['global_started_ns'],deadline_ns=outer['deadline_ns'],
                    launch_started_ns=time.perf_counter_ns())
    check_deadline(boundary)
    out = run/'outputs/coordinator';out.mkdir(exist_ok=False)
    save_exclusive(out/'entry.json', dict(identity=parent,process_group=group,session=session,
        started_ns=time.perf_counter_ns(),supervisor_start=outer))
    failure = None; finished = []; last_cache = {}; birth_ids = set(); audit_published_ns = None; audit_sha256 = None
    try:
        allocation = observe_allocation(run)
        save_exclusive(out/'allocation-observation.json',allocation)
        protocol = load_protocol(run/'code'/PROTOCOL)
        check_deadline(boundary)
        (run/'outputs/launch').mkdir(exist_ok=False)
        (run/'cache').mkdir(exist_ok=False)
        for spec in protocol['phases']:
            phase = spec['phase']; family = spec['cache_family']; cache = run/'cache'/family
            check_deadline(boundary)
            if not spec['reuse_disk_cache']:cache.mkdir(exist_ok=False)
            before = observe_cache(cache)
            require(identical(before,last_cache[family]) if spec['reuse_disk_cache'] else not before,
                    'Previous same-family cache changed before reuse')
            save_exclusive(out/f'phase_{phase}-cache-before.json',before)
            launch = dict(schema='serving-phase-launch-v1',run_root=str(run),phase=phase,
                protocol_sha256=PROTOCOL_SHA256,launch_token=uuid.uuid4().hex,parent_identity=parent,
                global_started_ns=boundary['global_started_ns'],deadline_ns=boundary['deadline_ns'],
                launch_started_ns=time.perf_counter_ns(),process_group=group,session=session,
                cache=dict(family=family,root=str(cache),reuse=spec['reuse_disk_cache']))
            save_exclusive(run/'outputs/launch'/f'phase_{phase}.json',launch)
            env=dict(os.environ,SERVING_LAUNCH_TOKEN=launch['launch_token'],
                TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'),CUDA_CACHE_PATH=str(cache/'cuda'))
            command=[sys.executable,'-B','-m','rapidtinyrec.serving_phase_cli','--run-root',str(run),
                     '--phase',str(phase),'--release-root',str(release_root),'--historical',str(historical)]
            child = None; observed = dict(phase=phase,command=command,returncode=None,error=None)
            try:
                with (out/f'phase_{phase}.log').open('xb') as log:
                    child=subprocess.Popen(command,cwd=run/'code',env=env,stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,close_fds=True,start_new_session=False)
                    identity=process_identity(child.pid)
                    observed.update(pid=child.pid,identity=identity,observed_ns=time.perf_counter_ns(),
                                    process_group=os.getpgid(child.pid),session=os.getsid(child.pid))
                    require(observed['process_group'] == observed['session'] == group,'Child process escaped supervisor')
                    birth=(identity['boot_id'],identity['pid'],identity['start_ticks'])
                    require(birth not in birth_ids,'Phase process reused');birth_ids.add(birth)
                    save_exclusive(out/f'phase_{phase}-spawn.json',observed)
                    # The external watchdog kills this coordinator and all
                    # inherited-group workers if even preflight blocks here.
                    os.waitid(os.P_PID,child.pid,os.WEXITED|os.WNOWAIT)
                    observed['exit_observed_ns']=time.perf_counter_ns()
                    observed['returncode']=child.wait()
            except BaseException as error:
                observed['error']=repr(error)
                if child is not None and child.returncode is None:
                    child.kill();observed['returncode']=child.wait(timeout=5)
                raise
            finally:
                save_exclusive(out/f'phase_{phase}-exit.json',observed)
            require(observed['returncode']==0,'Worker failed; later phases prohibited')
            check_deadline(boundary)
            root=run/'outputs'/f'phase_{phase}'
            after=observe_cache(cache);save_exclusive(out/f'phase_{phase}-cache-after.json',after)
            report=reconcile_worker(launch,observed,read_unique(root/'worker-entry.json'),read_unique(root/'worker-exit.json'),
                before,after,read_unique(root/'cache-before.json'),read_unique(root/'cache-after.json'))
            save_exclusive(out/f'phase_{phase}-binding.json',report)
            last_cache[family]=after;finished.append(phase)
            verify_source(run/'code',allocation)
        check_deadline(boundary)
        audit=final_auditor(run,historical)
        require(isinstance(audit,dict) and type(audit.get('evidence_gates_passed')) is bool,
                'Final run auditor must return its reconstructed evidence gates')
        save_exclusive(out/'final-audit.json',audit)
        audit_sha256=digest(out/'final-audit.json');audit_published_ns=time.perf_counter_ns()
        check_deadline(boundary)
        return audit
    except BaseException as error:
        failure=repr(error)
        raise
    finally:
        save_exclusive(out/'exit.json',dict(identity=parent,finished_ns=time.perf_counter_ns(),
            phases_completed=finished,complete=failure is None,error=failure,advancement_ready=False,
            audit_published_ns=audit_published_ns,audit_sha256=audit_sha256))
