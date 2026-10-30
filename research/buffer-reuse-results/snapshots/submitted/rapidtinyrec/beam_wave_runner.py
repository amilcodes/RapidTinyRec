"""Bounded allocation controller for the distinct serial-pair/wave experiment."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from .beam_wave_allocation import observe_allocation,read,digest,COMMAND
from .beam_wave_protocol import PROTOCOL_SHA256
from .precision_log_runner import supervise,recover_logs
from .precision_log_staging import filesystem_for

HISTORICAL='/oscar/scratch/aagraw73/agent-runs/minionerec-shared-prefill/benchmark-gqa-fp32-20261023T054059Z-7b4f0ee334cb/outputs/gqa_500_users'
RELEASE='/oscar/scratch/aagraw73/minionerec-shared-prefill/release'


def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def allowance(name,started,now):
    if (any(type(v) not in (int,float) or not math.isfinite(v) for v in (started,now))
            or started<0 or now<started):raise ValueError('Invalid original monotonic work clock')
    return max(0,min(dict(generation=1500,recovery=1620,audit=120)[name],1620-(now-started)))


def commands_for(code,out,local,python):
    return dict(generation=['bash',str(code/COMMAND[1]),'--child',str(local)],
        recovery=[python,'-m','rapidtinyrec.beam_wave_runner','recover','--local-root',str(local)],
        audit=[python,'-m','rapidtinyrec.audit_beam_wave','--root',str(out),'--code',str(code),
               '--historical',HISTORICAL,'--release',RELEASE,'--output',str(out/'beam-wave-audit.json')])


def validate_audit_start(root,record,now):
    from .audit_beam_wave import validate_work_identity
    start=validate_work_identity(read(root/'wave-work-start.json'),record)
    origin=start['started_monotonic'];deadline=start['deadline_monotonic'];previous=origin
    wrapper=read(root/'log-wrapper-start.json')
    commands=commands_for(Path(record['remote_dir'])/'code',Path(record['remote_dir'])/'outputs',
                          Path(wrapper['local_directory']),sys.executable)
    for name in ('generation','recovery'):
        entry=read(root/f'wave-{name}-exit.json');began=entry['started_monotonic'];ended=entry['ended_monotonic']
        validate_stage_identity(entry,name,record,commands[name])
        initial=read(root/f'wave-{name}-start.json')
        if initial!={k:v for k,v in entry.items() if k not in ('child','ended_monotonic','within_deadline')}:
            raise ValueError('Stage start and exit records differ')
        if (any(type(v) not in (int,float) or not math.isfinite(v) for v in (began,ended))
                or not previous<=began<ended<=deadline or entry['timeout_s']!=allowance(name,origin,began)
                or entry['timeout_s']<=0 or ended-began>entry['timeout_s'] or entry['within_deadline'] is not True):
            raise ValueError('Prerequisite stage violated original work deadline or ordering')
        child=entry['child']
        if (child['schema']!='precision-child-exit-v1' or child['completed_normally'] is not True
                or child['returncode']!=0 or child['reason']!='exited' or child['signal'] is not None
                or not 0<child['elapsed_s']<=ended-began):raise ValueError('Prerequisite child failed')
        previous=ended
    audit=read(root/'wave-audit-start.json');validate_stage_identity(audit,'audit',record,commands['audit'])
    began=audit['started_monotonic'];budget=allowance('audit',origin,began)
    if (not previous<=began or audit['timeout_s']!=budget or budget<=0
            or type(now) not in (int,float) or not math.isfinite(now) or not began<=now<began+budget<=deadline):
        raise ValueError('Audit exceeds its own or original inclusive deadline')
    return dict(start=start,audit_start=audit,effective_deadline_monotonic=began+budget)


def validate_stage_identity(entry,name,record,command):
    if (entry['schema']!='two-request-beam-wave-stage-v1' or entry['name']!=name
            or entry['job_id']!=record['job_id'] or entry['run_id']!=record['run_id']
            or entry['protocol_sha256']!=PROTOCOL_SHA256 or entry['command']!=command):
        raise ValueError('Original stage identity or executable command differs')


def run_stages(out,allocated,commands,env,started,*,supervisor=supervise,clock=time.monotonic):
    stages=[];report=dict(stages=stages,completed_normally=False)
    for name in ('generation','recovery','audit'):
        now=clock();budget=allowance(name,started,now)
        if budget<=0:report['deadline_before']=name;break
        if name=='audit' and not all(s['child']['completed_normally'] and s['within_deadline'] for s in stages):
            report['audit_skipped']='Generation or recovery failed';break
        entry=dict(schema='two-request-beam-wave-stage-v1',name=name,job_id=allocated['job_id'],
                   run_id=allocated['run_id'],protocol_sha256=PROTOCOL_SHA256,
                   started_monotonic=now,timeout_s=budget,command=commands[name])
        save(out/f'wave-{name}-start.json',entry)
        try:
            remaining=budget-(clock()-now)
            if remaining<=0:raise TimeoutError('Deadline before child launch')
            child=supervisor(commands[name],env,remaining)
        except BaseException as exc:
            child=dict(schema='precision-child-exit-v1',completed_normally=False,reason='supervisor_error',
                       error=repr(exc),traceback=traceback.format_exc())
        entry.update(child=child,ended_monotonic=clock());entry['within_deadline']=entry['ended_monotonic']-now<=budget
        stages.append(entry);save(out/f'wave-{name}-exit.json',entry)
        if name=='recovery':
            recovery=read(out/'log-recovery.json') if (out/'log-recovery.json').exists() else dict(recovery_complete=False)
            save(out/'log-wrapper-exit.json',dict(child=stages[0]['child'],recovery=recovery))
    report['completed_normally']=len(stages)==3 and all(s['child']['completed_normally'] and s['within_deadline'] for s in stages)
    if report['completed_normally']:
        report['completed_normally']=read(out/'beam-wave-audit.json')['execution_audit_passed'] is True
    return report


def recover(out,local,allocated):
    start=read(out/'log-wrapper-start.json');generation=read(out/'wave-generation-exit.json')
    if (str(start['job_id'])!=allocated['job_id'] or Path(start['local_directory']).resolve()!=local.resolve()
            or generation['job_id']!=allocated['job_id'] or generation['run_id']!=allocated['run_id']):
        raise ValueError('Recovery must follow the original stopped child in this allocation')
    result=recover_logs(out,local,start['filesystem']);save(out/'log-recovery.json',result)
    if result['recovery_complete']:local.rmdir()
    return result['recovery_complete']


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=('run','recover'))
    ap.add_argument('--local-root',type=Path);args=ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise ValueError('Allocated beam-wave controller only')
    run=Path(os.environ['OSCAR_RUN_DIR']).resolve();out=run/'outputs';code=run/'code'
    if args.action=='recover':
        if args.local_root is None:raise ValueError('Original private log directory required')
        allocated=observe_allocation(run,code,out)
        raise SystemExit(0 if recover(out,args.local_root,allocated) else 1)
    if (out/'wave-work-start.json').exists() or (out/'wave-work-exit.json').exists():
        raise ValueError('Preserve previous work; never restart generation')
    started=time.monotonic();report=dict(schema='two-request-beam-wave-work-exit-v1',completed_normally=False,
                                         started_monotonic=started,deadline_monotonic=started+1620)
    try:
        allocated=observe_allocation(run,code,out);save(out/'allocated-run-record.json',allocated)
        report.update(run_id=allocated['run_id'],job_id=allocated['job_id'],protocol_sha256=PROTOCOL_SHA256)
        save(out/'wave-work-start.json',dict(schema='two-request-beam-wave-work-v1',job_id=allocated['job_id'],
             run_id=allocated['run_id'],protocol_sha256=PROTOCOL_SHA256,started_monotonic=started,deadline_monotonic=started+1620))
        local=Path(tempfile.mkdtemp(prefix='rapidtinyrec-beam-wave-',dir='/tmp'))
        fs=filesystem_for(local,Path('/proc/self/mountinfo').read_text())
        commands=commands_for(code,out,local,sys.executable)
        save(out/'log-wrapper-start.json',dict(schema='precision-log-wrapper-v1',job_id=allocated['job_id'],
            local_directory=str(local),filesystem=fs,command=commands['generation'],child_timeout_s=1500))
        env=dict(os.environ,RAPIDTINYREC_LOG_STAGE_WRAPPER=str(local),PYTHONDONTWRITEBYTECODE='1')
        report.update(run_stages(out,allocated,commands,env,started))
    except BaseException as exc:
        report.update(error=repr(exc),traceback=traceback.format_exc(),completed_normally=False)
    finally:
        ended=time.monotonic();report.update(ended_monotonic=ended,elapsed_s=ended-started)
        if ended-started>1620:report['completed_normally']=False
        if (out/'beam-wave-audit.json').exists():report['audit_sha256']=digest(out/'beam-wave-audit.json')
        if not (out/'log-wrapper-exit.json').exists():report['log_recovery_incomplete']=True
        save(out/'wave-work-exit.json',report)
    if not report['completed_normally']:raise SystemExit(1)


if __name__=='__main__':main()
