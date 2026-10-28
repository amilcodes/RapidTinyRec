"""One allocation, one global deadline: generation, log recovery, then audit.

Only private child process groups are terminated. Forced termination, supervisor
KILL or node loss can lose buffered evidence; no recovery guarantee is implied.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

from copy import deepcopy
import subprocess
from .precision_allocation import read,digest,scheduler_fields,validate_scheduler,verify_source
from .bf16_full500 import PROTOCOL, PROTOCOL_SHA256, load_protocol
from .precision_log_runner import supervise, recover_logs
from .precision_log_staging import filesystem_for


PROJECT='rapidtinyrec-bf16-full500'
PROFILE='bf16-vs-tf32-full500-quality'
COMMAND=['bash','jobs/bf16-full500.sh']

def observe(run,code,out):
    assert code==run/'code' and out==run/'outputs'
    assert str(run)==os.environ['OSCAR_RUN_DIR'] and str(out)==os.environ['OSCAR_OUTPUT_DIR']
    assert not (run/'source-manifest.json').is_symlink() and not (run/'job-id').is_symlink()
    p=read(run/'source-manifest.json');job=(run/'job-id').read_text().strip()
    assert p['submission']=='prepared' and p['job_id'] is None
    assert len(p['files'])==len(set(p['files'])) and set(p['files'])==set(p['sha256'])
    assert job.isdecimal() and job==os.environ['SLURM_JOB_ID']
    assert p['project']==PROJECT and p['profile']==PROFILE
    assert p['remote_dir']==str(run)==f"/oscar/scratch/aagraw73/agent-runs/{PROJECT}/{p['run_id']}"
    protocol=load_protocol(code/PROTOCOL)
    resources=dict(protocol['resources'],setup=['module purge'],command=COMMAND)
    assert p['resources']==resources
    config=read(code/'oscar.bf16-full500.json')
    assert config['project']==PROJECT and config['profiles']=={PROFILE:resources}
    assert p['sha256'][PROTOCOL]==PROTOCOL_SHA256
    assert p['ssh_target']==config['ssh_target']=='aagraw73@sshcampus.ccv.brown.edu'
    verify_source(code,p)
    raw=subprocess.run(['scontrol','show','job','-o',job],text=True,capture_output=True,check=True,timeout=10).stdout
    fields=scheduler_fields(raw);validate_scheduler(fields,job,str(run))
    record=deepcopy(p);record.update(job_id=job,submission='allocated_observation',allocation_identity=dict(
        schema='precision-allocated-identity-v1',uploaded_manifest_sha256=digest(run/'source-manifest.json'),
        sbatch_job_id_file_sha256=digest(run/'job-id'),scheduler_command=['scontrol','show','job','-o',job],
        scheduler_stdout=raw,scheduler_fields=fields,requires_local_oscar_reconciliation=True))
    return record

def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def stage_allowance(name, started, now):
    assert all(type(x) in (float,int) and math.isfinite(x) for x in (started,now))
    assert now>=started
    remaining=1620-(now-started)
    return max(0,min({'generation':1500,'recovery':1620,'audit':120}[name],remaining))


def run_stages(output, started, commands, env, *, clock=time.monotonic, launch=supervise):
    """CPU-testable orchestration; all production commands are fixed by main."""
    stages=[];report=dict(schema='bf16-full500-work-exit-v1',stages=stages,completed_normally=False)
    generation=None;recovery=None
    try:
        for name in ('generation','recovery','audit'):
            if name=='audit' and not (generation['completed_normally'] and recovery['recovery_complete']
                    and all(s['within_deadline'] for s in stages)):
                report['audit_skipped']='Generation or recovery did not complete normally';break
            now=clock();allowance=stage_allowance(name,started,now)
            if allowance<=0:
                report['deadline_exhausted_before']=name;break
            entry=dict(name=name,started_monotonic=now,timeout_s=allowance,command=commands[name])
            save(output/f'bf16-full500-{name}-start.json',entry)
            # Record publication can block. It must consume this stage's budget,
            # rather than silently shifting the child deadline forward.
            launch_allowance=max(0,allowance-(clock()-now))
            if launch_allowance>0:
                child=launch(commands[name],env,launch_allowance)
            else:
                child=dict(completed_normally=False,reason='deadline_before_launch',returncode=None)
            entry.update(ended_monotonic=clock(),child=child);stages.append(entry)
            entry['within_deadline']=entry['ended_monotonic']-now<=allowance
            save(output/f'bf16-full500-{name}-exit.json',entry)
            if name=='generation':generation=child
            elif name=='recovery':
                recovery=read(output/'log-recovery.json') if child['completed_normally'] else {'recovery_complete':False,'locations':[]}
                save(output/'log-wrapper-exit.json',dict(child=generation,recovery=recovery))
                if not child['completed_normally']:break
        audit_path=output/'bf16-full500-audit.json'
        if audit_path.exists():report['audit_sha256']=digest(audit_path)
        report['ended_monotonic']=clock()
        report['elapsed_s']=report['ended_monotonic']-started
        report['completed_normally']=(len(stages)==3 and all(s['child']['completed_normally'] and s['within_deadline'] for s in stages)
            and report['elapsed_s']<=1620 and recovery['recovery_complete'])
        if report['completed_normally']:
            assert read(audit_path)['execution_audit_passed'] is True
        return report
    except BaseException as error:
        report.update(error=repr(error),traceback=traceback.format_exc(),completed_normally=False,
            ended_monotonic=clock(),elapsed_s=clock()-started)
        raise
    finally:
        save(output/'bf16-full500-work-exit.json',report)


def validate_audit_start(root, record, *, now=None):
    start=read(root/'bf16-full500-work-start.json');audit=read(root/'bf16-full500-audit-start.json')
    assert start['schema']=='bf16-full500-work-start-v1'
    assert start['job_id']==record['job_id'] and start['protocol_sha256']==PROTOCOL_SHA256
    assert start['global_work_deadline_s']==1620
    assert start['deadline_monotonic']==start['started_monotonic']+1620
    previous=start['started_monotonic']
    for name in ('generation','recovery'):
        entry=read(root/f'bf16-full500-{name}-exit.json')
        assert entry['name']==name and entry['started_monotonic']>=previous
        assert entry['ended_monotonic']>=entry['started_monotonic']
        assert entry['child']['completed_normally'] and entry['child']['reason']=='exited' and entry['child']['returncode']==0
        assert entry['within_deadline'] is True
        assert entry['timeout_s']==stage_allowance(name,start['started_monotonic'],entry['started_monotonic'])>0
        assert entry['ended_monotonic']-entry['started_monotonic']<=entry['timeout_s']
        previous=entry['ended_monotonic']
    assert audit['name']=='audit' and audit['started_monotonic']>=previous
    assert audit['timeout_s']==stage_allowance('audit',start['started_monotonic'],audit['started_monotonic'])>0
    if now is not None:assert audit['started_monotonic']<=now<min(start['deadline_monotonic'],audit['started_monotonic']+audit['timeout_s'])
    assert read(root/'log-recovery.json')['recovery_complete'] is True
    return start,audit


def reconcile_work(root, record):
    """Verify final stage order and result bytes locally, without resampling."""
    root=Path(root);start,audit=validate_audit_start(root,record)
    finish=read(root/'bf16-full500-work-exit.json');end=read(root/'bf16-full500-audit-exit.json')
    assert finish['schema']=='bf16-full500-work-exit-v1' and finish['completed_normally'] is True
    assert [s['name'] for s in finish['stages']]==['generation','recovery','audit']
    for entry in finish['stages']:assert entry==read(root/f"bf16-full500-{entry['name']}-exit.json")
    assert {k:end[k] for k in audit}==audit
    assert end['child']['completed_normally'] and end['child']['reason']=='exited' and end['child']['returncode']==0
    assert end['within_deadline'] is True
    assert 0<=end['ended_monotonic']-end['started_monotonic']<=audit['timeout_s']
    assert end['ended_monotonic']<=finish['ended_monotonic']<=start['deadline_monotonic']
    assert finish['elapsed_s']==finish['ended_monotonic']-start['started_monotonic']
    assert finish['audit_sha256']==digest(root/'bf16-full500-audit.json')
    result=read(root/'bf16-full500-audit.json')
    assert result['schema']=='bf16-vs-tf32-full500-audit-v1' and result['protocol_sha256']==PROTOCOL_SHA256
    assert result['job_id']==record['job_id'] and result['run_id']==record['run_id']
    assert result['allocated_record_sha256']==digest(root/'allocated-run-record.json')
    assert result['execution_audit_passed'] is True
    return dict(passed=True,elapsed_s=finish['elapsed_s'],audit_sha256=finish['audit_sha256'],bootstrap_executed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('run','recover'))
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--local-root',type=Path,default=Path('/tmp'))
    parser.add_argument('--historical',type=Path)
    parser.add_argument('--release-root',type=Path)
    args=parser.parse_args();started=time.monotonic()
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Allocated compute required')
    run=args.run_root.resolve(strict=True);output=run/'outputs';code=run/'code'
    assert str(run)==os.environ.get('OSCAR_RUN_DIR') and str(output)==os.environ.get('OSCAR_OUTPUT_DIR')
    if args.action=='recover':
        start=read(output/'log-wrapper-start.json');local=Path(start['local_directory'])
        report=recover_logs(output,local,start['filesystem']);save(output/'log-recovery.json',report)
        if report['recovery_complete']:local.rmdir()
        if not report['recovery_complete'] or len(report['locations'])!=4:raise SystemExit(1)
        return
    assert args.historical and args.release_root
    save(output/'bf16-full500-work-start.json',dict(schema='bf16-full500-work-start-v1',
        started_monotonic=started,deadline_monotonic=started+1620,global_work_deadline_s=1620,
        job_id=os.environ['SLURM_JOB_ID'],protocol_sha256=PROTOCOL_SHA256))
    try:
        record=observe(run,code,output)
        protocol=load_protocol(code/PROTOCOL)
        assert [protocol['budget'][k] for k in ('global_work_deadline_s','generation_child_deadline_s','audit_maximum_s')]==[1620,1500,120]
        save(output/'allocated-run-record.json',record)
        # Fail before generation on missing/mismatched catalog or held-out data.
        # This parses metadata only; the paired bootstrap runs later on allocation.
        from .audit_bf16_full500 import catalog_preflight
        preflight=catalog_preflight(args.release_root,read(args.historical/'manifest.json')['users'],protocol)
        save(output/'bf16-full500-catalog-preflight.json',preflight)
        for name in ('source-manifest.json','job-id'):
            with (output/('submitted-'+name)).open('xb') as f:f.write((run/name).read_bytes())
        fs=filesystem_for(args.local_root,Path('/proc/self/mountinfo').read_text())
        local=Path(tempfile.mkdtemp(prefix='rapidtinyrec-bf16-full500-',dir=args.local_root)).resolve()
        env=dict(os.environ,RAPIDTINYREC_LOG_STAGE_WRAPPER=str(local),PYTHONDONTWRITEBYTECODE='1')
        commands={'generation':['bash',str(code/'jobs/bf16-full500.sh'),'--child',str(local)],
            'recovery':[sys.executable,'-u','-m','rapidtinyrec.bf16_full500_runner','recover','--run-root',str(run)],
            'audit':[sys.executable,'-u','-m','rapidtinyrec.audit_bf16_full500','allocated',
                '--root',str(output),'--historical',str(args.historical),'--snapshot',str(code/'rapidtinyrec'),
                '--run-root',str(run),'--release-root',str(args.release_root),'--output',str(output/'bf16-full500-audit.json')]}
        save(output/'log-wrapper-start.json',dict(schema='precision-log-wrapper-v1',job_id=record['job_id'],
            local_directory=str(local),filesystem=fs,command=commands['generation'],child_timeout_s=1500))
        report=run_stages(output,started,commands,env)
        if not report['completed_normally']:raise SystemExit(1)
    except BaseException as error:
        save(output/'bf16-full500-controller-failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
            elapsed_s=time.monotonic()-started,promotion=False,noninferiority_established=False))
        raise


if __name__=='__main__':main()
