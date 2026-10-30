"""Bounded allocated BF16 pilot controller with distinct run and audit identity."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
from .precision_allocation import read,digest,scheduler_fields,validate_scheduler,verify_source
from .precision_log_runner import supervise,recover_logs
from .precision_log_staging import filesystem_for
from .bf16_protocol import PROTOCOL,PROTOCOL_SHA256,load_protocol

PROJECT='rapidtinyrec-bf16-pilot'
PROFILE='bf16-vs-tf32-pilot'
COMMAND=['bash','jobs/bf16-pilot.sh']

def save(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')

def observe(run,code,out):
    assert code==run/'code' and out==run/'outputs'
    assert str(run)==os.environ['OSCAR_RUN_DIR'] and str(out)==os.environ['OSCAR_OUTPUT_DIR']
    p=read(run/'source-manifest.json');job=(run/'job-id').read_text().strip()
    assert p['submission']=='prepared' and p['job_id'] is None
    assert job.isdecimal() and job==os.environ['SLURM_JOB_ID']
    assert p['project']==PROJECT and p['profile']==PROFILE
    assert p['remote_dir']==str(run)==f"/oscar/scratch/aagraw73/agent-runs/{PROJECT}/{p['run_id']}"
    protocol=load_protocol(code/PROTOCOL)
    resources=dict(protocol['resources'],setup=['module purge'],command=COMMAND)
    assert p['resources']==resources
    config=read(code/'oscar.bf16-pilot.json')
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

def allowance(name,started,now):
    return max(0,min(dict(generation=1500,recovery=1620,audit=120)[name],1620-(now-started)))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=('run','recover'))
    ap.add_argument('--local-root',type=Path)
    args=ap.parse_args()
    assert os.environ.get('SLURM_JOB_ID')
    run=Path(os.environ['OSCAR_RUN_DIR']).resolve();out=run/'outputs';code=run/'code'
    if args.action=='recover':
        start=read(out/'log-wrapper-start.json')
        recovered=recover_logs(out,args.local_root,start['filesystem'])
        save(out/'log-recovery.json',recovered)
        if recovered['recovery_complete']:args.local_root.rmdir()
        raise SystemExit(0 if recovered['recovery_complete'] else 1)
    started=time.monotonic();stages=[];report=dict(schema='bf16-pilot-work-v1',completed_normally=False,
        started_monotonic=started,deadline_monotonic=started+1620,stages=stages)
    try:
        allocated=observe(run,code,out);save(out/'allocated-run-record.json',allocated)
        save(out/'bf16-work-start.json',dict(report,job_id=allocated['job_id'],protocol_sha256=PROTOCOL_SHA256))
        local=Path(tempfile.mkdtemp(prefix='rapidtinyrec-bf16-',dir='/tmp'))
        fs=filesystem_for(local,Path('/proc/self/mountinfo').read_text())
        env=dict(os.environ,RAPIDTINYREC_LOG_STAGE_WRAPPER=str(local),PYTHONDONTWRITEBYTECODE='1')
        commands=dict(generation=['bash',str(code/COMMAND[1]),'--child',str(local)],
            recovery=[sys.executable,'-m','rapidtinyrec.bf16_runner','recover','--local-root',str(local)],
            audit=[sys.executable,'-m','rapidtinyrec.audit_bf16','--root',str(out),
                '--historical','/oscar/scratch/aagraw73/agent-runs/minionerec-shared-prefill/benchmark-gqa-fp32-20261023T054059Z-7b4f0ee334cb/outputs/gqa_500_users',
                '--snapshot',str(code/'rapidtinyrec'),'--run-record',str(out/'allocated-run-record.json'),
                '--release','/oscar/scratch/aagraw73/minionerec-shared-prefill/release','--output',str(out/'bf16-audit.json')])
        save(out/'log-wrapper-start.json',dict(schema='precision-log-wrapper-v1',job_id=allocated['job_id'],
            local_directory=str(local),filesystem=fs,command=commands['generation'],child_timeout_s=1500))
        for name in ('generation','recovery','audit'):
            now=time.monotonic();budget=allowance(name,started,now)
            if budget<=0:report['deadline_before']=name;break
            if name=='audit' and not all(s['child']['completed_normally'] and s['within_deadline'] for s in stages):
                report['audit_skipped']='Generation or recovery failed';break
            entry=dict(name=name,started_monotonic=now,timeout_s=budget,command=commands[name])
            save(out/f'bf16-{name}-start.json',entry)
            remaining=budget-(time.monotonic()-now)
            child=supervise(commands[name],env,remaining) if remaining>0 else dict(completed_normally=False,reason='deadline_before_launch')
            entry.update(child=child,ended_monotonic=time.monotonic());entry['within_deadline']=entry['ended_monotonic']-now<=budget
            stages.append(entry);save(out/f'bf16-{name}-exit.json',entry)
            if name=='recovery':
                recovery=read(out/'log-recovery.json') if (out/'log-recovery.json').exists() else dict(recovery_complete=False)
                save(out/'log-wrapper-exit.json',dict(child=stages[0]['child'],recovery=recovery))
        report['completed_normally']=len(stages)==3 and all(s['child']['completed_normally'] and s['within_deadline'] for s in stages)
        if report['completed_normally']:assert read(out/'bf16-audit.json')['execution_audit_passed']
    except BaseException as exc:
        report.update(error=repr(exc),traceback=traceback.format_exc(),completed_normally=False)
    finally:
        report.update(ended_monotonic=time.monotonic(),elapsed_s=time.monotonic()-started)
        if report['elapsed_s']>1620:report['completed_normally']=False
        if (out/'bf16-audit.json').exists():report['audit_sha256']=digest(out/'bf16-audit.json')
        save(out/'bf16-work-exit.json',report)
    if not report['completed_normally']:raise SystemExit(1)

if __name__=='__main__':main()
