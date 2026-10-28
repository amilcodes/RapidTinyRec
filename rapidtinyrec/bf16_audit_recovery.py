"""CPU allocation only: audit immutable BF16 pilot outputs without generation."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import time
import traceback

from .precision_allocation import read, scheduler_fields, reconcile_record, verify_source
from .bf16_saved_stages import validate_saved_stages

ORIGINAL_RUN='bf16-vs-tf32-pilot-20261028T040354Z-d98e01fbaca9'
ORIGINAL_JOB='8040768'
ORIGINAL_ROOT=Path('/oscar/scratch/aagraw73/agent-runs/rapidtinyrec-bf16-pilot')/ORIGINAL_RUN
PROJECT='rapidtinyrec-bf16-audit-recovery'
PROFILE='bf16-saved-output-audit-recovery'
PYTHON='/oscar/scratch/aagraw73/minionerec-shared-prefill/envs/torch214-cu126-transformers4554/bin/python'
RESOURCES=dict(partition='batch',gpus=0,cpus=4,memory_gb=32,time='00:30:00',
    setup=['module purge'],command=[PYTHON,'-u','-m','rapidtinyrec.bf16_audit_recovery'])
RELEASE=Path('/oscar/scratch/aagraw73/minionerec-shared-prefill/release')
HISTORICAL=Path('/oscar/scratch/aagraw73/agent-runs/minionerec-shared-prefill/benchmark-gqa-fp32-20261023T054059Z-7b4f0ee334cb/outputs/gqa_500_users')


def sha(path):
    result=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def verify_raw(root,rows):
    """Check complete raw inventory, rejecting alias paths and symlink parents."""
    root=root.resolve(strict=True);names=set()
    for row in rows:
        name=row['path'];rel=PurePosixPath(name)
        assert name==str(rel) and not rel.is_absolute() and '..' not in rel.parts
        assert name not in names,'Duplicate raw path'
        names.add(name)
        path=root/name
        assert not any((root/Path(*rel.parts[:i])).is_symlink() for i in range(1,len(rel.parts)+1))
        assert path.is_file() and path.resolve().is_relative_to(root)
        assert type(row['size']) is int and path.stat().st_size==row['size']
        assert sha(path)==row['sha256'],name
    observed={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and
        (p.relative_to(root).parts[0] in ('outputs','logs') or
         str(p.relative_to(root)) in ('job-id','job.sh','source-manifest.json'))}
    assert observed==names,'Raw inventory changed'


def validate_cpu(prepared,job,env,fields):
    assert prepared['submission']=='prepared' and prepared['job_id'] is None
    assert prepared['project']==PROJECT and prepared['profile']==PROFILE
    assert prepared['resources']==RESOURCES
    assert job.isdecimal() and job!=ORIGINAL_JOB and env.get('SLURM_JOB_ID')==job
    remote=f"/oscar/scratch/aagraw73/agent-runs/{PROJECT}/{prepared['run_id']}"
    assert prepared['remote_dir']==env['OSCAR_RUN_DIR']==remote
    assert env['OSCAR_OUTPUT_DIR']==remote+'/outputs'
    assert fields['JobId']==job and fields['JobState']=='RUNNING'
    assert fields['Partition']=='batch' and fields['NumCPUs']=='4' and fields['NumNodes']=='1'
    assert fields['TimeLimit']=='00:30:00'
    assert fields['WorkDir']==remote and fields['Command']==remote+'/job.sh'
    entries=[x.split('=',1) for x in fields['AllocTRES'].split(',')];tres=dict(entries)
    assert len(tres)==len(entries),'Duplicate allocated resource'
    assert tres['cpu']=='4' and tres['node']=='1' and tres['mem'] in ('32G','32768M')
    assert not any('gpu' in k and v!='0' for k,v in tres.items())
    assert 'gpu' not in fields.get('TresPerNode','').lower()
    assert fields['NodeList'] not in ('','(null)','None')


def timeout(signum,frame):
    raise TimeoutError('BF16 saved-output audit recovery exceeded 1620 seconds')


def main():
    assert os.environ.get('SLURM_JOB_ID'),'Allocated compute required; never local bootstrap'
    started=time.monotonic()
    run=Path(os.environ['OSCAR_RUN_DIR']).resolve();out=Path(os.environ['OSCAR_OUTPUT_DIR']).resolve()
    code=Path(__file__).resolve().parents[1]
    assert code==run/'code' and out==run/'outputs' and run!=ORIGINAL_ROOT
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1620)
    result=dict(schema='bf16-saved-output-recovery-v1',original_run_id=ORIGINAL_RUN,
        original_job_id=ORIGINAL_JOB,recovery_job_id=os.environ['SLURM_JOB_ID'],
        generation_executed=False,recovery_audit_passed=False,promotion=False,noninferiority_established=False)
    inventory=None
    try:
        prepared=read(run/'source-manifest.json');job=(run/'job-id').read_text().strip()
        assert sorted(prepared['files'])==sorted(prepared['sha256'])
        assert len(set(prepared['files']))==len(prepared['files'])
        verify_source(code,prepared)
        command=['scontrol','show','job','-o',job]
        raw=subprocess.run(command,text=True,capture_output=True,check=True,timeout=10).stdout
        fields=scheduler_fields(raw);validate_cpu(prepared,job,os.environ,fields)
        identity=dict(schema='bf16-recovery-allocation-v1',run_id=prepared['run_id'],job_id=job,
            uploaded_manifest_sha256=sha(run/'source-manifest.json'),job_id_file_sha256=sha(run/'job-id'),
            scheduler_command=command,scheduler_stdout=raw,scheduler_fields=fields,
            resources=RESOURCES,source_hashes=prepared['sha256'])
        save(out/'recovery-allocation.json',identity)
        save(out/'recovery-start.json',dict(started_monotonic=started,deadline_monotonic=started+1620))
        result['allocation_identity']=identity
        original=read(code/'original-run-record.json')
        assert original['run_id']==ORIGINAL_RUN and original['job_id']==ORIGINAL_JOB
        assert original['remote_dir']==str(ORIGINAL_ROOT)
        assert len(original['files'])==len(original['sha256'])==69
        original_code=ORIGINAL_ROOT/'code';original_out=ORIGINAL_ROOT/'outputs'
        # verify_source checks imported code; original source is separately hashed
        # because the corrected auditor intentionally belongs to the recovery run.
        for name,want in original['sha256'].items():
            rel=PurePosixPath(name)
            assert not rel.is_absolute() and '..' not in rel.parts
            path=original_code/name
            assert not any((original_code/Path(*rel.parts[:i])).is_symlink() for i in range(1,len(rel.parts)+1))
            assert sha(path)==want,name
        inventory=read(code/'original-raw-manifest.json')
        assert inventory['run_id']==ORIGINAL_RUN and inventory['job_id']==ORIGINAL_JOB
        verify_raw(ORIGINAL_ROOT,inventory['files'])
        allocation=read(original_out/'allocated-run-record.json')
        provenance=reconcile_record(allocation,original,(ORIGINAL_ROOT/'source-manifest.json').read_bytes(),
                                    (ORIGINAL_ROOT/'job-id').read_bytes())
        from .audit_precision_setup import validate_wrapper
        from .audit_bf16 import inspect_phases,evaluate_phases
        wrapper=validate_wrapper(original_out,allocation,phases=(0,1,2,3),child_timeout_s=1500)
        stages=validate_saved_stages(read(original_out/'bf16-work-start.json'),
            read(original_out/'bf16-generation-exit.json'),read(original_out/'bf16-recovery-exit.json'),allocation)
        failed=read(original_out/'bf16-audit.json')
        assert failed['execution_audit_passed'] is False
        assert 'UnboundLocalError' in failed['error'] and 'load_protocol' in failed['error']
        work=read(original_out/'bf16-work-exit.json')
        assert work['completed_normally'] is False
        assert work['audit_sha256']==sha(original_out/'bf16-audit.json')
        assert [s['name'] for s in work['stages']]==['generation','recovery','audit']
        for stage in work['stages']:
            assert stage==read(original_out/f"bf16-{stage['name']}-exit.json")
        result.update(original_provenance=provenance,original_wrapper=wrapper,original_stages=stages,
            original_failed_audit_sha256=work['audit_sha256'],original_failed_audit_preserved=True,
            original_work_exit_sha256=sha(original_out/'bf16-work-exit.json'),
            original_raw_manifest_sha256=sha(code/'original-raw-manifest.json'),
            original_raw_file_count=len(inventory['files']))
        phases=inspect_phases(original_out,HISTORICAL,original_code/'rapidtinyrec',original_out/'allocated-run-record.json')
        audit=evaluate_phases(phases,original_out,original_code/'rapidtinyrec',RELEASE,
                              wrapper=wrapper,allocation_identity=identity)
        assert audit['job_id']==ORIGINAL_JOB and audit['run_id']==ORIGINAL_RUN
        result.update(audit=audit,recovery_audit_passed=audit['execution_audit_passed'])
    except BaseException as exc:
        result.update(error=repr(exc),traceback=traceback.format_exc(),recovery_audit_passed=False)
    finally:
        # Also check immutable inputs following numerical failure; never overwrite
        # original evidence, and never cancel the deadline before this verification.
        if inventory is not None:
            try:
                verify_raw(ORIGINAL_ROOT,inventory['files'])
                result['original_raw_unchanged_after_audit']=True
            except BaseException as exc:
                result.update(raw_verification_error=repr(exc),recovery_audit_passed=False)
        signal.alarm(0)
        result.update(ended_monotonic=time.monotonic(),elapsed_s=time.monotonic()-started)
        if result['elapsed_s']>1620:result['recovery_audit_passed']=False
        save(out/'recovery-audit.json',result)
    print(json.dumps({k:result.get(k) for k in ('recovery_audit_passed','elapsed_s','error')}))
    return 0 if result['recovery_audit_passed'] else 1


if __name__=='__main__':raise SystemExit(main())
