"""Allocated primitive orchestration. Disabled until a new reviewed lock exists."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from allocation_contract import PROFILE,REQUIRED_FILES,GLOBAL_SECONDS
from whole_run import run_member,save
from guarded_whole import run_guarded_whole


def readiness(root):
    root=Path(root).resolve()
    lock=json.loads((root/'readiness-lock.json').read_text())
    if lock.get('schema')!='owned-block-readiness-v1' or lock.get('submission_ready') is not True or lock.get('profile')!=PROFILE:
        raise ValueError('New complete readiness review required')
    expected=lock['sha256']
    if set(expected)!=set(REQUIRED_FILES)-{'readiness-lock.json'}:raise ValueError('Readiness source inventory differs')
    for name,digest in expected.items():
        path=root/name
        if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Readiness source changed: '+name)
    return expected


def compute(root,output):
    expected=readiness(root)
    from native_binding import SOURCE_FILES
    from publication import run_guarded
    from allocation import observe
    import torch
    torch.set_num_threads(4)
    before=observe();save(output/'compute-allocation-before.json',before)
    result=run_guarded(root,{n:expected[n] for n in SOURCE_FILES},output/'publication')
    after=observe();save(output/'compute-allocation-after.json',after)
    # Numerical negatives are valid retained results. Structural/runtime failures
    # fail the process; strict numerical eligibility is recomputed by final audit.
    return 0 if result['compute_complete'] else 1


def worker(root,output):
    expected=readiness(root)
    output.mkdir(exist_ok=False)
    command=[sys.executable,str(root/'allocated_entry.py'),'compute']
    result=run_member(command,root,os.environ,output)
    if not result['completed_normally']:return 1
    # Metadata-only final audit executes in this same outer-bounded worker.
    from final_audit import audit_member,audit_publication,audit_allocation_pair,read_json
    from native_binding import SOURCE_FILES
    control=audit_member(result,output,command,float(os.environ['OWNED_BLOCK_GLOBAL_STARTED']),os.getpgrp())
    publication=audit_publication(output/'publication',{n:expected[n] for n in SOURCE_FILES})
    run=Path(os.environ['OSCAR_RUN_DIR'])
    identity=audit_allocation_pair(read_json(output/'compute-allocation-before.json'),read_json(output/'compute-allocation-after.json'),
                 (run/'source-manifest.json').read_bytes(),(run/'job-id').read_bytes(),read_json(root/'oscar.json'),read_json(root/'protocol.json'),root)
    if time.monotonic()>float(os.environ['OWNED_BLOCK_GLOBAL_DEADLINE']):raise TimeoutError('Original global deadline expired')
    save(output/'worker-audit.json',dict(schema='owned-block-worker-audit-v1',member=control,publication=publication,allocation=identity,
         records_passed=publication['records_passed'],outer_group_closure_required=True,execution_certified=False,submission_ready=False))
    return 0


def finalize_worker(root,output,reviewed,receipt):
    if receipt['returncode']!=0 or receipt['remaining_group_terminated'] is not False:
        return dict(worker_closure_passed=False,records_passed=False,execution_certified=False)
    if readiness(root)!=reviewed:raise ValueError('Final reviewed source binding changed')
    worker=Path(output)/'worker'
    pubnames={f'publication/case-{i:02d}.json' for i in range(16)}|{'publication/'+n for n in ('before.json','after.json','inventory.json','inputs.json','result.json')}
    names=pubnames|{'child.stdout','child.stderr','member-exit.json','compute-allocation-before.json','compute-allocation-after.json','worker-audit.json'}
    def inventory():
        actual={p.relative_to(worker).as_posix() for p in worker.rglob('*') if p.is_file()}
        if actual!=names or any(p.is_symlink() for p in worker.rglob('*')):raise ValueError('Exact final worker file inventory differs')
        return {n:dict(bytes=(worker/n).stat().st_size,sha256=hashlib.sha256((worker/n).read_bytes()).hexdigest()) for n in sorted(names)}
    manifest=inventory();audit=json.loads((worker/'worker-audit.json').read_text())
    member=json.loads((worker/'member-exit.json').read_text())
    if audit['schema']!='owned-block-worker-audit-v1' or audit['outer_group_closure_required'] is not True or audit['execution_certified'] is not False or audit['submission_ready'] is not False:raise ValueError('Worker audit closure differs')
    if member['group_id']!=receipt['pid'] or member['global_deadline']!=receipt['global_deadline'] or member['completed_normally'] is not True:raise ValueError('Member and whole-run identity/deadline differ')
    if audit['member']['logs']!=member['logs'] or member['logs']!={n:manifest[n] for n in ('child.stdout','child.stderr')}:raise ValueError('Final member logs differ')
    publication=audit['publication']
    if publication['files']!={n.removeprefix('publication/'):manifest[n] for n in pubnames} or publication['publication_metadata_passed'] is not True:raise ValueError('Final publication hash binding differs')
    if type(audit['records_passed']) is not bool or audit['records_passed'] is not publication['records_passed']:raise ValueError('Final strict numerical gate differs')
    identity=audit['allocation']
    if identity['allocated_identity_passed'] is not True or identity['job_id']!=os.environ['SLURM_JOB_ID']:raise ValueError('Final allocation identity differs')
    if inventory()!=manifest:raise ValueError('Worker evidence changed during closure')
    return dict(worker_closure_passed=True,records_passed=audit['records_passed'],worker_files=manifest,
                source_sha256=dict(reviewed,**{'readiness-lock.json':hashlib.sha256((Path(root)/'readiness-lock.json').read_bytes()).hexdigest()}),
                guardian_closure_required=True,execution_certified=False,submission_ready=False)


def main():
    started=time.monotonic()
    root=Path(__file__).resolve().parent
    # Reject the disabled local package before scheduler queries or CUDA imports.
    reviewed=readiness(root)
    output=Path(os.environ['OSCAR_OUTPUT_DIR'])/'owned-block'
    mode=sys.argv[1] if len(sys.argv)==2 else ''
    if mode=='compute':return compute(root,output/'worker')
    if mode=='worker':return worker(root,output/'worker')
    if mode!='outer':raise ValueError('Exact entry mode required')
    result=run_guarded_whole([sys.executable,str(root/'allocated_entry.py'),'worker'],root,os.environ,output,limit=GLOBAL_SECONDS,started=started,
                             finalize=lambda receipt:finalize_worker(root,output,reviewed,receipt))
    return 0 if result['completed_normally'] else 1


if __name__=='__main__':
    # Loaded-source guards must inspect the actually executing entry namespace.
    sys.modules['allocated_entry']=sys.modules[__name__]
    raise SystemExit(main())
