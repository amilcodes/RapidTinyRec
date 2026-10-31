"""Allocated-only full500 audit; local reconciliation never resamples quality."""
import argparse
import json
import os
from pathlib import Path
import traceback
import time

from .audit_precision import audit as paired_audit
from .audit_precision_setup import validate_wrapper
from .precision_allocation import observe_allocation, reconcile_record, read, digest
from .precision_full500 import PROTOCOL, PROTOCOL_SHA256, load_protocol
from .precision_full500_runner import validate_audit_start, reconcile_work


def validate_live_record(saved, live):
    assert saved['submission']==live['submission']=='allocated_observation'
    for key in ('run_id','job_id','project','profile','ssh_target','remote_dir','resources','files','sha256'):
        assert saved[key]==live[key],key
    for key in ('schema','uploaded_manifest_sha256','sbatch_job_id_file_sha256'):
        assert saved['allocation_identity'][key]==live['allocation_identity'][key],key
    assert saved['allocation_identity']['requires_local_oscar_reconciliation'] is True


def audit(root, historical, snapshot, run_root, release):
    # Real source-manifest, sbatch file and live scheduler query precede all work.
    assert os.environ.get('SLURM_JOB_ID'), 'Full500 audit requires allocated compute'
    root=Path(root).resolve(strict=True); snapshot=Path(snapshot).resolve(strict=True)
    live=observe_allocation(run_root,snapshot.parent,root)
    record_path=root/'allocated-run-record.json';saved=read(record_path)
    validate_live_record(saved,live)
    protocol=load_protocol(snapshot.parent/PROTOCOL)
    assert saved['sha256'][PROTOCOL]==PROTOCOL_SHA256
    wrapper=validate_wrapper(root,saved,phases=(0,1,2,3),
        child_timeout_s=protocol['budget']['generation_child_deadline_s'])
    validate_audit_start(root,saved,now=time.monotonic())
    assert {p.name for p in root.glob('phase_*') if p.is_dir()}=={f'phase_{i}' for i in range(4)}
    result=paired_audit(root,historical,snapshot,record_path,release,full500_quality=True)
    assert result['unique_users']==result['paired_requests']==500
    assert all(p['untimed_repeat_check']['untimed_repeat_requests']==250 for p in result['phases'])
    result.update(schema='precision-full500-quality-audit-v1',run_id=saved['run_id'],job_id=saved['job_id'],
        full500_audit_sha256=digest(Path(__file__)),protocol_sha256=PROTOCOL_SHA256,
        allocated_record_sha256=digest(record_path),allocation_identity=live['allocation_identity'],
        wrapper=wrapper,primary_pairs=500,untimed_repeat_requests=1000,
        source_reconciliation_required=True,local_source_reconciliation_passed=False,
        promotion=False,noninferiority_established=False)
    result['full500_execution_audit_passed']=all(result[k] for k in ('integrity_audit_passed','control_exact','stability_exact','dispatch_verified','validity_passed'))
    # Performance and quality remain reported even when disappointing. CLI failure
    # indicates execution/integrity failure, not a reason to discard a regression.
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    allocated=sub.add_parser('allocated')
    for name in ('root','historical','snapshot','run-root','release-root','output'):
        allocated.add_argument('--'+name,type=Path,required=True)
    local=sub.add_parser('reconcile')
    for name in ('allocated-record','run-record','uploaded-manifest','job-id-file','root','output'):
        local.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists(),'Never overwrite evidence'
    try:
        if args.action=='allocated':
            result=audit(args.root,args.historical,args.snapshot,args.run_root,args.release_root)
            passed=result['full500_execution_audit_passed']
        else:
            result=reconcile_record(read(args.allocated_record),read(args.run_record),
                args.uploaded_manifest.read_bytes(),args.job_id_file.read_bytes())
            assert args.allocated_record.resolve()==(args.root/'allocated-run-record.json').resolve()
            result['work']=reconcile_work(args.root,read(args.run_record))
            passed=result['matched']
    except Exception as exc:
        result=dict(schema='precision-full500-audit-failure-v1',action=args.action,
            error=repr(exc),traceback=traceback.format_exc(),integrity_audit_passed=False,
            promotion=False,noninferiority_established=False)
        passed=False
    with args.output.open('x') as file:
        json.dump(result,file,indent=2,allow_nan=False);file.write('\n')
    print(json.dumps({k:result.get(k) for k in ('full500_execution_audit_passed','matched','error')},indent=2))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
