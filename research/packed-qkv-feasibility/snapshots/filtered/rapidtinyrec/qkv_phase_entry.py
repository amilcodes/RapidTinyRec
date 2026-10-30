"""Fresh-process allocated phase entry; original work may never be restarted."""
import argparse,os,time
from pathlib import Path
from .qkv_allocation import read,verify_source
from .qkv_audit_headers import validate_work_deadline
from .qkv_pilot_runner import save,HISTORICAL,RELEASE

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',type=int,choices=range(4),required=True);p.add_argument('--api-log-local-root',type=Path,required=True);a=p.parse_args()
    run=Path(os.environ['OSCAR_RUN_DIR']).resolve(strict=True);out=run/'outputs';code=run/'code'
    from .qkv_readiness import require_reviewed_source
    require_reviewed_source(code)
    allocation=read(out/'allocated-run-record.json');work=read(out/'qkv-work-start.json')
    if allocation['job_id']!=os.environ.get('SLURM_JOB_ID') or allocation['remote_dir']!=str(run):raise ValueError('Wrong phase allocation')
    verify_source(code,allocation);validate_work_deadline(work,allocation)
    if work['protocol_sha256']!=allocation['sha256']['protocol.json']:raise ValueError('Wrong phase protocol')
    generation=read(out/'qkv-generation-start.json');deadline=min(generation['started_monotonic']+generation['timeout_s'],work['deadline_monotonic'])
    started=time.monotonic()
    if started>=deadline or (out/f'phase_{a.phase}').exists():raise ValueError('Late or repeated phase')
    if a.phase and read(out/f'qkv-phase-{a.phase-1}-exit.json')['complete'] is not True:raise ValueError('Previous phase incomplete')
    marker=dict(schema='packed-qkv-process-v1',phase=a.phase,pid=os.getpid(),job_id=allocation['job_id'],run_id=allocation['run_id'],protocol_sha256=work['protocol_sha256'],started_monotonic=started)
    save(out/f'qkv-phase-{a.phase}-start.json',marker)
    complete=False
    try:
        from .benchmark_packed_qkv import produce_phase
        a.output_root=out;a.release_root=Path(RELEASE);a.reference_results=Path(HISTORICAL);a.reuse_compiler_artifacts=a.phase>=2
        produce_phase(a,allocation,work)
        complete=time.monotonic()<deadline
        if not complete:raise TimeoutError('Phase exceeded generation deadline')
    finally:
        save(out/f'qkv-phase-{a.phase}-exit.json',dict(marker,complete=complete,ended_monotonic=time.monotonic()))
if __name__=='__main__':main()
