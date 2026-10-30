"""Allocated QKV evidence and stage integration, gated by exact readiness lock."""
import hashlib
import os
import time
from pathlib import Path
from .audit_qkv_outputs import require,paired_timing
from .audit_qkv_phase import read,audit_phase_evidence
from .qkv_allocation import observe_allocation,verify_source,scheduler_fields,validate_scheduler
from .qkv_allocation_contract import load_protocol
from .qkv_audit_headers import validate_reference,validate_phase_header,validate_cache_partitions,validate_closures
from .qkv_catalog import authenticate_catalog
from .qkv_control import load_historical,compare_cross_phase
from .qkv_inventory_schema import pinned_config
from .qkv_quality import evaluate_allocated
from .qkv_replay_audit import validate_runtime_records
from .qkv_native_dispatch import inspect_native_dispatch
from .audit_qkv_capture import validate_capture_binding
from .qkv_runtime_identity import observe_installed_runtime,validate_runtime_calls
from .qkv_load_observer import validate_load_identities


def validate_original_allocation(original,current):
    require(original['submission']=='allocated_observation' and current['submission']=='allocated_observation','Allocated source identities required')
    for key in ('run_id','job_id','project','profile','ssh_target','remote_dir','resources','files','sha256','source_bytes'):
        require(original[key]==current[key],'Allocation changed since work start: '+key)
    for key in ('uploaded_manifest_sha256','sbatch_job_id_file_sha256'):
        require(original['allocation_identity'][key]==current['allocation_identity'][key],'Source/job binding changed')
    for record in (original,current):
        identity=record['allocation_identity']
        require(identity['schema']=='packed-qkv-allocated-identity-v1' and identity['requires_local_oscar_reconciliation'] is True,'Allocation schema differs')
        require(identity['scheduler_command']==['scontrol','show','job','-o',record['job_id']],'Scheduler command differs')
        fields=scheduler_fields(identity['scheduler_stdout'])
        require(fields==identity['scheduler_fields'],'Scheduler text/field binding differs')
        validate_scheduler(fields,record['job_id'],record['remote_dir'])
    return original


def inspect_evidence(root,code,historical,release):
    require(bool(os.environ.get('SLURM_JOB_ID')),'Allocated evidence inspection only')
    root=Path(root).resolve(strict=True);code=Path(code).resolve(strict=True);release=Path(release)
    current=observe_allocation(root.parent,code,root)
    original=validate_original_allocation(read(root/'allocated-run-record.json'),current)
    require(original['job_id']==os.environ['SLURM_JOB_ID'],'Wrong allocated process')
    verify_source(code,original)
    protocol_bytes=(code/'protocol.json').read_bytes();protocol=load_protocol(code/'protocol.json')
    require(hashlib.sha256(protocol_bytes).hexdigest()==original['sha256']['protocol.json'],'Submitted protocol differs')
    expected=read(code/'rapidtinyrec/expected.json');reference=read(Path(historical)/'manifest.json')
    users=validate_reference(reference,expected)
    historical_rows,historical_proof=load_historical(code/'historical_bf16',users,expected)
    require(protocol['historical_control']['reference_sha256']==historical_proof['reference_lock_sha256'],'Protocol historical reference pin differs')
    config=pinned_config(code/'release-config.json',expected['release_sha256']['Office_ckpt/config.json'])
    constraints,root_prefix,catalog_proof=authenticate_catalog(release,users,expected)
    require({p.name for p in root.glob('phase_*') if p.is_dir()}=={f'phase_{p}' for p in range(4)},'All four phase directories required')
    work=read(root/'qkv-work-start.json')
    require(work['schema']=='packed-qkv-work-v1' and work['job_id']==original['job_id'] and work['run_id']==original['run_id']
            and work['protocol_sha256']==hashlib.sha256(protocol_bytes).hexdigest(),'Inclusive work identity differs')
    require(time.monotonic()<work['deadline_monotonic'],'Global allocation work deadline expired')
    sources={str(p.relative_to(code)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (code/'rapidtinyrec').glob('*.py')}
    installed_runtime=observe_installed_runtime(root)
    manifests={};setups={};closures={};phases={};paths={};requests={}
    for phase in range(4):
        path=root/f'phase_{phase}';paths[phase]=path
        m,s,w,c=(read(path/n) for n in ('manifest.json','summary.json','setup.json','compilation.json'))
        d=read(path/'dispatch.json')
        header=validate_phase_header(m,phase,original,protocol_bytes,users,expected,reference['manifest_sha256'],read(code/'rapidtinyrec/quality_resources.json'),sources)
        require(m['work_deadline']==work,'Phase work deadline replaced')
        runtime=validate_runtime_records(path,w,s,d,c)
        native=inspect_native_dispatch(path,d,m['variant'])
        evidence=audit_phase_evidence(path,phase,config,users,constraints,root_prefix)
        bindings=[read(path/f'capture-bindings-{stage}.json') for stage in ('before-measurement','after-measurement','after-profiles')]
        require(bindings[0]==bindings[1]==bindings[2],'Captured source identity changed after setup')
        runtime_identity=validate_runtime_calls(bindings[0],installed_runtime)
        loaded_binaries=validate_load_identities(path,bindings[0],installed_runtime)
        capture=validate_capture_binding(path,bindings[0],d,read(path/'storage-before-measurement.json'),m['variant'])
        manifests[phase]=m;setups[phase]=w;closures[phase]=read(path/'phase-closure.json');requests[phase]=read(path/'requests.json')
        phases[phase]=dict(header=header,runtime=runtime,runtime_identity=runtime_identity,loaded_binaries=loaded_binaries,native_dispatch=native,capture_binding=capture,evidence=evidence)
    caches=validate_cache_partitions(manifests,setups);closure=validate_closures(manifests,closures)
    controls=compare_cross_phase(paths,historical_rows);timing=paired_timing(requests)
    quality=evaluate_allocated(release,users,requests,current)
    require(time.monotonic()<work['deadline_monotonic'],'Inclusive work deadline exceeded by audit')
    return dict(schema='packed-qkv-component-inspection-v1',job_id=current['job_id'],run_id=current['run_id'],allocation=current,
                phases=phases,installed_runtime=installed_runtime,cache_partitions=caches,phase_closure=closure,catalog=catalog_proof,historical_reference=historical_proof,
                controls=controls,timing=timing,quality=quality,execution_audit_passed=False,complete_protocol_certified=False,
                reviewed_full500_eligible=False,remaining=['Fused compiled downstream Q/K/V index maps',
                    'Generation/audit stage exit receipts and budget/log-wrapper recovery publication closure'],promotion=False)


def audit(root,code,historical,release):
    from .qkv_readiness import require_reviewed_source
    from .qkv_pilot_runner import validate_audit_start,validate_process_receipts
    from .audit_precision_setup import validate_wrapper
    from .qkv_final_gates import finalize_components
    root=Path(root).resolve(strict=True);code=Path(code).resolve(strict=True)
    require_reviewed_source(code)
    record=read(root/'allocated-run-record.json')
    budget=validate_audit_start(root,record,time.monotonic())
    wrapper=validate_wrapper(root,record,phases=(0,1,2,3),child_timeout_s=1500)
    require(wrapper['child']==read(root/'qkv-generation-exit.json')['child'],'Wrapper/generation exit differs')
    require(wrapper['recovery']==read(root/'log-recovery.json'),'Wrapper/recovery payload differs')
    processes=validate_process_receipts(root,record)
    result=inspect_evidence(root,code,historical,release)
    require(time.monotonic()<budget['effective_deadline_monotonic'],'Allocated audit deadline expired')
    result.update(integrity_audit_passed=True,deadline_validation=budget,wrapper=wrapper,processes=processes)
    return finalize_components(result)

def main():
    import argparse,json,traceback
    from .qkv_pilot_runner import save
    p=argparse.ArgumentParser()
    for name in ('root','code','historical','release','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Never overwrite audit evidence')
    try:
        result=audit(a.root,a.code,a.historical,a.release);status=0 if result['execution_audit_passed'] else 1
    except Exception as exc:
        result=dict(execution_audit_passed=False,complete_protocol_certified=False,reviewed_full500_eligible=False,promotion=False,error=repr(exc),traceback=traceback.format_exc());status=1
    save(a.output,result)
    raise SystemExit(status)

if __name__=='__main__':main()
