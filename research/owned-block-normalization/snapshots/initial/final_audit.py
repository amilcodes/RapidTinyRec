"""Independent saved-record publication gate; no output recomputation.

The outer hard deadline and its final exit receipt remain separate requirements.
This module never certifies execution or permits submission by itself.
"""
import hashlib
import json
import math
from pathlib import Path
from allocation import require,validate_metadata,verify_source,scheduler_fields
from allocation_contract import COMPUTE_SECONDS,GLOBAL_SECONDS
from audit_evidence import audit_verified_inventory
from native_binding import SOURCE_FILES,FUNCTIONAL_SHA256,NATIVE_SCHEMA,validate_runtime


def file_record(path):
    path=Path(path)
    require(path.is_file() and not path.is_symlink(),'Regular published file required')
    raw=path.read_bytes()
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def same_json(left,right):
    # JSON booleans must not compare equal to integer 0/1.
    return json.dumps(left,sort_keys=True,allow_nan=False)==json.dumps(right,sort_keys=True,allow_nan=False)


def read_json(path):
    file_record(path)
    def reject(value):raise ValueError('Nonfinite JSON constant: '+value)
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'Duplicate JSON key: '+key);result[key]=value
        return result
    return json.loads(Path(path).read_text(),parse_constant=reject,object_pairs_hook=pairs)


def audit_allocation_pair(before,after,manifest_bytes,job_bytes,config,protocol,code):
    prepared=json.loads(manifest_bytes);job=job_bytes.decode().strip()
    for record in (before,after):
        require(record['schema']=='owned-block-allocation-v1','Allocation schema differs')
        require(record['prepared']==prepared and record['job_id']==job and record['run_id']==prepared['run_id'],'Allocation identity differs')
        require(record['uploaded_manifest_sha256']==hashlib.sha256(manifest_bytes).hexdigest() and
                record['job_id_sha256']==hashlib.sha256(job_bytes).hexdigest(),'Allocation byte binding differs')
        require(record['scheduler_command']==['scontrol','show','job','-o',job],'Wrong scheduler observation')
        fields=scheduler_fields(record['scheduler_stdout'])
        require(fields==record['scheduler_fields'],'Scheduler record differs')
        validate_metadata(prepared,config,protocol,job,record['environment'],fields)
    require(before['environment']==after['environment'],'Allocation environment changed')
    verify_source(code,prepared)
    return dict(allocated_identity_passed=True,job_id=job,run_id=prepared['run_id'])


def audit_child(receipt,logs,command,started):
    require(receipt['schema']=='owned-block-child-exit-v1' and receipt['command']==list(command),'Child command/schema differs')
    require(type(receipt['pid']) is int and receipt['pid']>0,'Child PID missing')
    require(type(receipt['returncode']) is int and receipt['returncode']==0 and receipt['reason']=='exited','Child did not exit normally')
    require(receipt['remaining_group_terminated'] is False and receipt['completed_normally'] is True and receipt['within_global_deadline'] is True,'Child closure failed')
    keys=('started_monotonic','ended_monotonic','compute_deadline','global_deadline')
    clocks=[started]+[receipt[k] for k in keys]
    require(all(type(v) in (int,float) and math.isfinite(v) for v in clocks),'Finite real clocks required')
    began,ended,compute,glob=(receipt[k] for k in keys)
    require(0<=started<=began<=ended<=min(compute,glob),'Child clock ordering differs')
    require(compute==began+COMPUTE_SECONDS and glob==started+GLOBAL_SECONDS,'Original deadlines differ')
    actual={n:file_record(Path(logs)/n) for n in ('child.stdout','child.stderr')}
    require(receipt['logs']==actual,'Closed child log hashes differ')
    return dict(child_normal_exit_and_logs_verified=True,logs=actual)


def audit_publication(output,expected_sources):
    output=Path(output)
    names={f'case-{i:02d}.json' for i in range(16)}|{'before.json','after.json','inventory.json','inputs.json','result.json'}
    require(output.is_dir() and not output.is_symlink(),'Publication directory differs')
    require({p.name for p in output.iterdir()}==names,'Complete exact publication inventory required')
    manifest={n:file_record(output/n) for n in sorted(names)}
    records=[read_json(output/f'case-{i:02d}.json') for i in range(16)]
    expected_order=[(seed,offset,mode) for seed in (17,29) for offset in range(4) for mode in ('eager','cuda_graph')]
    require([(r['seed'],r['offset'],r['mode']) for r in records]==expected_order,'Published case order differs')
    inventory=audit_verified_inventory(records)
    require(same_json(read_json(output/'inventory.json'),inventory),'Saved inventory differs from independent metadata audit')
    seeds=read_json(output/'inputs.json')
    require(same_json(seeds,dict(seed_generation_independently_verified=True,seeds=2,output_recomputed=False)),'Allocated input regeneration receipt differs')
    before=read_json(output/'before.json');after=read_json(output/'after.json')
    require(same_json(before,after) and set(before)=={'loaded','native','runtime'},'Before/after guards differ')
    loaded=before['loaded']
    require(set(expected_sources)==set(SOURCE_FILES) and loaded['source_sha256']==expected_sources,'Loaded source manifest differs')
    require(loaded['loaded_source_bindings_passed'] is True and loaded['execution_certified'] is False,'Loaded source proof differs')
    require(all(type(loaded[k]) is int and loaded[k]>0 for k in ('loaded_definitions','import_aliases')),'Loaded source coverage missing')
    native=dict(functional_sha256=FUNCTIONAL_SHA256,loaded_functional_code_matches=True,
                tensor_method_native_descriptor=True,override_modes_absent=True,native_schema=NATIVE_SCHEMA,execution_certified=False)
    require(same_json(before['native'],native),'Native entry receipt differs');validate_runtime(before['runtime'])
    result=read_json(output/'result.json')
    require(set(result)=={'schema','compute_complete','records_passed','execution_certified','submission_ready','published_cases','before_after_guards_match'},'Incomplete/error publication cannot close successfully')
    require(result['schema']=='request-block-guarded-publication-v1' and result['compute_complete'] is True and result['before_after_guards_match'] is True,'Publication incomplete')
    require(type(result['published_cases']) is int and result['published_cases']==16,'Published case count differs')
    require(type(result['records_passed']) is bool and result['records_passed']==inventory['records_passed'],'Numerical result flag differs')
    require(result['execution_certified'] is False and result['submission_ready'] is False,'Premature certificate forbidden')
    # Rehash after parsing/auditing to detect concurrent publication changes.
    require(manifest=={n:file_record(output/n) for n in sorted(names)} and {p.name for p in output.iterdir()}==names,'Published evidence changed during final audit')
    return dict(schema='owned-block-final-publication-audit-v1',cases=16,comparisons=320,
                records_passed=inventory['records_passed'],publication_metadata_passed=True,
                files=manifest,execution_certified=False,submission_ready=False,
                output_recomputed=False,formal_compiler_proof=False)


def audit_member(receipt,logs,command,started,group_id):
    require(receipt['schema']=='owned-block-member-exit-v1','Member schema differs')
    require(type(group_id) is int and group_id>0 and receipt['group_id']==group_id,'Member process group differs')
    require(receipt['group_closure_deferred_to_outer'] is True and receipt['execution_certified'] is False,'Premature member closure')
    require(receipt['reason']=='exited','Member deadline/failure')
    # Reuse strict clock/exit/log checks, with group closure explicitly deferred.
    # This is not an assertion that no descendants remain in the live worker.
    common=dict(receipt,schema='owned-block-child-exit-v1',remaining_group_terminated=False,
                within_global_deadline=receipt['ended_monotonic']<=receipt['global_deadline'])
    checked=audit_child(common,logs,command,started)
    checked.update(group_closure_deferred_to_outer=True,execution_certified=False)
    return checked
