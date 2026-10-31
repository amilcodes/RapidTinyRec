"""Saved-byte terminal reconciliation. No Torch, CUDA or remote operations.

Run only after one successful fetch has been frozen/hash inventoried. This
consumer checks recorded numerical evidence, never recomputes tensor outputs.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
from allocation import require,reconcile,verify_source
from allocation_contract import REQUIRED_FILES,GLOBAL_SECONDS,COMPUTE_SECONDS
from audit_evidence import audit_verified_inventory


def equal(a,b):
    return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)


def read(path):
    def constant(x):raise ValueError('Nonfinite JSON: '+x)
    def pairs(items):
        d={}
        for k,v in items:
            require(k not in d,'Duplicate JSON key');d[k]=v
        return d
    return json.loads(Path(path).read_text(),parse_constant=constant,object_pairs_hook=pairs)


def hash_file(path):
    p=Path(path);require(p.is_file() and not p.is_symlink(),'Regular evidence file required')
    raw=p.read_bytes();return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def inventory(root):
    root=Path(root);require(root.is_dir() and not root.is_symlink(),'Evidence directory required')
    paths=list(root.rglob('*'));require(not any(p.is_symlink() for p in paths),'Symlink evidence forbidden')
    return {p.relative_to(root).as_posix():hash_file(p) for p in paths if p.is_file()}


def literals(path):
    result={}
    for node in ast.parse(Path(path).read_bytes()).body:
        if isinstance(node,ast.Assign):
            try:v=ast.literal_eval(node.value)
            except (ValueError,TypeError):continue
            for target in node.targets:
                if isinstance(target,ast.Name):result[target.id]=v
    return result


def review(code,output,actual,manifest_bytes,job_bytes,terminal):
    code=Path(code);output=Path(output);raw=inventory(output)
    require(terminal['job_id']==actual['job_id'] and terminal['state']=='COMPLETED' and terminal['exit_code']=='0:0','Successful owned terminal state required for complete review')
    require(type(terminal['elapsed_seconds']) in (int,float) and math.isfinite(terminal['elapsed_seconds']) and 0<terminal['elapsed_seconds']<=600,'Terminal allocation duration differs')
    config=read(code/'oscar.json');protocol=read(code/'protocol.json')
    prepared=json.loads(manifest_bytes);verify_source(code,prepared)
    require(set(actual['sha256'])==set(actual['files'])==set(REQUIRED_FILES),'Complete submitted sources required')
    lock=read(code/'readiness-lock.json')
    require(lock['submission_ready'] is True and equal(lock['sha256'],{n:h for n,h in actual['sha256'].items() if n!='readiness-lock.json'}),'Submitted readiness differs')
    worker=output/'worker';w=read(output/'whole-exit.json');g=read(output/'guardian-exit.json');m=read(worker/'member-exit.json');a=read(worker/'worker-audit.json')
    require(w['schema']=='owned-block-whole-exit-v2' and g['schema']=='owned-block-guardian-exit-v1' and m['schema']=='owned-block-member-exit-v1','Closure schemas differ')
    for record in (w,m):
        require(type(record['returncode']) is int and record['returncode']==0 and record['reason']=='exited' and record['completed_normally'] is True,'Abnormal worker/member exit')
    require(w['error'] is None and g['error'] is None and g['reason']=='publication_closed','Guardian/whole failure')
    require(w['remaining_group_terminated'] is False and w['guardian_closure_required'] is True and m['group_closure_deferred_to_outer'] is True,'Missing group closure')
    ids=(w['pid'],w['guardian_pid'],g['parent_pid'],m['pid'])
    require(all(type(i) is int and i>0 for i in ids) and len(set(ids))==4,'Process identities differ')
    require(g['guardian_pid']==w['guardian_pid'] and m['group_id']==w['pid'],'Group/guardian identity mismatch')
    clocks=[w[k] for k in ('started_monotonic','ended_monotonic','publication_monotonic','global_deadline')]+[g['closed_monotonic'],m['started_monotonic'],m['ended_monotonic'],m['compute_deadline']]
    require(all(type(x) in (int,float) and math.isfinite(x) for x in clocks),'Finite clocks required')
    start=w['started_monotonic'];deadline=w['global_deadline']
    require(0<=start<=m['started_monotonic']<=m['ended_monotonic']<=w['ended_monotonic']<=w['publication_monotonic']<=g['closed_monotonic']<=deadline,'Execution/publication clock ordering differs')
    require(deadline==start+GLOBAL_SECONDS==g['global_deadline']==m['global_deadline'] and g['started_monotonic']==start,'Original global deadline differs')
    require(m['compute_deadline']==m['started_monotonic']+COMPUTE_SECONDS and m['ended_monotonic']<=m['compute_deadline'],'Compute deadline differs')
    require(g['closed_monotonic']-start<=terminal['elapsed_seconds']+1,'Scheduler duration shorter than execution')
    require(equal(g['worker'],dict(event='worker_closed',pid=w['pid'],returncode=0,remaining_group_terminated=False,ended_monotonic=w['ended_monotonic'])),'Guardian worker receipt differs')
    binary='/oscar/scratch/aagraw73/minionerec-shared-prefill/envs/torch214-cu126-transformers4554/bin/python'
    entry=actual['remote_dir']+'/code/allocated_entry.py'
    require(w['command']==[binary,entry,'worker'] and m['command']==[binary,entry,'compute'],'Actual executed command differs')
    require(equal(w['logs'],{n:raw[n] for n in ('whole.stdout','whole.stderr')}) and equal(m['logs'],{n:raw['worker/'+n] for n in ('child.stdout','child.stderr')}),'Closed log bytes differ')
    before=read(worker/'compute-allocation-before.json');after=read(worker/'compute-allocation-after.json')
    reconciled=[reconcile(x,actual,manifest_bytes,job_bytes,config,protocol) for x in (before,after)]
    require(a['schema']=='owned-block-worker-audit-v1' and a['outer_group_closure_required'] is True,'Worker audit missing')
    require(equal(a['allocation'],dict(allocated_identity_passed=True,job_id=actual['job_id'],run_id=actual['run_id'])),'Worker identity gate differs')
    require(a['member']['child_normal_exit_and_logs_verified'] is True and a['member']['group_closure_deferred_to_outer'] is True and equal(a['member']['logs'],m['logs']),'Worker member gate differs')
    pub=worker/'publication';names={f'case-{i:02d}.json' for i in range(16)}|{'before.json','after.json','inventory.json','inputs.json','result.json'}
    require(set(inventory(pub))==names,'Complete publication required')
    records=[read(pub/f'case-{i:02d}.json') for i in range(16)]
    require([(r['seed'],r['offset'],r['mode']) for r in records]==[(s,o,k) for s in (17,29) for o in range(4) for k in ('eager','cuda_graph')],'Ordered sixteen cases required')
    checked=audit_verified_inventory(records)
    require(equal(checked,read(pub/'inventory.json')),'Recorded independent inventory differs')
    require(equal(read(pub/'inputs.json'),dict(seed_generation_independently_verified=True,seeds=2,output_recomputed=False)),'Input regeneration receipt differs')
    guard=read(pub/'before.json');require(equal(guard,read(pub/'after.json')),'Runtime/source guard changed')
    pins=literals(code/'native_binding.py');loaded=guard['loaded']
    require(equal(loaded['source_sha256'],{n:actual['sha256'][n] for n in pins['SOURCE_FILES']}) and loaded['loaded_source_bindings_passed'] is True,'Loaded source bindings differ')
    require(all(type(loaded[k]) is int and loaded[k]>0 for k in ('loaded_definitions','import_aliases')),'Missing loaded definition coverage')
    require(equal(guard['native'],dict(functional_sha256=pins['FUNCTIONAL_SHA256'],loaded_functional_code_matches=True,tensor_method_native_descriptor=True,override_modes_absent=True,native_schema=pins['NATIVE_SCHEMA'],execution_certified=False)),'Native receipt differs')
    runtime=guard['runtime'];memory=runtime['total_memory_bytes']
    require(type(memory) is int and 47*2**30<=memory<=49*2**30,'Regular A6000 memory differs')
    expected_runtime=dict(torch_version='2.14.0+cu126',torch_git=pins['TORCH_GIT'],cuda_version='12.6',device_count=1,current_device=0,device_name='NVIDIA RTX A6000',capability=[8,6],num_threads=4,cuda_normalization_registered=True,capture_active=False,total_memory_bytes=memory)
    require(equal(runtime,expected_runtime),'Runtime hardware/source differs')
    publication=a['publication'];final=w['finalization'];result=read(pub/'result.json')
    require(publication['publication_metadata_passed'] is True and publication['cases']==16 and publication['comparisons']==320 and equal(publication['files'],inventory(pub)),'Worker publication gate differs')
    require(result['compute_complete'] is True and result['before_after_guards_match'] is True and type(result['published_cases']) is int and result['published_cases']==16,'Publication incomplete')
    for flag in (a['records_passed'],publication['records_passed'],final['records_passed'],result['records_passed']):
        require(type(flag) is bool and flag==checked['records_passed'],'Strict numerical gate differs')
    require(final['worker_closure_passed'] is True and final['guardian_closure_required'] is True,'Final closure missing')
    require(equal(final['worker_files'],inventory(worker)) and equal(final['source_sha256'],actual['sha256']),'Final worker/source hash binding differs')
    expected_names={'whole.stdout','whole.stderr','whole-exit.json','guardian-exit.json'}|{'worker/'+n for n in final['worker_files']}
    require(len(final['worker_files'])==27 and set(raw)==expected_names,'Unexpected/missing terminal evidence')
    for record in (w,g,m,a,publication,final,result,loaded,guard['native']):
        require(record['execution_certified'] is False,'Premature execution certificate')
    verify_source(code,prepared);require(equal(raw,inventory(output)),'Raw bytes changed during metadata review')
    return dict(schema='owned-block-terminal-metadata-v1',job_id=actual['job_id'],run_id=actual['run_id'],reconciled_records=reconciled,
                complete_protocol_metadata_verified=True,strict_bits_passed=checked['records_passed'],cases=16,comparisons=320,
                raw_files=len(raw),source_files=len(actual['files']),numerical_reaudit=False,model_repair_proved=False,
                production_latency_claim=False,promotion=False,formal_compiler_proof=False)
