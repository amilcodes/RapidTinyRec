"""Distinct allocated audit of complete serial-pair/wave ABBA raw evidence.

No numerical work is certified by relabeling an existing precision/serving
experiment. Shared helpers below only parse unchanged log/resource formats or
consume explicit evaluator control/candidate observations.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time
import traceback
from .audit_decode_pilot import read,digest,json_digest
from .audit_precision import validate_log_staging,TORCH_COMMIT
from .precision_provenance import validate_release
from .precision_policy import validate_state
from .beam_wave_protocol import PROTOCOL,PROTOCOL_SHA256,PHASES,load_protocol,compare_phases
from .beam_wave_allocation import observe_allocation,scheduler_fields,validate_scheduler
from .audit_beam_wave_outputs import validate_trials,validate_request,validate_request_group,compare_user_outputs
from .audit_beam_wave_dispatch import inspect_dispatch,validate_replay_counts
from .beam_wave_storage import validate_stages


def validate_header(m,phase,live,protocol,users,expected,reference):
    if (m['schema']!='two-request-beam-wave-phase-v1' or m['experiment']!=protocol['experiment']
            or m['phase']!=phase or m['half']!=phase//2 or m['variant']!=PHASES[phase]
            or m['phase_protocol']!=protocol['phases'][phase] or m['protocol_sha256']!=PROTOCOL_SHA256):
        raise ValueError('Distinct original wave phase identity required')
    if m['job_id']!=live['job_id'] or m['run_id']!=live['run_id']:raise ValueError('Phase job/run differs')
    for key in ('run_id','job_id','project','profile','files','sha256','resources','remote_dir'):
        if m['allocation'][key]!=live[key]:raise ValueError('Phase allocation/source binding differs: '+key)
    identity=m['allocation']['allocation_identity']
    if (identity['schema']!='beam-wave-allocated-identity-v1'
            or m['allocation']['submission']!='allocated_observation'
            or identity['requires_local_oscar_reconciliation'] is not True
            or identity['scheduler_command']!=['scontrol','show','job','-o',live['job_id']]):
        raise ValueError('Wrong original allocation identity')
    for key in ('uploaded_manifest_sha256','sbatch_job_id_file_sha256'):
        if identity[key]!=live['allocation_identity'][key]:raise ValueError('Original submitted identity differs')
    fields=scheduler_fields(identity['scheduler_stdout'])
    if fields!=identity['scheduler_fields']:raise ValueError('Saved scheduler text differs')
    validate_scheduler(fields,live['job_id'],live['remote_dir'])
    if (m['users']!=users or m['cohort_sha256']!=expected['users_sha256'] or m['reference_manifest_sha256']!=reference['manifest_sha256']):
        raise ValueError('Original user identity differs')
    if (m['dtype']!='float32' or m['tf32'] is not True or m['attention_layout']!='folded_query_heads'
            or m['hardware']!='NVIDIA RTX A6000' or m['torch_git']!=TORCH_COMMIT or m['cuda_version']!='12.6'
            or m['versions']!={'torch':'2.14.0+cu126','transformers':'4.55.4','tokenizers':'0.21.4'}
            or m['compiler_policy']!=dict(max_autotune=False,max_autotune_gemm=False,max_autotune_pointwise=False)):
        raise ValueError('Original runtime/precision/engine policy differs')
    if m['api_log_staging'] is not True:raise ValueError('Original log staging required')
    if m['timing_contract']!=protocol['timing']:raise ValueError('Pair timing contract differs')
    validate_state(m['precision_policy'],'folded_tf32');validate_release(m['release_verification'],expected['release_sha256'])


def validate_work_identity(work,live):
    if (work['schema']!='two-request-beam-wave-work-v1' or work['job_id']!=live['job_id']
            or work['run_id']!=live['run_id'] or work['protocol_sha256']!=PROTOCOL_SHA256):
        raise ValueError('Original work identity differs')
    start,end=work['started_monotonic'],work['deadline_monotonic']
    if (any(type(x) not in (int,float) or not math.isfinite(x) for x in (start,end))
            or start<0 or end-start!=1620):
        raise ValueError('Original inclusive work deadline differs')
    return work


def inspect_phases(root,code,historical,release,live):
    if not os.environ.get('SLURM_JOB_ID') or os.environ['SLURM_JOB_ID']!=live['job_id']:
        raise ValueError('Allocated wave raw audit only')
    protocol=load_protocol(code/PROTOCOL);expected=read(code/'rapidtinyrec/expected.json');reference=read(historical/'manifest.json')
    if reference['experiment']!='gqa_three_way' or reference['dtype']!='float32':raise ValueError('Wrong reference')
    if json_digest({k:v for k,v in reference.items() if k!='manifest_sha256'})!=reference['manifest_sha256']:
        raise ValueError('Reference manifest hash differs')
    if len(reference['users'])!=500 or json_digest(reference['users'])!=expected['users_sha256']:
        raise ValueError('Original500user identity differs')
    if expected['users_sha256']!=protocol['release_users_sha256']:raise ValueError('Protocol user identity differs')
    work=read(root/'wave-work-start.json')
    validate_work_identity(work,live)
    users=[next(u for u in reference['users'] if u['cohort_index']==i) for i in (1,2,6,10)]
    if {p.name for p in root.glob('phase_*') if p.is_dir()}!={f'phase_{i}' for i in range(4)}:raise ValueError('All four original phases required')
    config=read(release/'Office_ckpt/config.json')
    if digest(release/'Office_ckpt/config.json')!=expected['release_sha256']['Office_ckpt/config.json']:raise ValueError('Model config changed')
    layout=dict(layers=config['num_hidden_layers'],kv_heads=config['num_key_value_heads'],head_dim=config['hidden_size']//config['num_attention_heads'])
    vocab=config['vocab_size'];phases=[];timing={};joined={};seen={};all_controls=[];cache_paths={}
    for phase,variant in enumerate(PHASES):
        path=root/f'phase_{phase}';m,s,w,c=(read(path/n) for n in ('manifest.json','summary.json','setup.json','compilation.json'))
        dispatch=read(path/'dispatch.json');trials=read(path/'trials.json')
        validate_header(m,phase,live,protocol,users,expected,reference)
        if m['work_deadline']!=work:raise ValueError('Phase replaced original work deadline')
        if m['kv_layout']!=layout or m['vocab_size']!=vocab:raise ValueError('Model KV/vocabulary layout differs')
        validate_release(m['quality_release_verification'],read(code/'rapidtinyrec/quality_resources.json'))
        source={p.name:digest(p) for p in (code/'rapidtinyrec').glob('*.py')}
        if m['source_sha256']!=source:raise ValueError('Running source dependency changed')
        validate_log_staging(path,m)
        for obj in (w,s,dispatch):validate_state(obj['precision_policy'],'folded_tf32')
        if (c['phase']!='complete' or c['measurement_compilations']!=0 or not c['events']
                or not all(e.get('success') or e.get('expected_restart') for e in c['events'])
                or c['compiler_counters'].get('graph_break') or c['compiler_counters'].get('unimplemented')):
            raise ValueError('Incomplete compilation or compilation after freeze')
        artifacts=m['compiler_artifacts'];before=artifacts['before'];after=w['compiler_artifacts_after']
        if artifacts['reused']!=(phase>=2):
            raise ValueError('Wrong compiler cache inventory')
        if set(before)!={'TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR'} or set(after)!=set(before):raise ValueError('Compiler cache scope differs')
        paths={k:v['path'] for k,v in before.items()}
        if paths!={k:v['path'] for k,v in after.items()} or sum(v['files'] for v in after.values())<=0:raise ValueError('Compiler artifacts missing or moved')
        if phase<2:
            if any(v['files'] for v in before.values()):raise ValueError('Cold phase reused compiler artifacts')
            cache_paths[variant]=paths
        elif cache_paths[variant]!=paths or sum(v['files'] for v in before.values())<=0:raise ValueError('Fresh-process same-variant disk reuse differs')
        inventory=validate_trials(phase,trials,vocab);timing[phase]=[t['timing'] for t in trials]
        expected_files={f"{t['timing']['repeat']}_{t['timing']['pair_index']}.json" for t in trials}
        if {p.name for p in (path/'trials').glob('*.json')}!=expected_files:raise ValueError('Individual trial inventory differs')
        for t in trials:
            if read(path/'trials'/f"{t['timing']['repeat']}_{t['timing']['pair_index']}.json")!=t:raise ValueError('Trial differs from combined record')
        counts=validate_replay_counts(phase,w,s,dispatch)
        inventories=[read(path/f'storage-{stage}.json') for stage in ('before-measurement','after-measurement','after-profiles')]
        if any(r['variant']!=variant for r in inventories):raise ValueError('Storage inventory belongs to another arm')
        storage=validate_stages(inventories,**layout,vocab_size=vocab)
        observed_dispatch=inspect_dispatch(path,dispatch,variant)
        for item in dispatch['generated_code']:
            name=Path(item['file'])
            if name.is_absolute() or '..' in name.parts or digest(path/name)!=item['sha256']:raise ValueError('Generated code hash differs')
        if not dispatch['generated_code']:raise ValueError('Generated code missing')
        comparisons=[]
        def accept(row):
            key=(variant,row['cohort_index'])
            if key in seen:comparisons.append(compare_user_outputs(seen[key],row))
            else:seen[key]=row
            if variant=='serial_pair_tf32':all_controls.append(row)
        for t in trials:
            for row in t['requests']:
                accept(row)
                key=(t['timing']['repeat'],row['cohort_index'])
                joined.setdefault(key,dict(repeat=key[0],cohort_index=key[1],runs={} ))['runs'][variant]=row
        groups=[[1,2],[6,10]] if phase in (1,2) else [[1],[2],[6],[10]]
        warm_files=set()
        for group in groups:
            for mode in ('direct','replay','reverse','retained'):
                name='_'.join(map(str,group))+'_'+mode+'.json';warm_files.add(name);raw=read(path/'warmup_outputs'/name)
                order=group[::-1] if mode=='reverse' and phase in (1,2) else group
                if raw['users']!=order or len(raw['requests'])!=len(order):raise ValueError('Warm/reverse/retained group identity differs')
                if len(order)==2:validate_request_group(raw['requests'],order,True,vocab)
                else:validate_request(raw['requests'][0],order[0],0,1,vocab)
                for row in raw['requests']:accept(row)
        if {p.name for p in (path/'warmup_outputs').glob('*.json')}!=warm_files:raise ValueError('Warm output inventory differs')
        for p in dispatch['profiles']:
            name=Path(p['output'])
            if name.is_absolute() or '..' in name.parts:raise ValueError('Profile output outside phase')
            raw=read(path/name);order=p['cohort_indices']
            if raw['users']!=order:raise ValueError('Profile output pair differs')
            validate_request_group(raw['requests'],order,phase in (1,2),vocab)
            for row in raw['requests']:accept(row)
        for obj in (w,s):
            mem=obj['memory']
            if not 0<mem['allocated_bytes']<=mem['reserved_bytes']<46*1024**3:raise ValueError('Invalid residency evidence')
        phases.append(dict(phase=phase,variant=variant,output_inventory=inventory,replay=counts,storage=storage,
            dispatch=observed_dispatch,stability_comparisons=comparisons,stability_exact=all(all(x.values()) for x in comparisons),
            setup=w,summary=s))
    if len(joined)!=24 or any(set(r['runs'])!={'serial_pair_tf32','beam_wave_tf32'} for r in joined.values()):
        raise ValueError('Incomplete paired user inventory')
    return dict(protocol=protocol,users=users,phases=phases,joined=[joined[k] for k in sorted(joined)],
                controls=all_controls,timing=compare_phases(timing))


def evaluate_evidence(e,code,release,live):
    if os.environ.get('SLURM_JOB_ID')!=live['job_id']:raise ValueError('Allocated quality evaluation only')
    prior=e['protocol']['baseline'];oldroot=Path(prior['remote_dir']);reviewpath=Path(prior['recovered_audit_remote_path'])
    if digest(reviewpath)!=prior['recovered_audit_sha256']:raise ValueError('Audited TF32 reference changed')
    review=read(reviewpath)
    if not review['recovery_audit_passed'] or review['original_run_id']!=prior['run_id'] or review['original_job_id']!=prior['job_id']:
        raise ValueError('Wrong audited TF32 baseline')
    raw=code/'baseline-raw-manifest.json'
    if digest(raw)!=prior['raw_manifest_sha256']:raise ValueError('Baseline raw manifest differs')
    hashes={x['path']:x['sha256'] for x in read(raw)['files']};old={}
    for phase in (1,2):
        name=f'outputs/phase_{phase}/requests.json'
        if digest(oldroot/name)!=hashes[name]:raise ValueError('Baseline requests changed')
        for row in read(oldroot/name):
            if row['variant']!='folded_tf32' or row['cohort_index'] in old:raise ValueError('Baseline user inventory differs')
            old[row['cohort_index']]=dict(row['run'],cohort_index=row['cohort_index'])
    if len(old)!=500:raise ValueError('Incomplete audited baseline')
    control=[compare_user_outputs(old[r['cohort_index']],r) for r in e['controls']]
    from .precision_quality_adapter import quality_inputs
    from .quality_frontier import evaluate
    # Explicit evaluator role mapping only; original phase/run records stay intact.
    quality_rows=[dict(r,runs={'folded_fp32':r['runs']['serial_pair_tf32'],'folded_tf32':r['runs']['beam_wave_tf32']}) for r in e['joined']]
    quality=evaluate(**quality_inputs(release,e['users'],quality_rows))
    valid=quality['token_repeat_stability'] and all(all(v==0 for v in row.values()) for row in quality['output_validity'].values())
    result=dict(schema='two-request-beam-wave-audit-v1',run_id=live['run_id'],job_id=live['job_id'],unique_users=4,paired_user_observations=24,
        control_exact=all(all(x.values()) for x in control),control_comparisons=control,
        stability_exact=all(p['stability_exact'] for p in e['phases']),dispatch_verified=all(p['dispatch']['verified'] for p in e['phases']),
        validity_passed=valid,quality=quality,timing=e['timing'],phases=e['phases'],protocol_sha256=PROTOCOL_SHA256,
        cross_variant_comparisons=[compare_user_outputs(r['runs']['serial_pair_tf32'],r['runs']['beam_wave_tf32']) for r in e['joined']],
        promotion=False,noninferiority_established=False,note='Four users, finite equal-length pairs; no population equivalence or saturation claim. Full integrity additionally requires runner/deadline/log closure.')
    return result


def audit(root,code,historical,release):
    if not os.environ.get('SLURM_JOB_ID'):raise ValueError('Allocated beam-wave audit only')
    # Validate original generation/recovery stages and this audit's shorter deadline.
    from .beam_wave_runner import validate_audit_start
    from .audit_precision_setup import validate_wrapper
    root=Path(root).resolve();code=Path(code).resolve()
    live=observe_allocation(Path(os.environ['OSCAR_RUN_DIR']),code,root)
    budget=validate_audit_start(root,live,time.monotonic())
    wrapper=validate_wrapper(root,live,phases=(0,1,2,3),child_timeout_s=1500)
    e=inspect_phases(root,code,Path(historical),Path(release),live)
    result=evaluate_evidence(e,code,Path(release),live)
    if time.monotonic()>=budget['effective_deadline_monotonic']:raise TimeoutError('Original work/audit deadline expired')
    result.update(integrity_audit_passed=True,deadline_validation=budget,wrapper=wrapper,allocation_identity=live['allocation_identity'])
    result['execution_audit_passed']=all(result[k] for k in ('integrity_audit_passed','control_exact','stability_exact','dispatch_verified','validity_passed'))
    result['eligible_for_full500']=result['execution_audit_passed'] and result['timing']['timing_screen_passed']
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('root','code','historical','release','output'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise ValueError('Preserve existing audit evidence')
    try:result=audit(args.root,args.code,args.historical,args.release);exit_code=0 if result['execution_audit_passed'] else 1
    except Exception as exc:result=dict(integrity_audit_passed=False,error=repr(exc),traceback=traceback.format_exc());exit_code=1
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    raise SystemExit(exit_code)


if __name__=='__main__':main()
