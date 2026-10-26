"""Audit two TF32 setup-feasibility phases; never a paired quality/promotion gate."""
import argparse
import json
from pathlib import Path
import traceback
from .audit_precision import inspect_phases, read, digest, compare, exact, latency

PROTOCOL = 'benchmarks/preparation/quality-frontier/setup-feasibility-protocol.json'


def diagnostic_decision(stable, prior_exact, dispatch, normal_exit):
    return {'setup_diagnostic_passed': all((stable,prior_exact,dispatch,normal_exit)),
            'eligible_for_full_quality_review': False, 'promotion': False,
            'noninferiority_established': False,
            'full_abba_retry_authorized_by_diagnostic_alone': False}


def validate_wrapper(root, record, *, phases=(1,2), child_timeout_s=1620):
    """Require normal bounded exit and recovery of the exact staged phase logs."""
    assert phases in ((1,2),(0,1,2,3))
    assert child_timeout_s in (1500,1620)
    if child_timeout_s==1500:assert phases==(0,1,2,3)
    start=read(root/'log-wrapper-start.json'); finish=read(root/'log-wrapper-exit.json')
    assert start['schema']=='precision-log-wrapper-v1'
    assert str(start['job_id'])==str(record['job_id'])
    assert start['child_timeout_s']==child_timeout_s
    assert start['filesystem']['filesystem'] in ('tmpfs','ext4','xfs')
    child=finish['child']; recovery=finish['recovery']
    assert child['schema']=='precision-child-exit-v1' and recovery['schema']=='precision-log-recovery-v1'
    assert child['returncode']==0 and child['reason']=='exited' and child['completed_normally'] is True
    assert child['signal'] is None and 0<child['elapsed_s']<=child_timeout_s
    assert recovery['recovery_complete'] is True and not recovery['leftover_local_entries']
    assert len(recovery['locations'])==len(phases) and all(x['recovered'] is True for x in recovery['locations'])
    assert 'supervisor_error' not in finish
    recovered={x['location']:x for x in recovery['locations']}
    assert len(recovered)==len(phases)
    local_dirs=[]
    for phase in phases:
        m=read(root/f'phase_{phase}'/'manifest.json'); assert m['api_log_staging'] is True
        location=read(root/f'phase_{phase}'/'log-staging-location.json')
        assert location['schema']=='precision-log-location-v1'
        output=Path(location['output_directory'])
        assert output.name==f'phase_{phase}'
        assert output.parent==Path(read(root/f'phase_{phases[0]}'/'log-staging-location.json')['output_directory']).parent
        item=recovered[str(output/'log-staging-location.json')]
        publication=read(root/f'phase_{phase}'/'log-staging.json')['publication']
        assert item['publication']['files']==publication['files']
        assert item['publication']['filesystem']==location['filesystem']
        assert item['publication']['complete'] is True
        assert publication['filesystem']==location['filesystem']
        local_dirs.append(location['local_directory'])
        assert str(Path(location['local_directory']).parent)==start['local_directory']
        assert location['filesystem']==start['filesystem']
    assert len(set(local_dirs))==len(phases)
    return finish


def audit(root, historical, snapshot, run_record, prior_root, prior_manifest, prior_review):
    record=read(run_record)
    assert record['profile']=='precision-setup-feasibility'
    for name in (PROTOCOL,'rapidtinyrec/audit_precision_setup.py','rapidtinyrec/precision_log_runner.py',
                 'rapidtinyrec/benchmark_precision.py','jobs/precision-setup.sh'):
        assert name in record['sha256'],name
    assert digest(Path(__file__))==record['sha256']['rapidtinyrec/audit_precision_setup.py']
    protocol=read(snapshot.parent/PROTOCOL)
    assert protocol['schema']=='precision-setup-feasibility-v1'
    assert [p['phase'] for p in protocol['phases']]==[1,2]
    assert all(p['variant']=='folded_tf32' for p in protocol['phases'])
    assert protocol['supervisor_child_deadline_s']==1620
    assert set(p.name for p in root.glob('phase_*') if p.is_dir())=={'phase_1','phase_2'}
    evidence=inspect_phases(root,historical,snapshot,run_record,setup_diagnostic=True)
    assert evidence['full'] is False
    assert [u['cohort_index'] for u in evidence['users']]==protocol['cohort_indices']
    assert evidence['lengths']==protocol['prompt_lengths']
    finish=validate_wrapper(root,record)
    reference=protocol['prior_tf32_reference']
    assert digest(prior_manifest)==reference['raw_manifest_sha256']
    assert digest(prior_review)==reference['diagnostic_review_sha256']
    review=read(prior_review)
    assert review['run_id']==reference['run_id'] and review['partial_validity_passed'] is True
    hashes={r['path']:r['sha256'] for r in read(prior_manifest)['files']}
    def prior(name):
        p=prior_root/name; assert digest(p)==hashes[name],name
        return read(p)
    old_manifest=prior('outputs/phase_1/manifest.json')
    assert old_manifest['variant']=='folded_tf32' and old_manifest['users']==evidence['users']
    old_rows=prior('outputs/phase_1/requests.json')
    old={}
    for row in old_rows:
        user=row['cohort_index']
        assert row==prior(f"outputs/phase_1/users/{row['repeat']}_{user:04d}.json")
        old.setdefault(user,row['run'])
    assert len(old_rows)==12 and set(old)==set(protocol['cohort_indices'])
    rows=[evidence['joined'][k] for k in sorted(evidence['joined'])]
    assert len(rows)==24 and all(set(r['runs'])=={'folded_tf32'} for r in rows)
    prior_pairs=[(old[r['cohort_index']],r['runs']['folded_tf32']) for r in rows]
    prior_comparison=compare(prior_pairs)
    prior_ancestry=all(a['beam_indices']==b['beam_indices'] for a,b in prior_pairs)
    repeated=compare(evidence['repeated'])
    phases=evidence['phases']
    stable=exact(repeated) and evidence['ancestry_stable'] and all(exact(p[k]) for p in phases
        for k in ('repeat_comparison','profile_comparison','warmup_comparison'))
    dispatch=all(p['dispatch']['verified'] for p in phases)
    result={'schema':'precision-setup-feasibility-audit-v1','integrity_audit_passed':True,
        'experiment':'setup_feasibility_only','run_id':record['run_id'],'job_id':record['job_id'],
        'protocol_sha256':digest(snapshot.parent/PROTOCOL),'audit_sha256':digest(Path(__file__)),
        'source_files_verified':len(record['files']),'phases':phases,'unique_users':4,
        'requests':24,'prior_tf32_comparison':prior_comparison,'prior_tf32_ancestry_exact':prior_ancestry,
        'within_policy_comparison':repeated,'within_policy_ancestry_stable':evidence['ancestry_stable'],
        'validity_supported_by_exact_prior_outputs':exact(prior_comparison) and prior_ancestry,
        'historical_fp32_fidelity':compare(evidence['historical_pairs']['folded_tf32']),
        'dispatch_verified':dispatch,'wrapper':finish,
        'latency_ms_descriptive_only':latency(rows,('folded_tf32',)),
        'latency_by_length_ms_descriptive_only':{str(n):latency([r for r in rows if r['prompt_tokens']==n],('folded_tf32',)) for n in evidence['lengths']},
        'latency_by_user_ms_descriptive_only':{str(u['cohort_index']):latency([r for r in rows if r['cohort_index']==u['cohort_index']],('folded_tf32',)) for u in evidence['users']},
        'interpretation':'Only TF32 cold/reuse setup feasibility. No paired FP32 speed/quality estimate, no logging-causality claim against historical runs, no completed ABBA or automatic full500/promotion. Compiler/capture timings nest in setup. Review full paired-run budget separately.'}
    result.update(diagnostic_decision(stable,exact(prior_comparison) and prior_ancestry,dispatch,True))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    for name in ('historical','snapshot','run-record','prior-root','prior-manifest','prior-review','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    try:
        result=audit(args.root,args.historical,args.snapshot,args.run_record,args.prior_root,args.prior_manifest,args.prior_review)
    except Exception as error:
        result={'schema':'precision-setup-feasibility-audit-v1','integrity_audit_passed':False,
            'error':repr(error),'traceback':traceback.format_exc(),**diagnostic_decision(False,False,False,False)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))
    if not result.get('integrity_audit_passed'):raise SystemExit(1)


if __name__=='__main__':main()
