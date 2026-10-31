"""Explicit QKV arm mapping into the original paired SID quality evaluator."""
import os
from .audit_qkv_outputs import ARMS,USERS,require


def quality_observations(phases):
    require(set(phases)==set(range(4)),'Four quality partitions required')
    result=[]
    for phase in range(4):
        rows=phases[phase]
        expected={(r,i) for r in range(phase//2*3,phase//2*3+3) for i in USERS}
        require(len(rows)==12 and {(r['repeat'],r['cohort_index']) for r in rows}==expected,'Quality partition coverage differs')
        for row in rows:
            i=row['cohort_index']
            require(row['variant']==ARMS[phase] and row['prompt_tokens']==USERS[i],'Quality arm/user/length differs')
            require(row['run']['cohort_index']==i and row['run']['prompt_tokens']==USERS[i],'Quality output identity differs')
            result.append(dict(cohort_index=i,repeat=row['repeat'],variant='control' if phase in (0,3) else 'candidate',token_ids=row['run']['token_ids']))
    require(len(result)==48,'Expected24 paired quality requests')
    return result


def evaluate_allocated(release,users,phases,allocation):
    # The complete auditor must first authenticate this allocation, hashes,
    # live scheduler, all phases and full output evidence; env alone is not proof.
    require(allocation['submission']=='allocated_observation' and bool(os.environ.get('SLURM_JOB_ID'))
            and os.environ['SLURM_JOB_ID']==allocation['job_id'],'Allocated quality evaluation only')
    from .precision_quality_adapter import quality_inputs
    from .quality_frontier import evaluate
    # This existing resource parser independently hashes catalog/tokenmap/heldout
    # files. No old experiment's arm names are used for new observations.
    inputs=quality_inputs(release,users,[])
    require(inputs['observations']==[] and len(inputs['users'])==4,'Quality cohort resource preparation differs')
    inputs['observations']=quality_observations(phases)
    result=evaluate(**inputs)
    return dict(quality=result,arms=dict(control='original_bf16',candidate='packed_qkv_bf16'),
                independent_users=4,paired_requests=24,source_resources_verified=True,
                validity_passed=result['token_repeat_stability'] and all(all(v==0 for v in arm.values()) for arm in result['output_validity'].values()),
                population_equivalence_established=False,promotion=False)
