"""Distinct BF16 full500 protocol checks. No runner or submission entry point."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
PROTOCOL = 'benchmarks/preparation/quality-frontier/bf16-full500-protocol.json'
PROTOCOL_SHA256 = '206890755848e417a4fb31b88598c352e6acd75083b7ccd1f96ec3fef521641f'
PROFILE = 'bf16-vs-tf32-full500-quality'
PHASES = ('folded_tf32','folded_bf16','folded_bf16','folded_tf32')

def validate_protocol(p):
    assert p['schema']=='bf16-vs-tf32-full500-protocol-v1' and p['profile']==PROFILE
    assert p['cohort']['indices']==list(range(500)) and p['cohort']['unique_users']==500
    assert len(p['cohort']['lengths'])==10 and len(p['phases'])==4
    for i,s in enumerate(p['phases']):
        assert s['phase']==i and s['half']==i//2 and s['variant']==PHASES[i]
        assert s['primary_cohort_indices']==list(range(i//2,500,2))
        assert s['primary_requests']==s['untimed_repeat_requests']==250 and s['graphs']==40
        assert s['untimed_repeat_order']=='reverse-primary'
        assert s['disk_cache']==('cold' if i<2 else 'same-variant-reuse')
        assert all(s[k] is True for k in ('fresh_process','fresh_allocations','fresh_graphs'))
    resources=p['provenance']['resource_hashes']
    assert type(resources) is dict and len(resources)==5
    assert all(type(k) is str and type(v) is str and len(v)==64 and all(c in '0123456789abcdef' for c in v) for k,v in resources.items())
    assert p['promotion'] is False and p['noninferiority_established'] is False
    b=p['budget'];assert b['global_work_deadline_s']==1620 and b['generation_child_deadline_s']==1500 and b['audit_maximum_s']==120 and b['nominal_cleanup_s']==180
    assert b['planned_total_s']==b['planned_generation_setup_s']+b['planned_audit_s']<=1620
    assert p['quality']['bootstrap_execution']=='Allocated compute only'
    assert p['quality']['bootstrap_samples']==10000 and p['quality']['loss_budget'] is None
    return p

def load_protocol(path):
    raw=Path(path).read_bytes();assert hashlib.sha256(raw).hexdigest()==PROTOCOL_SHA256
    return validate_protocol(json.loads(raw))

def phase_record(p,phase,reused,users):
    validate_protocol(p)
    assert type(phase) is int and phase in range(4)
    assert type(reused) is bool and reused==(phase>=2)
    assert [u['cohort_index'] for u in users]==p['cohort']['indices']
    assert hashlib.sha256(json.dumps(users,sort_keys=True).encode()).hexdigest()==p['cohort']['users_sha256']
    assert sorted({u['prompt_tokens'] for u in users})==p['cohort']['lengths']
    return dict(schema='bf16-full500-phase-v1',scope=PROFILE,protocol_sha256=PROTOCOL_SHA256,phase=deepcopy(p['phases'][phase]),requests=250,untimed_repeats=250,graphs=40,quality_unit='one primary paired observation per unique user')


def select_users(reference_users,protocol):
    """Retain every original user; warm one per length without dropping strata."""
    phase_record(protocol,0,False,reference_users)
    return reference_users,[next(u for u in reference_users if u['prompt_tokens']==length)
                           for length in protocol['cohort']['lengths']]


def primary_trials(users,protocol,phase):
    phase_record(protocol,phase,phase>=2,users)
    by_index={u['cohort_index']:u for u in users}
    return [(0,by_index[i]) for i in protocol['phases'][phase]['primary_cohort_indices']]


def collect_repeats(users,protocol,phase,generate,unpack,synchronize,save):
    """Reuse only generic collection mechanics, under this distinct frozen schedule."""
    from .precision_full500 import collect_untimed_repeats
    trials=primary_trials(users,protocol,phase)
    return collect_untimed_repeats(users,trials,PHASES[phase],generate,unpack,synchronize,save)


def validate_phase(manifest,protocol):
    from .precision_provenance import validate_release
    assert manifest['experiment']=='bf16_vs_tf32_full500'
    phase=manifest['phase']
    assert manifest['phase_protocol']==phase_record(protocol,phase,manifest['compiler_artifacts']['reused'],manifest['users'])
    assert manifest['variant']==PHASES[phase] and manifest['half']==phase//2
    assert manifest['api_log_staging'] is True
    validate_release(manifest['release_verification'],protocol['provenance']['release_hashes'])
    validate_release(manifest['quality_resource_verification'],protocol['provenance']['resource_hashes'])


def validate_repeats(primary,repeats,phase_spec):
    from .precision_full500 import validate_repeats as validate_records
    assert all(r.get('measurement_role')=='primary_timed' for r in primary)
    result=validate_records(primary,repeats,phase_spec)
    return dict(result,schema='bf16-full500-untimed-repeats-v1')
