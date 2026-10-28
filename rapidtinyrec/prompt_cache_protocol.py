"""Distinct fixed-prompt-cache pilot identity and fail-closed phase metadata.

This validates the declared protocol and inventory, not numerical correctness.
The independent allocated auditor must still inspect raw outputs and runtime
evidence. No serving/BF16 validator is satisfied by relabeling these phases.
"""
import hashlib
import json
import math
from pathlib import Path

PROTOCOL = 'benchmarks/preparation/quality-frontier/prompt-cache-pilot-protocol.json'
PROTOCOL_SHA256 = 'a19324cbd8c0b990f2f84f64c1332f7f7f5e7e901b184a10b96e30acd6680fca'
PHASES = ('uncached_tf32', 'fixed69_tf32', 'fixed69_tf32', 'uncached_tf32')


def load_protocol(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise ValueError('Prompt-cache protocol hash differs')
    protocol = json.loads(raw)
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol):
    if (protocol['schema'] != 'fixed-prompt-cache-pilot-v1'
            or protocol['experiment'] != 'fixed_prompt_cache_tf32_pilot'
            or protocol['profile'] != 'fixed-prompt-cache-pilot'):
        raise ValueError('Distinct prompt-cache identity required')
    if protocol['phases'] != [dict(phase=i, variant=v, half=i//2,
                                  reuse_disk_artifacts=i>=2) for i,v in enumerate(PHASES)]:
        raise ValueError('Fresh-process uncached/cached/cached/uncached ABBA required')
    prefix = protocol['prefix_token_ids']
    if (len(prefix) != protocol['prefix_tokens'] or len(prefix) != 69
            or any(type(x) is not int or x < 0 for x in prefix)
            or hashlib.sha256(json.dumps(prefix,separators=(',', ':')).encode()).hexdigest()
                != protocol['prefix_token_sha256']):
        raise ValueError('Frozen 69-token identity differs')
    if (protocol['cohort_indices'] != [1,2,6,10] or protocol['prompt_lengths'] != [76,121]
            or protocol['requests_per_phase'] != 12 or protocol['graphs_per_phase'] != 8
            or protocol['repeats_per_half'] != 3):
        raise ValueError('Pilot cohort or request scope differs')
    budget = protocol['budget']
    if (budget['global_work_deadline_s'] != 1620
            or budget['generation_child_deadline_s'] != 1500
            or budget['audit_maximum_s'] != 120 or budget['nominal_cleanup_s'] != 180):
        raise ValueError('Work/audit/cleanup deadline contract differs')
    if protocol['promotion'] or protocol['noninferiority_established']:
        raise ValueError('Exploratory pilot cannot promote or establish noninferiority')


def phase_record(protocol, phase, reused, users):
    validate_protocol(protocol)
    if type(phase) is not int or phase not in range(4) or type(reused) is not bool:
        raise ValueError('Invalid phase identity')
    if reused != (phase >= 2):
        raise ValueError('Disk cache reuse does not match ABBA phase')
    if ([u['cohort_index'] for u in users] != protocol['cohort_indices']
            or [u['prompt_tokens'] for u in users] != [76,76,121,121]
            or any(len(u['prompt_token_ids']) != u['prompt_tokens']
                   or u['prompt_token_ids'][:69] != protocol['prefix_token_ids'] for u in users)):
        raise ValueError('Wrong cohort, length or fixed prompt prefix')
    return dict(schema='fixed-prompt-cache-phase-v1', protocol_sha256=PROTOCOL_SHA256,
        phase=protocol['phases'][phase], requests=12, graphs=8,
        prefix_tokens=69 if phase in (1,2) else 0,
        prefix_token_sha256=protocol['prefix_token_sha256'],
        suffix_lengths=[7,52] if phase in (1,2) else None,
        measurement='warm_generation_including_request_kv_copy')


def validate_request_inventory(protocol, phase, rows):
    """Check a phase's raw row identity without accepting its numerical results."""
    validate_protocol(protocol)
    if type(phase) is not int or phase not in range(4):
        raise ValueError('Invalid phase identity')
    half = phase//2
    expected = {(repeat, user) for repeat in range(half*3,half*3+3)
                for user in protocol['cohort_indices']}
    observed = [(r['repeat'], r['cohort_index']) for r in rows]
    if len(observed) != 12 or len(set(observed)) != 12 or set(observed) != expected:
        raise ValueError('Incomplete, duplicate or wrong-half request inventory')
    lengths = dict(zip(protocol['cohort_indices'], [76,76,121,121]))
    if any(r['variant'] != PHASES[phase] or r['prompt_tokens'] != lengths[r['cohort_index']]
           for r in rows):
        raise ValueError('Request variant or length differs from phase')


def audit_timeout(remaining_s):
    """Audit reservation is never an extension of the global work deadline."""
    if type(remaining_s) not in (int, float) or not math.isfinite(remaining_s):
        raise ValueError('Finite remaining work budget required')
    if remaining_s <= 0:
        raise TimeoutError('Global work deadline expired before audit')
    return min(120., remaining_s)
