"""Frozen full-cohort provenance and untimed-repeat checks; no inference entry point.

These checks are preparation building blocks. The allocated runner, source/run
identity reconciliation and complete audit must exist before submission.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .precision_provenance import validate_release
from .audit_precision import validate_output_storage, validate_run
from .audit_decode_pilot import compare
from .audit_prefix_profile import exact

PROTOCOL = 'benchmarks/preparation/quality-frontier/precision-full500-protocol.json'
PROTOCOL_SHA256 = '293d6a00d0c573dbb250a2bfa94d956b1b5c77fa17316da3726de9451c6dd654'
PROFILE = 'precision-full500-quality'


def load_protocol(path):
    raw = Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PROTOCOL_SHA256, 'Frozen full500 protocol changed'
    protocol = json.loads(raw)
    assert protocol['proposed_profile'] == PROFILE
    assert protocol['cohort']['indices'] == list(range(500))
    assert protocol['promotion'] is False and protocol['noninferiority_established'] is False
    return protocol


def validate_scope(*, full_cohort, full500_quality, paired_staged_pilot,
                   setup_feasibility, api_log_local_root):
    """Fail closed before CUDA initialization when the future scope is selected."""
    if full500_quality:
        assert full_cohort and not paired_staged_pilot and not setup_feasibility
        assert api_log_local_root is not None, 'Full500 requires staged logging'
    else:
        assert not full_cohort, 'Full-cohort execution requires explicit frozen quality scope'


def phase_record(protocol, phase, reused, users):
    assert type(phase) is int and phase in range(4)
    assert type(reused) is bool and reused == (phase >= 2)
    assert [u['cohort_index'] for u in users] == protocol['cohort']['indices']
    assert hashlib.sha256(json.dumps(users, sort_keys=True).encode()).hexdigest() == protocol['cohort']['users_sha256']
    assert sorted({u['prompt_tokens'] for u in users}) == protocol['cohort']['lengths']
    selected = protocol['phases'][phase]
    assert selected['phase'] == phase and selected['half'] == phase // 2
    assert selected['primary_cohort_indices'] == list(range(phase // 2, 500, 2))
    assert selected['primary_requests'] == selected['untimed_repeat_requests'] == 250
    assert selected['graphs'] == 40
    return {'schema': 'precision-full500-phase-v1', 'scope': PROFILE,
            'protocol_sha256': PROTOCOL_SHA256, 'phase': deepcopy(selected),
            'requests': 250, 'untimed_repeats': 250, 'graphs': 40,
            'quality_unit': 'one primary paired observation per unique user'}


def validate_phase(manifest, protocol, expected):
    assert manifest['experiment'] == 'precision_full500_quality'
    phase = manifest['phase']
    assert manifest['phase_protocol'] == phase_record(
        protocol, phase, manifest['compiler_artifacts']['reused'], manifest['users'])
    assert manifest['variant'] == protocol['phases'][phase]['variant']
    assert manifest['half'] == phase // 2 and manifest['api_log_staging'] is True
    assert expected == protocol['provenance']['release_hashes']
    validate_release(manifest['release_verification'], expected)


def validate_repeats(primary, repeats, phase_spec):
    """Validate saved records and return drift without silently accepting it.

    Repeated records have no elapsed_s field: they cannot be mixed into latency.
    Original primary elapsed_s is used only to reuse score/shape validation.
    Structural corruption raises; valid but unstable values are preserved in the
    returned comparison with passed=False, including signed-zero FP32 drift.
    """
    indices = phase_spec['primary_cohort_indices']
    assert len(primary) == phase_spec['primary_requests'] == len(indices)
    assert len(repeats) == phase_spec['untimed_repeat_requests'] == len(indices)
    assert len(set(indices)) == len(indices)
    assert [r['cohort_index'] for r in primary] == indices
    assert [r['cohort_index'] for r in repeats] == list(reversed(indices))
    originals = {r['cohort_index']: r for r in primary}
    pairs = []
    ancestry_stable = True
    for repeat in repeats:
        original = originals[repeat['cohort_index']]
        assert original['variant'] == repeat['variant'] == phase_spec['variant']
        assert original['repeat'] == 0
        assert repeat['repeat'] == 1 and repeat['measurement_role'] == 'untimed_stability'
        assert repeat['prompt_tokens'] == original['prompt_tokens']
        assert 'elapsed_s' not in repeat['run'], 'Untimed repeat must not supply primary latency'
        validate_run(original['run'])
        validate_output_storage(original['run'])
        check = dict(repeat['run'], elapsed_s=original['run']['elapsed_s'])
        validate_run(check)
        validate_output_storage(check)
        pairs.append((original['run'], repeat['run']))
        ancestry_stable &= original['run']['beam_indices'] == repeat['run']['beam_indices']
    comparison = compare(pairs)
    return {'schema': 'precision-full500-untimed-repeats-v1',
            'primary_requests': len(primary), 'untimed_repeat_requests': len(repeats),
            'comparison': comparison, 'ancestry_stable': ancestry_stable,
            'passed': exact(comparison) and ancestry_stable,
            'included_in_primary_latency': False, 'additional_quality_users': 0}


def collect_untimed_repeats(users, trials, variant, generate, unpack, synchronize, save):
    """Execute the frozen reverse pass after primary summary publication.

    Injected callbacks let CPU tests verify order/lifetime without loading Torch.
    Every record is saved immediately, including drift, before summary auditing.
    """
    by_index = {u['cohort_index']: u for u in users}
    selected = [u['cohort_index'] for repeat, u in trials]
    assert all(repeat == 0 for repeat, _ in trials)
    assert len(selected) == len(set(selected))
    rows = []
    for index in reversed(selected):
        user = by_index[index]
        output = generate(user)
        synchronize()
        raw = unpack(output, user)
        del output
        row = dict(cohort_index=index, prompt_tokens=user['prompt_tokens'],
                   variant=variant, repeat=1, measurement_role='untimed_stability', run=raw)
        assert 'elapsed_s' not in raw
        save(row)
        rows.append(row)
    return rows
