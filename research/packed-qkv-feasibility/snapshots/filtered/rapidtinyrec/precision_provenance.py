"""Untimed release verification and frozen paired-pilot protocol records."""
import hashlib
import json
import math
from pathlib import Path
import time

PROTOCOL = 'benchmarks/preparation/quality-frontier/precision-paired-staged-protocol.json'
PROTOCOL_SHA256 = '9bc657eb778ced7e77606e5ce92aaebb27e7811b568c832c0c254d46c5878e24'
PROFILE = 'precision-paired-staged-pilot'


def load_protocol(path):
    data = Path(path).read_bytes()
    assert hashlib.sha256(data).hexdigest() == PROTOCOL_SHA256, 'Frozen paired protocol changed'
    protocol = json.loads(data)
    assert protocol['proposed_profile'] == PROFILE
    return protocol


def verify_release(root, expected, digest):
    """Record the actual values from the existing pre-load reads, exactly once."""
    started = time.perf_counter()
    observed = {}
    for name, want in expected.items():
        observed[name] = digest(root / name)
        assert observed[name] == want, name
    return {'schema': 'precision-release-verification-v1',
            'observed_sha256': observed,
            'elapsed_s': time.perf_counter() - started,
            'boundary': 'before_tokenizer_and_model_load'}


def validate_release(record, expected):
    assert record['schema'] == 'precision-release-verification-v1'
    assert record['observed_sha256'] == expected
    assert record['boundary'] == 'before_tokenizer_and_model_load'
    assert math.isfinite(record['elapsed_s']) and record['elapsed_s'] > 0


def phase_record(protocol, phase, reused, users):
    selected = protocol['phases'][phase]
    assert selected['phase'] == phase
    assert reused == (phase >= 2)
    assert [u['cohort_index'] for u in users] == protocol['cohort_indices']
    assert sorted({u['prompt_tokens'] for u in users}) == protocol['prompt_lengths']
    return {'schema': 'precision-paired-staged-phase-v1', 'scope': PROFILE,
            'protocol_sha256': PROTOCOL_SHA256, 'phase': selected,
            'requests': protocol['requests_per_phase'], 'graphs': protocol['graphs_per_phase']}


def validate_phase(manifest, protocol, expected):
    assert manifest['experiment'] == 'precision_paired_staged_pilot'
    phase = manifest['phase']
    assert type(phase) is int and phase in range(4)
    assert manifest['phase_protocol'] == phase_record(
        protocol, phase, manifest['compiler_artifacts']['reused'], manifest['users'])
    assert manifest['variant'] == protocol['phases'][phase]['variant']
    assert manifest['half'] == protocol['phases'][phase]['half']
    assert manifest['api_log_staging'] is True
    assert expected == protocol['provenance']['release_hashes']
    validate_release(manifest['release_verification'], expected)
