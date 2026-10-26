"""CPU-only independent setup audit; no model execution or resampling.

Historical artifacts contain no beam ancestry. Ancestry is checked between all
new dense/replay/repeat/retained outputs, never claimed against history.
"""
import json
import math
import struct
from pathlib import Path

from .audit_decode_pilot import compare, digest, json_digest
from .audit_prefix_profile import exact
from .serving_storage import validate_inventory


def read_unique(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON evidence key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs)


def int_bits(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Finite FP32 scores required')
    return struct.unpack('!i', struct.pack('!f', value))[0]


def validate_raw(raw, *, historical=False):
    fields = ('token_ids', 'sequence_logp', 'transition_logp')
    if not historical:
        fields += ('sequence_bits', 'transition_bits', 'beam_indices')
    if any(not isinstance(raw[k], list) or len(raw[k]) != 50 for k in fields):
        raise ValueError('Exactly fifty ranks and score/ancestry rows required')
    for row in raw['token_ids']:
        if not isinstance(row, list) or len(row) != 5 or any(type(x) is not int or x < 0 for x in row):
            raise ValueError('Five integer output tokens required')
    for row in raw['transition_logp']:
        if not isinstance(row, list) or len(row) != 5:
            raise ValueError('Five transitions required')
    sequences = [int_bits(v) for v in raw['sequence_logp']]
    transitions = [[int_bits(v) for v in row] for row in raw['transition_logp']]
    if not historical:
        if (any(type(x) is not int for x in raw['sequence_bits']) or
                any(not isinstance(row, list) or len(row) != 5 or any(type(x) is not int for x in row)
                    for row in raw['transition_bits']) or
                sequences != raw['sequence_bits'] or transitions != raw['transition_bits']):
            raise ValueError('Explicit FP32 bits disagree with values')
        if any(not isinstance(row, list) or len(row) != 5 or
               any(type(x) is not int or not -1 <= x < 50 for x in row) for row in raw['beam_indices']):
            raise ValueError('Malformed five-step beam ancestry')
    sums = []
    for row in raw['transition_logp']:
        total = 0.
        for value in row:
            total = struct.unpack('!f', struct.pack('!f', total + value))[0]
        sums.append(int_bits(total))
    return (len({tuple(row) for row in raw['token_ids']}) == 50 and
            all(row[-2:] == [198, 151645] for row in raw['token_ids']) and sums == sequences)


def score_independence(record, retained_ids):
    """Require all retained original score storages distinct from all graph logits."""
    rows = record['storage']; validate_inventory(rows)
    by_name = {row['name']: row for row in rows}
    expected = {f'retained.{key}.{field}' for key in retained_ids
                for field in ('sequences', 'sequence_scores', 'beam_indices',
                              'score_0', 'score_1', 'score_2', 'score_3', 'score_4')}
    if {r['name'] for r in rows if r['role'] == 'retained' or r['name'].startswith('retained.')} != expected:
        raise ValueError('Complete named retained output inventory required')
    logits = [r for r in rows if r['role'] == 'graph_logits']
    if {r['name'] for r in logits} != {f'replay.{i}.output.logits' for i in range(8)}:
        raise ValueError('All eight graph logits required')
    scores = [by_name[f'retained.{key}.score_{i}'] for key in retained_ids for i in range(5)]
    for row in scores + logits:
        if row['device'] != record['identity']['device'] or not row['storage_bytes'] or not row['logical_bytes']:
            raise ValueError('Nonempty owner-device score/logit storage required')
    for i, left in enumerate(scores):
        for right in scores[i+1:] + logits:
            if max(left['address'], right['address']) < min(left['address']+left['storage_bytes'], right['address']+right['storage_bytes']):
                raise ValueError('Original HF score storages alias each other or graph logits')
    return dict(retained_outputs=len(retained_ids), independent_score_storages=len(scores), graph_logits=len(logits))


def audit_setup(root, historical, owners):
    """Audit initializer evidence and live owner snapshots before admission.

    Graph-pool and cross-lane storage audits are separately mandatory in the
    service lifecycle. Full source/run/allocation binding belongs to the harness.
    """
    root, historical = Path(root), Path(historical)
    if any(type(k) is not int for k in owners) or set(owners) not in ({0}, {0, 1}):
        raise ValueError('Explicit one/two-lane owner set required')
    expected = read_unique(Path(__file__).with_name('expected.json'))
    manifest = read_unique(historical/'manifest.json')
    if (manifest['experiment'] != 'gqa_three_way' or manifest['dtype'] != 'float32' or
            len(manifest['users']) != 500 or json_digest(manifest['users']) != expected['users_sha256'] or
            json_digest({k:v for k,v in manifest.items() if k != 'manifest_sha256'}) != manifest['manifest_sha256']):
        raise ValueError('Historical cohort/manifest identity mismatch')
    lengths = sorted({u['prompt_tokens'] for u in manifest['users']})
    users = [u for length in (lengths[0], lengths[-1])
             for u in [x for x in manifest['users'] if x['prompt_tokens'] == length][:2]]
    if len(users) != 4 or len({u['cohort_index'] for u in users}) != 4:
        raise ValueError('Four distinct frozen pilot users required')
    provenance = read_unique(root/'initialization-provenance.json')
    wanted = dict(users=users, observed_release_sha256=expected['release_sha256'],
                  reference_manifest_sha256=manifest['manifest_sha256'], torch_version='2.14.0+cu126',
                  torch_git='08187d9e0fba026dc8217405802ab5381dc88d90', transformers_version='4.55.4', tf32=False)
    runtime_keys = {'cuda_version', 'device_name', 'device_count', 'numerical_policy'}
    # Bare historical preparation records predate the runtime observation. A
    # complete allocated run requires these fields separately in its run audit.
    extra = set(provenance) - set(wanted)
    if extra and extra != runtime_keys:
        raise ValueError('Incomplete initializer runtime observation')
    if {k:v for k,v in provenance.items() if k not in runtime_keys} != wanted or provenance['tf32'] is not False:
        raise ValueError('Initializer provenance mismatch')
    if extra:
        from .serving_policy import validate_state
        if (provenance['cuda_version'] != '12.6' or provenance['device_name'] != 'NVIDIA RTX A6000' or
                type(provenance['device_count']) is not int or provenance['device_count'] != 1):
            raise ValueError('Initializer runtime/device mismatch')
        validate_state(provenance['numerical_policy'])
    retained_ids = [users[0]['cohort_index'], users[2]['cohort_index']]
    hashes = {}; history = {}; valid = True
    def read(path):
        hashes[str(path)] = digest(path)
        return read_unique(path)
    hashes[str(historical/'manifest.json')] = digest(historical/'manifest.json')
    hashes[str(root/'initialization-provenance.json')] = digest(root/'initialization-provenance.json')
    for user in users:
        key = user['cohort_index']; row = read(historical/'users'/f'{key:04d}.json')
        if any(row[k] != user[k] for k in ('cohort_index','source_row_index','user_id','prompt_tokens','target_sid')) or row['manifest_sha256'] != manifest['manifest_sha256']:
            raise ValueError('Historical row identity mismatch')
        history[key] = row['runs']['compiled_gqa']
        valid &= validate_raw(history[key], historical=True)
    historical_pairs = []; stability_pairs = []; ancestry = True; ownership = {}
    first_lane = {}; seen_files = set()
    for lane, record in owners.items():
        ownership[str(lane)] = score_independence(record, retained_ids)
        warm = root/f'lane_{lane}'/'warmup'
        if warm.is_symlink() or warm.parent.is_symlink():
            raise ValueError('Aliased warm evidence directory')
        names = {f"{u['cohort_index']}_{mode}.json" for u in users for mode in ('dense','replay','repeat')} | {'retained_after_mixed.json'}
        if {p.name for p in warm.iterdir()} != names or any(p.is_symlink() or not p.is_file() for p in warm.iterdir()):
            raise ValueError('Exact non-aliased warm evidence file set required')
        identities = {(p.stat().st_dev, p.stat().st_ino) for p in warm.iterdir()}
        if len(identities) != len(names) or identities & seen_files:
            raise ValueError('Aliased warm evidence files')
        seen_files.update(identities)
        retained = read(warm/'retained_after_mixed.json')
        if set(retained) != {str(k) for k in retained_ids}:
            raise ValueError('Retained user set mismatch')
        for user in users:
            key = user['cohort_index']; runs = [read(warm/f'{key}_{mode}.json') for mode in ('dense','replay','repeat')]
            if key in retained_ids:
                runs.append(retained[str(key)])
            for run in runs:
                valid &= validate_raw(run)
                historical_pairs.append((history[key], run))
            stability_pairs.extend((runs[0], run) for run in runs[1:])
            ancestry &= all(run['beam_indices'] == runs[0]['beam_indices'] for run in runs[1:])
            if key in first_lane:
                stability_pairs.append((first_lane[key], runs[0]))
                ancestry &= first_lane[key]['beam_indices'] == runs[0]['beam_indices']
            else:
                first_lane[key] = runs[0]
    h, s = compare(historical_pairs), compare(stability_pairs)
    return dict(schema='serving-setup-audit-v1', passed=bool(valid and exact(h) and exact(s) and ancestry),
                historical_comparison=h, stability_comparison=s, ancestry_stable=bool(ancestry),
                outputs_valid=bool(valid), score_ownership=ownership, evidence_sha256=hashes,
                historical_ancestry_available=False,
                scope='Setup only; measured timing/numerics/actual GPU overlap and source/allocation identity require separate audits.')


def setup_auditor(root, historical, output):
    """Mandatory callback for prepared_service; preserve failures exclusively."""
    def callback(owners):
        try:
            report = audit_setup(root, historical, owners)
        except Exception as error:
            report = dict(schema='serving-setup-audit-v1', passed=False, malformed=True, error=repr(error))
            with Path(output).open('x') as handle:
                json.dump(report, handle, indent=2, allow_nan=False)
            raise
        with Path(output).open('x') as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
        return report
    return callback
