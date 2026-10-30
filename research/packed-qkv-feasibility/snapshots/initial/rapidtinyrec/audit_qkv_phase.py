"""Independent phase evidence audit components for the allocated QKV auditor.

Caller supplies config, catalog and cohort from authenticated original files.
This is not the complete allocation/dispatch/historical/quality audit entrypoint.
"""
import json
from pathlib import Path
from .qkv_inventory_schema import expected_schema, validate_snapshot, validate_unchanged, validate_packing
from .qkv_storage import validate_stages
from .qkv_views import validate_coverage
from .audit_qkv_outputs import ARMS, require, audit_phase_outputs


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def storage_owners(record):
    result = set()
    def visit(value):
        if isinstance(value, dict):
            if 'storage_ptr' in value:
                result.add(value['storage_ptr'])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(record)
    return result


def validate_request_copies(path, requests):
    path = Path(path)
    expected_names = {f"{r['repeat']}_{r['cohort_index']:04d}.json" for r in requests}
    require(len(expected_names) == len(requests), 'Duplicate request file identity')
    require({p.name for p in path.iterdir()} == expected_names, 'Per-request file coverage differs')
    for row in requests:
        saved = read(path / f"{row['repeat']}_{row['cohort_index']:04d}.json")
        # Python float equality equates -0.0 and +0.0; serialized copies must
        # retain the score sign bit as well as the other JSON scalar values.
        require(json.dumps(saved, sort_keys=True, allow_nan=False) == json.dumps(row, sort_keys=True, allow_nan=False), 'Per-request copy differs')


def audit_phase_evidence(path, phase, config, users, constraints, root_prefix, *, device='cuda:0'):
    path = Path(path)
    require(type(phase) is int and phase in range(4), 'Unknown phase')
    original_schema = expected_schema(config, users, constraints, root_prefix)
    schema = expected_schema(config, users, constraints, root_prefix, packed=phase in (1, 2))
    before = read(path / 'immutable-original.json'); installed = read(path / 'immutable-installed.json')
    validate_snapshot(before, original_schema); validate_snapshot(installed, schema)
    require(all(r['device'] == device for r in before.values()) and all(r['device'] == device for r in installed.values()), 'Immutable device differs')
    packing = read(path / 'packing.json')
    if phase in (1, 2):
        packing_check = validate_packing(packing['proof'], before, installed, original_schema, schema, config)
    else:
        require(packing['proof'] is None, 'Original arm contains packed proof')
        validate_unchanged(before, installed, schema)
        packing_check = dict(extra_bytes=0, original_identity_preserved=True)
    for stage in ('setup', 'measurement', 'profiles'):
        validate_unchanged(installed, read(path / f'immutable-after-{stage}.json'), schema)
    storage = [read(path / f'storage-{stage}.json') for stage in ('before-measurement', 'after-measurement', 'after-profiles')]
    require(all(r['variant'] == ARMS[phase] for r in storage), 'Storage arm differs')
    storage_check = validate_stages(storage, immutable_snapshot=installed, immutable_schema=schema,
                                   layers=config.num_hidden_layers, kv_heads=config.num_key_value_heads,
                                   head_dim=config.hidden_size // config.num_attention_heads,
                                   vocab_size=config.vocab_size, device=device)
    views = read(path / 'packed-eager-views.json')
    require(views['eager_only'] is True and views['compiled_layout_verified'] is False, 'Eager layout mislabeled as compiled proof')
    if phase in (1, 2):
        view_check = validate_coverage(views['records'], config, (76, 121), device=device)
    else:
        require(views['records'] == [], 'Original arm contains packed view observations')
        view_check = dict(observations=0, eager_only=True, compiled_layout_verified=False)
    requests = read(path / 'requests.json')
    validate_request_copies(path / 'users', requests)
    warmups = {p.stem: read(p) for p in (path / 'warmup_outputs').iterdir() if p.suffix == '.json'}
    require(all(p.suffix == '.json' for p in (path / 'warmup_outputs').iterdir()), 'Unexpected warmup file')
    profiles = {length: read(path / f'profile-output-{length}.json') for length in (76, 121)}
    output_check = audit_phase_outputs(phase, requests, warmups, profiles, users, config.vocab_size, constraints, root_prefix,
                      required_immutable={r['storage_ptr'] for r in installed.values()},
                      required_supplied=storage_owners(storage[0]),
                      required_graph_logits={g['logits']['storage_ptr'] for g in storage[0]['graphs']})
    return dict(phase=phase, arm=ARMS[phase], packing=packing_check, storage=storage_check,
                views=view_check, outputs=output_check, submission_ready=False,
                complete_protocol_certified=False,
                remaining=['Authenticated source/allocation/work/deadline/log closure',
                           'Native BF16 dispatch/generated wrapper/graph correlation',
                           'Historical BF16 and cross-phase control, paired quality/timing orchestration'])
