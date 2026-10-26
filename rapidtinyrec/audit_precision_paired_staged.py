"""Complete paired ABBA audit plus frozen protocol, release and recovery evidence."""
import argparse
import json
from pathlib import Path
import traceback

from .audit_precision import audit as paired_audit, digest, read
from .audit_precision_setup import validate_wrapper
from .precision_provenance import PROTOCOL, PROFILE, PROTOCOL_SHA256, load_protocol, validate_phase


def validate_provenance(root, snapshot, record):
    assert record['profile'] == PROFILE
    assert record['resources']['command'] == ['bash', 'jobs/precision-paired-staged.sh']
    assert record['resources']['constraint'] == 'ampere'
    protocol_path = snapshot.parent / PROTOCOL
    assert record['sha256'][PROTOCOL] == PROTOCOL_SHA256 == digest(protocol_path)
    protocol = load_protocol(protocol_path)
    for key, value in protocol['resources'].items():
        assert record['resources'][key] == value, key
    # Every executable audit dependency must be from this submitted snapshot.
    for path in Path(__file__).parent.glob('*.py'):
        name = 'rapidtinyrec/' + path.name
        assert record['sha256'][name] == digest(path), name
    script = 'jobs/precision-paired-staged.sh'
    assert record['sha256'][script] == digest(snapshot.parent / script)
    assert set(p.name for p in root.glob('phase_*') if p.is_dir()) == {f'phase_{i}' for i in range(4)}
    expected = read(snapshot / 'expected.json')['release_sha256']
    observed = []
    for phase in range(4):
        manifest = read(root / f'phase_{phase}' / 'manifest.json')
        assert manifest['phase'] == phase
        validate_phase(manifest, protocol, expected)
        observed.append(manifest['release_verification']['observed_sha256'])
    assert all(x == observed[0] for x in observed)
    finish = validate_wrapper(root, record, phases=(0,1,2,3))
    return {'protocol_sha256': PROTOCOL_SHA256, 'release_sha256': observed[0], 'wrapper': finish}


def audit(root, historical, snapshot, run_record, release):
    record = read(run_record)
    provenance = validate_provenance(root, snapshot, record)
    result = paired_audit(root, historical, snapshot, run_record, release, staged_pilot=True)
    assert result['unique_users'] == 4 and result['paired_requests'] == 24
    result.update(schema='precision-paired-staged-audit-v1', run_id=record['run_id'],
                  job_id=record['job_id'], provenance=provenance,
                  staged_audit_sha256=digest(Path(__file__)))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root', type=Path)
    for name in ('historical','snapshot','run-record','release-root','output'):
        ap.add_argument('--' + name, type=Path, required=True)
    args = ap.parse_args()
    assert not args.output.exists(), 'Never overwrite audit evidence'
    try:
        result = audit(args.root,args.historical,args.snapshot,args.run_record,args.release_root)
    except Exception as exc:
        result = dict(schema='precision-paired-staged-audit-v1',integrity_audit_passed=False,
                      error=repr(exc),traceback=traceback.format_exc(),
                      eligible_for_full_quality_review=False,promotion=False,noninferiority_established=False)
    with args.output.open('x') as file:
        json.dump(result,file,indent=2,allow_nan=False); file.write('\n')
    print(json.dumps({k:result.get(k) for k in ('integrity_audit_passed','eligible_for_full_quality_review','error')},indent=2))
    if not result.get('eligible_for_full_quality_review'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
