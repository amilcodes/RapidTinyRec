"""Reconstruct four serving phases from raw files without trusting online gates.

Layout per phase: setup/ (initializer), preparation.json (service owners),
setup-audit.json (online diagnostic), execution/ (run_prepared_phase output).
This evidence audit deliberately cannot authorize advancement: source/config,
live allocation, fresh process/cache identity and external deadline binding must
still be verified by the eventual allocated-run harness.
"""
import copy
import math
from pathlib import Path

from .audit_decode_pilot import compare, digest
from .audit_prefix_profile import exact
from .serving_graph_ownership import audit_graph_owners
from .serving_admission_audit import audit_admission
from .serving_storage import audit_lane_storage
from .serving_setup_audit import audit_setup, read_unique
from .serving_measurement_audit import ORDERS, audit_burst, audit_measurements
from .serving_invariants import audit_interval, integer
from .serving_overlap_audit import audit_overlap
from .serving_initialize import save_exclusive


class Evidence:
    """Reject linked/duplicate files and detect mutation across the full audit."""
    def __init__(self):
        self.hashes = {}
        self.inodes = {}
        self.trees = {}

    def register(self, path):
        path = Path(path).absolute()
        if path in self.hashes:
            return
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Non-regular evidence file: {path}')
        stat = path.stat(); identity = (stat.st_dev, stat.st_ino)
        if stat.st_nlink != 1 or identity in self.inodes:
            raise ValueError('Aliased evidence file')
        self.inodes[identity] = path
        self.hashes[path] = digest(path)

    def tree(self, root):
        root = Path(root).absolute()
        if root.is_symlink() or not root.is_dir():
            raise ValueError('Regular evidence directory required')
        paths = set()
        for path in root.rglob('*'):
            if path.is_symlink():
                raise ValueError('Linked evidence directory/file')
            if not path.is_dir():
                self.register(path); paths.add(path.relative_to(root).as_posix())
        self.trees[root] = paths
        return paths

    def read(self, path):
        self.register(path)
        return read_unique(path)

    def finish(self):
        for root, expected in self.trees.items():
            current = set()
            for path in root.rglob('*'):
                if path.is_symlink():
                    raise ValueError('Evidence tree changed while being audited')
                if not path.is_dir():
                    current.add(path.relative_to(root).as_posix())
            if root.is_symlink() or current != expected:
                raise ValueError('Evidence tree changed while being audited')
        for path, expected in self.hashes.items():
            stat = path.stat()
            if path.is_symlink() or stat.st_nlink != 1 or self.inodes.get((stat.st_dev, stat.st_ino)) != path or digest(path) != expected:
                raise ValueError('Evidence changed while being audited')
        return {str(p): h for p, h in sorted(self.hashes.items())}


def owners(record, lanes):
    rows = record['owners']
    if set(rows) != {str(i) for i in range(lanes)}:
        raise ValueError('Exact serialized owner set required')
    result = {i: rows[str(i)] for i in range(lanes)}
    graph = audit_graph_owners(result, 8)
    storage = audit_lane_storage({i: row['storage'] for i, row in result.items()})
    if any(row['identity']['device'] != 'cuda:0' for row in result.values()):
        raise ValueError('Single indexed CUDA device required')
    return result, dict(graph=graph, storage=storage)


def elapsed(row, start, end, seconds):
    a, z = integer(row[start]), integer(row[end]); value = row[seconds]
    if a >= z or type(value) not in (float, int) or not math.isfinite(value) or not math.isclose(value, (z-a)/1e9, rel_tol=1e-12, abs_tol=0):
        raise ValueError('Setup elapsed disagrees with stage clocks')
    return a, z


def audit_phase(root, historical, index, evidence):
    root, historical = Path(root), Path(historical)
    lanes = (1, 2, 2, 1)[index]
    files = evidence.tree(root)
    required = set()
    def read(name):
        required.add(name)
        return evidence.read(root/name)
    # Online verdicts are retained and hashed, but never used as gate inputs.
    preparation = read('preparation.json'); read('setup-audit.json')
    setup_owners, setup_ownership = owners(preparation, lanes)
    provenance = read('setup/initialization-provenance.json')
    users = provenance['users']
    if len(users) != 4 or len({u['cohort_index'] for u in users}) != 4 or [u['prompt_tokens'] for u in users] != [76, 76, 121, 121]:
        raise ValueError('Frozen four-user cohort required')
    setup = audit_setup(root/'setup', historical, setup_owners)
    stages = read('setup/resource-verification-stages.json')
    first, previous = elapsed(stages, 'initializer_started_ns', 'resources_verified_ns', 'elapsed_s')
    stage_reports = []
    for lane in range(lanes):
        prefix = f'setup/lane_{lane}/'
        stage = read(prefix+'initialization-stages.json')
        a, z = elapsed(stage, 'allocation_started_ns', 'allocation_finished_ns', 'allocation_s')
        w, e = elapsed(stage, 'warmup_started_ns', 'warmup_finished_ns', 'warmup_s')
        if type(stage['lane']) is not int or stage['lane'] != lane or previous > a or z > w:
            raise ValueError('Lane allocation/warmup must follow resource checks and be serialized')
        previous = e; stage_reports.append(stage)
        for user in users:
            for mode in ('dense', 'replay', 'repeat'):
                read(prefix+f"warmup/{user['cohort_index']}_{mode}.json")
        read(prefix+'warmup/retained_after_mixed.json')
        compilation = read(prefix+'compilation.json')
        if compilation['phase'] != 'measurement' or compilation['measurement_compilations'] != 0 or not compilation['events'] or compilation['backend_graphs'] != len(compilation['events']):
            raise ValueError('Frozen compiler report required')
        required.add(prefix+'compiler_times.txt')
        for n, event in enumerate(compilation['events']):
            if type(event['index']) is not int or event['index'] != n:
                raise ValueError('Compiler event indices differ')
            required.add(prefix+f'compiled_graphs/fx_{n:03d}.py')
    execution = 'execution/'
    start = read(execution+'phase-start.json')
    complete = read(execution+'phase-complete.json')
    if (type(start['phase']) is not int or start['phase'] != index or start['users'] != users or
            complete['schema'] != 'serving-phase-completion-v1' or complete['complete'] is not True or
            type(complete['phase']) is not int or complete['phase'] != index or
            complete['started_ns'] != start['started_ns'] or previous >= integer(start['started_ns']) or
            integer(complete['finished_ns']) <= start['started_ns'] or
            complete['primary_requests'] != 12 or complete['secondary_requests'] != (12 if lanes == 2 else 0) or
            complete['profile_requests'] != 4):
        raise ValueError('Complete ordered phase envelope required')
    admission = audit_admission(preparation['admission_events'], setup_owners, stage_reports,
                                stages['resources_verified_ns'], start['started_ns'])
    live_reports = []
    expected_live = copy.deepcopy(setup_owners)
    for row in expected_live.values():
        row['storage'] = [r for r in row['storage'] if r['role'] != 'retained']
    for name in ('before', 'measured', 'after'):
        live, report = owners(read(execution+f'ownership-{name}.json'), lanes)
        if live != expected_live:
            raise ValueError('Live ownership changed beyond explicit retained-output release')
        live_reports.append(report)
    last = None
    def interval(label, records):
        nonlocal last
        before = read(execution+label+'-before.json')
        after = read(execution+label+'-after.json')
        if before['started_ns'] <= start['started_ns'] or after['finished_ns'] >= complete['finished_ns']:
            raise ValueError('Snapshot outside phase envelope')
        for snapshot in (before, after):
            if [v['identity'] for v in snapshot['lanes']] != [expected_live[i]['identity'] for i in range(lanes)]:
                raise ValueError('Snapshot/ownership identity mismatch')
        if last is not None:
            # No hidden work, changed compiler state or peak resets between bursts.
            audit_interval(last, before, {i: [] for i in range(lanes)})
        else:
            for i, lane in enumerate(before['lanes']):
                report = evidence.read(root/f'setup/lane_{i}/compilation.json')
                if report['events'] != lane['compiler']['events']:
                    raise ValueError('Compiler report differs from frozen snapshots')
        derived = audit_interval(before, after, {i: [r for r in records if r['lane'] == i] for i in range(lanes)})
        last = after
        return before, after, derived
    phase = dict(phase=index, lane_count=lanes, concurrency=lanes, primary=[], secondary=[])
    interval_reports = []
    measured = {}
    for kind in ('primary', 'secondary'):
        if kind == 'secondary' and lanes == 1:
            continue
        for local, order in enumerate(ORDERS):
            repeat = index//2*3+local; label = f'{kind}-{repeat}'
            records = read(execution+label+'-records.json')
            ordered = [users[i] for i in order]
            clocks = audit_burst(records, ordered, lane_count=lanes, concurrency=lanes if kind == 'primary' else 1)
            if [r['request_id'] for r in records] != [f"{kind}:{repeat}:{u['cohort_index']}" for u in ordered]:
                raise ValueError('Frozen request IDs differ')
            _, _, derived = interval(label, records)
            read(execution+label+'-audit.json')
            interval_reports.append(dict(label=label, clocks=clocks, invariants=derived))
            phase[kind].append(dict(repeat=repeat, records=records))
            for r in records:
                measured.setdefault(r['result']['cohort_index'], r['result']['raw'])
    if read(execution+'measurements.json') != phase:
        raise ValueError('Duplicate measurement summary differs from raw burst files')
    profile_start = read(execution+'profile/profile-start.json')
    profile_records = read(execution+'profile/profile-records.json')
    previous_completion = max(r['completed_ns'] for r in phase['secondary' if lanes == 2 else 'primary'][-1]['records'])
    expected_markers = {f"serving_profile/lane_{pos%lanes}/user_{u['cohort_index']}": pos%lanes for pos, u in enumerate(users)}
    if (type(profile_start['phase']) is not int or profile_start['phase'] != index or
            profile_start['last_measurement_completion_ns'] != previous_completion or
            profile_start['expected_markers'] != expected_markers or profile_start['performance_sample'] is not False or
            profile_start['owners'] != [expected_live[i]['identity'] for i in range(lanes)] or
            integer(profile_start['started_ns']) <= previous_completion or
            profile_start['started_ns'] >= profile_records[0]['arrival_ns'] or
            [r['request_id'] for r in profile_records] != [str(u['cohort_index']) for u in users]):
        raise ValueError('Profile schedule/owner binding differs')
    before, _, derived = interval('profile', profile_records)
    if before['finished_ns'] >= profile_start['started_ns']:
        raise ValueError('Profile started before pre-profile snapshot completed')
    read(execution+'profile-invariants.json'); read(execution+'profile/profile-audit.json')
    profile_clocks = audit_burst(profile_records, users, lane_count=lanes, concurrency=lanes)
    overlap = audit_overlap(read(execution+'profile/trace.json'), expected_markers)
    pairs = [(measured[r['result']['cohort_index']], r['result']['raw']) for r in profile_records]
    profile_fidelity = compare(pairs)
    profile_ancestry = all(a['beam_indices'] == b['beam_indices'] for a, b in pairs)
    # Historical artifacts lack ancestry. Bind measurement/profile ancestry to
    # the new dense setup reference, including across fresh ABBA processes below.
    dense = {u['cohort_index']: evidence.read(root/f"setup/lane_0/warmup/{u['cohort_index']}_dense.json") for u in users}
    setup_pairs = [(dense[r['result']['cohort_index']], r['result']['raw'])
                   for kind in ('primary', 'secondary') for burst in phase[kind] for r in burst['records']]
    setup_pairs += [(dense[r['result']['cohort_index']], r['result']['raw']) for r in profile_records]
    setup_fidelity = compare(setup_pairs)
    setup_ancestry = all(a['beam_indices'] == b['beam_indices'] for a, b in setup_pairs)
    if files != required:
        raise ValueError(f'Exact phase file set required; missing={sorted(required-files)}, extra={sorted(files-required)}')
    return dict(phase=phase, users=users, dense=dense, setup=setup,
                setup_ownership=setup_ownership, live_ownership=live_reports,
                intervals=interval_reports, profile_invariants=derived,
                profile_overlap=overlap, profile_comparison=profile_fidelity,
                profile_numerical_passed=bool(profile_clocks['outputs_valid'] and exact(profile_fidelity) and profile_ancestry),
                setup_to_execution_comparison=setup_fidelity,
                setup_to_execution_passed=bool(exact(setup_fidelity) and setup_ancestry),
                resource_stages=stages, lane_stages=stage_reports, admission=admission,
                started_ns=first, finished_ns=complete['finished_ns'])


def audit_run(phase_roots, historical):
    """Read-only four-phase reconstruction. No CUDA or bootstrap is imported/run."""
    if len(phase_roots) != 4 or len({Path(p).resolve() for p in phase_roots}) != 4:
        raise ValueError('Four distinct ordered ABBA evidence roots required')
    evidence = Evidence(); historical = Path(historical)
    if historical.is_symlink() or (historical/'users').is_symlink():
        raise ValueError('Linked historical evidence directory')
    evidence.read(historical/'manifest.json')
    # Hash the same historical rows the pinned setup auditor will consume.
    manifest = read_unique(historical/'manifest.json')
    lengths = sorted({u['prompt_tokens'] for u in manifest['users']})
    users = [u for n in (lengths[0], lengths[-1]) for u in [v for v in manifest['users'] if v['prompt_tokens'] == n][:2]]
    history = {u['cohort_index']: evidence.read(historical/'users'/f"{u['cohort_index']:04d}.json")['runs']['compiled_gqa'] for u in users}
    reports = [audit_phase(root, historical, i, evidence) for i, root in enumerate(phase_roots)]
    if any(r['users'] != users for r in reports) or any(a['finished_ns'] >= z['started_ns'] for a, z in zip(reports, reports[1:])):
        raise ValueError('ABBA phases overlap or cohort changes')
    measurement = audit_measurements([r['phase'] for r in reports], users, history)
    cross = [(reports[0]['dense'][key], r['dense'][key]) for r in reports[1:] for key in history]
    cross_report = compare(cross)
    cross_passed = exact(cross_report) and all(a['beam_indices'] == b['beam_indices'] for a, b in cross)
    gates = dict(setup=all(r['setup']['passed'] for r in reports),
                 setup_to_execution=all(r['setup_to_execution_passed'] for r in reports),
                 cross_phase_setup=bool(cross_passed), measured_numerics=measurement['numerical_passed'],
                 performance=measurement['performance_passed'], profile_numerics=all(r['profile_numerical_passed'] for r in reports),
                 actual_gpu_overlap=all(reports[i]['profile_overlap']['gpu_overlap_proven'] for i in (1, 2)))
    hashes = evidence.finish()
    return dict(schema='serving-disk-evidence-audit-v1', evidence_gates_passed=all(gates.values()), gates=gates,
                phases=reports, measurements=measurement, cross_phase_setup=cross_report, evidence_sha256=hashes,
                online_verdicts_trusted=False, historical_ancestry_available=False, advancement_ready=False,
                remaining='Source/config/protocol, live allocation, fresh process/cache identity, external global deadline/recovery and harness binding remain required.')


def save_audit(phase_roots, historical, output):
    """Preserve a failed verdict or malformed-evidence failure without overwrite."""
    output = Path(output).resolve()
    if any(output.is_relative_to(Path(p).resolve()) for p in phase_roots) or output.is_relative_to(Path(historical).resolve()):
        raise ValueError('Audit output must be outside immutable evidence roots')
    if output.exists():
        raise FileExistsError(output)
    try:
        report = audit_run(phase_roots, historical)
    except Exception as error:
        save_exclusive(output, dict(schema='serving-disk-evidence-audit-v1', evidence_gates_passed=False,
                                   malformed=True, error=repr(error), advancement_ready=False))
        raise
    save_exclusive(output, report)
    return report
