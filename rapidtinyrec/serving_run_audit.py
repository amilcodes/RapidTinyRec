"""Pre-exit disk reconstruction of serving source, allocation and parent records.

This runs inside the supervised coordinator, before final-audit.json or either
process exit record exists. Terminal submission/exit reconciliation is separate.
"""
from pathlib import Path
import time

from .serving_allocation import CONFIG, digest, identical, require, validate_observation
from .serving_contract import PROTOCOL, load_protocol
from .serving_coordinator import reconcile_worker
from .serving_disk_audit import Evidence, audit_run
from .serving_worker_audit import integer, validate_identity
from .serving_policy import validate_state


def audit_pre_exit(run_root, historical):
    started=time.perf_counter_ns();run=Path(run_root).absolute();historical=Path(historical).absolute()
    require(run.is_dir() and not run.is_symlink(),'Regular run evidence root required')
    evidence=Evidence();code=run/'code';outputs=run/'outputs';coord=outputs/'coordinator';sup=outputs/'supervision'
    require(outputs.is_dir() and not outputs.is_symlink(),'Regular outputs directory required')
    prepared=evidence.read(run/'source-manifest.json');evidence.register(run/'job-id')
    manifest_bytes=(run/'source-manifest.json').read_bytes();job_bytes=(run/'job-id').read_bytes()
    source_files=evidence.tree(code)
    require(source_files==set(prepared['files'])==set(prepared['sha256']),'Exact uploaded source inventory required')
    for name in source_files:
        path=Path(name)
        require(path.suffix not in ('.pyc','.pyo') and '__pycache__' not in path.parts,'Unreviewed source bytecode')
        require(evidence.hashes[(code/name).absolute()]==prepared['sha256'][name],'Uploaded source bytes changed')
    require('rapidtinyrec/source/LogitProcessor.py' in source_files,'Nested source dependency omitted')
    from .serving_harness import validate_provenance
    validate_provenance(evidence.read(code/'source-provenance.json'),prepared['sha256'])
    protocol=load_protocol(code/PROTOCOL);config=evidence.read(code/CONFIG)
    expected=evidence.read(code/'rapidtinyrec/expected.json')
    require(expected['users_sha256']==protocol['cohort']['full_users_sha256'] and
            expected['release_sha256']==protocol['cohort']['release_sha256'],'Frozen resource identity differs')
    evidence.register(historical/'manifest.json')
    require(digest(historical/'manifest.json')==protocol['cohort']['historical_manifest_file_sha256'],'Historical manifest bytes differ')
    coord_files={'entry.json','allocation-observation.json'}
    for i in range(4):
        coord_files.update(f'phase_{i}-{tag}.json' for tag in ('spawn','exit','cache-before','cache-after','binding'))
        coord_files.add(f'phase_{i}.log')
    require(evidence.tree(coord)==coord_files,'Exact pre-exit coordinator file set required')
    require(evidence.tree(outputs/'launch')=={f'phase_{i}.json' for i in range(4)},'Exact launch ticket set required')
    # Only this open stdout file may change while the coordinator audits itself.
    # It is not a source of gates; terminal reconciliation must hash it after exit.
    require(sup.is_dir() and not sup.is_symlink() and {p.name for p in sup.iterdir()}==
            {'supervisor-start.json','coordinator-launch.json','coordinator.log'},'Exact pre-exit supervisor layout required')
    live_log=sup/'coordinator.log';require(live_log.is_file() and not live_log.is_symlink() and live_log.stat().st_nlink==1,
                                        'Regular live coordinator stdout required')
    live_identity=(live_log.stat().st_dev,live_log.stat().st_ino)
    outer=evidence.read(sup/'supervisor-start.json');observed=evidence.read(sup/'coordinator-launch.json')
    entry=evidence.read(coord/'entry.json');parent=entry['identity'];validate_identity(parent)
    require(outer['schema']=='serving-supervisor-start-v1' and identical(entry['supervisor_start'],outer) and
            identical(observed['identity'],parent),'Coordinator/supervisor raw identity differs')
    require(parent['pid']==observed['pid']==observed['process_group']==entry['process_group']==entry['session'] and
            parent['parent_pid']==outer['supervisor_pid'],'Coordinator parent/session binding differs')
    start=integer(outer['global_started_ns']);deadline=integer(outer['deadline_ns'])
    require(deadline-start==1620*10**9 and start<=integer(observed['launched_ns'])<=integer(entry['started_ns']) and
            observed['launched_ns']<=integer(observed['observed_ns'])<started<deadline,'Outer deadline/preflight order differs')
    allocation=evidence.read(coord/'allocation-observation.json')
    bound=validate_observation(allocation,manifest_bytes,job_bytes,protocol)
    require(identical(allocation['allocation_identity']['config'],config),'Actual config bytes differ from observation')
    require(entry['started_ns']<=allocation['allocation_identity']['started_ns']<=allocation['allocation_identity']['verified_ns'],
            'Coordinator allocation verification precedes process entry')
    roots=[outputs/f'phase_{i}' for i in range(4)]
    for root in roots:evidence.tree(root)
    phases=audit_run(roots,historical)
    # Adopt the lower reader's exact input digests and recheck through the end of
    # this outer audit; do not copy/filter evidence or trust its saved verdicts.
    for path,want in phases['evidence_sha256'].items():
        evidence.register(path);require(evidence.hashes[Path(path).absolute()]==want,'Phase evidence changed between readers')
    previous=allocation['allocation_identity']['verified_ns'];bindings=[];tokens=set();command_base=None
    for i,phase in enumerate(phases['phases']):
        worker=phase['worker'];require(worker is not None,'Complete run requires raw worker wrappers')
        require(phase['users']==protocol['cohort']['users'],'Run user cohort differs from frozen protocol')
        root=roots[i];ticket=evidence.read(outputs/'launch'/f'phase_{i}.json')
        runtime=evidence.read(root/'setup/initialization-provenance.json');pinned=protocol['runtime']
        require((runtime['torch_version'],runtime['torch_git'],runtime['transformers_version'],runtime['cuda_version'],runtime['device_name'])==
                (pinned['torch'],pinned['torch_git'],pinned['transformers'],pinned['cuda'],pinned['device']) and
                type(runtime['device_count']) is int and runtime['device_count']==1,'Observed pinned runtime/device differs')
        validate_state(runtime['numerical_policy'])
        require(identical(ticket,worker['launch']) and identical(ticket['parent_identity'],parent) and
                ticket['global_started_ns']==start and ticket['deadline_ns']==deadline and
                previous<integer(ticket['launch_started_ns']),'Phase launch/parent/global clock differs')
        require(ticket['launch_token'] not in tokens,'Reused phase launch token');tokens.add(ticket['launch_token'])
        spawn=evidence.read(coord/f'phase_{i}-spawn.json');end=evidence.read(coord/f'phase_{i}-exit.json')
        require(spawn['returncode'] is None and spawn['error'] is None and end['error'] is None and
                set(end)==set(spawn)|{'exit_observed_ns'} and all(identical(spawn[k],end[k]) for k in spawn if k!='returncode'),
                'Parent spawn/exit record changed')
        require(type(end['phase']) is int and end['phase']==i,'Parent phase index differs')
        command=end['command'];require(len(command)==12 and command[1:4]==['-B','-m','rapidtinyrec.serving_phase_cli'] and
            command[4:8]==['--run-root',allocation['remote_dir'],'--phase',str(i)] and
            command[8]=='--release-root' and command[10]=='--historical' and
            all(Path(command[n]).is_absolute() for n in (0,9,11)), 'Unexpected worker command')
        common=[command[0],command[9],command[11]]
        if command_base is None:command_base=common
        require(common==command_base,'Worker interpreter/release/reference command changed')
        before=evidence.read(coord/f'phase_{i}-cache-before.json');after=evidence.read(coord/f'phase_{i}-cache-after.json')
        binding=reconcile_worker(ticket,end,evidence.read(root/'worker-entry.json'),evidence.read(root/'worker-exit.json'),
            before,after,evidence.read(root/'cache-before.json'),evidence.read(root/'cache-after.json'))
        observation=evidence.read(root/'allocation-observation.json')
        validate_observation(observation,manifest_bytes,job_bytes,protocol)
        require(identical(observation['allocation_identity']['config'],config),'Worker config bytes differ')
        previous=end['exit_observed_ns'];bindings.append(binding)
    expected_command=[command_base[0],'-B','-m','rapidtinyrec.serving_runner','--coordinator','--run-root',allocation['remote_dir'],
                      '--release-root',command_base[1],'--historical',command_base[2]]
    require(outer['command']==expected_command and outer['cwd']==allocation['remote_dir']+'/code',
            'Supervisor must launch the exact serving coordinator command')
    require(previous<started,'Allocated audit started before workers exited')
    hashes=evidence.finish()
    require((live_log.stat().st_dev,live_log.stat().st_ino)==live_identity and not live_log.is_symlink() and
            live_log.stat().st_nlink==1,'Live stdout identity changed')
    require({p.name for p in sup.iterdir()}=={'supervisor-start.json','coordinator-launch.json','coordinator.log'},
            'Supervisor layout changed during pre-exit audit')
    finished=time.perf_counter_ns();require(started<=finished<deadline,'Allocated audit exceeded global work deadline')
    def key(path):
        path=Path(path)
        if path.is_relative_to(run):return 'run/'+path.relative_to(run).as_posix()
        require(path.is_relative_to(historical),'Unbound external audit input')
        return 'historical/'+path.relative_to(historical).as_posix()
    return dict(schema='serving-allocated-pre-exit-audit-v1',started_ns=started,finished_ns=finished,
        evidence_gates_passed=phases['evidence_gates_passed'],gates=phases['gates'],phases=phases,
        allocation_binding=bound,worker_bindings=bindings,input_sha256={key(p):h for p,h in hashes.items()},
        live_stdout_deferred='run/outputs/supervision/coordinator.log',terminal_reconciliation_pending=True,
        original_submission_record_reconciled=False,advancement_ready=False)
