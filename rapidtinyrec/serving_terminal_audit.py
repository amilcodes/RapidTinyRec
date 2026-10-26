"""Metadata-only terminal reconciliation; never replay the allocated phase audit.

The caller supplies the separately saved submitted record and an externally
frozen hash inventory after terminal fetch. Neither input may live in the run.
This does not authorize submission, promotion or another run.
"""
import re
from pathlib import Path

from .serving_allocation import identical, reconcile_record, require
from .serving_contract import PROTOCOL, load_protocol
from .serving_disk_audit import Evidence
from .serving_worker_audit import integer

GATES = {'setup','setup_to_execution','cross_phase_setup','measured_numerics',
         'performance','profile_numerics','actual_gpu_overlap'}
TERMINAL = {'run/outputs/coordinator/final-audit.json', 'run/outputs/coordinator/exit.json',
            'run/outputs/supervision/supervisor-exit.json', 'run/outputs/supervision/coordinator.log'}


def reconcile_terminal(run_root, historical, submitted_record, frozen_manifest):
    """Verify saved bytes, publication/exit clocks and observed allocation only."""
    run=Path(run_root).absolute();history=Path(historical).absolute();evidence=Evidence()
    for path in (run, history):
        require(path.is_dir() and not path.is_symlink(),'Regular terminal evidence roots required')
    require(not run.is_relative_to(history) and not history.is_relative_to(run), 'Separate historical evidence root required')
    for path in (submitted_record,frozen_manifest):
        path=Path(path).absolute()
        require(not path.is_relative_to(run) and not path.is_relative_to(history),'External terminal metadata required')
    actual=evidence.read(submitted_record);freeze=evidence.read(frozen_manifest)
    require(freeze['schema']=='serving-terminal-freeze-v1' and freeze['run_id']==actual['run_id'] and
            freeze['job_id']==actual['job_id'],'Frozen terminal identity differs')
    require(evidence.hashes[Path(submitted_record).absolute()]==freeze['submitted_record_sha256'],
            'Submitted record changed after freeze')
    report=evidence.read(run/'outputs/coordinator/final-audit.json')
    require(report['schema']=='serving-allocated-pre-exit-audit-v1' and report['terminal_reconciliation_pending'] is True and
            report['original_submission_record_reconciled'] is False and report['advancement_ready'] is False,
            'Allocated pre-exit report scope differs')
    inputs=report['input_sha256'];frozen=freeze['input_sha256']
    require(isinstance(inputs,dict) and isinstance(frozen,dict) and set(frozen)==set(inputs)|TERMINAL and
            not set(inputs)&TERMINAL,'Frozen terminal input closure differs')
    def locate(key):
        require(isinstance(key,str) and '/' in key,'Invalid frozen input path')
        domain,relative=key.split('/',1);path=Path(relative)
        require(domain in ('run','historical') and relative and not path.is_absolute() and
                '..' not in path.parts and path.as_posix()==relative,'Invalid frozen input path')
        root=run if domain=='run' else history
        for parent in path.parents:
            require(not (root/parent).is_symlink(),'Linked frozen input directory')
        return root/path
    for key,want in frozen.items():
        require(isinstance(want,str) and re.fullmatch('[0-9a-f]{64}',want) is not None,'Invalid frozen digest')
        path=locate(key);evidence.register(path)
        require(evidence.hashes[path.absolute()]==want and (key not in inputs or inputs[key]==want),
                'Frozen input bytes changed: '+key)
    prepared=evidence.read(run/'source-manifest.json');evidence.register(run/'job-id')
    protocol=load_protocol(run/'code'/PROTOCOL)
    # Require every actual code/phase file, not merely whatever the saved report
    # elects to list. These are byte/inventory checks, never numerical re-audits.
    expected={'run/source-manifest.json','run/job-id','historical/manifest.json'}
    expected.update(f"historical/users/{u['cohort_index']:04d}.json" for u in protocol['cohort']['users'])
    for relative in ('code','outputs/coordinator','outputs/launch','outputs/supervision',
                     *(f'outputs/phase_{i}' for i in range(4))):
        expected.update('run/'+relative+'/'+name for name in evidence.tree(run/relative))
    require(set(frozen)==expected,'Complete terminal file inventory required')
    require({'run/code/'+name for name in prepared['files']}=={k for k in frozen if k.startswith('run/code/')},
            'Submitted source inventory differs')
    require(all(frozen['run/code/'+name]==want for name,want in prepared['sha256'].items()),
            'Submitted source bytes differ')
    observation=evidence.read(run/'outputs/coordinator/allocation-observation.json')
    binding=reconcile_record(observation,actual,(run/'source-manifest.json').read_bytes(),(run/'job-id').read_bytes(),protocol)
    for i in range(4):
        reconcile_record(evidence.read(run/f'outputs/phase_{i}/allocation-observation.json'),actual,
                         (run/'source-manifest.json').read_bytes(),(run/'job-id').read_bytes(),protocol)
    entry=evidence.read(run/'outputs/coordinator/entry.json');outer=evidence.read(run/'outputs/supervision/supervisor-start.json')
    launch=evidence.read(run/'outputs/supervision/coordinator-launch.json')
    end=evidence.read(run/'outputs/coordinator/exit.json');super_end=evidence.read(run/'outputs/supervision/supervisor-exit.json')
    require(identical(end['identity'],entry['identity']) and identical(entry['identity'],launch['identity']) and
            identical(entry['supervisor_start'],outer),'Terminal coordinator birth differs')
    require(end['complete'] is True and end['error'] is None and identical(end['phases_completed'],[0,1,2,3]) and
            end['advancement_ready'] is False,'Coordinator did not complete all four phases')
    require(end['audit_sha256']==frozen['run/outputs/coordinator/final-audit.json'],'Allocated audit publication hash differs')
    start=integer(outer['global_started_ns']);deadline=integer(outer['deadline_ns'])
    require(deadline-start==1620*10**9,'Global work deadline changed')
    clocks=[start,integer(entry['started_ns']),integer(report['started_ns']),integer(report['finished_ns']),
            integer(end['audit_published_ns']),integer(end['finished_ns']),integer(super_end['exit_observed_ns'])]
    require(all(a<=z for a,z in zip(clocks,clocks[1:])) and clocks[-1]<deadline,'Terminal work clocks escape deadline/order')
    finished=integer(super_end['finished_ns'])
    require(clocks[-1]<=finished<=start+1800*10**9,'Termination bookkeeping exceeds allocation ceiling')
    require(super_end['schema']=='serving-supervisor-exit-v1' and super_end['complete'] is True and
            super_end['error'] is None and super_end['returncode']==0 and type(super_end['returncode']) is int and
            super_end['child_pid']==entry['identity']['pid'] and super_end['global_started_ns']==start and
            super_end['deadline_ns']==deadline and super_end['deadline_exceeded'] is False and
            super_end['watchdog']=={} and type(super_end['group_cleanup_signal_sent']) is bool and
            super_end['advancement_ready'] is False,'Supervisor terminal failure or identity mismatch')
    gates=report['gates'];phase_report=report['phases']
    require(set(gates)==GATES and all(type(v) is bool for v in gates.values()) and
            identical(gates,phase_report['gates']) and type(report['evidence_gates_passed']) is bool and
            report['evidence_gates_passed']==phase_report['evidence_gates_passed']==all(gates.values()) and
            phase_report['online_verdicts_trusted'] is False and phase_report['advancement_ready'] is False,
            'Allocated gate summary is inconsistent')
    hashes=evidence.finish()
    return dict(schema='serving-terminal-reconciliation-v1',metadata_reconciled=True,
        allocation_binding=binding,allocated_evidence_gates_passed=report['evidence_gates_passed'],gates=gates,
        allocated_audit_sha256=frozen['run/outputs/coordinator/final-audit.json'],
        terminal_stdout_sha256=frozen['run/outputs/supervision/coordinator.log'],
        submitted_record_sha256=freeze['submitted_record_sha256'],
        frozen_manifest_sha256=hashes[str(Path(frozen_manifest).absolute())],
        numerical_audit_executed=False,model_executed=False,advancement_ready=False)
