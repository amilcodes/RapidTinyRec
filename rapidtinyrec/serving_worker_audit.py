"""Independent raw worker envelope/shutdown audit; run provenance is separate."""
import re

from .serving_allocation import identical, require
from .serving_contract import PROTOCOL_SHA256
from .serving_process import validate_launch, validate_cache_environment

WORKER_FILES = frozenset(('worker-entry.json','worker-exit.json','allocation-observation.json',
                         'allocation-persisted.json','cache-before.json','cache-after.json','lifecycle.json'))


def integer(value):
    require(type(value) is int and value > 0,'Positive integer worker clock/identity required')
    return value


def validate_identity(value):
    for name in ('pid','parent_pid','start_ticks','clock_ticks'):integer(value[name])
    require(isinstance(value['hostname'],str) and bool(value['hostname']) and
            isinstance(value['boot_id'],str) and re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',value['boot_id']),
            'Worker host/boot identity missing')


def validate_cache(value):
    from pathlib import PurePosixPath
    require(isinstance(value,dict),'Cache inventory must be a mapping')
    for name,row in value.items():
        path=PurePosixPath(name)
        require(not path.is_absolute() and path.as_posix()==name and '..' not in path.parts and name not in ('','.') and '\\' not in name,
                'Noncanonical cache member')
        require(set(row)=={'bytes','sha256'} and type(row['bytes']) is int and row['bytes']>=0 and
                isinstance(row['sha256'],str) and re.fullmatch(r'[0-9a-f]{64}',row['sha256']), 'Malformed cache digest/capacity')


def audit_worker(raw, *, phase, protocol, preparation, resource_stages, execution_finished_ns, allocation_sha256):
    require(set(raw)==WORKER_FILES,'Complete raw worker wrapper required')
    entry=raw['worker-entry.json'];end=raw['worker-exit.json'];allocated=raw['allocation-observation.json']
    persisted=raw['allocation-persisted.json'];launch=entry['launch'];identity=entry['identity']
    require(entry['schema']=='serving-worker-entry-v1' and end['schema']=='serving-worker-exit-v1', 'Worker schema differs')
    require(type(entry['phase']) is int and type(end['phase']) is int and entry['phase']==end['phase']==phase,
            'Worker phase identity differs')
    validate_identity(identity);validate_identity(launch['parent_identity'])
    require(identical(identity,end['identity']) and end['started_ns']==entry['started_ns'],
            'Worker birth/start changed before shutdown')
    spec=validate_launch(launch,allocated['remote_dir'],phase,protocol,identity,launch['parent_identity'],
                         launch['launch_token'],integer(entry['started_ns']))
    require(entry['process_group']==entry['session']==launch['process_group']==launch['session']==launch['parent_identity']['pid'],
            'Worker group/session differs from supervised parent')
    validate_cache_environment(launch['cache']['root'],entry['cache_environment'])
    require(end['complete'] is True and end['execution_complete'] is True and end['error'] is None,
            'Worker did not complete and close')
    before=raw['cache-before.json'];after=raw['cache-after.json'];validate_cache(before);validate_cache(after)
    require(bool(before)==spec['reuse_disk_cache'] and bool(after),'Worker cold/reuse cache evidence differs')
    proof=allocated['allocation_identity']
    require(allocated['submission']=='allocated_observation' and isinstance(allocated['job_id'],str) and
            re.fullmatch(r'[1-9][0-9]*',allocated['job_id']) and proof['schema']=='serving-allocated-identity-v1' and
            proof['protocol_sha256']==PROTOCOL_SHA256,'Raw allocation observation missing')
    require(persisted['schema']=='serving-allocation-persisted-v1' and persisted['sha256']==allocation_sha256,
            'Persisted allocation bytes differ')
    clocks=[entry['started_ns'],resource_stages['initializer_started_ns'],proof['started_ns'],proof['verified_ns'],
            resource_stages['resources_verified_ns'],persisted['published_ns'],preparation['admission_events'][0]['started_ns']]
    clocks=[integer(v) for v in clocks]
    require(all(a<=z for a,z in zip(clocks,clocks[1:])),'Allocation publication did not precede model initialization')
    journal=raw['lifecycle.json'];admission=preparation['admission_events'];lanes=spec['resident_lanes']
    require(isinstance(journal,list) and len(journal)==len(admission)+lanes and identical(journal[:len(admission)],admission),
            'Full lifecycle must preserve the exact admission prefix and all closes')
    previous=integer(execution_finished_ns)
    for lane,row in enumerate(journal[len(admission):]):
        require(row['operation']=='close' and type(row['lane']) is int and row['lane']==lane and
                row['thread_id']==preparation['owners'][str(lane)]['identity']['owner_thread_id'] and
                row['success'] is True and row['error'] is None,'Failed/missing/wrong-owner shutdown')
        start=integer(row['started_ns']);finish=integer(row['finished_ns'])
        require(previous<=start<=finish,'Owner shutdown overlaps requests or prior close');previous=finish
    require(previous<=integer(end['service_closed_ns'])<=integer(end['finished_ns'])<launch['deadline_ns'],
            'Worker shutdown escaped the global work deadline')
    return dict(schema='serving-worker-envelope-audit-v1',passed=True,phase=phase,identity=identity,
                launch=launch,started_ns=entry['started_ns'],finished_ns=end['finished_ns'],
                job_id=allocated['job_id'],run_id=allocated['run_id'],cache_before=before,cache_after=after,
                allocation_identity=proof,advancement_ready=False,
                remaining='Original submitted bytes/source/live scheduler proof, parent/coordinator cache/process observations and terminal supervisor deadline reconciliation remain required.')
