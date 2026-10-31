"""Bind a distinct packed-qkv run to OSCAR source and actual Slurm identity.

The uploaded source-manifest has no job ID because OSCAR writes it before sbatch.
The separate job-id file is written from sbatch's actual response. Neither file
is replaced here; the derived record identifies both sources explicitly.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess

from .qkv_allocation_contract import PROTOCOL, load_protocol, RESOURCES
PROFILE = 'packed-qkv-bf16-filtered-v1'

CONFIG = 'oscar.json'
PROJECT = 'rapidtinyrec-packed-qkv-filtered'
SSH_TARGET = 'aagraw73@sshcampus.ccv.brown.edu'
REMOTE_ROOT = '/oscar/scratch/aagraw73/agent-runs'
COMMAND = ['bash', 'jobs/packed-qkv-pilot.sh']


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scheduler_fields(raw):
    fields = {}
    for name,value in re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]*)',raw):
        assert name not in fields, 'Duplicate scheduler field'
        fields[name] = value
    return fields


def memory_mib(value):
    match=re.fullmatch(r'([0-9]+)([MG])',value)
    assert match, 'Unrecognized allocated memory unit'
    return int(match[1]) * (1024 if match[2]=='G' else 1)


def validate_scheduler(fields, job_id, remote):
    assert fields['JobId']==job_id and fields['JobState']=='RUNNING'
    assert fields['Partition']=='gpu' and fields['QOS']=='norm-gpu'
    assert fields['TimeLimit']=='00:30:00'
    assert fields['NumCPUs']=='4' and fields['NumNodes']=='1'
    assert fields['Command']==str(PurePosixPath(remote)/'job.sh')
    assert fields['WorkDir']==remote
    entries=[item.split('=',1) for item in fields['AllocTRES'].split(',')]
    tres=dict(entries)
    assert len(tres)==len(entries), 'Duplicate allocated resource'
    assert tres['cpu']=='4' and tres['node']=='1' and memory_mib(tres['mem'])==32768
    assert tres['gres/gpu']=='1'
    # Oscar accounting records generic GPU counts without typed AllocTRES.
    # Bind the scheduler's typed request separately; benchmark manifests still
    # independently require the actual hardware to be NVIDIA RTX A6000.
    assert fields['TresPerNode']=='gres/gpu:nvidia_rtx_a6000:1'
    if 'gres/gpu:nvidia_rtx_a6000' in tres:
        assert tres['gres/gpu:nvidia_rtx_a6000']=='1'
    assert fields['NodeList'] not in ('','(null)','None')


def validate_manifest(prepared, config, protocol, job_id_text, env, fields):
    assert prepared['submission']=='prepared' and prepared['job_id'] is None
    job_id=job_id_text.strip()
    assert re.fullmatch(r'[1-9][0-9]*',job_id), 'Missing or federated sbatch identity'
    assert env.get('SLURM_JOB_ID')==job_id
    remote=prepared['remote_dir']
    assert remote==str(PurePosixPath(REMOTE_ROOT)/PROJECT/prepared['run_id'])
    assert env.get('OSCAR_RUN_DIR')==remote
    assert env.get('OSCAR_OUTPUT_DIR')==str(PurePosixPath(remote)/'outputs')
    assert prepared['project']==config['project']==PROJECT
    assert prepared['profile']==PROFILE and prepared['ssh_target']==config['ssh_target']==SSH_TARGET
    assert config['remote_root']==REMOTE_ROOT
    resources=RESOURCES
    assert protocol['resources']==dict(gpu='regular RTX A6000',cpus=4,memory_gb=32,max_minutes=30,concurrency=1)
    assert prepared['resources']==resources
    assert config['profiles']=={PROFILE:resources}
    assert sorted(prepared['files'])==sorted(prepared['sha256'])
    assert len(set(prepared['files']))==len(prepared['files'])
    assert {CONFIG,PROTOCOL,COMMAND[1]} <= set(prepared['sha256'])
    validate_scheduler(fields,job_id,remote)
    return job_id


def verify_source(code, prepared):
    code=Path(code).resolve(strict=True)
    for name,want in prepared['sha256'].items():
        rel=PurePosixPath(name)
        assert not rel.is_absolute() and '..' not in rel.parts
        path=code/name
        assert path.is_file() and not path.is_symlink()
        assert not any(p.is_symlink() for p in path.parents if p!=code and code in p.parents)
        assert path.resolve().is_relative_to(code)
        assert digest(path)==want,name
    # The running auditor and all of its package dependencies must be submitted.
    for p in Path(__file__).parent.glob('*.py'):
        assert prepared['sha256']['rapidtinyrec/'+p.name]==digest(p),p.name


def verify_dependency_coverage(code, prepared):
    """Cover non-Python pins, nested source and historical inputs as well."""
    code=Path(code)
    required={CONFIG,PROTOCOL,COMMAND[1],'release-config.json','parent-source-lock.json',
              'rapidtinyrec/expected.json','rapidtinyrec/quality_resources.json','rapidtinyrec/qkv_forward_pin.txt'}
    required.update('historical_bf16/'+name for name in
                    ('reference.json','manifest.json','0001.json','0002.json','0006.json','0010.json'))
    for directory in ('rapidtinyrec','historical_bf16'):
        assert (code/directory).is_dir(),directory
        for path in (code/directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
                required.add(str(path.relative_to(code)))
    assert required <= set(prepared['sha256']), 'Unrecorded runtime dependency: '+repr(sorted(required-set(prepared['sha256'])))
    assert all((code/name).is_file() and not (code/name).is_symlink() for name in required)
    return {'required_dependencies':len(required),'python_only_coverage':False}


def observe_allocation(run_root, code_root, output_root):
    run_root=Path(run_root).resolve(strict=True)
    code_root=Path(code_root).resolve(strict=True)
    output_root=Path(output_root).resolve(strict=True)
    assert code_root==run_root/'code' and output_root==run_root/'outputs'
    assert str(run_root)==os.environ.get('OSCAR_RUN_DIR')
    prepared_path=run_root/'source-manifest.json'
    job_path=run_root/'job-id'
    assert not prepared_path.is_symlink() and not job_path.is_symlink()
    prepared=read(prepared_path); config=read(code_root/CONFIG)
    protocol=load_protocol(code_root/PROTOCOL)
    job_text=job_path.read_text()
    job_id=job_text.strip()
    assert re.fullmatch(r'[1-9][0-9]*',job_id) and os.environ.get('SLURM_JOB_ID')==job_id
    command=['scontrol','show','job','-o',job_id]
    raw=subprocess.run(command,check=True,text=True,capture_output=True,timeout=10).stdout
    fields=scheduler_fields(raw)
    validate_manifest(prepared,config,protocol,job_text,os.environ,fields)
    verify_source(code_root,prepared)
    verify_dependency_coverage(code_root,prepared)
    record=deepcopy(prepared)
    record['job_id']=job_id
    record['submission']='allocated_observation'
    record['allocation_identity']={
        'schema':'packed-qkv-allocated-identity-v1',
        'uploaded_manifest_sha256':digest(prepared_path),
        'sbatch_job_id_file_sha256':digest(job_path),
        'scheduler_command':command,'scheduler_stdout':raw,'scheduler_fields':fields,
        'source':'OSCAR uploaded prepared manifest plus actual sbatch job-id file and live scontrol query',
        'requires_local_oscar_reconciliation':True}
    return record


def reconcile_record(allocated, actual, uploaded_manifest_bytes, job_id_bytes):
    """Local metadata-only reconciliation after fetch; never re-run bootstrap."""
    identity=allocated['allocation_identity']
    assert identity['schema']=='packed-qkv-allocated-identity-v1'
    assert identity['requires_local_oscar_reconciliation'] is True
    assert identity['uploaded_manifest_sha256']==hashlib.sha256(uploaded_manifest_bytes).hexdigest()
    assert identity['sbatch_job_id_file_sha256']==hashlib.sha256(job_id_bytes).hexdigest()
    prepared=json.loads(uploaded_manifest_bytes)
    assert prepared['job_id'] is None and prepared['submission']=='prepared'
    assert actual['submission']=='submitted'
    assert allocated['submission']=='allocated_observation'
    assert allocated['job_id']==actual['job_id']==job_id_bytes.decode().strip()
    for key in ('run_id','project','profile','ssh_target','remote_dir','resources','files','sha256'):
        assert allocated[key]==actual[key]==prepared[key],key
    fields=scheduler_fields(identity['scheduler_stdout'])
    assert fields==identity['scheduler_fields']
    assert identity['scheduler_command']==['scontrol','show','job','-o',actual['job_id']]
    validate_scheduler(fields,actual['job_id'],actual['remote_dir'])
    return {'schema':'packed-qkv-allocation-reconciliation-v1','matched':True,
            'run_id':actual['run_id'],'job_id':actual['job_id'],
            'files':len(actual['files']),'bootstrap_executed':False}
