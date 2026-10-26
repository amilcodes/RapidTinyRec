"""Explicit pre-load source/Slurm binding. No submission or Torch imports.

The future outer supervisor must cover this preflight with its global deadline.
Hardware/runtime checks remain separate. Uploaded manifests are never rewritten.
"""
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time

from .serving_contract import PROFILE, PROTOCOL, PROTOCOL_SHA256, load_protocol
from .serving_policy import POLICY_ENV, validate_environment
from .serving_setup_audit import read_unique

CONFIG='oscar.serving-throughput.json'
PROJECT='rapidtinyrec-exact-serving-pilot'
SSH_TARGET='aagraw73@sshcampus.ccv.brown.edu'
REMOTE_ROOT='/oscar/scratch/aagraw73/agent-runs'
COMMAND=['bash','jobs/serving-throughput.sh']


def require(ok, message):
    if not ok: raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identical(left, right):
    return json.dumps(left,sort_keys=True,allow_nan=False)==json.dumps(right,sort_keys=True,allow_nan=False)


def scheduler_fields(raw):
    fields={}
    for name,value in re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]*)',raw):
        require(name not in fields,'Duplicate scheduler field');fields[name]=value
    return fields


def validate_scheduler(fields, job, remote):
    expected=dict(JobId=job,JobState='RUNNING',Partition='gpu',QOS='norm-gpu',
                  TimeLimit='00:30:00',NumCPUs='4',NumNodes='1',Command=remote+'/job.sh',WorkDir=remote,
                  TresPerNode='gres/gpu:nvidia_rtx_a6000:1')
    require(all(fields.get(k)==v for k,v in expected.items()),'Scheduler allocation/execution identity differs')
    entries=[part.split('=',1) for part in fields.get('AllocTRES','').split(',')]
    require(all(len(v)==2 for v in entries),'Malformed allocated resources')
    tres=dict(entries);require(len(tres)==len(entries),'Duplicate allocated resource')
    require(tres.get('cpu')=='4' and tres.get('node')=='1' and tres.get('mem') in ('32G','32768M') and tres.get('gres/gpu')=='1','Allocated resources differ')
    for key,value in tres.items():
        if key.startswith('gres/gpu:'):
            require(key=='gres/gpu:nvidia_rtx_a6000' and value=='1','Unexpected allocated GPU type/count')
    require(fields.get('NodeList') not in (None,'','(null)','None'),'Missing allocated node')


def validate_manifest(prepared, config, protocol, job_text, env, fields):
    job=job_text.strip()
    require(re.fullmatch(r'[1-9][0-9]*',job) is not None and env.get('SLURM_JOB_ID')==job,'Actual sbatch/environment job identity required')
    require(prepared.get('submission')=='prepared' and prepared.get('job_id') is None,'Original uploaded prepared manifest required')
    run=prepared['run_id'];require(isinstance(run,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',run) is not None,'Canonical run identifier required')
    remote=REMOTE_ROOT+'/'+PROJECT+'/'+run
    require(prepared['remote_dir']==remote==env.get('OSCAR_RUN_DIR') and env.get('OSCAR_OUTPUT_DIR')==remote+'/outputs','Run/code/output location differs')
    require(prepared['project']==config['project']==PROJECT and prepared['profile']==PROFILE,'Serving project/profile differs')
    require(prepared['ssh_target']==config['ssh_target']==SSH_TARGET and config['remote_root']==REMOTE_ROOT,'Remote identity differs')
    resources=dict(protocol['resources'],setup=['module purge'],command=COMMAND)
    require(identical(prepared['resources'],resources) and identical(config['profiles'],{PROFILE:resources}),'Frozen resource/config contract differs')
    names=prepared['files'];hashes=prepared['sha256']
    require(len(set(names))==len(names) and set(names)==set(hashes),'Source inventory differs')
    require({CONFIG,PROTOCOL,COMMAND[1],'source-provenance.json','rapidtinyrec/expected.json'}<=set(names),'Required source/config/protocol files omitted')
    for name,want in hashes.items():
        rel=PurePosixPath(name)
        require(not rel.is_absolute() and rel.as_posix()==name and '..' not in rel.parts and '\\' not in name,'Noncanonical uploaded path')
        require(isinstance(want,str) and re.fullmatch(r'[0-9a-f]{64}',want) is not None,'Malformed source SHA256')
    require(hashes[PROTOCOL]==PROTOCOL_SHA256,'Frozen protocol digest differs')
    validate_environment(env);validate_scheduler(fields,job,remote)
    return job


def verify_source(code, prepared):
    code=Path(code)
    require(not code.is_symlink() and code.is_dir(),'Regular uploaded code directory required')
    code=code.resolve(strict=True);running=Path(__file__).resolve().parent
    require(running==code/'rapidtinyrec','Running package is not the uploaded package')
    actual={};inodes=set()
    for path in code.rglob('*'):
        require(not path.is_symlink(),'Linked uploaded source path')
        if path.is_dir():continue
        require(path.is_file(),'Non-regular uploaded source')
        name=path.relative_to(code).as_posix();stat=path.stat();inode=(stat.st_dev,stat.st_ino)
        require(stat.st_nlink==1 and inode not in inodes,'Aliased source files');inodes.add(inode)
        require(path.suffix not in ('.pyc','.pyo') and '__pycache__' not in path.parts,'Bytecode cannot stand in for reviewed source')
        require(name in prepared['sha256'],'Undeclared uploaded file')
        actual[name]=digest(path)
    require(identical(actual,prepared['sha256']),'Uploaded source bytes differ')
    # Recursion includes source/LogitProcessor.py, not only top-level modules.
    require('rapidtinyrec/source/LogitProcessor.py' in actual,'Nested generation dependency omitted')
    for name,module in list(sys.modules.items()):
        if name=='rapidtinyrec' or name.startswith('rapidtinyrec.'):
            origin=getattr(module,'__file__',None)
            require(origin is not None,'Unbound imported package module')
            path=Path(origin).resolve()
            require(path.is_relative_to(code) and path.relative_to(code).as_posix() in actual,'Imported dependency outside uploaded inventory')
    return actual


def observe_allocation(run_root):
    started=time.perf_counter_ns();validate_environment(os.environ)
    require(sys.dont_write_bytecode,'Launch with -B; uploaded code must stay bytecode-free')
    run=Path(run_root)
    require(not run.is_symlink(),'Linked run directory');run=run.resolve(strict=True)
    require(str(run)==os.environ.get('OSCAR_RUN_DIR'),'Active run directory differs')
    code=run/'code';output=run/'outputs'
    require(output.is_dir() and not output.is_symlink() and str(output)==os.environ.get('OSCAR_OUTPUT_DIR'),'Active output directory differs')
    manifest_path=run/'source-manifest.json';job_path=run/'job-id'
    for path in (manifest_path,job_path):
        require(path.is_file() and not path.is_symlink() and path.stat().st_nlink==1,'Regular original submission identity files required')
    manifest_bytes=manifest_path.read_bytes();job_bytes=job_path.read_bytes()
    prepared=read_unique(manifest_path);config=read_unique(code/CONFIG);protocol=load_protocol(code/PROTOCOL)
    job_text=job_bytes.decode();job=job_text.strip()
    require(re.fullmatch(r'[1-9][0-9]*',job) is not None and os.environ.get('SLURM_JOB_ID')==job,'Actual sbatch identity missing')
    command=['scontrol','show','job','-o',job]
    raw=subprocess.run(command,check=True,text=True,capture_output=True,timeout=10).stdout
    fields=scheduler_fields(raw)
    validate_manifest(prepared,config,protocol,job_text,os.environ,fields)
    hashes=verify_source(code,prepared)
    require(manifest_path.read_bytes()==manifest_bytes and job_path.read_bytes()==job_bytes,'Submission identity changed during verification')
    record=copy.deepcopy(prepared);record.update(job_id=job,submission='allocated_observation')
    record['allocation_identity']=dict(schema='serving-allocated-identity-v1',
        uploaded_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),sbatch_job_id_file_sha256=hashlib.sha256(job_bytes).hexdigest(),
        scheduler_command=command,scheduler_stdout=raw,scheduler_fields=fields,observed_source_sha256=hashes,
        config=config,protocol_sha256=PROTOCOL_SHA256,started_ns=started,verified_ns=time.perf_counter_ns(),
        numerical_environment={key:os.environ.get(key) for key in POLICY_ENV},
        requires_local_oscar_reconciliation=True)
    return record


def validate_observation(allocated, manifest_bytes, job_bytes, protocol):
    """Bind observation to original prepared bytes, without inventing submission."""
    identity=allocated['allocation_identity']
    require(identity['schema']=='serving-allocated-identity-v1' and identity['requires_local_oscar_reconciliation'] is True,'Allocated proof schema differs')
    numerical=identity['numerical_environment']
    require(set(numerical)==set(POLICY_ENV) and all(v is None or isinstance(v,str) for v in numerical.values()),
            'Complete observed numerical environment required')
    validate_environment({k:v for k,v in numerical.items() if v is not None})
    require(identity['uploaded_manifest_sha256']==hashlib.sha256(manifest_bytes).hexdigest() and identity['sbatch_job_id_file_sha256']==hashlib.sha256(job_bytes).hexdigest(),'Submitted bytes differ')
    # Duplicate keys in uploaded JSON must be rejected too.
    def pairs(items):
        result={}
        for k,v in items:
            require(k not in result,'Duplicate manifest key');result[k]=v
        return result
    prepared=json.loads(manifest_bytes,object_pairs_hook=pairs);job=job_bytes.decode().strip()
    require(allocated['submission']=='allocated_observation','Observation state differs')
    require(allocated['job_id']==job,'Job binding differs')
    for key in ('run_id','project','profile','ssh_target','remote_dir','resources','files','sha256'):
        require(identical(allocated[key],prepared[key]),'Prepared record differs: '+key)
    raw=identity['scheduler_stdout'];fields=scheduler_fields(raw)
    require(fields==identity['scheduler_fields'] and identity['scheduler_command']==['scontrol','show','job','-o',job],'Scheduler proof differs')
    env=dict(SLURM_JOB_ID=job,OSCAR_RUN_DIR=prepared['remote_dir'],OSCAR_OUTPUT_DIR=prepared['remote_dir']+'/outputs')
    validate_manifest(prepared,identity['config'],protocol,job,env,fields)
    require(identical(identity['observed_source_sha256'],prepared['sha256']) and identity['protocol_sha256']==PROTOCOL_SHA256,'Observed source/protocol proof differs')
    require(type(identity['started_ns']) is int and type(identity['verified_ns']) is int and 0<identity['started_ns']<=identity['verified_ns'],'Invalid verification clocks')
    return dict(schema='serving-allocation-observation-audit-v1',matched=True,run_id=prepared['run_id'],job_id=job,
                model_executed=False,submitted_record_reconciled=False,advancement_ready=False)


def reconcile_record(allocated, actual, manifest_bytes, job_bytes, protocol):
    """Metadata-only terminal binding to the separately saved submission record."""
    report=validate_observation(allocated,manifest_bytes,job_bytes,protocol)
    require(actual['submission']=='submitted' and actual['job_id']==report['job_id'],'Submission states/job differ')
    for key in ('run_id','project','profile','ssh_target','remote_dir','resources','files','sha256'):
        require(identical(allocated[key],actual[key]),'Submitted record differs: '+key)
    return dict(report,schema='serving-allocation-reconciliation-v1',submitted_record_reconciled=True)
