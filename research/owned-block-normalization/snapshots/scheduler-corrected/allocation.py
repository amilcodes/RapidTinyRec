"""Pure metadata/source binding for a new owned-block primitive allocation.

Observation queries only its current owned job. Reconciliation uses saved bytes
and never performs scheduler queries, bootstrap or numerical work.
"""
import copy
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import re
import subprocess
from allocation_contract import PROFILE,PROJECT,SSH_TARGET,REMOTE_ROOT,RESOURCES,REQUIRED_FILES


def require(ok,message):
    if not ok:raise ValueError(message)


def scheduler_fields(raw):
    pairs=re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_/:]*)=([^\s]*)',raw)
    fields=dict(pairs);require(len(fields)==len(pairs),'Duplicate scheduler fields');return fields


def validate_metadata(prepared,config,protocol,job,env,fields):
    require(type(job) is str and re.fullmatch('[1-9][0-9]*',job) is not None,'Actual job ID required')
    require(prepared['submission']=='prepared' and prepared['job_id'] is None,'Uploaded prepared manifest required')
    run=prepared['run_id']
    require(type(run) is str and re.fullmatch(re.escape(PROFILE)+r'-[A-Za-z0-9-]+',run) is not None,'Distinct run identity required')
    remote=REMOTE_ROOT+'/'+PROJECT+'/'+run
    require(prepared['project']==config['project']==PROJECT and prepared['profile']==PROFILE,'Wrong experiment identity')
    require(prepared['ssh_target']==config['ssh_target']==SSH_TARGET and config['remote_root']==REMOTE_ROOT,'Wrong host/root')
    require(prepared['remote_dir']==env['OSCAR_RUN_DIR']==remote and env['OSCAR_OUTPUT_DIR']==remote+'/outputs','Wrong run/output directory')
    require(env['SLURM_JOB_ID']==job and env['SLURM_CPUS_PER_TASK']=='4','Allocation environment differs')
    require(prepared['resources']==RESOURCES and config['profiles']=={PROFILE:RESOURCES},'Resource/config contract differs')
    for resource in (prepared['resources'],config['profiles'][PROFILE]):
        require(all(type(resource[k]) is int for k in ('gpus','cpus','memory_gb')),'Integer resources required')
    require(sorted(config['extra_files'])==sorted(REQUIRED_FILES),'Exact configured upload inventory required')
    require(len(prepared['files'])==len(set(prepared['files'])) and set(prepared['files'])==set(prepared['sha256'])==set(REQUIRED_FILES),'Exact submitted source inventory required')
    for name,digest in prepared['sha256'].items():
        p=PurePosixPath(name)
        require(not p.is_absolute() and '..' not in p.parts and re.fullmatch('[0-9a-f]{64}',digest) is not None,'Unsafe source path/hash')
    expected=dict(schema='owned-request-block-normalization-intervention-v1',vocabulary=152265,
                  rows_per_request=50,requests=2,dtype='float32',seeds=[17,29],source_element_offsets=[0,1,2,3],
                  modes=['eager','cuda_graph'],declared_cases=16,orders=['forward','repeat','reverse','restored'],sites=['first','second'])
    for k,v in expected.items():require(type(protocol[k]) is type(v) and protocol[k]==v,'Protocol differs: '+k)
    require(all(type(v) is int for k in ('seeds','source_element_offsets') for v in protocol[k]),'Integer protocol lists required')
    require(all(type(protocol['resources'][k]) is int for k in ('cpus','memory_gb','max_allocation_minutes','concurrency')),'Integer protocol resources required')
    require(protocol['resources']==dict(gpu='regular RTX A6000',cpus=4,memory_gb=32,max_allocation_minutes=10,concurrency=1),'Protocol resource limits differ')
    require({k:protocol['budget'][k] for k in ('compute_max_s','global_max_s','allocation_s')}==dict(compute_max_s=480,global_max_s=540,allocation_s=600),'Protocol deadlines differ')
    require('CPUsPerTask' not in fields,'Unexpected scheduler CPU alias')
    required=dict(JobId=job,JobState='RUNNING',Partition='gpu',QOS='norm-gpu',TimeLimit='00:10:00',NumCPUs='4',NumNodes='1',WorkDir=remote,Command=remote+'/job.sh',TresPerNode='gres/gpu:nvidia_rtx_a6000:1')
    required['CPUs/Task']='4'
    for k,v in required.items():require(fields.get(k)==v,'Scheduler differs: '+k)
    pairs=[s.split('=',1) for s in fields['AllocTRES'].split(',')];tres=dict(pairs)
    require(len(pairs)==len(tres),'Duplicate allocated resource')
    require(tres.get('cpu')=='4' and tres.get('node')=='1' and tres.get('gres/gpu')=='1' and tres.get('mem') in ('32G','32768M'),'Allocated resource limits differ')
    if 'gres/gpu:nvidia_rtx_a6000' in tres:require(tres['gres/gpu:nvidia_rtx_a6000']=='1','Typed allocation differs')
    require(fields.get('NodeList') not in (None,'','(null)','None'),'Allocated node missing')
    return remote


def verify_source(code,prepared):
    code=Path(code).resolve(strict=True)
    for name,want in prepared['sha256'].items():
        p=code/name
        require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(code),'Unsafe/missing source')
        require(not any(parent.is_symlink() for parent in p.parents if parent!=code and code in parent.parents),'Symlink source directory')
        require(hashlib.sha256(p.read_bytes()).hexdigest()==want,'Source differs: '+name)
    return dict(files=len(prepared['files']),source_hashes_verified=True)


def validate_capture(observed,capture_bytes,stage):
    require(stage in ('before','after'),'Exact observation stage required')
    capture=json.loads(capture_bytes)
    require(capture['schema']=='owned-block-scheduler-capture-v1' and capture['stage']==stage,'Scheduler capture schema/stage differs')
    require(capture['validation_performed'] is False,'Capture must precede validation')
    require(type(capture['returncode']) is int and capture['returncode']==0 and capture['error'] is None,'Scheduler command failed')
    require(hashlib.sha256(capture_bytes).hexdigest()==observed['scheduler_capture_sha256'],'Scheduler capture bytes differ')
    for key in ('job_id','run_id','environment','uploaded_manifest_sha256','job_id_sha256','scheduler_command','scheduler_stdout'):
        require(capture[key]==observed[key],'Capture binding differs: '+key)
    return dict(scheduler_capture_bound=True,stage=stage)


def observe(stage):
    require(stage in ('before','after'),'Exact observation stage required')
    job=os.environ.get('SLURM_JOB_ID','');require(re.fullmatch('[1-9][0-9]*',job) is not None,'Allocated execution only')
    root=Path(os.environ['OSCAR_RUN_DIR']).resolve(strict=True);code=Path(__file__).resolve().parent
    require(code==root/'code','Source must be inside exact run')
    raw=(root/'source-manifest.json').read_bytes();job_bytes=(root/'job-id').read_bytes()
    require(job_bytes.decode().strip()==job,'Job file differs')
    prepared=json.loads(raw);config=json.loads((code/'oscar.json').read_text());protocol=json.loads((code/'protocol.json').read_text())
    command=['scontrol','show','job','-o',job]
    environment={k:os.environ.get(k) for k in ('SLURM_JOB_ID','SLURM_CPUS_PER_TASK','OSCAR_RUN_DIR','OSCAR_OUTPUT_DIR')}
    capture=dict(schema='owned-block-scheduler-capture-v1',stage=stage,job_id=job,run_id=prepared['run_id'],
                 environment=environment,uploaded_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                 job_id_sha256=hashlib.sha256(job_bytes).hexdigest(),scheduler_command=command,
                 scheduler_stdout='',scheduler_stderr='',returncode=None,error=None,validation_performed=False)
    # Exclusive creation occurs before the query; reuse never executes it again.
    # Flush/fsync even a failing command BEFORE parsing or resource validation.
    capture_path=root/'outputs'/'owned-block'/'worker'/('compute-scheduler-'+stage+'.json')
    failure=None
    with capture_path.open('x') as output:
        try:
            result=subprocess.run(command,check=False,capture_output=True,text=True,timeout=10)
            capture.update(scheduler_stdout=result.stdout,scheduler_stderr=result.stderr,returncode=result.returncode)
        except BaseException as exc:
            failure=exc
            def as_text(value):
                return value.decode(errors='replace') if isinstance(value,bytes) else (value or '')
            capture.update(scheduler_stdout=as_text(getattr(exc,'stdout',None)),scheduler_stderr=as_text(getattr(exc,'stderr',None)),error=repr(exc))
        output.write(json.dumps(capture,indent=2,allow_nan=False)+'\n');output.flush();os.fsync(output.fileno())
    if failure is not None:raise failure
    require(type(capture['returncode']) is int and capture['returncode']==0,'Scheduler command failed')
    stdout=capture['scheduler_stdout']
    fields=scheduler_fields(stdout);validate_metadata(prepared,config,protocol,job,os.environ,fields);verify_source(code,prepared)
    return dict(schema='owned-block-allocation-v1',job_id=job,run_id=prepared['run_id'],prepared=copy.deepcopy(prepared),
                environment={k:os.environ[k] for k in ('SLURM_JOB_ID','SLURM_CPUS_PER_TASK','OSCAR_RUN_DIR','OSCAR_OUTPUT_DIR')},
                uploaded_manifest_sha256=hashlib.sha256(raw).hexdigest(),job_id_sha256=hashlib.sha256(job_bytes).hexdigest(),
                scheduler_command=command,scheduler_stdout=stdout,scheduler_fields=fields,
                scheduler_capture_sha256=hashlib.sha256(capture_path.read_bytes()).hexdigest())


def reconcile(observed,actual,manifest_bytes,job_bytes,config,protocol):
    require(observed['schema']=='owned-block-allocation-v1','Wrong allocation schema')
    prepared=json.loads(manifest_bytes);job=job_bytes.decode().strip()
    require(prepared==observed['prepared'] and hashlib.sha256(manifest_bytes).hexdigest()==observed['uploaded_manifest_sha256'] and hashlib.sha256(job_bytes).hexdigest()==observed['job_id_sha256'],'Captured identity hashes differ')
    require(actual['submission']=='submitted' and actual['job_id']==observed['job_id']==job,'Submitted/allocated identity differs')
    for k in ('run_id','project','profile','ssh_target','remote_dir','resources','files','sha256'):
        require(actual[k]==prepared[k],'Actual run manifest differs: '+k)
    fields=scheduler_fields(observed['scheduler_stdout']);require(fields==observed['scheduler_fields'],'Captured scheduler differs')
    require(observed['scheduler_command']==['scontrol','show','job','-o',job],'Wrong scheduler query')
    require(observed['run_id']==prepared['run_id'],'Allocation run identity differs')
    validate_metadata(prepared,config,protocol,job,observed['environment'],fields)
    return dict(matched=True,job_id=job,run_id=prepared['run_id'],bootstrap_executed=False,numerical_reaudit=False)
