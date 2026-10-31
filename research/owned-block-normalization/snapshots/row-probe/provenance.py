"""Allocation/source validation before importing torch; no external credentials."""
import hashlib,json,os,pathlib,re,subprocess
PROFILE='row-permutation-diagnostic-v1'
PROJECT='rapidtinyrec-autoresearch-v2'
CONFIG='oscar.json'

def validate_metadata(record,config,job,env,fields):
    assert re.fullmatch(r'[1-9][0-9]*',job)
    assert record['submission']=='prepared' and record['job_id'] is None
    assert record['project']==config['project']==PROJECT and record['profile']==PROFILE
    assert record['ssh_target']==config['ssh_target']=='aagraw73@sshcampus.ccv.brown.edu'
    remote='/oscar/scratch/aagraw73/agent-runs/'+PROJECT+'/'+record['run_id']
    assert record['remote_dir']==remote and env['OSCAR_RUN_DIR']==remote and env['OSCAR_OUTPUT_DIR']==remote+'/outputs'
    assert env['SLURM_JOB_ID']==job and fields['JobId']==job and fields['JobState']=='RUNNING'
    resources=dict(partition='gpu',qos='norm-gpu',constraint='ampere',gpu_type='nvidia_rtx_a6000',gpus=1,cpus=4,memory_gb=32,time='00:10:00',setup=['module purge'],command=['bash','job.sh'])
    assert record['resources']==resources and config['profiles']=={PROFILE:resources}
    assert config['remote_root']=='/oscar/scratch/aagraw73/agent-runs'
    assert fields['Partition']=='gpu' and fields['QOS']=='norm-gpu' and fields['TimeLimit']=='00:10:00'
    assert fields['NumCPUs']=='4' and fields['NumNodes']=='1'
    assert fields['WorkDir']==remote and fields['Command']==remote+'/job.sh'
    assert fields['TresPerNode']=='gres/gpu:nvidia_rtx_a6000:1'
    items=[v.split('=',1) for v in fields['AllocTRES'].split(',')]; tres=dict(items);assert len(items)==len(tres)
    assert tres['cpu']=='4' and tres['node']=='1' and tres['gres/gpu']=='1' and tres['mem'] in ('32G','32768M')
    assert sorted(record['files'])==sorted(record['sha256']) and len(set(record['files']))==len(record['files'])
    for n in record['files']:
        p=pathlib.PurePosixPath(n);assert not p.is_absolute() and '..' not in p.parts
    assert {'probe.py','provenance.py','job.sh',CONFIG,'protocol.json'}<=set(record['files'])

def validate():
    job=os.environ.get('SLURM_JOB_ID','');assert re.fullmatch(r'[1-9][0-9]*',job),'Allocated CUDA diagnostic only'
    root=pathlib.Path(os.environ['OSCAR_RUN_DIR']);code=pathlib.Path(__file__).resolve().parent
    assert code==root/'code'
    raw=(root/'source-manifest.json').read_bytes();record=json.loads(raw);config=json.loads((code/CONFIG).read_text());jb=(root/'job-id').read_bytes();assert jb.decode().strip()==job
    stdout=subprocess.run(['scontrol','show','job','-o',job],check=True,capture_output=True,text=True,timeout=10).stdout
    pairs=re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]*)',stdout);fields=dict(pairs);assert len(pairs)==len(fields)
    validate_metadata(record,config,job,os.environ,fields)
    for n,want in record['sha256'].items():
        p=code/n;assert not p.is_symlink() and p.is_file() and p.resolve().is_relative_to(code)
        assert hashlib.sha256(p.read_bytes()).hexdigest()==want
    # Stable return value excludes volatile scheduler runtime fields.
    return dict(job_id=job,run_id=record['run_id'],uploaded_manifest_sha256=hashlib.sha256(raw).hexdigest(),job_id_sha256=hashlib.sha256(jb).hexdigest(),source_sha256=record['sha256'])
