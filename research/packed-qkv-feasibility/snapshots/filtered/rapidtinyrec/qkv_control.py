"""Fresh QKV cross-phase and historical BF16 control components.

Historical raw files are authenticated copies, never re-generated or re-audited.
Only the future allocated auditor may compare their scores with new measurements.
"""
import hashlib
import json
from pathlib import Path
from .audit_qkv_outputs import ARMS,USERS,require,compare_outputs
from .audit_qkv_phase import read
REFERENCE_SHA256='a2ca4057281f30a0c68daf54eb86a60647a71ed8a5c465ced6fe2e215e849464'
PARENT_JOB='8078579'
PARENT_RUN='bf16-vs-tf32-full500-quality-20261029T001048Z-32acac83dd02'


def digest_bytes(value):return hashlib.sha256(value).hexdigest()
def json_digest(value):return digest_bytes(json.dumps(value,sort_keys=True).encode())


def authenticate_reference_files(path):
    path=Path(path)
    raw=(path/'reference.json').read_bytes()
    require(digest_bytes(raw)==REFERENCE_SHA256,'Historical reference lock differs')
    lock=read(path/'reference.json')
    require(lock['schema']=='qkv-historical-bf16-reference-v1' and lock['job_id']==PARENT_JOB and lock['run_id']==PARENT_RUN,'Historical experiment differs')
    require(set(lock['files'])=={'manifest.json','0001.json','0002.json','0006.json','0010.json'},'Historical file coverage differs')
    for name,row in lock['files'].items():
        file=path/name
        require(not file.is_symlink() and file.is_file(),'Historical file missing or symlinked')
        data=file.read_bytes()
        require(len(data)==row['bytes'] and digest_bytes(data)==row['sha256'],'Historical raw bytes differ: '+name)
    return lock


def load_historical(path,users,expected):
    lock=authenticate_reference_files(path);path=Path(path)
    manifest=read(path/'manifest.json')
    require(manifest['variant']=='folded_bf16' and manifest['phase']==1 and str(manifest['job_id'])==PARENT_JOB,'Historical BF16 phase differs')
    require(len(manifest['users'])==500 and json_digest(manifest['users'])==expected['users_sha256']==manifest['cohort_sha256'],'Historical cohort identity differs')
    selected=[next(u for u in manifest['users'] if u['cohort_index']==i) for i in USERS]
    require(users==selected,'New cohort differs from historical user/prompt/target identities')
    result={}
    for user in users:
        i=user['cohort_index'];row=read(path/f'{i:04d}.json')
        require(row['cohort_index']==i and row['prompt_tokens']==user['prompt_tokens'] and row['variant']=='folded_bf16'
                and row['repeat']==0 and row['measurement_role']=='primary_timed','Historical primary observation differs')
        result[i]=row['run']
    return result,dict(reference_lock_sha256=REFERENCE_SHA256,files=lock['files'],selected_users=list(USERS),full_scores_available=False,numerical_reaudit_performed=False)


def portable_inventory(snapshot):
    """Cross-process equality ignores addresses/objects only; layout and bytes stay."""
    return {name:{k:row[k] for k in ('shape','stride','dtype','logical_bytes','sha256','storage_offset','storage_bytes','device')} for name,row in snapshot.items()}


def phase_outputs(path):
    path=Path(path);outputs=[]
    for file in sorted((path/'warmup_outputs').glob('*.json')):
        run=read(file);outputs.append((file.name,run['cohort_index'],run))
    for row in read(path/'requests.json'):
        outputs.append((f"timed_{row['repeat']}_{row['cohort_index']}",row['cohort_index'],row['run']))
    for length in (76,121):
        run=read(path/f'profile-output-{length}.json');outputs.append((f'profile_{length}',run['cohort_index'],run))
    require(len(outputs)==32,'Complete per-phase outputs required before cross-phase audit')
    return outputs


def compare_cross_phase(paths,historical):
    """Call only after independent per-phase coverage/validity/ownership checks."""
    require(set(paths)==set(range(4)) and set(historical)==set(USERS),'Four phases and four historical users required')
    originals=[portable_inventory(read(Path(paths[p])/'immutable-original.json')) for p in range(4)]
    require(all(v==originals[0] for v in originals[1:]),'Original input/model/catalog bytes or layout differ across phases')
    installed={p:portable_inventory(read(Path(paths[p])/'immutable-installed.json')) for p in range(4)}
    require(installed[0]==installed[3] and installed[1]==installed[2],'Installed same-arm bytes/layout differ across halves')
    anchors={p:{i:read(Path(paths[p])/'warmup_outputs'/f'{i}_direct.json') for i in USERS} for p in (0,1)}
    repeat_checks=[];historical_checks=[]
    for phase in range(4):
        for label,i,run in phase_outputs(paths[phase]):
            if phase in (2,3):
                reference_phase=1 if phase==2 else 0
                repeat_checks.append(dict(phase=phase,name=label,cohort_index=i,**compare_outputs(anchors[reference_phase][i],run)))
            if phase in (0,3):
                historical_checks.append(dict(phase=phase,name=label,cohort_index=i,**compare_outputs(historical[i],run,full_scores=False)))
    require(len(repeat_checks)==len(historical_checks)==64,'Control comparison coverage differs')
    return dict(cross_phase=repeat_checks,cross_phase_exact=all(x['exact'] for x in repeat_checks),
                historical=historical_checks,historical_exact=all(x['exact'] for x in historical_checks),
                original_bytes_match_across_phases=True,installed_bytes_match_within_arm=True,
                historical_full_scores_compared=False,population_equivalence_established=False)
