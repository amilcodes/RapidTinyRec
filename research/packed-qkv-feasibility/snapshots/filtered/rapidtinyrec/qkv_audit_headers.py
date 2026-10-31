"""QKV phase source/runtime/cache/deadline metadata checks for allocated audit.

Caller must independently authenticate live allocation and the source record.
These helpers neither contact the scheduler nor certify complete execution.
"""
import hashlib
import math
from pathlib import Path
from .audit_qkv_outputs import ARMS,USERS,require
from .audit_qkv_phase import read
from .bf16_policy import validate_state
from .precision_provenance import validate_release
from .qkv_control import json_digest
TORCH_COMMIT='08187d9e0fba026dc8217405802ab5381dc88d90'


def canonical_cache_path(value):
    require(isinstance(value,str) and Path(value).is_absolute() and '..' not in Path(value).parts
            and str(Path(value))==value,'Compiler cache path is not canonical absolute metadata')
    return Path(value)


def validate_reference(reference,expected):
    require(reference['experiment']=='gqa_three_way' and reference['dtype']=='float32','Original cohort reference experiment differs')
    require(json_digest({k:v for k,v in reference.items() if k!='manifest_sha256'})==reference['manifest_sha256'],'Original cohort manifest hash differs')
    require(len(reference['users'])==500 and json_digest(reference['users'])==expected['users_sha256'],'Original cohort user hash differs')
    users=[next(u for u in reference['users'] if u['cohort_index']==i) for i in USERS]
    require({u['cohort_index']:u['prompt_tokens'] for u in users}==USERS,'QKV pilot cohort differs')
    return users


def validate_work_deadline(work,allocation):
    require(work['job_id']==allocation['job_id'] and work['run_id']==allocation['run_id'],'Phase work deadline identity differs')
    start,end=work['started_monotonic'],work['deadline_monotonic']
    # Match the declared deadline construction exactly. Subtracting floats can
    # differ from1620 by one ULP even for end = start +1620; no tolerance needed.
    require(all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in (start,end))
            and end==start+1620,'Inclusive global work deadline differs')


def validate_phase_header(m,phase,allocation,protocol_bytes,users,expected,reference_sha,quality_hashes,package_sources):
    require(type(phase) is int and phase in range(4),'Unknown phase')
    require(m['schema']=='packed-qkv-phase-v1' and m['phase']==phase and m['half']==phase//2 and m['variant']==ARMS[phase],'QKV phase/arm identity differs')
    require(m['protocol_sha256']==hashlib.sha256(protocol_bytes).hexdigest(),'QKV protocol hash differs')
    import json
    protocol=json.loads(protocol_bytes)
    require(protocol['schema']=='packed-qkv-bf16-pilot-v1' and protocol['phases'][phase]==dict(phase=phase,arm=ARMS[phase],half=phase//2,reuse_disk_artifacts=phase>=2),'Protocol phase contract differs')
    require(allocation['submission']=='allocated_observation' and m['allocation']==allocation,'Phase allocation/source observation differs')
    require(m['job_id']==allocation['job_id'] and m['run_id']==allocation['run_id'],'Phase job/run differs')
    require(m['users']==users and m['cohort_sha256']==expected['users_sha256'] and m['reference_manifest_sha256']==reference_sha,'Phase cohort differs')
    require(m['source_sha256']==package_sources and all(allocation['sha256'].get(k)==v for k,v in package_sources.items()),'Phase source files differ')
    require(m['hardware']=='NVIDIA RTX A6000' and m['dtype']=='bfloat16' and m['attention_layout']=='folded_query_heads','Runtime hardware/precision/attention differs')
    require(m['versions']=={'torch':'2.14.0+cu126','transformers':'4.55.4','tokenizers':'0.21.4'} and m['torch_git']==TORCH_COMMIT and m['cuda_version']=='12.6','Pinned runtime differs')
    require(m['compiler_policy']==dict(max_autotune=False,max_autotune_gemm=False,max_autotune_pointwise=False),'Compiler algorithm policy differs')
    require(m['timing_contract']==protocol['timing'],'Original timing scope differs')
    require(type(m['model_load_s']) in (int,float) and math.isfinite(m['model_load_s']) and m['model_load_s']>0,'Missing load timing')
    validate_state(m['precision_policy'],'folded_bf16')
    validate_release(m['release_verification'],expected['release_sha256']);validate_release(m['quality_release_verification'],quality_hashes)
    validate_work_deadline(m['work_deadline'],allocation)
    return dict(phase=phase,source_files=len(package_sources),metadata_passed=True,allocation_independently_verified=False)


def validate_cache_partitions(manifests,setups):
    require(set(manifests)==set(setups)==set(range(4)),'Four cache partitions required')
    paths={}
    for phase in range(4):
        artifacts=manifests[phase]['compiler_artifacts'];before=artifacts['before'];after=setups[phase]['compiler_artifacts_after']
        require(artifacts['reused'] is (phase>=2),'Cold/reuse cache role differs')
        require(set(before)==set(after)=={'TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR'},'Compiler cache family differs')
        for family in before:
            a,b=before[family],after[family]
            require(a['path']==b['path'],'Compiler cache path changed')
            canonical_cache_path(a['path'])
            require(all(type(r['files']) is int and r['files']>=0 for r in (a,b)),'Invalid cache file count')
        require(sum(r['files'] for r in after.values())>0,'Missing compiler artifacts')
        require((sum(r['files'] for r in before.values())>0)==(phase>=2),'Incorrect cold/reuse artifact presence')
        paths[phase]={k:v['path'] for k,v in before.items()}
    require(paths[0]==paths[3] and paths[1]==paths[2],'Same-arm cache path differs across halves')
    distinct=[Path(p) for phase in (0,1) for p in paths[phase].values()]
    require(len(set(distinct))==4 and all(a not in b.parents and b not in a.parents for i,a in enumerate(distinct) for b in distinct[i+1:]),'Compiler cache trees overlap')
    return dict(fresh_phases=[0,1],reused_phases=[2,3],distinct_cache_trees=4)


def validate_closures(manifests,closures):
    require(set(manifests)==set(closures)==set(range(4)),'Four phase closures required')
    work=manifests[0]['work_deadline'];previous=work['started_monotonic']
    for phase in range(4):
        c=closures[phase]
        require(manifests[phase]['work_deadline']==work,'Different global deadline per process')
        require(c['complete'] is True and c['original_identity_restored'] is True,'Incomplete phase teardown')
        end=c['ended_monotonic']
        require(type(end) in (int,float) and math.isfinite(end) and previous<end<work['deadline_monotonic'],'Out-of-order or late phase closure')
        previous=end
    return dict(ordered=True,global_deadline_respected=True,allocation_stage_log_closure_verified=False)
