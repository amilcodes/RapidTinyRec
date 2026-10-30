"""Native BF16 geometry, generated-wrapper and each-graph kernel corroboration.

No getter-only proof, no prefill-only kernel evidence, no latency claim. Exact
fused compiled Q/K/V index maps remain an explicitly unverified separate gate.
"""
import hashlib
import json
import re
from pathlib import Path
from .audit_qkv_outputs import require
from .audit_bf16_dispatch import api_bf16_gemms
from .qkv_generated_matrix import inspect_wrapper


def graph_bf16_evidence(trace):
    events=trace['traceEvents']
    launches=[e for e in events if e.get('name')=='cudaGraphLaunch' and e.get('cat') in ('cuda_runtime','cuda_driver')]
    ids=[e.get('args',{}).get('correlation') for e in launches]
    require(len(ids)==4 and len(set(ids))==4 and all(type(i) is int and i>=0 for i in ids),'Exactly four distinct decode launches required')
    result=[]
    for correlation in ids:
        kernels=[e for e in events if e.get('cat')=='kernel' and e.get('dur',0)>0 and e.get('args',{}).get('correlation')==correlation]
        bf16=[e for e in kernels if re.search(r'bf16|bfloat16',e.get('name',''),re.I)]
        require(bool(kernels) and bool(bf16),'A decode graph lacks correlated BF16 kernel evidence')
        result.append(dict(correlation=correlation,kernel_events=len(kernels),bf16_kernel_events=len(bf16),bf16_symbols=sorted({e['name'] for e in bf16})))
    return dict(launches=4,per_launch=result,prefill_kernels_used=False,instruction_level_proof=False)


def inspect_native_dispatch(root,report,variant):
    root=Path(root).resolve(strict=True)
    require(variant in ('original_bf16','packed_qkv_bf16'),'Unknown QKV arm')
    packed=variant=='packed_qkv_bf16'
    def artifact(name):
        rel=Path(name);p=root/rel
        require(not rel.is_absolute() and '..' not in rel.parts and p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root),'Unsafe/missing dispatch artifact')
        return p
    logs=report['capture_logs']
    require(logs==dict(available=True,cublas='capture-cublas.log',cublaslt='capture-cublaslt.log'),'Native logger capture contract differs')
    calls=api_bf16_gemms(artifact(logs['cublas']).read_text())
    shapes={(8960,50,1536):56,(1536,50,8960):28}
    shapes.update({(2048,50,1536):28} if packed else {(1536,50,1536):28,(256,50,1536):56})
    counts={}
    for shape,minimum in shapes.items():
        matches=[c for c in calls if c['shape']==dict(zip(('m','n','k'),shape))]
        require(len(matches)>=minimum and all(c['bf16_inputs'] and c['bf16_output'] and c['fp32_compute'] for c in matches),'Missing or wrong native BF16 geometry/type/compute')
        counts[str(shape)]=len(matches)
    if packed:require(not any(c['shape']==dict(m=256,n=50,k=1536) for c in calls),'Standalone K/V decode GEMMs remain in packed arm')
    profiles=report['profiles']
    require([(p['prompt_tokens'],p['cohort_index']) for p in profiles]==[(76,1),(121,6)],'Both declared request profiles required')
    traces=[graph_bf16_evidence(json.loads(artifact(p['trace']).read_text())) for p in profiles]
    generated=report['generated_code'];require(bool(generated),'Missing generated code')
    unique={};wrappers=[]
    for record in generated:
        p=artifact(record['file']);data=p.read_bytes();digest=hashlib.sha256(data).hexdigest()
        require(digest==record['sha256'] and record['file']==f'generated-code/{digest}.py','Generated source hash/name differs')
        require(record['cache'] in ('TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR'),'Unexpected generated cache')
        if digest not in unique:
            unique[digest]=True;proof=inspect_wrapper(data.decode(),packed)
            if proof is not None:wrappers.append(dict(sha256=digest,**proof))
    require(bool(wrappers),'No complete model matrix wrapper verified')
    return dict(schema='qkv-native-dispatch-v1',variant=variant,native_bf16_geometry_verified=True,
                native_shape_counts=counts,profiles=traces,wrappers=wrappers,
                generated_matrix_inventory_verified=True,graph_bf16_corroborated=True,
                compiled_split_view_verified=False,complete_dispatch_verified=False,
                limitation='API geometry/types, complete generated matrix wrappers and per-launch model graph BF16 symbols are corroboration; exact fused downstream Q/K/V indexing and wrapper-to-captured-graph identity still require separate proof.')
