"""Require native BF16 GEMM types and model-graph BF16 kernel evidence."""
import json
import re
from pathlib import Path
from .audit_precision_dispatch import dispatch_evidence as tf32_dispatch, trace_kernels

def api_bf16_gemms(text):
    calls=[]
    for block in re.split(r'(?=I! cuBLAS .*? function )',text):
        if not re.search(r'function .*?cublasGemmEx\(',block):continue
        shape={}
        for key in ('m','n','k'):
            m=re.search(r'\b'+key+r': type=int; val=(\d+)',block)
            if m:shape[key]=int(m.group(1))
        def value(key):
            m=re.search(r'\b'+key+r': type=[^;\n]+; val=([^\n]+)',block,re.I)
            return m.group(1).strip() if m else None
        types={k:value(k) for k in ('Atype','Btype','Ctype','computeType')}
        bf16=lambda x:x is not None and bool(re.fullmatch(r'(?:CUDA_R_16BF(?:\(14\))?|14)',x))
        # Pinned ATen uses the legacy CUDA_R_32F overload on CUDA; newer
        # cuBLAS logs can print its equivalent CUBLAS_COMPUTE_32F value.
        fp32=lambda x:x is not None and bool(re.fullmatch(r'(?:CUBLAS_COMPUTE_32F(?:\(68\))?|68|CUDA_R_32F(?:\(0\))?|0)',x))
        calls.append(dict(shape=shape,types=types,bf16_inputs=bf16(types['Atype']) and bf16(types['Btype']),
            bf16_output=bf16(types['Ctype']),fp32_compute=fp32(types['computeType'])))
    return calls

def dispatch_evidence(root,report,candidate):
    if not candidate:return tf32_dispatch(root,report,True)
    root=Path(root);logs=report['capture_logs']
    calls=api_bf16_gemms((root/logs['cublas']).read_text()) if logs.get('available') else []
    shapes=[(8960,50,1536),(1536,50,8960)]
    covered={str(s):sum(c['shape']==dict(zip(('m','n','k'),s)) and c['bf16_inputs'] and c['bf16_output'] and c['fp32_compute'] for c in calls) for s in shapes}
    traces=[trace_kernels(json.loads((root/t['trace']).read_text())) for t in report['profiles']]
    symbols=[{n:c for n,c in t['graph_kernels'].items() if re.search(r'bf16|bfloat16',n,re.I)} for t in traces]
    replay=len(traces)==2 and all(t['cuda_graph_launches']>=4 and t['graph_kernel_events']>0 for t in traces)
    verified=bool(logs.get('available') and calls and all(covered.values()) and replay and all(symbols))
    return dict(verified=verified,api_gemm_calls=len(calls),required_shape_calls=covered,
        native_bf16_calls=calls,graph_bf16_kernel_symbols=symbols,replay_profiles=traces,
        limitation='Native GEMM BF16 operands/outputs with FP32 compute, corroborated by actual frozen-model graph BF16 symbols. No fallback assumption or instruction-level claim.')
