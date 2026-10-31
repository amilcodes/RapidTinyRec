"""Wave-specific native TF32 shape and actual frozen graph dispatch checks."""
from collections import Counter
import json
from pathlib import Path
from .audit_precision_dispatch import api_gemms,trace_kernels


def inspect_dispatch(root,report,variant):
    if variant not in ('serial_pair_tf32','beam_wave_tf32'):raise ValueError('Unknown pair variant')
    root=Path(root);wave=variant=='beam_wave_tf32';batch=100 if wave else 50
    profiles=report['profiles']
    if [(p['prompt_tokens'],p['cohort_indices']) for p in profiles]!=[(76,[1,2]),(121,[6,10])]:
        raise ValueError('Both frozen original model pair profiles required')
    def artifact(name):
        p=root/name
        if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):
            raise ValueError('Dispatch path escapes phase')
        return p
    logs=report['capture_logs'];calls=api_gemms(artifact(logs['cublas']).read_text()) if logs.get('available') else []
    required=[(8960,batch,1536),(1536,batch,8960)]
    mode='CUBLAS_TF32_TENSOR_OP_MATH'
    covered={str(shape):sum(c['shape']==dict(zip(('m','n','k'),shape)) and c['math_mode']==mode for c in calls) for shape in required}
    traces=[trace_kernels(json.loads(artifact(p['trace']).read_text())) for p in profiles]
    # Both serial requests issue four decode graph launches. A native batch2
    # wave issues four in total. Extra/missing launches invalidate this profile.
    launches=4 if wave else 8
    graph_ok=all(t['cuda_graph_launches']==launches and t['graph_kernel_events']>0 for t in traces)
    native_ok=bool(calls) and all(covered.values())
    tf32_ok=all(t['tf32_kernel_symbols'] for t in traces)
    return dict(schema='two-request-beam-wave-dispatch-v1',variant=variant,decode_batch=batch,
                required_shape_calls=covered,api_math_modes=dict(Counter(c['math_mode'] for c in calls)),
                api_gemm_calls=len(calls),profiles=traces,expected_launches_per_pair=launches,
                native_policy_verified=native_ok,graph_replay_verified=graph_ok,tf32_symbols_verified=tf32_ok,
                verified=bool(logs.get('available') and native_ok and graph_ok and tf32_ok),
                limitation='Actual model-sized API GEMM math modes and graph-correlated TF32 kernels, not getter-only or prefill probe evidence. Not every GEMM must accelerate; no speed claim.')


def validate_replay_counts(phase,setup,summary,dispatch):
    if type(phase) is not int or phase not in range(4):raise ValueError('Wrong wave phase')
    wave=phase in (1,2);batch=100 if wave else 50
    counts=[4,10,12] if wave else [8,20,24]
    reports=[(setup['replay'],setup['direct_reorder']), (summary['replay'],summary['direct_reorder']),
             (dispatch['replay_after_profiles'],dispatch['direct_reorder_after_profiles'])]
    for (replay,reorder),requests in zip(reports,counts):
        if replay['graphs']!=8 or replay['frozen'] is not True or replay['replays']!=4*requests:
            raise ValueError('Replay graph count/freeze/request inventory differs')
        events=replay['events']
        if len(events)!=8 or sorted((e['cache_tokens'],e['batch']) for e in events)!=sorted((length+i,batch) for length in (76,121) for i in range(4)):
            raise ValueError('Captured shapes do not cover both frozen prompt lengths')
        prefix=replay['physical_prefix']
        if (prefix['physical_prefix_batch']!=(2 if wave else 1) or prefix['suffix_batch']!=batch
                or prefix['max_prefix_tokens']!=121 or prefix['max_suffix_tokens']!=3
                or prefix['requests_loaded']!=(2*requests if wave else requests)):
            raise ValueError('Physical request/prefix ownership inventory differs')
        if wave and prefix['waves_loaded']!=requests:raise ValueError('Wave inventory differs')
        if (reorder['direct_calls']!=4*requests or reorder['fallback_calls']!=requests
                or reorder['prefix_bytes_reordered']!=0):
            raise ValueError('Five-step suffix reorder inventory differs')
    if summary['complete'] is not True or summary['measured_pairs']!=6 or summary['measured_requests']!=12:
        raise ValueError('Incomplete pair measurement')
    return dict(graphs_per_phase=8,request_major_decode_batch=batch,phase_inventory_verified=True)
