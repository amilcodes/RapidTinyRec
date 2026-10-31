"""Bind observed generated calls to every frozen graph and native matrix proof."""
import hashlib
from pathlib import Path
from .audit_qkv_outputs import require
from .qkv_capture_binding import source_call_fingerprint
from .qkv_generated_matrix import inspect_wrapper
from .qkv_wrapper_dataflow import packed_kernel_dataflow
from .qkv_semantic_routes import bind_attention_routes
from .qkv_capture_scalars import bind_capture_scalars
from .qkv_launch_binding import bind_launch_observations
from .qkv_attention_layout import validate_attention_layout
from .qkv_packed_maps import validate_packed_maps
from .qkv_cache_inputs import bind_cache_inputs,bind_cache_returns
from .qkv_cache_copy import validate_cache_copy


def validate_capture_binding(root,report,dispatch,storage,variant):
    root=Path(root)
    require(report['schema']=='packed-qkv-capture-bindings-v1' and report['profiling_scope']=='setup_before_first_compiled_warmup'
            and report['measurement_profiler_installed'] is False,'Capture observer scope differs')
    bindings=report['graphs'];require(len(bindings)==8 and len({b['graph_object_id'] for b in bindings})==8,'Eight distinct captured graph identities required')
    generated={(r['cache'],r['relative'],r['sha256']) for r in dispatch['generated_code']}
    remaining=list(storage['graphs']);proofs=[];packed_map_cache={};cache_copy_cache={};store_cache={}
    for binding in bindings:
        matches=[g for g in remaining if g['cache_tokens']==binding['cache_tokens'] and g['tokens']==binding['tokens'] and g['logits']==binding['logits']]
        require(len(matches)==1,'Captured code is not bound to exactly one supplied frozen graph')
        remaining.remove(matches[0]);calls=binding['wrappers'];require(bool(calls),'No actual call identity')
        model_wrappers=[]
        for call in calls:
            require((call['cache'],call['relative'],call['sha256']) in generated,'Capture wrapper missing from generated inventory')
            digest=call['sha256'];require(len(digest)==64 and all(c in '0123456789abcdef' for c in digest),'Invalid source digest')
            path=root/'generated-code'/f'{digest}.py';require(path.is_file() and not path.is_symlink(),'Missing source artifact')
            data=path.read_bytes();require(hashlib.sha256(data).hexdigest()==digest,'Capture source hash differs')
            require(source_call_fingerprint(data,str(path))==call['call_code_sha256'],'Captured call bytecode does not match source')
            proof=inspect_wrapper(data.decode(),variant=='packed_qkv_bf16')
            if proof is not None:
                cache_inputs=bind_cache_inputs(data.decode(),call,matches[0])
                cache_returns=bind_cache_returns(data.decode(),call,matches[0])
                scalars=bind_capture_scalars(data.decode(),call['scalar_observation'])
                launches=bind_launch_observations(data.decode(),scalars,call['launch_observations'])
                dataflow=packed_kernel_dataflow(data.decode()) if variant=='packed_qkv_bf16' else None
                roles=bind_attention_routes(data.decode(),dataflow) if dataflow is not None else None
                attention_layout=validate_attention_layout(data.decode(),scalars,launches,binding['cache_tokens'],store_cache)
                packed_maps=validate_packed_maps(data.decode(),roles,attention_layout,launches,matches[0],packed_map_cache) if roles is not None else None
                cache_copy=validate_cache_copy(data.decode(),launches,cache_inputs,cache_returns,attention_layout,matches[0],cache_copy_cache)
                model_wrappers.append(dict(sha256=digest,matrix_calls=proof['matrix_calls'],packed_dataflow=dataflow,semantic_routes=roles,capture_scalars=scalars,actual_launches=launches,attention_layout=attention_layout,packed_maps=packed_maps,cache_inputs=cache_inputs,cache_returns=cache_returns,cache_copy=cache_copy))
        require(len(model_wrappers)==1,'Expected exactly one complete model wrapper call per captured graph')
        proofs.append(dict(storage_graph_index=matches[0]['index'],cache_tokens=binding['cache_tokens'],model_wrapper=model_wrappers[0]))
    require(not remaining,'Frozen graph missing source identity')
    return dict(graphs=proofs,actual_capture_wrapper_identity_verified=True,compiled_split_view_verified=False,
                limitation='Executed generated call bytecode/source and supplied token/logit owners bind all eight captures. Fused downstream Q/K/V index expressions remain a separate gate; no instruction-level proof.')
