"""Aggregate independent proofs; runtime artifact provenance is not compiler proof."""
from .audit_qkv_outputs import require

def finalize_components(result):
    require(set(result['phases'])==set(range(4)),'Four verified phases required')
    for phase,p in result['phases'].items():
        native=p['native_dispatch'];runtime=p['runtime_identity'];loaded=p['loaded_binaries'];capture=p['capture_binding']
        require(all(native[k] is True for k in ('native_bf16_geometry_verified','generated_matrix_inventory_verified','graph_bf16_corroborated')),'Native dispatch corroboration missing')
        require(runtime['installed_source_and_run_code_verified'] is True and loaded['actual_binary_load_identity_verified'] is True,'Installed runtime or actual loaded artifact unbound')
        require(runtime['runtime_calls']==loaded['captured_runners']>0,'Runtime/load coverage differs')
        require(capture['actual_capture_wrapper_identity_verified'] is True and len(capture['graphs'])==8,'Capture identity coverage incomplete')
        require({g['storage_graph_index'] for g in capture['graphs']}==set(range(8)),'Repeated capture storage graph')
        for graph in capture['graphs']:
            m=graph['model_wrapper'];layout=m['attention_layout'];copy=m['cache_copy']
            require(m['matrix_calls']==(141 if phase in (1,2) else 197),'Arm matrix inventory differs')
            require(layout['independent_attention_extents_verified'] is True and layout['final_attention_store_coverage_verified'] is True and len(layout['attention'])==28,'Attention domain/store proof incomplete')
            require(copy['prefix_suffix_copy_verified'] is True and len(copy['copies'])==56,'Cache content copy proof incomplete')
            launches=m['actual_launches']
            require(launches['source_to_actual_launcher_parameters_bound'] is True and bool(launches['launches']),'Actual launch arguments missing')
            for launch in launches['launches']:
                binding=launch['compile_result_binding']
                require(binding['runtime_compile_result_and_jit_source_verified'] is True and binding['implicit_elision_verified'] is True,'Compile-result/source specialization missing')
                require(launch['runner_binding']['actual_launcher_global_and_forwarded_arguments_bound'] is True,'Actual runner forwarding missing')
            if phase in (1,2):
                maps=m['packed_maps'];require(maps['exact_packed_columns_and_masks'] is True and len(maps['maps'])==84,'Packed Q/K/V column/mask coverage incomplete')
            else:require(m['packed_maps'] is None,'Original arm has packed proof')
    result['runtime_artifact_provenance_verified']=True
    result['compiled_column_cache_store_maps_verified']=True
    result['complete_dispatch_verified']=True
    result['source_to_binary_compilation_verified']=False
    result['compiler_correctness_limitation']='Installed runtime, observed compile-result/JIT source, actual loaded artifact/handles and native dispatch are correlated. Compiler transformations are not independently reexecuted or formally verified. Strict full-score/output bits remain mandatory.'
    result['control_exact']=result['controls']['historical_exact'] is True and result['controls']['cross_phase_exact'] is True
    result['stability_exact']=all(p['evidence']['outputs']['same_arm_exact'] is True for p in result['phases'].values())
    result['transition_sum_exact']=all(p['evidence']['outputs']['sum_exact'] is True for p in result['phases'].values())
    result['cross_arm_exact']=result['timing']['cross_arm_exact'] is True
    result['validity_passed']=result['quality']['validity_passed'] is True and all(p['evidence']['outputs']['validity_passed'] is True for p in result['phases'].values())
    result['execution_audit_passed']=all(result[k] is True for k in ('integrity_audit_passed','complete_dispatch_verified','control_exact','stability_exact','transition_sum_exact','cross_arm_exact','validity_passed'))
    result['complete_protocol_certified']=result['execution_audit_passed']
    result['pilot_performance_gate_passed']=result['execution_audit_passed'] and result['timing']['performance_passed'] is True
    result['reviewed_full500_eligible']=False
    result['promotion']=False
    result['remaining']=['Terminal audit/work exit receipts and local metadata reconciliation before independent result review; no automatic full500 or adoption.']
    return result
