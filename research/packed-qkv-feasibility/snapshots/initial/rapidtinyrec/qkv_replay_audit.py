"""BF16 QKV frozen replay, compilation exclusion and log publication checks."""
from pathlib import Path
from .audit_qkv_outputs import require
from .audit_qkv_phase import read
from .audit_precision import validate_log_staging
from .bf16_policy import validate_state


def validate_runtime_records(path,setup,summary,dispatch,compilation):
    # Explicitly demand staging even though the phase manifest does not carry
    # the legacy optional api_log_staging flag.
    validate_log_staging(Path(path),{'api_log_staging':True})
    for record in (setup,summary,dispatch):validate_state(record['precision_policy'],'folded_bf16')
    require(compilation['phase']=='complete' and compilation['measurement_compilations']==0
            and bool(compilation['events']) and all(e.get('success') or e.get('expected_restart') for e in compilation['events'])
            and not compilation['compiler_counters'].get('graph_break') and not compilation['compiler_counters'].get('unimplemented'),'Compilation incomplete or occurred after freeze')
    require(dispatch['capture_logs']==dict(available=True,cublas='capture-cublas.log',cublaslt='capture-cublaslt.log'),'Original native capture log contract differs')
    require(summary['complete'] is True and summary['measured_requests']==12,'Incomplete measured requests')
    require(dispatch['profiles']==[dict(prompt_tokens=n,cohort_index=i,trace=f'profile-{n}.json',output=f'profile-output-{n}.json') for n,i in ((76,1),(121,6))],'Both frozen profile identities required')
    reports=[(setup['replay'],setup['direct_reorder'],8),(summary['replay'],summary['direct_reorder'],20),(dispatch['replay_after_profiles'],dispatch['direct_reorder_after_profiles'],22)]
    per_token=28*2*50*2*128*2;metadata=101*8
    expected_layouts=sorted((n+j,50) for n in (76,121) for j in range(4))
    first=reports[0][0]
    for replay,reorder,requests in reports:
        require(replay['graphs']==8 and replay['frozen'] is True and replay['replays']==4*requests,'Frozen replay count differs')
        require(replay['events']==first['events'] and len(replay['events'])==8 and sorted((e['cache_tokens'],e['batch']) for e in replay['events'])==expected_layouts,'Graph events changed or shapes incomplete')
        require(replay['shared_input_arena'] is True
                and replay['input_buffer_bytes']==sum(per_token//50*n+per_token*j+metadata for n in (76,121) for j in range(4))
                and replay['unique_input_storage_bytes']==per_token//50*121+per_token*3+8*metadata,'BF16 arena byte accounting differs')
        prefix=replay['physical_prefix']
        require(prefix['physical_prefix_batch']==1 and prefix['suffix_batch']==50 and prefix['max_prefix_tokens']==121
                and prefix['max_suffix_tokens']==3 and prefix['requests_loaded']==requests,'Physical prefix request inventory differs')
        require(reorder['direct_calls']==4*requests and reorder['fallback_calls']==requests and reorder['prefix_bytes_reordered']==0
                and reorder['suffix_bytes_written']==per_token*6*requests,'Native five-step suffix reorder differs')
        for event in replay['events']:
            n,j=next((n,j) for n in (76,121) for j in range(4) if n+j==event['cache_tokens'])
            require(event['input_buffer_bytes']==per_token//50*n+per_token*j+metadata,'Per-graph input byte accounting differs')
    require(summary['replay']['physical_prefix']['prefix_bytes_copied']-setup['replay']['physical_prefix']['prefix_bytes_copied']==per_token//50*3*(76+76+121+121),'Timed original prefix staging byte delta differs')
    return dict(graphs=8,measured_requests=12,post_profile_requests=22,measurement_compilations=0,
                supplied_replay_metadata_verified=True,native_dispatch_verified=False)
