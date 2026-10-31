"""Independent result-shape/bit-record/ownership checks, not an execution proof.

No imports from producer. CUDA graph binding review remains a separate required
gate; this component never marks a CUDA experiment submission-ready.
"""
import re

ORDERS=('forward','repeat','reverse','restored')
SITES=('first','second')
COMPARISONS=({f'control_repeat_{s}' for s in SITES} |
             {f'{o}_{s}' for o in ORDERS for s in SITES} |
             {f'{o}_retained_{s}' for o in ORDERS for s in SITES} |
             {'source_untouched','reverse_source_untouched'})


def require(value,message):
    if not value:raise ValueError(message)


def digest(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None,'Invalid bit digest')


def tensor(m, rows, width, device):
    require(set(m)=={'shape','stride','dtype','device','storage_base','storage_bytes','pointer','offset','pointer_mod16'},'Tensor metadata fields differ')
    require(m['shape']==[rows,width] and m['stride']==[width,1] and m['dtype']=='torch.float32' and m['device']==device,'Tensor geometry/device differs')
    for k in ('storage_base','storage_bytes','pointer','offset','pointer_mod16'):
        require(type(m[k]) is int and m[k]>=0,'Invalid integer metadata')
    require(m['storage_base']>0 and m['storage_base']%16==0,'Storage base alignment differs')
    require(m['pointer']==m['storage_base']+4*m['offset'] and m['pointer_mod16']==m['pointer']%16,'Pointer/offset mismatch')
    require(4*(m['offset']+rows*width)<=m['storage_bytes'],'View exceeds storage')


def disjoint(a,b):
    return a['storage_base']+a['storage_bytes']<=b['storage_base'] or b['storage_base']+b['storage_bytes']<=a['storage_base']


def audit_case(record, *, width=152265, device='cuda:0'):
    require(type(record['seed']) is int and record['seed'] in (17,29),'Seed not declared')
    require(type(record['offset']) is int and record['offset'] in range(4),'Shift not declared')
    require(record['vocabulary']==width and record['mode'] in ('eager','cuda_graph'),'Case geometry/mode differs')
    require(set(record['comparisons'])==COMPARISONS,'Missing/extra comparison')
    exact=True
    for name,c in record['comparisons'].items():
        require(set(c)=={'exact','differing_elements','left_sha256','right_sha256'},'Comparison fields differ')
        require(type(c['differing_elements']) is int and 0<=c['differing_elements']<=100*width,'Invalid difference count')
        digest(c['left_sha256']);digest(c['right_sha256'])
        expected=c['differing_elements']==0
        require(type(c['exact']) is bool and c['exact']==expected,'Contradictory exact flag')
        require((c['left_sha256']==c['right_sha256'])==expected,'Contradictory hashes')
        exact=exact and expected
    for site in SITES:
        reference=record['comparisons']['control_repeat_'+site]['left_sha256']
        require(all(record['comparisons'][order+'_'+site]['left_sha256']==reference for order in ORDERS),'Order comparisons use different controls')
    require([m['order'] for m in record['metadata']]==list(ORDERS),'Order coverage differs')
    for order in record['metadata']:
        observations=order['observations']
        require(len(observations)==4,'Two sites/two requests required')
        require([(m['site'],m['request_position']) for m in observations]==[(s,i) for s in SITES for i in range(2)],'Site/request ownership coverage differs')
        for n,m in enumerate(observations):
            require(type(m['request_position']) is int,'Request position must be integer')
            require(set(m)=={'site','step','request_position','source_slice','owned_input','normalized_output'} and type(m['step']) is int and m['step']==0,'Primitive observation fields/step differ')
            for k in ('source_slice','owned_input','normalized_output'):tensor(m[k],50,width,device)
            source,owned,out=(m[k] for k in ('source_slice','owned_input','normalized_output'))
            require(owned['offset']==0 and owned['pointer_mod16']==0,'Owned input is not independently aligned')
            require(out['offset']==0,'Native output must own its start')
            require(disjoint(source,owned) and disjoint(source,out) and disjoint(owned,out),'Live source/input/output alias')
            expected_offset=(record['offset'] if m['site']=='first' else 0)+m['request_position']*50*width
            require(source['offset']==expected_offset,'Source request slice differs')
            if m['request_position']==1:
                prior=observations[n-1]
                require(source['storage_base']==prior['source_slice']['storage_base'] and source['storage_bytes']==prior['source_slice']['storage_bytes'],'Request blocks do not share declared source')
                require(disjoint(prior['normalized_output'],owned) and disjoint(prior['normalized_output'],out),'Prior live normalized output aliased')
    for flag in ('live_outputs_disjoint','finite_outputs','passed','model_executed','production_latency_claim'):
        require(type(record[flag]) is bool,'Boolean flag type differs')
    require(record['model_executed'] is False and record['production_latency_claim'] is False,'Primitive scope overstated')
    passed=exact and record['live_outputs_disjoint'] and record['finite_outputs']
    require(record['passed']==passed,'Producer gate differs')
    return dict(case=(record['seed'],record['offset'],record['mode']),numerical_records_passed=passed,
                ownership_metadata_passed=True,execution_certified=False)


def audit_inventory(records, *, width=152265, device='cuda:0'):
    expected={(s,o,m) for s in (17,29) for o in range(4) for m in ('eager','cuda_graph')}
    observed=[(r['seed'],r['offset'],r['mode']) for r in records]
    require(len(observed)==16 and set(observed)==expected,'Exact sixteen-case inventory required')
    results=[audit_case(r,width=width,device=device) for r in records]
    return dict(cases=16,comparisons=320,records_passed=all(r['numerical_records_passed'] for r in results),
                execution_certified=False,cuda_graph_bindings_verified=False,submission_ready=False)


def audit_joined_case(record, *, width=152265, device='cuda:0'):
    result=audit_case(record,width=width,device=device)
    names={'source','reverse','source_saved','reverse_saved','serial_0','serial_1'}
    names|={label+'_'+s for label in ('reference','control_repeat') for s in SITES}
    names|={label+'_'+o+'_'+s for label in ('result','snapshot') for o in ORDERS for s in SITES}
    graph=record['mode']=='cuda_graph'
    if graph:names|={'graph_input','graph_first','graph_second'}
    live=record['live_tensors'];require(set(live)==names,'Complete live owner inventory required')
    for name,m in live.items():
        tensor(m,50 if name.startswith('serial_') else 100,width,device)
        offset=record['offset'] if name in ('source','reverse','graph_input') else 0
        require(m['offset']==offset,'Live owner offset differs')
    spans=list(live.values())
    independent_disjoint=all(disjoint(a,b) for i,a in enumerate(spans) for b in spans[i+1:])
    require(record['live_outputs_disjoint']==independent_disjoint,'Live disjoint flag differs from spans')
    ident=record['input_identity'];g=record['input_generation']
    require(set(ident)=={'source_sha256','reverse_sha256','source_blocks','serial_blocks','reverse_blocks'},'Input identity fields differ')
    for k in ('source_blocks','serial_blocks','reverse_blocks'):
        require(type(ident[k]) is list and len(ident[k])==2,'Input block coverage differs')
        for h in ident[k]:digest(h)
    for k in ('source_sha256','reverse_sha256'):digest(ident[k])
    require(ident['source_blocks']==ident['serial_blocks'] and ident['reverse_blocks']==ident['source_blocks'][::-1],'Controls or reverse inputs differ')
    require(record['comparisons']['source_untouched']['left_sha256']==ident['source_sha256'] and record['comparisons']['reverse_source_untouched']['left_sha256']==ident['reverse_sha256'],'Saved input comparison not bound')
    require(g['algorithm']=='torch.randn-dedicated-generator-v1' and g['seed']==record['seed'] and g['shape']==[100,width] and g['dtype']=='torch.float32' and g['device']==device,'Generation specification differs')
    for k in ('initial_generator_sha256','final_generator_sha256','generated_sha256','staged_sha256'):digest(g[k])
    require(g['generated_sha256']==g['staged_sha256']==ident['source_sha256'],'Generated/staged/source bytes differ')
    if graph:
        evidence=record['graph'];bindings=evidence['bindings']
        require(bindings==dict(input=live['graph_input'],outputs=[live['graph_first'],live['graph_second']]),'Static graph bindings differ')
        require(len(evidence['receipts'])==4,'Four replays required')
        for i,receipt in enumerate(evidence['receipts']):
            order=ORDERS[i];source='reverse' if order=='reverse' else 'source'
            require(type(receipt['replay']) is int and receipt['replay']==i and receipt['bindings']==bindings,'Replay identity/bindings differ')
            require(receipt['source']==live[source] and receipt['source_sha256']==receipt['staged_sha256']==ident[source+'_sha256'],'Replay source staging not bound')
            require(receipt['retained_outputs']==[live['result_'+order+'_'+s] for s in SITES],'Returned output owners differ')
            require(record['metadata'][i]['observations']==evidence['capture_observations'],'Capture metadata changed across replay')
    else:require(record['graph'] is None,'Eager case carries graph evidence')
    for order in record['metadata']:
        source=live['graph_input'] if graph else live['reverse' if order['order']=='reverse' else 'source']
        first=live['graph_first'] if graph else live['result_'+order['order']+'_first']
        for m in order['observations']:
            expected=source if m['site']=='first' else first
            observed=m['source_slice']
            require(observed['storage_base']==expected['storage_base'] and observed['storage_bytes']==expected['storage_bytes'],'Site source not bound to current owner')
    result.update(retained_live_spans_verified=True,source_replay_metadata_bound=True,
                  seed_generation_independently_verified=False,execution_certified=False)
    return result


def audit_joined_inventory(records, *, width=152265, device='cuda:0'):
    summary=audit_inventory(records,width=width,device=device)
    joined=[audit_joined_case(r,width=width,device=device) for r in records]
    for seed in (17,29):
        group=[r for r in records if r['seed']==seed]
        require(all(r['input_generation']==group[0]['input_generation'] for r in group),'Same-seed input differs across shifts/modes')
    summary.update(retained_live_spans_verified=True,source_replay_metadata_bound=True)
    return summary


def audit_verified_case(record, *, width=152265, device='cuda:0'):
    """Bind independent live numerical receipts; still not allocation proof."""
    result=audit_joined_case(record,width=width,device=device)
    evidence=record['live_numerical_audit'];live=record['live_tensors']
    require(set(evidence)=={'schema','comparisons','finite_counts','owners','owner_sha256','all_finite',
                            'all_comparisons_exact','output_recomputed','execution_certified'},'Live audit fields differ')
    require(evidence['schema']=='request-block-live-numerical-audit-v1','Live audit schema differs')
    require(evidence['comparisons']==record['comparisons'],'Independent live comparisons differ')
    require(evidence['owners']==live,'Independent audited owners differ')
    require(set(evidence['finite_counts'])==set(evidence['owner_sha256'])==set(live),'Finite/hash coverage differs')
    for name,m in live.items():
        count=evidence['finite_counts'][name]
        require(set(count)=={'elements','nonfinite_elements'},'Finite count fields differ')
        require(type(count['elements']) is int and count['elements']==m['shape'][0]*width,'Finite tensor extent differs')
        require(type(count['nonfinite_elements']) is int and 0<=count['nonfinite_elements']<=count['elements'],'Invalid nonfinite count')
        digest(evidence['owner_sha256'][name])
    finite=all(c['nonfinite_elements']==0 for c in evidence['finite_counts'].values())
    exact=all(c['exact'] for c in evidence['comparisons'].values())
    for k,v in [('all_finite',finite),('all_comparisons_exact',exact),('output_recomputed',False),('execution_certified',False)]:
        require(type(evidence[k]) is bool and evidence[k]==v,'Live audit flag differs: '+k)
    require(record['finite_outputs']==finite,'Producer finite flag differs from independent counts')
    hashes=evidence['owner_sha256'];comp=evidence['comparisons']
    require([hashes['serial_'+str(i)] for i in range(2)]==record['input_identity']['serial_blocks'],'Independent serial input hashes differ')
    if record['mode']=='cuda_graph':
        require(hashes['graph_input']==record['input_identity']['source_sha256'],'Final graph staging audit hash differs')
        require(all(hashes['graph_'+site]==hashes['result_restored_'+site] for site in SITES),'Final static/retained graph hashes differ')
    for site in SITES:
        require(comp['control_repeat_'+site]['left_sha256']==hashes['reference_'+site] and
                comp['control_repeat_'+site]['right_sha256']==hashes['control_repeat_'+site],'Control audit hash binding differs')
        for order in ORDERS:
            require(comp[order+'_retained_'+site]['left_sha256']==hashes['snapshot_'+order+'_'+site] and
                    comp[order+'_retained_'+site]['right_sha256']==hashes['result_'+order+'_'+site],'Retained audit hash binding differs')
            if order!='reverse':require(comp[order+'_'+site]['right_sha256']==hashes['result_'+order+'_'+site],'Result audit hash binding differs')
    for label in ('source','reverse_source'):
        owner='source' if label=='source' else 'reverse'
        require(comp[label+'_untouched']['left_sha256']==hashes[owner+'_saved'] and
                comp[label+'_untouched']['right_sha256']==hashes[owner],'Input audit hash binding differs')
    result.update(live_numerical_receipts_bound=True,all_live_tensors_finite=finite)
    return result


def audit_verified_inventory(records, *, width=152265, device='cuda:0'):
    summary=audit_joined_inventory(records,width=width,device=device)
    checked=[audit_verified_case(r,width=width,device=device) for r in records]
    summary.update(live_numerical_receipts_bound=True,
                   records_passed=all(r['numerical_records_passed'] for r in checked))
    return summary
