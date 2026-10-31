"""Complete synthetic raw artifacts; no release model or local bootstrap."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from rapidtinyrec import audit_bf16_full500 as audit
from rapidtinyrec.bf16_full500 import PROTOCOL, PROTOCOL_SHA256, load_protocol, phase_record, PHASES
from rapidtinyrec.audit_decode_pilot import digest,json_digest
from test_precision_full500 import run_record
ROOT=Path(__file__).resolve().parents[1]


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


class Full500RawAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.base=Path(cls.tmp.name)
        cls.raw=cls.base/'outputs';cls.history=cls.base/'historical'
        cls.snapshot=cls.base/'code'/'rapidtinyrec';cls.record=cls.base/'record.json'
        cls.protocol=load_protocol(ROOT/PROTOCOL)
        # Both halves have each length. These are invented users, never held-out data.
        cls.users=[dict(cohort_index=i,prompt_tokens=76+5*((i//2)%10)) for i in range(500)]
        cls.protocol['cohort']['users_sha256']=json_digest(cls.users)
        reference=dict(experiment='gqa_three_way',dtype='float32',users=cls.users)
        reference['manifest_sha256']=json_digest(reference)
        cls.protocol['cohort']['manifest_content_sha256']=reference['manifest_sha256']
        cls.expected=json.loads((ROOT/'rapidtinyrec/expected.json').read_text())
        cls.expected['users_sha256']=json_digest(cls.users)
        save(cls.history/'manifest.json',reference)
        cls.protocol['cohort']['manifest_file_sha256']=digest(cls.history/'manifest.json')
        raw=run_record();raw['graph_logit_storage_ptrs']=list(range(100,140))
        for u in cls.users:
            save(cls.history/'users'/f"{u['cohort_index']:04d}.json",dict(cohort_index=u['cohort_index'],manifest_sha256=reference['manifest_sha256'],runs={'compiled_gqa':raw}))
        names=['benchmark_bf16_full500.py','bf16_full500.py','precision_full500.py','bf16_policy.py','precision_provenance.py','precision_dispatch.py','precision_log_staging.py','precision_quality_adapter.py','quality_frontier.py','quality_resources.json','folded_attention.py','shared_prefix.py','prefix_replay.py','prefix_prefill.py','input_arena.py','decode_replay.py','compiled_forward.py','shared_prefill.py','gpu_constraints.py','expected.json','direct_reorder.py']
        cls.snapshot.mkdir(parents=True)
        for name in names:(cls.snapshot/name).write_bytes((ROOT/'rapidtinyrec'/name).read_bytes())
        save(cls.snapshot/'expected.json',cls.expected)
        (cls.snapshot.parent/PROTOCOL).parent.mkdir(parents=True)
        (cls.snapshot.parent/PROTOCOL).write_bytes((ROOT/PROTOCOL).read_bytes())
        hashes={str(p.relative_to(cls.snapshot.parent)):digest(p) for p in cls.snapshot.parent.rglob('*') if p.is_file()}
        save(cls.record,dict(job_id='123',run_id='synthetic',resources=cls.protocol['resources'],files=list(hashes),sha256=hashes))
        policy=dict(matmul_tf32=True,cudnn_tf32=False,float32_matmul_precision='high',blas_preference='_BlasBackend.Cublas',bf16_reduced_precision_reduction=False,bf16_split_k=True,math_sdpa_reduced_precision_reduction=False)
        lengths=cls.protocol['cohort']['lengths'];warm=[next(u['cohort_index'] for u in cls.users if u['prompt_tokens']==n) for n in lengths]
        memory=dict(allocated_bytes=10**9,reserved_bytes=2*10**9,peak_allocated_bytes=15*10**8,peak_reserved_bytes=2*10**9)
        for phase,variant in enumerate(PHASES):
            path=cls.raw/f'phase_{phase}';path.mkdir(parents=True)
            spec=cls.protocol['phases'][phase];bpt=28*2*50*2*128*(2 if variant=='folded_bf16' else 4)
            events=[dict(cache_tokens=n+j,batch=50,input_buffer_bytes=bpt//50*n+bpt*j+808,capture_and_warmup_s=1.) for n in lengths for j in range(4)]
            replay=dict(graphs=40,frozen=True,events=events,input_buffer_bytes=sum(e['input_buffer_bytes'] for e in events),unique_input_storage_bytes=bpt//50*max(lengths)+bpt*3+40*808,shared_input_arena=True,replays=0,capture_and_warmup_s=40.,output_suffix_bytes=bpt*10*10,physical_prefix=dict(physical_prefix_batch=1,suffix_batch=50,max_prefix_tokens=max(lengths),max_suffix_tokens=3,prefix_capacity_bytes=bpt//50*max(lengths),suffix_capacity_bytes=bpt*3,requests_loaded=0,prefix_bytes_copied=0))
            after=deepcopy(replay);after['replays']=1000;after['physical_prefix']['requests_loaded']=250
            after['physical_prefix']['prefix_bytes_copied']=sum(bpt//50*cls.users[i]['prompt_tokens'] for i in spec['primary_cohort_indices'])
            reorder=dict(direct_calls=0,fallback_calls=0,prefix_bytes_reordered=0,suffix_bytes_written=0)
            last=dict(direct_calls=1000,fallback_calls=250,prefix_bytes_reordered=0,suffix_bytes_written=bpt*6*250)
            caches={k:dict(path=f'/synthetic/{variant}/{k}',files=int(phase>=2)) for k in ('TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR')}
            verified=lambda hashes:dict(schema='precision-release-verification-v1',observed_sha256=hashes,elapsed_s=1.,boundary='before_tokenizer_and_model_load')
            manifest=dict(experiment='bf16_vs_tf32_full500',phase=phase,half=phase//2,variant=variant,attention_layout='folded_query_heads',compiler_artifacts=dict(reused=phase>=2,before=caches),hardware='NVIDIA RTX A6000',dtype='bfloat16' if phase in (1,2) else 'float32',tf32=True,compiler_policy=dict(max_autotune=False,max_autotune_gemm=False,max_autotune_pointwise=False),torch_git=audit.TORCH_COMMIT,cuda_version='12.6',job_id='123',model_load_s=1.,model_parameter_unique_bytes=100,model_parameter_logical_bytes=100,precision_policy=policy,versions={'torch':'2.14.0+cu126','transformers':'4.55.4','tokenizers':'0.21.4'},cohort_sha256=cls.expected['users_sha256'],reference_manifest_sha256=reference['manifest_sha256'],sha256={n:digest(cls.snapshot/n) for n in names},users=cls.users,phase_protocol=phase_record(cls.protocol,phase,phase>=2,cls.users),api_log_staging=True,release_verification=verified(cls.protocol['provenance']['release_hashes']),quality_resource_verification=verified(cls.protocol['provenance']['resource_hashes']))
            save(path/'manifest.json',manifest)
            setup=dict(precision_policy=policy,compiler_artifacts_after={k:dict(v,files=1) for k,v in caches.items()},warmup_cohort_indices=warm,replay=replay,combined_setup_and_warmup_s=50.,direct_reorder=reorder,memory=memory)
            save(path/'setup.json',setup)
            elapsed=.08 if variant=='folded_tf32' else .05
            save(path/'summary.json',dict(complete=True,precision_policy=policy,measured_requests=250,replay=after,direct_reorder=last,memory=memory,latency=dict(n=250,median_s=elapsed,p95_s=elapsed)))
            save(path/'compilation.json',dict(phase='complete',measurement_compilations=0,events=[{'success':True}],compiler_counters={},backend_compile_total_s=2.))
            rows=[dict(cohort_index=i,prompt_tokens=cls.users[i]['prompt_tokens'],repeat=0,variant=variant,measurement_role='primary_timed',run=dict(raw,elapsed_s=elapsed)) for i in spec['primary_cohort_indices']]
            save(path/'requests.json',rows)
            for r in rows:save(path/'users'/f"0_{r['cohort_index']:04d}.json",r)
            repeats=[dict(r,repeat=1,measurement_role='untimed_stability',run=raw) for r in reversed(rows)]
            save(path/'untimed-repeat-records.json',repeats)
            for r in repeats:save(path/'untimed-repeats'/f"{r['cohort_index']:04d}.json",r)
            save(path/'untimed-repeat-summary.json',dict(schema='bf16-full500-repeat-summary-v1',complete=True,requests=250,cohort_indices=list(reversed(spec['primary_cohort_indices'])),additional_quality_users=0,included_in_primary_latency=False))
            for i in warm:
                for mode in ('dense_forward',variant,'repeat','retained'):save(path/'warmup_outputs'/f'{i}_{mode}.json',raw)
            profiles=[]
            for n in (min(lengths),max(lengths)):
                i=next(u['cohort_index'] for u in cls.users if u['prompt_tokens']==n);name=f'profile-{n}.json';save(path/name,raw)
                profiles.append(dict(prompt_tokens=n,cohort_index=i,output=name))
            (path/'generated.py').write_text('# synthetic code bytes\n')
            post=deepcopy(after);post['replays']+=1008
            save(path/'dispatch.json',dict(precision_policy=policy,replay_after_profiles=post,profiles=profiles,generated_code=[dict(file='generated.py',sha256=digest(path/'generated.py'))]))
            logs=[]
            for name in ('capture-cublas.log','capture-cublaslt.log'):
                (path/name).write_text('synthetic capture\n');logs.append(dict(path=name,bytes=(path/name).stat().st_size,sha256=digest(path/name)))
            save(path/'log-staging.json',dict(schema='precision-staged-capture-v1',setup_succeeded=True,logs_disabled=True,published=True,measurement_allowed=True,setup_error=None,publication=dict(complete=True,filesystem={'filesystem':'tmpfs'},files=logs),copy_s=.1,session_wall_s=1.))
        cls.real_read=staticmethod(audit.read)
        # Only the synthetic cohort protocol and package pin are injected. All
        # raw files, raw validation, phase/source hashes and log checks are real.
        cls.loader=patch.object(audit,'load_protocol',return_value=cls.protocol);cls.loader.start()
        cls.reader=patch.object(audit,'read',side_effect=lambda p:cls.expected if Path(p)==ROOT/'rapidtinyrec/expected.json' else cls.real_read(p));cls.reader.start()
        # Tiny traces cannot prove actual GPU dispatch: report false deliberately.
        cls.dispatch=patch.object(audit,'dispatch_evidence',return_value={'verified':False,'fixture':True});cls.dispatch.start()

    @classmethod
    def tearDownClass(cls):
        cls.dispatch.stop();cls.reader.stop();cls.loader.stop();cls.tmp.cleanup()

    def inspect(self):return audit.inspect_phases(self.raw,self.history,self.snapshot,self.record)

    def test_complete_all500_raw_artifact_contract(self):
        e=self.inspect()
        self.assertEqual(len(e['joined']),500);self.assertEqual(len(e['phases']),4)
        self.assertTrue(all(p['untimed_repeat_check']['passed'] for p in e['phases']))
        self.assertFalse(any(p['dispatch']['verified'] for p in e['phases']))
        self.assertTrue({'PROTOCOL','load_protocol'}.isdisjoint(audit.inspect_phases.__code__.co_varnames))

    def test_corrupt_phase_source_storage_or_repeat_cannot_pass(self):
        mutations=[('manifest.json',lambda x:x.update(experiment='bf16_vs_tf32_pilot')),
                   ('manifest.json',lambda x:x['sha256'].update(**{'bf16_full500.py':'bad'})),
                   ('manifest.json',lambda x:x.pop('quality_resource_verification')),
                   ('summary.json',lambda x:x['replay'].update(graphs=8)),
                   ('requests.json',lambda x:x[0]['run']['saved_score_storage_ptrs'].__setitem__(0,100)),
                   ('untimed-repeat-summary.json',lambda x:x.update(included_in_primary_latency=True)),
                   ('untimed-repeat-records.json',lambda x:x.reverse())]
        for name,change in mutations:
            p=self.raw/'phase_0'/name;before=p.read_bytes();value=json.loads(before);change(value)
            try:
                save(p,value)
                with self.subTest(name=name),self.assertRaises((AssertionError,KeyError)):self.inspect()
            finally:p.write_bytes(before)

    def test_missing_repeat_file_and_signed_zero_drift(self):
        path=self.raw/'phase_0';p=path/'untimed-repeats/0498.json';before=p.read_bytes();p.unlink()
        try:
            with self.assertRaises(AssertionError):audit.inspect_repeats(path,self.real_read(path/'requests.json'),self.protocol['phases'][0])
        finally:p.write_bytes(before)
        aggregate=path/'untimed-repeat-records.json';saved=aggregate.read_bytes();rows=json.loads(saved)
        rows[0]['run']['transition_logp'][0][0]=-0.0
        try:
            save(aggregate,rows);save(p,rows[0])
            result=audit.inspect_repeats(path,self.real_read(path/'requests.json'),self.protocol['phases'][0])
            self.assertFalse(result['passed']);self.assertEqual(result['comparison']['matching_transition_values'],250*250-1)
        finally:aggregate.write_bytes(saved);p.write_bytes(before)

    def test_full_evaluation_pairs_once_and_preserves_negative_dispatch(self):
        evidence=self.inspect();protocol=deepcopy(self.protocol)
        baseline=self.base/'baseline';hashes=[]
        for phase,half in ((1,0),(2,1)):
            rows=[dict(cohort_index=i,variant='folded_tf32',run=evidence['joined'][(0,i)]['runs']['folded_tf32']) for i in range(half,500,2)]
            name=f'outputs/phase_{phase}/requests.json';save(baseline/name,rows)
            hashes.append(dict(path=name,sha256=digest(baseline/name)))
        save(baseline/'audit.json',dict(recovery_audit_passed=True,original_run_id='old',original_job_id='122'))
        save(self.snapshot.parent/'baseline-raw-manifest.json',dict(files=hashes))
        protocol['baseline']=dict(remote_dir=str(baseline),recovered_audit_remote_path=str(baseline/'audit.json'),recovered_audit_sha256=digest(baseline/'audit.json'),run_id='old',job_id='122',raw_manifest_sha256=digest(self.snapshot.parent/'baseline-raw-manifest.json'))
        def inputs(release,users,rows):
            self.assertEqual(users,self.users);self.assertEqual(len(rows),500)
            self.assertEqual([r['cohort_index'] for r in rows],list(range(500)))
            self.assertTrue(all(r['repeat']==0 and set(r['runs'])=={'folded_fp32','folded_tf32'} for r in rows))
            return dict(users=users,observations=rows,catalog_sids=[],eos_token_id=9)
        # Replacing only the expensive bootstrap: never execute 500-user
        # resampling on this host or infer actual recommendation quality.
        quality=dict(token_repeat_stability=True,output_validity={'control':{'invalid':0},'candidate':{'invalid':0}},fixture=True)
        with patch.dict(os.environ,{'SLURM_JOB_ID':'123'}),patch.object(audit,'load_protocol',return_value=protocol),patch.object(audit,'quality_inputs',side_effect=inputs),patch.object(audit,'evaluate',return_value=quality) as compute:
            result=audit.evaluate_phases(evidence,self.raw,self.snapshot,self.base,wrapper={},allocation_identity=dict(scheduler_fields=dict(JobId='123',JobState='RUNNING')))
        self.assertEqual(compute.call_args.kwargs['bootstrap_samples'],10000)
        self.assertEqual(compute.call_args.kwargs['bootstrap_seed'],20261026)
        self.assertEqual(result['paired_requests'],500);self.assertTrue(result['control_exact'])
        self.assertTrue(result['stability_exact']);self.assertTrue(result['performance_gate']['passed'])
        self.assertFalse(result['dispatch_verified']);self.assertFalse(result['execution_audit_passed'])
        self.assertFalse(result['promotion']);self.assertNotIn('eligible_for_full500',result)
        self.assertEqual(len(result['latency_by_length_half_ms']),20)
        self.assertEqual(result['latency_by_half_ms']['1']['folded_tf32']['n'],250)

    def test_allocation_guard_precedes_any_evaluation_io(self):
        identity=dict(scheduler_fields=dict(JobId='123',JobState='RUNNING'))
        for env in ({},{'SLURM_JOB_ID':'999'}):
            with patch.dict(os.environ,env,clear=True),self.assertRaises(AssertionError):
                audit.evaluate_phases(None,None,None,None,wrapper=None,allocation_identity=identity)

    def test_catalog_preflight_missing_resource_does_not_bootstrap(self):
        with patch.object(audit,'evaluate',side_effect=AssertionError('must not bootstrap')) as compute:
            with self.assertRaises(FileNotFoundError):audit.catalog_preflight(self.base/'missing-release',self.users,self.protocol)
            compute.assert_not_called()

if __name__=='__main__':unittest.main()
