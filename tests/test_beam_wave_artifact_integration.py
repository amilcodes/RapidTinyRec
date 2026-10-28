"""Tiny synthetic four-phase artifact integration; no model, GPU or bootstrap.

The protocol user digest and allocation environment are fixture identities, not
claims about a live job. All raw shape/storage/log/dispatch checks run normally.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import test_audit_beam_wave as header_fixture
from test_audit_beam_wave_outputs import trial,request
from test_beam_wave_dispatch import report,counters
from test_beam_wave_storage import inventory
from rapidtinyrec.audit_beam_wave import inspect_phases,digest,json_digest
from rapidtinyrec.beam_wave_protocol import PHASES,phase_schedule


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))


class ArtifactIntegrationTests(unittest.TestCase):
    def prepare(self,root):
        f=header_fixture.RawAuditBoundaryTests();f.setUp()
        code=root/'code';out=root/'outputs';historical=root/'historical';release=root/'release'
        config=dict(num_hidden_layers=1,num_key_value_heads=2,hidden_size=8,num_attention_heads=2,vocab_size=7)
        save(release/'Office_ckpt/config.json',config)
        users=[dict(cohort_index=i,prompt_tokens=76 if i<6 else 121) for i in range(500)]
        ref=dict(experiment='gqa_three_way',dtype='float32',users=users);ref['manifest_sha256']=json_digest(ref)
        save(historical/'manifest.json',ref);sha=json_digest(users)
        protocol=deepcopy(f.protocol);protocol['release_users_sha256']=sha
        expected=dict(users_sha256=sha,release_sha256={'Office_ckpt/config.json':digest(release/'Office_ckpt/config.json')})
        save(code/'rapidtinyrec/expected.json',expected);save(code/'rapidtinyrec/quality_resources.json',{})
        save(out/'wave-work-start.json',f.work)
        selected=[users[i] for i in (1,2,6,10)]
        for phase,variant in enumerate(PHASES):
            p=out/f'phase_{phase}';p.mkdir();wave=phase in (1,2)
            m=deepcopy(f.header);m.update(phase=phase,half=phase//2,variant=variant,phase_protocol=protocol['phases'][phase],
                users=selected,cohort_sha256=sha,reference_manifest_sha256=ref['manifest_sha256'],work_deadline=f.work,
                kv_layout=dict(layers=1,kv_heads=2,head_dim=4),vocab_size=7,source_sha256={})
            m['release_verification']['observed_sha256']=expected['release_sha256']
            m['quality_release_verification']=dict(m['release_verification'],observed_sha256={})
            before={k:dict(path='/synthetic/'+variant+'/'+k,files=0 if phase<2 else 1) for k in ('TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR')}
            m['compiler_artifacts']=dict(reused=phase>=2,before=before)
            w,s,d=counters(wave);d.update(report(p,wave));d['generated_code']=[dict(file='wrapper.py',sha256='')]
            (p/'wrapper.py').write_text('# synthetic generated wrapper');d['generated_code'][0]['sha256']=digest(p/'wrapper.py')
            for item in (w,s,d):item['precision_policy']=m['precision_policy']
            for item in (w,s):item['memory']=dict(allocated_bytes=100,reserved_bytes=200)
            w['compiler_artifacts_after']={k:dict(v,files=1) for k,v in before.items()}
            save(p/'manifest.json',m);save(p/'setup.json',w);save(p/'summary.json',s)
            save(p/'compilation.json',dict(phase='complete',measurement_compilations=0,events=[dict(success=True)],compiler_counters={}))
            logs=[]
            for name in ('capture-cublas.log','capture-cublaslt.log'):
                (p/name).write_text('fixture');logs.append(dict(path=name,bytes=7,sha256=digest(p/name)))
            save(p/'log-staging.json',dict(schema='precision-staged-capture-v1',setup_succeeded=True,logs_disabled=True,published=True,
                measurement_allowed=True,setup_error=None,copy_s=.01,session_wall_s=.1,
                publication=dict(complete=True,filesystem=dict(filesystem='tmpfs'),files=logs)))
            # Bind native dispatch parsing to the same hash-verified capture log.
            (p/'capture-cublas.log').write_bytes((p/'api.log').read_bytes())
            staging=json.loads((p/'log-staging.json').read_text());staging['publication']['files'][0].update(bytes=(p/'capture-cublas.log').stat().st_size,sha256=digest(p/'capture-cublas.log'))
            save(p/'log-staging.json',staging);d['capture_logs'].update(cublas='capture-cublas.log',cublaslt='capture-cublaslt.log')
            trials=[trial(spec) for spec in phase_schedule(phase)];save(p/'trials.json',trials)
            for t in trials:save(p/'trials'/f"{t['timing']['repeat']}_{t['timing']['pair_index']}.json",t)
            for stage in ('before-measurement','after-measurement','after-profiles'):
                save(p/f'storage-{stage}.json',dict(inventory(wave),stage=stage))
            def rows(group):
                return [request(u,i if wave else 0,2 if wave else 1,[100+k+(0 if wave else i*10) for k in range(5)]) for i,u in enumerate(group)]
            groups=[[1,2],[6,10]] if wave else [[1],[2],[6],[10]]
            for group in groups:
                for mode in ('direct','replay','reverse','retained'):
                    order=group[::-1] if mode=='reverse' and wave else group
                    save(p/'warmup_outputs'/('_'.join(map(str,group))+'_'+mode+'.json'),dict(users=order,requests=rows(order)))
            for profile in d['profiles']:
                profile['output']=f"profile-output-{profile['prompt_tokens']}.json"
                save(p/profile['output'],dict(users=profile['cohort_indices'],requests=rows(profile['cohort_indices'])))
            save(p/'dispatch.json',d)
        return (out,code,historical,release,f.live),protocol
    def inspect(self,args,protocol):
        with patch.dict(os.environ,{'SLURM_JOB_ID':args[-1]['job_id']}),patch('rapidtinyrec.audit_beam_wave.load_protocol',return_value=protocol):
            return inspect_phases(*args)
    def test_complete_four_phase_artifact_path_without_quality_bootstrap(self):
        with tempfile.TemporaryDirectory() as temp:
            args,p=self.prepare(Path(temp));e=self.inspect(args,p)
            self.assertEqual(len(e['joined']),24);self.assertEqual(len(e['phases']),4)
            self.assertTrue(all(x['stability_exact'] and x['dispatch']['verified'] for x in e['phases']))
            self.assertEqual({r['cohort_index'] for r in e['controls']},{1,2,6,10})
    def test_dispatch_cannot_use_an_unverified_copy_of_capture_log(self):
        with tempfile.TemporaryDirectory() as temp:
            args,p=self.prepare(Path(temp));file=args[0]/'phase_1/dispatch.json';d=json.loads(file.read_text());d['capture_logs']['cublas']='api.log';save(file,d)
            with self.assertRaisesRegex(ValueError,'hash-verified'):self.inspect(args,p)
    def test_phase_cannot_extend_original_deadline(self):
        with tempfile.TemporaryDirectory() as temp:
            args,p=self.prepare(Path(temp));file=args[0]/'phase_2/manifest.json';m=json.loads(file.read_text());m['work_deadline']['deadline_monotonic']+=1;save(file,m)
            with self.assertRaisesRegex(ValueError,'original work deadline'):self.inspect(args,p)
    def test_missing_trial_and_retained_drift_are_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            args,p=self.prepare(Path(temp));next((args[0]/'phase_1/trials').glob('*.json')).unlink()
            with self.assertRaisesRegex(ValueError,'trial inventory'):self.inspect(args,p)
        with tempfile.TemporaryDirectory() as temp:
            args,p=self.prepare(Path(temp));file=args[0]/'phase_1/warmup_outputs/1_2_retained.json';raw=json.loads(file.read_text());raw['requests'][0]['token_ids'][0][0]=99;save(file,raw)
            e=self.inspect(args,p);self.assertFalse(e['phases'][1]['stability_exact'])

if __name__=='__main__':unittest.main()
