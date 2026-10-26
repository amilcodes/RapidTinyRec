import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest.mock import patch

from rapidtinyrec.precision_provenance import (
    PROTOCOL, PROTOCOL_SHA256, load_protocol, phase_record, verify_release, validate_phase,
)
from rapidtinyrec.audit_precision_paired_staged import validate_provenance
from rapidtinyrec.audit_precision_setup import validate_wrapper
from rapidtinyrec.precision_log_staging import LogStage, LOG_NAMES, save_record
from rapidtinyrec.precision_log_runner import recover_logs

ROOT = Path(__file__).resolve().parents[1]


class PairedProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol(ROOT / PROTOCOL)
        self.expected = self.protocol['provenance']['release_hashes']
        self.users = [dict(cohort_index=i,prompt_tokens=n) for i,n in ((1,76),(2,76),(6,121),(10,121))]

    def manifest(self, phase):
        return dict(experiment='precision_paired_staged_pilot',phase=phase,
                    variant=self.protocol['phases'][phase]['variant'],half=phase//2,
                    compiler_artifacts={'reused':phase>=2},users=self.users,api_log_staging=True,
                    phase_protocol=phase_record(self.protocol,phase,phase>=2,self.users),
                    release_verification=dict(schema='precision-release-verification-v1',
                        observed_sha256=self.expected,elapsed_s=1.,boundary='before_tokenizer_and_model_load'))

    def test_actual_digest_reads_are_recorded_once_and_mismatch_fails(self):
        calls = []
        def digest(path):
            calls.append(path)
            return hashlib.sha256(path.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'checkpoint').write_bytes(b'actual checkpoint')
            want=hashlib.sha256(b'actual checkpoint').hexdigest()
            result=verify_release(root,{'checkpoint':want},digest)
            self.assertEqual(calls,[root/'checkpoint'])
            self.assertEqual(result['observed_sha256'],{'checkpoint':want})
            (root/'checkpoint').write_bytes(b'changed')
            with self.assertRaisesRegex(AssertionError,'checkpoint'):
                verify_release(root,{'checkpoint':want},digest)

    def test_each_frozen_phase_and_release_pass(self):
        for phase in range(4): validate_phase(self.manifest(phase),self.protocol,self.expected)

    def test_wrong_scope_protocol_cohort_cache_and_release_fail(self):
        m=self.manifest(2)
        mutations=[lambda x:x.update(experiment='precision_500'),
                   lambda x:x.update(variant='folded_fp32'),lambda x:x.update(half=0),
                   lambda x:x.update(api_log_staging=False),
                   lambda x:x['phase_protocol'].update(protocol_sha256='0'*64),
                   lambda x:x['phase_protocol']['phase'].update(fresh_graphs=False),
                   lambda x:x['compiler_artifacts'].update(reused=False),
                   lambda x:x['users'].pop(),
                   lambda x:x['release_verification']['observed_sha256'].pop('Office_ckpt/model.safetensors'),
                   lambda x:x['release_verification'].update(elapsed_s=float('nan')),
                   lambda x:x['release_verification'].update(boundary='after_model_load')]
        for change in mutations:
            bad=deepcopy(m);change(bad)
            with self.assertRaises(AssertionError):validate_phase(bad,self.protocol,self.expected)

    def test_changed_protocol_bytes_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'protocol.json';path.write_bytes((ROOT/PROTOCOL).read_bytes()+b' ')
            with self.assertRaises(AssertionError):load_protocol(path)

    def test_wrong_profile_is_rejected_before_reading_evidence(self):
        for profile in ('precision-pilot','precision-setup-feasibility'):
            with self.assertRaises(AssertionError):validate_provenance(None,None,{'profile':profile})

    def test_cli_scope_guard_precedes_cuda_validation(self):
        tree=ast.parse((ROOT/'rapidtinyrec/benchmark_precision.py').read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        # Execute the real argparse/guard prefix without importing GPU libraries.
        prefix=[]
        for node in main.body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='protocol' for t in node.targets):break
            prefix.append(node)
        import argparse
        code=compile(ast.fix_missing_locations(ast.Module(body=prefix,type_ignores=[])),'scope-guard','exec')
        common=['benchmark','--release-root','r','--reference-results','r','--output-root','o','--phase','0','--paired-staged-pilot']
        import contextlib,io
        for extra in ([],['--api-log-local-root','l','--full-cohort'],['--api-log-local-root','l','--setup-feasibility']):
            with patch('sys.argv',common+extra),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:exec(code,{'argparse':argparse,'Path':Path,'__doc__':'test'})
                self.assertEqual(caught.exception.code,2)


class FourPhaseRecoveryTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        base=Path(temp.name).resolve();self.root=base/'out';self.root.mkdir()
        local=base/'local';local.mkdir();fs={'mount':'/','filesystem':'tmpfs'}
        for phase in range(4):
            output=self.root/f'phase_{phase}';output.mkdir();stage=LogStage(local,output,fs)
            for name in LOG_NAMES:(stage.root/name).write_bytes(f'{phase}:{name}'.encode())
            save_record(output/'log-staging.json',{'publication':stage.publish(loggers_disabled=True,require_complete=True)})
            save_record(output/'manifest.json',{'api_log_staging':True})
            save_record(output/'log-staging-location.json',dict(schema='precision-log-location-v1',
                local_directory=str(stage.root),output_directory=str(output),filesystem=fs))
        save_record(self.root/'log-wrapper-start.json',dict(schema='precision-log-wrapper-v1',job_id='test',
            child_timeout_s=1620,filesystem=fs,local_directory=str(local)))
        self.finish=dict(child=dict(schema='precision-child-exit-v1',returncode=0,reason='exited',
            completed_normally=True,signal=None,elapsed_s=1500),recovery=recover_logs(self.root,local,fs))
        self.write()

    def write(self):save_record(self.root/'log-wrapper-exit.json',self.finish)

    def validate(self):return validate_wrapper(self.root,{'job_id':'test'},phases=(0,1,2,3))

    def test_four_actual_publications_and_recovery_pass(self):self.assertEqual(self.validate(),self.finish)

    def test_omitted_or_duplicate_phase_and_corrupt_recovery_fail(self):
        original=deepcopy(self.finish)
        for change in ('omitted','duplicate','incomplete','timeout'):
            self.finish=deepcopy(original)
            if change=='omitted':self.finish['recovery']['locations'].pop()
            if change=='duplicate':self.finish['recovery']['locations'][3]=deepcopy(self.finish['recovery']['locations'][0])
            if change=='incomplete':self.finish['recovery']['recovery_complete']=False
            if change=='timeout':self.finish['child']['reason']='deadline'
            self.write()
            with self.assertRaises(AssertionError):self.validate()

    def test_setup_auditor_does_not_accept_four_phase_scope(self):
        with self.assertRaises(AssertionError):validate_wrapper(self.root,{'job_id':'test'})


class ProvenanceBindingTests(FourPhaseRecoveryTests):
    def setUp(self):
        super().setUp()
        self.snapshot=self.root.parent/'snapshot'/'rapidtinyrec';self.snapshot.mkdir(parents=True)
        shutil.copyfile(ROOT/'rapidtinyrec/expected.json',self.snapshot/'expected.json')
        protocol_path=self.snapshot.parent/PROTOCOL;protocol_path.parent.mkdir(parents=True)
        shutil.copyfile(ROOT/PROTOCOL,protocol_path)
        script=self.snapshot.parent/'jobs/precision-paired-staged.sh';script.parent.mkdir()
        shutil.copyfile(ROOT/'benchmarks/preparation/quality-frontier/precision-paired-staged.sh',script)
        protocol=load_protocol(protocol_path)
        helper=PairedProvenanceTests();helper.setUp()
        for phase in range(4):save_record(self.root/f'phase_{phase}'/'manifest.json',helper.manifest(phase))
        self.record={'profile':'precision-paired-staged-pilot','job_id':'test',
                     'resources':{**protocol['resources'],'command':['bash','jobs/precision-paired-staged.sh']},
                     'sha256':{'rapidtinyrec/'+p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (ROOT/'rapidtinyrec').glob('*.py')}}
        self.record['sha256'][PROTOCOL]=PROTOCOL_SHA256
        self.record['sha256']['jobs/precision-paired-staged.sh']=hashlib.sha256(script.read_bytes()).hexdigest()

    def test_all_four_phase_release_and_source_binding_pass(self):
        result=validate_provenance(self.root,self.snapshot,self.record)
        self.assertEqual(result['protocol_sha256'],PROTOCOL_SHA256)
        self.assertIn('Office_ckpt/model.safetensors',result['release_sha256'])

    def test_wrong_resources_command_protocol_and_auditor_source_fail(self):
        changes=[lambda x:x['resources'].update(gpus=2),
                 lambda x:x['resources'].update(time='01:00:00'),
                 lambda x:x['resources'].update(command=['bash','jobs/precision-setup.sh']),
                 lambda x:x['sha256'].update({PROTOCOL:'0'*64}),
                 lambda x:x['sha256'].update({'rapidtinyrec/audit_precision.py':'0'*64})]
        for change in changes:
            record=deepcopy(self.record);change(record)
            with self.assertRaises(AssertionError):validate_provenance(self.root,self.snapshot,record)

    def test_missing_or_extra_phase_fail(self):
        extra=self.root/'phase_4';extra.mkdir()
        with self.assertRaises(AssertionError):validate_provenance(self.root,self.snapshot,self.record)
        extra.rmdir();(self.root/'phase_3').rename(self.root/'absent_phase')
        with self.assertRaises(AssertionError):validate_provenance(self.root,self.snapshot,self.record)


class NumericalBoundaryTests(unittest.TestCase):
    def test_generation_extraction_memory_selection_and_timed_loop_ast_unchanged(self):
        tree=ast.parse((ROOT/'rapidtinyrec/benchmark_precision.py').read_text())
        nodes={n.name:n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
        selected=[nodes[n] for n in ('select_users','generate','unpack','memory')]
        selected += [n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.target)=='(repeat, user)']
        self.assertEqual(len(selected),5)
        fixture=ROOT/'tests/fixtures/precision-original-boundaries.py.txt'
        self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(),'d1a05c050029781ed991512ce33e3a57783ae97ce9969786b2cdb43cb1024a45')
        # Parse both under this interpreter: FunctionDef/type_params and ast.dump
        # defaults differ between local Python3.14 and CI Python3.11.
        reference=ast.parse(fixture.read_text()).body
        self.assertEqual([ast.dump(n) for n in selected],[ast.dump(n) for n in reference])
