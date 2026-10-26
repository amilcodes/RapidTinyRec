import json
import tempfile
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
from rapidtinyrec.audit_precision import audit
from rapidtinyrec.audit_precision_setup import diagnostic_decision, validate_wrapper
from rapidtinyrec.precision_log_staging import LogStage, LOG_NAMES, save_record
from rapidtinyrec.precision_log_runner import recover_logs


class SetupScopeTests(unittest.TestCase):
    def test_success_never_grants_full_quality_or_promotion(self):
        result=diagnostic_decision(True,True,True,True)
        self.assertTrue(result['setup_diagnostic_passed'])
        for key in ('eligible_for_full_quality_review','promotion','noninferiority_established','full_abba_retry_authorized_by_diagnostic_alone'):
            self.assertIs(result[key],False)

    def test_any_failed_requirement_blocks_setup_pass(self):
        for i in range(4):
            flags=[True]*4;flags[i]=False
            self.assertFalse(diagnostic_decision(*flags)['setup_diagnostic_passed'])

    def test_full_quality_audit_cannot_accept_setup_phases(self):
        evidence=dict(record={},expected={},users=[],lengths=[],full=False,
            phases=[{'phase':1},{'phase':2}],joined={},repeated=[],ancestry_stable=True,historical_pairs={})
        with patch('rapidtinyrec.audit_precision.inspect_phases',return_value=evidence):
            with self.assertRaisesRegex(AssertionError,'complete ABBA'):
                audit(None,None,None,None,None)

    def test_frozen_protocol_has_separate_scope_and_budget(self):
        path=Path(__file__).resolve().parents[1]/'benchmarks/preparation/quality-frontier/setup-feasibility-protocol.json'
        p=json.loads(path.read_text())
        self.assertEqual([x['phase'] for x in p['phases']],[1,2])
        self.assertEqual(p['cohort_indices'],[1,2,6,10])
        self.assertEqual(p['supervisor_child_deadline_s']+p['minimum_reserved_cleanup_s'],1800)
        self.assertEqual(p['resources']['time'],'00:30:00')
        self.assertFalse(p['eligible_for_full_quality_review'])
        self.assertFalse(p['full_abba_retry_authorized_by_diagnostic_alone'])


class WrapperEvidenceTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        base=Path(temp.name).resolve();self.root=base/'out';self.root.mkdir()
        local=base/'local';local.mkdir()
        fs={'mount':'/','filesystem':'tmpfs'}
        for phase in (1,2):
            output=self.root/f'phase_{phase}';output.mkdir()
            stage=LogStage(local,output,fs)
            for name in LOG_NAMES:(stage.root/name).write_bytes(f'{phase}:{name}'.encode())
            publication=stage.publish(loggers_disabled=True,require_complete=True)
            save_record(output/'log-staging.json',{'publication':publication})
            save_record(output/'manifest.json',{'api_log_staging':True})
            save_record(output/'log-staging-location.json',{
                'schema':'precision-log-location-v1','local_directory':str(stage.root),
                'output_directory':str(output),'filesystem':fs})
        self.start={'schema':'precision-log-wrapper-v1','job_id':'test-job',
            'child_timeout_s':1620,'filesystem':fs,'local_directory':str(local)}
        self.finish={'child':{'schema':'precision-child-exit-v1','returncode':0,
            'reason':'exited','completed_normally':True,'signal':None,'elapsed_s':900},
            'recovery':recover_logs(self.root,local,fs)}
        self.record={'job_id':'test-job'}
        self.write()

    def write(self):
        save_record(self.root/'log-wrapper-start.json',self.start)
        save_record(self.root/'log-wrapper-exit.json',self.finish)

    def test_actual_published_and_recovered_logs_are_accepted(self):
        self.assertEqual(validate_wrapper(self.root,self.record),self.finish)

    def test_abnormal_exit_or_overbudget_never_passes(self):
        original=deepcopy(self.finish)
        for key,value in [('returncode',1),('reason','deadline'),('completed_normally',False),
                          ('signal',15),('elapsed_s',1621),('elapsed_s',float('nan'))]:
            with self.subTest(key=key,value=value):
                self.finish=deepcopy(original);self.finish['child'][key]=value
                # JSON disallows NaN on write; directly exercise a loaded corrupt record.
                if key=='elapsed_s' and value!=value:
                    with patch('rapidtinyrec.audit_precision_setup.read',side_effect=lambda p:
                        self.finish if p.name=='log-wrapper-exit.json' else json.loads(p.read_text())):
                        with self.assertRaises(AssertionError):validate_wrapper(self.root,self.record)
                else:
                    self.write()
                    with self.assertRaises(AssertionError):validate_wrapper(self.root,self.record)

    def test_deadline_job_and_filesystem_mismatch_fail(self):
        original=deepcopy(self.start)
        for key,value in [('job_id','wrong'),('child_timeout_s',1800),
                          ('filesystem',{'mount':'/','filesystem':'nfs'})]:
            self.start=deepcopy(original);self.start[key]=value;self.write()
            with self.assertRaises(AssertionError):validate_wrapper(self.root,self.record)

    def test_recovery_omission_duplicate_and_changed_hash_fail(self):
        original=deepcopy(self.finish)
        for change in ('omission','duplicate','hash','leftovers','incomplete'):
            self.finish=deepcopy(original);r=self.finish['recovery']
            if change=='omission':r['locations'].pop()
            if change=='duplicate':r['locations'][1]=deepcopy(r['locations'][0])
            if change=='hash':r['locations'][0]['publication']['files'][0]['sha256']='bad'
            if change=='leftovers':r['leftover_local_entries']=['not-preserved']
            if change=='incomplete':r['recovery_complete']=False
            self.write()
            with self.subTest(change=change),self.assertRaises(AssertionError):
                validate_wrapper(self.root,self.record)


if __name__=='__main__':unittest.main()

