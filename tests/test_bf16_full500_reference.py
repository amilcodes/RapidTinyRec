"""Real preserved historical manifest distinguishes file bytes from self-digest."""
from copy import deepcopy
import gzip,hashlib,json,os
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
from rapidtinyrec.bf16_full500 import PROTOCOL,load_protocol,validate_reference
from rapidtinyrec import bf16_full500_runner as runner
ROOT=Path(__file__).resolve().parents[1]

class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.raw=gzip.decompress((ROOT/'tests/fixtures/bf16-historical-reference-manifest.json.gz').read_bytes())
        self.protocol=load_protocol(ROOT/PROTOCOL)
    def test_original_manifest_file_and_content_pins_are_distinct(self):
        r=validate_reference(self.raw,self.protocol);p=self.protocol['cohort']
        self.assertEqual(hashlib.sha256(self.raw).hexdigest(),'9ebf231fcdfd5f92c31131d76cb41638f7f03d3d6258b8913e4c8da54d4b77fd')
        self.assertEqual(r['manifest_sha256'],'503ba174a92cde0028fb6a9ad1742fc2bd5f61651ef9eb9d217f1b69990e5bcc')
        self.assertNotEqual(p['manifest_file_sha256'],p['manifest_content_sha256'])
        self.assertEqual(len(r['users']),500)
    def test_swapped_or_legacy_ambiguous_hash_contract_rejected(self):
        p=deepcopy(self.protocol);a,b=(p['cohort'][k] for k in ('manifest_file_sha256','manifest_content_sha256'))
        p['cohort'].update(manifest_file_sha256=b,manifest_content_sha256=a)
        with self.assertRaises(AssertionError):validate_reference(self.raw,p)
        p=deepcopy(self.protocol);p['cohort']['manifest_sha256']=a
        with self.assertRaisesRegex(AssertionError,'Ambiguous'):validate_reference(self.raw,p)
    def test_file_padding_and_content_tampering_rejected_independently(self):
        with self.assertRaisesRegex(AssertionError,'file bytes'):validate_reference(self.raw+b' ',self.protocol)
        r=json.loads(self.raw);r['users'][0]['prompt_tokens']+=1
        altered=json.dumps(r).encode();p=deepcopy(self.protocol)
        p['cohort']['manifest_file_sha256']=hashlib.sha256(altered).hexdigest()
        with self.assertRaisesRegex(AssertionError,'content digest'):validate_reference(altered,p)
    def test_runner_checks_reference_before_catalog_or_generation_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory).resolve();(run/'outputs').mkdir();(run/'code').mkdir();hist=run/'historical';hist.mkdir()
            (hist/'manifest.json').write_bytes(self.raw+b' ')
            argv=['runner','run','--run-root',str(run),'--historical',str(hist),'--release-root',str(run/'release')]
            with patch('sys.argv',argv),patch.dict(os.environ,{'SLURM_JOB_ID':'123','OSCAR_RUN_DIR':str(run),'OSCAR_OUTPUT_DIR':str(run/'outputs')}),patch.object(runner,'observe',return_value={'job_id':'123'}),patch.object(runner,'load_protocol',return_value=self.protocol),patch.object(runner,'run_stages') as launch,patch('rapidtinyrec.audit_bf16_full500.catalog_preflight') as catalog:
                with self.assertRaisesRegex(AssertionError,'file bytes'):runner.main()
                launch.assert_not_called();catalog.assert_not_called()
            result=json.loads((run/'outputs/bf16-full500-controller-failure.json').read_text())
            self.assertIn('file bytes',result['error'])
            self.assertFalse((run/'outputs/bf16-full500-generation-start.json').exists())

if __name__=='__main__':unittest.main()
