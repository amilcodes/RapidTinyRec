import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from rapidtinyrec import serving_harness as a

ROOT=Path(__file__).resolve().parents[1]


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.repo=Path(self.tmp.name)/'repo';self.repo.mkdir();self.out=Path(self.tmp.name)/'harness'
        names=['rapidtinyrec/serving_runner.py','rapidtinyrec/serving_run_audit.py','rapidtinyrec/serving_terminal_audit.py',
               'rapidtinyrec/source/LogitProcessor.py','rapidtinyrec/expected.json',a.PROTOCOL,a.TEMPLATES+a.CONFIG,
               a.TEMPLATES+'serving-throughput.sh','tests/test_serving_example.py']
        for name in names:
            path=self.repo/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes((ROOT/name).read_bytes() if (ROOT/name).is_file() else b'# fixture\n')
        subprocess.run(['git','init','-q'],cwd=self.repo,check=True)
        self.git('add','.');self.git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture')

    def git(self,*args):return a.git(self.repo,*args)

    def test_clean_commit_recursive_source_and_exact_templates(self):
        report=a.prepare(self.repo,self.out)
        self.assertFalse(report['submission_ready'])
        self.assertIn('rapidtinyrec/source/LogitProcessor.py',report['sha256'])
        self.assertEqual(report['source_commit'],self.git('rev-parse','HEAD').decode().strip())
        self.assertEqual((self.out/a.COMMAND[1]).read_bytes(),(self.repo/(a.TEMPLATES+'serving-throughput.sh')).read_bytes())
        self.assertEqual(a.validate_snapshot(self.out,self.repo),report)

    def test_dirty_or_existing_harness_never_replaced(self):
        path=self.repo/'rapidtinyrec/serving_runner.py';path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'Clean committed'):a.prepare(self.repo,self.out)
        self.assertFalse(self.out.exists());self.git('checkout','--','rapidtinyrec/serving_runner.py')
        a.prepare(self.repo,self.out);saved=(self.out/'source-provenance.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'already exists'):a.prepare(self.repo,self.out)
        self.assertEqual((self.out/'source-provenance.json').read_bytes(),saved)

    def test_bytecode_or_missing_recursive_dependency_rejected(self):
        names=self.git('ls-tree','-r','--name-only','HEAD').decode().splitlines()
        for entries in (names+['rapidtinyrec/stale.pyc'],[n for n in names if n!='rapidtinyrec/source/LogitProcessor.py']):
            with self.assertRaises(ValueError):a.repository_mapping(entries)

    def test_changed_source_or_unlisted_file_fails_local_validation(self):
        a.prepare(self.repo,self.out);path=self.out/'rapidtinyrec/source/LogitProcessor.py';raw=path.read_bytes();path.write_bytes(raw+b' ')
        with self.assertRaisesRegex(ValueError,'source inventory'):a.validate_snapshot(self.out)
        path.write_bytes(raw);(self.out/'extra.py').write_text('# extra')
        with self.assertRaisesRegex(ValueError,'source inventory'):a.validate_snapshot(self.out)

    def test_provenance_revision_and_repository_mapping_are_explicit(self):
        report=a.prepare(self.repo,self.out);p=self.out/'source-provenance.json';value=json.loads(p.read_text())
        altered=copy.deepcopy(value);altered['source_commit']='unknown'
        with self.assertRaisesRegex(ValueError,'identity'):a.validate_provenance(altered,report['sha256'])
        altered=copy.deepcopy(value);altered['repository_paths'][a.CONFIG]='other-config.json'
        with self.assertRaisesRegex(ValueError,'path mapping'):a.validate_provenance(altered,report['sha256'])

    def test_job_template_only_launches_fixed_supervised_runner(self):
        path=ROOT/(a.TEMPLATES+'serving-throughput.sh');result=subprocess.run(['bash','-n',str(path)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        text=path.read_text();self.assertIn('"$python_bin" -B -m rapidtinyrec.serving_runner',text)
        self.assertNotIn('sbatch',text);self.assertNotIn('precision_full500',text)


if __name__=='__main__':unittest.main()
