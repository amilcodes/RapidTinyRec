import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from rapidtinyrec.bf16_audit_recovery import PROJECT,PROFILE,RESOURCES,main,sha,validate_cpu,verify_raw


class RecoveryProvenanceTests(unittest.TestCase):
    def allocation(self):
        remote=f'/oscar/scratch/aagraw73/agent-runs/{PROJECT}/recovery-test'
        prepared=dict(submission='prepared',job_id=None,project=PROJECT,profile=PROFILE,
                      resources=copy.deepcopy(RESOURCES),remote_dir=remote,run_id='recovery-test')
        env=dict(SLURM_JOB_ID='9000000',OSCAR_RUN_DIR=remote,OSCAR_OUTPUT_DIR=remote+'/outputs')
        fields=dict(JobId='9000000',JobState='RUNNING',Partition='batch',NumCPUs='4',NumNodes='1',
                    TimeLimit='00:30:00',WorkDir=remote,Command=remote+'/job.sh',
                    AllocTRES='cpu=4,mem=32G,node=1',NodeList='cpu01')
        return prepared,'9000000',env,fields

    def test_actual_cpu_allocation(self):
        validate_cpu(*self.allocation())

    def test_stale_identity_gpu_and_wrong_resources_rejected(self):
        for key,value in (('JobId','8040768'),('JobState','COMPLETED'),('Partition','gpu'),
                          ('TimeLimit','01:00:00'),('NumCPUs','8'),('WorkDir','/other'),
                          ('AllocTRES','cpu=4,mem=32G,node=1,gres/gpu=1'),
                          ('AllocTRES','cpu=4,mem=32G,node=1,cpu=4'),
                          ('TresPerNode','gres/gpu:nvidia_rtx_a6000:1'),('NodeList','(null)')):
            args=self.allocation();args[3][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(AssertionError):validate_cpu(*args)
        args=list(self.allocation());args[1]='8040768';args[2]['SLURM_JOB_ID']='8040768'
        with self.assertRaises(AssertionError):validate_cpu(*args)

    def fixture(self,root):
        (root/'outputs').mkdir();p=root/'outputs/raw.json';p.write_text('{}')
        return [dict(path='outputs/raw.json',size=2,sha256=sha(p))]

    def test_inventory_rechecked_after_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rows=self.fixture(root);verify_raw(root,rows)
            (root/'outputs/raw.json').write_text('[]')
            with self.assertRaises(AssertionError):verify_raw(root,rows)

    def test_missing_extra_duplicate_and_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rows=self.fixture(root)
            for bad in ([],rows+rows,[dict(rows[0],path='../raw.json')],
                        [dict(rows[0],path='/raw.json')],[dict(rows[0],path='outputs//raw.json')]):
                with self.assertRaises(AssertionError):verify_raw(root,bad)
            (root/'outputs/extra').write_text('')
            with self.assertRaises(AssertionError):verify_raw(root,rows)

    def test_symlink_parent_rejected_even_inside_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rows=self.fixture(root)
            (root/'alias').symlink_to(root/'outputs',target_is_directory=True)
            with self.assertRaises(AssertionError):verify_raw(root,[dict(rows[0],path='alias/raw.json')])

    def test_entrypoint_refuses_local_before_reading_files(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(AssertionError,'Allocated compute required'):main()


if __name__=='__main__':unittest.main()
