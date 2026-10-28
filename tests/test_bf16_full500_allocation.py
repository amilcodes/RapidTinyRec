"""Distinct BF16 full500 source/config/scheduler identity boundary."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from rapidtinyrec import bf16_full500_runner as runner
from rapidtinyrec.bf16_full500 import load_protocol,PROTOCOL,PROTOCOL_SHA256
ROOT=Path(__file__).resolve().parents[1]


class AllocationTests(unittest.TestCase):
    def setUp(self):
        self.run=Path('/oscar/scratch/aagraw73/agent-runs/rapidtinyrec-bf16-full500/fixture')
        self.protocol=load_protocol(ROOT/PROTOCOL)
        resources=dict(self.protocol['resources'],setup=['module purge'],command=['bash','jobs/bf16-full500.sh'])
        self.manifest=dict(submission='prepared',job_id=None,project=runner.PROJECT,profile=runner.PROFILE,run_id='fixture',remote_dir=str(self.run),resources=resources,files=[PROTOCOL],sha256={PROTOCOL:PROTOCOL_SHA256},ssh_target='aagraw73@sshcampus.ccv.brown.edu')
        self.config=dict(project=runner.PROJECT,ssh_target=self.manifest['ssh_target'],profiles={runner.PROFILE:resources})
        # Real scheduler parser/validator; only OS and SSH boundaries are synthetic.
        self.raw=f'JobId=123 JobState=RUNNING NodeList=synthetic-node Partition=gpu QOS=norm-gpu NumCPUs=4 NumNodes=1 MinMemoryNode=32G TimeLimit=00:30:00 WorkDir={self.run} Command={self.run}/job.sh AllocTRES=cpu=4,mem=32G,node=1,billing=1,gres/gpu=1 TresPerNode=gres/gpu:nvidia_rtx_a6000:1'

    def observe(self):
        with patch.dict(os.environ,{'OSCAR_RUN_DIR':str(self.run),'OSCAR_OUTPUT_DIR':str(self.run/'outputs'),'SLURM_JOB_ID':'123'}),patch.object(runner,'read',side_effect=lambda p:self.manifest if p.name=='source-manifest.json' else self.config),patch.object(runner,'load_protocol',return_value=self.protocol),patch.object(Path,'read_text',return_value='123\n'),patch.object(Path,'is_symlink',return_value=False),patch.object(runner,'verify_source') as verify,patch.object(runner,'digest',return_value='fixture-digest'),patch.object(runner.subprocess,'run',return_value=SimpleNamespace(stdout=self.raw)):
            result=runner.observe(self.run,self.run/'code',self.run/'outputs')
            verify.assert_called_once_with(self.run/'code',self.manifest)
            return result

    def test_distinct_identity_and_typed_live_allocation_required(self):
        result=self.observe();self.assertEqual(result['profile'],'bf16-vs-tf32-full500-quality')
        self.assertEqual(result['job_id'],'123');self.assertIsNone(self.manifest['job_id'])
        self.assertEqual(result['allocation_identity']['scheduler_fields']['JobState'],'RUNNING')
        self.raw=self.raw.replace('TresPerNode=gres/gpu:nvidia_rtx_a6000:1','TresPerNode=gres/gpu:1')
        with self.assertRaises(AssertionError):self.observe()

    def test_pilot_identity_or_stale_source_rejected(self):
        for key,value in (('profile','bf16-vs-tf32-pilot'),('project','rapidtinyrec-bf16-pilot'),('job_id','122'),('files',[PROTOCOL,PROTOCOL])):
            old=self.manifest[key];self.manifest[key]=value
            try:
                with self.subTest(key=key),self.assertRaises(AssertionError):self.observe()
            finally:self.manifest[key]=old
        self.manifest['sha256'][PROTOCOL]='other'
        with self.assertRaises(AssertionError):self.observe()

if __name__=='__main__':unittest.main()
