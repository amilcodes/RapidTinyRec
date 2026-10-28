import ast
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]


class BeamWaveBenchmarkBoundaryTests(unittest.TestCase):
    def test_local_entry_fails_before_heavy_import(self):
        env=dict(os.environ)
        for name in ('SLURM_JOB_ID','OSCAR_RUN_DIR'):env.pop(name,None)
        command=[sys.executable,'-m','rapidtinyrec.benchmark_beam_wave','--phase','0']
        for name in ('release-root','reference-results','output-root','api-log-local-root'):command+=['--'+name,'/nonexistent']
        result=subprocess.run(command,cwd=ROOT,env=env,text=True,capture_output=True,timeout=10)
        self.assertNotEqual(result.returncode,0);self.assertIn('actual allocated job',result.stderr)
        self.assertNotIn('No module named',result.stderr)

    def test_distinct_collector_and_manifest_are_used(self):
        tree=ast.parse((ROOT/'rapidtinyrec/benchmark_beam_wave.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        measured=[n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='measure_pair']
        self.assertEqual(len(measured),1)
        self.assertEqual([ast.unparse(a) for a in measured[0].args],
            ['spec','engine.generate','torch.cuda.synchronize','time.perf_counter','unpack'])
        names={n.func.id for n in calls if isinstance(n.func,ast.Name)}
        self.assertIn('observe_allocation',names);self.assertIn('unpack_requests',names)
        imports=[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any('audit_precision' in s or 'benchmark_precision' in s for s in imports))


if __name__=='__main__':unittest.main()
