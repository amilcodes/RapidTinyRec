import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rapidtinyrec import serving_initialize


class InitializeTests(unittest.TestCase):
    def test_allocation_statements_match_pinned_original(self):
        source = (Path(__file__).parent/'fixtures/serving_generation_original.py.txt').read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(),
                         '47925842d794afa854f3f6b0e12c47656346d61eaec1b719d327e9776c170190')
        original = ast.parse(source)
        main = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        original_body = next(n for n in main.body if isinstance(n, ast.With)).body
        def target(node, name):
            return isinstance(node, ast.Assign) and any(isinstance(n, ast.Name) and n.id == name for n in node.targets)
        a = next(i for i, n in enumerate(original_body) if target(n, 'tok'))
        b = next(i for i, n in enumerate(original_body) if target(n, 'manifest'))
        c = next(i for i, n in enumerate(original_body) if target(n, 'compiler'))
        d = next(i for i, n in enumerate(original_body) if isinstance(n, ast.FunctionDef) and n.name == 'generate')
        wanted = original_body[a:b] + original_body[c:d]
        current = ast.parse(Path(serving_initialize.__file__).read_text())
        allocate = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == '_allocate')
        normal_construction = next(n for n in allocate.body if isinstance(n, ast.With))
        self.assertEqual(ast.unparse(normal_construction.items[0].context_expr), 'torch.inference_mode(False)')
        self.assertEqual([ast.dump(n) for n in wanted], [ast.dump(n) for n in normal_construction.body])

    def test_local_initializer_is_rejected_before_torch_import_or_files(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'allocated compute'):
                serving_initialize.lane_initializer('/missing', '/missing', '/missing')

    def test_warm_schedule_retains_first_per_length_and_scans_after_reverse_repeats(self):
        events = []; retained = []
        users = [dict(cohort_index=i, prompt_tokens=76 if i < 2 else 121) for i in range(4)]
        class Lane:
            def warm(self, user, *, replay, retain=False):
                key = user['cohort_index']; events.append((key, replay, retain))
                if retain: retained.append(key)
                return {'bits': [key]}
            def scan_retained(self, rows):
                events.append('scan'); return {str(i): {'bits': [999]} for i in retained}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); serving_initialize.warm_lane(Lane(), users, root)
            self.assertEqual(retained, [0, 2])
            self.assertEqual(events[:8], [(i, replay, replay and i in (0, 2)) for i in range(4) for replay in (False, True)])
            self.assertEqual(events[8:], [(i, True, False) for i in (3, 2, 1, 0)]+['scan'])
            self.assertEqual(len(list(root.glob('*.json'))), 13)
            # A changed retained output is evidence, never silently overwritten
            # with the initial warm result or discarded by an early equality test.
            self.assertEqual(json.loads((root/'retained_after_mixed.json').read_text())['0']['bits'], [999])
            self.assertEqual(json.loads((root/'0_replay.json').read_text())['bits'], [0])

    def test_saved_outputs_are_exclusive_and_partial_failure_keeps_prior_rows(self):
        users = [dict(cohort_index=i, prompt_tokens=76) for i in range(2)]
        class Lane:
            def warm(self, user, *, replay, retain=False):
                if user['cohort_index'] == 1: raise ValueError('warm failure')
                return {'bits': [1]}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(ValueError): serving_initialize.warm_lane(Lane(), users, root)
            self.assertEqual(len(list(root.glob('*.json'))), 2)
            with self.assertRaises(FileExistsError): serving_initialize.save_exclusive(root/'0_dense.json', {})
            self.assertEqual(json.loads((root/'0_dense.json').read_text()), {'bits': [1]})


if __name__ == '__main__': unittest.main()
