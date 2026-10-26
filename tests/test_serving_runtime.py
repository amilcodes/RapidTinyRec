import ast
import hashlib
from pathlib import Path
import types
import unittest
from unittest.mock import patch

from rapidtinyrec import serving_generation, serving_runtime
from rapidtinyrec.serving_context import OwnedCudaContext
from test_serving_context import FakeBackend


class RuntimeTests(unittest.TestCase):
    def prepared(self):
        self.backend = FakeBackend()
        self.context = OwnedCudaContext(backend=self.backend)
        self.arena = object()
        self.pool = types.SimpleNamespace(arena=self.arena, entries={'graph': object()},
                                          replays=0, frozen=False)
        self.compiler = types.SimpleNamespace(events=[{}], phase='warmup')
        self.pool.freeze = lambda: setattr(self.pool, 'frozen', True)
        self.compiler.freeze = lambda: setattr(self.compiler, 'phase', 'measurement')
        self.calls = []
        def generate(user, replay, direct):
            self.calls.append(('generate', replay, direct, self.context.stream.drains))
            if replay: self.pool.replays += 4
            return object()
        def extract(model, output, user):
            self.calls.append(('extract', self.context.stream.drains))
            return {'ranks': [user['cohort_index']]}
        self.addCleanup(self.context.close)
        self.extract_patch = patch.object(serving_runtime, 'materialize', extract)
        self.extract_patch.start(); self.addCleanup(self.extract_patch.stop)
        with patch.object(serving_runtime, 'make_generation', return_value=generate):
            self.lane = serving_runtime.PreparedFoldedLane(
                self.context, object(), self.compiler, self.arena, self.pool,
                types.SimpleNamespace(pool=self.pool), object(), object(), {}, {})
        self.user = dict(cohort_index=7, prompt_tokens=76)
        return self.lane

    def test_original_generation_ast_is_preserved_under_current_python(self):
        path = Path(__file__).parent/'fixtures/serving_generation_original.py.txt'
        source = path.read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(),
                         '47925842d794afa854f3f6b0e12c47656346d61eaec1b719d327e9776c170190')
        def generation(text):
            return next(n for n in ast.walk(ast.parse(text))
                        if isinstance(n, ast.FunctionDef) and n.name == 'generate')
        original = generation(source.decode())
        current = generation(Path(serving_generation.__file__).read_text())
        self.assertEqual(ast.dump(original), ast.dump(current))

    def test_generation_boundary_precedes_materialization_and_final_drain(self):
        lane = self.prepared(); lane.freeze()
        before = self.context.stream.drains
        result = lane.run(self.user)
        self.assertEqual(self.calls, [('generate', True, True, before), ('extract', before+1)])
        self.assertEqual(self.context.stream.drains, before+2)
        self.assertGreater(result['generation_elapsed_s'], 0)
        self.assertEqual(result['raw'], {'ranks': [7]})
        self.assertEqual(result['cohort_index'], 7)

    def test_warm_retained_outputs_survive_until_explicit_release(self):
        lane = self.prepared()
        raw = lane.warm(self.user, replay=True, retain=True)
        output = lane.retained[7]
        lane.warm(dict(cohort_index=9, prompt_tokens=121), replay=False)
        self.assertIs(lane.retained[7], output)
        lane.freeze(); lane.release_retained()
        self.assertFalse(lane.retained)
        self.assertEqual(lane.run(self.user)['raw'], raw)

    def test_timing_before_freeze_or_before_retained_release_fails(self):
        for retained in (False, True):
            lane = self.prepared()
            if retained:
                lane.warm(self.user, replay=True, retain=True); lane.freeze()
            with self.assertRaises(RuntimeError): lane.run(self.user)
            self.assertTrue(self.context.failed)
            lane.close()

    def test_frozen_counter_or_replay_drift_rejects_sample(self):
        for kind in ('compile', 'graph', 'replay', 'phase'):
            lane = self.prepared(); lane.freeze()
            original = lane.generate
            def mutate(*args):
                output = original(*args)
                if kind == 'compile': self.compiler.events.append({})
                elif kind == 'graph': self.pool.entries['extra'] = object()
                elif kind == 'phase': self.compiler.phase = 'warmup'
                else: self.pool.replays += 1
                return output
            lane.generate = mutate
            with self.assertRaises(RuntimeError): lane.run(self.user)
            self.assertTrue(self.context.failed)
            lane.close()

    def test_duplicate_retention_and_late_warmup_fail_without_reuse(self):
        for kind in ('duplicate', 'frozen'):
            lane = self.prepared()
            lane.warm(self.user, replay=True, retain=True)
            if kind == 'frozen': lane.freeze()
            with self.assertRaises((ValueError, RuntimeError)):
                lane.warm(self.user, replay=True, retain=True)
            self.assertTrue(self.context.failed)
            lane.close()

    def test_materializer_preserves_fp32_bits_and_ancestry_on_cpu(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Real Torch check runs in decoding CI')
        # Signed zero distinguishes exact-bit extraction from numeric equality.
        sequence = torch.tensor([0.0, -0.0], dtype=torch.float32)
        transition = torch.tensor([[0., -0.], [-1., -2.]], dtype=torch.float32)
        output = types.SimpleNamespace(scores=tuple(torch.ones(2, 3) for _ in range(5)),
            sequences_scores=sequence, sequences=torch.tensor([[8, 9, 1, 2], [8, 9, 3, 4]]),
            beam_indices=torch.tensor([[0, 1], [1, 0]]))
        seen = []
        class Model:
            def compute_transition_scores(self, *args, **kwargs):
                seen.append(kwargs); return transition
        raw = serving_runtime.materialize(Model(), output, {'prompt_tokens': 2})
        self.assertEqual(seen, [{'normalize_logits': False}])
        self.assertEqual(raw['sequence_bits'], [0, -2147483648])
        self.assertEqual(raw['transition_bits'][0], [0, -2147483648])
        self.assertEqual(raw['beam_indices'], [[0, 1], [1, 0]])
        sequence.fill_(10); transition.fill_(10); output.beam_indices.fill_(99)
        self.assertEqual(raw['sequence_bits'], [0, -2147483648])
        self.assertEqual(raw['beam_indices'], [[0, 1], [1, 0]])


if __name__ == '__main__': unittest.main()
