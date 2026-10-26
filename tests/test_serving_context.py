import contextlib
import threading
import types
import unittest

from rapidtinyrec.serving_context import OwnedCudaContext
from rapidtinyrec.serving_lanes import LaneServer, Request


class FakeBackend:
    """CPU state machine only; deliberately has no device-wide synchronize."""
    def __init__(self):
        self.local = threading.local()
        self.cuda = self
        self.streams = []
        self.mode = False

    @contextlib.contextmanager
    def device(self, index):
        previous = getattr(self.local, 'device', 9)
        self.local.device = index
        try: yield
        finally: self.local.device = previous

    @contextlib.contextmanager
    def stream(self, stream):
        previous = getattr(self.local, 'stream', None)
        self.local.stream = stream
        try: yield
        finally: self.local.stream = previous

    @contextlib.contextmanager
    def inference_mode(self):
        previous = getattr(self.local, 'inference', False)
        self.local.inference = True
        try: yield
        finally: self.local.inference = previous

    def current_device(self): return self.local.device
    def current_stream(self, index): return self.local.stream

    def Stream(self, device):
        stream = types.SimpleNamespace(device=types.SimpleNamespace(type='cuda', index=device),
                                       cuda_stream=len(self.streams)+1, drains=0, error=None)
        def synchronize():
            stream.drains += 1
            if stream.error: raise stream.error
        stream.synchronize = synchronize
        self.streams.append(stream)
        return stream


class OwnedContextTests(unittest.TestCase):
    def test_context_and_materialization_finish_before_return(self):
        backend = FakeBackend(); ctx = OwnedCudaContext(1, backend=backend)
        events = []
        def action():
            self.assertEqual(backend.current_device(), 1)
            self.assertIs(backend.current_stream(1), ctx.stream)
            self.assertTrue(backend.local.inference)
            events.append('materialized')
            return [1, 2]
        self.assertEqual(ctx.call(action), [1, 2])
        self.assertEqual(events, ['materialized'])
        self.assertEqual(ctx.stream.drains, 1)
        self.assertEqual(backend.current_device(), 9)
        self.assertIsNone(backend.current_stream(9))
        self.assertFalse(backend.local.inference)
        ctx.close(); ctx.close()
        self.assertEqual(ctx.stream.drains, 2)
        with self.assertRaises(RuntimeError): ctx.call(action)

    def test_callback_error_drains_and_is_preserved_when_drain_also_fails(self):
        ctx = OwnedCudaContext(backend=FakeBackend())
        original = ValueError('generation failed')
        ctx.stream.error = RuntimeError('CUDA drain failed')
        def fail(): raise original
        with self.assertRaises(ValueError) as caught: ctx.call(fail)
        self.assertIs(caught.exception, original)
        self.assertIn('CUDA drain failed', original.__notes__[0])
        self.assertTrue(ctx.failed)
        with self.assertRaises(RuntimeError): ctx.call(lambda: None)
        with self.assertRaises(RuntimeError): ctx.close()
        self.assertTrue(ctx.closed)

    def test_drain_failure_never_returns_success(self):
        ctx = OwnedCudaContext(backend=FakeBackend())
        ctx.stream.error = RuntimeError('drain')
        with self.assertRaises(RuntimeError): ctx.call(lambda: 'not returned')
        self.assertTrue(ctx.failed)

    def test_wrong_thread_cannot_call_inspect_or_close(self):
        ctx = OwnedCudaContext(backend=FakeBackend()); errors = []
        def other():
            for action in (lambda: ctx.call(lambda: None), ctx.identity, ctx.close):
                try: action()
                except RuntimeError: errors.append(True)
        thread = threading.Thread(target=other); thread.start(); thread.join()
        self.assertEqual(len(errors), 3)
        self.assertFalse(ctx.closed); self.assertFalse(ctx.failed)
        ctx.close()

    def test_reentry_or_context_escape_fails_and_drains(self):
        for kind in ('reentry', 'stream', 'device'):
            backend = FakeBackend(); ctx = OwnedCudaContext(backend=backend)
            def action():
                if kind == 'reentry': ctx.call(lambda: None)
                elif kind == 'stream': backend.local.stream = None
                else: backend.local.device = 8
            with self.assertRaises(RuntimeError): ctx.call(action)
            self.assertTrue(ctx.failed); self.assertEqual(ctx.stream.drains, 1)
            ctx.close()

    def test_lane_server_keeps_context_lifecycle_on_owner_threads(self):
        backend = FakeBackend(); identities = []; barrier = threading.Barrier(2)
        def init(index): return OwnedCudaContext(backend=backend)
        def run(ctx, payload):
            def execute(): barrier.wait(timeout=2); return payload
            return ctx.call(execute)
        def freeze(ctx): identities.append(ctx.identity())
        with LaneServer(init, run, freeze, lambda ctx: ctx.close()) as server:
            server.freeze()
            rows = server.run_burst([Request('a', 1), Request('b', 2)], 2)
        self.assertEqual([r['result'] for r in rows], [1, 2])
        self.assertEqual(len({r['owner_thread_id'] for r in identities}), 2)
        self.assertEqual(len({r['stream_id'] for r in identities}), 2)
        self.assertEqual([s.drains for s in backend.streams], [2, 2])

    def test_invalid_device_or_nonprivate_stream_rejected(self):
        for index in (-1, True, 'cuda:0'):
            with self.assertRaises(ValueError): OwnedCudaContext(index, backend=FakeBackend())
        backend = FakeBackend(); original = backend.Stream
        def bad(device):
            stream = original(device); stream.cuda_stream = 0; return stream
        backend.Stream = bad
        with self.assertRaises(ValueError): OwnedCudaContext(backend=backend)


if __name__ == '__main__': unittest.main()
