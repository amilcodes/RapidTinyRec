"""Owner-thread CUDA execution envelope, separate from model arithmetic.

This is preparation for a serving runtime, not a complete runtime or ownership
audit. Callbacks must materialize independent outputs before returning. A
callback can still misuse other streams internally; profiles must verify that.
"""
import threading


class OwnedCudaContext:
    """One persistent thread, indexed device and private sequential stream.

    Every callback executes in inference mode and drains this stream before
    returning, including on failure. A failed context cannot execute again.
    No device-wide synchronization is performed. The optional backend exists
    for CPU lifecycle tests; production callers use the lazily imported Torch.
    """
    def __init__(self, device_index=0, *, backend=None):
        if type(device_index) is not int or device_index < 0:
            raise ValueError('Explicit nonnegative CUDA device index required')
        if backend is None:
            import torch
            backend = torch
        self.backend = backend
        self.device_index = device_index
        self.owner = threading.current_thread()
        self.busy = False
        self.failed = False
        self.closed = False
        with backend.cuda.device(device_index):
            self.stream = backend.cuda.Stream(device=device_index)
        if self.stream.device.type != 'cuda' or self.stream.device.index != device_index:
            raise ValueError('Stream device differs from the explicit owner device')
        if type(self.stream.cuda_stream) is not int or self.stream.cuda_stream <= 0:
            raise ValueError('A private nondefault CUDA stream is required')

    def _owner(self):
        if threading.current_thread() is not self.owner:
            raise RuntimeError('CUDA context used outside its persistent owner thread')
        if self.busy:
            raise RuntimeError('CUDA context cannot be reentered')

    def _current(self):
        cuda = self.backend.cuda
        if (cuda.current_device() != self.device_index or
                cuda.current_stream(self.device_index) != self.stream):
            raise RuntimeError('Callback changed the owner device or current stream')

    def call(self, action, *args, **kwargs):
        self._owner()
        if self.closed or self.failed:
            raise RuntimeError('Closed or failed CUDA context cannot execute')
        self.busy = True
        try:
            with self.backend.inference_mode(), self.backend.cuda.device(self.device_index), self.backend.cuda.stream(self.stream):
                try:
                    self._current()
                    result = action(*args, **kwargs)
                    self._current()
                except BaseException as error:
                    # Preserve the original callback error if draining also fails.
                    try:
                        self.stream.synchronize()
                    except BaseException as drain_error:
                        error.add_note(f'Owner stream drain also failed: {drain_error!r}')
                    raise
                self.stream.synchronize()
                return result
        except BaseException:
            self.failed = True
            raise
        finally:
            self.busy = False

    def identity(self):
        """Read only while idle; this is identity metadata, not GPU overlap proof."""
        self._owner()
        if self.closed:
            raise RuntimeError('Closed context has no live ownership record')
        return dict(owner_thread_id=self.owner.ident, device=f'cuda:{self.device_index}',
                    stream_id=self.stream.cuda_stream, failed=self.failed)

    def close(self):
        self._owner()
        if self.closed:
            return
        try:
            with self.backend.cuda.device(self.device_index):
                self.stream.synchronize()
        except BaseException:
            self.failed = True
            raise
        finally:
            self.closed = True
