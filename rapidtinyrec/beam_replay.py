"""Replay two unmodified HF beam-state helpers on one sequential CUDA stream.

Every helper/step/prompt shape owns inputs and outputs. Returned tensors are
borrowed until the generation lease closes, after final output extraction.
No vocabulary-wide arithmetic, constraints, selector or termination is changed.
"""
import functools
import hashlib
import inspect
import textwrap
import time

import torch
import transformers
from transformers.generation.utils import GenerationMixin

RUNNING = '_get_running_beams_for_next_iteration'
FINISHED = '_update_finished_beams'
METHODS = (RUNNING, FINISHED)
PINNED_SOURCE_SHA256 = {
    '_gather_beams': '4e29903be152b0c59a229a00b8b333466b19103330f242f52c25ccdb596e87e3',
    RUNNING: '97511a4b165780149cfebe20ed583f79e411b6dda676e2430fae57442d7e8a07',
    FINISHED: 'b2bd70364b53bdd154e4f54b22d0774846ac53f6720ccea319c8d77d38700d74',
    '_beam_search': '432fafa552a0b380f14340750057593112be902c03ae7eb4f29328d378a4d0c5',
}


def verify_helpers(model):
    if transformers.__version__ != '4.55.4':
        raise RuntimeError('Beam replay requires transformers 4.55.4')
    found = {}
    for name, expected in PINNED_SOURCE_SHA256.items():
        method = getattr(model, name)
        original = getattr(GenerationMixin, name)
        if getattr(method, '__func__', method) is not original:
            raise RuntimeError('Custom beam helpers are not supported: ' + name)
        source = textwrap.dedent(inspect.getsource(method)).strip()
        found[name] = hashlib.sha256(source.encode()).hexdigest()
        if found[name] != expected:
            raise RuntimeError(f'Pinned HF beam implementation changed: {name}: expected {expected}, observed {found[name]}')
    return found


def signature(arguments):
    result = []
    device = None
    for name, value in arguments.items():
        if isinstance(value, torch.Tensor):
            if value.requires_grad or not value.is_contiguous() or value.numel() == 0:
                raise ValueError('Expected nonempty contiguous inference tensors: ' + name)
            if value.dtype not in (torch.float32, torch.int64, torch.int32, torch.bool):
                raise ValueError('Unsupported beam tensor dtype: ' + name)
            if device is not None and device != value.device:
                raise ValueError('Beam tensors must share one device')
            device = value.device
            result.append((name, 'tensor', tuple(value.shape), tuple(value.stride()), str(value.dtype), str(value.device)))
        elif type(value) in (int, float, bool, str):
            result.append((name, type(value).__name__, value))
        else:
            raise ValueError('Unsupported beam argument: ' + name)
    if device is None:
        raise ValueError('No beam state tensors')
    return tuple(result)


class BeamReplayBuffers:
    """Owned copies of every dynamic argument, with unchanged shapes/strides."""
    def __init__(self, arguments):
        self.signature = signature(arguments)
        self.arguments = {name: torch.empty_like(value) if isinstance(value, torch.Tensor) else value
                          for name, value in arguments.items()}
        self.load(arguments)

    def load(self, arguments):
        if signature(arguments) != self.signature:
            raise ValueError('Beam input shape/stride/device/dtype or scalar argument changed')
        for name, target in self.arguments.items():
            if isinstance(target, torch.Tensor):
                target.copy_(arguments[name])

    def tensors(self):
        return tuple(v for v in self.arguments.values() if isinstance(v, torch.Tensor))


class BeamReplayLease:
    def __init__(self, pool, prompt_tokens):
        self.pool = pool
        self.prompt_tokens = prompt_tokens
        self.calls = 0
        self.closed = False
        self.entries = []

    def dispatch(self, method, arguments):
        if self.closed or self.pool.active is not self:
            raise RuntimeError('No active beam replay lease')
        if self.calls >= 10 or method != METHODS[self.calls % 2]:
            raise RuntimeError('Expected running/finished helper pairs for exactly five steps')
        step = self.calls // 2
        if method == FINISHED and (arguments['decoder_prompt_len'] != self.prompt_tokens or
                                  arguments['cur_len'] != self.prompt_tokens + step):
            raise ValueError('Beam step/prompt does not match active generation')
        output, key = self.pool.replay(method, self.prompt_tokens, step, arguments)
        self.entries.append(key)
        self.calls += 1
        return output

    def close(self):
        if self.closed or self.pool.active is not self:
            raise RuntimeError('Beam replay lease already closed')
        if self.calls != 10:
            raise RuntimeError('Expected five complete beam steps before extraction/release')
        self.closed = True
        self.pool.active = None

    def abort(self):
        """Cleanup after a failed request; this does not certify completion."""
        if self.closed or self.pool.active is not self:
            raise RuntimeError('Beam replay lease already closed')
        self.closed = True
        self.pool.active = None
        self.pool.aborted_requests += 1


class BeamReplayPool:
    def __init__(self, model, max_graphs=100):
        self.model = model
        self.helper_hashes = verify_helpers(model)
        self.originals = {name: getattr(model, name) for name in METHODS}
        self.signatures = {name: inspect.signature(method) for name, method in self.originals.items()}
        self.entries = {}
        self.events = []
        self.max_graphs = max_graphs
        self.active = None
        self.frozen = False
        self.stream_id = None
        self.device = None
        self.installed = False
        self.requests = self.replays = self.staged_bytes = self.aborted_requests = 0

    def install(self):
        if self.installed or self.active is not None:
            raise RuntimeError('Beam replay already installed or active')
        if any(name in self.model.__dict__ for name in METHODS):
            raise RuntimeError('Do not overwrite instance beam methods')
        for name in METHODS:
            @functools.wraps(self.originals[name])
            def wrapper(*args, _method=name, **kwargs):
                if self.active is None:
                    raise RuntimeError('Acquire a generation lease before beam search')
                arguments = self.signatures[_method].bind(*args, **kwargs).arguments
                return self.active.dispatch(_method, arguments)
            setattr(self.model, name, wrapper)
        self.installed = True
        return self

    def uninstall(self):
        if not self.installed or self.active is not None:
            raise RuntimeError('Close the beam lease before uninstalling')
        for name in METHODS:
            delattr(self.model, name)
        self.installed = False

    def request(self, prompt_tokens):
        if not self.installed or self.active is not None:
            raise RuntimeError('Install replay and consume/close the previous generation first')
        if type(prompt_tokens) is not int or prompt_tokens < 1:
            raise ValueError('Invalid prompt length')
        self.active = BeamReplayLease(self, prompt_tokens)
        self.requests += 1
        return self.active

    def _stream(self, arguments):
        tensor = next(v for v in arguments.values() if isinstance(v, torch.Tensor))
        if tensor.device.type != 'cuda':
            raise ValueError('Beam graph replay requires CUDA')
        stream_id = torch.cuda.current_stream(tensor.device).cuda_stream
        if self.stream_id is None:
            self.stream_id, self.device = stream_id, tensor.device
        elif self.stream_id != stream_id or self.device != tensor.device:
            raise RuntimeError('Beam replay is bound to one sequential CUDA stream/device')

    @torch.inference_mode()
    def prepare(self, key, arguments):
        if self.frozen or len(self.entries) >= self.max_graphs:
            raise RuntimeError('Beam capture after freeze or beyond graph bound')
        self._stream(arguments)
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        before = torch.cuda.memory_allocated(self.device)
        buffers = BeamReplayBuffers(arguments)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        method = self.originals[key[0]]
        with torch.cuda.stream(stream):
            for _ in range(3):
                warm = method(**buffers.arguments)
                del warm
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        # Independent private pool per helper, step and prompt length.
        with torch.cuda.graph(graph, stream=stream):
            output = method(**buffers.arguments)
        torch.cuda.synchronize(self.device)
        self.entries[key] = (buffers, graph, output)
        self.check_ownership()
        self.events.append(dict(method=key[0], prompt_tokens=key[1], step=key[2],
            signature=key[3], input_bytes=sum(t.numel()*t.element_size() for t in buffers.tensors()),
            output_bytes=sum(t.numel()*t.element_size() for t in output),
            capture_and_warmup_s=time.perf_counter()-started,
            allocated_delta_bytes=torch.cuda.memory_allocated(self.device)-before))

    @torch.inference_mode()
    def replay(self, method, prompt_tokens, step, arguments):
        if self.active is None:
            raise RuntimeError('Use an active generation lease')
        self._stream(arguments)
        key = (method, prompt_tokens, step, signature(arguments))
        if key not in self.entries:
            self.prepare(key, arguments)
        buffers, graph, output = self.entries[key]
        buffers.load(arguments)
        graph.replay()
        self.replays += 1
        self.staged_bytes += sum(t.numel()*t.element_size() for t in buffers.tensors())
        return output, key

    def check_ownership(self):
        tensors = [t for b, _, outputs in self.entries.values() for t in (*b.tensors(), *outputs)]
        identities = [(t.device, t.untyped_storage().data_ptr()) for t in tensors]
        if len(set(identities)) != len(identities):
            raise AssertionError('Beam input/output storage aliases across helpers/steps')

    def freeze(self, prompt_lengths):
        if self.active is not None or self.aborted_requests:
            raise RuntimeError('Complete and close all warmup leases before freeze')
        expected = {(m, n, s) for m in METHODS for n in prompt_lengths for s in range(5)}
        if len(self.entries) != len(expected) or {k[:3] for k in self.entries} != expected:
            raise RuntimeError('Expected both helpers at every length and five steps')
        self.check_ownership()
        self.frozen = True

    def report(self):
        self.check_ownership()
        inputs = [t for b, _, _ in self.entries.values() for t in b.tensors()]
        outputs = [t for _, _, o in self.entries.values() for t in o]
        unique = {(t.device, t.untyped_storage().data_ptr()): t.untyped_storage().nbytes() for t in inputs+outputs}
        return dict(graphs=len(self.entries), frozen=self.frozen, replays=self.replays, requests=self.requests,
            staged_bytes=self.staged_bytes, aborted_requests=self.aborted_requests,
            input_logical_bytes=sum(t.numel()*t.element_size() for t in inputs),
            output_logical_bytes=sum(t.numel()*t.element_size() for t in outputs),
            unique_input_output_storage_bytes=sum(unique.values()), disjoint_step_storage=True,
            active_request=self.active is not None, helper_source_sha256=self.helper_hashes,
            capture_and_warmup_s=sum(e['capture_and_warmup_s'] for e in self.events), events=self.events,
            scope='Original HF helpers; borrowed per-step output state until lease close. Timed staging. Unique inputs/outputs exclude graph intermediates and private pool slack; allocator measurements include them.')
