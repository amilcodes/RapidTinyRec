"""Replay unchanged SID constraint arithmetic on one sequential CUDA stream.

One independent graph/output per scored step preserves output_scores through all
five steps. Outputs are borrowed until the request lease is closed. The caller
must finish transition-score extraction before closing it or starting a request.
Staging copies and replay are request work; preparation is explicit warmup work.
No compiler, fused reduction, restricted normalization or beam selector is used.
"""
import time

import torch
from transformers import LogitsProcessor


def constrain(tables, step, input_ids, scores):
    """Same ATen arithmetic and order as GpuSidConstrainedLogitsProcessor."""
    scores = torch.nn.functional.log_softmax(scores, dim=-1)
    mask = torch.full_like(scores, float('-inf'))
    columns = tables.lookup(step, input_ids)
    mask.scatter_(1, columns, 0.0)
    return scores + mask


def validate_inputs(input_ids, scores):
    if scores.ndim != 2 or scores.dtype != torch.float32 or not scores.is_contiguous():
        raise ValueError('Expected contiguous two-dimensional FP32 scores')
    if input_ids.ndim != 2 or input_ids.shape[0] != scores.shape[0] or input_ids.shape[1] < 3:
        raise ValueError('Expected matching beam rows and at least three token IDs')
    if input_ids.dtype != torch.long or input_ids.device != scores.device:
        raise ValueError('Token IDs must be int64 on the scores device')
    if scores.requires_grad:
        raise ValueError('Inference-only score buffers')
    return (tuple(scores.shape), scores.dtype, scores.device)


class ConstraintReplayBuffers:
    def __init__(self, input_ids, scores):
        self.signature = validate_inputs(input_ids, scores)
        self.scores = torch.empty_like(scores)
        # Only token suffixes are consulted; integer lookup semantics are exact.
        self.ids = torch.empty((scores.shape[0], 3), dtype=torch.long, device=scores.device)
        self.load(input_ids, scores)

    def load(self, input_ids, scores):
        if validate_inputs(input_ids, scores) != self.signature:
            raise ValueError('Constraint score shape, dtype or device changed')
        self.scores.copy_(scores)
        self.ids.copy_(input_ids[:, -3:])

    def tensors(self):
        return (self.scores, self.ids)


class ConstraintReplayLease(LogitsProcessor):
    def __init__(self, pool):
        self.pool = pool
        self.count = 0
        self.closed = False

    def __call__(self, input_ids, scores):
        if self.closed or self.pool.active is not self:
            raise RuntimeError('Request lease is no longer active')
        if self.count >= 5:
            raise RuntimeError('Only the five validated SID/newline/EOS steps are supported')
        result = self.pool.replay(self.count, input_ids, scores)
        self.count += 1
        return result

    def close(self):
        """Release only after all borrowed scores/transition values are consumed."""
        if self.closed or self.pool.active is not self:
            raise RuntimeError('Request lease already closed or superseded')
        self.closed = True
        self.pool.active = None


class ConstraintReplayPool:
    def __init__(self, tables):
        self.tables = tables
        self.entries = {}
        self.events = []
        self.frozen = False
        self.active = None
        self.stream_id = None
        self.replays = 0
        self.staged_bytes = 0
        self.requests = 0

    def processor(self):
        if self.active is not None:
            raise RuntimeError('Consume saved scores and close the previous request first')
        self.active = ConstraintReplayLease(self)
        self.requests += 1
        return self.active

    def _stream(self, scores):
        if scores.device.type != 'cuda' or scores.device != self.tables.eos.device:
            raise ValueError('Constraint capture requires CUDA scores on the table device')
        stream_id = torch.cuda.current_stream(scores.device).cuda_stream
        if self.stream_id is None:
            self.stream_id = stream_id
        elif self.stream_id != stream_id:
            raise RuntimeError('Constraint replay is bound to one sequential CUDA stream')

    @torch.inference_mode()
    def prepare(self, step, input_ids, scores):
        if self.frozen:
            raise RuntimeError('Constraint capture after freeze is forbidden')
        if step not in range(5):
            raise ValueError('Unsupported constraint step')
        self._stream(scores)
        signature = validate_inputs(input_ids, scores)
        if step in self.entries:
            if self.entries[step][0].signature != signature:
                raise ValueError('One score shape per step is supported')
            return
        torch.cuda.synchronize(scores.device)
        started = time.perf_counter()
        before = torch.cuda.memory_allocated(scores.device)
        buffers = ConstraintReplayBuffers(input_ids, scores)
        stream = torch.cuda.Stream(device=scores.device)
        stream.wait_stream(torch.cuda.current_stream(scores.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                warm = constrain(self.tables, step, buffers.ids, buffers.scores)
                del warm
        torch.cuda.current_stream(scores.device).wait_stream(stream)
        torch.cuda.synchronize(scores.device)
        graph = torch.cuda.CUDAGraph()
        # Each graph uses its own private pool; no output aliases another step.
        with torch.cuda.graph(graph, stream=stream):
            output = constrain(self.tables, step, buffers.ids, buffers.scores)
        torch.cuda.synchronize(scores.device)
        self.entries[step] = (buffers, graph, output)
        self.check_ownership()
        self.events.append(dict(step=step, score_shape=list(scores.shape),
            capture_and_warmup_s=time.perf_counter()-started,
            allocated_delta_bytes=torch.cuda.memory_allocated(scores.device)-before))

    def check_ownership(self):
        tensors = [t for buffers, _, output in self.entries.values()
                   for t in (*buffers.tensors(), output)]
        identities = [(t.device, t.untyped_storage().data_ptr()) for t in tensors]
        if len(set(identities)) != len(identities):
            raise AssertionError('Constraint input/output storage aliases across steps')

    def freeze(self):
        if set(self.entries) != set(range(5)):
            raise RuntimeError('All five constraint steps must be captured before freeze')
        if self.active is not None:
            raise RuntimeError('Close warmup request before freeze')
        self.check_ownership()
        self.frozen = True

    @torch.inference_mode()
    def replay(self, step, input_ids, scores):
        if self.active is None or self.active.count != step:
            raise RuntimeError('Use the active request processor in step order')
        self._stream(scores)
        if step not in self.entries:
            if self.frozen:
                raise RuntimeError('Uncaptured constraint step during measurement')
            self.prepare(step, input_ids, scores)
        buffers, graph, output = self.entries[step]
        buffers.load(input_ids, scores)
        graph.replay()
        self.replays += 1
        self.staged_bytes += sum(t.numel()*t.element_size() for t in buffers.tensors())
        return output

    def report(self):
        self.check_ownership()
        inputs = [t for b, _, _ in self.entries.values() for t in b.tensors()]
        outputs = [o for _, _, o in self.entries.values()]
        all_tensors = inputs + outputs
        unique = {(t.device,t.untyped_storage().data_ptr()):t.untyped_storage().nbytes()
                  for t in all_tensors}
        return dict(graphs=len(self.entries), frozen=self.frozen, replays=self.replays,
            requests=self.requests, staged_bytes=self.staged_bytes,
            input_logical_bytes=sum(t.numel()*t.element_size() for t in inputs),
            output_logical_bytes=sum(t.numel()*t.element_size() for t in outputs),
            unique_input_output_storage_bytes=sum(unique.values()),
            disjoint_step_storage=True, active_request=self.active is not None,
            capture_and_warmup_s=sum(e['capture_and_warmup_s'] for e in self.events),
            events=self.events,
            scope='Five independent constraint graphs; same full-vocabulary arithmetic. Staging copies are timed. Output lifetime ends when the request lease closes. Unique storage excludes graph intermediates/private pool slack; allocator measurements include them.')
