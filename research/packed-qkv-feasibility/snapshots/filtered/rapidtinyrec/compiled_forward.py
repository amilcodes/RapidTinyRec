"""Instrumented TorchInductor forward compilation; eager generation/cache policy.

Backend timings include lowering/code generation but exclude Dynamo frontend
tracing. Cold warmup wall time and PyTorch's nested timing metrics are recorded
separately. Any compilation after freeze() is an error, never a timed sample.
"""
import collections
import contextlib
import inspect
import json
import time

import torch
import torch._dynamo
import torch._dynamo.utils
from torch._dynamo.exc import TensorifyScalarRestartAnalysis
import torch._inductor
import torch._inductor.metrics


SETTINGS = {'backend': 'inductor', 'dynamic': True, 'fullgraph': False,
            'triton.cudagraphs': False, 'recompile_limit': 64,
            'suppress_errors': False, 'fail_on_recompile_limit_hit': True,
            'scope': 'Model forward only; original DynamicCache, shared prefill, GPU constraints, HF beam search.'}


class CompiledForward:
    def __init__(self, model, outdir, announce):
        self.model = model
        self.eager = model.forward
        self.outdir = outdir
        self.announce = announce
        self.events = []
        self.phase = 'warmup'
        self.warmups = []
        self.setup_started = time.perf_counter()
        (outdir/'compiled_graphs').mkdir(exist_ok=True)
        torch._dynamo.config.recompile_limit = SETTINGS['recompile_limit']
        torch._dynamo.config.suppress_errors = False
        torch._dynamo.config.fail_on_recompile_limit_hit = True
        started = time.perf_counter()
        self.compiled = torch.compile(self.eager, backend=self._backend,
                                      dynamic=True, fullgraph=False)
        self.wrapper_creation_s = time.perf_counter()-started
        assert 'logits_to_keep' in inspect.signature(self.compiled).parameters

    def _backend(self, gm, example_inputs):
        if self.phase != 'warmup':
            raise RuntimeError('Compilation during measurement is forbidden')
        index = len(self.events)
        event = {'index': index, 'fx_nodes': len(list(gm.graph.nodes)),
            'ops': dict(collections.Counter(str(n.target) for n in gm.graph.nodes if n.op == 'call_function')),
            'input_shapes': [([str(d) for d in x.shape] if isinstance(x, torch.Tensor) else str(type(x)))
                             for x in example_inputs]}
        self.events.append(event)
        (self.outdir/'compiled_graphs'/f'fx_{index:03d}.py').write_text(gm.code)
        self.announce({'stage': 'compilation_started', 'graph': index, 'fx_nodes': event['fx_nodes']})
        started = time.perf_counter()
        kernels_before = torch._inductor.metrics.generated_kernel_count
        try:
            result = torch._inductor.compile(gm, example_inputs, options={'triton.cudagraphs': False})
            event['success'] = True
            return result
        except Exception as exc:
            event['success'] = False
            event['error'] = repr(exc)
            # Dynamo deliberately restarts tracing after scalar tensorification.
            # Re-raise so Dynamo performs that retry; never turn errors into eager fallback.
            event['expected_restart'] = isinstance(exc, TensorifyScalarRestartAnalysis)
            raise
        finally:
            event['backend_compile_s'] = time.perf_counter()-started
            event['generated_kernels'] = torch._inductor.metrics.generated_kernel_count-kernels_before
            self.write_report()
            self.announce({'stage': 'compilation_finished', 'graph': index,
                           'seconds': event['backend_compile_s'], 'success': event['success']})

    @contextlib.contextmanager
    def select(self, compiled):
        original = self.model.forward
        self.model.forward = self.compiled if compiled else self.eager
        try:
            yield
        finally:
            self.model.forward = original

    def add_warmup(self, record):
        self.warmups.append(record)
        self.write_report()

    def freeze(self):
        assert self.events and self.events[-1].get('success'), 'No completed compiled graph'
        assert all(e.get('success') or e.get('expected_restart') for e in self.events), 'Unexpected compiler error'
        assert not torch._dynamo.utils.counters.get('graph_break'), 'Forward graph breaks detected'
        assert not torch._dynamo.utils.counters.get('unimplemented'), 'Uncompiled forward paths detected'
        self.setup_wall_s = time.perf_counter()-self.setup_started
        self.frozen_event_count = len(self.events)
        self.phase = 'measurement'
        torch._dynamo.config.error_on_recompile = True
        self.write_report()

    def finish(self):
        assert len(self.events) == self.frozen_event_count
        self.phase = 'complete'
        self.write_report()

    def write_report(self):
        report = {'settings': SETTINGS, 'phase': self.phase,
            'wrapper_creation_s': self.wrapper_creation_s,
            'setup_wall_s': getattr(self, 'setup_wall_s', None),
            'backend_compile_total_s': sum(e.get('backend_compile_s', 0) for e in self.events),
            'backend_graphs': len(self.events), 'events': self.events, 'warmups': self.warmups,
            'successful_graphs': sum(bool(e.get('success')) for e in self.events),
            'expected_restarts': sum(bool(e.get('expected_restart')) for e in self.events),
            'compiled_warmup_total_s': sum(w['elapsed_s'] for w in self.warmups if w['variant'] == 'shared'),
            'eager_warmup_total_s': sum(w['elapsed_s'] for w in self.warmups if w['variant'] == 'baseline'),
            'measurement_compilations': len(self.events)-self.frozen_event_count
                if hasattr(self, 'frozen_event_count') else None,
            'compiler_counters': {str(k): {str(a): b for a,b in v.items()}
                                  for k,v in torch._dynamo.utils.counters.items()},
            'compiler_time_metrics_s': dict(torch._dynamo.utils.compilation_time_metrics),
            'generated_kernel_count': torch._inductor.metrics.generated_kernel_count,
            'timing_note': 'Cold compiled warmups include compilation, tracing, and inference. Backend totals exclude frontend tracing. Compiler timing categories are nested; do not sum them. All warmups and compilation are excluded from the 500-user latency statistics.'}
        temp = self.outdir/'compilation.json.tmp'
        temp.write_text(json.dumps(report, indent=2, allow_nan=False))
        temp.replace(self.outdir/'compilation.json')
        (self.outdir/'compiler_times.txt').write_text(torch._dynamo.utils.compile_times(aggregate=True))
