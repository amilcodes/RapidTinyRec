"""Setup-only source identity observer around unchanged original capture body."""
import sys
import contextlib
import torch
from .prefix_replay import PrefixReplayPool
from .qkv_capture_binding import observe_generated_capture,attest_observed
from .qkv_storage import tensor_record
from .qkv_load_observer import LoadObserver


class BoundPrefixReplayPool(PrefixReplayPool):
    def __init__(self,*args,cache_roots,**kwargs):
        super().__init__(*args,**kwargs);self.cache_roots=cache_roots;self.capture_bindings=[];self.setup_observation=None

    @contextlib.contextmanager
    def observe_setup(self,output_root):
        if self.setup_observation is not None:raise ValueError('Nested setup observer')
        loads=LoadObserver(torch.cuda.is_current_stream_capturing,output_root/'loaded-binaries')
        with observe_generated_capture(self.cache_roots,torch.cuda.is_current_stream_capturing,loads) as observed:
            self.setup_observation=(observed,loads)
            try:yield
            finally:self.setup_observation=None
        if loads.active:raise ValueError('Incomplete setup kernel load')

    def prepare(self,kwargs):
        key=self.signature(kwargs)
        if key in self.entries:return super().prepare(kwargs)
        if self.setup_observation is None:raise ValueError('Capture requires setup-wide observer before direct warmup')
        observed,loads=self.setup_observation;start=len(observed)
        super().prepare(kwargs)
        buffers,graph,output=self.entries[key]
        calls=[loads.bind(call) for call in attest_observed(observed[start:],self.cache_roots)]
        self.capture_bindings.append(dict(cache_tokens=output.cache_length()-1,graph_object_id=id(graph),
            tokens={k:tensor_record(x) for k,x in buffers.tensors.items()},logits=tensor_record(output.logits),
            wrappers=calls))

    def capture_report(self):
        if sys.getprofile() is not None:raise ValueError('Profiler remains installed after setup')
        if not self.frozen or len(self.capture_bindings)!=len(self.entries):raise ValueError('Complete frozen graph bindings required')
        current={id(graph):(buffers,output) for buffers,graph,output in self.entries.values()}
        for binding in self.capture_bindings:
            if binding['graph_object_id'] not in current:raise ValueError('Captured graph replaced')
            buffers,output=current[binding['graph_object_id']]
            if binding['tokens']!={k:tensor_record(x) for k,x in buffers.tensors.items()} or binding['logits']!=tensor_record(output.logits):raise ValueError('Graph binding storage changed')
        return dict(schema='packed-qkv-capture-bindings-v1',profiling_scope='setup_before_first_compiled_warmup',measurement_profiler_installed=False,
                    graphs=self.capture_bindings,compiled_split_view_verified=False)
