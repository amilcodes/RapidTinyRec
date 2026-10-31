"""CUDA-only primitive capture adapter; no executable experiment entry point.

Replay stages inputs before the graph, synchronizes, and copies outputs for
retention. Hashes and metadata here are validation overhead, never a timing claim.
"""
import copy
import torch
from compare_intervention import two_site,_shifted_copy,_hash
from owned_blocks import _metadata,_disjoint


def require_cuda(x):
    if (type(x) is not torch.Tensor or x.device.type!='cuda' or x.device.index!=0
            or x.dtype!=torch.float32 or x.ndim!=2 or x.shape[0]!=100
            or not x.is_contiguous() or x.requires_grad or x.storage_offset() not in range(4)):
        raise ValueError('Explicit cuda:0 shifted FP32 input required')
    if torch.cuda.current_device()!=0:raise ValueError('Current device must match cuda:0 input')
    if torch.cuda.is_current_stream_capturing():raise ValueError('Nested capture unsupported')


class CapturedTwoSite:
    def __init__(self,sample):
        require_cuda(sample)
        self.offset=sample.storage_offset()
        self.shape=tuple(sample.shape)
        self.device=sample.device
        self.static_input=_shifted_copy(sample,self.offset)
        self.observations=[]
        self.receipts=[]
        current=torch.cuda.current_stream(self.device)
        warm=torch.cuda.Stream(device=self.device)
        warm.wait_stream(current)
        with torch.cuda.stream(warm):
            for _ in range(3):
                outputs=two_site(self.static_input)
                del outputs
        current.wait_stream(warm)
        torch.cuda.synchronize(self.device)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_outputs=two_site(self.static_input,self.observations.append)
        torch.cuda.synchronize(self.device)
        if len(self.observations)!=4:raise ValueError('Capture lacks two-site ownership metadata')
        self.bindings=self._bindings()
        live=[self.bindings['input'],*self.bindings['outputs']]
        if not all(_disjoint(a,b) for i,a in enumerate(live) for b in live[i+1:]):
            raise ValueError('Static input/output storage alias')

    def _bindings(self):
        return dict(input=_metadata(self.static_input),outputs=[_metadata(t) for t in self.static_outputs])

    def __call__(self,x,metadata_sink=None):
        require_cuda(x)
        if tuple(x.shape)!=self.shape or x.device!=self.device or x.storage_offset()!=self.offset:
            raise ValueError('Replay geometry/device/shift differs')
        if self._bindings()!=self.bindings:raise ValueError('Graph bindings changed before replay')
        self.static_input.copy_(x)
        self.graph.replay()
        torch.cuda.synchronize(self.device)
        if self._bindings()!=self.bindings:raise ValueError('Graph bindings changed after replay')
        staged=_hash(self.static_input);source=_hash(x)
        if staged!=source:raise ValueError('Replay input staging differs')
        retained=tuple(t.clone() for t in self.static_outputs)
        torch.cuda.synchronize(self.device)
        live=[_metadata(x),self.bindings['input'],*self.bindings['outputs'],*[_metadata(t) for t in retained]]
        if not all(_disjoint(a,b) for i,a in enumerate(live) for b in live[i+1:]):
            raise ValueError('Retained output/input graph storage alias')
        self.receipts.append(dict(replay=len(self.receipts),source_sha256=source,staged_sha256=staged,
                                  source=_metadata(x),bindings=copy.deepcopy(self.bindings),
                                  retained_outputs=[_metadata(t) for t in retained]))
        if metadata_sink is not None:
            for record in self.observations:metadata_sink(copy.deepcopy(record))
        return retained
