"""Packaged one-line allocation candidate; pinned released score objective."""
import hashlib
from pathlib import Path
import torch
from .gpu_constraints import GpuSidConstrainedLogitsProcessor
PIN='0a5e903b7251c067456377c7f09f569ccdbbf6f2c456b6a4b072d3a2f26f710c'
if hashlib.sha256(Path(__file__).with_name('gpu_constraints.py').read_bytes()).hexdigest()!=PIN:
    raise RuntimeError('Released constraint source differs')
class ReuseNormalizedConstraintProcessor(GpuSidConstrainedLogitsProcessor):
    def __call__(self,input_ids,scores):
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, float('-inf'))
        columns = self.tables.lookup(self.count, input_ids)
        mask.scatter_(1, columns, 0.0)
        self.count += 1
        return scores.add_(mask)
