"""GPU-resident lookup for MiniOneRec's three SID tokens, newline, and EOS.

The catalog is compiled once, outside request timing. Runtime lookup uses only
device tensors: no .item(), .tolist(), CPU copies, or device tensor creation
from Python lists. The Python step counter is independent of token values.
Full-vocabulary score normalization and newline/EOS scoring are unchanged.
Unknown prefixes retain the EOS fallback; no synchronous Python warning is
emitted for them. This schema is checked against the catalog at construction.
"""
import collections
import math

import torch
from transformers import LogitsProcessor


class GpuSidConstraintTables:
    def __init__(self, constraints, root_prefix, newline_id, eos_id, device):
        root_prefix = tuple(root_prefix)
        assert len(root_prefix) == 3
        sids = sorted(key for key in constraints if len(key) == 3 and key != root_prefix)
        assert sids and all(len(sid) == 3 for sid in sids)
        canonical = collections.defaultdict(set)
        for sid in sids:
            canonical[root_prefix].add(sid[0])
            canonical[sid[:1]].add(sid[1])
            canonical[sid[:2]].add(sid[2])
            canonical[sid].add(newline_id)
            canonical[sid + (newline_id,)].add(eos_id)
        assert dict(canonical) == {k: set(v) for k, v in constraints.items()}, 'Unexpected SID grammar'
        offsets = [min(sid[i] for sid in sids) for i in range(3)]
        sizes = [max(sid[i] for sid in sids)-offsets[i]+1 for i in range(3)]
        assert math.prod(sizes) <= 100_000_000, 'Dense SID-validity table too large for this implementation'
        self.sizes_python = sizes
        self.offsets = torch.tensor(offsets, dtype=torch.long, device=device)
        self.sizes = torch.tensor(sizes, dtype=torch.long, device=device)
        self.strides = {n: torch.tensor([math.prod(sizes[j+1:n]) for j in range(n)],
                                        dtype=torch.long, device=device) for n in (1, 2, 3)}
        self.root_prefix = torch.tensor(root_prefix, dtype=torch.long, device=device)
        self.first_tokens = torch.tensor(sorted(canonical[root_prefix]), dtype=torch.long, device=device)
        self.eos = torch.tensor(eos_id, dtype=torch.long, device=device)
        self.newline = torch.tensor(newline_id, dtype=torch.long, device=device)
        self.tables = {}
        for n in (1, 2):
            width = max(len(v) for k, v in canonical.items() if len(k) == n)
            table = torch.full((math.prod(sizes[:n]), width), eos_id, dtype=torch.long)
            for key, allowed in canonical.items():
                if len(key) != n:
                    continue
                index = sum((key[j]-offsets[j])*math.prod(sizes[j+1:n]) for j in range(n))
                tokens = sorted(allowed)
                # Duplicate a valid token for padding. All scatter writes are 0,
                # so duplicates cannot change which vocabulary entries are valid.
                table[index] = torch.tensor(tokens + [tokens[0]]*(width-len(tokens)))
            self.tables[n] = table.to(device)
        valid = torch.zeros(math.prod(sizes), dtype=torch.bool)
        for sid in sids:
            index = sum((sid[j]-offsets[j])*math.prod(sizes[j+1:]) for j in range(3))
            valid[index] = True
        self.valid_sids = valid.to(device)
        tensors = [self.offsets, self.sizes, self.root_prefix, self.first_tokens,
                   self.eos, self.newline, self.valid_sids, *self.strides.values(), *self.tables.values()]
        self.metadata = {'unique_sids': len(sids), 'sid_offsets': offsets, 'sid_range_sizes': sizes,
            'device_table_bytes': sum(t.numel()*t.element_size() for t in tensors),
            'table_shapes': {str(n): list(t.shape) for n, t in self.tables.items()},
            'unknown_prefix_behavior': 'EOS fallback on device; Python warning omitted',
            'runtime_cpu_token_transfers': 0}

    def _index(self, tokens, n):
        digits = tokens - self.offsets[:n]
        in_range = ((digits >= 0) & (digits < self.sizes[:n])).all(dim=-1)
        index = (digits * self.strides[n]).sum(dim=-1)
        return index.clamp(0, math.prod(self.sizes_python[:n])-1), in_range

    def lookup(self, step, input_ids):
        if step in (1, 2):
            index, in_range = self._index(input_ids[:, -step:], step)
            return torch.where(in_range[:, None], self.tables[step][index], self.eos)
        if step in (0, 3):
            suffix = input_ids[:, -3:]
            index, in_range = self._index(suffix, 3)
            terminal = torch.where(in_range & self.valid_sids[index], self.newline, self.eos)
            is_root = (suffix == self.root_prefix).all(dim=-1)
            return torch.where(is_root[:, None], self.first_tokens, terminal[:, None])
        # Every length-four catalog prefix permits EOS, and every unknown
        # prefix also falls back to EOS. The same holds at later steps.
        return self.eos.expand(input_ids.shape[0], 1)


class GpuSidConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, tables):
        self.tables = tables
        self.count = 0

    def __call__(self, input_ids, scores):
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, float('-inf'))
        columns = self.tables.lookup(self.count, input_ids)
        mask.scatter_(1, columns, 0.0)
        self.count += 1
        return scores + mask
