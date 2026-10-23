"""Catalog-wide device-lookup checks, excluded from request timing."""
import itertools
import random
import warnings

import torch

from .bulk_constraints import BulkPrefixConstrainedLogitsProcessor
from .gpu_constraints import GpuSidConstrainedLogitsProcessor


@torch.inference_mode()
def validate(tables, constraints, root_prefix, eos_id, vocab_size, device):
    rng = random.Random(42)
    offsets = tables.metadata['sid_offsets']
    sizes = tables.metadata['sid_range_sizes']
    report = {'lookup_cases_by_step': {}, 'full_score_matrix_comparisons': 0}
    cases_by_step = {}
    for step in range(6):
        length = 3 if step == 0 else step
        cases = [key for key in constraints if len(key) == length]
        if step == 2:
            cases += list(itertools.product(range(offsets[0], offsets[0]+sizes[0]),
                                            range(offsets[1], offsets[1]+sizes[1])))
        # Out-of-range tokens, absent combinations of valid SID token types,
        # wrong marker, and EOS fallback after the grammar's final step.
        cases += [tuple(rng.randrange(vocab_size) for _ in range(length)) for _ in range(1000)]
        if length == 3:
            cases += [tuple(offsets[j]+rng.randrange(sizes[j]) for j in range(3)) for _ in range(1000)]
        cases += [(0,)*length, (vocab_size-1,)*length, (-1,)*length]
        cases = list(dict.fromkeys(cases))
        cases_by_step[step] = cases
        for start in range(0, len(cases), 512):
            chunk = cases[start:start+512]
            x = torch.tensor(chunk, dtype=torch.long, device=device)
            actual = tables.lookup(step, x).cpu().tolist()
            for prefix, got in zip(chunk, actual):
                expected = set(constraints.get(prefix, [eos_id]))
                assert set(got) == expected, (step, prefix, got, expected)
        report['lookup_cases_by_step'][str(step)] = len(cases)

    generator = torch.Generator(device=device).manual_seed(142)
    scores = torch.randn((50, vocab_size), device=device, dtype=torch.float32, generator=generator)
    for step, cases in cases_by_step.items():
        # Exercise valid and fallback rows in the same mask, including root at
        # depth three and repeated padding-token writes from the device table.
        chosen = cases[:20] + cases[-20:]
        if step in (0, 3):
            chosen.append(tuple(root_prefix))
        while len(chosen) < 50:
            chosen.append(cases[rng.randrange(len(cases))])
        x = torch.tensor(chosen[:50], dtype=torch.long, device=device)
        reference = BulkPrefixConstrainedLogitsProcessor(
            lambda batch, ids: constraints.get(tuple(ids), []), 50, base_model='Qwen2', eos_token_id=eos_id)
        reference.count = step
        candidate = GpuSidConstrainedLogitsProcessor(tables)
        candidate.count = step
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            expected = reference(x, scores)
        actual = candidate(x, scores)
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32)), ('score bits', step)
        report['full_score_matrix_comparisons'] += 1
    torch.cuda.synchronize()
    report['passed'] = True
    report['total_lookup_cases'] = sum(report['lookup_cases_by_step'].values())
    report['score_values_compared_bitwise'] = report['full_score_matrix_comparisons']*scores.numel()
    return report
