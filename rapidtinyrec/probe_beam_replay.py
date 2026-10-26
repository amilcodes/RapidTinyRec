"""Allocated-GPU probes of original helper arithmetic, ties and retained state."""
import gzip
import hashlib
import json

import torch
from transformers.generation.utils import GenerationMixin
from .beam_replay import BeamReplayPool, RUNNING, FINISHED

SCENARIOS = ('equal', 'kth_ties', 'signed_zero', 'near_ties', 'infinities', 'finished_flags', 'penalty')


def compare_tensors(left, right):
    assert left.shape == right.shape and left.dtype == right.dtype
    a, b = left.contiguous(), right.contiguous()
    floating = a.dtype == torch.float32
    av, bv = (a.view(torch.int32), b.view(torch.int32)) if floating else (a, b)
    finite = torch.isfinite(a) & torch.isfinite(b) if floating else torch.ones_like(a, dtype=torch.bool)
    delta = torch.where(finite, (a.to(torch.float64)-b.to(torch.float64)).abs(), 0.)
    # Hashes and scans are diagnostics outside generation timing.
    return dict(values=a.numel(), shape=list(a.shape), dtype=str(a.dtype),
        matching_bits=int((av == bv).sum()), max_abs_finite=float(delta.max()),
        finite_mask_equal=bool(torch.equal(torch.isfinite(a), torch.isfinite(b))) if floating else True,
        no_nan=bool(not torch.isnan(a).any() and not torch.isnan(b).any()) if floating else True,
        left_sha256=hashlib.sha256(a.cpu().numpy().tobytes()).hexdigest(),
        right_sha256=hashlib.sha256(b.cpu().numpy().tobytes()).hexdigest())


def inputs(prompt, step, scenario, request, device):
    beams, count = 50, 100
    i = torch.arange(count, device=device)
    scores = -(i // 7).to(torch.float32).reshape(1, count)
    if scenario == 'equal': scores.fill_(-1.)
    elif scenario == 'kth_ties': scores = -(i // 60).to(torch.float32).reshape(1, count)
    elif scenario == 'signed_zero': scores = torch.where(i % 2 == 0, 0., -0.).reshape(1, count)
    elif scenario == 'near_ties': scores = (-1. + i.to(torch.float32) * torch.finfo(torch.float32).eps).reshape(1, count)
    elif scenario == 'infinities':
        scores[0, :3] = float('inf'); scores[0, -3:] = float('-inf')
    if scenario != 'signed_zero': scores = scores + request / 16 + step / 32
    # Distinct sequence identities expose top-k tie index changes. Ancestry also has duplicates.
    sequences = (torch.arange(count*(prompt+8), dtype=torch.int64, device=device)
                 .reshape(1, count, prompt+8) + request * 100000 + step * 10000)
    indices = (torch.arange(count*8, dtype=torch.int32, device=device).reshape(1, count, 8) + request) % 17
    stopped = torch.zeros((1, count), dtype=torch.bool, device=device)
    if scenario in ('finished_flags', 'penalty', 'infinities'): stopped[0] = (i+request+step) % 3 == 0
    if step == 4: stopped.fill_(True)
    running = dict(topk_log_probs=scores, topk_running_sequences=sequences,
        topk_running_beam_indices=indices, next_token_hits_stopping_criteria=stopped, num_beams=beams)
    finished = dict(sequences=sequences[:, :beams].clone(), topk_running_sequences=sequences,
        beam_scores=torch.full((1, beams), -1e9, device=device), topk_log_probs=scores,
        beam_indices=indices[:, :beams].clone(), topk_running_beam_indices=indices,
        is_early_stop_heuristic_unsatisfied=torch.tensor([[request % 2 == 0]], device=device),
        is_sent_finished=((torch.arange(beams, device=device)+request) % 3 == 0).reshape(1, beams),
        next_token_hits_stopping_criteria=stopped, top_num_beam_mask=(i < beams),
        num_beams=beams, cur_len=prompt+step, decoder_prompt_len=prompt,
        length_penalty=1.2 if scenario == 'penalty' else 0., early_stopping=scenario == 'finished_flags')
    return running, finished


@torch.inference_mode()
def probe(output_path):
    rows, retained_rows, reports = [], [], []
    with gzip.open(output_path, 'wt') as raw:
        for scenario in SCENARIOS:
            model = GenerationMixin()
            pool = BeamReplayPool(model, max_graphs=20).install()
            for request, prompt in enumerate((76, 121, 121, 76)):
                lease = pool.request(prompt); held = []; previous = None
                for step in range(5):
                    running, finished = inputs(prompt, step, scenario, request, 'cuda')
                    if previous is not None:
                        for name, value in zip(('sequences','beam_scores','beam_indices','is_sent_finished'), previous):
                            finished[name] = value
                    for method, arguments in ((RUNNING, running), (FINISHED, finished)):
                        expected = pool.originals[method](**arguments)
                        repeated = pool.originals[method](**arguments)
                        actual = getattr(model, method)(**arguments)
                        detail = dict(scenario=scenario, request=request, prompt_tokens=prompt, step=step, method=method,
                            comparisons=[compare_tensors(a, b) for a, b in zip(expected, actual)],
                            direct_repeat_comparisons=[compare_tensors(a, b) for a, b in zip(expected,repeated)])
                        rows.append(detail)
                        raw.write(json.dumps(dict(**detail, direct=[t.cpu().tolist() for t in expected],
                            graph=[t.cpu().tolist() for t in actual],
                            direct_repeat=[t.cpu().tolist() for t in repeated]), allow_nan=True)+'\n')
                        held.append((detail, actual, tuple(t.clone() for t in actual)))
                        if method == FINISHED: previous = actual
                identities = [t.untyped_storage().data_ptr() for _, output, _ in held for t in output]
                independent = len(identities) == len(set(identities)) == 35
                for detail, actual, saved in held:
                    comparison = dict(scenario=scenario, request=request, prompt_tokens=prompt,
                        step=detail['step'], method=detail['method'], independent_step_outputs=independent,
                        comparisons=[compare_tensors(a, b) for a, b in zip(saved, actual)])
                    retained_rows.append(comparison)
                    # Always preserve retained comparisons; full changed values on any failure.
                    if any(c['matching_bits'] != c['values'] for c in comparison['comparisons']):
                        raw.write(json.dumps(dict(retained=comparison, graph=[t.cpu().tolist() for t in actual]), allow_nan=True)+'\n')
                lease.close()
                if request == 1: pool.freeze([76, 121])
            reports.append(dict(scenario=scenario, replay=pool.report()))
            pool.uninstall()
            del pool, lease, model, held, previous, actual, expected, repeated, saved, running, finished, arguments
    return dict(complete=True, scenarios=list(SCENARIOS), graph_direct=rows,
        retained_after_five_steps=retained_rows, reports=reports,
        raw_outputs_file=output_path.name,
        raw_outputs_sha256=hashlib.sha256(output_path.read_bytes()).hexdigest(),
        scope='Seven separate synthetic helper pools, four mixed-length requests each, five steps. Original pinned helpers, 100/150 candidate selections, exact float bits and integer indices. Probe capture/serialization is setup only; measured generation uses a distinct pool.')
