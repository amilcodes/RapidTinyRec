"""Prepared folded model adapter for LaneServer's owner-thread callbacks.

Model allocation/warmup orchestration, global patch lifetime, cross-lane audits
and the serving harness are still separate requirements. This adapter does not
authorize measurement merely because its local freeze has succeeded.
"""
import time

from .serving_generation import make_generation


def materialize(model, output, user):
    """Independent Python raw evidence; transition extraction stays untimed."""
    import torch
    if len(output.scores) != 5:
        raise ValueError('All five original HF score tensors are required')
    sequence = output.sequences_scores
    transitions = model.compute_transition_scores(
        output.sequences, output.scores, output.beam_indices, normalize_logits=False)
    if sequence.dtype != torch.float32 or transitions.dtype != torch.float32:
        raise ValueError('Exact serving requires FP32 sequence and transition scores')
    return dict(token_ids=output.sequences[:, user['prompt_tokens']:].cpu().tolist(),
                sequence_logp=sequence.cpu().tolist(),
                sequence_bits=sequence.contiguous().view(torch.int32).cpu().tolist(),
                transition_logp=transitions.cpu().tolist(),
                transition_bits=transitions.contiguous().view(torch.int32).cpu().tolist(),
                beam_indices=output.beam_indices.cpu().tolist())


class PreparedFoldedLane:
    """Bind already-owned components, created on this context's owner stream.

    Call warm() only during serialized setup. Freeze every lane only after all
    lanes finish setup. Retained warm outputs must be audited and released before
    run(); those audits and global ownership gates belong to the harness.
    """
    def __init__(self, context, model, compiler, arena, pool, direct, tok,
                 tables, inputs, masks):
        context.identity()  # Reject cross-thread construction.
        if pool.arena is not arena or direct.pool is not pool:
            raise ValueError('Lane components must share exactly their own replay arena')
        self.context = context
        self.model, self.compiler, self.arena = model, compiler, arena
        self.pool, self.direct, self.tables = pool, direct, tables
        self.inputs, self.masks = inputs, masks
        self.generate = make_generation(model, compiler, arena, pool, direct,
                                        tok, tables, inputs, masks)
        self.frozen = False
        self.retained = {}
        self.closed = False

    def warm(self, user, *, replay, retain=False):
        def action():
            if self.frozen or self.closed:
                raise RuntimeError('Warmup after freeze/close is forbidden')
            key = user['cohort_index']
            if retain and key in self.retained:
                raise ValueError('Do not overwrite retained evidence')
            output = self.generate(user, replay, replay)
            raw = materialize(self.model, output, user)
            if retain:
                self.retained[key] = output
            return raw
        return self.context.call(action)

    def freeze(self):
        def action():
            if self.frozen or self.closed:
                raise RuntimeError('Lane already frozen/closed')
            self.pool.freeze()
            self.compiler.freeze()
            self.graph_count = len(self.pool.entries)
            self.compile_count = len(self.compiler.events)
            self.frozen = True
        return self.context.call(action)

    def scan_retained(self, users):
        """Re-extract retained warm outputs after mixed requests, outside timing."""
        def action():
            if self.closed:
                raise RuntimeError('Cannot scan closed lane')
            by_id = {user['cohort_index']: user for user in users}
            if len(by_id) != len(users) or not self.retained or not set(self.retained) <= set(by_id):
                raise ValueError('Retained outputs need unique matching cohort rows')
            return {str(key): materialize(self.model, output, by_id[key])
                    for key, output in self.retained.items()}
        return self.context.call(action)

    def release_retained(self):
        def action():
            if not self.frozen:
                raise RuntimeError('Retain warm evidence through setup and freeze')
            self.retained.clear()
        return self.context.call(action)

    def ownership_snapshot(self, expected_graphs):
        """Call with all lanes idle; hold returned references through comparison."""
        from .serving_graph_ownership import snapshot_graphs
        from .serving_inventory import enumerate_folded_lane
        from .serving_storage import describe_tensors
        identity = self.context.identity()
        def action():
            if not self.frozen or self.closed:
                raise RuntimeError('Ownership snapshot needs a live frozen lane')
            if set(self.inputs) != set(self.masks):
                raise ValueError('Prompt and attention-mask owner sets differ')
            prompts = {str(key): {'input_ids': value, 'attention_mask': self.masks[key]}
                       for key, value in self.inputs.items()}
            retained = {}
            for key, output in self.retained.items():
                if len(output.scores) != 5:
                    raise ValueError('Retained output must include five HF score storages')
                retained[str(key)] = dict(sequences=output.sequences,
                    sequence_scores=output.sequences_scores, beam_indices=output.beam_indices,
                    **{f'score_{i}': score for i, score in enumerate(output.scores)})
            tensors = enumerate_folded_lane(self.model, self.tables, self.arena,
                                            self.pool, prompts, retained)
            graphs, graph_refs = snapshot_graphs(self.pool, expected_graphs)
            return dict(record=dict(identity=identity, graphs=graphs,
                                    storage=describe_tensors(tensors)),
                        tensor_refs=tensors, graph_refs=graph_refs)
        return self.context.call(action)

    def run(self, user):
        def action():
            if not self.frozen or self.closed or self.retained:
                raise RuntimeError('Freeze and audit/release retained outputs before timing')
            before_replays = self.pool.replays
            started = time.perf_counter_ns()
            output = self.generate(user, True, True)
            self.context.stream.synchronize()
            finished = time.perf_counter_ns()
            raw = materialize(self.model, output, user)
            if (len(self.pool.entries) != self.graph_count or
                    len(self.compiler.events) != self.compile_count or
                    not self.pool.frozen or self.compiler.phase != 'measurement'):
                raise RuntimeError('Frozen graph/compiler state changed')
            if self.pool.replays - before_replays != 4:
                raise RuntimeError('Expected four original decode graph replays')
            return dict(cohort_index=user['cohort_index'], prompt_tokens=user['prompt_tokens'],
                        generation_started_ns=started, generation_finished_ns=finished,
                        generation_elapsed_s=(finished-started)/1e9, raw=raw)
        # The scheduler's response timestamp follows CPU materialization and this
        # context's final stream drain. No device-wide synchronization is used.
        return self.context.call(action)

    def close(self):
        self.context.close()
        self.closed = True
