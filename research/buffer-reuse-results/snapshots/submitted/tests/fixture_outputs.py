import copy

import unittest

def request(user, slot, batch, pointers):
    span = 50 * 7 * 4
    return dict(full_score_hashes=[dict(shape=[50,7],dtype='torch.float32',sha256='a'*64) for _ in range(5)],cohort_index=user, request_slot=slot, original_batch_requests=batch, original_beam_row_offset=50 * slot, token_ids=[[1, 2, 3, 4, 5] for _ in range(50)], sequence_logp=[-15.0] * 50, transition_logp=[[-1.0, -2.0, -3.0, -4.0, -5.0] for _ in range(50)], beam_indices=[[0, 1, 2, 3, 4] for _ in range(50)], global_beam_indices=[[x + 50 * slot for x in range(5)] for _ in range(50)], saved_score_storage_ptrs=pointers, graph_logit_storage_ptrs=[999], saved_score_dtypes=['torch.float32'] * 5, score_row_extents=[dict(storage_ptr=p, byte_start=slot * span, byte_end=(slot + 1) * span, storage_bytes=batch * span) for p in pointers])
