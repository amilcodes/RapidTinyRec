"""Extract native HF request outputs without corrupting global beam ancestry.

Transitions are computed on the complete original batch before splitting.
Global ancestry and disjoint score row extents are retained alongside the
explicit local ancestry representation for per-user comparisons.
"""
import torch


def unpack_requests(model,output,users,prompt_tokens,graph_logit_storage_ptrs=()):
    count=len(users);beams=50;rows=count*beams
    if count not in (1,2) or len(set(users))!=count:raise ValueError('One or two distinct users required')
    if (output.sequences.ndim!=2 or output.sequences.shape!=(rows,prompt_tokens+5)
            or output.sequences_scores.shape!=(rows,) or output.beam_indices.shape!=(rows,5)
            or len(output.scores)!=5):
        raise ValueError('Original beam50/five-transition output shapes required')
    if (output.sequences_scores.dtype!=torch.float32 or any(s.dtype!=torch.float32 or s.ndim!=2
            or s.shape[0]!=rows or not s.is_contiguous() for s in output.scores)):
        raise ValueError('Original contiguous FP32 HF scores required')
    if len({s.shape[1] for s in output.scores})!=1:raise ValueError('Vocabulary changed between score steps')
    pointers=[s.untyped_storage().data_ptr() for s in output.scores]
    graph_ptrs=list(graph_logit_storage_ptrs)
    if len(set(pointers))!=5 or set(pointers)&set(graph_ptrs):
        raise ValueError('HF score tensors must be independent from each other and graph logits')
    # Validate ancestry BEFORE using it to gather score rows.
    for slot in range(count):
        ancestry=output.beam_indices[slot*beams:(slot+1)*beams]
        if ancestry.dtype not in (torch.int32,torch.int64) or not bool(
                ((ancestry==-1)|((ancestry>=slot*beams)&(ancestry<(slot+1)*beams))).all()):
            raise ValueError('Output ancestry crosses request boundary')
    transitions=model.compute_transition_scores(output.sequences,output.scores,output.beam_indices,normalize_logits=False)
    if transitions.shape!=(rows,5) or transitions.dtype!=torch.float32:
        raise ValueError('Original five FP32 transition scores required')
    if not bool(torch.isfinite(transitions).all()) or not bool(torch.isfinite(output.sequences_scores).all()):
        raise ValueError('Nonfinite final sequence or transition scores')
    result=[]
    for slot,user in enumerate(users):
        lo=slot*beams;hi=lo+beams;ancestry=output.beam_indices[lo:hi]
        local=torch.where(ancestry<0,ancestry,ancestry-lo)
        extents=[]
        for score in output.scores:
            begin=score.storage_offset()*score.element_size()+lo*score.shape[1]*score.element_size()
            end=begin+beams*score.shape[1]*score.element_size()
            if end>score.untyped_storage().nbytes():raise ValueError('Score row extent exceeds storage')
            extents.append(dict(storage_ptr=score.untyped_storage().data_ptr(),byte_start=begin,byte_end=end,
                                storage_bytes=score.untyped_storage().nbytes()))
        result.append(dict(cohort_index=user,request_slot=slot,original_batch_requests=count,
            original_beam_row_offset=lo,score_row_extents=extents,
            global_beam_indices=ancestry.cpu().tolist(),beam_indices=local.cpu().tolist(),
            token_ids=output.sequences[lo:hi,prompt_tokens:].cpu().tolist(),
            sequence_logp=output.sequences_scores[lo:hi].cpu().tolist(),transition_logp=transitions[lo:hi].cpu().tolist(),
            saved_score_storage_ptrs=pointers,saved_score_dtypes=[str(s.dtype) for s in output.scores],
            graph_logit_storage_ptrs=sorted(set(graph_ptrs))))
    return result
