"""QKV request schedule, strict five-score serialization and retained ownership."""
import torch
from .qkv_inventory import digest

def schedule(users, half):
    if half not in (0,1) or len(users)!=4 or [u['cohort_index'] for u in users]!=[1,2,6,10]:
        raise ValueError('Declared four-user cohort required')
    if [u['prompt_tokens'] for u in users]!=[76,76,121,121]:
        raise ValueError('Declared prompt lengths required')
    for repeat in range(half*3,half*3+3):
        ordered=[users[1],users[0],users[3],users[2]] if repeat%2 else users
        for user in ordered:
            yield repeat,user

def unpack(model, output, user, graph_ptrs=(), supplied_storage_ptrs=()):
    scores=output.scores
    if len(scores)!=5 or output.sequences.shape!=(50,user['prompt_tokens']+5):
        raise ValueError('Native beam50 and five terminal-scored steps required')
    if output.sequences_scores.dtype!=torch.float32 or any(s.dtype!=torch.float32 or s.shape!=(50,model.config.vocab_size) for s in scores):
        raise ValueError('Original full-vocabulary FP32 scores required')
    pointers=[s.untyped_storage().data_ptr() for s in scores]
    forbidden=set(supplied_storage_ptrs)|set(graph_ptrs)|{t.untyped_storage().data_ptr() for t in model.parameters()}|{t.untyped_storage().data_ptr() for t in model.buffers()}
    if len(set(pointers))!=5 or set(pointers)&forbidden:
        raise ValueError('Score storage aliases graph/model/other step')
    return dict(cohort_index=user['cohort_index'],prompt_tokens=user['prompt_tokens'],
        graph_logit_storage_ptrs=sorted(graph_ptrs),supplied_storage_ptrs=sorted(supplied_storage_ptrs),saved_score_storage_ptrs=pointers,
        saved_score_dtypes=[str(s.dtype) for s in scores],
        full_score_hashes=[dict(shape=list(s.shape),dtype=str(s.dtype),sha256=digest(s)) for s in scores],
        beam_indices=output.beam_indices.cpu().tolist(),token_ids=output.sequences[:,user['prompt_tokens']:].cpu().tolist(),
        sequence_logp=output.sequences_scores.float().cpu().tolist(),
        transition_logp=model.compute_transition_scores(output.sequences,scores,output.beam_indices,normalize_logits=False).float().cpu().tolist())

def check_retained(outputs):
    pointers=[s.untyped_storage().data_ptr() for out in outputs for s in out.scores]
    if len(pointers)!=len(outputs)*5 or len(set(pointers))!=len(pointers):
        raise ValueError('Retained score tensors share owners')
