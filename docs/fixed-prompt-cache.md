# Fixed common-prompt cache preparation

The 500 frozen prompts share 69 initial tokens. `FixedPromptCache` computes a
fixed prefix once, then computes each request's suffix at its original absolute
positions. This is cross-request reuse, additional to the existing within-request
beam prefix storage. The implementation remains a CPU-tested building block;
there is no CUDA performance or release-checkpoint quality result yet.

Each suffix uses an explicit additive mask: query at absolute position `69+i`
can see keys `0..69+i`, including all cached keys. Empty-cache causal shortcuts
must not be applied to this non-square attention. This explicit mask also changes
HF's GQA dispatch path, so shorter prefill is not assumed to be faster.

Checkpoint and tokenizer manifest digests are supplied after external verification;
the cache also binds model instance, parameter versions/dtype/device, configuration,
numerical policy and exact prefix tokens. Only static default RoPE and unpadded,
full-attention Qwen2 SDPA are supported. Requests get independent KV clones before
suffix execution. The clone cost and all request-time identity checks belong
inside warm-generation timing. Prefix fill, preparation and audit snapshots are
separate; report cold first request and warm service separately. No cache eviction,
general radix tree, model mutation, or concurrent use is supported.

`PromptCacheGenerator` binds a prepared input to the unchanged HF generation
loop, routes only its first call through masked suffix prefill, expands the
complete prompt KV into the existing physical beam arena, and delegates four
single-token calls to the supplied decode replay callable. It supports the
existing direct reorder context. All wrapper and KV-copy costs stay inside its
`generate` call. Exceptions restore the model's forward; no eager fallback is
introduced. The eventual pilot must supply and freeze actual compiled/replay
callables; CPU tests alone do not exercise CUDA graph capture.

Tiny CPU tests compare suffix logits and every KV tensor to full prompt prefill,
exercise different suffix lengths and repeat-after-other-user stability, mutate
and reorder request caches, reject mismatched identities, and exercise existing
50-beam generation with five independently stored FP32 score tensors. Tolerances
in CPU arithmetic checks are not an exact GPU claim or quality-loss tolerance.

`benchmark_prompt_cache` now supplies the separate TF32 control/cached-TF32
entrypoint with original scoring, retained output ownership checks, compilation
freeze, cold-build accounting, and raw persistent-KV snapshots before/after
warmup, timing and profiling. Initial prefix build includes any nested compilation;
it is not a steady-state cold-cache latency estimate. Generation peaks are frozen
before snapshot allocations. The allocated independent auditor, deadline runner
and submission package are still pending, so this entrypoint is not ready to run.
The baseline choice can be revised only
after BF16 evidence is audited; do not combine precision and caching changes in
the first experiment. No second GPU job while the BF16 job is active.
