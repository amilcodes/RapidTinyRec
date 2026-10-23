# Where the work goes

MiniOneRec recommends an item by generating its **semantic ID**: three tokens
from a learned, residual-quantized item code. A catalog trie limits each next
token to continuations that correspond to known items. Multiple items can share
a code; the decoder ranks codes rather than resolving those collisions.

The model receives a prompt containing a user's interaction history. Training
uses next-token cross-entropy to teach the model to predict the target code.
The released checkpoint and its training recipe are upstream MiniOneRec's work;
RapidTinyRec changes inference only.

For an output `a, b, c, newline, EOS`, the score is the sum of all five token
log-probabilities, each conditioned on the prompt and preceding output tokens.
Length penalty is zero. Even though the trie forces newline and EOS, their
probabilities still vary between candidates and can change the final ranking.
The implementation preserves both full-vocabulary normalizations in the released
Hugging Face beam-search / logits-processor path.

At every generation step, beam search adds allowed token scores to each beam's
accumulated score, selects the best continuations and reorders their KV caches.
It is an approximate search over the catalog. These optimizations keep its beam
width, scoring and selection policy fixed.

## Prefill once

The default generation path expands a single input history to 50 beam rows before
the first transformer forward. Every row has the same tokens, so the model does
the same history computation 50 times.

[`PrefillOnce`](../rapidtinyrec/shared_prefill.py) slices those inputs to one row,
runs the original forward and repeats the resulting logits and KV cache to 50
rows. Later forwards process one new token per beam. Each row then contains the
history followed by that beam's generated prefix. Selecting a parent twice copies
its cache into two child rows; their next tokens can diverge. This eliminates
repeated prefill computation while keeping Hugging Face's existing cache policy.

## Constraints and attention

The released constraint processor transfers each beam's prefix to the CPU
separately. The batched version performs one prefix transfer and batches mask
writes. The GPU version compiles the catalog into device tables once, then uses
tensor lookups without per-beam CPU transfers. Allowed tokens and score math
remain unchanged; invalid prefixes retain the EOS fallback.

Compiling the model forward exposed a second cost. This pinned Transformers
version's conservative mask handling can prevent native grouped-query attention,
expanding two KV heads into twelve. For unpadded prefill, causal attention supplies
the correct mask. For a one-token decode, every cached token is visible. Supplying
that information at the model boundary preserves native GQA and avoids the
expansion. The ordinary two-dimensional generation mask is retained for position
handling. See [`benchmark_gqa.py`](../rapidtinyrec/benchmark_gqa.py) for the shape
checks and model-boundary override.
