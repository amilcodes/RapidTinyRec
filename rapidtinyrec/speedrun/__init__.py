"""Speedrun engine: a hand-written MiniOneRec (Qwen2) beam-search engine.

The whole request -- prefill, three constrained beam steps, forced-suffix
scoring and the final sort -- runs as device-resident tensor code so that it
can be captured as a single CUDA graph per (batch, padded-length) bucket.
"""
