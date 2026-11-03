# Architecture

All autoregressive versions use ordinary decoder transformer tensors: token and
position embeddings, 18 causal self-attention blocks with 16 heads, residual
connections, LayerNorm, ReLU feed-forward layers and a final vocabulary projection.
Every parameter belongs to a named tensor; there are no file padding bytes.

Logical values are paired as `(x, -x)` in a 1024-dimensional residual stream.
A large constant anchor makes LayerNorm approximately preserve circuit values.
Token identities use 14 signed bits. Attention dot products implement equality
retrieval with a BOS sentinel that beats mismatching memories. Position scores
select the latest matching fact. ReLU gates copy bounded values conditionally.
The output projection compares an output code to vocabulary codes; argmax picks
one token. Output codes repeat through autoregressive forward passes.

This is compiled algorithmic computation inside a transformer. The sparse
connections and finite-state grammar are explicitly designed, not learned.
The compiler contains atomic subject/predicate/object records but no completed
response bank. New words receive fixed pointer IDs through tokenization; the
host remembers only their spelling, not their semantics. The host capitalizes
sentences and removes spaces before punctuation during detokenization.

In v5, layers 0-2 identify role boundaries, local fact patterns, the active query
and two numeric operands. Layers 3-9 retrieve seven relations. Layers 10-13
choose grammatical next tokens and semantic arguments. Layers 14-17 encode the
bounded sum into a numeric token. The integer-to-code circuit is a large
piecewise-linear construction, not yet a general arithmetic algorithm.

`engine.py` imports neither `knowledge.py` nor `build.py`. Generation is only
tokenization, tensor execution, causal masking, KV caching, argmax and text
formatting. It does not parse facts into a host dictionary or compute answers.
The external evaluator computes expected results but passes only each input
prompt to `Model.generate`. Ablations and causal/cache checks are recorded in
the original verification reports.
