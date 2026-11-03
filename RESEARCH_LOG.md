# Research log

## 2026-11-03 — Historical import

Preserved the rejected response-selector prototype (v0) and five direct-written
autoregressive builds. Their original construction took place before this Git
repository existed, so their original manifests have no Git commit IDs. Content
hashes identify their source and weights. The first repository commit imports
their source, metadata and historical evidence; no historical commits are invented.

The first autoregressive build stored some articles/punctuation as fact values.
v2 corrected fact-value eligibility, but conflicts averaged and multi-turn sums
reused an old operand. v3 added recency, turn selection and inheritance but had a
stale temporary register. v4 cleared it but exposed coefficient overwriting in a
gated-copy primitive. v5 merged coefficients correctly. Original probe outputs
and the final CUDA/CPU verification are retained under package `evidence/`.

## 2026-11-03 — Research protocol and first audit

Adopted the user's constraints: no training, pretrained weights, online
construction/inference, complete answer bank, host reasoning, or test-specific
answer circuits. Exactly 250M float32 parameters. Test before implementation;
freeze each candidate before new evaluations; retain failures and make separate
commits for meaningful tasks.

The weakest immediately actionable generalisation is relationship composition:
v5 inherits from an authored class such as `bird` but cannot follow a chain of
unfamiliar classes. The next experiment will test bounded graph traversal over
prompt facts, with shuffled fact order, conflicting facts, cycles, distractors,
missing links and a depth boundary. Test strings must never enter the compiler.
