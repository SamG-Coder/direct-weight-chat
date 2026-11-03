# Research log

## 2026-11-03: historical import

Preserved the rejected response-selector prototype (v0) and five direct-written
autoregressive builds. Their construction took place before this Git repository
existed, so their original manifests have no Git commit IDs. Content hashes
identify their source and weights. Commit `5be0768` imports original evidence;
no historical commits are invented.

v1 stored some articles/punctuation as fact values. v2 corrected eligibility,
but conflicts averaged and multi-turn sums reused an old operand. v3 added
recency, turn selection and inheritance but had a stale temporary register. v4
cleared it but exposed coefficient overwriting in a gated-copy primitive. v5
merged coefficients correctly. Original probes and v5 CUDA/CPU verification
remain under package `evidence/`.

## 2026-11-03: protocol, audit and frozen baseline

Adopted the user's constraints: no training, pretrained weights, online
construction/inference, complete answer bank, host reasoning, or test-specific
answer circuits. Exactly 250M float32 parameters. Test before implementation;
freeze candidates before fresh evaluations; retain failures and commit
independently meaningful tasks.

The retrospective 185-case development comparison passed 131, 177, 182, 182 and
185 cases for v1-v5. A new 79-case graph suite was created before modifying the
compiler: six property relations, shuffled assertions, one-to-three class links,
depth stress, nearer/later overrides, authored leaves, cycles, absent links,
dialogue, paraphrase and negation. Frozen v5 passed 12/79, including 0/48 novel
chains. Commit `b5f7465` contains tests, baseline and version comparison.

## 2026-11-03: v6a failed scratch allocation

Reallocated temporary features and replaced seven serial single-predicate
retrieval layers with four parallel graph lookups and three static-fact lookups.
Neural inspection found the inherited property, but generation stopped after
the subject: scratch reuse erased the previous token needed by the grammar.
Graph: 8/79; legacy: 1/185. Commit `1f7509f` preserves this failed candidate,
source, hashes and all regressions. This paragraph was added in the following
commit after a documentation encoding error; results were already preserved.

## 2026-11-03: v6b preserved grammar state

Moved previous-token detectors before scratch reuse and retained their nine
Boolean results through traversal. Graph: 68/79; legacy: 182/185. All one- and
two-link novel chains succeeded. Remaining implementation failures: the static
lookup sentinel's token identity incorrectly contributed to its score, defeating
exact matches; tiny residuals in reused output coordinates caused two trait
answers to repeat instead of terminating. Depth and language stress failures
remain. This candidate is frozen before the next correction.

## 2026-11-03: v6c bounded graph traversal

Neutralized the BOS token identity in static lookup keys and reinitialized
recycled output coordinates before decoding. Graph development: 74/79,
including 48/48 one-to-three-link chains, all precedence/cycle/missing-link
cases and all authored leaves. Legacy: 185/185, with no regressions.
Five retained failures are longer-than-three-link chains, an unsupported
paraphrase and negation. This candidate is frozen before selecting a new
random evaluation seed. It is a bounded graph algorithm, not unrestricted
reasoning or general conversation.
