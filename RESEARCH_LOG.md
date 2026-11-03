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

## 2026-11-03: post-freeze graph verification and arithmetic baseline

v6c, frozen at commit `95f9754`, passed 74/79 with fresh seed 965544940 on both
CUDA and CPU, including 48/48 novel chains. The seed was chosen after freezing
weights; the generator and grammatical structures were already known. This is
fresh-combination evidence, not a structurally independent benchmark. No weights
were changed in response to those results. Causal prefix difference was zero;
cache/full-prefix tokens agreed; graph ablation removed the inherited answer;
source/weight hashes, exact size, finite values and offline guard passed.

Created 120 arithmetic cases before implementation, including 88 subtraction
cases, addition regression, mixed turns and six unsupported-operation stresses.
v6c passed 22/120 (only addition cases). The next circuit will compute a signed
difference with ReLU arithmetic, reusing the existing integer decoder. The model
will still accept only two nonnegative operands of at most 999.

## 2026-11-03: v7a signed subtraction

The compiler now routes the current turn's operator through attention, calculates
absolute difference as ReLU(a-b)+ReLU(b-a), and derives a negative-result bit
with a bounded ramp. A negative result emits the minus token, followed by the
magnitude token. Python only removes the space between these generated tokens.
No operand-pair lookup table or new training procedure was added.

Development: 114/120 arithmetic (88/88 subtraction, 20/20 addition, 6/6 mixed
turns); legacy 185/185; graph 74/79. Six unsupported arithmetic stresses remain
failures, including negative operands and a chained expression that produce
incorrect answers. These failures are retained. The existing integer-to-token
piecewise decoder remains a limitation; this is not arbitrary-precision math.
Candidate v7a is frozen before generating new arithmetic combinations.

## 2026-11-03: final verification and release preparation

v7a arithmetic with post-freeze seed 2964068083 scored 114/120 on both CUDA and
CPU. That set contains 100 fresh random operand combinations and 20 repeated
boundary, dialogue and stress cases. The former v6c fresh graph set is now a
regression suite for v7a and remains 74/79. No frozen weights were tuned after
these evaluations. Arithmetic accumulator ablation changes `149 - 370` from
`-221.` to `-0.`, while graph and output-head ablations, causality, cache parity,
size, finiteness, hash and offline checks pass.

Rebuilding v7a from its archived compiler reproduces the exact weight and config
hashes. Ten checkpoint archives (selector v0, autoregressive v1-v5, v6a, v6b,
v6c and v7a) were compressed and their decompressed hashes/CRCs verified. Raw
1 GB tensors remain outside Git. The public experiment is MIT licensed.

Recommended next experiment: add an explicit neural validity/polarity
representation, test negative assertions and compound expressions before
implementation, and reject unsupported structures consistently. Broader language,
code generation and useful open-ended conversation are still unresolved. The
long-term research objective is not complete.

## 2026-11-03: publication verified

Public repository and prerelease `v0.1.0-experiment` are live. GitHub reports all
11 assets uploaded (ten checkpoint archives plus checksums); each ZIP's size
and SHA-256 match the local packaging report. A fresh clone of release commit
`33952a6` and a fresh downloaded v7a ZIP passed raw/config hash checks, 39/39
quick verification cases, and four graph/subtraction smoke checks across CUDA
and CPU. Evidence is in `results/publication.json` and `results/release-validation/`.
