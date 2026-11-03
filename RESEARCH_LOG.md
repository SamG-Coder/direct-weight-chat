# Research log

Dates in this log use UTC.

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

## 2026-11-03: polarity baseline before implementation

Defined fact polarity as a reusable field in subject/predicate/object memories.
Created 233 development cases before modifying the compiler: seven relations,
negative grammar variants, corrections in both directions, distractors, inherited
denials, negative membership, multi-sentence output, articles and dialogue.
Multiple values, yes/no questions, query/instruction scope and double negation
remain explicit stress tests. No test strings are supplied to construction.
The baseline outputs are in `results/polarity-v7a-baseline-cuda.json`.

## 2026-11-03: v8a fact polarity

Replaced unused per-property vowel flags with a negative-polarity bit and moved
article choice to a neural lexical classifier. A shared parser gate layout handles
four-token lookback, negative auxiliaries and contractions without storing answer
strings. Polarity follows facts through graph retrieval; negative class edges do
not license inheritance. Token grammar selects copular negation, cannot or
auxiliary does/not/base-verb forms from those semantic fields.

Polarity: 226/233 versus v7a's 53/233. Legacy: 185/185. Graph: 75/79; the former
negative-fact stress now succeeds. Arithmetic remains 114/120. Retained regression:
positive fly followed by negative swim returns the last denied action, exposing
the existing one-value-per-relation answer selection. One otherwise correct
negative sentence appends a token because tiny numerical residuals break the
implicit all-zero EOS tie. This candidate is archived before correction.

## 2026-11-03: v8b robust neural termination

Added a one-logit EOS bias, below the 16-logit margin between valid token codes,
to resolve the no-output state despite small floating-point residuals. Polarity:
227/233, with the six structural/scope/multiple-value stress failures retained.
Legacy remains 185/185, graph improves to 75/79, and arithmetic remains 114/120.
The previous fresh graph and arithmetic sets were rerun as regressions with the
same respective scores. This candidate is frozen before a fresh polarity seed.

## 2026-11-03: polarity verification and signed-operand baseline

Frozen v8b passed 227/233 with post-freeze seed 1760738774 on both CUDA and CPU.
Polarity ablation changes a denied ability into a positive ability and permits
inheritance from a denied class. Other integrity, causality, cache, graph and
arithmetic ablations pass. This evidence is bounded by the known generator;
the six scope/multiple-value stresses still fail, including one regression
relative to v7a's accidental pass when it ignored a negative assertion.

Created 159 signed-operand cases before implementation, covering negative inputs,
unary/binary signs, prefix addition, dialogue, bounds and unsupported operand
counts. Baseline v8b: 46/159. Syntax, multiplication, division and parentheses
stress cases remain in this suite; old mathematical expected answers remain
unchanged even when a new abstention check has a different purpose.

## 2026-11-03: v9a signed operands and operand-count validity

Expanded local neural lookback with number flags. Unary +/- no longer replace
the binary operator, and a gated linear circuit applies a unary minus to numeric
embeddings. A signed accumulator is formed in block 10; block 11 derives its
magnitude and sign. Bounds now test both sides of [-999,999]. An attention head
retrieves the second numeric position; comparison with the last position and
current-turn bounds rejects unsupported operand counts without Python parsing.

Signed suite: 153/159, including all 120 random signed cases and all unary,
prefix-addition, dialogue, range and operand-count cases. The three malformed
syntax cases still return numbers; multiplication, division and parentheses
remain failures. Legacy 185/185, graph 75/79, polarity 227/233, old arithmetic
115/120. The former negative-operand stress now succeeds; the chained-expression
math test still fails its mathematical answer even though the separate validity
test now correctly records abstention. All former fresh sets were rerun as
regressions. Candidate frozen before a new signed-arithmetic seed.

## 2026-11-03: v9a verification and release preparation

Post-freeze signed seed 3220192550 scored 153/159 on CUDA and CPU, including all
120 new random signed cases. Rebuilding from the frozen v9a compiler reproduces
raw weight and config hashes exactly. Parameter count/size, finite values,
causality, cached decoding and offline guard pass. Polarity, graph, accumulator,
unary-sign and forced-validity ablations change the corresponding behaviors.
Fourteen matched-suite comparisons identify no new v9a regressions against v8b;
the one multiple-action regression introduced in v8 remains recorded against v7a.

An exploratory conversation combines a negative fact, signed arithmetic,
context recall, a correction and another inherited property; exact token traces
are in `results/conversation-v9a.json`. It still cannot write the requested poem.
Prepared v8a, v8b and v9a archives for a second experimental release. All three
were streamed through CRC and decompressed SHA-256 verification. Earlier release
assets remain preserved. The general conversational research objective remains
incomplete. Next priority: object-specific truth queries and multi-value relation
retrieval, then broader compositional language and mathematical algorithms.

## 2026-11-03: v0.2 publication verified

Published v8a, v8b and v9a as `v0.2.0-experiment`. All four assets (three model
archives plus checksums) are uploaded; GitHub archive sizes/digests match local
records. A fresh clone of release commit `16a9ddc` and downloaded v9a archive
passed source/raw hashes, exact size, finite values, causal/cache checks, all
five circuit interventions and six CPU/CUDA smoke cases. The working release
uses the current default v9a checkpoint. The research goal is still active;
useful general-purpose conversation has not been achieved.

## 2026-11-04: object-specific truth baseline before implementation

Defined a three-valued proposition query: exact subject/relation/object evidence
returns yes or no according to its latest polarity; absence remains unknown.
The 239-case development suite was generated before compiler changes. It covers
four relation families, unfamiliar subjects and objects, per-object corrections,
distractors, inherited facts, question/assertion separation and dialogue. Eleven
composition/scope stresses preserve broader syntax and multiple-value limits.
Frozen v9a passes 45/239; most passes are existing abstentions. Exact baseline
outputs and timings are in `results/truth-v9a-baseline-cuda.json`.

## 2026-11-04: v10a proposition identity and three-valued queries

Compiled clause-boundary/auxiliary subject binding and distinct query events.
Four graph steps now also retrieve exact subject/predicate/object propositions;
per-object recency and nearest inherited evidence preserve polarity. Missing
evidence stays unknown. Permanent atomic facts use the same identity comparison.
The normal transformer runtime and parameter layout are unchanged.

Truth development improves from 45/239 to 229/239, including all 228 non-stress
cases. Ten composition/scope cases still fail. Legacy 185/185, graph 75/79,
arithmetic 115/120 and signed 153/159 retain their scores; polarity rises from
227/233 to 229/233 because its two earlier yes/no stresses now pass. All four
former post-freeze suites were rerun as regressions with the same respective
scores. The old open-question multiple-action bug remains.

Integrity, exact parameter size/hash, finite values, offline guard, causal
prefixes and cached/full decoding pass. Removing proposition retrieval changes
yes to unknown; removing only object identity changes yes to an unrelated no.
Other graph, polarity and arithmetic interventions retain their effects. Frozen
v10a has 732,615 nonzero parameters; raw SHA-256 is
`1734f1e0ed9bac539f5148e676dfc9b9792ccd881b848e5220c06b134208362f`.
This candidate is archived before choosing a fresh truth-suite seed.

## 2026-11-04: v10a fresh evaluation and query-negation baseline

After freeze commit `6bb1736`, truth seed 34865453 produced 229/239 on both CUDA
and CPU. The 200 main combinations vary subject/object bindings, corrections
and inheritance; the grammar was known before freezing. These are fresh
combinations, not an independent test design, and no v10a weights were adjusted.

Created 269 query-scope cases before another compiler change. They test all seven
existing predicates and positive/negative queries against positive, negative or
absent evidence. In this controlled grammar, query `not` negates the proposition;
that does not cover every pragmatic reading of negative English questions.
Frozen v10a baseline is 80/269. All eight new broader composition/scope stresses
remain in the score. The next candidate will reuse the proposition lookup while
extending predicate binding and applying a shared truth-negation operation.

## 2026-11-04: v11a predicate binding and proposition negation

Extended the reusable question-event representation to all seven predicates.
Query polarity is now separate from fact polarity; a shared equality/XOR circuit
produces the affirmative/negative truth token only when evidence is found.
Unknown remains unknown under negation. Three neural lookbacks support copular,
article, prepositional and negative-auxiliary inversion without a host parser.

Query scope improves 80/269 -> 262/269, including all 261 non-stress cases.
The one passing compound stress is accidental: a contrasting conjunction still
fails, so conjunction support is not claimed. Truth improves 229/239 -> 234/239;
legacy 185/185, graph 75/79, arithmetic 115/120, polarity 229/233 and signed
153/159 retain their scores. All five former post-freeze suites were rerun as
regressions. No new v11a pass/fail regressions were found.

The 22-suite matched comparison does reveal one v10a regression against v9a in
the new truth suite: a later instruction to ignore the question is disregarded
and the model returns yes. The earlier open-question multiple-action regression
also remains. Exact before/after outputs are preserved, not removed from scores.

All integrity checks and earlier interventions pass. Removing only the new
query-polarity bit changes yes to no for a known negative proposition. Frozen
v11a has 744,215 nonzero parameters and raw SHA-256
`f15e70fdeddf91f94bfac13a505b1f14157a9dae79bb4388febf1c62a2a637b4`.
The candidate is archived before selecting a new query-scope seed. No broad
language or general intelligence milestone is declared complete.

## 2026-11-04: v11a fresh evaluation and release preparation

Post-freeze query-scope seed 3050431372, selected after commit `55ab049`, scored
262/269 on CUDA and CPU. The 245 main generated cases vary bindings, corrections
and inheritance; 24 authored/conversation/stress cases retain the known structure.
This remains a known-generator evaluation. No candidate weights were adjusted.
Rebuilding the archived compiler reproduces raw weights and config byte for byte.
The default local model now references v11a; the formerly linked v9a artifact was
hashed afterward and remains unchanged.

Prepared v10a and v11a archives for `v0.3.0-experiment`, with streamed CRC and
decompressed SHA-256 verification. Earlier checkpoints and test failures remain
preserved. The next priority is to represent the active request and compound
scope so later instructions and conjunctions cannot silently use a partial
question; the old open wh-question multiple-value selection also needs work.
Useful flexible conversation remains an unfinished research goal.

## 2026-11-04: v0.3 publication and fresh-download validation

Published v10a and v11a under `v0.3.0-experiment`, release commit `11e2f62`.
All three assets (two archives plus checksums) are uploaded; archive sizes and
GitHub digests match the local packaging records. A fresh clone and downloaded
v11a archive pass source/raw hashes, parameter layout, finite values, causal/cache
checks, offline guard and all graph/arithmetic/polarity/truth interventions.

Fresh CPU and CUDA smoke runs each score 11/12. All 11 mixed-turn fact, truth,
negation, correction and arithmetic checks pass. The separately scored creative
request fails: `Write a poem about rain.` returns `Rain is a precipitation. It can
fall.` instead of the expected abstention. Local and freshly downloaded runs
agree. The failed check is preserved, not counted as success. Evidence is in
`results/publication-v0.3.json` and `results/release-validation-v0.3/`.
The research goal remains active and useful open-ended conversation is unachieved.

## 2026-11-04: proposition composition and request-scope baseline

Created 404 development cases before changing the compiler. They include full
three-valued two-proposition AND/OR tables across seven predicates, n-ary
reductions, independent query negation, mixed subjects and predicates, inherited
properties, dialogue, and later requests. Unsupported mixed operators, malformed
structures and paraphrases remain explicit stress cases. Frozen v11a scores
243/404; many passes come from using only one conjunct, so the truth-table
contrasts are essential. The next experiment will evaluate proposition events
in parallel, reduce their truth values neurally, and keep active clauses separate.

## 2026-11-04: v12a composition, scope and a preserved regression

Implemented per-position proposition events and strong three-valued AND/OR
reductions inside attention/MLPs. Clause IDs isolate the active request. The
inference runtime remains unchanged. Composition improves 243/404 -> 398/404,
including all 395 non-stress cases. Six structural stresses remain failures.
Graph retains 75/79 and legacy retains 185/185. Polarity improves to 230/233,
truth to 236/239 and query scope to 265/269. The former creative-request smoke
failure now correctly abstains; this does not demonstrate creative writing.

All prior suites were rerun. The new request-head filter rejects Please calculate,
causing 106 regressions across four arithmetic reports: development arithmetic
falls to 88/120 and signed operands to 121/159. Three additional pass/fail
regressions are unsupported composition stresses that previously passed by
ignoring part of the question. Exact comparisons are preserved in
`results/regressions-v12a.json`; the candidate is frozen, not overwritten.

Integrity and all earlier interventions pass with the shifted truth-head layout.
AND-to-OR, removed-unknown and removed-clause-key interventions change the
corresponding outputs. v12a has 788,721 nonzero parameters and raw SHA-256
`37cca688bea820ab2ca4e64b0dc0641f33df561a54ec402355a77c2e57aa14b3`.

Before correcting the filter, created 81 request-modifier cases, including
supported and unsupported requests, later clauses, two-modifier forms and
longer framing stresses. Frozen v12a scores 17/81. The correction will identify
the request head after transparent modifiers instead of labeling every modified
command as supported.

## 2026-11-04: v12b request-head modifiers and restored arithmetic

Compiled a bounded transparent-modifier grammar that reads the actual request
head and also binds modified inverted questions. Unsupported commands remain
unsupported after a modifier. Modifier development improves 17/81 -> 78/81.
Arithmetic returns to 115/120 and signed operands to 153/159; all 106 arithmetic
regressions across the four matched reports are repaired. Legacy remains 185/185,
graph 75/79, polarity 230/233, truth 236/239 and query scope 265/269.

Composition is 397/404. The corrected prefix handling exposes an unsupported
Please tell me whether paraphrase that v12a had rejected, losing one accidental
abstention pass. Three composition regressions against v11a remain: mixed
operator precedence, compound fact assertions and a request to answer only the
first part. They are retained as failures despite v11a's partial-answer passes.
The 30 adjacent-version comparisons and 14 direct v11a-to-v12b comparisons are
both recorded; no arithmetic regression is concealed by the aggregate gains.

All prior suites, the earlier release smoke suite and integrity/ablation checks
were rerun. v12b has 873,633 nonzero parameters; raw SHA-256 is
`07349cb626864c4ea77e32300af5670a6de69ee612662ec5e9d2d2cdba43cea5`.
The candidate is frozen before selecting fresh composition and modifier seeds.

## 2026-11-04: v12b fresh evaluation and v0.4 preparation

After freeze commit `49dd2ea`, composition seed 1676552291 scored 397/404 on
CUDA and CPU. Modifier seed 973315925 scored 78/81 on both; it adds 20 fresh
arithmetic combinations while the 61 structural cases repeat. The composition
generator also retains known syntax and truth-table structure. No frozen weights
were tuned after these runs, and independent generalization is not claimed.

The archived v12b compiler reproduces raw weight and config hashes exactly.
The default local model now references v12b; the former v11a raw file was hashed
afterward and is unchanged. Prepared v12a and v12b ZIPs for `v0.4.0-experiment`
and verified streamed decompression hashes/CRC. A 16-turn smoke conversation
passes locally, combining AND/OR, unknowns, corrections, inheritance, distinct
subjects/predicates, active requests, modified arithmetic and abstention.

Next priority: parse compound assertions and mixed-operator expressions into
composable structure, while retaining the malformed-input and framing failures.
The long-term goal of useful flexible conversation remains unachieved.

## 2026-11-04: v0.4 publication and fresh-download verification

Published v12a and v12b under `v0.4.0-experiment`, source commit `10fc451`.
GitHub reports all three assets uploaded; archive sizes/digests match the local
packaging report. The downloaded v12b ZIP and checksum file match their remote
digests. A fresh clone/extraction preserves a clean source tree and passes raw
and source hashes, exact size, finite values, causal/cache checks, offline guard
and every graph/arithmetic/polarity/truth/composition intervention.

The 16-turn fresh-download conversation scores 16/16 on both CUDA and CPU.
It exercises compound truth, unknown evidence, correction, inheritance, distinct
subjects/predicates, active request changes, transparent modifiers and arithmetic.
Creative and program requests correctly abstain in these checks; no generation
capability is inferred from that. Publication evidence is recorded in
`results/publication-v0.4.json` and `results/release-validation-v0.4/`.
The research goal remains active, with all structural failures and historical
regressions preserved.
