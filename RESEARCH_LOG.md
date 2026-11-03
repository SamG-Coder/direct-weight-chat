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

## 2026-11-04: compound-assertion baseline before implementation

Added a deterministic 390-case suite for shared-subject assertions across all
seven predicates, positive/negative coordination, predicate and subject changes,
object determiners, correction, inheritance and multi-turn question isolation.
The frozen v12b baseline is 158/390. In particular, shared-positive and
shared-negative assertions each score 21/49, mixed binding 16/112, object
determiners 0/24 and inherited compound properties 0/15. Four question-isolation
failures show that declarative fragments inside a question can become evidence.

Nine explicit stress cases retain disjunction, conditionals, multiple class
parents, comma lists, longer negation, double negation and open multi-value
generation. These score 1/9. The intended next circuit binds coordinated objects
to clause-local subject/predicate/polarity registers while rejecting question
clauses as assertions. The suite and exact baseline are committed before that
compiler change; this is development evidence, not independent generalization.

## 2026-11-04: v13a clause-local coordinated assertions

Moved full assertion events to block 1 and added separate lexical-predicate and
subject-frame attention. Fourteen bounded coordination layouts compile into
provisional object events; a valid same-clause subject frame completes them in
block 2. Predicate replacement and local negation have separate lifetimes.
Question-head exclusion prevents declarative fragments inside questions from
being stored as evidence. The inference runtime remains byte-for-byte unchanged.

Assertion development improves 158/390 -> 382/390 (381/381 targeted grammar,
1/9 stress). Replayed all nineteen other suites. All earlier passes are retained;
composition improves 397/404 -> 398/404 on both development and former fresh sets.
The nineteen matched comparisons retain one new-suite regression: a conditional
with a satisfied premise now abstains. Its previous Yes also appeared without
the premise, so it was not conditional reasoning. The no-premise case is repaired;
the rule-solving case remains failed rather than receiving a changed expectation.

Hash/layout/finite/causal/cache/offline checks and prior interventions pass.
Removing continuation validity or predicate binding changes a coordinated-fact
Yes to unknown. Removing question exclusion recreates a false remembered fact;
removing both clause bindings leaks an earlier subject/predicate into a bare
later continuation. These interventions affect runtime copies only.

v13a has 928,135 nonzero parameters; raw SHA-256 is
`dcd4cdeff8c750fc95a9071e3de833a8849b4874d8dfda984dceb885197858af`.
This candidate is frozen and committed before drawing a fresh assertion seed.
Disjunction, conditional application, multiple-parent traversal, open-list
generation and mixed Boolean precedence remain unresolved.

## 2026-11-04: v13a fresh combinations and adversarial boundary audit

After freeze commit `89e58f3`, seed 4176840521 reproduced 382/390 on CUDA and CPU.
The generator and grammar were known before freezing; this is fresh combination
evidence, not an independent language evaluation. Frozen v13a is not retuned.

An additional 76-case audit was authored after the freeze using seed 629104045.
CUDA scores 41/76. All eight new-subject positive controls pass, but the same
clauses also create eight false object assertions: `X has wings and Y needs
water` prematurely stores `X has Y`. Indirect `Tell me whether` requests create
and retain unasserted facts (sixteen failures); Could you forms pass. At 2,048
padding tokens and beyond, several fact and generation checks fail despite
fitting the context window. Exact short/long outputs and timings are preserved.

The next candidate will defer coordinated-object validation to the following
token and exclude directive request clauses from assertion storage. The long-
context numerical failures remain a separate unresolved issue. These published
boundaries become development/regression evidence for that new candidate;
they will not be relabeled as fresh evidence for it.

## 2026-11-04: preserved v13b deferred-commit failure

The deferred-object candidate fixes the observed false subject-as-object events
and Tell-me-whether assertion leaks, but loses valid final coordinated objects.
Its assertion score falls to 184/390, on both development and the former v13a
fresh set. The boundary audit stays 41/76 because new fixes are offset by lost
positive controls. Twenty-two matched suites record 422 regressions, including
repeated cases across suites. All older suites were rerun and exact reports kept.

The failure is a clause-phase mismatch at sentence-ending punctuation: the
pending object's query uses the old clause while a zero-valued current key uses
the new clause. The latter dominates attention and supplies no subject frame.
The integrity run fails its first assertion intervention because the original
answer is already unknown; the failure traceback is saved rather than called a
pass. Frozen v13b has raw SHA-256
`a9e0528b8f698db8e39caa696e3871c5d31c4ed3a19522f12f573c115c6ca724`.
The next candidate aligns both attention coordinates to the preceding token.

The completed v13a CPU boundary audit is 40/76 versus CUDA 41/76. The one-score
difference is in a long-context isolation case; several long outputs differ
between devices. Both expose unresolved numerical/context accuracy limits.

## 2026-11-04: v13c delayed-clause alignment and remaining precision error

Aligning both subject-frame attention coordinates restores 382/390 assertions
and improves the boundary suite to 61/76. The old suites are rerun; no additional
failures appear there. All integrity checks and interventions pass, including a
new intervention that forces following-token validity and recreates a spurious
subject-as-object fact. Frozen source reproduces raw/config hashes exactly.

Four boundary positive controls still regress against v13a when the new subject
has an article. An external token-by-token activation trace shows that the fuel
object binds to the earlier subject even though the later subject's full fact
exists. The recycled clause coordinate is stored at position/8192 scale after
clearing a +/-1 code; float32 cancellation changes its low bits, and the clause
penalty amplifies that error above the recency bonus. This is a representation
precision issue, not a missing article grammar rule. The candidate and diagnostic
trace are preserved before changing the coordinate scale.

v13c raw SHA-256:
`af534471a28ef79649013f09d7c645561b01aef23288de7e07d39e687c897292`.

## 2026-11-04: v13d coordinate precision and restored subject controls

Preserved the delayed clause coordinate at position/128 instead of position/8192
and changed the attention coefficient inversely. The ideal score is identical;
the residual write loses less float32 precision. Assertion development remains
382/390, and the boundary audit improves 61/76 -> 65/76 on CUDA. All 48 short
boundary cases now pass. The 11 remaining failures are long-context cases.

All 22 existing suites were rerun. No regression remains against v13a across
matched reports; the earlier conditional-stress regression against v12b remains
explicit. Legacy 185/185, graph 75/79, arithmetic 115/120, polarity 230/233,
signed 153/159, truth 236/239, query scope 265/269, composition 398/404 and
modifiers 78/81 are retained. All integrity/ablation checks pass, including
following-token validity and deferred-event cache parity. Frozen source
reproduces raw/config hashes exactly. The inference runtime is unchanged.

v13d has 929,381 nonzero parameters; raw SHA-256 is
`ff0b1e1af0909fc6237bdcfc7fc41e6df590cd678bc9552705bb3a943e2b549d`.
This candidate is committed before choosing its fresh assertion/boundary seeds.
Useful general chat remains unachieved. The next priority is the demonstrated
long-context positional/generation failure, followed by broader expression
structure, rather than treating the full storage window as reliable.

## 2026-11-04: v13d fresh inputs and v0.5 preparation

After freeze commit `4382108`, assertion seed 3178044486 scores 382/390 on CUDA
and CPU. Boundary seed 3284145273 scores 65/76 CUDA and 64/76 CPU, including
48/48 short controls on each. The discrepancy and all remaining failures occur
in long contexts. Generators and structures were known before freezing; these
are fresh combinations, not independent evidence of broad intelligence.
No v13d source or raw weights were tuned after these runs.

The default local model now references v13d; the previous v12b raw hash remains
unchanged. Prepared four archives for v0.5, preserving v13a, failed v13b,
intermediate v13c and v13d. Streamed decompression validates hashes and CRC for
every checkpoint. The 20-turn local publication smoke passes 20/20, exercising
deferred objects, subject isolation, correction, inherited properties, negative
facts, question/directive isolation, arithmetic and unsupported-request abstention.

## 2026-11-04: v0.5 public release and fresh-download validation

Published v0.5.0-experiment at source commit `81c7f7b`, with four checkpoint ZIPs
and SHA256SUMS. GitHub reports all five assets uploaded. All four downloaded
archives match the public digests, sizes, local package hashes and checksum
file. Streamed decompression verifies raw/config hashes and CRC; each frozen
source manifest matches the clean tagged clone. The v13d extraction leaves the
source tree clean.

Fresh-download integrity passes independently on CUDA and CPU, including every
graph/arithmetic/polarity/truth/composition/assertion intervention, causal-prefix
and cache checks, exact 1 GB layout, finite weights and offline guard. The 20-turn
smoke conversation scores 20/20 on each device. No checkpoint bytes change.
Evidence is in `results/publication-v0.5.json` and `results/release-validation-v0.5/`.

The publication checks do not erase the 8/390 assertion stresses or the 11/76
CUDA and 12/76 CPU fresh boundary failures. Those long-context failures are the
next investigation. The broader goal of useful flexible conversation is active
and remains unachieved.

## 2026-11-04: context-position baseline before representation changes

Added 126 deterministic position-shift cases covering existing generation,
signed arithmetic, near/distant compound facts, negation, distinct subjects,
three-hop inheritance, correction and active-question scope. Six requests run
near the declared context capacity, including a 5,007-token prefix. Frozen v13d
scores 61/126 on CUDA; some failures appear at 1,536 padding tokens.

A separate diagnostic captures actual Q/K activations for all nine local
lookback heads and runs attention with synthetic positional indices as V. Across
all 5,048 positions per head, 15,900/45,432 addresses deviate by more than 0.01
token, with errors as large as three tokens. This diagnostic does not alter
inference, weights or answers. The quadratic positional scores are large enough
that float32 error changes address selection. Tests and baseline evidence are
committed before replacing the representation with bounded positional features.

## 2026-11-04: v14a bounded positions, repaired context and correction regression

Replaced all nine local lookback scores with two bounded sinusoidal phases at
periods 64 and 8192. Fixed query rotations address earlier tokens; the minimum
ideal nonmatching score gap exceeds 39 across the complete window. Removed the
quadratic embedding and its delayed copy. The layout, capacity and runtime are
unchanged. The 45,432-address diagnostic has zero errors on both CPU and CUDA;
the prior CPU baseline has 7,491 errors. Context development improves 61/126 to
126/126, and both former boundary sets improve to 76/76 on CUDA.

All 26 suites were rerun. Sixteen correction cases regress in each of three
assertion reports (48 repeated regressions): assertion score falls to 366/390.
An activation trace shows delayed assertion codes with magnitudes around
1.000011 versus 1.000001 for a later direct correction. The sharp proposition
match amplifies this difference, weakening the intended recency advantage and
mixing opposite polarities. This is separate from positional addressing; the
address audit and all current integrity/ablation checks still pass.

The failed candidate is frozen with raw SHA-256
`2dd59a1e327deaa42f10d5cb18152052b52e19358dc5dd2bf1ad2acfc4a96915`.
The next candidate will canonicalize fact identity/metadata coordinates before
proposition retrieval, rather than add a correction-specific answer rule.

## 2026-11-04: v14b shared fact saturation restores corrections

Added a shared block-3 saturation barrier for subject/object identity and fact
metadata before proposition lookup. This removes the extra magnitude of the
delayed assertion path without changing any grammar rule. Assertions return to
382/390 in all three existing assertion suites. Context remains 126/126 and both
boundary sets are 76/76 on CUDA. All 26 matched v13d/v14b suites retain their
earlier passes; the adjacent-version report preserves all 48 v14a regressions.

The 45,432-address audit remains exact on CPU and CUDA. Integrity checks pass,
including phase-removal interventions, long-prefix full/cache parity and a new
controlled amplitude test: scaling delayed identity writes by 1.001 preserves
the later denial with saturation, but removing saturation makes the older
positive fact win. Diagnostic changes use restored tensor copies, not saved
weight updates. Frozen source reproduces raw/config hashes exactly.

v14b has 960,705 nonzero parameters; raw SHA-256 is
`61deda2326ad355a3b2921e608e37de3e47e7949f445753cb8a0b78ad7288646`.
The candidate is committed before choosing a fresh context seed. These numerical
repairs do not establish general language understanding; broader grammar,
conditional reasoning, creative writing and code generation remain unachieved.

## 2026-11-04: frozen v14b fresh context evaluation

After committing v14b at `b3b6e1bd76a7e007262d6cf16c27dd3ed4cc1c04`, selected
context seed 2063643282. The 126 fresh combinations score 126/126 on CUDA and
125/126 on CPU. CPU case `context-086` has a 4,720-token prefix, followed by
`A omnddrvr has wings and a omsmpnrl needs water and fuel. Does omsmpnrl need fuel?`
It answers `I do not know.` instead of `Yes.`. Exact output, token IDs and timing
are retained. The zero-error local-address audit therefore does not guarantee
all semantic binding at long positions. Frozen weights were not retuned.

This generator and its structures were known before freezing. Fresh spellings,
operands and twelve random offsets are useful combination checks; many fixed
offsets and all six near-capacity structures repeat. This is not independently
designed evidence of general language understanding. Median response time is
55.99 ms on CUDA and 1,230.27 ms on CPU, across varying prefix/output lengths.

A separate publication smoke adds 3,000 tokens of real declarative distractors
before the existing 20-turn conversation; it passes 20/20 locally on CUDA.
Both v14a and v14b are packaged, preserving v14a's correction regressions.
Streamed decompression verifies both raw/config hashes and ZIP CRC. The default
local checkpoint now references v14b; the frozen v13d raw hash is unchanged.
The remaining CPU binding failure is a suitable next diagnostic target before
extending logical structure or claiming reliable long conversations.

## 2026-11-04: v0.6 published and independently downloaded

Published v0.6.0-experiment at source commit
`ad07de31df757bcd7393cb3cb3e6210d400c3ec2`, with v14a/v14b ZIPs and SHA256SUMS.
Both downloaded archives match public digests, sizes, the local package report
and checksum entries. Streamed decompression verifies raw/config hashes and CRC;
both manifests and compiler hashes match the clean tagged clone. Extracted v14b
has exactly 1,000,000,000 raw weight bytes and leaves the source tree clean.

Fresh-download integrity and all interventions pass on CPU and CUDA, including
bounded-phase removal, long-prefix cache parity and fact-magnitude saturation.
The long 20-turn smoke conversation scores 20/20 CUDA and 19/20 CPU. CPU turn 2
asks `A neru has wings and a selv needs water and fuel. Does selv need fuel?`
at a 3,036-token prefix and answers `I do not know.` instead of `Yes.`. This is
another second-object/changed-subject binding failure. It remains in the public
report, not a passing publication smoke. Saved weights are unchanged.

Evidence is in `results/publication-v0.6.json` and `results/release-validation-v0.6/`.
The next useful experiment should trace and stress numerical subject/frame
binding across positions and devices. The general conversation objective remains
active and unachieved; addressing arithmetic or language expansion alone would
not resolve this demonstrated context reliability gap.

## 2026-11-04: subject/frame binding baseline and activation evidence

The previous goal turn made concrete progress: it froze and published v14a/v14b,
verified downloads, and preserved two CPU failures. The new 136-case development
suite tests changed subjects across all seven predicates, polarity, wrong-subject
isolation, single-subject controls, random positions, corrections and dialogue.
Before implementation, frozen v14b scores 136/136 CUDA and 133/136 CPU. CPU loses
the second object's new subject in two cases, and assigns that object to the
wrong subject in a follow-up (a false Yes, not just conservative abstention).

A reusable external trace captures residual fields and actual attention outputs
without modifying weights or inference. In original fresh case `context-086`,
the CPU clause coordinate at the older `wings` fact is 36.7265625; the newer
`water` fact is 36.7265586853. The difference is about 0.0000038. Multiplication
by the clause penalty 16,777,216 creates a 64-point bias that outweighs the
40-point recency advantage of the five-token-newer fact. At punctuation, block-2
head 1 consequently retrieves subject token 370 instead of 371, and commits
`fuel` under the old subject. Float64 score recomputation from the captured Q/K
also favors the wrong fact: the error already exists in the coordinate/key,
not only in the attention dot-product reduction. CUDA retains the right subject.

The next candidate will separate latest-event selection from clause validation:
attention retrieves the newest valid event with modest scores; neural gates
reject a retrieved event from an older clause using a low-scale coordinate
difference. This removes absolute clause precision from competition between
valid subjects. The source tests, baselines and traces are committed before
changing the compiler. v14b post-freeze cases become regression evidence for the
new candidate; they are not new held-out tests for that candidate.

## 2026-11-04: v15a separates recency selection from clause validation

Replaced both clause-penalized assertion heads with a shared two-stage primitive.
Attention uses `65536*valid + 8*position`, returning the selected event and its
group. Each dependent ReLU pattern, validity gate and bounded identity copy
includes `512*(retrieved_group-current_group)`. An older clause is rejected;
matching groups cannot bias attention's recency ranking. Predicate/frame scratch
fields are cleared before later query/grammar roles. The generic tensor runtime
is unchanged. Exact weight bytes remain 1,000,000,000; 961,697 parameters are
nonzero. Raw SHA-256 is
`6386f31062760a2f5d81c1197c734af7ddd73f888fe0d44c09180de7f026ad81`.

All 29 existing CUDA suites were rerun, with zero regressions against v14b.
This includes 136/136 subject binding, 126/126 context, 382/390 assertions and
76/76 boundary controls. The earlier unsupported structures remain failures.
The evaluator now permits an explicit provenance label: old post-freeze suites
are labeled regression evidence for new candidates, without modifying the
original suite or its historical reports.

The first integrity attempt stopped because removing recency on a two-subject
probe did not change its Yes answer. This ineffective intervention is preserved
in `recency-intervention-v15a-design.json`. Three-subject controls reveal both
the missed latest fact and a false fact attached to an earlier subject when
recency is removed. The revised diagnostic retains the original neutral probe
and records all outcomes; no model source or weights changed in response.
CUDA integrity then passes, including scope-gate removal and all older
interventions. CPU binding improves 133/136 to 136/136, repairing all three
failures. The former fresh context set improves 125/126 to 126/126 CPU; the long
conversation improves 19/20 to 20/20. Two matched root CPU reports have zero
regressions, and the separately compared conversation retains all old passes.

CPU integrity and interventions also pass. Both devices retain zero errors in
the 45,432-address audit. The repaired CPU trace retrieves subject 371: the newer
water fact scores 103215.9721 versus 103175.9721 for the older wings fact. Clause
rounding no longer competes with that 40-point recency gap. Archived source
reproduces exact raw/config hashes. The candidate is committed before any new
binding seed is selected; no general-conversation success is claimed.

## 2026-11-04: frozen v15a fresh combinations

After commit `a583d2b49325d3ec66c937c32f06c43fa0b19ac4`, selected seed 192095465.
The fresh binding suite scores 136/136 on CUDA and CPU. It varies unfamiliar
names, combinations and eight random offsets, while the seven fixed offsets,
controlled language structures and expected answer types are already known.
These are combination checks, not independent evidence of broad intelligence.
Frozen source and checkpoint bytes were not tuned after the results.

The default local model now references v15a; the frozen v14b raw hash remains
unchanged. The v0.7 archive passes streamed raw/config hash and ZIP CRC checks.
A 23-turn publication conversation adds three-subject binding, wrong-subject
isolation and correction to the previous long conversation, still following
3,000 tokens of declarative distractors. It is a post-freeze author-designed
smoke check, not an independent language benchmark.
The local CUDA run passes 23/23 before publication.

## 2026-11-04: v0.7 public release and fresh-checkout validation

Published v0.7.0-experiment at `afb8241c50d59cdfc405292aaf89024d631585de`.
The downloaded v15a ZIP and SHA256SUMS match public digests/sizes and the local
package report. Streamed decompression checks raw/config hashes and CRC; the
manifest/compiler hashes match a fresh tagged clone. Extraction leaves the
source tree clean. The checkpoint remains exactly 1 GB of float32 parameters.

Fresh-checkout integrity and every intervention pass on CPU and CUDA. The long
23-turn conversation scores 23/23 on each device, including new three-subject
binding, isolation and correction. Raw weight bytes remain unchanged. Evidence
is in `results/publication-v0.7.json` and `results/release-validation-v0.7/`.

The numerical subject-binding failures motivating this iteration are repaired
across development, older regressions, fresh known-generator combinations and
publication checks. This does not prove arbitrary language or context behavior.
The next capability experiment should address actual conditional-rule inference:
distinguish a rule from an asserted conclusion and require its premise, including
missing and negative premises. Useful flexible conversation remains unachieved;
the full research goal stays active.

## 2026-11-04: conditional reasoning tests before implementation

Added `tests/generate_conditionals.py` and 293 seeded development cases before
changing the compiler. The target is signed ground implications with explicit
subjects and one premise, including facts before or after the rule, corrections,
identity isolation, absent/false premises, and position shifts. Chains through
eight rules, alternative proofs, cycles, compound premises and paraphrases are
retained as stress cases. The external finite-closure oracle never enters the
inference process. Explicit facts take precedence over derived information.

Frozen v15a scores 152/293 on both CPU and CUDA. It passes 0/49 positive
predicate-pair implications and 0/49 signed implications; most passes are
correct unknown answers. Its two positive paraphrase passes do not establish
conditional understanding: unsupported syntax can be mistaken for assertions.
Exact outputs, timings and source/model hashes are in the two baseline reports.
The next compiler experiment will encode typed rule records, retrieve a rule
by its conclusion, and check its signed premise through neural fact attention.

### First conditional candidate v16a

The frozen first candidate scores 269/293 conditional cases on CUDA, including
49/49 positive predicate pairs, 49/49 signed cases, and all nine chain cases
through depth three. Its 24 remaining failures include deeper chains and an
older satisfied alternative hidden by a newer unsatisfied rule. Legacy checks
regress to 182/185: the reused context bank can accidentally activate rule
attention on open-answer requests. The candidate and exact repeated-token
outputs are retained. The correction must gate both rule attention and proof
acceptance explicitly by the binary-question mode, not just by a state field
that serves a different purpose in open-answer generation.

### v16b isolation correction and retained static-fact failure

v16b retains 269/293 conditional development cases and repairs the open-answer
field collision. All 33 saved suites were run on CUDA for both candidates
(the 36 guard cases were added after the first run began). The guards score
19/36 on v15a, 27/36 on v16a and 29/36 on v16b. They expose unsupported
paraphrases being treated as unconditional assertions, compound rules being
rejected and rule-derived open answers remaining unavailable.

The full v16b run still loses one truth case in each of two suites: `Does a
robot have sensors?` becomes unknown. The read-only activation trace finds a
spurious rule validity of about 493 at the robot token after block 2. Atomic
fact bits share the temporary rule-coordinate bank; a +1 static bit is
misread as a clause coordinate, adding +512 to the scope expression even
though the token is not a rule terminator. Saturation then makes it look like
a valid rule and static truth attention excludes it. This requires a general
record-type guard, not an exception for robot or sensors.

Six of seven conditional interventions pass. Removing only the object-match
dot product also removes its 14-point match reward, so the BOS sentinel wins
even on the intended ablated path. That ineffective intervention is preserved
in `conditional-integrity-v16b-cuda.json`. A corrected diagnostic must retain
the maximum object reward while removing dependence on the object's identity.
The model itself will not be changed to make an intervention pass.

### v16c typed rule records and completed CUDA regressions

The compiler adds `4096*(terminator-1)` to each rule-record copy/gate. Its
negative type penalty dominates the maximum +1024 spillover from the two
atomic bits occupying temporary group coordinates, plus the bounded gate
terms. The penalty cancels at real terminators where the original atomic
vector is zero. The corrected robot trace has rule validity zero and answers
Yes. No entity-specific weight or answer exception was added.

All 33 matching CUDA suites retain every v15a pass: 5,499/5,619 cases, 130
improvements and zero regressions. The count includes repeated structures
across historical suites. Conditional development is 269/293; guards 29/36;
assertions improve to 383/390; legacy is 185/185 and truth returns to 236/239.
v16a and v16b and their exact failures remain archived. All eight conditional
interventions now pass, alongside the existing integrity/intervention suite.
The 31-turn publication smoke passes on CUDA. CPU validation follows.

Rebuilding the frozen v16c compiler reproduces raw SHA-256
`385dff256910dec6eba802d3930452c2963e85223bbaa54251d2931b9b86b1dc`
and the unchanged config hash. It has 995,881 nonzero parameters, still exactly
250,000,000 float32 values / 1,000,000,000 bytes without padding. The ordinary
inference runtime is unchanged. Three release archives are prepared offline,
including the failed v16a/v16b candidates, with decompressed hashes and CRC
verified. The full research goal remains active.

### v16c CPU validation and long-conversation limitation

CPU matches conditional development 269/293, guards 29/36 and legacy 185/185.
All eight conditional interventions and the existing full integrity/intervention
suite pass. The new 31-turn conversation, however, scores 28/31 CPU versus
31/31 CUDA. CPU misses conditional-smoke-1, -3 and -4, returning unknown rather
than the required positive/negative conclusions late in a long conversation.
The previous 23 turns all retain their CPU passes. Exact failures are preserved;
this release must not claim reliable long-context CPU rule inference.

The source was frozen at `9c84c6ad4c4202764358ab0c48d4d2a4d7cc25e7`
before selecting conditional seed 1711846539. Its fresh combinations are being
evaluated without retuning the frozen checkpoint. The next experiment should
trace the CPU rule-record/proof states at the long-conversation failures before
adding alternative proof search or broader language structures.

Fresh conditional seed 1711846539 scores 269/293 on both CPU and CUDA, with
49/49 positive pairs, 49/49 signed applications and all tested one-to-three
rule chains. The same 24 structural stress failures remain. The generator was
known before freezing, so this is fresh-combination evidence, not an independent
language benchmark. The frozen model has not been retuned. The default model
was switched to v16c by replacing a hardlink, and the old v15a raw hash was
verified unchanged. Publication will retain the long CPU conversation failure.

### v0.8 published and independently downloaded

Published tag v0.8.0-experiment at `171f9a5fcc8c0166047b61e880d2c72a4a511f7b`,
including archives for v16a, v16b and v16c. Downloaded archive sizes/digests and
the checksum file match the public assets. All three raw/config hashes, ZIP
CRCs and frozen-source manifests match a clean tagged checkout. Extraction
leaves that source checkout clean.

Fresh CPU/CUDA integrity and all eight conditional interventions pass. The
31-turn replay reproduces 31/31 CUDA and 28/31 CPU with exactly the same three
CPU failures. `results/publication-v0.8.json` explicitly marks the conversation
as not all passed. No saved weights or frozen compiler were changed after the
fresh seed or publication tests. A bounded read-only dialogue trace is being
captured to localize the CPU difference before the next model experiment.

The completed matching CPU/CUDA traces replay the same conversation and token
prefix (3,398 tokens), copying only the final 100 positions for diagnostics.
CUDA records both rules; CPU loses the second rule at position 3,374. Its
newer literal's rule-context flag is 0.9990235567 versus 1.0 for the older
literal. The 131,072 ranking coefficient gives that older flag about 127.98
score advantage, exceeding the newer literal's 72-point recency advantage.
The head returns the older clause coordinate, which the scope guard correctly
rejects. These score contributions are derived externally from captured
residuals and frozen coefficients, not measured fused attention scores.

`results/rule-phase-diagnosis-v16c.json` summarizes the traces. This localizes
the next test-first improvement: vary positions/order of multiple conditional
rules, baseline the frozen version, then investigate a shared Boolean
saturation step before literal ranking. No frozen v16c source or weights have
been altered. The trace helper only replays prompts and reads activations;
diagnostic timings include capture overhead and are not speed benchmarks.
