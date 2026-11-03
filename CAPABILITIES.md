# Capabilities and limitations

## Current v17a

- Rule-position development improves from v16c 154/162 to 160/162 on CPU;
  CUDA stays 160/162. The 31-turn long conversation improves from 28/31 to
  31/31 CPU and remains 31/31 CUDA. The same two semantic stress failures
  remain: four-link inference and an older successful alternative rule.
- A shared saturating Boolean primitive repairs the measured phase-amplitude
  error before attention ranking. The actual compiled MLP passes 220/220
  synthetic perturbation checks on each device, versus 164/220 for v16c.
  Maximum measured output error is below 0.000001. The test's intended Boolean
  inputs vary by up to 1/8; it does not establish arbitrary error tolerance.
- Restoring only the old context gate makes the isolated CPU failure return.
  All saved weights remain unchanged during this intervention.
- All 36 matched CUDA suites retain identical outputs: 5,959/6,105. Four
  matched CPU suites gain nine passes with zero regressions. These totals
  repeat historical structures. Short conditional/guard scores remain
  269/293 and 29/36 on both devices.
- The existing 27 integrity check groups and eight conditional interventions
  pass on CPU and CUDA. The runtime, tokenization and tensor layout are
  unchanged. Frozen-source rebuilding reproduces exact raw/config hashes.
- After source commit `83a2f03afa98075286e7ec82b3e5c53f29cd4ceb`, fresh seed
  2165866013 scores 160/162 on each device. The known generator supplies new
  combinations/offsets and repeated fixed structures. No frozen weights were
  retuned after these results; this is not independent language evaluation.
- A new 104-case alternative-proof suite scores 29/104 on both devices.
  It includes two to eight alternatives, mixed branches, declaration order,
  corrections and dialogue. This is a baseline for the next capability,
  authored after v17a froze; it is not a held-out language score.

The repair improves measured numerical reliability. It does not add proof
search, deeper reasoning, variable binding, compound rule premises, open
answers from rules, broader mathematics, creative writing, code generation
or useful general conversation. Unsupported paraphrases can still produce
false assertions. The research objective remains active.

v17a has exactly 250,000,000 float32 parameters / 1,000,000,000 bytes, with
**995,905 nonzero parameters (0.398362%)**. Raw weight SHA-256:
`5cc52f3793a1546c620195c110a7d1c3a2f8d3417cb72f193003589b16e8e0d2`.

## Historical v16c

- Ground conditional development improves from v15a 152/293 to 269/293 on
  CPU and CUDA. All 49 positive predicate pairs and 49 signed applications
  pass. Missing/false premises, corrections, exact identity isolation and
  chains up to three rule links pass under the controlled grammar.
- Additional author-designed scope/dialogue guards score 29/36 on both
  devices. Their seven failures include unsupported compound rules,
  paraphrases that become false assertions, and open rule-derived answers.
- All 33 matched CUDA suites retain every v15a pass: 5,499/5,619, with 130
  improvements and zero regressions. Repeated cases/structures across suites
  are included. Earlier post-freeze sets are now regression evidence.
- v16a/v16b are preserved failed candidates. v16a corrupts some open answers;
  v16b still creates a false rule row at an atomic entity token. Read-only
  activation traces identify the field collisions; v16c uses question-mode
  isolation and a separately bounded rule-terminator penalty.
- The ordinary tensor runtime is unchanged. Rules and proof goals are neural
  activations/KV state, never a Python rule database or a parameter update.
  The frozen compiler reproduces the exact raw/config hashes.
- Eight conditional interventions pass on CPU and CUDA, including each rule depth,
  premise evidence, record validity, object identity, signed premises and the
  typed-record guard. The original ineffective object intervention remains
  recorded; the corrected intervention preserves its match-score reward.
- The existing integrity/intervention suite passes on both devices. Legacy
  CPU remains 185/185. The 31-turn long conversation passes 31/31 CUDA but
  28/31 CPU: three later conditional answers become unknown. The earlier
  23 turns retain their CPU passes. Long-context CPU rule inference remains
  an unresolved numerical boundary, not a completed reliability claim.
- After source freeze `9c84c6a`, conditional seed 1711846539 scores 269/293
  on CPU and CUDA. Fresh names/combinations use a known author-designed
  generator; fixed boundaries repeat. The frozen model was not retuned after
  these results, and they do not establish independent language generalization.
- Public v0.8 download validation checks all three archive digests/sizes,
  checksum entries, raw/config hashes, CRC and frozen source. A clean tagged
  checkout passes integrity and all eight conditional interventions on both
  devices, reproducing 31/31 CUDA and 28/31 CPU conversation scores. The same
  CPU failures remain explicit in `results/publication-v0.8.json`.

The circuit follows one latest matching rule at each step. It cannot find an
older successful alternative when a newer rule has no proof, traverse more
than three links, bind variables, evaluate compound rule premises or generate
open answers from rules. General conversation, creative writing and code
generation remain unachieved. Existing arithmetic, mixed query precedence,
disjunctive assertion and multiple-parent limitations also remain.

v16c has 250,000,000 float32 parameters / 1,000,000,000 bytes, with **995,881
nonzero parameters (0.3983524%)**. Raw weight SHA-256:
`385dff256910dec6eba802d3930452c2963e85223bbaa54251d2931b9b86b1dc`.

## Historical v15a

- The new 136-case binding development suite passes on CPU and CUDA. Frozen
  v14b scores 133/136 CPU and 136/136 CUDA. The three CPU repairs include two
  missed second objects and a false fact assigned to the previous subject.
- The same retrieval primitive serves lexical predicates and subject frames.
  Attention selects by validity/recency; following neural gates check clause
  membership. Tiny clause-coordinate errors no longer rank competing subjects.
- All 29 existing CUDA suites retain their v14b passes. Older post-freeze sets
  are explicitly labeled regression evidence for this candidate. Earlier
  unsupported language/reasoning structures remain failures.
- CUDA integrity and all interventions pass. The original ineffective recency
  probe remains recorded alongside three-subject controls that expose missed
  and falsely bound facts when recency is removed. The inference runtime is
  unchanged; all diagnostic interventions restore the runtime tensors.
- CPU regression checks also repair the original context failure (125/126 to
  126/126) and the long conversation (19/20 to 20/20). The old fresh inputs are
  now regression evidence. No previous CPU passes are lost in those checks.
- CPU integrity and all interventions pass. The local address audit remains
  exact at all 45,432 positions on both devices. A repaired CPU activation trace
  selects the newer subject with its expected 40-point recency advantage.
  Archived source reproduces the exact raw/config hashes. The candidate is
  committed before selecting a new binding seed.
- After freeze commit `a583d2b`, binding seed 192095465 passes 136/136 on CPU
  and CUDA. These are new names/combinations and eight random offsets from a
  known generator. The fixed offsets and language structures repeat; this is
  not independent evidence of general understanding. No frozen source or
  weights were retuned after the fresh results.
- The published v0.7 checkpoint passes download/checksum/source/raw/CRC
  verification. From a clean tagged checkout, CPU and CUDA each pass integrity,
  all interventions and a 23-turn long conversation with three-subject binding,
  isolation and correction. [Publication evidence](results/publication-v0.7.json)
  preserves all exact outputs and timings.

This is a numerical reliability improvement to the controlled grammar. It does
not add conditional-rule reasoning, disjunctive assertions, mixed-precedence
parsing, open multi-value answers, creative writing, code generation or useful
general conversation. Those capabilities remain unachieved.

v15a has 250,000,000 float32 parameters / 1,000,000,000 bytes, with **961,697
nonzero parameters (0.3846788%)**. Raw weight SHA-256:
`6386f31062760a2f5d81c1197c734af7ddd73f888fe0d44c09180de7f026ad81`.

## Historical v14b

- Context/position development scores 126/126 on CUDA, versus v13d 61/126.
  Assertions return to 382/390 after the v14a correction regression. The
  nine-head address audit has zero errors out of 45,432 on CPU and CUDA.
- Bounded positional phases replace quadratic local-address scores. A shared
  saturation step then removes fact-code magnitude bias before proposition
  retrieval. These changes preserve the 5,048-position window and ordinary
  tensor runtime; they do not add a host parser or answer fallback.
- Phase-removal interventions change the associated response. A controlled
  identity-magnitude perturbation leaves corrections intact with saturation;
  removing that step makes the older assertion win. Long-prefix cache/full
  decoding matches in the integrity check.
- All 26 matched suites retain their v13d passes. Both older boundary suites
  now score 76/76 on CUDA. The v14a history retains 48 correction regressions
  across three reports; all are repaired in v14b. Frozen source reproduces
  the exact raw/config hashes.
- After freeze commit `b3b6e1b`, context seed 2063643282 scores 126/126 on CUDA
  and 125/126 on CPU. CPU case `context-086` has a 4,720-token prefix and loses
  the second coordinated object of an unfamiliar subject: `I do not know.`
  instead of `Yes.`. This failure is retained; v14b was not retuned.
  These are fresh combinations from a known generator, not independent tests
  of broad language ability. The six near-capacity requests repeat.
- A separate 20-turn publication smoke conversation, after 3,000 initial tokens
  of declarative distractors, scores 20/20 on CUDA and 19/20 on CPU from a fresh
  tagged clone and downloaded checkpoint. CPU turn 2 loses the second object
  of a changed subject at a 3,036-token prefix. This corroborates the fresh
  context failure; no frozen weights were retuned. Both downloaded archives
  verify. Downloaded v14b passes integrity and all interventions on both devices.
  See [publication evidence](results/publication-v0.6.json).

This repairs the tested numerical failures, not every possible long-language
problem. The existing controlled grammar still fails disjunction, conditional
rules, mixed query precedence, repeated negation and open multi-value answers.
Useful open-ended conversation, creative writing and code generation remain
unachieved. Failed v14a is retained with its correction regressions.

v14b has 250,000,000 float32 parameters / 1,000,000,000 bytes, with **960,705
nonzero parameters (0.384282%)**. Raw weight SHA-256:
`61deda2326ad355a3b2921e608e37de3e47e7949f445753cb8a0b78ad7288646`.

## Historical v14a

- Context/position development improves v13d 61/126 -> v14a 126/126 on CUDA.
  The suite moves existing generation, arithmetic, compound facts, corrections,
  negation, subject binding and inheritance across the context window. It
  includes six requests near capacity, with prefixes up to 5,007 tokens.
- A full-window diagnostic checks all nine local lookback heads: v14a has zero
  incorrect addresses out of 45,432 on both CPU and CUDA. The v13d baseline has
  15,900 CUDA and 7,491 CPU errors, with deviations up to three tokens.
- Two bounded positional phases replace quadratic scores. The tensor layout,
  5,048-position capacity and generic inference runtime remain unchanged. No
  new linguistic grammar or answer rules are added by this change.
- All 26 suites were rerun. Assertion corrections regress by 16 cases in each
  of three reports, lowering assertions to 366/390. A trace attributes this to
  fact-code magnitude differences amplified by sharp proposition matching.
  v14a is preserved as a failed candidate despite its repaired positional scores.

These checks address the demonstrated positional failures. They do not establish
correctness for arbitrary long conversations or expand the existing language
and reasoning grammar. Disjunction, conditional rules, mixed query precedence,
repeated negation, open multi-value answers, creative writing, code generation
and useful flexible conversation remain unachieved.

v14a has 250,000,000 float32 parameters / 1,000,000,000 bytes, with **959,817
nonzero parameters (0.3839268%)**. Raw weight SHA-256:
`2dd59a1e327deaa42f10d5cb18152052b52e19358dc5dd2bf1ad2acfc4a96915`.

## Historical v13d

- Assertion development scores 382/390 (381/381 targeted grammar; 1/9 structural
  stress), versus v12b 158/390. The boundary audit improves v13a 41/76 -> v13d
  65/76 on CUDA. All 48 short boundary cases pass; 17/28 context-length cases pass.
- Shared coordinated assertions bind subject, predicate and polarity neurally.
  A following-token gate prevents `X has wings and Y needs water` from storing
  Y as an object of X. Predicate changes, object correction and inheritance
  remain supported within the tested grammar.
- Question and directive clauses do not create the observed unasserted facts.
  The Tell-me-whether checks demonstrate assertion isolation, not general
  understanding of indirect questions or arbitrary instruction following.
- Failed v13b (184/390 assertions, 41/76 boundaries) and intermediate v13c
  (382/390, 61/76) remain archived with exact results. v13b misaligns clause
  coordinates at punctuation; v13c loses low bits in a recycled float32 field.
  v13d preserves the coordinate scale while keeping the ideal attention score.
- After freeze commit `4382108`, assertion seed 3178044486 scores 382/390 on
  CPU and CUDA. Boundary seed 3284145273 scores 65/76 on CUDA and 64/76 on CPU;
  all 48 short cases pass on both. The device difference is in a long-context
  isolation case. These are fresh names/combinations from known generators,
  not independently designed generalization evidence. No frozen weights are tuned.
- All 22 matched v13a/v13d suites retain their earlier passes. The conditional
  stress regression against v12b remains in the direct comparison. All integrity
  checks and neural interventions pass; frozen source reproduces exact raw and
  config hashes. A separate 20-turn publication smoke conversation passes locally
  and on both CPU/CUDA from a fresh tagged clone and downloaded checkpoint.
  [Publication verification](results/publication-v0.5.json) checks all four archives.

The long-context audit fails some cases beginning at 2,048 padding tokens.
The 5,048-token storage window is **not** a demonstrated accuracy window.
At long positions, answers can change between CPU and CUDA or stop mid-sentence.
Disjunction, conditional rules, multiple class parents, comma lists, repeated
negation, open multi-value answers and mixed query precedence remain failures.
Useful flexible conversation, creative writing and code generation are unachieved.

v13d has 250,000,000 float32 parameters / 1,000,000,000 bytes, with **929,381
nonzero parameters (0.3717524%)**. Raw weight SHA-256:
`ff0b1e1af0909fc6237bdcfc7fc41e6df590cd678bc9552705bb3a943e2b549d`.

## Historical v13a

- Coordinated assertions improve from v12b's 158/390 to 382/390, including
  all 381 cases within the targeted grammar and 1/9 broader structural stresses.
- Shared positive/negative predicates work across all seven relations. Explicit
  verb changes replace the predicate and polarity; local not negates one object.
  Subject changes, object correction, object determiners, up to three inherited
  class links and multi-turn combinations pass the new development cases.
- Question clauses are excluded from assertion storage, including declarative
  fragments following AND. This repairs all four observed question-memory errors.
  It is bounded clause-head recognition, not general intent understanding.
- All nineteen other existing suites were rerun. Legacy remains 185/185; graph
  75/79; arithmetic 115/120; polarity 230/233; signed 153/159; truth 236/239;
  query scope 265/269; modifiers 78/81. Composition improves to 398/404.
- Nineteen matched v12b/v13a reports contain one regression: a conditional with
  its premise now abstains instead of accidentally asserting its conclusion.
  The matching no-premise case is repaired. Conditional reasoning itself remains
  unsupported. Exact changes are in [the comparison](results/regressions-v13a.json).
- Integrity, causal/cache parity, offline guard and all earlier interventions
  pass. Four new ablations show the roles of continuation validity, predicate
  binding, question exclusion and clause isolation. The runtime is unchanged.

Disjunctive assertions can still incorrectly assert their first alternative.
Conditional rules, multiple class parents, comma lists, longer negation forms,
double negation and open wh-answers listing multiple values remain failures.
Mixed query AND/OR precedence, pronouns, broader instruction scope and all
previous arithmetic/depth limits remain. Useful flexible conversation, creative
writing and code generation are unachieved.

v13a has 250,000,000 float32 parameters / 1,000,000,000 bytes, with **928,135
nonzero parameters (0.371254%)**. Raw weight SHA-256:
`dcd4cdeff8c750fc95a9071e3de833a8849b4874d8dfda984dceb885197858af`.

## Historical v12b

- Legacy 185/185; graph 75/79; arithmetic 115/120; polarity 230/233;
  signed operands 153/159; truth 236/239; query scope 265/269;
  composition 397/404; request modifiers 78/81.
- Applies strong three-valued AND/OR to independently evaluated propositions.
  False decides an AND expression; true decides an OR expression; unknown
  evidence is otherwise retained. Tested homogeneous sequences contain two
  through five atoms, including independent negation and unknown properties.
- Supports shared or distinct subjects and predicates, inherited properties,
  corrections and dialogue within the existing controlled grammar. This is
  query composition; compound fact assertions are still unsupported.
- Clause IDs isolate the active request. Recognized request heads can follow
  bounded please/briefly/simply/just prefixes. An unsupported head stays
  unsupported after a modifier. These are lexical grammar rules, not general
  intent understanding or arbitrary instruction following.
- Fresh seeds selected after the v12b source freeze reproduce composition
  397/404 and modifiers 78/81 on CPU and CUDA. These use known generators;
  modifier evaluation adds 20 new arithmetic inputs while its 61 structural
  cases repeat. No frozen candidate was tuned after those runs.
- v12a's Please calculate regression is preserved in its reports. v12b repairs
  all 106 affected cases across four arithmetic reports. Three composition
  regressions against v11a remain: mixed precedence, compound assertions and
  partial-question instructions. Their earlier partial-answer passes are not
  evidence of those capabilities. All exact outputs are retained.

Remaining failures include mixed AND/OR precedence, malformed trailing
connectives, repeated negation, general pronouns, compound assertions, unsupported
framing and requests to answer only part of a question. The original open
wh-question multiple-action bug remains. Longer graph chains and the earlier
arithmetic limits remain. Correct abstention on a creative request does not
demonstrate creative writing; useful flexible conversation and code generation
are still unachieved.

v12b has 250,000,000 float32 parameters / 1,000,000,000 bytes, with **873,633
nonzero parameters (0.3494532%)**. Raw weight SHA-256:
`07349cb626864c4ea77e32300af5670a6de69ee612662ec5e9d2d2cdba43cea5`.

## Historical v11a

- Legacy 185/185; graph 75/79; arithmetic 115/120; polarity 229/233;
  signed operands 153/159; object-specific truth 234/239; query scope 262/269.
  Raw outputs, timings and preserved failures are in the [results table](results/README.md).
- Queries individual subject/predicate/object propositions, distinguishing an
  explicit positive fact, an explicit negative fact and absent evidence. The
  latest correction for the same object wins without discarding another object.
- Binds all seven existing predicates in a restricted inversion grammar:
  can, need, have, produce, class, trait and location. Three causal lookbacks
  support the tested article, preposition and not forms. Query `not` is treated
  as logical proposition negation; missing information remains unknown.
- Tested facts can be inherited through up to three unfamiliar class links.
  A question is not added as an assertion. These behaviors compose with the
  tested fact corrections and signed arithmetic across user turns.
- v10a's post-freeze truth score is 229/239 on CPU and CUDA. v11a's post-freeze
  query-scope score is 262/269 on both. Generators and syntax were known before
  freezing; this is not an independently designed language/general-intelligence
  evaluation. No frozen weights were adjusted after fresh-seed evaluation.

**Remaining failures:** open wh-questions still select one value per relation,
so the earlier permitted-action/denied-action regression remains. A subsequent
instruction to ignore an already parsed question is disregarded (a v10a
regression against a v9a abstention). Conjunctions, general pronouns, quantifiers,
repeated negation and unsupported paraphrases fail. One compound test passes
by ignoring part of the question; contrasting failures rule out a conjunction
claim. No additional v11a regressions appear in the matched suites.

The arithmetic limits and failures described below remain. Useful open-ended
conversation, creative writing, program generation and broad general reasoning
are unachieved. This is still a sparse, hand-compiled neural program with a small
authored lexicon, not 1 GB of useful learned knowledge.

Fresh-download CPU/CUDA validation each scores 11/12: all mixed-conversation
checks pass, but `Write a poem about rain.` returns factual rain sentences
instead of the expected abstention. This is preserved as a failed creative-
request check in the publication evidence.

v11a has 250,000,000 float32 parameters / 1,000,000,000 bytes, with **744,215
nonzero parameters (0.297686%)**. Raw weight SHA-256:
`f15e70fdeddf91f94bfac13a505b1f14157a9dae79bb4388febf1c62a2a637b4`.

## Historical v9a

- Legacy 185/185; graph 75/79; old arithmetic 115/120; polarity 227/233;
  signed operands and validity 153/159. Exact outputs and stress failures are
  retained in the [results table](results/README.md).
- Represents negative predicate assertions across seven relation types. Tested
  forms include cannot/can't/can not, is not/isn't, and does not/doesn't with a
  base verb. Corrected facts and inherited negative properties retain polarity.
- A negative category assertion does not license inheritance from that category.
  Articles and multi-sentence positive/negative answers are generated from
  semantic fields through the transformer.
- Adds/subtracts two signed integers in [-999,999], distinguishes unary signs
  from binary operators, and retains tested prefix addition. A neural operand-
  count check rejects the tested expressions with too few or too many numbers.

Frozen v8b's fresh polarity cases scored 227/233 on CPU and CUDA. Frozen v9a's
fresh signed cases scored 153/159 on both, including all 120 newly generated
random signed combinations. Those generators and structures were known in
advance; this is not an independently designed language benchmark.

**Retained regression:** `X can fly. X cannot swim. What can X do?` selects the
last denied action instead of the earlier permitted one. The context stores
multiple assertion events, but the current answer retrieval selects one object
per predicate. v7a accidentally passed this case by ignoring the denial. Explicit
yes/no queries, query/instruction negation and double negation still fail.

**Remaining arithmetic errors:** malformed expressions such as `5 + * 3` and
`Calculate 3 4.` still produce numbers. Operand-count rejection is not a complete
syntax validator. Multiplication, division, parentheses, arbitrary precision and
multi-step evaluation remain unimplemented. A chained mathematical-answer test
still fails even when the model now abstains correctly in a separate validity test.

Creative writing, program generation, natural open-ended conversation, broad
world knowledge and general logical reasoning remain unachieved. The earlier
pointer/context/grammar limits still apply. No research milestone is declared
complete from these narrow scores.

v9a has 250,000,000 float32 parameters / 1,000,000,000 bytes; **720,343 parameters
(0.2881372%)** are nonzero. Raw weight SHA-256:
`397b76ee15d876d6d28d4f30ce59dd4d40069029018887133a173a4a67a74f53`.

## Historical v7a

- Legacy regression: 185/185. Graph development: 74/79. Arithmetic: 114/120.
- Follows up to three unfamiliar class links to a terminal property, across
  six property relations, with shuffled fact order and irrelevant facts.
- Uses nearest available properties along the chain and the latest assertion
  for a matching entity. Direct assertions override static facts for the queried
  subject. Known category leaves can supply authored properties.
- Handles the tested multi-turn corrections, unknown links and cycles without
  inventing a property. It does not implement a general cycle detector or
  unrestricted graph search; it executes four graph lookup steps.
- Adds and subtracts two nonnegative integers from 0 to 999. Subtraction emits
  negative results as a minus token and a magnitude token.

v6c's post-freeze graph cases scored 74/79 on both CPU and CUDA. v7a's post-freeze
arithmetic cases scored 114/120 on both, including all 80 new random subtraction
and 20 new addition combinations. The known generator/syntax limits these claims.
No code or weights were tuned against those post-freeze outputs.

Latest raw weight SHA-256:
`85a6b709cead89d261a1b1d412ad8e64bac19d4e12b9f9ec6bc947fff1869c2d`.
Only **717,554 parameters (0.2870216%)** are nonzero. All 250M float32 parameters
belong to standard transformer tensors, but most connections are unused.

Still fails: longer graph chains, unsupported paraphrases, negation, arbitrary
relations/events, comparisons, conditionals, plurals, creative writing, coding,
general conversation, algebra, fractions, multiplication and division. Negative
operands and chained subtraction can return incorrect numbers (for example,
`-5 - 3` and `10 - 3 - 2`); signed outputs do not imply signed-input support.
Knowledge is still seeded by 40 authored entity records and seven relation types.
There is no general pronoun/variable binding or durable memory outside the token
context. Static ancestor facts can have precedence interactions with explicit
ancestor facts that have not been exhaustively tested.

The research milestones remain partial. In particular, broad language generation,
code generation and useful open-ended conversation have not been achieved.

## Historical v5

Measured on the restricted development suite: 15/15 functional cases, 40/40
compositions containing new pointer words, 120/120 bounded additions, 5/5
multi-turn additions and 5/5 multi-turn fact queries. The separate short CPU
suite passed 39/39. These are narrow exact-output tests, not evidence of broad
language understanding.

Works within tested syntax: `X is a Y`, `X can V`, `X needs Y`, article selection,
latest explicit fact, one-level inheritance from a fixed known class, and sums
of two non-negative integers at most 999. Temporary facts live in the token
context. Weights do not change during conversation.

Fails: arbitrary paraphrases, negation, multi-hop novel classes, poetry, program
writing, broad factual questions, general explanations, multiplication and large
operands. Configured context length is 5,048; reliable behavior near its limit
has not been established. Pointer vocabulary has 64 vowel-initial and 64 other
slots; punctuation/unsupported numbers can consume slots. Capacity exhaustion
raises a visible error rather than silently changing meaning.

The model has 18 blocks, width 1024, 16 attention heads, ReLU FFN width 4096 and
vocabulary size 8832. Only 697,392 of v5's 250M parameters are nonzero (0.279%).
No claim is made that 1 GB of useful knowledge has been created.

The legacy selector v0 uses canned responses and does not satisfy the current
research constraints. Its successful routing tests are not conversational or
autoregressive language scores.
