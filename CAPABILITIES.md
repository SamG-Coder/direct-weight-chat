# Capabilities and limitations

## Current v11a

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
