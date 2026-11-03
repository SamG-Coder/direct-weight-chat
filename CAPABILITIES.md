# Capabilities and limitations

## Current v7a

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
