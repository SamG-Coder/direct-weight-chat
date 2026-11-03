# Capabilities and limitations

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
