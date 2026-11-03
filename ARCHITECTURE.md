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

## v6: a bounded graph algorithm

The answer subject and graph cursor occupy separate residual registers. After
local fact parsing, layers 3, 5, 7 and 9 use seven parallel attention heads to
look up all predicates for the current cursor. A direct prompt assertion can
override the queried subject's authored fact. Later ancestors fill properties
that are still absent. Each step copies the class object's code into the cursor.
Layers 4, 6 and 8 query the permanent atomic fields for that new cursor. Lookup
is causal with respect to the answer token, but does not require assertions to
appear in parent-first order. It is bounded to three class links before a
terminal explicit-property lookup.

For an exact fact match, signed identity bits contribute 14 and the valid
predicate contributes 5. A BOS sentinel scores 18.5; one mismatching identity
bit drops a candidate by 2. Recency breaks ties within the available context.
For static identity lookup, the sentinel's identity is explicitly zeroed so
its threshold of 13 sits between exact score 14 and mismatch score at most 12.

Temporary lookback buffers are recycled into a 112-feature relation buffer.
Nine grammar predicates are saved before that reuse. Decoding coordinates are
reinitialized to suppress floating-point residue before the implicit EOS tie.
The failed v6a and v6b snapshots demonstrate why these lifetime/score details
matter. Disabling later graph attention changes an inherited answer to an
unknown response without any host-side fallback.

## v7: signed subtraction

Attention retrieves the current query's arithmetic operator and numeric
operands. With operands `a,b` in `[0,999]`, the compiled MLP forms
`abs(a-b) = ReLU(a-b) + ReLU(b-a)` and a one-unit ramp for `b > a`. ReLU gates
select addition or subtraction using the operator feature. The existing
piecewise-linear integer decoder maps the magnitude to token-code bits.

The next-token circuit emits a minus token for a negative result, then magnitude
and punctuation in subsequent forward passes. Detokenization removes whitespace
between a generated minus and a generated number. No Python subtraction,
comparison, expression evaluation or answer rendering occurs inside inference.
The large bounded integer-to-code decoder is retained; a reusable arbitrary-
precision arithmetic algorithm remains future research.

## v8: polarity as part of a fact

Each relation slot now contains 14 object-code bits, a presence bit and a
negative-polarity bit. A neural lexical classifier derives article choice from
the selected class object's code, freeing the old vowel field. Four local
attention lookbacks recognize negative auxiliaries and contractions and compile
them into shared subject, predicate and polarity gates. This is still restricted
syntax; it is not a general dependency parser.

Retrieval carries polarity with the selected value. A negative class edge is
stored but cannot advance the graph cursor. During generation, the requested
relation and polarity determine copular negation, cannot, or does/not/base-verb
transitions. The grammar can mix positive and negative clauses. The EOS output
has a one-logit bias, smaller than the 16-logit valid-code margin, so tiny residual
noise cannot break termination when no semantic output code is present.

## v9: signed operands and bounded validity

Lookback fields include number indicators. A unary minus conditionally reverses
the numeric embedding; unary +/- tokens are excluded from binary-operator
selection. Block 10 forms `r = a + b - 2*b*subtract`; the product is implemented
by a gate because `subtract` is Boolean. Block 11 uses ReLU pairs to compute
`abs(r)` and a one-unit ramp for the negative-result bit. Operand bounds test
both sides of [-999,999]. No host arithmetic or expression evaluation is used.

Another neural gate marks numbers after the first in a user turn. Attention
retrieves the second numeric position; it must equal the last numeric position,
and the first must belong to the current turn. This rejects unsupported operand
counts, but does not parse arbitrary syntax or evaluate longer expressions.
Forcing that validity feature to true makes `10 - 3 - 2` produce the incorrect
first/last-operand result `8.` instead of abstaining. Disabling unary sign
changes `-149 - 370` from `-519.` to `-221.`. These interventions are recorded
alongside graph, polarity and accumulator ablations.

## v10a: object-specific proposition lookup

Clause-boundary and auxiliary-inversion detectors bind the subject of `Can X V?`
and `Does X need/have/produce Y?`. Optional articles use the same subject code.
The parser marks a query event separately from assertion events; an auxiliary
question cannot become evidence for its own answer. Request-reset sentinels
separate user turns and the tested following wh-questions. Lexical/role fields
are reused only after their previous readers have finished.

A new head at each of the four graph steps compares 14-bit subject identity,
14-bit queried-object identity, and predicate. A full match scores 33 before
scaling; a one-bit mismatch scores at most 31. The zero-value BOS sentinel
scores 32. Scores are scaled by 8192 and token position breaks ties. Thus the
latest assertion for that exact proposition wins without discarding other
objects of the same relation. Separate heads compare permanent atomic values.
Explicit evidence overrides a matching permanent value; the nearest graph node
with matching evidence supplies the truth value. No match remains unknown.

The retrieved found/polarity bits drive token-level yes/no generation, with
punctuation and EOS emitted in later forward passes. Unknown follows the existing
abstention grammar. There is no Python truth lookup or response selection, and
the inference runtime is unchanged. The model remains an ordinary causal
transformer with the same 250M-parameter layout.

An ablation removes the proposition head and changes `Yes.` to `I do not know.`.
A second removes only the object-key component (with the corresponding sentinel
adjustment): the same prompt then mistakes a later denial of another action for
the queried action and says `No.`. Open wh-questions still use the older one-value
retrieval path; this experiment does not fix that separate multiple-action bug.

## v11a: predicate binding and query negation

Three local inversion markers (can, does, is) and three causal lookbacks compile
questions into the existing seven-predicate representation. Article, preposition
and lexical adjective flags distinguish category, location and trait requests.
The question event carries its own predicate vector, so an object's lexical
query-word flag cannot accidentally choose a different requested operation.
The query predicate reuses REQUEST after its initial lexical reader; no extra
parameters, tokens, host grammar parser or fact records are required.

An independent query-polarity bit records explicit not. For found evidence,
the answer is affirmative exactly when fact polarity and query polarity agree;
otherwise it is negative. A missing proposition stays unknown under either
polarity. The two-bit equality/XOR operation is compiled as Boolean ReLU gates
and applies uniformly to all seven predicates. Negative membership continues
to block traversal, even when a negative question is asked about an inherited
property. In this controlled grammar, not means logical proposition negation;
ordinary conversational readings can be more ambiguous.

Removing only the query-polarity output from block 2 changes a true negative
query from yes to no. The earlier object-identity and retrieval ablations still
change their corresponding answers. Conjunction, general pronoun binding,
quantifiers, repeated negation and later instruction scope remain unsupported.

## v12a: proposition events, composition and clause scope

Clause-boundary attention supplies a monotonically increasing clause ID. The
first request word has an authored lexical support flag; unfamiliar content
words use the existing entity-pointer path. This separates the active request
from older queries, but does not establish general intent understanding.

Block 2 compiles shared-subject proposition continuations, optionally replacing
the predicate or local negation. Block 3 binds the newest complete atom. Explicit
and permanent proposition lookups move to blocks 4/6/8/10, using a separate
cursor lifetime so the existing graph traversal and normal answer subject remain
available. Each atom is evaluated at its own causal token position. Block 11
writes true, false or unknown only at genuine atom positions; later positions
carrying a copied query are not counted as new evidence.

Five block-12 attention heads reduce the presence of true, false, unknown, AND
and OR within the newest clause. A clause position contributes 65,536 per token
of clause age; an existential flag contributes 8,192. The newer clause therefore
dominates any earlier truth flag, while a present flag wins inside that clause.
The next-token MLP implements strong three-valued conjunction/disjunction:
false decides AND, true decides OR, and otherwise unknown is retained. This
works for a homogeneous sequence of any tested length without storing operand
combinations or answers. Mixed AND/OR precedence is deliberately not implemented.

Ablations redirect AND to OR (no becomes yes), remove unknown evidence (unknown
becomes yes), or remove clause IDs (an earlier false contaminates a later unknown
question). Cached/full decoding matches for compound unknown answers. The
inference runtime is unchanged and the raw checkpoint is never modified.

The first request-head filter is too strict: it treats a leading Please as an
unsupported request instead of reading the following verb. v12a is preserved
with this regression before a separate modifier-grammar correction.

## v12b: transparent request modifiers

A reusable lexical-conjunction compiler identifies the request head after a
bounded prefix of please, briefly, simply or just. The modifier itself does not
make a request supported: the following lexical head still determines that flag.
The same prefix grammar is applied to inverted-question subject binding.
Within the four-token lookback, request heads support up to three modifiers;
inverted binding supports two without an article or one with an article. More
complex framing remains unsupported and scored as failure.

This restores the Please calculate arithmetic behavior without allowing Please
write a poem to inherit a previous fact query. It adds no runtime parser, learned
values, test strings or answer entries. Grammar alternatives compile to Boolean
ReLU identity conjunctions. A broad paraphrase beginning Please tell me whether
still returns an incomplete factual answer; the extra incorrect abstention pass
in v12a is not retained as a claimed capability.
