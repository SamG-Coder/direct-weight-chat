# Historical v5 report

This document preserves the original v5 description and test results. The current
model is v7a; see the [root README](../README.md), [capabilities](../CAPABILITIES.md)
and [version table](../results/README.md) for current behavior. Commands below
that refer to the default model now load the current release. Use
`--model-dir versions/autoregressive_v5/model` to run the archived v5 checkpoint.
The original `verify.py` is historical; use `benchmarks.evaluate` for current
versioned results.

# Direct-weight autoregressive chat experiment

This is a concrete attempt to construct a generative neural chat model without
training, pretrained weights, or internet access. It uses a standard causal
transformer forward pass and predicts successive tokens from the prompt and the
tokens it has already emitted. There is no complete-response bank.

The active weight file is **exactly 1,000,000,000 bytes**, containing
**250,000,000 float32 parameters**. The architecture has 18 decoder blocks, width
1,024, 16 attention heads, ReLU feed-forward width 4,096, separate token embeddings
and output projection, and a vocabulary of 8,832 tokens.

**The result is still limited.** The weights encode explicit attention, grammar,
fact-retrieval, and arithmetic circuits. Only **697,392 parameters are nonzero
(0.279%)**; the remaining standard transformer connections are zero. This is a
sparse hand-programmed computation inside a large transformer architecture, not
the capability or independent knowledge of a trained 250M-parameter language
model. Its file size should not be interpreted as evidence of intelligence.

## Run it

Double-click `Chat.cmd`, or run:

```powershell
python -m chat1gb_autoregressive.chat
```

For a single prompt and a record of every next-token decision:

```powershell
python -m chat1gb_autoregressive.chat `
    --prompt 'A dax is a bird. What can a dax do?' --trace
```

Use `/reset` for a fresh context and `/quit` to exit. `-Cpu` selects the CPU.
The launcher uses the existing local PyTorch 2.8.0 CUDA installation at
`python` from your active environment. Nothing is downloaded or installed.

## Actual generated outputs

```text
User: A dax is a bird. What can a dax do?
Model: Dax can fly.

User: A dax is a bird. A dax can swim. What can a dax do?
Model: Dax can swim.

User: An ombre is an observatory. An ombre can shimmer. Explain an ombre.
Model: Ombre is an observatory. It can shimmer.

User: A dax can swim. A dax can fly. What can a dax do?
Model: Dax can fly.

User: 123 + 456
Model: 579.
```

Those new subjects and complete output sentences were not stored in the weights.
The model retrieves individual facts, selects relations, and composes the output
one token at a time. It can inherit an encoded property from a known class, and
an explicit prompt fact can override that property. It can also prefer a later
statement over an earlier conflicting statement. These are limited programmed
operations, not general reasoning.

## What runs inside the network

`engine.py` contains generic token embedding, position embedding, layer
normalization, causal scaled dot-product attention, residual connections, ReLU
MLPs, an output projection, and autoregressive decoding with a KV cache. It does
not import `knowledge.py`, `build.py`, the earlier response-selector project, or
an answer table. There are no Python branches that select sentences from the
meaning of a question, parse facts into a database, or add the requested numbers.

The directly constructed weights implement:

1. Role and position tracking, previous-token pointers, and prompt-subject selection.
2. Recognition of limited fact patterns such as `X is a Y`, `X can V`, and `X needs Y`.
3. Exact subject/relation retrieval from earlier user tokens, with a sentinel for no match.
4. Recency selection and one level of property inheritance from an encoded class.
5. Token-level grammar and the selection or copying of individual subject/object tokens.
6. Addition of two non-negative integers no larger than 999, computed through neural layers.
7. Neural conversion of the selected output representation into next-token logits.

The source knowledge consists of 40 small entity records with atomic relations,
and a small lexical vocabulary. The rest of the vocabulary includes numeric
tokens and 128 fixed pointer-token slots for new input words. The host assigns
new words to those slots and detokenizes them, but does not update any embedding
or model parameter. Prompt facts exist only as activations and KV-cache state.

The grammar is explicitly programmed in the weights. This is not open-ended
language learning: successful outputs generally use its limited clause forms.
It can compose new combinations within those forms, including unfamiliar words
introduced in the prompt, but it cannot freely invent explanations or stories.

## Frozen evaluation

The final checkpoint was held fixed for these tests. The new word forms and
arithmetic operands were selected after the checkpoint was written. They were
evaluation inputs only; no parameter was fitted or updated from them.

| CUDA evaluation category | Correct / total |
|---|---:|
| Core functional checks | 15 / 15 |
| New-word and fact combinations | 40 / 40 |
| Previously untested additions | 120 / 120 |
| Consecutive arithmetic questions in one chat | 5 / 5 |
| Multi-turn facts, overrides, and topic changes | 5 / 5 |
| **Total within the intended capabilities** | **185 / 185** |

The shorter CPU suite passed **39 / 39**. These are functional checks chosen for
the programmed skills, not an independent general language benchmark.

Additional checks passed:

- Exact file size, parameter count, contiguous tensor layout, and SHA-256 integrity.
- No change in prefix logits when future tokens were changed: tested difference 0.0.
- Cached decoding and full-prefix decoding chose the same successive tokens.
- Disabling neural fact retrieval changed `Kelv is a vehicle. It can shimmer.`
  to `I do not know.` for the same prompt.
- Disabling the output projection removed generated text.
- Inference did not import the authored fact/compiler modules.
- The Python socket guard rejected a connection before network I/O.
- The weight checksum stayed unchanged through evaluation.

The recorded median time for a complete response was about 11 ms on the RTX 5080
and 164 ms in the smaller CPU run. Outputs are short and the test sets differ;
these figures should not be treated as a comparison with trained chat systems.

## Retained broad-chat failures

| Request | Actual output |
|---|---|
| Write an original poem about a rainy city. | I do not know. |
| Explain why a fish cannot live on land. | Fish is an animal. It can swim. |
| Write a Python function that sorts a file. | I do not know. |
| Compare three people's heights transitively. | I do not know. |
| What is the capital of France? | I do not know. |
| Do not explain a cat. | Cat is an animal. It can move. |
| Follow two novel class links before inferring an ability. | I do not know. |
| 1234 + 5678 | I do not know. |
| 17 * 24 | I do not know. |

The experiment demonstrates real autoregressive composition and limited context
use without training. **It has not achieved useful general-purpose, open-ended
chat.** Negation, arbitrary instructions, creative writing, programming, and
general reasoning remain failures. The configured positional capacity is 5,048
tokens; reliable operation near that limit has not been established.

## Construction history

There were five analytic builds. I inspected generated results and debugged the
weight-construction code, with no optimizer or statistical fitting:

- First build: causal generation worked; the fact parser sometimes stored articles
  and punctuation as values.
- Second build: corrected fact-value eligibility; exposed averaging of conflicting
  facts and reuse of an old arithmetic operand across turns.
- Third build: added recency, user-turn selection, and property inheritance.
- Fourth build: corrected a reused attention register.
- Fifth build: corrected coefficient merging when a gated value shared its gate's
  input feature. This is the active checkpoint and passed the frozen checks above.

This is therefore an iteratively debugged, tool-assisted experiment, not a
single-response or no-feedback benchmark attempt. Earlier checkpoints and source
snapshots are preserved in `model-v1` through `model-v4` and `evidence/v*-source`.
Each archived checkpoint is another 1 GB; the active model itself is one GB.

## Files and reproduction

- `model/weights.f32`: the active raw little-endian float32 checkpoint.
- `model/config.json`: standard transformer dimensions, tensors, and tokenizer.
- `model/manifest.json`: provenance, hashes, sparsity, and circuit-unit counts.
- `knowledge.py`: atomic facts and lexicon; no complete-response bank.
- `spec.py`, `build.py`: architecture layout and direct numeric construction.
- `engine.py`, `chat.py`: transformer inference and interactive chat.
- `verify.py`: frozen evaluation, causality, cache parity, and ablation checks.
- `evidence/verification-cuda.json`: full test cases, token traces, and failures.
- `evidence/verification-cpu.json`: shorter CPU validation.
- `evidence/conversation.txt`: a recorded session from the final checkpoint.

```powershell
# Run from the repository root
python -m chat1gb_autoregressive.verify --device cuda
python -m chat1gb_autoregressive.verify --device cpu --quick
```

The compiler uses the existing Python 3.14/NumPy installation. Its command is
`python -m chat1gb_autoregressive.build`, and it refuses to overwrite an existing
`model` directory. There are no downloads, external datasets, gradient steps,
pretrained checkpoints, or remote inference calls. The Python socket audit guard
is an application restriction, not a system-wide firewall.

Active checkpoint SHA-256:

```text
87db14c26468b600d64737a840685dee8f0e4896e5ba7e6a7f2b3f798deb943f
```
