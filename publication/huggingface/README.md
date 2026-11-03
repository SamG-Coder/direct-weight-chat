---
license: mit
language:
- en
tags:
- pytorch
- experimental
- untrained
- neural-circuits
- analytically-constructed
- custom-architecture
model_name: direct-weight-chat v17a compact
---

# direct-weight-chat v17a compact

**Analytically constructed weights — no training**

[Architecture and construction](https://github.com/SamG-Coder/direct-weight-chat/blob/main/ARCHITECTURE.md)
· [Benchmark results for every version](https://github.com/SamG-Coder/direct-weight-chat/blob/main/results/README.md)
· [Matching v17a source code](https://github.com/SamG-Coder/direct-weight-chat/tree/83a2f03afa98075286e7ec82b3e5c53f29cd4ceb/versions/autoregressive_v17a)
· [Compact runtime](https://github.com/SamG-Coder/direct-weight-chat/blob/main/chat1gb_autoregressive/compact_engine.py)

Finished research experiment. The statement above describes how these parameter
tensors were produced; it does not claim that useful chat emerged without training.

> **We did not achieve useful, open-ended chat under these constraints.**
> This is a research artifact with narrow, compiled capabilities. It is not a
> general-purpose language assistant. v17a is the final checkpoint; the experiment
> ended on 4 November 2026.

The default download is now the **lossless compact edition** of the final v17a
checkpoint: **1,528,592 bytes of packed weights**. It stores 995,905
nonzero float32 coefficients and their positions, plus six negative-zero entries
needed for exact reconstruction. Most connections in the logical transformer layout are implicit zeros.

This exact size refers to `weights.sparse.zip`. Configuration and runtime source
files are additional downloads.

**This is the same frozen circuit, not a newly trained or more capable model.**
The parameters were written analytically through reusable neural-circuit
compilation. There was no gradient training, pretrained tensor transfer,
distillation or downloaded training dataset.

The compact loader retains **12,952,916 bytes of weight tensors** after
inference and reconstructs one requested dense tensor at a time. Total RAM/VRAM is higher because
Python/PyTorch, temporary tensors, activations and conversation state also need
memory. The logical architecture and compute dimensions remain unchanged.
See [compact format, measured memory and output comparisons](https://github.com/SamG-Coder/direct-weight-chat/blob/main/publication/COMPACT.md).

The original dense weights remain in the
[historical v0.9 GitHub release](https://github.com/SamG-Coder/direct-weight-chat/releases/tag/v0.9.0-experiment)
and Hugging Face revision `4a5bf383cbc6df4545f8f04fe8d51be208c29811`.
They are no longer part of the default download. File size is not a capability claim.

## Source code and complete experiment

- [GitHub repository](https://github.com/SamG-Coder/direct-weight-chat)
- [Final README and conclusion](https://github.com/SamG-Coder/direct-weight-chat#final-outcome)
- [Every version's test results](https://github.com/SamG-Coder/direct-weight-chat/blob/main/results/README.md)
- [Frozen v17a source](https://github.com/SamG-Coder/direct-weight-chat/tree/83a2f03afa98075286e7ec82b3e5c53f29cd4ceb/versions/autoregressive_v17a)
- [Architecture](https://github.com/SamG-Coder/direct-weight-chat/blob/main/ARCHITECTURE.md)
- [Research log and failures](https://github.com/SamG-Coder/direct-weight-chat/blob/main/RESEARCH_LOG.md)
- [Fresh-download release verification](https://github.com/SamG-Coder/direct-weight-chat/blob/main/results/publication-v0.9.json)

The `runtime.zip` in this model repository includes the matching Python runtime,
compiler source, compact loader, local chat UI, verification code and license. It is provided here so that
the model can be run without separately cloning GitHub. This is a **custom
PyTorch runtime**, not a Transformers `AutoModel` checkpoint or a GGUF model.
There is no hosted inference service attached to this upload.

## What it can do in the tested grammar

- Generate tokens through a normal transformer forward pass and output head.
- Use facts supplied in the current conversation, including unfamiliar entity
  names, explicit negation and later corrections.
- Answer restricted questions about attributes and relationships, including
  bounded category inheritance and simple Boolean query combinations.
- Apply signed, ground `If ... then ...` rules through at most three links along
  one latest-rule path; missing premises can result in `I do not know.`.
- Perform bounded two-operand signed addition/subtraction. The intended operand
  range is -999 to 999; this is not a general expression parser.

Inference uses tokenization, tensor operations, KV caching, argmax selection and
detokenization. It has no Python answer engine, response-selection fallback,
complete-answer bank or weight updates. The restricted language grammar and
atomic facts are nevertheless manually authored and compiled into weights.
Token-by-token output does not establish language understanding.

## Final results and limitations

| Measured suite | CPU | CUDA |
|---|---:|---:|
| Long conversation replay | 31/31 | 31/31 |
| Ground conditional development | 269/293 | 269/293 |
| Rule-position development | 160/162 | 160/162 |
| Fresh rule-position combinations after source freeze | 160/162 | 160/162 |
| Additional scope/dialogue guards | 29/36 | 29/36 |
| Alternative proof-search baseline | **29/104** | **29/104** |

The original 36 matched CUDA regression suites total 5,959/6,105, with no output changes
from v16c. Historical suites repeat structures. Fresh combinations use known,
author-designed generators; these are not independent tests of general language
understanding. Exact outputs and timings are linked above, including failures.

**Unsupported or unreliable:** useful general conversation, unrestricted English,
creative writing, code generation, general knowledge, multiplication/division, three-or-more-operand expressions,
conditional arithmetic or numerical if/else,
longer proof chains, searching alternative proofs, variable binding, compound
rule premises and open answers derived from rules. Some unsupported paraphrases
can be misread as assertions and produce false positives. An abstention on a
creative request is not successful creative generation.

The practical goal was not met. This experiment does not prove mathematical
impossibility for every conceivable architecture. It preserves what this
particular direct-construction attempt achieved and where it failed.

The compact loader also matches every text output, token ID and EOS decision
in all **6,371 cases across 38 retained CUDA suites**, including all 223 failures.
All **788 CPU cases across six suites** also match exactly, including all 110
failures. This measures storage/runtime parity, not new language ability. See the linked
[compact verification](https://github.com/SamG-Coder/direct-weight-chat/blob/main/publication/COMPACT.md)
for CPU comparisons and memory measurements.

## Download and run

Download the files with the Hugging Face CLI, or use the Files tab. Downloading
is a separate publication step; construction and inference run offline.

```bash
hf download SamGCoder/direct-weight-chat-v17a --local-dir direct-weight-chat-v17a
cd direct-weight-chat-v17a
python -m zipfile -e runtime.zip .
python -m chat1gb_autoregressive.compact_chat --model-dir . --device auto
```

Use an environment with **PyTorch already installed**. NumPy is also needed for
rebuilding weights, but not for inference. Original verification used Python
3.11.9 and PyTorch 2.8.0+cu128 on an RTX 5080, with separate CPU runs. Construction
used NumPy 2.5.3. CPU inference is supported. The runtime blocks outbound Python
socket connections at the application level; it is not an OS firewall.

The chat command keeps context between turns. `/reset` clears context; `/quit`
exits. The fixed window is 5,048 tokens, with 128 dynamic vocabulary slots for
unfamiliar spellings. Reset the conversation if either capacity is exhausted.

For one prompt:

```bash
python -m chat1gb_autoregressive.compact_chat --model-dir . --device cpu --prompt "149 - 370"
```

Or load the same runtime from Python after extracting `runtime.zip`:

```python
from chat1gb_autoregressive.compact_engine import Model

model = Model(directory=".", device="auto")
print(model.generate("A kelv can fly. What can a kelv do?")["text"])
print(model.generate("A kelv cannot fly. Can kelv fly?")["text"])
```

These calls execute the uploaded weights. No model or dataset download occurs
inside the runtime. The illustrative prompts are restricted examples, not
evidence that arbitrary conversation works.

## Integrity and provenance

`SHA256SUMS.txt` records the current file hashes. `compact.json` describes the
sparse artifact; `dense-manifest.json` preserves the original construction
metadata and source hashes. `weights.sparse.zip` is read directly by the compact
loader; do not extract it into a dense checkpoint.

Compact archive SHA-256:

```text
4739a3511f5c4b9170f5d09524bd8105f5853a7999cbeefdea7a8cbae01d737d
```

An independent reconstruction of every tensor from the serialized archive
reproduces the original raw `weights.f32` SHA-256:

```text
5cc52f3793a1546c620195c110a7d1c3a2f8d3417cb72f193003589b16e8e0d2
```

The tensor runtime SHA-256 is:

```text
f56d29cd23ba2b3018c4ba6a7ad6928dcb28c437e213929802aa4cba90329e10
```

The frozen compiler reproduces the exact raw/config hashes. Existing integrity
checks and ablations for the original dense checkpoint passed on CPU and CUDA. v17a stabilizes rule-phase flags
before sharp attention; restoring the older gate recreates a measured CPU failure.

Human and AI-assistant prior knowledge entered the authored algorithms and
atomic facts. “No pretrained weights” does not mean the design was created
without prior knowledge. No pretrained tensor values were transferred.

During development, authors revised the circuit code in response to tests. That
manual, test-guided engineering is distinct from a training loop: the provided
compiler performs no gradient updates, loss minimization or automated fitting
of parameters to examples. It writes coefficients from the supplied algorithms,
grammar and 40 authored entity records. A publication-time rebuild of the frozen
compiler reproduced the exact original weight hash; the compact archive preserves
those bytes losslessly. See the
[construction audit](https://github.com/SamG-Coder/direct-weight-chat/blob/main/publication/CONSTRUCTION-AUDIT.md)
for the evidence and limits of this claim.

MIT licensed. This model is provided for inspection and reproducible experiments,
not for factual advice or deployment as a dependable assistant.
