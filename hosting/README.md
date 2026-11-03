---
title: direct-weight-chat
emoji: 💬
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Analytically constructed weights. No training. Browser chat.
models:
  - SamGCoder/direct-weight-chat-v17a
---

# direct-weight-chat · v17a compact

**Analytically constructed weights — no training.** The experiment is finished.
**Useful open-ended chat was not achieved under these constraints.**

This is a live chat interface to the frozen experimental transformer. The model
generates tokens from its weights; the app does not select answers from a list.
It supports narrow fact statements, some corrections and conditional rules,
and addition/subtraction of two signed integers with magnitudes up to 999.
Unknown words use a bounded pointer vocabulary. Arbitrary language, general
reasoning, unrestricted arithmetic and reliable open-ended chat are unsupported.

The exact packed weight artifact is **1,528,592 bytes**. Config, browser runtime
and working memory are additional. The browser reads the same SHA-256 checked
`weights.sparse.zip` published in the model repository; no weights were trained
or modified for this Space.

- [Model card and Python downloads](https://huggingface.co/SamGCoder/direct-weight-chat-v17a)
- [GitHub source](https://github.com/SamG-Coder/direct-weight-chat)
- [Architecture](https://github.com/SamG-Coder/direct-weight-chat/blob/main/ARCHITECTURE.md)
- [Benchmark results and known failures](https://github.com/SamG-Coder/direct-weight-chat/blob/main/results/README.md)
- [Browser backend verification](https://github.com/SamG-Coder/direct-weight-chat/blob/main/publication/browser-suite-parity.json)

## Hosting and privacy

Hugging Face hosts the static files. Inference runs on the visitor's CPU in a
browser Web Worker. This is not a server inference API. No subscription, API key,
GPU or local Python installation is required to use it. After the assets load,
the app makes no inference network requests. It does not send messages to a
server or store chat in local storage. Each tab has a separate conversation;
refreshing or choosing **New conversation** clears it.

Use a current browser with Web Workers, Web Crypto and `DecompressionStream`
support for `deflate-raw`. Long conversations use more compute and memory.
The browser demo caps a conversation at 1,024 tokens, including 64 tokens
reserved for the next answer, to bound latency and memory on visitors' devices.
The Python runtime retains the model's full 5,048-token logical context.

## Browser implementation

`runtime.mjs` implements the original tokenizer, layer normalization, sparse
linear maps, causal attention, ReLU, residual connections and greedy generation.
It skips exact zero coefficients and computes attention only on dimensions that
can be nonzero according to the weight matrices. These are generic tensor
operations, not host fact parsing, arithmetic or rule solving.

The JavaScript backend accumulates dot products in double precision and stores
float32 activations. It is an independent numerical implementation, not a claim
of bitwise activation equivalence to PyTorch. The linked verification report
records exact text, token IDs and end-of-sequence comparisons against retained
v17a baseline runs, including their known incorrect answers.

The no-training claim describes weight construction: no optimizer, fitting,
training corpus or pretrained tensor transfer. Human/AI prior knowledge,
handwritten grammar, 40 authored atomic entities and manual test-guided
engineering were used. This experiment does not prove that useful chat is
mathematically impossible without training.
