# direct-weight-chat: compact runtime

**Analytically constructed weights - no training.** The experiment is finished.
Useful, open-ended chat was not achieved. The final published checkpoint is
**v17a compact**, with exactly **1,528,592 bytes of packed weights**.

This artifact stores 995,905 nonzero float32 coefficients and their coordinates.
Six negative-zero entries are retained for exact bitwise reconstruction.
Persistent runtime weight tensors use **12,952,916 bytes** after inference;
temporary tensors, activations, conversation state and Python/PyTorch require
additional memory. Download size is not total RAM or VRAM use.

The compact loader inherits the frozen tokenizer, transformer forward pass and
token-by-token generation unchanged. It reconstructs one requested tensor at a
time. It does not quantize coefficients, train, solve facts in Python, choose
completed responses or add capabilities.

## Run

From the extracted GitHub compact release, with PyTorch already installed:

```bash
python -m chat1gb_autoregressive.compact_chat --model-dir compact --device auto
```

From the Hugging Face download directory:

```bash
python -m zipfile -e runtime.zip .
python -m chat1gb_autoregressive.compact_chat --model-dir . --device auto
```

`/reset` starts a fresh conversation; `/quit` exits. CPU inference is supported.
This is a custom PyTorch runtime, not Transformers AutoModel or GGUF. NumPy is
needed for construction/packing, but not for inference.

## Measured scope

Restricted context facts, corrections, bounded signed two-operand addition and
subtraction, and short ground-rule chains work in the tested grammar. The model
cannot provide useful general conversation, unrestricted English, creative
writing, code generation or general mathematics. The alternative-proof baseline
remains 29/104; all 75 failures are retained. Extra operands and conditional
arithmetic are unsupported in the recorded probes.

[Compact format, memory and response parity](../publication/COMPACT.md)

[Architecture](../ARCHITECTURE.md)

[Every historical version's results](../results/README.md)

[Construction audit and claim limits](../publication/CONSTRUCTION-AUDIT.md)

[Model and matching source on Hugging Face](https://huggingface.co/SamGCoder/direct-weight-chat-v17a)

The original dense runtime and historical compilers remain for reproducibility.
Use the compact commands above for the current published artifact.
