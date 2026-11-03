# Final archive: v17a compact — experiment finished

**Analytically constructed weights — no training.** Useful, open-ended chat
was not achieved under the experiment's constraints. The research is finished;
this release publishes a compact representation of the same final checkpoint.

The packed weights occupy exactly **1,528,592 bytes**, storing **995,905 nonzero
coefficients**. The loader reconstructs individual tensors when needed and
retains **12,952,916 bytes of weight tensors** after inference. Temporary tensors,
activations and runtime overhead require additional RAM/VRAM. The architecture
and compute dimensions remain unchanged.

Every float32 bit reconstructs exactly, including six negative-zero entries.
There is no quantization, coefficient update, training, new grammar or added
capability. The original dense model and earlier failed candidates remain in
the historical releases.

The attached `direct-weight-chat-v17a-compact.zip` contains the compact model,
matching PyTorch runtime, frozen compiler source and local chat UI. Extract it
to a new directory and run with PyTorch already installed:

```bash
python -m chat1gb_autoregressive.compact_chat --model-dir compact --device auto
```

`/reset` clears context; `/quit` exits. The model supports narrow fact/correction
grammar, bounded signed addition/subtraction and short ground-rule chains. It
does not support useful general chat, creative writing, code generation,
general mathematics or searching alternative proofs. The alternative-proof
baseline remains **29/104**, with all failures preserved.

Verification preserves all text, token IDs and EOS decisions in **6,371 CUDA
cases and 788 CPU cases**, including the existing failures. Independent archive
reconstruction matches the frozen original weight hash exactly.

The [compact verification guide](https://github.com/SamG-Coder/direct-weight-chat/blob/main/publication/COMPACT.md)
links exact size, round-trip hash, CPU/CUDA response parity and memory evidence.
[Every historical version's results](https://github.com/SamG-Coder/direct-weight-chat/blob/main/results/README.md)
and the [construction claim audit](https://github.com/SamG-Coder/direct-weight-chat/blob/main/publication/CONSTRUCTION-AUDIT.md)
remain public.

The same compact weights and matching source are available on
[Hugging Face](https://huggingface.co/SamGCoder/direct-weight-chat-v17a).
This is a custom offline PyTorch runtime, not a Transformers AutoModel or GGUF
checkpoint. Checksums are attached as `SHA256SUMS.txt`.
