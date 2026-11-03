# v17a compact: the same frozen circuit, less stored zero space

**Analytically constructed weights — no training.** The experiment is finished.
Useful, open-ended chat was not achieved. This edition changes storage and
loading, not the checkpoint's coefficients, grammar, algorithms or capabilities.

| Quantity | Exact bytes |
|---|---:|
| Packed weight archive | **1,528,592** |
| Float32 values plus uint32 coordinates before ZIP compression | 7,967,288 |
| Persistent runtime sparse buffers and cached small vectors after inference | 12,952,916 |

The archive contains **995,905 nonzero coefficients**, plus six negative-zero
entries retained for exact bitwise reconstruction. Positive zeros are implicit.
There is no quantization or change in coefficient precision. The 250,000,000
positions describe the **original logical dense layout**, not 250 million
independently meaningful values stored by the compact edition.

The sparse artifact is consumed directly. It does not write a dense checkpoint
to disk or retain the entire dense weight array during inference.

## How it runs

`compact_engine.Model` inherits the frozen model's tokenizer, forward pass,
attention, KV caching, token selection and generation unchanged. Its weight
mapping stores coordinates and float32 values. When the original forward pass
requests a tensor, the mapping reconstructs that tensor in its original shape.
Large tensors are temporary; small normalization and bias vectors are cached.
This preserves the original dense matrix-operation dimensions and order.

The implementation remains an ordinary PyTorch transformer with analytically
constructed circuits. It is **not a host-side fact solver, arithmetic parser,
response selector or new trained language model**. Both the word/fact circuits
and the arithmetic circuits use this same loader. Compression adds no abilities.

12,952,916 bytes is the persistent **weight-tensor** footprint, not total application
memory. Python, PyTorch, temporary dense tensors, attention activations, logits,
KV state, allocator caches and CUDA context consume additional memory. The
logical width, layer count and context length remain unchanged. Long contexts
can still use substantial memory and compute.

In a fresh-process, one-short-prompt check on the test machine, the compact
model's CPU working set after the response was **507,301,888 bytes**, including
Python/PyTorch. This is not a maximum-context RAM requirement. See
[measured memory](compact-memory.json) for the measurements and their scope;
timing under concurrent verification is not a speed claim.

## Evidence

- [Packing record](compact-packing.json): source hash unchanged, nonzero count,
  stored coordinates and exact archive size.
- [Independent archive round trip](compact-storage-verification.json): all
  serialized tensors reconstruct the original dense SHA-256 without writing a
  dense checkpoint to disk.
- [CUDA output comparison](compact-parity-cuda.json) and
  [CPU output comparison](compact-parity-cpu.json): actual outputs, token IDs,
  end-of-sequence behavior, baseline hashes and retained failures.
- [Actual compact conversation capture](demo-transcript-compact.json): facts,
  corrections, short rule chains, two-operand arithmetic and failed requests.
- [Construction claim audit](CONSTRUCTION-AUDIT.md): no fitting/training loop,
  and disclosure of authored algorithms, facts and manual test-guided development.

The frozen dense SHA-256 remains
`5cc52f3793a1546c620195c110a7d1c3a2f8d3417cb72f193003589b16e8e0d2`.
The compact archive SHA-256 is
`4739a3511f5c4b9170f5d09524bd8105f5853a7999cbeefdea7a8cbae01d737d`.

The comparison reports test preservation of existing behavior. They are not new
independent capability benchmarks. Known incorrect answers remain incorrect.
Output parity on these tests does not establish identical behavior on every
possible prompt, hardware configuration or future PyTorch version.

The completed CUDA comparison covers **38 suites / 6,371 cases**, with identical
text, token IDs and EOS behavior in all 6,371. The score remains 6,148/6,371;
the 223 previously incorrect cases are still incorrect. These suites repeat
historical structures. The maximum CUDA allocation observed during those
long-context suites was about 2.08 GB, with up to 4.20 GB reserved by the
allocator, illustrating why the 12,952,916 bytes weight figure is not a total VRAM claim.

The completed CPU comparison covers **6 suites / 788 cases**, with all text,
token IDs and EOS decisions unchanged. The score remains 678/788, preserving
all 110 failures. Four suites completed in an interrupted initial run; the two
remaining long-context suites completed in separate fresh processes. The
aggregate report identifies all three source reports without duplicate counting.

## Run the compact artifact

Download the model repository from
[Hugging Face](https://huggingface.co/SamGCoder/direct-weight-chat-v17a), extract
`runtime.zip` and use an environment with PyTorch already installed:

```bash
python -m zipfile -e runtime.zip .
python -m chat1gb_autoregressive.compact_chat --model-dir . --device auto
```

The GitHub compact release instead puts weights in `compact/` alongside the
runtime source:

```bash
python -m chat1gb_autoregressive.compact_chat --model-dir compact --device auto
```

The local UI includes separately labelled captured replay and live offline
inference. Use the Python executable for an installed PyTorch environment:

```bash
python -m publication.serve_demo --python /path/to/python --model compact --port 8768
```

Open `http://127.0.0.1:8768/`. Context is shared by tabs connected to that one
local server; Reset starts a fresh conversation. There is no hosted chat service.

## Reproduce packing and comparison offline

With NumPy installed for construction/packing and PyTorch for inference:

```bash
python -m tools.build_version v17a --output build/v17a
python -m publication.pack_compact --source build/v17a --output build/v17a-compact
python -m publication.verify_compact_storage --model build/v17a-compact --output build/round-trip.json
python -m publication.verify_compact --device cuda --output build/parity-cuda.json
python -m publication.verify_compact --device cpu --output build/parity-cpu.json
```

These commands refuse to overwrite existing evidence. The original dense
[v0.9 release](https://github.com/SamG-Coder/direct-weight-chat/releases/tag/v0.9.0-experiment)
and earlier versions remain available for historical reproduction.
