# Construction claim audit: final v17a

**Analytically constructed weights — no training** accurately describes the
provided v17a construction pipeline. It does not imply general language ability,
absence of prior knowledge, or absence of manual engineering against development
tests. Useful, open-ended chat was not achieved.

This publication audit was performed on 4 November 2026:

1. Inspected the frozen `build.py`, `spec.py`, `knowledge.py`, `offline.py` and
   generic inference runtime. The compiler starts a new float32 memory-mapped
   file, fills it with zeroes, and writes coefficients using explicit arithmetic,
   code vectors, attention projections, gates and piecewise-linear circuits.
   It does not load an earlier or pretrained parameter file.
2. Confirmed that the compiler imports standard-library modules, NumPy and the
   local specification/knowledge/offline modules. Its construction path contains
   no gradient computation, optimizer, loss minimization, example-fitting loop,
   dataset loader or model-download operation. A text scan is supporting evidence,
   not a formal proof; the code and its inputs are available for inspection.
3. Rebuilt from the archived v17a source into a new output directory, without
   modifying the frozen source. The build uses the project's offline socket guard.
   All source hashes match the original manifest. The raw and config hashes match
   the published checkpoint exactly, including the copy downloaded from Hugging Face.
4. The rebuilt checkpoint contains 995,905 nonzero float32 values in its
   original named tensor layout, with no padding trailer. The current compact
   artifact preserves those bits in exactly 1,528,592 packed weight bytes.
   Its nonzero count comes from scanning the actual tensors after construction.
5. Inference runs the saved tensors with gradients disabled and no compiler/fact
   module imports. Existing ablations and before/after file hashes establish the
   tested runtime boundary; conversation facts do not update saved parameters.

Raw weight SHA-256:
`5cc52f3793a1546c620195c110a7d1c3a2f8d3417cb72f193003589b16e8e0d2`.

Config SHA-256:
`56df33d5842a0d3a1351dae78732c01589bde1ce55a33868f9d13dec171bee91`.

## What the claim does not mean

- The compiler, restricted grammar and 40 atomic entity records were authored
  with an AI coding assistant. Human/assistant prior knowledge is present in
  the design; pretrained parameter tensors were not transferred.
- Developers used test failures to revise general circuit code and numerical
  constants. “No training” here means no automated fitting/training of this
  model's parameters, not an absence of development feedback.
- The language behavior is manually constrained. There is no bank of complete
  answers in the current checkpoint, but hand-written token/state rules still
  determine a narrow output grammar, including short fixed expressions.
- Most logical connections are zero. Packed file size does not measure
  independently acquired conversational knowledge.
- The experiment's failure is empirical. It does not establish a theorem that
  all possible directly constructed neural architectures must fail.

Evidence: [frozen compiler](../versions/autoregressive_v17a/build.py),
[authored facts](../versions/autoregressive_v17a/knowledge.py),
[runtime](../chat1gb_autoregressive/engine.py),
[publication rebuild record](construction-audit.json),
[architecture](../ARCHITECTURE.md),
[all benchmark results](../results/README.md),
[original dense Hugging Face verification](huggingface-verification.json),
[compact storage and inference evidence](COMPACT.md).
