# v0.4 experiment: proposition composition and active request scope

Two new analytically compiled 1 GB checkpoints are preserved: v12a and corrected
v12b. Each contains exactly 250,000,000 float32 parameters / 1,000,000,000 raw
weight bytes. No training, pretrained parameters or downloaded knowledge is used;
construction and inference remain offline.

v12 compiles proposition events, evaluates each one neurally, and reduces true,
false and unknown values with homogeneous AND/OR. It supports the tested shared
and distinct subjects/predicates, inheritance and corrections. Clause IDs isolate
the active request. The generic transformer inference runtime is unchanged.

| Development suite | v11a | v12a | v12b |
|---|---:|---:|---:|
| Legacy | 185/185 | 185/185 | 185/185 |
| Graph | 75/79 | 75/79 | 75/79 |
| Arithmetic | 115/120 | 88/120 | 115/120 |
| Polarity | 229/233 | 230/233 | 230/233 |
| Signed operands | 153/159 | 121/159 | 153/159 |
| Object-specific truth | 234/239 | 236/239 | 236/239 |
| Query scope | 262/269 | 265/269 | 265/269 |
| Proposition composition | 243/404 | 398/404 | 397/404 |
| Request modifiers | Not run | 17/81 | 78/81 |

v12a rejects Please calculate because its request-head filter is too strict.
v12b uses a bounded transparent-modifier grammar, repairing all 106 arithmetic
regressions across four reports. It also exposes an unsupported paraphrase that
v12a had rejected. The failed candidate and every failing case remain archived.
Three composition regressions against v11a remain recorded; its earlier partial-
answer passes did not establish support for those structures.

After freezing v12b, fresh seeds reproduced 397/404 composition and 78/81 modifier
scores on CPU and CUDA. These are known-generator combinations, not independently
designed general-intelligence tests. Source hashes, raw model hashes, exact
outputs, timings and regression comparisons are in `results/`.

Integrity, causal-prefix and cached/full decoding checks pass. Changing AND to
OR, removing unknown evidence or removing clause IDs changes the corresponding
answer. Earlier graph, arithmetic, polarity and truth ablations still work.
Rebuilding from the frozen v12b source reproduces raw weights and config exactly.

**Remaining limits:** mixed-operator precedence, compound assertions, incomplete
connectives, repeated negation, general pronouns, longer framing and partial-
question instructions fail. The open wh-question multiple-action bug remains.
Open-ended conversation, creative writing and code generation are unachieved.
v12b has 873,633 nonzero parameters (0.3494532%); most connections remain unused.

Extract `autoregressive-v12b.zip` into the repository root and run:

```powershell
python -m chat1gb_autoregressive.chat --device auto
```

The v12a archive uses `versions/autoregressive_v12a/model`; select it explicitly
with `--model-dir`. Compare archive hashes with `SHA256SUMS.txt`, and raw/config
hashes with the included manifest. Small compressed archives reflect sparse,
repetitive weights; each extracted checkpoint is exactly 1 GB. NumPy/PyTorch
must already be installed; the build and runtime do not download dependencies.
