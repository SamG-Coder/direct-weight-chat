# v0.6 experiment: stable local attention and fact magnitudes

> Historical release notes. The current compact artifact is exactly **1,528,592 packed weight bytes**. See the [current release](RELEASE-COMPACT.md); the old assets below remain historical.

Two analytically compiled checkpoints are preserved: failed v14a and current
v14b. Each contains exactly 250,000,000 float32 parameters. Construction and inference remain offline, with no training,
pretrained tensors, downloaded knowledge, response bank or host answer logic.
The generic transformer runtime is unchanged.

Two bounded sinusoidal phases replace quadratic local attention addresses.
The new representation avoids large common score terms that lost precision at
long positions. v14b also saturates fact coordinates before proposition lookup,
preventing small magnitude differences from overriding a later correction.

| Development check | v13d baseline | Failed v14a | v14b |
|---|---:|---:|---:|
| Context positions and generation, CUDA | 61/126 | 126/126 | 126/126 |
| Assertion boundary audit, CUDA | 65/76 | 76/76 | 76/76 |
| Coordinated assertions, CUDA | 382/390 | 366/390 | 382/390 |
| Incorrect local addresses /45,432, CUDA | 15,900 | 0 | 0 |
| Incorrect local addresses /45,432, CPU | 7,491 | 0 | 0 |

The address audit checks nine heads at all 5,048 context positions. Context
tests move existing generation, arithmetic, facts, correction, negation, binding
and inheritance through the window, with six requests near capacity. These
checks add numerical evidence; they do not add general language understanding.

All 26 existing matched suites were rerun. v14b has no regressions against
v13d. It retains legacy 185/185, graph 75/79, arithmetic 115/120, polarity 230/233,
signed operands 153/159, truth 236/239, query scope 265/269, composition 398/404
and request modifiers 78/81. Failed v14a loses sixteen assertion corrections
in each of three reports. Its checkpoint, exact failures and diagnostic trace
are retained in the [version history](https://github.com/SamG-Coder/direct-weight-chat/blob/v0.6.0-experiment/results/README.md).

After freezing v14b at `b3b6e1bd76a7e007262d6cf16c27dd3ed4cc1c04`, fresh
context seed 2063643282 scores **126/126 on CUDA and 125/126 on CPU**. CPU case
`context-086`, with a 4,720-token prefix, loses an unfamiliar subject's second
coordinated object and answers `I do not know.` instead of `Yes.`. This remains
a failed test. No frozen weights were retuned. These are fresh combinations
from known structures and generators, not independent generalization evidence.

Integrity, finite/layout/hash, causal-prefix, cache and offline-guard checks
pass on the local CUDA run. Phase-removal interventions change the associated
response. A controlled fact-magnitude perturbation preserves a correction with
saturation; removing saturation makes the earlier assertion win. All earlier
graph/arithmetic/truth/composition/assertion interventions also pass. Frozen
v14b source reproduces exact raw and config hashes. A separate local 20-turn
publication smoke passes after 3,000 initial tokens of declarative distractors.

Fresh-download verification confirms both published archive digests, sizes,
raw/config hashes, ZIP CRC and frozen-source manifests. All integrity checks
and interventions pass on CPU and CUDA from the clean tagged checkout. The long
20-turn conversation scores **20/20 on CUDA and 19/20 on CPU**. At turn 2, with
a 3,036-token prefix, CPU loses a changed subject's second object and says
`I do not know.` instead of `Yes.`. This corroborates the separate fresh context
failure. Exact results remain in the [publication evidence](https://github.com/SamG-Coder/direct-weight-chat/blob/main/results/publication-v0.6.json).

**The model still cannot provide useful open-ended chat.** Conditional rules,
disjunctive assertions, mixed query precedence, multiple class parents, repeated
negation and open multi-value answers remain unsupported. Creative writing and
code generation remain unachieved. The remaining CPU binding failure prevents
a claim of semantic reliability throughout the context window. v14b has 960,705
nonzero parameters (0.384282%); most connections remain unused.

Extract `autoregressive-v14b.zip` into the repository root and run:

```powershell
python -m chat1gb_autoregressive.chat --device auto
```

The failed v14a archive extracts to `versions/autoregressive_v14a/model`; use
`--model-dir` to inspect it. Verify ZIP hashes against `SHA256SUMS.txt` and
raw/config hashes against each manifest. ZIPs are small because these weights
are sparse and repetitive; every extracted checkpoint contains the original dense float32 layout. NumPy and PyTorch must already be installed.
