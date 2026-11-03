# v0.5 experiment: coordinated assertions and deferred binding

Four analytically compiled 1 GB checkpoints are preserved: v13a, failed v13b,
intermediate v13c and current v13d. Each has exactly 250,000,000 float32 parameters
and 1,000,000,000 raw weight bytes. Construction and inference remain offline,
with no training, pretrained parameters or downloaded knowledge. The generic
transformer runtime is unchanged.

The new circuits bind shared assertion objects to clause-local subjects,
predicates and polarity. Deferred commitment waits for a following boundary
before storing an ambiguous coordinated object. Question/directive clauses are
excluded from the observed false memories. A numerical correction preserves
clause-coordinate precision when reusing a residual field.

| Suite | v12b baseline | v13a | v13b | v13c | v13d |
|---|---:|---:|---:|---:|---:|
| Coordinated assertions | 158/390 | 382/390 | 184/390 | 382/390 | 382/390 |
| Assertion boundary audit, CUDA | Not run | 41/76 | 41/76 | 61/76 | 65/76 |
| Proposition composition | 397/404 | 398/404 | 397/404 | 398/404 | 398/404 |

v13d passes all 381 targeted assertion cases and all 48 short boundary controls.
It retains legacy 185/185, graph 75/79, arithmetic 115/120, polarity 230/233,
signed operands 153/159, truth 236/239, query scope 265/269 and modifiers 78/81.
All prior suites were rerun. The failed candidates, exact outputs, timings,
regressions, activation diagnostic and source/model hashes remain public in the
[version table](https://github.com/SamG-Coder/direct-weight-chat/blob/v0.5.0-experiment/results/README.md).

v13b loses valid sentence-final objects through mismatched clause coordinates.
v13c repairs that but has four subject-binding regressions caused by float32
cancellation. v13d fixes the coordinate scale and has no regressions against
v13a across 22 matched reports. A conditional-stress regression against v12b
remains: the old Yes also occurred without the premise and did not demonstrate
conditional-rule application.

After freezing v13d at `4382108`, seed 3178044486 reproduces 382/390 assertions
on CPU and CUDA. Boundary seed 3284145273 scores 65/76 CUDA and 64/76 CPU;
all 48 short cases pass on both. These use known generators and structures,
not independently designed evidence of broad intelligence. Frozen weights were
not retuned after those results.

Integrity, finite/layout/hash, causal-prefix, cache and offline-guard checks pass.
Neural interventions remove predicate/frame binding, question exclusion,
continuation validity or following-token validation and change the corresponding
behavior. The earlier graph/arithmetic/truth/composition interventions still pass.
Frozen v13d source reproduces exact raw and config hashes. The local 20-turn
publication smoke conversation passes.

**Limits remain substantial.** Some cases fail from 2,048 padding tokens onward;
the 5,048-token storage window is not a demonstrated accuracy window. CPU/CUDA
can diverge at long positions, including incomplete output. Disjunctions,
conditional rules, multiple class parents, mixed query precedence, repeated
negation and open multi-value answers remain unsupported. Useful open-ended chat,
creative writing and code generation are unachieved. v13d has 929,381 nonzero
parameters (0.3717524%); most connections remain unused.

Extract `autoregressive-v13d.zip` into the repository root and run:

```powershell
python -m chat1gb_autoregressive.chat --device auto
```

The other archives extract to `versions/autoregressive_VERSION/model`; choose
that directory with `--model-dir`. Verify ZIP hashes using `SHA256SUMS.txt` and
raw/config hashes using each manifest. Small ZIPs reflect sparse/repetitive
weights; every extracted checkpoint is exactly 1 GB. NumPy and PyTorch must
already be installed; these commands do not download dependencies.
