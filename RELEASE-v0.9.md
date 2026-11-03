# v0.9 experiment: stable rule phases across long contexts

v17a repairs measured CPU failures in conditional-rule retrieval. Small errors
in a nominally Boolean phase flag previously outweighed literal recency in
sharp attention. A shared two-ReLU saturation primitive gives the rule-phase
flags flat zero/one regions before ranking. It adds two ReLU units and 24
nonzero parameters; the ordinary transformer runtime and layout are unchanged.

The checkpoint has exactly **250,000,000 float32 parameters / 1,000,000,000
bytes**, including **995,905 nonzero parameters (0.398362%)**. All parameters
belong to named model tensors. Construction uses no training, optimization,
pretrained weights, downloaded data or internet. Inference uses tensor
operations without a host rule engine, completed-response bank or weight update.
GitHub is used only to publish source and evidence.

| Check | v16c | v17a |
|---|---:|---:|
| Rule positions, CPU | 154/162 | 160/162 |
| Rule positions, CUDA | 160/162 | 160/162 |
| Long conversation, CPU | 28/31 | 31/31 |
| Long conversation, CUDA | 31/31 | 31/31 |
| Phase perturbations, each device | 164/220 | 220/220 |
| Ground conditionals, each device | 269/293 | 269/293 |
| Scope/dialogue guards, each device | 29/36 | 29/36 |

All 36 matched CUDA suites have identical outputs to v16c: **5,959/6,105**,
zero regressions. Four matched CPU suites gain nine passes with zero regressions.
Historical suites repeat structures; their totals are not counts of independent
reasoning problems. Exact prompts, outputs, expected answers, timings and
hashes remain in the [version table](https://github.com/SamG-Coder/direct-weight-chat/blob/main/results/README.md),
[CPU comparison](https://github.com/SamG-Coder/direct-weight-chat/blob/main/results/regressions-v17a-cpu.json)
and [CUDA comparison](https://github.com/SamG-Coder/direct-weight-chat/blob/main/results/regressions-v17a-cuda.json).

Tests and frozen baselines preceded the change. A targeted CPU intervention
restoring only the old context gate changes a repaired answer from `Yes.` back
to `I do not know.`. The saved checkpoint is unchanged. Both devices pass 27
existing integrity check groups and eight conditional interventions. Rebuilding
from the archived compiler reproduces exact raw/config hashes.

After freeze commit `83a2f03afa98075286e7ec82b3e5c53f29cd4ceb`, fresh seed
**2165866013** scores **160/162 on CPU and CUDA**. The two failures require
a fourth rule link or an older alternative proof. These are new combinations
and offsets under a known generator, with repeated fixed structures. They do
not establish independent language generalization. No frozen parameters were
retuned after observing these results.

**Useful general conversation remains unachieved.** The model still follows
one latest matching rule at each step. It cannot search earlier alternatives,
bind variables, evaluate compound premises or generate open answers from rules.
Creative writing, code generation, broader mathematics and unrestricted English
remain unsupported. Some paraphrases still produce false assertions. Passing
these numerical tests does not prove reliability on every possible prompt.

The next capability baseline scores **29/104 on CPU and CUDA** across
alternative rules, mixed branches, ordering and corrections. These tests were
authored after v17a froze, before any proof-search implementation. They are
development evidence for the next experiment; all 75 failures remain recorded.

Download `autoregressive-v17a.zip`, verify `SHA256SUMS.txt`, and extract into
the repository root. The compressed size reflects sparse/repetitive weights;
the raw parameter file is exactly 1 GB. With PyTorch already installed:

```powershell
python -m chat1gb_autoregressive.chat --device auto
```

NumPy is also required to rebuild from source. No dependencies or model data
are downloaded by construction or inference. All previous releases, failed
candidates and their original evidence remain available.

Raw weight SHA-256:
`5cc52f3793a1546c620195c110a7d1c3a2f8d3417cb72f193003589b16e8e0d2`.
