# v0.8 experiment: ground conditional reasoning inside the weights

v16c adds signed ground implications to the analytically compiled decoder.
Rules supplied during a conversation can connect unfamiliar subjects through
up to three links. Neural attention retrieves a rule by its conclusion, checks
the premise against current signed facts, and propagates proof through the
existing truth/query circuit. Missing or false premises remain unknown.

The checkpoint remains exactly **250,000,000 float32 parameters / 1,000,000,000
bytes**, with no padding. It uses 995,881 nonzero parameters (0.3983524%).
Construction uses no training, optimization, pretrained weights, downloaded
data or internet. Inference uses the unchanged generic transformer runtime;
there is no host rule engine, answer fallback, completed-response bank or
weight update. GitHub is used to publish the experiment and evidence.

| CUDA development check | v15a | v16a | v16b | v16c |
|---|---:|---:|---:|---:|
| Ground conditionals | 152/293 | 269/293 | 269/293 | 269/293 |
| Additional scope/dialogue guards | 19/36 | 27/36 | 29/36 | 29/36 |
| Original narrow cases | 185/185 | 182/185 | 185/185 | 185/185 |
| Coordinated assertions and stress | 382/390 | 368/390 | 383/390 | 383/390 |
| Object-specific truth | 236/239 | 229/239 | 235/239 | 236/239 |

All **33 matched CUDA suites** retain every v15a pass: **5,499/5,619 cases**,
130 improvements and zero regressions. Historical suites repeat some cases
and structures. The new conditional suite was committed before compiler
changes; the guard suite is explicitly author-designed development evidence.
CPU matches the conditional 269/293 and guard 29/36 scores. Exact prompts,
outputs, expected answers, timings and hashes are in the
[version table](https://github.com/SamG-Coder/direct-weight-chat/blob/main/results/README.md)
and [comparison](https://github.com/SamG-Coder/direct-weight-chat/blob/main/results/regressions-v16c.json).

The failed v16a/v16b sources, weights and measurements are retained. v16a
allowed proof state to interfere with ordinary open answers. v16b repaired
that isolation but could treat an atomic fact bit as a rule coordinate,
creating a false rule at an entity token. v16c adds a type penalty derived
from the maximum possible coordinate spillover. Read-only activation traces
and a penalty-removal intervention preserve evidence of the cause and fix.

Eight conditional interventions pass on CPU and CUDA, covering rule links, premise
evidence, record validity, object identity, polarity and the typed-record
guard. The initial object-matching intervention was ineffective because it
also removed the positive match score; its failed report remains public.
The corrected diagnostic retains that reward while removing identity
dependence. All interventions use temporary tensor copies and restore them;
the saved model is unchanged. Archived source rebuilds reproduce the exact
raw/config hashes. Both devices pass the existing integrity/intervention suite.
A 31-turn conversation after long declarative distractors scores **31/31 CUDA
and 28/31 CPU**. CPU misses three later conditional conclusions while retaining
all 23 earlier turns. This device/long-context difference is unresolved and is
the next numerical investigation. It is not hidden by the matching short-suite
scores or presented as a passed conversation test.

After source freeze commit `9c84c6ad4c4202764358ab0c48d4d2a4d7cc25e7`,
seed **1711846539** scores **269/293 on CPU and CUDA**, including 98/98
positive/signed applications and all tested chains through three links.
These are fresh combinations from the known generator; fixed structures and
boundaries repeat. They are not independent general-language evidence. No
frozen source or parameters were retuned after this evaluation.

**Useful open-ended conversation remains unachieved.** This is a bounded
single-path proof circuit. It fails chains longer than three rules and misses
older successful alternatives when a newer rule has no proof. Variables,
compound rule premises and open answers from rules are unsupported. Some
paraphrases, including `implies` and `whenever`, can be mistaken for assertions.
Creative writing, code generation, unrestricted English and broader arithmetic
remain failures. The experimental goal stays active.

Download `autoregressive-v16c.zip`, verify `SHA256SUMS.txt`, and extract it into
the repository root. The smaller archive reflects sparse/repetitive weights;
the decompressed parameter file is exactly 1 GB. NumPy/PyTorch must already be
installed; the project downloads no dependencies or model data.

```powershell
python -m chat1gb_autoregressive.chat --device auto
```

The v16a/v16b archives extract to their historical version directories.
Every earlier release remains available. v16c raw SHA-256:
`385dff256910dec6eba802d3930452c2963e85223bbaa54251d2931b9b86b1dc`.
