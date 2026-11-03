# Version results

All five autoregressive checkpoints were run through the same 185-case development suite using the same current generic transformer runtime on one RTX 5080. This is a retrospective comparison, not an independent benchmark. Whole-response medians exclude one warm-up; the original historical timings are retained separately.

| Version | Exact passes | Median response (ms) | Nonzero parameters | Raw outputs |
|---|---:|---:|---:|---|
| v1 | 131/185 | 7.59 | 674,640 | [legacy-v1-cuda.json](legacy-v1-cuda.json) |
| v2 | 177/185 | 7.62 | 691,848 | [legacy-v2-cuda.json](legacy-v2-cuda.json) |
| v3 | 182/185 | 7.71 | 697,264 | [legacy-v3-cuda.json](legacy-v3-cuda.json) |
| v4 | 182/185 | 7.54 | 697,392 | [legacy-v4-cuda.json](legacy-v4-cuda.json) |
| v5 | 185/185 | 7.61 | 697,392 | [legacy-v5-cuda.json](legacy-v5-cuda.json) |

The eleven broad request outputs are retained in each JSON under `probes`; they are not included in the 185 narrow test score. Original v5 failed all eleven intended broad requests, including poetry, code, negation and a two-link novel inheritance query.

The rejected selector v0 is a different architecture and must not be ranked on the autoregressive table. Its historical checks passed 67/68 direct routes, 84/86 follow-ups, 200/200 digit/carry cases and 260/260 full sums. See [CUDA evidence](../chat1gb/evidence/verification-torch-cuda.json) and [CPU evidence](../chat1gb/evidence/verification-numpy-cpu.json). It stores canned responses and does not satisfy the current task.

The first graph-composition baseline is [v5: 12/79](graph-v5-baseline-cuda.json). Its passes are direct overrides and appropriate unknown answers; none of the 48 one-to-three-link unfamiliar property queries succeeds.

Historical builds predate Git. Their source/weight hashes are archived in `versions/*/model/manifest.json`. Retrospective evaluations record the import commit and whether the working tree was dirty; their model/source hashes are the definitive artifact identities.
