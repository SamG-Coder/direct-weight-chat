# Version results

Every autoregressive checkpoint has exactly 250,000,000 float32 parameters and 1,000,000,000 weight bytes. Failed candidates remain in this table. A dash means the suite was not run for that version.

| Version | Legacy /185 | Graph /79 | Arithmetic /120 | Nonzero parameters | Source commit |
|---|---:|---:|---:|---:|---|
| v1 | [131/185](legacy-v1-cuda.json) | — | — | 674,640 | [5be0768](https://github.com/SamG-Coder/direct-weight-chat/commit/5be0768f6e2d61c1bacd30950cf343cd79173b74) |
| v2 | [177/185](legacy-v2-cuda.json) | — | — | 691,848 | [5be0768](https://github.com/SamG-Coder/direct-weight-chat/commit/5be0768f6e2d61c1bacd30950cf343cd79173b74) |
| v3 | [182/185](legacy-v3-cuda.json) | — | — | 697,264 | [5be0768](https://github.com/SamG-Coder/direct-weight-chat/commit/5be0768f6e2d61c1bacd30950cf343cd79173b74) |
| v4 | [182/185](legacy-v4-cuda.json) | — | — | 697,392 | [5be0768](https://github.com/SamG-Coder/direct-weight-chat/commit/5be0768f6e2d61c1bacd30950cf343cd79173b74) |
| v5 | [185/185](legacy-v5-cuda.json) | [12/79](graph-v5-baseline-cuda.json) | — | 697,392 | [5be0768](https://github.com/SamG-Coder/direct-weight-chat/commit/5be0768f6e2d61c1bacd30950cf343cd79173b74) |
| v6a | [1/185](legacy-v6a-cuda.json) | [8/79](graph-v6a-cuda.json) | — | 716,888 | [1f7509f](https://github.com/SamG-Coder/direct-weight-chat/commit/1f7509f7a7e9251386c445c722dab5701fd591ef) |
| v6b | [182/185](legacy-v6b-cuda.json) | [68/79](graph-v6b-cuda.json) | — | 716,960 | [ddfaa85](https://github.com/SamG-Coder/direct-weight-chat/commit/ddfaa85087e60339a058933b5aa09667f43906f1) |
| v6c | [185/185](legacy-v6c-cuda.json) | [74/79](graph-v6c-cuda.json) | [22/120](math-v6c-baseline-cuda.json) | 717,352 | [95f9754](https://github.com/SamG-Coder/direct-weight-chat/commit/95f9754da6d5cccc8d4170dc767875a4867a8edb) |
| v7a | [185/185](legacy-v7a-cuda.json) | [74/79](graph-v7a-cuda.json) | [114/120](math-v7a-cuda.json) | 717,554 | [60169ec](https://github.com/SamG-Coder/direct-weight-chat/commit/60169ec23186114b40069961c61c48dd01de51b9) |

## Interpretation

The legacy suite is a restricted development suite run retrospectively on v1–v4. It is not a general-chat benchmark. Its eleven broad probes are retained outside the exact-output score. v7a now answers the two-link inheritance probe; creative writing, code generation, negation, general knowledge and most other broad requests still fail.

The graph suite tests six properties through unfamiliar class links with shuffled facts, distractors, corrections, cycles and missing links. v6c and v7a pass all 74 cases within the implemented syntax/depth; all five depth/paraphrase/negation stress failures remain in the denominator.

The arithmetic suite contains 88 subtraction, 20 addition and six mixed-turn cases, plus six failing stress cases. v7a passes 114/120. Negative **results** work; negative **operands**, chained expressions, multiplication, division, larger inputs and an unsupported paraphrase fail. Some unsupported inputs produce incorrect numbers rather than an unknown response.

## Fresh combinations after freezing

- [v6c graph CUDA](graph-v6c-postfreeze-cuda.json) and [CPU](graph-v6c-postfreeze-cpu.json): 74/79, including 48/48 novel one-to-three-link chains. Seed 965544940 was chosen after commit `95f9754` froze the candidate.
- [v7a arithmetic CUDA](math-v7a-postfreeze-cuda.json) and [CPU](math-v7a-postfreeze-cpu.json): 114/120. Seed 2964068083 was chosen after commit `60169ec` froze the candidate. This adds 100 fresh operand combinations; the 20 fixed boundary, dialogue and stress cases repeat.
- [v7a graph regression](graph-v7a-regression-cuda.json): 74/79 on the former v6c fresh set, now treated as regression evidence.
- [v6c integrity](integrity-v6c-cuda.json) and [v7a integrity](integrity-v7a-cuda.json): hashes, exact layout/size, finite values, zero causal-prefix difference, cache parity, offline guard and graph/output-head ablations passed.

The generators and language structures were known before freezing. These are fresh combinations under a known grammar, not independently designed tests of general intelligence. No model was tuned after inspecting its own post-freeze results.

## Provenance and timing

[experiments.json](experiments.json) maps each checkpoint to the commit containing its exact compiler bytes, model hash and measured counts. Each raw report also records the Git HEAD and dirty flag at execution time. A dirty execution tree is not a claim that HEAD alone contains the candidate; use the indexed frozen source commit and hashes. Original v1–v5 builds predate this repository, so their source commit is the import commit.

Response timings include token generation and formatting, exclude one warm-up, and depend on prompt/output length and machine load. They are not a comparison against trained language models. Exact per-case timings and hardware are in each JSON. CUDA and CPU evaluations were run separately.

## Rejected selector v0

v0 is a different architecture with canned responses. Its routing checks passed 67/68 direct routes and 84/86 follow-ups; arithmetic passed 200/200 digit/carry and 260/260 full-sum checks. [CUDA evidence](../chat1gb/evidence/verification-torch-cuda.json) and [CPU evidence](../chat1gb/evidence/verification-numpy-cpu.json) are retained. Those scores are not autoregressive conversation scores and this prototype does not satisfy the current research constraints.
