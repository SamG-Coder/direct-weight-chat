# Version results

Every autoregressive checkpoint has exactly 250,000,000 float32 parameters and 1,000,000,000 weight bytes. Failed candidates remain in this table. A dash means the suite was not run for that version.

| Version | Legacy /185 | Graph /79 | Arithmetic /120 | Polarity /233 | Signed /159 | Nonzero parameters | Source commit |
|---|---:|---:|---:|---:|---:|---:|---|
| v1 | [131/185](legacy-v1-cuda.json) | — | — | — | — | 674,640 | [5be0768](https://github.com/SamG-Coder/direct-weight-chat/commit/5be0768f6e2d61c1bacd30950cf343cd79173b74) |
| v2 | [177/185](legacy-v2-cuda.json) | — | — | — | — | 691,848 | [5be0768](https://github.com/SamG-Coder/direct-weight-chat/commit/5be0768f6e2d61c1bacd30950cf343cd79173b74) |
| v3 | [182/185](legacy-v3-cuda.json) | — | — | — | — | 697,264 | [5be0768](https://github.com/SamG-Coder/direct-weight-chat/commit/5be0768f6e2d61c1bacd30950cf343cd79173b74) |
| v4 | [182/185](legacy-v4-cuda.json) | — | — | — | — | 697,392 | [5be0768](https://github.com/SamG-Coder/direct-weight-chat/commit/5be0768f6e2d61c1bacd30950cf343cd79173b74) |
| v5 | [185/185](legacy-v5-cuda.json) | [12/79](graph-v5-baseline-cuda.json) | — | — | — | 697,392 | [5be0768](https://github.com/SamG-Coder/direct-weight-chat/commit/5be0768f6e2d61c1bacd30950cf343cd79173b74) |
| v6a | [1/185](legacy-v6a-cuda.json) | [8/79](graph-v6a-cuda.json) | — | — | — | 716,888 | [1f7509f](https://github.com/SamG-Coder/direct-weight-chat/commit/1f7509f7a7e9251386c445c722dab5701fd591ef) |
| v6b | [182/185](legacy-v6b-cuda.json) | [68/79](graph-v6b-cuda.json) | — | — | — | 716,960 | [ddfaa85](https://github.com/SamG-Coder/direct-weight-chat/commit/ddfaa85087e60339a058933b5aa09667f43906f1) |
| v6c | [185/185](legacy-v6c-cuda.json) | [74/79](graph-v6c-cuda.json) | [22/120](math-v6c-baseline-cuda.json) | — | — | 717,352 | [95f9754](https://github.com/SamG-Coder/direct-weight-chat/commit/95f9754da6d5cccc8d4170dc767875a4867a8edb) |
| v7a | [185/185](legacy-v7a-cuda.json) | [74/79](graph-v7a-cuda.json) | [114/120](math-v7a-cuda.json) | [53/233](polarity-v7a-baseline-cuda.json) | — | 717,554 | [60169ec](https://github.com/SamG-Coder/direct-weight-chat/commit/60169ec23186114b40069961c61c48dd01de51b9) |
| v8a | [185/185](legacy-v8a-cuda.json) | [75/79](graph-v8a-cuda.json) | [114/120](math-v8a-cuda.json) | [226/233](polarity-v8a-cuda.json) | — | 720,110 | [b75234e](https://github.com/SamG-Coder/direct-weight-chat/commit/b75234e3f018db2b84031e2901c710589a7521b0) |
| v8b | [185/185](legacy-v8b-cuda.json) | [75/79](graph-v8b-cuda.json) | [114/120](math-v8b-cuda.json) | [227/233](polarity-v8b-cuda.json) | [46/159](signed-v8b-baseline-cuda.json) | 720,111 | [0e52ba9](https://github.com/SamG-Coder/direct-weight-chat/commit/0e52ba9101e736e2d1340ead31b91ff054c0207f) |
| v9a | [185/185](legacy-v9a-cuda.json) | [75/79](graph-v9a-cuda.json) | [115/120](math-v9a-cuda.json) | [227/233](polarity-v9a-cuda.json) | [153/159](signed-v9a-cuda.json) | 720,343 | [74a4b81](https://github.com/SamG-Coder/direct-weight-chat/commit/74a4b81ea210907fc60a027f9182758f6ae32f3f) |

## Interpretation

The legacy suite is a restricted development suite run retrospectively on v1–v4. It is not a general-chat benchmark. Its eleven broad probes are retained outside the exact-output score. v9a answers the two-link inheritance and negative-fact probes; creative writing, code generation, instruction negation, general knowledge and most other broad requests still fail.

The graph suite tests six properties through unfamiliar class links with shuffled facts, distractors, corrections, cycles and missing links. v6c/v7a score 74/79; v8/v9 also solve the negative-fact stress for 75/79. Longer chains and an unsupported paraphrase still fail.

The arithmetic suite contains 88 subtraction, 20 addition and six mixed-turn cases, plus six failing stress cases. v7a/v8 pass 114/120; v9a adds signed operands for 115/120. The chained-expression case remains a failure against its mathematical answer; its newly correct abstention is measured separately and does not change that expected answer.

Polarity v8b/v9a: 227/233, with six scope/multiple-value failures. One case regresses from v7a: it now selects a later denied action rather than an earlier permitted action. [Regression comparison](regressions-v9a.json) preserves exact before/after outputs.

Signed operands v9a: 153/159, including all random signed, unary/prefix, dialogue, count and range cases. Three malformed syntax cases still emit numbers; multiplication, division and parentheses remain failures. These are not claims of a complete expression parser.

## Fresh combinations after freezing

- [v6c graph CUDA](graph-v6c-postfreeze-cuda.json) and [CPU](graph-v6c-postfreeze-cpu.json): 74/79, including 48/48 novel one-to-three-link chains. Seed 965544940 was chosen after commit `95f9754` froze the candidate.
- [v7a arithmetic CUDA](math-v7a-postfreeze-cuda.json) and [CPU](math-v7a-postfreeze-cpu.json): 114/120. Seed 2964068083 was chosen after commit `60169ec` froze the candidate. This adds 100 fresh operand combinations; the 20 fixed boundary, dialogue and stress cases repeat.
- [v7a graph regression](graph-v7a-regression-cuda.json): 74/79 on the former v6c fresh set, now treated as regression evidence.
- [v6c integrity](integrity-v6c-cuda.json) and [v7a integrity](integrity-v7a-cuda.json): hashes, exact layout/size, finite values, zero causal-prefix difference, cache parity, offline guard and graph/output-head ablations passed.

- [v7a arithmetic ablation](integrity-v7a-arithmetic-cuda.json): disabling its neural accumulator changes `-221.` to `-0.`.
- [Public release verification](publication.json): all ten uploaded archive sizes/digests match; [fresh-download checks](release-validation/verification.json) passed 39/39 and [CPU/CUDA smoke cases](release-validation/smoke.json) passed 4/4.

The generators and language structures were known before freezing. These are fresh combinations under a known grammar, not independently designed tests of general intelligence. No model was tuned after inspecting its own post-freeze results.

## Additional frozen candidates

- v8b polarity: [CUDA](polarity-v8b-postfreeze-cuda.json) and [CPU](polarity-v8b-postfreeze-cpu.json), 227/233 with seed 1760738774 selected after commit `0e52ba9`.
- v9a signed operands: [CUDA](signed-v9a-postfreeze-cuda.json) and [CPU](signed-v9a-postfreeze-cpu.json), 153/159 with seed 3220192550 selected after commit `74a4b81`. There are 120 fresh random combinations and 39 fixed cases.
- [v9a integrity and ablations](integrity-v9a-cuda.json): polarity, unary sign, signed accumulator and operand-count interventions change the associated behavior. [Reproduction](reproduction-v9a.json) matches raw weight/config hashes.

## Provenance and timing

[experiments.json](experiments.json) maps each checkpoint to the commit containing its exact compiler bytes, model hash and measured counts. Each raw report also records the Git HEAD and dirty flag at execution time. A dirty execution tree is not a claim that HEAD alone contains the candidate; use the indexed frozen source commit and hashes. Original v1–v5 builds predate this repository, so their source commit is the import commit.

Response timings include token generation and formatting, exclude one warm-up, and depend on prompt/output length and machine load. They are not a comparison against trained language models. Exact per-case timings and hardware are in each JSON. CUDA and CPU evaluations were run separately.

## Rejected selector v0

v0 is a different architecture with canned responses. Its routing checks passed 67/68 direct routes and 84/86 follow-ups; arithmetic passed 200/200 digit/carry and 260/260 full-sum checks. [CUDA evidence](../chat1gb/evidence/verification-torch-cuda.json) and [CPU evidence](../chat1gb/evidence/verification-numpy-cpu.json) are retained. Those scores are not autoregressive conversation scores and this prototype does not satisfy the current research constraints.
