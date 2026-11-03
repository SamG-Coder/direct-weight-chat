# Version results

Every autoregressive checkpoint has exactly 250,000,000 float32 parameters and 1,000,000,000 weight bytes. Failed candidates remain in this table. A dash means the suite was not run for that version.

| Version | Legacy /185 | Graph /79 | Arithmetic /120 | Polarity /233 | Signed /159 | Truth /239 | Query scope /269 | Composition /404 | Modifiers /81 | Nonzero parameters | Source commit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v1 | [131/185](legacy-v1-cuda.json) | — | — | — | — | — | — | — | — | 674,640 | [5be0768](https://github.com/SamG-Coder/direct-weight-chat/commit/5be0768f6e2d61c1bacd30950cf343cd79173b74) |
| v2 | [177/185](legacy-v2-cuda.json) | — | — | — | — | — | — | — | — | 691,848 | [5be0768](https://github.com/SamG-Coder/direct-weight-chat/commit/5be0768f6e2d61c1bacd30950cf343cd79173b74) |
| v3 | [182/185](legacy-v3-cuda.json) | — | — | — | — | — | — | — | — | 697,264 | [5be0768](https://github.com/SamG-Coder/direct-weight-chat/commit/5be0768f6e2d61c1bacd30950cf343cd79173b74) |
| v4 | [182/185](legacy-v4-cuda.json) | — | — | — | — | — | — | — | — | 697,392 | [5be0768](https://github.com/SamG-Coder/direct-weight-chat/commit/5be0768f6e2d61c1bacd30950cf343cd79173b74) |
| v5 | [185/185](legacy-v5-cuda.json) | [12/79](graph-v5-baseline-cuda.json) | — | — | — | — | — | — | — | 697,392 | [5be0768](https://github.com/SamG-Coder/direct-weight-chat/commit/5be0768f6e2d61c1bacd30950cf343cd79173b74) |
| v6a | [1/185](legacy-v6a-cuda.json) | [8/79](graph-v6a-cuda.json) | — | — | — | — | — | — | — | 716,888 | [1f7509f](https://github.com/SamG-Coder/direct-weight-chat/commit/1f7509f7a7e9251386c445c722dab5701fd591ef) |
| v6b | [182/185](legacy-v6b-cuda.json) | [68/79](graph-v6b-cuda.json) | — | — | — | — | — | — | — | 716,960 | [ddfaa85](https://github.com/SamG-Coder/direct-weight-chat/commit/ddfaa85087e60339a058933b5aa09667f43906f1) |
| v6c | [185/185](legacy-v6c-cuda.json) | [74/79](graph-v6c-cuda.json) | [22/120](math-v6c-baseline-cuda.json) | — | — | — | — | — | — | 717,352 | [95f9754](https://github.com/SamG-Coder/direct-weight-chat/commit/95f9754da6d5cccc8d4170dc767875a4867a8edb) |
| v7a | [185/185](legacy-v7a-cuda.json) | [74/79](graph-v7a-cuda.json) | [114/120](math-v7a-cuda.json) | [53/233](polarity-v7a-baseline-cuda.json) | — | — | — | — | — | 717,554 | [60169ec](https://github.com/SamG-Coder/direct-weight-chat/commit/60169ec23186114b40069961c61c48dd01de51b9) |
| v8a | [185/185](legacy-v8a-cuda.json) | [75/79](graph-v8a-cuda.json) | [114/120](math-v8a-cuda.json) | [226/233](polarity-v8a-cuda.json) | — | — | — | — | — | 720,110 | [b75234e](https://github.com/SamG-Coder/direct-weight-chat/commit/b75234e3f018db2b84031e2901c710589a7521b0) |
| v8b | [185/185](legacy-v8b-cuda.json) | [75/79](graph-v8b-cuda.json) | [114/120](math-v8b-cuda.json) | [227/233](polarity-v8b-cuda.json) | [46/159](signed-v8b-baseline-cuda.json) | — | — | — | — | 720,111 | [0e52ba9](https://github.com/SamG-Coder/direct-weight-chat/commit/0e52ba9101e736e2d1340ead31b91ff054c0207f) |
| v9a | [185/185](legacy-v9a-cuda.json) | [75/79](graph-v9a-cuda.json) | [115/120](math-v9a-cuda.json) | [227/233](polarity-v9a-cuda.json) | [153/159](signed-v9a-cuda.json) | [45/239](truth-v9a-baseline-cuda.json) | — | — | — | 720,343 | [74a4b81](https://github.com/SamG-Coder/direct-weight-chat/commit/74a4b81ea210907fc60a027f9182758f6ae32f3f) |
| v10a | [185/185](legacy-v10a-cuda.json) | [75/79](graph-v10a-cuda.json) | [115/120](math-v10a-cuda.json) | [229/233](polarity-v10a-cuda.json) | [153/159](signed-v10a-cuda.json) | [229/239](truth-v10a-cuda.json) | [80/269](query-scope-v10a-baseline-cuda.json) | — | — | 732,615 | [6bb1736](https://github.com/SamG-Coder/direct-weight-chat/commit/6bb1736abc175a14774c796f5540f4078f879bed) |
| v11a | [185/185](legacy-v11a-cuda.json) | [75/79](graph-v11a-cuda.json) | [115/120](math-v11a-cuda.json) | [229/233](polarity-v11a-cuda.json) | [153/159](signed-v11a-cuda.json) | [234/239](truth-v11a-cuda.json) | [262/269](query-scope-v11a-cuda.json) | [243/404](composition-v11a-baseline-cuda.json) | — | 744,215 | [55ab049](https://github.com/SamG-Coder/direct-weight-chat/commit/55ab049010292835bb6eb83b74647b50df0df9d4) |
| v12a | [185/185](legacy-v12a-cuda.json) | [75/79](graph-v12a-cuda.json) | [88/120](math-v12a-cuda.json) | [230/233](polarity-v12a-cuda.json) | [121/159](signed-v12a-cuda.json) | [236/239](truth-v12a-cuda.json) | [265/269](query-scope-v12a-cuda.json) | [398/404](composition-v12a-cuda.json) | [17/81](request-modifiers-v12a-baseline-cuda.json) | 788,721 | [e1e262c](https://github.com/SamG-Coder/direct-weight-chat/commit/e1e262c3d8176fa53946a35030db6b575179b149) |
| v12b | [185/185](legacy-v12b-cuda.json) | [75/79](graph-v12b-cuda.json) | [115/120](math-v12b-cuda.json) | [230/233](polarity-v12b-cuda.json) | [153/159](signed-v12b-cuda.json) | [236/239](truth-v12b-cuda.json) | [265/269](query-scope-v12b-cuda.json) | [397/404](composition-v12b-cuda.json) | [78/81](request-modifiers-v12b-cuda.json) | 873,633 | [49dd2ea](https://github.com/SamG-Coder/direct-weight-chat/commit/49dd2eaaf0f4f34982713dc4711b820cb2768e2c) |

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

- [v0.2 publication](publication-v0.2.json): three uploaded archive hashes/sizes verified, then a fresh clone and download passed [integrity/ablations](release-validation-v0.2/integrity.json) and [six CPU/CUDA smoke cases](release-validation-v0.2/smoke.json).

## Object-specific queries

- v10a adds exact subject/predicate/object matching: truth 229/239 versus the frozen v9a baseline of 45/239. Its post-freeze seed 34865453 reproduced 229/239 on [CUDA](truth-v10a-postfreeze-cuda.json) and [CPU](truth-v10a-postfreeze-cpu.json), after source commit `6bb1736`.
- v11a extends predicate binding to all seven relations and applies query negation to known truth values. Missing evidence remains unknown. Truth scores 234/239 and query scope 262/269. After freeze commit `55ab049`, seed 3050431372 reproduced 262/269 on [CUDA](query-scope-v11a-postfreeze-cuda.json) and [CPU](query-scope-v11a-postfreeze-cpu.json). Unsupported conjunctions, pronouns and instruction scope remain in the score.
- [Matched-suite comparisons](regressions-v11a.json) record all output changes across v9a, v10a and v11a: 22 matched suites, one v10a regression on a later ignore-question instruction, and no further v11a regressions. The older open-question multiple-action regression also remains; object-specific queries do not fix the separate wh-answer selection path.
- [v10a ablations](integrity-v10a-cuda.json) and [v11a ablations](integrity-v11a-cuda.json) remove proposition retrieval, object identity or query negation. The corresponding answers change while the saved checkpoint stays unchanged.

- [v11a reproduction](reproduction-v11a.json) verifies identical raw weight/config hashes from the frozen source. [v0.3 packaging](release-assets-v0.3.json) records the two checkpoint archives and streamed decompression checks.

- [v0.3 publication verification](publication-v0.3.json): both uploaded archive hashes/sizes match. Fresh clone/download integrity and all interventions pass. [CUDA](release-validation-v0.3/fresh-smoke-cuda.json) and [CPU](release-validation-v0.3/fresh-smoke-cpu.json) each score 11/12: all 11 mixed-conversation checks pass; the creative-request abstention check fails, producing factual rain sentences instead of abstaining. It remains a failed check.

## Proposition composition and request scope

- v12a compiles atom events, three-valued AND/OR reductions and clause IDs: composition improves from v11a baseline 243/404 to 398/404. Its request-head filter rejects Please calculate, causing 106 arithmetic regressions across four reports. The failed candidate and [comparisons](regressions-v12a.json) remain preserved.
- v12b restores all 106 arithmetic cases through a bounded modifier grammar: arithmetic 115/120, signed 153/159, modifiers 78/81, composition 397/404. One unsupported paraphrase loses an accidental v12a abstention pass. [Adjacent-version comparisons](regressions-v12b.json) and [direct v11a-to-v12b comparisons](regressions-v11a-to-v12b.json) retain all changes, including three composition regressions against v11a.
- [v12a integrity](integrity-v12a-cuda.json) and [v12b integrity](integrity-v12b-cuda.json) include interventions on AND/OR, unknown evidence and clause scope, plus all earlier checks. The inference runtime is unchanged.

- After freeze commit `49dd2ea`, composition seed 1676552291 reproduced 397/404 on [CUDA](composition-v12b-postfreeze-cuda.json) and [CPU](composition-v12b-postfreeze-cpu.json). Modifier seed 973315925 reproduced 78/81 on [CUDA](request-modifiers-v12b-postfreeze-cuda.json) and [CPU](request-modifiers-v12b-postfreeze-cpu.json). The latter adds 20 fresh arithmetic combinations; its 61 structural cases repeat. Both generators were known before freezing.
- [Reproduction](reproduction-v12b.json) compares exact raw/config hashes from the archived v12b compiler; [v0.4 packaging](release-assets-v0.4.json) verifies both archived checkpoints after decompression.

- [v0.4 publication verification](publication-v0.4.json): both uploaded archive sizes/digests match. A fresh clone/download passed integrity and all interventions; its 16-turn smoke conversation scored 16/16 on [CUDA](release-validation-v0.4/fresh-smoke-cuda.json) and [CPU](release-validation-v0.4/fresh-smoke-cpu.json). Creative/code-request abstentions are not claims of generation capability.

## Provenance and timing

[experiments.json](experiments.json) maps each checkpoint to the commit containing its exact compiler bytes, model hash and measured counts. Each raw report also records the Git HEAD and dirty flag at execution time. A dirty execution tree is not a claim that HEAD alone contains the candidate; use the indexed frozen source commit and hashes. Original v1–v5 builds predate this repository, so their source commit is the import commit.

Response timings include token generation and formatting, exclude one warm-up, and depend on prompt/output length and machine load. They are not a comparison against trained language models. Exact per-case timings and hardware are in each JSON. CUDA and CPU evaluations were run separately.

## Rejected selector v0

v0 is a different architecture with canned responses. Its routing checks passed 67/68 direct routes and 84/86 follow-ups; arithmetic passed 200/200 digit/carry and 260/260 full-sum checks. [CUDA evidence](../chat1gb/evidence/verification-torch-cuda.json) and [CPU evidence](../chat1gb/evidence/verification-numpy-cpu.json) are retained. Those scores are not autoregressive conversation scores and this prototype does not satisfy the current research constraints.
