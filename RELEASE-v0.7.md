# v0.7 experiment: preserve subjects across long contexts

> Historical release notes. The current compact artifact is exactly **1,528,592 packed weight bytes**. See the [current release](RELEASE-COMPACT.md); the old assets below remain historical.

The v15a checkpoint fixes demonstrated CPU subject-binding failures through a
shared neural retrieval primitive. It still has exactly 250,000,000 float32
parameters, with no padding, training,
pretrained parameters or downloaded data. Construction and inference remain
offline; the ordinary transformer runtime is unchanged.

An activation trace found that two copies of one clause coordinate differed by
about 0.0000038. The previous attention coefficient amplified this into a
64-point bias, overpowering the newer subject's 40-point recency advantage.
The second object could then be stored under the first subject, causing both
missed facts and false positive answers about the wrong subject.

v15a retrieves the latest valid event using a modest recency score, then checks
clause membership through the following neural gates. The same primitive handles
lexical predicates and subject frames; it adds no prompt-specific answer rule.

| Check | v14b | v15a |
|---|---:|---:|
| Subject binding, CPU | 133/136 | 136/136 |
| Subject binding, CUDA | 136/136 | 136/136 |
| Previously published context set, CPU | 125/126 | 126/126 |
| Previously published context set, CUDA | 126/126 | 126/126 |
| Previous long 20-turn conversation, CPU | 19/20 | 20/20 |
| Previous long 20-turn conversation, CUDA | 20/20 | 20/20 |

All 29 existing CUDA suites were rerun, with no regressions against v14b. Two
matched CPU suites and the separately compared long conversation retain their
old passes. The original source tests, exact outputs, timings, traces, regressions
and model hashes are in the [version history](https://github.com/SamG-Coder/direct-weight-chat/blob/v0.7.0-experiment/results/README.md).
The old post-freeze cases are explicitly regression evidence for this version.

After freeze commit `a583d2b49325d3ec66c937c32f06c43fa0b19ac4`, binding seed
192095465 scores **136/136 on both CPU and CUDA**. It varies unfamiliar names,
combinations and eight random offsets from a known generator; fixed offsets and
structures repeat. This is not independently designed generalization evidence.
No frozen model source or weights were retuned after these results.

CPU and CUDA integrity checks pass, including causal/cache parity, offline
guard, exact layout/size, finite weights and all neural interventions. The local
address audit retains zero errors out of 45,432 on each device. Rebuilding from
the frozen v15a source reproduces the exact raw and config hashes.
A local 23-turn CUDA publication smoke passes after 3,000 tokens of declarative
distractors, including new three-subject binding, isolation and correction checks.

Fresh-download verification confirms the published archive size/digest, checksum
entry, raw/config hashes, ZIP CRC and frozen-source manifest. From the clean
tagged checkout, **CPU and CUDA each pass all integrity/intervention checks and
23/23 long-conversation turns**. The saved checkpoint remains unchanged. Exact
outputs and timings are in the [publication evidence](https://github.com/SamG-Coder/direct-weight-chat/blob/main/results/publication-v0.7.json).

The initial two-subject recency intervention left its answer unchanged. That
ineffective probe and the stopped integrity attempt remain documented. Additional
three-subject controls expose missed facts and false cross-subject assertions
when recency is removed. No model source or saved weights changed in response
to that diagnostic outcome.

**Useful open-ended conversation remains unachieved.** This release repairs
numerical binding within the existing controlled grammar. Conditional rules,
disjunctive assertions, mixed query precedence, multiple class parents, repeated
negation, open multi-value answers, creative writing and code generation remain
unsupported. v15a uses 961,697 nonzero parameters (0.3846788%); most connections
remain zero. Restricted tests do not establish broad language understanding.

Extract `autoregressive-v15a.zip` into the repository root and run:

```powershell
python -m chat1gb_autoregressive.chat --device auto
```

Verify the ZIP hash against `SHA256SUMS.txt` and raw/config hashes against the
included manifest. The small archive reflects sparse and repetitive weights;
the extracted checkpoint uses the original dense tensor layout. NumPy and PyTorch must already be
installed. Every previous release, including failed candidates, remains public.
