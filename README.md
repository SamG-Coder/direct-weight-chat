# direct-weight-chat

An experimental 1 GB neural network built by writing its weights analytically:
no training, pretrained parameters, downloaded datasets, or network access during
construction and inference. GitHub is used only to publish code and evidence.

**This is a research experiment, not a useful general-purpose chat model.** The
autoregressive model emits tokens through an ordinary causal transformer, with
compiled circuits for restricted grammar, facts, context retrieval and arithmetic.
It does not contain a bank of completed responses. Its narrow grammar is still
hand-programmed in weights; token-by-token generation does not establish language
understanding. Most parameters are zero. The older `chat1gb` response-selector
prototype is retained only as a rejected historical baseline.

Every checkpoint has exactly **250,000,000 float32 parameters**, occupying
**1,000,000,000 bytes** (decimal GB; 0.9313 GiB). File size is not a capability claim.

The current frozen version is **v17a**, supplied in the
[v0.9 experimental release](https://github.com/SamG-Coder/direct-weight-chat/releases/tag/v0.9.0-experiment).
It repairs CPU rule retrieval across long contexts. A shared saturating Boolean
circuit makes rule-phase flags stable before sharp attention ranking. Rules
supplied in the conversation still support signed ground implications through
up to three links. The ordinary tensor runtime, grammar and layout are unchanged.

| Development/regression check | v16c | Current v17a |
|---|---:|---:|
| Rule positions, CPU | 154/162 | 160/162 |
| Rule positions, CUDA | 160/162 | 160/162 |
| Long conversation, CPU | 28/31 | 31/31 |
| Long conversation, CUDA | 31/31 | 31/31 |
| Rule-phase perturbations, CPU and CUDA | 164/220 | 220/220 |
| Ground conditionals, CPU and CUDA | 269/293 | 269/293 |
| Additional scope/dialogue guards, CPU and CUDA | 29/36 | 29/36 |
| Original narrow cases, CUDA | 185/185 | 185/185 |

All **36 matched CUDA suites** have identical outputs to v16c: **5,959/6,105**,
with zero regressions. The four matched CPU suites gain nine passes with no
regressions. Historical suites repeat structures; these totals are not counts
of independent reasoning problems. Exact prompts, outputs, timings and failed
candidates are retained in the [version table](results/README.md),
[CUDA comparison](results/regressions-v17a-cuda.json) and
[CPU comparison](results/regressions-v17a-cpu.json).

Tests and frozen-v16c baselines preceded the compiler change. A read-only trace
located the CPU error: a small phase-flag error outweighed literal recency.
The new gate adds two ReLU units and 24 nonzero weights. In a targeted CPU
[intervention](results/rule-phase-ablation-v17a-cpu.json), restoring only the old
context gate changes the repaired answer from `Yes.` back to `I do not know.`.
Both devices pass the existing 27 integrity check groups and eight conditional
interventions. Archived source reproduces the exact raw/config hashes.

After source freeze commit `83a2f03`, seed **2165866013** scores **160/162 on
CPU and CUDA**. These are fresh combinations and offsets from a known generator,
not independently designed language tests. The two failures require a fourth
rule link or an older alternative proof. The frozen model was not retuned after
observing them. Passing these contexts does not prove reliability at every
possible prompt, numerical perturbation or device.

Useful open-ended conversation, creative writing and code generation remain
unachieved. This circuit follows the latest matching rule along one path; it
cannot search earlier alternatives or traverse more than three rule links.
Variables, compound rule premises and open answers derived from rules are
unsupported. Some paraphrases, including `implies` and `whenever`, can be
misread as unconditional assertions. Their false positives remain scored failures.
v17a uses **995,905 nonzero parameters (0.398362%)** of the 250 million slots.
Earlier releases preserve every failed checkpoint and its original results.
The next capability baseline, [alternative proof search](results/proof-search-v17a-cpu.json),
scores **29/104 on CPU and CUDA**. It tests whether a valid proof can be found
among multiple rules and mixed branches. No improvement has been implemented
for that suite yet; all 75 failures remain recorded.

## Run offline

Use a Python environment with NumPy for construction and PyTorch for inference
already installed. No packages or weights are downloaded by these commands.
The original runs used NumPy 2.5.3 and PyTorch 2.8.0+cu128, on an RTX 5080.

```powershell
python -m tools.build_version current --output build/current
python -m chat1gb_autoregressive.chat --model-dir build/current --device auto
python -m benchmarks.evaluate --model build/current --suite tests/legacy-185.json --output results/local-current.json --version current --device auto
```

For a historical build, replace `current` with an archived version such as `v5`,
`v6c`, `v7a`, `v8a`, `v8b`, `v9a`, `v10a`, `v11a`, `v12a`, `v12b`,
`v13a`, `v13b`, `v13c`, `v13d`, `v14a`, `v14b`, `v15a`, `v16a`, `v16b`, `v16c` or `v17a`.
`v0` builds the separate response-selector prototype. Builders refuse to overwrite
an existing directory. The archived source bytes and checkpoint hashes are
preserved. Raw weight files are excluded from Git; compressed checkpoints are
published as release assets.

To run a downloaded checkpoint, extract its ZIP in the repository root.
The v17a archive supplies `chat1gb_autoregressive/model/weights.f32`; then run
`python -m chat1gb_autoregressive.chat`. Historical archives use
`versions/autoregressive_VERSION/model`; select that directory with `--model-dir`.
Check archive SHA-256 against `SHA256SUMS.txt` and raw weight SHA-256 against the
included manifest. Small ZIPs reflect sparse/repetitive weights: every extracted
checkpoint still has exactly 1,000,000,000 parameter bytes.

```powershell
python -m benchmarks.evaluate --model build/current --suite tests/graph-development.json --output results/local-graph.json --version current
python -m benchmarks.evaluate --model build/current --suite tests/math-development.json --output results/local-math.json --version current
python -m benchmarks.evaluate --model build/current --suite tests/polarity-development.json --output results/local-polarity.json --version current
python -m benchmarks.evaluate --model build/current --suite tests/signed-development.json --output results/local-signed.json --version current
python -m benchmarks.evaluate --model build/current --suite tests/truth-development.json --output results/local-truth.json --version current
python -m benchmarks.evaluate --model build/current --suite tests/query-scope-development.json --output results/local-query-scope.json --version current
python -m benchmarks.evaluate --model build/current --suite tests/composition-development.json --output results/local-composition.json --version current
python -m benchmarks.evaluate --model build/current --suite tests/request-modifiers-development.json --output results/local-modifiers.json --version current
python -m benchmarks.evaluate --model build/current --suite tests/assertions-development.json --output results/local-assertions.json --version current
python -m benchmarks.evaluate --model build/current --suite tests/assertion-boundaries-v13a.json --output results/local-boundaries.json --version current
python -m benchmarks.evaluate --model build/current --suite tests/context-development.json --output results/local-context.json --version current
python -m benchmarks.lookbacks --model build/current --version current --device cuda --output results/local-lookbacks.json
python -m benchmarks.evaluate --model build/current --suite tests/binding-positions-development.json --output results/local-binding-cpu.json --version current --device cpu
python -m benchmarks.integrity --model build/current --source versions/autoregressive_v17a --output results/local-integrity.json --arithmetic --arithmetic-layer 10 --polarity --signed-operands --truth --query-negation --composition --compound-facts --deferred-assertions --periodic-lookbacks --fact-canonicalization --segmented-binding
python -m benchmarks.conditional_integrity --model build/current --version current --device cpu --output results/local-conditional-integrity.json
```

## Evidence and research

- [Version comparison](results/README.md): exact outputs and failures for each build.
- [Research log](RESEARCH_LOG.md): experiments, failures and next steps.
- [Capabilities](CAPABILITIES.md): measured abilities and limits.
- [Architecture](ARCHITECTURE.md): mathematical construction and runtime boundary.
- [v0.7 release verification](results/publication-v0.7.json): the downloaded archive matches public/local checksums, frozen source and decompressed hashes. Fresh tagged-checkout integrity and all interventions pass on CPU/CUDA; its 23-turn long conversation passes on both devices.
- [Earlier v0.6 release verification](results/publication-v0.6.json): the archived CPU failure remains public, alongside exact downloaded checkpoint checks.
- `tests/`: frozen development cases and separately identified post-freeze cases.
- `benchmarks/`: external evaluators; expected answers are never passed to inference.
- `versions/`: frozen historical compiler sources and checkpoint metadata.

The original v5 passed 185/185 restricted tests and 39/39 shorter CPU tests, while
failing the broad requests for poetry, code, negation and general reasoning. The
current model improves bounded relationship composition, fact/query negation, Boolean composition and signed arithmetic, but
still fails open-ended conversation, creative writing and code generation. The
185-case suite was evaluated retrospectively on earlier versions. It is a
development suite, not an independent general intelligence benchmark. All
observed failures are retained.

This work was inspired by the user's request to attempt direct weight writing
after seeing [weight-space-bench](https://github.com/eomcaleb/weight-space-bench).
It is a separate experiment; it is not an official benchmark submission or a
reproduction of that repository's rules. The construction code was written with
an AI coding assistant, so human/assistant prior knowledge enters the algorithms
and authored atomic facts. No pretrained tensor values are transferred.

MIT licensed. The Python network audit guard is application-level, not an OS
firewall. Inference uses tokenizer bookkeeping, generic tensor operations,
argmax token selection, KV caching and detokenization; it performs no host-side
reasoning or answer lookup.
