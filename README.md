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

The current frozen version is **v15a**, available in the
[v0.7 experimental release](https://github.com/SamG-Coder/direct-weight-chat/releases/tag/v0.7.0-experiment).
It repairs demonstrated CPU subject-binding errors in long conversations.
Attention selects the latest valid event, then neural gates check that it belongs
to the current clause. This prevents tiny coordinate errors from making an older
subject acquire a later object's fact. The tensor runtime remains unchanged.
Tests and failed intermediate versions are public in the [version table](results/README.md).

| Development/regression check | v14b | Current v15a |
|---|---:|---:|
| Subject binding, CPU | 133/136 | 136/136 |
| Subject binding, CUDA | 136/136 | 136/136 |
| Previously published context set, CPU | 125/126 | 126/126 |
| Previously published context set, CUDA | 126/126 | 126/126 |
| Previous long 20-turn conversation, CPU | 19/20 | 20/20 |
| Previous long 20-turn conversation, CUDA | 20/20 | 20/20 |
| Original narrow cases, CUDA | 185/185 | 185/185 |
| Coordinated assertions and stress, CUDA | 382/390 | 382/390 |

All 29 existing CUDA suites were rerun with no regressions against v14b. Graph
remains 75/79; arithmetic 115/120; polarity 230/233; signed operands 153/159;
truth 236/239; query scope 265/269; composition 398/404; modifiers 78/81;
boundary controls 76/76; context development 126/126. The older post-freeze sets
are regression evidence for this candidate, not new held-out tests.

A trace of the original CPU failure shows a coordinate difference of about
0.0000038 amplified into a 64-point attention bias, overpowering a 40-point
recency advantage. v15a separates those operations. The repaired trace selects
the correct later subject; the new suite checks all seven relations, polarity,
isolation, corrections and dialogue across positions. Integrity and neural
interventions pass on both devices, the 45,432-address audit retains zero errors,
and the archived source reproduces exact raw/config hashes.

After freezing v15a at `a583d2b`, seed 192095465 scores **136/136 on CPU and
CUDA** in the binding suite. It varies unfamiliar names, combinations and eight
random offsets under known structures; it is not an independently designed
general-language evaluation. The frozen model was not retuned after these results.

Useful open-ended conversation, creative writing and code generation remain
unachieved. Conditional rules, disjunctive assertions, mixed query precedence
and other language structures still fail. Passing the address diagnostic does
not establish semantic correctness throughout the context window. Correct
abstention on a poem request does not demonstrate poetry generation.
v15a uses **961,697 nonzero parameters (0.3846788%)** of the 250 million slots.

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
`v13a`, `v13b`, `v13c`, `v13d`, `v14a`, `v14b` or `v15a`.
`v0` builds the separate response-selector prototype. Builders refuse to overwrite
an existing directory. The archived source bytes and checkpoint hashes are
preserved. Raw weight files are excluded from Git; compressed checkpoints are
published as release assets.

To run a downloaded checkpoint, extract its ZIP in the repository root.
The v15a archive supplies `chat1gb_autoregressive/model/weights.f32`; then run
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
python -m benchmarks.integrity --model build/current --source versions/autoregressive_v15a --output results/local-integrity.json --arithmetic --arithmetic-layer 10 --polarity --signed-operands --truth --query-negation --composition --compound-facts --deferred-assertions --periodic-lookbacks --fact-canonicalization --segmented-binding
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
