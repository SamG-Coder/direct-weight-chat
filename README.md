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

The current frozen version is **v14b**, available with failed candidate v14a
in the [v0.6 experimental release](https://github.com/SamG-Coder/direct-weight-chat/releases/tag/v0.6.0-experiment).
It replaces numerically unstable local attention addresses with bounded
sinusoidal features, then saturates fact coordinates before proposition retrieval.
This improves existing grammar and reasoning at long context positions; it does
not add general language understanding. The tensor runtime remains unchanged.
Tests and failed intermediate versions are public in the [version table](results/README.md).

| Development suite, CUDA | v13d | Failed v14a | Current v14b |
|---|---:|---:|---:|
| Original narrow cases | 185/185 | 185/185 | 185/185 |
| Graph composition and stress | 75/79 | 75/79 | 75/79 |
| Arithmetic and stress | 115/120 | 115/120 | 115/120 |
| Fact polarity and scope stress | 230/233 | 230/233 | 230/233 |
| Signed operands and validity stress | 153/159 | 153/159 | 153/159 |
| Object-specific truth and stress | 236/239 | 236/239 | 236/239 |
| Query predicates, negation and stress | 265/269 | 265/269 | 265/269 |
| Proposition composition and scope | 398/404 | 398/404 | 398/404 |
| Request modifiers and stress | 78/81 | 78/81 | 78/81 |
| Coordinated assertions and stress | 382/390 | 366/390 | 382/390 |
| Assertion boundary audit | 65/76 | 76/76 | 76/76 |
| Context positions and generation | 61/126 | 126/126 | 126/126 |

The context tests move existing operations across the 5,048-position window,
including six requests near capacity. A separate diagnostic examines all nine
local lookback heads at every position: v13d has 15,900/45,432 incorrect addresses
on CUDA and 7,491/45,432 on CPU; v14a and v14b have zero on both devices.
v14a nevertheless regresses on sixteen assertion corrections. Shared fact
saturation repairs those in v14b, with no regressions against v13d across all
26 matched suites. Failed candidates and exact before/after outputs remain public.

After freeze commit `b3b6e1b`, context seed 2063643282 scores **126/126 on CUDA
and 125/126 on CPU**. The CPU failure has a 4,720-token prefix: an unfamiliar
subject's second coordinated object produces `I do not know.` instead of `Yes.`.
The failure is preserved and the frozen model was not retuned. Fresh-seed tests
use known generators and structures, so these are fresh combinations rather
than independently designed evidence of broad intelligence.

Useful open-ended conversation, creative writing and code generation remain
unachieved. Conditional rules, disjunctive assertions, mixed query precedence
and other language structures still fail. Passing the address diagnostic does
not establish semantic correctness throughout the context window. Correct
abstention on a poem request does not demonstrate poetry generation.
v14b uses **960,705 nonzero parameters (0.384282%)** of the 250 million slots.

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
`v13a`, `v13b`, `v13c`, `v13d`, `v14a` or `v14b`.
`v0` builds the separate response-selector prototype. Builders refuse to overwrite
an existing directory. The archived source bytes and checkpoint hashes are
preserved. Raw weight files are excluded from Git; compressed checkpoints are
published as release assets.

To run a downloaded checkpoint, extract its ZIP in the repository root.
The v14b archive supplies `chat1gb_autoregressive/model/weights.f32`; then run
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
python -m benchmarks.integrity --model build/current --source versions/autoregressive_v14b --output results/local-integrity.json --arithmetic --arithmetic-layer 10 --polarity --signed-operands --truth --query-negation --composition --compound-facts --deferred-assertions --periodic-lookbacks --fact-canonicalization
```

## Evidence and research

- [Version comparison](results/README.md): exact outputs and failures for each build.
- [Research log](RESEARCH_LOG.md): experiments, failures and next steps.
- [Capabilities](CAPABILITIES.md): measured abilities and limits.
- [Architecture](ARCHITECTURE.md): mathematical construction and runtime boundary.
- [v0.6 release verification](results/publication-v0.6.json): both downloaded archives match; integrity and all interventions pass on CPU/CUDA. The long 20-turn conversation scores 20/20 on CUDA and 19/20 on CPU. Its CPU binding failure remains public.
- Fresh v14b context evidence: [CUDA 126/126](results/context-v14b-postfreeze-cuda.json), [CPU 125/126](results/context-v14b-postfreeze-cpu.json), with the remaining CPU failure retained.
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
