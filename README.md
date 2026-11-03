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

The current frozen version is **v11a**, available with v10a in the
[v0.3 experimental release](https://github.com/SamG-Coder/direct-weight-chat/releases/tag/v0.3.0-experiment).
It adds object-specific truth queries across seven predicates and explicit
logical negation of the queried proposition. Multiple objects and their separate
corrections can be queried; missing evidence stays unknown. Tests and failed
intermediate versions are public in the [version table](results/README.md).

| Development suite | v9a | v10a | Current v11a |
|---|---:|---:|---:|
| Original narrow cases | 185/185 | 185/185 | 185/185 |
| Graph composition and stress | 75/79 | 75/79 | 75/79 |
| Arithmetic and stress | 115/120 | 115/120 | 115/120 |
| Fact polarity and scope stress | 227/233 | 229/233 | 229/233 |
| Signed operands and validity stress | 153/159 | 153/159 | 153/159 |
| Object-specific truth and stress | 45/239 | 229/239 | 234/239 |
| Query predicates, negation and stress | Not run | 80/269 | 262/269 |

Fresh seeds selected after candidate freezes reproduced v10a's 229/239 truth
score and v11a's 262/269 query-scope score on CUDA and CPU. Their structures were
already known; these are fresh combinations, not independent evidence of broad
intelligence. Failed stress cases remain in every score. The earlier open-question
multiple-action bug remains, and v10a introduced a later-instruction scope
regression. Exact before/after outputs are retained; v11a introduces no additional
pass/fail regressions across its matched suites.

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
`v6c`, `v7a`, `v8a`, `v8b`, `v9a`, `v10a` or `v11a`.
`v0` builds the separate response-selector prototype. Builders refuse to overwrite
an existing directory. The archived source bytes and checkpoint hashes are
preserved. Raw weight files are excluded from Git; compressed checkpoints are
published as release assets.

To run a downloaded checkpoint, extract its ZIP in the repository root.
The v11a archive supplies `chat1gb_autoregressive/model/weights.f32`; then run
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
python -m benchmarks.integrity --model build/current --source versions/autoregressive_v11a --output results/local-integrity.json --arithmetic --arithmetic-layer 10 --polarity --signed-operands --truth --query-negation
```

## Evidence and research

- [Version comparison](results/README.md): exact outputs and failures for each build.
- [Research log](RESEARCH_LOG.md): experiments, failures and next steps.
- [Capabilities](CAPABILITIES.md): measured abilities and limits.
- [Architecture](ARCHITECTURE.md): mathematical construction and runtime boundary.
- `tests/`: frozen development cases and separately identified post-freeze cases.
- `benchmarks/`: external evaluators; expected answers are never passed to inference.
- `versions/`: frozen historical compiler sources and checkpoint metadata.

The original v5 passed 185/185 restricted tests and 39/39 shorter CPU tests, while
failing the broad requests for poetry, code, negation and general reasoning. The
current model improves bounded relationship composition, fact/query negation and signed arithmetic, but
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
