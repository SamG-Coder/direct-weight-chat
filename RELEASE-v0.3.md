# v0.3 experiment: object-specific truth and query negation

Two new analytically compiled checkpoints, v10a and v11a, extend the existing
offline transformer. Each contains exactly 250,000,000 float32 parameters and
1,000,000,000 raw weight bytes. No training, pretrained weights, downloaded
knowledge or runtime answer generator is used.

- **v10a:** exact subject/predicate/object retrieval, three-valued truth,
  per-object corrections and separation of questions from assertions.
- **v11a (current):** all seven existing predicates, copular and auxiliary
  inversion, and a separate query-negation bit. Missing evidence stays unknown.

| Development suite | v9a | v10a | v11a |
|---|---:|---:|---:|
| Legacy | 185/185 | 185/185 | 185/185 |
| Graph | 75/79 | 75/79 | 75/79 |
| Arithmetic | 115/120 | 115/120 | 115/120 |
| Polarity | 227/233 | 229/233 | 229/233 |
| Signed operands | 153/159 | 153/159 | 153/159 |
| Object-specific truth | 45/239 | 229/239 | 234/239 |
| Query scope | Not run | 80/269 | 262/269 |

Seeds selected after freezing reproduce v10a truth 229/239 and v11a query scope
262/269 on both CPU and CUDA. These are fresh combinations under known grammar,
not independent general-intelligence tests. Exact outputs, timings, source
commits, hashes, failures and 22 matched-suite comparisons are in `results/`.

The runtime is unchanged. Causal-prefix, cached/full decoding, file-integrity,
offline-guard and neural ablation checks pass. Removing proposition retrieval,
object identity or query negation changes the associated answers. The archived
v11a compiler reproduces raw weight and config hashes exactly.

**Still an experiment:** most parameters are zero (v11a has 744,215 nonzero
parameters). Open-ended conversation, creative writing and code generation remain
unachieved. Conjunction, general pronouns, repeated negation and instruction scope
fail. v10a adds one later-instruction regression; v11a adds none across the matched
suites. The earlier open-question multiple-action regression is still present.
No failing case was removed and no broad research milestone is declared complete.

Extract `autoregressive-v11a.zip` into the repository root and run:

```powershell
python -m chat1gb_autoregressive.chat --device auto
```

It supplies the default model directory. The v10a archive uses
`versions/autoregressive_v10a/model`; select that path with `--model-dir`.
Check archive hashes against `SHA256SUMS.txt`, then raw/config hashes against
the included manifest. The small ZIPs reflect sparse/repetitive weights; every
unpacked checkpoint is exactly 1 GB. NumPy/PyTorch must already be installed;
construction and inference perform no downloads.
