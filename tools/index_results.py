"""Generate the public version table and source-commit provenance index."""
import json
import re
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    versions = sorted([p.name.removeprefix('autoregressive_') for p in (root / 'versions').glob('autoregressive_*')], key=lambda v: (int(re.search(r'v([0-9]+)', v).group(1)), v))
    rows = ['# Version results', '',
            'Every autoregressive checkpoint has exactly 250,000,000 float32 parameters and 1,000,000,000 weight bytes. Failed candidates remain in this table. A dash means the suite was not run for that version.', '',
            '| Version | Legacy /185 | Graph /79 | Arithmetic /120 | Polarity /233 | Signed /159 | Nonzero parameters | Source commit |',
            '|---|---:|---:|---:|---:|---:|---:|---|']
    index = []
    for v in versions:
        source = Path('versions') / ('autoregressive_' + v)
        m = json.loads((root / source / 'model' / 'manifest.json').read_text(encoding='utf-8'))
        commit = subprocess.check_output(['git', 'log', '--diff-filter=A', '--format=%H', '--', str(source / 'build.py')], cwd=root, text=True).strip().splitlines()[-1]
        scores, reports = [], []
        for suite in ('legacy', 'graph', 'math', 'polarity', 'signed'):
            p = root / 'results' / f'{suite}-{v}-cuda.json'
            if not p.exists():
                p = root / 'results' / f'{suite}-{v}-baseline-cuda.json'
            if p.exists():
                r = json.loads(p.read_text(encoding='utf-8'))
                score = r['summary']['all']
                scores.append(f"[{score['passed']}/{score['total']}]({p.name})")
                reports.append(dict(path='results/' + p.name, summary=r['summary'],
                                    execution_git=r['git'], model_sha256=r['model_sha256']))
            else:
                scores.append('—')
        rows.append(f"| {v} | {' | '.join(scores)} | {m['nonzero_parameters']:,} | [{commit[:7]}](https://github.com/SamG-Coder/direct-weight-chat/commit/{commit}) |")
        index.append(dict(version=v, source_commit=commit, source_path=source.as_posix(),
                          model_sha256=m['sha256']['weights.f32'], source_sha256=m['source_sha256'],
                          parameter_count=m['parameter_count'], weight_bytes=m['weight_bytes'],
                          nonzero_parameters=m['nonzero_parameters'], development_reports=reports))
    rows += ['', '## Interpretation', '',
             'The legacy suite is a restricted development suite run retrospectively on v1–v4. It is not a general-chat benchmark. Its eleven broad probes are retained outside the exact-output score. v9a answers the two-link inheritance and negative-fact probes; creative writing, code generation, instruction negation, general knowledge and most other broad requests still fail.', '',
             'The graph suite tests six properties through unfamiliar class links with shuffled facts, distractors, corrections, cycles and missing links. v6c/v7a score 74/79; v8/v9 also solve the negative-fact stress for 75/79. Longer chains and an unsupported paraphrase still fail.', '',
             'The arithmetic suite contains 88 subtraction, 20 addition and six mixed-turn cases, plus six failing stress cases. v7a/v8 pass 114/120; v9a adds signed operands for 115/120. The chained-expression case remains a failure against its mathematical answer; its newly correct abstention is measured separately and does not change that expected answer.', '',
             'Polarity v8b/v9a: 227/233, with six scope/multiple-value failures. One case regresses from v7a: it now selects a later denied action rather than an earlier permitted action. [Regression comparison](regressions-v9a.json) preserves exact before/after outputs.', '',
             'Signed operands v9a: 153/159, including all random signed, unary/prefix, dialogue, count and range cases. Three malformed syntax cases still emit numbers; multiplication, division and parentheses remain failures. These are not claims of a complete expression parser.', '',
             '## Fresh combinations after freezing', '',
             '- [v6c graph CUDA](graph-v6c-postfreeze-cuda.json) and [CPU](graph-v6c-postfreeze-cpu.json): 74/79, including 48/48 novel one-to-three-link chains. Seed 965544940 was chosen after commit `95f9754` froze the candidate.',
             '- [v7a arithmetic CUDA](math-v7a-postfreeze-cuda.json) and [CPU](math-v7a-postfreeze-cpu.json): 114/120. Seed 2964068083 was chosen after commit `60169ec` froze the candidate. This adds 100 fresh operand combinations; the 20 fixed boundary, dialogue and stress cases repeat.',
             '- [v7a graph regression](graph-v7a-regression-cuda.json): 74/79 on the former v6c fresh set, now treated as regression evidence.',
             '- [v6c integrity](integrity-v6c-cuda.json) and [v7a integrity](integrity-v7a-cuda.json): hashes, exact layout/size, finite values, zero causal-prefix difference, cache parity, offline guard and graph/output-head ablations passed.', '',
             '- [v7a arithmetic ablation](integrity-v7a-arithmetic-cuda.json): disabling its neural accumulator changes `-221.` to `-0.`.',
             '- [Public release verification](publication.json): all ten uploaded archive sizes/digests match; [fresh-download checks](release-validation/verification.json) passed 39/39 and [CPU/CUDA smoke cases](release-validation/smoke.json) passed 4/4.', '',
             'The generators and language structures were known before freezing. These are fresh combinations under a known grammar, not independently designed tests of general intelligence. No model was tuned after inspecting its own post-freeze results.', '',
             '## Additional frozen candidates', '',
             '- v8b polarity: [CUDA](polarity-v8b-postfreeze-cuda.json) and [CPU](polarity-v8b-postfreeze-cpu.json), 227/233 with seed 1760738774 selected after commit `0e52ba9`.',
             '- v9a signed operands: [CUDA](signed-v9a-postfreeze-cuda.json) and [CPU](signed-v9a-postfreeze-cpu.json), 153/159 with seed 3220192550 selected after commit `74a4b81`. There are 120 fresh random combinations and 39 fixed cases.',
             '- [v9a integrity and ablations](integrity-v9a-cuda.json): polarity, unary sign, signed accumulator and operand-count interventions change the associated behavior. [Reproduction](reproduction-v9a.json) matches raw weight/config hashes.', '',
             '- [v0.2 publication](publication-v0.2.json): three uploaded archive hashes/sizes verified, then a fresh clone and download passed [integrity/ablations](release-validation-v0.2/integrity.json) and [six CPU/CUDA smoke cases](release-validation-v0.2/smoke.json).', '',
             '## Provenance and timing', '',
             '[experiments.json](experiments.json) maps each checkpoint to the commit containing its exact compiler bytes, model hash and measured counts. Each raw report also records the Git HEAD and dirty flag at execution time. A dirty execution tree is not a claim that HEAD alone contains the candidate; use the indexed frozen source commit and hashes. Original v1–v5 builds predate this repository, so their source commit is the import commit.', '',
             'Response timings include token generation and formatting, exclude one warm-up, and depend on prompt/output length and machine load. They are not a comparison against trained language models. Exact per-case timings and hardware are in each JSON. CUDA and CPU evaluations were run separately.', '',
             '## Rejected selector v0', '',
             'v0 is a different architecture with canned responses. Its routing checks passed 67/68 direct routes and 84/86 follow-ups; arithmetic passed 200/200 digit/carry and 260/260 full-sum checks. [CUDA evidence](../chat1gb/evidence/verification-torch-cuda.json) and [CPU evidence](../chat1gb/evidence/verification-numpy-cpu.json) are retained. Those scores are not autoregressive conversation scores and this prototype does not satisfy the current research constraints.']
    (root / 'results' / 'README.md').write_text('\n'.join(rows) + '\n', encoding='utf-8', newline='\n')
    (root / 'results' / 'experiments.json').write_text(json.dumps(index, indent=2) + '\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
