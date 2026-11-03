"""Generate the public version table and source-commit provenance index."""
import json
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    versions = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6a', 'v6b', 'v6c', 'v7a']
    rows = ['# Version results', '',
            'Every autoregressive checkpoint has exactly 250,000,000 float32 parameters and 1,000,000,000 weight bytes. Failed candidates remain in this table. A dash means the suite was not run for that version.', '',
            '| Version | Legacy /185 | Graph /79 | Arithmetic /120 | Nonzero parameters | Source commit |',
            '|---|---:|---:|---:|---:|---|']
    index = []
    for v in versions:
        source = Path('versions') / ('autoregressive_' + v)
        m = json.loads((root / source / 'model' / 'manifest.json').read_text(encoding='utf-8'))
        commit = subprocess.check_output(['git', 'log', '--diff-filter=A', '--format=%H', '--', str(source / 'build.py')], cwd=root, text=True).strip().splitlines()[-1]
        scores, reports = [], []
        for suite in ('legacy', 'graph', 'math'):
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
             'The legacy suite is a restricted development suite run retrospectively on v1–v4. It is not a general-chat benchmark. Its eleven broad probes are retained outside the exact-output score. v7a now answers the two-link inheritance probe; creative writing, code generation, negation, general knowledge and most other broad requests still fail.', '',
             'The graph suite tests six properties through unfamiliar class links with shuffled facts, distractors, corrections, cycles and missing links. v6c and v7a pass all 74 cases within the implemented syntax/depth; all five depth/paraphrase/negation stress failures remain in the denominator.', '',
             'The arithmetic suite contains 88 subtraction, 20 addition and six mixed-turn cases, plus six failing stress cases. v7a passes 114/120. Negative **results** work; negative **operands**, chained expressions, multiplication, division, larger inputs and an unsupported paraphrase fail. Some unsupported inputs produce incorrect numbers rather than an unknown response.', '',
             '## Fresh combinations after freezing', '',
             '- [v6c graph CUDA](graph-v6c-postfreeze-cuda.json) and [CPU](graph-v6c-postfreeze-cpu.json): 74/79, including 48/48 novel one-to-three-link chains. Seed 965544940 was chosen after commit `95f9754` froze the candidate.',
             '- [v7a arithmetic CUDA](math-v7a-postfreeze-cuda.json) and [CPU](math-v7a-postfreeze-cpu.json): 114/120. Seed 2964068083 was chosen after commit `60169ec` froze the candidate. This adds 100 fresh operand combinations; the 20 fixed boundary, dialogue and stress cases repeat.',
             '- [v7a graph regression](graph-v7a-regression-cuda.json): 74/79 on the former v6c fresh set, now treated as regression evidence.',
             '- [v6c integrity](integrity-v6c-cuda.json) and [v7a integrity](integrity-v7a-cuda.json): hashes, exact layout/size, finite values, zero causal-prefix difference, cache parity, offline guard and graph/output-head ablations passed.', '',
             'The generators and language structures were known before freezing. These are fresh combinations under a known grammar, not independently designed tests of general intelligence. No model was tuned after inspecting its own post-freeze results.', '',
             '## Provenance and timing', '',
             '[experiments.json](experiments.json) maps each checkpoint to the commit containing its exact compiler bytes, model hash and measured counts. Each raw report also records the Git HEAD and dirty flag at execution time. A dirty execution tree is not a claim that HEAD alone contains the candidate; use the indexed frozen source commit and hashes. Original v1–v5 builds predate this repository, so their source commit is the import commit.', '',
             'Response timings include token generation and formatting, exclude one warm-up, and depend on prompt/output length and machine load. They are not a comparison against trained language models. Exact per-case timings and hardware are in each JSON. CUDA and CPU evaluations were run separately.', '',
             '## Rejected selector v0', '',
             'v0 is a different architecture with canned responses. Its routing checks passed 67/68 direct routes and 84/86 follow-ups; arithmetic passed 200/200 digit/carry and 260/260 full-sum checks. [CUDA evidence](../chat1gb/evidence/verification-torch-cuda.json) and [CPU evidence](../chat1gb/evidence/verification-numpy-cpu.json) are retained. Those scores are not autoregressive conversation scores and this prototype does not satisfy the current research constraints.']
    (root / 'results' / 'README.md').write_text('\n'.join(rows) + '\n', encoding='utf-8', newline='\n')
    (root / 'results' / 'experiments.json').write_text(json.dumps(index, indent=2) + '\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
