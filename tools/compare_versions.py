"""Compare external reports only when frozen suite hashes and devices agree."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--versions', nargs='+', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if len(a.versions) < 2 or a.output.exists():
        raise SystemExit('Need at least two versions and a new output path')
    reports = {}
    for path in Path('results').glob('*.json'):
        report = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(report, dict) or 'cases' not in report or report.get('device') != a.device:
            continue
        key = (report['version'], report['suite_sha256'])
        if key in reports:
            raise ValueError(f'Duplicate report key: {key}')
        reports[key] = (path, report)
    comparisons = []
    for before, after in zip(a.versions, a.versions[1:]):
        for (version, suite_hash), (before_path, old) in reports.items():
            if version != before or (after, suite_hash) not in reports:
                continue
            after_path, new = reports[(after, suite_hash)]
            assert len(old['cases']) == len(new['cases'])
            changes = []
            for left, right in zip(old['cases'], new['cases']):
                assert all(left[k] == right[k] for k in ('id', 'prompt', 'expected', 'reset'))
                if left['text'] != right['text'] or left['correct'] != right['correct']:
                    kind = ('regression' if left['correct'] and not right['correct'] else
                            'improvement' if right['correct'] and not left['correct'] else
                            'output_changed')
                    changes.append(dict(id=left['id'], kind=kind, prompt=left['prompt'], expected=left['expected'],
                                        before=left['text'], after=right['text'],
                                        before_correct=left['correct'], after_correct=right['correct']))
            comparisons.append(dict(before=before_path.as_posix(), after=after_path.as_posix(),
                                    suite_sha256=suite_hash, before_summary=old['summary']['all'],
                                    after_summary=new['summary']['all'],
                                    regressions=sum(c['kind'] == 'regression' for c in changes),
                                    improvements=sum(c['kind'] == 'improvement' for c in changes), changes=changes))
    result = dict(versions=a.versions, device=a.device, comparisons=comparisons,
                  regressions=sum(c['regressions'] for c in comparisons),
                  note='Only identical frozen suite hashes and devices are compared; repeated cases across suites remain visible.')
    a.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f"Compared {len(comparisons)} matched suites; {result['regressions']} regressions")


if __name__ == '__main__':
    main()
