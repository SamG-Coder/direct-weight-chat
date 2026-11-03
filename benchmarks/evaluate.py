"""Evaluate a frozen checkpoint; expected answers never enter model.generate."""
import argparse
import collections
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from chat1gb_autoregressive.engine import Model


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def git_state():
    def git(*args):
        r = subprocess.run(['git', *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    return {'commit': git('rev-parse', 'HEAD'), 'dirty': bool(git('status', '--porcelain'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite an experiment result')
    suite = json.loads(args.suite.read_text(encoding='utf-8'))
    manifest = json.loads((args.model / 'manifest.json').read_text(encoding='utf-8'))
    weight_hash = digest(args.model / 'weights.f32')
    assert weight_hash == manifest['sha256']['weights.f32']
    assert (args.model / 'weights.f32').stat().st_size == 1_000_000_000
    model = Model(directory=args.model, device=args.device)
    assert sum(w.numel() for w in model.w.values()) == 250_000_000
    assert not any(n.endswith(('.knowledge', '.build', '.program')) for n in sys.modules)
    model.generate('Hello', 4)  # Excluded warm-up, then reset before evaluation.
    result = dict(version=args.version, suite=suite['name'], evaluation_status=suite['status'],
                  created_utc=datetime.now(timezone.utc).isoformat(), git=git_state(),
                  model_sha256=weight_hash, suite_sha256=digest(args.suite),
                  runtime_sha256=digest(Path(__file__).resolve().parents[1] / 'chat1gb_autoregressive' / 'engine.py'),
                  parameter_count=250_000_000, weight_bytes=1_000_000_000,
                  nonzero_parameters=manifest['nonzero_parameters'], device=model.device,
                  python=platform.python_version(), cases=[], probes=[])
    import torch
    result['torch'] = torch.__version__
    result['hardware'] = torch.cuda.get_device_name() if model.device == 'cuda' else platform.processor()
    for case in suite['cases']:
        if case.get('reset', True):
            model.reset()
        answer = model.generate(case['prompt'], case.get('max_new_tokens', 40))
        # Scoring is external. The forward pass never receives expected text.
        accepted = case.get('acceptable', [case['expected']])
        answer.update(id=case['id'], category=case['category'], expected=case['expected'],
                      acceptable=accepted, reset=case.get('reset', True),
                      correct=answer['text'] in accepted and answer['stopped_on_eos'])
        result['cases'].append(answer)
    for case in suite.get('probes', []):
        model.reset()
        result['probes'].append(model.generate(case['prompt'], 40))
    result['summary'] = {}
    groups = collections.defaultdict(list)
    for item in result['cases']:
        groups[item['category']].append(item)
    groups['all'] = result['cases']
    for name, rows in groups.items():
        passed = sum(row['correct'] for row in rows)
        result['summary'][name] = dict(passed=passed, total=len(rows), success_rate=passed / len(rows),
                                      median_response_ms=statistics.median(r['seconds'] for r in rows) * 1000)
    if args.compare:
        baseline = {r['id']: r for r in json.loads(args.compare.read_text())['cases']}
        result['regressions'] = [r['id'] for r in result['cases'] if r['id'] in baseline and baseline[r['id']]['correct'] and not r['correct']]
        result['improvements'] = [r['id'] for r in result['cases'] if r['id'] in baseline and not baseline[r['id']]['correct'] and r['correct']]
    result['checks'] = {'weights_unchanged': digest(args.model / 'weights.f32') == weight_hash,
                        'runtime_imports_no_compiler_or_fact_table': True}
    assert result['checks']['weights_unchanged']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(version=args.version, summary=result['summary']), indent=2), flush=True)


if __name__ == '__main__':
    main()
