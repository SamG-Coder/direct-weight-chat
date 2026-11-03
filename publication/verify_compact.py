"""Compare compact inference with retained, unmodified dense-checkpoint results."""
import argparse
import hashlib
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from chat1gb_autoregressive.compact_engine import Model
from chat1gb_autoregressive.engine import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=Path('build/v17a-compact'))
    p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--baseline', type=Path, action='append')
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit('Refusing to overwrite verification evidence')
    suites = {sha(f): f for f in Path('tests').glob('*.json')}
    selected = []
    paths = a.baseline or sorted(Path('results').glob(f'*v17a*{a.device}.json'))
    for path in paths:
        baseline = json.loads(path.read_text())
        if baseline.get('suite_sha256') in suites and baseline.get('cases'):
            selected.append((path, baseline, suites[baseline['suite_sha256']]))
    if not selected:
        raise ValueError('No compatible baseline suites selected')
    model = Model(a.model, a.device)
    model.generate('Hello', 4)
    report = dict(kind='lossless-storage-runtime-parity', version='v17a-compact', device=model.device,
                  created_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                  torch=torch.__version__, hardware=torch.cuda.get_device_name() if model.device == 'cuda' else platform.processor(),
                  compact_sha256=sha(a.model / 'weights.sparse.zip'),
                  frozen_runtime_sha256=sha(Path('chat1gb_autoregressive/engine.py')),
                  compact_runtime_sha256=sha(Path('chat1gb_autoregressive/compact_engine.py')),
                  prompt_reference='Each record id selects its exact prompt from the SHA-256 pinned baseline and suite files.',
                  comparisons=[], differences=[], total=0, identical=0, scored_passes=0)
    for path, baseline, suite_path in selected:
        suite = json.loads(suite_path.read_text())
        assert len(baseline['cases']) == len(suite['cases'])
        rows = []
        for case, old in zip(suite['cases'], baseline['cases']):
            assert case['id'] == old['id'] and case['prompt'] == old['prompt']
            if case.get('reset', True):
                model.reset()
            answer = model.generate(case['prompt'], case.get('max_new_tokens', 40))
            same = all(answer[key] == old[key] for key in ('text', 'token_ids', 'stopped_on_eos'))
            accepted = case.get('acceptable', [case['expected']])
            correct = answer['text'] in accepted and answer['stopped_on_eos']
            row = dict(id=case['id'], text=answer['text'], token_ids=answer['token_ids'],
                       stopped_on_eos=answer['stopped_on_eos'], identical=same, correct=correct,
                       seconds=answer['seconds'], reset=case.get('reset', True))
            rows.append(row)
            if not same:
                report['differences'].append(dict(baseline=str(path), id=case['id'], old=old, compact=answer))
        entry = dict(baseline=path.as_posix(), baseline_sha256=sha(path), suite=suite_path.as_posix(),
                     suite_sha256=sha(suite_path), total=len(rows), identical=sum(r['identical'] for r in rows),
                     scored_passes=sum(r['correct'] for r in rows),
                     median_response_ms=1000 * statistics.median(r['seconds'] for r in rows), records=rows)
        report['comparisons'].append(entry)
        for key in ('total', 'identical', 'scored_passes'):
            report[key] += entry[key]
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({k: v for k, v in entry.items() if k != 'records'}), flush=True)
    report['storage_bytes_after_inference'] = model.storage_stats()
    report['runtime_imports_no_compiler'] = not any(n.endswith(('.build', '.knowledge', '.program')) for n in sys.modules)
    report['all_outputs_identical'] = report['total'] == report['identical']
    report['complete'] = True
    if model.device == 'cuda':
        report['cuda_peak_allocated_bytes_for_these_suites'] = torch.cuda.max_memory_allocated()
        report['cuda_peak_reserved_bytes_for_these_suites'] = torch.cuda.max_memory_reserved()
    assert report['runtime_imports_no_compiler']
    a.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('comparisons', 'differences')}, indent=2))
    if not report['all_outputs_identical']:
        raise SystemExit('Response parity failed; see retained differences')


if __name__ == '__main__':
    main()
