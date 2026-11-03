"""Run every saved exact-output suite, preserving existing report bytes.

This is an external orchestrator. Each evaluator runs in a separate process;
neither suite answers nor this report-selection code are imported by inference.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--versions', nargs='+', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    p.add_argument('--baseline', default='v15a')
    a = p.parse_args()
    saved = {}
    for path in Path('results').glob('*.json'):
        r = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(r, dict) and 'cases' in r and 'suite_sha256' in r and r.get('device') == a.device:
            key = (r['version'], r['suite_sha256'])
            if key in saved:
                raise ValueError(f'Duplicate saved report: {key}')
            saved[key] = path
    suites = sorted(Path('tests').glob('*.json'))
    for version in a.versions:
        for suite in suites:
            key = (version, digest(suite))
            if key in saved:
                print(f'{version}: preserved {saved[key]}', flush=True)
                continue
            stem = suite.stem.replace('-development', '').replace('legacy-185', 'legacy')
            found = re.search(r'-v[0-9]+[a-z]?', stem)
            if found:
                stem = stem[:found.start()] + '-' + version + stem[found.start():]
                stem = stem.replace('-postfreeze', '') + '-regression'
            else:
                stem += '-' + version
            output = Path('results') / (stem + '-' + a.device + '.json')
            command = [sys.executable, '-m', 'benchmarks.evaluate', '--model', 'build/' + version,
                       '--suite', str(suite), '--output', str(output), '--version', version,
                       '--device', a.device, '--evaluation-status', 'regression-on-previously-published-suite' if found else 'development']
            previous = saved.get((a.baseline, key[1]))
            if previous:
                command += ['--compare', str(previous)]
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            result = json.loads(output.read_text(encoding='utf-8'))
            score = result['summary']['all']
            print(f"{output.name}: {score['passed']}/{score['total']}, regressions={len(result.get('regressions', []))}", flush=True)
            saved[key] = output


if __name__ == '__main__':
    main()
