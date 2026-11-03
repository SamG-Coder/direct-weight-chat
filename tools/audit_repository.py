"""Offline metadata audit; no numerical dependencies and no weight loading."""
import hashlib
import json
import math
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    versions = sorted((root / 'versions').glob('autoregressive_*'))
    for version in versions:
        manifest = json.loads((version / 'model' / 'manifest.json').read_text(encoding='utf-8'))
        config_path = version / 'model' / 'config.json'
        config = json.loads(config_path.read_text(encoding='utf-8'))
        assert hashlib.sha256(config_path.read_bytes()).hexdigest() == manifest['sha256']['config.json']
        for name, expected in manifest['source_sha256'].items():
            assert hashlib.sha256((version / name).read_bytes()).hexdigest() == expected, (version.name, name)
        offset = count = 0
        for name, tensor in config['tensors'].items():
            assert tensor['offset_bytes'] == offset, (version.name, name)
            n = math.prod(tensor['shape'])
            assert tensor['nbytes'] == n * 4
            count += n
            offset += tensor['nbytes']
        assert count == 250_000_000 and offset == 1_000_000_000
    reports = 0
    for path in (root / 'results').glob('*.json'):
        r = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(r, dict) or 'cases' not in r:
            continue
        expected_total = len(r['cases'])
        passed = sum(c['text'] in c.get('acceptable', [c['expected']]) and c['stopped_on_eos'] for c in r['cases'])
        assert r['summary']['all']['total'] == expected_total
        assert r['summary']['all']['passed'] == passed
        assert all(c['correct'] == (c['text'] in c.get('acceptable', [c['expected']]) and c['stopped_on_eos']) for c in r['cases'])
        version = root / 'versions' / ('autoregressive_' + r['version'])
        manifest = json.loads((version / 'model' / 'manifest.json').read_text(encoding='utf-8'))
        assert r['model_sha256'] == manifest['sha256']['weights.f32']
        reports += 1
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    for name in filter(None, tracked):
        assert not name.endswith(('.f32', '.zip', '.pyc')) and '/__pycache__/' not in name, name
        assert (root / name).stat().st_size < 10_000_000, name
    print(f'PASS: {len(versions)} frozen compiler/config layouts; {reports} exact-output reports; no large artifacts tracked.')


if __name__ == '__main__':
    main()
