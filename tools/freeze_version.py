"""Archive compiler bytes and checkpoint metadata without the large weights."""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('version')
    p.add_argument('--model', type=Path, required=True)
    a = p.parse_args()
    if not re.fullmatch(r'v[1-9][0-9]*[a-z]?', a.version):
        raise SystemExit('Invalid version')
    root = Path(__file__).resolve().parents[1]
    target = root / 'versions' / ('autoregressive_' + a.version)
    if target.exists():
        raise SystemExit('Refusing to overwrite a frozen version')
    manifest = json.loads((a.model / 'manifest.json').read_text(encoding='utf-8'))
    source = root / 'chat1gb_autoregressive'
    for name, digest in manifest['source_sha256'].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest, name
    (target / 'model').mkdir(parents=True)
    (target / '__init__.py').write_bytes(b'')
    for name in manifest['source_sha256']:
        shutil.copy2(source / name, target / name)
    for name in ('manifest.json', 'config.json'):
        shutil.copy2(a.model / name, target / 'model' / name)
    print(f'Frozen {a.version}: {manifest["sha256"]["weights.f32"]}')


if __name__ == '__main__':
    main()
