"""Prepare static Space assets, preserving the exact published model bytes."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'build/huggingface-space'
DEST.mkdir(parents=True, exist_ok=True)
for name in ['index.html', 'app.mjs', 'worker.mjs', 'runtime.mjs', 'README.md']:
    shutil.copyfile(ROOT / 'hosting' / name, DEST / name)
for name in ['config.json', 'weights.sparse.zip']:
    shutil.copyfile(ROOT / 'build/v17a-compact' / name, DEST / name)
shutil.copyfile(ROOT / 'LICENSE', DEST / 'LICENSE')
assert hashlib.sha256((DEST / 'weights.sparse.zip').read_bytes()).hexdigest() == '4739a3511f5c4b9170f5d09524bd8105f5853a7999cbeefdea7a8cbae01d737d'
files = [dict(name=f.name, bytes=f.stat().st_size, sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(DEST.iterdir()) if f.is_file()]
(ROOT / 'publication/hosted-chat-package.json').write_text(json.dumps(dict(files=files, inference='browser CPU', model='frozen v17a compact'), indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(directory=str(DEST), files=len(files), total_bytes=sum(f['bytes'] for f in files))))
