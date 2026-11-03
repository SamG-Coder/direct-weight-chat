"""Verify a fresh public release download, then run its own offline runtime."""
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD = ROOT / 'build/compact-github-download'
DEST = ROOT / 'build/compact-github-extracted'
expected = json.loads((ROOT / 'publication/compact-package.json').read_text(encoding='utf-8'))
archive = DOWNLOAD / expected['github_archive']['name']
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert archive.stat().st_size == expected['github_archive']['bytes']
assert sha(archive) == expected['github_archive']['sha256']
assert (DOWNLOAD / 'SHA256SUMS.txt').read_text().strip() == sha(archive) + '  ' + archive.name
if DEST.exists():
    raise SystemExit('Refusing to overwrite the fresh release extraction')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert all((DEST / name).resolve().is_relative_to(DEST.resolve()) for name in z.namelist())
    z.extractall(DEST)
for name, digest in expected['runtime_source_sha256'].items():
    assert sha(DEST / name) == digest, name
for name in ['weights.sparse.zip', 'config.json', 'compact.json', 'dense-manifest.json']:
    entry = next(e for e in expected['files'] if e['name'] == name)
    assert sha(DEST / 'compact' / name) == entry['sha256'], name
code = '''import json,sys
from chat1gb_autoregressive.compact_engine import Model
m=Model('compact',device='cpu')
rows=[m.generate(p) for p in ['A kelv can fly. What can a kelv do?','A kelv cannot fly. Can kelv fly?','149 - 370']]
assert not any(n.endswith(('.build','.knowledge','.program')) for n in sys.modules)
print(json.dumps(rows))
'''
run = subprocess.run([r'D:\LittleBit\.venv\Scripts\python.exe', '-c', code], cwd=DEST, capture_output=True, text=True, encoding='utf-8', check=True)
rows = json.loads(run.stdout)
assert [r['text'] for r in rows] == ['Kelv can fly.', 'No.', '-221.']
assert all(r['stopped_on_eos'] for r in rows)
assert not list(DEST.rglob('weights.f32'))
report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
              release='https://github.com/SamG-Coder/direct-weight-chat/releases/tag/v0.10.0-compact-final',
              archive=expected['github_archive'], checksum_file_matches=True,
              zip_crc_valid=True, all_runtime_source_hashes_match=True,
              compact_weight_hash_matches=True, standalone_cpu_smoke_passed=True,
              no_dense_checkpoint_written=True, turns=rows)
(ROOT / 'publication/compact-github-verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(verified=True, archive_bytes=archive.stat().st_size, cpu_smoke='3/3')))
