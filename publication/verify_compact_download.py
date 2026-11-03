"""Verify the public compact package and execute its downloaded runtime offline."""
import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download', type=Path, required=True)
    p.add_argument('--python', required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    expected = json.loads((root / 'publication/compact-package.json').read_text(encoding='utf-8'))
    repo = 'SamGCoder/direct-weight-chat-v17a'
    info = HfApi().model_info(repo, files_metadata=True)
    assert not info.private
    assert 'weights.f32' not in [f.rfilename for f in info.siblings], 'Dense weights remain in default download'
    if a.download.exists():
        raise SystemExit('Refusing to overwrite the fresh download')
    snapshot_download(repo, revision=info.sha, local_dir=a.download,
                      allow_patterns=[e['name'] for e in expected['files']] + ['SHA256SUMS.txt'], max_workers=3)
    files = []
    for entry in expected['files']:
        path = a.download / entry['name']
        assert sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes'], entry['name']
        files.append(dict(**entry, download_verified=True))
    sums = (a.download / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines()
    assert dict((line.split('  ', 1)[1], line.split('  ', 1)[0]) for line in sums) == {e['name']: e['sha256'] for e in files}
    with zipfile.ZipFile(a.download / 'runtime.zip') as z:
        assert z.testzip() is None
        for name in z.namelist():
            assert (a.download / name).resolve().is_relative_to(a.download.resolve())
        z.extractall(a.download)
    for name, expected_hash in expected['runtime_source_sha256'].items():
        assert sha(a.download / name) == expected_hash, name
    program = '''import json,sys
from chat1gb_autoregressive.compact_engine import Model
m=Model(directory='.',device='cpu')
rows=[m.generate(p) for p in ['A kelv can fly. What can a kelv do?','A kelv cannot fly. Can kelv fly?','149 - 370']]
assert not any(n.endswith(('.build','.knowledge','.program')) for n in sys.modules)
print(json.dumps(dict(turns=rows,weight_tensor_bytes=m.storage_stats())))
'''
    run = subprocess.run([a.python, '-c', program], cwd=a.download, capture_output=True, text=True, encoding='utf-8', check=True)
    smoke = json.loads(run.stdout)
    assert [r['text'] for r in smoke['turns']] == ['Kelv can fly.', 'No.', '-221.']
    assert all(r['stopped_on_eos'] for r in smoke['turns'])
    assert not (a.download / 'weights.f32').exists()
    from publication.verify_compact_storage import verify
    round_trip = verify(a.download)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), repository=f'https://huggingface.co/{repo}',
                  revision=info.sha, public=True, dense_weights_removed_from_main=True, files=files,
                  checksum_file_matches=True, all_runtime_source_hashes_match=True,
                  extracted_runtime_crc_verified=True, standalone_cpu_smoke_passed=True,
                  smoke=smoke, serialized_round_trip=round_trip,
                  no_dense_checkpoint_written=True, compact_weights_unchanged=sha(a.download / 'weights.sparse.zip') == next(e['sha256'] for e in files if e['name'] == 'weights.sparse.zip'))
    assert result['compact_weights_unchanged']
    (root / 'publication/compact-huggingface-verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(repository=result['repository'], revision=info.sha, verified_files=len(files), cpu_smoke='3/3', round_trip_exact=True), indent=2))


if __name__ == '__main__':
    main()
