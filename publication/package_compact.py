"""Package the verified compact edition for Hugging Face and GitHub releases."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_files():
    files = {}
    for directory in ['chat1gb_autoregressive', 'versions/autoregressive_v17a']:
        for path in sorted((ROOT / directory).glob('*.py')):
            files[path.relative_to(ROOT).as_posix()] = path
    for name in ['config.json', 'manifest.json']:
        path = ROOT / 'versions/autoregressive_v17a/model' / name
        files[path.relative_to(ROOT).as_posix()] = path
    for name in ['LICENSE', 'requirements.txt', 'ARCHITECTURE.md', 'CAPABILITIES.md',
                 'publication/serve_demo.py', 'publication/model_worker.py',
                 'publication/capture_demo.py', 'publication/pack_compact.py',
                 'publication/verify_compact_storage.py', 'publication/CONSTRUCTION-AUDIT.md',
                 'publication/COMPACT.md', 'publication/demo-transcript-compact.json',
                 'publication/demo_ui/index.html']:
        files[name] = ROOT / name
    files['SOURCE-README.md'] = ROOT / 'README.md'
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--github-output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.github_output.exists():
        raise SystemExit('Refusing to overwrite publication packages')
    for device in ['cpu', 'cuda']:
        report = json.loads((ROOT / f'publication/compact-parity-{device}.json').read_text())
        assert report['complete'] and report['all_outputs_identical']
        assert report['compact_runtime_sha256'] == sha(ROOT / 'chat1gb_autoregressive/compact_engine.py')
    a.output.mkdir(parents=True)
    a.github_output.mkdir(parents=True)
    source = ROOT / 'build/v17a-compact'
    for name in ['weights.sparse.zip', 'config.json', 'compact.json', 'dense-manifest.json']:
        shutil.copyfile(source / name, a.output / name)
    shutil.copyfile(ROOT / 'publication/huggingface/README.md', a.output / 'README.md')
    shutil.copyfile(ROOT / 'LICENSE', a.output / 'LICENSE')
    files = runtime_files()
    with zipfile.ZipFile(a.output / 'runtime.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, path in files.items():
            z.write(path, name)
    with zipfile.ZipFile(a.output / 'runtime.zip') as z:
        assert z.testzip() is None
        assert hashlib.sha256(z.read('chat1gb_autoregressive/engine.py')).hexdigest() == 'f56d29cd23ba2b3018c4ba6a7ad6928dcb28c437e213929802aa4cba90329e10'
    entries = [dict(name=f.name, bytes=f.stat().st_size, sha256=sha(f)) for f in sorted(a.output.iterdir())]
    (a.output / 'SHA256SUMS.txt').write_text(''.join(f"{r['sha256']}  {r['name']}\n" for r in entries), encoding='utf-8')
    github_archive = a.github_output / 'direct-weight-chat-v17a-compact.zip'
    with zipfile.ZipFile(github_archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, path in files.items():
            z.write(path, name)
        for name in ['weights.sparse.zip', 'config.json', 'compact.json', 'dense-manifest.json']:
            z.write(a.output / name, 'compact/' + name)
    with zipfile.ZipFile(github_archive) as z:
        assert z.testzip() is None
    (a.github_output / 'SHA256SUMS.txt').write_text(f'{sha(github_archive)}  {github_archive.name}\n', encoding='utf-8')
    result = dict(version='v17a-compact', files=entries,
                  github_archive=dict(name=github_archive.name, bytes=github_archive.stat().st_size, sha256=sha(github_archive)),
                  original_dense_sha256='5cc52f3793a1546c620195c110a7d1c3a2f8d3417cb72f193003589b16e8e0d2',
                  runtime_source_sha256={name: sha(path) for name, path in files.items()})
    (ROOT / 'publication/compact-package.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(files=entries, github_archive=result['github_archive']), indent=2))


if __name__ == '__main__':
    main()
