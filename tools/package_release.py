"""Package local checkpoints and verify every decompressed byte, entirely offline."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def digest_stream(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(8 << 20), b''):
        h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    a.output.mkdir(parents=True, exist_ok=True)
    checkpoints = [('selector-v0', a.original_root / 'chat1gb' / 'model', 'chat1gb/model')]
    for v in range(1, 6):
        name = f'model-v{v}' if v < 5 else 'model'
        checkpoints.append((f'autoregressive-v{v}', a.original_root / 'chat1gb_autoregressive' / name,
                            f'versions/autoregressive_v{v}/model'))
    for v in ('v6a', 'v6b', 'v6c', 'v7a'):
        target = 'chat1gb_autoregressive/model' if v == 'v7a' else f'versions/autoregressive_{v}/model'
        checkpoints.append((f'autoregressive-{v}', root / 'build' / v, target))
    records = []
    for name, source, prefix in checkpoints:
        archive = a.output / (name + '.zip')
        if archive.exists():
            raise SystemExit(f'Refusing to overwrite {archive}')
        manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
        assert (source / 'weights.f32').stat().st_size == 1_000_000_000
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for file in [*manifest['sha256'], 'manifest.json']:
                z.write(source / file, prefix + '/' + file)
        # Streaming through ZipExtFile also validates CRC at EOF. Do not trust
        # the original manifest without comparing the decompressed bytes.
        with zipfile.ZipFile(archive) as z:
            for file, expected in manifest['sha256'].items():
                with z.open(prefix + '/' + file) as stream:
                    assert digest_stream(stream) == expected, (name, file)
        with open(archive, 'rb') as stream:
            checksum = digest_stream(stream)
        records.append(dict(name=archive.name, bytes=archive.stat().st_size, sha256=checksum,
                            model_sha256=manifest['sha256']['weights.f32'],
                            parameter_count=manifest['parameter_count'], weight_bytes=manifest['weight_bytes'],
                            extracted_path=prefix, decompressed_hashes_and_crc_verified=True))
        print(f'{archive.name}: {archive.stat().st_size:,} bytes; decompressed hashes verified', flush=True)
    (a.output / 'SHA256SUMS.txt').write_text(''.join(f"{r['sha256']}  {r['name']}\n" for r in records), encoding='ascii', newline='\n')
    a.report.write_text(json.dumps(dict(assets=records), indent=2) + '\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
