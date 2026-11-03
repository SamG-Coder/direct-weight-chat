"""Verify downloaded release bytes against public metadata and a clean clone.

No network is used here. Downloading and selecting the public release happen
separately; this tool checks and extracts only the explicitly selected version.
"""
import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


def digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(8 << 20), b''):
        h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package-report', type=Path, required=True)
    p.add_argument('--remote-metadata', type=Path, required=True)
    p.add_argument('--downloads', type=Path, required=True)
    p.add_argument('--clone', type=Path, required=True)
    p.add_argument('--expected-commit', required=True)
    p.add_argument('--extract-version', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit('Refusing to overwrite evidence')
    clone = a.clone.resolve()

    def git(*args):
        return subprocess.check_output(['git', '-C', str(clone), *args], text=True).strip()

    assert git('rev-parse', 'HEAD') == a.expected_commit
    assert not git('status', '--porcelain')
    package = json.loads(a.package_report.read_text(encoding='utf-8'))
    remote = json.loads(a.remote_metadata.read_text(encoding='utf-8'))
    assert not remote['draft'] and all(x['state'] == 'uploaded' for x in remote['assets'])
    expected = {x['name']: x for x in package['assets']}
    published = {x['name']: x for x in remote['assets']}
    assert set(published) == set(expected) | {'SHA256SUMS.txt'}
    for name, item in published.items():
        assert Path(name).name == name
        path = a.downloads / name
        with path.open('rb') as f:
            actual = digest(f)
        assert actual == item['digest'].removeprefix('sha256:')
        assert path.stat().st_size == item['size']
        if name in expected:
            assert actual == expected[name]['sha256'] and path.stat().st_size == expected[name]['bytes']
    sums = {}
    for line in (a.downloads / 'SHA256SUMS.txt').read_text(encoding='ascii').splitlines():
        checksum, name = line.split('  ', 1)
        sums[name] = checksum
    assert sums == {name: item['sha256'] for name, item in expected.items()}
    verified = []
    extracted = False
    for name, item in expected.items():
        version = name.removeprefix('autoregressive-').removesuffix('.zip')
        source = clone / 'versions' / ('autoregressive_' + version)
        prefix = item['extracted_path']
        with zipfile.ZipFile(a.downloads / name) as z:
            manifest = json.loads(z.read(prefix + '/manifest.json'))
            assert manifest == json.loads((source / 'model/manifest.json').read_text(encoding='utf-8'))
            assert manifest['parameter_count'] == 250_000_000 and manifest['weight_bytes'] == 1_000_000_000
            assert set(z.namelist()) == {prefix + '/' + f for f in [*manifest['sha256'], 'manifest.json']}
            for file, checksum in manifest['source_sha256'].items():
                with (source / file).open('rb') as f:
                    assert digest(f) == checksum
            for file, checksum in manifest['sha256'].items():
                with z.open(prefix + '/' + file) as f:
                    assert digest(f) == checksum  # Reads to EOF, validating ZIP CRC too.
            assert z.getinfo(prefix + '/weights.f32').file_size == 1_000_000_000
            if version == a.extract_version:
                for member in z.infolist():
                    assert (clone / member.filename).resolve().is_relative_to(clone)
                z.extractall(clone)
                extracted = True
        verified.append(dict(version=version, archive=name, sha256=item['sha256'],
                             model_sha256=manifest['sha256']['weights.f32'],
                             downloaded_hash_size_checksum_and_source_match=True,
                             decompressed_hashes_and_crc_verified=True))
    assert extracted and not git('status', '--porcelain')
    result = dict(release=remote['html_url'], release_commit=a.expected_commit,
                  downloaded_assets_match_public_digests=True, checksum_file_verified=True,
                  archives=verified, extracted_version=a.extract_version,
                  extraction_paths_checked=True, fresh_source_tree_clean=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'PASS: {len(verified)} downloaded checkpoints; source/raw hashes, CRC, size and clean extraction')


if __name__ == '__main__':
    main()
