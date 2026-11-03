"""Losslessly pack frozen float32 tensors; no fitting or coefficient changes.

NumPy is used only by this offline packaging/verification tool. The compact
inference loader uses the standard library and PyTorch, and never expands a
full dense checkpoint on disk or retains it as a single runtime allocation.
"""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_member(z, name, data):
    info = zipfile.ZipInfo(name, date_time=(2026, 11, 4, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    z.writestr(info, data, compresslevel=9)


def pack(source, output):
    if output.exists():
        raise ValueError('Refusing to overwrite an existing compact package')
    config_bytes = (source / 'config.json').read_bytes()
    config = json.loads(config_bytes)
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    raw = source / config['weights_file']
    expected = manifest['sha256']['weights.f32']
    if sha(raw) != expected or sha(source / 'config.json') != manifest['sha256']['config.json']:
        raise ValueError('Frozen checkpoint hashes do not match')
    mapped = np.memmap(raw, dtype='<u4', mode='r')
    output.mkdir(parents=True)
    metadata = dict(format='direct-causal-transformer-sparse-f32-v1', version='v17a-compact',
                    dense_weight_sha256=expected, dense_weight_bytes=raw.stat().st_size,
                    config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                    logical_parameter_count=config['parameter_count'],
                    nonzero_coefficients=0, stored_float32_entries=0, negative_zero_entries=0,
                    tensors={})
    reconstructed = hashlib.sha256()
    with zipfile.ZipFile(output / 'weights.sparse.zip', 'w') as z:
        for n, (name, item) in enumerate(config['tensors'].items()):
            start, length = item['offset_bytes'] // 4, item['nbytes'] // 4
            bits = mapped[start:start + length]
            # Preserve negative zero too: numeric nonzero alone is not bit-lossless.
            indices = np.flatnonzero(bits).astype('<u4')
            values = bits[indices].astype('<u4')
            prefix = f'tensors/{n:03d}'
            write_member(z, prefix + '.indices.u32', indices.tobytes())
            write_member(z, prefix + '.values.f32', values.tobytes())
            metadata['tensors'][name] = dict(shape=item['shape'], entries=len(indices), prefix=prefix)
            metadata['stored_float32_entries'] += len(indices)
            metadata['negative_zero_entries'] += int(np.count_nonzero(values == 0x80000000))
            restored = np.zeros(length, dtype='<u4')
            restored[indices] = values
            if not np.array_equal(restored, bits):
                raise AssertionError(f'Bitwise reconstruction failed: {name}')
            reconstructed.update(restored.tobytes())
        metadata['nonzero_coefficients'] = metadata['stored_float32_entries'] - metadata['negative_zero_entries']
        metadata['uncompressed_coordinate_and_value_bytes'] = metadata['stored_float32_entries'] * 8
        metadata['reconstructed_dense_sha256'] = reconstructed.hexdigest()
        assert metadata['reconstructed_dense_sha256'] == expected
        assert metadata['nonzero_coefficients'] == manifest['nonzero_parameters']
        write_member(z, 'compact.json', (json.dumps(metadata, indent=2) + '\n').encode())
        write_member(z, 'config.json', config_bytes)
    shutil.copyfile(source / 'config.json', output / 'config.json')
    shutil.copyfile(source / 'manifest.json', output / 'dense-manifest.json')
    (output / 'compact.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    with zipfile.ZipFile(output / 'weights.sparse.zip') as z:
        assert z.testzip() is None
    report = {k: v for k, v in metadata.items() if k != 'tensors'}
    report.update(archive_bytes=(output / 'weights.sparse.zip').stat().st_size,
                  archive_sha256=sha(output / 'weights.sparse.zip'),
                  tensor_count=len(metadata['tensors']), source_weights_unchanged=sha(raw) == expected,
                  method='Bitwise coordinates of every non-positive-zero float32 value; ZIP deflate; no quantization, training, or changed coefficients.')
    (output / 'packing-verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('build/v17a'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(pack(a.source, a.output), indent=2))
