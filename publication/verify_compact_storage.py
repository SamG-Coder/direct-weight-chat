"""Independently hash dense tensors reconstructed from the serialized archive."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np


def verify(directory):
    digest = hashlib.sha256()
    config_bytes = (directory / 'config.json').read_bytes()
    config = json.loads(config_bytes)
    with zipfile.ZipFile(directory / 'weights.sparse.zip') as z:
        assert z.testzip() is None
        meta = json.loads(z.read('compact.json'))
        assert z.read('config.json') == config_bytes
        assert hashlib.sha256(config_bytes).hexdigest() == meta['config_sha256']
        for name, layout in config['tensors'].items():
            item = meta['tensors'][name]
            indices = np.frombuffer(z.read(item['prefix'] + '.indices.u32'), dtype='<u4')
            bits = np.frombuffer(z.read(item['prefix'] + '.values.f32'), dtype='<u4')
            assert len(indices) == len(bits) == item['entries']
            dense = np.zeros(layout['nbytes'] // 4, dtype='<u4')
            dense[indices] = bits
            digest.update(dense.tobytes())
    assert digest.hexdigest() == meta['dense_weight_sha256']
    return dict(serialized_archive_round_trip_exact=True, reconstructed_sha256=digest.hexdigest(),
                reconstructed_bytes=sum(i['nbytes'] for i in config['tensors'].values()),
                archive_sha256=hashlib.sha256((directory / 'weights.sparse.zip').read_bytes()).hexdigest(),
                wrote_dense_checkpoint_to_disk=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=Path('build/v17a-compact'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit('Refusing to overwrite verification evidence')
    report = verify(a.model)
    a.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
