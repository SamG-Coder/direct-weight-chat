"""Lossless sparse storage with the frozen dense transformer's exact operations.

Only one requested tensor is materialized at a time. The original Model's
forward pass, attention, tokenizer, decoding, and generation are inherited
unchanged. This loader performs no prompt parsing, arithmetic, or rule solving.
It does not import NumPy, the compiler, or any pretrained model.
"""
from .engine import Model as DenseModel, Tokenizer, torch

import hashlib
import json
import math
import sys
import zipfile
from collections.abc import Mapping
from pathlib import Path


class SparseWeights(Mapping):
    def __init__(self, archive, config, device):
        self.items_by_name = {}
        self.cached_vectors = {}
        self.device = device
        if sys.byteorder != 'little':
            raise ValueError('This float32 artifact requires a little-endian host')
        with zipfile.ZipFile(archive) as z:
            self.metadata = json.loads(z.read('compact.json'))
            if self.metadata['format'] != 'direct-causal-transformer-sparse-f32-v1':
                raise ValueError('Unexpected compact format')
            for name, spec in self.metadata['tensors'].items():
                if spec['shape'] != config['tensors'][name]['shape']:
                    raise ValueError(f'Tensor shape mismatch: {name}')
                count = spec['entries']
                ib = z.read(spec['prefix'] + '.indices.u32')
                vb = z.read(spec['prefix'] + '.values.f32')
                if len(ib) != 4 * count or len(vb) != 4 * count:
                    raise ValueError(f'Truncated sparse tensor: {name}')
                if count:
                    indices = torch.frombuffer(bytearray(ib), dtype=torch.int32).to(device=device, dtype=torch.long)
                    values = torch.frombuffer(bytearray(vb), dtype=torch.float32).clone().to(device)
                    if int(indices[0]) < 0 or int(indices[-1]) >= math.prod(spec['shape']):
                        raise ValueError(f'Sparse index outside tensor: {name}')
                    if count > 1 and not bool((indices[1:] > indices[:-1]).all()):
                        raise ValueError(f'Nonunique or unordered sparse indices: {name}')
                else:
                    indices = torch.empty(0, device=device, dtype=torch.long)
                    values = torch.empty(0, device=device, dtype=torch.float32)
                self.items_by_name[name] = (tuple(spec['shape']), indices, values)
            if set(self.items_by_name) != set(config['tensors']):
                raise ValueError('Tensor inventory mismatch')

    def __iter__(self):
        return iter(self.items_by_name)

    def __len__(self):
        return len(self.items_by_name)

    def __getitem__(self, name):
        if name in self.cached_vectors:
            return self.cached_vectors[name]
        shape, indices, values = self.items_by_name[name]
        dense = torch.zeros(shape, dtype=torch.float32, device=self.device)
        if indices.numel():
            dense.view(-1).index_copy_(0, indices, values)
        # Cache only small vectors, never the unused padding or dense matrices.
        if len(shape) == 1 and math.prod(shape) <= 16384:
            self.cached_vectors[name] = dense
        return dense

    def resident_tensor_bytes(self):
        sparse = sum(t.numel() * t.element_size() for _, i, v in self.items_by_name.values() for t in (i, v))
        vectors = sum(t.numel() * t.element_size() for t in self.cached_vectors.values())
        return dict(sparse_buffers=sparse, cached_vectors=vectors, total=sparse + vectors)


class Model(DenseModel):
    def __init__(self, directory, device='auto'):
        self.directory = Path(directory)
        config_bytes = (self.directory / 'config.json').read_bytes()
        self.config = json.loads(config_bytes)
        if self.config['format'] != 'direct-causal-transformer-f32-v1':
            raise ValueError('Unexpected logical model format')
        torch.set_num_threads(8)
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = 'cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu' if device == 'auto' else device
        self.w = SparseWeights(self.directory / 'weights.sparse.zip', self.config, self.device)
        self.compact_metadata = self.w.metadata
        if hashlib.sha256(config_bytes).hexdigest() != self.compact_metadata['config_sha256']:
            raise ValueError('Config hash does not match compact artifact')
        self.reset()

    def storage_stats(self):
        return self.w.resident_tensor_bytes()
