"""Audit positional attention using actual Q/K activations and synthetic index V.

This external diagnostic replaces only the value input of a separate attention
calculation with position indices. It never changes model weights or inference
answers. Every declared lookback is checked across the complete storage window.
"""
import argparse
import json
from pathlib import Path

from .evaluate import digest, git_state
from chat1gb_autoregressive.engine import Model
import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--version', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit('Refusing to overwrite evidence')
    m = Model(a.model, device=a.device)
    initial_hash = digest(a.model / 'weights.f32')
    count = m.config['context_length']
    pattern = m.tokenizer.encode('A zorl can fly and swim. A telv does not need water and fuel. Does telv need fuel? Please calculate -149 - 370. Hello.')
    ids = [m.config['special']['bos'], m.config['special']['user']]
    ids += (pattern * (count // len(pattern) + 1))[:count - 2]
    captured = {}
    original = m.linear

    def capture(x, prefix):
        value = original(x, prefix)
        if prefix in [f'blocks.{layer}.attn.qkv' for layer in range(3)]:
            captured[prefix] = value.detach()
        return value

    m.linear = capture
    with torch.inference_mode():
        m.forward(ids)
    m.linear = original
    addresses = [(0, shift + 1, shift) for shift in range(1, 5)]
    addresses += [(1, shift + 4, shift) for shift in range(1, 4)] + [(1, 10, 1), (2, 2, 1)]
    queries, keys = [], []
    for layer, head, shift in addresses:
        q, k, _ = captured[f'blocks.{layer}.attn.qkv'].chunk(3, dim=-1)
        queries.append(q.reshape(1, count, 16, 64)[:, :, head])
        keys.append(k.reshape(1, count, 16, 64)[:, :, head])
    q, k = torch.stack(queries, dim=1), torch.stack(keys, dim=1)
    positions = torch.arange(count, device=a.device, dtype=torch.float32)
    v = positions.reshape(1, 1, count, 1).expand(1, len(addresses), count, 64).contiguous()
    with torch.inference_mode():
        selected = F.scaled_dot_product_attention(q, k, v, is_causal=True)[0, :, :, 0]
    reports = []
    for index, (layer, head, shift) in enumerate(addresses):
        target = torch.clamp(positions - shift, min=0)
        errors = (selected[index] - target).abs()
        bad = torch.where(errors > .01)[0].cpu().tolist()
        reports.append(dict(layer=layer, head=head, lookback=shift, tested_positions=count,
                            incorrect_positions=len(bad), max_index_error=float(errors.max().item()),
                            bad_positions=bad,
                            first_errors=[dict(position=j, expected=max(0, j-shift),
                                               selected=float(selected[index, j].item())) for j in bad[:20]]))
    assert digest(a.model / 'weights.f32') == initial_hash
    result = dict(version=a.version, git=git_state(), model_sha256=initial_hash,
                  device=a.device, context_length=count, heads=reports,
                  tested_addresses=count * len(addresses),
                  incorrect_addresses=sum(x['incorrect_positions'] for x in reports),
                  tolerance_tokens=.01, weights_unchanged=True,
                  note='Actual Q/K from a full causal forward pass; a separate SDPA call with synthetic index V measures the selected address. This is an external address diagnostic, not a host reasoning fallback or a semantic benchmark.')
    a.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['version', 'device', 'tested_addresses', 'incorrect_addresses']}))
    print([(x['layer'], x['head'], x['incorrect_positions'], x['max_index_error']) for x in reports])


if __name__ == '__main__':
    main()
