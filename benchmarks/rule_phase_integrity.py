"""External perturbation checks for the compiled Boolean phase gates.

This diagnostic injects synthetic residuals into the actual block-1 MLP. It
does not participate in normal inference or calculate a conversational answer.
"""
import argparse
import itertools
import json
from pathlib import Path

from .evaluate import digest, git_state
from chat1gb_autoregressive.engine import Model
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--version', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--require-saturation', action='store_true')
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit('Refusing to overwrite evidence')
    m = Model(a.model, device=a.device)
    before = digest(a.model / 'weights.f32')
    groups = (('current_context', (494, 52), 56), ('after_then', (495,), 59),
              ('previous_context', (496, 497), 60), ('compound_rule', (498,), 260))
    deltas = (-.125, -.001, 0, .001, .125)
    rows = []
    for name, sources, target in groups:
        for bits in itertools.product((0, 1), repeat=len(sources)):
            for errors in itertools.product(deltas, repeat=len(sources)):
                rows.append(dict(gate=name, sources=sources, target=target, bits=bits, errors=errors,
                                 expected=int(all(bits))))
    x = torch.zeros((1, len(rows), 1024), dtype=torch.float32, device=m.device)

    def feature(row, coordinate, value):
        x[0, row, coordinate*2] = value
        x[0, row, coordinate*2+1] = -value

    for i, row in enumerate(rows):
        feature(i, 0, 1_000_000)
        feature(i, 1, 1)
        feature(i, 2, 26.3)
        feature(i, row['target'], .375)  # Also check erasure of the previous role.
        for source, bit, error in zip(row['sources'], row['bits'], row['errors']):
            feature(i, source, bit + error)
    value = m.norm(x, 'blocks.1.ln2')
    value = torch.relu(m.linear(value, 'blocks.1.mlp.up'))
    value = x + m.linear(value, 'blocks.1.mlp.down')
    for i, row in enumerate(rows):
        coordinate = row['target'] * 2
        actual = float(((value[0, i, coordinate] - value[0, i, coordinate+1])/2).item())
        row['actual'] = actual
        row['absolute_error'] = abs(actual-row['expected'])
        row['passed'] = row['absolute_error'] < 1e-5
    result = dict(version=a.version, device=m.device, git=git_state(), model_sha256=before,
                  probes=rows, diagnostic='synthetic-residual-boolean-saturation',
                  passed=sum(r['passed'] for r in rows), total=len(rows),
                  maximum_absolute_error=max(r['absolute_error'] for r in rows),
                  weights_unchanged=digest(a.model/'weights.f32') == before,
                  note='Actual norm/MLP tensor operations on externally generated residuals; expected Booleans score the result only. No generated dialogue, compiler import, model update or normal-runtime alteration.')
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('version','device','passed','total','maximum_absolute_error')}))
    assert result['weights_unchanged']
    if a.require_saturation:
        assert result['passed'] == result['total']


if __name__ == '__main__':
    main()
