"""Inspect existing activations around assertion binding without changing weights.

The model alone produces the answer. This external diagnostic records residual
fields and attention inputs/outputs, and compares float32/float64 score arithmetic
after inference. Diagnostic timings include capture overhead and are not speed
benchmarks. Suite labels are never passed to the model.
"""
import argparse
import json
from pathlib import Path

from .evaluate import digest, git_state
from chat1gb_autoregressive.engine import Model
import torch


FIELDS = {
    'one': [1], 'position': [2], 'query_group': [3],
    'query_object': list(range(30, 44)), 'query_mode': [44],
    'query_truth': list(range(45, 50)),
    'previous_continuation': list(range(230, 253)), 'terminator': [253],
    'frame_group_scratch': [257], 'previous_clause_group': [258],
    'subject': list(range(400, 414)), 'clause_or_adjective': [414],
    'current_subject': list(range(415, 429)), 'request': list(range(430, 437)),
    'fact_subject': list(range(440, 454)), 'fact_object': list(range(454, 468)),
    'fact_relation': list(range(468, 475)), 'fact_valid': [475], 'fact_negative': [476],
    'predicate': list(range(484, 491)), 'predicate_negative': [491],
    'predicate_valid': [492], 'frame_valid': [493], 'continuation': [504],
    'question_clause': [506], 'predicate_group_scratch': [508],
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--version', required=True)
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--case-id', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    p.add_argument('--tail', type=int, default=28)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit('Refusing to overwrite evidence')
    suite = json.loads(a.suite.read_text(encoding='utf-8'))
    target = next(i for i, c in enumerate(suite['cases']) if c['id'] == a.case_id)
    first = target
    while first > 0 and not suite['cases'][first].get('reset', True):
        first -= 1
    model = Model(a.model, device=a.device)
    initial_hash = digest(a.model / 'weights.f32')
    replay = []
    for case in suite['cases'][first:target]:
        if case.get('reset', True):
            model.reset()
        replay.append(model.generate(case['prompt'], case.get('max_new_tokens', 40)))
    case = suite['cases'][target]
    if case.get('reset', True):
        model.reset()
    prefix = (model.history + [model.config['special']['user']]
              + model.tokenizer.encode(case['prompt']) + [model.config['special']['assistant']])
    n = len(prefix)
    begin = max(0, n-a.tail)
    records = {}
    qkv, attended = {}, {}
    original_norm, original_linear = model.norm, model.linear
    heads, width = model.config['heads'], model.config['width']
    head_dim = width // heads
    inspected_heads = {1: 9, 2: 1}

    def norm(x, name):
        if x.shape[1] == n and name in [f'blocks.{layer}.ln{part}' for layer in range(1, 6) for part in (1, 2)]:
            values = ((x[0, begin:, 0::2] - x[0, begin:, 1::2]) / 2).detach().cpu()
            records[name] = {key: values[:, indices].tolist() for key, indices in FIELDS.items()}
        return original_norm(x, name)

    def linear(x, name):
        for layer, head in inspected_heads.items():
            if x.shape[1] == n and name == f'blocks.{layer}.attn.out':
                attended[layer] = x[0, begin:, head*head_dim:(head+1)*head_dim].detach().cpu()
        out = original_linear(x, name)
        for layer, head in inspected_heads.items():
            if x.shape[1] == n and name == f'blocks.{layer}.attn.qkv':
                qkv[layer] = [v[0, :, head*head_dim:(head+1)*head_dim].detach().cpu()
                              for v in out.chunk(3, dim=-1)]
        return out

    model.norm, model.linear = norm, linear
    try:
        answer = model.generate(case['prompt'], case.get('max_new_tokens', 40))
    finally:
        model.norm, model.linear = original_norm, original_linear
    attention = []
    for layer, head in inspected_heads.items():
        q, k, v = qkv[layer]
        for position in range(begin, n):
            entry = dict(layer=layer, head=head, position=position,
                         actual_value_output=attended[layer][position-begin, :16].tolist())
            for dtype, label in ((torch.float32, 'float32_scores'), (torch.float64, 'float64_scores')):
                scores = (q[position].to(dtype) @ k[:position+1].to(dtype).T) / head_dim**.5
                values, indices = scores.topk(min(6, len(scores)))
                entry[label] = [dict(position=int(i), token=model.tokenizer.token_text(prefix[int(i)]),
                                     score=float(s), value=v[int(i), :16].tolist())
                                for s, i in zip(values, indices)]
            attention.append(entry)
    assert digest(a.model / 'weights.f32') == initial_hash
    result = dict(version=a.version, model_sha256=initial_hash, device=a.device, git=git_state(),
                  suite_sha256=digest(a.suite), case_id=a.case_id, expected=case['expected'],
                  replay=replay, answer=answer, prefix_tokens=n, tail_begin=begin,
                  tail_tokens=[dict(position=i, token_id=prefix[i], token=model.tokenizer.token_text(prefix[i]))
                               for i in range(begin, n)], fields=FIELDS, residuals=records, attention=attention,
                  weights_unchanged=True,
                  note='Inference is unmodified. Norm/linear wrappers copy activations for external diagnostics only. Separate score arithmetic does not replace actual attention or influence generated tokens.')
    a.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(version=a.version, device=a.device, case_id=a.case_id,
                          prefix_tokens=n, expected=case['expected'], text=answer['text'])))


if __name__ == '__main__':
    main()
