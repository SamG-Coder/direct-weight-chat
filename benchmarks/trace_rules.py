"""Read-only activation trace of typed rule and atomic-fact fields."""
import argparse
import json
from pathlib import Path

from .evaluate import digest, git_state
from chat1gb_autoregressive.engine import Model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--version', required=True)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--prompt')
    inputs.add_argument('--suite', type=Path)
    p.add_argument('--case-id')
    p.add_argument('--tail', type=int, default=80)
    p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit('Refusing to overwrite evidence')
    if a.tail < 1 or a.suite and not a.case_id:
        raise SystemExit('Need positive tail size and a case ID for a suite')
    m = Model(a.model, device=a.device)
    replay, expected, suite_hash = [], None, None
    if a.suite:
        suite = json.loads(a.suite.read_text(encoding='utf-8'))
        target = next(i for i, case in enumerate(suite['cases']) if case['id'] == a.case_id)
        first = target
        while first > 0 and not suite['cases'][first].get('reset', True):
            first -= 1
        for case in suite['cases'][first:target]:
            if case.get('reset', True):
                m.reset()
            replay.append(m.generate(case['prompt'], case.get('max_new_tokens', 40)))
        case = suite['cases'][target]
        if case.get('reset', True):
            m.reset()
        a.prompt, expected, suite_hash = case['prompt'], case['expected'], digest(a.suite)
    prefix = m.history + [m.config['special']['user']] + m.tokenizer.encode(a.prompt) + [m.config['special']['assistant']]
    begin = max(0, len(prefix) - a.tail)
    fields = {'rule_valid': [175], 'rule_context': [56, 59, 60], 'rule_terminator_join': [259, 260],
              'if_then_phase': [494, 495, 496, 497, 498],
              'parsing_premise_valid_group': [100, 101], 'parsing_conclusion_valid_group': [138, 139],
              'previous_clause_group': [258], 'query': list(range(30, 50)),
              'proof': list(range(288, 368)), 'static_has': list(range(112, 128))}
    records = {}
    original = m.norm

    def norm(x, name):
        if x.shape[1] == len(prefix) and name.startswith('blocks.') and int(name.split('.')[1]) <= 10:
            features = ((x[0, begin:, 0::2] - x[0, begin:, 1::2]) / 2).detach().cpu()
            records[name] = {key: features[:, coordinates].tolist() for key, coordinates in fields.items()}
        return original(x, name)

    initial = digest(a.model / 'weights.f32')
    m.norm = norm
    try:
        answer = m.generate(a.prompt)
    finally:
        m.norm = original
    result = dict(version=a.version, device=m.device, git=git_state(), model_sha256=initial,
                  answer=answer, expected=expected, case_id=a.case_id, suite_sha256=suite_hash, replay=replay,
                  prefix_tokens=len(prefix), tail_begin=begin,
                  tokens=[m.tokenizer.token_text(i) for i in prefix[begin:]], fields=fields,
                  residuals=records, weights_unchanged=digest(a.model / 'weights.f32') == initial,
                  note='External read-only norm hook copies residuals. It does not change activations, output decisions, or saved weights.')
    a.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(answer['text'])


if __name__ == '__main__':
    main()
