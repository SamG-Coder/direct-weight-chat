"""Post-freeze adversarial audit of subject ambiguity, request scope and length.

These structures were authored by the same engineer after the v13a freeze.
They are additional boundary evidence, not an independent blind evaluation.
"""
import argparse
import json
import random
from pathlib import Path


def generate(seed):
    rng = random.Random(seed)
    cases = []

    def name():
        return 'vel' + ''.join(rng.choice('bcdfghklmnprstvwz') for _ in range(6))

    def add(prompt, expected, category, reset=True):
        cases.append(dict(id=f'assertion-boundary-{len(cases):03}', prompt=prompt,
                          expected=expected, category=category, reset=reset))

    for _ in range(4):
        x, y = name(), name()
        for article in ('', 'a '):
            fact = f'A {x} has wings and {article}{y} needs water and fuel.'
            add(fact + f' Does {x} have {y}?', 'I do not know.', 'new_subject_is_not_an_object')
            add(fact + f' Does {y} need fuel?', 'Yes.', 'new_subject_positive_control')
        for question in (f'Could you tell me if {x} has wings?',
                         f'Tell me whether {x} has wings.',
                         f'Please tell me whether {x} has wings.'):
            add(question, 'I do not know.', 'indirect_question_scope')
            add(f'Does {x} have wings?', 'I do not know.', 'indirect_question_memory', False)
        add(f'A {x} can fly. And swim. Can {x} swim?', 'I do not know.', 'clause_frame_isolation')
        add(f'A {x} can fly and has wings and sensors. Does {x} have sensors?', 'Yes.', 'predicate_control')

    # Two lexical tokens per padding repetition. No new pointer identities are
    # consumed by the padding, and even the longest case fits the 5,048-token
    # model window with a 40-token generation allowance.
    for padding in (0, 64, 256, 512, 1024, 2048, 2375):
        x, y = name(), name()
        prefix = 'Hello. ' * padding
        examples = [
            (prefix + f'A {x} can fly and swim. Can {x} swim?', 'Yes.', 'context_shared_object'),
            (prefix + f'A {x} has wings and {y} needs water and fuel. Does {y} need fuel?', 'Yes.', 'context_subject_change'),
            (f'A {x} can fly. ' + prefix + f'And swim. Can {x} swim?', 'I do not know.', 'context_clause_isolation'),
            (prefix + f'A {x} cannot fly and can swim and glimmer. Can {x} glimmer?', 'Yes.', 'context_polarity_change'),
        ]
        for prompt, expected, category in examples:
            add(prompt, expected, category)
            cases[-1]['padding_tokens'] = 2 * padding
    return dict(name='assertion-boundary-audit', status='post-freeze-author-designed-boundaries',
                seed=seed, frozen_source_commit='89e58f3625b61c952c5cbac2284ce4cde8948279',
                cases=cases,
                objective='Audit false assertions from coordinated subjects and indirect questions, plus numerical/context-length boundaries. No v13a weights are changed after these results.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit('Refusing to overwrite a suite')
    suite = generate(a.seed)
    a.output.write_text(json.dumps(suite, indent=2) + '\n', encoding='utf-8')
    print(f"Wrote {len(suite['cases'])} post-freeze boundary cases")


if __name__ == '__main__':
    main()
