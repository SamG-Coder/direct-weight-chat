"""External checks of subject/frame binding across positions and devices.

These tests extend the v14b post-freeze failures into a declared development
suite before changing the model. Answers are external labels, never model input.
"""
import argparse
import json
import random
from pathlib import Path


def generate(seed, status):
    rng = random.Random(seed)
    cases = []
    relations = [
        ('class', 'is a', 'is not a', 'Is {s} a {v}?', ['vehicle', 'machine']),
        ('can', 'can', 'cannot', 'Can {s} {v}?', ['fly', 'swim']),
        ('needs', 'needs', 'does not need', 'Does {s} need {v}?', ['water', 'fuel']),
        ('has', 'has', 'does not have', 'Does {s} have {v}?', ['wings', 'sensors']),
        ('produces', 'produces', 'does not produce', 'Does {s} produce {v}?', ['heat', 'oxygen']),
        ('trait', 'is', 'is not', 'Is {s} {v}?', ['cold', 'green']),
        ('location', 'exists in', 'is not in', 'Is {s} in {v}?', ['water', 'space']),
    ]

    def name():
        return rng.choice(['vel', 'om']) + ''.join(rng.choice('bcdfghklmnprstvwz') for _ in range(6))

    def padding(length, kind='greetings'):
        assert length % 2 == 0
        if kind == 'facts':
            return ('A derv has wheels. A pelm needs water. ' * (length // 10)
                    + 'Hello. ' * ((length % 10) // 2))
        return 'Hello. ' * (length // 2)

    def add(prompt, expected, category, length, relation, reset=True):
        cases.append(dict(id=f'binding-{len(cases):03}', prompt=prompt, expected=expected,
                          category=category, padding_tokens=length, relation=relation, reset=reset))

    def fact(subject, relation, negative=False):
        label, positive, negated, query, values = relation
        second = ('a ' if label == 'class' else '') + values[1]
        return f'a {subject} {negated if negative else positive} {values[0]} and {second}.'

    for length in (0, 1024, 2048, 3072, 4096, 4700, 4900):
        for relation in relations:
            label, positive, negated, query, values = relation
            x, y = name(), name()
            for negative in (False, True):
                prompt = padding(length) + f'A {x} has wheels and ' + fact(y, relation, negative)
                prompt += ' ' + query.format(s=y, v=values[1])
                add(prompt, 'No.' if negative else 'Yes.',
                    'changed_subject_negative' if negative else 'changed_subject_positive', length, label)

    for relation in relations:
        label, positive, negated, query, values = relation
        x, y = name(), name()
        prefix = padding(4900) + f'A {x} has wheels and ' + fact(y, relation)
        add(prefix + ' ' + query.format(s=x, v=values[1]), 'I do not know.',
            'wrong_subject_isolation', 4900, label)
        add(padding(4900) + fact(y, relation).capitalize() + ' ' + query.format(s=y, v=values[1]),
            'Yes.', 'single_subject_control', 4900, label)

    for _ in range(8):
        length = 2 * rng.randrange(0, 2451)
        relation = rng.choice(relations)
        label, positive, negated, query, values = relation
        x, y = name(), name()
        clause = fact(y, relation)
        if rng.choice([False, True]):
            clause = clause.removeprefix('a ')
        prefix = padding(length, 'facts') + f'A {x} has wheels and ' + clause
        add(prefix + ' ' + query.format(s=y, v=values[1]), 'Yes.', 'random_offset_binding', length, label)
        correction = f'A {y} {negated} {values[1]}.'
        add(prefix + ' ' + correction + ' ' + query.format(s=y, v=values[1]),
            'No.', 'random_offset_correction', length, label)

    for kind in ('greetings', 'facts'):
        x, y = name(), name()
        add(padding(3000, kind) + f'A {x} can fly and swim. Can {x} swim?',
            'Yes.', 'long_conversation', 3000, 'can')
        add(f'A {x} has wings and a {y} needs water and fuel. Does {y} need fuel?',
            'Yes.', 'long_conversation', 3000, 'needs', False)
        add(f'Does {x} need fuel?', 'I do not know.', 'long_conversation', 3000, 'needs', False)
        add(f'A {y} does not need fuel. Does {y} need fuel?', 'No.', 'long_conversation', 3000, 'needs', False)
    return dict(name='subject-binding-across-positions', seed=seed, status=status, cases=cases,
                objective='Bind each coordinated object to the latest valid subject in its clause across positions, predicates, polarities and dialogue; preserve isolation and corrections on CPU and CUDA.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--status', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit('Refusing to overwrite evidence')
    suite = generate(a.seed, a.status)
    a.output.write_text(json.dumps(suite, indent=2) + '\n', encoding='utf-8')
    print(f"Wrote {len(suite['cases'])} subject-binding cases")


if __name__ == '__main__':
    main()
