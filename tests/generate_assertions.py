"""External tests for coordinated assertion events and clause-local binding.

In this controlled ellipsis grammar a negative shared predicate applies to each
coordinated object until an explicit predicate replaces it. Local not negates
one object. Disjunctive and conditional premises are separate stress cases.
"""
import argparse
import json
import random
from pathlib import Path


def generate(seed, status):
    rng = random.Random(seed)
    cases = []

    def name():
        return rng.choice(['vel', 'om']) + ''.join(rng.choice('bcdfghklmnprstvwz') for _ in range(6))

    def add(prompt, expected, category, reset=True):
        cases.append(dict(id=f'assertion-{len(cases):03}', prompt=prompt, expected=expected,
                          category=category, reset=reset))

    # Positive and negative assertion prefix, positive query prefix, query
    # object form and lexical values. Tests are never imported by construction.
    relations = [
        ('is a', 'is not a', 'Is {s}', 'a {v}', ['vehicle', 'machine', 'animal']),
        ('can', 'cannot', 'Can {s}', '{v}', ['fly', 'swim', 'glimmer']),
        ('needs', 'does not need', 'Does {s} need', '{v}', ['water', 'fuel', 'sunlight']),
        ('has', 'does not have', 'Does {s} have', '{v}', ['wings', 'sensors', 'wheels']),
        ('produces', 'does not produce', 'Does {s} produce', '{v}', ['heat', 'oxygen', 'light']),
        ('is', 'is not', 'Is {s}', '{v}', ['cold', 'green', 'furry']),
        ('exists in', 'is not in', 'Is {s}', 'in {v}', ['water', 'space', 'land']),
    ]
    for relation, (positive, negative, prefix, term, vocabulary) in enumerate(relations):
        for negated in (False, True):
            for variant in range(3):
                x = name()
                values = list(vocabulary[:2 + variant % 2]) if variant == 0 or relation == 5 else [name() for _ in range(2 + variant % 2)]
                assertion_terms = values if relation != 0 else [values[0]] + ['a ' + v for v in values[1:]]
                fact = f'A {x} {negative if negated else positive} ' + ' and '.join(assertion_terms) + '.'
                for value in values:
                    add(fact + ' ' + prefix.format(s=x) + ' ' + term.format(v=value) + '?',
                        'No.' if negated else 'Yes.', 'shared_negative' if negated else 'shared_positive')
                query = prefix.format(s=x) + ' ' + ' and '.join(term.format(v=v) for v in values) + '?'
                add(fact + ' ' + query, 'No.' if negated else 'Yes.', 'compound_query_of_compound_fact')
                add(fact + ' ' + prefix.format(s=name()) + ' ' + term.format(v=values[-1]) + '?',
                    'I do not know.', 'subject_isolation')
    for _ in range(8):
        x, y, value = name(), name(), name()
        examples = [
            (f'A {x} can fly and cannot swim and can {value}. Can {x} {value}?', 'Yes.'),
            (f'A {x} cannot fly and can swim and {value}. Can {x} {value}?', 'Yes.'),
            (f'A {x} can fly and not swim and {value}. Can {x} swim?', 'No.'),
            (f'A {x} can fly and not swim and {value}. Can {x} {value}?', 'Yes.'),
            (f'A {x} can fly and has wings and {value}. Does {x} have {value}?', 'Yes.'),
            (f'A {x} can fly and has wings and {value}. Can {x} {value}?', 'I do not know.'),
            (f'A {x} has wings and {y} needs water and {value}. Does {y} need {value}?', 'Yes.'),
            (f'A {x} has wings and {y} needs water and {value}. Does {x} need {value}?', 'I do not know.'),
            (f'A {x} can fly and does not need water. Does {x} need water?', 'No.'),
            (f'A {x} can fly and does not have wings. Does {x} have wings?', 'No.'),
            (f'A {x} can fly and does not produce heat. Does {x} produce heat?', 'No.'),
            (f"A {x} can fly and doesn't need water. Does {x} need water?", 'No.'),
            (f'A {x} is a bird and is not a machine. Is {x} a machine?', 'No.'),
            (f'A {x} is cold and is in water and space. Is {x} in space?', 'Yes.'),
        ]
        for prompt, expected in examples:
            add(prompt, expected, 'predicate_polarity_and_subject_changes')
        add(f'A {x} can fly and swim. A {x} cannot swim. Can {x} fly and swim?', 'No.', 'object_correction')
        add(f'A {x} cannot fly and swim. A {x} can swim. Can {x} swim?', 'Yes.', 'object_correction')
        add(f'A {x} has a sensor and a wheel. Does {x} have wheel?', 'Yes.', 'object_determiners')
        add(f"A {x} doesn't have a sensor and a wheel. Does {x} have wheel?", 'No.', 'object_determiners')
        add(f'A {x} exists in a cave and a forest. Is {x} in forest?', 'Yes.', 'object_determiners')
    for depth in (1, 2, 3):
        for _ in range(5):
            nodes = [name() for _ in range(depth + 1)]
            v1, v2 = name(), name()
            facts = [f'A {a} is a {b}.' for a, b in zip(nodes, nodes[1:])]
            facts.append(f'A {nodes[-1]} can {v1} and {v2}.')
            rng.shuffle(facts)
            add(' '.join(facts) + f' Can {nodes[0]} {v1} and {v2}?', 'Yes.', 'inherited_compound_properties')

    for _ in range(4):
        x, y = name(), name()
        add(f'Does {x} need water and {y} has wings?', 'I do not know.', 'question_is_not_assertion')
        add(f'Does {y} have wings?', 'I do not know.', 'question_is_not_assertion', False)
        add(f'Can {x} fly and swim?', 'I do not know.', 'question_is_not_assertion')
        add(f'Can {x} swim?', 'I do not know.', 'question_is_not_assertion', False)
        add(f'A {x} can fly and swim. Can {x} fly and swim?', 'Yes.', 'conversation')
        add(f'A {x} cannot swim and has wings and sensors. Can {x} swim?', 'No.', 'conversation', False)
        add('Please calculate -149 - 370.', '-519.', 'conversation', False)
        add(f'Does {x} have sensors?', 'Yes.', 'conversation', False)
    x = name()
    for prompt, expected in [
        (f'A {x} can fly or swim. Can {x} fly?', 'I do not know.'),
        (f'A {x} has wings or sensors. Does {x} have wings?', 'I do not know.'),
        (f'If {x} can fly then {x} has wings. Does {x} have wings?', 'I do not know.'),
        (f'If {x} can fly then {x} has wings. A {x} can fly. Does {x} have wings?', 'Yes.'),
        (f'A {x} is a bird and a fish. Can {x} fly and swim?', 'Yes.'),
        (f'A {x} has wings, sensors and wheels. Does {x} have wings and sensors and wheels?', 'Yes.'),
        (f'A {x} does not have a sensor and a wheel. Does {x} have wheel?', 'No.'),
        (f'A {x} can not not fly and swim. Can {x} fly?', 'Yes.'),
        (f'A {x} can fly and swim. What can {x} do?', f'{x.capitalize()} can fly and swim.'),
    ]:
        add(prompt, expected, 'assertion_structure_stress')
    return dict(name='coordinated-assertion-binding', status=status, seed=seed, cases=cases,
                objective='Compile each coordinated assertion into its own subject/predicate/object/polarity event, preserve explicit changes and clause scope, inherit properties, and never treat a question as an assertion. Disjunction, conditionals, multiple parents and broader structures remain scored stresses.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--status', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit('Refusing to overwrite a suite')
    suite = generate(a.seed, a.status)
    a.output.write_text(json.dumps(suite, indent=2) + '\n', encoding='utf-8')
    print(f"Wrote {len(suite['cases'])} coordinated assertion cases")


if __name__ == '__main__':
    main()
