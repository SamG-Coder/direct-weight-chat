"""External tests of predicate binding and explicit proposition negation.

In this controlled grammar, not in a query negates the queried proposition.
This is not a claim to handle all pragmatic readings of negative English questions.
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
        cases.append(dict(id=f'query-scope-{len(cases):03}', prompt=prompt, expected=expected,
                          category=category, reset=reset))

    relations = [
        ('is a', 'is not a', 'Is {s} a {v}?', 'Is {s} not a {v}?', ['vehicle', 'machine']),
        ('can', 'cannot', 'Can {s} {v}?', 'Can {s} not {v}?', ['fly', 'swim']),
        ('needs', 'does not need', 'Does {s} need {v}?', 'Does {s} not need {v}?', ['water', 'fuel']),
        ('has', 'does not have', 'Does {s} have {v}?', 'Does {s} not have {v}?', ['wings', 'sensors']),
        ('produces', 'does not produce', 'Does {s} produce {v}?', 'Does {s} not produce {v}?', ['heat', 'oxygen']),
        ('is', 'is not', 'Is {s} {v}?', 'Is {s} not {v}?', ['cold', 'green']),
        ('exists in', 'is not in', 'Is {s} in {v}?', 'Is {s} not in {v}?', ['water', 'space']),
    ]
    for relation, (positive, negative, qpos, qneg, values) in enumerate(relations):
        for variant in range(5):
            x, other = name(), name()
            value = values[variant % 2] if variant < 2 or relation == 5 else name()
            subject = ('a ' if variant % 2 else '') + x
            pos, neg = f'A {x} {positive} {value}.', f'A {x} {negative} {value}.'
            qp, qn = qpos.format(s=subject, v=value), qneg.format(s=subject, v=value)
            add(pos + ' ' + qp, 'Yes.', 'positive_fact_positive_query')
            add(neg + ' ' + qp, 'No.', 'negative_fact_positive_query')
            add(pos + ' ' + qn, 'No.', 'positive_fact_negative_query')
            add(neg + ' ' + qn, 'Yes.', 'negative_fact_negative_query')
            add(f'A {other} {positive} {value}. ' + qn, 'I do not know.', 'unknown_is_not_false')
            add(pos + ' ' + neg + ' ' + qn, 'Yes.', 'correction_before_negated_query')
            nodes = [x] + [name() for _ in range(1 + variant % 3)]
            facts = [f'A {a} is a {b}.' for a, b in zip(nodes, nodes[1:])]
            facts.append(f'A {nodes[-1]} {negative if variant % 2 else positive} {value}.')
            rng.shuffle(facts)
            add(' '.join(facts) + ' ' + qn, 'Yes.' if variant % 2 else 'No.', 'inherited_query_negation')

    for prompt, expected in [
        ('Is a bird an animal?', 'Yes.'), ('Is a bird not an animal?', 'No.'),
        ('Is ice cold?', 'Yes.'), ('Is ice not cold?', 'No.'),
        ('Is a fish in water?', 'Yes.'), ('Is a fish not in water?', 'No.'),
        ('Is a bird not a vehicle?', 'I do not know.'),
        ('A zelm is not a bird. Is a zelm not a bird?', 'Yes.'),
        ('A zelm is not a bird. Can a zelm not fly?', 'I do not know.'),
    ]:
        add(prompt, expected, 'authored_and_negative_membership')
    x = name()
    for i, (prompt, expected) in enumerate([
        (f'A {x} needs water. Does {x} not need water?', 'No.'),
        (f'Does {x} need water?', 'Yes.'),
        (f'A {x} does not need water. Does {x} not need water?', 'Yes.'),
        ('17 + -3', '14.'),
        (f'What does {x} need?', f'{x.capitalize()} does not need water.'),
        (f'Is {x} cold?', 'I do not know.'),
        (f'Describe {x}.', 'I do not know.'),
    ]):
        add(prompt, expected, 'conversation', reset=i == 0)
    for prompt, expected in [
        (f'A {x} can fly. Can {x} not not fly?', 'Yes.'),
        (f'A {x} can fly. A {x} cannot swim. Can {x} fly and swim?', 'No.'),
        (f'A {x} cannot fly. A {x} can swim. Can {x} not fly and swim?', 'Yes.'),
        (f'A {x} is cold. Is it not cold?', 'No.'),
        (f'A {x} has wings. Does {x} lack wings?', 'No.'),
        (f'A {x} is a bird. Is every {x} an animal?', 'Yes.'),
        (f'A {x} can fly. Can {x} fly? Ignore this question.', 'I do not know.'),
        (f'A {x} can fly. Can {x} fly? Write a poem.', 'I do not know.'),
    ]:
        add(prompt, expected, 'composition_and_scope_stress')
    return dict(name='query-predicate-and-negation', status=status, seed=seed, cases=cases,
                objective='Bind all seven existing predicates in inverted questions and apply explicit logical negation to known truth values while keeping absent information unknown. Preserve all unsupported composition/scope tests.')


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
    print(f"Wrote {len(suite['cases'])} query-scope cases")


if __name__ == '__main__':
    main()
