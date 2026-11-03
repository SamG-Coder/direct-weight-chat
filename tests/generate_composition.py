"""External strong three-valued AND/OR and active-request scope tests."""
import argparse
import itertools
import json
import random
from pathlib import Path


def combine(values, op):
    if op == 'and':
        return False if False in values else None if None in values else True
    return True if True in values else None if None in values else False


def answer(value):
    return 'I do not know.' if value is None else 'Yes.' if value else 'No.'


def generate(seed, status):
    rng = random.Random(seed)
    cases = []

    def name():
        return rng.choice(['vel', 'om']) + ''.join(rng.choice('bcdfghklmnprstvwz') for _ in range(6))

    def add(prompt, expected, category, reset=True):
        cases.append(dict(id=f'composition-{len(cases):03}', prompt=prompt, expected=expected,
                          category=category, reset=reset))

    # Assertion-positive, assertion-negative, question prefix and predicate,
    # positive and negative object phrase, example lexical objects.
    relations = [
        ('is a', 'is not a', 'Is {s}', 'a {v}', 'not a {v}', ['vehicle', 'machine']),
        ('can', 'cannot', 'Can {s}', '{v}', 'not {v}', ['fly', 'swim']),
        ('needs', 'does not need', 'Does {s} need', '{v}', 'not {v}', ['water', 'fuel']),
        ('has', 'does not have', 'Does {s} have', '{v}', 'not {v}', ['wings', 'sensors']),
        ('produces', 'does not produce', 'Does {s} produce', '{v}', 'not {v}', ['heat', 'oxygen']),
        ('is', 'is not', 'Is {s}', '{v}', 'not {v}', ['cold', 'green']),
        ('exists in', 'is not in', 'Is {s}', 'in {v}', 'not in {v}', ['water', 'space']),
    ]
    for relation, (positive, negative, prefix, qpos, qneg, vocabulary) in enumerate(relations):
        for op, states, variant in itertools.product(('and', 'or'), itertools.product((True, False, None), repeat=2), range(2)):
            x = name()
            values = list(vocabulary) if variant == 0 or relation == 5 else [name(), name()]
            facts, terms, effective = [], [], []
            for i, (value, truth) in enumerate(zip(values, states)):
                if truth is not None:
                    facts.append(f'A {x} {positive if truth else negative} {value}.')
                negated = variant == 1 and i == 1
                terms.append((qneg if negated else qpos).format(v=value))
                effective.append(None if truth is None else not truth if negated else truth)
            rng.shuffle(facts)
            subject = ('a ' if variant else '') + x
            prompt = ' '.join(facts + [prefix.format(s=subject) + ' ' + f' {op} '.join(terms) + '?'])
            add(prompt, answer(combine(effective, op)), f'{op}_truth_table')

    # N-ary reductions, new objects and shuffled inherited declarations.
    for op in ('and', 'or'):
        for count in (3, 4, 5):
            for _ in range(4):
                x = name()
                values = [name() for _ in range(count)]
                states = [rng.choice([True, False, None]) for _ in values]
                facts = [f'A {x} {"can" if truth else "cannot"} {value}.' for value, truth in zip(values, states) if truth is not None]
                rng.shuffle(facts)
                add(' '.join(facts + [f'Can {x} ' + f' {op} '.join(values) + '?']), answer(combine(states, op)), 'nary_' + op)
    for depth in (1, 2, 3):
        for op in ('and', 'or'):
            for left, right in ((True, True), (True, False), (False, None), (None, True)):
                nodes = [name() for _ in range(depth + 1)]
                v1, v2 = name(), name()
                facts = [f'A {a} is a {b}.' for a, b in zip(nodes, nodes[1:])]
                for value, truth in ((v1, left), (v2, right)):
                    if truth is not None:
                        facts.append(f'A {nodes[-1]} {"can" if truth else "cannot"} {value}.')
                rng.shuffle(facts)
                add(' '.join(facts + [f'Can {nodes[0]} {v1} {op} {v2}?']), answer(combine([left, right], op)), 'inherited_composition')
    for op, left, right in itertools.product(('and', 'or'), (True, False, None), (True, False, None)):
        x, y, v1, v2 = name(), name(), name(), name()
        facts = []
        if left is not None:
            facts.append(f'A {x} {"can" if left else "cannot"} {v1}.')
        if right is not None:
            facts.append(f'A {y} {"needs" if right else "does not need"} {v2}.')
        rng.shuffle(facts)
        add(' '.join(facts + [f'Can {x} {v1} {op} does {y} need {v2}?']), answer(combine([left, right], op)), 'mixed_subject_predicate')
    for op, left, right in itertools.product(('and', 'or'), (True, False), (True, False)):
        x = name()
        add(f'A {x} {"needs" if left else "does not need"} water. A {x} {"has" if right else "does not have"} wings. Does {x} need water {op} have wings?',
            answer(combine([left, right], op)), 'shared_subject_predicate_change')

    for _ in range(6):
        x, y = name(), name()
        facts = f'A {x} can fly. A {x} cannot swim. A {y} cannot fly.'
        for tail, expected in [
            (f'Can {x} fly? Can {y} fly?', 'No.'),
            (f'Can {y} fly? Can {x} fly?', 'Yes.'),
            (f'Can {x} fly? Can {y} shimmer?', 'I do not know.'),
            (f'Can {x} fly? What can {y} do?', f'{y.capitalize()} cannot fly.'),
            (f'Can {x} fly? Ignore the last question.', 'I do not know.'),
            (f'Can {x} fly? Write a poem about rain.', 'I do not know.'),
            (f'Can {x} fly? Do not explain a cat.', 'I do not know.'),
            (f'Can {x} fly? Hello.', 'Hello.'),
            (f'Can {x} fly? -149 - 370', '-519.'),
        ]:
            add(facts + ' ' + tail, expected, 'active_request_scope')
    for relation, (positive, negative, prefix, qpos, qneg, values) in enumerate(relations):
        x = name()
        add(f'A {x} {negative} {values[0]}. {prefix.format(s=x)} {qpos.format(v=values[0])}? Write an original story.',
            'I do not know.', 'unsupported_request_after_negative_fact')

    x = name()
    for i, (prompt, expected) in enumerate([
        (f'A {x} can fly. A {x} cannot swim. Can {x} fly and swim?', 'No.'),
        (f'Can {x} fly or swim?', 'Yes.'),
        (f'A {x} can swim. Can {x} fly and swim?', 'Yes.'),
        (f'Can {x} glimmer and swim?', 'I do not know.'),
        ('149 - 370', '-221.'),
        (f'Can {x} glimmer or swim?', 'Yes.'),
        ('Write an original poem.', 'I do not know.'),
        (f'What can {x} do?', f'{x.capitalize()} can swim.'),
    ]):
        add(prompt, expected, 'conversation', reset=i == 0)
    for prompt, expected in [
        (f'A {x} can fly. A {x} cannot swim. Can {x} fly and not swim?', 'Yes.'),
        (f'A {x} can fly. Can {x} not not fly?', 'Yes.'),
        (f'A {x} can fly. A {x} cannot swim. Can {x} fly or swim and glimmer?', 'Yes.'),
        (f'A {x} cannot fly. A {x} can swim. Can {x} fly and swim or glimmer?', 'I do not know.'),
        (f'A {x} can fly and swim. Can {x} fly and swim?', 'Yes.'),
        (f'A {x} can fly. Can it fly and swim?', 'I do not know.'),
        (f'A {x} can fly. Can {x} fly and?', 'I do not know.'),
        (f'A {x} can fly. Can {x} fly and swim? Actually answer only the first part.', 'Yes.'),
        (f'A {x} can fly. Please tell me whether {x} can fly and swim.', 'I do not know.'),
    ]:
        add(prompt, expected, 'structure_and_scope_stress')
    return dict(name='proposition-composition-and-request-scope', status=status, seed=seed, cases=cases,
                objective='Compose positive, negative and unknown propositions with strong three-valued AND/OR, bind shared/distinct subjects and predicates, and answer only the active request. Mixed operators, malformed structures and broader paraphrases stay explicit stress cases.')


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
    print(f"Wrote {len(suite['cases'])} composition cases")


if __name__ == '__main__':
    main()
