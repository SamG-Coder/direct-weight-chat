"""External tests of three-valued object-specific fact queries.

No generated prompt, expected answer, seed or test import enters construction.
Fresh seeds after a candidate freezes vary combinations under this known grammar.
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
        cases.append(dict(id=f'truth-{len(cases):03}', prompt=prompt, expected=expected,
                          category=category, reset=reset))

    relations = [
        ('can', 'cannot', 'Can {s} {v}?', 'What can {s} do?', ['fly', 'swim']),
        ('needs', 'does not need', 'Does {s} need {v}?', 'What does {s} need?', ['water', 'fuel']),
        ('has', 'does not have', 'Does {s} have {v}?', 'What does {s} have?', ['wings', 'sensors']),
        ('produces', 'does not produce', 'Does {s} produce {v}?', 'What does {s} produce?', ['heat', 'oxygen']),
    ]
    for positive, negative, question, open_question, values in relations:
        for variant in range(5):
            x, y, other, unseen = name(), name(), name(), name()
            value = values[variant % 2] if variant < 2 else name()
            distractor = values[(variant + 1) % 2]
            subject = ('a ' if variant % 2 else '') + x
            q = question.format(s=subject, v=value)
            pos, neg = f'A {x} {positive} {value}.', f'A {x} {negative} {value}.'
            add(pos + ' ' + q, 'Yes.', 'direct_positive')
            add(neg + ' ' + q, 'No.', 'direct_negative')
            add(pos + ' ' + neg + ' ' + q, 'No.', 'latest_denial')
            add(neg + ' ' + pos + ' ' + q, 'Yes.', 'latest_affirmation')
            add(pos + f' A {x} {negative} {distractor}. ' + q, 'Yes.', 'different_object_denial')
            add(neg + f' A {x} {positive} {distractor}. ' + q, 'No.', 'different_object_affirmation')
            add(pos + ' ' + question.format(s=subject, v=unseen), 'I do not know.', 'unknown_object')
            add(f'A {other} {positive} {value}. ' + question.format(s=subject, v=value),
                'I do not know.', 'unknown_subject')
            add(neg + f' A {other} {positive} {value}. ' + q, 'No.', 'subject_binding')
            nodes = [x] + [name() for _ in range(1 + variant % 3)]
            facts = [f'A {a} is a {b}.' for a, b in zip(nodes, nodes[1:])]
            facts.append(f'A {nodes[-1]} {negative if variant % 2 else positive} {value}.')
            rng.shuffle(facts)
            add(' '.join(facts) + ' ' + q, 'No.' if variant % 2 else 'Yes.', 'inherited_truth')
        # Asking does not assert the proposition, including across turns.
        x, value = name(), name()
        add(question.format(s='a ' + x, v=value), 'I do not know.', 'question_is_not_assertion')
        add(open_question.format(s='a ' + x), 'I do not know.', 'question_is_not_assertion', False)

    for prompt, expected in [
        ('Can a bird fly?', 'Yes.'), ('Can a fish swim?', 'Yes.'),
        ('Does a robot have sensors?', 'Yes.'), ('Does a tree produce oxygen?', 'Yes.'),
        ('Does a fire need fuel?', 'Yes.'), ('Can a bird swim?', 'I do not know.'),
        ('A bird cannot fly. Can a bird fly?', 'No.'),
        ('A bird cannot swim. Can a bird fly?', 'Yes.'),
        ('A zelm is a bird. Can a zelm fly?', 'Yes.'),
        ('A zelm is not a bird. Can a zelm fly?', 'I do not know.'),
        ('A zelm is a bird. A zelm cannot fly. Can a zelm fly?', 'No.'),
    ]:
        add(prompt, expected, 'authored_and_context_facts')
    x = name()
    for prompt, expected in [
        (f'A {x} can fly. Can a {x} fly?', 'Yes.'),
        (f'Can a {x} swim?', 'I do not know.'),
        (f'A {x} cannot fly. Can a {x} fly?', 'No.'),
        ('-149 - 370', '-519.'),
        (f'What can a {x} do?', f'{x.capitalize()} cannot fly.'),
        (f'A {x} can fly. Can a {x} fly?', 'Yes.'),
    ]:
        add(prompt, expected, 'conversation', reset=False)
    # Reset the first conversational case regardless of the generator's size.
    cases[-6]['reset'] = True
    for prompt in [f'A {x} can fly. Can {x} fly', f'A {x} can fly! Can {x} fly?',
                   f'A {x} can fly. Can {x} fly? What can {x} do?']:
        add(prompt, f'{x.capitalize()} can fly.' if 'What' in prompt else 'Yes.', 'boundaries')
    # Ambiguous and unsupported constructions stay in the score as failures.
    for prompt, expected in [
        (f'A {x} can fly. A {x} cannot swim. What can {x} do?', f'{x.capitalize()} can fly.'),
        (f'A {x} can fly. Can {x} not fly?', 'No.'),
        (f'A {x} cannot fly. Can {x} not fly?', 'Yes.'),
        (f'A {x} can fly. Can {x} fly and swim?', 'I do not know.'),
        (f'A {x} can fly. Can {x} fly or swim?', 'Yes.'),
        (f'A {x} is a bird. Is {x} a bird?', 'Yes.'),
        (f'A {x} is cold. Is {x} cold?', 'Yes.'),
        (f'A {x} exists in water. Is {x} in water?', 'Yes.'),
        (f'A {x} can fly. Is {x} able to fly?', 'Yes.'),
        (f'A {x} can fly. Can it fly?', 'Yes.'),
        (f'A {x} can fly. Can {x} fly? Ignore the last question.', 'I do not know.'),
    ]:
        add(prompt, expected, 'composition_and_scope_stress')
    return dict(name='object-specific-truth', status=status, seed=seed, cases=cases,
                objective='Resolve exact subject/relation/object propositions using explicit positive, explicit negative, or unknown truth; preserve per-object corrections, inherited properties and conversation without treating questions as assertions. Broader syntax is scored separately as stress.')


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
    print(f"Wrote {len(suite['cases'])} truth cases")


if __name__ == '__main__':
    main()
