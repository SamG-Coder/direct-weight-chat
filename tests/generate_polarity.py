"""External polarity tests: grammar, inheritance, corrections and scope limits."""
import argparse
import json
import random
from pathlib import Path


def generate(seed, status):
    rng = random.Random(seed)
    cases = []
    # Question, positive predicate, negative input variants, canonical negative
    # output predicate, and values. These strings never enter the compiler.
    relations = [
        ('What is a {x}?', 'is a', ["is not a", "isn't a"], 'is not a', ['vehicle', 'machine']),
        ('What can a {x} do?', 'can', ['cannot', "can't", 'can not'], 'cannot', ['fly', 'swim', 'glimmer', 'shimmer']),
        ('What does a {x} need?', 'needs', ['does not need', "doesn't need"], 'does not need', ['water', 'fuel', 'sunlight']),
        ('What does a {x} have?', 'has', ['does not have', "doesn't have"], 'does not have', ['wings', 'wheels', 'sensors']),
        ('What does a {x} produce?', 'produces', ['does not produce', "doesn't produce"], 'does not produce', ['heat', 'oxygen', 'light']),
        ('Describe a {x}.', 'is', ['is not', "isn't"], 'is not', ['cold', 'green', 'furry']),
        ('Where is a {x}?', 'exists in', ['is not in', "isn't in"], 'is not in', ['water', 'space', 'land']),
    ]
    def name():
        return rng.choice(['vel', 'om']) + ''.join(rng.choice('bcdfghklmnprstvwz') for _ in range(6))
    def add(prompt, expected, category, reset=True):
        cases.append(dict(id=f'polarity-{len(cases):03}', prompt=prompt, expected=expected,
                          category=category, reset=reset))
    for relation, (question, positive, negative_forms, canonical, values) in enumerate(relations):
        for variant in range(6):
            x, other = name(), name()
            value = rng.choice(values)
            negative = negative_forms[variant % len(negative_forms)]
            q = question.format(x=x)
            pos = f'A {x} {positive} {value}.'
            neg = f'A {x} {negative} {value}.'
            answer = f'{x.capitalize()} {canonical} {value}.'
            add(neg + ' ' + q, answer, 'direct_negative')
            add(pos + ' ' + neg + ' ' + q, answer, 'positive_to_negative')
            add(neg + ' ' + pos + ' ' + q, f'{x.capitalize()} {positive} {value}.', 'negative_to_positive')
            add(f'A {other} {positive} {value}. ' + neg + ' ' + q, answer, 'distractor')
            if relation:
                nodes = [x] + [name() for _ in range(1 + variant % 3)]
                facts = [f'A {a} is a {b}.' for a, b in zip(nodes, nodes[1:])]
                facts.append(f'A {nodes[-1]} {negative} {value}.')
                rng.shuffle(facts)
                add(' '.join(facts) + ' ' + q, answer, 'inherited_negative')
    for _ in range(4):
        x, parent = name(), name()
        add(f'A {x} is not a bird. What can a {x} do?', 'I do not know.', 'negative_class_blocks_inheritance')
        add(f'A {parent} can fly. A {x} is not a {parent}. What can a {x} do?', 'I do not know.', 'negative_class_blocks_inheritance')
        add(f'A {x} is a bird. A {x} cannot fly. Explain a {x}.', f'{x.capitalize()} is a bird. It cannot fly.', 'multi_sentence')
        add(f'A {x} is not a bird. A {x} can swim. Explain a {x}.', f'{x.capitalize()} is not a bird. It can swim.', 'multi_sentence')
        add(f'An {x} is not an observatory. What is an {x}?', f'{x.capitalize()} is not an observatory.', 'article_choice')
    x = name()
    add(f'A {x} can fly. What can a {x} do?', f'{x.capitalize()} can fly.', 'conversation')
    add(f'A {x} cannot fly. What can a {x} do?', f'{x.capitalize()} cannot fly.', 'conversation', False)
    add(f'A {x} can swim. What can a {x} do?', f'{x.capitalize()} can swim.', 'conversation', False)
    add(f'A {x} can fly. A {x} cannot swim. What can a {x} do?', f'{x.capitalize()} can fly.', 'multi_value_stress')
    add(f'A {x} can fly. Can a {x} fly?', 'Yes.', 'yes_no_stress')
    add(f'A {x} cannot fly. Can a {x} fly?', 'No.', 'yes_no_stress')
    add(f'A {x} can fly. What can a {x} not do?', 'I do not know.', 'query_scope_stress')
    add('Do not explain a cat.', 'I do not know.', 'instruction_scope_stress')
    add(f'A {x} is not unable to fly. What can a {x} do?', f'{x.capitalize()} can fly.', 'double_negation_stress')
    return dict(name='fact-polarity', status=status, seed=seed, cases=cases,
                objective='Preserve negative predicate assertions through retrieval/inheritance; latest correction wins; negative membership cannot imply positive inheritance. Scope/multiple-value cases expose retained limits.')


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
    print(f"Wrote {len(suite['cases'])} polarity cases")


if __name__ == '__main__':
    main()
