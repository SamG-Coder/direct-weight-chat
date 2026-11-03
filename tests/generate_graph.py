"""External graph benchmark generator. Never imported by construction/inference."""
import argparse
import json
import random
from pathlib import Path


def generate(seed, status, count=48):
    rng = random.Random(seed)
    cases = []
    relations = [
        ('can', 'can', 'What can a {name} do?', ['swim', 'pivot', 'glimmer', 'shimmer']),
        ('needs', 'needs', 'What does a {name} need?', ['water', 'fuel', 'nectar', 'sunlight']),
        ('has', 'has', 'What does a {name} have?', ['wheels', 'sensors', 'fins', 'pages']),
        ('produces', 'produces', 'What does a {name} produce?', ['heat', 'light', 'oxygen', 'pollen']),
        ('is', 'is', 'Describe a {name}.', ['cold', 'hot', 'green', 'furry']),
        ('exists in', 'exists in', 'Where is a {name}?', ['water', 'space', 'sky', 'land']),
    ]
    def name():
        return 'zel' + ''.join(rng.choice('bcdfghklmnprstvwz') for _ in range(6))
    def add(prompt, expected, category, reset=True):
        cases.append(dict(id=f'graph-{len(cases):03}', prompt=prompt, expected=expected,
                          category=category, reset=reset))
    for i in range(count):
        depth = 1 + i % 3
        nodes = [name() for _ in range(depth + 1)]
        rel, answer_rel, question, values = relations[(i // 3) % len(relations)]
        value = rng.choice(values)
        facts = [f'A {a} is a {b}.' for a, b in zip(nodes, nodes[1:])]
        facts.append(f'A {nodes[-1]} {rel} {value}.')
        distractor = name()
        facts.append(f'A {distractor} {rel} {rng.choice(values)}.')
        rng.shuffle(facts)
        add(' '.join(facts) + ' ' + question.format(name=nodes[0]),
            f'{nodes[0].capitalize()} {answer_rel} {value}.', f'chain_depth_{depth}')
    for depth in (4, 5, 8):
        nodes = [name() for _ in range(depth + 1)]
        facts = [f'A {a} is a {b}.' for a, b in zip(nodes, nodes[1:])]
        facts.append(f'A {nodes[-1]} can shimmer.')
        rng.shuffle(facts)
        add(' '.join(facts) + f' What can a {nodes[0]} do?',
            f'{nodes[0].capitalize()} can shimmer.', 'depth_stress')
    for _ in range(4):
        a, b, c = name(), name(), name()
        add(f'A {a} is a {b}. A {b} is a {c}. A {c} can fly. A {a} can swim. What can a {a} do?',
            f'{a.capitalize()} can swim.', 'nearest_override')
        add(f'A {a} is a {b}. A {b} is a {c}. A {c} can fly. A {b} can pivot. What can a {a} do?',
            f'{a.capitalize()} can pivot.', 'nearest_override')
        add(f'A {a} is a {b}. A {b} is a {c}. A {c} can fly. A {c} can swim. What can a {a} do?',
            f'{a.capitalize()} can swim.', 'latest_ancestor_fact')
        add(f'A {a} is a {b}. A {b} is a bird. What can a {a} do?',
            f'{a.capitalize()} can fly.', 'authored_leaf')
        add(f'A {a} is a {b}. A {b} is a {a}. What can a {a} do?',
            'I do not know.', 'cycle_without_fact')
        add(f'A {a} is a {b}. A {c} can fly. What can a {a} do?',
            'I do not know.', 'missing_link')
    a, b, c = name(), name(), name()
    add(f'A {a} is a {b}. A {b} is a {c}. A {c} can fly. What can a {a} do?',
        f'{a.capitalize()} can fly.', 'conversation')
    add(f'A {c} can swim. What can a {a} do?', f'{a.capitalize()} can swim.', 'conversation', False)
    # Explicit stress tests remain failures until the syntax itself generalises.
    add(f'{a} belongs to {b}. {b} can fly. What can a {a} do?',
        f'{a.capitalize()} can fly.', 'paraphrase_stress')
    add(f'A {a} is a {b}. A {b} cannot fly. What can a {a} do?',
        f'{a.capitalize()} cannot fly.', 'negation_stress')
    return dict(name='graph-composition', status=status, seed=seed, cases=cases,
                objective='Nearest available property along directed class links; latest assertion within an entity; reject unrelated facts.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--status', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit('Refusing to overwrite a frozen suite')
    suite = generate(a.seed, a.status)
    a.output.write_text(json.dumps(suite, indent=2) + '\n', encoding='utf-8')
    print(f"Wrote {len(suite['cases'])} graph cases")


if __name__ == '__main__':
    main()
