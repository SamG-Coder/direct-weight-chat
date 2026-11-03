"""External position/length tests across the existing controlled capabilities.

Padding changes where the same reusable operations execute; expected values
and arithmetic are computed only here, never by the inference runtime.
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

    def padding(count):
        return 'Hello. ' * (count // 2)

    def add(prompt, expected, category, pad):
        cases.append(dict(id=f'context-{len(cases):03}', prompt=prompt, expected=expected,
                          category=category, padding_tokens=pad, reset=True))

    for length in (0, 32, 128, 384, 768, 1024, 1536, 2048, 3072, 4096, 4700, 4900):
        x, y, z, w = [name() for _ in range(4)]
        a, b = [rng.randint(-999, 999) for _ in range(2)]
        pad = padding(length)
        add(pad + 'Please explain a bird.', 'Bird is an animal. It can fly.', 'static_generation', length)
        add(pad + f'Please calculate {a} - {b}.', f'{a-b}.', 'signed_arithmetic', length)
        add(pad + f'A {x} can fly and swim. Can {x} swim?', 'Yes.', 'late_compound_fact', length)
        add(f'A {x} can fly and swim. ' + pad + f'Can {x} swim?', 'Yes.', 'distant_compound_fact', length)
        add(pad + f'Can {x} glimmer?', 'I do not know.', 'unknown_generation', length)
        add(pad + f'A {x} does not need water and fuel. Does {x} not need fuel?', 'Yes.', 'negative_fact_and_query', length)
        add(pad + f'A {x} has wings and a {y} needs water and fuel. Does {y} need fuel?', 'Yes.', 'distinct_subject_binding', length)
        add(f'A {x} is a {y}. A {y} is a {z}. A {z} is a {w}. A {w} can glimmer. ' + pad + f'Can {x} glimmer?', 'Yes.', 'distant_three_hop_inheritance', length)

    for _ in range(12):
        length = 2 * rng.randrange(0, 2451)
        x, y = name(), name()
        pad = padding(length)
        add(f'A {x} can fly. ' + pad + f'A {x} cannot fly. Can {x} fly?', 'No.', 'distant_correction', length)
        add(f'A {x} cannot fly. ' + pad + f'Can {x} fly? Can {y} swim and glimmer?', 'I do not know.', 'active_question_scope', length)

    # These short requests plus a 40-token generation allowance still fit the
    # declared 5,048-position window, including BOS/user/assistant delimiters.
    for length in (4980, 4990, 5000):
        add(padding(length) + 'Hello.', 'Hello.', 'near_capacity_greeting', length)
        add(padding(length) + 'Explain a bird.', 'Bird is an animal. It can fly.', 'near_capacity_generation', length)
    return dict(name='context-position-and-generation', seed=seed, status=status, cases=cases,
                objective='Preserve existing semantic operations and complete generation under position shifts up to the declared context capacity, without reducing the window or changing the host runtime.')


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
    print(f"Wrote {len(suite['cases'])} context/position cases")


if __name__ == '__main__':
    main()
