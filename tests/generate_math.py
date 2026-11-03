"""External arithmetic tests; answers are calculated only by the evaluator."""
import argparse
import json
import random
from pathlib import Path


def generate(seed, status):
    rng = random.Random(seed)
    cases = []
    def add(prompt, answer, category, reset=True):
        cases.append(dict(id=f'math-{len(cases):03}', prompt=prompt, expected=f'{answer}.', category=category, reset=reset))
    for a, b in [(0, 0), (0, 999), (999, 0), (999, 999), (1, 2), (9, 10), (100, 99), (500, 501)]:
        add(f'{a} - {b}', a - b, 'subtraction_boundary')
    for i in range(80):
        a, b = rng.randrange(1000), rng.randrange(1000)
        prompt = rng.choice([f'{a}-{b}', f'{a} - {b}', f'Calculate {a} - {b}.', f'Please calculate {a} - {b}.'])
        add(prompt, a - b, 'subtraction_random')
    for i in range(20):
        a, b = rng.randrange(1000), rng.randrange(1000)
        add(f'{a} + {b}', a + b, 'addition_regression')
    for i, (a, op, b) in enumerate([(17, '-', 48), (125, '+', 456), (999, '-', 999), (0, '-', 999), (12, '+', 80), (81, '-', 12)]):
        add(f'{a} {op} {b}', a + b if op == '+' else a - b, 'mixed_conversation', reset=i == 0)
    add('1000 - 1', 999, 'range_stress')
    add('-5 - 3', -8, 'negative_operand_stress')
    add('17 * 24', 408, 'multiplication_stress')
    add('81 / 9', 9, 'division_stress')
    add('10 - 3 - 2', 5, 'multi_step_stress')
    add('Subtract 3 from 8.', 5, 'paraphrase_stress')
    return dict(name='signed-subtraction', status=status, seed=seed, cases=cases,
                objective='Two nonnegative integer operands at most 999; signed results; preserve addition and per-turn operator selection. Unsupported operations remain stress failures.')


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
    print(f"Wrote {len(suite['cases'])} math cases")


if __name__ == '__main__':
    main()
