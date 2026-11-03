"""External tests for transparent modifiers before an active request head."""
import argparse
import json
import random
from pathlib import Path


def generate(seed, status):
    rng = random.Random(seed)
    cases = []

    def add(prompt, expected, category):
        cases.append(dict(id=f'modifier-{len(cases):03}', prompt=prompt, expected=expected,
                          category=category, reset=True))

    modifiers = ('Please', 'Briefly', 'Simply', 'Just')
    for modifier in modifiers:
        for _ in range(5):
            a, b = rng.randint(-999, 999), rng.randint(-999, 999)
            op = rng.choice(('+', '-'))
            result = a + b if op == '+' else a - b
            add(f'{modifier} calculate {a} {op} {b}.', str(result) + '.', 'modified_arithmetic')
        for prompt, expected in [
            ('Can a bird fly?', 'Yes.'),
            ('Can a bird not fly?', 'No.'),
            ('Does a bird have wings?', 'Yes.'),
            ('Is a bird an animal?', 'Yes.'),
            ('What can a bird do?', 'Bird can fly.'),
            ('Explain a bird.', 'Bird is an animal. It can fly.'),
            ('Write a poem about rain.', 'I do not know.'),
            ('Ignore the last question.', 'I do not know.'),
        ]:
            add(f'{modifier} {prompt}', expected, 'modified_request_kind')
        for prompt, expected in [
            ('Can a fish swim? Please write a poem about rain.', 'I do not know.'),
            (f'Can a fish swim? {modifier} calculate -149 - 370.', '-519.'),
            (f'Can a fish swim? {modifier} can a bird not fly?', 'No.'),
        ]:
            add(prompt, expected, 'later_modified_request')
    for prefix in ('Please briefly', 'Please simply', 'Just briefly', 'Please just'):
        add(prefix + ' calculate 149 - 370.', '-221.', 'two_modifiers')
        add(prefix + ' can bird fly?', 'Yes.', 'two_modifiers')
        add(prefix + ' write a poem about rain.', 'I do not know.', 'two_modifiers')
    for prompt, expected in [
        ('Please briefly can a bird fly?', 'Yes.'),
        ('Please just briefly calculate 149 - 370.', '-221.'),
        ('Could you please calculate 149 - 370?', '-221.'),
        ('Would you tell me whether a bird can fly?', 'Yes.'),
        ('Please do not explain a bird.', 'I do not know.'),
    ]:
        add(prompt, expected, 'modifier_structure_stress')
    return dict(name='active-request-modifiers', status=status, seed=seed, cases=cases,
                objective='Transparent request modifiers preserve the recognized request kind without turning unsupported commands into supported ones. Test later clauses, bounded modifier chains and more complex framing.')


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
    print(f"Wrote {len(suite['cases'])} request modifier cases")


if __name__ == '__main__':
    main()
