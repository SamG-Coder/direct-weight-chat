"""Signed-operand arithmetic and external validity stress tests."""
import argparse
import json
import random
from pathlib import Path


def generate(seed, status):
    rng = random.Random(seed)
    cases = []
    def add(prompt, expected, category, reset=True):
        cases.append(dict(id=f'signed-{len(cases):03}', prompt=prompt, expected=expected, category=category, reset=reset))
    boundaries = [(-999, '+', -999), (-999, '-', 999), (999, '-', -999), (-999, '+', 999),
                  (0, '-', -1), (-1, '+', 0), (0, '+', 0), (-1, '-', -1), (1, '-', -2), (1, '+', -2)]
    for a, op, b in boundaries:
        add(f'{a} {op} {b}', f'{a+b if op=="+" else a-b}.', 'signed_boundary')
    for _ in range(120):
        a, b = rng.randrange(-999, 1000), rng.randrange(-999, 1000)
        op = rng.choice(['+', '-'])
        prompt = rng.choice([f'{a}{op}{b}', f'{a} {op} {b}', f'Calculate {a} {op} {b}.', f'Please calculate {a} {op} {b}.'])
        add(prompt, f'{a+b if op=="+" else a-b}.', 'signed_random')
    for prompt, expected in [('5 + +3','8.'), ('5 - +3','2.'), ('+7 + +9','16.'), ('-0 - -0','0.'),
                             ('Add -3 and 4.','1.'), ('Sum -17 and -5.','-22.'), ('Add 7 and 9.','16.')]:
        add(prompt, expected, 'unary_and_prefix')
    for i, (a, op, b) in enumerate([(-5,'-',3),(-5,'+',-3),(999,'-',-999),(8,'+',7),(-3,'-',-11),(0,'+',0)]):
        add(f'{a} {op} {b}', f'{a+b if op=="+" else a-b}.', 'signed_dialogue', i==0)
    for prompt in ['10 - 3 - 2', '4 + 5 + 6', 'Calculate 7 + 8 + 9.', 'Calculate 3 + 4 and then 5.', 'Calculate 4.', 'Calculate.']:
        add(prompt, 'I do not know.', 'unsupported_arity_abstention')
    for prompt in ['-1000 + 1', '999 + -1000', '1000 - 1', '-1000 - -1000']:
        add(prompt, 'I do not know.', 'range_abstention')
    for prompt in ['5 + * 3', 'Calculate 3 4.', '5 + monkey + 3']:
        add(prompt, 'I do not know.', 'syntax_stress')
    add('17 * -24', '-408.', 'multiplication_stress')
    add('-81 / 9', '-9.', 'division_stress')
    add('(-5 + 3) - 2', '-4.', 'parentheses_stress')
    return dict(name='signed-operands', status=status, seed=seed, cases=cases,
                objective='Two signed operands with magnitude <=999; unary signs are not binary operators; retain prefix addition; reject unsupported operand counts and ranges. Syntax and other-operation failures remain visible.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--status',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise SystemExit('Refusing to overwrite a suite')
    suite=generate(a.seed,a.status)
    a.output.write_text(json.dumps(suite,indent=2)+'\n',encoding='utf-8')
    print(f"Wrote {len(suite['cases'])} signed arithmetic cases")


if __name__=='__main__': main()
