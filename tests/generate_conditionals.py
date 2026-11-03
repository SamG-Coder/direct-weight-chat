"""External ground-implication tests. No oracle or examples enter inference.

The bounded target is explicit-subject, single-premise implications. Tests also
retain deeper chains, alternative proofs, compound premises and paraphrases as
stress cases. Unknown is open-world: an unsatisfied premise proves neither
the conclusion nor its negation. Explicit facts override derived information.
"""
import argparse
import json
import random
from pathlib import Path


def assertion(literal):
    subject, relation, obj, negative = literal
    positive = (f'{subject} is a {obj}', f'{subject} can {obj}',
                f'{subject} needs {obj}', f'{subject} has {obj}',
                f'{subject} produces {obj}', f'{subject} is {obj}',
                f'{subject} exists in {obj}')
    negated = (f'{subject} is not a {obj}', f'{subject} cannot {obj}',
               f'{subject} does not need {obj}', f'{subject} does not have {obj}',
               f'{subject} does not produce {obj}', f'{subject} is not {obj}',
               f'{subject} is not in {obj}')
    return (negated if negative else positive)[relation]


def question(literal):
    subject, relation, obj, negative = literal
    n = 'not ' if negative else ''
    return (f'Is {subject} {n}a {obj}?', f'Can {subject} {n}{obj}?',
            f'Does {subject} {n}need {obj}?', f'Does {subject} {n}have {obj}?',
            f'Does {subject} {n}produce {obj}?', f'Is {subject} {n}{obj}?',
            f'Is {subject} {n}in {obj}?')[relation]


def solve(facts, rules, query):
    direct = {tuple(f[:3]): bool(f[3]) for f in facts}
    known = dict(direct)
    # Finite ground closure is an external scoring oracle, not model code.
    for _ in range(len(rules) + 1):
        changed = False
        for premise, conclusion in rules:
            key = tuple(conclusion[:3])
            if known.get(tuple(premise[:3])) == premise[3] and key not in known:
                known[key] = bool(conclusion[3])
                changed = True
        if not changed:
            break
    value = known.get(tuple(query[:3]))
    return 'I do not know.' if value is None else 'Yes.' if value == query[3] else 'No.'


def generate(seed, status):
    rng = random.Random(seed)
    cases = []

    def name():
        return rng.choice(('vel', 'om')) + ''.join(rng.choice('bcdfghklmnprstvwz') for _ in range(7))

    def literal(relation=None, negative=False):
        r = rng.randrange(7) if relation is None else relation
        return (name(), r, rng.choice(('cold', 'green', 'small', 'hot')) if r == 5 else name(), negative)

    def rule(p, q):
        return f'If {assertion(p)} then {assertion(q)}.'

    def add(facts, rules, query, category, reverse=False, prefix='', extra='', expected=None):
        fs = [assertion(f) + '.' for f in facts]
        rs = [rule(*r) for r in rules]
        prompt = prefix + ' '.join((rs + fs if reverse else fs + rs) + [extra, question(query)]).replace('  ', ' ')
        cases.append(dict(id=f'conditional-{len(cases):03}', prompt=prompt,
                          expected=solve(facts, rules, query) if expected is None else expected,
                          category=category, reset=True))

    for a in range(7):
        for b in range(7):
            p, q = literal(a), literal(b)
            add([p], [(p, q)], q, 'single_rule_all_predicate_pairs', reverse=bool((a+b) % 2))
            p, q = literal(a, True), literal(b, bool((a+b) % 2))
            add([p], [(p, q)], q[:3] + (not q[3],), 'signed_premise_conclusion_query', reverse=True)
            add([], [(p, q)], q, 'missing_premise')
            add([p[:3] + (not p[3],)], [(p, q)], q, 'false_premise')

    for _ in range(7):
        p, q, unrelated = literal(), literal(), literal()
        add([], [(p, q)], p, 'rule_does_not_assert_premise')
        add([p], [(p, q)], unrelated, 'conclusion_identity_isolation')
        add([p, p[:3] + (not p[3],)], [(p, q)], q, 'premise_correction', reverse=True)
        add([p, q[:3] + (not q[3],)], [(p, q)], q, 'direct_fact_precedence')

    for depth in range(1, 9):
        literals = [literal(1 if i % 2 == 0 else 3, bool(rng.randrange(2))) for i in range(depth+1)]
        rules = list(zip(literals, literals[1:]))
        for ordering in ('forward', 'reverse', 'shuffled'):
            ordered = list(rules)
            if ordering == 'reverse':
                ordered.reverse()
            if ordering == 'shuffled':
                rng.shuffle(ordered)
            add(literals[:1], ordered, literals[-1], f'chain_depth_{depth}', reverse=ordering != 'forward')

    for pad in (0, 128, 1024, 3072):
        p, q = literal(1), literal(3)
        add([p], [(p, q)], q, 'position_shift', prefix='Hello. ' * (pad // 2))
        add([], [(p, q)], q, 'position_shift_missing', prefix='Hello. ' * (pad // 2))

    for _ in range(6):
        p, q, alternative = literal(1), literal(3), literal(2)
        add([p], [(p, q), (alternative, q)], q, 'alternative_proof_older_satisfied')
        add([p], [(alternative, q), (p, q)], q, 'alternative_proof_newer_satisfied')
        add([], [(p, q), (q, p)], q, 'unfounded_cycle')
        add([q], [(p, q)], p, 'no_converse')
        add([q[:3] + (True,)], [(p, q)], p, 'no_contraposition')

    p, q, r = literal(1), literal(3), literal(2)
    compound = f'If {assertion(p)} and {assertion(r)} then {assertion(q)}.'
    add([p], [], q, 'compound_premise_missing', extra=compound, expected='I do not know.')
    add([p, r], [], q, 'compound_premise_satisfied', extra=compound, expected='Yes.')
    add([p], [], q, 'disjunctive_premise_satisfied', extra=compound.replace(' and ', ' or '), expected='Yes.')
    add([p], [], q, 'paraphrase_implies', extra=f'{assertion(p)} implies {assertion(q)}.', expected='Yes.')
    add([p], [], q, 'paraphrase_whenever', extra=f'Whenever {assertion(p)}, {assertion(q)}.', expected='Yes.')
    add([], [], q, 'conditional_question_not_assertion', extra=f'If {assertion(p)} then {assertion(q)}?', expected='I do not know.')
    add([p], [(p, q)], q, 'rule_without_period', expected='Yes.')
    cases[-1]['prompt'] = cases[-1]['prompt'].replace(rule(p, q), rule(p, q)[:-1])
    return dict(name='ground-conditional-reasoning', seed=seed, status=status, cases=cases,
                objective='Apply explicit ground implications only when a signed premise is known; retain scope, polarity, corrections and stress failures. Fresh seeds from this author-designed generator are not independent language generalization.')


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
    print(f"Wrote {len(suite['cases'])} conditional cases")


if __name__ == '__main__':
    main()
