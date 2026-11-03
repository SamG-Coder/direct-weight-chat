"""External tests of multi-rule inference under position/order changes.

Expected closure is computed only by the external suite generator. The model
receives plain prompts, including unfamiliar names, and never receives labels.
"""
import argparse
import json
import random
from pathlib import Path

from .generate_conditionals import assertion, question, solve


def generate(seed, status):
    rng = random.Random(seed)
    cases = []

    def name():
        return rng.choice(('vel', 'om')) + ''.join(rng.choice('bcdfghklmnprstvwz') for _ in range(7))

    def literal(relation, negative=False):
        return (name(), relation, name(), negative)

    def add(facts, rules, query, category, offset, ordering='forward'):
        ordered = list(rules)
        if ordering == 'reverse':
            ordered.reverse()
        elif ordering == 'shuffled':
            rng.shuffle(ordered)
        rule_text = [f'If {assertion(p)} then {assertion(q)}.' for p, q in ordered]
        fact_text = [assertion(f) + '.' for f in facts]
        prefix = 'Hello. ' * (offset // 2)
        clauses = rule_text + fact_text if ordering != 'forward' else fact_text + rule_text
        cases.append(dict(id=f'rule-position-{len(cases):03}', prompt=prefix + ' '.join(clauses + [question(query)]),
                          expected=solve(facts, rules, query), category=category,
                          requested_padding_tokens=offset, ordering=ordering, reset=True))

    fixed = (0, 32, 128, 512, 1024, 2048, 3072, 3328, 4096, 4608, 4832)
    offsets = list(fixed) + [2*rng.randrange(16, 2417) for _ in range(8)]
    for offset in offsets:
        # All signs and identities vary independently of absolute position.
        chain = [literal(r, bool(rng.randrange(2))) for r in (1, 3, 2, 4)]
        rules = list(zip(chain, chain[1:]))
        for depth in (1, 2, 3):
            add(chain[:1], rules[:depth], chain[depth], f'chain_depth_{depth}', offset)
        add(chain[:1], rules, chain[-1], 'reverse_rule_order', offset, 'reverse')
        add(chain[:1], rules, chain[-1], 'shuffled_rules_before_fact', offset, 'shuffled')
        add([], rules, chain[-1], 'missing_premise', offset)
        add([chain[0][:3] + (not chain[0][3],)], rules, chain[-1], 'opposite_premise', offset)
        add([chain[0], chain[0][:3] + (not chain[0][3],)], rules, chain[-1], 'premise_correction', offset, 'reverse')

    for depth in (2, 3):
        offset = 3328
        chain = [literal((1, 3, 2, 4)[i]) for i in range(depth + 1)]
        rules = list(zip(chain, chain[1:]))
        add([], rules, chain[-1], 'dialogue_rule_before_fact', offset)
        for negative, expected in ((False, 'Yes.'), (True, 'I do not know.'), (False, 'Yes.')):
            fact = chain[0][:3] + (negative,)
            cases.append(dict(id=f'rule-position-{len(cases):03}',
                              prompt=assertion(fact) + '. ' + question(chain[-1]), expected=expected,
                              category='dialogue_correction', reset=False))

    # Existing semantic limits remain stress failures; they are not silently
    # relabeled as abstention successes when measuring numerical reliability.
    chain = [literal(1 if i % 2 == 0 else 3) for i in range(5)]
    add(chain[:1], list(zip(chain, chain[1:])), chain[-1], 'four_link_stress', 4096)
    alternative = literal(2)
    add(chain[:1], [(chain[0], chain[1]), (alternative, chain[1])], chain[1], 'older_alternative_stress', 4096)
    return dict(name='position-varied-ground-rule-inference', seed=seed, status=status, cases=cases,
                fixed_offsets=list(fixed), random_offsets=offsets[len(fixed):],
                objective='Keep one-to-three-link signed ground inference invariant under rule ordering, long prefixes and dialogue corrections; retain deeper/alternative-proof limitations. Fresh seeds use known structures and are not independent language tests.')


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
    print(f"Wrote {len(suite['cases'])} rule-position cases")


if __name__ == '__main__':
    main()
