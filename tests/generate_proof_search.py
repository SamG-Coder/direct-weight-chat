"""External tests for alternative ground proofs, independent of rule order.

All expected closure is external. Positive proof cases have consistent derived
signs; explicit facts take precedence. Unsupported compositions remain stress
failures instead of being relabeled as successful abstention.
"""
import argparse
import itertools
import json
import random
from pathlib import Path

from .generate_conditionals import assertion, question, solve


def generate(seed, status):
    rng = random.Random(seed)
    cases = []

    def name():
        return rng.choice(('vel', 'om')) + ''.join(rng.choice('bcdfghklmnprstvwz') for _ in range(7))

    def literal(relation=None):
        relation = rng.randrange(7) if relation is None else relation
        obj = rng.choice(('cold', 'hot', 'small', 'green')) if relation == 5 else name()
        return (name(), relation, obj, bool(rng.randrange(2)))

    def add(facts, rules, query, category, offset=0, facts_first=False, expected=None):
        fs = [assertion(f)+'.' for f in facts]
        rs = [f'If {assertion(p)} then {assertion(q)}.' for p,q in rules]
        prompt = 'Hello. '*(offset//2) + ' '.join((fs+rs if facts_first else rs+fs) + [question(query)])
        cases.append(dict(id=f'proof-search-{len(cases):03}', prompt=prompt,
                          expected=solve(facts,rules,query) if expected is None else expected,
                          category=category, reset=True, rule_count=len(rules), padding_tokens=offset))

    for count in (2,3,4,6,8):
        for winner in range(count):
            premises = [literal() for _ in range(count)]
            conclusion = literal()
            facts = [p if i == winner else p[:3]+(not p[3],) for i,p in enumerate(premises)]
            query = conclusion[:3] + (bool(rng.randrange(2)),)
            add(facts, [(p,conclusion) for p in premises], query, f'{count}_alternatives_winner_{winner}', facts_first=bool(winner%2))
        premises, conclusion = [literal() for _ in range(count)], literal()
        rules = [(p,conclusion) for p in premises]
        add([],rules,conclusion,'all_premises_missing')
        add([p[:3]+(not p[3],) for p in premises],rules,conclusion,'all_premises_opposite')
        add(premises+[conclusion[:3]+(not conclusion[3],)],rules,conclusion,'direct_fact_overrides_proofs')

    # The satisfied branch may be older at one step and newer at another.
    # A single earliest/latest path is therefore insufficient for this family.
    for branching in (2,3):
        for depth in (2,3):
            for choices in itertools.product(range(branching),repeat=depth):
                path = [literal(rng.choice((1,2,3,4))) for _ in range(depth+1)]
                rules = []
                for level, selected in enumerate(choices):
                    for branch in range(branching):
                        premise = path[level] if branch == selected else literal()
                        rules.append((premise,path[level+1]))
                add(path[:1],rules,path[-1],f'mixed_path_depth_{depth}_branching_{branching}')

    for offset in (0,1024,3072):
        path = [literal(1),literal(3),literal(2),literal(4)]
        rules = list(zip(path,path[1:])) + [(literal(),path[i]) for i in (1,2,3)]
        for _ in range(3):
            rng.shuffle(rules)
            add(path[:1],rules,path[-1],'shuffled_branched_chain',offset)

    p,q,target = literal(1),literal(2),literal(3)
    rules = [(p,target),(q,target)]
    add([],rules,target,'dialogue_alternatives')
    known = []
    for fact in (p,q,p[:3]+(not p[3],),q[:3]+(not q[3],),p):
        known.append(fact)
        cases.append(dict(id=f'proof-search-{len(cases):03}',
                          prompt=assertion(fact)+'. '+question(target),
                          expected=solve(known,rules,target), category='dialogue_alternatives',reset=False))

    for depth in (4,5):
        path = [literal(1 if i%2 == 0 else 3) for i in range(depth+1)]
        add(path[:1],list(zip(path,path[1:])),path[-1],f'depth_{depth}_stress')
    # A class edge derived by a rule should compose with a known property.
    a,b,action = name(),name(),name()
    trigger = literal(2)
    membership = (a,0,b,False)
    add([trigger,(b,1,action,False)],[(trigger,membership)],(a,1,action,False),
        'derived_class_inheritance_stress',expected='Yes.')
    return dict(name='alternative-ground-proof-search',seed=seed,status=status,cases=cases,
                objective='Find any supported signed ground proof among multiple alternatives, including mixed branches, declaration order, corrections and facts after rules. Depth and inheritance composition stresses remain visible.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--status',required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit('Refusing to overwrite evidence')
    suite=generate(a.seed,a.status)
    a.output.write_text(json.dumps(suite,indent=2)+'\n',encoding='utf-8')
    print(f"Wrote {len(suite['cases'])} alternative-proof cases")


if __name__ == '__main__':
    main()
