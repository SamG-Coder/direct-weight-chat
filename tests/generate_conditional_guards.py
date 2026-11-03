"""Additional author-designed diagnostics for conditional scope and dialogue.

Written after the first candidate; these are development checks, not held-out
evidence. Positive unsupported paraphrases need negative controls to distinguish
inference from accidental unconditional assertion.
"""
import json
from pathlib import Path


def main():
    path = Path('tests/conditional-guards-development.json')
    if path.exists():
        raise SystemExit('Refusing to overwrite evidence')
    cases = []

    def add(prompt, expected, category, reset=True):
        cases.append(dict(id=f'rule-guard-{len(cases):03}', prompt=prompt,
                          expected=expected, category=category, reset=reset))

    for context, expected in (('Bird can fly.', 'Yes.'), ('Bird cannot fly.', 'I do not know.'), ('', 'Yes.')):
        add(context + ' If bird can fly then kelv has fins. Does kelv have fins?', expected, 'atomic_premise_and_override')
    for context in ('Kelv can fly. ', ''):
        for rule in ('If kelv can fly then zor has wings?',
                     'If kelv can fly then zor has wings and tal has scales.',
                     'If kelv can fly or kelv can swim then zor has wings.'):
            add(context + rule + ' Does zor have wings?', 'I do not know.' if rule.endswith('?') or not context else 'Yes.', 'scope_or_unsupported_compound')
    for rule in ('Kelv can fly implies zor has wings.', 'Whenever kelv can fly, zor has wings.'):
        add(rule + ' Does zor have wings?', 'I do not know.', 'unsupported_paraphrase_missing_premise')
        add('Kelv cannot fly. ' + rule + ' Does zor have wings?', 'I do not know.', 'unsupported_paraphrase_false_premise')
    for fact, rule, query, expected in (
        ('Kelv can fly.', 'If kelv can fly then zor has wings.', 'Does zor have scales?', 'I do not know.'),
        ('Kelv can fly.', 'If kelv can fly then zor has wings.', 'Does tal have wings?', 'I do not know.'),
        ('Kelv can fly.', 'If kelv can fly then zor has wings.', 'Does zor need wings?', 'I do not know.'),
        ('Kelv can swim.', 'If kelv can fly then zor has wings.', 'Does zor have wings?', 'I do not know.'),
        ('Tal can fly.', 'If kelv can fly then zor has wings.', 'Does zor have wings?', 'I do not know.'),
        ('Kelv needs fly.', 'If kelv can fly then zor has wings.', 'Does zor have wings?', 'I do not know.'),
        ('Kelv cannot fly.', 'If kelv cannot fly then zor does not have wings.', 'Does zor have wings?', 'No.'),
        ('Kelv can fly.', 'If kelv cannot fly then zor has wings.', 'Does zor have wings?', 'I do not know.'),
    ):
        add(fact + ' ' + rule + ' ' + query, expected, 'literal_field_isolation')
    chain = 'If kelv can fly then zor has wings. If zor has wings then tal needs water. '
    add(chain + 'Kelv can fly. Does tal need water?', 'Yes.', 'multi_turn_correction')
    add('Kelv cannot fly. Does tal need water?', 'I do not know.', 'multi_turn_correction', False)
    add('Kelv can fly. Does tal need water?', 'Yes.', 'multi_turn_correction', False)
    add('Tal does not need water. Does tal need water?', 'No.', 'multi_turn_correction', False)
    add('Does zor have wings?', 'Yes.', 'multi_turn_correction', False)
    for query, expected in (('Does zor have wings and does tal need water?', 'Yes.'),
                            ('Does zor have scales or does tal need water?', 'Yes.'),
                            ('Does zor have scales and does tal need water?', 'I do not know.'),
                            ('Does zor not have wings?', 'No.')):
        add(chain + 'Kelv can fly. ' + query, expected, 'query_composition')
    for query, expected in (('What does a computer have?', 'Computer has processor.'),
                            ('Explain a computer.', 'Computer is a machine. It can calculate.'),
                            ('Explain a bird.', 'Bird is an animal. It can fly.')):
        add(chain + 'Kelv can fly. ' + query, expected, 'open_request_isolation')
    add('If kelv can fly then zor has wings. Kelv can fly. What does zor have?', 'Zor has wings.', 'open_rule_answer_stress')
    add('If kelv can fly then zor has wings. If zor has wings then kelv can fly. Does zor have wings?', 'I do not know.', 'cyclic_unknown')
    add('If kelv can fly then zor has wings. If zor has wings then kelv can fly. Kelv can fly. Does zor have wings?', 'Yes.', 'cyclic_grounded')
    suite = dict(name='conditional-scope-and-dialogue-guards', status='development-author-designed-after-v16a',
                 cases=cases, objective=__doc__)
    path.write_text(json.dumps(suite, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {len(cases)} guards')


if __name__ == '__main__':
    main()
