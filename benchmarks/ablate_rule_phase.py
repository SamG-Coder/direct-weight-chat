"""Restore the old unsaturated context gate in temporary runtime weights.

This intervention is external to inference. All other candidate parameters
remain unchanged, and the saved checkpoint is verified unchanged afterward.
"""
import argparse
import json
from pathlib import Path

from .evaluate import digest, git_state
from chat1gb_autoregressive.engine import Model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--version', required=True)
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--case-id', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--require-change', action='store_true')
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit('Refusing to overwrite evidence')
    suite = json.loads(a.suite.read_text(encoding='utf-8'))
    case = next(c for c in suite['cases'] if c['id'] == a.case_id)
    if not case.get('reset', True):
        raise SystemExit('This isolated intervention requires a reset case')
    m = Model(a.model, device=a.device)
    initial = digest(a.model/'weights.f32')
    normal = m.generate(case['prompt'], case.get('max_new_tokens', 40))
    name = 'blocks.1.mlp.up.weight'
    saved = m.w[name]
    down = m.w['blocks.1.mlp.down.weight']
    # Paired readers store half each logical coefficient. The new Boolean
    # context units read IF_ACTIVE and USER_STATE with coefficient 4.
    candidates = ((saved[:,988] == 2) & (saved[:,104] == 2) & (down[112].abs() == 1)).nonzero().flatten().tolist()
    assert len(candidates) == 2, candidates
    positive = next(i for i in candidates if down[112,i] == 1)
    for row in candidates:
        assert down[:,row].nonzero().flatten().tolist() == [112,113]
    try:
        m.w[name] = saved.clone()
        m.w[name][candidates,:] = 0
        # Restore ReLU(2*IF_ACTIVE + 2*USER_STATE - 3), the v16c gate.
        for coordinate, coefficient in ((494,2), (52,2), (1,-3)):
            m.w[name][positive,coordinate*2] = coefficient/2
            m.w[name][positive,coordinate*2+1] = -coefficient/2
        m.reset()
        ablated = m.generate(case['prompt'], case.get('max_new_tokens', 40))
    finally:
        m.w[name] = saved
    result = dict(version=a.version, device=m.device, git=git_state(), model_sha256=initial,
                  suite_sha256=digest(a.suite), case_id=a.case_id, expected=case['expected'],
                  normal=normal, unsaturated_context_gate=ablated,
                  changed=normal['text'] != ablated['text'],
                  normal_correct=normal['text'] == case['expected'] and normal['stopped_on_eos'],
                  ablated_correct=ablated['text'] == case['expected'] and ablated['stopped_on_eos'],
                  altered_up_rows=candidates, weights_unchanged=digest(a.model/'weights.f32') == initial,
                  note='Only the two current-rule-context MLP units are temporarily replaced by the old unsaturated function. Tokenization, full tensor execution and token decisions remain in the normal model.')
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('version','device','changed','normal_correct','ablated_correct')}))
    assert result['normal_correct'] and result['weights_unchanged']
    if a.require_change:
        assert result['changed'] and not result['ablated_correct']


if __name__ == '__main__':
    main()
