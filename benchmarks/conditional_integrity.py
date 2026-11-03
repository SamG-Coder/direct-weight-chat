"""External conditional-circuit interventions; never imported by inference."""
import argparse
import json
from pathlib import Path

from .evaluate import digest, git_state
from chat1gb_autoregressive.engine import Model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--version', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit('Refusing to overwrite evidence')
    initial = digest(a.model / 'weights.f32')
    m = Model(a.model, device=a.device)
    checks = []

    def check(name, prompt, expected, ablated_expected, changes):
        m.reset()
        normal = m.generate(prompt)
        saved = {}
        try:
            for tensor, rows, columns in changes:
                if tensor not in saved:
                    saved[tensor] = m.w[tensor]
                    m.w[tensor] = saved[tensor].clone()
                m.w[tensor][rows, columns] = 0
            m.reset()
            altered = m.generate(prompt)
        finally:
            m.w.update(saved)
        checks.append(dict(name=name, original=normal, ablated=altered,
                           expected=expected, ablated_expected=ablated_expected,
                           passed=normal['text'] == expected and altered['text'] == ablated_expected))

    chain = ('Kelv can shimmer. If kelv can shimmer then zor has fins. '
             'If zor has fins then tal needs water. If tal needs water then pel can glow. ')
    for layer, query in ((4, 'Does zor have fins?'), (6, 'Does tal need water?'), (8, 'Can pel glow?')):
        check(f'rule_link_{(layer-2)//2}', chain + query, 'Yes.', 'I do not know.',
              [(f'blocks.{layer}.attn.out.weight', slice(None), slice(10*64, 11*64))])
    check('premise_evidence_attention', chain + 'Does zor have fins?', 'Yes.', 'I do not know.',
          [(f'blocks.{layer}.attn.out.weight', slice(None), slice(7*64, 15*64)) for layer in (5, 7, 9)])
    check('rule_record_validity', chain + 'Does zor have fins?', 'Yes.', 'I do not know.',
          [('blocks.2.mlp.down.weight', slice(350, 352), slice(None))])
    check('consequent_object_identity', chain + 'Does zor have scales?', 'I do not know.', 'Yes.',
          [(f'blocks.{layer}.attn.qkv.weight', slice(10*64+14, 10*64+28), slice(None)) for layer in (4, 6, 8)])
    check('signed_premise', 'Kelv can shimmer. If kelv cannot shimmer then zor has fins. Does zor have fins?',
          'I do not know.', 'Yes.',
          [(f'blocks.{layer}.mlp.down.weight', slice(646, 648), slice(None)) for layer in (4, 6, 8)])

    m.reset()
    s = m.config['special']
    prefix = [s['bos'], s['user']] + m.tokenizer.encode(chain + 'Can pel glow?') + [s['assistant']]
    logits, cache, _ = m.forward(prefix)
    generated = []
    cache_matches = True
    for _ in range(12):
        full, _, _ = m.forward(prefix + generated)
        token = int(logits[-1].argmax().item())
        cache_matches &= token == int(full[-1].argmax().item())
        generated.append(token)
        if token == s['eos']:
            break
        logits, cache, _ = m.forward([token], cache)
    result = dict(version=a.version, device=m.device, git=git_state(), model_sha256=initial,
                  interventions=checks, cached_full_tokens_match=cache_matches,
                  cached_generated_tokens=generated, weights_unchanged=digest(a.model / 'weights.f32') == initial,
                  note='Only temporary runtime tensor copies are ablated; saved weights and the normal generic inference code are unchanged. Expected text is used only for external scoring.')
    a.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(passed=sum(c['passed'] for c in checks), total=len(checks), cache_matches=cache_matches)))
    assert all(c['passed'] for c in checks) and cache_matches and result['weights_unchanged']


if __name__ == '__main__':
    main()
