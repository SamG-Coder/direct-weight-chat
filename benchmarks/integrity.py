"""Check artifact integrity, causal execution, cache parity and neural ablations."""
import argparse
import json
import socket
import sys
from pathlib import Path

from .evaluate import digest, git_state
from chat1gb_autoregressive.engine import Model
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--arithmetic', action='store_true', help='Also ablate the v7 signed arithmetic accumulator')
    p.add_argument('--polarity', action='store_true', help='Also ablate the v8 fact-polarity register')
    p.add_argument('--arithmetic-layer', type=int, choices=[10, 11], default=11)
    p.add_argument('--signed-operands', action='store_true', help='Ablate unary sign and operand-count validity in v9')
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit('Refusing to overwrite evidence')
    manifest = json.loads((a.model / 'manifest.json').read_text(encoding='utf-8'))
    for file, expected in manifest['sha256'].items():
        assert digest(a.model / file) == expected, file
    for file, expected in manifest['source_sha256'].items():
        assert digest(a.source / file) == expected, file
    m = Model(a.model, device=a.device)
    assert not any(n.endswith(('.knowledge', '.build', '.program')) for n in sys.modules)
    checks = dict(file_and_source_hashes=True, runtime_imports_no_compiler=True,
                  parameter_count=sum(t.numel() for t in m.w.values()),
                  weight_bytes=(a.model / 'weights.f32').stat().st_size)
    checks['nonzero_parameters'] = int(torch.count_nonzero(m.storage).item())
    checks['finite_weights'] = bool(torch.isfinite(m.storage).all().item())
    assert checks['nonzero_parameters'] == manifest['nonzero_parameters']
    assert checks['parameter_count'] == 250_000_000 and checks['weight_bytes'] == 1_000_000_000
    assert checks['finite_weights']
    with socket.socket() as connection:
        try:
            connection.connect(('127.0.0.1', 9))
        except PermissionError:
            checks['socket_audit_guard'] = True
        else:
            raise AssertionError('Network guard missing')
    s = m.config['special']
    common = [s['bos'], s['user']] + m.tokenizer.encode('A lavrik is a pelvon.')
    left = common + m.tokenizer.encode('A pelvon can pivot.') + [s['assistant']]
    right = common + m.tokenizer.encode('A pelvon can shimmer.') + [s['assistant']]
    x, _, _ = m.forward(left)
    y, _, _ = m.forward(right)
    checks['causal_prefix_max_logit_difference'] = float((x[:len(common)] - y[:len(common)]).abs().max().item())
    assert checks['causal_prefix_max_logit_difference'] < .01
    m.reset()
    prompt = 'A lavrik is a pelvon. A pelvon is a nebvik. A nebvik can shimmer. What can a lavrik do?'
    prefix = [s['bos'], s['user']] + m.tokenizer.encode(prompt) + [s['assistant']]
    logits, cache, _ = m.forward(prefix)
    generated = []
    for _ in range(20):
        full, _, _ = m.forward(prefix + generated)
        token = int(logits[-1].argmax().item())
        assert token == int(full[-1].argmax().item())
        generated.append(token)
        if token == s['eos']:
            break
        logits, cache, _ = m.forward([token], cache)
    checks['cached_and_full_prefix_tokens_match'] = True
    m.reset()
    normal = m.generate(prompt)
    saved = {}
    try:
        for layer in (5, 7, 9):
            name = f'blocks.{layer}.attn.out.weight'
            saved[name] = m.w[name]
            m.w[name] = torch.zeros_like(m.w[name])
        m.reset()
        ablated = m.generate(prompt)
    finally:
        m.w.update(saved)
    checks['multi_hop_ablation'] = dict(prompt=prompt, original=normal['text'], ablated=ablated['text'],
                                        changed=normal['text'] != ablated['text'], layers=[5, 7, 9])
    assert normal['text'] == 'Lavrik can shimmer.' and ablated['text'] != normal['text']
    saved_head = m.w['lm_head.weight']
    try:
        m.w['lm_head.weight'] = torch.zeros_like(saved_head)
        m.reset()
        checks['output_head_ablation_removes_text'] = m.generate('Hello', 4)['text'] == ''
    finally:
        m.w['lm_head.weight'] = saved_head
    assert checks['output_head_ablation_removes_text']
    if a.polarity:
        probes = [
            ('A kelv cannot fly. What can a kelv do?', 'Kelv cannot fly.'),
            ('A kelv is not a bird. What can a kelv do?', 'I do not know.'),
        ]
        checks['polarity_ablation'] = []
        for prompt, expected in probes:
            m.reset()
            original = m.generate(prompt)
            name = 'blocks.2.mlp.down.weight'
            saved = m.w[name]
            try:
                m.w[name] = saved.clone()
                m.w[name][952:954] = 0  # Fact polarity is logical feature 476.
                m.reset()
                ablated = m.generate(prompt)
            finally:
                m.w[name] = saved
            assert original['text'] == expected and ablated['text'] != expected
            checks['polarity_ablation'].append(dict(prompt=prompt, original=original['text'], ablated=ablated['text']))
    if a.arithmetic:
        m.reset()
        original_math = m.generate('149 - 370')
        # Logical accumulator 5 is paired in physical rows 10 and 11. This is
        # a diagnostic intervention, not an alternate inference path.
        name = f'blocks.{a.arithmetic_layer}.mlp.down.weight'
        saved_math = m.w[name]
        try:
            m.w[name] = saved_math.clone()
            m.w[name][10:12] = 0
            m.reset()
            ablated_math = m.generate('149 - 370')
        finally:
            m.w[name] = saved_math
        checks['arithmetic_ablation'] = dict(prompt='149 - 370', original=original_math['text'],
                                            ablated=ablated_math['text'])
        assert original_math['text'] == '-221.' and ablated_math['text'] != original_math['text']
    if a.signed_operands:
        m.reset()
        original_sign = m.generate('-149 - 370')
        name = 'blocks.0.mlp.down.weight'
        saved = m.w[name]
        try:
            m.w[name] = saved.clone()
            m.w[name][48:50] = 0  # Unary sign writes the numeric-value feature 24.
            m.reset()
            ablated_sign = m.generate('-149 - 370')
        finally:
            m.w[name] = saved
        assert original_sign['text'] == '-519.' and ablated_sign['text'] == '-221.'
        checks['unary_sign_ablation'] = dict(original=original_sign['text'], ablated=ablated_sign['text'])
        m.reset()
        original_arity = m.generate('10 - 3 - 2')
        weight_name, bias_name = 'blocks.10.mlp.down.weight', 'blocks.10.mlp.down.bias'
        saved_w, saved_b = m.w[weight_name], m.w[bias_name]
        try:
            m.w[weight_name], m.w[bias_name] = saved_w.clone(), saved_b.clone()
            m.w[weight_name][936:938] = 0  # Validity feature 468 at decode time.
            m.w[bias_name][936], m.w[bias_name][937] = 1, -1
            m.reset()
            forced_arity = m.generate('10 - 3 - 2')
        finally:
            m.w[weight_name], m.w[bias_name] = saved_w, saved_b
        assert original_arity['text'] == 'I do not know.' and forced_arity['text'] == '8.'
        checks['arity_gate_forced_valid'] = dict(original=original_arity['text'], forced_valid=forced_arity['text'])
    checks['weight_file_unchanged_after_ablations'] = digest(a.model / 'weights.f32') == manifest['sha256']['weights.f32']
    assert checks['weight_file_unchanged_after_ablations']
    result = dict(git=git_state(), model_sha256=manifest['sha256']['weights.f32'], device=m.device, checks=checks,
                  note='Ablations replace selected runtime tensor references with zero copies, then restore them. The saved checkpoint is never modified.')
    a.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
