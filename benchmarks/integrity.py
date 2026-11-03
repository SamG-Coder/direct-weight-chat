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
    p.add_argument('--truth', action='store_true', help='Ablate proposition retrieval and its object identity in v10')
    p.add_argument('--query-negation', action='store_true', help='Ablate the separate v11 query-polarity bit')
    p.add_argument('--composition', action='store_true', help='Use v12 truth layout and ablate connectives, unknown truth and clause scope')
    p.add_argument('--compound-facts', action='store_true', help='Use v13 assertion layout and ablate continuation, predicate, and question/clause binding')
    p.add_argument('--deferred-assertions', action='store_true', help='Use v13c delayed-clause layout and ablate following-token validation')
    p.add_argument('--periodic-lookbacks', action='store_true', help='Ablate the two v14 positional phases and check long-prefix cache parity')
    p.add_argument('--fact-canonicalization', action='store_true', help='Test the v14b fact-coordinate saturation under a controlled magnitude perturbation')
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
            names = ['blocks.1.mlp.down.weight', 'blocks.2.mlp.down.weight'] if a.compound_facts else ['blocks.2.mlp.down.weight']
            saved = {name: m.w[name] for name in names}
            try:
                for name in names:
                    m.w[name] = saved[name].clone()
                    m.w[name][952:954] = 0  # Fact polarity is logical feature 476.
                m.reset()
                ablated = m.generate(prompt)
            finally:
                m.w.update(saved)
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
    if a.truth:
        prompt = 'A kelv can fly. A kelv cannot swim. Can a kelv fly?'
        m.reset()
        original = m.generate(prompt, trace=True)
        assert original['text'] == 'Yes.'
        interventions = []
        for intervention in ('remove_proposition_head', 'remove_object_identity'):
            saved = {}
            try:
                truth_layers = (4, 6, 8, 10) if a.composition else (3, 5, 7, 9)
                truth_head = 2 if a.composition else 8
                for layer in truth_layers:
                    if intervention == 'remove_proposition_head':
                        name = f'blocks.{layer}.attn.out.weight'
                        saved[name] = m.w[name]
                        m.w[name] = saved[name].clone()
                        m.w[name][:, truth_head * 64:(truth_head + 1) * 64] = 0
                    else:
                        name = f'blocks.{layer}.attn.qkv.weight'
                        saved[name] = m.w[name]
                        m.w[name] = saved[name].clone()
                        # Exact proposition head: subject[14], object[14],
                        # relation[7], validity/sentinel, recency. Remove the
                        # object component and lower the sentinel by 14 so
                        # lookup now accepts any object of the right relation.
                        m.w[name][truth_head * 64 + 14:truth_head * 64 + 28] = 0
                        sentinel_row = 1024 + truth_head * 64 + 35
                        m.w[name][sentinel_row, 8] = 9
                        m.w[name][sentinel_row, 9] = -9
                m.reset()
                ablated = m.generate(prompt, trace=True)
            finally:
                m.w.update(saved)
            expected = 'I do not know.' if intervention == 'remove_proposition_head' else 'No.'
            assert ablated['text'] == expected, (intervention, ablated['text'])
            interventions.append(dict(intervention=intervention, original=original, ablated=ablated))
        checks['truth_ablations'] = interventions
        m.reset()
        prefix = [s['bos'], s['user']] + m.tokenizer.encode(prompt) + [s['assistant']]
        logits, cache, _ = m.forward(prefix)
        generated = []
        for _ in range(10):
            full, _, _ = m.forward(prefix + generated)
            token = int(logits[-1].argmax().item())
            assert token == int(full[-1].argmax().item())
            generated.append(token)
            if token == s['eos']:
                break
            logits, cache, _ = m.forward([token], cache)
        checks['truth_cached_and_full_prefix_tokens_match'] = True
    if a.query_negation:
        prompt = 'A kelv cannot fly. Can a kelv not fly?'
        m.reset()
        original = m.generate(prompt)
        name = 'blocks.2.attn.out.weight'
        saved = m.w[name]
        try:
            m.w[name] = saved.clone()
            m.w[name][110:112] = 0  # Query polarity is logical feature 55.
            m.reset()
            ablated = m.generate(prompt)
        finally:
            m.w[name] = saved
        assert original['text'] == 'Yes.' and ablated['text'] == 'No.'
        checks['query_negation_ablation'] = dict(prompt=prompt, original=original['text'], ablated=ablated['text'])
    if a.composition:
        checks['composition_ablations'] = []
        probes = [
            ('and_to_or', 'A kelv can fly. A kelv cannot swim. Can kelv fly and swim?', 'No.', 'Yes.'),
            ('remove_unknown', 'A kelv can fly. Can kelv fly and glimmer?', 'I do not know.', 'Yes.'),
            ('remove_clause_scope', 'A kelv cannot fly. A kelv can swim. Can kelv fly? Can kelv swim and glimmer?', 'I do not know.', 'No.'),
        ]
        for intervention, prompt, expected, changed in probes:
            m.reset()
            original = m.generate(prompt, trace=True)
            saved = {}
            try:
                if intervention == 'and_to_or':
                    name = 'blocks.12.attn.out.weight'
                    saved[name] = m.w[name]
                    m.w[name] = saved[name].clone()
                    # Send the AND-head value to the OR flag instead.
                    m.w[name][142:144, 3 * 64:4 * 64] = saved[name][140:142, 3 * 64:4 * 64]
                    m.w[name][140:142, 3 * 64:4 * 64] = 0
                elif intervention == 'remove_unknown':
                    name = 'blocks.11.mlp.down.weight'
                    saved[name] = m.w[name]
                    m.w[name] = saved[name].clone()
                    m.w[name][132:134] = 0  # Atom-unknown feature 66.
                else:
                    name = 'blocks.12.attn.qkv.weight'
                    saved[name] = m.w[name]
                    m.w[name] = saved[name].clone()
                    for head in range(5):
                        m.w[name][1024 + head * 64, 6:8] = 0  # Clause-ID key feature 3.
                m.reset()
                ablated = m.generate(prompt, trace=True)
            finally:
                m.w.update(saved)
            assert original['text'] == expected and ablated['text'] == changed, (intervention, original['text'], ablated['text'])
            checks['composition_ablations'].append(dict(intervention=intervention, original=original, ablated=ablated))
        m.reset()
        prompt = probes[1][1]
        prefix = [s['bos'], s['user']] + m.tokenizer.encode(prompt) + [s['assistant']]
        logits, cache, _ = m.forward(prefix)
        generated = []
        for _ in range(12):
            full, _, _ = m.forward(prefix + generated)
            token = int(logits[-1].argmax().item())
            assert token == int(full[-1].argmax().item())
            generated.append(token)
            if token == s['eos']:
                break
            logits, cache, _ = m.forward([token], cache)
        checks['compound_unknown_cached_and_full_prefix_tokens_match'] = True
    if a.compound_facts:
        checks['assertion_ablations'] = []
        probes = [
            ('remove_continuation_validity', ['A kelv has wings and sensors. Does kelv have sensors?'], 'Yes.', 'I do not know.'),
            ('remove_predicate_binding', ['A kelv can fly and has wings and sensors. Does kelv have sensors?'], 'Yes.', 'I do not know.'),
            ('remove_question_exclusion', ['Does kelv need water and lavrik has wings?', 'Does lavrik have wings?'], 'I do not know.', 'Yes.'),
            ('remove_assertion_clause_scope', ['A kelv can fly. And swim. Can kelv swim?'], 'I do not know.', 'Yes.'),
        ]
        if a.deferred_assertions:
            probes.append(('force_following_token_valid', ['A kelv has wings and lavrik needs water. Does kelv have lavrik?'],
                           'I do not know.', 'Yes.'))
        for intervention, prompts, expected, changed in probes:
            m.reset()
            original = [m.generate(prompt, trace=True) for prompt in prompts]
            saved = {}

            def clone(name):
                if name not in saved:
                    saved[name] = m.w[name]
                    m.w[name] = saved[name].clone()
                return m.w[name]

            try:
                if intervention == 'remove_continuation_validity':
                    clone('blocks.2.mlp.down.weight')[950:952] = 0
                elif intervention == 'remove_predicate_binding':
                    clone('blocks.1.attn.out.weight')[:, 9 * 64:10 * 64] = 0
                elif intervention == 'remove_question_exclusion':
                    clone('blocks.1.attn.out.weight')[1012:1014, 8 * 64:9 * 64] = 0
                elif intervention == 'force_following_token_valid':
                    clone('blocks.1.mlp.down.weight')[506:508] = 0
                    bias = clone('blocks.1.mlp.down.bias')
                    bias[506], bias[507] = 1, -1  # Following-token validity feature 253.
                else:
                    # Remove the group-ID key in both lexical-predicate and
                    # full-subject attention. The query offset is common to
                    # all keys, so older valid clauses now become selectable.
                    clone('blocks.1.attn.qkv.weight')[1024 + 9 * 64, 828:830] = 0
                    group = 258 if a.deferred_assertions else 24
                    clone('blocks.2.attn.qkv.weight')[1024 + 1 * 64, 2 * group:2 * group + 2] = 0
                m.reset()
                ablated = [m.generate(prompt, trace=True) for prompt in prompts]
            finally:
                m.w.update(saved)
            assert original[-1]['text'] == expected and ablated[-1]['text'] == changed, (intervention, original[-1]['text'], ablated[-1]['text'])
            checks['assertion_ablations'].append(dict(intervention=intervention, original=original, ablated=ablated))
        if a.deferred_assertions:
            m.reset()
            prompt = 'A kelv cannot fly and can swim and glimmer. Can kelv swim and glimmer?'
            prefix = [s['bos'], s['user']] + m.tokenizer.encode(prompt) + [s['assistant']]
            logits, cache, _ = m.forward(prefix)
            generated = []
            for _ in range(12):
                full, _, _ = m.forward(prefix + generated)
                token = int(logits[-1].argmax().item())
                assert token == int(full[-1].argmax().item())
                generated.append(token)
                if token == s['eos']:
                    break
                logits, cache, _ = m.forward([token], cache)
            assert m.tokenizer.decode(generated) == 'Yes.'
            checks['deferred_assertion_cached_and_full_prefix_tokens_match'] = True
    if a.periodic_lookbacks:
        prompt = 'Hello. ' * 128 + 'A kelv can fly and swim. Can kelv swim?'
        heads = [(0, h) for h in (2, 3, 4, 5)] + [(1, h) for h in (5, 6, 7, 10)] + [(2, 2)]
        checks['periodic_position_ablations'] = []
        for phase, first in [('fast', 0), ('slow', 2)]:
            m.reset()
            original = m.generate(prompt)
            saved = {}
            try:
                for layer, head in heads:
                    name = f'blocks.{layer}.attn.qkv.weight'
                    if name not in saved:
                        saved[name] = m.w[name]
                        m.w[name] = saved[name].clone()
                    m.w[name][head * 64 + first:head * 64 + first + 2] = 0
                m.reset()
                ablated = m.generate(prompt)
            finally:
                m.w.update(saved)
            assert original['text'] == 'Yes.' and ablated['text'] != 'Yes.', (phase, original['text'], ablated['text'])
            checks['periodic_position_ablations'].append(dict(removed_phase=phase, original=original, ablated=ablated))
        m.reset()
        prompt = 'Hello. ' * 1024 + 'Can kelv glimmer?'
        prefix = [s['bos'], s['user']] + m.tokenizer.encode(prompt) + [s['assistant']]
        logits, cache, _ = m.forward(prefix)
        generated = []
        for _ in range(12):
            full, _, _ = m.forward(prefix + generated)
            token = int(logits[-1].argmax().item())
            assert token == int(full[-1].argmax().item())
            generated.append(token)
            if token == s['eos']:
                break
            logits, cache, _ = m.forward([token], cache)
        assert m.tokenizer.decode(generated) == 'I do not know.'
        checks['long_prefix_cached_and_full_tokens_match'] = dict(prefix_tokens=len(prefix), text=m.tokenizer.decode(generated))
    if a.fact_canonicalization:
        prompt = 'A kelv can fly and swim. A kelv cannot swim. Can kelv swim?'
        m.reset()
        original = m.generate(prompt)
        saved = {}
        try:
            name = 'blocks.2.mlp.down.weight'
            saved[name] = m.w[name]
            m.w[name] = saved[name].clone()
            m.w[name][880:936] *= 1.001  # Small extra magnitude in delayed identity codes.
            m.reset()
            protected = m.generate(prompt)
            name = 'blocks.3.mlp.down.weight'
            saved[name] = m.w[name]
            m.w[name] = saved[name].clone()
            m.w[name][880:954] = 0  # Remove the shared saturation barrier.
            m.reset()
            unprotected = m.generate(prompt)
        finally:
            m.w.update(saved)
        assert original['text'] == 'No.' and protected['text'] == 'No.' and unprotected['text'] == 'Yes.', (original['text'], protected['text'], unprotected['text'])
        checks['fact_coordinate_amplitude_intervention'] = dict(original=original, protected=protected,
                                                               unprotected=unprotected, identity_scale=1.001)
    checks['weight_file_unchanged_after_ablations'] = digest(a.model / 'weights.f32') == manifest['sha256']['weights.f32']
    assert checks['weight_file_unchanged_after_ablations']
    result = dict(git=git_state(), model_sha256=manifest['sha256']['weights.f32'], device=m.device, checks=checks,
                  note='Ablations replace selected runtime tensor references with zero copies, then restore them. The saved checkpoint is never modified.')
    a.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
