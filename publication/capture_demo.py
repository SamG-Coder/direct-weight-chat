"""Capture real frozen-model conversations for a clearly labelled presentation.

No expected answer is supplied to the model. This is demonstration evidence,
not a newly scored generalization benchmark. All prompts and outputs are kept.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from chat1gb_autoregressive.engine import Model


CONVERSATIONS = [
    dict(id='memory', title='Facts, memory & corrections',
         subtitle='One continuous conversation. Facts arrive in the prompt.',
         note='Restricted grammar. No Python answer lookup or weight updates.',
         prompts=[
             'A kelv can fly. What can a kelv do?',
             'Can kelv swim?',
             'A kelv can swim. Can kelv swim?',
             'A kelv cannot swim. Can kelv swim?',
             'Can kelv fly?',
         ]),
    dict(id='rules', title='Reasoning from supplied rules',
         subtitle='One continuous conversation. A missing premise becomes known.',
         note='Signed ground rules. At most three links along one selected path.',
         prompts=[
             'If kelv can shimmer then tal needs water. If tal needs water then pel cannot glimmer. Can pel glimmer?',
             'Kelv can shimmer. Does tal need water?',
             'Can pel glimmer?',
             'Can pel not glimmer?',
         ]),
    dict(id='arithmetic', title='Bounded signed arithmetic',
         subtitle='Two operands. Addition and subtraction in the compiled range.',
         note='Arithmetic happens inside the neural circuit. Not a general math model.',
         prompts=['999 - -999', '-999 + -999']),
    dict(id='limits', title='Where it still fails',
         subtitle='A valid older proof is missed. Open-ended writing is unsupported.',
         note='These outputs are retained. Useful general chat was not achieved.',
         prompts=[
             'If kelv can fly then tal needs water. If pel can swim then tal needs water. Kelv can fly. Does tal need water?',
             'Write a short story about a robot learning to paint.',
         ]),
]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=Path('build/v17a'))
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--output', type=Path, default=Path('publication/demo-transcript.json'))
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit('Refusing to overwrite captured evidence')
    compact = (a.model / 'weights.sparse.zip').exists()
    manifest = json.loads((a.model / ('dense-manifest.json' if compact else 'manifest.json')).read_text(encoding='utf-8'))
    weight_file = a.model / ('weights.sparse.zip' if compact else 'weights.f32')
    before = digest(weight_file)
    if compact:
        from chat1gb_autoregressive.compact_engine import Model as CompactModel
        model = CompactModel(a.model, device=a.device)
        assert model.compact_metadata['dense_weight_sha256'] == manifest['sha256']['weights.f32']
    else:
        assert before == manifest['sha256']['weights.f32']
        model = Model(a.model, device=a.device)
    model.generate('Hello', 4)
    conversations = []
    for item in CONVERSATIONS:
        model.reset()
        rows = []
        for prompt in item['prompts']:
            result = model.generate(prompt, max_new_tokens=64, trace=True)
            rows.append(result)
            print(json.dumps(dict(conversation=item['id'], prompt=prompt, text=result['text']), ensure_ascii=False), flush=True)
        conversations.append({**{k:v for k,v in item.items() if k != 'prompts'}, 'turns':rows})
    assert digest(weight_file) == before
    assert not any(n.endswith(('.build', '.knowledge', '.program')) for n in sys.modules)
    import torch
    data = dict(version='v17a-compact' if compact else 'v17a', status='final-frozen-model-demonstration-not-a-benchmark',
                created_utc=datetime.now(timezone.utc).isoformat(),
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                model_sha256=manifest['sha256']['weights.f32'], runtime_sha256=digest(Path('chat1gb_autoregressive/engine.py')),
                parameter_count=manifest['parameter_count'], weight_bytes=manifest['weight_bytes'],
                nonzero_parameters=manifest['nonzero_parameters'], device=model.device,
                python=platform.python_version(), torch=torch.__version__,
                hardware=torch.cuda.get_device_name() if model.device == 'cuda' else platform.processor(),
                weights_unchanged=True, runtime_imports_no_compiler=True,
                presentation='Actual outputs, replayed with edited pacing for readability. Not a live speed demonstration. Context resets only between the four named conversations.',
                conversations=conversations)
    if compact:
        data['compact'] = dict(archive_sha256=before, archive_bytes=weight_file.stat().st_size,
                               runtime_sha256=digest(Path('chat1gb_autoregressive/compact_engine.py')),
                               resident_weight_tensor_bytes=model.storage_stats(),
                               method='Lossless sparse coefficients; original tensor operations; one tensor materialized at a time.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
