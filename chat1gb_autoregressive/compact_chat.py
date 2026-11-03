"""Offline CLI for the compact, untrained v17a circuit model."""
import argparse
import json
from pathlib import Path

from .compact_engine import Model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, default=Path('compact'))
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--prompt')
    p.add_argument('--max-new-tokens', type=int, default=64)
    p.add_argument('--trace', action='store_true')
    a = p.parse_args()
    model = Model(a.model_dir, a.device)
    if a.prompt is not None:
        result = model.generate(a.prompt, a.max_new_tokens, a.trace)
        print(json.dumps(result, indent=2) if a.trace else result['text'])
        return
    print(f'Direct-weight-chat v17a compact | {model.device} | analytically constructed weights - no training')
    print('Restricted facts, corrections, rules, and two-operand +/-. Useful open-ended chat was not achieved.')
    print('/reset starts a fresh conversation; /quit exits.\n')
    while True:
        try:
            prompt = input('You> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt == '/quit':
            break
        if prompt == '/reset':
            model.reset()
            print('Context reset.\n')
        elif prompt:
            try:
                result = model.generate(prompt, a.max_new_tokens, a.trace)
                print('Model> ' + result['text'] + '\n')
                if a.trace:
                    print(json.dumps(result, indent=2))
            except ValueError as e:
                print(str(e))


if __name__ == '__main__':
    main()
