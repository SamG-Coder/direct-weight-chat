"""Build a checkpoint offline, refusing to replace an existing output directory."""
import argparse
import importlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('version', help='v0, v1-v5, or current')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.version == 'v0':
        name = 'chat1gb.build'
    elif args.version == 'current':
        name = 'chat1gb_autoregressive.build'
    elif args.version in ('v1', 'v2', 'v3', 'v4', 'v5'):
        name = 'versions.autoregressive_' + args.version + '.build'
    else:
        raise SystemExit('Unknown version')
    compiler = importlib.import_module(name)
    compiler.MODEL = args.output.resolve()
    compiler.main()


if __name__ == '__main__':
    main()
