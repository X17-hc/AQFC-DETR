import argparse
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util.config_validation import validate_config


def load_config(path):
    path = Path(path).resolve()
    namespace = runpy.run_path(str(path))
    config = {}
    bases = namespace.get('_base_', [])
    if isinstance(bases, str):
        bases = [bases]
    for base in bases:
        config.update(load_config(path.parent / base))
    config.update({key: value for key, value in namespace.items()
                   if not key.startswith('__') and key != '_base_'})
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    print(f'Configuration is valid: {args.config}')


if __name__ == '__main__':
    main()
