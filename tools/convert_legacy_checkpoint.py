import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util.checkpoint_migration import convert_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Convert legacy detector weights to AQFC warm-start format')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prefer-ema', action='store_true')
    args = parser.parse_args()
    output = convert_checkpoint(args.input, args.output, prefer_ema=args.prefer_ema)
    print(f'Converted warm-start checkpoint: {output}')


if __name__ == '__main__':
    main()
