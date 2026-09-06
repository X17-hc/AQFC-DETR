#!/usr/bin/env python
"""TTA launcher with explicit paths.

Each augmentation is evaluated independently through the standard evaluation
entrypoint. This preserves per-branch dynamic query statistics. Prediction
fusion is intentionally kept as a separate analysis step because score
calibration/WBF settings are dataset-specific.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--scales', nargs='+', type=int, default=[736, 800, 864])
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    output_root = Path(args.output_dir)
    for scale in args.scales:
        branch = output_root / f'scale_{scale}'
        command = [sys.executable, 'main.py', '--config', args.config,
                   '--data-root', args.data_root, '--resume', args.checkpoint,
                   '--output-dir', str(branch), '--device', args.device, '--eval',
                   '--options', f'data_aug_scales=[{scale}]']
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
