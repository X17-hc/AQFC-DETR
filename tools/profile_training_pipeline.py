"""Manual-only bounded profiler launcher; never runs at import time."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--pretrained')
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.steps <= 100:
        parser.error('Profile steps must be between 1 and 100')
    command = [sys.executable, 'main.py', '--config', args.config, '--data-root', args.data_root,
               '--output-dir', args.output_dir, '--unique-output-dir', '--profile-trace', 'training_trace.json',
               '--num_workers', '0', '--max-train-steps', str(args.steps),
               '--options', 'epochs=1', 'val_epoch=[]', 'run_purpose=benchmark']
    command += ['--pretrained', args.pretrained] if args.pretrained else ['--no-pretrained']
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)


if __name__ == '__main__':
    main()
