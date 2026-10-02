"""H1 worker matrix. Does not change global num_workers default (stays 0)."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX = (
    dict(num_workers=0, persistent_workers=False),
    dict(num_workers=2, persistent_workers=False),
    dict(num_workers=2, persistent_workers=True),
    dict(num_workers=4, persistent_workers=True),
)


def commands(config, data_root, pretrained, output_dir):
    rows = []
    for spec in MATRIX:
        argv = [
            'python', '-u', str(ROOT / 'tools/run_joint_training.py'),
            '--config', config, '--data-root', data_root, '--pretrained', pretrained,
            '--output-dir', output_dir, '--unique-output-dir', '--device', 'cuda',
            '--amp', '--seed', '42', '--max-train-steps', '50', '--max-eval-steps', '0',
            '--stop-after-epochs', '1', '--max-consecutive-skipped-steps', '20',
            '--num_workers', str(spec['num_workers']),
        ]
        if spec['persistent_workers']:
            argv.append('--persistent-workers')
        rows.append(dict(spec=spec, argv=argv))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/legacy_joint/h1_smoke.py'))
    parser.add_argument('--data-root', default='/workspace/DQDetr/data/path/AITODv2')
    parser.add_argument('--pretrained', default='/workspace/AQFC-DETR/weights/legacy/dqdetr_best305.pth')
    parser.add_argument('--output-dir', default='/workspace/AQFC-DETR/outputs/legacy_joint/h1_workers')
    parser.add_argument('--write', default='')
    args = parser.parse_args()
    rows = commands(args.config, args.data_root, args.pretrained, args.output_dir)
    text = json.dumps(dict(
        note='Keep Mosaic on. Do not treat mosaic_p=0 short tests as H1 epoch time. '
             'Windows local stays num_workers=0. Change a server launcher default only after this matrix.',
        default_num_workers=0,
        default_persistent_workers=False,
        matrix=rows,
    ), indent=2)
    print(text)
    if args.write:
        Path(args.write).write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
