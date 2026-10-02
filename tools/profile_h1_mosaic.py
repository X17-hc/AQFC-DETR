"""H1 Mosaic profile commands. Does not start training."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H1_CFG = ROOT / 'configs/legacy_joint/h1_smoke.py'
H1_PRETRAINED = '/workspace/AQFC-DETR/weights/legacy/dqdetr_best305.pth'


def commands(config, data_root, pretrained, output_dir):
    chrome = [
        'python', '-u', str(ROOT / 'tools/profile_training_pipeline.py'),
        '--config', config, '--data-root', data_root, '--pretrained', pretrained,
        '--output-dir', output_dir, '--steps', '50',
    ]
    throughput = [
        'python', '-u', str(ROOT / 'tools/run_joint_training.py'),
        '--config', config, '--data-root', data_root, '--pretrained', pretrained,
        '--output-dir', output_dir, '--unique-output-dir', '--device', 'cuda',
        '--amp', '--seed', '42', '--num_workers', '2',
        '--max-train-steps', '250', '--max-eval-steps', '0',
        '--stop-after-epochs', '1', '--max-consecutive-skipped-steps', '20',
    ]
    return dict(chrome_trace_50=chrome, mosaic_250_steps=throughput)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(H1_CFG))
    parser.add_argument('--data-root', default='/workspace/DQDetr/data/path/AITODv2')
    parser.add_argument('--pretrained', default=str(H1_PRETRAINED))
    parser.add_argument('--output-dir', default='/workspace/AQFC-DETR/outputs/legacy_joint/h1_profile')
    parser.add_argument('--write', default='')
    args = parser.parse_args()
    payload = dict(
        note='Keep Mosaic on. 50-step Chrome trace plus 50 warmup + 200 measure '
             '(250 steps on H1 smoke). Do not treat mosaic_p=0 shorts as epoch time. '
             'Windows local stays num_workers=0.',
        commands=commands(args.config, args.data_root, args.pretrained, args.output_dir),
    )
    text = json.dumps(payload, indent=2)
    print(text)
    if args.write:
        Path(args.write).write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
