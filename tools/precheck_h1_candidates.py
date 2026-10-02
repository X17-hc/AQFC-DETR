"""32 then 1000 image H1 candidate precheck. No training, no formal checkpoint writes."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H1_SUBSET = '/workspace/AQFC-DETR/outputs/legacy24_diagnosis_20260917/subset_ids.json'
H1_CKPT = '/workspace/AQFC-DETR/outputs/legacy_joint/h1_epoch0/20260927_012928_172150_7b068fc7650e/checkpoint0023.pth'
H1_CFG = ROOT / 'configs/legacy_joint/h1_24e.py'
PAUSE_AP = 1.0
PAUSE_APVT = 1.0


def load_ids(path):
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    ids = raw if isinstance(raw, list) else raw.get('image_ids', raw.get('ids'))
    if not isinstance(ids, list) or len(set(ids)) != len(ids):
        raise ValueError('image id list must be unique')
    return [int(x) for x in ids]


def first_n_ids(path, count):
    ids = load_ids(path)
    if len(ids) < count:
        raise ValueError(f'need {count} ids, got {len(ids)}')
    return ids[:count]


def ids_sha256(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def risk_pause(baseline, candidate, ap_pp=PAUSE_AP, apvt_pp=PAUSE_APVT, cover_pp=2.0):
    d_ap = (baseline['AP'] - candidate['AP']) * 100
    d_vt = (baseline['APvt'] - candidate['APvt']) * 100
    d_cover = None
    if 'vt_cover_50' in baseline and 'vt_cover_50' in candidate:
        d_cover = (baseline['vt_cover_50'] - candidate['vt_cover_50']) * 100
    pause = d_ap > ap_pp or d_vt > apvt_pp or (d_cover is not None and d_cover > cover_pp)
    return pause, dict(d_ap_pp=d_ap, d_apvt_pp=d_vt, d_cover_pp=d_cover)


def parse_bbox_metrics(metrics_path):
    payload = json.loads(Path(metrics_path).read_text(encoding='utf-8'))
    box = payload.get('bbox_metrics') or payload.get('test_bbox_metrics') or payload
    return dict(AP=float(box['AP']), APvt=float(box['APvt']))


def build_eval_argv(args, ids_file, amp):
    argv = [
        sys.executable, '-u', str(ROOT / 'main.py'),
        '--config', str(args.config),
        '--data-root', args.data_root,
        '--pretrained', args.pretrained,
        '--output-dir', args.output_dir,
        '--unique-output-dir',
        '--eval', '--eval-boxes-only',
        '--eval-image-ids', str(ids_file),
        '--device', 'cuda', '--seed', '42', '--num_workers', '2',
    ]
    argv.append('--amp' if amp else '--no-amp')
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['32', '1000'], required=True)
    parser.add_argument('--config', default=str(H1_CFG))
    parser.add_argument('--data-root', default='/workspace/DQDetr/data/path/AITODv2')
    parser.add_argument('--pretrained', default=str(H1_CKPT))
    parser.add_argument('--subset-ids', default=str(H1_SUBSET))
    parser.add_argument('--image-ids', default='', help='Override JSON list')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--baseline', default='', help='JSON with AP/APvt for the 1000-image gate')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    count = 32 if args.stage == '32' else 1000
    source = Path(args.image_ids) if args.image_ids else Path(args.subset_ids)
    ids = first_n_ids(source, count) if count == 32 and not args.image_ids else load_ids(source)
    if len(ids) != count:
        raise ValueError(f'{args.stage} precheck requires {count} ids')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    id_file = out / f'precheck_{args.stage}_ids.json'
    id_file.write_text(json.dumps(ids), encoding='utf-8')
    record = dict(stage=args.stage, count=len(ids), ids_file=str(id_file),
                  ids_sha256=ids_sha256(ids), pretrained=args.pretrained, config=args.config)
    print('image_ids sha256', record['ids_sha256'], flush=True)
    if args.dry_run:
        (out / 'precheck_plan.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(json.dumps(record, indent=2))
        return
    for amp in (False, True):
        name = 'amp' if amp else 'fp32'
        command = build_eval_argv(args, id_file, amp)
        print('Running', name, flush=True)
        subprocess.check_call(command, cwd=ROOT)
    if args.stage == '1000' and args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding='utf-8'))
        # Caller fills candidate after eval; this file only records the gate function inputs.
        record['baseline'] = baseline
        record['pause_rule'] = 'AP or APvt drop > 1pp vs baseline'
    (out / 'precheck_launch.json').write_text(json.dumps(record, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
