"""Manual evaluation-only C0/C1 pipeline. Never trains or changes checkpoints."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ANNOTATION_SHA = '9817a75f9bc4a84015881f2ddbf39bcc29635654a0266b210fd382743c927d98'
IDS_SHA = 'ca8ea6d5b2db447a466bcba7ae0f768ec1fb420a33a8d6c2b87f6543f2ce9182'
INITIAL_SHA = 'c447367e2d19411319a990985e0127af8db08bfb7eec0329b2065b095ccc64eb'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding='utf-8')


def validate_pair(configs, manifests):
    allowed = {'output_dir', 'config_file', 'geometry_loss_weight'}
    differences = {k for k in configs[0].keys() | configs[1].keys()
                   if configs[0].get(k) != configs[1].get(k)}
    if differences - allowed:
        raise ValueError(f'Unexpected pair differences: {sorted(differences - allowed)}')
    for index, (cfg, manifest) in enumerate(zip(configs, manifests)):
        if cfg.get('geometry_loss_weight') != (0.0, 0.05)[index]:
            raise ValueError('C0/C1 geometry weight mismatch')
        if manifest['initialization']['sha256'] != INITIAL_SHA:
            raise ValueError('Unexpected training initialization')
        if cfg.get('eval_split') != 'test' or cfg.get('proposal_selection_mode') != 'spatial':
            raise ValueError('Expected test / dynamic spatial configuration')
        if cfg.get('force_query_budget') is not None:
            raise ValueError('Do not change query budget for this comparison')


def eval_command(root, config, checkpoint, data, ids, output):
    # --eval is mandatory; strict ordinary-model loading, not partial warm-start.
    return [sys.executable, '-u', str(root / 'main.py'), '--config', str(config),
            '--data-root', str(data), '--resume', str(checkpoint), '--no-pretrained',
            '--eval', '--output-dir', str(output), '--device', 'cuda', '--amp',
            '--seed', '42', '--num_workers', '0', '--max-eval-steps', '0',
            '--eval-image-ids', str(ids), '--export-predictions', '--export-diagnostics',
            '--options', 'data_aug_scales=[800]', 'data_aug_max_size=1333']


def run_logged(command, log, root):
    print('COMMAND:', subprocess.list2cmdline(command), flush=True)
    with log.open('x', encoding='utf-8') as stream:
        stream.write(subprocess.list2cmdline(command) + '\n')
        stream.flush()
        with subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                              errors='replace', bufsize=1) as process:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end='', flush=True)
            code = process.wait()
        if code:
            raise RuntimeError(f'Stage exited {code}; see {log}; no automatic retry')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--c0-run', type=Path, required=True)
    p.add_argument('--c1-run', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--image-ids', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--check-only', action='store_true', help='CPU checks only; no output directory or model execution')
    return p


def main():
    args = parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    runs = [args.c0_run.resolve(), args.c1_run.resolve()]
    if runs[0] == runs[1]:
        raise ValueError('C0 and C1 must be different runs')
    annotation = args.data_root / 'annotations/aitodv2_test.json'
    if digest(annotation) != ANNOTATION_SHA or digest(args.image_ids) != IDS_SHA:
        raise ValueError('Frozen annotation or subset SHA256 mismatch')
    ids, ann = read(args.image_ids), read(annotation)
    if len(ids) != 1000 or len(set(ids)) != 1000 or not set(ids) <= {i['id'] for i in ann['images']}:
        raise ValueError('Expected original 1000 unique image IDs')
    id_set = set(ids)
    gt = [g for g in ann['annotations'] if g['image_id'] in id_set]
    if len(gt) != 90536 or len({g['id'] for g in gt}) != len(gt) or any(g.get('ignore', 0) or g.get('iscrowd', 0) for g in gt):
        raise ValueError('Expected 90536 unique, non-ignore/non-crowd GT')
    configs = [read(p / 'config_args_all.json') for p in runs]
    manifests = [read(p / 'experiment_manifest.json') for p in runs]
    validate_pair(configs, manifests)
    checkpoints = [p / 'checkpoint0002.pth' for p in runs]
    cfg_files = [root / 'configs/geometry_v1' / f'{tag}_3e.py' for tag in ('c0', 'c1')]
    # Refuse changed training configs except the explicit evaluation overrides.
    sys.path.insert(0, str(root))
    from util.slconfig import SLConfig
    for file, saved in zip(cfg_files, configs):
        # Saved JSON serializes tuples as lists; compare the same representation.
        current = json.loads(json.dumps(SLConfig.fromfile(str(file))._cfg_dict.to_dict()))
        changed = [k for k, v in current.items() if saved.get(k) != v]
        if changed:
            raise ValueError(f'Config differs from training record: {changed}')
    import torch
    for checkpoint in checkpoints:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        metadata = saved.get('run_metadata', {})
        if saved.get('epoch') != 2 or metadata.get('epoch_complete') is not True or metadata.get('smoke_test'):
            raise ValueError(f'Not a completed third-epoch checkpoint: {checkpoint}')
        del saved
    sources = [root / name for name in ('main.py', 'engine.py')]
    for folder in ('models', 'util', 'datasets', 'configs', 'tools'):
        sources.extend(sorted((root / folder).rglob('*.py')))
    inputs = checkpoints + [annotation, args.image_ids] + sources
    for run in runs:
        inputs.extend(run / name for name in ('config_args_all.json', 'experiment_manifest.json'))
    hashes = {str(p): digest(p) for p in inputs}
    print('Preflight OK: 1000 images / 90536 GT; two complete epoch2 checkpoints.', flush=True)
    if args.check_only:
        return
    if len([v for v in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if v.strip()]) != 1:
        raise ValueError('Set exactly one CUDA_VISIBLE_DEVICES in the run configuration')
    output = args.output_dir / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    print('Output:', output, flush=True)
    state = dict(status='running', inputs_sha256=hashes,
                 labels={'epoch0': 'C0 checkpoint0002', 'epoch1': 'C1 checkpoint0002'},
                 purpose='test engineering diagnosis; native truncated predictions, not all decoder logits',
                 cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), command=sys.argv)
    save(output / 'pair_manifest.json', state)
    try:
        for i, label in enumerate(('C0', 'C1')):
            state['stage'] = label
            save(output / 'pair_manifest.json', state)
            run_logged(eval_command(root, cfg_files[i], checkpoints[i], args.data_root,
                                    args.image_ids, output / label), output / f'{label}_console.log', root)
        state['stage'] = 'paired_analysis'
        save(output / 'pair_manifest.json', state)
        run_logged([sys.executable, '-u', str(root / 'tools/analyze_paired_head_errors.py'),
                    '--before', str(output / 'C0'), '--after', str(output / 'C1'),
                    '--annotations', str(annotation), '--image-ids', str(args.image_ids),
                    '--expected-images', '1000', '--expected-gt', '90536',
                    '--output', str(output / 'paired')], output / 'paired_console.log', root)
        if any(digest(Path(p)) != h for p, h in hashes.items()):
            raise ValueError('Input/source changed during the run')
        state['status'] = 'completed'
        state['inputs_unchanged'] = True
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        save(output / 'pair_manifest.json', state)
    print('Completed; paired/summary.json and paired/paired_gt.jsonl. epoch0=C0, epoch1=C1.', flush=True)


if __name__ == '__main__':
    main()
