"""Bounded AI-TOD2 continuation: retain four epochs, then evaluate only epoch 5."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_factor_diagnosis import DATA, artifacts, read, save, source_hashes
from util.experiment import sha256

SOURCE = ROOT / 'outputs/continuation_20260910_012534/train_epoch1/checkpoint.pth'
EXPECTED = 'f0c25506797d3390b602e3b1b90de1f783ed9146dfb5e6ec0295ec7fb2eab702'


def command(output, evaluation=False):
    result = [sys.executable, '-u', str(ROOT / 'main.py'), '--config',
              str(output / 'training_config.py'), '--data-root', str(DATA),
              '--no-pretrained', '--resume',
              str(output / 'train/checkpoint0005.pth' if evaluation else SOURCE),
              '--output-dir', str(output / ('final640' if evaluation else 'train')),
              '--device', 'cuda', '--amp', '--seed', '42', '--num_workers', '0',
              '--max-train-steps', '0', '--max-eval-steps', '0']
    if evaluation:
        result += ['--eval', '--export-predictions', '--export-diagnostics',
                   '--options', 'data_aug_scales=[640]', 'data_aug_max_size=640']
    else:
        # epochs is the cumulative stop index; 2,3,4,5 are the four new epochs.
        result += ['--max-consecutive-skipped-steps', '20', '--options',
                   'epochs=6', 'val_epoch=[]', 'save_checkpoint_interval=1']
    return result


def check_checkpoint(path, epoch):
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    required = {'model', 'optimizer', 'lr_scheduler', 'scaler', 'rng_states',
                'epoch', 'variant_signature', 'best_metrics', 'run_metadata'}
    if not required.issubset(checkpoint) or checkpoint['epoch'] != epoch:
        raise ValueError(f'Incomplete or incorrect resume checkpoint: {path}')
    metadata = checkpoint['run_metadata']
    if not metadata.get('epoch_complete') or metadata.get('smoke_test'):
        raise ValueError(f'Checkpoint is partial/smoke: {path}')
    if not checkpoint['rng_states']:
        raise ValueError('Missing RNG states')
    return dict(path=str(path), epoch=epoch, sha256=sha256(path),
                size_bytes=path.stat().st_size)


def prepare(output):
    if output.exists():
        raise FileExistsError(f'Refusing existing output: {output}')
    if sha256(SOURCE) != EXPECTED:
        raise ValueError('Source checkpoint hash mismatch')
    source_info = check_checkpoint(SOURCE, 1)
    if shutil.disk_usage(ROOT).free < 8 * 1024**3:
        raise OSError('At least 8 GiB free required before starting')
    config = SOURCE.parent / 'config_cfg.py'
    values = {}
    exec(compile(config.read_text(encoding='utf-8'), str(config), 'exec'), values)
    if (values['dataset_file'], values['train_split'], values['eval_split']) != ('aitodv2', 'trainval', 'test'):
        raise ValueError('Unexpected dataset protocol')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config, output / 'training_config.py')
    inputs = [SOURCE, output / 'training_config.py',
              DATA / 'annotations/aitodv2_trainval.json', DATA / 'annotations/aitodv2_test.json']
    save(output / 'manifest.json', dict(created_at=datetime.now().astimezone().isoformat(),
         source=source_hashes(), inputs={str(p): sha256(p) for p in inputs},
         source_checkpoint=source_info, dataset='AI-TOD2 / AI-TOD-V2',
         purpose='engineering_check; trainval/test; not independent research validation',
         epochs=[2, 3, 4, 5], evaluation_epochs=[5], evaluation_size=640,
         commands=[command(output), command(output, True)],
         schedule_note='Original teacher schedule retained; allocator total_epochs becomes 6.'))
    save(output / 'state.json', dict(status='prepared', runs=[]))
    print(output, flush=True)


def run(output):
    with (output / 'RUN_STARTED').open('x', encoding='utf-8') as stream:
        stream.write(datetime.now().astimezone().isoformat())
    manifest = read(output / 'manifest.json')
    state = read(output / 'state.json')
    state['queue_pid'] = os.getpid()

    def integrity():
        if source_hashes() != manifest['source']:
            raise ValueError('Source changed since preparation; queue paused')
        for path, digest in manifest['inputs'].items():
            if sha256(path) != digest:
                raise ValueError(f'Input changed: {path}')
        if shutil.disk_usage(output).free < 4 * 1024**3:
            raise OSError('Less than 4 GiB free; queue paused')

    def execute(name, evaluation=False):
        integrity()
        destination = output / name
        destination.mkdir(exist_ok=False)
        record = dict(name=name, command=command(output, evaluation),
                      status='running', started_at=datetime.now().astimezone().isoformat())
        state['runs'].append(record)
        state.update(status='running', current=name)
        start = time.monotonic()
        with (destination / 'console.log').open('w', encoding='utf-8') as stdout, \
             (destination / 'stderr.log').open('w', encoding='utf-8') as stderr:
            process = subprocess.Popen(record['command'], cwd=ROOT, stdout=stdout, stderr=stderr,
                env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1'))
            record['pid'] = process.pid
            save(output / 'state.json', state)
            record['exit_code'] = process.wait()
        record['seconds'] = time.monotonic() - start
        if record['exit_code']:
            raise RuntimeError(f'{name} failed; no automatic retry')
        record['status'] = 'process_finished'
        save(output / 'state.json', state)
        return record

    try:
        training = execute('train')
        rows = [json.loads(line) for line in (output / 'train/log.txt').read_text().splitlines() if line]
        if [row['epoch'] for row in rows] != [2, 3, 4, 5]:
            raise ValueError('Four expected epochs were not completed')
        for row in rows:
            if row['train_train_iterations'] != 14018 or row['train_optimizer_steps'] <= 0:
                raise ValueError('Incomplete training iteration/update coverage')
        training['checkpoints'] = [check_checkpoint(output / f'train/checkpoint{e:04}.pth', e)
                                   for e in range(2, 6)]
        training['stats'] = rows
        training['status'] = 'completed'
        save(output / 'state.json', state)
        evaluation = execute('final640', True)
        ids = [image['id'] for image in read(DATA / 'annotations/aitodv2_test.json')['images']]
        if len(ids) != 14018:
            raise ValueError('Unexpected test population')
        stats, _ = artifacts(output / 'final640', ids, True)
        evaluation.update(status='completed', stats=stats)
        integrity()
        state.update(status='completed', completed_at=datetime.now().astimezone().isoformat())
        save(output / 'state.json', state)
        print('Four epochs and final640 evaluation completed.', flush=True)
    except Exception as error:
        state.update(status='paused_error', error=f'{type(error).__name__}: {error}')
        save(output / 'state.json', state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--run', action='store_true')
    args = parser.parse_args()
    (prepare if args.prepare else run)(args.output.resolve())
