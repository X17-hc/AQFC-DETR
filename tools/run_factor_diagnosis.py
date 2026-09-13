"""Bounded, fail-closed 640-only evaluation queue approved for 2026-09-09.

No training, retries, overwrites, dependency installation or automatic model changes.
Run --prepare first, inspect/test, then --run against that exact directory.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from util.experiment import sha256

DATA = Path('D:/PythonProject/DQ-DETR/DQ-DETR/data/path/AITODv2')
UPDATED = ROOT / 'outputs/pycharm_UpdateEpoch/20260908_083932_152642/checkpoint.pth'
STANDARD = ROOT / 'outputs/codex_BaselineControl_20260908/20260908_135455_470539/checkpoint.pth'
HISTORY = ROOT / 'outputs/codex_ScaleIsolation_20260908/scale640'
EXPECTED = '6294174fdea6d042d3c1d6c90c04ef70869feec0ff2e5f4f053421239dff3e08'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def source_hashes():
    files = [ROOT / 'main.py', ROOT / 'engine.py']
    for name in ('models', 'datasets', 'util', 'configs', 'tools'):
        files.extend((ROOT / name).rglob('*.py'))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)}


def bucket(count):
    return 0 if count == 0 else 1 if count <= 100 else 2 if count <= 300 else 3 if count <= 900 else 4


def sample_ids(annotation, total=1000):
    counts = Counter(a['image_id'] for a in annotation['annotations'] if not a.get('iscrowd', 0))
    groups = [[] for _ in range(5)]
    for image in sorted(annotation['images'], key=lambda x: x['id']):
        groups[bucket(counts[image['id']])].append(image['id'])
    if sum(map(len, groups)) < total:
        raise ValueError('Not enough test images')
    quota = [min(100, len(g)) for g in groups]
    remaining = total - sum(quota)
    capacity = [len(g) - q for g, q in zip(groups, quota)]
    shares = [remaining * c / sum(capacity) for c in capacity]
    extra = [math.floor(x) for x in shares]
    for i in sorted(range(5), key=lambda i: (-(shares[i] - extra[i]), i))[:remaining-sum(extra)]:
        extra[i] += 1
    quota = [a+b for a,b in zip(quota, extra)]
    rng = random.Random(42)
    selected = sorted(x for g,q in zip(groups, quota) for x in rng.sample(g, q))
    return selected, dict(population=[len(g) for g in groups], sample=quota,
                         buckets=['0', '1-100', '101-300', '301-900', '>900'],
                         count_definition='non-crowd annotations', seed=42)


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    annotations = DATA / 'annotations/aitodv2_test.json'
    if sha256(UPDATED) != EXPECTED:
        raise ValueError('Updated checkpoint differs')
    ids, strata = sample_ids(read(annotations))
    save(output / 'subset_ids.json', ids)
    save(output / 'precheck_ids.json', sorted(random.Random(42).sample(ids, 32)))
    save(output / 'sampling.json', strata)
    shutil.copy2(HISTORY / 'config_cfg.py', output / 'updated_config.py')
    shutil.copy2(STANDARD.parent / 'config_cfg.py', output / 'standard_config.py')
    manifest = dict(started_at='2026-09-09T08:45:29+08:00',
                    started_epoch=datetime.fromisoformat('2026-09-09T08:45:29+08:00').timestamp(),
                    prepared_at=datetime.now().astimezone().isoformat(),
                    source=source_hashes(), checkpoint_hashes={str(p): sha256(p) for p in (UPDATED, STANDARD)},
                    annotation=dict(path=str(annotations), sha256=sha256(annotations)),
                    subset_sha256=sha256(output/'subset_ids.json'),
                    config_hashes={name: sha256(output/name) for name in ('updated_config.py','standard_config.py')},
                    history=str(HISTORY), purpose='engineering_check; test diagnostic, not research selection')
    save(output/'queue_manifest.json', manifest)
    save(output/'queue_state.json', dict(status='prepared', runs=[]))
    print(json.dumps(dict(output=str(output), sampling=strata), ensure_ascii=False), flush=True)


def command(output, name, variant, ids, diagnostic=True):
    standard = variant == 'standard'
    cmd = [sys.executable, '-u', str(ROOT/'main.py'), '--config',
           str(output/('standard_config.py' if standard else 'updated_config.py')),
           '--data-root', str(DATA), '--no-pretrained', '--resume', str(STANDARD if standard else UPDATED),
           '--eval', '--output-dir', str(output/name), '--device', 'cuda', '--amp', '--seed', '42',
           '--num_workers', '0', '--max-train-steps', '0', '--max-eval-steps', '0', '--export-predictions']
    if diagnostic:
        cmd += ['--export-diagnostics']
    if ids:
        cmd += ['--eval-image-ids', str(output/ids)]
    cmd += ['--options', 'data_aug_scales=[640]', 'data_aug_max_size=640']
    if variant in ('fixed_spatial', 'fixed_semantic'):
        cmd += ['force_query_budget=900']
    if variant == 'fixed_semantic':
        cmd += ['proposal_selection_mode=semantic']
    return cmd


def artifacts(path, expected_ids, diagnostic):
    stats = json.loads((path/'log.txt').read_text(encoding='utf-8').splitlines()[-1])
    predictions = list(path.glob('predictions_*/predictions.json'))
    if len(predictions) != 1:
        raise ValueError('Missing or ambiguous prediction export')
    images = [json.loads(line) for line in predictions[0].with_name('images.jsonl').read_text().splitlines()]
    actual = [row['image_id'] for row in images]
    if len(actual) != len(set(actual)) or set(actual) != set(expected_ids):
        raise ValueError('Image coverage mismatch')
    if stats['test_evaluated_images'] != len(expected_ids) or not (path/'eval.pth').is_file():
        raise ValueError('Incomplete evaluation')
    if diagnostic:
        if not (path/'class_metrics.json').is_file():
            raise ValueError('Missing class metrics')
        if any(row['query_mask_count'] != row['executed_query_count'] for row in images):
            raise ValueError('Query mask mismatch')
    return stats, predictions[0]


def check_preflight(output):
    import numpy as np
    paths = [next((output/name).glob('predictions_*/predictions.json'))
             for name in ('precheck_off', 'precheck_on')]
    before, after = [read(p) for p in paths]
    if len(before) != len(after):
        raise ValueError('Precheck prediction lengths differ')
    for a,b in zip(before, after):
        if (a['image_id'], a['category_id']) != (b['image_id'], b['category_id']):
            raise ValueError('Precheck prediction IDs/order differ')
        np.testing.assert_allclose([*a['bbox'], a['score']], [*b['bbox'], b['score']], atol=1e-5, rtol=1e-4)
    save(output/'precheck_comparison.json', dict(passed=True, images=32, predictions=len(before), atol=1e-5, rtol=1e-4))


def run(output):
    # An exclusive marker prevents accidental restart/retry of a partially executed queue.
    with (output/'RUN_STARTED').open('x') as stream:
        stream.write(datetime.now().astimezone().isoformat())
    manifest = read(output/'queue_manifest.json')
    state = read(output/'queue_state.json')
    all_ids = [i['id'] for i in read(manifest['annotation']['path'])['images']]
    def execute(name, variant, ids='subset_ids.json', diagnostic=True):
        elapsed = time.time()-manifest['started_epoch']
        if elapsed >= 18*3600:
            raise TimeoutError('18-hour new-evaluation cutoff reached')
        if source_hashes() != manifest['source']:
            raise ValueError('Source changed after preparation')
        for path, digest in manifest['checkpoint_hashes'].items():
            if sha256(path) != digest:
                raise ValueError('Checkpoint changed')
        if sha256(manifest['annotation']['path']) != manifest['annotation']['sha256']:
            raise ValueError('Annotation changed')
        if sha256(output/'subset_ids.json') != manifest['subset_sha256']:
            raise ValueError('Subset changed')
        for name_cfg, digest in manifest['config_hashes'].items():
            if sha256(output/name_cfg) != digest:
                raise ValueError('Configuration changed')
        if shutil.disk_usage(output).free < 6*1024**3:
            raise OSError('Less than 6 GiB free disk space; queue paused')
        destination = output/name
        destination.mkdir(exist_ok=False)
        cmd = command(output, name, variant, ids, diagnostic)
        record = dict(name=name, variant=variant, command=cmd, start=time.time(), status='running')
        state['runs'].append(record)
        state.update(status='running', current=name, queue_pid=os.getpid())
        save(output/'queue_state.json', state)
        env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
        with (destination/'console.log').open('w', encoding='utf-8') as stdout, (destination/'stderr.log').open('w', encoding='utf-8') as stderr:
            child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            record['pid'] = child.pid
            save(output/'queue_state.json', state)
            code = child.wait()  # hourly heartbeat monitors; never auto-kill or retry
        record.update(exit_code=code, seconds=time.time()-record['start'])
        if code:
            raise RuntimeError(f'{name} exited with code {code}; no retry')
        stats, _ = artifacts(destination, read(output/ids) if ids else all_ids, diagnostic)
        record.update(status='completed', metrics=stats['test_bbox_metrics'],
                      images=stats['test_evaluated_images'], queries=stats['test_decoder_query_tokens'])
        save(output/'queue_state.json', state)
        print(f'{name} completed: AP={record["metrics"]["AP"]:.8f}', flush=True)
        return record
    try:
        execute('precheck_off', 'dynamic', 'precheck_ids.json', False)
        execute('precheck_on', 'dynamic', 'precheck_ids.json')
        check_preflight(output)
        records = [execute(name, variant) for name,variant in
                   [('S1','dynamic'),('S2','fixed_spatial'),('S3','fixed_semantic'),('S4','standard')]]
        ap = [r['metrics']['AP'] for r in records]
        if ap[1]-ap[0] >= .003:
            full = [('F_budget','fixed_spatial')]
            reason = 'S2-S1 >= 0.3 percentage points'
        elif ap[2]-ap[1] >= .003:
            full = [('F_candidates_spatial','fixed_spatial'),('F_candidates_semantic','fixed_semantic')]
            reason = 'S3-S2 >= 0.3 percentage points'
        elif abs(ap[0]-ap[3]) >= .003:
            full = [('F_standard','standard')]
            reason = 'abs(S1-S4) >= 0.3 percentage points'
        else:
            full, reason = [], 'No screening threshold met'
        # Include both candidate-confirmation jobs in the deadline estimate, never start half a pair.
        seconds_per_image = max(r['seconds']/r['images'] for r in records)
        estimate = seconds_per_image*len(all_ids)*len(full)*1.5 + 600*len(full)
        state.update(selection_reason=reason, planned_confirmation=full, estimated_confirmation_seconds=estimate)
        # Manifest/protocol comparison is reviewed by the agent before permitting reuse of history.
        if full and not (output/'HISTORY_COMPARABLE.json').exists():
            state.update(status='awaiting_history_review')
            save(output/'queue_state.json', state)
            return
        if time.time()+estimate > manifest['started_epoch']+22*3600:
            state.update(status='completed_subset_only', confirmation_note='22-hour completion budget would be exceeded')
        else:
            for name,variant in full:
                execute(name, variant, ids=None)
            state['status'] = 'completed'
        save(output/'queue_state.json', state)
    except Exception as exc:
        state.update(status='paused_error', error=f'{type(exc).__name__}: {exc}')
        save(output/'queue_state.json', state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    args = parser.parse_args()
    (prepare if args.prepare else run)(args.output.resolve())
