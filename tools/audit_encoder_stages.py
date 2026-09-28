"""Read-only detector-stage diagnosis. No optimizer, training or checkpoint writes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = '40dc24835aa6f9b6b211f1d8034da4e370518bf28c2ac11cbb47e4cd8e941928'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def select_ids(annotation, count):
    ids = sorted(int(row['id']) for row in annotation['images'])
    if len(ids) != len(set(ids)) or not 2 <= count <= len(ids):
        raise ValueError('Invalid image IDs/count')
    # Include the original smoke images, then deterministic additional images.
    return ids[:2] + sorted(random.Random(42).sample(ids[2:], count - 2))


def check_exports(directory, ids):
    files = list(directory.rglob('images.jsonl'))
    if len(files) != 1:
        raise RuntimeError(f'Expected one image export in {directory}')
    rows = [json.loads(line) for line in files[0].read_text().splitlines()]
    if [row['image_id'] for row in rows] != ids:
        raise RuntimeError('Image identity/order/coverage mismatch')
    return rows


def quality_review(results):
    baseline = results['f1_6']['metrics']['test_coco_eval_bbox'][0]
    review = dict(threshold_ap_points=0.3, stages={}, formal_training_acceptance='NOT_ESTABLISHED')
    for name in ('f1_4', 'f1_2'):
        ap = results[name]['metrics']['test_coco_eval_bbox'][0]
        review['stages'][name] = dict(AP_points=ap*100,
            change_from_six_points=(ap-baseline)*100, review_required=(baseline-ap)*100 > 0.3)
    return review


def run(args):
    if digest(args.pretrained) != EXPECTED:
        raise ValueError('S2 checkpoint SHA-256 mismatch')
    annotation = Path(args.data_root) / 'annotations/aitodv2_test.json'
    annotation_hash = digest(annotation)
    ids = select_ids(json.loads(annotation.read_text()), args.images)
    out = Path(args.output_dir) / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    ids_file = out / 'image_ids.json'
    ids_file.write_text(json.dumps(ids))
    manifest = dict(checkpoint=str(args.pretrained), checkpoint_sha256=EXPECTED,
                    annotation_sha256=annotation_hash, image_ids=ids,
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    source_hashes={str(p.relative_to(ROOT)): digest(p) for p in
                        [ROOT/'main.py', ROOT/'engine.py', ROOT/'util/precision24.py',
                         ROOT/'models/aqfcdetr/transformer.py', Path(__file__)]},
                    warning='32-image engineering diagnosis, not full-test AP or speed benchmark',
                    commands=[], results={})
    print(f'Output: {out}', flush=True)
    for name, config, depth in [('f0_6','f0_smoke',6), ('f1_6','f1_smoke',6),
                                 ('f1_4','f1_smoke',4), ('f1_2','f1_smoke',2)]:
        target = out / name
        target.mkdir()
        command = [sys.executable, '-u', str(ROOT/'main.py'), '--config',
                   str(ROOT/f'configs/precision24/{config}.py'), '--pretrained', args.pretrained,
                   '--data-root', args.data_root, '--output-dir', str(target), '--device', args.device,
                   '--eval', '--num_workers', '0', '--seed', '42', '--eval-image-ids', str(ids_file),
                   '--export-predictions', '--export-diagnostics']
        if args.amp:
            command.append('--amp')
        if name != 'f0_6':
            command += ['--options', f'precision24_stage_override={depth}']
        manifest['commands'].append(command)
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
        print(f'[{name}] fresh S2 initialization, evaluation only; {target}/console.log', flush=True)
        with (target/'console.log').open('x', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{name} failed ({result.returncode}); inspect {target}/console.log')
        rows = check_exports(target, ids)
        metrics = json.loads((target/'metrics.jsonl').read_text().splitlines()[-1])
        manifest['results'][name] = dict(metrics=metrics, images=rows)
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
        print(json.dumps(dict(variant=name, metrics={k:v for k,v in metrics.items()
            if 'coco_eval' in k or 'query' in k or 'seconds' in k})), flush=True)
    if digest(args.pretrained) != EXPECTED or digest(annotation) != annotation_hash:
        raise RuntimeError('Inputs changed during audit')
    # Compare identical initialized 6-layer predictions, without assuming 2-layer equivalence.
    import numpy as np
    def predictions(name):
        p = next((out/name).rglob('predictions.json'))
        return json.loads(p.read_text())
    a, b = predictions('f0_6'), predictions('f1_6')
    if len(a) != len(b) or any((x['image_id'],x['category_id']) != (y['image_id'],y['category_id']) for x,y in zip(a,b)):
        raise AssertionError('Initialized F0/F1 prediction identity mismatch')
    xa = np.array([[x['score'], *x['bbox']] for x in a]).reshape(-1,5)
    xb = np.array([[x['score'], *x['bbox']] for x in b]).reshape(-1,5)
    np.testing.assert_allclose(xa, xb, atol=1e-5, rtol=1e-4)
    manifest['initial_six_layer_export_equivalence'] = dict(passed=True,
        max_abs=float(np.max(np.abs(xa-xb))) if xa.size else 0., atol=1e-5, rtol=1e-4)
    # This is an engineering warning threshold, not a statistical significance test.
    # A successful subprocess must never be mistaken for architecture acceptance.
    manifest['quality_review'] = quality_review(manifest['results'])
    needs_review = any(r['review_required'] for r in manifest['quality_review']['stages'].values())
    manifest['status'] = 'COMPLETED_REVIEW_REQUIRED' if needs_review else 'COMPLETED_SUBSET_ONLY'
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest['quality_review'], indent=2), flush=True)
    if needs_review:
        print('WARNING: stage-transfer quality dropped. Do not treat exit code 0 as permission/acceptance for 24-epoch training.', flush=True)
    print(manifest['status'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--images', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--amp', action='store_true')
    run(parser.parse_args())
