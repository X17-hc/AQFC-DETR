"""Offline diagnostic counts, not a replacement for benchmark AP/LRP."""
from collections import defaultdict
import numpy as np


def density_bucket(count):
    return ('0' if count == 0 else '1-100' if count <= 100 else '101-300'
            if count <= 300 else '301-900' if count <= 900 else '>900')


def area_bucket(area):
    return 'verytiny' if area < 64 else 'tiny' if area < 256 else 'small' if area < 1024 else 'medium'


def box_iou_xywh(a, b):
    a, b = np.asarray(a, dtype=float).reshape(-1, 4), np.asarray(b, dtype=float).reshape(-1, 4)
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, :2] + a[:, None, 2:], b[None, :, :2] + b[None, :, 2:])
    intersection = np.maximum(hi-lo, 0).prod(-1)
    return intersection / np.maximum(a[:, 2:].prod(-1)[:, None] + b[:, 2:].prod(-1)[None] - intersection, 1e-12)


def analyze(records, annotation, image_ids, confidence=.25, iou_threshold=.5):
    categories = sorted(annotation['categories'], key=lambda c: c['id'])
    index = {c['id']: i for i, c in enumerate(categories)}
    ids = set(image_ids)
    predictions, ground_truth = defaultdict(list), defaultdict(list)
    for record in records:
        if record['category_id'] not in index:
            raise ValueError(f"Unknown category {record['category_id']}")
        if record['image_id'] not in ids:
            raise ValueError('Prediction image is outside evaluated image_ids')
        if record['score'] >= confidence:
            predictions[record['image_id']].append(record)
    ignored = 0
    for record in annotation['annotations']:
        if record['image_id'] in ids:
            if record.get('iscrowd', 0) or record.get('ignore', 0):
                ignored += 1
            else:
                ground_truth[record['image_id']].append(record)
    counts = {str(c['id']): dict(name=c['name'], gt=0, tp=0, fp=0, fn=0, duplicate=0) for c in categories}
    scales = defaultdict(lambda: dict(gt=0, tp=0, fn=0))
    buckets = defaultdict(lambda: dict(images=0, gt=0, tp=0, fp=0, fn=0))
    confusion = np.zeros((len(index)+1, len(index)+1), dtype=int)
    image_results = []
    for image_id in image_ids:
        gt = ground_truth[image_id]
        pred = sorted(predictions[image_id], key=lambda p: -p['score'])
        ious = box_iou_xywh([p['bbox'] for p in pred], [g['bbox'] for g in gt])
        matched, statuses = set(), []
        bucket = buckets[density_bucket(len(gt))]
        bucket['images'] += 1; bucket['gt'] += len(gt)
        for p, overlaps in zip(pred, ious):
            compatible = [j for j, g in enumerate(gt) if g['category_id'] == p['category_id']
                          and overlaps[j] >= iou_threshold]
            free = [j for j in compatible if j not in matched]
            count = counts[str(p['category_id'])]
            if free:
                j = max(free, key=lambda j: (overlaps[j], -j))
                matched.add(j); count['tp'] += 1; bucket['tp'] += 1
                statuses.append('tp')
            else:
                count['fp'] += 1; bucket['fp'] += 1
                status = 'duplicate' if compatible else 'fp'
                count['duplicate'] += int(bool(compatible)); statuses.append(status)
        for j, g in enumerate(gt):
            count = counts[str(g['category_id'])]
            count['gt'] += 1
            scale = scales[area_bucket(g.get('area', g['bbox'][2]*g['bbox'][3]))]
            scale['gt'] += 1; scale['tp'] += int(j in matched); scale['fn'] += int(j not in matched)
            if j not in matched:
                count['fn'] += 1; bucket['fn'] += 1
        # Separate class-agnostic one-to-one matching for confusion only.
        used = set()
        for p, overlaps in zip(pred, ious):
            free = [j for j in range(len(gt)) if j not in used and overlaps[j] >= iou_threshold]
            if free:
                j = max(free, key=lambda j: (overlaps[j], -j)); used.add(j)
                confusion[index[gt[j]['category_id']], index[p['category_id']]] += 1
            else:
                confusion[-1, index[p['category_id']]] += 1
        for j, g in enumerate(gt):
            if j not in used:
                confusion[index[g['category_id']], -1] += 1
        image_results.append(dict(image_id=image_id, predictions=pred, prediction_status=statuses,
                                  ground_truth=gt, missed_gt=[j for j in range(len(gt)) if j not in matched]))
    for count in counts.values():
        count['support_status'] = 'no_gt' if not count['gt'] else 'no_tp_at_diagnostic_threshold' if not count['tp'] else 'supported'
    return dict(confidence=confidence, iou_threshold=iou_threshold, category_counts=counts,
                area_counts=dict(scales), density_counts=dict(buckets), confusion=confusion.tolist(),
                categories=categories, images=image_results, ignored_gt=ignored,
                warning='Diagnostic matching only; crowd/ignore GT excluded, not official COCO crowd matching or LRP')
