"""One full-test forward that records why AP stayed flat.

The official COCO pass is unchanged. The ranking diagnostic reuses that
pass's IoU matrix with scores replaced by the maximum same-class IoU, so
the wall clock stays one forward plus one matching sweep.
"""
import argparse
import array
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from util.box_ops import box_iou

IOU_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
HIGH_THRESHOLDS = (0.75, 0.80, 0.85, 0.90, 0.95)
LOW_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70)
SIZE_NAMES = ('verytiny', 'tiny', 'small', 'medium')
BIN_NAMES = ('unmatched', 'loose', 'tight')
MAX_DETS = 1500
PUBLISHED_AP = 0.322


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def size_name(area):
    if area < 64:
        return 'verytiny'
    if area < 256:
        return 'tiny'
    if area < 1024:
        return 'small'
    return 'medium'


def iou_bin(value):
    if value < 0.5:
        return 'unmatched'
    if value < 0.75:
        return 'loose'
    return 'tight'


def _median(values):
    if len(values) == 0:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


class GeometryTable:
    def __init__(self):
        self.counts = {name: {bin_name: 0 for bin_name in BIN_NAMES} for name in SIZE_NAMES}
        self._center = {name: {bin_name: array.array('f') for bin_name in ('loose', 'tight')}
                        for name in SIZE_NAMES}
        self._wh = {name: {bin_name: array.array('f') for bin_name in ('loose', 'tight')}
                    for name in SIZE_NAMES}

    def add(self, det_boxes, det_scores, det_labels, gt_boxes, gt_labels, gt_areas):
        if gt_boxes.numel() == 0:
            return
        for label in torch.unique(gt_labels).tolist():
            gt_index = gt_labels == label
            chosen_boxes, best = _best_same_class(
                det_boxes, det_scores, det_labels, gt_boxes[gt_index], int(label))
            areas = gt_areas[gt_index]
            gt_sel = gt_boxes[gt_index]
            for row in range(gt_sel.shape[0]):
                name = size_name(float(areas[row]))
                quality = iou_bin(float(best[row]))
                self.counts[name][quality] += 1
                if chosen_boxes is None:
                    continue
                center, wh = _box_errors(chosen_boxes[row], gt_sel[row])
                if quality in ('loose', 'tight'):
                    self._center[name][quality].append(center)
                    self._wh[name][quality].append(wh)

    def as_dict(self):
        medians = {}
        for name in SIZE_NAMES:
            medians[name] = {}
            for quality in ('loose', 'tight'):
                medians[name][quality] = {
                    'center_px': _median(self._center[name][quality]),
                    'wh_rel': _median(self._wh[name][quality]),
                    'n': len(self._center[name][quality]),
                }
        return {'counts': self.counts, 'medians': medians}


def _best_same_class(det_boxes, det_scores, det_labels, gt_boxes, label):
    """Highest-IoU box among the top 1500 same-class detections by score."""
    empty = gt_boxes.new_zeros(gt_boxes.shape[0])
    if det_boxes.numel() == 0:
        return None, empty
    keep = det_labels == label
    if not bool(keep.any()):
        return None, empty
    scores = det_scores[keep]
    order = torch.argsort(scores, descending=True)[:MAX_DETS]
    chosen = det_boxes[keep][order]
    iou, _ = box_iou(chosen, gt_boxes)
    best_iou, best_index = iou.max(dim=0)
    return chosen[best_index], best_iou


def _box_errors(pred, gt):
    pred_wh = (pred[2:] - pred[:2]).clamp(min=0)
    gt_wh = (gt[2:] - gt[:2]).clamp(min=0)
    pred_c = (pred[:2] + pred[2:]) * 0.5
    gt_c = (gt[:2] + gt[2:]) * 0.5
    center = torch.linalg.norm(pred_c - gt_c).item()
    denom = gt_wh.clamp(min=1e-6)
    wh = torch.max(torch.abs(pred_wh - gt_wh) / denom).item()
    return float(center), float(wh)


def rescore_by_iou(detections, ious):
    """Reorder one class's score-sorted detections by max IoU with a ground-truth box."""
    if len(detections) == 0:
        return [], np.zeros((0, 0))
    if isinstance(ious, list) and len(ious) == 0:
        matrix = np.zeros((len(detections), 0))
    else:
        matrix = np.asarray(ious, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(len(detections), -1)
    if matrix.shape[0] != len(detections):
        raise ValueError('IoU rows do not match the score-sorted detections')
    if matrix.size == 0 or matrix.shape[1] == 0:
        scores = np.zeros(len(detections), dtype=np.float64)
    else:
        scores = matrix.max(axis=1)
    order = np.argsort(-scores, kind='mergesort')
    rescored = []
    for row in order:
        item = dict(detections[int(row)])
        item['score'] = float(scores[int(row)])
        rescored.append(item)
    return rescored, matrix[order]


def oracle_eval_imgs(official, oracle):
    """Score detections by IoU and match only at 0.75 over the full area range.

    The IoU matrix is the one just computed for the official ranking, so this
    does not run a second pairwise IoU.
    """
    params = official.params
    cat_ids = params.catIds if params.useCats else [-1]
    max_det = params.maxDets[-1]
    oracle.params = copy.deepcopy(params)
    oracle.params.iouThrs = np.array([0.75])
    oracle.params.areaRng = [list(params.areaRng[0])]
    oracle.params.areaRngLbl = ['all']
    oracle._gts = official._gts
    oracle._dts = {}
    oracle.ious = {}
    for img_id in params.imgIds:
        for cat_id in cat_ids:
            key = (img_id, cat_id)
            detections = official._dts[key]
            inds = np.argsort([-float(det['score']) for det in detections], kind='mergesort')
            ordered = [detections[int(index)] for index in inds[:max_det]]
            rescored, matrix = rescore_by_iou(ordered, official.ious[key])
            oracle._dts[key] = rescored
            oracle.ious[key] = matrix
    records = [
        oracle.evaluateImg(img_id, cat_id, oracle.params.areaRng[0], max_det)
        for cat_id in cat_ids
        for img_id in params.imgIds
    ]
    return np.asarray(records).reshape(len(cat_ids), 1, len(params.imgIds))


def _mean_valid(values):
    valid = values[values > -1]
    if valid.size == 0:
        return None
    return float(valid.mean())


def metric_tables(coco_eval):
    precision = coco_eval.eval['precision']
    recall = coco_eval.eval['recall']
    det_index = list(coco_eval.params.maxDets).index(MAX_DETS)
    labels = list(coco_eval.params.areaRngLbl)
    grid = {}
    for thr_index, thr in enumerate(IOU_THRESHOLDS):
        for area_index, label in enumerate(labels):
            grid[f'{thr:.2f}:{label}'] = _mean_valid(precision[thr_index, :, :, area_index, det_index])
    recall_50 = _mean_valid(recall[0, :, 0, det_index])
    recall_75 = _mean_valid(recall[list(IOU_THRESHOLDS).index(0.75), :, 0, det_index])
    stats = coco_eval.stats
    return {
        'ap_grid': grid,
        'ap': float(stats[0]),
        'ap50': float(stats[2]),
        'ap75': float(stats[3]),
        'recall_50': recall_50,
        'recall_75': recall_75,
    }


def _pp(new, old):
    return (float(new) - float(old)) * 100.0


def _cell(summary, thr, area):
    return summary['ap_grid'][f'{thr:.2f}:{area}']


def _relative(new, old):
    if new is None or old is None or old == 0:
        return None
    return (new - old) / old


def _tight_relative(new_counts, old_counts, size):
    new = new_counts[size]['tight']
    old = old_counts[size]['tight']
    if old == 0:
        return 0.0 if new == 0 else float('inf')
    return (new - old) / old


def _median_pair(summary, size, quality, key):
    return summary['medians'][size][quality][key]


def decide(h1, current):
    """Return the first matching cause. Deltas are current minus H1, in percentage points."""
    checks = []
    cells = []
    for thr in IOU_THRESHOLDS:
        for size in SIZE_NAMES:
            delta = _pp(_cell(current, thr, size), _cell(h1, thr, size))
            cells.append({'iou': thr, 'size': size, 'pp': delta})
    risen = [cell for cell in cells if cell['pp'] >= 0.3]
    fallen = [cell for cell in cells if cell['pp'] <= -0.3]
    ap_pp = _pp(current['ap'], h1['ap'])
    checks.append({'rule': 'cancellation', 'ap_pp': ap_pp, 'risen': risen, 'fallen': fallen})
    if risen and fallen and abs(ap_pp) <= 0.15:
        up = max(risen, key=lambda cell: cell['pp'])
        down = min(fallen, key=lambda cell: cell['pp'])
        sentence = (
            f"AP 被抵消：IoU {up['iou']:.2f} 的 {up['size']} 升高 {up['pp']:.2f} 个百分点，"
            f"IoU {down['iou']:.2f} 的 {down['size']} 下降 {abs(down['pp']):.2f} 个百分点。"
            f"{_falling_geometry(h1, current, down['size'])}"
        )
        return {'cause': 'cancellation', 'sentence': sentence, 'checks': checks}

    high = [_pp(_cell(current, thr, 'all'), _cell(h1, thr, 'all')) for thr in HIGH_THRESHOLDS]
    low = [_pp(_cell(current, thr, 'all'), _cell(h1, thr, 'all')) for thr in LOW_THRESHOLDS]
    tight_rel = {size: _tight_relative(current['counts'], h1['counts'], size) for size in SIZE_NAMES}
    center_rel = _relative(
        _median_pair(current, 'verytiny', 'loose', 'center_px'),
        _median_pair(h1, 'verytiny', 'loose', 'center_px'))
    wh_rel = _relative(
        _median_pair(current, 'verytiny', 'loose', 'wh_rel'),
        _median_pair(h1, 'verytiny', 'loose', 'wh_rel'))
    loose_up = current['counts']['verytiny']['loose'] > h1['counts']['verytiny']['loose']
    high_flat = all(abs(value) <= 0.15 for value in high)
    low_up = any(value >= 0.3 for value in low)
    tight_flat = all(value <= 0.01 for value in tight_rel.values())
    checks.append({
        'rule': 'high_iou_flat', 'high_pp': high, 'low_pp': low,
        'tight_relative': tight_rel, 'verytiny_loose_up': loose_up,
        'verytiny_loose_center_relative': center_rel,
        'verytiny_loose_wh_relative': wh_rel,
    })
    if high_flat and low_up and tight_flat and center_rel is not None and wh_rel is not None:
        if loose_up and center_rel <= -0.10 and abs(wh_rel) < 0.10:
            return {
                'cause': 'center_without_edge',
                'sentence': '新检到的极小目标中心更近，宽高误差仍把交并比挡在 0.75 以下。',
                'checks': checks,
            }
        if loose_up and abs(center_rel) < 0.10 and abs(wh_rel) < 0.10:
            return {
                'cause': 'localization_unchanged',
                'sentence': '定位误差分布没有移动，只是宽松匹配变多，0.75 以上没有新增真阳性。',
                'checks': checks,
            }

    total_new = sum(current['counts'][size]['tight'] for size in SIZE_NAMES)
    total_old = sum(h1['counts'][size]['tight'] for size in SIZE_NAMES)
    total_rel = 0.0 if total_old == 0 and total_new == 0 else (
        float('inf') if total_old == 0 else (total_new - total_old) / total_old)
    ap75_pp = _pp(current['ap75'], h1['ap75'])
    recall_pp = _pp(current['recall_75'], h1['recall_75'])
    checks.append({
        'rule': 'precision_eaten', 'tight_relative_total': total_rel,
        'ap75_pp': ap75_pp, 'recall75_pp': recall_pp,
    })
    if total_rel >= 0.02 and abs(ap75_pp) <= 0.15 and recall_pp > 0:
        return {
            'cause': 'precision_eaten',
            'sentence': '高分误检在 0.75 档变成假阳性，召回增加没有变成 PR 曲线下的面积。',
            'checks': checks,
        }

    oracle_pp = _pp(current['oracle_ap75'], h1['oracle_ap75'])
    checks.append({'rule': 'ranking', 'ap75_pp': ap75_pp, 'oracle_ap75_pp': oracle_pp})
    if abs(ap75_pp) <= 0.15 and oracle_pp >= 0.5:
        return {
            'cause': 'ranking',
            'sentence': '更紧的框已经在输出里，分类分数把更松的框排在前面。',
            'checks': checks,
        }
    if abs(ap75_pp) <= 0.15 and abs(oracle_pp) <= 0.15:
        return {
            'cause': 'ranking_not_responsible',
            'sentence': '排序没有藏住紧框。几何条件也没有单独成立，见 checks 里的原始差值。',
            'checks': checks,
        }
    return {
        'cause': 'unresolved',
        'sentence': '四条判定都没有成立，见 checks 里的原始差值。',
        'checks': checks,
    }


def _falling_geometry(h1, current, size):
    center = _relative(
        _median_pair(current, size, 'loose', 'center_px'),
        _median_pair(h1, size, 'loose', 'center_px'))
    wh = _relative(
        _median_pair(current, size, 'loose', 'wh_rel'),
        _median_pair(h1, size, 'loose', 'wh_rel'))
    if center is None or wh is None:
        return '下降一侧的宽松框中位数不足。'
    if wh > 0 and wh >= center:
        return f'下降的 {size} 一侧宽高相对误差变大。'
    if center > 0:
        return f'下降的 {size} 一侧中心误差变大。'
    return f'下降的 {size} 一侧宽松框误差没有变大。'


def compare_summaries(h1, current):
    report = decide(h1, current)
    h1_ok = abs(h1['ap'] - PUBLISHED_AP) <= 0.005
    current_ok = abs(current['ap'] - PUBLISHED_AP) <= 0.005
    report['headline_reproduced'] = bool(h1_ok and current_ok)
    report['h1_ap'] = h1['ap']
    report['current_ap'] = current['ap']
    report['h1_ap50'] = h1['ap50']
    report['current_ap50'] = current['ap50']
    report['h1_ap75'] = h1['ap75']
    report['current_ap75'] = current['ap75']
    if not report['headline_reproduced']:
        report['sentence'] = (
            f"复测 AP 与公布的 32.2 不一致（H1 {h1['ap']:.4f}，这次 {current['ap']:.4f}）。"
            f"在这份复测上，规则给出的句子是：{report['sentence']}"
        )
    return report


def _gt_index(coco_gt):
    grouped = {}
    for ann in coco_gt.dataset.get('annotations', []):
        if ann.get('iscrowd', 0):
            continue
        grouped.setdefault(int(ann['image_id']), []).append(ann)
    return grouped


def _gt_tensors(annotations, device):
    if not annotations:
        empty = torch.zeros((0, 4), device=device)
        return empty, torch.zeros(0, dtype=torch.long, device=device), torch.zeros(0, device=device)
    boxes = []
    labels = []
    areas = []
    for ann in annotations:
        x, y, w, h = ann['bbox']
        boxes.append([x, y, x + w, y + h])
        labels.append(int(ann['category_id']))
        areas.append(float(ann['area']))
    return (
        torch.tensor(boxes, dtype=torch.float32, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
        torch.tensor(areas, dtype=torch.float32, device=device),
    )


def _detection_results(predictions):
    """COCO xywh records, matching datasets.coco_eval.prepare_for_coco_detection."""
    from datasets.coco_eval import convert_to_xywh
    records = []
    for image_id, prediction in predictions.items():
        if prediction['boxes'].numel() == 0:
            continue
        boxes = convert_to_xywh(prediction['boxes']).tolist()
        scores = prediction['scores'].tolist()
        labels = prediction['labels'].tolist()
        records.extend(
            {'image_id': int(image_id), 'category_id': int(labels[index]),
             'bbox': box, 'score': float(scores[index])}
            for index, box in enumerate(boxes))
    return records


def _image_id(target):
    image_id = target['image_id']
    if torch.is_tensor(image_id):
        return int(image_id.reshape(-1)[0].item())
    return int(image_id)


def _oracle_ap75(coco_eval):
    precision = coco_eval.eval['precision']
    det_index = list(coco_eval.params.maxDets).index(MAX_DETS)
    return _mean_valid(precision[0, :, :, 0, det_index])


def diagnose_evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir,
                      wo_class_error=False, args=None, logger=None):
    del criterion, wo_class_error, logger
    from aitodpycocotools.coco import COCO
    from aitodpycocotools.cocoeval import COCOeval
    from datasets.coco_eval import create_common_coco_eval, evaluate as coco_evaluate
    import os
    import contextlib

    model.eval()
    raw_model = model.module if hasattr(model, 'module') else model
    transformer = getattr(raw_model, 'transformer', None)
    if transformer is not None:
        transformer.density_gate_scale = 1.0
    try:
        need_targets = bool(args.use_dn)
    except AttributeError:
        need_targets = False

    official = COCOeval(base_ds, iouType='bbox')
    oracle = COCOeval(base_ds, iouType='bbox')
    img_ids = []
    official_imgs = []
    oracle_imgs = []
    geometry = GeometryTable()
    ground_truth = _gt_index(base_ds)
    started = time.perf_counter()
    seen = 0
    limit = int(getattr(args, 'max_eval_steps', 0) or 0)
    for samples, targets in data_loader:
        if limit and seen >= limit:
            break
        samples = samples.to(device, non_blocking=getattr(args, 'non_blocking_transfer', False))
        targets = [{key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in target.items()} for target in targets]
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=args.amp):
                outputs = model(samples, targets) if need_targets else model(samples)
            orig_sizes = torch.stack([target['orig_size'] for target in targets], dim=0)
            results = postprocessors['bbox'](outputs, orig_sizes)
        predictions = {_image_id(target): result for target, result in zip(targets, results)}
        for image_id, result in predictions.items():
            gt_boxes, gt_labels, gt_areas = _gt_tensors(
                ground_truth.get(image_id, []), result['boxes'].device)
            geometry.add(
                result['boxes'].float(), result['scores'].float(), result['labels'].long(),
                gt_boxes, gt_labels, gt_areas)

        records = _detection_results(predictions)
        with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink):
            detected = COCO.loadRes(base_ds, records) if records else COCO()
        batch_ids = list(predictions.keys())
        official.cocoDt = detected
        official.params.imgIds = list(batch_ids)
        official.params.useCats = True
        batch_ids, batch_imgs = coco_evaluate(official)
        img_ids.extend(batch_ids)
        official_imgs.append(batch_imgs)
        oracle_imgs.append(oracle_eval_imgs(official, oracle))
        seen += 1
        if seen % 200 == 0 or seen == 1:
            elapsed = time.perf_counter() - started
            print(f'Test [{seen}] elapsed={elapsed:.1f}s per_image={elapsed / seen:.3f}s', flush=True)

    elapsed = time.perf_counter() - started
    official_cat = np.concatenate(official_imgs, 2)
    oracle_cat = np.concatenate(oracle_imgs, 2)
    create_common_coco_eval(official, img_ids, official_cat)
    create_common_coco_eval(oracle, img_ids, oracle_cat)
    official.accumulate()
    with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink):
        official.summarize()
    oracle.accumulate()
    summary = metric_tables(official)
    summary['oracle_ap75'] = _oracle_ap75(oracle)
    summary.update(geometry.as_dict())
    summary.update({
        'role': args.role,
        'checkpoint': str(Path(args.resume).resolve()),
        'sha256': sha256_file(args.resume),
        'config': str(args.config_file),
        'images': seen,
        'partial': bool(limit),
        'elapsed_seconds': elapsed,
        'per_image_seconds': elapsed / max(seen, 1),
        'nms_iou_threshold': float(getattr(args, 'nms_iou_threshold', -1)),
        'amp': bool(args.amp),
    })
    destination = Path(output_dir) / 'stall_summary.json'
    destination.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(
        f"STALL_SUMMARY images={seen} per_image={summary['per_image_seconds']:.3f} "
        f"ap={summary['ap']:.4f} ap75={summary['ap75']:.4f} oracle_ap75={summary['oracle_ap75']:.4f}",
        flush=True)

    class _SavedEval:
        eval = {'stall_summary': 'stall_summary.json'}

    class _EvaluatorView:
        coco_eval = {'bbox': _SavedEval()}

    return summary, _EvaluatorView()


def compare_main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--compare', nargs=2, metavar=('H1_JSON', 'CURRENT_JSON'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    h1 = json.loads(Path(args.compare[0]).read_text(encoding='utf-8'))
    current = json.loads(Path(args.compare[1]).read_text(encoding='utf-8'))
    if h1.get('partial') or current.get('partial'):
        raise SystemExit('PARTIAL_SUMMARY')
    if h1.get('images') != current.get('images'):
        raise SystemExit('IMAGE_COUNT_MISMATCH')
    report = compare_summaries(h1, current)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(report['sentence'])
    print('CAUSE', report['cause'], 'HEADLINE', report['headline_reproduced'])


def eval_main(argv):
    import main as main_mod
    import engine
    parser = main_mod.get_args_parser()
    parser.add_argument('--role', choices=('h1', 'current'), required=True)
    parser.add_argument('--expected-sha256', required=True)
    args = parser.parse_args(argv)
    digest = sha256_file(args.resume)
    if digest != args.expected_sha256:
        raise SystemExit(f'SHA_MISMATCH {digest}')
    engine.evaluate = diagnose_evaluate
    main_mod.evaluate = diagnose_evaluate
    main_mod.main(args)


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == '--compare':
        compare_main(argv)
        return
    eval_main(argv)


if __name__ == '__main__':
    main()
