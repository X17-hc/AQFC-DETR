"""CPU-only diagnostics on exported detections; not a COCO/AP implementation.

GT features permit many-to-one coverage. Matching results are strictly one-to-one.
Prediction indices are original within-image export indices, never query IDs.
"""
from collections import Counter
import numpy as np
from util.error_analysis import box_iou_xywh, area_bucket, density_bucket

THRESHOLDS = {'50': .5, '75': .75}
MODES = {f'{name}_{key}': (cutoff, threshold)
         for name, cutoff in [('all', None), ('score025', .25)]
         for key, threshold in THRESHOLDS.items()}
FP_TYPES = ('duplicate', 'class_confusion', 'localization',
            'class_and_localization', 'background')


def iou_bin(value):
    return int(np.searchsorted([.1, .5, .75], value, side='right'))


def prepare_image(predictions, ground_truth):
    """Sort GT by stable annotation ID; same-score predictions keep export order."""
    gt = sorted(ground_truth, key=lambda g: g['id'])
    order = sorted(range(len(predictions)), key=lambda j: -predictions[j]['score'])
    pred = [predictions[j] for j in order]
    ious = box_iou_xywh([p['bbox'] for p in pred], [g['bbox'] for g in gt])
    pc = np.array([p['category_id'] for p in pred], dtype=np.int64)
    gc = np.array([g['category_id'] for g in gt], dtype=np.int64)
    return gt, pred, order, ious, pc, gc


def geometry_errors(pred, gt):
    x, y, w, h = pred['bbox']
    a, b, gw, gh = gt['bbox']
    return dict(center_dx=(x+w/2-a-gw/2)/gw,
                center_dy=(y+h/2-b-gh/2)/gh,
                width_ratio=w/gw, height_ratio=h/gh,
                aspect_ratio_relative_error=(w/h)/(gw/gh)-1)


def gt_features(gt, pred, order, ious, pc):
    features = []
    for j, g in enumerate(gt):
        same = np.flatnonzero(pc == g['category_id'])
        overlaps = ious[:, j]
        best = int(same[np.argmax(overlaps[same])]) if len(same) else None
        any_best = int(np.argmax(overlaps)) if len(pred) else None
        neighbors = same[overlaps[same] >= .1]
        neighbor = int(neighbors[0]) if len(neighbors) else None
        record = dict(best_same_iou=float(overlaps[best]) if best is not None else 0.,
            best_any_iou=float(overlaps[any_best]) if any_best is not None else 0.,
            best_any_category=pred[any_best]['category_id'] if any_best is not None else None,
            best_same_prediction_index=order[best] if best is not None else None,
            best_same_score=pred[best]['score'] if best is not None else None,
            best_same_rank=best+1 if best is not None else None,
            best_same_class_rank=int(np.flatnonzero(same == best)[0])+1 if best is not None else None,
            best_any_prediction_index=order[any_best] if any_best is not None else None,
            neighbor_prediction_index=order[neighbor] if neighbor is not None else None,
            neighbor_iou=float(overlaps[neighbor]) if neighbor is not None else None,
            geometry=geometry_errors(pred[best], g) if best is not None else None,
            thresholds={})
        for key, threshold in THRESHOLDS.items():
            good = same[overlaps[same] >= threshold]
            score = pred[int(good[0])]['score'] if len(good) else None
            low = bool(len(good) and score < .25)
            inconsistent = bool(len(good) and overlaps[neighbor] < threshold)
            record['thresholds'][key] = dict(count=int(len(good)), max_score=score,
                best_score_rank=int(good[0])+1 if len(good) else None,
                covered=bool(len(good)), geometry_insufficient=not bool(len(good)),
                low_score=low, ranking_inconsistent=inconsistent,
                low_and_ranking=low and inconsistent,
                covered_top300=bool(np.any(good < 300)))
        features.append(record)
    return features


def match_mode(gt, pred, order, ious, pc, gc, cutoff, threshold, features):
    matched = np.zeros(len(gt), dtype=bool)
    gt_prediction = [None]*len(gt)
    labels = []
    for i, p in enumerate(pred):
        if cutoff is not None and p['score'] < cutoff:
            labels.append(dict(label='excluded', matched_gt_id=None, associated_gt_id=None))
            continue
        overlaps = ious[i]
        same = pc[i] == gc
        compatible = same & (overlaps >= threshold)
        free = np.flatnonzero(compatible & ~matched)
        associated = None
        if len(free):
            j = int(free[np.argmax(overlaps[free])])
            matched[j] = True
            gt_prediction[j] = order[i]
            label = 'tp'
            associated = gt[j]['id']
        else:
            if np.any(compatible & matched):
                label = 'duplicate'
                candidates = np.flatnonzero(compatible & matched)
            elif np.any((~same) & (overlaps >= threshold)):
                label = 'class_confusion'
                candidates = np.flatnonzero((~same) & (overlaps >= threshold))
            elif np.any(same & (overlaps >= .1)):
                label = 'localization'
                candidates = np.flatnonzero(same & (overlaps >= .1))
            elif np.any((~same) & (overlaps >= .1)):
                label = 'class_and_localization'
                candidates = np.flatnonzero((~same) & (overlaps >= .1))
            else:
                label = 'background'
                candidates = []
            if len(candidates):
                associated = gt[int(candidates[np.argmax(overlaps[candidates])])]['id']
        labels.append(dict(label=label, matched_gt_id=associated if label=='tp' else None,
                           associated_gt_id=associated))
    key = '50' if threshold == .5 else '75'
    misses = []
    for j, feature in enumerate(features):
        t = feature['thresholds'][key]
        available = t['covered'] and (cutoff is None or t['max_score'] >= cutoff)
        misses.append(dict(matched=bool(matched[j]), prediction_index=gt_prediction[j],
            fn_geometry_insufficient=bool(not matched[j] and not t['covered']),
            fn_low_score=bool(not matched[j] and t['low_score']),
            fn_matching_competition=bool(not matched[j] and available)))
    counts = Counter(x['label'] for x in labels)
    assert int(matched.sum()) == counts['tp']
    assert counts['tp']+sum(counts[k] for k in FP_TYPES) == len(pred)-counts['excluded']
    assert counts['tp']+sum(not x['matched'] for x in misses) == len(gt)
    return labels, misses


def analyze_image(predictions, ground_truth):
    gt, pred, order, ious, pc, gc = prepare_image(predictions, ground_truth)
    features = gt_features(gt, pred, order, ious, pc)
    label_modes = {}
    for mode, (cutoff, threshold) in MODES.items():
        labels, misses = match_mode(gt, pred, order, ious, pc, gc, cutoff, threshold, features)
        label_modes[mode] = labels
        for feature, miss in zip(features, misses):
            feature.setdefault('matching', {})[mode] = miss
    predictions_out = [dict(prediction_index=original, rank=i+1, category_id=pred[i]['category_id'],
        score=pred[i]['score'], bbox=pred[i]['bbox'],
        modes={mode: labels[i] for mode, labels in label_modes.items()}) for i, original in enumerate(order)]
    return gt, features, predictions_out


def new_group():
    return dict(gt=0, epoch0=Counter(), epoch1=Counter(),
                transitions=[[0]*4 for _ in range(4)], paired=Counter())


def add_pair(group, before, after):
    group['gt'] += 1
    group['transitions'][iou_bin(before['best_same_iou'])][iou_bin(after['best_same_iou'])] += 1
    for version, f in [('epoch0', before), ('epoch1', after)]:
        counts = group[version]
        counts['sum_best_same_iou'] += f['best_same_iou']
        if f['geometry'] is not None:
            counts['geometry_support'] += 1
            for key, value in f['geometry'].items():
                counts['sum_abs_'+key] += abs(value-1) if key in ('width_ratio','height_ratio') else abs(value)
        for key, t in f['thresholds'].items():
            for flag in ('covered','geometry_insufficient','low_score','ranking_inconsistent','low_and_ranking','covered_top300'):
                counts[key+'_'+flag] += int(t[flag])
        for mode, m in f['matching'].items():
            counts[mode+'_tp'] += int(m['matched'])
            counts[mode+'_fn'] += int(not m['matched'])
            for flag in ('fn_geometry_insufficient','fn_low_score','fn_matching_competition'):
                counts[mode+'_'+flag] += int(m[flag])
    for key in THRESHOLDS:
        b, a = before['thresholds'][key], after['thresholds'][key]
        group['paired'][key+'_lost_coverage'] += int(b['covered'] and not a['covered'])
        group['paired'][key+'_gained_coverage'] += int(a['covered'] and not b['covered'])
        if b['covered'] and a['covered']:
            group['paired'][key+'_both_covered'] += 1
            group['paired'][key+'_new_ranking_issue'] += int(not b['ranking_inconsistent'] and a['ranking_inconsistent'])
            group['paired'][key+'_resolved_ranking_issue'] += int(b['ranking_inconsistent'] and not a['ranking_inconsistent'])


def gt_groups(gt, n_gt, original_k):
    return ['overall', 'class:'+str(gt['category_id']),
            'size:'+area_bucket(gt.get('area',gt['bbox'][2]*gt['bbox'][3])),
            'density:'+density_bucket(n_gt), 'original_K:'+str(original_k)]
