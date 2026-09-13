"""Opt-in, post-forward engineering diagnostics; never routes using ground truth."""
import json
from pathlib import Path

import numpy as np
import torch


def restrict_eval_dataset(dataset, filename):
    ids = json.loads(Path(filename).read_text(encoding='utf-8'))
    if not isinstance(ids, list) or not ids or any(type(x) is not int for x in ids):
        raise ValueError('Evaluation IDs must be a nonempty JSON list of integers')
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate evaluation image IDs')
    unknown = set(ids) - set(dataset.ids)
    if unknown:
        raise ValueError(f'Unknown evaluation image IDs: {sorted(unknown)[:10]}')
    # Keep the full COCO annotation object: evaluator synchronizes the actual image IDs.
    dataset.ids = list(ids)


def finite_json(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@torch.no_grad()
def image_diagnostics(outputs, target, index):
    from util.box_ops import box_cxcywh_to_xyxy, box_iou
    allocator = outputs.get('allocator_outputs', {})
    row = {'gt_count': len(target['boxes'])}
    for key in ('predicted_count', 'raw_count', 'boundaries', 'routing_count',
                'raw_query_counts', 'requested_query_counts', 'valid_token_counts'):
        value = allocator.get(key)
        row[key] = finite_json(value[index]) if value is not None else None
    valid = outputs['query_valid_mask'][index]
    proposals = outputs['interm_outputs']['pred_boxes'][index][valid].detach().float()
    gt = target['boxes'].detach().float()
    best = gt.new_zeros(len(gt))
    # Bounded matrix memory, even for dense images; no input/output is modified.
    for chunk in proposals.split(256):
        if len(gt):
            best = torch.maximum(best, box_iou(box_cxcywh_to_xyxy(chunk),
                                              box_cxcywh_to_xyxy(gt))[0].amax(0))
    row['proposal_stage'] = 'selected_encoder_regressed_boxes_before_decoder'
    row['proposal_hits_025'] = int((best >= .25).sum())
    row['proposal_hits_050'] = int((best >= .5).sum())
    row['proposal_recall_025'] = row['proposal_hits_025'] / len(gt) if len(gt) else None
    row['proposal_recall_050'] = row['proposal_hits_050'] / len(gt) if len(gt) else None
    row['query_mask_count'] = int(valid.sum())
    return row


def export_class_metrics(evaluator, destination):
    """Extract in-memory numeric arrays, without deserializing evaluation pickle files."""
    precision, recall = evaluator.eval['precision'], evaluator.eval['recall']
    params = evaluator.params
    image_ids = set(params.imgIds)
    def mean_valid(array):
        values = array[np.isfinite(array) & (array >= 0)]
        return float(values.mean()) if values.size else None
    rows = []
    for k, category_id in enumerate(params.catIds):
        category = evaluator.cocoGt.cats[category_id]
        annotations = [a for a in evaluator.cocoGt.catToImgs.get(category_id, []) if a in image_ids]
        gt = [a for a in evaluator.cocoGt.anns.values()
              if a['category_id'] == category_id and a['image_id'] in image_ids]
        row = dict(category_id=int(category_id), name=category['name'],
                   images_with_gt=len(set(annotations)), gt_annotations=len(gt),
                   noncrowd_gt=sum(not a.get('iscrowd', 0) for a in gt))
        row['areas'] = {name: dict(AP=mean_valid(precision[:, :, k, a, -1]),
                                   AR=mean_valid(recall[:, k, a, -1]))
                        for a, name in enumerate(params.areaRngLbl)}
        rows.append(row)
    Path(destination).write_text(json.dumps(dict(image_ids=sorted(int(x) for x in image_ids),
        max_dets=[int(x) for x in params.maxDets], iou_thresholds=params.iouThrs.tolist(), classes=rows),
        ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
