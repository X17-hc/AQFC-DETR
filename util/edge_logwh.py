"""Log-width loss for refined boxes whose IoU is still below 0.60."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from util.box_ops import box_cxcywh_to_xyxy, box_iou


def low_iou_logwh_loss(outputs, targets, indices, num_boxes, threshold=0.60):
    """Smooth L1 of log(pred_wh / gt_wh) on matched refined boxes below the IoU cut.

    IoU is detached. A matched box at or above the cut contributes nothing, so
    the existing box loss keeps those edges where they are.
    """
    pred_boxes = outputs['pred_boxes']
    pieces = []
    for batch, (src, dst) in enumerate(indices):
        if src.numel() == 0:
            continue
        pred = pred_boxes[batch, src]
        truth = targets[batch]['boxes'][dst]
        with torch.no_grad():
            paired = box_iou(box_cxcywh_to_xyxy(pred.detach()), box_cxcywh_to_xyxy(truth.detach()))[0].diag()
            selected = paired < threshold
        with torch.autocast(device_type=pred.device.type, enabled=False):
            log_ratio = pred[:, 2:].float().clamp_min(1e-6).log() - truth[:, 2:].float().clamp_min(1e-6).log()
            per = F.smooth_l1_loss(log_ratio, torch.zeros_like(log_ratio), reduction='none').mean(-1)
        pieces.append(per * selected.to(dtype=per.dtype))
    if not pieces:
        return pred_boxes.float().reshape(-1)[:1].sum() * 0
    denom = num_boxes if torch.is_tensor(num_boxes) else pred_boxes.new_tensor(float(num_boxes))
    return torch.cat(pieces).sum() / denom.clamp_min(1).to(dtype=torch.float32)


def _wh_rel(pred, gt):
    pred_wh = (pred[2:] - pred[:2]).clamp(min=0)
    gt_wh = (gt[2:] - gt[:2]).clamp(min=1e-6)
    return float((pred_wh - gt_wh).abs().div(gt_wh).max())


def _best_box(det_boxes, det_scores, det_labels, gt_boxes, label):
    empty = gt_boxes.new_zeros(gt_boxes.shape[0])
    if det_boxes.numel() == 0:
        return None, empty
    keep = det_labels == label
    if not bool(keep.any()):
        return None, empty
    order = torch.argsort(det_scores[keep], descending=True)[:1500]
    chosen = det_boxes[keep][order]
    iou, _ = box_iou(chosen, gt_boxes)
    best_iou, best_index = iou.max(dim=0)
    return chosen[best_index], best_iou


class EdgeScreen:
    """Share of targets whose best same-class IoU is below 0.60, and their width error."""

    def __init__(self):
        self.n_gt = 0
        self.n_below = 0
        self.wh = []

    def add(self, det_boxes, det_scores, det_labels, gt_boxes, gt_labels):
        if gt_boxes.numel() == 0:
            return
        det_boxes = det_boxes.detach().float()
        det_scores = det_scores.detach().float()
        det_labels = det_labels.detach()
        gt_boxes = gt_boxes.detach().float()
        gt_labels = gt_labels.detach()
        for label in torch.unique(gt_labels).tolist():
            gt_index = gt_labels == label
            chosen, best = _best_box(
                det_boxes, det_scores, det_labels, gt_boxes[gt_index], int(label))
            selected = gt_boxes[gt_index]
            for row in range(selected.shape[0]):
                self.n_gt += 1
                if float(best[row]) >= 0.60:
                    continue
                self.n_below += 1
                if chosen is None:
                    continue
                self.wh.append(_wh_rel(chosen[row], selected[row]))

    def add_batch(self, results, targets):
        for result, target in zip(results, targets):
            gt = box_cxcywh_to_xyxy(target['boxes'].detach().float())
            height, width = target['orig_size'].detach().float().flatten()[:2]
            scale = torch.stack([width, height, width, height]).to(device=gt.device)
            self.add(result['boxes'], result['scores'], result['labels'], gt * scale, target['labels'])

    def summary(self):
        ordered = sorted(self.wh)
        median = None if not ordered else float(ordered[len(ordered) // 2])
        return {
            'n_gt': self.n_gt,
            'n_best_iou_below_060': self.n_below,
            'fraction_below_060': (self.n_below / self.n_gt) if self.n_gt else None,
            'wh_rel_median_below_060': median,
        }

    def merged_summary(self):
        import util.misc as utils
        packed = utils.all_gather({'n_gt': self.n_gt, 'n_below': self.n_below, 'wh': list(self.wh)})
        merged = EdgeScreen()
        for item in packed:
            merged.n_gt += int(item['n_gt'])
            merged.n_below += int(item['n_below'])
            merged.wh.extend(item['wh'])
        return merged.summary()

    def write(self, path):
        Path(path).write_text(json.dumps(self.merged_summary(), indent=2), encoding='utf-8')
