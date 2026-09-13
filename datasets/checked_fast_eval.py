"""Opt-in parity gate. Legacy LRP remains authoritative; all costs are counted."""
import copy
import time
import numpy as np
from .coco_eval import CocoEvaluator
from util.misc import all_gather


def require_fast():
    try:
        from faster_coco_eval_aitod import COCO, COCOeval_faster
    except ImportError as exc:
        raise RuntimeError('faster-coco-eval-aitod is unavailable; legacy evaluation is unchanged') from exc
    return COCO, COCOeval_faster


def compare_arrays(reference, candidate, tolerance=1e-6):
    if reference.shape != candidate.shape:
        raise ValueError(f'Evaluator shapes differ: {reference.shape} != {candidate.shape}')
    if not np.array_equal(reference < 0, candidate < 0):
        raise ValueError('Evaluator unavailable-value masks differ')
    if not np.array_equal(np.isfinite(reference), np.isfinite(candidate)):
        raise ValueError('Evaluator finite-value masks differ')
    valid = np.isfinite(reference) & (reference >= 0)
    difference = float(np.max(np.abs(reference[valid] - candidate[valid]))) if valid.any() else 0.
    if difference > tolerance:
        raise ValueError(f'Evaluator parity failed: maximum difference {difference} > {tolerance}')
    return difference


class CheckedFastEvaluator(CocoEvaluator):
    """Run both backends: validates AP tensors, does NOT claim an acceleration yet."""
    def __init__(self, coco_gt, iou_types, useCats=True):
        self.fast_types = require_fast()
        if tuple(iou_types) != ('bbox',):
            raise ValueError('Checked faster_aitod currently supports bbox only')
        super().__init__(coco_gt, iou_types, useCats)
        self.records = {}
        self.started = time.perf_counter()
        self.legacy_seconds = 0.

    def update(self, predictions):
        start = time.perf_counter()
        super().update(predictions)
        self.legacy_seconds += time.perf_counter()-start
        for image_id, prediction in predictions.items():
            self.records[int(image_id)] = self.prepare({image_id: prediction}, 'bbox')

    def synchronize_between_processes(self):
        super().synchronize_between_processes()
        merged = {}
        for records in all_gather(self.records):
            merged.update(records)
        self.records = merged

    def accumulate(self):
        start = time.perf_counter()
        super().accumulate()
        self.legacy_seconds += time.perf_counter()-start
        legacy_done = time.perf_counter()
        COCO, Evaluator = self.fast_types
        gt = COCO()
        gt.dataset = copy.deepcopy(self.coco_gt.dataset)
        gt.createIndex()
        records = [r for rows in self.records.values() for r in rows]
        if records:
            dt = gt.loadRes(records)
        else:
            dt = COCO()
            dt.dataset = {**copy.deepcopy(gt.dataset), 'annotations': []}
            dt.createIndex()
        differences = {}
        for name, reference in [('standard', self.coco_eval['bbox']),
                                ('ap25', self.coco_eval25['bbox'])]:
            fast = Evaluator(gt, dt, iouType='bbox')
            for field in ('imgIds', 'catIds', 'iouThrs', 'recThrs', 'maxDets',
                          'areaRng', 'areaRngLbl', 'useCats'):
                setattr(fast.params, field, copy.deepcopy(getattr(reference.params, field)))
            fast.evaluate()
            fast.accumulate()
            for field in ('precision', 'recall'):
                differences[f'{name}_{field}'] = compare_arrays(reference.eval[field], fast.eval[field])
        self.backend_report = dict(status='parity_passed', differences=differences,
            legacy_evaluation_seconds=self.legacy_seconds,
            extra_fast_seconds=time.perf_counter()-legacy_done,
            wall_seconds_since_evaluator_creation=time.perf_counter()-self.started,
            lrp_backend='legacy', execution='dual_backend_validation_not_acceleration')

    def summarize(self):
        super().summarize()
        print('Backend parity:', self.backend_report)
