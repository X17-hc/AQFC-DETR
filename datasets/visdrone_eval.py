"""COCO bbox validation proxy for converted VisDrone, not the official DET toolkit.

No AI-TOD tiny-area or LRP naming is reused. Original category IDs are retained.
"""
import copy
import numpy as np
from pycocotools.cocoeval import COCOeval
from .coco_eval import CocoEvaluator


class VisDroneCocoEvaluator(CocoEvaluator):
    def __init__(self, coco_gt, iou_types, useCats=True):
        self.coco_gt = copy.deepcopy(coco_gt)
        self.iou_types = iou_types
        self.useCats = useCats
        self.coco_eval = {kind: COCOeval(self.coco_gt, iouType=kind) for kind in iou_types}
        self.coco_eval25 = {}
        self.img_ids = []
        self.eval_imgs = {kind: [] for kind in iou_types}
        self.eval_imgs25 = {}
        # COCO ignores regions via iscrowd; preserve input source and operate on a copy.
        for annotation in self.coco_gt.dataset.get('annotations', []):
            if annotation.get('ignore', 0):
                annotation['iscrowd'] = 1

    def summarize(self):
        print('VisDrone COCO bbox proxy: maxDets=100; not the official VisDrone DET score.')
        for evaluator in self.coco_eval.values():
            evaluator.summarize()

    def named_metrics(self, iou_type='bbox'):
        names = ('AP', 'AP50', 'AP75', 'APs', 'APm', 'APl',
                 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl')
        return {key: float(value) if np.isfinite(value) and value >= 0 else None
                for key, value in zip(names, self.coco_eval[iou_type].stats)}
