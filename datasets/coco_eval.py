# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import io
import numpy as np
import torch

from aitodpycocotools.cocoeval import COCOeval
from aitodpycocotools.coco import COCO
import aitodpycocotools.mask as mask_util

from util.misc import all_gather


class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types, useCats=True):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)  # coco_gt是ground truth标注数据，使用deepcopy确保原始数据不被修改
        self.coco_gt = coco_gt

        self.iou_types = iou_types  # 为每种IoU类型创建独立的COCO_eval对象，这是评估的核心
        self.coco_eval = {}
        self.coco_eval25 = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)
            self.coco_eval[iou_type].useCats = useCats
            if iou_type in ('bbox', 'segm'):
                self.coco_eval25[iou_type] = COCOeval(coco_gt, iouType=iou_type)
                self.coco_eval25[iou_type].params.iouThrs = np.array([.25])

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}
        self.eval_imgs25 = {k: [] for k in self.coco_eval25}
        self.useCats = useCats  # useCats参数控制是否按类别进行评估，对类别不平衡问题尤为重要

    def update(self, predictions):  # 评估核心流程：准备结果→加载结果→设置参数→执行评估
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):  # 它重定向标准输出到/dev/null，避免pycocotools在评估过程中产生大量无用输出
                    coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            coco_eval.params.useCats = self.useCats
            img_ids, eval_imgs = evaluate(coco_eval)  # evaluate函数是重写的版本，支持分布式评估，这是与原始pycocotools的主要区别

            self.eval_imgs[iou_type].append(eval_imgs)
            if iou_type in self.coco_eval25:
                auxiliary = self.coco_eval25[iou_type]
                auxiliary.cocoDt = coco_dt
                auxiliary.params.imgIds = list(img_ids)
                auxiliary.params.useCats = self.useCats
                _, auxiliary_imgs = evaluate(auxiliary)
                self.eval_imgs25[iou_type].append(auxiliary_imgs)

    def synchronize_between_processes(self):  # 分布式评估机制，可以将各进程的评估结果合并
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)  # np.concatenate(..., 2)沿第三维拼接，保持评估结果的多维结构
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])
        for iou_type, evaluator in self.coco_eval25.items():
            images = np.concatenate(self.eval_imgs25[iou_type], 2)
            create_common_coco_eval(evaluator, self.img_ids, images)

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()
        for coco_eval in self.coco_eval25.values():
            # AP25 is independent: standard AP and oLRP keep the original .50:.95 grid.
            with contextlib.redirect_stdout(io.StringIO()):
                coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            if iou_type not in self.coco_eval25:
                coco_eval.summarize()
                continue
            original_summary = io.StringIO()
            with contextlib.redirect_stdout(original_summary):
                coco_eval.summarize()
            precision = self.coco_eval25[iou_type].eval['precision'][:, :, :, 0, -1]
            valid = precision[precision >= 0]
            coco_eval.stats[1] = float(valid.mean()) if valid.size else -1.
            # Restore the native AP/AR/area/maxDets/LRP table. Its AP25 row was
            # computed on the .50:.95 grid, so replace ONLY that displayed value
            # with the independent AP25 result; never change standard AP or LRP.
            for line in original_summary.getvalue().splitlines():
                if line.startswith('Average Precision') and 'IoU=0.25' in line:
                    line = line.rsplit('=', 1)[0] + f'= {coco_eval.stats[1]:0.3f}'
                print(line)

    def named_metrics(self, iou_type='bbox'):
        """Stable named fields; unavailable area/class metrics are JSON null, not scores."""
        names = ('AP', 'AP25', 'AP50', 'AP75', 'APvt', 'APt', 'APs', 'APm',
                 'AR1', 'AR100', 'AR1500', 'ARvt', 'ARt', 'ARs', 'ARm',
                 'oLRP', 'oLRP_loc', 'oLRP_FP', 'oLRP_FN')
        return {name: float(value) if np.isfinite(value) and value >= 0 else None
                for name, value in zip(names, self.coco_eval[iou_type].stats)}

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):  # 此方法将模型预测转换为COCO评估所需的标准格式
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:  # 对空预测的处理
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            if not isinstance(prediction["scores"], list):
                scores = prediction["scores"].tolist()
            else:
                scores = prediction["scores"]
            if not isinstance(prediction["labels"], list):
                labels = prediction["labels"].tolist()
            else:
                labels = prediction["labels"]

        
            try:
                coco_results.extend(
                    [
                        {
                            "image_id": original_id,
                            "category_id": labels[k],
                            "bbox": box,
                            "score": scores[k],
                        }
                        for k, box in enumerate(boxes)
                    ]
                )
            except:
                import ipdb; ipdb.set_trace()
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [  # RLE（Run-Length Encoding）是COCO分割标注的标准压缩格式
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)  # all_gather是分布式训练中的关键操作，收集所有进程的数据
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################


def evaluate(self):  # 此函数是对原始pycocotools.evaluate方法的重写，专为分布式环境优化
    '''
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    

    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]

    
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]

    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)

    return p.imgIds, evalImgs

#################################################################
# end of straight copy from pycocotools, just removing the prints
#################################################################
