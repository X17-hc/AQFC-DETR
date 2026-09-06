"""
AI-TOD SOTA 切片推理脚本 (SAHI) - 修正版 v2
修复内容：
1. 回滚类别ID映射 (AP=0 的原因)
2. 加入全图推理 (Global Inference)，保证基础性能
3. 优化坐标映射逻辑
"""
import argparse
import json
import os
import time
from pathlib import Path
import sys

# 环境变量设置
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
np.float = float
np.int = int

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.ops import nms

import util.misc as utils
from datasets.coco import CocoDetection
from datasets.coco_eval import CocoEvaluator
from util.box_ops import box_cxcywh_to_xyxy
from util.slconfig import SLConfig
from main_aitod import get_args_parser

# ================= 配置区域 =================
# 切片大小：AI-TOD 微小目标建议使用 480 或 512
SLICE_SIZE = 480
# 重叠率：0.25
OVERLAP_RATIO = 0.25
# 置信度阈值：降低到 0.05 以提升召回率
CONF_THRESHOLD = 0.05
# NMS 阈值
NMS_THRESHOLD = 0.03
# 全图推理尺寸：与训练时的 max_size 保持一致 (通常 800 或 1333)
GLOBAL_SIZE = 1400
# ===========================================

def build_model_main(args):
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

def get_slices(image_pil, slice_h, slice_w, overlap_ratio=0.2):
    w, h = image_pil.size
    slices = []

    stride_h = int(slice_h * (1 - overlap_ratio))
    stride_w = int(slice_w * (1 - overlap_ratio))

    if w <= slice_w or h <= slice_h:
        return [] # 如果图太小，直接由全图推理处理

    y_steps = list(range(0, h - slice_h, stride_h))
    if (h - slice_h) not in y_steps:
        y_steps.append(h - slice_h)

    x_steps = list(range(0, w - slice_w, stride_w))
    if (w - slice_w) not in x_steps:
        x_steps.append(w - slice_w)

    for y in y_steps:
        for x in x_steps:
            box = (x, y, x + slice_w, y + slice_h)
            crop = image_pil.crop(box)
            slices.append({
                'crop': crop,
                'rect': [x, y, slice_w, slice_h]
            })

    return slices

@torch.no_grad()
def infer_image_with_slices(model, image_pil, device):
    w_orig, h_orig = image_pil.size

    # 1. 准备切片列表
    slices = get_slices(image_pil, SLICE_SIZE, SLICE_SIZE, OVERLAP_RATIO)

    # 2. 【关键】加入全图推理 (Global Context)
    # 将原图缩放到训练时的常见尺寸 (如 800x800) 进行一次整体预测
    # 这能找回大目标和被切片切断的目标，是保分的关键
    slices.append({
        'crop': image_pil.resize((GLOBAL_SIZE, GLOBAL_SIZE)),
        'rect': None  # 标记为全图
    })

    all_boxes = []
    all_scores = []
    all_labels = []

    model.eval()

    for slc in slices:
        crop = slc['crop']
        rect = slc['rect']

        # 预处理
        img_tensor = F.to_tensor(crop)
        img_tensor = F.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_tensor = img_tensor.to(device).unsqueeze(0)

        # 推理
        outputs = model(img_tensor)

        pred_logits = outputs['pred_logits'][0]
        pred_boxes = outputs['pred_boxes'][0]

        prob = pred_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(-1), 150) # 每个切片取 Top 150
        scores = topk_values
        topk_boxes_cat = topk_indexes // prob.shape[1]
        labels = topk_indexes % prob.shape[1]

        boxes_norm = pred_boxes[topk_boxes_cat]
        boxes_abs = box_cxcywh_to_xyxy(boxes_norm)

        crop_w, crop_h = crop.size

        # 坐标映射逻辑
        if rect is not None:
            # 切片模式：还原到原图绝对坐标
            boxes_abs[:, 0] *= crop_w
            boxes_abs[:, 2] *= crop_w
            boxes_abs[:, 1] *= crop_h
            boxes_abs[:, 3] *= crop_h

            off_x, off_y = rect[0], rect[1]
            boxes_abs[:, 0] += off_x
            boxes_abs[:, 2] += off_x
            boxes_abs[:, 1] += off_y
            boxes_abs[:, 3] += off_y
        else:
            # 全图模式：crop是resize后的图，需要映射回 w_orig, h_orig
            # 因为 boxes_norm 是归一化的 (0-1)，直接乘以原图尺寸即可还原
            boxes_abs[:, 0] *= w_orig
            boxes_abs[:, 2] *= w_orig
            boxes_abs[:, 1] *= h_orig
            boxes_abs[:, 3] *= h_orig

        keep = scores > CONF_THRESHOLD
        all_boxes.append(boxes_abs[keep])
        all_scores.append(scores[keep])
        all_labels.append(labels[keep])

    if len(all_boxes) == 0:
        return torch.empty((0, 4)), torch.empty(0), torch.empty(0)

    global_boxes = torch.cat(all_boxes, dim=0)
    global_scores = torch.cat(all_scores, dim=0)
    global_labels = torch.cat(all_labels, dim=0)

    # 全局 NMS
    final_boxes = []
    final_scores = []
    final_labels = []

    for cls_id in global_labels.unique():
        cls_mask = (global_labels == cls_id)
        cls_boxes = global_boxes[cls_mask]
        cls_scores = global_scores[cls_mask]

        keep_indices = nms(cls_boxes, cls_scores, NMS_THRESHOLD)

        final_boxes.append(cls_boxes[keep_indices])
        final_scores.append(cls_scores[keep_indices])
        final_labels.append(global_labels[cls_mask][keep_indices])

    if len(final_boxes) == 0:
        return torch.empty((0, 4)), torch.empty(0), torch.empty(0)

    return torch.cat(final_boxes), torch.cat(final_scores), torch.cat(final_labels)

def main(args):
    print(f"Loading config file from {args.config_file}")
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)

    if not getattr(args, 'use_ema', None): args.use_ema = False
    if not getattr(args, 'debug', None): args.debug = False

    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    print(f"Building model: {args.modelname}")
    model, _, _ = build_model_main(args)
    model.to(device)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if args.use_ema and 'ema_model' in checkpoint:
            print("Loading EMA weights...")
            model.load_state_dict(checkpoint['ema_model'])
        else:
            print("Loading standard weights...")
            model.load_state_dict(checkpoint['model'])
        print(f"Loaded checkpoint from {args.resume}")
    else:
        return

    model.eval()

    root = Path(args.coco_path)
    img_folder = root / "images/test/images"
    ann_file = root / "annotations" / "aitodv2_test.json"
    if not ann_file.exists(): ann_file = root / "annotations" / "val.json"

    print(f"Loading annotations from {ann_file}")
    dataset_val = CocoDetection(str(img_folder), str(ann_file), transforms=None, return_masks=False)

    print(f"Start SAHI + Global Inference...")
    print(f"Config: Slice={SLICE_SIZE}, Global={GLOBAL_SIZE}, Thresh={CONF_THRESHOLD}")

    coco_results = []

    for i in range(len(dataset_val)):
        image, target = dataset_val[i]
        image_id = target['image_id'].item()

        boxes, scores, labels = infer_image_with_slices(model, image, device)

        boxes_w = boxes[:, 2] - boxes[:, 0]
        boxes_h = boxes[:, 3] - boxes[:, 1]
        boxes[:, 2] = boxes_w
        boxes[:, 3] = boxes_h

        for box, score, label in zip(boxes, scores, labels):
            res = {
                "image_id": image_id,
                # 【回滚】去掉+1，改回原始逻辑
                "category_id": int(label.item()),
                "bbox": [round(b, 2) for b in box.tolist()],
                "score": float(score.item())
            }
            coco_results.append(res)

        if i % 100 == 0:
            print(f"Processed {i}/{len(dataset_val)}...")

    if len(coco_results) > 0:
        coco_true = dataset_val.coco
        coco_dt = coco_true.loadRes(coco_results)
        coco_eval = CocoEvaluator(coco_true, ['bbox'])
        coco_eval.coco_eval['bbox'].cocoDt = coco_dt
        coco_eval.coco_eval['bbox'].params.imgIds = sorted(coco_true.getImgIds())
        print("Evaluating...")
        coco_eval.coco_eval['bbox'].evaluate()
        coco_eval.coco_eval['bbox'].accumulate()
        coco_eval.coco_eval['bbox'].summarize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SAHI Inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)