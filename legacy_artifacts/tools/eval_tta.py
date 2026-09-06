# """
# Test-Time Augmentation evaluation script.
# Usage:
#     first:
#     --output_dir logs/tta_eval
#     -c config/DQ_5scale.py
#     --coco_path /workspace/DQDetr/data/path/AITODv2
#     --checkpoint
#     "/workspace/DQDetr/logs/DQDETR_ver4/0325-接着0324的12个epoch/checkpoint0011.pth"
#     --scales 736,800,864
#     --flip
#     --wbf_iou_thr 0.55
#     --wbf_skip_thr 0.001
# """
#
# import argparse
# import json
# import os
# import sys
# import copy
# from pathlib import Path
#
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, SequentialSampler
#
# # 尝试导入 ensemble_boxes，若未安装则给出提示
# try:
#     from ensemble_boxes import weighted_boxes_fusion
#     HAS_WBF = True
# except ImportError:
#     print("[WARNING] ensemble_boxes 未安装，将使用 soft-NMS 替代 WBF")
#     print("  安装命令: pip install ensemble-boxes --break-system-packages")
#     HAS_WBF = False
#
# from util.slconfig import SLConfig, DictAction
# from util.misc import nested_tensor_from_tensor_list
# import datasets.transforms as T
# from datasets import build_dataset, get_coco_api_from_dataset
# from datasets.coco_eval import CocoEvaluator
# import util.misc as utils
#
#
# # ─────────────────────────────────────────────
# # 1. 命令行参数
# # ─────────────────────────────────────────────
#
# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-c', '--config_file', type=str, required=True)
#     parser.add_argument('--checkpoint', type=str, required=True)
#     parser.add_argument('--coco_path', type=str, required=True)
#     parser.add_argument('--output_dir', type=str, default='logs/tta_eval')
#     parser.add_argument('--dataset_file', type=str, default='aitodv2')
#     parser.add_argument('--device', type=str, default='cuda')
#     parser.add_argument('--num_workers', type=int, default=4)
#     parser.add_argument('--options', nargs='+', action=DictAction, default=None)
#     # TTA 控制
#     parser.add_argument('--scales', nargs='+', type=int,
#                         default=[736, 800, 864],
#                         help='推理尺度列表，推荐 [736, 800, 864]')
#     parser.add_argument('--flip', action='store_true', default=True,
#                         help='是否加入水平翻转增强')
#     parser.add_argument('--wbf_iou_thr', type=float, default=0.55,
#                         help='WBF 合并的 IoU 阈值，小目标建议 0.5~0.6')
#     parser.add_argument('--wbf_skip_thr', type=float, default=0.001,
#                         help='低于该分数的框直接丢弃')
#     return parser.parse_args()
#
#
# # ─────────────────────────────────────────────
# # 2. 构建单尺度 transform（不用随机裁剪）
# # ─────────────────────────────────────────────
#
# def build_val_transform(scale: int, max_size: int = 1333, flip: bool = False):
#     """返回一个确定性的 val transform，支持指定尺度和水平翻转"""
#     normalize = T.Compose([
#         T.ToTensor(),
#         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#     ops = []
#     if flip:
#         # 确定性翻转（p=1.0）
#         ops.append(T.RandomHorizontalFlip(p=1.0))
#     ops.append(T.RandomResize([scale], max_size=max_size))
#     ops.append(normalize)
#     return T.Compose(ops)
#
#
# # ─────────────────────────────────────────────
# # 3. 单次推理，返回 {image_id: {'boxes','scores','labels'}} 格式
# # ─────────────────────────────────────────────
#
# @torch.no_grad()
# def run_inference_single_aug(model, data_loader, device, scale, flip,
#                              num_select_override=900):
#     """
#     对整个验证集跑一次推理。
#     返回 dict: image_id -> {'boxes': tensor(N,4) xyxy 绝对坐标,
#                             'scores': tensor(N,),
#                             'labels': tensor(N,)}
#     """
#     model.eval()
#     results = {}
#
#     for samples, targets in data_loader:
#         samples = samples.to(device)
#         targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
#                     for k, v in t.items()} for t in targets]
#
#         outputs = model(samples)
#
#         # 取 num_select — 强制最低 900
#         ns = outputs.get('num_select', None)
#         if ns is None:
#             num_select = num_select_override
#         elif isinstance(ns, torch.Tensor):
#             num_select = max(int(ns.max().item()), num_select_override)
#         else:
#             num_select = max(int(ns), num_select_override)
#         num_select = min(num_select, 1500)
#
#         orig_sizes = torch.stack([t['orig_size'] for t in targets], 0)
#         batch_results = postprocess_outputs(
#             outputs, orig_sizes, num_select, flip=flip)
#
#         for target, res in zip(targets, batch_results):
#             image_id = int(target['image_id'].item())
#             results[image_id] = res
#
#     return results
#
#
# def postprocess_outputs(outputs, orig_target_sizes, num_select, flip=False):
#     """将模型输出转为绝对坐标 xyxy 格式的框列表"""
#     out_logits = outputs['pred_logits']   # [bs, nq, nc]
#     out_bbox   = outputs['pred_boxes']    # [bs, nq, 4] cxcywh normalized
#
#     bs = out_logits.shape[0]
#     prob = out_logits.sigmoid()
#
#     # topk
#     topk_val, topk_idx = torch.topk(
#         prob.view(bs, -1), num_select, dim=1)
#     scores = topk_val
#     topk_boxes = topk_idx // out_logits.shape[2]
#     labels = topk_idx % out_logits.shape[2]
#
#     # cx cy w h -> x1 y1 x2 y2
#     boxes = box_cxcywh_to_xyxy(out_bbox)
#     boxes = torch.gather(boxes, 1,
#                          topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
#
#     # 还原到原始图像尺寸
#     img_h, img_w = orig_target_sizes.unbind(1)
#     scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
#     boxes = boxes * scale_fct[:, None, :]
#
#     # 如果是翻转推理，把框 x 坐标翻回来
#     if flip:
#         x1 = boxes[..., 0].clone()
#         x2 = boxes[..., 2].clone()
#         boxes[..., 0] = img_w.unsqueeze(1) - x2
#         boxes[..., 2] = img_w.unsqueeze(1) - x1
#
#     batch_out = []
#     for s, l, b in zip(scores, labels, boxes):
#         batch_out.append({'scores': s, 'labels': l, 'boxes': b})
#     return batch_out
#
#
# def box_cxcywh_to_xyxy(x):
#     cx, cy, w, h = x.unbind(-1)
#     return torch.stack([cx - 0.5*w, cy - 0.5*h,
#                         cx + 0.5*w, cy + 0.5*h], dim=-1)
#
#
# # ─────────────────────────────────────────────
# # 4. WBF 合并多次推理结果
# # ─────────────────────────────────────────────
#
# def merge_predictions_wbf(all_preds_list, img_sizes,
#                            iou_thr=0.55, skip_box_thr=0.001):
#     """
#     all_preds_list: List[dict]  每个元素是一次增强的全部结果
#                     {image_id: {'boxes','scores','labels'}}
#     img_sizes:      {image_id: (H, W)}
#     返回:           {image_id: {'boxes','scores','labels'}}
#     """
#     all_image_ids = set()
#     for preds in all_preds_list:
#         all_image_ids.update(preds.keys())
#
#     merged = {}
#     for img_id in all_image_ids:
#         H, W = img_sizes[img_id]
#
#         boxes_list, scores_list, labels_list = [], [], []
#         for preds in all_preds_list:
#             if img_id not in preds:
#                 continue
#             res = preds[img_id]
#             b = res['boxes'].cpu().numpy()    # [N,4] xyxy abs
#             s = res['scores'].cpu().numpy()   # [N,]
#             l = res['labels'].cpu().numpy()   # [N,]
#
#             # WBF 需要归一化坐标 [0,1]
#             b[:, [0, 2]] /= W
#             b[:, [1, 3]] /= H
#             b = np.clip(b, 0., 1.)
#
#             boxes_list.append(b.tolist())
#             scores_list.append(s.tolist())
#             labels_list.append(l.tolist())
#
#         if not boxes_list:
#             merged[img_id] = {'boxes': torch.zeros(0,4),
#                               'scores': torch.zeros(0),
#                               'labels': torch.zeros(0, dtype=torch.long)}
#             continue
#
#         if HAS_WBF:
#             boxes_wbf, scores_wbf, labels_wbf = weighted_boxes_fusion(
#                 boxes_list, scores_list, labels_list,
#                 iou_thr=iou_thr,
#                 skip_box_thr=skip_box_thr,
#                 weights=None        # 等权重
#             )
#         else:
#             # fallback: 简单拼接 + score-NMS
#             boxes_wbf  = np.concatenate([np.array(b) for b in boxes_list], 0)
#             scores_wbf = np.concatenate([np.array(s) for s in scores_list], 0)
#             labels_wbf = np.concatenate([np.array(l) for l in labels_list], 0)
#
#         # 还原回绝对坐标
#         boxes_abs = boxes_wbf.copy()
#         boxes_abs[:, [0, 2]] *= W
#         boxes_abs[:, [1, 3]] *= H
#
#         merged[img_id] = {
#             'boxes':  torch.from_numpy(boxes_abs).float(),
#             'scores': torch.from_numpy(scores_wbf).float(),
#             'labels': torch.from_numpy(labels_wbf.astype(np.int64)).long(),
#         }
#     return merged
#
#
# # ─────────────────────────────────────────────
# # 5. 主流程
# # ─────────────────────────────────────────────
#
# def main():
#     args = get_args()
#     os.makedirs(args.output_dir, exist_ok=True)
#
#     # 加载 config
#     cfg = SLConfig.fromfile(args.config_file)
#     if args.options:
#         cfg.merge_from_dict(args.options)
#     cfg_dict = cfg._cfg_dict.to_dict()
#     for k, v in cfg_dict.items():
#         if not hasattr(args, k):
#             setattr(args, k, v)
#
#     # 补全必要属性
#     for attr, default in [('masks', False), ('fix_size', False),
#                           ('strong_aug', False), ('distributed', False),
#                           ('amp', False), ('save_results', False),
#                           ('debug', False), ('rank', 0), ('local_rank', 0)]:
#         if not hasattr(args, attr):
#             setattr(args, attr, default)
#
#     device = torch.device(args.device)
#
#     # 构建模型
#     from main_aitod import build_model_main
#     model, criterion, postprocessors = build_model_main(args)
#     model.to(device)
#     model.eval()
#
#     # 加载 checkpoint
#     ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
#     state = ckpt.get('ema_model', ckpt.get('model', ckpt))
#     if hasattr(state, 'items'):
#         from util.misc import clean_state_dict
#         model.load_state_dict(clean_state_dict(state), strict=False)
#     print(f"[TTA] Loaded checkpoint: {args.checkpoint}")
#
#     # 确定所有 TTA 变体
#     # (scale, flip)
#     tta_variants = []
#     for scale in args.scales:
#         tta_variants.append((scale, False))         # 正常
#         if args.flip:
#             tta_variants.append((scale, True))      # 翻转
#
#     print(f"[TTA] Variants: {tta_variants}")
#
#     # 构建基础数据集（只用来获取标注和 image_id→size 映射）
#     args_for_data = copy.deepcopy(args)
#     dataset_base = build_dataset(image_set='test', args=args_for_data)
#     base_ds = get_coco_api_from_dataset(dataset_base)
#
#     # 收集 image_id → 原始尺寸的映射
#     img_sizes = {}
#     for img_info in base_ds.dataset['images']:
#         img_sizes[img_info['id']] = (img_info['height'], img_info['width'])
#
#     # ── 对每个 TTA 变体跑推理 ──
#     all_preds_list = []
#     for scale, flip in tta_variants:
#         tag = f"scale{scale}" + ("_flip" if flip else "")
#         print(f"[TTA] Running augmentation: {tag}")
#
#         # 临时替换数据集的 transform
#         dataset_aug = build_dataset(image_set='test', args=copy.deepcopy(args))
#         dataset_aug._transforms = build_val_transform(scale, flip=flip)
#
#         sampler = SequentialSampler(dataset_aug)
#         loader = DataLoader(
#             dataset_aug, batch_size=1, sampler=sampler,
#             num_workers=args.num_workers,
#             collate_fn=utils.collate_fn,
#             pin_memory=True
#         )
#
#         preds = run_inference_single_aug(
#             model, loader, device,
#             scale=scale, flip=flip,
#             num_select_override=900
#         )
#         all_preds_list.append(preds)
#         print(f"[TTA]   → {len(preds)} images predicted")
#
#     # ── WBF 合并 ──
#     print("[TTA] Merging predictions with WBF ...")
#     merged = merge_predictions_wbf(
#         all_preds_list, img_sizes,
#         iou_thr=args.wbf_iou_thr,
#         skip_box_thr=args.wbf_skip_thr
#     )
#
#     # ── COCO 评估 ──
#     print("[TTA] Running COCO evaluation ...")
#     coco_evaluator = CocoEvaluator(base_ds, ['bbox'], useCats=True)
#
#     # 转换为 coco evaluator 格式
#     # boxes 需要是 xyxy 格式，evaluator 内部会转 xywh
#     coco_evaluator.update(merged)
#     coco_evaluator.synchronize_between_processes()
#     coco_evaluator.accumulate()
#     coco_evaluator.summarize()
#
#     # 保存合并结果
#     out_path = os.path.join(args.output_dir, 'tta_merged_results.json')
#     save_list = []
#     for img_id, res in merged.items():
#         for box, score, label in zip(
#                 res['boxes'].tolist(),
#                 res['scores'].tolist(),
#                 res['labels'].tolist()):
#             x1, y1, x2, y2 = box
#             save_list.append({
#                 'image_id': img_id,
#                 'category_id': int(label),
#                 'bbox': [x1, y1, x2 - x1, y2 - y1],
#                 'score': float(score)
#             })
#     with open(out_path, 'w') as f:
#         json.dump(save_list, f)
#     print(f"[TTA] Saved merged results to {out_path}")
#     print("[TTA] Done.")
#
#
# if __name__ == '__main__':
#     main()
#


"""
Test-Time Augmentation evaluation script for DQ-DETR on AI-TOD-v2.

Usage example (ver6, 3 scales):
    python eval_tta.py \
        -c config/DQ_5scale_v6.py \
        --checkpoint logs/DQDETR_ver6_CopyPaste/0330-NewRecord/checkpoint0023.pth \
        --coco_path /workspace/DQDetr/data/path/AITODv2 \
        --dataset_file aitodv2 \
        --output_dir logs/tta_eval_ver6 \
        --scales 736 800 864 \
        --flip \
        --wbf_iou_thr 0.55 \
        --wbf_skip_thr 0.001

Usage example (5 scales, 10 variants):
        --scales 704 736 800 864 896 \
        --wbf_iou_thr 0.58   (更多变体时用稍高的阈值防止过度合并)

关于 num_classes=9 与 8 个 LRP 阈值的关系：
  AI-TOD-v2 有 8 个前景类 (index 0-7)。
  DQ-DETR 输出 9 个 logit 维度：前 8 维是前景类，第 9 维 (index 8)
  是 DETR 风格的隐式背景位，SetCriterion 用 num_classes=9 作为
  no-object 的 target label，它从未作为真实前景被训练，sigmoid 分数
  始终接近 0。LRP 只对 8 个前景类报告最优阈值，这是正确的。

Trick A 正确实现：CLASS_OPTIMAL_THRESHOLDS 必须有 9 个值：
  - 前 8 个 = 8 个前景类的 LRP 最优阈值（每次训练后从评估结果更新）
  - 第 9 个 = 1.0  ← 背景位固定为 1.0，禁止改成 0 或小值！
    原因：背景位 sigmoid 分数接近 0，若阈值为 0.1 则归一化后变成 0/0.1=0
    (仍然是 0)，看似无害；但若分子有微小数值如 0.05，则 0.05/0.01=5，
    直接入选 topk，导致大量虚假框（LRP FP 暴增的根本原因）。
    设为 1.0 则 0.05/1.0=0.05，保持压制。
"""

import argparse
import json
import os
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torchvision.ops import nms as torchvision_nms

try:
    from ensemble_boxes import weighted_boxes_fusion
    HAS_WBF = True
except ImportError:
    print("[WARNING] ensemble_boxes 未安装，将使用简单拼接替代 WBF")
    print("  安装命令: pip install ensemble-boxes --break-system-packages")
    HAS_WBF = False

from util.slconfig import SLConfig, DictAction
import datasets.transforms as T
from datasets import build_dataset, get_coco_api_from_dataset
from datasets.coco_eval import CocoEvaluator
import util.misc as utils


# ─────────────────────────────────────────────
# Trick A：类别自适应推理阈值
#
# ★ 使用新模型前，把单模型评估输出最后一行
#   "Class-specific LRP-Optimal Thresholds" 的 8 个数填到前 8 位。
#   第 9 位固定为 1.0，不要修改。
#
# 当前：ver6 的阈值
#   来源：[0.384 0.425 0.395 0.393 0.419 0.388 0.405 0.330]
# ─────────────────────────────────────────────
CLASS_OPTIMAL_THRESHOLDS = torch.tensor(
    [0.384, 0.425, 0.395, 0.393, 0.419, 0.388, 0.405, 0.330,
     1.0],   # ← index 8：背景位，固定 1.0，严禁修改
    dtype=torch.float32
)


# ─────────────────────────────────────────────
# 1. 命令行参数
# ─────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(
        description='TTA evaluation for DQ-DETR on AI-TOD-v2')
    parser.add_argument('-c', '--config_file', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--coco_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='logs/tta_eval')
    parser.add_argument('--dataset_file', type=str, default='aitodv2')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--options', nargs='+', action=DictAction, default=None)

    # TTA 变体
    parser.add_argument('--scales', nargs='+', type=int,
                        default=[736, 800, 864])
    parser.add_argument('--flip', action='store_true', default=True)

    # WBF
    parser.add_argument('--wbf_iou_thr', type=float, default=0.55)
    parser.add_argument('--wbf_skip_thr', type=float, default=0.001)

    # Trick A
    parser.add_argument('--use_cls_threshold', action='store_true', default=True)
    parser.add_argument('--no_cls_threshold', dest='use_cls_threshold',
                        action='store_false')

    # Trick F
    parser.add_argument('--use_cls_nms', action='store_true', default=True)
    parser.add_argument('--no_cls_nms', dest='use_cls_nms',
                        action='store_false')
    parser.add_argument('--cls_nms_iou_thr', type=float, default=0.50)

    # num_select 下限
    parser.add_argument('--num_select_override', type=int, default=900)

    return parser.parse_args()


# ─────────────────────────────────────────────
# 2. 构建单尺度 transform
# ─────────────────────────────────────────────

def build_val_transform(scale: int, max_size: int = 1333, flip: bool = False):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ops = []
    if flip:
        ops.append(T.RandomHorizontalFlip(p=1.0))
    ops.append(T.RandomResize([scale], max_size=max_size))
    ops.append(normalize)
    return T.Compose(ops)


# ─────────────────────────────────────────────
# 3. 坐标转换
# ─────────────────────────────────────────────

def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - 0.5*w, cy - 0.5*h,
                        cx + 0.5*w, cy + 0.5*h], dim=-1)


# ─────────────────────────────────────────────
# 4. 后处理（含 Trick A）
# ─────────────────────────────────────────────

def postprocess_outputs(outputs, orig_target_sizes, num_select,
                        flip=False, use_cls_threshold=True):
    """
    将模型输出转为绝对坐标 xyxy 格式的预测框列表。

    Trick A：
      - 把 prob 除以各类别 LRP 最优阈值，在归一化空间里做 topk
      - topk 选完后，scores 换回原始 prob（保持评估语义不变）
      - CLASS_OPTIMAL_THRESHOLDS[8] = 1.0 确保背景位不被放大
    """
    out_logits = outputs['pred_logits']   # [bs, nq, nc]，nc=9
    out_bbox   = outputs['pred_boxes']    # [bs, nq, 4] cxcywh normalized

    bs     = out_logits.shape[0]
    nc     = out_logits.shape[2]          # 9
    device = out_logits.device

    prob = out_logits.sigmoid()           # [bs, nq, 9]

    if use_cls_threshold:
        thr = CLASS_OPTIMAL_THRESHOLDS.to(device)   # [9]
        if thr.shape[0] != nc:
            # 长度仍不匹配：打印明确错误信息，回退到原始 prob
            print(
                f"\n[ERROR] CLASS_OPTIMAL_THRESHOLDS 长度 ({thr.shape[0]}) "
                f"!= pred_logits nc ({nc})。\n"
                f"  正确做法：前 {nc-1} 位填 LRP 阈值，第 {nc} 位固定 1.0，"
                f"共 {nc} 个元素。\n"
                f"  已自动关闭 Trick A。"
            )
            prob_for_topk = prob
        else:
            # clamp 下限 0.05，防止极小阈值导致数值爆炸
            # thr[8]=1.0 不受影响
            thr_safe      = thr.clamp(min=0.05)
            prob_for_topk = prob / thr_safe.view(1, 1, nc)  # [bs, nq, nc]
    else:
        prob_for_topk = prob

    # topk 选取（在归一化空间）
    _, topk_idx = torch.topk(prob_for_topk.view(bs, -1), num_select, dim=1)

    topk_boxes_idx = topk_idx // nc
    labels         = topk_idx % nc

    # 用原始 prob 取分数
    scores = torch.gather(prob.view(bs, -1), 1, topk_idx)

    # 坐标处理
    boxes = box_cxcywh_to_xyxy(out_bbox)
    boxes = torch.gather(
        boxes, 1,
        topk_boxes_idx.unsqueeze(-1).repeat(1, 1, 4))

    img_h, img_w = orig_target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]

    if flip:
        x1 = boxes[..., 0].clone()
        x2 = boxes[..., 2].clone()
        boxes[..., 0] = img_w.unsqueeze(1) - x2
        boxes[..., 2] = img_w.unsqueeze(1) - x1

    return [{'scores': s, 'labels': l, 'boxes': b}
            for s, l, b in zip(scores, labels, boxes)]


# ─────────────────────────────────────────────
# 5. Trick F：WBF 后 per-class NMS
# ─────────────────────────────────────────────

def apply_per_class_nms(boxes_abs, scores_arr, labels_arr,
                        iou_threshold=0.50):
    if len(scores_arr) == 0:
        return boxes_abs, scores_arr, labels_arr

    boxes_t  = torch.from_numpy(boxes_abs).float()
    scores_t = torch.from_numpy(np.asarray(scores_arr)).float()
    labels_t = torch.from_numpy(np.asarray(labels_arr).astype(np.int64)).long()

    keep_indices = []
    for cls in labels_t.unique():
        mask       = (labels_t == cls)
        keep       = torchvision_nms(boxes_t[mask], scores_t[mask],
                                     iou_threshold)
        keep_indices.append(torch.where(mask)[0][keep])

    if not keep_indices:
        return boxes_abs, scores_arr, labels_arr

    keep_all = torch.cat(keep_indices)
    order    = scores_t[keep_all].argsort(descending=True)
    keep_all = keep_all[order]

    return (boxes_t[keep_all].numpy(),
            scores_t[keep_all].numpy(),
            labels_t[keep_all].numpy())


# ─────────────────────────────────────────────
# 6. 单次推理
# ─────────────────────────────────────────────

@torch.no_grad()
def run_inference_single_aug(model, data_loader, device, flip,
                             num_select_override=900,
                             use_cls_threshold=True):
    model.eval()
    results = {}

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]

        outputs = model(samples)

        ns = outputs.get('num_select', None)
        if ns is None:
            num_select = num_select_override
        elif isinstance(ns, torch.Tensor):
            num_select = max(int(ns.max().item()), num_select_override)
        else:
            num_select = max(int(ns), num_select_override)
        num_select = min(num_select, 1500)

        orig_sizes    = torch.stack([t['orig_size'] for t in targets], 0)
        batch_results = postprocess_outputs(
            outputs, orig_sizes, num_select,
            flip=flip,
            use_cls_threshold=use_cls_threshold)

        for target, res in zip(targets, batch_results):
            results[int(target['image_id'].item())] = res

    return results


# ─────────────────────────────────────────────
# 7. WBF 合并
# ─────────────────────────────────────────────

def merge_predictions_wbf(all_preds_list, img_sizes,
                           iou_thr=0.55, skip_box_thr=0.001,
                           use_cls_nms=True, cls_nms_iou_thr=0.50):
    all_image_ids = set()
    for preds in all_preds_list:
        all_image_ids.update(preds.keys())

    merged = {}
    for img_id in all_image_ids:
        H, W = img_sizes[img_id]

        boxes_list, scores_list, labels_list = [], [], []
        for preds in all_preds_list:
            if img_id not in preds:
                continue
            res = preds[img_id]
            b = res['boxes'].cpu().numpy().copy()
            s = res['scores'].cpu().numpy()
            l = res['labels'].cpu().numpy()

            b[:, [0, 2]] /= W
            b[:, [1, 3]] /= H
            b = np.clip(b, 0., 1.)

            boxes_list.append(b.tolist())
            scores_list.append(s.tolist())
            labels_list.append(l.tolist())

        if not boxes_list:
            merged[img_id] = {
                'boxes':  torch.zeros(0, 4),
                'scores': torch.zeros(0),
                'labels': torch.zeros(0, dtype=torch.long),
            }
            continue

        if HAS_WBF:
            boxes_wbf, scores_wbf, labels_wbf = weighted_boxes_fusion(
                boxes_list, scores_list, labels_list,
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr,
                weights=None,
            )
        else:
            boxes_wbf  = np.concatenate([np.array(b) for b in boxes_list])
            scores_wbf = np.concatenate([np.array(s) for s in scores_list])
            labels_wbf = np.concatenate([np.array(l) for l in labels_list])

        boxes_abs = boxes_wbf.copy()
        boxes_abs[:, [0, 2]] *= W
        boxes_abs[:, [1, 3]] *= H

        if use_cls_nms and len(scores_wbf) > 0:
            boxes_abs, scores_wbf, labels_wbf = apply_per_class_nms(
                boxes_abs, scores_wbf, labels_wbf,
                iou_threshold=cls_nms_iou_thr)

        merged[img_id] = {
            'boxes':  torch.from_numpy(boxes_abs).float(),
            'scores': torch.from_numpy(np.asarray(scores_wbf)).float(),
            'labels': torch.from_numpy(
                np.asarray(labels_wbf).astype(np.int64)).long(),
        }

    return merged


# ─────────────────────────────────────────────
# 8. 主流程
# ─────────────────────────────────────────────

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = SLConfig.fromfile(args.config_file)
    if args.options:
        cfg.merge_from_dict(args.options)
    for k, v in cfg._cfg_dict.to_dict().items():
        if not hasattr(args, k):
            setattr(args, k, v)

    for attr, default in [
        ('masks', False), ('fix_size', False), ('strong_aug', False),
        ('distributed', False), ('amp', False), ('save_results', False),
        ('debug', False), ('rank', 0), ('local_rank', 0),
    ]:
        if not hasattr(args, attr):
            setattr(args, attr, default)

    device = torch.device(args.device)

    from main_aitod import build_model_main
    model, _, _ = build_model_main(args)
    model.to(device)
    model.eval()

    ckpt  = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = ckpt.get('ema_model', ckpt.get('model', ckpt))
    if hasattr(state, 'items'):
        from util.misc import clean_state_dict
        model.load_state_dict(clean_state_dict(state), strict=False)

    # 确定 TTA 变体
    tta_variants = []
    for scale in args.scales:
        tta_variants.append((scale, False))
        if args.flip:
            tta_variants.append((scale, True))

    print(f"[TTA] Checkpoint      : {args.checkpoint}")
    print(f"[TTA] Variants        : {len(tta_variants)} → {tta_variants}")
    print(f"[TTA] Trick A         : {args.use_cls_threshold}")
    print(f"[TTA] Trick F         : {args.use_cls_nms} "
          f"(nms_iou={args.cls_nms_iou_thr})")
    print(f"[TTA] WBF iou_thr     : {args.wbf_iou_thr}")
    print(f"[TTA] WBF wbf_skip_thr: {args.wbf_skip_thr}")
    if args.use_cls_threshold:
        print(f"[TTA] Thresholds (9)  : {CLASS_OPTIMAL_THRESHOLDS.tolist()}")

    # 基础数据集
    dataset_base = build_dataset(image_set='test', args=copy.deepcopy(args))
    base_ds      = get_coco_api_from_dataset(dataset_base)
    img_sizes    = {img['id']: (img['height'], img['width'])
                    for img in base_ds.dataset['images']}

    # 逐变体推理
    all_preds_list = []
    for i, (scale, flip) in enumerate(tta_variants):
        tag = f"scale{scale}" + ("_flip" if flip else "")
        print(f"\n[TTA] [{i+1}/{len(tta_variants)}] {tag}")

        dataset_aug   = build_dataset(image_set='test', args=copy.deepcopy(args))
        new_transform = build_val_transform(scale, flip=flip)

        if hasattr(dataset_aug, '_transforms'):
            dataset_aug._transforms = new_transform
        elif (hasattr(dataset_aug, 'dataset') and
              hasattr(dataset_aug.dataset, '_transforms')):
            dataset_aug.dataset._transforms = new_transform
        else:
            print(f"  [WARNING] 无法替换 transform，跳过 {tag}")
            continue

        loader = DataLoader(
            dataset_aug, batch_size=1,
            sampler=SequentialSampler(dataset_aug),
            num_workers=args.num_workers,
            collate_fn=utils.collate_fn,
            pin_memory=True,
        )

        preds = run_inference_single_aug(
            model, loader, device,
            flip=flip,
            num_select_override=args.num_select_override,
            use_cls_threshold=args.use_cls_threshold,
        )
        all_preds_list.append(preds)
        # 在推理循环结束后，WBF 之前，保存各变体预测供参数搜索复用
        variants_save_path = os.path.join(args.output_dir, 'variants_raw.pt')
        # 把 tensor 转为可序列化的格式
        save_variants = []
        for preds_dict in all_preds_list:
            serialized = {}
            for img_id, res in preds_dict.items():
                serialized[img_id] = {
                    'boxes':  res['boxes'].cpu(),
                    'scores': res['scores'].cpu(),
                    'labels': res['labels'].cpu(),
                }
            save_variants.append(serialized)
        torch.save(save_variants, variants_save_path)
        print(f"[TTA] Saved raw variants → {variants_save_path}")
        print(f"  → {len(preds)} images done")

    # WBF 合并
    print(f"\n[TTA] Merging {len(all_preds_list)} variant(s) ...")
    merged = merge_predictions_wbf(
        all_preds_list, img_sizes,
        iou_thr=args.wbf_iou_thr,
        skip_box_thr=args.wbf_skip_thr,
        use_cls_nms=args.use_cls_nms,
        cls_nms_iou_thr=args.cls_nms_iou_thr,
    )

    # COCO 评估
    print("[TTA] Running COCO evaluation ...")
    coco_evaluator = CocoEvaluator(base_ds, ['bbox'], useCats=True)
    coco_evaluator.update(merged)
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    # 保存预测结果
    out_path  = os.path.join(args.output_dir, 'tta_merged_results.json')
    save_list = []
    for img_id, res in merged.items():
        for box, score, label in zip(
                res['boxes'].tolist(),
                res['scores'].tolist(),
                res['labels'].tolist()):
            x1, y1, x2, y2 = box
            save_list.append({
                'image_id':    img_id,
                'category_id': int(label),
                'bbox':        [x1, y1, x2 - x1, y2 - y1],
                'score':       float(score),
            })
    with open(out_path, 'w') as f:
        json.dump(save_list, f)
    print(f"[TTA] Saved → {out_path}")
    print("[TTA] Done.")


if __name__ == '__main__':
    main()
