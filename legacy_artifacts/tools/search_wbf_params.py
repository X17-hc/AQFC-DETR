# # # search_wbf_params.py
# # """
# # 加载已保存的变体预测，快速枚举 WBF 参数组合。
# # 不需要重新跑模型推理，每组参数只需几秒。
# #
# # 使用方法：
# #     python search_wbf_params.py \
# #         --variants_path logs/tta_eval_ver6_clean/variants_raw.pt \
# #         --ann_file /workspace/DQDetr/data/path/AITODv2/annotations/aitodv2_test.json \
# #         --dataset_file aitodv2 \
# #         --coco_path /workspace/DQDetr/data/path/AITODv2
# # """
# # import argparse, json, os, copy
# # import numpy as np
# # import torch
# # from ensemble_boxes import weighted_boxes_fusion
# # from torchvision.ops import nms as torchvision_nms
# #
# # # ── 复用 eval_tta.py 里的函数 ──
# # def apply_per_class_nms(boxes_abs, scores_arr, labels_arr, iou_threshold=0.50):
# #     if len(scores_arr) == 0:
# #         return boxes_abs, scores_arr, labels_arr
# #     boxes_t  = torch.from_numpy(boxes_abs).float()
# #     scores_t = torch.from_numpy(np.asarray(scores_arr)).float()
# #     labels_t = torch.from_numpy(np.asarray(labels_arr).astype(np.int64)).long()
# #     keep_indices = []
# #     for cls in labels_t.unique():
# #         mask = (labels_t == cls)
# #         keep = torchvision_nms(boxes_t[mask], scores_t[mask], iou_threshold)
# #         keep_indices.append(torch.where(mask)[0][keep])
# #     if not keep_indices:
# #         return boxes_abs, scores_arr, labels_arr
# #     keep_all = torch.cat(keep_indices)
# #     order    = scores_t[keep_all].argsort(descending=True)
# #     return (boxes_t[keep_all[order]].numpy(),
# #             scores_t[keep_all[order]].numpy(),
# #             labels_t[keep_all[order]].numpy())
# #
# #
# # def merge_and_eval(all_preds_list, img_sizes, base_ds,
# #                    iou_thr, skip_thr, nms_iou):
# #     """单次合并+评估，返回 AP"""
# #     from datasets.coco_eval import CocoEvaluator
# #
# #     all_image_ids = set()
# #     for p in all_preds_list:
# #         all_image_ids.update(p.keys())
# #
# #     merged = {}
# #     for img_id in all_image_ids:
# #         H, W = img_sizes[img_id]
# #         boxes_list, scores_list, labels_list = [], [], []
# #         for preds in all_preds_list:
# #             if img_id not in preds:
# #                 continue
# #             b = preds[img_id]['boxes'].numpy().copy()
# #             s = preds[img_id]['scores'].numpy()
# #             l = preds[img_id]['labels'].numpy()
# #             b[:, [0,2]] /= W;  b[:, [1,3]] /= H
# #             b = np.clip(b, 0., 1.)
# #             boxes_list.append(b.tolist())
# #             scores_list.append(s.tolist())
# #             labels_list.append(l.tolist())
# #
# #         if not boxes_list:
# #             merged[img_id] = {'boxes': torch.zeros(0,4),
# #                                'scores': torch.zeros(0),
# #                                'labels': torch.zeros(0, dtype=torch.long)}
# #             continue
# #
# #         bw, sw, lw = weighted_boxes_fusion(
# #             boxes_list, scores_list, labels_list,
# #             iou_thr=iou_thr, skip_box_thr=skip_thr, weights=None)
# #
# #         ba = bw.copy()
# #         ba[:, [0,2]] *= W;  ba[:, [1,3]] *= H
# #
# #         if nms_iou is not None and len(sw) > 0:
# #             ba, sw, lw = apply_per_class_nms(ba, sw, lw, nms_iou)
# #
# #         merged[img_id] = {
# #             'boxes':  torch.from_numpy(ba).float(),
# #             'scores': torch.from_numpy(np.asarray(sw)).float(),
# #             'labels': torch.from_numpy(np.asarray(lw).astype(np.int64)).long(),
# #         }
# #
# #     import io, contextlib
# #     ev = CocoEvaluator(base_ds, ['bbox'], useCats=True)
# #     ev.update(merged)
# #     ev.synchronize_between_processes()
# #     ev.accumulate()
# #     # 静默输出，只取 AP
# #     buf = io.StringIO()
# #     with contextlib.redirect_stdout(buf):
# #         ev.summarize()
# #     output = buf.getvalue()
# #     # 第一行是 AP@0.50:0.95
# #     for line in output.split('\n'):
# #         if 'IoU=0.50:0.95' in line and 'area=   all' in line:
# #             ap = float(line.strip().split('=')[-1].strip())
# #             return ap
# #     return 0.0
# #
# #
# # def main():
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument('--variants_path', required=True)
# #     parser.add_argument('--ann_file', required=True)
# #     parser.add_argument('--dataset_file', default='aitodv2')
# #     parser.add_argument('--coco_path', required=True)
# #     parser.add_argument('-c', '--config_file', default='config/DQ_5scale_v6.py')
# #     args = parser.parse_args()
# #
# #     # 加载变体预测
# #     print(f"Loading variants from {args.variants_path} ...")
# #     all_preds_list = torch.load(args.variants_path, map_location='cpu')
# #     print(f"  {len(all_preds_list)} variants loaded")
# #
# #     # 构建评估用数据集
# #     from util.slconfig import SLConfig
# #     from datasets import build_dataset, get_coco_api_from_dataset
# #     cfg = SLConfig.fromfile(args.config_file)
# #     eval_args = argparse.Namespace()
# #     for k, v in cfg._cfg_dict.to_dict().items():
# #         setattr(eval_args, k, v)
# #     eval_args.dataset_file = args.dataset_file
# #     eval_args.coco_path    = args.coco_path
# #     for attr in ['masks','fix_size','strong_aug','distributed',
# #                  'amp','save_results','debug','rank','local_rank']:
# #         setattr(eval_args, attr, False if attr != 'rank' else 0)
# #
# #     dataset_base = build_dataset(image_set='test', args=eval_args)
# #     base_ds      = get_coco_api_from_dataset(dataset_base)
# #     img_sizes    = {img['id']: (img['height'], img['width'])
# #                     for img in base_ds.dataset['images']}
# #
# #     # 网格搜索
# #     iou_thrs  = [0.40, 0.45, 0.50, 0.55, 0.60]
# #     skip_thrs = [0.001, 0.003, 0.005, 0.010]
# #     nms_ious  = [None, 0.45, 0.50, 0.55]
# #
# #     results = []
# #     total = len(iou_thrs) * len(skip_thrs) * len(nms_ious)
# #     done  = 0
# #
# #     print(f"\nSearching {total} parameter combinations ...\n")
# #     print(f"{'iou_thr':>8} {'skip_thr':>9} {'nms_iou':>8}  {'AP':>6}")
# #     print("-" * 40)
# #
# #     best_ap, best_params = 0.0, {}
# #     for iou_thr in iou_thrs:
# #         for skip_thr in skip_thrs:
# #             for nms_iou in nms_ious:
# #                 ap = merge_and_eval(
# #                     all_preds_list, img_sizes, base_ds,
# #                     iou_thr, skip_thr, nms_iou)
# #                 nms_str = f"{nms_iou:.2f}" if nms_iou else "None"
# #                 marker  = " ★" if ap > best_ap else ""
# #                 print(f"{iou_thr:>8.2f} {skip_thr:>9.3f} {nms_str:>8}  "
# #                       f"{ap:>6.4f}{marker}")
# #                 if ap > best_ap:
# #                     best_ap = ap
# #                     best_params = dict(iou_thr=iou_thr,
# #                                        skip_thr=skip_thr,
# #                                        nms_iou=nms_iou)
# #                 done += 1
# #
# #     print(f"\n{'='*40}")
# #     print(f"Best AP = {best_ap:.4f}")
# #     print(f"Best params: {best_params}")
# #     print(f"\nRun command:")
# #     nms_flag = (f"--cls_nms_iou_thr {best_params['nms_iou']}"
# #                 if best_params['nms_iou'] else "--no_cls_nms")
# #     print(f"  python eval_tta.py ... "
# #           f"--wbf_iou_thr {best_params['iou_thr']} "
# #           f"--wbf_skip_thr {best_params['skip_thr']} "
# #           f"{nms_flag} --no_cls_threshold")
# #
# #
# # if __name__ == '__main__':
# #     main()
#
# """
# WBF 参数网格搜索脚本。
# 加载已保存的各变体预测（variants_raw.pt），快速枚举 WBF 参数组合。
# 不需要重新跑模型推理，每组参数只需几秒。
#
# 使用方法：
#     python search_wbf_params.py \
#         --variants_path logs/tta_variants_ver6/variants_raw.pt \
#         --ann_file /workspace/DQDetr/data/path/AITODv2/annotations/aitodv2_test.json \
#         --dataset_file aitodv2 \
#         --coco_path /workspace/DQDetr/data/path/AITODv2 \
#         -c config/DQ_5scale_v6.py
# """
#
# import argparse
# import copy
# import os
# import sys
# import time
#
# import numpy as np
#
# # ====== 修复 aitodpycocotools 与 numpy>=1.24 的兼容性问题 ======
# if not hasattr(np, 'float'):
#     np.float = float
# # =========================================================
#
# import torch
# from ensemble_boxes import weighted_boxes_fusion
# from torchvision.ops import nms as torchvision_nms
#
#
# # ─────────────────────────────────────────────
# # 工具函数
# # ─────────────────────────────────────────────
#
# def apply_per_class_nms(boxes_abs, scores_arr, labels_arr, iou_threshold=0.50):
#     if len(scores_arr) == 0:
#         return boxes_abs, scores_arr, labels_arr
#     boxes_t  = torch.from_numpy(np.asarray(boxes_abs)).float()
#     scores_t = torch.from_numpy(np.asarray(scores_arr)).float()
#     labels_t = torch.from_numpy(np.asarray(labels_arr).astype(np.int64)).long()
#     keep_indices = []
#     for cls in labels_t.unique():
#         mask = (labels_t == cls)
#         keep = torchvision_nms(boxes_t[mask], scores_t[mask], iou_threshold)
#         keep_indices.append(torch.where(mask)[0][keep])
#     if not keep_indices:
#         return boxes_abs, scores_arr, labels_arr
#     keep_all = torch.cat(keep_indices)
#     order    = scores_t[keep_all].argsort(descending=True)
#     keep_all = keep_all[order]
#     return (boxes_t[keep_all].numpy(),
#             scores_t[keep_all].numpy(),
#             labels_t[keep_all].numpy())
#
#
# def merge_predictions(all_preds_list, img_sizes,
#                       iou_thr, skip_thr, nms_iou):
#     """合并所有变体预测，返回 {image_id: {'boxes','scores','labels'}}"""
#     all_image_ids = set()
#     for p in all_preds_list:
#         all_image_ids.update(p.keys())
#
#     merged = {}
#     for img_id in all_image_ids:
#         H, W = img_sizes[img_id]
#         boxes_list, scores_list, labels_list = [], [], []
#
#         for preds in all_preds_list:
#             if img_id not in preds:
#                 continue
#             b = preds[img_id]['boxes'].numpy().copy()
#             s = preds[img_id]['scores'].numpy()
#             l = preds[img_id]['labels'].numpy()
#             b[:, [0, 2]] /= W
#             b[:, [1, 3]] /= H
#             b = np.clip(b, 0., 1.)
#             boxes_list.append(b.tolist())
#             scores_list.append(s.tolist())
#             labels_list.append(l.tolist())
#
#         if not boxes_list:
#             merged[img_id] = {
#                 'boxes':  torch.zeros(0, 4),
#                 'scores': torch.zeros(0),
#                 'labels': torch.zeros(0, dtype=torch.long),
#             }
#             continue
#
#         bw, sw, lw = weighted_boxes_fusion(
#             boxes_list, scores_list, labels_list,
#             iou_thr=iou_thr,
#             skip_box_thr=skip_thr,
#             weights=None,
#         )
#
#         ba = bw.copy()
#         ba[:, [0, 2]] *= W
#         ba[:, [1, 3]] *= H
#
#         if nms_iou is not None and len(sw) > 0:
#             ba, sw, lw = apply_per_class_nms(ba, sw, lw, nms_iou)
#
#         merged[img_id] = {
#             'boxes':  torch.from_numpy(ba).float(),
#             'scores': torch.from_numpy(np.asarray(sw)).float(),
#             'labels': torch.from_numpy(
#                 np.asarray(lw).astype(np.int64)).long(),
#         }
#     return merged
#
#
# def eval_merged(merged, coco_gt):
#     """
#     直接调用 aitodpycocotools 评估，通过 .stats[0] 取 AP。
#     不依赖 stdout 解析，完全避免输出重定向问题。
#     """
#     import copy
#     import contextlib
#     import io
#     from aitodpycocotools.coco import COCO
#     from aitodpycocotools.cocoeval import COCOeval
#
#     # 构建预测结果列表（COCO 格式）
#     coco_results = []
#     for img_id, res in merged.items():
#         boxes  = res['boxes'].numpy()
#         scores = res['scores'].numpy()
#         labels = res['labels'].numpy()
#         for box, score, label in zip(boxes, scores, labels):
#             x1, y1, x2, y2 = box
#             coco_results.append({
#                 'image_id':    int(img_id),
#                 'category_id': int(label),
#                 'bbox':        [float(x1), float(y1),
#                                 float(x2 - x1), float(y2 - y1)],
#                 'score':       float(score),
#             })
#
#     if len(coco_results) == 0:
#         return 0.0
#
#     # 加载预测结果（静默）
#     with contextlib.redirect_stdout(io.StringIO()):
#         coco_dt = coco_gt.loadRes(coco_results)
#
#     # 创建评估对象并评估（静默）
#     coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
#     coco_eval.params.imgIds = list(merged.keys())
#
#     with contextlib.redirect_stdout(io.StringIO()):
#         coco_eval.evaluate()
#         coco_eval.accumulate()
#         coco_eval.summarize()
#
#     # stats[0] = AP@0.50:0.95, all areas, maxDets=1500
#     ap = float(coco_eval.stats[0])
#     return ap
#
#
# # ─────────────────────────────────────────────
# # 主流程
# # ─────────────────────────────────────────────
#
# def main():
#     parser = argparse.ArgumentParser(
#         description='WBF parameter grid search for TTA')
#     parser.add_argument('--variants_path', required=True,
#                         help='variants_raw.pt 的路径')
#     parser.add_argument('--ann_file', required=True,
#                         help='测试集标注文件路径 (aitodv2_test.json)')
#     parser.add_argument('--dataset_file', default='aitodv2')
#     parser.add_argument('--coco_path', required=True)
#     parser.add_argument('-c', '--config_file',
#                         default='config/DQ_5scale.py')
#
#     # 自定义搜索空间（可选，默认使用预设范围）
#     parser.add_argument('--iou_thrs', nargs='+', type=float,
#                         default=[0.40, 0.45, 0.50, 0.55, 0.60])
#     parser.add_argument('--skip_thrs', nargs='+', type=float,
#                         default=[0.001, 0.003, 0.005, 0.010])
#     parser.add_argument('--nms_ious', nargs='+', type=float,
#                         default=[0.0, 0.45, 0.50, 0.55],
#                         help='per-class NMS IoU 阈值，0.0 表示不做 NMS')
#
#     args = parser.parse_args()
#
#     # ── 加载变体预测 ──
#     print(f"[Search] Loading variants from {args.variants_path} ...")
#     all_preds_list = torch.load(args.variants_path, map_location='cpu',
#                                 weights_only=False)
#     print(f"[Search]   {len(all_preds_list)} variants loaded, "
#           f"{len(all_preds_list[0])} images each")
#
#     # ── 加载 COCO GT（只做一次）──
#     print(f"[Search] Loading COCO GT from {args.ann_file} ...")
#     from aitodpycocotools.coco import COCO
#     import contextlib, io
#     with contextlib.redirect_stdout(io.StringIO()):
#         coco_gt = COCO(args.ann_file)
#
#     # ── 构建 image_id → size 映射 ──
#     img_sizes = {
#         img['id']: (img['height'], img['width'])
#         for img in coco_gt.dataset['images']
#     }
#     print(f"[Search]   {len(img_sizes)} images in GT")
#
#     # ── 构建搜索网格 ──
#     # nms_iou=0.0 表示不做 per-class NMS
#     nms_ious = [None if v == 0.0 else v for v in args.nms_ious]
#     combos   = [(i, s, n)
#                 for i in args.iou_thrs
#                 for s in args.skip_thrs
#                 for n in nms_ious]
#     total = len(combos)
#
#     print(f"\n[Search] Grid: {len(args.iou_thrs)} iou_thrs × "
#           f"{len(args.skip_thrs)} skip_thrs × "
#           f"{len(nms_ious)} nms_ious = {total} combinations\n")
#
#     header = (f"{'#':>4}  {'iou_thr':>7}  {'skip_thr':>8}  "
#               f"{'nms_iou':>7}  {'AP':>7}  {'time':>5}")
#     print(header)
#     print("-" * len(header))
#
#     results    = []
#     best_ap    = 0.0
#     best_combo = None
#
#     for idx, (iou_thr, skip_thr, nms_iou) in enumerate(combos, 1):
#         t0 = time.time()
#
#         merged = merge_predictions(
#             all_preds_list, img_sizes,
#             iou_thr=iou_thr,
#             skip_thr=skip_thr,
#             nms_iou=nms_iou,
#         )
#         ap = eval_merged(merged, coco_gt)
#
#         elapsed  = time.time() - t0
#         nms_str  = f"{nms_iou:.2f}" if nms_iou is not None else "None"
#         is_best  = ap > best_ap
#         marker   = " ★" if is_best else ""
#
#         print(f"{idx:>4}  {iou_thr:>7.2f}  {skip_thr:>8.3f}  "
#               f"{nms_str:>7}  {ap:>7.4f}  {elapsed:>4.1f}s{marker}")
#         sys.stdout.flush()   # 确保每行立即输出
#
#         results.append((ap, iou_thr, skip_thr, nms_iou))
#         if is_best:
#             best_ap    = ap
#             best_combo = (iou_thr, skip_thr, nms_iou)
#
#     # ── 输出最优结果 ──
#     print(f"\n{'='*50}")
#     print(f"Best AP = {best_ap:.4f}")
#     iou_thr, skip_thr, nms_iou = best_combo
#     nms_str = f"{nms_iou:.2f}" if nms_iou is not None else "None"
#     print(f"  wbf_iou_thr  = {iou_thr}")
#     print(f"  wbf_skip_thr = {skip_thr}")
#     print(f"  cls_nms_iou  = {nms_str}")
#
#     nms_flag = (f"--cls_nms_iou_thr {nms_iou}" if nms_iou is not None
#                 else "--no_cls_nms")
#     print(f"\nRun command:")
#     print(f"  python eval_tta.py \\")
#     print(f"    -c config/DQ_5scale_v6.py \\")
#     print(f"    --checkpoint <your_checkpoint> \\")
#     print(f"    --coco_path {args.coco_path} \\")
#     print(f"    --dataset_file aitodv2 \\")
#     print(f"    --output_dir logs/tta_eval_best \\")
#     print(f"    --scales 736 800 864 \\")
#     print(f"    --flip \\")
#     print(f"    --wbf_iou_thr {iou_thr} \\")
#     print(f"    --wbf_skip_thr {skip_thr} \\")
#     print(f"    {nms_flag} \\")
#     print(f"    --no_cls_threshold")
#
#     # ── 保存完整搜索结果 ──
#     save_path = os.path.join(
#         os.path.dirname(args.variants_path), 'wbf_search_results.txt')
#     with open(save_path, 'w') as f:
#         f.write(header + '\n')
#         f.write('-' * len(header) + '\n')
#         for i, (ap, iou_thr, skip_thr, nms_iou) in enumerate(results, 1):
#             nms_str = f"{nms_iou:.2f}" if nms_iou is not None else "None"
#             marker  = " ★" if (iou_thr, skip_thr, nms_iou) == best_combo else ""
#             f.write(f"{i:>4}  {iou_thr:>7.2f}  {skip_thr:>8.3f}  "
#                     f"{nms_str:>7}  {ap:>7.4f}{marker}\n")
#         f.write(f"\nBest AP = {best_ap:.4f}, params = {best_combo}\n")
#     print(f"\n[Search] Full results saved → {save_path}")
#
#
# if __name__ == '__main__':
#     main()

"""
WBF 参数网格搜索脚本。
加载已保存的各变体预测（variants_raw.pt），快速枚举 WBF 参数组合。
引入了子集抽样搜索（Strategy 4）与进度条（Strategy 1），将耗时从数小时缩短至几分钟。
"""

import argparse
import copy
import os
import sys
import time
import random

import numpy as np

# ====== 修复 aitodpycocotools 与 numpy>=1.24 的兼容性问题 ======
if not hasattr(np, 'float'):
    np.float = float
# =========================================================

import torch
from ensemble_boxes import weighted_boxes_fusion
from torchvision.ops import nms as torchvision_nms
from tqdm import tqdm


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def apply_per_class_nms(boxes_abs, scores_arr, labels_arr, iou_threshold=0.50):
    if len(scores_arr) == 0:
        return boxes_abs, scores_arr, labels_arr
    boxes_t = torch.from_numpy(np.asarray(boxes_abs)).float()
    scores_t = torch.from_numpy(np.asarray(scores_arr)).float()
    labels_t = torch.from_numpy(np.asarray(labels_arr).astype(np.int64)).long()
    keep_indices = []
    for cls in labels_t.unique():
        mask = (labels_t == cls)
        keep = torchvision_nms(boxes_t[mask], scores_t[mask], iou_threshold)
        keep_indices.append(torch.where(mask)[0][keep])
    if not keep_indices:
        return boxes_abs, scores_arr, labels_arr
    keep_all = torch.cat(keep_indices)
    order = scores_t[keep_all].argsort(descending=True)
    keep_all = keep_all[order]
    return (boxes_t[keep_all].numpy(),
            scores_t[keep_all].numpy(),
            labels_t[keep_all].numpy())


def merge_predictions(all_preds_list, img_sizes,
                      iou_thr, skip_thr, nms_iou):
    """合并所有变体预测，返回 {image_id: {'boxes','scores','labels'}}"""
    all_image_ids = set()
    for p in all_preds_list:
        all_image_ids.update(p.keys())

    merged = {}

    # 【策略一】：引入 tqdm 进度条，消除黑盒焦虑
    for img_id in tqdm(list(all_image_ids), desc="Merging WBF", leave=False):
        H, W = img_sizes[img_id]
        boxes_list, scores_list, labels_list = [], [], []

        for preds in all_preds_list:
            if img_id not in preds:
                continue
            b = preds[img_id]['boxes'].numpy().copy()
            s = preds[img_id]['scores'].numpy()
            l = preds[img_id]['labels'].numpy()
            b[:, [0, 2]] /= W
            b[:, [1, 3]] /= H
            b = np.clip(b, 0., 1.)
            boxes_list.append(b.tolist())
            scores_list.append(s.tolist())
            labels_list.append(l.tolist())

        if not boxes_list:
            merged[img_id] = {
                'boxes': torch.zeros(0, 4),
                'scores': torch.zeros(0),
                'labels': torch.zeros(0, dtype=torch.long),
            }
            continue

        bw, sw, lw = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            iou_thr=iou_thr,
            skip_box_thr=skip_thr,
            weights=None,
        )

        ba = bw.copy()
        ba[:, [0, 2]] *= W
        ba[:, [1, 3]] *= H

        if nms_iou is not None and len(sw) > 0:
            ba, sw, lw = apply_per_class_nms(ba, sw, lw, nms_iou)

        merged[img_id] = {
            'boxes': torch.from_numpy(ba).float(),
            'scores': torch.from_numpy(np.asarray(sw)).float(),
            'labels': torch.from_numpy(
                np.asarray(lw).astype(np.int64)).long(),
        }
    return merged


def eval_merged(merged, coco_gt):
    """
    直接调用 aitodpycocotools 评估，通过 .stats[0] 取 AP。
    不依赖 stdout 解析，完全避免输出重定向问题。
    """
    import contextlib
    import io
    from aitodpycocotools.cocoeval import COCOeval

    # 构建预测结果列表（COCO 格式）
    coco_results = []
    for img_id, res in merged.items():
        boxes = res['boxes'].numpy()
        scores = res['scores'].numpy()
        labels = res['labels'].numpy()
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            coco_results.append({
                'image_id': int(img_id),
                'category_id': int(label),
                'bbox': [float(x1), float(y1),
                         float(x2 - x1), float(y2 - y1)],
                'score': float(score),
            })

    if len(coco_results) == 0:
        return 0.0

    # 加载预测结果（静默）
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(coco_results)

    # 创建评估对象并评估（静默）
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.imgIds = list(merged.keys())

    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    # stats[0] = AP@0.50:0.95, all areas, maxDets=1500
    ap = float(coco_eval.stats[0])
    return ap


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='WBF parameter grid search for TTA')
    parser.add_argument('--variants_path', required=True,
                        help='variants_raw.pt 的路径')
    parser.add_argument('--ann_file', required=True,
                        help='测试集标注文件路径 (aitodv2_test.json)')
    parser.add_argument('--dataset_file', default='aitodv2')
    parser.add_argument('--coco_path', required=True)
    parser.add_argument('-c', '--config_file',
                        default='config/DQ_5scale.py')

    # 自定义搜索空间（可选，默认使用预设范围）
    parser.add_argument('--iou_thrs', nargs='+', type=float,
                        default=[0.80])
    parser.add_argument('--skip_thrs', nargs='+', type=float,
                        default=[0.080,0.085,0.090,0.095])
    parser.add_argument('--nms_ious', nargs='+', type=float,
                        default=[0.0],
                        help='per-class NMS IoU 阈值，0.0 表示不做 NMS')

    args = parser.parse_args()

    # ── 加载变体预测 ──
    print(f"[Search] Loading variants from {args.variants_path} ...")
    all_preds_list = torch.load(args.variants_path, map_location='cpu',
                                weights_only=False)
    print(f"[Search]   {len(all_preds_list)} variants loaded, "
          f"{len(all_preds_list[0])} images each")

    # ── 加载 COCO GT（只做一次）──
    print(f"[Search] Loading COCO GT from {args.ann_file} ...")
    from aitodpycocotools.coco import COCO
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(args.ann_file)

    # ── 构建 image_id → size 映射 ──
    img_sizes = {
        img['id']: (img['height'], img['width'])
        for img in coco_gt.dataset['images']
    }
    print(f"[Search]   {len(img_sizes)} images in GT")

    # =================================================================
    # 【策略四】：抽样网格子集 (Subsampling) 用于快速搜索
    # =================================================================
    SEARCH_SUBSET_SIZE = 1000
    all_img_ids = list(img_sizes.keys())

    if len(all_img_ids) > SEARCH_SUBSET_SIZE:
        print(f"\n[Search] 🚀 Randomly subsampling {SEARCH_SUBSET_SIZE} images for fast grid search...")
        # 固定随机种子以便复现
        random.seed(42)
        subset_ids = set(random.sample(all_img_ids, SEARCH_SUBSET_SIZE))

        # 1. 裁剪变体预测列表
        search_preds_list = []
        for p in all_preds_list:
            search_preds_list.append({k: v for k, v in p.items() if k in subset_ids})

        # 2. 裁剪 img_sizes
        search_img_sizes = {k: v for k, v in img_sizes.items() if k in subset_ids}

        # 3. 裁剪 COCO GT，构建局部的 ground truth 索引
        search_gt = COCO()
        search_gt.dataset['images'] = [img for img in coco_gt.dataset['images'] if img['id'] in subset_ids]
        search_gt.dataset['annotations'] = [ann for ann in coco_gt.dataset['annotations'] if
                                            ann['image_id'] in subset_ids]
        search_gt.dataset['categories'] = coco_gt.dataset['categories']
        search_gt.createIndex()
    else:
        search_preds_list = all_preds_list
        search_img_sizes = img_sizes
        search_gt = coco_gt

    # ── 构建搜索网格 ──
    nms_ious = [None if v == 0.0 else v for v in args.nms_ious]
    combos = [(i, s, n)
              for i in args.iou_thrs
              for s in args.skip_thrs
              for n in nms_ious]
    total = len(combos)

    print(f"\n[Search] Grid: {len(args.iou_thrs)} iou_thrs × "
          f"{len(args.skip_thrs)} skip_thrs × "
          f"{len(nms_ious)} nms_ious = {total} combinations\n")

    header = (f"{'#':>4}  {'iou_thr':>7}  {'skip_thr':>8}  "
              f"{'nms_iou':>7}  {'AP(Sub)':>8}  {'time':>5}")
    print(header)
    print("-" * len(header))

    results = []
    best_ap = 0.0
    best_combo = None

    for idx, (iou_thr, skip_thr, nms_iou) in enumerate(combos, 1):
        t0 = time.time()

        # 使用裁剪后的子集进行 WBF 合并
        merged = merge_predictions(
            search_preds_list, search_img_sizes,
            iou_thr=iou_thr,
            skip_thr=skip_thr,
            nms_iou=nms_iou,
        )
        # 使用裁剪后的子集进行评估
        ap = eval_merged(merged, search_gt)

        elapsed = time.time() - t0
        nms_str = f"{nms_iou:.2f}" if nms_iou is not None else "None"
        is_best = ap > best_ap
        marker = " ★" if is_best else ""

        print(f"{idx:>4}  {iou_thr:>7.2f}  {skip_thr:>8.3f}  "
              f"{nms_str:>7}  {ap:>8.4f}  {elapsed:>4.1f}s{marker}")
        sys.stdout.flush()

        results.append((ap, iou_thr, skip_thr, nms_iou))
        if is_best:
            best_ap = ap
            best_combo = (iou_thr, skip_thr, nms_iou)

    # ── 最终验证：将最优参数应用到全量数据集 ──
    print(f"\n{'=' * 60}")
    print(f"🌟 Fast search complete. Evaluating best parameters on FULL dataset ({len(img_sizes)} images)...")
    iou_thr, skip_thr, nms_iou = best_combo

    t0_full = time.time()
    # 传入原始的 full list 和 full GT
    full_merged = merge_predictions(
        all_preds_list, img_sizes,
        iou_thr=iou_thr,
        skip_thr=skip_thr,
        nms_iou=nms_iou,
    )
    final_full_ap = eval_merged(full_merged, coco_gt)
    full_elapsed = time.time() - t0_full

    print(f"✅ Full Dataset Eval Done in {full_elapsed:.1f}s")
    print(f"👑 FINAL FULL DATASET AP = {final_full_ap:.4f}")

    nms_str = f"{nms_iou:.2f}" if nms_iou is not None else "None"
    print(f"  wbf_iou_thr  = {iou_thr}")
    print(f"  wbf_skip_thr = {skip_thr}")
    print(f"  cls_nms_iou  = {nms_str}")

    nms_flag = (f"--cls_nms_iou_thr {nms_iou}" if nms_iou is not None
                else "--no_cls_nms")
    print(f"\nRun command for eval_tta.py:")
    print(f"  python eval_tta.py \\")
    print(f"    -c config/DQ_5scale_v6.py \\")
    print(f"    --checkpoint <your_checkpoint> \\")
    print(f"    --coco_path {args.coco_path} \\")
    print(f"    --dataset_file aitodv2 \\")
    print(f"    --output_dir logs/tta_eval_best \\")
    print(f"    --scales 736 800 864 \\")
    print(f"    --flip \\")
    print(f"    --wbf_iou_thr {iou_thr} \\")
    print(f"    --wbf_skip_thr {skip_thr} \\")
    print(f"    {nms_flag} \\")
    print(f"    --no_cls_threshold")

    # ── 保存完整搜索结果 ──
    save_path = os.path.join(
        os.path.dirname(args.variants_path), 'wbf_search_results.txt')
    with open(save_path, 'w') as f:
        f.write(f"Subset Search Size: {SEARCH_SUBSET_SIZE}\n")
        f.write(header + '\n')
        f.write('-' * len(header) + '\n')
        for i, (ap, iou, skip, nms) in enumerate(results, 1):
            nms_str = f"{nms:.2f}" if nms is not None else "None"
            marker = " ★" if (iou, skip, nms) == best_combo else ""
            f.write(f"{i:>4}  {iou:>7.2f}  {skip:>8.3f}  "
                    f"{nms_str:>7}  {ap:>8.4f}{marker}\n")
        f.write(f"\nSubset Best AP = {best_ap:.4f}, params = {best_combo}\n")
        f.write(f"FULL DATASET FINAL AP = {final_full_ap:.4f}\n")

    print(f"\n[Search] Full results saved → {save_path}")


if __name__ == '__main__':
    main()