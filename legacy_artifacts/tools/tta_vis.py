import os
import cv2
import json
import torch
import numpy as np
import torchvision.ops as ops

# --- 1. 色彩与显示方案配置（OpenCV BGR 格式）---
# 融合后的 TP/FP/FN 色彩映射
COLOR_MAP = {
    'TP': (0, 255, 0),  # 绿色 — 真阳性 (预测正确的最终框)
    'FP': (0, 0, 255),  # 红色 — 假阳性 (误检/多余的预测)
    'FN': (255, 0, 0),  # 蓝色 — 假阴性 (漏检的真实框)
}
BOX_THICKNESS = 2


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────
def get_image_id_from_coco(coco_json_path, img_filename):
    """通过文件名在 COCO 标注文件中查找 image_id。"""
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    for img in coco_data['images']:
        if img['file_name'] == img_filename:
            return img['id']
    raise ValueError(f"在 {coco_json_path} 中找不到图片文件: {img_filename}")


def get_gt_boxes_from_coco(json_path, img_id, device, dup_iou_thresh=0.85):
    """从 COCO 提取指定图片的真实标注框，并使用 NMS 消除重复打标。"""
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    gt_list = []
    for ann in coco_data['annotations']:
        if ann['image_id'] == img_id:
            x, y, w, h = ann['bbox']
            gt_list.append([x, y, x + w, y + h])

    gt_tensor = torch.tensor(gt_list, dtype=torch.float32, device=device)

    # 去重：过滤掉数据集中重合度极高的标注
    if len(gt_tensor) > 0:
        dummy_scores = torch.ones(len(gt_tensor), device=device)
        keep_idx = ops.nms(gt_tensor, dummy_scores, dup_iou_thresh)
        gt_tensor = gt_tensor[keep_idx]

    return gt_tensor


def match_predictions(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """预测框与 GT 框进行贪婪匹配，区分 TP, FP, FN。"""
    if len(gt_boxes) == 0:
        sorted_order = torch.argsort(pred_scores, descending=True).tolist()
        return sorted_order, [], list(range(len(sorted_order))), []
    if len(pred_boxes) == 0:
        return [], [], [], list(range(len(gt_boxes)))

    sorted_order = torch.argsort(pred_scores, descending=True).tolist()
    ious = ops.box_iou(pred_boxes, gt_boxes)  # [Num_preds, Num_GTs]

    matched_gt = set()
    tp_pos, fp_pos = [], []

    for pos, orig_i in enumerate(sorted_order):
        iou_row = ious[orig_i]
        max_iou, max_gt_idx = torch.max(iou_row, dim=0)
        max_gt_idx = max_gt_idx.item()

        if max_iou.item() > iou_threshold and max_gt_idx not in matched_gt:
            tp_pos.append(pos)
            matched_gt.add(max_gt_idx)
        else:
            fp_pos.append(pos)

    fn_idx = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    return sorted_order, tp_pos, fp_pos, fn_idx


def draw_legend_advanced(img, position=(10, 10)):
    """绘制图例 (TP/FP/FN 指示)"""
    legend_items = [
        ('TP: Correct Det     (Green)', COLOR_MAP['TP']),
        ('FP: Wrong/Extra Det (Red)', COLOR_MAP['FP']),
        ('FN: Missed GT       (Blue)', COLOR_MAP['FN']),
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.50
    thickness = 1
    line_h = 24
    box_w, box_h = 16, 12
    pad = 8

    bx, by = position
    legend_h = len(legend_items) * line_h + pad * 2
    legend_w = 320

    # 半透明背景
    overlay = img.copy()
    cv2.rectangle(overlay, (bx, by), (bx + legend_w, by + legend_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    cv2.rectangle(img, (bx, by), (bx + legend_w, by + legend_h), (180, 180, 180), 1)

    for i, (label, color) in enumerate(legend_items):
        cy = by + pad + i * line_h + line_h // 2

        # 绘制图例的色块
        cv2.rectangle(img, (bx + pad, cy - box_h // 2), (bx + pad + box_w, cy + box_h // 2), color, -1)

        cv2.putText(img, label,
                    (bx + pad + box_w + 8, cy + 5),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# 主可视化函数
# ──────────────────────────────────────────────────────────────────────────────
def visualize_tta_process(img_path, img_id, coco_json_path, merged_json_path,
                          out_path, score_thr=0.4, iou_thr=0.5, gt_dup_thr=0.85):
    """可视化 WBF 最终预测结果与 GT 的比对评估 (TP/FP/FN)，不带得分。"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"正在处理: {os.path.basename(img_path)}  (image_id={img_id})")

    # ── 1. 加载原图 ──────────────────────────────────────────────────────────
    raw_img = cv2.imread(img_path)
    if raw_img is None:
        raise FileNotFoundError(f"无法读取图片: {img_path}")

    draw_img = raw_img.copy()

    # ── 2. 提取 GT 框并获取 WBF 最终预测进行对比匹配 ─────────────────────────
    print("-> 提取去重 Ground Truth 框及 WBF 预测框并进行匹配...")
    gt_boxes = get_gt_boxes_from_coco(coco_json_path, img_id, device, dup_iou_thresh=gt_dup_thr)

    with open(merged_json_path, 'r') as f:
        merged_data = json.load(f)

    p_boxes_list, p_scores_list = [], []
    for ann in merged_data:
        if ann['image_id'] == img_id and ann['score'] > score_thr:
            x, y, bw, bh = ann['bbox']  # COCO 格式: [x_min, y_min, w, h]
            p_boxes_list.append([x, y, x + bw, y + bh])
            p_scores_list.append(ann['score'])

    if len(p_boxes_list) > 0:
        p_boxes = torch.tensor(p_boxes_list, dtype=torch.float32, device=device)
        p_scores = torch.tensor(p_scores_list, dtype=torch.float32, device=device)
    else:
        p_boxes = torch.empty((0, 4), device=device)
        p_scores = torch.empty((0,), device=device)

    # 使用 IoU 贪婪匹配进行分类
    sorted_order, tp_pos, fp_pos, fn_idx = match_predictions(
        p_boxes, p_scores, gt_boxes, iou_threshold=iou_thr
    )

    # ── 3. 在画布上绘制 TP, FP 和 FN ──────────────────────────────────────────
    def draw_box(img, box, color):
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS)

    # 画 TP (绿框)
    for pos in tp_pos:
        orig_i = sorted_order[pos]
        b = p_boxes[orig_i].int().tolist()
        draw_box(draw_img, b, COLOR_MAP['TP'])

    # 画 FP (红框)
    for pos in fp_pos:
        orig_i = sorted_order[pos]
        b = p_boxes[orig_i].int().tolist()
        draw_box(draw_img, b, COLOR_MAP['FP'])

    # 画 FN (蓝框)
    for i in fn_idx:
        b = gt_boxes[i].int().tolist()
        draw_box(draw_img, b, COLOR_MAP['FN'])

    print(f"   最终匹配结果: TP={len(tp_pos)}, FP={len(fp_pos)}, FN={len(fn_idx)}")

    # ── 4. 绘制图例并保存 ────────────────────────────────────────────────────
    draw_legend_advanced(draw_img)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(out_path, draw_img)
    print(f"可视化完成！结果已保存至: {out_path}\n")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ================= 配置区（请修改为你的实际路径） =================
    COCO_JSON_PATH = "/workspace/DQDetr/data/path/AITODv2/annotations/aitodv2_test.json"
    INPUT_FOLDER = "/workspace/DQDetr/data/path/AITODv2/images/test/images"
    MERGED_JSON = "/workspace/DQDetr/logs/tta_variants_ver9/iou80-skip0.06/tta_merged_results.json"
    OUTPUT_FOLDER = "/workspace/DQDetr/logs/vis_tta"

    VIS_SCORE_THR = 0.4
    IOU_THR = 0.5  # TP/FP 的匹配 IoU 阈值
    GT_DUP_THR = 0.85  # 标注数据重复过滤的 NMS 阈值
    # ==================================================================

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    image_files = sorted([
        f for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith(image_extensions)
    ])
    print(f"在 {INPUT_FOLDER} 中找到 {len(image_files)} 张图片，开始批量可视化...")

    for idx, img_filename in enumerate(image_files, start=1):
        img_path = os.path.join(INPUT_FOLDER, img_filename)
        out_filename = os.path.splitext(img_filename)[0] + "_Result.jpg"
        out_path = os.path.join(OUTPUT_FOLDER, out_filename)

        print(f"\n[{idx}/{len(image_files)}] 处理: {img_filename}")

        try:
            image_id = get_image_id_from_coco(COCO_JSON_PATH, img_filename)
            visualize_tta_process(img_path, image_id, COCO_JSON_PATH, MERGED_JSON,
                                  out_path,
                                  score_thr=VIS_SCORE_THR,
                                  iou_thr=IOU_THR,
                                  gt_dup_thr=GT_DUP_THR)
        except Exception as e:
            print(f"  发生错误: {e}")

    print(f"\n批量可视化完成！共处理 {len(image_files)} 张图片，输出至 {OUTPUT_FOLDER}")