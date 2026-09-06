import torch
import cv2
import numpy as np
import json
import os
import sys
from torchvision.ops import nms

# 将 origin/ 加入 sys.path，使 Baseline 模型的 models/ 包优先被找到
_ORIGIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'origin')
if _ORIGIN_DIR not in sys.path:
    sys.path.insert(0, _ORIGIN_DIR)

from main_aitod import build_model_main  # origin 版本的 build_model_main

from util.box_ops import box_cxcywh_to_xyxy, box_iou
from util.slconfig import SLConfig
from PIL import Image
import torchvision.transforms as T

# --- 全局配置（OpenCV 使用 BGR 格式）---
COLOR_MAP = {
    'TP': (0, 255, 0),  # 绿色 — 真阳性 (预测正确)
    'FP': (0, 0, 255),  # 红色 — 假阳性 (误检/多余的预测)
    'FN': (255, 0, 0),  # 蓝色 — 假阴性 (漏检的真实框)
}
BOX_THICKNESS = 2


# ──────────────────────────────────────────────────────────────────────────────
# 1. 预测框与 GT 框匹配（贪婪匹配）
# ──────────────────────────────────────────────────────────────────────────────
def match_predictions(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    if len(gt_boxes) == 0:
        sorted_order = torch.argsort(pred_scores, descending=True).tolist()
        return sorted_order, [], list(range(len(sorted_order))), []
    if len(pred_boxes) == 0:
        return [], [], [], list(range(len(gt_boxes)))

    sorted_order = torch.argsort(pred_scores, descending=True).tolist()
    ious = box_iou(pred_boxes, gt_boxes)[0]

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


# ──────────────────────────────────────────────────────────────────────────────
# 2. 从 COCO JSON 提取 GT 框 (应用 NMS 消除数据集重复打标)
# ──────────────────────────────────────────────────────────────────────────────
def get_gt_boxes_from_coco(json_path, img_filename, device, dup_iou_thresh=0.85):
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    img_id = None
    for img in coco_data['images']:
        if img['file_name'] == img_filename:
            img_id = img['id']
            break

    if img_id is None:
        print(f"警告: 在标注文件中未找到图片 '{img_filename}'")
        return torch.empty((0, 4), device=device)

    gt_list = []
    for ann in coco_data['annotations']:
        if ann['image_id'] == img_id:
            x, y, w, h = ann['bbox']
            gt_list.append([x, y, x + w, y + h])

    gt_tensor = torch.tensor(gt_list, dtype=torch.float32, device=device)

    if len(gt_tensor) > 0:
        dummy_scores = torch.ones(len(gt_tensor), device=device)
        keep_idx = nms(gt_tensor, dummy_scores, dup_iou_thresh)
        gt_tensor = gt_tensor[keep_idx]

    print(f"   找到 {len(gt_tensor)} 个有效真实标注框 (已去重)。")
    return gt_tensor


# ──────────────────────────────────────────────────────────────────────────────
# 3. 图例绘制
# ──────────────────────────────────────────────────────────────────────────────
def draw_legend(img, position=(10, 10)):
    legend_items = [
        ('TP: Correct Det   (Green)', COLOR_MAP['TP']),
        ('FP: Wrong/Extra Det (Red)', COLOR_MAP['FP']),
        ('FN: Missed GT       (Blue)', COLOR_MAP['FN']),
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.50
    thickness = 1
    line_h = 24
    box_size = 14
    pad = 8

    bx, by = position
    legend_h = len(legend_items) * line_h + pad * 2
    legend_w = 280

    overlay = img.copy()
    cv2.rectangle(overlay, (bx, by), (bx + legend_w, by + legend_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    cv2.rectangle(img, (bx, by), (bx + legend_w, by + legend_h), (180, 180, 180), 1)

    for i, (label, color) in enumerate(legend_items):
        cy = by + pad + i * line_h + box_size // 2
        cv2.rectangle(img, (bx + pad, cy - box_size // 2),
                      (bx + pad + box_size, cy + box_size // 2), color, -1)
        cv2.putText(img, label, (bx + pad + box_size + 8, cy + 5),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# 4. 主可视化函数
# ──────────────────────────────────────────────────────────────────────────────
def run_baseline_vis(img_path, gt_boxes, model, device,
                     score_threshold=0.4, iou_threshold=0.5, pred_nms_thresh=0.5):
    raw_img = cv2.imread(img_path)
    if raw_img is None:
        raise FileNotFoundError(f"无法读取图片: {img_path}")
    h, w = raw_img.shape[:2]

    img_pil = Image.open(img_path).convert("RGB")
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    probs = outputs['pred_logits'].sigmoid()[0]
    scores, _ = probs.max(-1)
    keep = scores > score_threshold

    p_scores = scores[keep]
    p_boxes = box_cxcywh_to_xyxy(outputs['pred_boxes'][0][keep]) \
              * torch.tensor([w, h, w, h], dtype=torch.float32, device=device)

    # 应用 NMS 清除模型产生的重复预测框
    if len(p_boxes) > 0:
        keep_nms = nms(p_boxes, p_scores, pred_nms_thresh)
        p_boxes = p_boxes[keep_nms]
        p_scores = p_scores[keep_nms]

    sorted_order, tp_pos, fp_pos, fn_idx = match_predictions(
        p_boxes, p_scores, gt_boxes, iou_threshold=iou_threshold
    )

    draw_img = raw_img.copy()

    # 画 TP (绿框)
    for pos in tp_pos:
        orig_i = sorted_order[pos]
        b = p_boxes[orig_i].int().tolist()
        cv2.rectangle(draw_img, (b[0], b[1]), (b[2], b[3]), COLOR_MAP['TP'], BOX_THICKNESS)

    # 画 FP (红框)
    for pos in fp_pos:
        orig_i = sorted_order[pos]
        b = p_boxes[orig_i].int().tolist()
        cv2.rectangle(draw_img, (b[0], b[1]), (b[2], b[3]), COLOR_MAP['FP'], BOX_THICKNESS)

    # 画 FN (蓝框)
    for i in fn_idx:
        b = gt_boxes[i].int().tolist()
        cv2.rectangle(draw_img, (b[0], b[1]), (b[2], b[3]), COLOR_MAP['FN'], BOX_THICKNESS)

    draw_legend(draw_img)
    return draw_img


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ================= 配置区 =================
    GT_JSON_PATH = "/workspace/DQDetr/data/path/AITODv2/annotations/aitodv2_test.json"
    CONFIG_FILE = "/workspace/DQDetr/origin/config/DQ_5scale.py"
    WEIGHTS_FILE = "/workspace/DQDetr/models/dqdetr_best305.pth"
    INPUT_FOLDER = "/workspace/DQDetr/data/path/AITODv2/images/test/images"
    OUTPUT_FOLDER = "/workspace/DQDetr/logs/vis_origin"

    SCORE_THR = 0.4
    IOU_THR = 0.5
    PRED_NMS_THR = 0.5  # 预测框去重阈值
    GT_DUP_THR = 0.85   # 数据集重复标注去重阈值
    # ==================================================================

    print("1. 正在初始化 Baseline 模型...")
    args = SLConfig.fromfile(CONFIG_FILE)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    model, _, _ = build_model_main(args)
    checkpoint = torch.load(WEIGHTS_FILE, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    image_files = sorted([
        f for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith(image_extensions)
    ])
    print(f"2. 在 {INPUT_FOLDER} 中找到 {len(image_files)} 张图片，开始批量可视化...")

    for idx, img_filename in enumerate(image_files, start=1):
        img_path = os.path.join(INPUT_FOLDER, img_filename)
        out_filename = os.path.splitext(img_filename)[0] + "_Result.jpg"
        out_path = os.path.join(OUTPUT_FOLDER, out_filename)

        print(f"\n[{idx}/{len(image_files)}] 处理: {img_filename}")

        gt_boxes = get_gt_boxes_from_coco(GT_JSON_PATH, img_filename, device, dup_iou_thresh=GT_DUP_THR)

        res_img = run_baseline_vis(img_path, gt_boxes, model, device,
                                   score_threshold=SCORE_THR,
                                   iou_threshold=IOU_THR,
                                   pred_nms_thresh=PRED_NMS_THR)

        cv2.imwrite(out_path, res_img)
        print(f"  已保存至 {out_path}")

    print(f"\n3. 批量可视化完成！共处理 {len(image_files)} 张图片，输出至 {OUTPUT_FOLDER}")
