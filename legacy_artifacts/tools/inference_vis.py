import os
import torch
import cv2
import json
import numpy as np
import torchvision.transforms as T
from PIL import Image

from util.slconfig import SLConfig
from util.box_ops import box_cxcywh_to_xyxy
from main_aitod import build_model_main

# 图像预处理流水线（与训练时保持一致）
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 为 AI-TOD 8 类分配独立颜色（BGR 格式），提升多类别可读性
# AI-TOD 类别顺序: airplane, ship, storage-tank, baseball-diamond,
#                  tennis-court, basketball-court, ground-track-field, harbor
CATEGORY_COLORS = [
    (0,   255, 0  ),   # 0 - 绿色   (airplane)
    (0,   165, 255),   # 1 - 橙色   (ship)
    (255, 0,   0  ),   # 2 - 蓝色   (storage-tank)
    (0,   0,   255),   # 3 - 红色   (baseball-diamond)
    (255, 255, 0  ),   # 4 - 青色   (tennis-court)
    (255, 0,   255),   # 5 - 洋红   (basketball-court)
    (128, 0,   128),   # 6 - 紫色   (ground-track-field)
    (0,   128, 255),   # 7 - 天蓝   (harbor)
]


# ──────────────────────────────────────────────────────────────────────────────
# 模型加载
# ──────────────────────────────────────────────────────────────────────────────
def load_model(config_path, checkpoint_path):
    """加载配置与模型权重，返回 (model, device, postprocessors)。"""
    args        = SLConfig.fromfile(config_path)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, _, postprocessors = build_model_main(args)

    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint['model'])
    model.to(args.device)
    model.eval()

    return model, args.device, postprocessors


# ──────────────────────────────────────────────────────────────────────────────
# 推理与可视化
# ──────────────────────────────────────────────────────────────────────────────
def visualize(image_path, model, device, threshold=0.5, id2name=None):
    """对单张图片进行推理，将检测结果绘制在原图上并返回。

    【BUG 修复 1】原版预测框未按置信度排序就绘制，导致低分框可能覆盖高分框。
    【BUG 修复 2】AI-TOD 类别 ID 在 COCO JSON 中从 1 开始，而 sigmoid 输出的
                  label 下标从 0 开始，直接用 str(label) 查询会错位一位。
                  修复：查询时统一 +1 对齐 COCO 标注约定。

    Args:
        image_path: 输入图片路径
        model:      已加载的检测模型
        device:     运行设备
        threshold:  置信度过滤阈值
        id2name:    {str(coco_id): class_name} 字典，可为 None

    Returns:
        img_cv2: 绘制结果的 BGR 图像
    """
    img_cv2 = cv2.imread(image_path)
    if img_cv2 is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    h, w, _ = img_cv2.shape

    img_pil    = Image.open(image_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        outputs = model(img_tensor)

    pred_logits = outputs['pred_logits'][0]   # [num_queries, num_classes]
    pred_boxes  = outputs['pred_boxes'][0]    # [num_queries, 4]  (cxcywh, 归一化)

    # Sigmoid 激活（适用于 Focal Loss / BCE 训练的 DETR 变体）
    probas         = pred_logits.sigmoid()
    scores, labels = probas.max(-1)           # 每个 query 的最高类别分数及类别 id

    # 置信度过滤
    keep   = scores > threshold
    boxes  = pred_boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    # ★ 按置信度降序排列，高分框优先绘制（避免被低分框遮挡）
    sort_idx = torch.argsort(scores, descending=True)
    boxes    = boxes[sort_idx]
    labels   = labels[sort_idx]
    scores   = scores[sort_idx]

    # 坐标转换：cxcywh(归一化) → xyxy(绝对像素)
    boxes = box_cxcywh_to_xyxy(boxes)
    boxes = boxes * torch.tensor([w, h, w, h], dtype=torch.float32, device=device)

    # 绘制
    for box, label, score in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
        xmin, ymin, xmax, ymax = map(int, box)
        label_int = int(label)

        # ★ AI-TOD（COCO 格式）类别 ID 从 1 起，sigmoid 输出从 0 起，+1 对齐
        coco_id   = str(label_int + 1)
        class_name = id2name.get(coco_id, f"cls_{label_int}") if id2name else str(label_int)
        text = f"{class_name}: {score:.2f}"

        color = CATEGORY_COLORS[label_int % len(CATEGORY_COLORS)]

        # 画框
        cv2.rectangle(img_cv2, (xmin, ymin), (xmax, ymax), color, 2)

        # 标签背景 + 文字（防止标签超出图像顶部）
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        label_y = max(ymin, th + baseline + 2)
        cv2.rectangle(img_cv2,
                      (xmin, label_y - th - baseline - 2),
                      (xmin + tw, label_y),
                      color, -1)
        cv2.putText(img_cv2, text,
                    (xmin, label_y - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    return img_cv2


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ================= 配置区（请修改为你的实际路径） =================
    CONFIG_FILE  = "/workspace/DQDetr/config/DQ_5scale_fullV2.py"
    WEIGHTS_FILE = "/workspace/DQDetr/logs/DQDETR_ver7/0412-mosaic+finetunev2/checkpoint0017.pth"
    INPUT_IMAGE  = "/workspace/DQDetr/data/path/AITODv2/images/test/images/0000126_02076_d_0000125__1120_0.png"
    OUTPUT_IMAGE = "/workspace/DQDetr/logs/vis/0000126_02076_d_0000125__1120_0_vis.jpg"
    SCORE_THR    = 0.4
    # ==================================================================

    with open("util/coco_id2name.json", "r") as f:
        coco_id2name = json.load(f)

    print("正在加载模型...")
    model, device, _ = load_model(CONFIG_FILE, WEIGHTS_FILE)

    print("正在进行推理和可视化...")
    res_img = visualize(INPUT_IMAGE, model, device,
                        threshold=SCORE_THR, id2name=coco_id2name)

    out_dir = os.path.dirname(OUTPUT_IMAGE)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(OUTPUT_IMAGE, res_img)
    print(f"可视化完成！结果已保存至 {OUTPUT_IMAGE}")