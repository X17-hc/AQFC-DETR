# datasets/mosaic.py
"""
Mosaic 2×2 augmentation，专为 AI-TOD 极小目标设计。

调用时机：在标准 transforms 之前（PIL Image 层面），
此时 target['boxes'] 为 xyxy 像素坐标（FloatTensor）。

设计要点：
  1. 以"中心点是否落在格子内"作为保留条件，而非面积比。
     对于 verytiny（2~16 px）目标，任何裁剪都几乎摧毁目标，
     中心判断更稳健。
  2. 格子中心抖动范围设为 [0.35, 0.65]，防止某一格子退化为极细条带。
  3. 不固定输出尺寸，继承锚图像的原始尺寸，保证后续 RandomResize
     的尺度统计与单图训练一致。
  4. 背景填充 RGB=114（ImageNet 灰），与 YOLOv5 一致。
"""

import random
import torch
from PIL import Image


class MosaicDetection:
    """
    2×2 Mosaic 增强。

    调用约定（与 CocoDetection._get_raw_item 配套）：
        输入：4 个 (PIL.Image, target_dict) 元组的列表，
              target_dict 来自 ConvertCocoPolysToMask，包含：
                boxes     FloatTensor [N, 4]  xyxy 像素坐标
                labels    LongTensor  [N]     原始 category_id
                area      FloatTensor [N]
                iscrowd   LongTensor  [N]
                image_id  LongTensor  [1]
                size      LongTensor  [2]  (H, W)
                orig_size LongTensor  [2]  (H, W)
        输出：(PIL.Image, target_dict)，与输入 schema 相同，
              可直接送入 make_coco_transforms。

    参数：
        p                  触发概率（由 CocoDetection.__getitem__ 判断）
        center_ratio_range Mosaic 中心点位置的合法范围（相对于画布）
        fill_value         背景填充灰度值
    """

    def __init__(
        self,
        p: float = 0.5,
        center_ratio_range=(0.35, 0.65),
        fill_value: int = 114,
    ):
        assert 0.0 < center_ratio_range[0] < center_ratio_range[1] < 1.0
        self.p = p
        self.center_ratio_range = center_ratio_range
        self.fill_value = fill_value

    # ------------------------------------------------------------------ #
    #  内部工具                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _paste_tile(canvas, img_resized, tx1, ty1):
        """把缩放后的小图贴到画布指定位置（居中于格子内）。"""
        canvas.paste(img_resized, (tx1, ty1))

    # ------------------------------------------------------------------ #
    #  主接口                                                               #
    # ------------------------------------------------------------------ #

    def __call__(self, imgs_and_targets):
        """
        Args:
            imgs_and_targets: list[4] of (PIL.Image, dict)
        Returns:
            canvas  : PIL.Image
            out_tgt : dict
        """
        assert len(imgs_and_targets) == 4, (
            f"Mosaic 需要恰好 4 张图，收到 {len(imgs_and_targets)}"
        )

        if any('masks' in target or 'keypoints' in target for _, target in imgs_and_targets):
            raise ValueError('Mosaic supports detection boxes only, not masks/keypoints')
        anchor_img, anchor_tgt = imgs_and_targets[0]
        out_w, out_h = anchor_img.size          # PIL: (width, height)

        # ── 画布 ──────────────────────────────────────────────────────
        canvas = Image.new("RGB", (out_w, out_h), (self.fill_value,) * 3)

        # ── Mosaic 中心（像素） ───────────────────────────────────────
        lo, hi = self.center_ratio_range
        cx = int(random.uniform(lo, hi) * out_w)
        cy = int(random.uniform(lo, hi) * out_h)

        # 4 个格子在画布中的区域 (x1, y1, x2, y2)
        # 顺序：TL → TR → BL → BR，与 imgs_and_targets 对应
        tile_regions = [
            (0,   0,   cx,    cy),       # TL
            (cx,  0,   out_w, cy),       # TR
            (0,   cy,  cx,    out_h),    # BL
            (cx,  cy,  out_w, out_h),    # BR
        ]

        all_boxes   = []
        all_labels  = []
        all_areas   = []
        all_iscrowd = []

        for (img, tgt), (rx1, ry1, rx2, ry2) in zip(
            imgs_and_targets, tile_regions
        ):
            tw = rx2 - rx1
            th = ry2 - ry1
            if tw <= 0 or th <= 0:
                continue  # 退化格子，跳过

            iw, ih = img.size

            # ── 等比缩放，使图像填满格子（取较小缩放系数） ─────────
            scale = min(tw / iw, th / ih)
            sw = max(int(round(iw * scale)), 1)
            sh = max(int(round(ih * scale)), 1)

            # 在格子内居中放置
            ox = rx1 + (tw - sw) // 2   # paste 位置 x
            oy = ry1 + (th - sh) // 2   # paste 位置 y

            img_r = img.resize((sw, sh), Image.BILINEAR)
            canvas.paste(img_r, (ox, oy))

            # ── 转换 boxes 到画布坐标系 ──────────────────────────────
            if tgt["boxes"].shape[0] == 0:
                continue

            boxes  = tgt["boxes"].float().clone()   # [N, 4] xyxy px
            labels = tgt["labels"].clone()

            # 先缩放，再平移到贴图位置
            # PIL uses rounded integer sizes: each axis has its own true scale.
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * (sw / iw) + ox
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * (sh / ih) + oy

            # ── 中心点过滤（核心）────────────────────────────────────
            # 极小目标任何面积裁剪都接近全损，改用中心判断：
            # 只要目标中心落在本格子区域内就保留。
            cx_b = (boxes[:, 0] + boxes[:, 2]) * 0.5
            cy_b = (boxes[:, 1] + boxes[:, 3]) * 0.5
            keep = (
                (cx_b >= rx1) & (cx_b < rx2) &
                (cy_b >= ry1) & (cy_b < ry2)
            )
            if keep.sum() == 0:
                continue

            boxes  = boxes[keep]
            labels = labels[keep]

            # 裁剪到格子范围（边缘可能略有溢出）
            boxes[:, 0].clamp_(min=rx1, max=rx2 - 1)
            boxes[:, 1].clamp_(min=ry1, max=ry2 - 1)
            boxes[:, 2].clamp_(min=rx1 + 1, max=rx2)
            boxes[:, 3].clamp_(min=ry1 + 1, max=ry2)

            # 丢弃裁剪后面积为 0 的框
            valid = (
                (boxes[:, 2] - boxes[:, 0] > 0) &
                (boxes[:, 3] - boxes[:, 1] > 0)
            )
            if valid.sum() == 0:
                continue

            boxes  = boxes[valid]
            labels = labels[valid]

            all_boxes.append(boxes)
            all_labels.append(labels)

            all_areas.append((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
            if "iscrowd" in tgt and tgt["iscrowd"].shape[0] > 0:
                all_iscrowd.append(tgt["iscrowd"][keep][valid])
            else:
                all_iscrowd.append(torch.zeros(len(boxes), dtype=torch.int64))

        # ── 拼合 target ───────────────────────────────────────────────
        out_tgt = {
            "image_id": anchor_tgt["image_id"],
            "orig_size": torch.tensor([out_h, out_w]),
            "size":      torch.tensor([out_h, out_w]),
        }

        if all_boxes:
            out_tgt["boxes"]  = torch.cat(all_boxes,  dim=0)
            out_tgt["labels"] = torch.cat(all_labels, dim=0)
        else:
            out_tgt["boxes"]  = torch.zeros((0, 4), dtype=torch.float32)
            out_tgt["labels"] = torch.zeros(0,      dtype=torch.int64)

        out_tgt["area"] = (
            torch.cat(all_areas, dim=0)
            if all_areas else torch.zeros(0, dtype=torch.float32)
        )
        out_tgt["iscrowd"] = (
            torch.cat(all_iscrowd, dim=0)
            if all_iscrowd else torch.zeros(0, dtype=torch.int64)
        )

        return canvas, out_tgt
