# validate_boundaries.py
import torch
from datasets import build_dataset
from util.box_ops import compute_box_size_distribution
import numpy as np


def analyze_dataset():
    args = type('Args', (), {
        'dataset_file': 'aitodv2',
        'coco_path': 'data/path/AITODv2',
        'device': 'cuda',
        'modelname': 'test01',
        'fix_size': 'store_true',
        'masks': False
    })()

    dataset = build_dataset(image_set='trainval', args=args)

    all_areas = []
    for i in range(min(1000, len(dataset))):  # 采样1000张图
        img, target = dataset[i]
        boxes = target['boxes']
        areas = (boxes[:, 2] * boxes[:, 3]) * (800 * 800)  # 转像素面积
        all_areas.extend(areas.tolist())

    all_areas = np.array(all_areas)

    print(f"数据集面积统计:")
    print(f"  P60: {np.percentile(all_areas, 60):.1f} 像素")
    print(f"  P85: {np.percentile(all_areas, 85):.1f} 像素")
    print(f"  P95: {np.percentile(all_areas, 95):.1f} 像素")
    print(f"\n建议边界范围:")
    print(f"  b1 ≈ {np.sqrt(np.percentile(all_areas, 60)):.1f} 像素")
    print(f"  b2 ≈ {np.sqrt(np.percentile(all_areas, 85)):.1f} 像素")
    print(f"  b3 ≈ {np.sqrt(np.percentile(all_areas, 95)):.1f} 像素")


if __name__ == '__main__':
    analyze_dataset()