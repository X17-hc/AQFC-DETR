# # A+B方案
# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# """
# COCO dataset which returns image_id for evaluation.
#
# Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
# """
# if __name__ == "__main__":
#     # for debug only
#     import os, sys
#
#     sys.path.append(os.path.dirname(sys.path[0]))
#
# import json
# from pathlib import Path
# import random
# import os
#
# import torch
# import torch.utils.data
# import torchvision
# from pycocotools import mask as coco_mask
#
# from datasets.data_util import preparing_dataset
# import datasets.transforms as T
# from util.box_ops import box_cxcywh_to_xyxy, box_iou
#
# __all__ = ['build']
#
#
# # 核心创新 钩子
# # label2compat label_compat2onehot box_label_catter —— 基础钩子，标签重映射与格式化；
# RandomSelectBoxlabels and BboxPerturber: denoising-training hooks.
# # RandomDrop 与 RandomCutout —— 其他钩子
# class label2compat():
#     # COCO 原始ID到连续ID的映射
#     def __init__(self) -> None:
#         self.category_map_str = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
#                                  "11": 11, "13": 12, "14": 13, "15": 14, "16": 15, "17": 16, "18": 17, "19": 18,
#                                  "20": 19, "21": 20, "22": 21, "23": 22, "24": 23, "25": 24, "27": 25, "28": 26,
#                                  "31": 27, "32": 28, "33": 29, "34": 30, "35": 31, "36": 32, "37": 33, "38": 34,
#                                  "39": 35, "40": 36, "41": 37, "42": 38, "43": 39, "44": 40, "46": 41, "47": 42,
#                                  "48": 43, "49": 44, "50": 45, "51": 46, "52": 47, "53": 48, "54": 49, "55": 50,
#                                  "56": 51, "57": 52, "58": 53, "59": 54, "60": 55, "61": 56, "62": 57, "63": 58,
#                                  "64": 59, "65": 60, "67": 61, "70": 62, "72": 63, "73": 64, "74": 65, "75": 66,
#                                  "76": 67, "77": 68, "78": 69, "79": 70, "80": 71, "81": 72, "82": 73, "84": 74,
#                                  "85": 75, "86": 76, "87": 77, "88": 78, "89": 79, "90": 80}
#         self.category_map = {int(k): v for k, v in self.category_map_str.items()}
#
#     def __call__(self, target, img=None):
#         labels = target['labels']
#         res = torch.zeros(labels.shape, dtype=labels.dtype)
#         for idx, item in enumerate(labels):
#             res[idx] = self.category_map[item.item()] - 1
#         # 将target['labels']中的原始COCO ID映射为连续的训练ID
#         target['label_compat'] = res  # 存储映射后的结果
#         if img is not None:
#             return target, img
#         else:
#             return target
#
#
# class label_compat2onehot():
#     def __init__(self, num_class=80, num_output_objs=1):
#         self.num_class = num_class
#         self.num_output_objs = num_output_objs
#         if num_output_objs != 1:
#             raise DeprecationWarning("num_output_objs!=1, which is only used for comparison")
#
#     def __call__(self, target, img=None):
#         # 将离散的标签`label_compat`转换为多标签one-hot格式`label_compat_onehot`
#         # 例如，如果图像中有类别1和3的物体，则生成的one-hot向量在位置1和3上为1，其余为0。
#         labels = target['label_compat']
#         place_dict = {k: 0 for k in range(self.num_class)}
#         if self.num_output_objs == 1:
#             res = torch.zeros(self.num_class)
#             for i in labels:
#                 itm = i.item()
#                 res[itm] = 1.0
#         else:
#             # compat with baseline
#             res = torch.zeros(self.num_class, self.num_output_objs)
#             for i in labels:
#                 itm = i.item()
#                 res[itm][place_dict[itm]] = 1.0
#                 place_dict[itm] += 1
#         target['label_compat_onehot'] = res
#         if img is not None:
#             return target, img
#         else:
#             return target
#
#
# class box_label_catter():
#     def __init__(self):
#         pass
#
#     def __call__(self, target, img=None):
#         labels = target['label_compat']
#         boxes = target['boxes']
#         # 将boxes(N,4)和labels(N)拼接成一个Tensor(N,5)，最后一维是类别标签
#         box_label = torch.cat((boxes, labels.unsqueeze(-1)), 1)
#         target['box_label'] = box_label
#         if img is not None:
#             return target, img
#         else:
#             return target
#
#
# class RandomSelectBoxlabels():
#     # 配置四种去噪任务的概率
#     def __init__(self, num_classes, leave_one_out=False, blank_prob=0.8,
#                  prob_first_item=0.0,
#                  prob_random_item=0.0,
#                  prob_last_item=0.8,
#                  prob_stop_sign=0.2
#                  ) -> None:
#         self.num_classes = num_classes
#         self.leave_one_out = leave_one_out
#         self.blank_prob = blank_prob
#
#         self.set_state(prob_first_item, prob_random_item, prob_last_item, prob_stop_sign)
#
#     def get_state(self):
#         return [self.prob_first_item, self.prob_random_item, self.prob_last_item, self.prob_stop_sign]
#
#     def set_state(self, prob_first_item, prob_random_item, prob_last_item, prob_stop_sign):
#         sum_prob = prob_first_item + prob_random_item + prob_last_item + prob_stop_sign
#         assert sum_prob - 1 < 1e-6, \
#             f"Sum up all prob = {sum_prob}. prob_first_item:{prob_first_item}" \
#             + f"prob_random_item:{prob_random_item}, prob_last_item:{prob_last_item}" \
#             + f"prob_stop_sign:{prob_stop_sign}"
#
#         self.prob_first_item = prob_first_item
#         self.prob_random_item = prob_random_item
#         self.prob_last_item = prob_last_item
#         self.prob_stop_sign = prob_stop_sign
#
#     def sample_for_pred_first_item(self, box_label: torch.FloatTensor):
#         box_label_known = torch.Tensor(0, 5)
#         box_label_unknown = box_label
#         return box_label_known, box_label_unknown
#
#     def sample_for_pred_random_item(self, box_label: torch.FloatTensor):
#         n_select = int(random.random() * box_label.shape[0])
#         box_label = box_label[torch.randperm(box_label.shape[0])]
#         box_label_known = box_label[:n_select]
#         box_label_unknown = box_label[n_select:]
#         return box_label_known, box_label_unknown
#
#     def sample_for_pred_last_item(self, box_label: torch.FloatTensor):
#         box_label_perm = box_label[torch.randperm(box_label.shape[0])]
#         known_label_list = []
#         box_label_known = []
#         box_label_unknown = []
#         for item in box_label_perm:
#             label_i = item[4].item()
#             if label_i in known_label_list:
#                 box_label_known.append(item)
#             else:
#                 # first item
#                 box_label_unknown.append(item)
#                 known_label_list.append(label_i)
#         box_label_known = torch.stack(box_label_known) if len(box_label_known) > 0 else torch.Tensor(0, 5)
#         box_label_unknown = torch.stack(box_label_unknown) if len(box_label_unknown) > 0 else torch.Tensor(0, 5)
#         return box_label_known, box_label_unknown
#
#     def sample_for_pred_stop_sign(self, box_label: torch.FloatTensor):
#         box_label_unknown = torch.Tensor(0, 5)
#         box_label_known = box_label
#         return box_label_known, box_label_unknown
#
#     def __call__(self, target, img=None):
#         box_label = target['box_label']  # K, 5
#
#         dice_number = random.random()
#         # 根据概率dice，选择一种去噪任务来分割已知框和未知框
#         if dice_number < self.prob_first_item:  # 任务1：所有框未知
#             box_label_known, box_label_unknown = self.sample_for_pred_first_item(box_label)
#         elif dice_number < self.prob_first_item + self.prob_random_item:  # 随机分割
#             box_label_known, box_label_unknown = self.sample_for_pred_random_item(box_label)
#         elif dice_number < self.prob_first_item + self.prob_random_item + self.prob_last_item:  # 任务3：每个类别的第一个出现框为未知，其余为已知
#             box_label_known, box_label_unknown = self.sample_for_pred_last_item(box_label)
#         else:  # 任务4：所有框已知
#             box_label_known, box_label_unknown = self.sample_for_pred_stop_sign(box_label)
#
#         target['label_onehot_known'] = label2onehot(box_label_known[:, -1], self.num_classes)
#         target['label_onehot_unknown'] = label2onehot(box_label_unknown[:, -1], self.num_classes)
#         target['box_label_known'] = box_label_known
#         target['box_label_unknown'] = box_label_unknown
#
#         return target, img
#
#
# class RandomDrop():
#     def __init__(self, p=0.2) -> None:
#         self.p = p
#
#     def __call__(self, target, img=None):
#         # 以概率p随机丢弃一些已知框，增加任务的难度和多样性。
#         known_box = target['box_label_known']
#         num_known_box = known_box.size(0)
#         idxs = torch.rand(num_known_box)
#         # indices = torch.randperm(num_known_box)[:int((1-self).p*num_known_box + 0.5 + random.random())]
#         target['box_label_known'] = known_box[idxs > self.p]
#         return target, img
#
#
# class BboxPertuber():
#     def __init__(self, max_ratio=0.02, generate_samples=1000) -> None:
#         self.max_ratio = max_ratio
#         self.generate_samples = generate_samples
#         self.samples = self.generate_pertube_samples()
#         self.idx = 0
#
#     def generate_pertube_samples(self):
#         import torch
#         samples = (torch.rand(self.generate_samples, 5) - 0.5) * 2 * self.max_ratio
#         return samples
#
#     def __call__(self, target, img):
#         known_box = target['box_label_known']  # Tensor(K,5), K known bbox # 已知框
#         K = known_box.shape[0]
#         known_box_pertube = torch.zeros(K, 6)  # 4:bbox, 1:prob, 1:label
#         if K == 0:
#             pass
#         else:
#             if self.idx + K > self.generate_samples:
#                 self.idx = 0
#             delta = self.samples[self.idx: self.idx + K, :]
#             # 对已知框添加随机噪声（位置和尺寸）
#             known_box_pertube[:, :4] = known_box[:, :4] + delta[:, :4]
#             # 计算加噪后框与原始框的IoU，作为“置信度”标签
#             iou = (torch.diag(
#                 box_iou(box_cxcywh_to_xyxy(known_box[:, :4]), box_cxcywh_to_xyxy(known_box_pertube[:, :4]))[0])) * (
#                           1 + delta[:, -1])
#             known_box_pertube[:, 4].copy_(iou)
#             known_box_pertube[:, -1].copy_(known_box[:, -1])
#
#         target['box_label_known_pertube'] = known_box_pertube  # 存储加噪后的已知框
#         return target, img
#
#
# # RandomSelectBoxlabels：将真实标注分割为“已知”和“未知”两部分。
# # 这模拟了不同的噪声情况：pred_first_item（无先验知识）、pred_random_item（部分知识）、pred_last_item（类别首次出现）、pred_stop_sign（全知识）
# # BboxPertuber：对“已知”部分添加噪声，生成噪声输入 box_label_known_pertube。这个带噪声的框和它的 IoU（作为重建目标）将作为额外的“去噪查询”输入给 Transformer 解码器
#
# class RandomCutout():
#     def __init__(self, factor=0.5) -> None:
#         self.factor = factor
#
#     def __call__(self, target, img=None):
#         # 将一些未知框变小并作为已知框加入，模拟困难样本。
#         unknown_box = target['box_label_unknown']  # Ku, 5
#         known_box = target['box_label_known_pertube']  # Kk, 6
#         Ku = unknown_box.size(0)
#
#         known_box_add = torch.zeros(Ku, 6)  # Ku, 6
#         known_box_add[:, :5] = unknown_box
#         known_box_add[:, 5].uniform_(0.5, 1)
#
#         known_box_add[:, :2] += known_box_add[:, 2:4] * (torch.rand(Ku, 2) - 0.5) / 2
#         known_box_add[:, 2:4] /= 2
#
#         target['box_label_known_pertube'] = torch.cat((known_box, known_box_add))
#         return target, img
#
#
# class RandomSelectBoxes():
#     def __init__(self, num_class=80) -> None:
#         Warning("This is such a slow function and will be deprecated soon!!!")
#         self.num_class = num_class
#
#     def __call__(self, target, img=None):
#         boxes = target['boxes']
#         labels = target['label_compat']
#
#         # transform to list of tensors
#         boxs_list = [[] for i in range(self.num_class)]
#         for idx, item in enumerate(boxes):
#             label = labels[idx].item()
#             boxs_list[label].append(item)
#         boxs_list_tensor = [torch.stack(i) if len(i) > 0 else torch.Tensor(0, 4) for i in boxs_list]
#
#         # random selection
#         box_known = []
#         box_unknown = []
#         for idx, item in enumerate(boxs_list_tensor):
#             ncnt = item.shape[0]
#             nselect = int(random.random() * ncnt)  # close in both sides, much faster than random.randint
#
#             item = item[torch.randperm(ncnt)]
#             # random.shuffle(item)
#             box_known.append(item[:nselect])
#             box_unknown.append(item[nselect:])
#
#         # box_known_tensor = [torch.stack(i) if len(i) > 0 else torch.Tensor(0,4) for i in box_known]
#         # box_unknown_tensor = [torch.stack(i) if len(i) > 0 else torch.Tensor(0,4) for i in box_unknown]
#         # print('box_unknown_tensor:', box_unknown_tensor)
#         target['known_box'] = box_known
#         target['unknown_box'] = box_unknown
#         return target, img
#
#
# def label2onehot(label, num_classes):
#     """
#     label: Tensor(K)
#     """
#     res = torch.zeros(num_classes)
#     for i in label:
#         itm = int(i.item())
#         res[itm] = 1.0
#     return res
#
#
# class MaskCrop():
#     def __init__(self) -> None:
#         pass
#
#     def __call__(self, target, img):
#         known_box = target['known_box']
#         h, w = img.shape[1:]  # h,w
#         # imgsize = target['orig_size'] # h,w
#
#         scale = torch.Tensor([w, h, w, h])
#
#         # _cnt = 0
#         for boxes in known_box:
#             if boxes.shape[0] == 0:
#                 continue
#             box_xyxy = box_cxcywh_to_xyxy(boxes) * scale
#             for box in box_xyxy:
#                 x1, y1, x2, y2 = [int(i) for i in box.tolist()]
#                 img[:, y1:y2, x1:x2] = 0
#                 # _cnt += 1
#         # print("_cnt:", _cnt)
#         return target, img
#
#
# dataset_hook_register = {
#     'label2compat': label2compat,
#     'label_compat2onehot': label_compat2onehot,
#     'box_label_catter': box_label_catter,
#     'RandomSelectBoxlabels': RandomSelectBoxlabels,
#     'RandomSelectBoxes': RandomSelectBoxes,
#     'MaskCrop': MaskCrop,
#     'BboxPertuber': BboxPertuber,
# }
#
#
# class CocoDetection(torchvision.datasets.CocoDetection):
#     def __init__(self, img_folder, ann_file, transforms, return_masks,
#                  aux_target_hacks=None, filter_empty_gt=False,
#                  copy_paste=None, dataset_name='coco'):
#         super(CocoDetection, self).__init__(img_folder, ann_file)
#         self._transforms = transforms
#         self.prepare = ConvertCocoPolysToMask(return_masks,dataset_name = dataset_name)
#         self.aux_target_hacks = aux_target_hacks
#         self.filter_empty_gt = filter_empty_gt
#         self.copy_paste = copy_paste
#
#     def __getitem__(self, idx):
#         try:
#             img, target = super(CocoDetection, self).__getitem__(idx)
#         except:
#             print("Error idx: {}".format(idx))
#             idx += 1
#             img, target = super(CocoDetection, self).__getitem__(idx)
#
#         image_id = self.ids[idx]
#         target = {'image_id': image_id, 'annotations': target}
#         img, target = self.prepare(img, target)
#
#         if self._transforms is not None:
#             img, target = self._transforms(img, target)
#
#         if self.filter_empty_gt and len(target['boxes']) == 0:
#             return self.__getitem__(random.randint(0, len(self) - 1))
#
#         # ★ 在 aux_target_hacks 之前执行 Copy-Paste
#         # 此时 img 是归一化后的 tensor，boxes 是 cxcywh normalized
#         if self.copy_paste is not None:
#             img, target = self.copy_paste(img, target)
#
#         if self.aux_target_hacks is not None:
#             for hack_runner in self.aux_target_hacks:
#                 target, img = hack_runner(target, img=img)
#
#         return img, target
#
#
# def convert_coco_poly_to_mask(segmentations, height, width):
#     masks = []
#     for polygons in segmentations:
#         rles = coco_mask.frPyObjects(polygons, height, width)
#         mask = coco_mask.decode(rles)
#         if len(mask.shape) < 3:
#             mask = mask[..., None]
#         mask = torch.as_tensor(mask, dtype=torch.uint8)
#         mask = mask.any(dim=2)
#         masks.append(mask)
#     if masks:
#         masks = torch.stack(masks, dim=0)
#     else:
#         masks = torch.zeros((0, height, width), dtype=torch.uint8)
#     return masks
#
#
# # 数据准备核心
# class ConvertCocoPolysToMask():
#     def __init__(self, return_masks=False, dataset_name='coco'):
#         self.return_masks = return_masks
#         self.dataset_name = dataset_name
#
#     def __call__(self, image, target):
#         w, h = image.size
#
#         image_id = target["image_id"]
#         image_id = torch.tensor([image_id])
#
#         anno = target["annotations"]
#
#         anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
#
#         # 获取image_id,处理annotations
#         boxes = [obj["bbox"] for obj in anno]
#         # guard against no boxes via resizing
#         boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
#         boxes[:, 2:] += boxes[:, :2]  # 转换: [x, y, w, h] -> [x0, y0, x1, y1] （坐标转换与清洗）
#         boxes[:, 0::2].clamp_(min=0, max=w)  # 确保坐标不超出图像范围
#         boxes[:, 1::2].clamp_(min=0, max=h)
#
#         classes = [obj["category_id"] for obj in anno]
#         classes = torch.tensor(classes, dtype=torch.int64)
#
#         # Normalize category_id to [0, num_classes-1] based on dataset
#         if len(classes) > 0:
#             if self.dataset_name in ['aitod', 'aitod_v1', 'aitodv2']:
#                 # AI-TOD: category_id 1-8 -> 0-7
#                 classes = classes - 1
#             elif self.dataset_name == 'visdrone':
#                 # VisDrone: category_id 1-10 -> 0-9 (category_id=0 is ignored)
#                 classes = classes - 1
#             # COCO and others: keep as is (assuming already 0-indexed or mapped)
#
#         if self.return_masks:
#             segmentations = [obj["segmentation"] for obj in anno]
#             masks = convert_coco_poly_to_mask(segmentations, h, w)
#
#         keypoints = None
#         if anno and "keypoints" in anno[0]:
#             keypoints = [obj["keypoints"] for obj in anno]
#             keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
#             num_keypoints = keypoints.shape[0]
#             if num_keypoints:
#                 keypoints = keypoints.view(num_keypoints, -1, 3)
#
#         # ... [处理classes, masks, keypoints] ...
#         keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])  # 过滤无效框（高度或宽度为0）
#         boxes = boxes[keep]
#         classes = classes[keep]
#         if self.return_masks:
#             masks = masks[keep]
#         if keypoints is not None:
#             keypoints = keypoints[keep]
#
#         # ... [构建最终的target字典] ...
#         target = {}
#         target["boxes"] = boxes
#         target["labels"] = classes
#         if self.return_masks:
#             target["masks"] = masks
#         target["image_id"] = image_id
#         if keypoints is not None:
#             target["keypoints"] = keypoints
#
#         # for conversion to coco api
#         area = torch.tensor([obj["area"] for obj in anno])
#         iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
#         target["area"] = area[keep]
#         target["iscrowd"] = iscrowd[keep]
#
#         target["orig_size"] = torch.as_tensor([int(h), int(w)])
#         target["size"] = torch.as_tensor([int(h), int(w)])
#
#         return image, target
#
#
# # 数据增强管道
# def make_coco_transforms(image_set, fix_size=False, strong_aug=False, args=None):
#     normalize = T.Compose([
#         T.ToTensor(),
#         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#
#     # config the params for data aug
#     scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
#     max_size = 1333
#     scales2_resize = [400, 500, 600]
#     scales2_crop = [384, 600]
#
#     # update args from config files
#     scales = getattr(args, 'data_aug_scales', scales)
#     max_size = getattr(args, 'data_aug_max_size', max_size)
#     scales2_resize = getattr(args, 'data_aug_scales2_resize', scales2_resize)
#     scales2_crop = getattr(args, 'data_aug_scales2_crop', scales2_crop)
#
#     # resize them
#     data_aug_scale_overlap = getattr(args, 'data_aug_scale_overlap', None)
#     if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
#         data_aug_scale_overlap = float(data_aug_scale_overlap)
#         scales = [int(i * data_aug_scale_overlap) for i in scales]
#         max_size = int(max_size * data_aug_scale_overlap)
#         scales2_resize = [int(i * data_aug_scale_overlap) for i in scales2_resize]
#         scales2_crop = [int(i * data_aug_scale_overlap) for i in scales2_crop]
#
#     datadict_for_print = {
#         'scales': scales,
#         'max_size': max_size,
#         'scales2_resize': scales2_resize,
#         'scales2_crop': scales2_crop
#     }
#     print("data_aug_params:", json.dumps(datadict_for_print, indent=2))
#
#     if image_set in ['train', 'trainval', 'debug']:
#         # 固定尺寸训练
#         if fix_size:
#             return T.Compose([
#                 T.RandomHorizontalFlip(),
#                 T.RandomResize([(max_size, max(scales))]),
#                 # T.RandomResize([(512, 512)]),
#                 normalize,
#             ])
#
#         # 数据增强
#         if strong_aug:
#             import datasets.sltransform as SLT
#
#             return T.Compose([
#                 T.RandomHorizontalFlip(),  # 随机水平翻转
#                 T.RandomSelect(  # 随机选择两种增强策略之一
#                     T.RandomResize(scales, max_size=max_size),  # 策略1：仅多尺度缩放
#                     T.Compose([  # 策略2: 随机缩放 -> 随机裁剪 -> 缩放回原尺度
#                         T.RandomResize(scales2_resize),
#                         T.RandomSizeCrop(*scales2_crop),
#                         T.RandomResize(scales, max_size=max_size),
#                     ])
#                 ),
#                 SLT.RandomSelectMulti([
#                     SLT.RandomCrop(),
#                     SLT.LightingNoise(),
#                     SLT.AdjustBrightness(2),
#                     SLT.AdjustContrast(2),
#                 ]),
#                 normalize,
#             ])
#
#         return T.Compose([
#             T.RandomHorizontalFlip(),
#             T.RandomSelect(
#                 T.RandomResize(scales, max_size=max_size),
#                 T.Compose([
#                     T.RandomResize(scales2_resize),
#                     T.RandomSizeCrop(*scales2_crop),
#                     T.RandomResize(scales, max_size=max_size),
#                 ])
#             ),
#             normalize,
#         ])
#
#     if image_set in ['val', 'eval_debug', 'train_reg']:
#
#         if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == 'INFO':
#             print("Under debug mode for flops calculation only!!!!!!!!!!!!!!!!")
#             return T.Compose([
#                 T.ResizeDebug((1280, 800)),
#                 normalize,
#             ])
#
#         return T.Compose([
#             T.RandomResize([max(scales)], max_size=max_size),
#             normalize,
#         ])
#
#     if image_set in ['test']:
#
#         if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == 'INFO':
#             print("Under debug mode for flops calculation only!!!!!!!!!!!!!!!!")
#             return T.Compose([
#                 normalize,
#             ])
#
#         return T.Compose([
#             T.RandomResize([max(scales)], max_size=max_size),
#             normalize,
#         ])
#
#     raise ValueError(f'unknown {image_set}')
#
#
# # 组装与构建
# def get_aux_target_hacks_list(image_set, args):
#     if args.modelname in ['q2bs_mask', 'q2bs']:
#         aux_target_hacks_list = [
#             label2compat(),
#             label_compat2onehot(),
#             RandomSelectBoxes(num_class=args.num_classes)
#         ]
#         if args.masked_data and image_set == 'train':
#             # aux_target_hacks_list.append()
#             aux_target_hacks_list.append(MaskCrop())
#     elif args.modelname in ['q2bm_v2', 'q2bs_ce', 'q2op', 'q2ofocal', 'q2opclip', 'q2ocqonly']:
#         aux_target_hacks_list = [
#             label2compat(),
#             label_compat2onehot(),
#             box_label_catter(),
#             RandomSelectBoxlabels(num_classes=args.num_classes,
#                                   prob_first_item=args.prob_first_item,
#                                   prob_random_item=args.prob_random_item,
#                                   prob_last_item=args.prob_last_item,
#                                   prob_stop_sign=args.prob_stop_sign,
#                                   ),
#             BboxPertuber(max_ratio=0.02, generate_samples=1000),  # DN-Detr去噪
#         ]
#     # 模型的配置
#     elif args.modelname in ['q2omask', 'q2osa']:
#         if args.coco_aug:
#             aux_target_hacks_list = [
#                 label2compat(),
#                 label_compat2onehot(),
#                 box_label_catter(),
#                 RandomSelectBoxlabels(num_classes=args.num_classes,
#                                       prob_first_item=args.prob_first_item,
#                                       prob_random_item=args.prob_random_item,
#                                       prob_last_item=args.prob_last_item,
#                                       prob_stop_sign=args.prob_stop_sign,
#                                       ),
#                 RandomDrop(p=0.2),
#                 BboxPertuber(max_ratio=0.02, generate_samples=1000),
#                 RandomCutout(factor=0.5)
#             ]
#         else:
#             aux_target_hacks_list = [
#                 label2compat(),
#                 label_compat2onehot(),
#                 box_label_catter(),
#                 RandomSelectBoxlabels(num_classes=args.num_classes,
#                                       prob_first_item=args.prob_first_item,
#                                       prob_random_item=args.prob_random_item,
#                                       prob_last_item=args.prob_last_item,
#                                       prob_stop_sign=args.prob_stop_sign,
#                                       ),
#                 BboxPertuber(max_ratio=0.02, generate_samples=1000),
#             ]
#     else:
#         aux_target_hacks_list = None
#
#     return aux_target_hacks_list
#
#
# def build(image_set, args):
#     root = Path(args.coco_path)
#     mode = 'instances'
#
#     if args.dataset_file == 'aitod':
#         PATHS = {
#             "train": (root / "images/train/images", root / "annotations" / 'aitod_train_v1.json'),
#             "trainval": (root / "images/trainval/images", root / "annotations" / 'aitod_trainval_v1.json'),
#             "val": (root / "images/val/images", root / "annotations" / 'aitod_val_v1.json'),
#             "eval_debug": (root / "images/val/images", root / "annotations" / 'aitod_val_v1.json'),
#             "test": (root / "images/test/images", root / "annotations" / 'aitod_test_v1.json'),
#         }
#     if args.dataset_file == 'aitodv2':
#         PATHS = {
#             "train": (root / "images/train/images", root / "annotations" / 'aitodv2_train.json'),
#             "trainval": (root / "images/trainval/images", root / "annotations" / 'aitodv2_trainval.json'),
#             "val": (root / "images/val/images", root / "annotations" / 'aitodv2_val.json'),
#             "eval_debug": (root / "images/val/images", root / "annotations" / 'aitodv2_val.json'),
#             "test": (root / "images/test/images", root / "annotations" / 'aitodv2_test.json'),
#         }
#     if args.dataset_file == 'coco':
#         PATHS = {
#             "train": (root / "images/train/images", root / "annotations" / 'instances_train2017.json'),
#             "trainval": (root / "images/trainval/images", root / "annotations" / 'instances_train2017.json'),
#             "val": (root / "images/val/images", root / "annotations" / 'instances_val2017.json'),
#             "eval_debug": (root / "images/val/images", root / "annotations" / 'instances_val2017.json'),
#             "test": (root / "images/test/images", root / "annotations" / 'instances_train2017.json'),
#         }
#     if args.dataset_file == 'visdrone':
#         PATHS = {
#             "train": (root / "VisDrone2019-DET-train/images", root / "annotations_coco/VisDrone2019-DET_train_coco.json"),
#             "val": (root / "VisDrone2019-DET-val/images", root / "annotations_coco/VisDrone2019-DET_val_coco.json"),
#             "test": (root / "VisDrone2019-DET-test-dev/images", root / "VisDrone2019-DET-test-dev/annotations"),
#         }
#
#     # add some hooks to datasets
#     aux_target_hacks_list = get_aux_target_hacks_list(image_set, args)
#     img_folder, ann_file = PATHS[image_set]
#
#     # copy to local path
#     if os.environ.get('DATA_COPY_SHILONG') == 'INFO':
#         preparing_dataset(dict(img_folder=img_folder, ann_file=ann_file), image_set, args)
#
#     try:
#         strong_aug = args.strong_aug
#     except:
#         strong_aug = False
#
#     # 构建 Copy-Paste 增强器（只在训练集启用）
#     copy_paste = None
#     if image_set in ['train', 'trainval']:
#         from datasets.copy_paste import CopyPasteSmallObjects
#         copy_paste = CopyPasteSmallObjects(
#             p=0.5,
#             max_paste=6,
#             area_threshold=0.0016,  # 约对应 32x32 像素在 800px 图中
#             min_area=0.000025,
#             iou_threshold=0.15,
#             cache_size=300
#         )
#
#     # 设置过滤标志，仅在训练集启用
#     filter_empty_gt = image_set in ['train', 'trainval']
#
#     dataset = CocoDetection(img_folder, ann_file,
#                             transforms=make_coco_transforms(image_set, fix_size=args.fix_size, strong_aug=strong_aug,
#                                                             args=args),
#                             return_masks=args.masks,
#                             aux_target_hacks=aux_target_hacks_list,
#                             filter_empty_gt=filter_empty_gt,
#                             copy_paste=copy_paste,  # ← 新增传参
#                             dataset_name=args.dataset_file  # ← 传递数据集名称
#                             )
#
#     return dataset
#
#
# if __name__ == "__main__":
#     # Objects365 Val example
#     dataset_o365 = CocoDetection(
#         '/path/Objects365/train/',
#         "/path/Objects365/slannos/anno_preprocess_train_v2.json",
#         transforms=None,
#         return_masks=False,
#     )
#     print('len(dataset_o365):', len(dataset_o365))


# 4.12：ap=0.320使用的代码
# datasets/coco.py
# A+B方案 + Mosaic增强（修复版）
# 核心修复：
#   1. Mosaic 与 Copy-Paste 互斥触发，杜绝双重强增强叠加
#   2. 支持外部动态调整 mosaic.p（用于 epoch-aware warmup）
#   3. filter_empty_gt 移到 Mosaic 之后、transforms 之前，避免
#      Mosaic 生成的合法稠密样本被误删

if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(sys.path[0]))

import json
from pathlib import Path
import random
import os

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

from datasets.data_util import preparing_dataset
import datasets.transforms as T
from util.box_ops import box_cxcywh_to_xyxy, box_iou

__all__ = ['build']


# ======================================================================
# 数据集钩子（与原版完全相同）
# ======================================================================

class label2compat():
    def __init__(self) -> None:
        self.category_map_str = {
            "1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"10":10,
            "11":11,"13":12,"14":13,"15":14,"16":15,"17":16,"18":17,"19":18,
            "20":19,"21":20,"22":21,"23":22,"24":23,"25":24,"27":25,"28":26,
            "31":27,"32":28,"33":29,"34":30,"35":31,"36":32,"37":33,"38":34,
            "39":35,"40":36,"41":37,"42":38,"43":39,"44":40,"46":41,"47":42,
            "48":43,"49":44,"50":45,"51":46,"52":47,"53":48,"54":49,"55":50,
            "56":51,"57":52,"58":53,"59":54,"60":55,"61":56,"62":57,"63":58,
            "64":59,"65":60,"67":61,"70":62,"72":63,"73":64,"74":65,"75":66,
            "76":67,"77":68,"78":69,"79":70,"80":71,"81":72,"82":73,"84":74,
            "85":75,"86":76,"87":77,"88":78,"89":79,"90":80,
        }
        self.category_map = {int(k): v for k, v in self.category_map_str.items()}

    def __call__(self, target, img=None):
        labels = target['labels']
        res = torch.zeros(labels.shape, dtype=labels.dtype)
        for idx, item in enumerate(labels):
            res[idx] = self.category_map[item.item()] - 1
        target['label_compat'] = res
        if img is not None:
            return target, img
        return target


class label_compat2onehot():
    def __init__(self, num_class=80, num_output_objs=1):
        self.num_class = num_class
        self.num_output_objs = num_output_objs
        if num_output_objs != 1:
            raise DeprecationWarning("num_output_objs!=1")

    def __call__(self, target, img=None):
        labels = target['label_compat']
        place_dict = {k: 0 for k in range(self.num_class)}
        if self.num_output_objs == 1:
            res = torch.zeros(self.num_class)
            for i in labels:
                res[i.item()] = 1.0
        else:
            res = torch.zeros(self.num_class, self.num_output_objs)
            for i in labels:
                itm = i.item()
                res[itm][place_dict[itm]] = 1.0
                place_dict[itm] += 1
        target['label_compat_onehot'] = res
        if img is not None:
            return target, img
        return target


class box_label_catter():
    def __init__(self): pass

    def __call__(self, target, img=None):
        labels = target['label_compat']
        boxes  = target['boxes']
        box_label = torch.cat((boxes, labels.unsqueeze(-1)), 1)
        target['box_label'] = box_label
        if img is not None:
            return target, img
        return target


class RandomSelectBoxlabels():
    def __init__(self, num_classes, leave_one_out=False, blank_prob=0.8,
                 prob_first_item=0.0, prob_random_item=0.0,
                 prob_last_item=0.8, prob_stop_sign=0.2) -> None:
        self.num_classes = num_classes
        self.leave_one_out = leave_one_out
        self.blank_prob = blank_prob
        self.set_state(prob_first_item, prob_random_item, prob_last_item, prob_stop_sign)

    def get_state(self):
        return [self.prob_first_item, self.prob_random_item,
                self.prob_last_item, self.prob_stop_sign]

    def set_state(self, prob_first_item, prob_random_item, prob_last_item, prob_stop_sign):
        s = prob_first_item + prob_random_item + prob_last_item + prob_stop_sign
        assert s - 1 < 1e-6
        self.prob_first_item  = prob_first_item
        self.prob_random_item = prob_random_item
        self.prob_last_item   = prob_last_item
        self.prob_stop_sign   = prob_stop_sign

    def sample_for_pred_first_item(self, box_label):
        return torch.Tensor(0, 5), box_label

    def sample_for_pred_random_item(self, box_label):
        n    = int(random.random() * box_label.shape[0])
        perm = box_label[torch.randperm(box_label.shape[0])]
        return perm[:n], perm[n:]

    def sample_for_pred_last_item(self, box_label):
        perm = box_label[torch.randperm(box_label.shape[0])]
        known, unknown, seen = [], [], []
        for item in perm:
            l = item[4].item()
            if l in seen:
                known.append(item)
            else:
                unknown.append(item)
                seen.append(l)
        known   = torch.stack(known)   if known   else torch.Tensor(0, 5)
        unknown = torch.stack(unknown) if unknown else torch.Tensor(0, 5)
        return known, unknown

    def sample_for_pred_stop_sign(self, box_label):
        return box_label, torch.Tensor(0, 5)

    def __call__(self, target, img=None):
        box_label = target['box_label']
        d = random.random()
        if d < self.prob_first_item:
            kn, un = self.sample_for_pred_first_item(box_label)
        elif d < self.prob_first_item + self.prob_random_item:
            kn, un = self.sample_for_pred_random_item(box_label)
        elif d < self.prob_first_item + self.prob_random_item + self.prob_last_item:
            kn, un = self.sample_for_pred_last_item(box_label)
        else:
            kn, un = self.sample_for_pred_stop_sign(box_label)
        target['label_onehot_known']   = label2onehot(kn[:, -1], self.num_classes)
        target['label_onehot_unknown'] = label2onehot(un[:, -1], self.num_classes)
        target['box_label_known']      = kn
        target['box_label_unknown']    = un
        return target, img


class RandomDrop():
    def __init__(self, p=0.2): self.p = p

    def __call__(self, target, img=None):
        known = target['box_label_known']
        idxs  = torch.rand(known.size(0))
        target['box_label_known'] = known[idxs > self.p]
        return target, img


class BboxPertuber():
    def __init__(self, max_ratio=0.02, generate_samples=1000):
        self.max_ratio = max_ratio
        self.generate_samples = generate_samples
        self.samples = self.generate_pertube_samples()
        self.idx = 0

    def generate_pertube_samples(self):
        return (torch.rand(self.generate_samples, 5) - 0.5) * 2 * self.max_ratio

    def __call__(self, target, img):
        known = target['box_label_known']
        K = known.shape[0]
        known_pertube = torch.zeros(K, 6)
        if K > 0:
            if self.idx + K > self.generate_samples:
                self.idx = 0
            delta = self.samples[self.idx: self.idx + K, :]
            known_pertube[:, :4] = known[:, :4] + delta[:, :4]
            iou = (
                torch.diag(
                    box_iou(
                        box_cxcywh_to_xyxy(known[:, :4]),
                        box_cxcywh_to_xyxy(known_pertube[:, :4])
                    )[0]
                ) * (1 + delta[:, -1])
            )
            known_pertube[:, 4].copy_(iou)
            known_pertube[:, -1].copy_(known[:, -1])
        target['box_label_known_pertube'] = known_pertube
        return target, img


class RandomCutout():
    def __init__(self, factor=0.5): self.factor = factor

    def __call__(self, target, img=None):
        unknown = target['box_label_unknown']
        known   = target['box_label_known_pertube']
        Ku = unknown.size(0)
        add = torch.zeros(Ku, 6)
        add[:, :5] = unknown
        add[:, 5].uniform_(0.5, 1)
        add[:, :2] += add[:, 2:4] * (torch.rand(Ku, 2) - 0.5) / 2
        add[:, 2:4] /= 2
        target['box_label_known_pertube'] = torch.cat((known, add))
        return target, img


class RandomSelectBoxes():
    def __init__(self, num_class=80):
        Warning("This is such a slow function and will be deprecated soon!!!")
        self.num_class = num_class

    def __call__(self, target, img=None):
        boxes  = target['boxes']
        labels = target['label_compat']
        boxs_list = [[] for _ in range(self.num_class)]
        for idx, item in enumerate(boxes):
            boxs_list[labels[idx].item()].append(item)
        boxs_list_tensor = [
            torch.stack(i) if i else torch.Tensor(0, 4)
            for i in boxs_list
        ]
        box_known, box_unknown = [], []
        for item in boxs_list_tensor:
            nc = item.shape[0]
            ns = int(random.random() * nc)
            item = item[torch.randperm(nc)]
            box_known.append(item[:ns])
            box_unknown.append(item[ns:])
        target['known_box']   = box_known
        target['unknown_box'] = box_unknown
        return target, img


def label2onehot(label, num_classes):
    res = torch.zeros(num_classes)
    for i in label:
        res[int(i.item())] = 1.0
    return res


class MaskCrop():
    def __init__(self): pass

    def __call__(self, target, img):
        known = target['known_box']
        h, w  = img.shape[1:]
        scale = torch.Tensor([w, h, w, h])
        for boxes in known:
            if boxes.shape[0] == 0:
                continue
            box_xyxy = box_cxcywh_to_xyxy(boxes) * scale
            for box in box_xyxy:
                x1, y1, x2, y2 = [int(i) for i in box.tolist()]
                img[:, y1:y2, x1:x2] = 0
        return target, img


dataset_hook_register = {
    'label2compat':          label2compat,
    'label_compat2onehot':   label_compat2onehot,
    'box_label_catter':      box_label_catter,
    'RandomSelectBoxlabels': RandomSelectBoxlabels,
    'RandomSelectBoxes':     RandomSelectBoxes,
    'MaskCrop':              MaskCrop,
    'BboxPertuber':          BboxPertuber,
}


# ======================================================================
# CocoDetection（核心修复在此）
# ======================================================================

class CocoDetection(torchvision.datasets.CocoDetection):
    """
    修复说明（相对上一版本）：

    [修复1] Mosaic 与 Copy-Paste 互斥触发
        上一版本：Mosaic(p=0.5) 和 Copy-Paste(p=0.5) 各自独立采样，
                  25% 的图同时经历两种强增强 → FP暴增。
        本版本：  用一次统一的随机数决定使用哪种增强：
                  [0,   mosaic_p)         → 触发 Mosaic，跳过 Copy-Paste
                  [mosaic_p, mosaic_p+cp_p) → 触发 Copy-Paste，跳过 Mosaic
                  [mosaic_p+cp_p, 1.0)   → 两者都不触发（纯净样本）

    [修复2] 支持外部动态调整 mosaic.p
        engine.py 中可在每个 epoch 开始时调用：
            dataset_train.mosaic.p = scheduler.get_mosaic_p(epoch)
        从而实现 Mosaic warmup：前期低概率，中期高概率，后期逐渐退出。

    [修复3] filter_empty_gt 仅过滤增强前的原始空图，最多重采样32次。
        增强后合法出现的空图保留；不递归调用 __getitem__。
        损坏图片直接报错，避免验证时静默替换 image_id。
    """

    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        return_masks,
        aux_target_hacks=None,
        filter_empty_gt=False,
        copy_paste=None,
        mosaic=None,
        # 互斥控制参数（外部可覆盖）
        copy_paste_p: float = 0.5,   # Copy-Paste 的独立触发概率
    ):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms      = transforms
        self.prepare          = ConvertCocoPolysToMask(return_masks)
        self.aux_target_hacks = aux_target_hacks
        self.filter_empty_gt  = filter_empty_gt
        self.copy_paste       = copy_paste
        self.mosaic           = mosaic
        self.copy_paste_p     = copy_paste_p

    # ------------------------------------------------------------------
    # 私有方法：纯 PIL 加载 + prepare()，不触发任何增强
    # 供 Mosaic 内部采样邻图时复用
    # ------------------------------------------------------------------

    def _get_raw_item(self, idx: int):
        """
        返回 (PIL.Image, target_dict)。
        target['boxes'] 为 xyxy 像素坐标（FloatTensor）。
        不触发 Mosaic / transforms / copy_paste。
        """
        try:
            img, target = super(CocoDetection, self).__getitem__(idx)
        except (OSError, ValueError) as exc:
            raise OSError(f'Failed to load image_id={self.ids[idx]}, dataset index={idx}') from exc

        image_id = self.ids[idx]
        target   = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        return img, target

    # ------------------------------------------------------------------
    # __getitem__：完整数据流（互斥增强版）
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int):

        # ── 1. 加载锚图（PIL + prepare）────────────────────────────
        img, target = self._get_raw_item(idx)
        if self.filter_empty_gt:
            for _ in range(32):
                if len(target['boxes']):
                    break
                img, target = self._get_raw_item(random.randrange(len(self)))
            if len(target['boxes']) == 0:
                raise ValueError('Could not find a non-empty raw image after 32 retries; '
                                 'check annotations or disable filter_empty_gt')

        # ── 2. 互斥增强决策 ─────────────────────────────────────────
        # 用一次随机数统一决定本样本走哪条增强路径，避免叠加。
        #
        #   dice ∈ [0,            mosaic_p)        → Mosaic
        #   dice ∈ [mosaic_p,     mosaic_p+cp_p)   → Copy-Paste（在transforms后执行）
        #   dice ∈ [mosaic_p+cp_p, 1.0)            → 纯净样本
        #
        # 其中 mosaic_p 由 self.mosaic.p 动态控制（epoch-aware warmup）。

        mosaic_p = self.mosaic.p if self.mosaic is not None else 0.0
        cp_p     = self.copy_paste_p if self.copy_paste is not None else 0.0

        # Do not silently clip one augmentation's probability behind another.
        if not (0 <= mosaic_p <= 1 and 0 <= cp_p <= 1 and mosaic_p + cp_p <= 1):
            raise ValueError('mosaic_p and copy_paste_p must be probabilities with sum <= 1')
        dice = random.random()

        use_mosaic     = (dice < mosaic_p) and (self.mosaic is not None)
        use_copy_paste = (not use_mosaic) and \
                         (dice < mosaic_p + cp_p) and \
                         (self.copy_paste is not None)

        # ── 3. Mosaic（PIL 层面，在 transforms 之前）────────────────
        if use_mosaic:
            other_idxs      = [random.randint(0, len(self) - 1) for _ in range(3)]
            others          = [self._get_raw_item(i) for i in other_idxs]
            imgs_and_targets = [(img, target)] + others
            img, target     = self.mosaic(imgs_and_targets)

            # An empty result of augmentation is a valid background example.

        # ── 4. 标准 transforms（resize / flip / normalize）────────
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        # ── 5. Copy-Paste（tensor 层面，归一化 cxcywh）────────────
        #   只在 use_copy_paste=True 时触发，与 Mosaic 互斥
        if use_copy_paste:
            img, target = self.copy_paste(img, target)

        # ── 6. Aux target hacks（DN-DETR 去噪等）─────────────────
        if self.aux_target_hacks is not None:
            for hack_runner in self.aux_target_hacks:
                target, img = hack_runner(target, img=img)

        return img, target


# ======================================================================
# 坐标转换工具（原版）
# ======================================================================

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, valid_category_ids=None):
        self.return_masks = return_masks
        self.valid_category_ids = None if valid_category_ids is None else set(valid_category_ids)

    def __call__(self, image, target):
        w, h     = image.size
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        anno     = target["annotations"]
        anno     = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        if self.valid_category_ids is not None:
            anno = [obj for obj in anno if not obj.get('ignore', 0)]
            unknown = {obj['category_id'] for obj in anno} - self.valid_category_ids
            if unknown:
                raise ValueError(f'Unknown non-ignored VisDrone categories: {sorted(unknown)}')

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_kp = keypoints.shape[0]
            if num_kp:
                keypoints = keypoints.view(num_kp, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes   = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"]     = boxes
        target["labels"]    = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"]  = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        area    = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"]      = area[keep]
        target["iscrowd"]   = iscrowd[keep]
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"]      = torch.as_tensor([int(h), int(w)])

        return image, target


# ======================================================================
# 数据增强管道（原版）
# ======================================================================

def make_coco_transforms(image_set, fix_size=False, strong_aug=False, args=None):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales         = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size       = 1333
    scales2_resize = [400, 500, 600]
    scales2_crop   = [384, 600]

    scales         = getattr(args, 'data_aug_scales',        scales)
    max_size       = getattr(args, 'data_aug_max_size',      max_size)
    scales2_resize = getattr(args, 'data_aug_scales2_resize', scales2_resize)
    scales2_crop   = getattr(args, 'data_aug_scales2_crop',  scales2_crop)

    data_aug_scale_overlap = getattr(args, 'data_aug_scale_overlap', None)
    if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
        data_aug_scale_overlap = float(data_aug_scale_overlap)
        scales         = [int(i * data_aug_scale_overlap) for i in scales]
        max_size       = int(max_size * data_aug_scale_overlap)
        scales2_resize = [int(i * data_aug_scale_overlap) for i in scales2_resize]
        scales2_crop   = [int(i * data_aug_scale_overlap) for i in scales2_crop]

    datadict_for_print = {
        'scales': scales, 'max_size': max_size,
        'scales2_resize': scales2_resize, 'scales2_crop': scales2_crop
    }
    print("data_aug_params:", json.dumps(datadict_for_print, indent=2))

    if image_set in ['train', 'trainval', 'debug']:
        if fix_size:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResize([(max_size, max(scales))]),
                normalize,
            ])
        if strong_aug:
            import datasets.sltransform as SLT
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose([
                        T.RandomResize(scales2_resize),
                        T.RandomSizeCrop(*scales2_crop),
                        T.RandomResize(scales, max_size=max_size),
                    ])
                ),
                SLT.RandomSelectMulti([
                    SLT.RandomCrop(),
                    SLT.LightingNoise(),
                    SLT.AdjustBrightness(2),
                    SLT.AdjustContrast(2),
                ]),
                normalize,
            ])

        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(scales2_resize),
                    T.RandomSizeCrop(*scales2_crop),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])

    if image_set in ['val', 'eval_debug', 'train_reg']:
        if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == 'INFO':
            return T.Compose([T.ResizeDebug((1280, 800)), normalize])
        return T.Compose([
            T.RandomResize([max(scales)], max_size=max_size),
            normalize,
        ])

    if image_set in ['test']:
        if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == 'INFO':
            return T.Compose([normalize])
        return T.Compose([
            T.RandomResize([max(scales)], max_size=max_size),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


# ======================================================================
# aux_target_hacks（原版）
# ======================================================================

def get_aux_target_hacks_list(image_set, args):
    if args.modelname in ['q2bs_mask', 'q2bs']:
        aux_target_hacks_list = [
            label2compat(), label_compat2onehot(),
            RandomSelectBoxes(num_class=args.num_classes)
        ]
        if args.masked_data and image_set == 'train':
            aux_target_hacks_list.append(MaskCrop())
    elif args.modelname in ['q2bm_v2', 'q2bs_ce', 'q2op', 'q2ofocal', 'q2opclip', 'q2ocqonly']:
        aux_target_hacks_list = [
            label2compat(), label_compat2onehot(), box_label_catter(),
            RandomSelectBoxlabels(
                num_classes=args.num_classes,
                prob_first_item=args.prob_first_item,
                prob_random_item=args.prob_random_item,
                prob_last_item=args.prob_last_item,
                prob_stop_sign=args.prob_stop_sign,
            ),
            BboxPertuber(max_ratio=0.02, generate_samples=1000),
        ]
    elif args.modelname in ['q2omask', 'q2osa']:
        if args.coco_aug:
            aux_target_hacks_list = [
                label2compat(), label_compat2onehot(), box_label_catter(),
                RandomSelectBoxlabels(
                    num_classes=args.num_classes,
                    prob_first_item=args.prob_first_item,
                    prob_random_item=args.prob_random_item,
                    prob_last_item=args.prob_last_item,
                    prob_stop_sign=args.prob_stop_sign,
                ),
                RandomDrop(p=0.2),
                BboxPertuber(max_ratio=0.02, generate_samples=1000),
                RandomCutout(factor=0.5),
            ]
        else:
            aux_target_hacks_list = [
                label2compat(), label_compat2onehot(), box_label_catter(),
                RandomSelectBoxlabels(
                    num_classes=args.num_classes,
                    prob_first_item=args.prob_first_item,
                    prob_random_item=args.prob_random_item,
                    prob_last_item=args.prob_last_item,
                    prob_stop_sign=args.prob_stop_sign,
                ),
                BboxPertuber(max_ratio=0.02, generate_samples=1000),
            ]
    else:
        aux_target_hacks_list = None
    return aux_target_hacks_list


# ======================================================================
# build()
# ======================================================================

def build(image_set, args):
    root = Path(args.coco_path)

    if args.dataset_file == 'aitod':
        PATHS = {
            "train":      (root / "images/train/images",    root / "annotations" / 'aitod_train_v1.json'),
            "trainval":   (root / "images/trainval/images", root / "annotations" / 'aitod_trainval_v1.json'),
            "val":        (root / "images/val/images",      root / "annotations" / 'aitod_val_v1.json'),
            "eval_debug": (root / "images/val/images",      root / "annotations" / 'aitod_val_v1.json'),
            "test":       (root / "images/test/images",     root / "annotations" / 'aitod_test_v1.json'),
        }
    if args.dataset_file == 'aitodv2':
        PATHS = {
            "train":      (root / "images/train/images",    root / "annotations" / 'aitodv2_train.json'),
            "trainval":   (root / "images/trainval/images", root / "annotations" / 'aitodv2_trainval.json'),
            "val":        (root / "images/val/images",      root / "annotations" / 'aitodv2_val.json'),
            "eval_debug": (root / "images/val/images",      root / "annotations" / 'aitodv2_val.json'),
            "test":       (root / "images/test/images",     root / "annotations" / 'aitodv2_test.json'),
        }
    if args.dataset_file == 'coco':
        PATHS = {
            "train":      (root / "images/train/images",    root / "annotations" / 'instances_train2017.json'),
            "trainval":   (root / "images/trainval/images", root / "annotations" / 'instances_train2017.json'),
            "val":        (root / "images/val/images",      root / "annotations" / 'instances_val2017.json'),
            "eval_debug": (root / "images/val/images",      root / "annotations" / 'instances_val2017.json'),
            "test":       (root / "images/test/images",     root / "annotations" / 'instances_train2017.json'),
        }

    aux_target_hacks_list = get_aux_target_hacks_list(image_set, args)
    if args.dataset_file == 'visdrone':
        from util.dataset_paths import visdrone_paths
        img_folder, ann_file = visdrone_paths(root, image_set)
    else:
        img_folder, ann_file = PATHS[image_set]

    if os.environ.get('DATA_COPY_SHILONG') == 'INFO':
        preparing_dataset(dict(img_folder=img_folder, ann_file=ann_file), image_set, args)

    try:
        strong_aug = args.strong_aug
    except Exception:
        strong_aug = False

    # ── Copy-Paste 对象（只在训练集构建，p 由 copy_paste_p 控制）──
    copy_paste    = None
    copy_paste_p  = 0.0
    if image_set in ['train', 'trainval']:
        from datasets.copy_paste import CopyPasteSmallObjects
        # Copy-Paste 对象本身不持有 p，p 由 CocoDetection 统一管理
        copy_paste = CopyPasteSmallObjects(
            p=1.0,           # ← 设为 1.0：触发决策已移到 CocoDetection
            max_paste=6,
            area_threshold=0.0016,
            min_area=0.000025,
            iou_threshold=0.15,
            cache_size=300,
        )
        copy_paste_p = getattr(args, 'copy_paste_p', 0.5)

    # ── Mosaic 对象（只在训练集构建）────────────────────────────────
    mosaic   = None
    mosaic_p = getattr(args, 'mosaic_p', 0.0)
    if image_set in ['train', 'trainval'] and mosaic_p > 0.0:
        from datasets.mosaic import MosaicDetection
        mosaic = MosaicDetection(
            p=mosaic_p,
            center_ratio_range=getattr(args, 'mosaic_center_ratio', (0.35, 0.65)),
            fill_value=getattr(args, 'mosaic_fill_value', 114),
        )
        print(
            f"[Dataset] Mosaic 已启用：p={mosaic_p}，"
            f"Copy-Paste p={copy_paste_p}，两者互斥触发"
        )
        # 增强分布概览（日志）
        total = min(mosaic_p + copy_paste_p, 1.0)
        pure  = 1.0 - total
        print(
            f"  期望样本比例：Mosaic={mosaic_p:.0%}  "
            f"CopyPaste={copy_paste_p:.0%}  "
            f"纯净={pure:.0%}"
        )

    filter_empty_gt = image_set in ['train', 'trainval']

    dataset = CocoDetection(
        img_folder, ann_file,
        transforms=make_coco_transforms(
            image_set, fix_size=args.fix_size,
            strong_aug=strong_aug, args=args,
        ),
        return_masks=args.masks,
        aux_target_hacks=aux_target_hacks_list,
        filter_empty_gt=filter_empty_gt,
        copy_paste=copy_paste,
        mosaic=mosaic,
        copy_paste_p=copy_paste_p,
    )

    if args.dataset_file == 'visdrone':
        if set(dataset.coco.getCatIds()) != set(range(1, 11)):
            raise ValueError('VisDrone annotations must preserve category IDs 1..10')
        # Preserve evaluation annotations; remove ignored regions only in targets.
        dataset.prepare = ConvertCocoPolysToMask(return_masks=args.masks,
                                                valid_category_ids=range(1, 11))
    return dataset


if __name__ == "__main__":
    dataset_o365 = CocoDetection(
        '/path/Objects365/train/',
        "/path/Objects365/slannos/anno_preprocess_train_v2.json",
        transforms=None,
        return_masks=False,
    )
    print('len(dataset_o365):', len(dataset_o365))
