# # ------------------------------------------------------------------------
# # DINO
# # Copyright (c) 2022 IDEA. All Rights Reserved.
# # Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# # ------------------------------------------------------------------------
# # Modules to compute the matching cost and solve the corresponding LSAP.
# # Copyright (c) 2021 Microsoft. All Rights Reserved.
# # Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# # ------------------------------------------------------------------------
# # Modified from DETR (https://github.com/facebookresearch/detr)
# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# # ------------------------------------------------------------------------
# # Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# # Copyright (c) 2020 SenseTime. All Rights Reserved.
# # ------------------------------------------------------------------------
#
#
# import torch, os
# from torch import nn
# from scipy.optimize import linear_sum_assignment
#
# from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
#
#
# class HungarianMatcher(nn.Module):
#     """This class computes an assignment between the targets and the predictions of the network
#     For efficiency reasons, the targets don't include the no_object. Because of this, in general,
#     there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
#     while the others are un-matched (and thus treated as non-objects).
#     """
#
#     def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, focal_alpha = 0.25):
#         """Creates the matcher
#         Params:
#             cost_class: This is the relative weight of the classification error in the matching cost
#             cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
#             cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
#         """
#         super().__init__()
#         self.cost_class = cost_class
#         self.cost_bbox = cost_bbox
#         self.cost_giou = cost_giou
#         assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
#
#         self.focal_alpha = focal_alpha
#
#     @torch.no_grad()
#     def forward(self, outputs, targets):
#         """ Performs the matching
#         Params:
#             outputs: This is a dict that contains at least these entries:
#                  "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
#                  "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
#             targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
#                  "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
#                            objects in the target) containing the class labels
#                  "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
#         Returns:
#             A list of size batch_size, containing tuples of (index_i, index_j) where:
#                 - index_i is the indices of the selected predictions (in order)
#                 - index_j is the indices of the corresponding selected targets (in order)
#             For each batch element, it holds:
#                 len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
#         """
#
#         bs, num_queries = outputs["pred_logits"].shape[:2]
#
#         # We flatten to compute the cost matrices in a batch
#         out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
#         out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
#
#         # Also concat the target labels and boxes
#         tgt_ids = torch.cat([v["labels"] for v in targets])
#         tgt_bbox = torch.cat([v["boxes"] for v in targets])
#
#         # Compute the classification cost.
#         alpha = self.focal_alpha
#         gamma = 2.0
#         neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
#         pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
#         cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
#
#         # Compute the L1 cost between boxes
#         cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
#
#         # Compute the giou cost betwen boxes
#         cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
#
#         # Final cost matrix
#         C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
#         C = C.view(bs, num_queries, -1).cpu()
#
#         sizes = [len(v["boxes"]) for v in targets]
#         indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
#         return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
#
#
# class SimpleMinsumMatcher(nn.Module):
#     """This class computes an assignment between the targets and the predictions of the network
#     For efficiency reasons, the targets don't include the no_object. Because of this, in general,
#     there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
#     while the others are un-matched (and thus treated as non-objects).
#     """
#
#     def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, focal_alpha = 0.25):
#         """Creates the matcher
#         Params:
#             cost_class: This is the relative weight of the classification error in the matching cost
#             cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
#             cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
#         """
#         super().__init__()
#         self.cost_class = cost_class
#         self.cost_bbox = cost_bbox
#         self.cost_giou = cost_giou
#         assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
#
#         self.focal_alpha = focal_alpha
#
#     @torch.no_grad()
#     def forward(self, outputs, targets):
#         """ Performs the matching
#         Params:
#             outputs: This is a dict that contains at least these entries:
#                  "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
#                  "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
#             targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
#                  "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
#                            objects in the target) containing the class labels
#                  "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
#         Returns:
#             A list of size batch_size, containing tuples of (index_i, index_j) where:
#                 - index_i is the indices of the selected predictions (in order)
#                 - index_j is the indices of the corresponding selected targets (in order)
#             For each batch element, it holds:
#                 len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
#         """
#
#         bs, num_queries = outputs["pred_logits"].shape[:2]
#
#         # We flatten to compute the cost matrices in a batch
#         out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
#         out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
#
#         # Also concat the target labels and boxes
#         tgt_ids = torch.cat([v["labels"] for v in targets])
#         tgt_bbox = torch.cat([v["boxes"] for v in targets])
#
#         # Compute the classification cost.
#         alpha = self.focal_alpha
#         gamma = 2.0
#         neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
#         pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
#         cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
#
#         # Compute the L1 cost between boxes
#         cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
#
#         # Compute the giou cost betwen boxes
#         cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
#
#         # Final cost matrix
#         C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
#         C = C.view(bs, num_queries, -1)
#
#         sizes = [len(v["boxes"]) for v in targets]
#         indices = []
#         device = C.device
#         for i, (c, _size) in enumerate(zip(C.split(sizes, -1), sizes)):
#             weight_mat = c[i]
#             idx_i = weight_mat.min(0)[1]
#             idx_j = torch.arange(_size).to(device)
#             indices.append((idx_i, idx_j))
#
#         return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
#
#
# def build_matcher(args):
#     assert args.matcher_type in ['HungarianMatcher', 'SimpleMinsumMatcher'], "Unknown args.matcher_type: {}".format(args.matcher_type)
#     if args.matcher_type == 'HungarianMatcher':
#         return HungarianMatcher(
#             cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou,
#             focal_alpha=args.focal_alpha
#         )
#     elif args.matcher_type == 'SimpleMinsumMatcher':
#         return SimpleMinsumMatcher(
#             cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou,
#             focal_alpha=args.focal_alpha
#         )
#     else:
#         raise NotImplementedError("Unknown args.matcher_type: {}".format(args.matcher_type))


# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modules to compute the matching cost and solve the corresponding LSAP.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------


import torch, os
from torch import nn
from scipy.optimize import linear_sum_assignment

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, focal_alpha=0.25):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

        self.focal_alpha = focal_alpha

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        batch_size, num_queries = outputs["pred_logits"].shape[:2]
        valid_masks = outputs.get(
            "query_valid_mask",
            torch.ones(batch_size, num_queries, dtype=torch.bool,
                       device=outputs["pred_logits"].device))
        assignments = []
        for batch_index, target in enumerate(targets):
            valid_indices = torch.nonzero(valid_masks[batch_index], as_tuple=False).flatten()
            target_ids = target["labels"]
            target_boxes = target["boxes"].float()
            if valid_indices.numel() == 0 or target_ids.numel() == 0:
                empty = torch.empty(0, dtype=torch.int64)
                assignments.append((empty, empty.clone()))
                continue

            logits = outputs["pred_logits"][batch_index, valid_indices].float()
            probability = logits.sigmoid()
            predicted_boxes = outputs["pred_boxes"][batch_index, valid_indices].float()
            if ((target_ids < 0) | (target_ids >= probability.shape[1])).any():
                raise ValueError('Target label is outside the configured class range')
            alpha, gamma = self.focal_alpha, 2.0
            # Stable -log(sigmoid) forms: half-precision probabilities can round to 0/1.
            negative = (1 - alpha) * probability.pow(gamma) * torch.nn.functional.softplus(logits)
            positive = alpha * (1 - probability).pow(gamma) * torch.nn.functional.softplus(-logits)
            class_cost = positive[:, target_ids] - negative[:, target_ids]
            bbox_cost = torch.cdist(predicted_boxes, target_boxes, p=1)
            giou_cost = -generalized_box_iou(
                box_cxcywh_to_xyxy(predicted_boxes), box_cxcywh_to_xyxy(target_boxes))
            cost = (self.cost_bbox * bbox_cost + self.cost_class * class_cost +
                    self.cost_giou * giou_cost)
            local_prediction, target_index = linear_sum_assignment(cost.cpu())
            prediction_index = valid_indices[torch.as_tensor(
                local_prediction, dtype=torch.long, device=valid_indices.device)].cpu()
            assignments.append((prediction_index, torch.as_tensor(target_index, dtype=torch.int64)))
        return assignments


class SimpleMinsumMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, focal_alpha=0.25):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

        self.focal_alpha = focal_alpha

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        if 'query_valid_mask' in outputs:
            assignments = []
            for batch_index, target in enumerate(targets):
                valid_indices = torch.nonzero(
                    outputs['query_valid_mask'][batch_index], as_tuple=False).flatten()
                if valid_indices.numel() == 0 or target['labels'].numel() == 0:
                    empty = torch.empty(0, dtype=torch.int64)
                    assignments.append((empty, empty.clone()))
                    continue
                compact = {
                    'pred_logits': outputs['pred_logits'][batch_index:batch_index + 1, valid_indices],
                    'pred_boxes': outputs['pred_boxes'][batch_index:batch_index + 1, valid_indices],
                }
                local_predictions, target_indices = self.forward(compact, [target])[0]
                assignments.append((valid_indices[local_predictions.to(valid_indices.device)].cpu(),
                                    target_indices.cpu()))
            return assignments

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        # Safety check: ensure tgt_ids are within valid range
        num_classes = out_prob.shape[1]
        if tgt_ids.numel() > 0:
            max_label = tgt_ids.max().item()
            min_label = tgt_ids.min().item()
            if max_label >= num_classes or min_label < 0:
                print(
                    f"[Matcher Warning] Invalid label detected! Max: {max_label}, Min: {min_label}, num_classes: {num_classes}. Clamping to valid range.")
                tgt_ids = tgt_ids.clamp(min=0, max=num_classes - 1)

        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1)

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        device = C.device
        for i, (c, _size) in enumerate(zip(C.split(sizes, -1), sizes)):
            weight_mat = c[i]
            idx_i = weight_mat.min(0)[1]
            idx_j = torch.arange(_size).to(device)
            indices.append((idx_i, idx_j))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    assert args.matcher_type in ['HungarianMatcher', 'SimpleMinsumMatcher'], "Unknown args.matcher_type: {}".format(
        args.matcher_type)
    if args.matcher_type == 'HungarianMatcher':
        return HungarianMatcher(
            cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha
        )
    elif args.matcher_type == 'SimpleMinsumMatcher':
        return SimpleMinsumMatcher(
            cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha
        )
    else:
        raise NotImplementedError("Unknown args.matcher_type: {}".format(args.matcher_type))
