"""Opt-in H1 modules. No ground truth is consumed by any inference module.

Distribution supervision is inspired by GFL/D-FINE; candidate auxiliary matching
uses the training-only principle of H-DETR/Co-DETR, not their full architectures.
"""
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from .precision_modules import ChannelLayerNorm, LocalBoxRefiner
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class SemanticDetailBridge(nn.Module):
    def __init__(self, hidden=256, channels=64):
        super().__init__()
        self.detail = nn.ModuleList([nn.Conv2d(hidden, channels, 1) for _ in range(2)])
        self.semantic = nn.ModuleList([nn.Conv2d(hidden, channels, 1) for _ in range(2)])
        self.fusion = nn.ModuleList([nn.Sequential(nn.Conv2d(3*channels, channels, 1),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            ChannelLayerNorm(channels), nn.SiLU()) for _ in range(2)])
        self.output = nn.ModuleList([nn.Conv2d(channels, hidden, 1) for _ in range(2)])
        for layer in self.output:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, memory, sources, masks):
        pieces, features, ratios = [], [], []
        offset = 0
        for i in range(2):
            b, c, h, w = sources[i].shape
            encoded = memory[:, offset:offset+h*w].transpose(1, 2).reshape(b, c, h, w)
            offset += h*w
            valid = (~masks[i])[:, None]
            detail = self.detail[i](sources[i].masked_fill(~valid, 0)).masked_fill(~valid, 0)
            semantic = self.semantic[i](encoded.masked_fill(~valid, 0)).masked_fill(~valid, 0)
            support = F.avg_pool2d(valid.float(), 3, 1, 1)
            average = F.avg_pool2d(detail.float(), 3, 1, 1) / support.clamp_min(1e-6)
            contrast = (detail - average.to(detail)).masked_fill(~valid, 0)
            feature = self.fusion[i](torch.cat((detail, contrast, semantic), 1)).masked_fill(~valid, 0)
            residual = .1 * self.output[i](feature).masked_fill(~valid, 0)
            features.append(feature)
            pieces.append((encoded + residual).flatten(2).transpose(1, 2))
            with torch.no_grad():
                ratios.append(residual.float().square().sum() /
                              encoded.float().masked_fill(~valid, 0).square().sum().clamp_min(1e-8))
        return torch.cat([*pieces, memory[:, offset:]], 1), features, torch.stack(ratios).sqrt()


class CandidateAuxiliaryHead(nn.Module):
    def __init__(self, classification, regression):
        super().__init__()
        self.classification = copy.deepcopy(classification)
        self.regression = copy.deepcopy(regression)

    def initialize_from(self, transformer):
        self.classification.load_state_dict(transformer.enc_out_class_embed.state_dict(), strict=True)
        self.regression.load_state_dict(transformer.enc_out_bbox_embed.state_dict(), strict=True)

    def forward(self, memory, proposals, mask):
        # Proposals are finite for valid tokens; sanitize padding before sigmoid.
        anchors = proposals.masked_fill(~mask[..., None], 0)
        return dict(pred_logits=self.classification(memory),
                    pred_boxes=(self.regression(memory) + anchors).sigmoid(), query_valid_mask=mask)


def distribution_support(device=None):
    t = (torch.arange(33, dtype=torch.float32, device=device)-16)/16
    shape = t.sign()*t.square()
    return torch.stack([.5*shape, .5*shape, math.log(2)*shape, math.log(2)*shape])


def distribution_mean(logits):
    with torch.autocast(device_type=logits.device.type, enabled=False):
        p = logits.float().softmax(-1)
        support = distribution_support(logits.device)
        # Equal symmetric probabilities cancel exactly at initialization.
        return ((p[..., 17:] - p[..., :16].flip(-1)) * support[:, 17:]).sum(-1)


class LocalDistributionRefiner(nn.Module):
    def __init__(self, hidden=256, channels=64, chunk=256):
        super().__init__()
        self.grid_size, self.chunk_size = 5, chunk
        self.encoders = nn.ModuleList([nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1), ChannelLayerNorm(channels), nn.SiLU(),
            nn.Flatten(), nn.Linear(channels*25, 128)) for _ in range(2)])
        self.query_proj = nn.Linear(hidden, 128)
        self.head = nn.Sequential(nn.Linear(384, 256), nn.SiLU(), nn.Linear(256, 132))
        nn.init.zeros_(self.head[-1].weight)
        with torch.no_grad():
            prior = -.5*((torch.arange(33)-16)/4).square()
            self.head[-1].bias.copy_(prior.repeat(4))
        axis = (torch.arange(5, dtype=torch.float32)+.5)/5-.5
        y, x = torch.meshgrid(axis, axis, indexing='ij')
        self.register_buffer('offsets', torch.stack([x, y], -1), persistent=False)

    # The audited sampler uses continuous input-pixel coordinates, padded feature
    # strides, sampled valid-support normalization and detached box grids.
    sample = LocalBoxRefiner.sample

    def forward(self, queries, boxes, features, masks, image_sizes, valid):
        if queries.shape[:2] != boxes.shape[:2] or valid.shape != boxes.shape[:2]:
            raise ValueError('Distribution query/box/mask alignment differs')
        scale = image_sizes[:, [1, 0, 1, 0]].float()[:, None]
        base = boxes.float()
        pixels = base*scale
        all_logits, all_boxes = [], []
        for start in range(0, boxes.shape[1], self.chunk_size):
            end = min(start+self.chunk_size, boxes.shape[1])
            parts = []
            for i, (stride, expansion, minimum) in enumerate(((4, 1.25, 8.), (8, 3., 24.))):
                roi = self.sample(features[i], masks[i], pixels[:, start:end], stride, expansion, minimum)
                parts.append(self.encoders[i](roi).reshape(boxes.shape[0], end-start, 128))
            parts.append(self.query_proj(queries[:, start:end]))
            logits = self.head(torch.cat(parts, -1)).reshape(boxes.shape[0], end-start, 4, 33)
            delta = distribution_mean(logits)
            with torch.autocast(device_type=boxes.device.type, enabled=False):
                coarse = base[:, start:end]
                centre_scale = pixels[:, start:end, 2:].detach().clamp_min(4.)/scale[..., :2]
                refined = torch.cat([coarse[..., :2]+centre_scale*delta[..., :2],
                                     coarse[..., 2:]*delta[..., 2:].exp()], -1)
                refined = torch.where(valid[:, start:end, None], refined, coarse)
            all_logits.append(logits)
            all_boxes.append(refined)
        if not all_boxes:
            return base, base.new_empty((*base.shape[:2], 4, 33))
        return torch.cat(all_boxes, 1), torch.cat(all_logits, 1)


@torch.no_grad()
def auxiliary_matches(outputs, targets, matcher, rounds=3):
    """One FP32 cost computation/CPU copy per image; remove rows, not GTs."""
    matches = []
    for b, target in enumerate(targets):
        valid = outputs['query_valid_mask'][b].nonzero().flatten()
        labels = target['labels']
        if not valid.numel() or not labels.numel():
            empty = torch.empty(0, dtype=torch.long)
            matches.append((empty, empty.clone()))
            continue
        with torch.autocast(device_type=outputs['pred_logits'].device.type, enabled=False):
            logits = outputs['pred_logits'][b, valid].float()
            if ((labels < 0) | (labels >= logits.shape[-1])).any():
                raise ValueError('Auxiliary target class outside configured range')
            p = logits.sigmoid()
            positive = matcher.focal_alpha*(1-p).square()*F.softplus(-logits)
            negative = (1-matcher.focal_alpha)*p.square()*F.softplus(logits)
            boxes = outputs['pred_boxes'][b, valid].float()
            truth = target['boxes'].float()
            cost = (matcher.cost_class*(positive[:, labels]-negative[:, labels]) +
                matcher.cost_bbox*torch.cdist(boxes, truth, p=1) -
                matcher.cost_giou*generalized_box_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(truth)))
        matrix = cost.cpu().numpy()
        if not np.isfinite(matrix).all():
            raise FloatingPointError('Non-finite auxiliary matching cost')
        rows = np.arange(len(valid))
        src, dst = [], []
        for _ in range(rounds):
            if not len(rows):
                break
            i, j = linear_sum_assignment(matrix[rows])
            src.extend(rows[i].tolist()); dst.extend(j.tolist())
            rows = np.delete(rows, i)
        matches.append((valid.cpu()[torch.tensor(src, dtype=torch.long)], torch.tensor(dst, dtype=torch.long)))
    return matches


def distribution_loss(outputs, targets, indices, num_boxes):
    logits_parts, coarse_parts, truth_parts, sizes = [], [], [], []
    for b, (src, dst) in enumerate(indices):
        logits_parts.append(outputs['refine_distribution_logits'][b, src])
        coarse_parts.append(outputs['pred_boxes_coarse'][b, src].detach())
        truth_parts.append(targets[b]['boxes'][dst].detach())
        sizes.append(outputs['joint_image_sizes'][b].expand(len(src), 2))
    logits = torch.cat(logits_parts).float()
    if not logits.shape[0]:
        return logits.sum()*0, {'distribution_outside_ratio': logits.new_zeros(())}
    with torch.autocast(device_type=logits.device.type, enabled=False):
        coarse, truth = torch.cat(coarse_parts).float(), torch.cat(truth_parts).float()
        wh = torch.cat(sizes).float()[:, [1, 0]]
        centre_scale = (coarse[:, 2:]*wh).clamp_min(4.)
        residual = torch.cat([(truth[:, :2]-coarse[:, :2])*wh/centre_scale,
            (truth[:, 2:].clamp_min(1e-8)/coarse[:, 2:].clamp_min(1e-8)).log()], -1).detach()
        support = distribution_support(logits.device)
        inside = (residual >= support[:, 0]) & (residual <= support[:, -1]) & torch.isfinite(residual)
        safe = torch.nan_to_num(residual).clamp(min=support[:, 0], max=support[:, -1])
        hi = torch.searchsorted(support.contiguous(), safe.T.contiguous()).T.clamp(1, 32)
        lo = hi-1
        lower = support[None].expand(len(logits), -1, -1).gather(2, lo[..., None]).squeeze(-1)
        upper = support[None].expand(len(logits), -1, -1).gather(2, hi[..., None]).squeeze(-1)
        fraction = (safe-lower)/(upper-lower)
        logp = logits.log_softmax(-1)
        loss = -(1-fraction)*logp.gather(2, lo[..., None]).squeeze(-1) - fraction*logp.gather(2, hi[..., None]).squeeze(-1)
        per_gt = (loss*inside).sum(-1)/inside.sum(-1).clamp_min(1)
        diagnostics = {'distribution_outside_ratio': (~inside).float().mean().detach()}
        for d, name in enumerate(('cx', 'cy', 'w', 'h')):
            diagnostics['distribution_outside_'+name] = (~inside[:, d]).float().mean().detach()
        # Training-view size only: crops/Mosaic cannot be inverted using orig_size.
        # This is not the official evaluation-area definition.
        vt = (truth[:, 2:]*wh).prod(-1) < 64
        diagnostics['distribution_training_vt_support'] = vt.sum().float()
        diagnostics['distribution_training_vt_outside_count'] = (~inside[vt]).sum().float()
        diagnostics['distribution_entropy'] = -(logp.exp()*logp).sum(-1).mean().detach()
        return per_gt.sum()/num_boxes, diagnostics
