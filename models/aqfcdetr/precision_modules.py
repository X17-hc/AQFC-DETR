"""Opt-in detail/context fusion and continuous local box refinement.

No ground truth, class-specific routing, or score threshold is used here.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DetailContextFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.context = nn.Conv2d(channels, channels, 1)
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.norm = ChannelLayerNorm(channels)
        self.output = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, memory, shapes, mask):
        h2, w2 = (int(v) for v in shapes[0])
        h3, w3 = (int(v) for v in shapes[1])
        n2, n3 = h2 * w2, h3 * w3
        b, _, c = memory.shape
        p2 = memory[:, :n2].transpose(1, 2).reshape(b, c, h2, w2)
        p3 = memory[:, n2:n2+n3].transpose(1, 2).reshape(b, c, h3, w3)
        m2 = mask[:, :n2].reshape(b, 1, h2, w2)
        m3 = mask[:, n2:n2+n3].reshape(b, 1, h3, w3)
        context = self.context(p3.masked_fill(m3, 0)).masked_fill(m3, 0)
        context = F.interpolate(context, (h2, w2), mode='bilinear', align_corners=False)
        residual = self.output(F.silu(self.norm(self.depthwise(
            (p2 + context).masked_fill(m2, 0))))).masked_fill(m2, 0)
        return torch.cat([(p2 + residual).flatten(2).transpose(1, 2), memory[:, n2:]], 1)


class LocalBoxRefiner(nn.Module):
    def __init__(self, hidden_dim=256, channels=128, grid_size=5, chunk_size=256):
        super().__init__()
        self.grid_size, self.chunk_size = grid_size, chunk_size
        self.projections = nn.ModuleList([nn.Conv2d(hidden_dim, channels, 1) for _ in range(2)])
        self.encoders = nn.ModuleList([nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1), ChannelLayerNorm(channels), nn.SiLU(),
            nn.Flatten(), nn.Linear(channels * grid_size**2, channels)) for _ in range(2)])
        self.query_proj = nn.Linear(hidden_dim, channels)
        self.head = nn.Sequential(nn.Linear(channels * 3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 4))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        axis = (torch.arange(grid_size, dtype=torch.float32) + .5) / grid_size - .5
        y, x = torch.meshgrid(axis, axis, indexing='ij')
        self.register_buffer('offsets', torch.stack([x, y], -1), persistent=False)

    def sample(self, feature, mask, boxes_pixels, stride, expansion, minimum):
        # Grid depends on predictions, but coordinate gradients are deliberately stopped.
        b, c, h, w = feature.shape
        k = boxes_pixels.shape[1]
        centre = boxes_pixels[..., :2].detach()
        extent = (boxes_pixels[..., 2:].detach() * expansion).clamp_min(minimum)
        xy = centre[:, :, None, None] + extent[:, :, None, None] * self.offsets
        denom = xy.new_tensor([stride * w, stride * h])
        grid = (2 * xy / denom - 1).reshape(b, k*self.grid_size, self.grid_size, 2)
        # FP32 grid_sample is supported on CPU/CUDA and protects small-coordinate geometry.
        with torch.autocast(device_type=feature.device.type, enabled=False):
            valid = (~mask).float()[:, None]
            sampled = F.grid_sample(feature.float() * valid, grid.float(), align_corners=False)
            support = F.grid_sample(valid, grid.float(), align_corners=False)
            sampled = torch.where(support > 1e-6, sampled / support.clamp_min(1e-6), 0.)
        return sampled.reshape(b, c, k, self.grid_size, self.grid_size).permute(0, 2, 1, 3, 4).reshape(
            b*k, c, self.grid_size, self.grid_size)

    def forward(self, queries, boxes, features, masks, image_sizes, query_valid_mask):
        if queries.shape[:2] != boxes.shape[:2] or query_valid_mask.shape != boxes.shape[:2]:
            raise ValueError('matching query, box and mask indices must correspond')
        # image_sizes is (height,width) of the actual image, not padded batch dimensions.
        scale = image_sizes[:, [1, 0, 1, 0]].to(boxes).float()[:, None]
        coarse = boxes.float()
        pixels = coarse * scale
        projected = [proj(feat.masked_fill(mask[:, None], 0)).masked_fill(mask[:, None], 0)
                     for proj, feat, mask in zip(self.projections, features[:2], masks[:2])]
        results = []
        for start in range(0, boxes.shape[1], self.chunk_size):
            end = min(start+self.chunk_size, boxes.shape[1])
            parts = []
            for i, (stride, expansion, minimum) in enumerate(((4, 1.5, 16.), (8, 3., 32.))):
                roi = self.sample(projected[i], masks[i], pixels[:, start:end], stride, expansion, minimum)
                parts.append(self.encoders[i](roi).reshape(boxes.shape[0], end-start, -1))
            parts.append(self.query_proj(queries[:, start:end]))
            delta = self.head(torch.cat(parts, -1)).float().tanh()
            with torch.autocast(device_type=boxes.device.type, enabled=False):
                base = coarse[:, start:end]
                wh_pixels = pixels[:, start:end, 2:]
                # Algebraically identity at delta=0, without a pixel roundtrip of the base box.
                centre = base[..., :2] + .25 * wh_pixels.clamp_min(4.) / scale[..., :2] * delta[..., :2]
                wh = base[..., 2:] * torch.exp(math.log(1.5) * delta[..., 2:])
                refined = torch.cat([centre, wh], -1)
                results.append(torch.where(query_valid_mask[:, start:end, None], refined, base))
        return torch.cat(results, 1) if results else coarse
