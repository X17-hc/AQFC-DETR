"""Chunked, FP32 Gaussian targets; no per-object host/device synchronization."""
from functools import lru_cache

import torch


@lru_cache(maxsize=32)
def _kernels(device):
    axis = torch.arange(-4, 5, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    offsets = torch.stack((xx.flatten(), yy.flatten()), -1).long()
    radii = torch.arange(1, 5, device=device, dtype=torch.float32)
    values = torch.exp(-(xx.flatten()[None] ** 2 + yy.flatten()[None] ** 2)
                       / (2 * (radii[:, None] / 3) ** 2))
    support = offsets.abs().amax(-1)[None] <= radii[:, None]
    return offsets, values, support


@torch.no_grad()
def draw_gaussians(heatmap, boxes, valid_height, valid_width, chunk_size=512):
    """Modify a single HxW target using the reference rounding/max convention."""
    if chunk_size <= 0:
        raise ValueError('density_target_chunk_size must be positive')
    boxes = boxes.to(device=heatmap.device, dtype=torch.float32)
    offsets, kernels, support = _kernels(str(heatmap.device))
    for chunk in boxes.split(chunk_size):
        centers = chunk[:, :2] * chunk.new_tensor([valid_width, valid_height])
        centers = torch.minimum(centers.clamp(min=0),
                                chunk.new_tensor([valid_width - 1, valid_height - 1])).long()
        radii = (0.5 * (chunk[:, 2:] * chunk.new_tensor(
            [valid_width, valid_height])).amax(-1)).round().clamp(1, 4).long()
        xy = centers[:, None] + offsets[None]
        valid = (support[radii - 1] & (xy[..., 0] >= 0) &
                 (xy[..., 0] < valid_width) & (xy[..., 1] >= 0) &
                 (xy[..., 1] < valid_height))
        indices = xy[..., 1] * heatmap.shape[-1] + xy[..., 0]
        heatmap.view(-1).scatter_reduce_(0, indices[valid], kernels[radii - 1][valid],
                                        reduce='amax', include_self=True)
