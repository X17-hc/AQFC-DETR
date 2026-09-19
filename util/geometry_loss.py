"""Matched, last-decoder geometry supervision. No inference or model parameters."""
import math

import torch
import torch.nn.functional as F


DEFAULTS = dict(geometry_loss_weight=0.0, geometry_min_size_pixels=4.0,
                geometry_smooth_l1_beta=0.1, geometry_warmup_epochs=1,
                training_phase_fixed_epoch=None)


def validate_geometry(config):
    errors = []
    for key in config:
        if key.startswith('geometry_') and key not in DEFAULTS:
            errors.append(f'Unknown geometry field: {key}')
    for key in ('geometry_loss_weight', 'geometry_min_size_pixels', 'geometry_smooth_l1_beta'):
        v = config.get(key, DEFAULTS[key])
        if (type(v) not in (int, float) or not math.isfinite(v) or
                (v < 0 if key == 'geometry_loss_weight' else v <= 0)):
            errors.append(f'{key} must be finite and {"non-negative" if key == "geometry_loss_weight" else "positive"}')
    v = config.get('geometry_warmup_epochs', 1)
    if type(v) is not int or v < 1:
        errors.append('geometry_warmup_epochs must be a positive integer')
    phase = config.get('training_phase_fixed_epoch')
    total = config.get('training_phase_total_epochs') or config.get('epochs', 1)
    if phase is not None and (type(phase) is not int or phase < 0 or phase >= total):
        errors.append('training_phase_fixed_epoch must be within the phase schedule')
    return errors


def matched_geometry_loss(pred_boxes, targets, indices, num_boxes,
                          min_size_pixels=4.0, beta=0.1):
    """cxcywh is normalized to each transformed (NOT padded/original) image.

    Smooth-L1 bounds the residual derivative by 1; the pixel floor bounds
    d(residual)/d(pixel coordinate) by 1/min_size_pixels. Normalized-coordinate
    derivatives still include W/H, so the existing global gradient clip remains.
    GT and all scale factors are detached. No clamp on prediction residuals.
    """
    with torch.autocast(device_type=pred_boxes.device.type, enabled=False):
        chunks = []
        for b, (src, dst) in enumerate(indices):
            if src.numel() == 0:
                continue
            size = targets[b]['size'].detach().to(device=pred_boxes.device, dtype=torch.float32)
            if size.shape != (2,):
                raise ValueError('geometry requires transformed target size [height, width]')
            scale = size[[1, 0, 1, 0]]
            truth = targets[b]['boxes'][dst].detach().float() * scale
            pred = pred_boxes[b, src].float() * scale
            denominator = truth[:, 2:].clamp_min(min_size_pixels).repeat(1, 2)
            residual = (pred - truth) / denominator
            chunks.append(F.smooth_l1_loss(residual, torch.zeros_like(residual),
                                          beta=beta, reduction='none').mean(-1))
        if not chunks:
            # A differentiable empty sum cannot turn invalid padding NaNs into NaN*0.
            return pred_boxes.reshape(-1)[:0].float().sum()
        return torch.cat(chunks).sum() / num_boxes


def initialize_geometry_progress(criterion, args, steps_per_epoch):
    if getattr(args, 'geometry_loss_weight', 0.0) <= 0:
        return
    steps = steps_per_epoch * getattr(args, 'geometry_warmup_epochs', 1)
    if steps < 1:
        raise ValueError('Geometry warmup requires a non-empty training loader')
    restored = getattr(args, 'geometry_warmup_steps', None)
    if restored is not None and restored != steps:
        raise ValueError('Geometry resume warmup denominator differs; do not change loader length')
    criterion.geometry_warmup_steps = steps


def update_geometry_weight(criterion):
    if criterion.geometry_max_weight > 0:
        if criterion.geometry_warmup_steps is None:
            raise ValueError('Geometry warmup progress has not been initialized')
        criterion.weight_dict['loss_geometry'] = criterion.geometry_max_weight * min(
            1.0, criterion.quality_successful_updates / criterion.geometry_warmup_steps)


def criterion_progress(criterion):
    state = {'successful_updates': criterion.quality_successful_updates}
    if criterion.geometry_max_weight > 0:
        state['geometry_warmup_steps'] = criterion.geometry_warmup_steps
    return state
