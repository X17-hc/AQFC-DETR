from pathlib import Path

import torch

from .checkpoint_migration import extract_state_dict, is_legacy_state_dict


def load_native_resume(model, checkpoint_path, optimizer=None, scheduler=None, ema=None):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = extract_state_dict(checkpoint)
    if is_legacy_state_dict(state):
        raise ValueError('Legacy checkpoints are warm-start weights; use --pretrained instead of --resume')
    model.load_state_dict(state, strict=True)
    required = []
    if optimizer is not None and 'optimizer' not in checkpoint:
        required.append('optimizer')
    if scheduler is not None and 'lr_scheduler' not in checkpoint:
        required.append('lr_scheduler')
    if ema is not None and 'ema_model' not in checkpoint:
        required.append('ema_model')
    if required:
        raise KeyError(f"Native resume checkpoint is missing: {', '.join(required)}")
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
    if ema is not None:
        ema.module.load_state_dict(checkpoint['ema_model'], strict=True)
    return int(checkpoint.get('epoch', -1)) + 1


def migration_report_path(output_dir):
    return Path(output_dir) / 'checkpoint_migration_report.json'
