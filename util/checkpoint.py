from pathlib import Path
import random
import warnings

import numpy as np
import torch

from .checkpoint_migration import extract_state_dict, is_legacy_state_dict


def capture_rng_state():
    """Capture the calling rank's random streams at an epoch boundary."""
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Checkpoint CUDA RNG device count differs from this host')
        torch.cuda.set_rng_state_all(state['cuda'])


def load_native_resume(model, checkpoint_path, optimizer=None, scheduler=None, ema=None, scaler=None,
                       best_metrics=None):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = extract_state_dict(checkpoint)
    if is_legacy_state_dict(state):
        raise ValueError('Legacy checkpoints are warm-start weights; use --pretrained instead of --resume')
    metadata = checkpoint.get('run_metadata', {})
    old_args = checkpoint.get('args')
    old_limits = any(getattr(old_args, name, 0) for name in ('max_train_steps', 'max_eval_steps'))
    if optimizer is not None and (metadata.get('smoke_test') or
                                  metadata.get('epoch_complete') is False or old_limits):
        raise ValueError('A smoke/partial checkpoint cannot resume a full epoch; use --pretrained')
    required = []
    if (optimizer is not None or scheduler is not None) and 'epoch' not in checkpoint:
        required.append('epoch')
    if optimizer is not None and 'optimizer' not in checkpoint:
        required.append('optimizer')
    if scheduler is not None and 'lr_scheduler' not in checkpoint:
        required.append('lr_scheduler')
    if ema is not None and 'ema_model' not in checkpoint:
        required.append('ema_model')
    if scaler is not None and 'scaler' not in checkpoint:
        required.append('scaler')
    if best_metrics is not None and optimizer is not None and 'best_metrics' not in checkpoint:
        required.append('best_metrics (use --pretrained for older initialization-only checkpoints)')
    if required:
        raise KeyError(f"Native resume checkpoint is missing: {', '.join(required)}")
    if optimizer is not None and 'epoch' in checkpoint:
        if type(checkpoint['epoch']) is not int or checkpoint['epoch'] < 0:
            raise ValueError('Resume epoch must be a non-negative integer')
    rng_states = checkpoint.get('rng_states')
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size() if distributed else 1
    rank = torch.distributed.get_rank() if distributed else 0
    if optimizer is not None and rng_states is not None and len(rng_states) != world_size:
        raise ValueError('Checkpoint RNG rank count differs from current world size; use --pretrained')
    model.load_state_dict(state, strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
    if ema is not None:
        ema.module.load_state_dict(checkpoint['ema_model'], strict=True)
    if scaler is not None and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
    if best_metrics is not None:
        best_metrics.update(checkpoint.get('best_metrics', {}))
    if optimizer is not None:
        if rng_states is not None:
            restore_rng_state(rng_states[rank])
        else:
            warnings.warn('Checkpoint has no RNG state: resume is not random-stream reproducible',
                          RuntimeWarning)
    return int(checkpoint.get('epoch', -1)) + 1


def migration_report_path(output_dir):
    return Path(output_dir) / 'checkpoint_migration_report.json'
