"""Explicit opt-in architecture/training contract for the F0/F1 experiment."""
import math
import torch

DEFAULTS = dict(precision24_enabled=False, detail_context_encoder=False,
                local_refiner_enabled=False, precision24_stage_override=None,
                precision24_new_lr=1e-4, precision24_warmup_updates=500)
NEW_PREFIXES = ('local_refiner.', 'transformer.encoder.detail_fusion.', 'transformer.encoder.p2_substitutes.')


def is_fixed_six_refiner(args):
    """F2 is an ordinary encoder + local refiner, NOT transfer reference mode."""
    v = args if isinstance(args, dict) else vars(args)
    return (v.get('precision24_enabled', False) and v.get('local_refiner_enabled', False)
            and not v.get('detail_context_encoder', False)
            and v.get('p2_transfer_version') is None)


def validate(config, errors):
    for key in config:
        if key.startswith(('precision24_', 'detail_context_', 'local_refiner_')) and key not in DEFAULTS:
            errors.append(f'Unknown precision24 field: {key}')
    for key in ('precision24_enabled', 'detail_context_encoder', 'local_refiner_enabled'):
        if type(config.get(key, False)) is not bool:
            errors.append(f'{key} must be boolean')
    active = config.get('precision24_enabled', False)
    if (config.get('detail_context_encoder', False) or config.get('local_refiner_enabled', False)) and not active:
        errors.append('precision24_enabled is required for new architecture modules')
    if not active:
        return
    if config.get('num_feature_levels') != 5 or config.get('enc_layers') != 6 or config.get('hidden_dim') != 256:
        errors.append('precision24 requires five levels, six encoder layers, hidden_dim=256')
    if config.get('return_interm_indices') != [0, 1, 2, 3] or config.get('backbone') != 'resnet50' or config.get('dilation', False):
        errors.append('precision24 requires the undilated ResNet50 stride-4/8 pyramid')
    if config.get('enc_layer_dropout_prob') is not None or config.get('enc_layer_share', False):
        errors.append('precision24 requires unshared, non-dropped encoder layers')
    if (config.get('masks', False) or config.get('geometry_loss_weight', 0) or
            config.get('onecyclelr', False) or config.get('use_ema', False)):
        errors.append('precision24 is detection-only, without geometry_v1, OneCycle or EMA')
    if config.get('two_stage_type') != 'standard':
        errors.append('precision24 requires the standard two-stage proposal path')
    if config.get('param_dict_type', 'default') != 'default':
        errors.append('precision24 requires explicit default parameter groups')
    override = config.get('precision24_stage_override')
    if is_fixed_six_refiner(config) and override is not None:
        errors.append('Fixed-six refinement cannot use an encoder stage override')
    if override is not None and (type(override) is not int or override not in (2, 4, 6)):
        errors.append('precision24_stage_override must be None, 2, 4 or 6')
    if override is not None and config.get('epochs') != 1:
        errors.append('precision24_stage_override is restricted to one-epoch smoke configs')
    if type(config.get('precision24_warmup_updates', 500)) is not int or config.get('precision24_warmup_updates', 500) < 1:
        errors.append('precision24_warmup_updates must be a positive integer')
    lr = config.get('precision24_new_lr', 1e-4)
    if not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        errors.append('precision24_new_lr must be positive and finite')


def signature(args):
    values = args if isinstance(args, dict) else vars(args)
    if not values.get('precision24_enabled', False):
        return {}
    result = {**{k: values.get(k, v) for k, v in DEFAULTS.items()},
            'precision24_revision': 'detail_local_v1',
            'precision24_recipe': 'epoch_0_1_6_2_3_4_4plus_2_lr16_22',
            'precision24_epochs': values.get('epochs'),
            'precision24_batch_size': values.get('batch_size'),
            'precision24_lr_drop_list': values.get('lr_drop_list')}
    if values.get('p2_transfer_version') == 'layerwise_v2':
        result['precision24_revision'] = 'p2_layerwise_v2_local_v1'
        result['precision24_recipe'] = ('fixed_2_heavy_4_light_lr16_22'
            if values.get('p2_transfer_mode') == 'student' else 'reference_six_full_layers')
    elif is_fixed_six_refiner(values):
        result['precision24_revision'] = 'fixed_six_local_v1'
        result['precision24_recipe'] = 'six_full_layers_all_epochs_local_refiner_lr16_22'
    return result


def configure(model, args):
    if not getattr(args, 'precision24_enabled', False):
        return
    from models.aqfcdetr.precision_modules import DetailContextFusion, LocalBoxRefiner
    # Called after every original module is constructed, preserving its initialization/RNG.
    with torch.random.fork_rng(devices=[]):
        if getattr(args, 'detail_context_encoder', False):
            model.transformer.encoder.detail_fusion = DetailContextFusion(model.hidden_dim)
        if getattr(args, 'local_refiner_enabled', False):
            model.local_refiner = LocalBoxRefiner(model.hidden_dim)
    model.precision24_enabled = True
    model.precision24_stage_override = getattr(args, 'precision24_stage_override', None)
    from .p2_transfer import configure as configure_transfer
    configure_transfer(model, args)
    set_epoch(model, 0)


def set_epoch(model, epoch):
    if not getattr(model, 'precision24_enabled', False):
        return
    model.precision24_local_epoch = int(epoch)
    encoder = model.transformer.encoder
    if hasattr(encoder, 'p2_substitutes'):
        encoder.detail_full_layers = 6 if encoder.p2_transfer_mode == 'reference' else 2
        return
    depth = 6 if epoch < 2 else 4 if epoch < 4 else 2
    if not hasattr(encoder, 'detail_fusion'):
        depth = 6
    elif model.precision24_stage_override is not None:
        depth = model.precision24_stage_override
    encoder.detail_full_layers = depth


def state(model, successful_updates):
    if not getattr(model, 'precision24_enabled', False):
        return None
    result = dict(local_epoch=model.precision24_local_epoch,
                detail_full_layers=model.transformer.encoder.detail_full_layers,
                successful_updates=int(successful_updates))
    if hasattr(model, 'p2_transfer_version'):
        result.update(p2_transfer_version=model.p2_transfer_version,
                      p2_transfer_mode=model.transformer.encoder.p2_transfer_mode)
    return result


def restore(model, checkpoint):
    if not getattr(model, 'precision24_enabled', False):
        if checkpoint.get('precision24_state') is not None:
            raise ValueError('precision24 checkpoint requires the matching config even for evaluation')
        return
    saved = checkpoint.get('precision24_state')
    if not isinstance(saved, dict) or saved.get('local_epoch') != checkpoint.get('epoch'):
        raise ValueError('precision24 resume/eval requires matching saved structural epoch')
    if saved.get('successful_updates') != checkpoint.get('criterion_progress', {}).get('successful_updates'):
        raise ValueError('precision24 successful-update state differs')
    if (saved.get('p2_transfer_version') != getattr(model, 'p2_transfer_version', None) or
            saved.get('p2_transfer_mode') != getattr(model.transformer.encoder, 'p2_transfer_mode', None)):
        raise ValueError('P2 transfer structure/mode differs from checkpoint')
    set_epoch(model, saved['local_epoch'])
    if saved.get('detail_full_layers') != model.transformer.encoder.detail_full_layers:
        raise ValueError('precision24 encoder stage differs from checkpoint')


def param_groups(args, model):
    buckets = {'loaded': [], 'backbone': [], 'new': []}
    names = {key: [] for key in buckets}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            key = 'new' if name.startswith(NEW_PREFIXES) else 'backbone' if 'backbone' in name else 'loaded'
            buckets[key].append(parameter)
            names[key].append(name)
    rates = dict(loaded=args.lr, backbone=args.lr_backbone, new=args.precision24_new_lr)
    return [dict(params=parameters, lr=rates[name], precision24_peak_lr=rates[name],
                 precision24_group=name, precision24_names=names[name])
            for name, parameters in buckets.items() if parameters]


def update_lr(optimizer, args, epoch, successful_updates):
    if not getattr(args, 'precision24_enabled', False):
        return
    warmup = min(1., .1 + .9 * successful_updates / args.precision24_warmup_updates)
    decay = .01 if epoch >= 22 else .1 if epoch >= 16 else 1.
    for group in optimizer.param_groups:
        group['lr'] = group['precision24_peak_lr'] * warmup * decay
