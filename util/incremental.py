"""Weight-preserving update defaults and separate phase/update clocks."""
DEFAULTS = dict(aligned_box_loss=False, batched_metric_transfer=False,
                non_blocking_transfer=False, classification_loss_type='focal',
                quality_blend_max=.25, quality_blend_warmup_epochs=1,
                training_phase_epoch_offset=0, training_phase_total_epochs=None,
                strict_warmstart=False, expected_pretrained_epoch=None,
                expected_pretrained_sha256='')


def training_phase(args, epoch):
    return (epoch + getattr(args, 'training_phase_epoch_offset', 0),
            getattr(args, 'training_phase_total_epochs', None) or args.epochs)


def quality_progress(args, successful_updates, steps_per_epoch):
    denominator = max(1, steps_per_epoch * getattr(args, 'quality_blend_warmup_epochs', 1))
    return getattr(args, 'quality_blend_max', .25) * min(1., successful_updates / denominator)


def validate_incremental(config):
    import math
    errors=[]
    for key in config:
        if key.startswith(('quality_', 'training_phase_', 'classification_loss_', 'aligned_box_',
                           'batched_metric_', 'non_blocking_', 'expected_pretrained_', 'strict_warm')) and key not in DEFAULTS:
            errors.append(f'Unknown incremental field: {key}')
    for key in ('aligned_box_loss','batched_metric_transfer','non_blocking_transfer','strict_warmstart'):
        if type(config.get(key,DEFAULTS[key])) is not bool:
            errors.append(f'{key} must be boolean')
    if config.get('classification_loss_type','focal') not in ('focal','quality_blend'):
        errors.append('classification_loss_type must be focal or quality_blend')
    value=config.get('quality_blend_max',.25)
    if type(value) not in (float,int) or not math.isfinite(value) or not 0 <= value <= 1:
        errors.append('quality_blend_max must be finite in [0,1]')
    for key, minimum in [('quality_blend_warmup_epochs',1),('training_phase_epoch_offset',0)]:
        v=config.get(key,DEFAULTS[key])
        if type(v) is not int or v < minimum:
            errors.append(f'{key} must be integer >= {minimum}')
    total=config.get('training_phase_total_epochs')
    offset=config.get('training_phase_epoch_offset',0)
    if total is not None and (type(total) is not int or total < 1 or
            (type(offset) is int and total < offset + config.get('epochs',1))):
        errors.append('training_phase_total_epochs must cover offset + epochs')
    source_epoch=config.get('expected_pretrained_epoch')
    if source_epoch is not None and (type(source_epoch) is not int or source_epoch < 0):
        errors.append('expected_pretrained_epoch must be a non-negative integer')
    digest=config.get('expected_pretrained_sha256','')
    if digest and (not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest)):
        errors.append('expected_pretrained_sha256 must be a lowercase SHA-256')
    return errors
