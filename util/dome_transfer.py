"""Opt-in, SHA-pinned mature-H1 fine-tuning; never changes historical contracts."""
import json
import math
from pathlib import Path
import torch

SOURCE_SHA = '15862ce9658c871c0152c5120f1562675a5ce294bf7029a4b7ed28b4bbbd9aee'
DEFAULTS = dict(dome_transfer_recipe=None, dome_transfer_smoke=False, density_underestimate_weight=0.,
                density_underestimate_warmup_updates=500,
                protected_density_valid_classes=list(range(8)))

def values(args):
    return args if isinstance(args, dict) else vars(args)

def active(args):
    return values(args).get('dome_transfer_recipe') in ('d0', 'd1')

def validate(v, errors):
    for key in v:
        if key.startswith(('dome_', 'density_underestimate_', 'protected_density_')) and key not in DEFAULTS:
            errors.append('Unknown Dome adaptation field: '+key)
    if v.get('dome_transfer_recipe') not in (None, 'd0', 'd1'):
        errors.append('Unknown dome_transfer_recipe')
    weight = v.get('density_underestimate_weight', 0.)
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
        errors.append('density_underestimate_weight must be finite and nonnegative')
    steps = v.get('density_underestimate_warmup_updates', 500)
    if type(steps) is not int or steps < 1:
        errors.append('density_underestimate_warmup_updates must be a positive integer')
    classes = v.get('protected_density_valid_classes', list(range(8)))
    if (not isinstance(classes, (list, tuple)) or not classes
            or any(type(x) is not int or x < 0 or x >= v.get('num_classes', 9) for x in classes)
            or len(set(classes)) != len(classes)):
        errors.append('protected_density_valid_classes must contain unique valid head indices')
    if not active(v):
        return
    if type(v.get('dome_transfer_smoke',False)) is not bool:
        errors.append('dome_transfer_smoke must be boolean')
    expected = dict(architecture_variant='legacy_joint_v2', epochs=1 if v.get('dome_transfer_smoke') else 3, batch_size=2,
        lr=1e-5, lr_backbone=1e-6, lr_linear_proj_mult=.1, joint_new_lr=1e-5,
        joint_warmup_updates=500, joint_subset_epochs=[], lr_drop_list=[], multi_step_lr=True,
        training_phase_fixed_epoch=23, training_phase_total_epochs=24,
        quality_blend_max=.25, quality_blend_warmup_epochs=0,
        train_transform_mode='native_multiscale', data_aug_scales=[640,704,768,800],
        data_aug_max_size=1333, train_split='trainval', eval_split='test',
        expected_pretrained_sha256=SOURCE_SHA,
        density_underestimate_weight=.1 if v['dome_transfer_recipe']=='d1' else 0.,
        proposal_selection_mode='protected_density' if v['dome_transfer_recipe']=='d1' else 'spatial')
    for key, value in expected.items():
        if v.get(key) != value:
            errors.append(f'Dome fine-tune requires {key}={value!r}')

def signature(args):
    v = values(args)
    # Omit disabled defaults: historical signed checkpoints stay unchanged.
    if not active(v) and not v.get('density_underestimate_weight',0.) and v.get('proposal_selection_mode')!='protected_density':
        return {}
    return {**{k:v.get(k,d) for k,d in DEFAULTS.items()}, 'dome_revision':'protected_density_v1'}

def underestimate_loss(prediction, target, valid):
    p, y, mask = prediction.float(), target.detach().float(), valid.detach().bool()
    # Mask before arithmetic: ignored padding must not propagate NaN or gradient.
    p, y = p.masked_fill(~mask,0), y.masked_fill(~mask,0)
    support = y.flatten(1).sum(1)
    numerator = (y * (y-p).clamp_min(0).square()).flatten(1).sum(1)
    return (numerator/support.clamp_min(1)).mean(), support.mean()

def under_weight(args, successful_updates):
    v = values(args)
    return v.get('density_underestimate_weight',0.) * min(1., max(0,successful_updates)/v.get('density_underestimate_warmup_updates',500))

def initialize(model, args):
    from .experiment import sha256
    if sha256(args.pretrain_model_path) != SOURCE_SHA:
        raise ValueError('Dome fine-tune requires the frozen H1 epoch23 SHA-256')
    checkpoint = torch.load(args.pretrain_model_path, map_location='cpu', weights_only=False)
    if checkpoint.get('epoch') != 23 or 'model' not in checkpoint:
        raise ValueError('Expected the ordinary H1 epoch23 model')
    state = checkpoint['model']
    if any(not torch.isfinite(x).all() for x in state.values() if x.is_floating_point()):
        raise ValueError('Nonfinite source checkpoint')
    model.load_state_dict(state, strict=True)
    from .legacy_joint import tensor_digest
    model.joint_common_sha = tensor_digest(state)
    report = dict(source_checkpoint=args.pretrain_model_path, source_sha256=SOURCE_SHA,
        source_epoch=23, ordinary_model=True, strict=True, loaded_keys=sorted(state),
        coverage_by_numel=1., missing_keys=[], unexpected_keys=[], shape_mismatched_keys=[],
        common_parameter_sha256=model.joint_common_sha, optimizer_restored=False)
    Path(args.output_dir,'checkpoint_migration_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
