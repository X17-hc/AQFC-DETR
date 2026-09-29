"""H1 versioned contract, legacy allowlist, independent LR and checkpoint checks."""
import hashlib
import json
import math
from pathlib import Path
import torch

VARIANTS = ('legacy_joint_v2', 'legacy_joint_control_v2')
SOURCE_SHA = 'f854eec1f0d0ebf6e131d01e0b2826ce5690b593a5ed7f3fb8203453a62936e4'
NEW_PREFIXES = ('transformer.semantic_detail_bridge.', 'transformer.candidate_auxiliary_head.',
                'distribution_refiner.')
DEFAULTS = dict(joint_warmup_updates=500, joint_new_lr=1e-4,
                joint_subset_epochs=[0, 3, 7, 12],
                joint_subset_ids='/workspace/AQFC-DETR/outputs/legacy24_diagnosis_20260917/subset_ids.json')


def active(args):
    return (args.get('architecture_variant') if isinstance(args, dict) else
            getattr(args, 'architecture_variant', None)) in VARIANTS


def validate(v, errors):
    if v.get('architecture_variant') is not None and v['architecture_variant'] not in VARIANTS:
        errors.append('Unknown architecture_variant')
    for k in v:
        if k.startswith('joint_') and k not in DEFAULTS:
            errors.append('Unknown joint field: '+k)
    if not active(v):
        return
    required = dict(backbone='resnet50', hidden_dim=256, num_feature_levels=5, enc_layers=6,
        dec_layers=6, two_stage_type='standard', matcher_type='HungarianMatcher',
        return_interm_indices=[0, 1, 2, 3], param_dict_type='default',
        allocator_encoder_type='light_dw', density_target_backend='vectorized',
        proposal_selection_mode='spatial', use_ema=False, onecyclelr=False,
        allocator_enabled=True, calibrator_enabled=True)
    for k, expected in required.items():
        if v.get(k) != expected:
            errors.append(f'H1 requires {k}={expected!r}')
    if any(v.get(k) for k in ('precision24_enabled', 'p2_transfer_version', 'dilation', 'masks',
                              'geometry_loss_weight', 'two_stage_keep_all_tokens', 'two_stage_add_query_num',
                              'two_stage_pat_embed')):
        errors.append('H1 cannot combine old precision/transfer/geometry or alternate proposal paths')
    n = v.get('joint_warmup_updates', 500)
    if type(n) is not int or n < 1:
        errors.append('joint_warmup_updates must be positive integer')
    lr = v.get('joint_new_lr', 1e-4)
    if type(lr) not in (float, int) or not math.isfinite(lr) or lr <= 0:
        errors.append('joint_new_lr must be positive finite')
    if v.get('lr_drop_list') != [13, 23] or not v.get('multi_step_lr'):
        errors.append('H1 uses the explicit epoch13/23 schedule')
    if any(type(x) is not int or x < 0 for x in v.get('joint_subset_epochs', [0,3,7,12])):
        errors.append('Invalid joint_subset_epochs')


def signature(args):
    v = args if isinstance(args, dict) else vars(args)
    if not active(v):
        return {}
    return dict(architecture_variant=v['architecture_variant'], joint_revision='v2_1',
        joint_contract='six_full_detail64_r0.1_o2m3_aux0.25_distribution33_dfl0.25_mix0.5',
        joint_warmup_updates=v.get('joint_warmup_updates', 500), joint_new_lr=v.get('joint_new_lr', 1e-4),
        joint_epochs=v.get('epochs'), joint_batch_size=v.get('batch_size'), joint_seed=v.get('seed', 42),
        joint_lr_drop_list=v.get('lr_drop_list'), joint_source_sha=v.get('expected_pretrained_sha256'),
        joint_scales=v.get('data_aug_scales'), joint_max_size=v.get('data_aug_max_size'),
        joint_mosaic=[v.get('mosaic_p'), v.get('mosaic_warmup_end'), v.get('mosaic_decay_start')],
        joint_copy_paste=v.get('copy_paste_p'), joint_lr_projection_mult=v.get('lr_linear_proj_mult', .1))


def configure(model, criterion, args):
    if not active(args):
        return
    model.joint_variant = args.architecture_variant
    criterion.joint_enabled = args.architecture_variant == VARIANTS[0]
    criterion.joint_warmup_updates = getattr(args, 'joint_warmup_updates', 500)
    if not criterion.joint_enabled:
        return
    from models.aqfcdetr.joint_modules import (SemanticDetailBridge, CandidateAuxiliaryHead,
                                              LocalDistributionRefiner)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(getattr(args, 'seed', 42)+1701)
        model.transformer.semantic_detail_bridge = SemanticDetailBridge()
        model.transformer.candidate_auxiliary_head = CandidateAuxiliaryHead(
            model.transformer.enc_out_class_embed, model.transformer.enc_out_bbox_embed)
        model.distribution_refiner = LocalDistributionRefiner()
    for key in ('loss_ce', 'loss_bbox', 'loss_giou', 'loss_nwd'):
        criterion.weight_dict[key+'_joint_o2m'] = criterion.weight_dict[key]
    criterion.weight_dict['loss_joint_dfl'] = 1.


def parameter_groups(args, model):
    names, params = {}, {}
    rates = dict(loaded=args.lr, backbone=args.lr_backbone,
                 projection=args.lr*getattr(args, 'lr_linear_proj_mult', .1),
                 new=getattr(args, 'joint_new_lr', 1e-4))
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        kind = ('new' if name.startswith(NEW_PREFIXES) else 'backbone' if 'backbone' in name else
                'projection' if any(k in name for k in args.lr_linear_proj_names) else 'loaded')
        params.setdefault(kind, []).append(p); names.setdefault(kind, []).append(name)
    return [dict(params=ps, lr=rates[k], joint_peak_lr=rates[k], joint_group=k, joint_names=names[k])
            for k, ps in params.items()]


def update_lr(optimizer, args, epoch, updates):
    factor = (.1+.9*min(1., updates/getattr(args, 'joint_warmup_updates', 500)))
    factor *= .1 ** sum(epoch >= milestone for milestone in args.lr_drop_list)
    for group in optimizer.param_groups:
        group['lr'] = group['joint_peak_lr']*factor


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        t = value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(t.dtype).encode()); digest.update(str(tuple(t.shape)).encode())
        digest.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def initialize(model, args):
    """Verified, frozen legacy migration allowlist; never accepts a different source."""
    from .checkpoint_migration import extract_state_dict, migrate_state_dict, load_legacy_pretrained
    from .experiment import sha256
    if sha256(args.pretrain_model_path) != SOURCE_SHA:
        raise ValueError('H1 requires the approved historical SHA-256')
    source = migrate_state_dict(extract_state_dict(torch.load(args.pretrain_model_path,
                                   map_location='cpu', weights_only=False)))
    current = model.state_dict()
    # Audited against the original paper checkpoint; only the named AQBA changes.
    root = 'transformer.query_allocator.'
    missing_exact = {root+k for k in ('ema_log_boundaries', 'ema_initialized', 'boundary_history',
        'history_ptr', 'training_steps', 'density_head.weight', 'density_head.bias')}
    missing_exact.add('transformer.feature_calibrator.channel_gate.scale_factor')
    for branch, layers in [('boundary_head', (1,2,5)), ('count_regressor', (2,3,6))]:
        missing_exact.update(root+f'{branch}.{i}.{suffix}' for i in layers for suffix in ('weight','bias'))
    missing_exact.update(k for k in current if k.startswith(root+'light_density_encoder.'))
    allowed_missing = missing_exact | {k for k in current if k.startswith(NEW_PREFIXES)}
    allowed_extra = {root+f'density_encoder.{i}.conv.{s}' for i in range(6) for s in ('weight','bias')}
    allowed_extra |= {root+'density_conv1.'+s for s in ('weight','bias')}
    mismatched = [k for k in source.keys() & current.keys() if source[k].shape != current[k].shape]
    if mismatched or set(current)-set(source) != allowed_missing or set(source)-set(current) != allowed_extra:
        raise ValueError('Historical migration differs from frozen allowlist: '+str(dict(
            mismatched=mismatched, missing=sorted(set(current)-set(source)-allowed_missing),
            extra=sorted(set(source)-set(current)-allowed_extra))))
    if any(not torch.isfinite(x).all() for x in source.values() if x.is_floating_point()):
        raise ValueError('Non-finite historical weight')
    report = load_legacy_pretrained(model, args.pretrain_model_path, expected_sha256=SOURCE_SHA)
    if hasattr(model.transformer, 'candidate_auxiliary_head'):
        model.transformer.candidate_auxiliary_head.initialize_from(model.transformer)
    common = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith(NEW_PREFIXES)}
    model.joint_common_sha = tensor_digest(common)
    from .epoch_boundary import atomic_save
    out = Path(args.output_dir)
    atomic_save(dict(model=common, format='legacy_joint_common_initialization', source_sha256=SOURCE_SHA,
                     parameter_sha256=model.joint_common_sha, seed=args.seed), out/'common_initialization.pth')
    report.update(common_parameter_sha256=model.joint_common_sha, allowlist_verified=True)
    (out/'checkpoint_migration_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def checkpoint_state(model, args, optimizer, epoch, updates):
    return dict(signature=signature(args), common_sha256=model.joint_common_sha,
                epoch=epoch, successful_updates=updates,
                parameter_groups=[{k:v for k,v in g.items() if k.startswith('joint_')}
                                  for g in optimizer.param_groups])


def restore(model, checkpoint, args, optimizer=None):
    saved = checkpoint.get('joint_state')
    if not active(args):
        if saved is not None:
            raise ValueError('H1 checkpoint requires H1 architecture')
        return
    if not saved or saved.get('signature') != signature(args):
        raise ValueError('H1 structure/recipe/source signature mismatch')
    if saved.get('epoch') != checkpoint.get('epoch') or saved.get('successful_updates') != checkpoint.get('criterion_progress',{}).get('successful_updates'):
        raise ValueError('H1 epoch/update counters mismatch')
    if optimizer is not None:
        if checkpoint.get('rng_states') is None or not checkpoint.get('joint_data_rng'):
            raise ValueError('H1 resume requires model and loader RNG states')
        current = [{k:v for k,v in g.items() if k.startswith('joint_')} for g in optimizer.param_groups]
        if current != saved.get('parameter_groups'):
            raise ValueError('H1 optimizer groups differ')
        args['_joint_data_rng'] = checkpoint['joint_data_rng']
    model.joint_common_sha = saved['common_sha256']


def extra_losses(criterion, outputs, targets, indices, num_boxes):
    from models.aqfcdetr.joint_modules import auxiliary_matches, distribution_loss
    ramp = min(1., criterion.quality_successful_updates/criterion.joint_warmup_updates)
    losses = {}
    if 'refine_distribution_logits' in outputs:
        value, diagnostics = distribution_loss(outputs, targets, indices, num_boxes)
        losses['loss_joint_dfl'] = .25*ramp*value
        losses.update(diagnostics)
    if 'auxiliary_o2m_outputs' in outputs:
        auxiliary = outputs['auxiliary_o2m_outputs']
        pairs = auxiliary_matches(auxiliary, targets, criterion.matcher)
        total = sum(len(i) for i,j in pairs)
        values = criterion.loss_labels(auxiliary, targets, pairs, max(total,1), log=False, quality_eligible=False)
        values.update(criterion.loss_boxes(auxiliary, targets, pairs, max(total,1)))
        losses.update({k+'_joint_o2m': .25*ramp*v for k,v in values.items()
                       if k in ('loss_ce','loss_bbox','loss_giou','loss_nwd')})
        losses['joint_aux_positives'] = outputs['pred_boxes'].new_tensor(float(total))
    losses['joint_refinement_mix'] = outputs['pred_boxes'].new_tensor(.5*ramp)
    return losses
