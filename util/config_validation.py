import math

LEGACY_FIELDS = {
    'dynamic_query_list': 'query_budget_levels',
    'ccm_cls_num': 'derived from len(query_budget_levels)',
    'ccm_loss_coef': 'allocator_loss_weight',
    'coverage_loss_coef': 'coverage_loss_weight',
    'spacing_loss_coef': 'spacing_loss_weight',
    'interval_loss_coef': 'interval_loss_weight',
    'cgfe_no_spatial': 'calibrator_use_spatial',
    'gate_type': 'calibrator_gate_type',
}


def validate_config(config):
    errors = []
    for old, new in LEGACY_FIELDS.items():
        if old in config:
            errors.append(f"Legacy field '{old}' is not supported; use '{new}'")
    levels = list(config.get('query_budget_levels', []))
    if (len(levels) != 4 or any(type(value) is not int or value <= 0 for value in levels)
            or levels != sorted(set(levels))):
        errors.append('query_budget_levels must contain four strictly increasing integers')
    if levels and max(levels) > int(config.get('max_query_budget', 1500)):
        errors.append('maximum query budget exceeds max_query_budget')
    fallback = int(config.get('allocator_fallback_queries', 900))
    if levels and fallback not in levels:
        errors.append('allocator_fallback_queries must be one of query_budget_levels')
    alphas = config.get('calibrator_spatial_alphas', [])
    feature_levels = int(config.get('num_feature_levels', len(alphas)))
    if len(alphas) != feature_levels:
        errors.append('calibrator_spatial_alphas length must equal num_feature_levels')
    if any(not math.isfinite(float(alpha)) or float(alpha) < 0 for alpha in alphas):
        errors.append('calibrator_spatial_alphas must be finite and non-negative')
    forced = config.get('force_query_budget')
    if forced is not None and (type(forced) is not int or not levels or
                               not min(levels) <= forced <= max(levels)):
        errors.append('force_query_budget must be an integer within query_budget_levels bounds')
    # Inherited transformer options do not implement the dynamic-query mask
    # contract. Reject them before model construction, rather than crash later.
    if config.get('two_stage_type', 'standard') != 'standard':
        errors.append('AQFC-DETR requires two_stage_type=standard')
    for key in ('two_stage_pat_embed', 'two_stage_add_query_num'):
        if config.get(key, 0) != 0:
            errors.append(f'{key} must be zero with dynamic query selection')
    if config.get('two_stage_keep_all_tokens', False):
        errors.append('two_stage_keep_all_tokens is incompatible with query masks')
    if config.get('dec_layer_number') is not None:
        errors.append('dec_layer_number must be None with dynamic query masks')
    if config.get('embed_init_tgt', False) and levels and config.get('num_queries', 900) < max(levels):
        errors.append('embed_init_tgt requires num_queries >= the maximum query budget')
    if config.get('calibrator_gate_type', 'tanh') not in {'tanh', 'sigmoid', 'se', 'swish', 'glu'}:
        errors.append('calibrator_gate_type is not a supported gate')
    mosaic_p = float(config.get('mosaic_p', 0))
    copy_paste_p = float(config.get('copy_paste_p', 0))
    if not (0 <= mosaic_p <= 1 and 0 <= copy_paste_p <= 1 and mosaic_p + copy_paste_p <= 1):
        errors.append('mosaic_p and copy_paste_p must be probabilities with sum <= 1')
    if config.get('masks', False) and (mosaic_p or copy_paste_p):
        errors.append('Mosaic/Copy-Paste support detection boxes only; disable them for masks')
    if config.get('proposal_selection_mode') not in {'semantic', 'fused', 'mixed'}:
        errors.append("proposal_selection_mode must be 'semantic', 'fused', or 'mixed'")
    ratio = float(config.get('mixed_density_ratio', 0.25))
    if not 0.0 <= ratio <= 1.0:
        errors.append('mixed_density_ratio must be in [0, 1]')
    weight_fields = [key for key in config if key.endswith('_loss_weight')]
    for key in weight_fields:
        if not math.isfinite(float(config[key])) or float(config[key]) < 0:
            errors.append(f'{key} must be non-negative')
    for key, value in [('proposal_density_weight', config.get('proposal_density_weight', .25)),
                       ('amp_init_scale', config.get('amp_init_scale', 128.))]:
        if not math.isfinite(float(value)) or float(value) < 0 or (key == 'amp_init_scale' and value == 0):
            errors.append(f'{key} must be finite and positive (density weight may be zero)')
    if config.get('allocator_teacher_epochs', 6) != 6:
        errors.append('allocator_teacher_epochs currently supports the fixed six-epoch schedule only')
    if config.get('train_split', 'trainval') not in ('train', 'trainval'):
        errors.append('train_split must be train or trainval')
    if config.get('eval_split', 'test') not in ('val', 'test', 'eval_debug'):
        errors.append('eval_split must be val, test or eval_debug')
    if config.get('train_split', 'trainval') == 'trainval' and config.get('eval_split') in ('val', 'eval_debug'):
        errors.append('trainval includes validation data; select train_split=train for validation selection')
    if 'allocator_use_boundary_ema' in config and type(config['allocator_use_boundary_ema']) is not bool:
        errors.append('allocator_use_boundary_ema must be boolean')
    if errors:
        raise ValueError('Invalid AQFC-DETR configuration:\n- ' + '\n- '.join(errors))
    return True
