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
    if len(levels) != 4 or levels != sorted(set(levels)):
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
    if config.get('proposal_selection_mode') not in {'semantic', 'fused', 'mixed'}:
        errors.append("proposal_selection_mode must be 'semantic', 'fused', or 'mixed'")
    ratio = float(config.get('mixed_density_ratio', 0.25))
    if not 0.0 <= ratio <= 1.0:
        errors.append('mixed_density_ratio must be in [0, 1]')
    weight_fields = [key for key in config if key.endswith('_loss_weight')]
    for key in weight_fields:
        if float(config[key]) < 0:
            errors.append(f'{key} must be non-negative')
    if errors:
        raise ValueError('Invalid AQFC-DETR configuration:\n- ' + '\n- '.join(errors))
    return True
