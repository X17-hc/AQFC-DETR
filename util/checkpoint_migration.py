"""Legacy checkpoint migration for AQFC-DETR.

Legacy names are intentionally isolated in this module so the active model API
does not depend on them.
"""
from collections import OrderedDict
import json
from pathlib import Path
import re

import torch


LEGACY_KEY_REPLACEMENTS = (
    ('transformer.CCM.conv1.', 'transformer.query_allocator.density_conv1.'),
    ('transformer.CCM.linear.', 'transformer.query_allocator.budget_classifier.'),
    ('transformer.CGFE.SpatialGate.spatial.conv.',
     'transformer.feature_calibrator.spatial_attention.conv.conv.'),
    ('transformer.CGFE.SpatialGate.spatial.bn.',
     'transformer.feature_calibrator.spatial_attention.conv.bn.'),
    ('transformer.CCM.', 'transformer.query_allocator.'),
    ('transformer.CGFE.', 'transformer.feature_calibrator.'),
    ('transformer.multiscale.', 'transformer.density_pyramid.'),
    ('ccm_backbone.', 'density_encoder.'),
    ('ccm_pool.', 'budget_pool.'),
    ('ccm_classifier.', 'budget_classifier.'),
    ('ref_point_conv.', 'density_head.'),
    ('ChannelGate.', 'channel_gate.'),
    ('SpatialAttn.', 'spatial_attention.'),
)


def extract_state_dict(checkpoint, prefer_ema=False):
    if not isinstance(checkpoint, dict):
        raise TypeError('Checkpoint must be a dictionary')
    if prefer_ema and isinstance(checkpoint.get('ema_model'), dict):
        return checkpoint['ema_model']
    if isinstance(checkpoint.get('model'), dict):
        return checkpoint['model']
    if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint
    raise KeyError('Checkpoint contains neither model nor ema_model state')


def strip_parallel_prefix(key):
    while key.startswith('module.'):
        key = key[len('module.'):]
    return key


def is_legacy_state_dict(state_dict):
    return any('transformer.CCM.' in key or 'transformer.CGFE.' in key
               for key in state_dict)


def migrate_key(key):
    migrated = strip_parallel_prefix(key)
    legacy_density_layer = re.match(r'transformer\.CCM\.ccm\.(\d+)\.(weight|bias)$', migrated)
    if legacy_density_layer:
        old_index = int(legacy_density_layer.group(1))
        new_index = old_index // 2
        parameter = legacy_density_layer.group(2)
        migrated = (f'transformer.query_allocator.density_encoder.{new_index}.'
                    f'conv.{parameter}')
    for old, new in LEGACY_KEY_REPLACEMENTS:
        migrated = migrated.replace(old, new)
    return migrated


def migrate_state_dict(state_dict):
    return OrderedDict((migrate_key(key), value) for key, value in state_dict.items())


def load_legacy_pretrained(model, checkpoint_path, report_path=None, prefer_ema=False):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    source = extract_state_dict(checkpoint, prefer_ema=prefer_ema)
    migrated = migrate_state_dict(source)
    current = model.state_dict()
    compatible = OrderedDict()
    unexpected = []
    mismatched = []
    for key, value in migrated.items():
        if key not in current:
            unexpected.append(key)
        elif current[key].shape != value.shape:
            mismatched.append({
                'key': key,
                'checkpoint_shape': list(value.shape),
                'model_shape': list(current[key].shape),
            })
        else:
            compatible[key] = value
    load_result = model.load_state_dict(compatible, strict=False)
    loaded_numel = sum(current[key].numel() for key in compatible)
    total_numel = sum(value.numel() for value in current.values())
    module_coverage = {}
    for key, value in current.items():
        module = '.'.join(key.split('.')[:2]) if key.startswith('transformer.') else key.split('.')[0]
        counts = module_coverage.setdefault(module, {'loaded_numel': 0, 'total_numel': 0})
        counts['total_numel'] += value.numel()
        if key in compatible:
            counts['loaded_numel'] += value.numel()
    for counts in module_coverage.values():
        counts['coverage'] = counts['loaded_numel'] / max(counts['total_numel'], 1)
    report = {
        'parameter_only_loaded_numel': sum(p.numel() for k, p in model.named_parameters() if k in compatible),
        'parameter_only_total_numel': sum(p.numel() for p in model.parameters()),
        'source_checkpoint': str(checkpoint_path.resolve()),
        'source_format': 'legacy_dqdetr' if is_legacy_state_dict(source) else 'compatible_pretrained',
        'loaded_keys': list(compatible),
        'missing_keys': list(load_result.missing_keys),
        'unexpected_keys': unexpected + list(load_result.unexpected_keys),
        'shape_mismatched_keys': mismatched,
        'loaded_parameter_count': loaded_numel,
        'total_parameter_count': total_numel,
        'coverage_by_numel': loaded_numel / max(total_numel, 1),
        'coverage_definition': 'state_dict tensor elements, including buffers; not functional equivalence',
        'coverage_by_module': module_coverage,
    }
    report['parameter_only_coverage'] = (report['parameter_only_loaded_numel'] /
                                       max(report['parameter_only_total_numel'], 1))
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report


def convert_checkpoint(input_path, output_path, prefer_ema=False):
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
    state = migrate_state_dict(extract_state_dict(checkpoint, prefer_ema=prefer_ema))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': state, 'format': 'aqfcdetr_warmstart'}, output_path)
    return output_path
