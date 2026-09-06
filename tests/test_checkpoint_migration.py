from collections import OrderedDict

import pytest
import torch

from util.checkpoint import load_native_resume
from util.checkpoint_migration import migrate_state_dict


def test_legacy_key_mapping_and_parallel_prefix():
    source = OrderedDict({
        'module.transformer.CCM.ccm_backbone.0.weight': torch.ones(1),
        'module.transformer.CGFE.ChannelGate.weight': torch.ones(1),
        'transformer.multiscale.ref_point_conv.weight': torch.ones(1),
    })
    migrated = migrate_state_dict(source)
    assert 'transformer.query_allocator.density_encoder.0.weight' in migrated
    assert 'transformer.feature_calibrator.channel_gate.weight' in migrated
    assert 'transformer.density_pyramid.density_head.weight' in migrated


def test_legacy_checkpoint_is_rejected_by_resume(tmp_path):
    checkpoint = tmp_path / 'legacy.pth'
    torch.save({'model': {'transformer.CCM.x': torch.ones(1)}}, checkpoint)
    with pytest.raises(ValueError, match='pretrained'):
        load_native_resume(torch.nn.Linear(1, 1), checkpoint)
