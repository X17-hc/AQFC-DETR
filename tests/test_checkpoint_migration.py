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


def test_native_resume_restores_optimizer_scheduler_ema_and_scaler(tmp_path):
    import copy
    from types import SimpleNamespace
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = torch.amp.GradScaler('cpu', init_scale=128.)
    scaler.scale(model(torch.ones(1,2)).sum()).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    expected = copy.deepcopy(model.state_dict())
    ema = SimpleNamespace(module=copy.deepcopy(model))
    path = tmp_path / 'native.pth'
    torch.save(dict(model=expected, optimizer=optimizer.state_dict(),
                    lr_scheduler=scheduler.state_dict(), ema_model=ema.module.state_dict(),
                    scaler=scaler.state_dict(), epoch=2,
                    best_metrics={'regular_AP': .42, 'regular_epoch': 1}), path)
    restored = torch.nn.Linear(2,1)
    new_optimizer = torch.optim.AdamW(restored.parameters(), lr=1.)
    new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=1)
    new_scaler = torch.amp.GradScaler('cpu')
    best_metrics = {}
    # Older native checkpoints remain loadable, but lack reproducible RNG state.
    with pytest.warns(RuntimeWarning, match='RNG state'):
        next_epoch = load_native_resume(restored, path, new_optimizer, new_scheduler, ema, new_scaler,
                                        best_metrics=best_metrics)
    assert best_metrics == {'regular_AP': .42, 'regular_epoch': 1}
    assert next_epoch == 3
    assert new_scaler.state_dict() == scaler.state_dict()
    assert new_scheduler.state_dict() == scheduler.state_dict()
    assert new_optimizer.param_groups[0]['lr'] == optimizer.param_groups[0]['lr']
    assert all(int(s['step']) == 1 for s in new_optimizer.state.values())
    for key in expected:
        torch.testing.assert_close(restored.state_dict()[key], expected[key], rtol=0, atol=0)
        torch.testing.assert_close(ema.module.state_dict()[key], expected[key], rtol=0, atol=0)
