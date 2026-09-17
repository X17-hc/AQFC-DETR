from pathlib import Path
from types import SimpleNamespace
import hashlib
import pytest
import torch
from util.incremental_checkpoint import validate_warmstart
from util.checkpoint_migration import load_legacy_pretrained


def test_frozen_epoch10_rejects_different_checkpoint(tmp_path):
    source = tmp_path / 'legacy.pth'
    torch.save({'model': torch.nn.Linear(1, 1).state_dict()}, source)
    with pytest.raises(ValueError, match='SHA-256'):
        validate_warmstart(torch.nn.Linear(1, 1), SimpleNamespace(
            pretrain_model_path=str(source), expected_pretrained_sha256='0' * 64))


def test_partial_migration_still_checks_identity_before_model_mutation(tmp_path):
    model = torch.nn.Linear(1, 1)
    source = tmp_path / 'legacy.pth'
    torch.save({'model': {'weight': torch.full_like(model.weight, 7), 'unused': torch.ones(1)}}, source)
    before = model.weight.detach().clone()
    with pytest.raises(ValueError, match='SHA-256'):
        load_legacy_pretrained(model, source, expected_sha256='0' * 64)
    assert torch.equal(model.weight, before)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    report = load_legacy_pretrained(model, source, expected_sha256=digest)
    assert model.weight.item() == 7
    assert report['missing_keys'] == ['bias']
    assert report['unexpected_keys'] == ['unused']
    assert report['source_sha256'] == digest


def test_legacy_curriculum_and_original_strict_config_stay_separate():
    import util
    from util.slconfig import SLConfig
    from util.config_validation import validate_config
    root = Path(util.__file__).resolve().parents[1]
    legacy = SLConfig.fromfile(str(root / 'configs/incremental_v2/p2_24e_legacy.py'))._cfg_dict.to_dict()
    validate_config(legacy)
    assert legacy['training_phase_epoch_offset'] == 0
    assert legacy['training_phase_total_epochs'] == legacy['epochs'] == 24
    assert legacy['val_epoch'] == [23] and legacy['save_checkpoint_interval'] == 1
    assert legacy['classification_loss_type'] == 'quality_blend'
    assert legacy['allocator_encoder_type'] == 'light_dw'
    assert legacy['lr'] == 1e-4 and legacy['lr_drop_list'] == [13, 23]
    assert legacy['strict_warmstart'] is False and legacy['expected_pretrained_epoch'] is None
    native = SLConfig.fromfile(str(root / 'configs/incremental_v2/p2_24e.py'))
    assert native.strict_warmstart and native.expected_pretrained_epoch == 10
    assert native.training_phase_epoch_offset == 11
