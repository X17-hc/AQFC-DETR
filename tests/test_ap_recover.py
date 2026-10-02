"""Decoder-head recovery: weight 2, verytiny unchanged, encoder heads frozen."""
import argparse
import sys
import types
from pathlib import Path

import torch
from torch import nn

if 'MultiScaleDeformableAttention' not in sys.modules:
    sys.modules['MultiScaleDeformableAttention'] = types.ModuleType(
        'MultiScaleDeformableAttention')

from models.aqfcdetr.matcher import HungarianMatcher
from models.aqfcdetr.model import SetCriterion
from util.config_validation import validate_config
from util.slconfig import SLConfig
from util import legacy_joint as joint


def _criterion(tighten, weight=2.0):
    criterion = SetCriterion(
        8, HungarianMatcher(), {}, 0.25, ['boxes'], aligned_box_loss=True)
    criterion.box_tighten = tighten
    criterion.box_tighten_weight = weight
    return criterion


def _case(pred, target, orig_hw):
    pred = pred.clone().detach().requires_grad_(True)
    outputs = dict(pred_boxes=pred.view(1, 1, 4))
    targets = [dict(boxes=target.view(1, 4), orig_size=orig_hw)]
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    return outputs, targets, indices, pred


def _losses(tighten, pred, target, orig_hw, weight=2.0):
    outputs, targets, indices, pred = _case(pred, target, orig_hw)
    losses = _criterion(tighten, weight).loss_boxes(outputs, targets, indices, 1)
    return losses, pred


def test_weight_two_leaves_verytiny_loss_unchanged():
    orig = torch.tensor([100., 100.])
    pred = torch.tensor([0.512, 0.5, 0.04, 0.04])
    target = torch.tensor([0.5, 0.5, 0.04, 0.04])
    off, _ = _losses(False, pred, target, orig)
    on, _ = _losses(True, pred, target, orig)
    for key in ('loss_bbox', 'loss_giou', 'loss_nwd', 'loss_xy', 'loss_hw'):
        torch.testing.assert_close(on[key], off[key])


def test_weight_two_scales_only_wh_and_giou():
    target = torch.tensor([0.4, 0.4, 0.4, 0.4])
    pred = torch.tensor([0.492, 0.4, 0.4, 0.4])
    orig = torch.tensor([100., 100.])
    off, pred_off = _losses(False, pred, target, orig)
    on, pred_on = _losses(True, pred, target, orig)
    torch.testing.assert_close(on['loss_giou'], off['loss_giou'] * 2)
    torch.testing.assert_close(on['loss_xy'], off['loss_xy'])
    torch.testing.assert_close(on['loss_hw'], off['loss_hw'] * 2)
    torch.testing.assert_close(on['loss_nwd'], off['loss_nwd'])
    off['loss_bbox'].backward()
    on['loss_bbox'].backward()
    torch.testing.assert_close(pred_on.grad[:2], pred_off.grad[:2])
    torch.testing.assert_close(pred_on.grad[2:], pred_off.grad[2:] * 2)


def test_decoder_class_head_excludes_encoder_class_embed():
    assert joint._decoder_class_head('transformer.decoder.class_embed.0.weight')
    assert joint._decoder_class_head('class_embed.0.weight')
    assert not joint._decoder_class_head('transformer.enc_out_class_embed.layers.0.weight')
    assert not joint._decoder_class_head('transformer.enc_out_bbox_embed.layers.0.weight')

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.bbox_embed = nn.ModuleList([nn.Linear(2, 4)])
            self.class_embed = nn.ModuleList([nn.Linear(2, 8)])
            self.transformer = nn.Module()
            self.transformer.enc_out_class_embed = nn.Linear(2, 8)
            self.transformer.enc_out_bbox_embed = nn.Linear(2, 4)
            self.backbone = nn.Linear(2, 2)

    model = Tiny()
    args = argparse.Namespace(
        decoder_heads_only=True, box_head_only=False, lr=1e-5, lr_backbone=1e-5,
        lr_linear_proj_mult=0.1, lr_linear_proj_names=[], joint_new_lr=1e-5)
    groups = joint.parameter_groups(args, model)
    names = [name for group in groups for name in group['joint_names']]
    assert 'class_embed.0.weight' in names
    assert 'bbox_embed.0.weight' in names
    assert 'transformer.enc_out_class_embed.weight' not in names
    assert 'transformer.enc_out_bbox_embed.weight' not in names
    assert 'backbone.weight' not in names
    assert model.class_embed[0].weight.requires_grad
    assert model.bbox_embed[0].weight.requires_grad
    assert not model.transformer.enc_out_class_embed.weight.requires_grad
    assert not model.backbone.weight.requires_grad
    assert groups


def test_recover_configs_validate():
    root = Path('configs/legacy_joint')
    control = SLConfig.fromfile(str(root / 'h1_ap_recover_control_1e.py'))._cfg_dict.to_dict()
    treat = SLConfig.fromfile(str(root / 'h1_ap_recover_treat_1e.py'))._cfg_dict.to_dict()
    full = SLConfig.fromfile(str(root / 'h1_ap_recover_24e.py'))._cfg_dict.to_dict()
    for cfg in (control, treat, full):
        validate_config(cfg)
        assert cfg['decoder_heads_only'] and not cfg['box_head_only']
        assert cfg['native_weight_finetune']
        assert cfg['lr'] == 1e-5 and cfg['lr_drop_list'] == []
        assert cfg['joint_warmup_updates'] == 1 and cfg['joint_subset_epochs'] == []
        assert cfg['val_epoch'] == [] and not cfg['multi_step_lr']
        assert cfg['expected_native_sha256'] == (
            '222b7a0dd797d25c2105084bc6fa4fb8374ef01967acfff0d3d1ce0eca2d0313')
        assert cfg['calibrator_density_spatial'] and cfg['allocator_quantile_boundaries']
    assert not control['box_tighten']
    assert treat['box_tighten'] and treat['box_tighten_weight'] == 2.0
    assert full['box_tighten'] and full['box_tighten_weight'] == 2.0
    assert control['epochs'] == treat['epochs'] == 1
    assert full['epochs'] == 24
