"""Near-miss box reweight: verytiny and already-tight matches stay at weight 1."""
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


def _criterion(tighten):
    criterion = SetCriterion(
        8, HungarianMatcher(), {}, 0.25, ['boxes'], aligned_box_loss=True)
    criterion.box_tighten = tighten
    return criterion


def _case(pred, target, orig_hw):
    pred = pred.clone().detach().requires_grad_(True)
    outputs = dict(pred_boxes=pred.view(1, 1, 4))
    targets = [dict(boxes=target.view(1, 4), orig_size=orig_hw)]
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    return outputs, targets, indices, pred


def _losses(tighten, pred, target, orig_hw):
    outputs, targets, indices, pred = _case(pred, target, orig_hw)
    losses = _criterion(tighten).loss_boxes(outputs, targets, indices, 1)
    return losses, pred


def test_verytiny_and_tight_boxes_keep_the_original_loss():
    orig = torch.tensor([100., 100.])
    cases = [
        (torch.tensor([0.512, 0.5, 0.04, 0.04]), torch.tensor([0.5, 0.5, 0.04, 0.04])),
        (torch.tensor([0.5, 0.5, 0.04, 0.04]), torch.tensor([0.5, 0.5, 0.04, 0.04])),
        (torch.tensor([0.4, 0.4, 0.4, 0.4]), torch.tensor([0.4, 0.4, 0.4, 0.4])),
    ]
    for pred, target in cases:
        off, _ = _losses(False, pred, target, orig)
        on, _ = _losses(True, pred, target, orig)
        for key in ('loss_bbox', 'loss_giou', 'loss_nwd'):
            torch.testing.assert_close(on[key], off[key])


def test_near_miss_large_box_scales_giou_and_wh_only():
    target = torch.tensor([0.4, 0.4, 0.4, 0.4])
    pred = torch.tensor([0.492, 0.4, 0.4, 0.4])
    orig = torch.tensor([100., 100.])
    off, pred_off = _losses(False, pred, target, orig)
    on, pred_on = _losses(True, pred, target, orig)
    torch.testing.assert_close(on['loss_giou'], off['loss_giou'] * 3)
    torch.testing.assert_close(on['loss_xy'], off['loss_xy'])
    torch.testing.assert_close(on['loss_hw'], off['loss_hw'] * 3)
    off['loss_bbox'].backward()
    on['loss_bbox'].backward()
    torch.testing.assert_close(pred_on.grad[:2], pred_off.grad[:2])
    torch.testing.assert_close(pred_on.grad[2:], pred_off.grad[2:] * 3)


def test_decoder_box_head_name_skips_the_encoder_box_embed():
    assert joint._decoder_box_head('transformer.decoder.bbox_embed.0.layers.0.weight')
    assert joint._decoder_box_head('bbox_embed.0.weight')
    assert not joint._decoder_box_head('transformer.enc_out_bbox_embed.layers.0.weight')
    assert not joint._decoder_box_head('backbone.0.weight')
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.bbox_embed = nn.ModuleList([nn.Linear(2, 4)])
            self.backbone = nn.Linear(2, 2)

    model = Tiny()
    args = argparse.Namespace(
        box_head_only=True, lr=1e-5, lr_backbone=1e-5,
        lr_linear_proj_mult=0.1, lr_linear_proj_names=[], joint_new_lr=1e-5)
    groups = joint.parameter_groups(args, model)
    names = [name for group in groups for name in group['joint_names']]
    assert names == ['bbox_embed.0.weight', 'bbox_embed.0.bias']
    assert model.bbox_embed[0].weight.requires_grad
    assert not model.backbone.weight.requires_grad


def test_box_finetune_configs_validate():
    root = Path('configs/legacy_joint')
    control = SLConfig.fromfile(str(root / 'h1_ap_box_control_3e.py'))._cfg_dict.to_dict()
    tighten = SLConfig.fromfile(str(root / 'h1_ap_box_tighten_3e.py'))._cfg_dict.to_dict()
    validate_config(control)
    validate_config(tighten)
    assert control['box_head_only'] and not control['box_tighten']
    assert tighten['box_tighten'] and tighten['box_head_only']
    assert control['epochs'] == tighten['epochs'] == 3
    assert control['lr'] == tighten['lr'] == 1e-5
    assert control['lr_drop_list'] == [] and control['joint_subset_epochs'] == [2]
    assert control['calibrator_density_spatial'] and control['allocator_quantile_boundaries']
    assert not control.get('joint_finetune_from_h1')
