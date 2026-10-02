"""Distribution-refiner edge loss: IoU below 0.60, frozen detection set."""
import argparse
from pathlib import Path

import torch
from torch import nn

from util.config_validation import validate_config
from util.edge_logwh import EdgeScreen, low_iou_logwh_loss
from util.slconfig import SLConfig
from util import legacy_joint as joint


def _pair(pred, target):
    pred = pred.clone().detach().requires_grad_(True)
    outputs = dict(pred_boxes=pred.view(1, 1, 4))
    targets = [dict(boxes=target.view(1, 4))]
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    return outputs, targets, indices, pred


def test_low_iou_logwh_moves_only_width_and_height():
    outputs, targets, indices, pred = _pair(
        torch.tensor([0.5, 0.5, 0.30, 0.30]),
        torch.tensor([0.5, 0.5, 0.20, 0.20]))
    loss = low_iou_logwh_loss(outputs, targets, indices, 1)
    assert float(loss.detach()) > 0
    loss.backward()
    assert float(pred.grad[:2].abs().sum()) == 0
    assert float(pred.grad[2:].abs().sum()) > 0


def test_iou_at_least_060_contributes_nothing():
    outputs, targets, indices, pred = _pair(
        torch.tensor([0.5, 0.5, 0.210, 0.210]),
        torch.tensor([0.5, 0.5, 0.200, 0.200]))
    loss = low_iou_logwh_loss(outputs, targets, indices, 1)
    assert float(loss.detach()) == 0
    loss.backward()
    assert float(pred.grad.abs().sum()) == 0


def test_high_iou_box_stays_zero_beside_a_low_iou_box():
    pred = torch.tensor([
        [0.5, 0.5, 0.30, 0.30],
        [0.5, 0.5, 0.21, 0.21],
    ], requires_grad=True)
    target = torch.tensor([
        [0.5, 0.5, 0.20, 0.20],
        [0.5, 0.5, 0.20, 0.20],
    ])
    outputs = dict(pred_boxes=pred.view(1, 2, 4))
    targets = [dict(boxes=target)]
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    low_iou_logwh_loss(outputs, targets, indices, 2).backward()
    assert float(pred.grad[0, 2:].abs().sum()) > 0
    assert float(pred.grad[1].abs().sum()) == 0


def test_edge_flag_off_does_not_add_the_loss():
    class Criterion:
        edge_logwh = False
        quality_successful_updates = 1
        joint_warmup_updates = 1

    outputs = dict(pred_boxes=torch.zeros(1, 1, 4))
    targets = [dict(boxes=torch.zeros(1, 4))]
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    losses = joint.extra_losses(Criterion(), outputs, targets, indices, 1)
    assert 'loss_edge_logwh' not in losses


def test_only_the_distribution_refiner_trains():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.distribution_refiner = nn.Linear(4, 4)
            self.bbox_embed = nn.ModuleList([nn.Linear(2, 4)])
            self.class_embed = nn.ModuleList([nn.Linear(2, 8)])
            self.backbone = nn.Linear(2, 2)
            self.query_allocator = nn.Linear(2, 2)
            self.transformer = nn.Module()
            self.transformer.enc_out_class_embed = nn.Linear(2, 8)
            self.transformer.enc_out_bbox_embed = nn.Linear(2, 4)

    model = Tiny()
    args = argparse.Namespace(
        edge_refine_only=True, decoder_heads_only=False, box_head_only=False,
        lr=1e-5, lr_backbone=1e-5, lr_linear_proj_mult=0.1,
        lr_linear_proj_names=[], joint_new_lr=1e-5)
    groups = joint.parameter_groups(args, model)
    names = [name for group in groups for name in group['joint_names']]
    assert names == ['distribution_refiner.weight', 'distribution_refiner.bias']
    assert groups[0]['lr'] == 1e-5
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad is name.startswith('distribution_refiner.')


def test_edge_screen_counts_best_iou_below_060():
    screen = EdgeScreen()
    det_boxes = torch.tensor([[0., 0., 10., 10.], [20., 20., 40., 40.]])
    det_scores = torch.tensor([0.9, 0.2])
    det_labels = torch.tensor([1, 1])
    gt_boxes = torch.tensor([[0., 0., 10., 10.], [0., 0., 20., 20.]])
    gt_labels = torch.tensor([1, 1])
    screen.add(det_boxes, det_scores, det_labels, gt_boxes, gt_labels)
    summary = screen.summary()
    assert summary['n_gt'] == 2
    assert summary['n_best_iou_below_060'] == 1
    assert summary['fraction_below_060'] == 0.5
    assert summary['wh_rel_median_below_060'] == 0.5


def test_edge_configs_validate():
    root = Path('configs/legacy_joint')
    control = SLConfig.fromfile(str(root / 'h1_ap_edge_control_1e.py'))._cfg_dict.to_dict()
    treat = SLConfig.fromfile(str(root / 'h1_ap_edge_treat_1e.py'))._cfg_dict.to_dict()
    full = SLConfig.fromfile(str(root / 'h1_ap_edge_24e.py'))._cfg_dict.to_dict()
    for cfg in (control, treat, full):
        validate_config(cfg)
        assert cfg['edge_refine_only'] and not cfg['decoder_heads_only']
        assert not cfg['box_head_only'] and not cfg['box_tighten']
        assert cfg['native_weight_finetune']
        assert cfg['lr'] == 1e-5 and cfg['joint_new_lr'] == 1e-5
        assert cfg['lr_drop_list'] == [] and not cfg['multi_step_lr']
        assert cfg['joint_warmup_updates'] == 1 and cfg['joint_subset_epochs'] == []
        assert cfg['val_epoch'] == []
        assert cfg['expected_native_sha256'] == (
            '222b7a0dd797d25c2105084bc6fa4fb8374ef01967acfff0d3d1ce0eca2d0313')
    assert control['edge_logwh'] is False
    assert treat['edge_logwh'] is True
    assert full['edge_logwh'] is True and full['epochs'] == 24
    assert control['epochs'] == 1 and treat['epochs'] == 1
