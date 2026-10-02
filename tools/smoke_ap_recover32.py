"""32-image mechanism check for the decoder-head recovery recipe.

Writes the first 32 locked test ids, then checks three facts on the real
checkpoint: verytiny matched box loss is unchanged, area>=64 and IoU in
[0.5, 0.75) has a width/height gradient of 2, and the decoder class head
receives a gradient while enc_out_class_embed does not.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import build_dataset
from main import build_model_main, get_args_parser, resolve_launch_defaults
from util.config_validation import validate_config
from util.factor_diagnostics import restrict_eval_dataset
from util.get_param_dicts import get_param_dict
from util.misc import collate_fn
from util.slconfig import SLConfig


def _locked_ids(path):
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    ids = raw if isinstance(raw, list) else raw.get('image_ids', raw.get('ids'))
    if not isinstance(ids, list) or len(ids) < 32:
        raise ValueError('Locked subset must contain at least 32 image ids')
    chosen = [int(value) for value in ids[:32]]
    if len(set(chosen)) != 32:
        raise ValueError('First 32 locked ids are not unique')
    return chosen


def _load_args(config, data_root, pretrained, output_dir, device):
    parser = argparse.ArgumentParser('recover smoke', parents=[get_args_parser()])
    args = parser.parse_args([
        '--config', config,
        '--data-root', data_root,
        '--pretrained', pretrained,
        '--output-dir', output_dir,
        '--device', device,
        '--num_workers', '0',
        '--no-amp',
    ])
    args = resolve_launch_defaults(args)
    cfg = SLConfig.fromfile(args.config_file)
    cfg_dict = cfg._cfg_dict.to_dict()
    validate_config(cfg_dict)
    for key, value in cfg_dict.items():
        if key not in vars(args) or vars(args)[key] is None:
            setattr(args, key, value)
    return args


def _move_targets(targets, device):
    moved = []
    for target in targets:
        item = {}
        for key, value in target.items():
            item[key] = value.to(device) if torch.is_tensor(value) else value
        moved.append(item)
    return moved


def _match_rows(indices):
    rows = []
    for batch_index, (src_idx, tgt_idx) in enumerate(indices):
        for src, tgt in zip(src_idx.tolist(), tgt_idx.tolist()):
            rows.append((batch_index, int(src), int(tgt)))
    return rows


def _areas(targets, rows):
    values = []
    for batch_index, _, tgt in rows:
        box = targets[batch_index]['boxes'][tgt].detach().float()
        size = targets[batch_index]['orig_size'].detach().float().flatten()
        values.append(float(box[2] * size[1] * box[3] * size[0]))
    return values


def _isolated(criterion, pred, target_box, orig_size, tighten):
    pred = pred.detach().float().cpu().clone().requires_grad_(True)
    box = target_box.detach().float().cpu()
    size = orig_size.detach().float().cpu()
    outputs = dict(pred_boxes=pred.view(1, 1, 4))
    targets = [dict(boxes=box.view(1, 4), orig_size=size)]
    indices = [(torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long))]
    saved = criterion.box_tighten
    criterion.box_tighten = tighten
    try:
        losses = criterion.loss_boxes(outputs, targets, indices, 1)
    finally:
        criterion.box_tighten = saved
    return losses, pred


def _check_pair(criterion, pred, target_box, orig_size, selected):
    off, pred_off = _isolated(criterion, pred, target_box, orig_size, False)
    on, pred_on = _isolated(criterion, pred, target_box, orig_size, True)
    if not selected:
        for key in ('loss_bbox', 'loss_giou', 'loss_nwd', 'loss_xy', 'loss_hw'):
            torch.testing.assert_close(on[key], off[key])
        return
    torch.testing.assert_close(on['loss_giou'], off['loss_giou'] * 2)
    torch.testing.assert_close(on['loss_hw'], off['loss_hw'] * 2)
    torch.testing.assert_close(on['loss_xy'], off['loss_xy'])
    torch.testing.assert_close(on['loss_nwd'], off['loss_nwd'])
    off['loss_bbox'].backward()
    on['loss_bbox'].backward()
    torch.testing.assert_close(pred_on.grad[:2], pred_off.grad[:2])
    torch.testing.assert_close(pred_on.grad[2:], pred_off.grad[2:] * 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--subset', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda')
    smoke = parser.parse_args()
    output = Path(smoke.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ids = _locked_ids(smoke.subset)
    (output / 'ids.json').write_text(json.dumps(ids), encoding='utf-8')

    args = _load_args(smoke.config, smoke.data_root, smoke.pretrained, str(output), smoke.device)
    args.eval_split = getattr(args, 'eval_split', 'test')
    if not args.decoder_heads_only or not args.box_tighten:
        raise SystemExit('Smoke requires the weight-2 decoder-head config')
    if abs(float(args.box_tighten_weight) - 2.0) > 1e-12:
        raise SystemExit('Smoke requires box_tighten_weight 2')

    device = torch.device(args.device)
    model, criterion, _ = build_model_main(args)
    model.to(device)
    from util import legacy_joint as joint
    joint.load_native_finetune(model, args)
    groups = get_param_dict(args, model)
    if not groups or not any(group['params'] for group in groups):
        raise SystemExit('Decoder-head parameter groups are empty')
    model.eval()
    criterion.eval()
    model.transformer.density_gate_scale = 1.0
    for name, parameter in model.named_parameters():
        if 'enc_out_class_embed' in name or 'enc_out_bbox_embed' in name:
            if parameter.requires_grad:
                raise SystemExit(f'{name} is trainable')
        if name.startswith('backbone') and parameter.requires_grad:
            raise SystemExit(f'{name} is trainable')

    dataset = build_dataset(image_set=args.eval_split, args=args)
    restrict_eval_dataset(dataset, str(output / 'ids.json'))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    verytiny = 0
    selected = 0
    class_checked = False
    for samples, targets in loader:
        samples = samples.to(device)
        targets = _move_targets(targets, device)
        grad_enabled = not class_checked
        with torch.set_grad_enabled(grad_enabled):
            outputs = model(samples)
        matched = {key: value for key, value in outputs.items() if key != 'aux_outputs'}
        indices = criterion.matcher(matched, targets)
        rows = _match_rows(indices)
        if rows and not class_checked:
            model.zero_grad(set_to_none=True)
            num_boxes = max(1, sum(len(target['labels']) for target in targets))
            losses = criterion.loss_labels(outputs, targets, indices, num_boxes, log=False)
            losses['loss_ce'].backward()
            class_grads = [
                parameter.grad for name, parameter in model.named_parameters()
                if joint._decoder_class_head(name) and parameter.grad is not None]
            if not class_grads or not any(float(grad.detach().abs().sum()) > 0 for grad in class_grads):
                raise SystemExit('Decoder class head has no gradient')
            for name, parameter in model.named_parameters():
                if 'enc_out_class_embed' in name and parameter.grad is not None:
                    raise SystemExit(f'{name} received a gradient')
            class_checked = True
        areas = _areas(targets, rows)
        src_boxes = outputs['pred_boxes'].detach()
        criterion.box_tighten = True
        criterion.box_tighten_weight = 2.0
        flat_src = torch.stack([src_boxes[b, s] for b, s, _ in rows]) if rows else src_boxes.new_zeros((0, 4))
        flat_tgt = torch.stack([targets[b]['boxes'][t] for b, _, t in rows]) if rows else src_boxes.new_zeros((0, 4))
        weights = criterion._box_tighten_weights(flat_src, flat_tgt, targets, indices)
        for row, area, (batch_index, src, tgt) in zip(range(len(rows)), areas, rows):
            weight = 1.0 if weights is None else float(weights[row])
            is_selected = weight > 1.0
            is_verytiny = area < 64
            if not is_selected and not is_verytiny:
                continue
            if is_selected and verytiny and selected:
                continue
            if is_verytiny and verytiny and not is_selected:
                continue
            _check_pair(
                criterion, src_boxes[batch_index, src], targets[batch_index]['boxes'][tgt],
                targets[batch_index]['orig_size'], is_selected)
            verytiny += int(is_verytiny)
            selected += int(is_selected)
        del outputs
        if class_checked and verytiny and selected:
            break

    report = {
        'verytiny_matches': verytiny,
        'selected_matches': selected,
        'verytiny_loss_unchanged': verytiny > 0,
        'wh_grad_ratio': 2.0 if selected else None,
        'class_head_grad': class_checked,
        'enc_out_class_embed_grad': False if class_checked else None,
        'density_gate_scale': float(model.transformer.density_gate_scale),
        'pass': bool(class_checked and verytiny and selected and model.transformer.density_gate_scale == 1.0),
    }
    (output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('SMOKE', json.dumps(report))
    if not report['pass']:
        raise SystemExit('SMOKE_FAIL')
    print('SMOKE_PASS')


if __name__ == '__main__':
    main()
