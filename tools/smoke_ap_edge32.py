"""32-image mechanism check for the distribution-refiner edge loss.

Writes the first 32 locked test ids. A matched refined box below IoU 0.60
must send a width/height gradient into distribution_refiner. A matched box at
or above 0.60 contributes zero of this loss. Frozen modules stay without a
gradient. density_gate_scale stays 1.
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
from util.box_ops import box_cxcywh_to_xyxy, box_iou
from util.config_validation import validate_config
from util.edge_logwh import low_iou_logwh_loss
from util.factor_diagnostics import restrict_eval_dataset
from util.get_param_dicts import get_param_dict
from util.misc import collate_fn
from util.slconfig import SLConfig
from util import legacy_joint as joint


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
    parser = argparse.ArgumentParser('edge smoke', parents=[get_args_parser()])
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


def _indices_for(rows, batch, device):
    grouped = {index: ([], []) for index in range(batch)}
    for batch_index, src, tgt in rows:
        grouped[batch_index][0].append(src)
        grouped[batch_index][1].append(tgt)
    indices = []
    for index in range(batch):
        src, tgt = grouped[index]
        indices.append((
            torch.tensor(src, dtype=torch.long, device=device),
            torch.tensor(tgt, dtype=torch.long, device=device)))
    return indices


def _pair_loss(outputs, targets, row, device):
    return low_iou_logwh_loss(
        outputs, targets, _indices_for([row], len(targets), device), 1)


def _refiner_grad(model):
    grads = [
        parameter.grad for name, parameter in model.named_parameters()
        if joint._distribution_refiner(name) and parameter.grad is not None]
    return bool(grads) and any(float(grad.detach().abs().sum()) > 0 for grad in grads)


def _frozen_grad(model):
    hit = []
    for name, parameter in model.named_parameters():
        if joint._distribution_refiner(name):
            continue
        if parameter.requires_grad or parameter.grad is not None:
            hit.append(name)
    return hit


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
    if not args.edge_refine_only or not args.edge_logwh:
        raise SystemExit('Smoke requires the log-width refiner config')

    device = torch.device(args.device)
    model, criterion, _ = build_model_main(args)
    model.to(device)
    joint.load_native_finetune(model, args)
    groups = get_param_dict(args, model)
    names = [name for group in groups for name in group.get('joint_names', [])]
    if not names or any(not joint._distribution_refiner(name) for name in names):
        raise SystemExit(f'Refiner groups are wrong: {names[:8]}')
    model.eval()
    model.distribution_refiner.train()
    criterion.eval()
    model.transformer.density_gate_scale = 1.0

    dataset = build_dataset(image_set=args.eval_split, args=args)
    restrict_eval_dataset(dataset, str(output / 'ids.json'))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    high_zero = None
    verytiny_grad = None
    area64_grad = None
    frozen = []
    for samples, targets in loader:
        if high_zero is True and verytiny_grad is True and area64_grad is True:
            break
        samples = samples.to(device)
        targets = _move_targets(targets, device)
        outputs = model(samples, targets)
        matched = {key: value for key, value in outputs.items() if key != 'aux_outputs'}
        matched['pred_boxes'] = outputs['pred_boxes_coarse']
        indices = criterion.matcher(matched, targets)
        pending = []
        for batch_index, (src_idx, tgt_idx) in enumerate(indices):
            for src, tgt in zip(src_idx.tolist(), tgt_idx.tolist()):
                pred = outputs['pred_boxes'][batch_index, src]
                truth = targets[batch_index]['boxes'][tgt]
                with torch.no_grad():
                    iou = box_iou(
                        box_cxcywh_to_xyxy(pred.detach()[None]),
                        box_cxcywh_to_xyxy(truth.detach()[None]))[0].reshape(-1)[0]
                box = truth.detach().float()
                size = targets[batch_index]['orig_size'].detach().float().flatten()
                area = float(box[2] * size[1] * box[3] * size[0])
                pending.append((float(iou), area, batch_index, int(src), int(tgt)))
        if high_zero is None:
            high_rows = [row for row in pending if row[0] >= 0.60]
            if high_rows:
                loss = _pair_loss(outputs, targets, high_rows[0][2:], device)
                high_zero = float(loss.detach()) == 0.0
                del loss
        low_rows = [row for row in pending if row[0] < 0.60]
        for want_verytiny in (True, False):
            recorded = verytiny_grad if want_verytiny else area64_grad
            if recorded is True:
                continue
            for row in low_rows:
                if (row[1] < 64) is not want_verytiny:
                    continue
                model.zero_grad(set_to_none=True)
                loss = _pair_loss(outputs, targets, row[2:], device)
                if float(loss.detach()) == 0.0:
                    del loss
                    continue
                loss.backward(retain_graph=True)
                got = _refiner_grad(model)
                frozen = _frozen_grad(model)
                del loss
                if want_verytiny:
                    verytiny_grad = got
                else:
                    area64_grad = got
                if got:
                    break
        del outputs

    report = {
        'high_iou_loss_zero': high_zero,
        'verytiny_grad': verytiny_grad,
        'area64_grad': area64_grad,
        'frozen_with_grad': frozen,
        'density_gate_scale': float(model.transformer.density_gate_scale),
        'pass': bool(
            high_zero is True
            and area64_grad is True
            and not frozen
            and float(model.transformer.density_gate_scale) == 1.0),
    }
    (output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('SMOKE', json.dumps(report))
    if verytiny_grad is not True and area64_grad is not True:
        raise SystemExit('SMOKE_FAIL neither size class produced a refiner gradient')
    if not report['pass']:
        raise SystemExit('SMOKE_FAIL')
    print('SMOKE_PASS')


if __name__ == '__main__':
    main()
