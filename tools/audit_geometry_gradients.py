"""Manual bounded gradient diagnostic. No optimizer, training updates or checkpoint writes."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def gradient_summary(parts):
    import torch
    tensors = [x.detach().float() for x in parts if x is not None]
    if any(not torch.isfinite(x).all() for x in tensors):
        raise FloatingPointError('Non-finite gradient; stopping, no fallback')
    return sum(float(x.double().square().sum()) for x in tensors) ** .5


def cosine(parts, other):
    na, nb = gradient_summary(parts), gradient_summary(other)
    if not na or not nb:
        return None
    return sum(float((a.detach().double().cpu() * b.detach().double().cpu()).sum())
               for a, b in zip(parts, other) if a is not None and b is not None) / (na * nb)


class Tee:
    def __init__(self, terminal, file):
        self.terminal, self.file = terminal, file

    def write(self, text):
        self.terminal.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()


def audit(cli, output, record):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from main import build_model_main
    from datasets import build_dataset
    from datasets.coco import make_coco_transforms
    from util.slconfig import SLConfig
    from util.config_validation import validate_config
    from util.incremental_checkpoint import validate_warmstart
    from util.experiment import sha256
    from util.misc import collate_fn
    from engine import AllocatorWeightScheduler
    from models.aqfcdetr.query_allocator import QueryBudgetLoss

    config = SLConfig.fromfile(cli.config)._cfg_dict.to_dict()
    validate_config(config)
    config.update(coco_path=cli.data_root, device='cuda', distributed=False, rank=0,
                  fix_size=False, masks=False, pretrain_model_path=cli.pretrained,
                  num_workers=0, data_aug_scales=[800], data_aug_max_size=1333,
                  mosaic_p=0., copy_paste_p=0.)
    args = argparse.Namespace(**config)
    if args.geometry_loss_weight != .05 or args.training_phase_fixed_epoch != 23:
        raise ValueError('This diagnostic requires frozen C1 geometry_v1 settings')
    annotation = Path(cli.data_root) / 'annotations/aitodv2_trainval.json'
    checkpoint_hash = sha256(cli.pretrained)
    annotation_hash = sha256(annotation)
    if checkpoint_hash != args.expected_pretrained_sha256:
        raise ValueError('Checkpoint hash differs from the common epoch23 starting point')
    ids = sorted(x['id'] for x in json.loads(annotation.read_text())['images'])
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate image IDs')
    random.Random(42).shuffle(ids)
    ids = ids[:2 * cli.batches]
    if len(ids) != 2 * cli.batches:
        raise ValueError('Insufficient images')
    record.update(resolved_config=config, image_ids=ids, checkpoint_sha256=checkpoint_hash,
                  annotation_sha256=annotation_hash, cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
                  torch=torch.__version__, cuda=torch.version.cuda,
                  comparison='C0 vs C1 on SAME forward, SAME matching, geometry weight at plateau 0.05',
                  precision_note='FP32/AMP use paired RNG/buffers, but independently matched predictions; not a numerical equivalence test',
                  limitation='Unscaled autograd; does not reproduce GradScaler overflow or optimizer update. Deterministic validation resize, not augmented training distribution.')
    if cli.check_only:
        state = torch.load(cli.pretrained, map_location='cpu', weights_only=False)
        if state.get('epoch') != 23 or not state.get('model'):
            raise ValueError('Invalid checkpoint metadata')
        record['status'] = 'preflight_only'
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback')
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    model, criterion, _ = build_model_main(args)
    validate_warmstart(model, args)
    state = torch.load(cli.pretrained, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    del state
    model.cuda().train(); model.set_epoch(23); criterion.cuda().train()
    criterion.quality_lambda = .25
    criterion.weight_dict['loss_geometry'] = .05
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    head = [(n, p) for n, p in params if 'bbox_embed' in n]
    if not head:
        raise ValueError('No bbox head found')
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    versions = {n: p._version for n, p in params}
    budget = QueryBudgetLoss(coverage_weight=args.coverage_loss_weight,
        spacing_weight=args.spacing_loss_weight, count_weight=args.count_loss_weight,
        interval_weight=args.interval_loss_weight, boundary_guide_weight=args.boundary_guide_loss_weight,
        density_weight=args.density_map_loss_weight, density_target_backend=args.density_target_backend,
        density_target_chunk_size=args.density_target_chunk_size).cuda()
    weight = AllocatorWeightScheduler(**args.allocator_schedule, total_epochs=24).get_weight(23, 0, 1) * args.allocator_loss_weight
    dataset = build_dataset('trainval', args)
    dataset._transforms = make_coco_transforms('val', fix_size=False, strong_aug=False, args=args)
    dataset.filter_empty_gt = False
    index = {v: i for i, v in enumerate(dataset.ids)}
    loader = DataLoader(Subset(dataset, [index[i] for i in ids]), batch_size=2,
                        num_workers=0, collate_fn=collate_fn)
    record.update(gpu=str(torch.cuda.get_device_properties(0)), allocator_weight=weight,
                  head_parameters=[n for n, _ in head], gradient_clip=args.clip_max_norm)
    with (output / 'batches.jsonl').open('x', encoding='utf-8') as stream:
        for batch, (samples, targets) in enumerate(loader):
            samples = samples.to('cuda')
            targets = [{k: v.to('cuda') if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
            for mode in cli.precisions.split(','):
                with torch.no_grad():
                    for n, b in model.named_buffers():
                        b.copy_(buffers[n])
                random.seed(42 + batch); np.random.seed(42 + batch)
                torch.manual_seed(42 + batch); torch.cuda.manual_seed_all(42 + batch)
                with torch.amp.autocast('cuda', enabled=mode == 'amp'):
                    out = model(samples, targets)
                    losses = criterion(out, targets)
                    alloc = budget(out['allocator_outputs'], dict(targets=targets,
                        real_counts=torch.tensor([len(t['labels']) for t in targets], device='cuda')))
                    base = sum(v * criterion.weight_dict[k] for k, v in losses.items()
                               if k in criterion.weight_dict and k != 'loss_geometry') + weight * alloc['loss_allocator_total']
                    geometry = losses['loss_geometry'] * .05
                if not torch.isfinite(base + geometry):
                    raise FloatingPointError('Non-finite loss')
                row = dict(batch=batch, precision=mode, image_ids=[int(t['image_id'].item()) for t in targets],
                           gt_count=sum(len(t['labels']) for t in targets),
                           query_counts=out['executed_query_counts'].tolist(), losses={}, terms={})
                grads = {}
                for key in ('loss_bbox', 'loss_giou', 'loss_nwd', 'loss_geometry'):
                    term = losses[key] * criterion.weight_dict[key]
                    g = torch.autograd.grad(term, [out['pred_boxes']] + [p for _, p in head],
                                            retain_graph=True, allow_unused=True)
                    grads[key] = [None if x is None else x.detach().float().cpu() for x in g]
                    row['losses'][key] = float(term.detach())
                    row['terms'][key] = dict(output_norm=gradient_summary(g[:1]), head_norm=gradient_summary(g[1:]),
                        coordinate_norms={name: gradient_summary([g[0][..., i]]) for i, name in enumerate(('cx', 'cy', 'w', 'h'))})
                row['geometry_cosines'] = {key: dict(output=cosine(grads['loss_geometry'][:1], grads[key][:1]),
                    head=cosine(grads['loss_geometry'][1:], grads[key][1:])) for key in ('loss_bbox', 'loss_giou', 'loss_nwd')}
                totals = []
                for label, loss in (('c0', base), ('c1', base + geometry)):
                    g = torch.autograd.grad(loss, [p for _, p in params], retain_graph=label == 'c0', allow_unused=True)
                    norm = gradient_summary(g)
                    factor = min(1., args.clip_max_norm / (norm + 1e-6)) if args.clip_max_norm > 0 else 1.
                    row[label] = dict(total_loss=float(loss.detach()), global_norm=norm, clip_factor=factor,
                                      post_clip_norm=norm * factor)
                    totals.append([None if x is None else x.detach().float().cpu() for x in g])
                    del g
                row['total_gradient_cosine'] = cosine(*totals)
                row['global_gradient_change_norm'] = gradient_summary([
                    (b if a is None else -a if b is None else b - a)
                    for a, b in zip(*totals) if a is not None or b is not None])
                stream.write(json.dumps(row, allow_nan=False) + '\n'); stream.flush()
                print(f'[gradient] batch {batch + 1}/{cli.batches} {mode}: C0 norm={row["c0"]["global_norm"]:.6g}, C1 norm={row["c1"]["global_norm"]:.6g}', flush=True)
                del totals, grads, out, losses, alloc, base, geometry, term, loss
    with torch.no_grad():
        for n, b in model.named_buffers():
            b.copy_(buffers[n])
    if any(p._version != versions[n] or p.grad is not None for n, p in params):
        raise AssertionError('Unexpected parameter mutation or accumulated gradients')
    if sha256(cli.pretrained) != checkpoint_hash or sha256(annotation) != annotation_hash:
        raise AssertionError('Input changed during diagnostic')
    record.update(status='completed', optimizer_updates=0, checkpoint_writes=0, inputs_unchanged=True,
                  parameters_unchanged=True, buffers_restored=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--pretrained', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--batches', type=int, default=8)
    p.add_argument('--precisions', choices=['fp32', 'amp', 'fp32,amp'], default='fp32,amp')
    p.add_argument('--check-only', action='store_true')
    cli = p.parse_args()
    if not 1 <= cli.batches <= 16:
        p.error('Bounded diagnostic requires 1..16 batches')
    from datetime import datetime, timezone
    import uuid
    output = Path(cli.output_dir) / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    record = dict(status='running', command=sys.argv, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    started = time.perf_counter()
    with (output / 'console.log').open('x', encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print(f'Output: {output}\nNo optimizer updates; no checkpoint writes.')
            try:
                audit(cli, output, record)
            except Exception as exc:
                import traceback
                record.update(status='failed', error=repr(exc)); traceback.print_exc()
                raise
            finally:
                record['elapsed_seconds'] = time.perf_counter() - started
                (output / 'manifest.json').write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
            print(f'Status: {record["status"]}; elapsed {record["elapsed_seconds"]:.2f}s')


if __name__ == '__main__':
    main()
