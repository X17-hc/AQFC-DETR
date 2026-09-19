"""Fixed-32, evaluation-only observer; never starts training or changes predictions."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

FIXED_IDS_SHA = '29bf79c41a519d7acc059522b499ece6d4eeb3207eba9837f8862acbad08a139'
VERSION = 'localization_trajectory_v1'


def native_query_mapping(processor, outputs, sizes, result):
    """Reconstruct native TopK on its original device; reject any disagreement.

    This is an observer, NOT a replacement postprocessor. Ties follow the same
    device TopK implementation; no CPU re-sorting or fabricated query mapping.
    """
    import torch
    from util.box_ops import box_cxcywh_to_xyxy
    if processor.nms_iou_threshold > 0 or len(result) != 1:
        raise ValueError('Only native batch=1, no-NMS postprocessing is supported')
    mask = outputs['query_valid_mask'][0]
    query_ids = torch.where(mask)[0]
    k = int(outputs['executed_query_counts'][0])
    if k != len(query_ids):
        raise ValueError('Query mask/count mismatch')
    logits = outputs['pred_logits'][0, mask]
    classes = (torch.arange(logits.shape[-1], device=logits.device)
               if processor.valid_category_ids is None else
               torch.as_tensor(processor.valid_category_ids, device=logits.device))
    logits = logits[:, classes]
    scores, indexes = torch.topk(logits.sigmoid().flatten(), min(k, logits.numel()))
    selected = query_ids[indexes // len(classes)]
    labels = classes[indexes % len(classes)]
    boxes = box_cxcywh_to_xyxy(outputs['pred_boxes'])[0, selected]
    h, w = sizes[0]
    boxes = boxes * torch.stack([w, h, w, h])
    for key, value in [('scores', scores), ('labels', labels), ('boxes', boxes)]:
        if not torch.equal(value, result[0][key]):
            raise ValueError('Native postprocess mapping differs: ' + key)
    return selected.detach().cpu(), labels.detach().cpu()


def geometry(boxes, gt_boxes, query_ids):
    """Class-agnostic coverage; errors only for best IoU >= .10, not remote boxes."""
    import numpy as np
    from util.error_analysis import box_iou_xywh
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    if not np.isfinite(boxes).all() or (boxes[:, 2:] <= 0).any():
        raise ValueError('Invalid predicted geometry')
    if len(boxes) != len(query_ids):
        raise ValueError('Geometry/index mismatch')
    if not len(gt_boxes):
        return []
    if not np.isfinite(gt_boxes).all() or (gt_boxes[:, 2:] <= 0).any():
        raise ValueError('Invalid GT geometry')
    overlap = box_iou_xywh(boxes, gt_boxes)
    rows = []
    for j, g in enumerate(gt_boxes):
        i = int(overlap[:, j].argmax()) if len(boxes) else None
        best = float(overlap[i, j]) if i is not None else 0.
        row = dict(best_iou=best, best_query_id=int(query_ids[i]) if best > 0 else None,
                   center_error_xy=None, size_ratio_wh=None)
        if best >= .1:
            b = boxes[i]
            row['center_error_xy'] = ((b[:2]+b[2:]/2-g[:2]-g[2:]/2)/g[2:]).tolist()
            row['size_ratio_wh'] = (b[2:]/g[2:]).tolist()
        rows.append(row)
    return rows


def collect(outputs, sizes, result, processor, image, gt, layer_count):
    import numpy as np
    import torch
    from util.error_analysis import area_bucket, density_bucket
    selected, labels = native_query_mapping(processor, outputs, sizes, result)
    mask = outputs['query_valid_mask'][0].detach().cpu()
    ids = torch.where(mask)[0]
    layers = list(outputs.get('aux_outputs', [])) + [outputs]
    if len(layers) != layer_count or layer_count < 1:
        raise ValueError('All Decoder auxiliary layers must be available')
    stages = [('encoder', outputs['interm_outputs'])] + [
        (f'decoder_{i+1}', layer) for i, layer in enumerate(layers)]
    h, w = sizes[0].detach().cpu().tolist()
    if (h, w) != (image['height'], image['width']):
        raise ValueError('Original image dimensions differ')
    raw = dict(image_id=image['id'], query_ids=ids, native_query_ids=selected,
               native_labels=labels, layers={}, coordinate_format='normalized_cxcywh')
    per_stage = {}
    gt_boxes = [g['bbox'] for g in gt]
    for name, layer in stages:
        if not torch.equal(layer['query_valid_mask'][0].detach().cpu(), mask):
            raise ValueError('Layer query identity/mask differs')
        boxes = layer['pred_boxes'][0].detach().cpu()[mask].clone()
        logits = layer['pred_logits'][0].detach().cpu()[mask].clone()
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite classification logits')
        raw['layers'][name] = dict(boxes=boxes, logits=logits)
        xywh = boxes.float().numpy().astype(float)
        xywh[:, :2] -= xywh[:, 2:]/2
        xywh *= np.array([w, h, w, h])
        per_stage[name] = geometry(xywh, gt_boxes, ids.tolist())
    native = result[0]
    native_boxes = native['boxes'].detach().float().cpu().numpy().astype(float)
    native_boxes[:, 2:] -= native_boxes[:, :2]
    raw['native_predictions'] = {key: native[key].detach().cpu().clone()
                                 for key in ('boxes', 'scores', 'labels')}
    per_stage['native_any_class'] = geometry(native_boxes, gt_boxes, selected.tolist())
    per_stage['native_same_class'] = []
    for g in gt:
        valid = labels.numpy() == g['category_id']
        per_stage['native_same_class'].extend(geometry(
            native_boxes[valid], [g['bbox']], selected.numpy()[valid].tolist()))
    rows = [dict(image_id=image['id'], annotation_id=g['id'], category_id=g['category_id'],
                 area=g['area'], size_bucket=area_bucket(g['area']),
                 density_bucket=density_bucket(len(gt)), query_count=len(ids),
                 stages={name: values[i] for name, values in per_stage.items()})
            for i, g in enumerate(gt)]
    info = dict(image_id=image['id'], gt_count=len(gt), valid_queries=len(ids),
                native_detections=len(selected), native_unique_queries=len(set(selected.tolist())),
                omitted_queries=len(ids)-len(set(selected.tolist())), mapping_exact=True)
    return raw, rows, info


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        for key in ['all', 'category/'+str(row['category_id']), 'size/'+row['size_bucket'],
                    'density/'+row['density_bucket'], 'query/'+str(row['query_count'])]:
            groups[key].append(row)
    summary = {}
    for name, group in groups.items():
        stages = {}
        for stage in group[0]['stages']:
            values = [r['stages'][stage]['best_iou'] for r in group]
            stages[stage] = {str(t): dict(hits=sum(v >= t for v in values),
                              coverage=sum(v >= t for v in values)/len(values)) for t in (.5, .75)}
        last = [s for s in stages if s.startswith('decoder_')][-1]
        deltas = {}
        for first, second in [('encoder', last), (last, 'native_any_class'),
                              ('native_any_class', 'native_same_class')]:
            deltas[first+' -> '+second] = {}
            for t in (.5, .75):
                gain = sum(r['stages'][first]['best_iou'] < t <= r['stages'][second]['best_iou'] for r in group)
                loss = sum(r['stages'][second]['best_iou'] < t <= r['stages'][first]['best_iou'] for r in group)
                deltas[first+' -> '+second][str(t)] = dict(gained=gain, lost=loss, net=gain-loss)
        summary[name] = dict(gt_support=len(group), low_support=len(group)<100, stages=stages, changes=deltas)
    return summary


def worker(a):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import legacy24_diagnosis as legacy
    import main as entry
    import torch
    from util.slconfig import SLConfig
    if legacy.digest(a.image_ids) != FIXED_IDS_SHA:
        raise ValueError('Use the frozen 32-image precheck list')
    ids = legacy.read(a.image_ids)
    if len(ids) != 32 or len(set(ids)) != 32:
        raise ValueError('Exactly 32 unique images required')
    cfg = SLConfig.fromfile(str(a.config))
    if cfg.proposal_selection_mode != 'spatial' or not cfg.aux_loss or cfg.use_ema:
        raise ValueError('Expected D0 spatial, auxiliary layers enabled, ordinary model')
    if getattr(cfg, 'force_query_budget', None) not in (None, 0):
        raise ValueError('Expected dynamic queries')
    ann = legacy.read(a.data_root/'annotations/aitodv2_test.json')
    images = {i['id']: i for i in ann['images']}
    gt = defaultdict(list)
    for g in ann['annotations']:
        if g['image_id'] in ids:
            gt[g['image_id']].append(g)
    out = a.output_dir/'trajectory'
    out.mkdir()
    original = entry.evaluate
    rows, image_rows = [], []

    def observed(model, criterion, processors, loader, *rest, **kwargs):
        if list(loader.dataset.ids) != ids or loader.batch_size != 1:
            raise ValueError('Unexpected evaluation order/batch')
        count = 0
        with (out/'paired_gt.jsonl').open('x', encoding='utf-8') as stream:
            def hook(module, inputs, predictions):
                nonlocal count
                if count >= 32 or len(inputs) != 2:
                    raise ValueError('Unexpected native postprocess call')
                image_id = ids[count]
                raw, records, info = collect(inputs[0], inputs[1], predictions, module,
                                             images[image_id], gt[image_id], cfg.dec_layers)
                torch.save(raw, out/f'{image_id}.pth')
                for record in records:
                    stream.write(json.dumps(record, allow_nan=False)+'\n')
                stream.flush()
                rows.extend(records); image_rows.append(info); count += 1
                print(f'[trajectory] {count}/32 image={image_id} K={info["valid_queries"]}', flush=True)
                # Returning None preserves native output, tensor identity and ordering.
            handle = processors['bbox'].register_forward_hook(hook)
            try:
                result = original(model, criterion, processors, loader, *rest, **kwargs)
            finally:
                handle.remove()
        if count != 32:
            raise ValueError('Incomplete trajectory export')
        return result

    entry.evaluate = observed
    saved_argv = sys.argv
    sys.argv = [str(root/'tools/legacy24_diagnosis.py'), '--mode', 'evaluate', '--group', 'D0',
                '--checkpoint', str(a.checkpoint), '--config', str(a.config),
                '--data-root', str(a.data_root), '--output-dir', str(a.output_dir/'evaluation'),
                '--image-ids', str(a.image_ids), '--export-proposals']
    try:
        legacy.main()
    finally:
        sys.argv = saved_argv
        entry.evaluate = original
    legacy.save(out/'images.json', image_rows)
    legacy.save(out/'summary.json', dict(version=VERSION, images=len(image_rows), gt=len(rows),
                strata=summarize(rows), boundary='32-image engineering diagnosis; geometry is not TP/AP',
                error_scope='Offsets/size ratios only when best IoU >= 0.10; diagnostic area bins <64/<256/<1024'))
    (out/'README.md').write_text(
        '# 逐层定位轨迹诊断\n\n仅32图，不用于总体AP结论。\n\n'
        '- paired_gt.jsonl：同GT各层最佳IoU与条件几何误差。\n'
        '- summary.json：分层覆盖、encoder→末层→原生预测的增减。\n'
        '- 每图pth：全部有效query逐层框/logits、原生query—类别映射。\n'
        '- 原生映射在原设备重建，并与实际后处理输出逐元素严格核对。\n'
        '- native_any_class为保留框的几何覆盖；native_same_class再要求类别正确。\n'
        '- 不是一对一TP；最佳query可跨层变化，完整pth保留稳定query索引供追踪。\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint', 'config', 'data-root', 'image-ids', 'output-dir']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.worker:
        worker(a)
        return
    # Parent captures ALL native stdout/stderr; each manual run gets a new directory.
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
    out = a.output_dir/stamp
    out.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker']
    for name in ['checkpoint', 'config', 'data-root', 'image-ids']:
        command += ['--'+name, str(getattr(a, name.replace('-', '_')).resolve())]
    command += ['--output-dir', str(out.resolve())]
    print('Output: '+str(out), flush=True)
    start = time.time()
    with (out/'console.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace', bufsize=1)
        for line in process.stdout:
            log.write(line); log.flush(); print(line, end='', flush=True)
        code = process.wait()
    (out/'run_status.json').write_text(json.dumps(dict(version=VERSION, command=command,
        exit_code=code, status='completed' if code==0 else 'failed', elapsed_seconds=time.time()-start),
        ensure_ascii=False, indent=2), encoding='utf-8')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
