"""Stress-test one real dense AI-TOD-V2 image without dropping annotations."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from datasets.coco import build, make_coco_transforms
from models.aqfcdetr import build_aqfcdetr
from models.aqfcdetr.query_allocator import QueryBudgetLoss
from util.slconfig import SLConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--config', default='configs/aitodv2/aqfc_r50_5scale_local8gb.py')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = SLConfig.fromfile(args.config)._cfg_dict.to_dict()
    cfg.update(device='cuda', coco_path=args.data_root, fix_size=False,
               mosaic_p=0., copy_paste_p=0.)
    settings = argparse.Namespace(**cfg)
    dataset = build('trainval', settings)
    # Use deterministic full-image resize: no random crop can hide the dense case.
    dataset._transforms = make_coco_transforms('val', args=settings)
    counts = Counter(a['image_id'] for a in dataset.coco.anns.values())
    image_id, original_count = counts.most_common(1)[0]
    sample, target = dataset[dataset.ids.index(image_id)]
    target = {k:v.cuda() for k,v in target.items()}
    model, criterion, _ = build_aqfcdetr(settings)
    model.cuda().train().set_epoch(6)
    model.transformer.force_query_budget = 1500
    criterion.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda'):
        output = model([sample.cuda()], [target])
        losses = criterion(output, [target])
        allocator = QueryBudgetLoss().cuda()(output['allocator_outputs'], {
            'real_counts': torch.tensor([len(target['labels'])], device='cuda'),
            'targets': [target]})
        loss = sum(v*criterion.weight_dict[k] for k,v in losses.items()
                   if k in criterion.weight_dict) + allocator['loss_allocator_total']
    assert torch.isfinite(loss), loss
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    report = dict(status='passed', image_id=image_id, original_annotations=original_count,
                  used_annotations=len(target['labels']), image_shape=list(sample.shape),
                  query_count=int(output['executed_query_counts'][0]), loss=float(loss.detach()),
                  peak_gpu_memory_mb=torch.cuda.max_memory_allocated()/2**20)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
