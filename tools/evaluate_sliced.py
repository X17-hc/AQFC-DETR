"""Explicit, manual tiled inference. No training; original category IDs preserved."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--image-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--image-ids', help='JSON list; omitted means all annotation images')
    parser.add_argument('--slice-wh', type=int, default=320)
    parser.add_argument('--overlap-wh', type=int, default=64)
    parser.add_argument('--merge', choices=['nms','nmm'], default='nms')
    parser.add_argument('--overlap-metric', choices=['iou','ios'], default='iou')
    parser.add_argument('--iou', type=float, default=.5)
    parser.add_argument('--include-full-image', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    import supervision as sv
    from PIL import Image
    from util.inference_runner import InferenceRunner
    from util.prediction_export import to_supervision, prediction_records, save_predictions, category_mapping
    from aitodpycocotools.coco import COCO
    from datasets.coco_eval import CocoEvaluator
    annotation = json.loads(Path(args.annotations).read_text(encoding='utf-8'))
    categories = annotation['categories']
    inverse = {v:k for k,v in category_mapping(categories).items()}
    selected = set(json.loads(Path(args.image_ids).read_text())) if args.image_ids else None
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    runner = InferenceRunner(args.config, args.checkpoint, args.device, args.amp)
    evaluator = CocoEvaluator(COCO(args.annotations), ('bbox',))
    callback = lambda tile: to_supervision(runner(tile), categories)
    # Merge once globally, including the optional full-image branch.
    slicer = sv.InferenceSlicer(callback=callback, slice_wh=args.slice_wh,
        overlap_wh=args.overlap_wh, overlap_filter=sv.OverlapFilter.NONE, thread_workers=1)
    records, images = [], []
    for item in annotation['images']:
        if selected is not None and item['id'] not in selected:
            continue
        image = np.array(Image.open(Path(args.image_root)/item['file_name']).convert('RGB'))
        start = time.perf_counter(); runner.records.clear()
        detections = slicer(image)
        if args.include_full_image:
            detections = sv.Detections.merge([detections, callback(image)])
        before_merge = time.perf_counter()
        metric = sv.OverlapMetric.IOU if args.overlap_metric == 'iou' else sv.OverlapMetric.IOS
        detections = (detections.with_nms if args.merge == 'nms' else detections.with_nmm)(
            threshold=args.iou, class_agnostic=False, overlap_metric=metric)
        result = dict(boxes=torch.as_tensor(detections.xyxy), scores=torch.as_tensor(detections.confidence),
                      labels=torch.tensor([inverse[int(i)] for i in detections.class_id], dtype=torch.long))
        records.extend(prediction_records(item['id'], result, categories))
        elapsed = time.perf_counter()
        images.append(dict(image_id=item['id'], width=image.shape[1], height=image.shape[0],
            slice_count=len(runner.records)-int(args.include_full_image), branches=list(runner.records),
            total_query_count=sum(r['query_count'] for r in runner.records),
            end_to_end_ms=(elapsed-start)*1000, merge_ms=(elapsed-before_merge)*1000))
        evaluator.update({item['id']: result})
    if not images:
        raise ValueError('No images selected')
    save_predictions(output, records, images, categories, vars(args))
    evaluator.synchronize_between_processes(); evaluator.accumulate(); evaluator.summarize()
    (output/'metrics.json').write_text(json.dumps(evaluator.named_metrics(), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
