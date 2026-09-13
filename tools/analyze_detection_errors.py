"""Analyze existing prediction exports, optionally render images using Supervision."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from util.error_analysis import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--image-root')
    parser.add_argument('--diagnostics-dir', help='Optional exported .npz density/proposal directory')
    parser.add_argument('--confidence', type=float, default=.25)
    parser.add_argument('--iou', type=float, default=.5)
    parser.add_argument('--max-visualizations', type=int, default=30)
    args = parser.parse_args()
    read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
    annotations = read(args.annotations)
    result = analyze(read(args.predictions), annotations, read(args.metadata)['image_ids'], args.confidence, args.iou)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output/'error_analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.diagnostics_dir:
        import numpy as np
        from PIL import Image, ImageDraw
        from util.error_analysis import box_iou_xywh
        coverage = []
        for row in result['images']:
            source = Path(args.diagnostics_dir)/f"{row['image_id']}.npz"
            if not source.exists():
                continue
            with np.load(source, allow_pickle=False) as data:
                h,w = data['original_size']
                boxes = data['proposals_cxcywh'].copy()
                boxes[:,:2] -= boxes[:,2:]/2
                boxes *= np.array([w,h,w,h])
                gt = [g['bbox'] for g in row['ground_truth']]
                ious = box_iou_xywh(boxes, gt)
                best = ious.max(0) if len(boxes) else np.zeros(len(gt))
                coverage.append(dict(image_id=row['image_id'], gt=len(gt), proposals=len(boxes),
                                     recalled_025=int((best>=.25).sum()), recalled_050=int((best>=.5).sum())))
                if 'density' in data:
                    density, valid = data['density'], data['valid_mask']
                    density = density[:valid.any(1).sum(), :valid.any(0).sum()]
                    red = (np.clip(density,0,1)*255).astype(np.uint8)
                    rgb = np.stack([red, np.zeros_like(red), 255-red], -1)
                    canvas = Image.fromarray(rgb).resize((int(w),int(h)))
                    draw = ImageDraw.Draw(canvas)
                    for x,y,bw,bh in boxes:
                        cx,cy=x+bw/2,y+bh/2
                        draw.ellipse((cx-1,cy-1,cx+1,cy+1),fill='white')
                    canvas.save(output/f"{row['image_id']}_density_candidates.png")
        (output/'proposal_recall.json').write_text(json.dumps(coverage,indent=2),encoding='utf-8')
    if args.image_root:
        import numpy as np
        import torch
        from PIL import Image, ImageDraw
        from util.prediction_export import to_supervision
        import supervision as sv
        files = {i['id']: i['file_name'] for i in annotations['images']}
        for row in result['images'][:args.max_visualizations]:
            image = Image.open(Path(args.image_root)/files[row['image_id']]).convert('RGB')
            pred = row['predictions']
            boxes = torch.tensor([p['bbox'] for p in pred]).reshape(-1, 4)
            boxes[:, 2:] += boxes[:, :2]
            detections = to_supervision(dict(boxes=boxes, scores=[p['score'] for p in pred],
                                              labels=[p['category_id'] for p in pred]), annotations['categories'])
            scene = sv.BoxAnnotator().annotate(np.array(image), detections)
            canvas = Image.fromarray(scene); draw = ImageDraw.Draw(canvas)
            for p, status in zip(pred, row['prediction_status']):
                draw.text(tuple(p['bbox'][:2]), status, fill='yellow')
            for j, gt in enumerate(row['ground_truth']):
                x,y,w,h=gt['bbox']
                draw.rectangle((x,y,x+w,y+h), outline='red' if j in row['missed_gt'] else 'lime')
            canvas.save(output/f"{row['image_id']}.png")


if __name__ == '__main__':
    main()
