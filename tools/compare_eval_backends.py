"""Compare fixed predictions; never executes a detector or training."""
import argparse
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aitodpycocotools.coco import COCO
from datasets.checked_fast_eval import CheckedFastEvaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--metadata', required=True, help='Export metadata with evaluated image_ids')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    gt = COCO(args.annotations)
    records = json.loads(Path(args.predictions).read_text(encoding='utf-8'))
    ids = json.loads(Path(args.metadata).read_text(encoding='utf-8'))['image_ids']
    evaluator = CheckedFastEvaluator(gt, ('bbox',))
    grouped = {int(i): [] for i in ids}
    for record in records:
        grouped[record['image_id']].append(record)
    for image_id, rows in grouped.items():
        boxes = torch.tensor([r['bbox'] for r in rows], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        evaluator.update({image_id: dict(boxes=boxes, scores=torch.tensor([r['score'] for r in rows]),
            labels=torch.tensor([r['category_id'] for r in rows], dtype=torch.long))})
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    Path(args.output).write_text(json.dumps(evaluator.backend_report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
