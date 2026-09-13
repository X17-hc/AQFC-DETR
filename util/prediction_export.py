"""COCO export preserving original IDs, including explicit empty-image records."""
import json
from pathlib import Path
import torch


def category_mapping(categories):
    ids = sorted(int(c['id']) for c in categories)
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate category IDs')
    return {category: index for index, category in enumerate(ids)}


def prediction_records(image_id, result, categories):
    mapping = category_mapping(categories)
    boxes = torch.as_tensor(result['boxes']).detach().cpu().reshape(-1, 4)
    labels = torch.as_tensor(result['labels']).detach().cpu().reshape(-1)
    scores = torch.as_tensor(result['scores']).detach().cpu().reshape(-1)
    if not (len(boxes) == len(labels) == len(scores)):
        raise ValueError('Prediction box/label/score lengths differ')
    unknown = sorted(set(labels.tolist()) - set(mapping))
    if unknown:
        raise ValueError(f'Unknown prediction categories {unknown}; no implicit class remapping allowed')
    if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
        raise ValueError('Nonfinite predictions cannot be exported')
    records = []
    for box, label, score in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
        x1, y1, x2, y2 = box
        records.append(dict(image_id=int(image_id), category_id=int(label),
                            bbox=[x1, y1, x2-x1, y2-y1], score=float(score)))
    return records


def save_predictions(directory, records, images, categories, metadata=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'predictions.json').write_text(json.dumps(records, allow_nan=False), encoding='utf-8')
    (directory / 'images.jsonl').write_text(''.join(json.dumps(row, allow_nan=False) + '\n'
                                                  for row in images), encoding='utf-8')
    data = dict(categories=categories, category_to_index=category_mapping(categories),
                image_ids=[row['image_id'] for row in images], metadata=metadata or {})
    (directory / 'metadata.json').write_text(json.dumps(data, indent=2, ensure_ascii=False,
                                                       default=str), encoding='utf-8')


def to_supervision(result, categories):
    # Import is isolated from model/train imports, and IDs remain reversible.
    try:
        import supervision as sv
    except ImportError as exc:
        raise RuntimeError('Optional supervision==0.30.1 is unavailable; see requirements-analysis.txt') from exc
    mapping = category_mapping(categories)
    prediction_records(0, result, categories)  # validate before indexing
    import numpy as np
    return sv.Detections(xyxy=torch.as_tensor(result['boxes']).detach().cpu().numpy().reshape(-1, 4),
                         confidence=torch.as_tensor(result['scores']).detach().cpu().numpy(),
                         class_id=np.array([mapping[int(x)] for x in result['labels']], dtype=int))
