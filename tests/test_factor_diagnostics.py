import json
from types import SimpleNamespace

import pytest
import torch

from util.factor_diagnostics import restrict_eval_dataset, image_diagnostics, export_class_metrics


@pytest.mark.parametrize('ids', [[1, 1], [4], [], [True], 'bad'])
def test_invalid_ids(tmp_path, ids):
    path = tmp_path / 'ids.json'
    path.write_text(json.dumps(ids))
    dataset = SimpleNamespace(ids=[1, 2, 3])
    with pytest.raises(ValueError):
        restrict_eval_dataset(dataset, path)
    assert dataset.ids == [1, 2, 3]


def test_order_preserved(tmp_path):
    path = tmp_path / 'ids.json'
    path.write_text('[3, 1]')
    dataset = SimpleNamespace(ids=[1, 2, 3], coco=object())
    original = dataset.coco
    restrict_eval_dataset(dataset, path)
    assert dataset.ids == [3, 1] and dataset.coco is original


@pytest.mark.parametrize('empty', [False, True])
def test_diagnostic_is_readonly_and_masks_padding(empty):
    boxes = torch.tensor([[[.2, .2, .1, .1], [.8, .8, .1, .1]]])
    output = {'query_valid_mask': torch.tensor([[True, False]]),
              'interm_outputs': {'pred_boxes': boxes},
              'allocator_outputs': {'raw_query_counts': torch.tensor([300])}}
    target = {'boxes': boxes[0, 1:].clone() if not empty else torch.empty(0, 4)}
    before = boxes.clone()
    row = image_diagnostics(output, target, 0)
    assert row['proposal_hits_050'] == 0 and row['query_mask_count'] == 1
    assert row['proposal_recall_050'] == (None if empty else 0.)
    assert torch.equal(before, boxes)
    json.dumps(row, allow_nan=False)


def test_class_metrics_ignore_invalid(tmp_path):
    import numpy as np
    evaluator = SimpleNamespace(
        eval={'precision': np.array([[[[[.5]], [[-1.]]]]]),
              'recall': np.array([[[[.7]], [[-1.]]]])},
        params=SimpleNamespace(imgIds=[1], catIds=[0, 1], areaRngLbl=['all'],
                               maxDets=[1500], iouThrs=np.array([.5])),
        cocoGt=SimpleNamespace(cats={0: {'name': 'a'}, 1: {'name': 'b'}},
                               catToImgs={0: [1]}, anns={}))
    path = tmp_path / 'classes.json'
    export_class_metrics(evaluator, path)
    rows = json.loads(path.read_text())['classes']
    assert rows[0]['areas']['all']['AP'] == .5
    assert rows[1]['areas']['all']['AP'] is None


def test_sampling_deterministic():
    from tools.run_factor_diagnosis import sample_ids
    annotation = dict(images=[dict(id=i) for i in range(1400)], annotations=[])
    ids, strata = sample_ids(annotation)
    assert len(ids) == len(set(ids)) == 1000
    assert (ids, strata) == sample_ids(annotation)
    assert strata['sample'] == [1000, 0, 0, 0, 0]


def test_cli_default_off():
    from main import get_args_parser
    args = get_args_parser().parse_args([])
    assert args.eval_image_ids == ''
