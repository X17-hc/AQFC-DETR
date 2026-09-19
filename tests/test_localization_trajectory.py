import copy
import importlib.util
from pathlib import Path

import pytest
import torch
from models.aqfcdetr.model import PostProcess

source = Path(__file__).with_name('diagnose_localization_trajectory.py')
if not source.exists():
    source = Path(__file__).resolve().parents[1]/'tools/diagnose_localization_trajectory.py'
spec = importlib.util.spec_from_file_location('trajectory', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(ties=False):
    logits = torch.zeros(1, 4, 4) if ties else torch.tensor([[
        [1., 8., 2., 7.], [100., 100., 100., 100.],
        [0., 2., 0., 1.], [0., 1., 0., 2.]]])
    boxes = torch.tensor([[[.5, .5, .2, .2], [.2, .2, .1, .1],
                           [.2, .2, .1, .1], [.7, .8, .1, .1]]])
    mask = torch.tensor([[True, False, True, True]])
    layer = dict(pred_logits=logits, pred_boxes=boxes, query_valid_mask=mask)
    out = dict(**layer, executed_query_counts=torch.tensor([3]),
               aux_outputs=[copy.deepcopy(layer)], interm_outputs=copy.deepcopy(layer))
    return out, torch.tensor([[100, 200]]), PostProcess(valid_category_ids=[1, 3])


@pytest.mark.parametrize('ties', [False, True])
def test_native_mapping_exact_and_nonmutating(ties):
    out, sizes, processor = fixture(ties)
    before = copy.deepcopy(out)
    result = processor(out, sizes)
    ids, labels = module.native_query_mapping(processor, out, sizes, result)
    assert 1 not in ids and set(labels.tolist()) <= {1, 3}
    assert torch.equal(out['pred_boxes'], before['pred_boxes'])
    assert torch.equal(out['pred_logits'], before['pred_logits'])
    again = processor(out, sizes)
    for key in ['scores', 'boxes', 'labels']:
        assert torch.equal(result[0][key], again[0][key])


def test_reject_wrong_mapping_and_nms():
    out, sizes, processor = fixture()
    result = processor(out, sizes)
    result[0]['scores'][0] -= .1
    with pytest.raises(ValueError, match='mapping differs'):
        module.native_query_mapping(processor, out, sizes, result)
    processor.nms_iou_threshold = .5
    with pytest.raises(ValueError, match='no-NMS'):
        module.native_query_mapping(processor, out, sizes, result)


def test_empty_and_remote_geometry():
    assert module.geometry([], [], []) == []
    empty = module.geometry([], [[0, 0, 2, 2]], [])[0]
    assert empty['best_iou'] == 0 and empty['center_error_xy'] is None
    remote = module.geometry([[50, 50, 2, 2]], [[0, 0, 2, 2]], [8])[0]
    assert remote['best_query_id'] is None and remote['size_ratio_wh'] is None
    exact = module.geometry([[0, 0, 2, 2]], [[0, 0, 2, 2]], [8])[0]
    assert exact['best_iou'] == 1 and exact['center_error_xy'] == [0, 0]


def test_collection_stages_index_and_summary():
    out, sizes, processor = fixture()
    result = processor(out, sizes)
    image = dict(id=10, width=200, height=100)
    gt = [dict(id=12, category_id=1, bbox=[80, 40, 40, 20], area=800)]
    raw, rows, info = module.collect(out, sizes, result, processor, image, gt, 2)
    assert raw['query_ids'].tolist() == [0, 2, 3]
    assert list(raw['layers']) == ['encoder', 'decoder_1', 'decoder_2']
    assert info['native_unique_queries'] < info['native_detections']
    assert rows[0]['stages']['decoder_2']['best_iou'] > .99999
    stats = module.summarize(rows)['all']
    assert stats['gt_support'] == 1
    assert stats['changes']['encoder -> decoder_2']['0.75']['net'] == 0
    with pytest.raises(ValueError, match='auxiliary'):
        module.collect(out, sizes, result, processor, image, gt, 6)
    out['aux_outputs'][0]['query_valid_mask'][0, 0] = False
    with pytest.raises(ValueError, match='identity'):
        module.collect(out, sizes, result, processor, image, gt, 2)


def test_exact_threshold_and_lost_coverage():
    row = dict(category_id=1, size_bucket='verytiny', density_bucket='1-100', query_count=3,
               stages={name: dict(best_iou=value) for name, value in [
                   ('encoder', .5), ('decoder_1', .75),
                   ('native_any_class', .5), ('native_same_class', .1)]})
    s = module.summarize([row])['all']
    assert s['stages']['encoder']['0.5']['hits'] == 1
    assert s['changes']['encoder -> decoder_1']['0.75']['gained'] == 1
    assert s['changes']['decoder_1 -> native_any_class']['0.75']['lost'] == 1


def test_hook_preserves_native_tensor_identity():
    out, sizes, processor = fixture()
    captured = []
    def observer(m, args, result):
        module.native_query_mapping(m, *args, result)
        captured.append(result)
    handle = processor.register_forward_hook(observer)
    try:
        result = processor(out, sizes)
    finally:
        handle.remove()
    assert result is captured[0]
