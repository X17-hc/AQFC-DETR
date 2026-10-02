import importlib.util
from pathlib import Path

import torch

_PATH = Path(__file__).resolve().parents[1] / 'tools' / 'diagnose_ap_stall.py'
_SPEC = importlib.util.spec_from_file_location('diagnose_ap_stall', _PATH)
diag = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diag)


def _summary(ap=0.322, ap50=0.682, ap75=0.263, oracle=0.30, recall75=0.40):
    grid = {}
    for thr in diag.IOU_THRESHOLDS:
        for area in ('all',) + diag.SIZE_NAMES:
            grid[f'{thr:.2f}:{area}'] = ap
    counts = {name: {'unmatched': 10, 'loose': 10, 'tight': 100} for name in diag.SIZE_NAMES}
    medians = {
        name: {
            'loose': {'center_px': 2.0, 'wh_rel': 0.40, 'n': 10},
            'tight': {'center_px': 1.0, 'wh_rel': 0.20, 'n': 100},
        }
        for name in diag.SIZE_NAMES
    }
    return {
        'ap': ap, 'ap50': ap50, 'ap75': ap75, 'oracle_ap75': oracle,
        'recall_50': 0.70, 'recall_75': recall75, 'ap_grid': grid,
        'counts': counts, 'medians': medians,
    }


def test_rescore_puts_tighter_box_first():
    detections = [{'score': 0.9, 'id': 1}, {'score': 0.2, 'id': 2}]
    ious = [[0.20], [0.80]]
    rescored, matrix = diag.rescore_by_iou(detections, ious)
    assert [item['id'] for item in rescored] == [2, 1]
    assert rescored[0]['score'] == 0.8
    assert matrix[0, 0] == 0.8


def test_loose_box_records_center_error_without_counting_as_tight():
    table = diag.GeometryTable()
    gt = torch.tensor([[0., 0., 10., 10.]])
    pred = torch.tensor([[1., 1., 11., 11.]])
    table.add(pred, torch.tensor([0.9]), torch.tensor([3]), gt, torch.tensor([3]), torch.tensor([100.]))
    result = table.as_dict()
    assert result['counts']['tiny']['loose'] == 1
    assert result['counts']['tiny']['tight'] == 0
    assert result['medians']['tiny']['loose']['center_px'] > 1.0
    assert result['medians']['tiny']['loose']['wh_rel'] == 0.0


def test_cancellation_names_the_opposing_cells():
    h1 = _summary()
    current = _summary()
    current['ap_grid']['0.50:verytiny'] = 0.326
    current['ap_grid']['0.80:small'] = 0.318
    report = diag.decide(h1, current)
    assert report['cause'] == 'cancellation'
    assert 'verytiny' in report['sentence']
    assert 'small' in report['sentence']


def test_center_improvement_without_an_edge_change():
    h1 = _summary()
    current = _summary()
    current['ap_grid']['0.50:all'] = 0.326
    current['counts']['verytiny']['loose'] = 12
    current['medians']['verytiny']['loose']['center_px'] = 1.7
    report = diag.decide(h1, current)
    assert report['cause'] == 'center_without_edge'


def test_flat_errors_mean_localization_did_not_move():
    h1 = _summary()
    current = _summary()
    current['ap_grid']['0.50:all'] = 0.326
    current['counts']['verytiny']['loose'] = 12
    current['medians']['verytiny']['loose']['center_px'] = 2.05
    report = diag.decide(h1, current)
    assert report['cause'] == 'localization_unchanged'


def test_extra_tight_boxes_with_flat_ap75_are_eaten_by_false_positives():
    h1 = _summary()
    current = _summary()
    current['ap_grid']['0.80:all'] = 0.325
    for name in diag.SIZE_NAMES:
        current['counts'][name]['tight'] = 110
    current['recall_75'] = 0.42
    report = diag.decide(h1, current)
    assert report['cause'] == 'precision_eaten'


def test_oracle_gain_with_flat_ap75_is_ranking():
    h1 = _summary()
    current = _summary(oracle=0.306)
    current['ap_grid']['0.80:all'] = 0.325
    report = diag.decide(h1, current)
    assert report['cause'] == 'ranking'


def test_headline_mismatch_is_reported_with_the_rule():
    h1 = _summary(ap=0.300)
    current = _summary(ap=0.300, oracle=0.306)
    current['ap_grid']['0.80:all'] = 0.303
    report = diag.compare_summaries(h1, current)
    assert report['headline_reproduced'] is False
    assert report['cause'] == 'ranking'
    assert '32.2' in report['sentence']
