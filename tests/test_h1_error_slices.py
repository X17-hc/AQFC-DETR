import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from summarize_size_class_ap import (
    H1_CLASSES,
    compare_runs,
    hypotheses_from_evidence,
    load_class_metrics,
    summarize,
)


def _class_metrics(tmp_path, name, ap_all, ap_vt):
    rows = []
    for i, cls in enumerate(H1_CLASSES):
        rows.append(dict(
            category_id=i, name=cls, images_with_gt=10, gt_annotations=20, noncrowd_gt=20,
            areas={
                'all': dict(AP=ap_all[i], AR=0.4),
                'verytiny': dict(AP=ap_vt[i], AR=0.2),
            }))
    path = tmp_path / f'{name}.json'
    path.write_text(json.dumps(dict(image_ids=list(range(5)), classes=rows)), encoding='utf-8')
    return path


def test_load_and_summarize_eight_classes(tmp_path):
    ap = [0.4, 0.3, 0.2, 0.25, 0.1, 0.35, 0.15, 0.05]
    vt = [0.2, 0.1, 0.05, 0.08, 0.02, 0.18, 0.12, 0.01]
    path = _class_metrics(tmp_path, 'h1', ap, vt)
    loaded = load_class_metrics(path)
    summary = summarize(loaded)
    assert [row['name'] for row in summary['classes']] == list(H1_CLASSES)
    assert summary['classes'][5]['name'] == 'vehicle'
    assert summary['classes'][6]['name'] == 'person'
    assert summary['mean_AP'] == pytest.approx(sum(ap) / 8)
    assert summary['mean_APvt'] == pytest.approx(sum(vt) / 8)


def test_compare_marks_vehicle_person_vt(tmp_path):
    base = load_class_metrics(_class_metrics(tmp_path, 'h1', [0.3] * 8, [0.10] * 8))
    cand = load_class_metrics(_class_metrics(
        tmp_path, 'd1', [0.3] * 8, [0.10, 0.10, 0.10, 0.10, 0.10, 0.16, 0.10, 0.10]))
    delta = compare_runs(base, cand)
    vehicle = next(row for row in delta['classes'] if row['name'] == 'vehicle')
    assert vehicle['dAPvt'] == pytest.approx(0.06)


def test_hypotheses_are_falsifiable():
    rows = hypotheses_from_evidence(dict(
        h1_full=dict(AP=0.322, AP50=0.682, AP75=0.263, APvt=0.156),
        d0_full=dict(AP=0.322, AP50=0.677, AP75=0.265, APvt=0.160),
        d1_full=dict(AP=0.324, AP50=0.678, AP75=0.268, APvt=0.161, APs=0.377),
    ))
    assert len(rows) == 3
    for row in rows:
        assert row['falsify']
        assert row['probe']
