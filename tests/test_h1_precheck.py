import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from precheck_h1_candidates import first_n_ids, ids_sha256, risk_pause
from benchmark_h1_workers import MATRIX, commands
from profile_h1_mosaic import commands as profile_commands
from util.runtime import coco_places_agree, compute_eval_criterion, training_eval_weights


def test_first_n_ids(tmp_path):
    path = tmp_path / 'ids.json'
    path.write_text(json.dumps(list(range(1000))))
    assert first_n_ids(path, 32) == list(range(32))


def test_first_n_ids_short(tmp_path):
    path = tmp_path / 'short.json'
    path.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ValueError):
        first_n_ids(path, 32)


def test_risk_pause_one_point():
    base = dict(AP=0.360, APvt=0.232)
    ok, _ = risk_pause(base, dict(AP=0.358, APvt=0.231))
    assert ok is False
    pause, delta = risk_pause(base, dict(AP=0.348, APvt=0.232))
    assert pause is True
    assert delta['d_ap_pp'] == pytest.approx(1.2)


def test_worker_matrix_keeps_default_off():
    assert MATRIX[0] == dict(num_workers=0, persistent_workers=False)
    rows = commands('cfg.py', 'data', 'w.pth', 'out')
    assert all('--num_workers' in row['argv'] for row in rows)
    assert any('--persistent-workers' in row['argv'] for row in rows)
    assert '--persistent-workers' not in rows[0]['argv']


def test_risk_pause_vt_coverage():
    base = dict(AP=0.360, APvt=0.232, vt_cover_50=0.80)
    ok, _ = risk_pause(base, dict(AP=0.360, APvt=0.232, vt_cover_50=0.79))
    assert ok is False
    pause, delta = risk_pause(base, dict(AP=0.360, APvt=0.232, vt_cover_50=0.77))
    assert pause is True
    assert delta['d_cover_pp'] == pytest.approx(3.0)


def test_ids_sha256_is_stable():
    assert ids_sha256([1, 2, 3]) == ids_sha256([1, 2, 3])
    assert ids_sha256([1, 2, 3]) != ids_sha256([1, 2, 4])


def test_h1_mosaic_profile_keeps_mosaic_and_bounds():
    rows = profile_commands('cfg.py', 'data', 'w.pth', 'out')
    assert rows['chrome_trace_50'][-1] == '50'
    assert '--max-train-steps' in rows['mosaic_250_steps']
    assert rows['mosaic_250_steps'][rows['mosaic_250_steps'].index('--max-train-steps') + 1] == '250'


def test_boxes_only_and_ema_eval_targets():
    assert compute_eval_criterion(SimpleNamespace()) is True
    assert compute_eval_criterion(SimpleNamespace(eval_boxes_only=True)) is False
    assert training_eval_weights(SimpleNamespace()) == ('model',)
    assert training_eval_weights(SimpleNamespace(use_ema=True)) == ('model', 'ema')
    assert training_eval_weights(SimpleNamespace(use_ema=True, eval_ema_only=True)) == ('ema',)
    with pytest.raises(ValueError):
        training_eval_weights(SimpleNamespace(eval_ema_only=True))


def test_coco_places_agree_three_decimals():
    left = dict(AP=0.3224, AP50=0.6821, AP75=0.2630, APvt=0.1562)
    right = dict(AP=0.3221, AP50=0.6824, AP75=0.2634, APvt=0.1564)
    assert coco_places_agree(left, right) is True
    assert coco_places_agree(left, dict(AP=0.3214, AP50=0.6821, AP75=0.2630, APvt=0.1562)) is False


def test_h1_run_config_names_are_unique():
    import xml.etree.ElementTree as ET
    root = Path(__file__).resolve().parents[1] / '.run'
    names = []
    for path in root.glob('*.run.xml'):
        config = ET.parse(path).getroot().find('configuration')
        names.append(config.get('name'))
    assert len(names) == len(set(names))
    assert 'AQFC-DETR H1候选预检32图' in names
    assert 'AQFC-DETR H1探针EMA3轮' in names
