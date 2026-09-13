import argparse
import json
import numpy as np
import pytest
import torch
from util.error_analysis import analyze, density_bucket
from util.checkpoint import load_native_resume
from util.experiment import variant_signature


def test_error_analysis_duplicate_empty_and_confusion():
    gt = dict(categories=[dict(id=0,name='a'),dict(id=7,name='b')],
              annotations=[dict(image_id=1,category_id=0,bbox=[0,0,4,4])])
    predictions = [dict(image_id=1,category_id=0,bbox=[0,0,4,4],score=.9),
                   dict(image_id=1,category_id=0,bbox=[0,0,4,4],score=.8)]
    result = analyze(predictions, gt, [1,2])
    assert result['category_counts']['0']['tp'] == 1
    assert result['category_counts']['0']['duplicate'] == 1
    assert result['category_counts']['7']['support_status'] == 'no_gt'
    assert result['density_counts']['0']['images'] == 1
    assert density_bucket(901) == '>900'


def test_signature_resume_refuses_cross_variant_before_loading(tmp_path):
    model = torch.nn.Linear(1,1)
    optimizer = torch.optim.AdamW(model.parameters())
    source = tmp_path/'checkpoint.pth'
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=0,
                    variant_signature=variant_signature({})), source)
    with pytest.raises(ValueError, match='signature'):
        load_native_resume(model, source, optimizer=optimizer,
                           expected_args=dict(allocator_encoder_type='light_dw'))
    assert load_native_resume(model, source, optimizer=optimizer, expected_args={}) == 1


def test_supervision_slicer_coordinates_and_merge():
    sv = pytest.importorskip('supervision')
    calls = []
    def callback(tile):
        calls.append(tile.shape)
        return sv.Detections(xyxy=np.array([[1.,2.,5.,6.]]), confidence=np.array([.9]),
                             class_id=np.array([0]))
    slicer = sv.InferenceSlicer(callback, slice_wh=16, overlap_wh=4,
                               overlap_filter=sv.OverlapFilter.NONE, thread_workers=1)
    result = slicer(np.zeros((24,28,3), dtype=np.uint8))
    assert len(calls) > 1 and len(result) == len(calls)
    assert result.xyxy[:,0].max() > 1 and result.xyxy[:,1].max() > 2
    duplicate = sv.Detections.merge([result, result])
    merged = duplicate.with_nms(threshold=.5, class_agnostic=False, overlap_metric=sv.OverlapMetric.IOU)
    assert len(merged) == len(result)


def test_optional_supervision_not_imported_by_model():
    import subprocess, sys
    command = "import sys; import models.aqfcdetr.model; assert 'supervision' not in sys.modules"
    result = subprocess.run([sys.executable, '-c', command], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_no_fast_backend_silent_fallback():
    import importlib.util
    if importlib.util.find_spec('faster_coco_eval_aitod') is None:
        from datasets.checked_fast_eval import require_fast
        with pytest.raises(RuntimeError, match='unavailable'):
            require_fast()


def test_default_allocator_matches_preupdate_snapshot():
    import hashlib
    from pathlib import Path
    from models.aqfcdetr.query_allocator import AdaptiveQueryBudgetAllocator
    # Exact git-show snapshot from b47369d24ccba813e6310362297f783f5db5637f.
    # Package the immutable fixture so deployment does not require a .git directory.
    source = Path(__file__).with_name('reference_allocator_b47369d.py').read_text(encoding='utf-8')
    assert hashlib.sha256(source.encode('utf-8')).hexdigest() == 'd31d42770c9b383687484bd2c063fea78001fc9bcf3e5ea70c00da7c9ade1bff'
    namespace = {'__name__': 'models.aqfcdetr._reference_allocator'}
    exec(compile(source, '<preupdate allocator>', 'exec'), namespace)
    torch.manual_seed(42)
    reference = namespace['AdaptiveQueryBudgetAllocator']().eval()
    torch.manual_seed(42)
    current = AdaptiveQueryBudgetAllocator().eval()
    assert reference.state_dict().keys() == current.state_dict().keys()
    for key, value in reference.state_dict().items():
        assert torch.equal(value, current.state_dict()[key]), key
    with torch.no_grad():
        image = torch.rand(1,256,5,7)
        expected, actual = reference(image), current(image)
    for key in ('density_prior','predicted_count','query_counts','boundaries'):
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)


def test_inactive_profile_has_no_hooks_or_trace(tmp_path):
    from util.profiling import training_profile, region
    model = torch.nn.Linear(1,1)
    with training_profile('', model, None), region('test'):
        model(torch.ones(1,1))
    assert not model._forward_hooks and not list(tmp_path.iterdir())


def test_active_profile_writes_trace_and_removes_hooks(tmp_path):
    from types import SimpleNamespace
    from util.profiling import training_profile
    backbone = torch.nn.Linear(2,2)
    transformer = SimpleNamespace(encoder=torch.nn.Linear(2,2), decoder=torch.nn.Linear(2,2),
        query_allocator=None, feature_calibrator=None, density_pyramid=None)
    model = SimpleNamespace(backbone=backbone, transformer=transformer)
    criterion = torch.nn.Sequential(torch.nn.Linear(2,1))
    criterion.matcher = torch.nn.Identity()
    path = tmp_path/'trace.json'
    with training_profile(path, model, criterion):
        value = backbone(torch.ones(1,2))
        criterion(transformer.decoder(transformer.encoder(value))).sum().backward()
    assert path.is_file() and path.with_suffix('.summary.json').is_file()
    assert not backbone._forward_hooks
