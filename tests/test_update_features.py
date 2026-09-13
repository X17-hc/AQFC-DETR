import argparse
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
import torch

from models.aqfcdetr.query_allocator import AdaptiveQueryBudgetAllocator, QueryBudgetLoss
from models.aqfcdetr.proposal_selection import select_proposal_indices
from models.aqfcdetr.spatial_selection import capped_quotas
from util.experiment import unique_output, variant_signature
from util.prediction_export import prediction_records, category_mapping, save_predictions
from util.config_validation import validate_config
from util.slconfig import SLConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('chunk', [1, 7, 512])
def test_vectorized_targets_match_reference(device, chunk):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    generator = torch.Generator().manual_seed(42)
    boxes = torch.rand(41, 4, generator=generator).to(device)
    boxes = torch.cat([boxes, boxes[:5], boxes.new_tensor([[0, 0, .01, .01], [1, 1, 1, 1]])])
    targets = [dict(boxes=boxes), dict(boxes=boxes[:0])]
    mask = torch.ones(2, 1, 17, 29, device=device, dtype=torch.bool)
    mask[0, :, 15:, :] = False
    mask[0, :, :, 24:] = False
    kwargs = dict(return_valid_mask=True, spatial_valid_mask=mask)
    reference, refmask = QueryBudgetLoss.build_density_targets(targets, (17, 29), device, **kwargs)
    actual, actualmask = QueryBudgetLoss.build_density_targets(
        targets, (17, 29), device, backend='vectorized', chunk_size=chunk, **kwargs)
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-6)
    assert torch.equal(refmask, actualmask)
    assert not actual[1].any()


def test_light_allocator_backward_and_roundtrip():
    model = AdaptiveQueryBudgetAllocator(encoder_type='light_dw').train()
    feature = torch.randn(2, 256, 6, 9, requires_grad=True)
    out = model(feature, real_counts=torch.tensor([3., 500.]), epoch=4)
    loss = QueryBudgetLoss(density_target_backend='vectorized')(out,
        dict(real_counts=torch.tensor([3., 500.]), targets=[{'boxes': torch.rand(3, 4)},
                                                            {'boxes': torch.rand(7, 4)}]))
    loss['loss_allocator_total'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.light_density_encoder.parameters())
    assert not any(k.startswith('density_encoder.') for k in model.state_dict())
    other = AdaptiveQueryBudgetAllocator(encoder_type='light_dw')
    other.load_state_dict(model.state_dict(), strict=True)
    model.eval(); other.eval()
    with torch.no_grad():
        expected = model(feature)['density_prior']
        actual = other(feature.flatten(2).transpose(1, 2), torch.tensor([[6, 9]]))['density_prior']
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('density_value', [0., .5, float('nan')])
def test_spatial_selection_valid_deterministic(density_value):
    logits = torch.zeros(2, 32, 3)
    density = torch.full((2, 32), density_value)
    mask = torch.zeros(2, 32, dtype=torch.bool)
    mask[1, 20:] = True
    reports = []
    kwargs = dict(mode='spatial', spatial_shapes=[[4, 8]], spatial_grid_size=(2, 4),
                  diagnostics=reports)
    indices = select_proposal_indices(logits, density, mask, 16, **kwargs)
    again = select_proposal_indices(logits, density, mask, 16, **kwargs)
    assert torch.equal(indices, again)
    assert not mask.gather(1, indices).any()
    assert all(len(row.unique()) == 16 for row in indices)
    if density_value == 0 or density_value != density_value:
        assert torch.equal(indices[0], torch.arange(16))
        assert reports[0]['fallback'] == 1


def test_capped_largest_remainder():
    assert capped_quotas([.9, .1], [1, 9], 8) == [1, 7]
    assert capped_quotas([1, 1, 1], [9, 9, 9], 5) == [2, 2, 1]


def test_unique_output_and_resume(tmp_path):
    args = argparse.Namespace(output_dir=str(tmp_path), unique_output_dir=True, resume='')
    unique_output(args)
    first = args.output_dir
    args.output_dir = str(tmp_path)
    unique_output(args)
    assert first != args.output_dir
    args.resume = 'checkpoint.pth'
    with pytest.raises(ValueError, match='resume'):
        unique_output(args)


def test_export_preserves_ids_and_empty_images(tmp_path):
    categories = [dict(id=7, name='wind'), dict(id=0, name='air')]
    assert category_mapping(categories) == {0: 0, 7: 1}
    result = dict(boxes=torch.tensor([[1., 2., 5., 8.]]), scores=torch.tensor([.2]), labels=torch.tensor([7]))
    rows = prediction_records(99, result, categories)
    assert rows[0]['category_id'] == 7 and rows[0]['bbox'] == [1., 2., 4., 6.]
    save_predictions(tmp_path, rows, [dict(image_id=99), dict(image_id=100)], categories)
    assert json.loads((tmp_path / 'metadata.json').read_text())['image_ids'] == [99, 100]
    result['labels'] = torch.tensor([8])
    with pytest.raises(ValueError, match='Unknown'):
        prediction_records(99, result, categories)


def test_all_experiment_configs_and_pycharm_xml():
    paths = list((ROOT / 'configs/experiments').glob('*/*.py'))
    assert len(paths) == 12
    for path in paths:
        config = SLConfig.fromfile(str(path))._cfg_dict.to_dict()
        validate_config(config)
        if 'research' in path.parts:
            assert config['train_split'] == 'train' and config['eval_split'] == 'val'
    for name in ('UpdateSmoke', 'BaselineEpoch', 'LightEpoch', 'SpatialEpoch', 'UpdateEpoch'):
        config = ET.parse(ROOT / '.run' / f'AQFC-DETR_{name}.run.xml').getroot().find('configuration')
        options = {x.attrib['name']: x.attrib.get('value') for x in config.findall('option')}
        # JetBrains may serialize the same Windows interpreter using a project macro.
        # Normalize as Windows syntax even when the tests run on the Linux server.
        import ntpath
        sdk = options['SDK_HOME'].replace('$PROJECT_DIR$', 'D:/PythonProject/AQFC-DETR')
        assert ntpath.normpath(sdk) == ntpath.normpath('D:/venv/AQFC-DETR/Scripts/python.exe')
        assert '--unique-output-dir' in options['PARAMETERS']


def test_variant_signature_changes():
    standard = variant_signature({})
    assert standard != variant_signature({'allocator_encoder_type': 'light_dw'})


def test_update_config_typo_is_not_silently_ignored():
    config = SLConfig.fromfile(str(ROOT/'configs/experiments/local8gb/baseline.py'))._cfg_dict.to_dict()
    config['allocator_encodre_type'] = 'light_dw'
    with pytest.raises(ValueError, match='Unknown update'):
        validate_config(config)


def test_fast_array_gate():
    import numpy as np
    from datasets.checked_fast_eval import compare_arrays
    assert compare_arrays(np.array([-1., .5]), np.array([-1., .5])) == 0
    with pytest.raises(ValueError):
        compare_arrays(np.array([-1., .5]), np.array([0., .5]))
