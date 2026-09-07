"""Second-round audit: state integrity, augmentation geometry and safe configuration."""
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from util.utils import ModelEma
from util.checkpoint import load_native_resume
from util.config_validation import validate_config
from util.slconfig import SLConfig
from datasets.coco import CocoDetection
from datasets.mosaic import MosaicDetection
from datasets.copy_paste import CopyPasteSmallObjects


def test_model_ema_copies_integer_buffers():
    model = torch.nn.BatchNorm1d(2)
    ema = ModelEma(model, decay=.9)
    model.num_batches_tracked.fill_(7)
    ema.update(model)
    assert ema.module.num_batches_tracked.item() == 7


def test_model_ema_averages_parameters_but_copies_running_state():
    model = torch.nn.BatchNorm1d(2)
    ema = ModelEma(model, decay=.9)
    with torch.no_grad():
        model.weight.fill_(3)
        model.running_mean.fill_(5)
    ema.update(model)
    torch.testing.assert_close(ema.module.weight, torch.full((2,), 1.2))
    torch.testing.assert_close(ema.module.running_mean, model.running_mean)
    ema.set(model)
    torch.testing.assert_close(ema.module.weight, model.weight)


def test_empty_dataset_sampling_is_bounded():
    dataset = CocoDetection.__new__(CocoDetection)
    dataset.ids = [0]
    dataset.mosaic = dataset.copy_paste = dataset._transforms = dataset.aux_target_hacks = None
    dataset.filter_empty_gt = True
    dataset.copy_paste_p = 0.
    calls = []
    def raw(index):
        calls.append(index)
        return Image.new('RGB',(8,8)), {'boxes':torch.empty(0,4), 'labels':torch.empty(0,dtype=torch.long)}
    dataset._get_raw_item = raw
    with pytest.raises(ValueError, match='empty|non-empty'):
        dataset[0]
    assert len(calls) <= 33


def test_corrupt_image_is_not_replaced_by_another_image():
    dataset = CocoDetection.__new__(CocoDetection)
    dataset.ids = [12,13]
    with patch('torchvision.datasets.CocoDetection.__getitem__', side_effect=OSError('corrupt')) as read:
        with pytest.raises(OSError):
            dataset._get_raw_item(0)
    assert read.call_count == 1


def test_mosaic_uses_actual_rounded_resize_geometry():
    # 13x9 -> tile 5x5 -> actual resized 5x3. Y scale is 3/9, not 5/13.
    target = dict(boxes=torch.tensor([[0.,0.,13.,9.]]), labels=torch.tensor([1]),
                  image_id=torch.tensor([0]), area=torch.tensor([117.]), iscrowd=torch.tensor([0]))
    anchor = dict(target, boxes=torch.empty(0,4), labels=torch.empty(0,dtype=torch.long),
                  area=torch.empty(0), iscrowd=torch.empty(0,dtype=torch.long))
    items = [(Image.new('RGB',(10,10)),anchor)] + [(Image.new('RGB',(13,9)),target)]*3
    with patch('datasets.mosaic.random.uniform',return_value=.5):
        _, result = MosaicDetection()(items)
    torch.testing.assert_close(result['boxes'][0], torch.tensor([5.,1.,10.,4.]))
    torch.testing.assert_close(result['area'], torch.full((3,),15.))


def test_copy_paste_does_not_cover_existing_tiny_object():
    augmentation = CopyPasteSmallObjects()
    # IoU=.01 but the small object would be completely overwritten.
    overlap = augmentation._overlap_with_existing(torch.tensor([0.,0.,10.,10.]),
                                               torch.tensor([[4.,4.,5.,5.]]))
    assert overlap > augmentation.iou_threshold


def test_resume_requires_epoch_for_optimizer_state(tmp_path):
    model = torch.nn.Linear(1,1)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path/'no_epoch.pth'
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict()),path)
    with pytest.raises(KeyError,match='epoch'):
        load_native_resume(model,path,optimizer=optimizer)


@pytest.mark.parametrize('override', [dict(force_query_budget=-1), dict(force_query_budget=100),
    dict(two_stage_type='no'),dict(two_stage_pat_embed=2),dict(two_stage_keep_all_tokens=True),
    dict(mosaic_p=.8,copy_paste_p=.5),dict(calibrator_spatial_alphas=[0,0,0,0,float('nan')])])
def test_unsafe_configuration_is_rejected(override):
    cfg=SLConfig.fromfile('configs/aitodv2/aqfc_r50_5scale_local8gb.py')._cfg_dict.to_dict()
    cfg.update(override)
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_eval_loader_does_not_load_training_data(monkeypatch):
    import main
    splits = []
    def build(image_set, args):
        splits.append(image_set)
        assert image_set == 'test', 'Evaluation must not require training data'
        return torch.utils.data.TensorDataset(torch.ones(2, 1))
    monkeypatch.setattr(main, 'build_dataset', build)
    args = SimpleNamespace(eval=True, train_split='trainval', eval_split='test',
                           distributed=False, batch_size=1, num_workers=0)
    train, val, train_loader, val_loader, sampler = main.build_data_loaders(args)
    assert train is train_loader is sampler is None
    assert len(val_loader) == 2 and splits == ['test']


def test_distributed_step_statistics_do_not_mix_local_and_global_counts():
    from util.runtime import optimizer_step_statistics
    result = optimizer_step_statistics(10, 9, SimpleNamespace(total=18, count=20))
    assert result == dict(train_iterations=10, optimizer_steps=9, amp_skipped_steps=1,
                          global_optimizer_steps=18, global_train_iterations=20)


def test_checkpoint_restores_random_generators(tmp_path):
    from util.checkpoint import capture_rng_state
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path/'rng.pth'
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=2,
                    rng_states=[capture_rng_state()]), path)
    expected = (random.random(), np.random.rand(), torch.rand(2))
    for _ in range(10):
        random.random(); np.random.rand(); torch.rand(2)
    assert load_native_resume(model, path, optimizer=optimizer) == 3
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(2), expected[2], rtol=0, atol=0)


def test_eval_ema_is_explicit_and_requires_native_resume():
    from main import get_args_parser, resolve_launch_defaults
    parser = get_args_parser()
    assert parser.parse_args(['--eval', '--eval-ema']).eval_ema
    with pytest.raises(ValueError, match='eval-ema'):
        resolve_launch_defaults(parser.parse_args(['--eval-ema']))
    args = resolve_launch_defaults(parser.parse_args(['--eval', '--eval-ema', '--resume', 'model.pth']))
    assert args.pretrain_model_path == ''


def test_density_supervision_uses_encoder_padding_mask():
    from models.aqfcdetr.query_allocator import QueryBudgetLoss
    targets = [dict(size=torch.tensor([4, 4]), boxes=torch.tensor([[.75,.75,.1,.1]])),
               dict(size=torch.tensor([9, 9]), boxes=torch.empty(0, 4))]
    # Nearest-neighbor mask resize 9 -> 3 samples pixels 0,3,6:
    # the 4x4 image has TWO valid rows/columns, although round(4/9*3)==1.
    actual = torch.zeros(2, 1, 3, 3, dtype=torch.bool)
    actual[0, :, :2, :2] = True
    actual[1] = True
    heatmap, valid = QueryBudgetLoss.build_density_targets(
        targets, (3, 3), 'cpu', return_valid_mask=True, spatial_valid_mask=actual)
    assert torch.equal(valid, actual)
    assert heatmap[0, 0, 1, 1] == 1
    assert not heatmap[0, 0, 2].any()
