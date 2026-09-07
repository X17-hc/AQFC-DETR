import numpy as np
import torch

from aitodpycocotools.coco import COCO
from datasets.coco_eval import CocoEvaluator
from util.runtime import LimitedLoader, can_select_best
from types import SimpleNamespace
import pytest
from util.config_validation import validate_config
from util.slconfig import SLConfig


def test_ap25_is_computed_without_changing_standard_ap():
    gt = COCO()
    gt.dataset = dict(images=[dict(id=1, width=100, height=100)],
                      categories=[dict(id=1, name='object')],
                      annotations=[dict(id=1, image_id=1, category_id=1,
                                        bbox=[10, 10, 10, 10], area=100, iscrowd=0)])
    gt.createIndex()
    evaluator = CocoEvaluator(gt, ['bbox'])
    # IoU=1/3: AP25=1, while standard AP50:95 stays zero.
    evaluator.update({1: dict(boxes=torch.tensor([[15.,10.,25.,20.]]),
                              scores=torch.tensor([.9]), labels=torch.tensor([1]))})
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.coco_eval['bbox'].stats
    assert stats[0] == 0
    assert stats[1] > .99
    assert np.isclose(evaluator.coco_eval['bbox'].params.iouThrs[0], .5)
    assert evaluator.named_metrics('bbox')['APs'] is None


def test_limited_loader_does_not_fetch_extra_batch():
    seen = []
    class Loader:
        def __len__(self): return 100
        def __iter__(self):
            for i in range(100):
                seen.append(i)
                yield i
    loader = LimitedLoader(Loader(), 3)
    assert len(loader) == 3
    assert list(loader) == [0, 1, 2]
    assert seen == [0, 1, 2]


def test_test_split_and_partial_eval_cannot_select_best():
    args = SimpleNamespace(eval_split='test', debug=False, max_eval_steps=0, max_train_steps=0)
    assert not can_select_best(args)
    args.eval_split = 'val'
    assert can_select_best(args)
    args.max_eval_steps = 2
    assert not can_select_best(args)


@pytest.mark.parametrize('changes', [
    {'count_loss_weight': float('nan')}, {'query_budget_levels': [0,500,900,1500]},
    {'train_split': 'trainval', 'eval_split': 'val'}, {'allocator_teacher_epochs': 8},
    {'allocator_use_boundary_ema': 'False'}])
def test_invalid_config_fails_early(changes):
    cfg = SLConfig.fromfile('configs/aitodv2/aqfc_r50_5scale_local8gb.py')._cfg_dict.to_dict()
    cfg.update(changes)
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_default_trainval_cannot_overlap_validation():
    cfg = SLConfig.fromfile('configs/aitodv2/aqfc_r50_5scale_local8gb.py')._cfg_dict.to_dict()
    cfg.pop('train_split')
    cfg['eval_split'] = 'val'
    with pytest.raises(ValueError, match='includes validation'):
        validate_config(cfg)
