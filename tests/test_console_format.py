"""Keep legacy console presentation without reverting corrected evaluation math."""
import torch

from aitodpycocotools.coco import COCO
from datasets.coco_eval import CocoEvaluator
from util.misc import MetricLogger
from engine import DISPLAY_KEYS


def test_original_coco_table_keeps_correct_ap25(capsys):
    gt = COCO()
    gt.dataset = dict(images=[dict(id=1, width=100, height=100)],
                      categories=[dict(id=1, name='object')],
                      annotations=[dict(id=1, image_id=1, category_id=1,
                                        bbox=[10, 10, 10, 10], area=100, iscrowd=0)])
    gt.createIndex()
    evaluator = CocoEvaluator(gt, ['bbox'])
    evaluator.update({1: dict(boxes=torch.tensor([[15., 10., 25., 20.]]),
                              scores=torch.tensor([.9]), labels=torch.tensor([1]))})
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    capsys.readouterr()
    evaluator.summarize()
    printed = capsys.readouterr().out
    assert 'Average Precision' in printed and 'Average Recall' in printed
    assert 'Optimal LRP' in printed
    rows = [line for line in printed.splitlines() if 'Average Precision' in line]
    assert next(line for line in rows if 'IoU=0.25' in line).endswith('= 1.000')
    assert next(line for line in rows if 'IoU=0.50:0.95' in line).endswith('= 0.000')
    assert evaluator.coco_eval['bbox'].stats[0] == 0
    assert evaluator.coco_eval['bbox'].stats[1] > .99
    assert evaluator.named_metrics()['APs'] is None


def test_original_iteration_format_includes_auxiliary_and_unscaled_losses(capsys):
    meter = MetricLogger()
    meter.update(loss=3., loss_ce_dn_0_unscaled=2., loss_density_map_unscaled=1.)
    for _ in meter.log_every([0], 1, 'Test:', display_keys=DISPLAY_KEYS):
        pass
    printed = capsys.readouterr().out
    assert 'loss_ce_dn_0_unscaled:' in printed
    assert 'loss_density_map_unscaled:' in printed
    assert 'loss_ce_dn_0_unscaled:' in meter.format_meters(DISPLAY_KEYS)
