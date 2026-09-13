"""Server data contracts: real builder, original VisDrone IDs, no hidden remap."""
import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from datasets.coco import build
from models.aqfcdetr.model import PostProcess


def fixture_args(tmp_path, split='train'):
    folder = tmp_path / f'VisDrone2019-DET-{split}' / 'images'
    folder.mkdir(parents=True)
    Image.new('RGB', (64, 64)).save(folder / 'sample.jpg')
    annotation_dir = tmp_path / 'annotations_coco'
    annotation_dir.mkdir(exist_ok=True)
    annotations = [dict(id=1, image_id=1, category_id=10, bbox=[4, 5, 8, 9], area=72, iscrowd=0),
                   dict(id=2, image_id=1, category_id=0, bbox=[0, 0, 30, 30], area=900, iscrowd=1, ignore=1)]
    (annotation_dir / f'VisDrone2019-DET_{split}_coco.json').write_text(json.dumps(dict(
        images=[dict(id=1, file_name='sample.jpg', width=64, height=64)],
        annotations=annotations, categories=[dict(id=i, name=str(i)) for i in range(1, 11)])))
    return SimpleNamespace(coco_path=str(tmp_path), dataset_file='visdrone', modelname='aqfcdetr',
        masks=False, fix_size=False, strong_aug=False, mosaic_p=0., copy_paste_p=0.,
        data_aug_scales=[64], data_aug_max_size=64,
        data_aug_scales2_resize=[64], data_aug_scales2_crop=[32, 64])


@pytest.mark.parametrize('split', ['train', 'val'])
def test_visdrone_real_builder(tmp_path, split):
    dataset = build(split, fixture_args(tmp_path, split))
    image, target = dataset[0]
    assert len(dataset) == 1
    assert set(target['labels'].tolist()) <= {10}
    if split == 'val':
        assert target['labels'].tolist() == [10]
    assert dataset.coco.getCatIds() == list(range(1, 11))


def test_visdrone_rejects_trainval(tmp_path):
    with pytest.raises(ValueError, match='VisDrone'):
        build('trainval', fixture_args(tmp_path))


def test_visdrone_postprocess_never_exports_unused_channels():
    post = PostProcess(valid_category_ids=range(1, 11))
    logits = torch.zeros(1, 2, 12)
    logits[:, :, 0] = 100
    logits[:, :, 11] = 100
    result = post(dict(pred_logits=logits, pred_boxes=torch.ones(1, 2, 4) * .5),
                  torch.tensor([[64, 64]]))[0]
    assert len(result['scores']) == 2
    assert set(result['labels'].tolist()) <= set(range(1, 11))


def test_visdrone_evaluation_is_not_aitod(tmp_path):
    from datasets.visdrone_eval import VisDroneCocoEvaluator
    ds = build('val', fixture_args(tmp_path, 'val'))
    evaluator = VisDroneCocoEvaluator(ds.coco, ('bbox',))
    evaluator.update({1: dict(boxes=torch.tensor([[4., 5., 12., 14.]]),
                              labels=torch.tensor([10]), scores=torch.tensor([.9]))})
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    assert evaluator.named_metrics()['AP'] == pytest.approx(1.)
    assert 'APvt' not in evaluator.named_metrics()
    assert evaluator.coco_eval['bbox'].params.maxDets == [1, 10, 100]
