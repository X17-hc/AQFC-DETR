"""Regression contracts for the weight-preserving incremental update."""
from types import SimpleNamespace

import pytest
import torch

from util import box_ops
from util.misc import NestedTensor, MetricLogger
from models.aqfcdetr.model import SetCriterion


def criterion(**options):
    return SetCriterion(9, None, {}, .25, ['labels', 'boxes'], **options)


def sample():
    logits = torch.randn(1, 3, 9, requires_grad=True)
    boxes = torch.tensor([[[.5,.5,.2,.2], [.4,.4,.1,.1], [.8,.8,.1,.1]]], requires_grad=True)
    out = dict(pred_logits=logits, pred_boxes=boxes, query_valid_mask=torch.tensor([[True, True, False]]))
    gt = [dict(labels=torch.tensor([2]), boxes=torch.tensor([[.5,.5,.2,.2]]))]
    idx = [(torch.tensor([0]), torch.tensor([0]))]
    return out, gt, idx


@pytest.mark.parametrize('n', [0, 1, 23])
def test_aligned_boxes_match_pairwise_value_and_gradient(n):
    torch.manual_seed(42)
    a = torch.rand(n,4, requires_grad=True)
    b = torch.rand(n,4)
    xy_a, xy_b = box_ops.box_cxcywh_to_xyxy(a), box_ops.box_cxcywh_to_xyxy(b)
    for actual, expected in [
        (box_ops.aligned_generalized_box_iou(xy_a, xy_b), box_ops.generalized_box_iou(xy_a, xy_b).diag()),
        (box_ops.aligned_box_nwd(a,b,.03), box_ops.box_nwd(a,b,.03).diag())]:
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        ga = torch.autograd.grad(actual.sum(), a, retain_graph=True)[0]
        ge = torch.autograd.grad(expected.sum(), a, retain_graph=True)[0]
        torch.testing.assert_close(ga, ge, atol=1e-5, rtol=1e-4)


def test_nested_tensor_nonblocking_and_mask_none():
    for mask in [None, torch.zeros(1,3,4,dtype=torch.bool)]:
        sample = NestedTensor(torch.randn(1,3,3,4),mask)
        moved = sample.to('cpu', non_blocking=True)
        torch.testing.assert_close(moved.tensors,sample.tensors)
        assert (moved.mask is None) == (mask is None)


def test_quality_zero_is_original_and_dn_is_unchanged():
    out,gt,idx = sample()
    old = criterion().loss_labels(out,gt,idx,1)['loss_ce']
    new = criterion(classification_loss_type='quality_blend')
    torch.testing.assert_close(new.loss_labels(out,gt,idx,1)['loss_ce'],old,atol=0,rtol=0)
    new.quality_lambda = .25
    torch.testing.assert_close(new.loss_labels(out,gt,idx,1,quality_eligible=False)['loss_ce'],old,atol=0,rtol=0)


def test_quality_mask_and_stop_gradient():
    out,gt,idx = sample()
    obj=criterion(classification_loss_type='quality_blend')
    obj.quality_lambda=.25
    loss=obj.loss_labels(out,gt,idx,1)['loss_ce']
    loss.backward()
    assert torch.isfinite(loss)
    assert out['pred_logits'].grad[0,2].eq(0).all()
    assert out['pred_boxes'].grad is None


def test_scalar_logging_preserves_values():
    log=MetricLogger(batched_transfer=True)
    log.update(a=torch.tensor(2.),b=torch.tensor(3.),c=4.)
    assert [log.meters[k].global_avg for k in ('a','b','c')] == [2.,3.,4.]


def test_phase_offset_and_quality_progress():
    from util.incremental import training_phase, quality_progress
    args=SimpleNamespace(epochs=3,training_phase_epoch_offset=11,training_phase_total_epochs=24,
                         quality_blend_warmup_epochs=1,quality_blend_max=.25)
    assert training_phase(args,0) == (11,24)
    assert quality_progress(args,50,100) == .125
    assert quality_progress(args,100,100) == .25


def test_degenerate_iou_is_finite():
    # A legal zero-area prediction pair must not introduce NaNs in diagnostics.
    b=torch.zeros(2,4,requires_grad=True)
    result=box_ops.generalized_box_iou(b,b)
    assert torch.isfinite(result).all()
    result.sum().backward()
    assert torch.isfinite(b.grad).all()


def test_grouped_scatter_preserves_decoder_precision():
    from models.aqfcdetr.transformer import DeformableTransformer
    class FakeDecoder:
        def __call__(self, **kw):
            assert kw.get('tgt_key_padding_mask') is not None
            assert not kw['tgt_key_padding_mask'].any()
            target=kw['tgt'].transpose(0,1).float()+.0003
            ref=kw['refpoints_unsigmoid'].transpose(0,1).float().sigmoid()
            return [target], [ref]
    fake=SimpleNamespace(embed_init_tgt=False,decoder=FakeDecoder(),
        select_proposal_indices=lambda cls, density, padding, count, **kw:
            torch.arange(count).expand(cls.shape[0],-1))
    memory=torch.ones(2,4,8,dtype=torch.float16)
    proposals=torch.zeros(2,4,4)
    result=DeformableTransformer._decode_grouped_inference(fake,memory,proposals,
        torch.ones(2,4,9),proposals,torch.ones(2,4),torch.zeros(2,4,dtype=torch.bool),
        memory,memory,torch.tensor([0]),torch.tensor([[2,2]]),torch.ones(2,1,2),torch.tensor([2,3]))
    assert result[0][0].dtype == torch.float32
    torch.testing.assert_close(result[0][0][0,:2],torch.full((2,8),1.0003),atol=0,rtol=0)


def test_aitod_builder_excludes_unused_ninth_channel():
    from unittest.mock import patch
    import argparse
    import torchvision
    from main import build_model_main
    from util.slconfig import SLConfig
    config=SLConfig.fromfile('configs/experiments/local8gb/light_spatial.py')._cfg_dict.to_dict()
    config.update(device='cpu',distributed=False)
    original=torchvision.models.resnet50
    def no_download(*a,**kw):
        kw['weights']=None
        return original(*a,**kw)
    with patch('torchvision.models.resnet50',side_effect=no_download):
        model,_,post=build_model_main(argparse.Namespace(**config))
    assert post['bbox'].valid_category_ids == tuple(range(8))


@pytest.mark.parametrize('variant',['p1','p2'])
def test_actual_training_engine_synthetic_update(variant):
    if not torch.cuda.is_available(): pytest.skip('CUDA training engine')
    from unittest.mock import patch
    import argparse
    import torchvision
    from main import build_model_main
    from util.slconfig import SLConfig
    from engine import train_one_epoch, MosaicPScheduler
    from util.misc import nested_tensor_from_tensor_list
    cfg=SLConfig.fromfile(f'configs/incremental_v2/{variant}.py')._cfg_dict.to_dict()
    cfg.update(device='cuda',distributed=False,rank=0,amp=True,max_train_steps=1,max_eval_steps=0,
               profile_trace='',num_workers=0)
    args=argparse.Namespace(**cfg)
    original=torchvision.models.resnet50
    def no_download(*a,**kw):
        kw['weights']=None
        return original(*a,**kw)
    with patch('torchvision.models.resnet50',side_effect=no_download):
        model,crit,_=build_model_main(args)
    model.cuda(); crit.cuda()
    image=torch.rand(3,128,128)
    targets=[dict(labels=torch.tensor([0]),boxes=torch.tensor([[.4,.4,.05,.06]]),size=torch.tensor([128,128]))]
    class Loader:
        dataset=SimpleNamespace(mosaic=SimpleNamespace(p=0.))
        def __len__(self): return 1
        def __iter__(self): yield nested_tensor_from_tensor_list([image]),targets
    loader=Loader()
    optim=torch.optim.AdamW(model.parameters(),lr=1e-5)
    schedule=MosaicPScheduler(peak_p=.25,warmup_end=4,decay_start=20,total_epochs=24)
    before=next(model.parameters()).detach().clone()
    stats=train_one_epoch(model,crit,loader,optim,torch.device('cuda'),0,args.clip_max_norm,
                          args=args,mosaic_scheduler=schedule)
    assert stats['optimizer_steps']==1 and stats['training_phase_epoch']==11
    assert stats['training_seconds']>0 and crit.quality_successful_updates==1
    assert loader.dataset.mosaic.p==.25
    assert crit.quality_lambda==0
    assert not torch.equal(before,next(model.parameters()))
    if variant=='p2':
        stats=train_one_epoch(model,crit,loader,optim,torch.device('cuda'),1,args.clip_max_norm,args=args,mosaic_scheduler=schedule)
        assert crit.quality_lambda==.25


def test_unique_directory_with_frozen_clock(tmp_path,monkeypatch):
    import util.experiment as experiment
    class Clock:
        @staticmethod
        def now(): return SimpleNamespace(strftime=lambda fmt:'frozen')
    monkeypatch.setattr(experiment,'datetime',Clock)
    paths=[]
    for _ in range(3):
        args=SimpleNamespace(output_dir=str(tmp_path),unique_output_dir=True,resume='')
        experiment.unique_output(args); paths.append(args.output_dir)
    assert len(set(paths))==3


@pytest.mark.parametrize('field,value',[('quality_blend_max',float('nan')),('aligned_box_loss',1),
                                      ('training_phase_epoch_offset',-1),('quality_wrong',1)])
def test_incremental_config_validation(field,value):
    from util.incremental import validate_incremental
    assert validate_incremental({field:value})
