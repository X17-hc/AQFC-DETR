"""Synthetic-only contracts; these tests never load real images or start training jobs."""
import argparse
import copy
from pathlib import Path
import shlex
import xml.etree.ElementTree as ET

import pytest
import torch

from models.aqfcdetr.model import SetCriterion
from models.aqfcdetr.matcher import HungarianMatcher
from util.geometry_loss import (matched_geometry_loss, initialize_geometry_progress,
                                update_geometry_weight, criterion_progress)
from util.incremental import quality_progress, training_phase, validate_incremental
from util.experiment import variant_signature
from util.checkpoint import load_native_resume


def fixture():
    pred = torch.tensor([[[.51,.48,.025,.045],[.3,.4,.1,.1],[.7,.6,.2,.2]]],requires_grad=True)
    target = [dict(labels=torch.tensor([0]), boxes=torch.tensor([[.5,.5,.01,.02]],requires_grad=True),
                   size=torch.tensor([200,400]), orig_size=torch.tensor([999,999]))]
    indices = [(torch.tensor([0]),torch.tensor([0]))]
    return pred,target,indices


def test_manual_formula_pixel_floor_and_gt_detach():
    pred,t,idx = fixture()
    loss = matched_geometry_loss(pred,t,idx,1)
    residual = torch.tensor([4/4,-4/4,6/4,5/4])
    expected = torch.nn.functional.smooth_l1_loss(residual,torch.zeros(4),beta=.1)
    torch.testing.assert_close(loss,expected,atol=1e-5,rtol=1e-4)
    loss.backward()
    assert t[0]['boxes'].grad is None
    assert pred.grad[0,1:].eq(0).all()
    assert torch.isfinite(pred.grad).all()
    # Per normalized coordinate gradient upper bound = dimension/(4*d*num_boxes).
    assert (pred.grad[0,0].abs() <= torch.tensor([25,12.5,25,12.5])).all()


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
def test_fp32_loss_extreme_small_target_and_finite_gradient(dtype):
    pred,t,idx=fixture(); pred=pred.detach().to(dtype).requires_grad_()
    t[0]['boxes']=torch.tensor([[.0,.0,1e-10,1e-10]])
    with torch.autocast('cpu',dtype=torch.bfloat16):
        loss=matched_geometry_loss(pred,t,idx,1)
    assert loss.dtype==torch.float32 and torch.isfinite(loss)
    loss.backward(); assert torch.isfinite(pred.grad).all()


def test_empty_is_differentiable_and_padding_nan_does_not_leak():
    pred,t,_=fixture(); pred=pred.detach(); pred[0,2]=float('nan'); pred.requires_grad_()
    empty=[(torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long))]
    loss=matched_geometry_loss(pred,t,empty,1)
    assert loss.item()==0
    loss.backward(); assert pred.grad.eq(0).all()


def test_scale_invariance_above_floor_and_batch_gt_normalization():
    pred,t,idx=fixture()
    original=matched_geometry_loss(pred,t,idx,1)
    t2=copy.deepcopy(t); t2[0]['size']*=2
    torch.testing.assert_close(original,matched_geometry_loss(pred,t2,idx,1))
    doubled=pred.repeat(2,1,1)
    torch.testing.assert_close(original,matched_geometry_loss(doubled,t+t,idx+idx,2))


def make_criterion(weight):
    return SetCriterion(9,HungarianMatcher(),{'loss_ce':1,'loss_bbox':5,'loss_giou':2,'loss_nwd':1},
                        .25,['labels','boxes','cardinality'],aligned_box_loss=True,
                        classification_loss_type='quality_blend',geometry_loss_weight=weight)


def outputs():
    pred,t,idx=fixture()
    out=dict(pred_boxes=pred,pred_logits=torch.randn(1,3,9,requires_grad=True),
             query_valid_mask=torch.tensor([[True,True,False]]),dn_meta=None)
    aux={k:v.detach().clone().requires_grad_(v.is_floating_point()) for k,v in out.items() if isinstance(v,torch.Tensor)}
    out['aux_outputs']=[aux]
    out['interm_outputs']=aux
    out['enc_outputs']=[aux]
    return out,t


def test_only_last_layer_geometry_and_zero_weight_original_gradient():
    out,t=outputs(); a,b=make_criterion(0),make_criterion(.05)
    a.quality_lambda=b.quality_lambda=.25
    old,new=a(out,t),b(out,t)
    assert set(new)-set(old)=={'loss_geometry'}
    assert [k for k in b.weight_dict if 'geometry' in k]==['loss_geometry']
    old_total=sum(old[k]*a.weight_dict[k] for k in old if k in a.weight_dict)
    new_total=sum(new[k]*b.weight_dict[k] for k in new if k in b.weight_dict)
    torch.testing.assert_close(old_total,new_total,atol=0,rtol=0)
    ga=torch.autograd.grad(old_total,out['pred_boxes'],retain_graph=True)[0]
    gb=torch.autograd.grad(new_total,out['pred_boxes'],retain_graph=True)[0]
    torch.testing.assert_close(ga,gb,atol=0,rtol=0)
    gg=torch.autograd.grad(new['loss_geometry'],out['pred_boxes'])[0]
    assert gg[0,2].eq(0).all()
    assert out['aux_outputs'][0]['pred_boxes'].grad is None


def test_progress_skip_restore_and_cross_variant_rejection(tmp_path):
    args=argparse.Namespace(geometry_loss_weight=.05,geometry_warmup_epochs=1,
                            quality_blend_warmup_epochs=0,classification_loss_type='quality_blend')
    crit=make_criterion(.05); initialize_geometry_progress(crit,args,10)
    for successful,expected in [(0,0),(5,.025),(5,.025),(10,.05),(20,.05)]:
        crit.quality_successful_updates=successful; update_geometry_weight(crit)
        assert crit.weight_dict['loss_geometry']==pytest.approx(expected)
    model=torch.nn.Linear(2,1); optimizer=torch.optim.AdamW(model.parameters())
    model(torch.ones(1,2)).sum().backward(); optimizer.step()
    path=tmp_path/'native.pth'
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=0,
                    variant_signature=variant_signature(args),criterion_progress=criterion_progress(crit)),path)
    with pytest.warns(RuntimeWarning,match='RNG'):
        assert load_native_resume(model,path,optimizer=optimizer,expected_args=args)==1
    restored=make_criterion(.05); restored.quality_successful_updates=args.quality_successful_updates
    initialize_geometry_progress(restored,args,10); update_geometry_weight(restored)
    assert restored.weight_dict['loss_geometry']==.05
    with pytest.raises(ValueError,match='denominator'): initialize_geometry_progress(restored,args,11)
    args.geometry_loss_weight=0
    with pytest.raises(ValueError,match='signature'): load_native_resume(model,path,optimizer=optimizer,expected_args=args)


@pytest.mark.parametrize('field,value',[('geometry_loss_weight',-1),('geometry_min_size_pixels',0),
    ('geometry_smooth_l1_beta',float('nan')),('geometry_warmup_epochs',0),('geometry_typo',True),
    ('training_phase_fixed_epoch',24)])
def test_validation(field,value):
    assert validate_incremental({'training_phase_total_epochs':24,field:value})


def test_old_signature_unchanged_by_neutral_new_defaults():
    from util.geometry_loss import DEFAULTS
    old=variant_signature({})
    assert not set(old)&set(DEFAULTS)
    assert variant_signature(DEFAULTS)==old


def test_configs_and_xml_identical_except_geometry():
    from util.slconfig import SLConfig
    from util.config_validation import validate_config
    from main import get_args_parser
    configs=[]
    for arm in ('c0','c1'):
        cfg=SLConfig.fromfile(f'configs/geometry_v1/{arm}_3e.py')._cfg_dict.to_dict()
        validate_config(cfg); configs.append(cfg)
        assert cfg['epochs']==3 and cfg['val_epoch']==[2] and cfg['strict_warmstart']
        assert cfg['expected_pretrained_epoch']==23 and not cfg['use_ema']
        args=argparse.Namespace(**cfg)
        for epoch in range(3): assert training_phase(args,epoch)==(23,24)
        assert quality_progress(args,0,7009)==.25
        doc=ET.parse(f'.run/AQFC-DETR_Geometry_{arm}.run.xml')
        opts={x.get('name'):x.get('value') for x in doc.findall('.//option')}
        assert opts['WORKING_DIRECTORY']=='/workspace/AQFC-DETR'
        assert opts['RUN_WITH_PYTHON_CONSOLE']=='false'
        cli=get_args_parser().parse_args(shlex.split(opts['PARAMETERS']))
        assert cli.num_workers==2 and cli.seed==42 and cli.amp and cli.unique_output_dir
        assert cli.pretrain_model_path.endswith('resume_epoch11_20260916/checkpoint0023.pth')
        assert not cli.resume and cli.max_train_steps==cli.max_eval_steps==0
        assert not set(cfg)&{k for k,v in vars(cli).items() if v is not None}
    assert {k for k in configs[0] if configs[0][k]!=configs[1][k]}=={'geometry_loss_weight'}


def test_synthetic_optimizer_changes_parameters_with_geometry():
    out,t=outputs(); criterion=make_criterion(.05); criterion.weight_dict['loss_geometry']=.05
    criterion.quality_lambda=.25
    optimizer=torch.optim.AdamW([out['pred_boxes'],out['pred_logits']],lr=1e-5)
    before=out['pred_boxes'].detach().clone()
    losses=criterion(out,t)
    total=sum(losses[k]*criterion.weight_dict[k] for k in losses if k in criterion.weight_dict)
    total.backward()
    assert torch.isfinite(out['pred_boxes'].grad).all()
    torch.nn.utils.clip_grad_norm_([out['pred_boxes'],out['pred_logits']],.1,error_if_nonfinite=True)
    optimizer.step()
    assert not torch.equal(before,out['pred_boxes'])


@pytest.mark.parametrize('arm',['c0','c1'])
def test_real_model_synthetic_engine_update(arm):
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required for synthetic model backward')
    from unittest.mock import patch
    import torchvision
    from main import build_model_main
    from engine import train_one_epoch, MosaicPScheduler
    from util.slconfig import SLConfig
    from util.misc import nested_tensor_from_tensor_list
    from types import SimpleNamespace
    cfg=SLConfig.fromfile(f'configs/geometry_v1/{arm}_3e.py')._cfg_dict.to_dict()
    cfg.update(device='cuda',distributed=False,rank=0,amp=True,max_train_steps=1,
               max_eval_steps=0,profile_trace='',num_workers=0)
    args=argparse.Namespace(**cfg)
    original=torchvision.models.resnet50
    def no_download(*a,**kw):
        kw['weights']=None
        return original(*a,**kw)
    with patch('torchvision.models.resnet50',side_effect=no_download):
        model,crit,_=build_model_main(args)
    model.cuda(); crit.cuda()
    image=torch.rand(3,128,128)
    target=[dict(labels=torch.tensor([0]),boxes=torch.tensor([[.4,.4,.05,.06]]),size=torch.tensor([128,128]))]
    class Loader:
        dataset=SimpleNamespace(mosaic=SimpleNamespace(p=0.))
        def __len__(self): return 1
        def __iter__(self): yield nested_tensor_from_tensor_list([image]),target
    loader=Loader(); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
    schedule=MosaicPScheduler(peak_p=.25,warmup_end=4,decay_start=20,total_epochs=24)
    # Simulate one successful update: test C1 at full auxiliary weight, not just zero.
    crit.quality_successful_updates=1
    stats=train_one_epoch(model,crit,loader,optimizer,torch.device('cuda'),0,args.clip_max_norm,
                         args=args,mosaic_scheduler=schedule)
    assert stats['optimizer_steps']==1 and crit.quality_successful_updates==2
    assert stats['training_phase_epoch']==23 and crit.quality_lambda==.25
    assert stats['teacher_ratio']==0
    if arm=='c1':
        assert stats['geometry_weight']==.05 and stats['loss_geometry']>0
    else:
        assert 'loss_geometry' not in stats
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
