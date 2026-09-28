"""H1 opt-in numerical, index, random-stream and continuation contracts."""
import argparse
import copy
import random
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import torch
from torch import nn
from util import legacy_joint as joint
from util.config_validation import validate_config
from util.slconfig import SLConfig
from models.aqfcdetr.joint_modules import (
    SemanticDetailBridge, LocalDistributionRefiner, distribution_mean,
    distribution_support, distribution_loss, auxiliary_matches)


def recipe(name='h1_24e'):
    values=SLConfig.fromfile(f'configs/legacy_joint/{name}.py')._cfg_dict.to_dict()
    validate_config(values)
    return argparse.Namespace(**values)


def test_recipe_and_compatibility():
    a=recipe(); first=recipe('h1_epoch0'); smoke=recipe('h1_smoke')
    assert a.epochs==first.epochs==smoke.epochs==24
    assert a.val_epoch==[23] and first.val_epoch==[0,23]
    assert joint.signature(a)==joint.signature(first)
    assert not a.precision24_enabled and a.lr_drop_list==[13,23]
    assert a.lr==1e-4 and a.lr_backbone==1e-5 and a.training_phase_epoch_offset==0
    assert a.expected_pretrained_sha256==joint.SOURCE_SHA
    assert joint.signature({})=={}
    with pytest.raises(ValueError):validate_config(dict(vars(a),joint_mistake=1))
    with pytest.raises(ValueError):validate_config(dict(vars(a),enc_layers=2))


def test_bridge_padding_identity_gradient():
    torch.manual_seed(4)
    mod=SemanticDetailBridge(16,8)
    src=[torch.randn(2,16,7,9,requires_grad=True),torch.randn(2,16,4,5,requires_grad=True)]
    masks=[torch.zeros(2,*x.shape[-2:],dtype=torch.bool) for x in src]
    masks[0][0,-1:]=True
    mem=torch.randn(2,86,16,requires_grad=True)
    out,feats,ratios=mod(mem,src,masks)
    torch.testing.assert_close(out,mem,atol=0,rtol=0)
    assert (feats[0][0,:,-1]==0).all() and (ratios==0).all()
    out.square().mean().backward()
    assert mod.output[0].weight.grad.abs().sum()>0
    optimizer=torch.optim.SGD(mod.parameters(),lr=.1);optimizer.step();optimizer.zero_grad()
    out,_,_=mod(mem,src,masks);out.square().mean().backward()
    assert mod.detail[0].weight.grad.abs().sum()>0


def test_distribution_identity_and_mask_gradients():
    refiner=LocalDistributionRefiner(16,8,chunk=2)
    boxes=torch.tensor([[[.5,.5,.04,.08],[.1,.1,.01,.01],[.7,.8,.2,.3]]],requires_grad=True)
    query=torch.randn(1,3,16,requires_grad=True)
    features=[torch.randn(1,8,9,13,requires_grad=True),torch.randn(1,8,5,7,requires_grad=True)]
    masks=[torch.zeros(1,*x.shape[-2:],dtype=torch.bool) for x in features]
    valid=torch.tensor([[True,False,True]])
    out,logits=refiner(query,boxes,features,masks,torch.tensor([[33,49]]),valid)
    torch.testing.assert_close(out,boxes,atol=0,rtol=0)
    assert (distribution_mean(logits)==0).all()
    out[valid].sum().backward()
    assert refiner.head[-1].weight.grad.abs().sum()>0
    assert (boxes.grad[~valid]==0).all()
    assert torch.equal(boxes.grad[valid],torch.ones_like(boxes.grad[valid]))
    # Detached sampling coordinates, but differentiable features.
    pixels=(boxes.detach()*torch.tensor([49,33,49,33])).requires_grad_()
    refiner.sample(features[0],masks[0],pixels,4,1.25,8).sum().backward()
    assert pixels.grad is None and features[0].grad.abs().sum()>0


def test_distribution_supervision_boundary_and_empty():
    logits=torch.randn(1,3,4,33,requires_grad=True)
    boxes=torch.tensor([[[.5,.5,.1,.1],[.5,.5,.1,.1],[.5,.5,.1,.1]]],requires_grad=True)
    out=dict(refine_distribution_logits=logits,pred_boxes_coarse=boxes,joint_image_sizes=torch.tensor([[800,800]]))
    gt=boxes.detach()[0].clone();gt[0,0]+=.05;gt[1,0]+=.051;gt[2,2]*=3
    targets=[dict(boxes=gt,labels=torch.zeros(3,dtype=torch.long))]
    pairs=[(torch.arange(3),torch.arange(3))]
    loss,stats=distribution_loss(out,targets,pairs,3)
    assert torch.isfinite(loss) and stats['distribution_outside_ratio']>0
    loss.backward()
    assert boxes.grad is None and (logits.grad[0,1,0]==0).all() and (logits.grad[0,2,2]==0).all()
    empty=[(torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long))]
    loss,_=distribution_loss(out,targets,empty,1);assert loss.item()==0


def test_auxiliary_capacity_ties_empty_and_mask():
    from models.aqfcdetr.matcher import HungarianMatcher
    matcher=HungarianMatcher(cost_class=2,cost_bbox=5,cost_giou=2)
    out=dict(pred_logits=torch.zeros(2,7,9),pred_boxes=torch.full((2,7,4),.2),
        query_valid_mask=torch.tensor([[1,1,1,1,0,0,0],[1,1,1,1,1,1,1]],dtype=torch.bool))
    targets=[dict(labels=torch.tensor([0,1]),boxes=torch.full((2,4),.2)),
             dict(labels=torch.empty(0,dtype=torch.long),boxes=torch.empty(0,4))]
    pairs=auxiliary_matches(out,targets,matcher)
    assert len(pairs[0][0])==4 and len(pairs[0][0].unique())==4
    assert (pairs[0][0]<4).all() and len(pairs[1][0])==0
    assert torch.bincount(pairs[0][1]).max()<=3
    assert all(torch.equal(a,b) for p,q in zip(pairs,auxiliary_matches(out,targets,matcher)) for a,b in zip(p,q))


class RandomDataset(torch.utils.data.Dataset):
    def __len__(self):return 8
    def __getitem__(self,index):return index,random.random(),np.random.random(),torch.rand(())


def test_loader_random_state_and_complete_boundary():
    from util.joint_runtime import SeededDataset,loader_state,restore_loader
    def make(workers):
        dataset=SeededDataset(RandomDataset(),42)
        sampler=torch.utils.data.RandomSampler(dataset,generator=torch.Generator().manual_seed(1))
        return torch.utils.data.DataLoader(dataset,batch_sampler=torch.utils.data.BatchSampler(sampler,2,False),
            num_workers=workers,generator=torch.Generator().manual_seed(2))
    a=make(0); initial=torch.get_rng_state();rows=list(a)
    assert torch.equal(initial,torch.get_rng_state())
    state=loader_state(a);a.dataset.epoch=1;expected=list(a)
    b=make(0);restore_loader(b,state);b.dataset.epoch=1
    assert all(torch.equal(x,y) for xs,ys in zip(expected,list(b)) for x,y in zip(xs,ys))
    c=make(2)
    assert all(torch.equal(x,y) for xs,ys in zip(rows,list(c)) for x,y in zip(xs,ys))


def test_parameter_groups_scheduler_and_restore():
    a=recipe();m=nn.Module();m.backbone=nn.Linear(2,2);m.sampling_offsets=nn.Linear(2,2)
    m.distribution_refiner=nn.Linear(2,2);m.other=nn.Linear(2,2);m.joint_common_sha='verified'
    opt=torch.optim.AdamW(joint.parameter_groups(a,m))
    flat=[id(p) for g in opt.param_groups for p in g['params']]
    assert len(flat)==len(set(flat))==len(list(m.parameters()))
    joint.update_lr(opt,a,0,0);assert opt.param_groups[0]['lr']==pytest.approx(a.lr_backbone*.1)
    joint.update_lr(opt,a,13,500);assert opt.param_groups[0]['lr']==pytest.approx(a.lr_backbone*.1)
    joint.update_lr(opt,a,23,501);assert opt.param_groups[0]['lr']==pytest.approx(a.lr_backbone*.01)
    cp=dict(epoch=0,joint_state=joint.checkpoint_state(m,a,opt,0,10),
        criterion_progress=dict(successful_updates=10),rng_states=[{}],joint_data_rng=dict(sampler=1,workers=2))
    fresh=copy.deepcopy(a);joint.restore(m,cp,vars(fresh),opt)
    assert fresh._joint_data_rng==cp['joint_data_rng']
    fresh.architecture_variant='legacy_joint_control_v2'
    with pytest.raises(ValueError):joint.restore(m,cp,vars(fresh),opt)


@pytest.mark.parametrize('amp',[False,True])
def test_full_synthetic_identity_dn_backward(amp):
    # CPU uses the existing reference attention implementation; native CUDA is a
    # separately reported validation path, not an automatic real-image experiment.
    import torchvision
    from main import build_model_main
    from models.aqfcdetr.ops.functions.ms_deform_attn_func import MSDeformAttnFunction,ms_deform_attn_core_pytorch
    from models.aqfcdetr.query_allocator import QueryBudgetLoss
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if amp and device=='cpu':pytest.skip('CUDA AMP integration requires CUDA')
    args=recipe();args.device=device;args.distributed=False
    original=torchvision.models.resnet50
    def no_download(*a,**kw):kw['weights']=None;return original(*a,**kw)
    with patch('torchvision.models.resnet50',side_effect=no_download):model,criterion,post=build_model_main(args)
    model.to(device).eval();criterion.to(device);criterion.quality_successful_updates=500
    import contextlib
    ctx=(patch.object(MSDeformAttnFunction,'apply',side_effect=lambda v,s,st,l,w,i:ms_deform_attn_core_pytorch(v,s,l,w))
         if device=='cpu' else contextlib.nullcontext())
    with ctx:
        images=[torch.rand(3,65,81,device=device),torch.rand(3,80,96,device=device)]
        with torch.no_grad(),torch.autocast(device_type=device,enabled=amp):
            enhanced=model(images)
            bridge=model.transformer.semantic_detail_bridge;refiner=model.distribution_refiner
            del model.transformer.semantic_detail_bridge;del model.distribution_refiner
            plain=model(images)
            model.transformer.semantic_detail_bridge=bridge;model.distribution_refiner=refiner
        for key in ('pred_logits','pred_boxes'):
            maximum=(enhanced[key].float()-plain[key].float()).abs().max().item()
            print('H1 initialization',device,'AMP',amp,key,'max_abs',maximum)
            torch.testing.assert_close(enhanced[key].float(),plain[key].float(),atol=1e-5,rtol=1e-4)
        targets=[dict(boxes=torch.tensor([[.5,.5,.08,.08]],device=device),labels=torch.tensor([0],device=device)),
            dict(boxes=torch.empty(0,4,device=device),labels=torch.empty(0,dtype=torch.long,device=device))]
        model.train();model.set_epoch(0);criterion.train();criterion.quality_lambda=.25
        opt=torch.optim.AdamW(joint.parameter_groups(args,model));budget=QueryBudgetLoss(density_target_backend='vectorized').to(device)
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device,enabled=amp):
                out=model(images,targets);losses=criterion(out,targets)
                assert out['dn_meta']['pad_size']>0 and 'auxiliary_o2m_outputs' in out
                loss=sum(v*criterion.weight_dict[k] for k,v in losses.items() if k in criterion.weight_dict)
                loss+=budget(out['allocator_outputs'],dict(targets=targets,real_counts=torch.tensor([1.,0.],device=device)))['loss_allocator_total']
            assert torch.isfinite(loss);loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            assert refiner.head[-1].weight.grad.abs().sum()>0
            assert bridge.output[0].weight.grad.abs().sum()>0
            assert model.transformer.candidate_auxiliary_head.classification.weight.grad.abs().sum()>0
            if step:assert bridge.fusion[0][0].weight.grad.abs().sum()>0
            nn.utils.clip_grad_norm_(model.parameters(),.1);opt.step()
        model.eval()
        def forbidden(*a,**kw):raise AssertionError('Training auxiliary head executed during eval')
        with patch.object(model.transformer.candidate_auxiliary_head,'forward',side_effect=forbidden),torch.no_grad():
            result=model(images)
        assert 'auxiliary_o2m_outputs' not in result
        predictions=post['bbox'](result,torch.tensor([[65,81],[80,96]],device=device))
        assert all(torch.isfinite(p['boxes']).all() for p in predictions)
