import argparse
import copy
import pytest
import torch
from util.dome_transfer import underestimate_loss, under_weight, signature
from util.slconfig import SLConfig
from util.config_validation import validate_config
from util import legacy_joint
from models.aqfcdetr.protected_selection import redundancy_mask
from models.aqfcdetr.proposal_selection import select_proposal_indices
from util.box_ops import box_iou, box_cxcywh_to_xyxy

def recipe(name='d1_3e'):
    v=SLConfig.fromfile(f'configs/dome_transfer/{name}.py')._cfg_dict.to_dict()
    validate_config(v)
    return argparse.Namespace(**v)

def test_recipe_clocks_and_old_contract():
    a,b=recipe('d0_3e'),recipe()
    diff={k for k in vars(a) if getattr(a,k)!=getattr(b,k)}
    assert diff=={'dome_transfer_recipe','density_underestimate_weight','proposal_selection_mode'}
    assert signature(a)!=signature(b) and signature({})=={}
    from util.incremental import quality_progress,training_phase
    assert quality_progress(b,0,7009)==.25 and training_phase(b,0)==(23,24)
    assert under_weight(b,0)==0 and under_weight(b,250)==.05 and under_weight(b,501)==.1
    old=SLConfig.fromfile('configs/legacy_joint/h1_24e.py')._cfg_dict.to_dict()
    validate_config(old)
    with pytest.raises(ValueError):validate_config(dict(old,proposal_selection_mode='protected_density'))
    with pytest.raises(ValueError):validate_config(dict(vars(b),density_underestimate_typo=True))

def test_under_gradient_padding_empty_and_per_image_normalization():
    p=torch.tensor([[[[.1,.9,.2]]],[[[.2,.2,.2]]]],requires_grad=True)
    y=torch.tensor([[[[1.,.5,1.]]],[[[0.,0.,0.]]]],requires_grad=True)
    mask=torch.tensor([[[[True,True,False]]],[[[True,True,True]]]])
    loss,support=underestimate_loss(p,y,mask)
    assert loss.item()==pytest.approx(.81/1.5/2)
    loss.backward()
    assert p.grad[0,0,0,0]<0 and torch.count_nonzero(p.grad)==1 and y.grad is None
    assert support.item()==.75
    with torch.autocast('cpu',dtype=torch.bfloat16):
        l,_=underestimate_loss(p,y,mask)
    assert l.dtype==torch.float32 and torch.isfinite(l)

@pytest.mark.parametrize('chunk',[1,3,256])
def test_redundancy_reference_ties_and_invalid_class(chunk):
    torch.manual_seed(1)
    boxes=torch.rand(21,4);boxes[:,2:]=.6
    boxes[1]=boxes[0];boxes[2]=boxes[0]
    labels=torch.zeros(21,dtype=torch.long);labels[2]=8
    density=torch.rand(21)
    got=redundancy_mask(boxes,labels,density,range(8),chunk)
    iou=box_iou(box_cxcywh_to_xyxy(boxes),box_cxcywh_to_xyxy(boxes))[0]
    expected=torch.tensor([any(labels[i]==labels[j] and labels[i]<8 and
        iou[i,j]>.4+.5*max(density[i],density[j]) for j in range(i)) for i in range(21)])
    assert torch.equal(got,expected) and not got[2]

def sample(mode='protected_density',density=None,diagnostics=None,counts=None):
    logits=torch.zeros(2,30,9);logits[:,:,0]=torch.arange(30,0,-1)
    boxes=torch.full((2,30,4),.3)
    pad=torch.zeros(2,30,dtype=torch.bool);pad[1,8:10]=True
    return select_proposal_indices(logits,torch.ones(2,30)*.2 if density is None else density,
        pad,12,mode=mode,proposal_boxes=torch.zeros_like(boxes),spatial_shapes=[[3,8],[2,3]],
        decoded_proposal_boxes=boxes,spatial_grid_size=(2,3),query_counts=counts,diagnostics=diagnostics)

def test_core_counts_padding_determinism_and_mixed_budget():
    report=[];a=sample(diagnostics=report)
    assert torch.equal(a,sample()) and a.shape==(2,12)
    assert len(a[0].unique())==12 and set(range(9)).issubset(a[0].tolist())
    assert not {8,9}&set(a[1].tolist())
    assert all(r['core_count']+r['nonredundant_added']+r['redundant_added']+r['global_filled']==12 for r in report)
    b=sample(counts=torch.tensor([5,12]));assert len(b[0,:5].unique())==5
    assert (b[0,5:]==b[0,0]).all()

def test_zero_density_fallback_and_nonfinite_rejected():
    a=sample(density=torch.zeros(2,30));assert a[0].tolist()==list(range(12))
    d=torch.ones(2,30);d[0,0]=float('nan')
    with pytest.raises(FloatingPointError):sample(density=d)

def test_optimizer_groups_and_constant_peak():
    a=recipe();m=torch.nn.Module();m.backbone=torch.nn.Linear(2,2)
    m.distribution_refiner=torch.nn.Linear(2,2);m.sampling_offsets=torch.nn.Linear(2,2)
    opt=torch.optim.AdamW(legacy_joint.parameter_groups(a,m))
    assert len({id(p) for g in opt.param_groups for p in g['params']})==len(list(m.parameters()))
    legacy_joint.update_lr(opt,a,0,0);assert max(g['lr'] for g in opt.param_groups)==pytest.approx(1e-6)
    legacy_joint.update_lr(opt,a,2,501);assert max(g['lr'] for g in opt.param_groups)==pytest.approx(1e-5)

def test_selection_reference_final_indices(monkeypatch):
    from models.aqfcdetr import protected_selection as p
    from models.aqfcdetr.spatial_selection import _quota_additions_reference
    actual=sample()
    def naive(boxes,labels,density,classes,chunk=256):
        xy=box_cxcywh_to_xyxy(boxes.float());iou=box_iou(xy,xy)[0]
        return torch.tensor([any(int(labels[i]) in classes and labels[i]==labels[j] and
            iou[i,j]>.4+.5*max(density[i],density[j]) for j in range(i)) for i in range(len(boxes))],device=boxes.device)
    monkeypatch.setattr(p,'redundancy_mask',naive)
    monkeypatch.setattr(p,'_quota_additions',_quota_additions_reference)
    assert torch.equal(actual,sample())

@pytest.mark.parametrize('name',['d0_3e','d1_3e'])
def test_full_synthetic_dn_backward_and_grouped_inference(name):
    # Reuse the full H1 synthetic contract with the new independently validated recipe.
    import runpy
    context=runpy.run_path('tests/test_legacy_joint.py')
    function=context['test_full_synthetic_identity_dn_backward']
    function.__globals__['recipe']=lambda:recipe(name)
    function(False)

def test_resume_variant_isolation():
    from util.experiment import variant_signature
    a,b=recipe('d0_3e'),recipe()
    assert variant_signature(a)!=variant_signature(b)
    assert variant_signature(a)==variant_signature(copy.deepcopy(a))
    # Optimizer clock is a separate integer; successful updates alone advance warmup.
    state=dict(successful_updates=250)
    assert under_weight(b,state['successful_updates'])==.05
    assert under_weight(b,state['successful_updates'])==.05  # AMP skipped: no advance
    state['successful_updates']+=1
    assert under_weight(b,state['successful_updates'])>.05

def test_native_checkpoint_roundtrip_and_cross_recipe_rejection(tmp_path):
    from util.checkpoint import load_native_resume,capture_rng_state
    from util.experiment import variant_signature
    a=recipe();m=torch.nn.Linear(2,2);m.joint_common_sha='synthetic-verified'
    opt=torch.optim.AdamW(legacy_joint.parameter_groups(a,m))
    m(torch.ones(1,2)).sum().backward();opt.step()
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    cp=dict(model=m.state_dict(),epoch=0,optimizer=opt.state_dict(),lr_scheduler=scheduler.state_dict(),
        scaler=scaler.state_dict(),variant_signature=variant_signature(a),
        criterion_progress=dict(successful_updates=17),rng_states=[capture_rng_state()],
        joint_data_rng=dict(sampler=torch.get_rng_state(),workers=torch.get_rng_state()),
        joint_state=legacy_joint.checkpoint_state(m,a,opt,0,17),run_metadata=dict(epoch_complete=True))
    path=tmp_path/'roundtrip.pth';torch.save(cp,path)
    fresh=torch.nn.Linear(2,2);other=torch.optim.AdamW(legacy_joint.parameter_groups(a,fresh))
    args=copy.deepcopy(a)
    assert load_native_resume(fresh,path,optimizer=other,scaler=scaler,expected_args=args)==1
    assert args.quality_successful_updates==17
    for x,y in zip(m.parameters(),fresh.parameters()):assert torch.equal(x,y)
    for old,new in zip(opt.state.values(),other.state.values()):
        assert torch.equal(old['exp_avg'],new['exp_avg'])
    with pytest.raises(ValueError):load_native_resume(fresh,path,optimizer=other,expected_args=recipe('d0_3e'))

def test_new_manual_xml_and_smoke_recipe():
    import xml.etree.ElementTree as ET
    from pathlib import Path
    files=[p for p in Path('.run').glob('*.xml') if any(x in p.name for x in ('Dome借鉴版','D0原H1','D1密度候选','D0-D1同卡'))]
    assert len(files)==6
    for path in files:
        cfg=ET.parse(path).getroot().find('configuration')
        options={o.attrib['name']:o.attrib.get('value','') for o in cfg.findall('option')}
        assert options['WORKING_DIRECTORY']=='/workspace/AQFC-DETR'
        assert options['SDK_HOME']=='/opt/conda/envs/AQFC-DETR/bin/python'
        assert '--resume' not in options['PARAMETERS']
    a=recipe('d1_smoke');assert a.epochs==1 and a.val_epoch==[0] and a.dome_transfer_smoke

def test_actual_engine_includes_under_loss_once_and_advances_updates():
    import torchvision
    from unittest.mock import patch
    from main import build_model_main
    from engine import train_one_epoch
    from util.misc import collate_fn
    from models.aqfcdetr.ops.functions.ms_deform_attn_func import MSDeformAttnFunction,ms_deform_attn_core_pytorch
    a=recipe();a.device='cpu';a.distributed=False;a.rank=0;a.amp=False
    a.max_train_steps=2;a.max_eval_steps=0;a.profile_trace='';a.max_consecutive_skipped_steps=20
    original=torchvision.models.resnet50
    def no_download(*args,**kwargs):kwargs['weights']=None;return original(*args,**kwargs)
    with patch('torchvision.models.resnet50',side_effect=no_download):model,criterion,_=build_model_main(a)
    data=[(torch.rand(3,65,81),dict(labels=torch.tensor([0]),
        boxes=torch.tensor([[.5,.5,.1,.1]]),size=torch.tensor([65,81]))) for _ in range(4)]
    loader=torch.utils.data.DataLoader(data,batch_size=2,collate_fn=collate_fn)
    opt=torch.optim.AdamW(legacy_joint.parameter_groups(a,model))
    criterion.quality_successful_updates=500
    with patch.object(MSDeformAttnFunction,'apply',side_effect=lambda v,s,st,l,w,i:ms_deform_attn_core_pytorch(v,s,l,w)):
        stats=train_one_epoch(model,criterion,loader,opt,torch.device('cpu'),0,
            max_norm=.1,args=a)
    assert stats['optimizer_steps']==2 and criterion.quality_successful_updates==502
    assert stats['loss_density_under_raw_unscaled']>0
    assert stats['loss_density_under']==pytest.approx(.1*stats['loss_density_under_raw_unscaled'])
    assert stats['quality_lambda']==.25 and stats['training_phase_epoch']==23
