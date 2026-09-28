import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from torch import nn

from models.aqfcdetr.p2_transfer import P2Substitute,transfer_loss,adaptation_loss,step_student
from util import precision24
from util.p2_transfer import (VERSION,SOURCE_SHA,PREFIX,freeze_for_adaptation,frozen_digest,
                              extract_inputs,load_artifact,initialize_student)
from util.slconfig import SLConfig
from util.config_validation import validate_config


def recipe(name='reference'):
    values=SLConfig.fromfile(f'configs/p2_transfer/{name}.py')._cfg_dict.to_dict()
    validate_config(values)
    return argparse.Namespace(**values)


def test_config_legacy_signature_and_modes():
    from util.experiment import variant_signature
    a=recipe(); b=recipe('student_24e')
    assert a.p2_transfer_mode=='reference' and b.p2_transfer_mode=='student'
    assert not a.detail_context_encoder and a.local_refiner_enabled
    assert variant_signature(a)['p2_transfer_version']==VERSION
    assert variant_signature(b)['precision24_recipe']=='fixed_2_heavy_4_light_lr16_22'
    old=SLConfig.fromfile('configs/precision24/f1_24e.py')._cfg_dict.to_dict()
    assert 'p2_transfer_version' not in variant_signature(old)
    for key,value in [('p2_transfer_mode','bad'),('p2_transfer_typo',1),('detail_context_encoder',True),('precision24_stage_override',2)]:
        with pytest.raises(ValueError): validate_config(dict(vars(a),**{key:value}))


def test_adaptation_recipe_checks_executed_resize():
    from tools.train_p2_transfer import validate_recipe
    values=vars(recipe('adapt_2e'))
    validate_recipe(values)
    with pytest.raises(ValueError): validate_recipe(dict(values,lr=2e-4))


def test_adapter_mask_identity_and_later_gradients():
    torch.manual_seed(42)
    shapes=torch.tensor([[5,7],[3,4],[2,2],[1,1],[1,1]])
    x=torch.randn(2,53,16,requires_grad=True)
    mask=torch.zeros(2,53,dtype=torch.bool); mask[0,28:35]=True
    module=P2Substitute(16,8)
    out=module(x,shapes,mask)
    torch.testing.assert_close(out,x[:,:35],atol=0,rtol=0)
    optimizer=torch.optim.AdamW(module.parameters(),lr=.01)
    for _ in range(2):
        optimizer.zero_grad(); out=module(x,shapes,mask)
        ((out-1).square()*(~mask[:,:35,None])).mean().backward(); optimizer.step()
    assert module.detail.weight.grad.abs().sum()>0
    assert module.context.weight.grad.abs().sum()>0
    torch.testing.assert_close(module(x,shapes,mask)[mask[:,:35]],x[:,:35][mask[:,:35]])


def test_loss_detaches_teacher_ignores_padding_and_is_finite():
    teacher=torch.randn(2,12,16,requires_grad=True)
    student=teacher.detach().clone().requires_grad_()
    valid=torch.ones(2,12,dtype=torch.bool); valid[:,8:]=False
    weight=torch.ones(2,12)
    assert transfer_loss(student,teacher,weight,valid).abs()<1e-6
    modified=student.detach().clone(); modified[:,8:]=1000
    torch.testing.assert_close(transfer_loss(modified,teacher,weight,valid),transfer_loss(student,teacher,weight,valid))
    loss=transfer_loss(student+1,teacher,weight,valid); loss.backward()
    assert teacher.grad is None and torch.isfinite(student.grad).all()
    assert student.grad[:,8:].abs().sum()==0


def test_same_input_for_both_branches():
    seen=[]
    class Adapter(nn.Module):
        def forward(self,m,s,p): seen.append(m); return m[:,:4]+1
    class Layer(nn.Module):
        def forward_queries(self,q,pos,refs,m,*args): seen.append(m); return q+2
    encoder=nn.Module(); encoder.p2_substitutes=nn.ModuleList([Adapter()]*4)
    encoder.layers=nn.ModuleList([Layer()]*6)
    memory=torch.randn(1,5,4)
    out=step_student(encoder,2,memory,torch.zeros_like(memory),torch.zeros(1,5,2,2),
                     torch.tensor([[2,2],[1,1]]),torch.tensor([0,4]),torch.zeros(1,5,dtype=torch.bool))
    assert seen[0] is memory and seen[1] is memory
    torch.testing.assert_close(out[:,:4],memory[:,:4]+1)
    torch.testing.assert_close(out[:,4:],memory[:,4:]+2)


def dummy():
    m=nn.Module(); m.hidden_dim=16; m.backbone=nn.Linear(3,16)
    m.transformer=nn.Module(); m.transformer.encoder=nn.Module()
    precision24.configure(m,recipe())
    return m


def test_freeze_schedule_and_parameter_groups():
    m=dummy(); initial=frozen_digest(m); freeze_for_adaptation(m)
    assert all(p.requires_grad==n.startswith(PREFIX) for n,p in m.named_parameters())
    for epoch in (0,2,23):
        precision24.set_epoch(m,epoch); assert m.transformer.encoder.detail_full_layers==6
    m.transformer.encoder.p2_transfer_mode='student'
    for epoch in (0,2,23):
        precision24.set_epoch(m,epoch); assert m.transformer.encoder.detail_full_layers==2
    assert frozen_digest(m)==initial
    groups=precision24.param_groups(recipe(),m)
    assert len(groups)==1 and groups[0]['precision24_group']=='new'
    assert len(groups[0]['params'])==len(list(m.transformer.encoder.p2_substitutes.parameters()))


def test_artifact_and_formal_gate(tmp_path):
    m=dummy(); file=tmp_path/'adapter.pth'
    data=dict(kind='p2_adapter_initialization',version=VERSION,source_sha256=SOURCE_SHA,
              complete_two_epochs=True,adapters=m.transformer.encoder.p2_substitutes.state_dict())
    torch.save(data,file); load_artifact(m,file)
    with pytest.raises(ValueError): load_artifact(m,file,require_receipt=True)
    args=recipe('student_24e'); args.eval=False; args.resume=''; args.p2_adapter=''
    with pytest.raises(ValueError): initialize_student(m,args)
    data['complete_two_epochs']=False; torch.save(data,file)
    with pytest.raises(ValueError): load_artifact(m,file)


def test_adaptation_resume_roundtrip_and_partial_rejection(tmp_path):
    from tools.train_p2_transfer import restore_adapter_training
    from util.checkpoint import capture_rng_state
    m=dummy(); freeze_for_adaptation(m)
    opt=torch.optim.AdamW(m.transformer.encoder.p2_substitutes.parameters())
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    state=dict(kind='p2_adaptation_resume',epoch_complete=True,signature={'test':1},epoch=0,
        frozen_digest=frozen_digest(m),adapters=m.transformer.encoder.p2_substitutes.state_dict(),
        optimizer=opt.state_dict(),scaler=scaler.state_dict(),rng=capture_rng_state(),successful_updates=10)
    file=tmp_path/'resume.pth'; torch.save(state,file)
    assert restore_adapter_training(file,m,opt,scaler,{'test':1})==(1,10)
    with pytest.raises(ValueError): restore_adapter_training(file,m,opt,scaler,{'test':2})
    state['epoch_complete']=False; torch.save(state,file)
    with pytest.raises(ValueError): restore_adapter_training(file,m,opt,scaler,{'test':1})


def test_old_new_resume_cannot_mix():
    m=dummy(); saved=precision24.state(m,3)
    m.transformer.encoder.p2_transfer_mode='student'
    with pytest.raises(ValueError):
        precision24.restore(m,dict(epoch=0,precision24_state=saved,criterion_progress={'successful_updates':3}))


def test_quality_gate_all_conditions():
    from tools.check_p2_transfer import gate
    r=dict(AP=40.,APvt=20.,proposal_050=90.,count_mae=2.,mean_queries=400.)
    assert gate(r,r)['accepted']
    for key,value in [('AP',38.),('APvt',18.),('proposal_050',87.),('count_mae',4.),('mean_queries',481.)]:
        assert not gate(r,dict(r,**{key:value}))['accepted']


@pytest.mark.parametrize('amp',[False,True])
def test_reference_child_precision_is_explicit(tmp_path,amp):
    from tools.check_p2_transfer import evaluate
    cli=argparse.Namespace(pretrained='frozen.pth',data_root='data',amp=amp)
    # Stop at the subprocess seam: verify actual entry command, not a metadata label.
    with patch('tools.check_p2_transfer.subprocess.run') as child:
        child.return_value.returncode=1
        with pytest.raises(RuntimeError): evaluate(cli,tmp_path,'child',[1],'configs/p2_transfer/reference.py')
        assert ('--amp' if amp else '--no-amp') in child.call_args.args[0]


def test_manual_run_configs():
    import shlex
    import xml.etree.ElementTree as ET
    paths=list(Path('.run').glob('AQFC-DETR P2迁移V2*.run.xml'))
    assert len(paths)==5
    for path in paths:
        root=ET.parse(path).getroot()
        options={n.attrib['name']:n.attrib['value'] for n in root.findall('.//option')}
        env={n.attrib['name']:n.attrib['value'] for n in root.findall('.//env')}
        assert options['WORKING_DIRECTORY']=='/workspace/AQFC-DETR'
        assert options['SDK_HOME']=='/opt/conda/envs/AQFC-DETR/bin/python'
        assert env['CUDA_VISIBLE_DEVICES']=='2'
        assert options['SCRIPT_NAME'].startswith('$PROJECT_DIR$/tools/')
        tokens=shlex.split(options['PARAMETERS'])
        assert '--resume' not in tokens
        assert '/outputs/p2_transfer_v2/' in options['PARAMETERS']
        if '适配20步' in path.name:
            assert tokens[tokens.index('--max-train-steps')+1]=='20'
        if '独立适配2轮' in path.name:
            assert tokens[tokens.index('--max-train-steps')+1]=='0'


@pytest.mark.parametrize('amp',[False,True])
def test_native_full_reference_adaptation_and_student_dn(amp,tmp_path):
    if not torch.cuda.is_available(): pytest.skip('Native CUDA integration')
    import torchvision
    from main import build_model_main
    from util.misc import nested_tensor_from_tensor_list
    from models.aqfcdetr.query_allocator import QueryBudgetLoss
    args=recipe(); args.device='cuda'; args.distributed=False
    original=torchvision.models.resnet50
    def no_download(*a,**kw): kw['weights']=None; return original(*a,**kw)
    with patch('torchvision.models.resnet50',side_effect=no_download): model,criterion,post=build_model_main(args)
    model.cuda().eval(); criterion.cuda()
    images=nested_tensor_from_tensor_list([torch.rand(3,65,81,device='cuda'),torch.rand(3,80,96,device='cuda')])
    targets=[dict(boxes=torch.tensor([[.01,.98,.02,.03]],device='cuda'),labels=torch.tensor([0],device='cuda'),size=torch.tensor([65,81],device='cuda')),
             dict(boxes=torch.empty(0,4,device='cuda'),labels=torch.empty(0,dtype=torch.long,device='cuda'),size=torch.tensor([80,96],device='cuda'))]
    encoder=model.transformer.encoder
    with torch.no_grad(),torch.autocast('cuda',enabled=amp):
        new=model(images)
        subs=encoder.p2_substitutes; refiner=model.local_refiner
        del encoder.p2_substitutes; del model.local_refiner
        old=model(images)
        encoder.p2_substitutes=subs; model.local_refiner=refiner
    for key in ('pred_boxes','pred_logits'):
        torch.testing.assert_close(old[key],new[key],atol=1e-5,rtol=1e-4)
    freeze_for_adaptation(model); before=frozen_digest(model)
    opt=torch.optim.AdamW(subs.parameters(),lr=1e-3)
    # No teacher decoder/AQBA execution, and no frozen state updates.
    with patch.object(model.transformer.decoder,'forward',side_effect=AssertionError('Decoder must not run')):
        for phase in ('teacher_forced','teacher_forced','free_running'):
            opt.zero_grad()
            with torch.autocast('cuda',enabled=amp):
                inputs=extract_inputs(model,images)
                loss,_=adaptation_loss(encoder,inputs,targets,phase)
            assert torch.isfinite(loss); loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in subs.parameters())
            opt.step()
    assert frozen_digest(model)==before
    assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith(PREFIX))
    encoder.p2_transfer_mode='student'
    for p in model.parameters(): p.requires_grad_(True)
    model.train(); model.set_epoch(23); precision24.set_epoch(model,0)
    criterion.train(); criterion.quality_lambda=.25
    with torch.autocast('cuda',enabled=amp):
        out=model(images,targets); losses=criterion(out,targets)
        loss=sum(v*criterion.weight_dict[k] for k,v in losses.items() if k in criterion.weight_dict)
        loss+=QueryBudgetLoss(density_target_backend='vectorized').cuda()(out['allocator_outputs'],
            dict(targets=targets,real_counts=torch.tensor([1,0],device='cuda')))['loss_allocator_total']
    assert torch.isfinite(loss) and out['dn_meta']['pad_size']>0
    loss.backward()
    assert torch.isfinite(subs[0].output.weight.grad).all()
    torch.nn.utils.clip_grad_norm_(model.parameters(),.1)
    optimizer=torch.optim.AdamW(precision24.param_groups(args,model)); optimizer.step()
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',enabled=amp):
        result=model(images)
    predictions=post['bbox'](result,torch.tensor([[65,81],[80,96]],device='cuda'))
    assert all(torch.isfinite(r['boxes']).all() for r in predictions)
