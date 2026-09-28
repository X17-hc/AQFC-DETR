"""F2 fixed-six, complete-epoch interruption and exact-resume contracts."""
import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

from util import precision24 as contract
from util.epoch_boundary import validate_stop, reached_stop, atomic_save, first_epoch_review
from util.experiment import variant_signature
from util.slconfig import SLConfig
from util.config_validation import validate_config


def recipe(name='f2_24e'):
    values=SLConfig.fromfile(f'configs/precision24/{name}.py')._cfg_dict.to_dict()
    validate_config(values)
    return argparse.Namespace(**values)


def toy(args):
    model=nn.Module();model.hidden_dim=16
    model.backbone=nn.Linear(3,16)
    model.transformer=nn.Module();model.transformer.encoder=nn.Module()
    contract.configure(model,args)
    return model


def test_f2_structure_recipe_signature_and_rng():
    a=recipe(); b=recipe('f0_24e')
    assert a.epochs==24 and a.val_epoch==[0,23] and a.save_checkpoint_interval==1
    assert a.lr==3e-5 and a.lr_backbone==3e-6 and a.precision24_new_lr==1e-4
    assert a.quality_blend_warmup_epochs==0 and a.quality_blend_max==.25
    assert a.training_phase_fixed_epoch==23 and a.train_transform_mode=='native_multiscale'
    assert a.geometry_loss_weight==0 and not a.use_ema and a.dn_number==100
    # Only model and evaluation schedule change from the controlled 24e base.
    assert {k for k in vars(b) if getattr(a,k)!=getattr(b,k)}=={'local_refiner_enabled','val_epoch'}
    torch.manual_seed(42);m=toy(a)
    assert hasattr(m,'local_refiner') and not hasattr(m.transformer.encoder,'p2_substitutes')
    assert not hasattr(m.transformer.encoder,'detail_fusion')
    for epoch in [0,1,2,4,16,23]:
        contract.set_epoch(m,epoch);assert m.transformer.encoder.detail_full_layers==6
    sig=variant_signature(a)
    assert sig['precision24_recipe']=='six_full_layers_all_epochs_local_refiner_lr16_22'
    a.stop_after_epochs=1
    assert variant_signature(a)==sig
    a.stop_after_epochs=0
    assert variant_signature(a)==sig
    assert variant_signature(recipe('f1_24e'))!=sig
    groups=contract.param_groups(a,m)
    flat=[id(p) for g in groups for p in g['params']]
    assert len(flat)==len(set(flat))==len(list(m.parameters()))
    assert all(n.startswith('local_refiner.') for g in groups if g['precision24_group']=='new' for n in g['precision24_names'])
    with pytest.raises(ValueError):validate_config(dict(vars(a),precision24_stage_override=6))


def stop_args(**changes):
    return argparse.Namespace(**dict(dict(stop_after_epochs=1,eval=False,max_train_steps=0,
        max_eval_steps=0,debug=False,output_dir='out',start_epoch=0),**changes))


def test_stop_runtime_validation():
    a=stop_args();validate_stop(a);assert reached_stop(a,0)
    a=stop_args(start_epoch=4,stop_after_epochs=2)
    assert not reached_stop(a,4) and reached_stop(a,5)
    assert not reached_stop(stop_args(stop_after_epochs=0),23)
    for change in [dict(stop_after_epochs=-1),dict(max_train_steps=20),dict(max_eval_steps=2),
                   dict(eval=True),dict(debug=True),dict(output_dir='')]:
        with pytest.raises(ValueError):validate_stop(stop_args(**change))


def test_atomic_save_preserves_previous_on_failure(tmp_path):
    p=tmp_path/'checkpoint.pth';atomic_save({'a':torch.tensor(1)},p)
    before=p.read_bytes()
    with patch('torch.save',side_effect=OSError('disk error')):
        with pytest.raises(OSError):atomic_save({'a':torch.tensor(2)},p)
    assert before==p.read_bytes() and not list(tmp_path.glob('*.tmp'))


def test_full_resume_matches_uninterrupted_update(tmp_path):
    from util.checkpoint import capture_rng_state, load_native_resume
    args=recipe();m=toy(args)
    opt=torch.optim.AdamW(contract.param_groups(args,m))
    sched=torch.optim.lr_scheduler.MultiStepLR(opt,[16,22])
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    def step(model,optimizer,epoch,count):
        contract.set_epoch(model,epoch);contract.update_lr(optimizer,args,epoch,count)
        optimizer.zero_grad();loss=sum((p*torch.randn_like(p)).sum() for p in model.parameters())
        loss.backward();optimizer.step()
    step(m,opt,0,0);sched.step()
    cp=dict(model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),
        lr_scheduler=sched.state_dict(),scaler=scaler.state_dict(),epoch=0,args=args,
        variant_signature=variant_signature(args),criterion_progress=dict(successful_updates=1),
        precision24_state=contract.state(m,1),rng_states=[capture_rng_state()],best_metrics={},
        precision24_parameter_groups=[{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups],
        run_metadata=dict(epoch_complete=True,smoke_test=False),evaluation_state='complete')
    path=tmp_path/'checkpoint0000.pth';atomic_save(cp,path)
    step(m,opt,1,1)
    expected_rng=torch.rand(3)
    fresh=toy(args);new_opt=torch.optim.AdamW(contract.param_groups(args,fresh))
    new_sched=torch.optim.lr_scheduler.MultiStepLR(new_opt,[16,22])
    start=load_native_resume(fresh,path,optimizer=new_opt,scheduler=new_sched,scaler=scaler,best_metrics={},expected_args=args)
    assert start==1 and args.quality_successful_updates==1
    step(fresh,new_opt,start,args.quality_successful_updates)
    assert torch.equal(expected_rng,torch.rand(3))
    for x,y in zip(m.parameters(),fresh.parameters()):torch.testing.assert_close(x,y,atol=0,rtol=0)
    assert opt.param_groups[0]['lr']==new_opt.param_groups[0]['lr']
    for variant in ['f1_24e','f0_24e']:
        with pytest.raises(ValueError,match='signature'):
            load_native_resume(fresh,path,optimizer=new_opt,expected_args=recipe(variant))


def test_f2_warmstart_only_allows_refiner(tmp_path):
    from util.incremental_checkpoint import validate_warmstart
    a=recipe();a.expected_pretrained_sha256='';a.pretrain_model_path=str(tmp_path/'source.pth')
    m=toy(a)
    state={k:v for k,v in m.state_dict().items() if not k.startswith('local_refiner.')}
    source=dict(model=state,epoch=2,run_metadata=dict(epoch_complete=True),variant_signature=variant_signature(a))
    atomic_save(source,a.pretrain_model_path);validate_warmstart(m,a)
    m.transformer.encoder.detail_fusion=nn.Linear(2,2)
    with pytest.raises(ValueError,match='coverage'):validate_warmstart(m,a)


def test_review_requires_full_coverage_and_no_large_drop(tmp_path):
    args=stop_args(output_dir=str(tmp_path))
    args.config_file=str(Path('configs/precision24/f2_24e.py').resolve())
    args.coco_path='/data';args.device='cuda';args.amp=True;args.seed=42;args.num_workers=2
    args.max_consecutive_skipped_steps=20
    cp=tmp_path/'checkpoint0000.pth';atomic_save({'epoch':0},cp)
    train=dict(train_iterations=7009,optimizer_steps=7009,amp_skipped_steps=0)
    evaluation=dict(evaluated_images=14018,coco_eval_bbox=[.32,0,0,0,.16])
    r=first_epoch_review(args,train,evaluation,7009,14018,range(14018),cp,Path('/workspace/AQFC-DETR'))
    assert r['status']=='READY_FOR_MANUAL_REVIEW'
    assert (tmp_path/'F2_resume_24e.run.xml').is_file()
    command=r['resume_command']
    assert '--resume' in command and '--pretrained' not in command and '--unique-output-dir' not in command
    assert command[command.index('--stop-after-epochs')+1]=='0'
    for bad in [None,dict(evaluation,coco_eval_bbox=[.1,0,0,0,.16]),dict(evaluation,evaluated_images=32)]:
        folder=tmp_path/str(len(list(tmp_path.iterdir())));folder.mkdir()
        args.output_dir=str(folder)
        r=first_epoch_review(args,train,bad,7009,14018,range(14018),cp,Path('/workspace/AQFC-DETR'))
        assert r['status']=='PAUSE_AND_REVIEW' and not (folder/'F2_resume_24e.run.xml').exists()


def test_f2_launchers():
    import shlex
    import xml.etree.ElementTree as ET
    from main import get_args_parser
    paths=list(Path('.run').glob('AQFC-DETR F2*.run.xml'))
    assert len(paths)==3
    for path in paths:
        c=ET.parse(path).getroot().find('configuration')
        o={v.get('name'):v.get('value') for v in c.findall('option')}
        assert o['WORKING_DIRECTORY']=='/workspace/AQFC-DETR'
        assert o['SDK_HOME']=='/opt/conda/envs/AQFC-DETR/bin/python'
        assert c.find("envs/env[@name='CUDA_VISIBLE_DEVICES']").get('value')=='2'
        if '检查' in path.name:continue
        a=get_args_parser().parse_args(shlex.split(o['PARAMETERS']))
        assert a.unique_output_dir and not a.resume and not a.p2_adapter
        assert a.num_workers==2 and a.amp
        assert a.stop_after_epochs==(0 if '20步' in path.name else 1)
        assert a.max_train_steps==(20 if '20步' in path.name else 0)
        assert a.max_eval_steps==(2 if '20步' in path.name else 0)


@pytest.mark.parametrize('interrupt_eval',[False,True])
def test_main_epoch_boundary_with_simulated_loaders(tmp_path,monkeypatch,interrupt_eval):
    """Exercise actual main control flow; fixture counts are not real-data validation."""
    import main as entry
    from types import SimpleNamespace
    from util import incremental_checkpoint
    args=entry.get_args_parser().parse_args(['--config',str(Path('configs/precision24/f2_24e.py').resolve()),
        '--output-dir',str(tmp_path),'--pretrained',str(tmp_path/'fake_source.pth'),
        '--device','cpu','--no-amp','--stop-after-epochs','1'])
    monkeypatch.setattr(entry.utils,'init_distributed_mode',lambda a: vars(a).update(distributed=False,rank=0))
    monkeypatch.setattr(entry,'write_manifest',lambda *a,**kw:None)
    monkeypatch.setattr(entry,'load_legacy_pretrained',lambda *a,**kw:dict(coverage_by_numel=1.,missing_keys=[],shape_mismatched_keys=[]))
    monkeypatch.setattr(incremental_checkpoint,'validate_warmstart',lambda *a:None)
    monkeypatch.setattr(entry,'get_coco_api_from_dataset',lambda _:None)
    class Sized:
        def __init__(self,n):self.n=n
        def __len__(self):return self.n
    monkeypatch.setattr(entry,'build_data_loaders',lambda a:(Sized(14018),Sized(14018),Sized(7009),Sized(14018),None))
    def build(a):
        c=nn.Module();c.quality_successful_updates=0
        c.geometry_warmup_steps=1;c.geometry_max_weight=0
        return toy(a),c,{}
    monkeypatch.setattr(entry,'build_model_main',build)
    calls=[]
    def train(model,criterion,loader,optimizer,device,epoch,*a,**kw):
        calls.append(epoch);contract.set_epoch(model,epoch)
        sum(p.square().mean() for p in model.parameters()).backward();optimizer.step();optimizer.zero_grad()
        criterion.quality_successful_updates+=1
        return dict(train_iterations=7009,optimizer_steps=7009,amp_skipped_steps=0,training_seconds=.01)
    monkeypatch.setattr(entry,'train_one_epoch',train)
    def evaluate(*a,**kw):
        pending=tmp_path/'checkpoint0000_pending_eval.pth'
        assert pending.is_file()
        cp=torch.load(pending,weights_only=False)
        assert cp['run_metadata']['epoch_complete'] and cp['evaluation_state']=='pending'
        if interrupt_eval:raise RuntimeError('simulated evaluation interruption')
        return dict(coco_eval_bbox=[.32,0,0,0,.16],evaluated_images=14018),SimpleNamespace(
            coco_eval={'bbox':SimpleNamespace(eval={},params=SimpleNamespace(imgIds=list(range(14018))))})
    monkeypatch.setattr(entry,'evaluate',evaluate)
    if interrupt_eval:
        with pytest.raises(RuntimeError,match='simulated'):entry.main(args)
        assert not (tmp_path/'first_epoch_review.json').exists()
        assert not (tmp_path/'F2_resume_24e.run.xml').exists()
    else:
        entry.main(args)
        assert calls==[0]
        cp=torch.load(tmp_path/'checkpoint0000.pth',weights_only=False)
        assert cp['args'].epochs==24 and cp['evaluation_state']=='complete'
        assert cp['precision24_state']['detail_full_layers']==6
        assert json.loads((tmp_path/'first_epoch_review.json').read_text())['status']=='READY_FOR_MANUAL_REVIEW'
