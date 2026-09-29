"""H1 launcher and real main-loop control flow on simulated data only."""
import argparse
import copy
import json
import shlex
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from util import legacy_joint as joint
from util.joint_runtime import SeededDataset,loader_state
from util.slconfig import SLConfig


def recipe(name='h1_epoch0'):
    return argparse.Namespace(**SLConfig.fromfile(f'configs/legacy_joint/{name}.py')._cfg_dict.to_dict())


def test_five_manual_configs():
    from main import get_args_parser
    from tools.benchmark_interleaved import parser as bench_parser
    files=[p for p in Path('.run').glob('*.run.xml') if any(x in p.name for x in ('H1','历史初始化与联合结构'))]
    assert len(files)==5
    for path in files:
        c=ET.parse(path).getroot().find('configuration');options={o.get('name'):o.get('value') for o in c.findall('option')}
        assert options['WORKING_DIRECTORY']=='/workspace/AQFC-DETR'
        assert options['SDK_HOME']=='/opt/conda/envs/AQFC-DETR/bin/python'
        assert options['EMULATE_TERMINAL']=='false'
        assert c.find("envs/env[@name='CUDA_VISIBLE_DEVICES']").get('value')=='2'
        argv=shlex.split(options['PARAMETERS'])
        if '性能' in path.name:
            a=bench_parser().parse_args(argv)
            assert a.comparison=='legacy_joint' and a.order=='joint_control,joint_h1,joint_h1,joint_control'
            assert a.workers==2 and a.warmup==50 and a.steps==200 and not a.profile
        elif '结构检查' not in path.name:
            a=get_args_parser().parse_args(argv)
            assert a.pretrain_model_path.endswith('/weights/legacy/dqdetr_best305.pth') and not a.resume
            assert a.unique_output_dir and a.num_workers==2 and a.amp
            assert a.stop_after_epochs==(1 if 'epoch0' in path.name else 0)
            assert a.max_train_steps==(20 if '20步' in path.name else 0)
            assert a.max_eval_steps==(2 if '20步' in path.name else 0)


def test_exact_joint_checkpoint_resume(tmp_path):
    from util.checkpoint import capture_rng_state,load_native_resume
    from util.experiment import variant_signature
    a=recipe();a.seed=42
    m=nn.Module();m.backbone=nn.Linear(2,2);m.head=nn.Linear(2,2);m.joint_common_sha='common'
    opt=torch.optim.AdamW(joint.parameter_groups(a,m));sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    def step(model,optimizer,epoch,updates):
        joint.update_lr(optimizer,a,epoch,updates);optimizer.zero_grad()
        sum((p*torch.randn_like(p)).square().sum() for p in model.parameters()).backward();optimizer.step()
    step(m,opt,0,10)
    cp=dict(model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),
        lr_scheduler=sched.state_dict(),epoch=0,args=a,variant_signature=variant_signature(a),
        criterion_progress=dict(successful_updates=11),joint_state=joint.checkpoint_state(m,a,opt,0,11),
        rng_states=[capture_rng_state()],joint_data_rng=dict(sampler=torch.Generator().get_state(),workers=torch.Generator().get_state()),
        run_metadata=dict(epoch_complete=True,smoke_test=False),evaluation_state='complete')
    path=tmp_path/'checkpoint.pth';torch.save(cp,path)
    step(m,opt,1,11);expected_rng=torch.rand(3)
    fresh=copy.deepcopy(m);fresh_opt=torch.optim.AdamW(joint.parameter_groups(a,fresh))
    fresh_sched=torch.optim.lr_scheduler.LambdaLR(fresh_opt,lambda _:1.)
    assert load_native_resume(fresh,path,optimizer=fresh_opt,scheduler=fresh_sched,expected_args=a)==1
    assert a.quality_successful_updates==11
    step(fresh,fresh_opt,1,11)
    assert torch.equal(expected_rng,torch.rand(3))
    for x,y in zip(m.parameters(),fresh.parameters()):torch.testing.assert_close(x,y,atol=0,rtol=0)
    wrong=copy.copy(a);wrong.architecture_variant='legacy_joint_control_v2'
    with pytest.raises(ValueError):load_native_resume(fresh,path,optimizer=fresh_opt,expected_args=wrong)


class TinyDataset(torch.utils.data.Dataset):
    def __len__(self):return 4
    def __getitem__(self,i):return i


@pytest.mark.parametrize('mode',['epoch0','smoke','eval_interruption'])
def test_main_h1_boundary_and_smoke(tmp_path,monkeypatch,mode):
    import main as entry
    args=entry.get_args_parser().parse_args(['--config',str(Path('configs/legacy_joint/'+('h1_smoke.py' if mode=='smoke' else 'h1_epoch0.py')).resolve()),
        '--output-dir',str(tmp_path),'--pretrained',str(tmp_path/'source.pth'),'--device','cpu','--no-amp',
        '--stop-after-epochs','0' if mode=='smoke' else '1','--max-train-steps','1' if mode=='smoke' else '0',
        '--max-eval-steps','2' if mode=='smoke' else '0'])
    monkeypatch.setattr(entry.utils,'init_distributed_mode',lambda a:vars(a).update(distributed=False,rank=0))
    monkeypatch.setattr(entry,'write_manifest',lambda *a,**kw:None)
    monkeypatch.setattr(entry,'get_coco_api_from_dataset',lambda _:None)
    dataset=SeededDataset(TinyDataset(),42)
    sampler=torch.utils.data.RandomSampler(dataset,generator=torch.Generator().manual_seed(2))
    loader=torch.utils.data.DataLoader(dataset,batch_sampler=torch.utils.data.BatchSampler(sampler,2,False),
        generator=torch.Generator().manual_seed(3))
    val=SimpleNamespace(ids=list(range(14018)))
    class Sized:
        def __len__(self):return 14018
    val=Sized()
    monkeypatch.setattr(entry,'build_data_loaders',lambda a:(dataset,val,loader,val,sampler))
    def initialize(model,a):
        model.joint_common_sha='simulated'
        return dict(coverage_by_numel=1.,missing_keys=[],shape_mismatched_keys=[])
    monkeypatch.setattr(joint,'initialize',initialize)
    def build(a):
        m=nn.Module();m.layer=nn.Linear(2,2);m.transformer=nn.Module()
        c=nn.Module();c.quality_successful_updates=0;c.geometry_warmup_steps=1;c.geometry_max_weight=0
        return m,c,{}
    monkeypatch.setattr(entry,'build_model_main',build)
    calls=[]
    def train(model,criterion,*a,**kw):
        calls.append(1);criterion.quality_successful_updates+=1
        return dict(train_iterations=1 if mode=='smoke' else 2,optimizer_steps=1 if mode=='smoke' else 2,
                    amp_skipped_steps=0,training_seconds=.01)
    monkeypatch.setattr(entry,'train_one_epoch',train)
    def evaluate(*a,**kw):
        pending=tmp_path/'checkpoint0000_pending_eval.pth'
        if mode!='smoke':assert pending.is_file()
        if mode=='eval_interruption':raise RuntimeError('injected evaluation failure')
        return dict(coco_eval_bbox=[.3,0,0,0,.15],evaluated_images=2 if mode=='smoke' else 14018),SimpleNamespace(
            coco_eval={'bbox':SimpleNamespace(eval={},params=SimpleNamespace(imgIds=list(range(14018))))})
    monkeypatch.setattr(entry,'evaluate',evaluate)
    if mode=='eval_interruption':
        with pytest.raises(RuntimeError,match='injected'):entry.main(args)
        assert not (tmp_path/'checkpoint0000.pth').exists()
    else:
        entry.main(args);assert len(calls)==1
        cp=torch.load(tmp_path/('checkpoint.pth' if mode=='smoke' else 'checkpoint0000.pth'),weights_only=False)
        assert cp['args'].epochs==24 and cp['joint_state']['epoch']==0
        assert cp['run_metadata']['epoch_complete']==(mode!='smoke')
        assert cp['evaluation_state']==('partial' if mode=='smoke' else 'complete')


def test_benchmark_summary_step_overhead_not_throughput():
    from tools.benchmark_control import summarize_joint
    r=summarize_joint([dict(variant='joint_control',images_per_second=3.),dict(variant='joint_h1',images_per_second=2.5)])
    assert r['h1_step_time_overhead_percent']==pytest.approx(20.)
    assert r['h1_throughput_penalty_percent']==pytest.approx(100/6)
