"""S2 synthetic contracts; no real data or GPU training."""
import copy
import random
import shlex
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import pytest
import torch
from PIL import Image
from tests.test_scale_v1 import cfg, target, ROOT
from datasets.coco import make_coco_transforms
from datasets.native_resize import NativeMultiScaleResize
from datasets import transforms as T
from util.config_validation import validate_config
from util.experiment import variant_signature
from util.box_ops import box_cxcywh_to_xyxy
from util.checkpoint import load_native_resume

def test_recipe():
    a=cfg('scale_v1/s1_3e.py'); b=cfg('scale_v1/s2_3e.py')
    assert a.pop('train_transform_mode')=='native800'
    assert b.pop('train_transform_mode')=='native_multiscale' and a==b
    assert validate_config(cfg('scale_v1/s2_3e.py'))
    assert variant_signature(dict(train_transform_mode='native800')) != variant_signature(dict(train_transform_mode='native_multiscale'))
    smoke=cfg('scale_v1/s2_smoke.py')
    assert smoke['epochs']==1 and smoke['val_epoch']==[0]

@pytest.mark.parametrize('short',[640,704,768,800])
@pytest.mark.parametrize('empty',[False,True])
def test_coordinates(short,empty,monkeypatch):
    monkeypatch.setattr(random,'choice',lambda sizes: short)
    for size in [(101,63),(2001,801),(63,101)]:
        w,h=size; gt=target(w,h,empty); before=copy.deepcopy(gt)
        pipeline=make_coco_transforms('train',args=SimpleNamespace(train_transform_mode='native_multiscale'))
        assert [type(x) for x in pipeline.transforms]==[T.RandomHorizontalFlip,NativeMultiScaleResize,T.Compose]
        random.seed(2)
        image,out=pipeline(Image.new('RGB',size),gt)
        oh,ow=image.shape[-2:]; assert max(oh,ow)<=1333 and min(oh,ow)<=short
        assert abs(ow/oh-w/h)<.01
        restored=box_cxcywh_to_xyxy(out['boxes'])*torch.tensor([w,h,w,h])
        torch.testing.assert_close(restored,gt['boxes'],atol=1e-3,rtol=1e-5)
        assert torch.equal(out['labels'],gt['labels'])
        for key in before: assert torch.equal(gt[key],before[key])

@pytest.mark.parametrize('flag',['fix_size','strong_aug'])
def test_conflicts(flag):
    a=cfg('scale_v1/s2_3e.py'); a[flag]=True
    with pytest.raises(ValueError,match='native_multiscale'): validate_config(a)
    with pytest.raises(ValueError,match='native_multiscale'):
        make_coco_transforms('train',args=SimpleNamespace(**a))

def test_eval_unchanged():
    results=[]
    for mode in ('legacy','native800','native_multiscale'):
        random.seed(42)
        results.append(make_coco_transforms('test',args=SimpleNamespace(train_transform_mode=mode))(Image.new('RGB',(107,71)),target(107,71)))
    for image,gt in results[1:]:
        torch.testing.assert_close(image,results[0][0],atol=0,rtol=0)
        for k in gt: torch.testing.assert_close(gt[k],results[0][1][k],atol=0,rtol=0)

def test_resume_rejects_cross_mode(tmp_path):
    model=torch.nn.Linear(2,1); optimizer=torch.optim.AdamW(model.parameters())
    for old in ('legacy','native800','native_multiscale'):
        path=tmp_path/'model.pth'
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=0,
                        variant_signature=variant_signature(dict(train_transform_mode=old))),path)
        args=dict(train_transform_mode='native_multiscale')
        if old=='native_multiscale':
            assert load_native_resume(model,path,optimizer=optimizer,expected_args=args)==1
        else:
            with pytest.raises(ValueError,match='signature'):
                load_native_resume(model,path,optimizer=optimizer,expected_args=args)

def test_xml():
    paths=list((ROOT/'.run').glob('*S2*.run.xml')); assert len(paths)==2
    for p in paths:
        c=ET.parse(p).getroot().find('configuration')
        o={x.get('name'):x.get('value') for x in c.findall('option')}
        e={x.get('name'):x.get('value') for x in c.findall('envs/env')}
        assert o['WORKING_DIRECTORY']=='/workspace/AQFC-DETR'
        assert o['SDK_HOME']=='/opt/conda/envs/AQFC-DETR/bin/python'
        assert e['CUDA_VISIBLE_DEVICES']=='0'
        a=shlex.split(o['PARAMETERS']); smoke='20步' in p.name
        assert '--resume' not in a and '--unique-output-dir' in a
        assert a[a.index('--max-train-steps')+1]==('20' if smoke else '0')
        assert a[a.index('--max-eval-steps')+1]==('2' if smoke else '0')
        assert a[a.index('--output-dir')+1].endswith('/s2_smoke' if smoke else '/s2')
