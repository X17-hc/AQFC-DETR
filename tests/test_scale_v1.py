"""S1 contracts, using synthetic images and checkpoints only."""
import copy
import random
import shlex
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest
import torch
from PIL import Image
from datasets.coco import make_coco_transforms
from datasets import transforms as T
from datasets.native_resize import Native800Resize
from util.box_ops import box_cxcywh_to_xyxy
from util.misc import nested_tensor_from_tensor_list
from util.experiment import variant_signature, original_variant
from util.config_validation import validate_config
from util.checkpoint import load_native_resume
from util.slconfig import SLConfig

ROOT = Path(__file__).resolve().parents[1]

def cfg(name):
    return SLConfig.fromfile(str(ROOT / 'configs' / name))._cfg_dict.to_dict()

def target(w, h, empty=False):
    boxes = torch.tensor([[0., 0., float(w), float(h)], [w-3., h-2., float(w), float(h)]])
    if empty:
        boxes = boxes[:0]
    return dict(boxes=boxes, labels=torch.arange(len(boxes)),
                area=(boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1]),
                size=torch.tensor([h,w]), orig_size=torch.tensor([h,w]),
                image_id=torch.tensor([7]))

def test_config_only_changes_mode():
    baseline=cfg('geometry_v1/c0_3e.py'); formal=cfg('scale_v1/s1_3e.py')
    assert formal.pop('train_transform_mode') == 'native800'
    assert formal == baseline
    formal['train_transform_mode']='native800'
    assert validate_config(formal)
    smoke=cfg('scale_v1/s1_smoke.py')
    assert smoke.pop('epochs') == 1 and smoke.pop('val_epoch') == [0]
    formal.pop('epochs'); formal.pop('val_epoch')
    assert smoke == formal

@pytest.mark.parametrize('field', ['fix_size','strong_aug'])
def test_conflicts_rejected(field):
    values=cfg('scale_v1/s1_3e.py'); values[field]=True
    with pytest.raises(ValueError,match='native800'):
        validate_config(values)
    with pytest.raises(ValueError,match='native800'):
        make_coco_transforms('train',args=SimpleNamespace(train_transform_mode='native800'),**{field:True})

def test_invalid_mode():
    with pytest.raises(ValueError,match='train_transform_mode'):
        make_coco_transforms('train',args=SimpleNamespace(train_transform_mode='typo'))

@pytest.mark.parametrize('size',[(101,63),(2001,801),(63,101)])
@pytest.mark.parametrize('empty',[False,True])
def test_native_coordinates_labels_aspect_padding(size,empty):
    w,h=size; source=target(w,h,empty); before=copy.deepcopy(source)
    transform=make_coco_transforms('train',args=SimpleNamespace(train_transform_mode='native800'))
    assert [type(t) for t in transform.transforms] == [T.RandomHorizontalFlip,Native800Resize,T.Compose]
    assert transform.transforms[1].sizes == [800] and transform.transforms[1].max_size == 1333
    random.seed(2)  # no flip; test exact normalized coordinate restoration
    image, out=transform(Image.new('RGB',size),source)
    oh,ow=image.shape[-2:]
    assert max(oh,ow)<=1333 and min(oh,ow)<=800
    assert abs(ow/oh-w/h)<.01
    restored=box_cxcywh_to_xyxy(out['boxes'])*torch.tensor([w,h,w,h])
    torch.testing.assert_close(restored,source['boxes'],atol=1e-3,rtol=1e-5)
    assert torch.equal(out['labels'],source['labels'])
    assert out['size'].tolist()==[oh,ow]
    for k in before: assert torch.equal(source[k],before[k])
    nested=nested_tensor_from_tensor_list([image,image[:,:oh-1,:ow-1]])
    assert not nested.mask[0].any() and nested.mask[1,-1].all() and nested.mask[1,:,-1].all()

@pytest.mark.parametrize('split',['train','test'])
def test_legacy_unchanged_and_eval_mode_independent(split):
    image=Image.new('RGB',(107,71)); gt=target(107,71)
    args=[SimpleNamespace(),SimpleNamespace(train_transform_mode='legacy')]
    if split=='test': args.append(SimpleNamespace(train_transform_mode='native800'))
    outputs=[]
    for a in args:
        random.seed(42); torch.manual_seed(42)
        outputs.append(make_coco_transforms(split,args=a)(image,copy.deepcopy(gt)))
    for x,y in outputs[1:]:
        torch.testing.assert_close(x,outputs[0][0],rtol=0,atol=0)
        for k in y: torch.testing.assert_close(y[k],outputs[0][1][k],rtol=0,atol=0)

def test_legacy_signature_and_unsigned_guard():
    assert variant_signature({})==variant_signature(dict(train_transform_mode='legacy'))
    assert 'train_transform_mode' not in variant_signature({})
    assert original_variant({})
    assert not original_variant(dict(train_transform_mode='native800'))

@pytest.mark.parametrize('old,new', [('legacy','legacy'),('native800','native800'),('legacy','native800'),('native800','legacy'),(None,'native800')])
def test_resume_modes(tmp_path,old,new):
    model=torch.nn.Linear(2,1); optimizer=torch.optim.AdamW(model.parameters())
    state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=0)
    if old is not None: state['variant_signature']=variant_signature(dict(train_transform_mode=old))
    path=tmp_path/'checkpoint.pth'; torch.save(state,path)
    args=dict(train_transform_mode=new)
    if old!=new:
        with pytest.raises(ValueError,match='signature'):
            load_native_resume(model,path,optimizer=optimizer,expected_args=args)
    else:
        assert load_native_resume(model,path,optimizer=optimizer,expected_args=args)==1

def test_run_configurations():
    paths=list((ROOT/'.run').glob('*S1*.run.xml')); assert len(paths)==2
    for path in paths:
        config=ET.parse(path).getroot().find('configuration')
        options={x.attrib['name']:x.attrib['value'] for x in config.findall('option')}
        env={x.attrib['name']:x.attrib['value'] for x in config.findall('envs/env')}
        assert options['WORKING_DIRECTORY']=='/workspace/AQFC-DETR'
        assert options['SDK_HOME']=='/opt/conda/envs/AQFC-DETR/bin/python'
        assert env['CUDA_VISIBLE_DEVICES']=='2' and env['PYTHONUNBUFFERED']=='1'
        argv=shlex.split(options['PARAMETERS']); smoke='20步' in path.name
        assert '--resume' not in argv and '--unique-output-dir' in argv
        assert argv[argv.index('--max-train-steps')+1]==('20' if smoke else '0')
        assert argv[argv.index('--max-eval-steps')+1]==('2' if smoke else '0')
        assert argv[argv.index('--pretrained')+1].endswith('/resume_epoch11_20260916/checkpoint0023.pth')
        assert argv[argv.index('--output-dir')+1].endswith('/s1_smoke' if smoke else '/s1')
