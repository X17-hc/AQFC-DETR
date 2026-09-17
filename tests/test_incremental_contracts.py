"""Resume, data audit and actual CLI contracts for the incremental release."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest
import torch
from util.experiment import variant_signature
from util.checkpoint import load_native_resume
from util.incremental_checkpoint import validate_warmstart


def test_quality_resume_roundtrip_and_cross_variant_rejected(tmp_path):
    model=torch.nn.Linear(2,1); optimizer=torch.optim.AdamW(model.parameters())
    args=argparse.Namespace(classification_loss_type='quality_blend',training_phase_epoch_offset=11,
                            training_phase_total_epochs=24)
    source=tmp_path/'native.pth'
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=0,
                    variant_signature=variant_signature(args),criterion_progress={'successful_updates':71}),source)
    assert load_native_resume(model,source,optimizer=optimizer,expected_args=args)==1
    assert args.quality_successful_updates==71
    args.classification_loss_type='focal'
    with pytest.raises(ValueError,match='signature'):
        load_native_resume(model,source,optimizer=optimizer,expected_args=args)


def test_warmstart_rejects_incomplete_and_mismatched_hash(tmp_path):
    model=torch.nn.Linear(2,1); source=tmp_path/'source.pth'
    torch.save(dict(model=model.state_dict(),epoch=10,run_metadata={'epoch_complete':False}),source)
    args=argparse.Namespace(pretrain_model_path=str(source),expected_pretrained_epoch=10)
    with pytest.raises(ValueError,match='complete'): validate_warmstart(model,args)
    args.expected_pretrained_sha256='0'*64
    with pytest.raises(ValueError,match='SHA-256'): validate_warmstart(model,args)


def test_pycharm_real_parser_and_config_no_conflicts():
    from main import get_args_parser
    from util.slconfig import SLConfig
    from util.config_validation import validate_config
    import shlex
    root=Path(__file__).resolve().parents[1]
    paths=list((root/'.run').glob('AQFC-DETR_Incremental*.xml'))
    assert len(paths)==6
    for path in paths:
        doc=ET.parse(path); options={x.get('name'):x.get('value') for x in doc.findall('.//option')}
        assert options['WORKING_DIRECTORY']=='/workspace/AQFC-DETR'
        assert doc.find('.//env[@name="CUDA_VISIBLE_DEVICES"]').get('value')=='2'
        if options['SCRIPT_NAME'].endswith('/main.py'):
            argv=shlex.split(options['PARAMETERS'].replace('/workspace/AQFC-DETR',root.as_posix()))
            args=get_args_parser().parse_args(argv)
            cfg=SLConfig.fromfile(args.config_file)
            if args.options: cfg.merge_from_dict(args.options)
            values=cfg._cfg_dict.to_dict(); validate_config(values)
            assert not {k for k in values if k in vars(args) and vars(args)[k] is not None}


def test_data_audit_records_missing_image_without_substitution(tmp_path):
    from tools.audit_incremental_data import audit
    ann=tmp_path/'annotations'; ann.mkdir()
    data=dict(images=[dict(id=1,file_name='absent.jpg',width=8,height=8)],
        categories=[dict(id=i,name=str(i)) for i in range(8)],annotations=[])
    for split in ('train','val','trainval','test'):
        (ann/f'aitodv2_{split}.json').write_text(json.dumps(data))
    report=audit(tmp_path,'aitodv2')
    assert report['status']=='blocked'
    errors=[e for e in report['issues'] if e['error']=='unreadable_or_size_mismatch']
    assert len(errors)==4 and all(e['image_id']==1 for e in errors)


def test_data_audit_split_local_ids_and_nonfinite_boxes(tmp_path):
    from PIL import Image
    from tools.audit_incremental_data import audit
    ann=tmp_path/'annotations'; ann.mkdir()
    for split,names in {'train':['a.jpg'],'val':['b.jpg'],
                        'trainval':['a.jpg','b.jpg'],'test':['c.jpg']}.items():
        folder=tmp_path/'images'/split/'images'; folder.mkdir(parents=True)
        for name in names: Image.new('RGB',(8,8)).save(folder/name)
        data=dict(images=[dict(id=i,file_name=name,width=8,height=8) for i,name in enumerate(names)],
                  categories=[dict(id=i) for i in range(8)],annotations=[])
        (ann/f'aitodv2_{split}.json').write_text(json.dumps(data))
    result=audit(tmp_path,'aitodv2')
    assert result['status']=='passed'
    assert result['split_relationships']==dict(train_val_overlap=0,trainval_test_overlap=0,trainval_equals_union=True)
    path=ann/'aitodv2_train.json'; data=json.loads(path.read_text())
    data['annotations']=[dict(id=1,image_id=0,category_id=0,bbox=[0,0,float('nan'),1])]
    path.write_text(json.dumps(data))
    result=audit(tmp_path,'aitodv2')
    assert result['status']=='blocked'
    assert any(x['error']=='malformed_or_nonfinite_box' for x in result['issues'])
