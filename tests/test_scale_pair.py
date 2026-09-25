import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import shlex
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import diagnose_scale_pair as pair


def fixture():
    return dict(images=[{'id':i} for i in range(4)],annotations=[
        dict(id=1,image_id=0,category_id=5,area=20),
        dict(id=2,image_id=1,category_id=0,area=64),
        dict(id=3,image_id=2,category_id=7,area=65),
        dict(id=4,image_id=3,category_id=1,area=4,ignore=1)])


def test_selection_exact_boundary_and_no_ignore():
    ids,s=pair.select_images(fixture(),[0])
    assert ids==[0,1] and s['added_ids']==[1] and s['rare_gt_ids']==[2]
    assert pair.select_images(fixture(),[0])==(ids,s)


@pytest.mark.parametrize('ids',[[0,0],[99],[],[True]])
def test_invalid_ids(ids):
    with pytest.raises(ValueError): pair.select_images(fixture(),ids)


def test_configs_and_evaluation_only_xml():
    from util.slconfig import SLConfig
    from main import get_args_parser
    configs=[SLConfig.fromfile(str(ROOT/p))._cfg_dict.to_dict() for p in pair.CONFIGS.values()]
    manifests=[dict(initialization=dict(sha256=pair.INITIAL_SHA),annotations=[]) for _ in range(2)]
    pair.validate_pair(configs,manifests)
    configs[1]['geometry_loss_weight']=.1
    with pytest.raises(ValueError): pair.validate_pair(configs,manifests)
    xml=ET.parse(ROOT/'.run/AQFC-DETR C0与S1同图同GT定向诊断.run.xml').getroot().find('configuration')
    options={x.attrib['name']:x.attrib['value'] for x in xml.findall('option')}
    args=pair.parser().parse_args(shlex.split(options['PARAMETERS']))
    assert options['WORKING_DIRECTORY']=='/workspace/AQFC-DETR'
    assert args.s1_run.name=='20260924_022157_677735_077b90195559'
    cmd=pair.eval_command(ROOT,ROOT/pair.CONFIGS['S1'],Path('checkpoint'),Path('data'),Path('ids'),Path('out'))[3:]
    parsed=get_args_parser().parse_args(cmd)
    assert parsed.eval and parsed.no_pretrained and not parsed.eval_ema
    assert parsed.max_eval_steps==0 and parsed.num_workers==0 and parsed.amp


def test_proposal_observer_does_not_modify_outputs():
    boxes=torch.tensor([[[.5,.5,.2,.2],[.2,.2,.1,.1]]])
    outputs=dict(query_valid_mask=torch.tensor([[True,False]]),executed_query_counts=torch.tensor([1]),
                 interm_outputs=dict(pred_boxes=boxes))
    before=boxes.clone()
    record=pair.proposal_record(outputs,dict(id=1,width=100,height=50),
                [dict(id=7,category_id=0,area=200,bbox=[40,20,20,10])])
    assert record['executed_query_count']==1 and len(record['proposals'])==1
    assert record['ground_truth'][0]['best_proposal_iou']==pytest.approx(1,abs=1e-6)
    assert torch.equal(boxes,before)


def test_targeted_join_separates_cohorts_and_includes_area64(tmp_path):
    from util.paired_head_errors import analyze_image
    gt=dict(id=7,image_id=1,category_id=0,area=64,bbox=[0,0,8,8])
    _,features,_=analyze_image([dict(category_id=0,score=.9,bbox=[0,0,8,8])],[gt])
    row=dict(image_id=1,annotation_id=7,category_id=0,area=64,epoch0=features[0],epoch1=features[0])
    (tmp_path/'paired').mkdir()
    (tmp_path/'paired/paired_gt.jsonl').write_text(json.dumps(row)+'\n')
    for label in ('C0','S1'):
        (tmp_path/label).mkdir()
        p=dict(image_id=1,ground_truth=[dict(annotation_id=7,best_proposal_iou=.6)])
        (tmp_path/label/'proposals.jsonl').write_text(json.dumps(p)+'\n')
    sampling=dict(base_ids=[],added_ids=[1],rare_gt_ids=[7],images=1,gt=1)
    pair.targeted_summary(tmp_path,sampling)
    result=pair.read(tmp_path/'targeted_summary.json')
    assert result['groups']['official_vt/class:0']['gt']==1
    assert result['groups']['added_images']['gt']==1
    assert result['proposal_coverage']['union']['S1_hits050']==1
