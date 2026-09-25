import json
import shlex
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import diagnose_scale_s2 as tool
import diagnose_scale_pair as pair

def test_xml_and_eval_only():
    from util.slconfig import SLConfig
    from main import get_args_parser
    x=ET.parse(ROOT/'.run/AQFC-DETR C0与S2同图同GT的VT定向诊断.run.xml').getroot().find('configuration')
    options={a.get('name'):a.get('value') for a in x.findall('option')}
    args=tool.parser().parse_args(shlex.split(options['PARAMETERS']))
    assert args.s2_run.name=='20260924_115851_063163_8e60e76c05c3'
    assert args.reuse_pair.name=='20260924T093854_91c0d93f'
    assert options['WORKING_DIRECTORY']=='/workspace/AQFC-DETR'
    assert args.output_dir.name=='c0_s2_vt_diagnosis'
    cfg=[SLConfig.fromfile(str(ROOT/p))._cfg_dict.to_dict() for p in [pair.CONFIGS['C0'],tool.S2_CONFIG]]
    m=[dict(initialization=dict(sha256=pair.INITIAL_SHA),annotations=[]) for _ in cfg]
    pair.validate_pair(cfg,m,after_mode='native_multiscale')
    with pytest.raises(ValueError): pair.validate_pair(cfg,m)
    cmd=pair.eval_command(ROOT,ROOT/tool.S2_CONFIG,Path('ckpt'),Path('data'),Path('ids'),Path('out'))[3:]
    parsed=get_args_parser().parse_args(cmd)
    assert parsed.eval and parsed.no_pretrained and not parsed.eval_ema
    assert parsed.num_workers==0 and parsed.max_eval_steps==0 and parsed.amp

def test_targeted_s2_reuses_c0_without_copy(tmp_path):
    from util.paired_head_errors import analyze_image
    gt=dict(id=7,image_id=1,category_id=5,area=64,bbox=[0,0,8,8])
    _,features,_=analyze_image([dict(category_id=5,score=.9,bbox=[0,0,8,8])],[gt])
    row=dict(image_id=1,annotation_id=7,category_id=5,area=64,epoch0=features[0],epoch1=features[0])
    (tmp_path/'paired').mkdir(); (tmp_path/'paired/paired_gt.jsonl').write_text(json.dumps(row)+'\n')
    old=tmp_path/'existing'; old.mkdir(); (tmp_path/'S2').mkdir()
    p=dict(image_id=1,ground_truth=[dict(annotation_id=7,best_proposal_iou=.6)])
    for folder in [old,tmp_path/'S2']: (folder/'proposals.jsonl').write_text(json.dumps(p)+'\n')
    before=(old/'proposals.jsonl').read_bytes()
    pair.targeted_summary(tmp_path,dict(base_ids=[1],added_ids=[],rare_gt_ids=[],images=1,gt=1),after_label='S2',c0_root=old)
    result=pair.read(tmp_path/'targeted_summary.json')
    assert result['labels']==dict(epoch0='C0',epoch1='S2')
    assert result['groups']['official_vt/class:5']['gt']==1
    assert result['proposal_coverage']['union']['S2_hits050']==1
    assert 'S1_hits050' not in result['proposal_coverage']['union']
    assert (old/'proposals.jsonl').read_bytes()==before and not (tmp_path/'C0').exists()

def test_incomplete_reuse_rejected(tmp_path):
    from types import SimpleNamespace
    (tmp_path/'pair_manifest.json').write_text(json.dumps(dict(status='failed')))
    with pytest.raises(ValueError,match='not complete'):
        tool.check_reuse(SimpleNamespace(reuse_pair=tmp_path),[],{}, {})
