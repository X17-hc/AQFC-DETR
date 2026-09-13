"""Hermetic fixtures: no model, dataset, CUDA or optional analysis packages needed."""
import copy
import json
from argparse import Namespace
from pathlib import Path
import numpy as np
import pytest
from util.paired_head_errors import analyze_image, add_pair, new_group, iou_bin, MODES, FP_TYPES
from util.error_analysis import analyze, box_iou_xywh
from tools.analyze_paired_head_errors import run, sha256


def gt(i=1, c=0, box=None):
    return dict(id=i,image_id=1,category_id=c,bbox=box or [0,0,10,10],area=100)


def pred(c=0, score=.8, box=None):
    return dict(image_id=1,category_id=c,bbox=box or [0,0,10,10],score=score)


def test_empty_inputs():
    assert analyze_image([],[]) == ([],[],[])
    _,f,_=analyze_image([],[gt()])
    assert f[0]['best_same_rank'] is None and f[0]['thresholds']['75']['geometry_insufficient']
    _,_,p=analyze_image([pred()],[])
    assert all(v['label']=='background' for v in p[0]['modes'].values())


@pytest.mark.parametrize('threshold,width,mode',[(.5,5,'all_50'),(.75,7.5,'all_75')])
def test_inclusive_iou_threshold(threshold,width,mode):
    _,f,p=analyze_image([pred(box=[0,0,width,10])],[gt()])
    assert f[0]['best_same_iou']==threshold
    assert p[0]['modes'][mode]['label']=='tp'


def test_iou_bins_and_score_inclusive():
    assert [iou_bin(v) for v in [0,.1,.5,.75,1]]==[0,1,2,3,3]
    _,_,p=analyze_image([pred(score=.25)],[gt()])
    assert p[0]['modes']['score025_75']['label']=='tp'


def test_score_and_annotation_ties_and_competition():
    ground=[gt(9),gt(2)]
    g,f,p=analyze_image([pred() ],ground)
    assert [x['id'] for x in g]==[2,9]
    assert p[0]['modes']['all_75']['matched_gt_id']==2
    assert f[1]['thresholds']['75']['covered']
    assert f[1]['matching']['all_75']['fn_matching_competition']
    _,_,ps=analyze_image([pred(),pred()],[gt()])
    assert [p['prediction_index'] for p in ps]==[0,1]
    assert [p['modes']['all_75']['label'] for p in ps]==['tp','duplicate']


def test_low_score_and_bad_ranking_intersection():
    _,f,_=analyze_image([pred(score=.8,box=[0,0,4,10]),pred(score=.2)],[gt()])
    t=f[0]['thresholds']['75']
    assert t['covered'] and t['low_score'] and t['ranking_inconsistent'] and t['low_and_ranking']
    assert f[0]['best_same_rank']==2 and f[0]['best_same_class_rank']==2
    assert f[0]['matching']['all_75']['matched']
    assert not f[0]['matching']['score025_75']['matched']


@pytest.mark.parametrize('p,expected',[
    (pred(c=1),'class_confusion'),
    (pred(box=[0,0,4,10]),'localization'),
    (pred(c=1,box=[0,0,4,10]),'class_and_localization'),
    (pred(box=[30,30,10,10]),'background')])
def test_exclusive_fp_types(p,expected):
    _,_,ps=analyze_image([p],[gt()])
    assert ps[0]['modes']['all_75']['label']==expected


def test_top300_is_score_rank_not_query_id():
    records=[pred(score=.9,box=[30,30,10,10]) for _ in range(300)]+[pred(score=.8)]
    _,f,_=analyze_image(records,[gt()])
    assert f[0]['thresholds']['75']['covered']
    assert not f[0]['thresholds']['75']['covered_top300']


def test_geometry_errors():
    _,f,_=analyze_image([pred(box=[2,1,20,5])],[gt()])
    e=f[0]['geometry']
    assert e['center_dx']==.7 and e['center_dy']==-.15
    assert e['width_ratio']==2 and e['height_ratio']==.5 and e['aspect_ratio_relative_error']==3


def test_identity_and_determinism():
    records=[pred(),pred(c=1),pred(score=.2)]
    ground=[gt(),gt(2,box=[20,20,10,10])]
    first=analyze_image(records,ground)
    assert first==analyze_image(copy.deepcopy(records),copy.deepcopy(ground))
    group=new_group()
    for f in first[1]: add_pair(group,f,f)
    assert all(v==0 for k,v in group['paired'].items() if 'both_covered' not in k)
    assert sum(group['transitions'][i][i] for i in range(4))==len(ground)


def test_matching_matches_existing_reference():
    rng=np.random.default_rng(42)
    ground=[gt(i+1,i%2,[float(i*4),0,10,10]) for i in range(8)]
    predictions=[pred(i%2,float(rng.random()),[float(rng.uniform(0,35)),0,10,10]) for i in range(40)]
    ann=dict(categories=[dict(id=i,name=str(i)) for i in range(2)],annotations=ground)
    _,features,ps=analyze_image(predictions,ground)
    for mode,(cutoff,t) in MODES.items():
        reference=analyze(predictions,ann,[1],confidence=0 if cutoff is None else cutoff,iou_threshold=t)
        tp=sum(c['tp'] for c in reference['category_counts'].values())
        assert sum(p['modes'][mode]['label']=='tp' for p in ps)==tp
        assert sum(f['matching'][mode]['matched'] for f in features)==tp
        assert sum(p['modes'][mode]['label'] in FP_TYPES for p in ps)==sum(c['fp'] for c in reference['category_counts'].values())


def fixture(tmp_path):
    ann=tmp_path/'annotation.json'; ids=tmp_path/'ids.json'
    ann.write_text(json.dumps(dict(images=[dict(id=1,width=100,height=100)],
        categories=[dict(id=0,name='object')],annotations=[gt()])))
    ids.write_text('[1]')
    export=tmp_path/'export';export.mkdir()
    (export/'predictions.json').write_text(json.dumps([pred()]))
    (export/'metadata.json').write_text(json.dumps(dict(categories=[dict(id=0,name='object')],image_ids=[1])))
    (export/'images.jsonl').write_text(json.dumps(dict(image_id=1,width=100,height=100,
        gt_count=1,executed_query_count=1,query_mask_count=1,raw_query_counts=300))+'\n')
    return Namespace(before=export,after=export,annotations=ann,image_ids=ids,output=tmp_path/'out',
                     image_root=None,expected_images=1,expected_gt=1)


def test_end_to_end_repeat_inputs_unchanged_and_no_overwrite(tmp_path):
    args=fixture(tmp_path)
    before={str(p):sha256(p) for p in tmp_path.rglob('*.json*')}
    first=run(args)
    args.output=tmp_path/'out2'
    second=run(args)
    assert first==second
    for name in ['paired_gt.jsonl','prediction_errors.jsonl','summary.json']:
        assert (tmp_path/'out'/name).read_bytes()==(tmp_path/'out2'/name).read_bytes()
    assert all(sha256(Path(p))==h for p,h in before.items())
    with pytest.raises(FileExistsError): run(args)


@pytest.mark.parametrize('fault',['duplicate_ids','mask','category','nan','count','crowd','duplicate_gt'])
def test_input_fail_closed(tmp_path,fault):
    args=fixture(tmp_path)
    if fault=='duplicate_ids': args.image_ids.write_text('[1,1]')
    elif fault in ['crowd','duplicate_gt']:
        ann=json.loads(args.annotations.read_text())
        if fault=='crowd': ann['annotations'][0]['iscrowd']=1
        else: ann['annotations'].append(copy.deepcopy(ann['annotations'][0]))
        args.annotations.write_text(json.dumps(ann))
    elif fault=='mask':
        p=args.before/'images.jsonl';r=json.loads(p.read_text());r['query_mask_count']=0;p.write_text(json.dumps(r))
    else:
        ps=[pred()]
        if fault=='category': ps[0]['category_id']=99
        if fault=='nan': ps[0]['score']=float('nan')
        if fault=='count': ps=[]
        (args.before/'predictions.json').write_text(json.dumps(ps))
    with pytest.raises(ValueError): run(args)
    assert not args.output.exists()
