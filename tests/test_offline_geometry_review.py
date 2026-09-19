import importlib.util
from pathlib import Path
import pytest

source=Path(__file__).with_name('offline_geometry_review.py')
if not source.exists(): source=Path(__file__).resolve().parents[1]/'tools/offline_geometry_review.py'
spec=importlib.util.spec_from_file_location('geometry_review',source)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def record(box,category=1,score=.9): return dict(bbox=box,category_id=category,score=score)

def test_empty_and_nonoverlap():
    assert m.best_matches([],[])==[]
    r=m.best_matches([], [record([0,0,2,2])])[0]
    assert r['best_iou']==0 and r['error'] is None and r['prediction_index'] is None
    r=m.best_matches([record([10,10,2,2])],[record([0,0,2,2])])[0]
    assert r['error'] is None and r['prediction_index'] is None

def test_class_and_tie_order():
    gt=[record([0,0,2,2])]
    pred=[record([0,0,2,2],2,.1),record([0,0,2,2],1,.8),record([0,0,2,2],1,.9)]
    assert m.best_matches(pred,gt)[0]['prediction_index']==0
    assert m.best_matches(pred,gt,True)[0]['prediction_index']==1
    assert m.best_matches(pred,gt,True,1)==m.best_matches(pred,gt,True,256)

def test_relative_geometry_and_score_not_filtered():
    r=m.best_matches([record([-1,-1,4,4],score=.0001)], [record([0,0,2,2])])[0]
    assert r['best_iou']==.25
    assert r['error']['width_ratio']==2 and r['error']['dx_relative']==0
    assert r['score']==.0001

def test_threshold_inclusive_and_many_to_one():
    gt=[record([0,0,2,2]),record([0,0,2,2])]
    r=m.best_matches([record([0,0,1,2])],gt)
    assert len(r)==2 and all(x['best_iou']==.5 for x in r)
    rows=[dict(image_id=1,any_class=x) for x in r]
    s=m.aggregate(rows,'any_class')
    assert s['hit50']==2 and s['neighbor_iou_ge_0.5']['gt_support']==2

@pytest.mark.parametrize('box',[[0,0,0,1],[0,0,1,-1],[0,float('nan'),1,1],[0,0,1]])
def test_invalid_boxes(box):
    with pytest.raises(ValueError): m.validate_box(box)

def test_ids_invalid():
    for ids in ([1,1],[1,3],[1],['1',2]):
        with pytest.raises(ValueError): m.validate_ids(ids,[1,2],2)
    m.validate_ids([1,2],[1,2,3],2)

def test_empty_category_and_image_balancing():
    r=m.best_matches([record([0,0,2,2])],[record([0,0,2,2])])[0]
    row=dict(image_id=1,category_id=1,area=4,image_gt_count=1,query_count=300,in_precheck32=True,any_class=r,same_class=r)
    s=m.summarize([row],{0:'absent',1:'present'})
    assert s['category/0']['any_class']['gt']==0
    assert s['category/0']['any_class']['neighbor_iou_ge_0.1']['measures']['width_ratio']['median'] is None
    assert s['cohort_size/original32/verytiny']['any_class']['gt']==1
    assert s['all']['any_class']['neighbor_iou_ge_0.1']['measures']['image_balanced_medians']['width_ratio']['median']==1

def test_no_model_dependency():
    import ast
    tree=ast.parse(source.read_text(encoding='utf-8'))
    roots=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): roots.extend(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node,ast.ImportFrom): roots.append((node.module or '').split('.')[0])
    assert not {'torch','models','main','engine'} & set(roots)

def test_synthetic_pipeline_and_input_hashes(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    run=tmp_path/'D0'; pred=run/'predictions_fixture'; pred.mkdir(parents=True)
    out=tmp_path/'out'; out.mkdir()
    cats=[dict(id=i,name=str(i)) for i in range(8)]
    ann=tmp_path/'annotation.json'; ids=tmp_path/'ids.json'; pre=tmp_path/'pre.json'
    m.save(ann,dict(images=[dict(id=i,width=20,height=20) for i in (1,2)],categories=cats,
                    annotations=[dict(id=10,image_id=1,category_id=1,bbox=[0,0,2,2],area=4)]))
    m.save(ids,[1,2]); m.save(pre,[1])
    m.save(pred/'predictions.json',[dict(image_id=1,**record([0,0,2,2]))])
    image_rows=[dict(image_id=i,width=20,height=20,gt_count=int(i==1),executed_query_count=int(i==1)) for i in (1,2)]
    (pred/'images.jsonl').write_text('\n'.join(json.dumps(x) for x in image_rows))
    m.save(pred/'metadata.json',dict(categories=cats,image_ids=[1,2]))
    m.save(run/'diagnosis_status.json',dict(status='completed',images=2))
    expected={k:m.digest(v) for k,v in dict(annotation=ann,image_ids=ids,precheck_ids=pre,predictions=pred/'predictions.json').items()}
    m.save(run/'diagnosis_manifest.json',dict(checkpoint_sha256=m.CHECKPOINT,group='D0',annotation_sha256=expected['annotation'],subset_sha256=expected['image_ids']))
    monkeypatch.setattr(m,'EXPECTED',expected)
    monkeypatch.setattr(m,'IMAGE_COUNT',2); monkeypatch.setattr(m,'PRECHECK_COUNT',1); monkeypatch.setattr(m,'GT_COUNT',1)
    a=SimpleNamespace(run_dir=run,annotation=ann,image_ids=ids,precheck_ids=pre)
    m.execute(a,out,lambda message:None)
    result=m.read(out/'summary.json')
    assert result['gt']==1 and result['empty_gt_images']==1
    assert result['strata']['all']['same_class']['hit75']==1
    assert m.digest(ann)==expected['annotation']
    # Corrupted provenance fails before any analysis; never silently switches input.
    m.save(ids,[2,1])
    with pytest.raises(ValueError,match='SHA-256'): m.execute(a,tmp_path/'bad',lambda message:None)
