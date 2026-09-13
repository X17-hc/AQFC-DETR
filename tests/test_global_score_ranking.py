import numpy as np
import pytest
from tools.analyze_global_score_ranking import rank_curve, official_eval


def test_fp_before_tp():
    r=rank_curve([.9,.8],[False,True],[False,False],1)
    np.testing.assert_allclose(r['q'],.5)
    assert r['AP75']==.5


def test_stable_ties_cross_image_order():
    r=rank_curve([.8,.8],[False,True],[False,False],1)
    assert r['order'].tolist()==[0,1] and r['AP75']==.5


def test_ignore_propagation():
    r=rank_curve([.99,.8],[False,True],[True,False],1)
    assert r['AP75']>1-1e-12 and r['fp'][-1]==0


def test_empty_and_absent_category():
    assert rank_curve([],[],[],1)['AP75']==0
    assert rank_curve([.8],[False],[False],0)['AP75'] is None


@pytest.mark.parametrize('records',[
    [],
    [dict(image_id=1,category_id=0,bbox=[0,0,10,10],score=.8)],
    [dict(image_id=2,category_id=0,bbox=[0,0,10,10],score=.8),
     dict(image_id=1,category_id=0,bbox=[0,0,10,10],score=.8)],
])
def test_official_precision_reconstruction(records):
    annotation=dict(images=[dict(id=1),dict(id=2)],categories=[dict(id=0,name='a'),dict(id=1,name='empty')],
        annotations=[dict(id=1,image_id=1,category_id=0,bbox=[0,0,10,10],area=100,iscrowd=0)])
    e=official_eval(annotation,records,[1,2])
    entries=[x for x in e.evalImgs if x and x['category_id']==0]
    scores=np.concatenate([x['dtScores'] for x in entries])
    matched=np.concatenate([x['dtMatches'][0] for x in entries])
    ignore=np.concatenate([x['dtIgnore'][0] for x in entries])
    r=rank_curve(scores,matched,ignore,1)
    np.testing.assert_allclose(r['q'],e.eval['precision'][0,:,0,0,0],atol=1e-12,rtol=0)
    assert (e.eval['precision'][0,:,1,0,0]==-1).all()
