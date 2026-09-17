import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from legacy24_diagnosis import sample,proposal_record

def test_sampling():
    ann={'images':[{'id':i} for i in range(1100)],'annotations':[]}
    a,s=sample(ann); b,_=sample(ann)
    assert a==b and len(a)==len(set(a))==1000
    assert s['sample']==[1000,0,0,0,0]
    with pytest.raises(ValueError): sample({'images':[{'id':1}]*1000,'annotations':[]})

def outputs():
    return {'query_valid_mask':torch.tensor([[True,False]]),'executed_query_counts':torch.tensor([1]),
            'interm_outputs':{'pred_boxes':torch.tensor([[[.5,.5,.2,.4],[1.,1.,1.,1.]]])}}

def test_rectangular_conversion_mask_and_no_mutation():
    out=outputs(); old=out['interm_outputs']['pred_boxes'].clone()
    row=proposal_record(out,{'id':8,'width':101,'height':57},[{'id':99,'bbox':[40.4,17.1,20.2,22.8],'area':460.56,'category_id':5}])
    assert row['executed_query_count']==len(row['proposals'])==1
    assert row['ground_truth'][0]['annotation_id']==99
    assert row['ground_truth'][0]['best_proposal_iou']==pytest.approx(1,abs=1e-6)
    assert torch.equal(old,out['interm_outputs']['pred_boxes'])

def test_empty_gt_and_invalid_mask():
    row=proposal_record(outputs(),{'id':1,'width':7,'height':9},[])
    assert row['ground_truth']==[]
    out=outputs(); out['executed_query_counts'][0]=2
    with pytest.raises(ValueError): proposal_record(out,{'id':1,'width':7,'height':9},[])

def test_empty_proposals():
    out=outputs(); out['query_valid_mask'][:]=False; out['executed_query_counts'][0]=0
    row=proposal_record(out,{'id':1,'width':7,'height':9},[{'id':2,'bbox':[0,0,1,1],'area':1,'category_id':0}])
    assert row['ground_truth'][0]['best_proposal_iou']==0

def test_nonfinite():
    out=outputs(); out['interm_outputs']['pred_boxes'][0,0,0]=float('nan')
    with pytest.raises(ValueError): proposal_record(out,{'id':1,'width':7,'height':9},[])
