import torch
import pytest
from tools.diagnose_f2_boxes import paired_results


def test_same_outputs_and_mask_are_preserved():
    outputs = dict(pred_boxes=torch.rand(1,3,4), pred_boxes_coarse=torch.rand(1,3,4),
                   pred_logits=torch.randn(1,3,2),query_valid_mask=torch.tensor([[True,True,False]]),
                   executed_query_counts=torch.tensor([2]))
    snapshot = {k:v.clone() for k,v in outputs.items()}
    def post(o, sizes):
        v=o['query_valid_mask'][0]
        scores,labels=o['pred_logits'][0][v].sigmoid().max(-1)
        return [dict(scores=scores,labels=labels,boxes=o['pred_boxes'][0][v])]
    a,b=paired_results(outputs,torch.tensor([[100,200]]),post)
    assert torch.equal(a[0]['boxes'],outputs['pred_boxes_coarse'][0,:2])
    assert torch.equal(b[0]['boxes'],outputs['pred_boxes'][0,:2])
    assert all(torch.equal(v,snapshot[k]) for k,v in outputs.items())
    outputs['executed_query_counts']=torch.tensor([3])
    with pytest.raises(ValueError,match='mask'): paired_results(outputs,None,post)
