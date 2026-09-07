import torch

from models.aqfcdetr.model import SetCriterion, PostProcess
from models.aqfcdetr.matcher import HungarianMatcher


def test_actual_classification_loss_has_zero_padding_gradient_with_empty_image():
    criterion = SetCriterion(2, HungarianMatcher(), {}, .25, ['labels'])
    logits = torch.tensor([[[2.,-1.],[1.,0.],[100.,100.]],
                           [[0.,0.],[100.,100.],[100.,100.]]], requires_grad=True)
    mask = torch.tensor([[True,True,False],[True,False,False]])
    targets = [dict(labels=torch.tensor([0])), dict(labels=torch.empty(0,dtype=torch.long))]
    empty = torch.empty(0,dtype=torch.long)
    indices = [(torch.tensor([0]),torch.tensor([0])), (empty,empty)]
    output = dict(pred_logits=logits, query_valid_mask=mask)
    loss = criterion.loss_labels(output, targets, indices, 1.)['loss_ce']
    loss.backward()
    assert torch.isfinite(loss)
    assert (logits.grad[~mask] == 0).all()
    assert logits.grad[1,0].abs().sum() > 0  # valid background still learns


def test_actual_postprocessor_never_returns_padding_boxes():
    output = dict(pred_logits=torch.tensor([[[1.],[100.]]]),
                  pred_boxes=torch.tensor([[[.25,.25,.1,.1],[.9,.9,.1,.1]]]),
                  query_valid_mask=torch.tensor([[True,False]]))
    results = PostProcess()(output, torch.tensor([[100,100]]))
    assert len(results[0]['scores']) == 1
    assert results[0]['executed_query_count'] == 1
    assert results[0]['boxes'][0,0] < 50
