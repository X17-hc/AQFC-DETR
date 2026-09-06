import torch

from models.aqfcdetr.proposal_selection import build_query_valid_mask
from models.aqfcdetr.matcher import HungarianMatcher


def test_query_valid_mask_for_all_budget_levels():
    counts = torch.tensor([300, 500, 900, 1500])
    mask = build_query_valid_mask(counts)
    assert mask.shape == (4, 1500)
    assert torch.equal(mask.sum(1), counts)
    assert mask[0, 299] and not mask[0, 300]


def test_invalid_query_logits_cannot_enter_topk():
    valid = torch.tensor([[True, True, False, False]])
    logits = torch.tensor([[[1.], [2.], [100.], [200.]]])
    logits = logits.masked_fill(~valid[:, :, None], float('-inf'))
    assert set(torch.topk(logits.flatten(1), 2).indices[0].tolist()) == {0, 1}


def test_matcher_never_selects_padding_queries():
    outputs = {
        'pred_logits': torch.tensor([[[0.], [0.], [100.], [100.]]]),
        'pred_boxes': torch.tensor([[[.5, .5, .2, .2], [.1, .1, .1, .1],
                                     [.5, .5, .2, .2], [.5, .5, .2, .2]]]),
        'query_valid_mask': torch.tensor([[True, True, False, False]]),
    }
    targets = [{'labels': torch.tensor([0]),
                'boxes': torch.tensor([[.5, .5, .2, .2]])}]
    prediction_indices, _ = HungarianMatcher()(outputs, targets)[0]
    assert set(prediction_indices.tolist()).issubset({0, 1})
