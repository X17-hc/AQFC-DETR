import torch

from models.aqfcdetr.proposal_selection import select_proposal_indices


def _inputs():
    classes = torch.tensor([[[1., 0.], [4., 0.], [3., 0.], [2., 0.], [99., 0.]]])
    density = torch.tensor([[0.99, 0.01, 0.90, 0.80, 1.00]])
    padding = torch.tensor([[False, False, False, False, True]])
    return classes, density, padding


def test_zero_density_weight_matches_semantic_topk_and_excludes_padding():
    classes, density, padding = _inputs()
    selected = select_proposal_indices(
        classes, density, padding, 3, mode='fused', density_weight=0.0)
    expected = torch.topk(classes.max(-1).values.masked_fill(padding, float('-inf')), 3).indices
    assert torch.equal(selected, expected)
    assert 4 not in selected[0].tolist()


def test_mixed_selection_is_unique_deterministic_and_complete():
    classes, density, padding = _inputs()
    first = select_proposal_indices(classes, density, padding, 4, mode='mixed')
    second = select_proposal_indices(classes, density, padding, 4, mode='mixed')
    assert torch.equal(first, second)
    assert len(set(first[0].tolist())) == 4
    assert 4 not in first[0].tolist()
