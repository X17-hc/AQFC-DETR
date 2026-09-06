import torch

from models.aqfcdetr.proposal_selection import build_query_valid_mask


def test_group_partition_restores_original_order_and_reduces_tokens():
    counts = torch.tensor([900, 300, 1500, 500])
    restored = torch.empty_like(counts)
    calls = []
    for level in torch.unique(counts, sorted=True):
        indices = torch.nonzero(counts == level, as_tuple=False).flatten()
        calls.append(int(level))
        restored[indices] = level
    assert calls == [300, 500, 900, 1500]
    assert torch.equal(restored, counts)
    assert counts.sum().item() < counts.max().item() * counts.numel()
    assert torch.equal(build_query_valid_mask(counts).sum(1), counts)
