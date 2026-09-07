import torch

from models.aqfcdetr.query_allocator import AdaptiveQueryBudgetAllocator, QueryBudgetLoss
from models.aqfcdetr.proposal_selection import select_proposal_indices


def test_boundary_loss_trains_raw_head_with_ema_enabled():
    allocator = AdaptiveQueryBudgetAllocator(use_ema=True).train()
    result = allocator(torch.randn(2,256,8,8), real_counts=torch.tensor([2.,100.]))
    losses = QueryBudgetLoss()(result, torch.tensor([2.,100.]))
    losses['loss_boundary_guide'].backward()
    gradient = allocator.boundary_head[-1].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_unequal_encoder_valid_lengths_keep_batch_padding_shape():
    logits = torch.randn(2,10,3)
    density = torch.rand(2,10)
    mask = torch.zeros(2,10,dtype=torch.bool)
    mask[0,4:] = True
    indices = select_proposal_indices(logits, density, mask, 8)
    assert indices.shape == (2,8)
    assert (indices[0,:4] < 4).all()
    assert indices[0,:4].unique().numel() == 4


def test_cli_single_element_list_and_nested_options():
    import argparse
    from util.slconfig import DictAction
    parser = argparse.ArgumentParser()
    parser.add_argument('--options', nargs='+', action=DictAction)
    options = parser.parse_args(['--options', 'val_epoch=[1]',
                                 'epochs=2', 'use_ema=False', 'scales=256,320']).options
    assert options == dict(val_epoch=[1], epochs=2, use_ema=False, scales=[256,320])
