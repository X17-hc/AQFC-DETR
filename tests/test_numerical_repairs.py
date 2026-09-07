"""Regression tests for the failures found in the September smoke run."""
import math

import pytest
import torch

from models.aqfcdetr.query_allocator import AdaptiveQueryBudgetAllocator, QueryBudgetLoss
from models.aqfcdetr.transformer import DeformableTransformer
from util.checkpoint import load_native_resume
from models.aqfcdetr.matcher import HungarianMatcher
from models.aqfcdetr.proposal_selection import select_proposal_indices


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('target_value', [0., 1.])
def test_saturated_density_loss_and_gradient_are_finite(dtype, target_value):
    prediction = torch.tensor([[[[0., 1.]]]], dtype=dtype, requires_grad=True)
    target = torch.full_like(prediction, target_value)
    loss = QueryBudgetLoss.density_focal_loss(prediction, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_parent_preserves_allocator_special_initialization():
    model = DeformableTransformer(
        deformable_encoder=True, deformable_decoder=True,
        num_encoder_layers=1, num_decoder_layers=1,
        return_intermediate_dec=True, decoder_sa_type='sa', learnable_tgt_init=True)
    allocator = model.query_allocator
    assert allocator.boundary_head[-1].weight.std() < .001
    assert allocator.count_regressor[-1].weight.std() < .001
    assert allocator.density_head.weight.std() < .02


def test_teacher_and_prediction_use_same_routing_units():
    allocator = AdaptiveQueryBudgetAllocator(use_ema=False)
    with torch.no_grad():
        allocator.count_regressor[-1].weight.zero_()
        allocator.count_regressor[-1].bias.fill_(math.log(20.))
        x = torch.randn(1, 256, 4, 4)
        teacher = allocator.train()(x, real_counts=torch.tensor([20.]), epoch=0)
        predicted = allocator.eval()(x)
    torch.testing.assert_close(teacher['routing_count'], predicted['routing_count'])


@pytest.mark.parametrize('bad', [float('inf'), float('-inf'), float('nan')])
def test_invalid_count_fallback_even_during_teacher_routing(bad):
    allocator = AdaptiveQueryBudgetAllocator(use_ema=False).train()
    with torch.no_grad():
        allocator.count_regressor[-1].bias.fill_(bad)
        result = allocator(torch.randn(1, 256, 4, 4), real_counts=torch.tensor([20.]))
    assert result['query_counts'].item() == 900
    assert result['invalid_fallback_count'].item() == 1


def test_invalid_boundary_does_not_poison_ema():
    allocator = AdaptiveQueryBudgetAllocator().train()
    before = allocator.ema_log_boundaries.clone()
    with torch.no_grad():
        allocator.boundary_head[-1].bias.fill_(float('nan'))
        result = allocator(torch.randn(1, 256, 4, 4))
    assert torch.isfinite(result['boundaries']).all()
    torch.testing.assert_close(allocator.ema_log_boundaries, before)
    assert result['invalid_boundary_fallback_count'].item() == 1


def test_partial_checkpoint_cannot_resume_as_complete_epoch(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = tmp_path / 'partial.pth'
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    epoch=0, run_metadata={'smoke_test': True, 'epoch_complete': False}), checkpoint)
    with pytest.raises(ValueError, match='smoke|partial'):
        load_native_resume(model, checkpoint, optimizer=optimizer)


def test_missing_scaler_is_not_silently_accepted(tmp_path):
    model = torch.nn.Linear(1, 1)
    checkpoint = tmp_path / 'missing.pth'
    torch.save(dict(model=model.state_dict(), epoch=0), checkpoint)
    with pytest.raises(KeyError, match='scaler'):
        load_native_resume(model, checkpoint, scaler=torch.amp.GradScaler('cpu'))


def test_matcher_accepts_saturated_half_logits():
    output = dict(pred_logits=torch.tensor([[[100.], [-100.]]], dtype=torch.float16),
                  pred_boxes=torch.tensor([[[.5,.5,.2,.2], [.1,.1,.1,.1]]]))
    target = [dict(labels=torch.tensor([0]), boxes=torch.tensor([[.5,.5,.2,.2]]))]
    result = HungarianMatcher()(output, target)
    assert result[0][0].tolist() == [0]


def test_matcher_rejects_invalid_labels_instead_of_relabeling():
    output = dict(pred_logits=torch.zeros(1,1,1), pred_boxes=torch.ones(1,1,4)*.1)
    target = [dict(labels=torch.tensor([5]), boxes=torch.ones(1,4)*.1)]
    with pytest.raises(ValueError, match='label'):
        HungarianMatcher()(output, target)


def test_half_density_ranking_retains_semantic_order_when_saturated():
    logits = torch.tensor([[[1.], [2.]]], dtype=torch.float16)
    density = torch.ones(1,2,dtype=torch.float16)
    indices = select_proposal_indices(logits, density, torch.zeros(1,2,dtype=torch.bool), 1)
    assert indices.item() == 1


def test_nonfinite_encoder_proposal_is_never_selected():
    logits = torch.tensor([[[1.], [100.]]])
    proposals = torch.tensor([[[0.,0.,0.,0.], [float('inf')]*4]])
    indices = select_proposal_indices(logits, torch.ones(1,2)*.5,
                                      torch.zeros(1,2,dtype=torch.bool), 1,
                                      proposal_boxes=proposals)
    assert indices.item() == 0
