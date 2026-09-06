import torch

from models.aqfcdetr.query_allocator import QueryBudgetLoss


def test_density_targets_cover_small_overlap_boundary_and_empty_images():
    targets = [
        {'boxes': torch.tensor([[0.50, 0.50, 0.01, 0.01],
                                [0.51, 0.50, 0.20, 0.10],
                                [0.00, 1.00, 0.90, 0.90]])},
        {'boxes': torch.empty(0, 4)},
    ]
    heatmap = QueryBudgetLoss.build_density_targets(targets, (16, 20), torch.device('cpu'))
    assert heatmap.shape == (2, 1, 16, 20)
    assert heatmap[0].max().item() == 1.0
    assert heatmap[0, 0, 15, 0].item() == 1.0
    assert heatmap[1].sum().item() == 0.0


def test_density_focal_loss_is_finite_and_differentiable():
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 4, 4] = 1.0
    logits = torch.full_like(target, -4.0, requires_grad=True)
    prediction = logits.sigmoid()
    loss = QueryBudgetLoss.density_focal_loss(prediction, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_perfect_density_prediction_has_near_zero_loss():
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, 2, 2] = 1.0
    prediction = target * (1 - 1e-6) + (1 - target) * 1e-6
    assert QueryBudgetLoss.density_focal_loss(prediction, target).item() < 1e-8


def test_padding_region_is_excluded_from_density_loss():
    targets = [
        {'boxes': torch.empty(0, 4), 'size': torch.tensor([8, 8])},
        {'boxes': torch.empty(0, 4), 'size': torch.tensor([16, 16])},
    ]
    target, valid = QueryBudgetLoss.build_density_targets(
        targets, (8, 8), torch.device('cpu'), return_valid_mask=True)
    prediction = torch.full_like(target, 0.01)
    prediction[0, :, 4:, :] = 0.99
    prediction[0, :, :, 4:] = 0.99
    masked = QueryBudgetLoss.density_focal_loss(prediction, target, valid)
    unmasked = QueryBudgetLoss.density_focal_loss(prediction, target)
    assert masked < unmasked
    assert valid[0].sum().item() == 16
