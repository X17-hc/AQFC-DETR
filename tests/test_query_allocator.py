import torch

from models.aqfcdetr.query_allocator import AdaptiveQueryBudgetAllocator, QueryBudgetLoss


def test_teacher_schedule_and_budget_levels():
    assert [AdaptiveQueryBudgetAllocator.teacher_ratio(epoch) for epoch in range(7)] == [
        1.0, 1.0, 1.0, 0.75, 0.5, 0.25, 0.0]
    model = AdaptiveQueryBudgetAllocator(boundary_warmup_steps=1)
    model.train()
    outputs = model(torch.randn(4, 256, 8, 8),
                    real_counts=torch.tensor([1., 30., 100., 500.]), epoch=0)
    assert torch.all(outputs['boundaries'][:, 1:] > outputs['boundaries'][:, :-1])
    assert set(outputs['query_counts'].tolist()).issubset({300, 500, 900, 1500})
    assert outputs['query_counts'][0].item() == 300


def test_eval_has_no_900_query_floor_and_ema_is_stable():
    model = AdaptiveQueryBudgetAllocator(boundary_warmup_steps=1)
    model.eval()
    with torch.no_grad():
        model.count_regressor[-1].weight.zero_()
        model.count_regressor[-1].bias.fill_(1.0)
        before = model.ema_log_boundaries.clone()
        outputs = model(torch.randn(1, 256, 6, 6), epoch=99)
    assert outputs['query_counts'].item() == 300
    assert outputs['teacher_ratio'] == 0.0
    assert torch.equal(before, model.ema_log_boundaries)


def test_non_finite_count_falls_back_to_900():
    model = AdaptiveQueryBudgetAllocator(boundary_warmup_steps=1)
    model.eval()
    with torch.no_grad():
        model.count_regressor[-1].bias.fill_(float('nan'))
        outputs = model(torch.randn(1, 256, 4, 4))
    assert outputs['query_counts'].item() == 900
    assert outputs['invalid_fallback_count'].item() == 1


def test_forward_does_not_return_density_peaks():
    model = AdaptiveQueryBudgetAllocator(boundary_warmup_steps=1)
    outputs = model(torch.randn(2, 256, 8, 8), real_counts=torch.tensor([10., 80.]))
    assert 'density_peaks' not in outputs
    assert 'density_prior' in outputs


def test_quantile_off_keeps_constant_boundary_targets():
    loss = QueryBudgetLoss(enable_adaptive_targets=True, quantile_boundaries=False).train()
    counts = torch.full((50,), 800.)
    expected = torch.log(torch.tensor([60.0, 150.0, 350.0]))
    for _ in range(20):
        target, _ = loss._compute_adaptive_targets(counts, counts.device)
    assert torch.allclose(target[0], expected)


def test_quantile_targets_stay_constant_before_2000_images():
    loss = QueryBudgetLoss(enable_adaptive_targets=True, quantile_boundaries=True).train()
    counts = torch.full((50,), 5.)
    expected = torch.log(torch.tensor([60.0, 150.0, 350.0]))
    for _ in range(20):
        target, _ = loss._compute_adaptive_targets(counts, counts.device)
    assert int(loss.quantile_samples.item()) == 1000
    assert torch.allclose(target[0], expected)


def test_quantile_targets_never_rise_above_design_boundaries():
    loss = QueryBudgetLoss(enable_adaptive_targets=True, quantile_boundaries=True).train()
    counts = torch.full((100,), 800.)
    expected = torch.log(torch.tensor([60.0, 150.0, 350.0]))
    target = expected
    for _ in range(25):
        target, _ = loss._compute_adaptive_targets(counts, counts.device)
    assert int(loss.quantile_samples.item()) >= 2000
    assert loss.quantile_updates_this_epoch == 1
    assert torch.all(target[0] <= expected + 1e-5)


def test_lowered_boundaries_do_not_increase_the_300_bin():
    from models.aqfcdetr.query_allocator import count_to_routing
    loss = QueryBudgetLoss(enable_adaptive_targets=True, quantile_boundaries=True).train()
    counts = torch.tensor([1., 2., 3., 4.])
    for _ in range(500):
        loss._compute_adaptive_targets(counts, counts.device)
    route = count_to_routing(counts)
    old_low = (route < 60).float().mean()
    new_low = (route < torch.exp(loss.quantile_log_boundaries.detach())[0]).float().mean()
    assert new_low <= old_low + 1e-6


def test_allocator_uses_at_least_two_query_levels_on_spread_counts():
    import math
    model = AdaptiveQueryBudgetAllocator(boundary_warmup_steps=1, use_ema=False).train()
    with torch.no_grad():
        model.boundary_head[-1].weight.zero_()
        model.boundary_head[-1].bias.copy_(torch.tensor([
            math.log(60.0), -0.16, -0.32]))
    outputs = model(torch.randn(4, 256, 8, 8),
                    real_counts=torch.tensor([1., 40., 120., 400.]), epoch=0)
    assert len(set(outputs['query_counts'].tolist())) >= 2


