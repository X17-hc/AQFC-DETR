import torch

from models.aqfcdetr.query_allocator import AdaptiveQueryBudgetAllocator


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
