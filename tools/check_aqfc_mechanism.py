"""Mechanism checks for scheme 2: peak-only DGFC and capped AQBA.

DGFC may change only the finest level's top 10 percent. AQBA keeps the
60/150/350 targets until 2000 images are stored, and never raises a boundary.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from models.aqfcdetr.feature_calibrator import DensityGuidedFeatureCalibrator
from models.aqfcdetr.query_allocator import QueryBudgetLoss, count_to_routing


def check_dgfc():
    module = DensityGuidedFeatureCalibrator(
        gate_channels=32, num_feature_levels=1, use_spatial=False,
        gate_type='tanh', level_spatial_alphas=[0.0], density_spatial=True)
    for parameter in module.channel_gate.mlp.parameters():
        parameter.data.zero_()
    memory = torch.rand(1, 100, 32) + 0.5
    prior = torch.linspace(0, 1, 100).view(1, 1, 10, 10)
    closed = module(None, memory, [(10, 10)], density_prior=torch.zeros_like(prior))
    opened = module(None, memory, [(10, 10)], density_prior=prior)
    delta = (opened - closed).abs().mean(-1).reshape(-1)
    order = prior.reshape(-1)
    bottom = delta[order.topk(90, largest=False).indices]
    top = delta[order.topk(10).indices]
    if not torch.all(bottom == 0) or not float(top.mean()) > 0:
        raise SystemExit('DGFC changed positions outside the top 10 percent')
    module.density_gate_scale = 0.0
    scaled = module(None, memory, [(10, 10)], density_prior=prior)
    if not torch.allclose(scaled, closed, atol=1e-6):
        raise SystemExit('DGFC scale 0 changed features')
    print(f'DGFC_OK top10={float(top.mean().detach()):.6f} bottom90=0 '
          f'alpha={float(module.level_alpha[0].detach()):.4f}')


def check_aqba():
    loss = QueryBudgetLoss(quantile_boundaries=True).train()
    small = torch.tensor([1., 2., 3., 4.])
    expected = torch.log(torch.tensor([60.0, 150.0, 350.0]))
    target = expected
    for _ in range(100):
        target, _ = loss._compute_adaptive_targets(small, small.device)
    if int(loss.quantile_samples.item()) != 400 or not torch.allclose(target[0], expected):
        raise SystemExit('AQBA moved its target before 2000 images')
    for _ in range(400):
        target, _ = loss._compute_adaptive_targets(small, small.device)
    route = count_to_routing(small)
    old_low = float((route < 60).float().mean())
    new_boundary = float(torch.exp(loss.quantile_log_boundaries.detach())[0])
    new_low = float((route < new_boundary).float().mean())
    if int(loss.quantile_samples.item()) < 2000:
        raise SystemExit(f'AQBA saw only {int(loss.quantile_samples)} samples')
    if torch.any(loss.quantile_log_boundaries.detach().cpu() > expected + 1e-5):
        raise SystemExit(f'AQBA raised a boundary: {loss.quantile_log_boundaries.tolist()}')
    if new_low > old_low + 1e-6:
        raise SystemExit(f'AQBA increased the 300 bin: {new_low:.3f} > {old_low:.3f}')
    crowded = torch.full((100,), 800.)
    crowded_loss = QueryBudgetLoss(quantile_boundaries=True).train()
    for _ in range(25):
        crowded_target, _ = crowded_loss._compute_adaptive_targets(crowded, crowded.device)
    if torch.any(crowded_target[0].cpu() > expected + 1e-5):
        raise SystemExit('AQBA raised the 500-bin entrance on crowded counts')
    print(f'AQBA_OK samples={int(loss.quantile_samples)} '
          f'low_boundary={new_boundary:.2f} low_bin={new_low:.3f}')


def main():
    check_dgfc()
    check_aqba()
    print('MECHANISM_OK')


if __name__ == '__main__':
    main()
