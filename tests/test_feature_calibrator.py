import pytest
import torch

from models.aqfcdetr.feature_calibrator import (
    DensityGuidedFeatureCalibrator, DensityPyramidAdapter)


def test_tanh_channel_gate_is_identity_at_zero_response():
    module = DensityGuidedFeatureCalibrator(
        gate_channels=256, num_feature_levels=1, use_spatial=False,
        gate_type='tanh', level_spatial_alphas=[0.0])
    for parameter in module.channel_gate.mlp.parameters():
        parameter.data.zero_()
    memory = torch.randn(2, 16, 256)
    output = module([torch.randn(2, 256, 4, 4)], memory, torch.tensor([[4, 4]]))
    assert torch.allclose(output, memory)


@pytest.mark.parametrize('gate_type', ['tanh', 'sigmoid', 'se', 'swish', 'glu'])
def test_all_gate_variants_share_shape_and_have_finite_gradients(gate_type):
    module = DensityGuidedFeatureCalibrator(
        gate_channels=256, num_feature_levels=1, use_spatial=False,
        gate_type=gate_type, level_spatial_alphas=[0.0])
    memory = torch.randn(1, 9, 256, requires_grad=True)
    output = module([torch.randn(1, 256, 3, 3)], memory, torch.tensor([[3, 3]]))
    output.mean().backward()
    assert output.shape == memory.shape
    assert torch.isfinite(memory.grad).all()


def test_density_pyramid_has_one_feature_per_level():
    pyramid = DensityPyramidAdapter(is_5_scale=True)
    outputs = pyramid(torch.randn(1, 256, 32, 32))
    assert [feature.shape[-2:] for feature in outputs] == [
        (32, 32), (16, 16), (8, 8), (4, 4), (2, 2)]


def _zero_channel(module):
    for parameter in module.channel_gate.mlp.parameters():
        parameter.data.zero_()


def test_density_spatial_off_finest_level_matches_channel_gate():
    torch.manual_seed(0)
    module = DensityGuidedFeatureCalibrator(
        gate_channels=32, num_feature_levels=2, gate_type='tanh',
        level_spatial_alphas=[0.0, 0.05], density_spatial=False)
    memory = torch.randn(1, 8, 32)
    aux = [torch.randn(1, 32, 2, 2), torch.randn(1, 32, 2, 2)]
    output = module(aux, memory, [(2, 2), (2, 2)])
    finest = memory[:, :4].transpose(1, 2).reshape(1, 32, 2, 2)
    expected = module.channel_gate(finest).flatten(2).transpose(1, 2)
    assert torch.allclose(output[:, :4], expected, atol=1e-6)


def test_density_peak_raises_that_location_and_zero_prior_is_channel_only():
    module = DensityGuidedFeatureCalibrator(
        gate_channels=32, num_feature_levels=1, use_spatial=False,
        gate_type='tanh', level_spatial_alphas=[0.0], density_spatial=True)
    _zero_channel(module)
    assert module.level_alpha[0].detach().item() == pytest.approx(0.05)
    memory = torch.ones(1, 4, 32)
    prior = torch.zeros(1, 1, 2, 2)
    prior[0, 0, 0, 0] = 1.0
    peaked = module(None, memory, [(2, 2)], density_prior=prior)
    assert torch.all(peaked[0, 0] > peaked[0, 1])
    zeros = module(None, memory, [(2, 2)], density_prior=torch.zeros_like(prior))
    expected = module.channel_gate(memory.transpose(1, 2).reshape(1, 32, 2, 2))
    expected = expected.flatten(2).transpose(1, 2)
    assert torch.allclose(zeros, expected, atol=1e-6)


def test_high_density_locations_change_more_than_background():
    module = DensityGuidedFeatureCalibrator(
        gate_channels=32, num_feature_levels=1, use_spatial=False,
        gate_type='tanh', level_spatial_alphas=[0.0], density_spatial=True)
    _zero_channel(module)
    memory = torch.rand(1, 100, 32) + 0.5
    prior = torch.linspace(0, 1, 100, dtype=torch.float32).view(1, 1, 10, 10)
    baseline = module(None, memory, [(10, 10)], density_prior=torch.zeros_like(prior))
    moved = module(None, memory, [(10, 10)], density_prior=prior)
    delta = (moved - baseline).abs().mean(dim=-1).reshape(-1)
    order = prior.reshape(-1)
    top = delta[order.topk(10).indices]
    bottom = delta[order.topk(90, largest=False).indices]
    assert torch.all(bottom == 0)
    assert top.mean() > 0


def test_density_gate_scale_zero_leaves_features_unchanged():
    module = DensityGuidedFeatureCalibrator(
        gate_channels=32, num_feature_levels=1, use_spatial=False,
        gate_type='tanh', level_spatial_alphas=[0.0], density_spatial=True)
    _zero_channel(module)
    module.density_gate_scale = 0.0
    memory = torch.rand(1, 16, 32) + 0.5
    prior = torch.zeros(1, 1, 4, 4)
    prior[0, 0, 0, 0] = 1.0
    output = module(None, memory, [(4, 4)], density_prior=prior)
    expected = module.channel_gate(memory.transpose(1, 2).reshape(1, 32, 4, 4))
    expected = expected.flatten(2).transpose(1, 2)
    assert torch.allclose(output, expected, atol=1e-6)


def test_coarser_levels_keep_the_old_spatial_path():
    torch.manual_seed(0)
    shared = dict(gate_channels=32, num_feature_levels=2, gate_type='tanh',
                  level_spatial_alphas=[0.0, 0.05])
    old = DensityGuidedFeatureCalibrator(density_spatial=False, **shared)
    new = DensityGuidedFeatureCalibrator(density_spatial=True, **shared)
    new.load_state_dict(old.state_dict(), strict=False)
    memory = torch.randn(1, 8, 32)
    aux = [torch.randn(1, 32, 2, 2), torch.randn(1, 32, 2, 2)]
    prior = torch.rand(1, 1, 2, 2)
    old_out = old(aux, memory, [(2, 2), (2, 2)])
    new_out = new(aux, memory, [(2, 2), (2, 2)], density_prior=prior)
    assert torch.allclose(new_out[:, 4:], old_out[:, 4:], atol=1e-5)

