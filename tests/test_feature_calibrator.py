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
