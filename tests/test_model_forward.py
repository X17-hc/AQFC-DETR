import importlib

import pytest


def test_model_module_imports_when_native_extension_is_available():
    if importlib.util.find_spec('MultiScaleDeformableAttention') is None:
        pytest.skip('native deformable-attention extension not installed')
    importlib.import_module('models.aqfcdetr.model')
