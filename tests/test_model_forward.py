import importlib

import pytest


def test_model_module_imports_when_native_extension_is_available():
    try:
        importlib.import_module('models.aqfcdetr.model')
    except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
        pytest.skip(f'native deformable-attention extension unavailable: {exc}')
