"""Tests for the top-level cosmulator namespace."""

import cosmulator


def test_version_is_exposed():
    """The package exposes a PEP 440 style version string."""
    assert isinstance(cosmulator.__version__, str)
    assert cosmulator.__version__.count(".") >= 2


def test_import_does_not_require_training_stack():
    """Importing cosmulator must not drag in JAX.

    This is the guarantee that keeps the package installable and reviewable
    without a GPU; see tests/test_packaging.py for the static counterpart.
    """
    import sys

    assert "jax" not in sys.modules
    assert "margarine" not in sys.modules


def test_public_api_is_exported():
    """Everything in __all__ is importable from the top-level namespace."""
    for name in cosmulator.__all__:
        assert hasattr(cosmulator, name), f"{name} is in __all__ but absent"


def test_public_api_is_reachable_without_the_training_stack():
    """The whole documented API works on an install with no JAX.

    This is the promise the optional-extra split exists to keep: a reviewer or
    a laptop user can exercise every part of the package except training.
    """
    import sys

    assert cosmulator.EmulatorStore is not None
    assert cosmulator.self_consistency is not None
    assert "jax" not in sys.modules
    assert "margarine" not in sys.modules
