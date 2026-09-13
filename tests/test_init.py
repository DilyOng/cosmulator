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
