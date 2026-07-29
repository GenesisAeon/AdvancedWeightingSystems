"""Tests for package-level __init__ exports."""

from __future__ import annotations

import advanced_weighting_systems as aws


class TestPackageExports:
    def test_version(self) -> None:
        assert isinstance(aws.__version__, str)
        assert aws.__version__

    def test_aeon_layer_exported(self) -> None:
        from advanced_weighting_systems import AeonLayer  # noqa: F401

    def test_symbolic_mirror_exported(self) -> None:
        from advanced_weighting_systems import SymbolicMirror  # noqa: F401

    def test_weighting_engine_exported(self) -> None:
        from advanced_weighting_systems import WeightingEngine  # noqa: F401

    def test_all_list(self) -> None:
        assert "AeonLayer" in aws.__all__
        assert "SymbolicMirror" in aws.__all__
        assert "WeightingEngine" in aws.__all__
