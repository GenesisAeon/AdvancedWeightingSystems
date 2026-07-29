"""Advanced Weighting Systems — resonance-based symbolic coupling of heterogeneous NN models.

GenesisAeon project: AeonLayer + CREP + Sigillin from unified-mandala.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from advanced_weighting_systems.aeon_layer import AeonLayer
from advanced_weighting_systems.symbolic_mirror import SymbolicMirror
from advanced_weighting_systems.weighting_engine import WeightingEngine

try:
    __version__ = _version("advanced-weighting-systems")
except PackageNotFoundError:
    # Not installed, e.g. running from source.
    __version__ = "0.0.0+unknown"

__all__ = ["AeonLayer", "SymbolicMirror", "WeightingEngine", "__version__"]
