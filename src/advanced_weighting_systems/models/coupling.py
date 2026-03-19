"""ResonanceCoupling — unified coupling of adapters through AeonLayer.

Wires SymbolicMirror + WeightingEngine + AeonLayer into a single forward pass:

    1. SymbolicMirror.reflect(activations) → MirrorResult (M_i, R_i)
    2. WeightingEngine.compute(H, C, rho)  → WeightingResult (w_i, CREP_i)
    3. AeonLayer.update_weights(w_i)
    4. AeonLayer.forward(M_i, R_i)         → AeonLayerState (L_Aeon)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from advanced_weighting_systems.aeon_layer import AeonLayer, AeonLayerConfig, AeonLayerState
from advanced_weighting_systems.symbolic_mirror import MirrorResult, NNAdapter, SymbolicMirror
from advanced_weighting_systems.weighting_engine import (
    WeightingConfig,
    WeightingEngine,
    WeightingResult,
)


@dataclass
class CouplingOutput:
    """Full output of one ResonanceCoupling.step call.

    Attributes:
        mirror_result:    Intermediate SymbolicMirror output.
        weighting_result: Intermediate WeightingEngine output.
        aeon_state:       Final AeonLayerState (contains L_Aeon).
        energy:           Frobenius norm of L_Aeon (scalar resonance energy).
    """

    mirror_result: MirrorResult
    weighting_result: WeightingResult
    aeon_state: AeonLayerState
    energy: float


class ResonanceCoupling:
    """End-to-end resonance coupling of heterogeneous NN adapters.

    Parameters
    ----------
    adapters:
        List of NNAdapter instances to couple.
    dim:
        Shared embedding dimension d.
    aeon_config:
        Configuration for the AeonLayer (beta, theta).
    weighting_config:
        Configuration for the WeightingEngine (regime, lam, kappa, tau, …).
    sigil_seed:
        RNG seed for SymbolicMirror's Sigillin phase matrix.
    """

    def __init__(
        self,
        adapters: list[NNAdapter],
        dim: int = 16,
        aeon_config: AeonLayerConfig | None = None,
        weighting_config: WeightingConfig | None = None,
        sigil_seed: int = 42,
    ) -> None:
        n = len(adapters)
        self._mirror = SymbolicMirror(dim=dim, sigil_seed=sigil_seed)
        for adapter in adapters:
            self._mirror.register(adapter)
        self._engine = WeightingEngine(config=weighting_config)
        aeon_cfg = aeon_config or AeonLayerConfig(n_models=n)
        aeon_cfg.n_models = n
        self._aeon = AeonLayer(config=aeon_cfg)

    # ------------------------------------------------------------------

    def step(
        self,
        activations_list: list[NDArray[np.float64]],
        entropies: NDArray[np.float64],
        coherences: NDArray[np.float64],
        rho: NDArray[np.float64],
    ) -> CouplingOutput:
        """Perform one full resonance coupling step.

        Parameters
        ----------
        activations_list:
            Activation arrays for each adapter.
        entropies:
            Per-model entropy H_i.
        coherences:
            Per-model coherence C_i.
        rho:
            Per-model resonance correlation rho_i.

        Returns
        -------
        CouplingOutput with all intermediate and final results.
        """
        mirror_result = self._mirror.reflect(activations_list)
        weighting_result = self._engine.compute(entropies, coherences, rho)
        self._aeon.update_weights(weighting_result.weights)
        aeon_state = self._aeon.forward(
            mirror_result.mirror_matrices,
            mirror_result.resonance_signals,
        )
        energy = self._aeon.resonance_energy(aeon_state)
        return CouplingOutput(
            mirror_result=mirror_result,
            weighting_result=weighting_result,
            aeon_state=aeon_state,
            energy=energy,
        )

    @property
    def n_models(self) -> int:
        """Number of coupled models."""
        return self._mirror.n_adapters

    def __repr__(self) -> str:
        return (
            f"ResonanceCoupling(n_models={self.n_models}, "
            f"aeon={self._aeon!r}, engine={self._engine!r})"
        )
