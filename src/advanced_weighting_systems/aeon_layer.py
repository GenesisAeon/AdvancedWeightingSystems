"""AeonLayer — central resonance aggregation layer for the GenesisAeon stack.

The AeonLayer implements the fundamental aggregation formula:

    L_Aeon = sum_i  w_i * M_i * sigma(beta * (R_i - Theta))

where:
    w_i   : dynamic resonance weights (from WeightingEngine)
    M_i   : Mirror-Matrix for model i (from mirror-machine / SymbolicMirror)
    R_i   : raw resonance signal for model i
    Theta : global resonance threshold
    beta  : sharpness / inverse temperature
    sigma : logistic sigmoid activation

KaTeX:
    L_{\\text{Aeon}} = \\sum_i w_i \\cdot M_i \\cdot \\sigma\\!\\left(\\beta(R_i - \\Theta)\\right)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Element-wise logistic sigmoid: sigma(x) = 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-x))  # type: ignore[return-value]


@dataclass
class AeonLayerConfig:
    """Configuration for the AeonLayer.

    Attributes:
        beta:    Sharpness / inverse temperature for the sigmoid gate.
        theta:   Global resonance threshold Theta.
        n_models: Number of coupled models (sets dimension of w and M).
    """

    beta: float = 1.0
    theta: float = 0.0
    n_models: int = 1


@dataclass
class AeonLayerState:
    """Runtime state produced by one AeonLayer forward pass.

    Attributes:
        weights:        Dynamic resonance weights w_i  (shape: [n_models]).
        mirror_matrices: Mirror-matrices M_i           (shape: [n_models, d, d]).
        resonance_signals: Raw signals R_i             (shape: [n_models]).
        gate_values:    sigma(beta*(R_i - Theta))     (shape: [n_models]).
        layer_output:   L_Aeon aggregation             (shape: [d, d]).
    """

    weights: NDArray[np.float64]
    mirror_matrices: NDArray[np.float64]
    resonance_signals: NDArray[np.float64]
    gate_values: NDArray[np.float64]
    layer_output: NDArray[np.float64]


class AeonLayer:
    """Resonance aggregation layer: L_Aeon = sum_i w_i * M_i * sigma(beta*(R_i - Theta)).

    Parameters
    ----------
    config:
        AeonLayerConfig instance controlling beta, theta, and n_models.
    weights:
        Initial dynamic resonance weights (1-D array of length n_models).
        If None, weights are initialised uniformly.
    """

    def __init__(
        self,
        config: AeonLayerConfig | None = None,
        weights: NDArray[np.float64] | None = None,
    ) -> None:
        self.config: AeonLayerConfig = config or AeonLayerConfig()
        n = self.config.n_models
        if weights is not None:
            if weights.shape != (n,):
                msg = f"weights must have shape ({n},), got {weights.shape}"
                raise ValueError(msg)
            self._weights: NDArray[np.float64] = weights.copy()
        else:
            self._weights = np.full(n, 1.0 / n, dtype=np.float64)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def weights(self) -> NDArray[np.float64]:
        """Current dynamic resonance weights (read-only copy)."""
        return self._weights.copy()

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        mirror_matrices: NDArray[np.float64],
        resonance_signals: NDArray[np.float64],
    ) -> AeonLayerState:
        """Compute L_Aeon = sum_i w_i * M_i * sigma(beta*(R_i - Theta)).

        Parameters
        ----------
        mirror_matrices:
            Array of shape [n_models, d, d] — Mirror-Matrix M_i for each model.
        resonance_signals:
            1-D array of shape [n_models] — raw resonance signal R_i per model.

        Returns
        -------
        AeonLayerState with all intermediate values and the final layer_output.
        """
        n = self.config.n_models
        if mirror_matrices.ndim != 3 or mirror_matrices.shape[0] != n:
            msg = f"mirror_matrices must have shape [{n}, d, d], got {mirror_matrices.shape}"
            raise ValueError(msg)
        if resonance_signals.shape != (n,):
            msg = f"resonance_signals must have shape ({n},), got {resonance_signals.shape}"
            raise ValueError(msg)

        beta = self.config.beta
        theta = self.config.theta
        gate: NDArray[np.float64] = _sigmoid(beta * (resonance_signals - theta))

        d = mirror_matrices.shape[1]
        layer_output: NDArray[np.float64] = np.zeros((d, d), dtype=np.float64)
        for i in range(n):
            layer_output += self._weights[i] * mirror_matrices[i] * gate[i]

        return AeonLayerState(
            weights=self._weights.copy(),
            mirror_matrices=mirror_matrices.copy(),
            resonance_signals=resonance_signals.copy(),
            gate_values=gate,
            layer_output=layer_output,
        )

    # ------------------------------------------------------------------
    # Weight update
    # ------------------------------------------------------------------

    def update_weights(self, new_weights: NDArray[np.float64]) -> None:
        """Replace dynamic resonance weights.

        Parameters
        ----------
        new_weights:
            1-D array of length n_models.  Values are L1-normalised automatically.
        """
        n = self.config.n_models
        if new_weights.shape != (n,):
            msg = f"new_weights must have shape ({n},), got {new_weights.shape}"
            raise ValueError(msg)
        total = float(np.sum(np.abs(new_weights)))
        if total == 0.0:
            msg = "new_weights must not be all-zero"
            raise ValueError(msg)
        self._weights = new_weights / total

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def resonance_energy(self, state: AeonLayerState) -> float:
        """Scalar resonance energy E = ||L_Aeon||_F (Frobenius norm)."""
        return float(np.linalg.norm(state.layer_output, ord="fro"))

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"AeonLayer(n_models={cfg.n_models}, beta={cfg.beta}, theta={cfg.theta})"
        )
