"""WeightingEngine — resonance weighting via entropy-governance + UTAC-Logistic + CREP.

Implements two resonance regimes from unified-mandala:

    S ∝ A  (area-proportional entropy):   w_i = exp(-lambda * H_i) / Z
    S ∝ V  (volume-proportional entropy): w_i = exp(-lambda * H_i^(d/2)) / Z

UTAC-Logistic gate (Unified Topology Activation Criterion):

    u_i = 1 / (1 + exp(-kappa * (C_i - tau)))

CREP coherence score (Coherence-Resonance-Entropy Product):

    crep_i = (1 - H_i / H_max) * rho_i * u_i

KaTeX:
    w_i^{(A)} = \\frac{\\exp(-\\lambda H_i)}{\\sum_j \\exp(-\\lambda H_j)}

    w_i^{(V)} = exp(-lambda * H_i^(d/2)) / sum_j exp(-lambda * H_j^(d/2))

    u_i = \\sigma\\!\\left(\\kappa(C_i - \\tau)\\right)

    \\mathrm{CREP}_i = \\left(1 - \\frac{H_i}{H_{\\max}}\\right) \\cdot \\rho_i \\cdot u_i
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class EntropyRegime(str, Enum):
    """Resonance weighting regime."""

    AREA = "area"
    VOLUME = "volume"


@dataclass
class WeightingConfig:
    """Configuration for WeightingEngine.

    Attributes:
        regime:     Entropy scaling regime (``area`` or ``volume``).
        lam:        Entropy penalty strength lambda (>0).
        kappa:      UTAC sharpness parameter kappa.
        tau:        UTAC coherence threshold tau.
        dim:        Embedding dimension d (used in volume regime, H^(d/2)).
        h_max:      Maximum entropy reference H_max for CREP normalisation.
    """

    regime: EntropyRegime = EntropyRegime.AREA
    lam: float = 1.0
    kappa: float = 2.0
    tau: float = 0.5
    dim: int = 2
    h_max: float = 1.0


@dataclass
class WeightingResult:
    """Output of WeightingEngine.compute.

    Attributes:
        weights:        Resonance weights w_i (normalised, shape [n]).
        utac_gates:     UTAC logistic gates u_i (shape [n]).
        crep_scores:    CREP coherence scores crep_i (shape [n]).
        entropy_scaled: Entropy values after regime scaling (shape [n]).
    """

    weights: NDArray[np.float64]
    utac_gates: NDArray[np.float64]
    crep_scores: NDArray[np.float64]
    entropy_scaled: NDArray[np.float64]


def _softmax(x: NDArray[np.float64]) -> NDArray[np.float64]:
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()  # type: ignore[return-value]


def _sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-x))  # type: ignore[return-value]


class WeightingEngine:
    """Compute resonance weights via entropy-governance, UTAC, and CREP.

    Parameters
    ----------
    config:
        WeightingConfig controlling regime, lambda, kappa, tau, dim, h_max.
    """

    def __init__(self, config: WeightingConfig | None = None) -> None:
        self.config: WeightingConfig = config or WeightingConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        entropies: NDArray[np.float64],
        coherences: NDArray[np.float64],
        rho: NDArray[np.float64],
    ) -> WeightingResult:
        """Compute weights, UTAC gates, and CREP scores.

        Parameters
        ----------
        entropies:
            Per-model entropy values H_i (shape [n], values in [0, h_max]).
        coherences:
            Per-model coherence values C_i (shape [n], values in [0, 1]).
        rho:
            Per-model resonance correlation rho_i (shape [n], values in [0, 1]).

        Returns
        -------
        WeightingResult with weights, utac_gates, crep_scores, entropy_scaled.
        """
        self._validate_inputs(entropies, coherences, rho)
        cfg = self.config

        h_scaled = self._scale_entropy(entropies)
        weights = _softmax(-cfg.lam * h_scaled)
        utac_gates = _sigmoid(cfg.kappa * (coherences - cfg.tau))
        h_norm = entropies / max(cfg.h_max, 1e-12)
        crep_scores = (1.0 - h_norm) * rho * utac_gates

        return WeightingResult(
            weights=weights,
            utac_gates=utac_gates,
            crep_scores=crep_scores,
            entropy_scaled=h_scaled,
        )

    def top_k_models(self, result: WeightingResult, k: int) -> NDArray[np.intp]:
        """Return indices of the top-k models by CREP score (descending)."""
        k = min(k, len(result.crep_scores))
        return np.argsort(result.crep_scores)[::-1][:k]  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scale_entropy(self, h: NDArray[np.float64]) -> NDArray[np.float64]:
        regime = self.config.regime
        d = self.config.dim
        if regime == EntropyRegime.AREA:
            return h
        return h ** (d / 2.0)  # type: ignore[return-value]

    @staticmethod
    def _validate_inputs(
        entropies: NDArray[np.float64],
        coherences: NDArray[np.float64],
        rho: NDArray[np.float64],
    ) -> None:
        for name, arr in [("entropies", entropies), ("coherences", coherences), ("rho", rho)]:
            if arr.ndim != 1:
                msg = f"{name} must be 1-D, got ndim={arr.ndim}"
                raise ValueError(msg)
        if not (entropies.shape == coherences.shape == rho.shape):
            msg = "entropies, coherences, and rho must have identical shapes"
            raise ValueError(msg)

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"WeightingEngine(regime={cfg.regime.value!r}, "
            f"lam={cfg.lam}, kappa={cfg.kappa}, tau={cfg.tau})"
        )
