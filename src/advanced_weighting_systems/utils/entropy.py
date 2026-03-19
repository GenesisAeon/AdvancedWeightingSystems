"""Entropy utilities integrating entropy-table and entropy-governance semantics.

Provides:
    - shannon_entropy: discrete Shannon entropy H(p) = -sum p_i log p_i
    - EntropyTable:    lookup/cache structure compatible with entropy-table package
    - entropy_governance_weight: governance-weighted entropy for AeonLayer inputs

KaTeX:
    H(p) = -\\sum_i p_i \\ln p_i

    w^{\\mathrm{gov}}_i = \\frac{\\exp(-\\lambda H_i)}{\\sum_j \\exp(-\\lambda H_j)}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


def shannon_entropy(
    probs: NDArray[np.float64],
    base: float = float(np.e),
    epsilon: float = 1e-12,
) -> float:
    """Compute the Shannon entropy H(p) = -sum_i p_i * log_base(p_i).

    Parameters
    ----------
    probs:
        Probability distribution (non-negative, will be L1-normalised).
    base:
        Logarithm base (default: natural log, base=e).
    epsilon:
        Small value added before log to avoid log(0).

    Returns
    -------
    Scalar entropy value H >= 0.
    """
    p = np.asarray(probs, dtype=np.float64)
    if p.sum() <= 0.0:
        msg = "probs must sum to a positive value"
        raise ValueError(msg)
    p = p / p.sum()
    log_base = np.log(base) if base != float(np.e) else 1.0
    h = -np.sum(p * np.log(p + epsilon)) / log_base
    return float(max(h, 0.0))


def entropy_governance_weight(
    entropies: NDArray[np.float64],
    lam: float = 1.0,
) -> NDArray[np.float64]:
    """Compute governance weights from entropy: w_i = softmax(-lambda * H_i).

    Parameters
    ----------
    entropies:
        Per-model entropy values H_i (1-D, non-negative).
    lam:
        Penalty strength lambda > 0.

    Returns
    -------
    1-D weight array (same shape as entropies), L1-normalised.
    """
    h = np.asarray(entropies, dtype=np.float64)
    logits = -lam * h
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()  # type: ignore[return-value]


@dataclass
class EntropyTableEntry:
    """One row in the EntropyTable.

    Attributes:
        model_id:  Unique string identifier for the model.
        entropy:   Current Shannon entropy H_i.
        metadata:  Arbitrary key-value metadata (e.g., layer name, step).
    """

    model_id: str
    entropy: float
    metadata: dict[str, Any] = field(default_factory=dict)


class EntropyTable:
    """In-memory entropy-table compatible with entropy-governance package semantics.

    Acts as a mutable registry of per-model entropy values, supporting
    bulk export for WeightingEngine.compute.

    Parameters
    ----------
    h_max:
        Maximum entropy reference used for CREP normalisation.
    """

    def __init__(self, h_max: float = 1.0) -> None:
        self.h_max = h_max
        self._entries: dict[str, EntropyTableEntry] = {}

    # ------------------------------------------------------------------

    def upsert(
        self,
        model_id: str,
        entropy: float,
        **metadata: Any,  # noqa: ANN401
    ) -> None:
        """Insert or update the entropy entry for model_id.

        Parameters
        ----------
        model_id:  Unique identifier.
        entropy:   New entropy value (must be >= 0).
        **metadata: Arbitrary key-value annotations.
        """
        if entropy < 0.0:
            msg = f"entropy must be >= 0, got {entropy}"
            raise ValueError(msg)
        self._entries[model_id] = EntropyTableEntry(
            model_id=model_id,
            entropy=min(entropy, self.h_max),
            metadata=dict(metadata),
        )

    def get(self, model_id: str) -> EntropyTableEntry:
        """Retrieve entropy entry by model_id (raises KeyError if absent)."""
        return self._entries[model_id]

    def remove(self, model_id: str) -> None:
        """Remove the entry for model_id (no-op if absent)."""
        self._entries.pop(model_id, None)

    def export_entropies(self, model_ids: list[str]) -> NDArray[np.float64]:
        """Export entropy values in the order given by model_ids.

        Parameters
        ----------
        model_ids:
            Ordered list of model identifiers.

        Returns
        -------
        1-D float64 array of entropy values.
        """
        values = []
        for mid in model_ids:
            entry = self._entries.get(mid)
            if entry is None:
                msg = f"Model ID {mid!r} not found in EntropyTable"
                raise KeyError(msg)
            values.append(entry.entropy)
        return np.array(values, dtype=np.float64)

    @property
    def model_ids(self) -> list[str]:
        """Ordered list of registered model IDs."""
        return list(self._entries.keys())

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"EntropyTable(n_models={len(self)}, h_max={self.h_max})"
