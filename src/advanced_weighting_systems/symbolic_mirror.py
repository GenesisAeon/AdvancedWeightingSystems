"""SymbolicMirror — Sigillin-based reflection for heterogeneous NN adapters.

Implements Sigillin-Bridges and MandalaMap-Topology to build Mirror-Matrices
M_i for any registered NN adapter (Transformer, CNN, RNN, GraphNN, Spike, …).

The Mirror-Matrix M_i is constructed as:

    M_i = P_i @ Phi_sigil @ P_i^T

where:
    P_i     : adapter-specific projection matrix (shape [d, d])
    Phi_sigil: Sigillin phase matrix (shape [d, d]) — encodes the mandala topology

KaTeX:
    M_i = P_i \\, \\Phi_{\\text{sigil}} \\, P_i^\\top
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Sigillin phase matrix
# ---------------------------------------------------------------------------


def build_sigillin_phase_matrix(dim: int, seed: int = 42) -> NDArray[np.float64]:
    """Build the Sigillin phase matrix Phi_sigil of shape [dim, dim].

    The phase matrix encodes the MandalaMap topology as a symmetric positive
    semi-definite matrix whose off-diagonal entries decay with index distance.

    Parameters
    ----------
    dim:  Embedding dimension d.
    seed: RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((dim, dim))
    phi = (base + base.T) / 2.0
    i_idx, j_idx = np.indices((dim, dim))
    decay = np.exp(-0.5 * np.abs(i_idx - j_idx).astype(np.float64))
    phi = phi * decay
    return phi / (np.linalg.norm(phi, ord="fro") + 1e-12)


# ---------------------------------------------------------------------------
# Adapter base class
# ---------------------------------------------------------------------------


class NNAdapter(ABC):
    """Base class for heterogeneous NN adapters used in SymbolicMirror."""

    adapter_type: ClassVar[str] = "base"

    def __init__(self, dim: int) -> None:
        self.dim = dim

    @abstractmethod
    def projection_matrix(self) -> NDArray[np.float64]:
        """Return the adapter-specific projection P_i (shape [dim, dim])."""

    @abstractmethod
    def resonance_signal(self, activations: NDArray[np.float64]) -> float:
        """Compute a scalar resonance signal R_i from layer activations."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim})"


# ---------------------------------------------------------------------------
# Concrete adapters
# ---------------------------------------------------------------------------


class TransformerAdapter(NNAdapter):
    """Adapter for Transformer-based models (attention resonance)."""

    adapter_type: ClassVar[str] = "transformer"

    def __init__(self, dim: int, n_heads: int = 4) -> None:
        super().__init__(dim)
        self.n_heads = n_heads
        rng = np.random.default_rng(seed=0)
        raw = rng.standard_normal((dim, dim))
        u, _, vt = np.linalg.svd(raw)
        self._proj: NDArray[np.float64] = u @ vt

    def projection_matrix(self) -> NDArray[np.float64]:
        return self._proj

    def resonance_signal(self, activations: NDArray[np.float64]) -> float:
        return float(np.mean(np.tanh(activations)))


class CNNAdapter(NNAdapter):
    """Adapter for Convolutional Neural Networks (spatial resonance)."""

    adapter_type: ClassVar[str] = "cnn"

    def __init__(self, dim: int, kernel_size: int = 3) -> None:
        super().__init__(dim)
        self.kernel_size = kernel_size
        rng = np.random.default_rng(seed=1)
        noise = rng.standard_normal((dim, dim))
        self._proj: NDArray[np.float64] = np.eye(dim, dtype=np.float64) + 0.1 * noise

    def projection_matrix(self) -> NDArray[np.float64]:
        return self._proj

    def resonance_signal(self, activations: NDArray[np.float64]) -> float:
        return float(np.max(np.abs(activations)))


class RNNAdapter(NNAdapter):
    """Adapter for Recurrent Neural Networks (temporal resonance)."""

    adapter_type: ClassVar[str] = "rnn"

    def __init__(self, dim: int, hidden_size: int = 64) -> None:
        super().__init__(dim)
        self.hidden_size = hidden_size
        rng = np.random.default_rng(seed=2)
        raw = rng.standard_normal((dim, dim))
        u, _, vt = np.linalg.svd(raw)
        self._proj: NDArray[np.float64] = u @ vt

    def projection_matrix(self) -> NDArray[np.float64]:
        return self._proj

    def resonance_signal(self, activations: NDArray[np.float64]) -> float:
        return float(np.std(activations))


class GraphNNAdapter(NNAdapter):
    """Adapter for Graph Neural Networks (topology resonance)."""

    adapter_type: ClassVar[str] = "graph"

    def __init__(self, dim: int, n_nodes: int = 16) -> None:
        super().__init__(dim)
        self.n_nodes = n_nodes
        rng = np.random.default_rng(seed=3)
        raw = rng.standard_normal((dim, dim))
        self._proj: NDArray[np.float64] = raw / (np.linalg.norm(raw, ord=2) + 1e-12)

    def projection_matrix(self) -> NDArray[np.float64]:
        return self._proj

    def resonance_signal(self, activations: NDArray[np.float64]) -> float:
        return float(np.sum(activations**2) / (activations.size + 1e-12))


class SpikeAdapter(NNAdapter):
    """Adapter for Spiking Neural Networks (temporal spike resonance)."""

    adapter_type: ClassVar[str] = "spike"

    def __init__(self, dim: int, threshold: float = 0.5) -> None:
        super().__init__(dim)
        self.threshold = threshold
        rng = np.random.default_rng(seed=4)
        diag = rng.uniform(0.5, 1.5, size=dim)
        self._proj: NDArray[np.float64] = np.diag(diag)

    def projection_matrix(self) -> NDArray[np.float64]:
        return self._proj

    def resonance_signal(self, activations: NDArray[np.float64]) -> float:
        spikes = (activations > self.threshold).astype(np.float64)
        return float(np.mean(spikes))


# ---------------------------------------------------------------------------
# SymbolicMirror
# ---------------------------------------------------------------------------


@dataclass
class MirrorResult:
    """Output of SymbolicMirror.reflect.

    Attributes:
        mirror_matrices:   Stack of M_i matrices (shape [n, d, d]).
        resonance_signals: Scalar R_i per adapter (shape [n]).
        adapter_types:     List of adapter type strings.
    """

    mirror_matrices: NDArray[np.float64]
    resonance_signals: NDArray[np.float64]
    adapter_types: list[str]


class SymbolicMirror:
    """Build Mirror-Matrices M_i = P_i @ Phi_sigil @ P_i^T for all adapters.

    Parameters
    ----------
    dim:
        Embedding dimension d shared by all adapters.
    sigil_seed:
        RNG seed for the Sigillin phase matrix Phi_sigil.
    """

    def __init__(self, dim: int = 16, sigil_seed: int = 42) -> None:
        self.dim = dim
        self._phi: NDArray[np.float64] = build_sigillin_phase_matrix(dim, sigil_seed)
        self._adapters: list[NNAdapter] = []

    # ------------------------------------------------------------------

    def register(self, adapter: NNAdapter) -> SymbolicMirror:
        """Register an NNAdapter (fluent API)."""
        if adapter.dim != self.dim:
            msg = (
                f"Adapter dim {adapter.dim} != SymbolicMirror dim {self.dim}. "
                "All adapters must share the same embedding dimension."
            )
            raise ValueError(msg)
        self._adapters.append(adapter)
        return self

    @property
    def n_adapters(self) -> int:
        """Number of registered adapters."""
        return len(self._adapters)

    # ------------------------------------------------------------------

    def reflect(
        self, activations_list: list[NDArray[np.float64]]
    ) -> MirrorResult:
        """Compute Mirror-Matrices and resonance signals for all adapters.

        Parameters
        ----------
        activations_list:
            One activation array per registered adapter, in registration order.

        Returns
        -------
        MirrorResult with mirror_matrices [n, d, d], resonance_signals [n].
        """
        n = self.n_adapters
        if len(activations_list) != n:
            msg = f"Expected {n} activation arrays, got {len(activations_list)}"
            raise ValueError(msg)

        d = self.dim
        mirror_matrices = np.empty((n, d, d), dtype=np.float64)
        resonance_signals = np.empty(n, dtype=np.float64)
        adapter_types: list[str] = []

        for i, (adapter, acts) in enumerate(zip(self._adapters, activations_list, strict=True)):
            p = adapter.projection_matrix()
            mirror_matrices[i] = p @ self._phi @ p.T
            resonance_signals[i] = adapter.resonance_signal(acts)
            adapter_types.append(adapter.adapter_type)

        return MirrorResult(
            mirror_matrices=mirror_matrices,
            resonance_signals=resonance_signals,
            adapter_types=adapter_types,
        )

    def mandala_topology_matrix(self) -> NDArray[np.float64]:
        """Return the global MandalaMap topology (Phi_sigil) as read-only view."""
        return self._phi.copy()

    def __repr__(self) -> str:
        types = [a.adapter_type for a in self._adapters]
        return f"SymbolicMirror(dim={self.dim}, adapters={types})"
