"""Tests for ResonanceCoupling end-to-end pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from advanced_weighting_systems.aeon_layer import AeonLayerConfig
from advanced_weighting_systems.models.coupling import CouplingOutput, ResonanceCoupling
from advanced_weighting_systems.symbolic_mirror import (
    CNNAdapter,
    GraphNNAdapter,
    RNNAdapter,
    SpikeAdapter,
    TransformerAdapter,
)
from advanced_weighting_systems.weighting_engine import WeightingConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


DIM = 8


@pytest.fixture()
def adapters() -> list:
    return [
        TransformerAdapter(dim=DIM),
        CNNAdapter(dim=DIM),
        RNNAdapter(dim=DIM),
        GraphNNAdapter(dim=DIM),
    ]


@pytest.fixture()
def coupling(adapters: list) -> ResonanceCoupling:
    return ResonanceCoupling(
        adapters=adapters,
        dim=DIM,
        aeon_config=AeonLayerConfig(n_models=4, beta=1.5),
        weighting_config=WeightingConfig(lam=1.0),
    )


@pytest.fixture()
def step_inputs() -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    n = 4
    rng = np.random.default_rng(7)
    acts = [rng.standard_normal(DIM) for _ in range(n)]
    h = rng.uniform(0.1, 0.9, size=n)
    c = rng.uniform(0.3, 0.8, size=n)
    rho = rng.uniform(0.4, 1.0, size=n)
    return acts, h, c, rho


# ---------------------------------------------------------------------------
# ResonanceCoupling.step
# ---------------------------------------------------------------------------


class TestResonanceCouplingStep:
    def test_returns_coupling_output(
        self, coupling: ResonanceCoupling, step_inputs: tuple
    ) -> None:
        acts, h, c, rho = step_inputs
        out = coupling.step(acts, h, c, rho)
        assert isinstance(out, CouplingOutput)

    def test_energy_non_negative(
        self, coupling: ResonanceCoupling, step_inputs: tuple
    ) -> None:
        acts, h, c, rho = step_inputs
        out = coupling.step(acts, h, c, rho)
        assert out.energy >= 0.0

    def test_layer_output_shape(
        self, coupling: ResonanceCoupling, step_inputs: tuple
    ) -> None:
        acts, h, c, rho = step_inputs
        out = coupling.step(acts, h, c, rho)
        assert out.aeon_state.layer_output.shape == (DIM, DIM)

    def test_weights_sum_to_one(
        self, coupling: ResonanceCoupling, step_inputs: tuple
    ) -> None:
        acts, h, c, rho = step_inputs
        out = coupling.step(acts, h, c, rho)
        np.testing.assert_allclose(out.weighting_result.weights.sum(), 1.0, atol=1e-10)

    def test_mirror_matrices_count(
        self, coupling: ResonanceCoupling, step_inputs: tuple
    ) -> None:
        acts, h, c, rho = step_inputs
        out = coupling.step(acts, h, c, rho)
        assert out.mirror_result.mirror_matrices.shape[0] == 4

    def test_deterministic_with_fixed_inputs(
        self, coupling: ResonanceCoupling, step_inputs: tuple
    ) -> None:
        acts, h, c, rho = step_inputs
        out1 = coupling.step(acts, h, c, rho)
        out2 = coupling.step(acts, h, c, rho)
        np.testing.assert_allclose(out1.energy, out2.energy, atol=1e-12)


# ---------------------------------------------------------------------------
# ResonanceCoupling properties
# ---------------------------------------------------------------------------


class TestResonanceCouplingProperties:
    def test_n_models(self, coupling: ResonanceCoupling) -> None:
        assert coupling.n_models == 4

    def test_repr(self, coupling: ResonanceCoupling) -> None:
        r = repr(coupling)
        assert "ResonanceCoupling" in r
        assert "n_models=4" in r


# ---------------------------------------------------------------------------
# Contract tests — mirror-machine compatible interface
# ---------------------------------------------------------------------------


class TestMirrorMachineContract:
    """Verify SymbolicMirror output satisfies mirror-machine structural contracts."""

    def test_mirror_matrix_shape_contract(self, coupling: ResonanceCoupling) -> None:
        rng = np.random.default_rng(0)
        acts = [rng.standard_normal(DIM) for _ in range(4)]
        h = np.ones(4) * 0.5
        c = np.ones(4) * 0.5
        rho = np.ones(4)
        out = coupling.step(acts, h, c, rho)
        m = out.mirror_result.mirror_matrices
        assert m.ndim == 3
        assert m.shape[1] == m.shape[2]

    def test_resonance_signals_are_finite(self, coupling: ResonanceCoupling) -> None:
        rng = np.random.default_rng(0)
        acts = [rng.standard_normal(DIM) for _ in range(4)]
        h = np.ones(4) * 0.5
        c = np.ones(4) * 0.5
        rho = np.ones(4)
        out = coupling.step(acts, h, c, rho)
        assert np.all(np.isfinite(out.mirror_result.resonance_signals))


# ---------------------------------------------------------------------------
# Contract tests — entropy-governance compatible
# ---------------------------------------------------------------------------


class TestEntropyGovernanceContract:
    """Verify WeightingEngine honours entropy-governance monotonicity contract."""

    def test_monotone_weight_wrt_entropy(self) -> None:
        from advanced_weighting_systems.weighting_engine import WeightingConfig, WeightingEngine

        engine = WeightingEngine(WeightingConfig(lam=2.0))
        h_low = np.array([0.1, 0.5])
        h_high = np.array([0.9, 0.5])
        c = np.array([0.5, 0.5])
        rho = np.ones(2)

        r_low = engine.compute(h_low, c, rho)
        r_high = engine.compute(h_high, c, rho)

        # Model 0 has lower entropy in h_low → should receive higher weight
        assert r_low.weights[0] > r_high.weights[0]


# ---------------------------------------------------------------------------
# Contract tests — utac-core compatible
# ---------------------------------------------------------------------------


class TestUTACCoreContract:
    """Verify UTAC gates satisfy monotonicity and range contracts."""

    def test_utac_monotone_wrt_coherence(self) -> None:
        from advanced_weighting_systems.weighting_engine import WeightingConfig, WeightingEngine

        engine = WeightingEngine(WeightingConfig(kappa=5.0, tau=0.5))
        h = np.array([0.5, 0.5])
        c_low = np.array([0.1, 0.5])
        c_high = np.array([0.9, 0.5])
        rho = np.ones(2)

        r_low = engine.compute(h, c_low, rho)
        r_high = engine.compute(h, c_high, rho)

        assert r_high.utac_gates[0] > r_low.utac_gates[0]

    def test_utac_range(self) -> None:
        from advanced_weighting_systems.weighting_engine import WeightingConfig, WeightingEngine

        engine = WeightingEngine(WeightingConfig())
        h = np.random.default_rng(0).uniform(0, 1, size=10)
        c = np.random.default_rng(1).uniform(0, 1, size=10)
        rho = np.ones(10)
        result = engine.compute(h, c, rho)
        assert np.all(result.utac_gates >= 0.0)
        assert np.all(result.utac_gates <= 1.0)


# ---------------------------------------------------------------------------
# Full stack — spike + all adapters
# ---------------------------------------------------------------------------


class TestFullStackAllAdapters:
    def test_all_five_adapter_types(self) -> None:
        adapters = [
            TransformerAdapter(dim=DIM),
            CNNAdapter(dim=DIM),
            RNNAdapter(dim=DIM),
            GraphNNAdapter(dim=DIM),
            SpikeAdapter(dim=DIM),
        ]
        coupling = ResonanceCoupling(adapters=adapters, dim=DIM)
        rng = np.random.default_rng(42)
        acts = [rng.standard_normal(DIM) for _ in range(5)]
        h = rng.uniform(0.1, 0.9, size=5)
        c = rng.uniform(0.3, 0.8, size=5)
        rho = rng.uniform(0.4, 1.0, size=5)
        out = coupling.step(acts, h, c, rho)
        assert out.aeon_state.layer_output.shape == (DIM, DIM)
        assert np.isfinite(out.energy)
