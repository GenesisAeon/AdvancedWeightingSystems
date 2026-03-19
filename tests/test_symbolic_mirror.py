"""Tests for SymbolicMirror and all NNAdapter implementations."""

from __future__ import annotations

import numpy as np
import pytest

from advanced_weighting_systems.symbolic_mirror import (
    CNNAdapter,
    GraphNNAdapter,
    NNAdapter,
    RNNAdapter,
    SpikeAdapter,
    SymbolicMirror,
    TransformerAdapter,
    build_sigillin_phase_matrix,
)

# ---------------------------------------------------------------------------
# build_sigillin_phase_matrix
# ---------------------------------------------------------------------------


class TestBuildSigillinPhaseMatrix:
    def test_shape(self) -> None:
        phi = build_sigillin_phase_matrix(8)
        assert phi.shape == (8, 8)

    def test_symmetric(self) -> None:
        phi = build_sigillin_phase_matrix(6)
        np.testing.assert_allclose(phi, phi.T, atol=1e-12)

    def test_frobenius_norm_unit(self) -> None:
        phi = build_sigillin_phase_matrix(5)
        np.testing.assert_allclose(np.linalg.norm(phi, ord="fro"), 1.0, atol=1e-10)

    def test_reproducible_with_same_seed(self) -> None:
        p1 = build_sigillin_phase_matrix(4, seed=99)
        p2 = build_sigillin_phase_matrix(4, seed=99)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_differ(self) -> None:
        p1 = build_sigillin_phase_matrix(4, seed=0)
        p2 = build_sigillin_phase_matrix(4, seed=1)
        assert not np.allclose(p1, p2)


# ---------------------------------------------------------------------------
# Adapter fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=["trans", "cnn", "rnn", "graph", "spike"])
def adapter(request: pytest.FixtureRequest) -> NNAdapter:
    dim = 8
    adapters = {
        "trans": TransformerAdapter(dim=dim, n_heads=2),
        "cnn": CNNAdapter(dim=dim, kernel_size=3),
        "rnn": RNNAdapter(dim=dim, hidden_size=32),
        "graph": GraphNNAdapter(dim=dim, n_nodes=8),
        "spike": SpikeAdapter(dim=dim, threshold=0.3),
    }
    return adapters[request.param]


# ---------------------------------------------------------------------------
# NNAdapter interface
# ---------------------------------------------------------------------------


class TestNNAdapters:
    def test_projection_matrix_shape(self, adapter: NNAdapter) -> None:
        p = adapter.projection_matrix()
        assert p.shape == (adapter.dim, adapter.dim)

    def test_resonance_signal_scalar(self, adapter: NNAdapter) -> None:
        rng = np.random.default_rng(0)
        acts = rng.standard_normal(adapter.dim)
        r = adapter.resonance_signal(acts)
        assert isinstance(r, float)

    def test_repr(self, adapter: NNAdapter) -> None:
        assert "dim=" in repr(adapter)

    def test_adapter_type_set(self, adapter: NNAdapter) -> None:
        assert isinstance(adapter.adapter_type, str)
        assert len(adapter.adapter_type) > 0


class TestTransformerAdapter:
    def test_resonance_bounded(self) -> None:
        adapter = TransformerAdapter(dim=8)
        acts = np.array([0.0] * 8)
        r = adapter.resonance_signal(acts)
        assert r == pytest.approx(0.0, abs=1e-10)


class TestCNNAdapter:
    def test_resonance_is_max_abs(self) -> None:
        adapter = CNNAdapter(dim=4)
        acts = np.array([0.0, 3.0, -5.0, 1.0])
        assert adapter.resonance_signal(acts) == pytest.approx(5.0, abs=1e-10)


class TestRNNAdapter:
    def test_resonance_is_std(self) -> None:
        adapter = RNNAdapter(dim=4)
        acts = np.array([1.0, 1.0, 1.0, 1.0])
        assert adapter.resonance_signal(acts) == pytest.approx(0.0, abs=1e-10)


class TestGraphNNAdapter:
    def test_resonance_non_negative(self) -> None:
        adapter = GraphNNAdapter(dim=6)
        acts = np.random.default_rng(0).standard_normal(6)
        assert adapter.resonance_signal(acts) >= 0.0


class TestSpikeAdapter:
    def test_resonance_rate_all_zero_acts(self) -> None:
        adapter = SpikeAdapter(dim=4, threshold=0.5)
        acts = np.zeros(4)
        assert adapter.resonance_signal(acts) == pytest.approx(0.0, abs=1e-10)

    def test_resonance_rate_all_above_threshold(self) -> None:
        adapter = SpikeAdapter(dim=4, threshold=0.0)
        acts = np.ones(4)
        assert adapter.resonance_signal(acts) == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# SymbolicMirror
# ---------------------------------------------------------------------------


@pytest.fixture()
def mirror() -> SymbolicMirror:
    sm = SymbolicMirror(dim=8, sigil_seed=42)
    sm.register(TransformerAdapter(dim=8))
    sm.register(CNNAdapter(dim=8))
    sm.register(RNNAdapter(dim=8))
    return sm


class TestSymbolicMirror:
    def test_register_fluent_returns_self(self) -> None:
        sm = SymbolicMirror(dim=8)
        result = sm.register(TransformerAdapter(dim=8))
        assert result is sm

    def test_n_adapters(self, mirror: SymbolicMirror) -> None:
        assert mirror.n_adapters == 3

    def test_register_wrong_dim_raises(self) -> None:
        sm = SymbolicMirror(dim=8)
        with pytest.raises(ValueError, match="Adapter dim"):
            sm.register(TransformerAdapter(dim=4))

    def test_reflect_returns_correct_shapes(self, mirror: SymbolicMirror) -> None:
        rng = np.random.default_rng(0)
        acts = [rng.standard_normal(8) for _ in range(3)]
        result = mirror.reflect(acts)
        assert result.mirror_matrices.shape == (3, 8, 8)
        assert result.resonance_signals.shape == (3,)
        assert len(result.adapter_types) == 3

    def test_reflect_wrong_number_of_acts_raises(self, mirror: SymbolicMirror) -> None:
        rng = np.random.default_rng(0)
        acts = [rng.standard_normal(8) for _ in range(2)]
        with pytest.raises(ValueError, match="Expected 3 activation arrays"):
            mirror.reflect(acts)

    def test_mirror_matrices_symmetric_if_projection_orthogonal(self) -> None:
        sm = SymbolicMirror(dim=8, sigil_seed=0)
        sm.register(RNNAdapter(dim=8))  # orthogonal projection
        acts = [np.random.default_rng(0).standard_normal(8)]
        result = sm.reflect(acts)
        m = result.mirror_matrices[0]
        np.testing.assert_allclose(m, m.T, atol=1e-10)

    def test_adapter_types_correct(self, mirror: SymbolicMirror) -> None:
        rng = np.random.default_rng(0)
        acts = [rng.standard_normal(8) for _ in range(3)]
        result = mirror.reflect(acts)
        assert result.adapter_types == ["transformer", "cnn", "rnn"]

    def test_mandala_topology_matrix_shape(self, mirror: SymbolicMirror) -> None:
        phi = mirror.mandala_topology_matrix()
        assert phi.shape == (8, 8)

    def test_repr_contains_adapter_types(self, mirror: SymbolicMirror) -> None:
        r = repr(mirror)
        assert "transformer" in r
        assert "cnn" in r
