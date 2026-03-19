"""Tests for WeightingEngine — entropy-governance, UTAC, CREP."""

from __future__ import annotations

import numpy as np
import pytest

from advanced_weighting_systems.weighting_engine import (
    EntropyRegime,
    WeightingConfig,
    WeightingEngine,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine() -> WeightingEngine:
    return WeightingEngine()


@pytest.fixture()
def sample_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = 4
    rng = np.random.default_rng(42)
    entropies = rng.uniform(0.1, 0.9, size=n)
    coherences = rng.uniform(0.3, 0.8, size=n)
    rho = rng.uniform(0.4, 1.0, size=n)
    return entropies, coherences, rho


# ---------------------------------------------------------------------------
# WeightingConfig
# ---------------------------------------------------------------------------


class TestWeightingConfig:
    def test_defaults(self) -> None:
        cfg = WeightingConfig()
        assert cfg.regime == EntropyRegime.AREA
        assert cfg.lam == 1.0
        assert cfg.kappa == 2.0
        assert cfg.tau == 0.5
        assert cfg.dim == 2
        assert cfg.h_max == 1.0

    def test_volume_regime(self) -> None:
        cfg = WeightingConfig(regime=EntropyRegime.VOLUME)
        assert cfg.regime == EntropyRegime.VOLUME


# ---------------------------------------------------------------------------
# WeightingEngine.compute — output types and shapes
# ---------------------------------------------------------------------------


class TestWeightingEngineCompute:
    def test_weights_shape(
        self, engine: WeightingEngine, sample_inputs: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        h, c, rho = sample_inputs
        result = engine.compute(h, c, rho)
        assert result.weights.shape == h.shape

    def test_weights_sum_to_one(
        self, engine: WeightingEngine, sample_inputs: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        h, c, rho = sample_inputs
        result = engine.compute(h, c, rho)
        np.testing.assert_allclose(result.weights.sum(), 1.0, atol=1e-10)

    def test_weights_non_negative(
        self, engine: WeightingEngine, sample_inputs: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        h, c, rho = sample_inputs
        result = engine.compute(h, c, rho)
        assert np.all(result.weights >= 0.0)

    def test_utac_gates_in_unit_interval(
        self, engine: WeightingEngine, sample_inputs: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        h, c, rho = sample_inputs
        result = engine.compute(h, c, rho)
        assert np.all(result.utac_gates >= 0.0)
        assert np.all(result.utac_gates <= 1.0)

    def test_crep_non_negative(
        self, engine: WeightingEngine, sample_inputs: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        h, c, rho = sample_inputs
        result = engine.compute(h, c, rho)
        assert np.all(result.crep_scores >= 0.0)

    def test_high_entropy_gets_low_weight(self) -> None:
        engine = WeightingEngine(config=WeightingConfig(lam=5.0))
        h = np.array([0.9, 0.1])
        c = np.array([0.5, 0.5])
        rho = np.array([0.5, 0.5])
        result = engine.compute(h, c, rho)
        assert result.weights[1] > result.weights[0]

    def test_utac_gate_below_tau_is_less_than_half(self) -> None:
        engine = WeightingEngine(config=WeightingConfig(kappa=10.0, tau=0.5))
        h = np.array([0.5])
        c = np.array([0.1])  # well below tau=0.5
        rho = np.array([0.5])
        result = engine.compute(h, c, rho)
        assert result.utac_gates[0] < 0.5

    def test_utac_gate_above_tau_is_greater_than_half(self) -> None:
        engine = WeightingEngine(config=WeightingConfig(kappa=10.0, tau=0.5))
        h = np.array([0.5])
        c = np.array([0.9])  # well above tau=0.5
        rho = np.array([0.5])
        result = engine.compute(h, c, rho)
        assert result.utac_gates[0] > 0.5


# ---------------------------------------------------------------------------
# EntropyRegime — volume vs area
# ---------------------------------------------------------------------------


class TestEntropyRegimes:
    def test_area_and_volume_differ_for_nonlinear_entropy(self) -> None:
        h = np.array([0.5, 0.3])
        c = np.array([0.5, 0.5])
        rho = np.array([0.5, 0.5])

        area_engine = WeightingEngine(WeightingConfig(regime=EntropyRegime.AREA, dim=4))
        vol_engine = WeightingEngine(WeightingConfig(regime=EntropyRegime.VOLUME, dim=4))

        r_a = area_engine.compute(h, c, rho)
        r_v = vol_engine.compute(h, c, rho)

        # They should differ in general (h^(d/2) != h for d != 2)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(r_a.weights, r_v.weights, atol=1e-10)

    def test_volume_regime_dim2_equals_area(self) -> None:
        h = np.array([0.5, 0.3])
        c = np.array([0.5, 0.5])
        rho = np.array([0.5, 0.5])

        area_engine = WeightingEngine(WeightingConfig(regime=EntropyRegime.AREA, dim=2))
        vol_engine = WeightingEngine(WeightingConfig(regime=EntropyRegime.VOLUME, dim=2))

        r_a = area_engine.compute(h, c, rho)
        r_v = vol_engine.compute(h, c, rho)
        np.testing.assert_allclose(r_a.weights, r_v.weights, atol=1e-10)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestWeightingEngineValidation:
    def test_2d_entropy_raises(self, engine: WeightingEngine) -> None:
        with pytest.raises(ValueError, match="entropies must be 1-D"):
            engine.compute(
                np.zeros((2, 2)),
                np.zeros(2),
                np.zeros(2),
            )

    def test_mismatched_shapes_raises(self, engine: WeightingEngine) -> None:
        with pytest.raises(ValueError, match="identical shapes"):
            engine.compute(
                np.zeros(3),
                np.zeros(3),
                np.zeros(4),
            )


# ---------------------------------------------------------------------------
# top_k_models
# ---------------------------------------------------------------------------


class TestTopKModels:
    def test_top_k_returns_correct_length(self, engine: WeightingEngine) -> None:
        h = np.array([0.5, 0.3, 0.8, 0.1])
        c = np.array([0.6, 0.7, 0.4, 0.9])
        rho = np.ones(4)
        result = engine.compute(h, c, rho)
        top = engine.top_k_models(result, k=2)
        assert len(top) == 2

    def test_top_k_capped_at_n(self, engine: WeightingEngine) -> None:
        h = np.array([0.5, 0.3])
        c = np.array([0.6, 0.7])
        rho = np.ones(2)
        result = engine.compute(h, c, rho)
        top = engine.top_k_models(result, k=100)
        assert len(top) == 2

    def test_top_1_is_highest_crep(self, engine: WeightingEngine) -> None:
        h = np.array([0.1, 0.9, 0.5])
        c = np.array([0.9, 0.1, 0.5])
        rho = np.ones(3)
        result = engine.compute(h, c, rho)
        top1 = engine.top_k_models(result, k=1)[0]
        assert result.crep_scores[top1] == result.crep_scores.max()


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestWeightingEngineRepr:
    def test_repr(self, engine: WeightingEngine) -> None:
        r = repr(engine)
        assert "WeightingEngine" in r
        assert "area" in r
