"""Tests for AeonLayer — contract: L_Aeon = sum_i w_i * M_i * sigma(beta*(R_i - Theta))."""

from __future__ import annotations

import numpy as np
import pytest

from advanced_weighting_systems.aeon_layer import AeonLayer, AeonLayerConfig, AeonLayerState

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_config() -> AeonLayerConfig:
    return AeonLayerConfig(beta=1.0, theta=0.0, n_models=3)


@pytest.fixture()
def layer(simple_config: AeonLayerConfig) -> AeonLayer:
    return AeonLayer(config=simple_config)


@pytest.fixture()
def mirror_matrices() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((3, 4, 4))


@pytest.fixture()
def resonance_signals() -> np.ndarray:
    return np.array([0.5, -0.3, 1.2])


# ---------------------------------------------------------------------------
# AeonLayerConfig
# ---------------------------------------------------------------------------


class TestAeonLayerConfig:
    def test_defaults(self) -> None:
        cfg = AeonLayerConfig()
        assert cfg.beta == 1.0
        assert cfg.theta == 0.0
        assert cfg.n_models == 1

    def test_custom(self) -> None:
        cfg = AeonLayerConfig(beta=2.0, theta=0.5, n_models=5)
        assert cfg.beta == 2.0
        assert cfg.n_models == 5


# ---------------------------------------------------------------------------
# AeonLayer construction
# ---------------------------------------------------------------------------


class TestAeonLayerConstruction:
    def test_default_weights_uniform(self) -> None:
        layer = AeonLayer(config=AeonLayerConfig(n_models=4))
        w = layer.weights
        assert w.shape == (4,)
        np.testing.assert_allclose(w, 0.25, atol=1e-12)

    def test_custom_weights_accepted(self) -> None:
        w = np.array([0.1, 0.4, 0.5])
        layer = AeonLayer(config=AeonLayerConfig(n_models=3), weights=w)
        np.testing.assert_allclose(layer.weights, w, atol=1e-12)

    def test_wrong_weight_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="weights must have shape"):
            AeonLayer(config=AeonLayerConfig(n_models=3), weights=np.array([0.5, 0.5]))

    def test_weights_are_copy(self) -> None:
        w = np.array([0.5, 0.5])
        layer = AeonLayer(config=AeonLayerConfig(n_models=2), weights=w)
        w[0] = 99.0
        assert layer.weights[0] != 99.0


# ---------------------------------------------------------------------------
# AeonLayer.forward
# ---------------------------------------------------------------------------


class TestAeonLayerForward:
    def test_output_shape(
        self,
        layer: AeonLayer,
        mirror_matrices: np.ndarray,
        resonance_signals: np.ndarray,
    ) -> None:
        state = layer.forward(mirror_matrices, resonance_signals)
        assert state.layer_output.shape == (4, 4)

    def test_gate_values_in_unit_interval(
        self,
        layer: AeonLayer,
        mirror_matrices: np.ndarray,
        resonance_signals: np.ndarray,
    ) -> None:
        state = layer.forward(mirror_matrices, resonance_signals)
        assert np.all(state.gate_values >= 0.0)
        assert np.all(state.gate_values <= 1.0)

    def test_gate_at_zero_signal_equals_half(self) -> None:
        cfg = AeonLayerConfig(beta=1.0, theta=0.0, n_models=1)
        layer = AeonLayer(config=cfg)
        m = np.eye(2)[np.newaxis]  # shape (1, 2, 2)
        r = np.array([0.0])
        state = layer.forward(m, r)
        np.testing.assert_allclose(state.gate_values[0], 0.5, atol=1e-10)

    def test_formula_single_model(self) -> None:
        cfg = AeonLayerConfig(beta=2.0, theta=0.1, n_models=1)
        layer = AeonLayer(config=cfg, weights=np.array([1.0]))
        m = np.eye(3)[np.newaxis]
        r = np.array([0.5])
        state = layer.forward(m, r)
        expected_gate = 1.0 / (1.0 + np.exp(-2.0 * (0.5 - 0.1)))
        expected_out = 1.0 * np.eye(3) * expected_gate
        np.testing.assert_allclose(state.layer_output, expected_out, atol=1e-10)

    def test_state_stores_inputs(
        self,
        layer: AeonLayer,
        mirror_matrices: np.ndarray,
        resonance_signals: np.ndarray,
    ) -> None:
        state = layer.forward(mirror_matrices, resonance_signals)
        np.testing.assert_array_equal(state.mirror_matrices, mirror_matrices)
        np.testing.assert_array_equal(state.resonance_signals, resonance_signals)

    def test_wrong_mirror_shape_raises(self, layer: AeonLayer) -> None:
        with pytest.raises(ValueError, match="mirror_matrices must have shape"):
            layer.forward(np.zeros((2, 4, 4)), np.zeros(3))

    def test_wrong_signal_shape_raises(
        self, layer: AeonLayer, mirror_matrices: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="resonance_signals must have shape"):
            layer.forward(mirror_matrices, np.zeros(2))

    def test_beta_zero_gives_half_gate(self) -> None:
        cfg = AeonLayerConfig(beta=0.0, theta=0.0, n_models=2)
        layer = AeonLayer(config=cfg)
        m = np.ones((2, 2, 2))
        r = np.array([10.0, -10.0])
        state = layer.forward(m, r)
        np.testing.assert_allclose(state.gate_values, 0.5, atol=1e-10)


# ---------------------------------------------------------------------------
# AeonLayer.update_weights
# ---------------------------------------------------------------------------


class TestAeonLayerUpdateWeights:
    def test_normalises_weights(self, layer: AeonLayer) -> None:
        layer.update_weights(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(layer.weights.sum(), 1.0, atol=1e-10)

    def test_wrong_shape_raises(self, layer: AeonLayer) -> None:
        with pytest.raises(ValueError, match="new_weights must have shape"):
            layer.update_weights(np.array([0.5, 0.5]))

    def test_zero_weights_raises(self, layer: AeonLayer) -> None:
        with pytest.raises(ValueError, match="must not be all-zero"):
            layer.update_weights(np.array([0.0, 0.0, 0.0]))


# ---------------------------------------------------------------------------
# AeonLayer.resonance_energy
# ---------------------------------------------------------------------------


class TestAeonLayerResonanceEnergy:
    def test_energy_non_negative(
        self,
        layer: AeonLayer,
        mirror_matrices: np.ndarray,
        resonance_signals: np.ndarray,
    ) -> None:
        state = layer.forward(mirror_matrices, resonance_signals)
        assert layer.resonance_energy(state) >= 0.0

    def test_energy_zero_for_zero_output(self, layer: AeonLayer) -> None:
        state = AeonLayerState(
            weights=layer.weights,
            mirror_matrices=np.zeros((3, 4, 4)),
            resonance_signals=np.zeros(3),
            gate_values=np.zeros(3),
            layer_output=np.zeros((4, 4)),
        )
        assert layer.resonance_energy(state) == 0.0


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestAeonLayerRepr:
    def test_repr_contains_key_info(self, layer: AeonLayer) -> None:
        r = repr(layer)
        assert "AeonLayer" in r
        assert "n_models=3" in r
