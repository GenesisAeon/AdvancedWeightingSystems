"""Tests for entropy utilities — shannon_entropy, EntropyTable, entropy_governance_weight."""

from __future__ import annotations

import numpy as np
import pytest

from advanced_weighting_systems.utils.entropy import (
    EntropyTable,
    entropy_governance_weight,
    shannon_entropy,
)

# ---------------------------------------------------------------------------
# shannon_entropy
# ---------------------------------------------------------------------------


class TestShannonEntropy:
    def test_uniform_distribution_max_entropy(self) -> None:
        p = np.ones(4) / 4.0
        h = shannon_entropy(p)
        expected = np.log(4.0)
        np.testing.assert_allclose(h, expected, atol=1e-10)

    def test_deterministic_distribution_zero_entropy(self) -> None:
        p = np.array([1.0, 0.0, 0.0, 0.0])
        h = shannon_entropy(p)
        np.testing.assert_allclose(h, 0.0, atol=1e-6)

    def test_non_negative_output(self) -> None:
        rng = np.random.default_rng(0)
        p = rng.dirichlet(np.ones(5))
        assert shannon_entropy(p) >= 0.0

    def test_unnormalised_probs_accepted(self) -> None:
        p = np.array([2.0, 2.0])
        h = shannon_entropy(p)
        assert h >= 0.0

    def test_zero_sum_raises(self) -> None:
        with pytest.raises(ValueError, match="must sum to a positive value"):
            shannon_entropy(np.zeros(3))

    def test_base_2_entropy_bits(self) -> None:
        p = np.array([0.5, 0.5])
        h = shannon_entropy(p, base=2.0)
        np.testing.assert_allclose(h, 1.0, atol=1e-10)

    def test_single_element_zero_entropy(self) -> None:
        p = np.array([1.0])
        h = shannon_entropy(p)
        np.testing.assert_allclose(h, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# entropy_governance_weight
# ---------------------------------------------------------------------------


class TestEntropyGovernanceWeight:
    def test_weights_sum_to_one(self) -> None:
        h = np.array([0.5, 0.3, 0.8])
        w = entropy_governance_weight(h)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-10)

    def test_lower_entropy_gets_higher_weight(self) -> None:
        h = np.array([0.9, 0.1])
        w = entropy_governance_weight(h)
        assert w[1] > w[0]

    def test_equal_entropies_uniform_weights(self) -> None:
        h = np.ones(4) * 0.5
        w = entropy_governance_weight(h)
        np.testing.assert_allclose(w, 0.25, atol=1e-10)

    def test_high_lambda_sharpens_distribution(self) -> None:
        h = np.array([0.3, 0.7])
        w_low = entropy_governance_weight(h, lam=0.1)
        w_high = entropy_governance_weight(h, lam=10.0)
        assert w_high[0] > w_low[0]

    def test_output_shape(self) -> None:
        h = np.linspace(0.1, 0.9, 6)
        w = entropy_governance_weight(h)
        assert w.shape == (6,)


# ---------------------------------------------------------------------------
# EntropyTable
# ---------------------------------------------------------------------------


class TestEntropyTable:
    def test_upsert_and_get(self) -> None:
        et = EntropyTable()
        et.upsert("m0", 0.5)
        entry = et.get("m0")
        assert entry.model_id == "m0"
        assert entry.entropy == pytest.approx(0.5)

    def test_upsert_capped_at_h_max(self) -> None:
        et = EntropyTable(h_max=0.8)
        et.upsert("m0", 1.0)
        assert et.get("m0").entropy == pytest.approx(0.8)

    def test_upsert_negative_entropy_raises(self) -> None:
        et = EntropyTable()
        with pytest.raises(ValueError, match="entropy must be >= 0"):
            et.upsert("m0", -0.1)

    def test_get_missing_key_raises(self) -> None:
        et = EntropyTable()
        with pytest.raises(KeyError):
            et.get("nonexistent")

    def test_remove_existing(self) -> None:
        et = EntropyTable()
        et.upsert("m0", 0.5)
        et.remove("m0")
        with pytest.raises(KeyError):
            et.get("m0")

    def test_remove_nonexistent_no_op(self) -> None:
        et = EntropyTable()
        et.remove("nonexistent")  # should not raise

    def test_export_entropies_order(self) -> None:
        et = EntropyTable()
        et.upsert("a", 0.2)
        et.upsert("b", 0.7)
        et.upsert("c", 0.4)
        arr = et.export_entropies(["c", "a", "b"])
        np.testing.assert_allclose(arr, [0.4, 0.2, 0.7], atol=1e-10)

    def test_export_missing_key_raises(self) -> None:
        et = EntropyTable()
        et.upsert("a", 0.5)
        with pytest.raises(KeyError):
            et.export_entropies(["a", "missing"])

    def test_model_ids_property(self) -> None:
        et = EntropyTable()
        et.upsert("x", 0.1)
        et.upsert("y", 0.2)
        assert "x" in et.model_ids
        assert "y" in et.model_ids

    def test_len(self) -> None:
        et = EntropyTable()
        assert len(et) == 0
        et.upsert("a", 0.5)
        assert len(et) == 1

    def test_metadata_stored(self) -> None:
        et = EntropyTable()
        et.upsert("m0", 0.5, layer="conv1", step=10)
        entry = et.get("m0")
        assert entry.metadata["layer"] == "conv1"
        assert entry.metadata["step"] == 10

    def test_repr(self) -> None:
        et = EntropyTable(h_max=2.0)
        r = repr(et)
        assert "EntropyTable" in r
        assert "h_max=2.0" in r
