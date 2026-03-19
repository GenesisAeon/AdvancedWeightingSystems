"""Tests for the Typer CLI (aws couple / aws weight)."""

from __future__ import annotations

from typer.testing import CliRunner

from advanced_weighting_systems.cli import app

runner = CliRunner()


class TestCoupleCommand:
    def test_couple_default(self) -> None:
        result = runner.invoke(app, ["couple"])
        assert result.exit_code == 0
        assert "Resonance energy" in result.output

    def test_couple_all_adapters(self) -> None:
        result = runner.invoke(app, ["couple", "--models", "trans,cnn,rnn,graph,spike"])
        assert result.exit_code == 0

    def test_couple_with_entropy(self) -> None:
        result = runner.invoke(app, ["couple", "--entropy", "0.3"])
        assert result.exit_code == 0

    def test_couple_with_visualize(self) -> None:
        result = runner.invoke(app, ["couple", "--visualize"])
        assert result.exit_code == 0
        assert "MandalaMap" in result.output

    def test_couple_no_aeon_layer(self) -> None:
        result = runner.invoke(app, ["couple", "--no-aeon-layer"])
        assert result.exit_code == 0

    def test_couple_unknown_model_raises(self) -> None:
        result = runner.invoke(app, ["couple", "--models", "unknown"])
        assert result.exit_code != 0

    def test_couple_single_model(self) -> None:
        result = runner.invoke(app, ["couple", "--models", "spike"])
        assert result.exit_code == 0


class TestWeightCommand:
    def test_weight_default(self) -> None:
        result = runner.invoke(app, ["weight"])
        assert result.exit_code == 0
        assert "Weighting Result" in result.output

    def test_weight_custom_ids(self) -> None:
        result = runner.invoke(app, ["weight", "--model-ids", "alpha,beta,gamma"])
        assert result.exit_code == 0

    def test_weight_volume_regime(self) -> None:
        result = runner.invoke(app, ["weight", "--regime", "volume"])
        assert result.exit_code == 0

    def test_weight_invalid_regime(self) -> None:
        result = runner.invoke(app, ["weight", "--regime", "invalid"])
        assert result.exit_code != 0

    def test_weight_with_visualize(self) -> None:
        result = runner.invoke(app, ["weight", "--visualize"])
        assert result.exit_code == 0
        assert "CREP Score" in result.output

    def test_weight_custom_lam(self) -> None:
        result = runner.invoke(app, ["weight", "--lam", "3.0"])
        assert result.exit_code == 0

    def test_weight_single_model(self) -> None:
        result = runner.invoke(app, ["weight", "--model-ids", "solo"])
        assert result.exit_code == 0


class TestCLIHelp:
    def test_main_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "couple" in result.output
        assert "weight" in result.output

    def test_couple_help(self) -> None:
        result = runner.invoke(app, ["couple", "--help"])
        assert result.exit_code == 0

    def test_weight_help(self) -> None:
        result = runner.invoke(app, ["weight", "--help"])
        assert result.exit_code == 0
