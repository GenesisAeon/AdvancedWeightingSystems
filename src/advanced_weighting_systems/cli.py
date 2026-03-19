"""Typer CLI for advanced-weighting-systems.

Entry-point: ``aws``

Commands
--------
aws couple  --models trans,cnn,rnn,graph  [--aeon-layer] [--entropy FLOAT] [--visualize]
aws weight  --model-ids id1,id2,...       [--visualize]
"""

from __future__ import annotations

from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from advanced_weighting_systems.aeon_layer import AeonLayerConfig
from advanced_weighting_systems.models.coupling import ResonanceCoupling
from advanced_weighting_systems.symbolic_mirror import (
    CNNAdapter,
    GraphNNAdapter,
    RNNAdapter,
    SpikeAdapter,
    TransformerAdapter,
)
from advanced_weighting_systems.utils.entropy import EntropyTable
from advanced_weighting_systems.weighting_engine import (
    EntropyRegime,
    WeightingConfig,
    WeightingEngine,
)

app = typer.Typer(
    name="aws",
    help="Advanced Weighting Systems — resonance coupling of heterogeneous NN models.",
    add_completion=False,
)
console = Console()

_ADAPTER_MAP = {
    "trans": TransformerAdapter,
    "cnn": CNNAdapter,
    "rnn": RNNAdapter,
    "graph": GraphNNAdapter,
    "spike": SpikeAdapter,
}

_DIM = 8  # embedding dimension for CLI demos


def _parse_models(models_str: str) -> list[str]:
    keys = [m.strip().lower() for m in models_str.split(",") if m.strip()]
    for k in keys:
        if k not in _ADAPTER_MAP:
            raise typer.BadParameter(
                f"Unknown model type {k!r}. Choose from: {', '.join(_ADAPTER_MAP)}"
            )
    return keys


@app.command()
def couple(
    models: Annotated[
        str,
        typer.Option("--models", help="Comma-separated adapter types: trans,cnn,rnn,graph,spike"),
    ] = "trans,cnn",
    aeon_layer: Annotated[
        bool,
        typer.Option("--aeon-layer/--no-aeon-layer", help="Enable AeonLayer aggregation"),
    ] = True,
    entropy: Annotated[
        float,
        typer.Option("--entropy", help="Uniform entropy value H_i for all models (demo)"),
    ] = 0.5,
    visualize: Annotated[
        bool,
        typer.Option("--visualize/--no-visualize", help="Print mandala topology (ASCII)"),
    ] = False,
) -> None:
    """Couple heterogeneous NN models via SymbolicMirror + AeonLayer."""
    model_keys = _parse_models(models)
    n = len(model_keys)

    adapters = [_ADAPTER_MAP[k](dim=_DIM) for k in model_keys]
    coupling = ResonanceCoupling(
        adapters=adapters,
        dim=_DIM,
        aeon_config=AeonLayerConfig(n_models=n, beta=1.5, theta=0.0),
    )

    rng = np.random.default_rng(seed=0)
    acts = [rng.standard_normal(_DIM) for _ in model_keys]
    entropies = np.full(n, entropy)
    coherences = rng.uniform(0.3, 0.9, size=n)
    rho = rng.uniform(0.4, 1.0, size=n)

    output = coupling.step(acts, entropies, coherences, rho)

    table = Table(title="Resonance Coupling Result", style="cyan")
    table.add_column("Model", style="bold")
    table.add_column("Weight w_i", justify="right")
    table.add_column("Gate sigma", justify="right")
    table.add_column("CREP", justify="right")
    table.add_column("R_i", justify="right")

    for i, key in enumerate(model_keys):
        table.add_row(
            key,
            f"{output.weighting_result.weights[i]:.4f}",
            f"{output.aeon_state.gate_values[i]:.4f}",
            f"{output.weighting_result.crep_scores[i]:.4f}",
            f"{output.mirror_result.resonance_signals[i]:.4f}",
        )

    console.print(table)
    console.print(f"[bold green]Resonance energy ‖L_Aeon‖_F = {output.energy:.6f}[/]")

    if visualize:
        _print_mandala(coupling._mirror.mandala_topology_matrix())


@app.command()
def weight(
    model_ids: Annotated[
        str,
        typer.Option("--model-ids", help="Comma-separated model IDs"),
    ] = "m0,m1,m2",
    regime: Annotated[
        str,
        typer.Option("--regime", help="Entropy regime: area or volume"),
    ] = "area",
    lam: Annotated[
        float,
        typer.Option("--lam", help="Entropy penalty strength lambda"),
    ] = 1.0,
    visualize: Annotated[
        bool,
        typer.Option("--visualize/--no-visualize", help="Show CREP bar chart (ASCII)"),
    ] = False,
) -> None:
    """Compute resonance weights from entropy-governed EntropyTable."""
    ids = [m.strip() for m in model_ids.split(",") if m.strip()]
    n = len(ids)

    table_e = EntropyTable(h_max=1.0)
    rng = np.random.default_rng(seed=7)
    for mid in ids:
        table_e.upsert(mid, float(rng.uniform(0.1, 0.9)))

    entropies = table_e.export_entropies(ids)
    coherences = rng.uniform(0.3, 0.9, size=n)
    rho = rng.uniform(0.4, 1.0, size=n)

    try:
        regime_enum = EntropyRegime(regime)
    except ValueError:
        raise typer.BadParameter(f"regime must be 'area' or 'volume', got {regime!r}") from None

    engine = WeightingEngine(
        config=WeightingConfig(regime=regime_enum, lam=lam, h_max=table_e.h_max)
    )
    result = engine.compute(entropies, coherences, rho)

    tbl = Table(title="Weighting Result", style="magenta")
    tbl.add_column("Model ID", style="bold")
    tbl.add_column("H_i", justify="right")
    tbl.add_column("w_i", justify="right")
    tbl.add_column("UTAC u_i", justify="right")
    tbl.add_column("CREP", justify="right")

    for i, mid in enumerate(ids):
        tbl.add_row(
            mid,
            f"{entropies[i]:.4f}",
            f"{result.weights[i]:.4f}",
            f"{result.utac_gates[i]:.4f}",
            f"{result.crep_scores[i]:.4f}",
        )

    console.print(tbl)

    if visualize:
        _print_crep_bars(ids, result.crep_scores)


# ---------------------------------------------------------------------------
# ASCII visualisation helpers
# ---------------------------------------------------------------------------


def _print_mandala(phi: np.ndarray) -> None:  # type: ignore[type-arg]
    """Print an ASCII heatmap of the Sigillin phase matrix Phi_sigil."""
    console.print("\n[bold yellow]MandalaMap Topology (Phi_sigil):[/]")
    d = phi.shape[0]
    chars = " ░▒▓█"
    vmin, vmax = phi.min(), phi.max()
    rng = vmax - vmin + 1e-12
    for row in range(d):
        line = ""
        for col in range(d):
            idx = int((phi[row, col] - vmin) / rng * (len(chars) - 1))
            line += chars[idx] * 2
        console.print(line)


def _print_crep_bars(ids: list[str], crep: np.ndarray) -> None:  # type: ignore[type-arg]
    """Print ASCII bar chart of CREP scores."""
    console.print("\n[bold yellow]CREP Score Distribution:[/]")
    max_val = float(crep.max()) + 1e-12
    bar_width = 40
    for mid, score in zip(ids, crep, strict=True):
        filled = int(score / max_val * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        console.print(f"{mid:>10} │{bar}│ {score:.4f}")


if __name__ == "__main__":
    app()
