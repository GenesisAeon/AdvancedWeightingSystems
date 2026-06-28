[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19110330.svg)](https://doi.org/10.5281/zenodo.19110330)
[![PyPI version](https://img.shields.io/pypi/v/advanced-weighting-systems)](https://pypi.org/project/advanced-weighting-systems/)
[![Tests](https://img.shields.io/badge/tests-99.59%25-green)](https://github.com/GenesisAeon/AdvancedWeightingSystems/actions)
# advanced-weighting-systems
**Resonance-based symbolic coupling of heterogeneous neural network models in the AeonLayer**

Version 0.1.0 – GenesisAeon Project
**DOI**: 10.5281/zenodo.19110330
**Zenodo Record**: https://zenodo.org/records/19110330

Resonance-based, symbolically mirrored coupling of heterogeneous neural network models (Transformer, CNN, RNN, GraphNN, Spiking) in the AeonLayer for AeonAI. Built on CREP + Sigillin from unified-mandala.

---

## Mathematical Foundation

### AeonLayer Aggregation

The central formula of the GenesisAeon stack:

$$
L_{\text{Aeon}} = \sum_i w_i \cdot M_i \cdot \sigma\!\left(\beta(R_i - \Theta)\right)
$$

| Symbol | Meaning |
|---|---|
| $w_i$ | Dynamic resonance weight for model $i$ (from WeightingEngine) |
| $M_i$ | Mirror-Matrix $M_i = P_i \Phi_{\text{sigil}} P_i^\top$ (from SymbolicMirror) |
| $R_i$ | Raw resonance signal for model $i$ (adapter-specific) |
| $\Theta$ | Global resonance threshold |
| $\beta$ | Sharpness / inverse temperature |
| $\sigma$ | Logistic sigmoid |

### WeightingEngine — Entropy-Governance

**Area regime** ($S \propto A$):

$$
w_i^{(A)} = \frac{\exp(-\lambda H_i)}{\sum_j \exp(-\lambda H_j)}
$$

**Volume regime** ($S \propto V$, dimension $d$):

$$
w_i^{(V)} = \frac{\exp\!\left(-\lambda H_i^{d/2}\right)}{\sum_j \exp\!\left(-\lambda H_j^{d/2}\right)}
$$

### UTAC-Logistic Gate

$$
u_i = \sigma\!\left(\kappa(C_i - \tau)\right)
$$

### CREP — Coherence-Resonance-Entropy Product

$$
\mathrm{CREP}_i = \left(1 - \frac{H_i}{H_{\max}}\right) \cdot \rho_i \cdot u_i
$$

### Sigillin Phase Matrix and Mirror-Matrix

$$
M_i = P_i \, \Phi_{\text{sigil}} \, P_i^\top, \qquad
[\Phi_{\text{sigil}}]_{jk} \propto e^{-\frac{1}{2}|j-k|}
$$

### Resonance Energy

$$
E = \| L_{\text{Aeon}} \|_F
$$

---

## Installation

```bash
pip install advanced-weighting-systems
```

### Full GenesisAeon Stack

```bash
pip install "advanced-weighting-systems[stack]"
```

Installs: `mirror-machine`, `entropy-governance`, `sigillin`, `mandala-visualizer`,
`utac-core`, `cosmic-web`.

### Development

```bash
git clone https://github.com/GenesisAeon/AdvancedWeightingSystems
cd AdvancedWeightingSystems
pip install -e ".[dev]"
```

---

## Quick Start

```python
import numpy as np
from advanced_weighting_systems.aeon_layer import AeonLayerConfig
from advanced_weighting_systems.symbolic_mirror import (
    TransformerAdapter, CNNAdapter, RNNAdapter, GraphNNAdapter
)
from advanced_weighting_systems.weighting_engine import WeightingConfig
from advanced_weighting_systems.models.coupling import ResonanceCoupling

DIM = 16

coupling = ResonanceCoupling(
    adapters=[
        TransformerAdapter(dim=DIM),
        CNNAdapter(dim=DIM),
        RNNAdapter(dim=DIM),
        GraphNNAdapter(dim=DIM),
    ],
    dim=DIM,
    aeon_config=AeonLayerConfig(beta=1.5, theta=0.0, n_models=4),
    weighting_config=WeightingConfig(lam=1.0),
)

rng = np.random.default_rng(0)
activations = [rng.standard_normal(DIM) for _ in range(4)]
entropies   = rng.uniform(0.1, 0.9, size=4)
coherences  = rng.uniform(0.3, 0.8, size=4)
rho         = rng.uniform(0.4, 1.0, size=4)

output = coupling.step(activations, entropies, coherences, rho)

print(f"L_Aeon shape          : {output.aeon_state.layer_output.shape}")
print(f"Resonance energy      : {output.energy:.6f}")
print(f"CREP scores           : {output.weighting_result.crep_scores}")
```

---

## CLI

```bash
# Couple four adapter types with AeonLayer + mandala ASCII visualisation
aws couple --models trans,cnn,rnn,graph --aeon-layer --entropy 0.4 --visualize

# Compute resonance weights for named models with CREP bar chart
aws weight --model-ids model_a,model_b,model_c --regime volume --lam 2.0 --visualize
```

---

## Architecture

```
advanced_weighting_systems/
├── aeon_layer.py        # L_Aeon aggregation (AeonLayer, AeonLayerConfig)
├── weighting_engine.py  # Entropy-governance, UTAC, CREP (WeightingEngine)
├── symbolic_mirror.py   # Sigillin + MandalaMap -> M_i (SymbolicMirror, adapters)
├── cli.py               # Typer CLI (aws couple / aws weight)
├── models/
│   ├── __init__.py      # Re-exports all adapters
│   └── coupling.py      # ResonanceCoupling end-to-end pipeline
└── utils/
    ├── __init__.py
    └── entropy.py       # Shannon entropy, EntropyTable, governance weights
```

### Supported NN Adapters

| Adapter | `--models` key | Resonance Signal |
|---|---|---|
| `TransformerAdapter` | `trans` | mean(tanh(activations)) |
| `CNNAdapter` | `cnn` | max(abs(activations)) |
| `RNNAdapter` | `rnn` | std(activations) |
| `GraphNNAdapter` | `graph` | mean(activations^2) |
| `SpikeAdapter` | `spike` | spike rate above threshold |

---

## Testing

```bash
pytest                          # run all tests with coverage
pytest --cov-report=html        # HTML coverage report
```

Coverage target: **99 %+** (enforced via `--cov-fail-under=99`).

Contract tests verify interface compatibility with:
- `mirror-machine` — Mirror-Matrix shape and finite-value contracts
- `entropy-governance` — monotonicity of weights w.r.t. entropy
- `utac-core` — UTAC gate range and monotonicity w.r.t. coherence

---

## Documentation

```bash
mkdocs serve          # local preview
mkdocs build --strict # production build (zero warnings)
```

Published at: <https://genesisaeon.github.io/AdvancedWeightingSystems>

---

## Citation

```bibtex
@software{genesisaeon_aws_2026,
  author    = {GenesisAeon},
  title     = {Advanced Weighting Systems: Resonance-based Coupling of Heterogeneous NN Models},
  year      = {2026},
  version   = {0.1.0},
  doi       = {10.5281/zenodo.19110330},
  url       = {https://github.com/GenesisAeon/AdvancedWeightingSystems}
}
```

---

## License

Dual-licensed:

- **Code** (`src/`, `tests/`, scripts, configuration) — [GPL-3.0-or-later](LICENSE-CODE)
- **Documentation** (README, `docs/`, RELEASE_GUIDE.md, CONTRIBUTING.md, CHANGELOG.md, templates) — [CC BY 4.0](LICENSE-DOCS)

See [LICENSE](LICENSE) for the full split. — GenesisAeon Project, 2026.
