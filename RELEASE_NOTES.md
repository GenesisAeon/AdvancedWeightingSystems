### Overview

`advanced-weighting-systems` v0.1.0 provides the foundational resonance engine for the GenesisAeon stack. It enables dynamic, self-reflective coupling of arbitrary neural network architectures (Transformer, CNN, RNN, GraphNN, Spiking) inside a single **AeonLayer** using symbolic mirroring and entropy-governed weighting.

Built with `diamond-setup --template genesis`. 100 % English. ruff-clean. 138 tests @ 99.59 % coverage.

### Core Features

- **AeonLayer** – central resonance container

$$L_{\text{Aeon}} = \sum_i w_i \cdot M_i \cdot \sigma(\beta(R_i - \Theta))$$

- **WeightingEngine** – dual entropy regimes (S∝A vs S∝V) + UTAC-Logistic + CREP coherence
- **SymbolicMirror** – Sigillin-based reflection with 5 production-ready adapters (`TransformerAdapter`, `CNNAdapter`, `RNNAdapter`, `GraphNNAdapter`, `SpikeAdapter`)
- **CLI** – `aws couple --models trans,cnn,rnn --aeon-layer --entropy 0.37 --visualize`
- **stack extra** – seamless integration with `mirror-machine`, `entropy-governance`, `sigillin`, `utac-core`, `mandala-visualizer`, `cosmic-web`

### Installation

```bash
pip install advanced-weighting-systems[stack]
```

### Documentation

- Full reference & KaTeX formulas: https://advanced-weighting-systems.readthedocs.io
- CLI usage & examples in `docs/cli.md`

### Next milestones (already in unified-mandala backlog)

- v0.2.0: real-time phase-transition detection
- v0.3.0: native integration into AeonAI self-reflection loop

**DOI will be added post-Zenodo release.**

Part of the GenesisAeon project – from modular entropy governance to full cosmic simulation.
