# Changelog

## [0.1.0] — 2026-03-19

### Added

- `AeonLayer`: central resonance aggregation $L_{\text{Aeon}} = \sum_i w_i M_i \sigma(\beta(R_i - \Theta))$
- `WeightingEngine`: entropy-governance ($S \propto A$ and $S \propto V$), UTAC-Logistic gates, CREP scores
- `SymbolicMirror`: Sigillin phase matrix + MandalaMap-Topology, Mirror-Matrix construction $M_i = P_i \Phi_{\text{sigil}} P_i^\top$
- Five NN adapters: `TransformerAdapter`, `CNNAdapter`, `RNNAdapter`, `GraphNNAdapter`, `SpikeAdapter`
- `ResonanceCoupling`: end-to-end pipeline wiring all components
- `EntropyTable`: in-memory entropy registry compatible with entropy-governance semantics
- CLI (`aws couple`, `aws weight`) with ASCII visualisation
- 99 %+ test coverage with contract tests for mirror-machine, entropy-governance, utac-core
