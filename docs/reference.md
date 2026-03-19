# API Reference — Mathematical Foundations

## 1. AeonLayer

The central aggregation formula of the GenesisAeon stack:

$$
L_{\text{Aeon}} = \sum_i w_i \cdot M_i \cdot \sigma\!\left(\beta(R_i - \Theta)\right)
$$

| Symbol | Meaning |
|---|---|
| $w_i$ | Dynamic resonance weight for model $i$ |
| $M_i$ | Mirror-Matrix for model $i$ (from SymbolicMirror) |
| $R_i$ | Raw resonance signal for model $i$ |
| $\Theta$ | Global resonance threshold |
| $\beta$ | Sharpness / inverse temperature |
| $\sigma$ | Logistic sigmoid $\sigma(x) = \frac{1}{1+e^{-x}}$ |

### Resonance Energy

$$
E = \| L_{\text{Aeon}} \|_F = \sqrt{\sum_{j,k} (L_{\text{Aeon}})_{jk}^2}
$$

::: advanced_weighting_systems.aeon_layer.AeonLayer

---

## 2. WeightingEngine

### 2.1 Entropy-Governance — Area Regime ($S \propto A$)

$$
w_i^{(A)} = \frac{\exp(-\lambda H_i)}{\sum_j \exp(-\lambda H_j)}
$$

### 2.2 Entropy-Governance — Volume Regime ($S \propto V$)

$$
w_i^{(V)} = \frac{\exp\!\left(-\lambda H_i^{d/2}\right)}{\sum_j \exp\!\left(-\lambda H_j^{d/2}\right)}
$$

### 2.3 UTAC-Logistic Gate

The Unified Topology Activation Criterion:

$$
u_i = \sigma\!\left(\kappa(C_i - \tau)\right)
$$

| Symbol | Meaning |
|---|---|
| $C_i$ | Coherence value for model $i \in [0, 1]$ |
| $\kappa$ | Gate sharpness |
| $\tau$ | Coherence threshold |

### 2.4 CREP — Coherence-Resonance-Entropy Product

$$
\mathrm{CREP}_i = \left(1 - \frac{H_i}{H_{\max}}\right) \cdot \rho_i \cdot u_i
$$

| Symbol | Meaning |
|---|---|
| $H_i$ | Per-model entropy |
| $H_{\max}$ | Maximum entropy reference |
| $\rho_i$ | Resonance correlation |
| $u_i$ | UTAC gate |

::: advanced_weighting_systems.weighting_engine.WeightingEngine

---

## 3. SymbolicMirror

### 3.1 Sigillin Phase Matrix

$$
\Phi_{\text{sigil}} \in \mathbb{R}^{d \times d}, \quad
[\Phi_{\text{sigil}}]_{jk} \propto e^{-\frac{1}{2}|j-k|}
$$

(symmetric, unit Frobenius norm, MandalaMap-Topology encoding)

### 3.2 Mirror-Matrix Construction

$$
M_i = P_i \, \Phi_{\text{sigil}} \, P_i^\top
$$

| Symbol | Meaning |
|---|---|
| $P_i$ | Adapter-specific projection matrix $\in \mathbb{R}^{d \times d}$ |
| $\Phi_{\text{sigil}}$ | Shared Sigillin phase matrix |

::: advanced_weighting_systems.symbolic_mirror.SymbolicMirror

---

## 4. Entropy Utilities

### 4.1 Shannon Entropy

$$
H(p) = -\sum_i p_i \ln p_i
$$

### 4.2 Governance Weights

$$
w_i^{\mathrm{gov}} = \frac{\exp(-\lambda H_i)}{\sum_j \exp(-\lambda H_j)}
$$

::: advanced_weighting_systems.utils.entropy.EntropyTable

::: advanced_weighting_systems.utils.entropy.shannon_entropy

::: advanced_weighting_systems.utils.entropy.entropy_governance_weight

---

## 5. ResonanceCoupling

Full pipeline: SymbolicMirror → WeightingEngine → AeonLayer.

::: advanced_weighting_systems.models.coupling.ResonanceCoupling
