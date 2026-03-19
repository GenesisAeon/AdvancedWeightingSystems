# CLI Reference

The `aws` command-line interface provides two primary subcommands.

## aws couple

Couple heterogeneous NN models through SymbolicMirror + AeonLayer.

```
aws couple [OPTIONS]

Options:
  --models TEXT          Comma-separated adapter types: trans,cnn,rnn,graph,spike
                         [default: trans,cnn]
  --aeon-layer / --no-aeon-layer
                         Enable AeonLayer aggregation  [default: aeon-layer]
  --entropy FLOAT        Uniform entropy value H_i for all models (demo)
                         [default: 0.5]
  --visualize / --no-visualize
                         Print MandalaMap topology as ASCII  [default: no-visualize]
  --help                 Show this message and exit.
```

### Example

```bash
aws couple --models trans,cnn,rnn,graph --aeon-layer --entropy 0.4 --visualize
```

---

## aws weight

Compute resonance weights from an entropy-governed EntropyTable.

```
aws weight [OPTIONS]

Options:
  --model-ids TEXT       Comma-separated model IDs  [default: m0,m1,m2]
  --regime TEXT          Entropy regime: area or volume  [default: area]
  --lam FLOAT            Entropy penalty strength lambda  [default: 1.0]
  --visualize / --no-visualize
                         Show CREP bar chart (ASCII)  [default: no-visualize]
  --help                 Show this message and exit.
```

### Example

```bash
aws weight --model-ids model_a,model_b,model_c --regime volume --lam 2.0 --visualize
```
