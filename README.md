# Fourier Neural Operators for Robust Domain Adaptation with Limited Target Labels

Code accompanying the paper *"Fourier Neural Operators for Robust Domain Adaptation with Limited Target Labels"* (submitted to IEEE MLSP 2025).

We study bearing fault diagnosis under two deployment-time distribution shifts:

1. **Sensor quality degradation** — training data recorded at a high sampling rate, deployed on lower-rate sensors.
2. **Operating-condition shift** — the test-time machine runs at a different speed / torque / radial force, with limited target labels.

Both shifts share a single architectural root cause: a model whose inductive bias is aligned with the global frequency structure of bearing fault signatures (Fourier Neural Operators + UNet) is more robust to both shifts than a conventional CNN under identical training settings.

---

## Repository contents

```
.
├── pu_cross_length.py              # H1a: cross-length control (same weights, different input shape)
├── pu_cross_sampling_rate.py       # H1b: cross-rate on one condition
├── pu_cross_rate_all_conditions.py # H1c + ablation: all 4 conditions, plus PlainFNO
├── pu_hypothesis2_coral.py         # H2: CORAL adaptation, fraction sweep
├── aggregate_seeds.py              # Combine multi-seed runs into mean ± std tables
├── mcnemar_per_seed.py             # Per-seed McNemar tests + Stouffer combination
├── requirements.txt
├── .gitignore
├── outputs/                        # Default output location; not pushed to repo
└── README.md
```

Each training script is **self-contained** (imports, model definitions, data loading, training loop, evaluation in one file).

---

## Dataset

We use the **Paderborn University (PU) bearing dataset** (Lessmeier et al., 2016):

> https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter/data-sets-and-download

Place the unpacked bearing folders (`K001`, `K002`, …, `KA01`, …, `KI01`, …) under a single directory, then point the scripts at it via `PU_DATA_DIR`.

Three balanced fault classes: healthy (K001–K003), outer race (KA01, KA03, KA04), inner race (KI01, KI03, KI04). Compound-fault KB bearings excluded. Reps 1–14 train, 15–17 validation, 18–20 fixed test.

---

## Quick start

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.9 and a CUDA-capable GPU. Training was performed on RTX PRO 6000 and A100 (40/80 GB).

### Run a single experiment

```bash
export PU_DATA_DIR=/path/to/PU_Data
export OUTPUTS_DIR=./outputs

SEED=42 python pu_cross_rate_all_conditions.py     # ~5-15 min
SEED=42 python pu_hypothesis2_coral.py             # ~1 hour
```

### Reproduce paper's multi-seed results

```bash
for SEED in 42 0 1; do
    SEED=$SEED python pu_cross_rate_all_conditions.py
    SEED=$SEED python pu_hypothesis2_coral.py
done
```

### Aggregate and run statistical tests

```bash
python aggregate_seeds.py          # mean ± std tables, collapse counts
python mcnemar_per_seed.py         # per-seed McNemar + Stouffer combined p-values
```

Outputs land in `$OUTPUTS_DIR/AGGREGATED/`:
- `h1_mean_std.csv` — H1 cross-rate accuracy table
- `h1_drop_4x.csv` — H1 4× downsampling drop summary
- `h2_mean_std.csv` — H2 target accuracy (mean ± std)
- `h2_collapse_rates.csv` — chance-level training collapses per (model, fraction)
- `h2_mcnemar_per_seed.csv` — paired McNemar test (statistically valid per-seed version)

---

## Configuration

Each script has a `USER CONFIG` block at the top:

```python
SEED        = int(os.environ.get('SEED',         42))
PU_DATA_DIR =     os.environ.get('PU_DATA_DIR',  '/path/to/PU_Data')
OUTPUTS_DIR =     os.environ.get('OUTPUTS_DIR',  './outputs')
```

Hyperparameters are defined as Python constants in each script. Defaults reproduce the paper's results:

- **H1 U-FNO**: WIDTH=64, DEPTH=2, MODES=12×12, UNET_DEPTH=1 → 1.34 M params
- **H2 U-FNO**: WIDTH=32, DEPTH=4, MODES=16×16, UNET_DEPTH=3 → 1.27 M params
- **CNN (both)**: 3 double-conv blocks, AdaptiveAvgPool2d(4,4), 0.81 M params

CNN and U-FNO share identical optimiser, schedule, augmentation, and batch size — any performance gap reflects the feature extractor under a controlled shared training procedure.

---

## Statistical methodology

We report results across three seeds (42, 0, 1). Significance claims use **per-seed McNemar tests** combined via Stouffer's method (`mcnemar_per_seed.py`).

We deliberately compute McNemar **per seed** rather than pooling predictions across seeds: pooled paired observations are correlated through sample-intrinsic difficulty and would violate McNemar's independence assumption. Seed 42 is omitted from McNemar because per-sample predictions for that seed were not retained by the original H2 training run; per-seed tests on seeds 0 and 1 (each n=482 paired samples) are combined into a per-fraction p-value.

---

## Reproducibility notes

- All scripts set Python, NumPy, and PyTorch random seeds and enable `torch.backends.cudnn.deterministic = True`.
- Exact bit-for-bit reproducibility across hardware is not guaranteed; expect results within ~1 pp on the same hardware/software stack.
- CNN training exhibits seed sensitivity in some (seed, condition) and (seed, fraction) pairs that U-FNO does not — this is itself one of our reported findings.

---

## License

MIT.

---

## Citation

```
@inproceedings{ufno_da_2025,
  title     = {Fourier Neural Operators for Robust Domain Adaptation
               with Limited Target Labels},
  author    = {Anonymous},
  booktitle = {IEEE Workshop on Machine Learning for Signal Processing (MLSP)},
  year      = {2025},
  note      = {Submitted}
}
```
