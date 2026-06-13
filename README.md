# Fourier Neural Operators for Robust Domain Adaptation with Limited Target Labels

Code accompanying the paper *"Fourier Neural Operators for Robust Domain Adaptation with Limited Target Labels"* (submitted to IEEE MLSP 2025).

We study bearing fault diagnosis under two deployment-time distribution shifts:

1. **Sensor quality degradation** — training data recorded at high sampling rates is deployed on lower-rate sensors.
2. **Operating-condition shift** — the test-time machine runs at a different speed / torque / radial force, with limited target labels.

Our central claim is that both shifts share a single architectural root cause, and that a model whose inductive bias is aligned with the global frequency structure of bearing fault signatures (Fourier Neural Operators + UNet) is more robust to both shifts than a conventional CNN under identical training settings.

---

## Repository contents

```
.
├── pu_cross_length.py              # H1a: cross-length control (1s -> 1s+2s)
├── pu_cross_sampling_rate.py       # H1b: cross-rate on one condition
├── pu_cross_rate_all_conditions.py # H1c + ablation: all 4 conditions, plus PlainFNO
├── pu_hypothesis2_coral.py         # H2: CORAL adaptation, fraction sweep
├── aggregate_seeds.py              # Combine multi-seed runs into final tables
├── requirements.txt
└── README.md
```

Each training script is **self-contained**: imports, model definitions, data loading, training loop, and evaluation are in a single file.

---

## Dataset

We use the **Paderborn University (PU) bearing dataset** (Lessmeier et al., 2016):

> https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter/data-sets-and-download

Place the unpacked bearing folders (`K001`, `K002`, ..., `KA01`, ..., `KI01`, ...) under a single directory, then point the scripts at it via the `PU_DATA_DIR` environment variable.

The PU dataset is recorded at 64 kHz. Three balanced fault classes are used: healthy (K001–K003), outer race fault (KA01, KA03, KA04), and inner race fault (KI01, KI03, KI04). Compound-fault KB bearings are excluded. Of the 20 measurement repetitions per bearing, reps 1–14 are training, 15–17 validation, and 18–20 the fixed test set.

---

## Quick start

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.9 and a CUDA-capable GPU (training was performed on RTX PRO 6000 and A100 40/80 GB).

### H1 — cross-rate / sensor quality robustness

```bash
export PU_DATA_DIR=/path/to/PU_Data
export OUTPUTS_DIR=./outputs

SEED=42 python pu_cross_rate_all_conditions.py
SEED=0  python pu_cross_rate_all_conditions.py
SEED=1  python pu_cross_rate_all_conditions.py
```

~5–15 minutes per seed. Outputs go to `$OUTPUTS_DIR/cross_rate_all_conditions/seed_{SEED}/`.

### H2 — CORAL adaptation, fraction sweep

```bash
SEED=42 python pu_hypothesis2_coral.py
SEED=0  python pu_hypothesis2_coral.py
SEED=1  python pu_hypothesis2_coral.py
```

~1 hour per seed. Outputs go to `$OUTPUTS_DIR/hypothesis2_coral_cnn_vs_ufno/seed_{SEED}/`.

### Aggregate across seeds

```bash
python aggregate_seeds.py
```

Combines `seed_{42,0,1}/` results into mean ± std tables, collapse-rate counts, and paired McNemar tests using the per-sample predictions saved in `predictions.json`.

---

## Reproducing the paper

The paper reports mean ± std over three seeds: **42, 0, 1**.

| Experiment | Script | Seeds | Total wall time |
|---|---|---|---|
| H1 cross-rate (4 conditions) | `pu_cross_rate_all_conditions.py` | {42, 0, 1} | ~30 min |
| H2 CORAL adaptation (5 fractions) | `pu_hypothesis2_coral.py` | {42, 0, 1} | ~3 hours |
| Cross-length control | `pu_cross_length.py` | {42} | ~3 min |

After all runs, `aggregate_seeds.py` produces the tables that appear in the paper.

---

## Configuration

Each script has a clearly marked `USER CONFIG` block near the top:

```python
SEED        = int(os.environ.get('SEED',         42))
PU_DATA_DIR =     os.environ.get('PU_DATA_DIR',  '/path/to/PU_Data')
OUTPUTS_DIR =     os.environ.get('OUTPUTS_DIR',  './outputs')
```

Hyperparameters (model width / depth / modes, learning rate, batch size, etc.) are defined as Python constants in each script. Defaults reproduce the paper's results:

- **H1 U-FNO**: WIDTH=64, DEPTH=2, MODES=12×12, UNET_DEPTH=1 → 1.34 M params
- **H2 U-FNO**: WIDTH=32, DEPTH=4, MODES=16×16, UNET_DEPTH=3 → 1.27 M params
- **CNN (both)**: 3 double-conv blocks, AdaptiveAvgPool2d(4,4), 0.81 M params

CNN and U-FNO share identical optimizer, schedule, augmentation, batch size, and pooled-vector dimensionality consumed by the CORAL head — any performance gap reflects the feature extractor alone.

---

## Notes on reproducibility

- All scripts set Python, NumPy, and PyTorch random seeds and enable `torch.backends.cudnn.deterministic = True`.
- Despite this, exact bit-for-bit reproducibility across hardware is not guaranteed.
- The paper reports **3 random seeds**; CNN training exhibits seed sensitivity in some (seed, condition/fraction) pairs that does not occur with U-FNO.

---

## License

Code released under the MIT license. The PU dataset has its own license; consult the dataset provider.

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
