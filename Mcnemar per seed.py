# ============================================================
# Per-seed McNemar test (statistically valid version)
#
# The original aggregator pooled predictions across seeds, which
# violates McNemar's independence assumption (same test sample
# predicted by both seed-0 and seed-1 models -> correlated).
#
# This script:
#   1. Computes per-seed McNemar (n=482 independent paired
#      samples per seed)
#   2. Combines per-seed p-values via Stouffer's method
#   3. Outputs a clean table for the paper
#
# Usage:
#   set OUTPUTS_DIR env var and run.
# ============================================================

import os, json
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

OUT_BASE = Path(os.environ.get('OUTPUTS_DIR',
                               '/content/drive/MyDrive/PU_Outputs'))
AGG = OUT_BASE / 'AGGREGATED'
AGG.mkdir(parents=True, exist_ok=True)

SEEDS_WITH_PREDS = [0, 1]   # seed 42 did not save predictions
TARGET_OPS = ['N15_M01_F10', 'N15_M07_F04', 'N09_M07_F10']
FRACTIONS  = [0.1, 0.3, 0.5, 0.7, 0.9]

# scipy is optional; provide fallback
try:
    from scipy import stats as sps
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def mcnemar(yt, cnn_pred, ufno_pred):
    """Continuity-corrected McNemar test on a single contingency."""
    yt = np.asarray(yt); cp = np.asarray(cnn_pred); up = np.asarray(ufno_pred)
    cnn_right  = (cp == yt)
    ufno_right = (up == yt)
    b = int(((~cnn_right) & ( ufno_right)).sum())   # CNN wrong, UFNO right
    c = int(( cnn_right  & (~ufno_right)).sum())    # CNN right, UFNO wrong
    n = len(yt)
    if b + c > 0:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        if HAVE_SCIPY:
            p = 1.0 - sps.chi2.cdf(chi2, df=1)
        else:
            p = float(np.exp(-chi2 / 2))   # crude approx
    else:
        chi2 = 0.0; p = 1.0
    return n, b, c, chi2, p


def stouffer(pvals):
    """Combine p-values via Stouffer's Z-score method (equal weights)."""
    if not HAVE_SCIPY:
        # fallback: report only Fisher-style sum-of-log-p
        return None, None
    z = sps.norm.ppf(1 - np.array(pvals) + 1e-300)
    z_combined = z.sum() / np.sqrt(len(z))
    p_combined = 1 - sps.norm.cdf(z_combined)
    return float(z_combined), float(p_combined)


# Per-seed contingency, then combine
rows = []
for f in FRACTIONS:
    per_seed = []
    pooled_b = 0; pooled_c = 0; pooled_n = 0
    for seed in SEEDS_WITH_PREDS:
        pj = OUT_BASE / 'hypothesis2_coral_cnn_vs_ufno' / f'seed_{seed}' / 'predictions.json'
        if not pj.exists():
            print(f'  missing: {pj}'); continue
        data = json.loads(open(pj).read())
        preds = data['predictions']

        all_yt = []; all_cnn = []; all_ufno = []
        for op in TARGET_OPS:
            ck = f'cnn__frac{f}__{op}'
            uk = f'ufno__frac{f}__{op}'
            if ck not in preds or uk not in preds:
                continue
            all_yt.extend(preds[ck]['yt'])
            all_cnn.extend(preds[ck]['yp'])
            all_ufno.extend(preds[uk]['yp'])

        n, b, c, chi2, p = mcnemar(all_yt, all_cnn, all_ufno)
        per_seed.append({'seed': seed, 'n': n, 'b': b, 'c': c,
                         'chi2': chi2, 'p': p})
        pooled_b += b; pooled_c += c; pooled_n += n

    # Combined p-value across seeds (Stouffer)
    if len(per_seed) == len(SEEDS_WITH_PREDS):
        ps = [s['p'] for s in per_seed]
        z_c, p_c = stouffer(ps)
    else:
        z_c, p_c = None, None

    # Print for this fraction
    print(f'\nFraction f={f}')
    for s in per_seed:
        print(f"  seed={s['seed']}  n={s['n']}  b(CNN wrong, UFNO right)={s['b']}  "
              f"c(CNN right, UFNO wrong)={s['c']}  chi2={s['chi2']:.2f}  p={s['p']:.4g}")
    if z_c is not None:
        print(f"  Stouffer combined: Z={z_c:.2f}  p={p_c:.4g}")
    print(f"  (For reference, OLD pooled would be: b={pooled_b}  c={pooled_c}  "
          f"chi2={(abs(pooled_b-pooled_c)-1)**2/max(1,pooled_b+pooled_c):.1f})")

    row = {'fraction': f}
    for s in per_seed:
        row[f'n_s{s["seed"]}']     = s['n']
        row[f'b_s{s["seed"]}']     = s['b']
        row[f'c_s{s["seed"]}']     = s['c']
        row[f'chi2_s{s["seed"]}']  = round(s['chi2'], 3)
        row[f'p_s{s["seed"]}']     = round(s['p'], 6)
    if z_c is not None:
        row['Z_combined']   = round(z_c, 3)
        row['p_combined']   = round(p_c, 6)
    rows.append(row)

df = pd.DataFrame(rows)
out_csv = AGG / 'h2_mcnemar_per_seed.csv'
df.to_csv(out_csv, index=False)
print(f'\nSaved: {out_csv}')
print('\n', df.to_string(index=False))