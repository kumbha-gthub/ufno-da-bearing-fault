# ============================================================
# Per-seed McNemar test with DIRECTIONAL Stouffer combination
#
# CORRECTNESS NOTE: an earlier version of this script combined
# two-sided p-values via non-directional Stouffer, which
# INFLATES significance when seeds disagree about which model
# wins. This version uses directional (one-sided) Stouffer
# in the "U-FNO better than CNN" direction, which is the
# scientifically correct test for our hypothesis.
#
# Usage:
#   set OUTPUTS_DIR env var and run.
# ============================================================

import os, json, math
from pathlib import Path
import numpy as np
import pandas as pd

OUT_BASE = Path(os.environ.get('OUTPUTS_DIR',
                               '/content/drive/MyDrive/PU_Outputs'))
AGG = OUT_BASE / 'AGGREGATED'
AGG.mkdir(parents=True, exist_ok=True)

SEEDS_WITH_PREDS = [0, 1]
TARGET_OPS = ['N15_M01_F10', 'N15_M07_F04', 'N09_M07_F10']
FRACTIONS  = [0.1, 0.3, 0.5, 0.7, 0.9]

try:
    from scipy import stats as sps
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def mcnemar(yt, cnn_pred, ufno_pred):
    yt = np.asarray(yt); cp = np.asarray(cnn_pred); up = np.asarray(ufno_pred)
    cnn_right  = (cp == yt)
    ufno_right = (up == yt)
    b = int(((~cnn_right) & ( ufno_right)).sum())
    c = int(( cnn_right  & (~ufno_right)).sum())
    n = len(yt)
    if b + c > 0:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_two = math.erfc(math.sqrt(chi2 / 2))
    else:
        chi2 = 0.0; p_two = 1.0
    return n, b, c, chi2, p_two


def directional_one_sided_p(b, c, p_two):
    if b > c:    return p_two / 2
    elif b < c:  return 1 - p_two / 2
    else:        return 0.5


def norm_ppf(p):
    if HAVE_SCIPY:
        return float(sps.norm.ppf(p))
    p = max(min(p, 1 - 1e-15), 1e-15)
    plow, phigh = 0.02425, 0.97575
    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239]
    bb = [-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572]
    c = [-7.784894002430293e-03, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [7.784695709041462e-03, 0.3224671290700398,
         2.445134137142996, 3.754408661907416]
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= phigh:
        q = p - 0.5; r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
               (((((bb[0]*r+bb[1])*r+bb[2])*r+bb[3])*r+bb[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def norm_sf(z):
    if HAVE_SCIPY: return float(sps.norm.sf(z))
    return 0.5 * math.erfc(z / math.sqrt(2))


def directional_stouffer(per_seed_results):
    zs = [norm_ppf(1 - r['p_one_ufno']) for r in per_seed_results]
    z_comb = sum(zs) / math.sqrt(len(zs))
    return z_comb, norm_sf(z_comb)


rows = []
for f in FRACTIONS:
    per_seed = []
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
            if ck not in preds or uk not in preds: continue
            all_yt.extend(preds[ck]['yt'])
            all_cnn.extend(preds[ck]['yp'])
            all_ufno.extend(preds[uk]['yp'])

        n, b, c, chi2, p_two = mcnemar(all_yt, all_cnn, all_ufno)
        p_one_ufno = directional_one_sided_p(b, c, p_two)
        per_seed.append({
            'seed': seed, 'n': n, 'b': b, 'c': c,
            'chi2': chi2, 'p_two_sided': p_two,
            'direction': '+UF' if b > c else ('-UF' if b < c else '0'),
            'p_one_ufno': p_one_ufno,
        })

    if len(per_seed) == len(SEEDS_WITH_PREDS):
        z_c, p_c_dir = directional_stouffer(per_seed)
    else:
        z_c, p_c_dir = None, None

    print(f'\nFraction f={f}')
    for s in per_seed:
        print(f"  seed={s['seed']}  b={s['b']:>3} c={s['c']:>3}  "
              f"chi2={s['chi2']:>6.2f}  p_2s={s['p_two_sided']:.4g}  "
              f"dir={s['direction']}")
    if z_c is not None:
        print(f"  Directional Stouffer (UFNO > CNN): Z={z_c:.2f}  p={p_c_dir:.4g}")

    row = {'fraction': f}
    for s in per_seed:
        row[f'b_s{s["seed"]}']     = s['b']
        row[f'c_s{s["seed"]}']     = s['c']
        row[f'chi2_s{s["seed"]}']  = round(s['chi2'], 3)
        row[f'p_two_s{s["seed"]}'] = round(s['p_two_sided'], 6)
        row[f'dir_s{s["seed"]}']   = s['direction']
    if z_c is not None:
        row['Z_directional']     = round(z_c, 3)
        row['p_directional']     = round(p_c_dir, 6)
    rows.append(row)

df = pd.DataFrame(rows)
out_csv = AGG / 'h2_mcnemar_per_seed.csv'
df.to_csv(out_csv, index=False)
print(f'\nSaved: {out_csv}')
print('\n', df.to_string(index=False))