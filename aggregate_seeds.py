# ============================================================
# Aggregate results across seeds for H1 and H2
# Run AFTER all training jobs (seed=42, 0, 1) have finished.
#
# Usage:
#   python aggregate_seeds.py
#
# Expects outputs at:
#   ./outputs/hypothesis2_coral_cnn_vs_ufno/seed_{42,0,1}/sweep_results.csv
#   ./outputs/cross_rate_all_conditions/seed_{42,0,1}/all_conditions_results.csv
#
# Produces:
#   ./outputs/AGGREGATED/h2_mean_std.csv           (Table 3 with mean ± std)
#   ./outputs/AGGREGATED/h2_collapse_rates.csv     (k/n collapses per cell)
#   ./outputs/AGGREGATED/h1_mean_std.csv           (Table 2 with mean ± std)
#   ./outputs/AGGREGATED/h2_mcnemar.csv            (paired test at each fraction)
# ============================================================

import os, json
from pathlib import Path
import numpy as np
import pandas as pd

OUT_BASE = Path(os.environ.get('OUTPUTS_DIR', './outputs'))
AGG = OUT_BASE / 'AGGREGATED'
AGG.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 0, 1]

# -----------------------------------------------------------
# H2: HYPOTHESIS 2
# -----------------------------------------------------------
print('=' * 70)
print('AGGREGATING H2 (Hypothesis 2)')
print('=' * 70)

h2_rows = []
for seed in SEEDS:
    p = OUT_BASE / 'hypothesis2_coral_cnn_vs_ufno' / f'seed_{seed}' / 'sweep_results.csv'
    if not p.exists():
        print(f'  MISSING: {p}')
        continue
    df = pd.read_csv(p)
    if 'seed' not in df.columns:
        df['seed'] = seed
    h2_rows.append(df)
    print(f'  loaded {p} ({len(df)} rows)')

if h2_rows:
    h2 = pd.concat(h2_rows, ignore_index=True)
    n_seeds = h2['seed'].nunique()
    print(f'\n  Aggregating over {n_seeds} seeds: {sorted(h2["seed"].unique())}')

    # Mean ± std target-domain accuracy per (model, fraction)
    tgt = h2[~h2['is_source']].copy()
    # First average over the 3 target OPs PER SEED (since each seed has 3 targets)
    per_seed = (tgt.groupby(['seed', 'model', 'fraction'])['overall_acc']
                   .mean().reset_index())
    # Then mean ± std across seeds
    agg = (per_seed.groupby(['model', 'fraction'])['overall_acc']
                   .agg(['mean', 'std', 'count']).round(4).reset_index())
    agg.to_csv(AGG / 'h2_mean_std.csv', index=False)
    print('\n  H2 mean ± std (target avg over 3 OPs):')
    print(agg.to_string(index=False))

    # Collapse rate: count seeds where source acc < 0.5 at each (model, fraction)
    src = h2[h2['is_source']].copy()
    collapse = (src.groupby(['model', 'fraction'])
                   .apply(lambda g: pd.Series({
                       'n_seeds':   len(g),
                       'n_collapse': int((g['overall_acc'] < 0.5).sum()),
                       'rate':      f"{int((g['overall_acc'] < 0.5).sum())}/{len(g)}",
                   }))
                   .reset_index())
    collapse.to_csv(AGG / 'h2_collapse_rates.csv', index=False)
    print('\n  H2 collapse rates (source acc < 0.5):')
    print(collapse.to_string(index=False))

    # McNemar test: paired CNN vs U-FNO on the same fixed test set, per seed
    # Combine predictions from all seeds, all target OPs
    from collections import defaultdict
    paired = defaultdict(lambda: {'cnn_correct': 0, 'ufno_correct': 0,
                                  'cnn_wrong': 0, 'ufno_wrong': 0,
                                  'both_correct': 0, 'both_wrong': 0,
                                  'cnn_only': 0, 'ufno_only': 0,
                                  'total': 0})
    for seed in SEEDS:
        pj = OUT_BASE / 'hypothesis2_coral_cnn_vs_ufno' / f'seed_{seed}' / 'predictions.json'
        if not pj.exists(): continue
        data = json.loads(open(pj).read())
        preds = data['predictions']
        # For each fraction, pool across the 3 target OPs and across seeds
        for key, val in preds.items():
            # key format: '{mname}__frac{frac}__{op}'
            parts = key.split('__')
            mname = parts[0]; frac = parts[1].replace('frac',''); op = parts[2]
            if op == 'N15_M07_F10': continue   # source: skip
            paired[(frac, op, seed)][f'{mname}_yt'] = np.array(val['yt'])
            paired[(frac, op, seed)][f'{mname}_yp'] = np.array(val['yp'])

    # Aggregate pooled McNemar across OPs+seeds per fraction
    pooled = defaultdict(lambda: {'b': 0, 'c': 0})  # b: CNN wrong & U-FNO right; c: opposite
    for (frac, op, seed), d in paired.items():
        if 'cnn_yp' not in d or 'ufno_yp' not in d: continue
        yt = d['cnn_yt']  # same test set, same labels
        cnn_correct  = (d['cnn_yp']  == yt)
        ufno_correct = (d['ufno_yp'] == yt)
        pooled[frac]['b'] += int(((~cnn_correct) & ( ufno_correct)).sum())
        pooled[frac]['c'] += int(( cnn_correct  & (~ufno_correct)).sum())
        pooled[frac]['n'] = pooled[frac].get('n', 0) + len(yt)

    mc_rows = []
    for frac, d in sorted(pooled.items(), key=lambda x: float(x[0])):
        b, c = d['b'], d['c']
        # Continuity-corrected McNemar
        if b + c > 0:
            chi2 = (abs(b - c) - 1) ** 2 / (b + c)
            # p-value via chi2 with df=1
            try:
                from scipy import stats
                p = 1 - stats.chi2.cdf(chi2, df=1)
            except ImportError:
                # crude approximation
                p = np.exp(-chi2/2)
        else:
            chi2 = 0.0; p = 1.0
        mc_rows.append({
            'fraction': float(frac),
            'n_total': d.get('n', 0),
            'CNN_wrong_UFNO_right (b)': b,
            'CNN_right_UFNO_wrong (c)': c,
            'chi2': round(chi2, 3),
            'p_value': round(p, 6),
        })
    mc = pd.DataFrame(mc_rows)
    mc.to_csv(AGG / 'h2_mcnemar.csv', index=False)
    print('\n  H2 McNemar (paired CNN vs U-FNO, pooled across 3 targets + seeds):')
    print(mc.to_string(index=False))

# -----------------------------------------------------------
# H1: HYPOTHESIS 1
# -----------------------------------------------------------
print('\n' + '=' * 70)
print('AGGREGATING H1 (Hypothesis 1)')
print('=' * 70)

h1_rows = []
for seed in SEEDS:
    p = OUT_BASE / 'cross_rate_all_conditions' / f'seed_{seed}' / 'all_conditions_results.csv'
    if not p.exists():
        print(f'  MISSING: {p}')
        continue
    df = pd.read_csv(p)
    if 'seed' not in df.columns:
        df['seed'] = seed
    h1_rows.append(df)
    print(f'  loaded {p} ({len(df)} rows)')

if h1_rows:
    h1 = pd.concat(h1_rows, ignore_index=True)
    # Mean ± std accuracy per (model, condition, rate)
    agg = (h1.groupby(['model','condition','rate_hz'])
              ['acc'].agg(['mean','std','count']).round(4).reset_index())
    agg.to_csv(AGG / 'h1_mean_std.csv', index=False)
    print('\n  H1 mean ± std (cross-rate accuracy):')
    print(agg.to_string(index=False))

    # Mean drop at 4× (16 kHz) across all 4 conditions + seeds, per model
    drops = h1[h1['rate_hz']==16000].copy()
    drops_agg = (drops.groupby('model')['drop']
                       .agg(['mean','std','count']).round(4).reset_index())
    print('\n  H1 4x downsampling drop, pooled across conditions + seeds:')
    print(drops_agg.to_string(index=False))
    drops_agg.to_csv(AGG / 'h1_drop_4x.csv', index=False)

print('\n' + '=' * 70)
print(f'All aggregated outputs in {AGG}')
print('=' * 70)
