"""
Validation tests for anchor calibration (priority 8).

Tests PCM calibrate_anchor() and MFRM calibrate_anchor() / anchor_raters=
parameter in fit_statistics() and stats_df methods.

PCM checks:
  - calibrate_anchor() sets expected attributes
  - anchor items have difficulties exactly equal to supplied anchor values
  - non-anchor items are shifted by anchor_trans_constant
  - anchor_correlation and anchor_sd_ratio are in plausible ranges
  - anchors_keep + anchors_drop = all supplied anchors
  - thresholds_uncentred_anchor exists and has correct structure

MFRM checks:
  - calibrate_anchor() adjusts severities so anchor raters have mean severity 0
  - fit_statistics(anchor_raters=...) runs without error
  - item_stats_df(anchor_raters=...) runs and produces a table
  - rater_stats_df(anchor_raters=...) runs and produces a table

Run with:
    python tests/validation_anchor_calibration.py
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raschpy.simulation.pcm_sim import PCM_Sim
from raschpy.simulation.mfrm_sim import MFRM_Sim_Global
from raschpy.pcm import PCM
from raschpy.mfrm import MFRM

# ── Parameters ────────────────────────────────────────────────────────────
SEED             = 42
NO_ITEMS         = 10
NO_PERSONS       = 500
MAX_SCORE_VECTOR = [3] * NO_ITEMS
NO_RATERS        = 5
MAX_SCORE        = 3

# ── Test infrastructure ───────────────────────────────────────────────────
passed = 0
failed = 0
errors = []

def check(condition, label):
    global passed, failed
    if condition:
        passed += 1
        print(f'  ✓ {label}')
    else:
        failed += 1
        errors.append(label)
        print(f'  ✗ FAIL: {label}')

def section(title):
    print(f'\n{"─" * 60}')
    print(f'  {title}')
    print(f'{"─" * 60}')

# ── Simulate and calibrate PCM ────────────────────────────────────────────
section('PCM — setup: calibrate then anchor')

np.random.seed(SEED)
sim = PCM_Sim(no_of_items=NO_ITEMS, no_of_persons=NO_PERSONS,
              max_score_vector=MAX_SCORE_VECTOR)
data = sim.scores

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    m = PCM(data, max_score_vector=MAX_SCORE_VECTOR)
    m.calibrate()

check(hasattr(m, 'central_diffs'), 'PCM: calibrate() sets central_diffs')
check(len(m.central_diffs) == NO_ITEMS, f'PCM: central_diffs has {NO_ITEMS} items')

# Use a subset of items as anchors with known values
anchor_items = list(m.items[:6])  # use first 6 items as anchors
# Anchor values = calibrated values + a known offset, so we know the expected shift
OFFSET = 0.5
anchor_values = m.central_diffs[anchor_items] + OFFSET
anchors = pd.Series(anchor_values.values, index=anchor_items)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    m.calibrate_anchor(anchors, min_anchors=4)

# ── PCM anchor attributes ─────────────────────────────────────────────────
section('PCM — calibrate_anchor() attributes')

for attr in ['anchor_trans_constant', 'anchor_correlation', 'anchor_sd_ratio',
             'anchors_keep', 'anchors_drop', 'anchor_robust_z',
             'central_diffs_anchor', 'thresholds_uncentred_anchor']:
    check(hasattr(m, attr), f'PCM: attribute {attr!r} set')

# anchors_keep + anchors_drop = all supplied anchors
all_supplied = set(anchor_items)
all_accounted = set(m.anchors_keep) | set(m.anchors_drop)
check(all_supplied == all_accounted,
      'PCM: anchors_keep ∪ anchors_drop = all supplied anchors')

# anchor_correlation in [0, 1]
check(0 <= m.anchor_correlation <= 1,
      f'PCM: anchor_correlation in [0, 1] (got {m.anchor_correlation:.3f})')

# anchor_sd_ratio > 0
check(m.anchor_sd_ratio > 0,
      f'PCM: anchor_sd_ratio > 0 (got {m.anchor_sd_ratio:.3f})')

# anchor_robust_z is a Series indexed by anchor items
check(isinstance(m.anchor_robust_z, pd.Series),
      'PCM: anchor_robust_z is pandas Series')
check(set(m.anchor_robust_z.index) == set(anchor_items),
      'PCM: anchor_robust_z indexed by anchor item names')

# ── PCM anchor difficulty values ──────────────────────────────────────────
section('PCM — central_diffs_anchor values')

check(len(m.central_diffs_anchor) == NO_ITEMS,
      f'PCM: central_diffs_anchor has {NO_ITEMS} items')

# Kept anchor items should have values from the supplied anchors Series
for item in m.anchors_keep:
    expected = anchors[item]
    actual   = m.central_diffs_anchor[item]
    check(abs(actual - expected) < 1e-6,
          f'PCM: central_diffs_anchor[{item!r}] = supplied anchor value')

# Non-anchor items should equal central_diffs + anchor_trans_constant
non_anchors = [item for item in m.items if item not in anchor_items]
for item in non_anchors[:3]:  # spot-check three
    expected = m.central_diffs[item] + m.anchor_trans_constant
    actual   = m.central_diffs_anchor[item]
    check(abs(actual - expected) < 1e-6,
          f'PCM: central_diffs_anchor[{item!r}] = central_diffs + trans_constant')

# ── PCM thresholds_uncentred_anchor structure ─────────────────────────────
section('PCM — thresholds_uncentred_anchor structure')

check(isinstance(m.thresholds_uncentred_anchor, dict),
      'PCM: thresholds_uncentred_anchor is dict')
check(set(m.thresholds_uncentred_anchor.keys()) == set(m.items),
      'PCM: thresholds_uncentred_anchor has all items as keys')

for item in list(m.items)[:3]:
    thr = m.thresholds_uncentred_anchor[item]
    check(len(thr) == MAX_SCORE_VECTOR[0],
          f'PCM: thresholds_uncentred_anchor[{item!r}] length = max_score')

# ── PCM stats_df after anchor calibration ────────────────────────────────
section('PCM — stats_df methods run after anchor calibration')

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    m.item_stats_df()
    m.threshold_stats_df()
    m.person_stats_df()
    m.test_stats_df()

check(hasattr(m, 'item_stats') and isinstance(m.item_stats, pd.DataFrame),
      'PCM: item_stats_df() runs after anchor calibration')
check(m.item_stats.shape[0] == NO_ITEMS,
      f'PCM: item_stats has {NO_ITEMS} rows')
check(hasattr(m, 'threshold_stats_uncentred') and isinstance(m.threshold_stats_uncentred, pd.DataFrame),
      'PCM: threshold_stats_df() runs after anchor calibration')
check(hasattr(m, 'person_stats') and isinstance(m.person_stats, pd.DataFrame),
      'PCM: person_stats_df() runs after anchor calibration')
check(hasattr(m, 'test_stats') and isinstance(m.test_stats, pd.DataFrame),
      'PCM: test_stats_df() runs after anchor calibration')

# ── MFRM anchor calibration ───────────────────────────────────────────────
section('MFRM — setup: calibrate then anchor')

np.random.seed(SEED)
mfrm_sim = MFRM_Sim_Global(no_of_items=6, no_of_persons=NO_PERSONS,
                            no_of_raters=NO_RATERS, max_score=MAX_SCORE)
mfrm_data = mfrm_sim.scores

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    mf = MFRM(mfrm_data)
    mf.calibrate(model='global')

check(hasattr(mf, 'severities_global'), 'MFRM: calibrate sets severities_global')
check(len(mf.severities_global) == NO_RATERS,
      f'MFRM: severities_global has {NO_RATERS} raters')

# Use first 3 raters as anchors
anchor_raters = list(mf.raters[:3])
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    mf.calibrate_anchor('global', anchor_raters)

check(hasattr(mf, 'anchor_raters_global'), 'MFRM: anchor_raters_global set')
check(mf.anchor_raters_global == anchor_raters,
      'MFRM: anchor_raters_global matches supplied list')

# calibrate_anchor stores adjusted severities in anchor_severities_global
# (not severities_global). Mean of anchor raters in anchor_severities_global = 0.
check(hasattr(mf, 'anchor_severities_global'),
      'MFRM: anchor_severities_global set by calibrate_anchor')
anchor_sev_mean = mf.anchor_severities_global.loc[anchor_raters].mean()
check(abs(anchor_sev_mean) < 1e-6,
      f'MFRM: mean of anchor rater anchor_severities ≈ 0 (got {anchor_sev_mean:.2e})')

# ── MFRM fit_statistics with anchor_raters ────────────────────────────────
section('MFRM — fit_statistics(anchor_raters=...) and stats_df methods')

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    mf.fit_statistics(model='global', anchor_raters=anchor_raters)

check(hasattr(mf, 'item_infit_ms_global'),
      'MFRM: fit_statistics with anchor_raters sets item_infit_ms_global')
check(hasattr(mf, 'person_infit_ms_global'),
      'MFRM: fit_statistics with anchor_raters sets person_infit_ms_global')

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    mf.item_stats_df(model='global', anchor_raters=anchor_raters)
    mf.rater_stats_df(model='global', anchor_raters=anchor_raters)
    # person_stats_df with anchor_raters requires anchor abilities to be pre-computed
    mf.person_abils(model='global', anchor=True)
    mf.person_stats_df(model='global', anchor_raters=anchor_raters)
    mf.test_stats_df(model='global')

check(hasattr(mf, 'item_stats_global') and isinstance(mf.item_stats_global, pd.DataFrame),
      'MFRM: item_stats_df(anchor_raters=...) produces DataFrame')
check(mf.item_stats_global.shape[0] == 6,
      'MFRM: item_stats_global has 6 rows')
check(hasattr(mf, 'rater_stats_global') and isinstance(mf.rater_stats_global, pd.DataFrame),
      'MFRM: rater_stats_df(anchor_raters=...) produces DataFrame')
check(hasattr(mf, 'person_stats_global') and isinstance(mf.person_stats_global, pd.DataFrame),
      'MFRM: person_stats_df(anchor_raters=...) produces DataFrame')
check(hasattr(mf, 'test_stats_global') and isinstance(mf.test_stats_global, pd.DataFrame),
      'MFRM: test_stats_df() produces DataFrame')

# Item difficulties should be the same with and without anchor raters
# (anchor_raters only affects severity zero-point, not item difficulties)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    mf2 = MFRM(mfrm_data)
    mf2.calibrate(model='global')
    mf2.item_stats_df(model='global')

check(np.allclose(mf.item_stats_global['Estimate'].values,
                  mf2.item_stats_global['Estimate'].values, atol=1e-3),
      'MFRM: item difficulties unchanged by anchor_raters parameter')

# ── Summary ───────────────────────────────────────────────────────────────
section('Summary')
total = passed + failed
print(f'\n  {passed}/{total} checks passed')
if errors:
    print(f'\n  Failed checks:')
    for e in errors:
        print(f'    ✗ {e}')
    sys.exit(1)
else:
    print('\n  All checks passed. ✓')
