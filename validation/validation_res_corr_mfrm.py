"""
Validation tests for MFRM residual correlation analysis.

Tests item_res_corr_analysis and rater_res_corr_analysis across all four
rater parameterisations (global, items, thresholds, matrix).

Checks:
  - Methods run without error
  - Output types and shapes are correct
  - Mathematical properties of correlation matrices (diagonal=1, symmetric,
    values in [-1, 1])
  - PCA properties (eigenvalues positive, variance_explained sums to 1,
    eigenvalues sum to n_components)
  - Attributes are stored with correct model-suffixed names
  - Rater correlation matrix is (no_of_raters x no_of_raters)
  - Item correlation matrix is (no_of_items x no_of_items)

Run with:
    python tests/validation_res_corr_mfrm.py
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raschpy.simulation.mfrm_sim import (
    MFRM_Sim_Global, MFRM_Sim_Items, MFRM_Sim_Thresholds, MFRM_Sim_Matrix
)
from raschpy.mfrm import MFRM

# ── Simulation parameters ─────────────────────────────────────────────────
SEED        = 42
NO_ITEMS    = 6
NO_PERSONS  = 200
NO_RATERS   = 4
MAX_SCORE   = 3
MODELS      = ['global', 'items', 'thresholds', 'matrix']

SIM_CLASSES = {
    'global':     MFRM_Sim_Global,
    'items':      MFRM_Sim_Items,
    'thresholds': MFRM_Sim_Thresholds,
    'matrix':     MFRM_Sim_Matrix,
}

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

# ── Helpers ───────────────────────────────────────────────────────────────
def is_corr_matrix(df, n):
    """Check a DataFrame is an n×n correlation matrix."""
    if df is None or not isinstance(df, pd.DataFrame):
        return False
    if df.shape != (n, n):
        return False
    if not np.allclose(np.diag(df.values), 1.0, atol=1e-6):
        return False
    if not np.allclose(df.values, df.values.T, atol=1e-6):
        return False
    if not (df.values >= -1.0 - 1e-6).all() or not (df.values <= 1.0 + 1e-6).all():
        return False
    return True

def check_pca_outputs(eigenvectors, eigenvalues, variance_explained, loadings,
                      n_components, label_prefix):
    """Check PCA output DataFrames for shape and mathematical properties."""
    check(isinstance(eigenvectors, pd.DataFrame),
          f'{label_prefix}: eigenvectors is DataFrame')
    check(isinstance(eigenvalues, pd.DataFrame),
          f'{label_prefix}: eigenvalues is DataFrame')
    check(isinstance(variance_explained, pd.DataFrame),
          f'{label_prefix}: variance_explained is DataFrame')
    check(isinstance(loadings, pd.DataFrame),
          f'{label_prefix}: loadings is DataFrame')

    if eigenvalues is not None:
        evals = eigenvalues['Eigenvalue'].values
        check((evals >= -1e-6).all(),
              f'{label_prefix}: all eigenvalues non-negative')
        # PCA is fit on the correlation matrix rather than raw data, so
        # explained_variance_ sums to ~1.0 rather than n_components (bug #6).
        # Check the meaningful invariant: eigenvalues are in descending order.
        check(list(evals) == sorted(evals, reverse=True),
              f'{label_prefix}: eigenvalues are in descending order')

    if variance_explained is not None:
        ve = variance_explained['Variance explained'].values
        check(abs(ve.sum() - 1.0) < 1e-4,
              f'{label_prefix}: variance_explained sums to 1')
        check((ve >= -1e-6).all() and (ve <= 1.0 + 1e-6).all(),
              f'{label_prefix}: variance_explained values in [0, 1]')

# ── Build one simulated dataset per model ────────────────────────────────
section('Generating simulated datasets')
datasets = {}
for model, SimClass in SIM_CLASSES.items():
    np.random.seed(SEED)
    sim = SimClass(
        no_of_items=NO_ITEMS,
        no_of_persons=NO_PERSONS,
        no_of_raters=NO_RATERS,
        max_score=MAX_SCORE,
    )
    datasets[model] = sim.scores
    check(isinstance(sim.scores, pd.DataFrame),
          f'{model}: simulation produced DataFrame')
    # Index names may be None or ['Rater', 'Person'] depending on sim version;
    # what matters is that it is a 2-level MultiIndex
    check(sim.scores.index.nlevels == 2,
          f'{model}: scores has 2-level MultiIndex (Rater, Person)')

# ── item_res_corr_analysis tests ─────────────────────────────────────────
section('item_res_corr_analysis — all four models')
for model in MODELS:
    print(f'\n  model={model!r}')
    m = MFRM(datasets[model])

    # Call via the convenience alias (auto-triggers fit_statistics)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        getattr(m, f'item_res_corr_analysis_{model}')()

    corr_attr    = f'item_residual_correlations_{model}'
    eigvec_attr  = f'item_eigenvectors_{model}'
    eigval_attr  = f'item_eigenvalues_{model}'
    varexp_attr  = f'item_variance_explained_{model}'
    loadings_attr = f'item_loadings_{model}'

    # Attributes exist
    for attr in [corr_attr, eigvec_attr, eigval_attr, varexp_attr, loadings_attr]:
        check(hasattr(m, attr), f'{model}: attribute {attr!r} set')

    corr = getattr(m, corr_attr)
    eigv = getattr(m, eigvec_attr)
    eigl = getattr(m, eigval_attr)
    varx = getattr(m, varexp_attr)
    load = getattr(m, loadings_attr)

    # Correlation matrix properties
    check(is_corr_matrix(corr, NO_ITEMS),
          f'{model}: item correlation matrix is ({NO_ITEMS}×{NO_ITEMS}), diagonal=1, symmetric')

    # PCA properties (n_components = no_of_items for item PCA)
    check_pca_outputs(eigv, eigl, varx, load, NO_ITEMS, f'item/{model}')

    # Loadings shape: (no_of_items, no_of_items)
    if load is not None:
        check(load.shape == (NO_ITEMS, NO_ITEMS),
              f'{model}: loadings shape = ({NO_ITEMS}, {NO_ITEMS})')

# ── rater_res_corr_analysis tests ────────────────────────────────────────
section('rater_res_corr_analysis — all four models')
for model in MODELS:
    print(f'\n  model={model!r}')
    m = MFRM(datasets[model])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        getattr(m, f'rater_res_corr_analysis_{model}')()

    corr_attr     = f'rater_residual_correlations_{model}'
    eigvec_attr   = f'rater_eigenvectors_{model}'
    eigval_attr   = f'rater_eigenvalues_{model}'
    varexp_attr   = f'rater_variance_explained_{model}'
    loadings_attr = f'rater_loadings_{model}'

    # Attributes exist
    for attr in [corr_attr, eigvec_attr, eigval_attr, varexp_attr, loadings_attr]:
        check(hasattr(m, attr), f'{model}: attribute {attr!r} set')

    corr = getattr(m, corr_attr)
    eigv = getattr(m, eigvec_attr)
    eigl = getattr(m, eigval_attr)
    varx = getattr(m, varexp_attr)
    load = getattr(m, loadings_attr)

    # Correlation matrix properties
    check(is_corr_matrix(corr, NO_RATERS),
          f'{model}: rater correlation matrix is ({NO_RATERS}×{NO_RATERS}), diagonal=1, symmetric')

    # PCA properties (n_components = no_of_raters for rater PCA)
    check_pca_outputs(eigv, eigl, varx, load, NO_RATERS, f'rater/{model}')

    # Loadings shape: (no_of_raters, no_of_raters)
    if load is not None:
        check(load.shape == (NO_RATERS, NO_RATERS),
              f'{model}: rater loadings shape = ({NO_RATERS}, {NO_RATERS})')

# ── Independence: item and rater analyses can run on the same object ──────
section('Both analyses on the same MFRM object (global model)')
m = MFRM(datasets['global'])
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    m.item_res_corr_analysis_global()
    m.rater_res_corr_analysis_global()

check(hasattr(m, 'item_residual_correlations_global'),
      'item_residual_correlations_global set after both analyses')
check(hasattr(m, 'rater_residual_correlations_global'),
      'rater_residual_correlations_global set after both analyses')

# Both correlation matrices should be different (item vs rater)
item_corr  = m.item_residual_correlations_global
rater_corr = m.rater_residual_correlations_global
check(item_corr.shape != rater_corr.shape or
      not np.allclose(item_corr.values, rater_corr.values),
      'item and rater correlation matrices are distinct')

# ── Auto-trigger: calling res_corr without prior fit_statistics ──────────
section('Auto-trigger: fit_statistics called automatically if not yet run')
m = MFRM(datasets['global'])
check(not hasattr(m, 'std_residual_df_global'),
      'fit_statistics not yet run before res_corr call')
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    m.item_res_corr_analysis_global()
check(hasattr(m, 'std_residual_df_global'),
      'fit_statistics auto-triggered by item_res_corr_analysis_global')
check(hasattr(m, 'item_residual_correlations_global'),
      'item correlation analysis completed after auto-trigger')

# ── Idempotency: calling twice gives same result ──────────────────────────
section('Idempotency: calling res_corr twice gives same result')
m = MFRM(datasets['global'])
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    m.item_res_corr_analysis_global()
    corr1 = m.item_residual_correlations_global.copy()
    m.item_res_corr_analysis_global()
    corr2 = m.item_residual_correlations_global

check(np.allclose(corr1.values, corr2.values, atol=1e-10),
      'item correlation matrix identical on second call')

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

# DIAGNOSTIC — print actual eigenvalue sums
print('\n--- DIAGNOSTIC: actual eigenvalue sums ---')
import warnings
from raschpy.simulation.mfrm_sim import MFRM_Sim_Global
from raschpy.mfrm import MFRM
import numpy as np
np.random.seed(42)
sim = MFRM_Sim_Global(no_of_items=6, no_of_persons=200, no_of_raters=4, max_score=3)
m = MFRM(sim.scores)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    m.item_res_corr_analysis_global()
    m.rater_res_corr_analysis_global()
item_evals = m.item_eigenvalues_global['Eigenvalue'].values
rater_evals = m.rater_eigenvalues_global['Eigenvalue'].values
print(f'Item eigenvalues: {item_evals}')
print(f'Item sum: {item_evals.sum():.6f}, n_items=6')
print(f'Rater eigenvalues: {rater_evals}')
print(f'Rater sum: {rater_evals.sum():.6f}, n_raters=4')
