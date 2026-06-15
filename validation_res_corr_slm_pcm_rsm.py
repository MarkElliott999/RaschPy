"""
Validation tests for residual correlation analysis in SLM, PCM, and RSM.

Tests res_corr_analysis() for each model and checks:
  - Method runs without error (auto-triggers fit_statistics if needed)
  - Output attributes exist and have correct types
  - Correlation matrix is square, symmetric, diagonal=1, values in [-1, 1]
  - PCA eigenvalues are non-negative and in descending order
  - variance_explained sums to 1 and values are in [0, 1]
  - Loadings shape matches (no_of_items × no_of_items)
  - pca_fail attribute is set to False on success
  - SLM-specific: item_residual_corr and person_residual_corr are Series
  - Auto-trigger: fit_statistics called if not yet run
  - Idempotency: calling twice gives same result

Run with:
    python tests/validation_res_corr_slm_pcm_rsm.py
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raschpy.simulation.slm_sim import SLM_Sim
from raschpy.simulation.pcm_sim import PCM_Sim
from raschpy.simulation.rsm_sim import RSM_Sim
from raschpy.slm import SLM
from raschpy.pcm import PCM
from raschpy.rsm import RSM

# ── Simulation parameters ─────────────────────────────────────────────────
SEED        = 42
NO_ITEMS    = 8
NO_PERSONS  = 400
MAX_SCORE   = 4
MAX_SCORE_VECTOR = [3, 3, 4, 4, 3, 3, 4, 4]

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
    if not isinstance(df, pd.DataFrame):
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

def check_pca_outputs(m, n, prefix):
    """Check PCA attributes on a fitted model object."""
    # pca_fail is only set in the except branch — absence means success
    check(not getattr(m, 'pca_fail', False),
          f'{prefix}: pca_fail absent or False (PCA succeeded)')
    check(isinstance(m.eigenvectors, pd.DataFrame),
          f'{prefix}: eigenvectors is DataFrame')
    check(isinstance(m.eigenvalues, pd.DataFrame),
          f'{prefix}: eigenvalues is DataFrame')
    check(isinstance(m.variance_explained, pd.DataFrame),
          f'{prefix}: variance_explained is DataFrame')
    check(isinstance(m.loadings, pd.DataFrame),
          f'{prefix}: loadings is DataFrame')

    if m.eigenvalues is not None:
        evals = m.eigenvalues['Eigenvalue'].values
        check((evals >= -1e-6).all(),
              f'{prefix}: all eigenvalues non-negative')
        check(list(evals) == sorted(evals, reverse=True),
              f'{prefix}: eigenvalues in descending order')

    if m.variance_explained is not None:
        ve = m.variance_explained['Variance explained'].values
        check(abs(ve.sum() - 1.0) < 1e-4,
              f'{prefix}: variance_explained sums to 1')
        check((ve >= -1e-6).all() and (ve <= 1.0 + 1e-6).all(),
              f'{prefix}: variance_explained values in [0, 1]')

    if m.loadings is not None:
        check(m.loadings.shape == (n, n),
              f'{prefix}: loadings shape = ({n}, {n})')

# ── Generate data ─────────────────────────────────────────────────────────
section('Generating simulated datasets')

np.random.seed(SEED)
slm_sim = SLM_Sim(no_of_items=NO_ITEMS, no_of_persons=NO_PERSONS)
check(isinstance(slm_sim.scores, pd.DataFrame), 'SLM sim produced DataFrame')

np.random.seed(SEED)
pcm_sim = PCM_Sim(no_of_items=NO_ITEMS, no_of_persons=NO_PERSONS,
                  max_score_vector=MAX_SCORE_VECTOR)
check(isinstance(pcm_sim.scores, pd.DataFrame), 'PCM sim produced DataFrame')

np.random.seed(SEED)
rsm_sim = RSM_Sim(no_of_items=NO_ITEMS, no_of_persons=NO_PERSONS,
                  max_score=MAX_SCORE)
check(isinstance(rsm_sim.scores, pd.DataFrame), 'RSM sim produced DataFrame')

# ── SLM res_corr_analysis ─────────────────────────────────────────────────
section('SLM — res_corr_analysis')

slm = SLM(slm_sim.scores)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    slm.res_corr_analysis()

# Standard PCA attributes
for attr in ['residual_correlations', 'eigenvectors', 'eigenvalues',
             'variance_explained', 'loadings']:
    check(hasattr(slm, attr), f'SLM: attribute {attr!r} set')

check(is_corr_matrix(slm.residual_correlations, NO_ITEMS),
      f'SLM: residual_correlations is ({NO_ITEMS}×{NO_ITEMS}), diagonal=1, symmetric')

check_pca_outputs(slm, NO_ITEMS, 'SLM')

# SLM-specific: item_residual_corr and person_residual_corr
for attr in ['item_residual_corr', 'person_residual_corr']:
    check(hasattr(slm, attr), f'SLM: attribute {attr!r} set')

check(isinstance(slm.item_residual_corr, pd.Series),
      'SLM: item_residual_corr is Series')
# item_residual_corr = corrwith(diffs, axis=1): one value per non-extreme person
# (how much each person's residual pattern tracks item difficulty order).
# Length equals no_of_persons minus extreme scorers, so check it's in (0, NO_PERSONS].
check(0 < len(slm.item_residual_corr) <= NO_PERSONS,
      f'SLM: item_residual_corr length in (0, {NO_PERSONS}]')
check(isinstance(slm.person_residual_corr, pd.Series),
      'SLM: person_residual_corr is Series')
# person_residual_corr = corrwith(abilities, axis=0): correlation per item column
check(len(slm.person_residual_corr) == NO_ITEMS,
      f'SLM: person_residual_corr length = no_of_items ({NO_ITEMS})')
check((slm.item_residual_corr.abs() <= 1.0 + 1e-6).all(),
      'SLM: item_residual_corr values in [-1, 1]')
check((slm.person_residual_corr.abs() <= 1.0 + 1e-6).all(),
      'SLM: person_residual_corr values in [-1, 1]')

# ── PCM res_corr_analysis ─────────────────────────────────────────────────
section('PCM — res_corr_analysis')

pcm = PCM(pcm_sim.scores, max_score_vector=MAX_SCORE_VECTOR)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    pcm.res_corr_analysis()

for attr in ['residual_correlations', 'eigenvectors', 'eigenvalues',
             'variance_explained', 'loadings']:
    check(hasattr(pcm, attr), f'PCM: attribute {attr!r} set')

check(is_corr_matrix(pcm.residual_correlations, NO_ITEMS),
      f'PCM: residual_correlations is ({NO_ITEMS}×{NO_ITEMS}), diagonal=1, symmetric')

check_pca_outputs(pcm, NO_ITEMS, 'PCM')

# ── RSM res_corr_analysis ─────────────────────────────────────────────────
section('RSM — res_corr_analysis')

rsm = RSM(rsm_sim.scores, max_score=MAX_SCORE)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    rsm.res_corr_analysis()

for attr in ['residual_correlations', 'eigenvectors', 'eigenvalues',
             'variance_explained', 'loadings']:
    check(hasattr(rsm, attr), f'RSM: attribute {attr!r} set')

check(is_corr_matrix(rsm.residual_correlations, NO_ITEMS),
      f'RSM: residual_correlations is ({NO_ITEMS}×{NO_ITEMS}), diagonal=1, symmetric')

check_pca_outputs(rsm, NO_ITEMS, 'RSM')

# ── Auto-trigger tests ────────────────────────────────────────────────────
section('Auto-trigger: fit_statistics called automatically if not yet run')

for ModelClass, data, name, kwargs in [
    (SLM, slm_sim.scores, 'SLM', {}),
    (PCM, pcm_sim.scores, 'PCM', {'max_score_vector': MAX_SCORE_VECTOR}),
    (RSM, rsm_sim.scores, 'RSM', {'max_score': MAX_SCORE}),
]:
    m = ModelClass(data, **kwargs)
    check(not hasattr(m, 'std_residual_df'),
          f'{name}: fit_statistics not yet run')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.res_corr_analysis()
    check(hasattr(m, 'std_residual_df'),
          f'{name}: fit_statistics auto-triggered by res_corr_analysis')
    check(hasattr(m, 'residual_correlations'),
          f'{name}: residual_correlations set after auto-trigger')

# ── Idempotency ───────────────────────────────────────────────────────────
section('Idempotency: calling res_corr_analysis twice gives same result')

for ModelClass, data, name, kwargs in [
    (SLM, slm_sim.scores, 'SLM', {}),
    (PCM, pcm_sim.scores, 'PCM', {'max_score_vector': MAX_SCORE_VECTOR}),
    (RSM, rsm_sim.scores, 'RSM', {'max_score': MAX_SCORE}),
]:
    m = ModelClass(data, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.res_corr_analysis()
        corr1 = m.residual_correlations.copy()
        m.res_corr_analysis()
        corr2 = m.residual_correlations
    check(np.allclose(corr1.values, corr2.values, atol=1e-10),
          f'{name}: residual_correlations identical on second call')

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
