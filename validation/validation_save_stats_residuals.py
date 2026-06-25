"""
Validation tests for save_stats() and save_residuals() across SLM, PCM, RSM, MFRM.

Checks:
  - Files are created and non-empty
  - CSV files are readable as DataFrames with plausible shape
  - XLSX files contain the expected sheet names
  - Auto-trigger: methods run without prior calibration
  - Both single=True and single=False variants of save_residuals
  - MFRM: all four parameterisations (global, items, thresholds, matrix)

All output files are written to a temporary directory and cleaned up afterwards.

Run with:
    python tests/validation_save_stats_residuals.py
"""

import sys
import os
import tempfile
import shutil
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raschpy.simulation.slm_sim import SLM_Sim
from raschpy.simulation.pcm_sim import PCM_Sim
from raschpy.simulation.rsm_sim import RSM_Sim
from raschpy.simulation.mfrm_sim import MFRM_Sim_Global, MFRM_Sim_Items, \
    MFRM_Sim_Thresholds, MFRM_Sim_Matrix
from raschpy.slm import SLM
from raschpy.pcm import PCM
from raschpy.rsm import RSM
from raschpy.mfrm import MFRM

# ── Parameters ────────────────────────────────────────────────────────────
SEED             = 42
NO_ITEMS         = 6
NO_PERSONS       = 200
MAX_SCORE        = 3
MAX_SCORE_VECTOR = [3, 3, 3, 3, 3, 3]
NO_RATERS        = 3
MFRM_MODELS      = ['global', 'items', 'thresholds', 'matrix']

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
def file_exists_nonempty(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0

def csv_readable(path, min_rows=1, min_cols=1):
    """Check a CSV file is readable with at least min_rows and min_cols."""
    try:
        df = pd.read_csv(path, index_col=0)
        return df.shape[0] >= min_rows and df.shape[1] >= min_cols
    except Exception:
        return False

def xlsx_has_sheets(path, expected_sheets):
    """Check an XLSX file contains all expected sheet names."""
    try:
        xl = pd.ExcelFile(path)
        return all(s in xl.sheet_names for s in expected_sheets)
    except Exception:
        return False

def xlsx_sheet_nonempty(path, sheet):
    """Check an XLSX sheet has at least one row of data."""
    try:
        df = pd.read_excel(path, sheet_name=sheet, index_col=0)
        return df.shape[0] >= 1
    except Exception:
        return False

# ── Simulate data ─────────────────────────────────────────────────────────
section('Generating simulated datasets')
np.random.seed(SEED)
slm_data = SLM_Sim(no_of_items=NO_ITEMS, no_of_persons=NO_PERSONS).scores
check(isinstance(slm_data, pd.DataFrame), 'SLM data generated')

np.random.seed(SEED)
pcm_data = PCM_Sim(no_of_items=NO_ITEMS, no_of_persons=NO_PERSONS,
                   max_score_vector=MAX_SCORE_VECTOR).scores
check(isinstance(pcm_data, pd.DataFrame), 'PCM data generated')

np.random.seed(SEED)
rsm_data = RSM_Sim(no_of_items=NO_ITEMS, no_of_persons=NO_PERSONS,
                   max_score=MAX_SCORE).scores
check(isinstance(rsm_data, pd.DataFrame), 'RSM data generated')

mfrm_data = {}
sim_classes = {
    'global': MFRM_Sim_Global, 'items': MFRM_Sim_Items,
    'thresholds': MFRM_Sim_Thresholds, 'matrix': MFRM_Sim_Matrix
}
for model, SimClass in sim_classes.items():
    np.random.seed(SEED)
    mfrm_data[model] = SimClass(no_of_items=NO_ITEMS, no_of_persons=NO_PERSONS,
                                no_of_raters=NO_RATERS, max_score=MAX_SCORE).scores
    check(isinstance(mfrm_data[model], pd.DataFrame), f'MFRM {model} data generated')

# ── Use one temp dir for all output files ────────────────────────────────
tmpdir = tempfile.mkdtemp(prefix='raschpy_test_')
print(f'\n  Output directory: {tmpdir}')

try:

    # ── SLM save_stats ────────────────────────────────────────────────────
    section('SLM — save_stats')
    slm = SLM(slm_data)

    # CSV
    base = os.path.join(tmpdir, 'slm')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        slm.save_stats(base, format='csv')

    for suffix, label in [('_item_stats.csv', 'item stats'),
                           ('_person_stats.csv', 'person stats'),
                           ('_test_stats.csv', 'test stats')]:
        path = base + suffix
        check(file_exists_nonempty(path), f'SLM CSV {label} file created')
        check(csv_readable(path), f'SLM CSV {label} is readable DataFrame')

    # XLSX
    xlsx = os.path.join(tmpdir, 'slm.xlsx')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        slm.save_stats(xlsx, format='xlsx')

    check(file_exists_nonempty(xlsx), 'SLM XLSX file created')
    check(xlsx_has_sheets(xlsx, ['Item statistics', 'Person statistics', 'Test statistics']),
          'SLM XLSX has correct sheet names')
    for sheet in ['Item statistics', 'Person statistics', 'Test statistics']:
        check(xlsx_sheet_nonempty(xlsx, sheet), f'SLM XLSX sheet {sheet!r} non-empty')

    # ── SLM save_residuals ────────────────────────────────────────────────
    section('SLM — save_residuals')
    slm2 = SLM(slm_data)

    # single=True, CSV
    path = os.path.join(tmpdir, 'slm_res_single.csv')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        slm2.save_residuals(path, format='csv', single=True)
    check(file_exists_nonempty(path), 'SLM save_residuals single CSV created')

    # single=False, CSV
    base = os.path.join(tmpdir, 'slm_res_multi')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        slm2.save_residuals(base, format='csv', single=False)
    for suffix in ['_eigenvectors.csv', '_eigenvalues.csv',
                   '_variance_explained.csv', '_principal_component_loadings.csv']:
        check(file_exists_nonempty(base + suffix),
              f'SLM save_residuals multi CSV {suffix} created')

    # single=True, XLSX
    xlsx = os.path.join(tmpdir, 'slm_res_single.xlsx')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        slm2.save_residuals(xlsx, format='xlsx', single=True)
    check(file_exists_nonempty(xlsx), 'SLM save_residuals single XLSX created')
    check(xlsx_has_sheets(xlsx, ['Item residual analysis']),
          "SLM save_residuals single XLSX has 'Item residual analysis' sheet")

    # single=False, XLSX
    xlsx = os.path.join(tmpdir, 'slm_res_multi.xlsx')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        slm2.save_residuals(xlsx, format='xlsx', single=False)
    check(file_exists_nonempty(xlsx), 'SLM save_residuals multi XLSX created')
    check(xlsx_has_sheets(xlsx, ['Eigenvectors', 'Eigenvalues',
                                 'Variance explained', 'Principal Component loadings']),
          'SLM save_residuals multi XLSX has correct sheets')

    # ── PCM save_stats ────────────────────────────────────────────────────
    section('PCM — save_stats')
    pcm = PCM(pcm_data, max_score_vector=MAX_SCORE_VECTOR)

    base = os.path.join(tmpdir, 'pcm')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pcm.save_stats(base, format='csv')

    # PCM has two threshold tables (uncentred + centred)
    for suffix in ['_item_stats.csv', '_threshold_stats_uncentred.csv',
                   '_threshold_stats_centred.csv', '_person_stats.csv', '_test_stats.csv']:
        path = base + suffix
        check(file_exists_nonempty(path), f'PCM CSV {suffix} created')
        check(csv_readable(path), f'PCM CSV {suffix} readable')

    xlsx = os.path.join(tmpdir, 'pcm.xlsx')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pcm.save_stats(xlsx, format='xlsx')
    check(file_exists_nonempty(xlsx), 'PCM XLSX created')
    check(xlsx_has_sheets(xlsx, ['Item statistics', 'Person statistics', 'Test statistics']),
          'PCM XLSX has core sheets')

    # PCM save_residuals
    section('PCM — save_residuals')
    pcm2 = PCM(pcm_data, max_score_vector=MAX_SCORE_VECTOR)
    path = os.path.join(tmpdir, 'pcm_res.csv')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pcm2.save_residuals(path, format='csv', single=True)
    check(file_exists_nonempty(path), 'PCM save_residuals CSV created')

    # ── RSM save_stats ────────────────────────────────────────────────────
    section('RSM — save_stats')
    rsm = RSM(rsm_data, max_score=MAX_SCORE)

    base = os.path.join(tmpdir, 'rsm')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rsm.save_stats(base, format='csv')

    for suffix in ['_item_stats.csv', '_threshold_stats.csv',
                   '_person_stats.csv', '_test_stats.csv']:
        path = base + suffix
        check(file_exists_nonempty(path), f'RSM CSV {suffix} created')
        check(csv_readable(path), f'RSM CSV {suffix} readable')

    xlsx = os.path.join(tmpdir, 'rsm.xlsx')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rsm.save_stats(xlsx, format='xlsx')
    check(file_exists_nonempty(xlsx), 'RSM XLSX created')
    check(xlsx_has_sheets(xlsx, ['Item statistics', 'Person statistics', 'Test statistics']),
          'RSM XLSX has core sheets')

    # RSM save_residuals
    section('RSM — save_residuals')
    rsm2 = RSM(rsm_data, max_score=MAX_SCORE)
    path = os.path.join(tmpdir, 'rsm_res.csv')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rsm2.save_residuals(path, format='csv', single=True)
    check(file_exists_nonempty(path), 'RSM save_residuals CSV created')

    # ── MFRM save_stats ───────────────────────────────────────────────────
    section('MFRM — save_stats (all four models)')
    for model in MFRM_MODELS:
        m = MFRM(mfrm_data[model])
        base = os.path.join(tmpdir, f'mfrm_{model}')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m.save_stats(model=model, filename=base, format='csv')

        for suffix in ['_item_stats.csv', '_threshold_stats.csv',
                       '_rater_stats.csv', '_person_stats.csv', '_test_stats.csv']:
            path = base + suffix
            check(file_exists_nonempty(path), f'MFRM {model} CSV {suffix} created')
            check(csv_readable(path), f'MFRM {model} CSV {suffix} readable')

        xlsx = os.path.join(tmpdir, f'mfrm_{model}.xlsx')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m.save_stats(model=model, filename=xlsx, format='xlsx')
        check(file_exists_nonempty(xlsx), f'MFRM {model} XLSX created')
        check(xlsx_has_sheets(xlsx, ['Item statistics', 'Person statistics']),
              f'MFRM {model} XLSX has core sheets')

    # ── MFRM save_residuals ───────────────────────────────────────────────
    section('MFRM — save_residuals (item + rater, global model)')
    m = MFRM(mfrm_data['global'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.item_res_corr_analysis_global()
        m.rater_res_corr_analysis_global()

    # item residuals
    path = os.path.join(tmpdir, 'mfrm_item_res.csv')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.save_residuals_items_global(path, format='csv', single=True)
    check(file_exists_nonempty(path), 'MFRM save_residuals_items_global CSV created')

    # rater residuals
    path = os.path.join(tmpdir, 'mfrm_rater_res.csv')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m.save_residuals_raters_global(path, format='csv', single=True)
    check(file_exists_nonempty(path), 'MFRM save_residuals_raters_global CSV created')

finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f'\n  Temp files cleaned up.')

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
