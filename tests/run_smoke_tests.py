"""
run_smoke_tests.py
------------------
Unified smoke test runner for RaschPy: SLM, PCM, RSM, MFRM (all four models).

Sections per model
  1. Simulate → instantiate → calibrate          (bail on failure)
  2. stats_df methods                             (non-empty, no all-NaN cols)
  3. person_abils                                 (Series, no all-NaN)
  4. std_errors                                   (runs + SE attrs set)
  5. res_corr_analysis                            (runs + shapes correct)
  6. save_stats / save_residuals                  (files exist, non-empty)
  7. Plot methods via Agg backend                 (no crash)

MFRM additional section
  8. Anchor calibration                           (item_low/item_high → calibrate,
                                                   anchor_persons_{model} set)

Run with:
    python run_smoke_tests.py                     # all models
    python run_smoke_tests.py -v                  # verbose (print passing checks too)
    python run_smoke_tests.py --model slm         # single model
    python run_smoke_tests.py --model mfrm -v     # MFRM verbose
"""

import sys
import os
import glob
import traceback
import argparse
import tempfile
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import raschpy as rp
from raschpy.simulation import SLM_Sim, PCM_Sim, RSM_Sim
from raschpy.simulation import (
    MFRM_Sim_Global,
    MFRM_Sim_Items,
    MFRM_Sim_Thresholds,
    MFRM_Sim_Matrix,
    MFRM_Sim_Bivector,
    MFRM_Sim_Centrality,
    MFRM_Sim_PseudoHalo,
    MFRM_Sim_Bistretch,
)

warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────────

SIM_SEED    = 42
N_PERSONS   = 200
N_ITEMS     = 6
N_RATERS    = 4
MAX_SCORE   = 3
RATER_RANGE = 1.0

MFRM_MODELS = [
    ('Global',     'global',     MFRM_Sim_Global),
    ('Items',      'items',      MFRM_Sim_Items),
    ('Thresholds', 'thresholds', MFRM_Sim_Thresholds),
    ('Matrix',     'matrix',     MFRM_Sim_Matrix),
    ('Centrality', 'centrality', MFRM_Sim_Centrality),
    ('PseudoHalo', 'pseudo_halo', MFRM_Sim_PseudoHalo),
    ('Bistretch',  'bistretch',  MFRM_Sim_Bistretch),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS_STR = '\033[32mPASS\033[0m'
FAIL_STR = '\033[31mFAIL\033[0m'

results = []   # (name, passed, traceback_or_msg)


def check(name, expr, msg='', verbose=False):
    if expr:
        results.append((name, True, ''))
        if verbose:
            print(f'  {PASS_STR}  {name}')
    else:
        results.append((name, False, msg))
        short = msg.strip().splitlines()[-1] if msg else ''
        print(f'  {FAIL_STR}  {name}' + (f' — {short}' if short else ''))


def assert_df(name, df, verbose=False):
    if not isinstance(df, pd.DataFrame):
        check(name, False, f'expected DataFrame, got {type(df).__name__}', verbose)
        return
    check(name + ' — non-empty', len(df) >= 1, f'got {len(df)} rows', verbose)
    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    check(name + ' — no all-NaN cols', len(all_nan_cols) == 0,
          f'all-NaN: {all_nan_cols}', verbose)


def files_ok(*paths):
    return all(os.path.exists(p) and os.path.getsize(p) > 0 for p in paths)


def section(title):
    print(f'\n{"─" * 60}')
    print(f'  {title}')
    print(f'{"─" * 60}')


# ── SLM ───────────────────────────────────────────────────────────────────────

def run_slm(verbose=False):
    section('SLM')
    tag = 'SLM'

    # 1. Simulate → instantiate → calibrate
    try:
        np.random.seed(SIM_SEED)
        sim  = SLM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS)
        data = sim.responses
        check(f'{tag} simulate', True, verbose=verbose)
    except Exception:
        check(f'{tag} simulate', False, traceback.format_exc()); return

    try:
        m = rp.SLM(data)
        check(f'{tag} instantiate', True, verbose=verbose)
    except Exception:
        check(f'{tag} instantiate', False, traceback.format_exc()); return

    try:
        m.calibrate()
        check(f'{tag} calibrate', True, verbose=verbose)
    except Exception:
        check(f'{tag} calibrate', False, traceback.format_exc()); return

    # 2. stats_df
    for method, attr in [
        ('item_stats_df',   'item_stats'),
        ('person_stats_df', 'person_stats'),
        ('test_stats_df',   'test_stats'),
    ]:
        try:
            getattr(m, method)()
            assert_df(f'{tag} {method}', getattr(m, attr, None), verbose)
        except Exception:
            check(f'{tag} {method}', False, traceback.format_exc(), verbose)

    # 3. person_abils
    try:
        m.person_estimates()
        abils = getattr(m, 'persons', None)
        check(f'{tag} person_abils — Series', isinstance(abils, pd.Series), verbose=verbose)
        if isinstance(abils, pd.Series):
            check(f'{tag} person_abils — no all-NaN', not abils.isna().all(),
                  f'{abils.isna().sum()} NaNs of {len(abils)}', verbose)
    except Exception:
        check(f'{tag} person_abils', False, traceback.format_exc(), verbose)

    # 4. std_errors
    try:
        m.std_errors(no_of_samples=10)
        check(f'{tag} std_errors — runs', True, verbose=verbose)
        se = getattr(m, 'item_se', None)
        check(f'{tag} std_errors — item_se set',
              se is not None and not pd.Series(se).isna().all(), verbose=verbose)
    except Exception:
        check(f'{tag} std_errors', False, traceback.format_exc(), verbose)

    # 5. res_corr_analysis
    try:
        m.res_corr_analysis()
        check(f'{tag} res_corr_analysis — runs', True, verbose=verbose)
        check(f'{tag} res_corr_analysis — eigenvectors shape',
              m.eigenvectors is not None and
              m.eigenvectors.shape == (N_ITEMS - 1, N_ITEMS),
              f'shape={getattr(m.eigenvectors, "shape", None)}', verbose)
        check(f'{tag} res_corr_analysis — no zero eigenvalue',
              float(m.eigenvalues['Eigenvalue'].min()) > 1e-10, verbose=verbose)
        check(f'{tag} res_corr_analysis — loadings shape',
              m.loadings.shape == (N_ITEMS, N_ITEMS - 1),
              f'shape={m.loadings.shape}', verbose)
    except Exception:
        check(f'{tag} res_corr_analysis', False, traceback.format_exc(), verbose)

    # 6. save_stats / save_residuals
    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, 'slm')
        try:
            m.save_stats(filename=base, format='csv', no_of_samples=10)
            expected = [f'{base}_{t}_stats.csv' for t in ['item', 'person', 'test']]
            check(f'{tag} save_stats csv', files_ok(*expected), verbose=verbose)
        except Exception:
            check(f'{tag} save_stats csv', False, traceback.format_exc(), verbose)

        try:
            m.save_stats(filename=base, format='xlsx', no_of_samples=10)
            check(f'{tag} save_stats xlsx', files_ok(f'{base}.xlsx'), verbose=verbose)
        except Exception:
            check(f'{tag} save_stats xlsx', False, traceback.format_exc(), verbose)

        try:
            m.save_residuals(filename=base + '_res', format='csv', single=True)
            check(f'{tag} save_residuals csv single',
                  files_ok(base + '_res.csv'), verbose=verbose)
        except Exception:
            check(f'{tag} save_residuals csv single', False, traceback.format_exc(), verbose)

        try:
            m.save_residuals(filename=base + '_res_multi', format='xlsx', single=False)
            check(f'{tag} save_residuals xlsx multi',
                  files_ok(base + '_res_multi.xlsx'), verbose=verbose)
        except Exception:
            check(f'{tag} save_residuals xlsx multi', False, traceback.format_exc(), verbose)

    # 7. Plots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    item = m.responses.columns[0]
    for plot_name, call in [
        ('icc',                lambda: m.icc(item=item)),
        ('iic',                lambda: m.iic(item=item)),
        ('tcc',                lambda: m.tcc()),
        ('test_info',          lambda: m.test_info()),
        ('test_csem',          lambda: m.test_csem()),
        ('std_residuals_plot', lambda: m.std_residuals_plot()),
        ('wright_map',                   lambda: m.wright_map()),
        ('wright_map/person_scaling',    lambda: m.wright_map(person_scaling=5)),
        ('wright_map/prop+scaling',      lambda: m.wright_map(prop=True, person_scaling=3)),
        ('wright_map/horiz+scaling',     lambda: m.wright_map(orientation='horizontal', person_scaling=5)),
        ('wright_map/item_labels',       lambda: m.wright_map(item_labels=True, person_scaling=5)),
        ('wright_map/item_distribution', lambda: m.wright_map(item_distribution=True, person_scaling=5)),
    ]:
        try:
            call(); plt.close('all')
            check(f'{tag} plot/{plot_name}', True, verbose=verbose)
        except Exception:
            plt.close('all')
            check(f'{tag} plot/{plot_name}', False, traceback.format_exc(), verbose)


# ── PCM ───────────────────────────────────────────────────────────────────────

def run_pcm(verbose=False):
    section('PCM')
    tag = 'PCM'
    max_score_vec = [MAX_SCORE] * N_ITEMS

    try:
        np.random.seed(SIM_SEED)
        sim  = PCM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS,
                       max_score_vector=max_score_vec)
        data = sim.responses
        check(f'{tag} simulate', True, verbose=verbose)
    except Exception:
        check(f'{tag} simulate', False, traceback.format_exc()); return

    try:
        m = rp.PCM(data, max_score_vector=max_score_vec)
        check(f'{tag} instantiate', True, verbose=verbose)
    except Exception:
        check(f'{tag} instantiate', False, traceback.format_exc()); return

    try:
        m.calibrate()
        check(f'{tag} calibrate', True, verbose=verbose)
    except Exception:
        check(f'{tag} calibrate', False, traceback.format_exc()); return

    # stats_df — PCM has two threshold attrs
    for method, attr in [
        ('item_stats_df',   'item_stats'),
        ('person_stats_df', 'person_stats'),
        ('test_stats_df',   'test_stats'),
    ]:
        try:
            getattr(m, method)()
            assert_df(f'{tag} {method}', getattr(m, attr, None), verbose)
        except Exception:
            check(f'{tag} {method}', False, traceback.format_exc(), verbose)

    try:
        m.threshold_stats_df()
        for attr in ('threshold_stats_uncentred', 'threshold_stats'):
            assert_df(f'{tag} threshold_stats_df — {attr}',
                      getattr(m, attr, None), verbose)
    except Exception:
        check(f'{tag} threshold_stats_df', False, traceback.format_exc(), verbose)

    try:
        m.person_estimates()
        abils = getattr(m, 'persons', None)
        check(f'{tag} person_abils — Series', isinstance(abils, pd.Series), verbose=verbose)
        if isinstance(abils, pd.Series):
            check(f'{tag} person_abils — no all-NaN', not abils.isna().all(),
                  f'{abils.isna().sum()} NaNs of {len(abils)}', verbose)
    except Exception:
        check(f'{tag} person_abils', False, traceback.format_exc(), verbose)

    try:
        m.std_errors(no_of_samples=10)
        check(f'{tag} std_errors — runs', True, verbose=verbose)
        se = getattr(m, 'item_se', None)
        check(f'{tag} std_errors — item_se set',
              se is not None and not pd.Series(se).isna().all(), verbose=verbose)
    except Exception:
        check(f'{tag} std_errors', False, traceback.format_exc(), verbose)

    try:
        m.res_corr_analysis()
        check(f'{tag} res_corr_analysis — runs', True, verbose=verbose)
        check(f'{tag} res_corr_analysis — eigenvectors shape',
              m.eigenvectors is not None and
              m.eigenvectors.shape == (N_ITEMS - 1, N_ITEMS),
              f'shape={getattr(m.eigenvectors, "shape", None)}', verbose)
        check(f'{tag} res_corr_analysis — no zero eigenvalue',
              float(m.eigenvalues['Eigenvalue'].min()) > 1e-10, verbose=verbose)
        check(f'{tag} res_corr_analysis — loadings shape',
              m.loadings.shape == (N_ITEMS, N_ITEMS - 1),
              f'shape={m.loadings.shape}', verbose)
    except Exception:
        check(f'{tag} res_corr_analysis', False, traceback.format_exc(), verbose)

    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, 'pcm')
        try:
            m.save_stats(filename=base, format='csv', no_of_samples=10)
            expected = [f'{base}_{t}_stats.csv' for t in ['item', 'person', 'test']]
            check(f'{tag} save_stats csv', files_ok(*expected), verbose=verbose)
        except Exception:
            check(f'{tag} save_stats csv', False, traceback.format_exc(), verbose)

        try:
            m.save_stats(filename=base, format='xlsx', no_of_samples=10)
            check(f'{tag} save_stats xlsx', files_ok(f'{base}.xlsx'), verbose=verbose)
        except Exception:
            check(f'{tag} save_stats xlsx', False, traceback.format_exc(), verbose)

        try:
            m.save_residuals(filename=base + '_res', format='csv', single=True)
            check(f'{tag} save_residuals csv single',
                  files_ok(base + '_res.csv'), verbose=verbose)
        except Exception:
            check(f'{tag} save_residuals csv single', False, traceback.format_exc(), verbose)

        try:
            m.save_residuals(filename=base + '_res_multi', format='xlsx', single=False)
            check(f'{tag} save_residuals xlsx multi',
                  files_ok(base + '_res_multi.xlsx'), verbose=verbose)
        except Exception:
            check(f'{tag} save_residuals xlsx multi', False, traceback.format_exc(), verbose)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    item = m.responses.columns[0]
    for plot_name, call in [
        ('icc',                lambda: m.icc(item=item)),
        ('crcs',               lambda: m.crcs(item=item)),
        ('threshold_ccs',      lambda: m.threshold_ccs(item=item)),
        ('iic',                lambda: m.iic(item=item)),
        ('tcc',                lambda: m.tcc()),
        ('test_info',          lambda: m.test_info()),
        ('test_csem',          lambda: m.test_csem()),
        ('std_residuals_plot', lambda: m.std_residuals_plot()),
        ('wright_map',                   lambda: m.wright_map()),
        ('wright_map/person_scaling',    lambda: m.wright_map(person_scaling=5)),
        ('wright_map/prop+scaling',      lambda: m.wright_map(prop=True, person_scaling=3)),
        ('wright_map/horiz+scaling',     lambda: m.wright_map(orientation='horizontal', person_scaling=5)),
        ('wright_map/item_labels',       lambda: m.wright_map(item_labels=True, person_scaling=5)),
        ('wright_map/item_distribution', lambda: m.wright_map(item_distribution=True, person_scaling=5)),
    ]:
        try:
            call(); plt.close('all')
            check(f'{tag} plot/{plot_name}', True, verbose=verbose)
        except Exception:
            plt.close('all')
            check(f'{tag} plot/{plot_name}', False, traceback.format_exc(), verbose)


# ── RSM ───────────────────────────────────────────────────────────────────────

def run_rsm(verbose=False):
    section('RSM')
    tag = 'RSM'

    try:
        np.random.seed(SIM_SEED)
        sim  = RSM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS,
                       max_score=MAX_SCORE)
        data = sim.responses
        check(f'{tag} simulate', True, verbose=verbose)
    except Exception:
        check(f'{tag} simulate', False, traceback.format_exc()); return

    try:
        m = rp.RSM(data, max_score=MAX_SCORE)
        check(f'{tag} instantiate', True, verbose=verbose)
    except Exception:
        check(f'{tag} instantiate', False, traceback.format_exc()); return

    try:
        m.calibrate()
        check(f'{tag} calibrate', True, verbose=verbose)
    except Exception:
        check(f'{tag} calibrate', False, traceback.format_exc()); return

    for method, attr in [
        ('item_stats_df',      'item_stats'),
        ('threshold_stats_df', 'threshold_stats'),
        ('person_stats_df',    'person_stats'),
        ('test_stats_df',      'test_stats'),
    ]:
        try:
            getattr(m, method)()
            assert_df(f'{tag} {method}', getattr(m, attr, None), verbose)
        except Exception:
            check(f'{tag} {method}', False, traceback.format_exc(), verbose)

    try:
        m.person_estimates()
        abils = getattr(m, 'persons', None)
        check(f'{tag} person_abils — Series', isinstance(abils, pd.Series), verbose=verbose)
        if isinstance(abils, pd.Series):
            check(f'{tag} person_abils — no all-NaN', not abils.isna().all(),
                  f'{abils.isna().sum()} NaNs of {len(abils)}', verbose)
    except Exception:
        check(f'{tag} person_abils', False, traceback.format_exc(), verbose)

    try:
        m.std_errors(no_of_samples=10)
        check(f'{tag} std_errors — runs', True, verbose=verbose)
        se = getattr(m, 'item_se', None)
        check(f'{tag} std_errors — item_se set',
              se is not None and not pd.Series(se).isna().all(), verbose=verbose)
    except Exception:
        check(f'{tag} std_errors', False, traceback.format_exc(), verbose)

    try:
        m.res_corr_analysis()
        check(f'{tag} res_corr_analysis — runs', True, verbose=verbose)
        check(f'{tag} res_corr_analysis — eigenvectors shape',
              m.eigenvectors is not None and
              m.eigenvectors.shape == (N_ITEMS - 1, N_ITEMS),
              f'shape={getattr(m.eigenvectors, "shape", None)}', verbose)
        check(f'{tag} res_corr_analysis — no zero eigenvalue',
              float(m.eigenvalues['Eigenvalue'].min()) > 1e-10, verbose=verbose)
        check(f'{tag} res_corr_analysis — loadings shape',
              m.loadings.shape == (N_ITEMS, N_ITEMS - 1),
              f'shape={m.loadings.shape}', verbose)
    except Exception:
        check(f'{tag} res_corr_analysis', False, traceback.format_exc(), verbose)

    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, 'rsm')
        try:
            m.save_stats(filename=base, format='csv', no_of_samples=10)
            expected = [f'{base}_{t}_stats.csv' for t in ['item', 'person', 'test']]
            check(f'{tag} save_stats csv', files_ok(*expected), verbose=verbose)
        except Exception:
            check(f'{tag} save_stats csv', False, traceback.format_exc(), verbose)

        try:
            m.save_stats(filename=base, format='xlsx', no_of_samples=10)
            check(f'{tag} save_stats xlsx', files_ok(f'{base}.xlsx'), verbose=verbose)
        except Exception:
            check(f'{tag} save_stats xlsx', False, traceback.format_exc(), verbose)

        try:
            m.save_residuals(filename=base + '_res', format='csv', single=True)
            check(f'{tag} save_residuals csv single',
                  files_ok(base + '_res.csv'), verbose=verbose)
        except Exception:
            check(f'{tag} save_residuals csv single', False, traceback.format_exc(), verbose)

        try:
            m.save_residuals(filename=base + '_res_multi', format='xlsx', single=False)
            check(f'{tag} save_residuals xlsx multi',
                  files_ok(base + '_res_multi.xlsx'), verbose=verbose)
        except Exception:
            check(f'{tag} save_residuals xlsx multi', False, traceback.format_exc(), verbose)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    item = m.responses.columns[0]
    for plot_name, call in [
        ('icc',                lambda: m.icc(item=item)),
        ('crcs',               lambda: m.crcs(item=item)),
        ('threshold_ccs',      lambda: m.threshold_ccs(item=item)),
        ('iic',                lambda: m.iic(item=item)),
        ('tcc',                lambda: m.tcc()),
        ('test_info',          lambda: m.test_info()),
        ('test_csem',          lambda: m.test_csem()),
        ('std_residuals_plot', lambda: m.std_residuals_plot()),
        ('wright_map',                   lambda: m.wright_map()),
        ('wright_map/person_scaling',    lambda: m.wright_map(person_scaling=5)),
        ('wright_map/prop+scaling',      lambda: m.wright_map(prop=True, person_scaling=3)),
        ('wright_map/horiz+scaling',     lambda: m.wright_map(orientation='horizontal', person_scaling=5)),
        ('wright_map/item_labels',       lambda: m.wright_map(item_labels=True, person_scaling=5)),
        ('wright_map/item_distribution', lambda: m.wright_map(item_distribution=True, person_scaling=5)),
    ]:
        try:
            call(); plt.close('all')
            check(f'{tag} plot/{plot_name}', True, verbose=verbose)
        except Exception:
            plt.close('all')
            check(f'{tag} plot/{plot_name}', False, traceback.format_exc(), verbose)


# ── MFRM ─────────────────────────────────────────────────────────────────────

def run_mfrm_model(label, model_name, sim_cls, verbose=False):
    section(f'MFRM — {label}  (model="{model_name}")')
    tag = f'MFRM/{label}'

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 1. Simulate → instantiate → calibrate
    # centrality/pseudo_halo/bistretch renamed facet_range -> global_range
    range_kw = {'global_range': RATER_RANGE} if model_name in ('centrality', 'pseudo_halo', 'bistretch') else {'facet_range': RATER_RANGE}
    try:
        np.random.seed(SIM_SEED)
        sim = sim_cls(no_of_items=N_ITEMS, no_of_persons=N_PERSONS,
                      no_of_facet_elements=N_RATERS, max_score=MAX_SCORE,
                      **range_kw)
        data = sim.responses
        check(f'{tag} simulate', True, verbose=verbose)
    except Exception:
        check(f'{tag} simulate', False, traceback.format_exc()); return

    try:
        m = rp.MFRM(data, max_score=MAX_SCORE)
        check(f'{tag} instantiate', True, verbose=verbose)
    except Exception:
        check(f'{tag} instantiate', False, traceback.format_exc()); return

    try:
        m.calibrate(model=model_name)
        check(f'{tag} calibrate', True, verbose=verbose)
    except Exception:
        check(f'{tag} calibrate', False, traceback.format_exc()); return

    # 2. stats_df
    for method, attr in [
        ('item_stats_df',      f'item_stats_{model_name}'),
        ('threshold_stats_df', f'threshold_stats_{model_name}'),
        ('person_stats_df',    f'person_stats_{model_name}'),
        ('test_stats_df',      f'test_stats_{model_name}'),
    ]:
        try:
            getattr(m, method)(model=model_name)
            assert_df(f'{tag} {method}', getattr(m, attr, None), verbose)
        except Exception:
            check(f'{tag} {method}', False, traceback.format_exc(), verbose)

    try:
        m.rater_stats_df(model=model_name, marginal=True)
        assert_df(f'{tag} rater_stats_df(marginal=True)',
                  getattr(m, f'rater_stats_{model_name}', None), verbose)
    except Exception:
        check(f'{tag} rater_stats_df(marginal=True)', False, traceback.format_exc(), verbose)

    if model_name == 'matrix':
        try:
            m.rater_stats_df(model=model_name, marginal=False)
            assert_df(f'{tag} rater_stats_df(marginal=False)',
                      getattr(m, f'rater_stats_{model_name}', None), verbose)
        except Exception:
            check(f'{tag} rater_stats_df(marginal=False)', False,
                  traceback.format_exc(), verbose)

    # 3. person_abils
    try:
        m.person_estimates(model=model_name)
        abils = getattr(m, f'persons_{model_name}', None)
        check(f'{tag} person_abils — Series', isinstance(abils, pd.Series), verbose=verbose)
        if isinstance(abils, pd.Series):
            check(f'{tag} person_abils — no all-NaN', not abils.isna().all(),
                  f'{abils.isna().sum()} NaNs of {len(abils)}', verbose)
    except Exception:
        check(f'{tag} person_abils', False, traceback.format_exc(), verbose)

    # 4. std_errors
    try:
        m.std_errors(model=model_name, no_of_samples=10)
        check(f'{tag} std_errors — runs', True, verbose=verbose)
        se = getattr(m, 'item_se', None)
        check(f'{tag} std_errors — item_se set',
              se is not None and not pd.Series(se).isna().all(), verbose=verbose)
    except Exception:
        check(f'{tag} std_errors', False, traceback.format_exc(), verbose)

    # 5. res_corr_analysis
    try:
        getattr(m, f'item_res_corr_analysis_{model_name}')()
        eigv = getattr(m, f'item_eigenvectors_{model_name}')
        eigl = getattr(m, f'item_eigenvalues_{model_name}')
        load = getattr(m, f'item_loadings_{model_name}')
        check(f'{tag} item_res_corr_analysis — runs', True, verbose=verbose)
        check(f'{tag} item_res_corr_analysis — eigenvectors shape',
              eigv.shape == (N_ITEMS - 1, N_ITEMS),
              f'shape={eigv.shape}', verbose)
        check(f'{tag} item_res_corr_analysis — no zero eigenvalue',
              float(eigl['Eigenvalue'].min()) > 1e-10, verbose=verbose)
        check(f'{tag} item_res_corr_analysis — loadings shape',
              load.shape == (N_ITEMS, N_ITEMS - 1),
              f'shape={load.shape}', verbose)
    except Exception:
        check(f'{tag} item_res_corr_analysis', False, traceback.format_exc(), verbose)

    try:
        getattr(m, f'facet_res_corr_analysis_{model_name}')()
        eigv = getattr(m, f'rater_eigenvectors_{model_name}')
        eigl = getattr(m, f'rater_eigenvalues_{model_name}')
        load = getattr(m, f'rater_loadings_{model_name}')
        check(f'{tag} facet_res_corr_analysis — runs', True, verbose=verbose)
        check(f'{tag} facet_res_corr_analysis — eigenvectors shape',
              eigv.shape == (N_RATERS - 1, N_RATERS),
              f'shape={eigv.shape}', verbose)
        check(f'{tag} facet_res_corr_analysis — no zero eigenvalue',
              float(eigl['Eigenvalue'].min()) > 1e-10, verbose=verbose)
        check(f'{tag} facet_res_corr_analysis — loadings shape',
              load.shape == (N_RATERS, N_RATERS - 1),
              f'shape={load.shape}', verbose)
    except Exception:
        check(f'{tag} rater_res_corr_analysis', False, traceback.format_exc(), verbose)

    # 6. save_stats / save_residuals
    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, f'mfrm_{model_name}')

        try:
            m.save_stats(model=model_name, filename=base, format='csv', no_of_samples=10)
            expected = [f'{base}_{t}_stats.csv'
                        for t in ['item', 'threshold', 'rater', 'person', 'test']]
            check(f'{tag} save_stats csv', files_ok(*expected), verbose=verbose)
        except Exception:
            check(f'{tag} save_stats csv', False, traceback.format_exc(), verbose)

        try:
            m.save_stats(model=model_name, filename=base, format='xlsx', no_of_samples=10)
            check(f'{tag} save_stats xlsx', files_ok(f'{base}.xlsx'), verbose=verbose)
        except Exception:
            check(f'{tag} save_stats xlsx', False, traceback.format_exc(), verbose)

        try:
            getattr(m, f'save_residuals_items_{model_name}')(
                filename=base + '_item_res', format='csv', single=True)
            check(f'{tag} save_residuals items csv single',
                  files_ok(base + '_item_res.csv'), verbose=verbose)
        except Exception:
            check(f'{tag} save_residuals items csv single',
                  False, traceback.format_exc(), verbose)

        try:
            getattr(m, f'save_residuals_items_{model_name}')(
                filename=base + '_item_res_multi', format='xlsx', single=False)
            check(f'{tag} save_residuals items xlsx multi',
                  files_ok(base + '_item_res_multi.xlsx'), verbose=verbose)
        except Exception:
            check(f'{tag} save_residuals items xlsx multi',
                  False, traceback.format_exc(), verbose)

        try:
            getattr(m, f'save_residuals_raters_{model_name}')(
                filename=base + '_rater_res', format='csv', single=True)
            check(f'{tag} save_residuals raters csv single',
                  files_ok(base + '_rater_res.csv'), verbose=verbose)
        except Exception:
            check(f'{tag} save_residuals raters csv single',
                  False, traceback.format_exc(), verbose)

    # 7. Plots
    item  = m.item_names[0]
    facet_element = m.rater_names[0]
    for plot_name, call in [
        ('icc',                lambda: m.icc(item=item, facet_element=facet_element)),
        ('crcs',               lambda: m.crcs(item=item, facet_element=facet_element)),
        ('threshold_ccs',      lambda: m.threshold_ccs(item=item, facet_element=facet_element)),
        ('iic',                lambda: m.iic(item=item)),
        ('tcc',                lambda: m.tcc()),
        ('test_info',          lambda: m.test_info()),
        ('test_csem',          lambda: m.test_csem()),
        ('std_residuals_plot', lambda: m.std_residuals_plot()),
        ('wright_map',                  lambda: m.wright_map(model_name)),
        ('wright_map/person_scaling',   lambda: m.wright_map(model_name, person_scaling=5)),
        ('wright_map/overlay+scaling',  lambda: m.wright_map(model_name, overlay_facet=True, person_scaling=5)),
        ('wright_map/prop+scaling',     lambda: m.wright_map(model_name, prop=True, person_scaling=3)),
        ('wright_map/item_labels',      lambda: m.wright_map(model_name, item_labels=True, person_scaling=5)),
    ]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                call()
            plt.close('all')
            check(f'{tag} plot/{plot_name}', True, verbose=verbose)
        except Exception:
            plt.close('all')
            check(f'{tag} plot/{plot_name}', False, traceback.format_exc(), verbose)

    # 8. Anchor calibration
    _run_mfrm_anchor(m, data, model_name, tag, verbose)


def _run_mfrm_anchor(base_model, data, model_name, tag, verbose):
    """
    Anchor smoke test.

    Flow:
      1. calibrate() has already been run on base_model.
      2. calibrate_anchor(model, anchor_raters) anchors mean facet_effect of the
         anchor raters to zero and adjusts diffs/thresholds accordingly.
      3. person_abils(model, anchor=True) produces anchor_persons_{model}.

    Strategy: use the first two raters as anchor raters — arbitrary but
    consistent, and guaranteed to exist given N_RATERS=4.
    """
    section(f'  └─ Anchor calibration ({tag})')

    anchors = list(base_model.rater_names[:2])

    # calibrate_anchor() operates in-place on base_model (no re-instantiation needed)
    try:
        base_model.calibrate_anchor(model=model_name, anchors=anchors)
        check(f'{tag} anchor — calibrate_anchor runs', True, verbose=verbose)
    except Exception:
        check(f'{tag} anchor — calibrate_anchor runs', False,
              traceback.format_exc()); return

    # anchor_raters_{model} should be stored
    try:
        stored = getattr(base_model, f'anchor_rater_names_{model_name}', None)
        check(f'{tag} anchor — anchor_raters_{model_name} set',
              stored == anchors, f'got {stored}', verbose)
    except Exception:
        check(f'{tag} anchor — anchor_raters_{model_name} set', False,
              traceback.format_exc(), verbose)

    # person_abils with anchor=True triggers anchor_persons_{model}
    try:
        base_model.person_estimates(model=model_name, anchor=True)
        anc_abils = getattr(base_model, f'anchor_persons_{model_name}', None)
        check(f'{tag} anchor — anchor_persons_{model_name} set',
              anc_abils is not None, verbose=verbose)
        if isinstance(anc_abils, pd.Series):
            check(f'{tag} anchor — anchor_abils no all-NaN',
                  not anc_abils.isna().all(),
                  f'{anc_abils.isna().sum()} NaNs of {len(anc_abils)}', verbose)
    except Exception:
        check(f'{tag} anchor — person_abils(anchor=True)', False,
              traceback.format_exc(), verbose)

    # item_stats_df should still work post-anchor
    try:
        base_model.item_stats_df(model=model_name)
        df = getattr(base_model, f'item_stats_{model_name}', None)
        assert_df(f'{tag} anchor — item_stats_df', df, verbose)
    except Exception:
        check(f'{tag} anchor — item_stats_df', False, traceback.format_exc(), verbose)

    # person_stats_df with anchor_raters
    try:
        base_model.person_stats_df(model=model_name, anchors=anchors)
        df = getattr(base_model, f'person_stats_{model_name}', None)
        assert_df(f'{tag} anchor — person_stats_df', df, verbose)
    except Exception:
        check(f'{tag} anchor — person_stats_df', False,
              traceback.format_exc(), verbose)

    # anchor_std_errors — slow path (no stored bootstrap)
    try:
        base_model.anchor_std_errors(model=model_name)
        check(f'{tag} anchor — anchor_std_errors runs', True, verbose=verbose)
        anc_se = getattr(base_model, 'anchor_item_se', None)
        check(f'{tag} anchor — anchor_item_se set',
              isinstance(anc_se, pd.Series) and not anc_se.isna().all(),
              verbose=verbose)
        anc_thr_se = getattr(base_model, f'anchor_threshold_se_{model_name}', None)
        check(f'{tag} anchor — anchor_threshold_se_{model_name} set',
              anc_thr_se is not None, verbose=verbose)
        anc_rater_se = getattr(base_model, f'anchor_rater_se_{model_name}', None)
        check(f'{tag} anchor — anchor_rater_se_{model_name} set',
              anc_rater_se is not None, verbose=verbose)
    except Exception:
        check(f'{tag} anchor — anchor_std_errors runs', False,
              traceback.format_exc(), verbose)

    # anchor_std_errors — fast path (store_bootstrap=True)
    try:
        # Re-instantiate to get a clean model with stored bootstrap
        import raschpy as rp
        m2 = rp.MFRM(base_model.responses, max_score=base_model.max_score)
        m2.calibrate(model=model_name)
        m2.std_errors(model=model_name, no_of_samples=10, store_bootstrap=True)
        check(f'{tag} anchor — _bootstrap_stored_{model_name} flag set',
              getattr(m2, f'_bootstrap_stored_{model_name}', False), verbose=verbose)
        m2.calibrate_anchor(model=model_name, anchors=anchors)
        m2.anchor_std_errors(model=model_name)
        check(f'{tag} anchor — anchor_std_errors fast path runs', True, verbose=verbose)
        anc_se2 = getattr(m2, 'anchor_item_se', None)
        check(f'{tag} anchor — anchor_item_se set (fast path)',
              isinstance(anc_se2, pd.Series) and not anc_se2.isna().all(),
              verbose=verbose)
    except Exception:
        check(f'{tag} anchor — anchor_std_errors fast path runs', False,
              traceback.format_exc(), verbose)


def run_mfrm(verbose=False):
    for label, model_name, sim_cls in MFRM_MODELS:
        run_mfrm_model(label, model_name, sim_cls, verbose=verbose)



def run_bivector(verbose=False):
    """
    Bivector smoke tests — structural and anchoring checks.

    Covers:
      1. Instantiate and calibrate
      2. rater_stats_df (marginal=True)
      3. person_abils
      4. std_errors
      5. facet_res_corr_analysis
      6. Anchor calibration — antisymmetry, CSEMs, person shift
      7. Plot methods
    """
    section('MFRM — Bivector')
    tag = 'MFRM/Bivector'

    # 1. Simulate and calibrate
    try:
        np.random.seed(SIM_SEED)
        sim = MFRM_Sim_Bivector(no_of_items=N_ITEMS, no_of_persons=N_PERSONS,
                                no_of_facet_elements=N_RATERS, max_score=MAX_SCORE)
        data = sim.responses
        check(f'{tag} simulate', True, verbose=verbose)
    except Exception:
        check(f'{tag} simulate', False, traceback.format_exc()); return

    try:
        m = rp.MFRM(data, max_score=MAX_SCORE)
        m.calibrate(model='bivector')
        check(f'{tag} calibrate', True, verbose=verbose)
    except Exception:
        check(f'{tag} calibrate', False, traceback.format_exc()); return

    # 2. rater_stats_df
    try:
        m.rater_stats_df(model='bivector', marginal=True)
        assert_df(f'{tag} rater_stats_df(marginal=True)',
                  getattr(m, 'rater_stats_bivector', None), verbose)
    except Exception:
        check(f'{tag} rater_stats_df', False, traceback.format_exc(), verbose)

    # 3. person_estimates
    try:
        m.person_estimates(model='bivector')
        abils = getattr(m, 'persons_bivector', None)
        check(f'{tag} person_estimates — Series', isinstance(abils, pd.Series), verbose=verbose)
        if isinstance(abils, pd.Series):
            check(f'{tag} person_estimates — no all-NaN', not abils.isna().all(),
                  f'{abils.isna().sum()} NaNs of {len(abils)}', verbose)
    except Exception:
        check(f'{tag} person_estimates', False, traceback.format_exc(), verbose)

    # 4. std_errors
    try:
        m.std_errors(model='bivector', no_of_samples=10)
        check(f'{tag} std_errors — runs', True, verbose=verbose)
        se = getattr(m, 'item_se', None)
        check(f'{tag} std_errors — item_se set',
              se is not None and not pd.Series(se).isna().all(), verbose=verbose)
    except Exception:
        check(f'{tag} std_errors', False, traceback.format_exc(), verbose)

    # 5. facet_res_corr_analysis
    try:
        m.facet_res_corr_analysis_bivector()
        eigv = getattr(m, 'rater_eigenvectors_bivector', None)
        check(f'{tag} facet_res_corr_analysis — runs', True, verbose=verbose)
        check(f'{tag} facet_res_corr_analysis — eigenvectors set',
              eigv is not None, verbose=verbose)
    except Exception:
        check(f'{tag} facet_res_corr_analysis', False, traceback.format_exc(), verbose)

    # 6. Anchor calibration
    anchors = list(m.rater_names[:2])
    try:
        m.calibrate_anchor(model='bivector', anchors=anchors)
        check(f'{tag} anchor — calibrate_anchor runs', True, verbose=verbose)
        stored = getattr(m, 'anchor_rater_names_bivector', None)
        check(f'{tag} anchor — anchor_rater_names_bivector set',
              stored == anchors, f'got {stored}', verbose)
    except Exception:
        check(f'{tag} anchor — calibrate_anchor runs', False,
              traceback.format_exc()); 

    try:
        m.person_estimates(model='bivector', anchor=True)
        anc_abils = getattr(m, 'anchor_persons_bivector', None)
        check(f'{tag} anchor — anchor_persons_bivector set',
              anc_abils is not None, verbose=verbose)
        if isinstance(anc_abils, pd.Series):
            check(f'{tag} anchor — anchor_abils no all-NaN',
                  not anc_abils.isna().all(), verbose=verbose)
    except Exception:
        check(f'{tag} anchor — person_estimates(anchor=True)', False,
              traceback.format_exc(), verbose)

    try:
        m.anchor_std_errors(model='bivector', anchors=anchors, no_of_samples=10)
        check(f'{tag} anchor — anchor_std_errors runs', True, verbose=verbose)
        anc_se = getattr(m, 'anchor_item_se', None)
        check(f'{tag} anchor — anchor_item_se set',
              isinstance(anc_se, pd.Series) and not anc_se.isna().all(),
              verbose=verbose)
    except Exception:
        check(f'{tag} anchor — anchor_std_errors', False,
              traceback.format_exc(), verbose)

    # 7. Plots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    item           = m.item_names[0]
    facet_element  = m.rater_names[0]
    for plot_name, call in [
        ('icc',          lambda: m.icc(item=item, facet_element=facet_element, model='bivector')),
        ('crcs',         lambda: m.crcs(item=item, facet_element=facet_element, model='bivector')),
        ('threshold_ccs',lambda: m.threshold_ccs(item=item, facet_element=facet_element, model='bivector')),
        ('iic',          lambda: m.iic(item=item, model='bivector')),
        ('tcc',          lambda: m.tcc(model='bivector')),
        ('test_info',    lambda: m.test_info(model='bivector')),
        ('test_csem',    lambda: m.test_csem(model='bivector')),
        ('std_residuals_plot', lambda: m.std_residuals_plot(model='bivector')),
        ('wright_map',                 lambda: m.wright_map('bivector')),
        ('wright_map/person_scaling',  lambda: m.wright_map('bivector', person_scaling=5)),
        ('wright_map/overlay+scaling', lambda: m.wright_map('bivector', overlay_facet=True, person_scaling=5)),
    ]:
        try:
            call()
            plt.close('all')
            check(f'{tag} plot/{plot_name}', True, verbose=verbose)
        except Exception:
            check(f'{tag} plot/{plot_name}', False, traceback.format_exc(), verbose)
            plt.close('all')


# ── wright_map person_scaling behaviour ──────────────────────────────────────

def run_wright_map_scaling(verbose=False):
    """
    person_scaling must enlarge the item side of the mirrored (item_labels=
    False) wright map by exactly that factor, leave the person side and the
    figure size untouched, and never change the item-side tick *labels*.
    """
    section('wright_map / person_scaling')
    tag = 'wright_map'

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def _capture(make_plot):
        """Return dict of figure/axis metrics for one mirrored wright map.

        Reads ax.dataLim (the raw data extent, before autoscale margins) so
        the person-side max and item-side min are directly comparable across
        person_scaling values regardless of how each model sets its limits.
        """
        grabbed = {}
        real_close = plt.close

        def spy(fig=None):
            f = fig if hasattr(fig, 'get_size_inches') else plt.gcf()
            if 'size' not in grabbed and getattr(f, 'axes', None):
                ax0 = f.axes[0]
                dl = ax0.dataLim
                grabbed['size'] = tuple(round(v, 4) for v in f.get_size_inches())
                grabbed['data_lo'] = round(dl.y0, 6)   # item side (negative)
                grabbed['data_hi'] = round(dl.y1, 6)   # person side (positive)
                # largest labelled value on the item side (negative ticks) --
                # must stay a TRUE count, never the person_scaling-inflated one
                item_vals = []
                for pos, lab in zip(ax0.get_yticks(), ax0.get_yticklabels()):
                    txt = lab.get_text().replace('−', '-').strip()
                    if pos < -1e-9 and txt:
                        try:
                            item_vals.append(abs(float(txt)))
                        except ValueError:
                            pass
                grabbed['item_label_max'] = max(item_vals) if item_vals else 0.0
            return real_close(fig)

        plt.close = spy
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                make_plot()
        finally:
            plt.close = real_close
            real_close('all')
        return grabbed

    K = 4

    builders = []
    try:
        np.random.seed(SIM_SEED)
        s = rp.SLM(SLM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS).responses)
        s.calibrate()
        builders.append(('SLM', lambda ps: s.wright_map(person_scaling=ps)))
    except Exception:
        check(f'{tag} SLM setup', False, traceback.format_exc(), verbose)

    try:
        np.random.seed(SIM_SEED)
        max_score_vec = [MAX_SCORE] * N_ITEMS
        p = rp.PCM(PCM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS,
                           max_score_vector=max_score_vec).responses,
                   max_score_vector=max_score_vec)
        p.calibrate()
        builders.append(('PCM', lambda ps: p.wright_map(person_scaling=ps)))
    except Exception:
        check(f'{tag} PCM setup', False, traceback.format_exc(), verbose)

    try:
        np.random.seed(SIM_SEED)
        r = rp.RSM(RSM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS,
                           max_score=MAX_SCORE).responses, max_score=MAX_SCORE)
        r.calibrate()
        builders.append(('RSM', lambda ps: r.wright_map(person_scaling=ps)))
    except Exception:
        check(f'{tag} RSM setup', False, traceback.format_exc(), verbose)

    try:
        np.random.seed(SIM_SEED)
        mm = rp.MFRM(MFRM_Sim_Global(no_of_items=N_ITEMS, no_of_persons=N_PERSONS,
                                     no_of_facet_elements=N_RATERS,
                                     max_score=MAX_SCORE).responses,
                     max_score=MAX_SCORE)
        mm.calibrate(model='global')
        builders.append(('MFRM', lambda ps: mm.wright_map('global', overlay_facet=True,
                                                          person_scaling=ps)))
    except Exception:
        check(f'{tag} MFRM setup', False, traceback.format_exc(), verbose)

    for name, build in builders:
        try:
            base = _capture(lambda: build(1))
            scaled = _capture(lambda: build(K))

            # person side (max person count) untouched by person_scaling
            check(f'{tag} {name} person side unchanged',
                  abs(base['data_hi'] - scaled['data_hi']) < 1e-4,
                  f"{base['data_hi']} vs {scaled['data_hi']}", verbose)

            # item side (max item count, drawn negative) stretched by K
            ratio = scaled['data_lo'] / base['data_lo']
            check(f'{tag} {name} item side x{K}',
                  abs(ratio - K) < 1e-3, f'ratio={ratio:.4f}', verbose)

            # figure size unchanged (mirrored map never grows)
            check(f'{tag} {name} figure size unchanged',
                  base['size'] == scaled['size'],
                  f"{base['size']} vs {scaled['size']}", verbose)

            # item-side tick labels stay TRUE counts: even though the item
            # side is drawn x{K} larger, its largest label must NOT grow
            # with K -- it stays ~the same true-count ceiling as at ps=1
            check(f'{tag} {name} item tick labels stay true counts',
                  scaled['item_label_max'] >= 1
                  and scaled['item_label_max'] <= base['item_label_max'] * 1.5 + 0.5,
                  f"max item label {base['item_label_max']} (ps=1) -> "
                  f"{scaled['item_label_max']} (ps={K})", verbose)
        except Exception:
            check(f'{tag} {name} person_scaling behaviour', False,
                  traceback.format_exc(), verbose)


# ── Entry point ───────────────────────────────────────────────────────────────

RUNNERS = {
    'slm':      run_slm,
    'pcm':      run_pcm,
    'rsm':      run_rsm,
    'mfrm':     run_mfrm,
    'bivector': run_bivector,
    'wright_map': run_wright_map_scaling,
}


def main():
    parser = argparse.ArgumentParser(description='RaschPy unified smoke tests')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print each passing check')
    parser.add_argument('--model', choices=list(RUNNERS), default=None,
                        help='Run only this model (default: all)')
    args = parser.parse_args()

    to_run = [args.model] if args.model else list(RUNNERS)

    print(f'\n{"=" * 60}')
    print(f'  RaschPy smoke tests — {", ".join(r.upper() for r in to_run)}')
    print(f'{"=" * 60}')

    for name in to_run:
        RUNNERS[name](verbose=args.verbose)

    # Summary
    total  = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = total - passed

    print(f'\n{"=" * 60}')
    print(f'  {passed}/{total} checks passed', end='')
    if failed:
        print(f'  ({failed} failed)\n')
        print('  Failed checks:')
        for name, ok, msg in results:
            if not ok:
                short = msg.strip().splitlines()[-1] if msg else ''
                print(f'    ✗  {name}' + (f'  [{short}]' if short else ''))
                if args.verbose and msg:
                    for line in msg.strip().splitlines():
                        print(f'         {line}')
    else:
        print('  — all good.\n')

    print(f'{"=" * 60}\n')
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
