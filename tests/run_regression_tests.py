"""
run_regression_tests.py
-----------------------
Unified numerical regression tests for SLM, PCM, RSM, and MFRM (all four models).

First run — generate known-good fixtures:
    python run_regression_tests.py --generate
    python run_regression_tests.py --generate --model slm   # single model

Subsequent runs — assert against fixtures:
    python run_regression_tests.py
    python run_regression_tests.py -v
    python run_regression_tests.py --model mfrm -v

Fixtures are saved to regression_data/ at the project root, one subfolder per
model (slm/, pcm/, rsm/, global/, items/, thresholds/, matrix/).

All numeric comparisons use atol=1e-6 via np.testing.assert_allclose.
Commit regression_data/ to version control after generating.
"""

import sys
import os
import json
import traceback
import argparse
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
)
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────────

SIM_SEED      = 42
N_PERSONS     = 200
N_ITEMS       = 6
N_RATERS      = 4
MAX_SCORE     = 3
MAX_SCORE_VEC = [3] * 6
RATER_RANGE   = 1.0
ATOL          = 1e-6
RECOVERY_R_MIN = 0.85   # minimum Pearson r for parameter recovery checks

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'regression_data')

MFRM_MODELS = [
    ('Global',     'global',     MFRM_Sim_Global),
    ('Items',      'items',      MFRM_Sim_Items),
    ('Thresholds', 'thresholds', MFRM_Sim_Thresholds),
    ('Matrix',     'matrix',     MFRM_Sim_Matrix),
]

# ── Reporting helpers ─────────────────────────────────────────────────────────

PASS_STR = '\033[32mPASS\033[0m'
FAIL_STR = '\033[31mFAIL\033[0m'

results = []


def check(name, expr, msg='', verbose=False):
    if expr:
        results.append((name, True, ''))
        if verbose:
            print(f'  {PASS_STR}  {name}')
    else:
        results.append((name, False, msg))
        short = msg.strip().splitlines()[-1] if msg else ''
        print(f'  {FAIL_STR}  {name}' + (f' — {short}' if short else ''))


def section(title):
    print(f'\n{"─" * 60}')
    print(f'  {title}')
    print(f'{"─" * 60}')


# ── Serialisation helpers ─────────────────────────────────────────────────────

def flatten_columns(df):
    """Flatten MultiIndex columns to 'a | b' strings for CSV round-trip."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [' | '.join(str(s) for s in col) for col in df.columns]
    return df


def save_df(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    flatten_columns(df).to_csv(path)


def load_df(path):
    return pd.read_csv(path, index_col=0)


def save_series(s, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    s.to_csv(path, header=True)


def load_series(path):
    return pd.read_csv(path, index_col=0).iloc[:, 0]


def save_array(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def load_array(path):
    return np.load(path + '.npy')


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_plot_lines(axes, path):
    data = {}
    for ax_idx, ax in enumerate(axes):
        for line_idx, line in enumerate(ax.lines):
            key = f'ax{ax_idx}_line{line_idx}'
            data[key] = {'x': line.get_xdata().tolist(),
                         'y': line.get_ydata().tolist()}
    save_json(data, path)


# ── Assertion helpers ─────────────────────────────────────────────────────────

def assert_df(name, df, fixture_csv, verbose):
    try:
        ref = load_df(fixture_csv)
    except FileNotFoundError:
        check(name, False, f'fixture not found: {fixture_csv}', verbose)
        return
    try:
        df = flatten_columns(df)
        shared   = ref.columns.intersection(df.columns)
        num_cols = ref[shared].select_dtypes(include='number').columns
        check(name + ' — columns match',
              set(ref.columns) == set(df.columns),
              f'ref={set(ref.columns)} got={set(df.columns)}', verbose)
        check(name + ' — index match',
              list(ref.index.astype(str)) == list(df.index.astype(str)),
              'index mismatch', verbose)
        np.testing.assert_allclose(
            df[num_cols].values.astype(float),
            ref[num_cols].values.astype(float),
            atol=ATOL, equal_nan=True)
        check(name + ' — numeric within 1e-6', True, verbose=verbose)
    except AssertionError as e:
        check(name + ' — numeric within 1e-6', False, str(e), verbose)
    except Exception:
        check(name + ' — numeric within 1e-6', False, traceback.format_exc(), verbose)


def assert_series(name, s, fixture_csv, verbose):
    try:
        ref = load_series(fixture_csv)
    except FileNotFoundError:
        check(name, False, f'fixture not found: {fixture_csv}', verbose)
        return
    try:
        np.testing.assert_allclose(
            s.values.astype(float), ref.values.astype(float),
            atol=ATOL, equal_nan=True)
        check(name + ' — values within 1e-6', True, verbose=verbose)
    except AssertionError as e:
        check(name + ' — values within 1e-6', False, str(e), verbose)


def assert_array(name, arr, fixture_npy, verbose):
    try:
        ref = load_array(fixture_npy)
    except FileNotFoundError:
        check(name, False, f'fixture not found: {fixture_npy}.npy', verbose)
        return
    try:
        np.testing.assert_allclose(arr, ref, atol=ATOL, equal_nan=True)
        check(name + ' — values within 1e-6', True, verbose=verbose)
    except AssertionError as e:
        check(name + ' — values within 1e-6', False, str(e), verbose)


def run_plot_checks(plot_calls, base, verbose):
    """Generate plots, compare line data against JSON fixtures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for plot_name, call in plot_calls:
            fixture = os.path.join(base, 'plots', f'{plot_name}.json')
            try:
                call()
                axes = plt.gcf().get_axes()
                plt.close('all')
            except Exception:
                plt.close('all')
                check(f'plot/{plot_name} — renders', False,
                      traceback.format_exc(), verbose)
                continue

            try:
                ref = load_json(fixture)
            except FileNotFoundError:
                check(f'plot/{plot_name}', False,
                      f'fixture not found: {fixture}', verbose)
                continue

            current = {}
            for ax_idx, ax in enumerate(axes):
                for line_idx, line in enumerate(ax.lines):
                    key = f'ax{ax_idx}_line{line_idx}'
                    current[key] = {'x': line.get_xdata().tolist(),
                                    'y': line.get_ydata().tolist()}

            keys_match = set(ref.keys()) == set(current.keys())
            check(f'plot/{plot_name} — line count', keys_match,
                  f'ref={set(ref.keys())} got={set(current.keys())}', verbose)
            if not keys_match:
                continue

            all_ok = True
            for key in ref:
                try:
                    np.testing.assert_allclose(
                        np.array(current[key]['x']),
                        np.array(ref[key]['x']),
                        atol=ATOL, equal_nan=True)
                    np.testing.assert_allclose(
                        np.array(current[key]['y']),
                        np.array(ref[key]['y']),
                        atol=ATOL, equal_nan=True)
                except AssertionError as e:
                    check(f'plot/{plot_name}/{key} — within 1e-6', False,
                          str(e), verbose)
                    all_ok = False
            if all_ok:
                check(f'plot/{plot_name} — all lines within 1e-6', True, verbose=verbose)


def save_plot_fixtures(plot_calls, base):
    """Generate plots and save line data as JSON fixtures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for plot_name, call in plot_calls:
            try:
                call()
                axes = plt.gcf().get_axes()
                save_plot_lines(axes,
                    os.path.join(base, 'plots', f'{plot_name}.json'))
                plt.close('all')
            except Exception:
                plt.close('all')
                print(f'  WARNING: could not save plot fixture for {plot_name}')


# ── SLM ───────────────────────────────────────────────────────────────────────

def build_slm():
    np.random.seed(SIM_SEED)
    sim = SLM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS)
    m   = rp.SLM(sim.responses)
    m.calibrate()
    m.item_stats_df()
    m.person_stats_df()
    m.test_stats_df()
    m.person_estimates()
    m.std_errors(no_of_samples=100)
    return m


def slm_plot_calls(m):
    item = m.responses.columns[0]
    return [
        ('icc',                lambda: m.icc(item=item)),
        ('iic',                lambda: m.iic(item=item)),
        ('tcc',                lambda: m.tcc()),
        ('test_info',          lambda: m.test_info()),
        ('test_csem',          lambda: m.test_csem()),
        ('std_residuals_plot', lambda: m.std_residuals_plot()),
    ]


def generate_slm(m):
    base = os.path.join(FIXTURE_DIR, 'slm')
    save_df(m.item_stats,       os.path.join(base, 'item_stats.csv'))
    save_df(m.person_stats,     os.path.join(base, 'person_stats.csv'))
    save_df(m.test_stats,       os.path.join(base, 'test_stats.csv'))
    save_series(m.persons, os.path.join(base, 'person_abilities.csv'))
    save_series(m.item_se,      os.path.join(base, 'item_se.csv'))
    save_plot_fixtures(slm_plot_calls(m), base)
    print('  Fixtures saved → regression_data/slm/')


def assert_slm(m, verbose):
    base = os.path.join(FIXTURE_DIR, 'slm')
    assert_df('[SLM] item_stats',   m.item_stats,
              os.path.join(base, 'item_stats.csv'), verbose)
    assert_df('[SLM] person_stats', m.person_stats,
              os.path.join(base, 'person_stats.csv'), verbose)
    assert_df('[SLM] test_stats',   m.test_stats,
              os.path.join(base, 'test_stats.csv'), verbose)
    assert_series('[SLM] person_abilities', m.persons,
                  os.path.join(base, 'person_abilities.csv'), verbose)
    assert_series('[SLM] item_se', m.item_se,
                  os.path.join(base, 'item_se.csv'), verbose)
    run_plot_checks(slm_plot_calls(m), base, verbose)


# ── PCM ───────────────────────────────────────────────────────────────────────

def build_pcm():
    np.random.seed(SIM_SEED)
    sim = PCM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS,
                  max_score_vector=MAX_SCORE_VEC)
    m   = rp.PCM(sim.responses, max_score_vector=MAX_SCORE_VEC)
    m.calibrate()
    m.item_stats_df()
    m.person_stats_df()
    m.test_stats_df()
    m.person_estimates()
    m.std_errors(no_of_samples=100)
    return m


def pcm_plot_calls(m):
    item = m.responses.columns[0]
    return [
        ('icc',                lambda: m.icc(item=item)),
        ('crcs',               lambda: m.crcs(item=item)),
        ('iic',                lambda: m.iic(item=item)),
        ('tcc',                lambda: m.tcc()),
        ('test_info',          lambda: m.test_info()),
        ('test_csem',          lambda: m.test_csem()),
        ('std_residuals_plot', lambda: m.std_residuals_plot()),
    ]


def generate_pcm(m):
    base = os.path.join(FIXTURE_DIR, 'pcm')
    save_df(m.item_stats,       os.path.join(base, 'item_stats.csv'))
    save_df(m.person_stats,     os.path.join(base, 'person_stats.csv'))
    save_df(m.test_stats,       os.path.join(base, 'test_stats.csv'))
    save_series(m.persons, os.path.join(base, 'person_abilities.csv'))
    save_series(m.item_se,      os.path.join(base, 'item_se.csv'))
    thr_dir = os.path.join(base, 'threshold_se_by_item')
    for item, arr in m.threshold_se.items():
        save_array(arr, os.path.join(thr_dir, str(item)))
    save_plot_fixtures(pcm_plot_calls(m), base)
    print('  Fixtures saved → regression_data/pcm/')


def assert_pcm(m, verbose):
    base = os.path.join(FIXTURE_DIR, 'pcm')
    assert_df('[PCM] item_stats',   m.item_stats,
              os.path.join(base, 'item_stats.csv'), verbose)
    assert_df('[PCM] person_stats', m.person_stats,
              os.path.join(base, 'person_stats.csv'), verbose)
    assert_df('[PCM] test_stats',   m.test_stats,
              os.path.join(base, 'test_stats.csv'), verbose)
    assert_series('[PCM] person_abilities', m.persons,
                  os.path.join(base, 'person_abilities.csv'), verbose)
    assert_series('[PCM] item_se', m.item_se,
                  os.path.join(base, 'item_se.csv'), verbose)
    thr_dir = os.path.join(base, 'threshold_se_by_item')
    for item, arr in m.threshold_se.items():
        assert_array(f'[PCM] threshold_se[{item}]', arr,
                     os.path.join(thr_dir, str(item)), verbose)
    run_plot_checks(pcm_plot_calls(m), base, verbose)


# ── RSM ───────────────────────────────────────────────────────────────────────

def build_rsm():
    np.random.seed(SIM_SEED)
    sim = RSM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS,
                  max_score=MAX_SCORE)
    m   = rp.RSM(sim.responses, max_score=MAX_SCORE)
    m.calibrate()
    m.item_stats_df()
    m.person_stats_df()
    m.test_stats_df()
    m.person_estimates()
    m.std_errors(no_of_samples=100)
    return m


def rsm_plot_calls(m):
    item = m.responses.columns[0]
    return [
        ('icc',                lambda: m.icc(item=item)),
        ('crcs',               lambda: m.crcs(item=item)),
        ('threshold_ccs',      lambda: m.threshold_ccs(item=item)),
        ('iic',                lambda: m.iic(item=item)),
        ('tcc',                lambda: m.tcc()),
        ('test_info',          lambda: m.test_info()),
        ('test_csem',          lambda: m.test_csem()),
        ('std_residuals_plot', lambda: m.std_residuals_plot()),
    ]


def generate_rsm(m):
    base = os.path.join(FIXTURE_DIR, 'rsm')
    save_df(m.item_stats,       os.path.join(base, 'item_stats.csv'))
    save_df(m.person_stats,     os.path.join(base, 'person_stats.csv'))
    save_df(m.test_stats,       os.path.join(base, 'test_stats.csv'))
    save_series(m.persons, os.path.join(base, 'person_abilities.csv'))
    save_series(m.item_se,      os.path.join(base, 'item_se.csv'))
    # threshold_se: Series or dict of arrays depending on RSM implementation
    if isinstance(m.threshold_se, dict):
        thr_dir = os.path.join(base, 'threshold_se_by_item')
        for item, arr in m.threshold_se.items():
            save_array(arr, os.path.join(thr_dir, str(item)))
    else:
        thr_se = m.threshold_se if isinstance(m.threshold_se, pd.Series) \
                 else pd.Series(m.threshold_se)
        save_series(thr_se, os.path.join(base, 'threshold_se.csv'))
    save_series(m.cat_width_se, os.path.join(base, 'cat_width_se.csv'))
    save_plot_fixtures(rsm_plot_calls(m), base)
    print('  Fixtures saved → regression_data/rsm/')


def assert_rsm(m, verbose):
    base = os.path.join(FIXTURE_DIR, 'rsm')
    assert_df('[RSM] item_stats',   m.item_stats,
              os.path.join(base, 'item_stats.csv'), verbose)
    assert_df('[RSM] person_stats', m.person_stats,
              os.path.join(base, 'person_stats.csv'), verbose)
    assert_df('[RSM] test_stats',   m.test_stats,
              os.path.join(base, 'test_stats.csv'), verbose)
    assert_series('[RSM] person_abilities', m.persons,
                  os.path.join(base, 'person_abilities.csv'), verbose)
    assert_series('[RSM] item_se', m.item_se,
                  os.path.join(base, 'item_se.csv'), verbose)
    if isinstance(m.threshold_se, dict):
        thr_dir = os.path.join(base, 'threshold_se_by_item')
        for item, arr in m.threshold_se.items():
            assert_array(f'[RSM] threshold_se[{item}]', arr,
                         os.path.join(thr_dir, str(item)), verbose)
    else:
        thr_se = m.threshold_se if isinstance(m.threshold_se, pd.Series) \
                 else pd.Series(m.threshold_se)
        assert_series('[RSM] threshold_se', thr_se,
                      os.path.join(base, 'threshold_se.csv'), verbose)
    assert_series('[RSM] cat_width_se', m.cat_width_se,
                  os.path.join(base, 'cat_width_se.csv'), verbose)
    run_plot_checks(rsm_plot_calls(m), base, verbose)


# ── MFRM ─────────────────────────────────────────────────────────────────────

def build_mfrm(sim_cls, model_name):
    np.random.seed(SIM_SEED)
    sim = sim_cls(no_of_items=N_ITEMS, no_of_persons=N_PERSONS,
                  no_of_raters=N_RATERS, max_score=MAX_SCORE,
                  facet_range=RATER_RANGE)
    m = rp.MFRM(sim.responses, max_score=MAX_SCORE)
    m.calibrate(model=model_name)
    m.item_stats_df(model=model_name)
    m.threshold_stats_df(model=model_name)
    m.person_stats_df(model=model_name)
    m.test_stats_df(model=model_name)
    m.rater_stats_df(model=model_name, marginal=True)
    if model_name == 'matrix':
        m.rater_stats_df(model=model_name, marginal=False)
    m.person_estimates(model=model_name)
    m.std_errors(model=model_name, no_of_samples=100)
    # Anchor: use first two raters
    anchors = list(m.rater_names[:2])
    m.calibrate_anchor(model=model_name, anchors=anchors)
    m.anchor_std_errors(model=model_name)
    return m


def mfrm_plot_calls(m):
    item  = m.item_names[0]
    facet_element = m.rater_names[0]
    return [
        ('icc',                lambda: m.icc(item=item, facet_element=facet_element)),
        ('crcs',               lambda: m.crcs(item=item, facet_element=facet_element)),
        ('threshold_ccs',      lambda: m.threshold_ccs(item=item, facet_element=facet_element)),
        ('iic',                lambda: m.iic(item=item)),
        ('tcc',                lambda: m.tcc()),
        ('test_info',          lambda: m.test_info()),
        ('test_csem',          lambda: m.test_csem()),
        ('std_residuals_plot', lambda: m.std_residuals_plot()),
    ]


def _save_rater_se(m, model_name, base, anchor=False):
    prefix = 'anchor_' if anchor else ''
    rater_se = getattr(m, f'{prefix}rater_se_{model_name}', None)
    if rater_se is None:
        return
    path = os.path.join(base, f'{prefix}rater_se_{model_name}')
    if model_name == 'global':
        save_series(rater_se, path + '.csv')
    elif model_name == 'items':
        for rater, s in rater_se.items():
            save_series(s, os.path.join(path, f'{rater}.csv'))
    elif model_name == 'thresholds':
        for rater, arr in rater_se.items():
            save_array(arr, os.path.join(path, str(rater)))
    elif model_name == 'matrix':
        np.save(path + '.npy', rater_se.values)


def _assert_rater_se(m, model_name, base, verbose, anchor=False):
    prefix = 'anchor_' if anchor else ''
    rater_se = getattr(m, f'{prefix}rater_se_{model_name}', None)
    if rater_se is None:
        check(f'{prefix}rater_se_{model_name} — exists', False, 'attribute not found', verbose)
        return
    path = os.path.join(base, f'{prefix}rater_se_{model_name}')
    tag  = f'[MFRM/{model_name}] {prefix}rater_se'
    if model_name == 'global':
        assert_series(f'{tag}', rater_se, path + '.csv', verbose)
    elif model_name == 'items':
        for rater, s in rater_se.items():
            assert_series(f'{tag}[{rater}]', s,
                          os.path.join(path, f'{rater}.csv'), verbose)
    elif model_name == 'thresholds':
        for rater, arr in rater_se.items():
            assert_array(f'{tag}[{rater}]', arr,
                         os.path.join(path, str(rater)), verbose)
    elif model_name == 'matrix':
        try:
            ref = np.load(path + '.npy')
            np.testing.assert_allclose(rater_se.values, ref, atol=ATOL, equal_nan=True)
            check(f'{tag} — numeric within 1e-6', True, verbose=verbose)
        except FileNotFoundError:
            check(f'{tag}', False, f'fixture not found: {path}.npy', verbose)
        except AssertionError as e:
            check(f'{tag} — numeric within 1e-6', False, str(e), verbose)


def generate_mfrm(m, model_name):
    base = os.path.join(FIXTURE_DIR, model_name)
    os.makedirs(base, exist_ok=True)

    for attr in (
        f'item_stats_{model_name}',
        f'threshold_stats_{model_name}',
        f'person_stats_{model_name}',
        f'test_stats_{model_name}',
        f'rater_stats_{model_name}',
    ):
        df = getattr(m, attr, None)
        if isinstance(df, pd.DataFrame):
            save_df(df, os.path.join(base, f'{attr}.csv'))

    abils = getattr(m, f'persons_{model_name}', None)
    if isinstance(abils, pd.Series):
        save_series(abils, os.path.join(base, f'persons_{model_name}.csv'))

    if isinstance(getattr(m, 'item_se', None), pd.Series):
        save_series(m.item_se, os.path.join(base, 'item_se.csv'))

    thr_se = getattr(m, f'threshold_se_{model_name}', None)
    if isinstance(thr_se, np.ndarray):
        save_array(thr_se, os.path.join(base, f'threshold_se_{model_name}'))

    _save_rater_se(m, model_name, base)
    save_plot_fixtures(mfrm_plot_calls(m), base)

    # Anchor SE fixtures
    if isinstance(getattr(m, 'anchor_item_se', None), pd.Series):
        save_series(m.anchor_item_se,
                    os.path.join(base, 'anchor_item_se.csv'))
    anc_thr_se = getattr(m, f'anchor_threshold_se_{model_name}', None)
    if isinstance(anc_thr_se, np.ndarray):
        save_array(anc_thr_se,
                   os.path.join(base, f'anchor_threshold_se_{model_name}'))
    _save_rater_se(m, model_name, base, anchor=True)

    print(f'  Fixtures saved → regression_data/{model_name}/')


def assert_mfrm(m, model_name, label, verbose):
    base = os.path.join(FIXTURE_DIR, model_name)
    tag  = f'MFRM/{label}'

    for attr in (
        f'item_stats_{model_name}',
        f'threshold_stats_{model_name}',
        f'person_stats_{model_name}',
        f'test_stats_{model_name}',
        f'rater_stats_{model_name}',
    ):
        df = getattr(m, attr, None)
        if isinstance(df, pd.DataFrame):
            assert_df(f'[{tag}] {attr}', df,
                      os.path.join(base, f'{attr}.csv'), verbose)

    abils = getattr(m, f'persons_{model_name}', None)
    if isinstance(abils, pd.Series):
        assert_series(f'[{tag}] persons_{model_name}', abils,
                      os.path.join(base, f'persons_{model_name}.csv'), verbose)

    item_se = getattr(m, 'item_se', None)
    if isinstance(item_se, pd.Series):
        assert_series(f'[{tag}] item_se', item_se,
                      os.path.join(base, 'item_se.csv'), verbose)

    thr_se = getattr(m, f'threshold_se_{model_name}', None)
    if isinstance(thr_se, np.ndarray):
        assert_array(f'[{tag}] threshold_se_{model_name}', thr_se,
                     os.path.join(base, f'threshold_se_{model_name}'), verbose)

    _assert_rater_se(m, model_name, base, verbose)

    # Anchor SE assertions
    anc_item_se = getattr(m, 'anchor_item_se', None)
    if isinstance(anc_item_se, pd.Series):
        assert_series(f'[{tag}] anchor_item_se', anc_item_se,
                      os.path.join(base, 'anchor_item_se.csv'), verbose)

    anc_thr_se = getattr(m, f'anchor_threshold_se_{model_name}', None)
    if isinstance(anc_thr_se, np.ndarray):
        assert_array(f'[{tag}] anchor_threshold_se_{model_name}', anc_thr_se,
                     os.path.join(base, f'anchor_threshold_se_{model_name}'), verbose)

    _assert_rater_se(m, model_name, base, verbose, anchor=True)

    run_plot_checks(mfrm_plot_calls(m), base, verbose)


# ── Per-model runners ─────────────────────────────────────────────────────────

def run_slm(generate=False, verbose=False):
    section('SLM')
    try:
        m = build_slm()
    except Exception:
        check('[SLM] build', False, traceback.format_exc())
        return
    if generate:
        generate_slm(m)
    else:
        assert_slm(m, verbose)


def run_pcm(generate=False, verbose=False):
    section('PCM')
    try:
        m = build_pcm()
    except Exception:
        check('[PCM] build', False, traceback.format_exc())
        return
    if generate:
        generate_pcm(m)
    else:
        assert_pcm(m, verbose)


def run_rsm(generate=False, verbose=False):
    section('RSM')
    try:
        m = build_rsm()
    except Exception:
        check('[RSM] build', False, traceback.format_exc())
        return
    if generate:
        generate_rsm(m)
    else:
        assert_rsm(m, verbose)


def run_mfrm(generate=False, verbose=False):
    for label, model_name, sim_cls in MFRM_MODELS:
        section(f'MFRM — {label}  (model="{model_name}")')
        try:
            m = build_mfrm(sim_cls, model_name)
        except Exception:
            check(f'[MFRM/{label}] build', False, traceback.format_exc())
            continue
        if generate:
            generate_mfrm(m, model_name)
        else:
            assert_mfrm(m, model_name, label, verbose)



# ── Bivector: parameter recovery and consistency ──────────────────────────────

def build_bivector():
    np.random.seed(SIM_SEED)
    sim = MFRM_Sim_Bivector(
        no_of_items=N_ITEMS, no_of_persons=N_PERSONS,
        no_of_raters=N_RATERS, max_score=MAX_SCORE,
    )
    m = rp.MFRM(sim.responses, max_score=MAX_SCORE)
    m.calibrate(model='bivector')
    m.person_estimates(model='bivector')
    m.fit_statistics(model='bivector')
    return sim, m


def generate_bivector(sim, m):
    base = os.path.join(FIXTURE_DIR, 'bivector')
    os.makedirs(base, exist_ok=True)
    save_series(m.items,    os.path.join(base, 'items.csv'))
    save_series(pd.Series(m.thresholds), os.path.join(base, 'thresholds.csv'))
    save_series(m.persons_bivector, os.path.join(base, 'persons_bivector.csv'))
    save_series(m.item_infit_ms_bivector,
                os.path.join(base, 'item_infit_ms_bivector.csv'))
    # Recovery correlations (computed fresh each time, saved for reference)
    true_item, est_item = [], []
    for r in m.rater_names:
        for item in m.item_names:
            true_item.append(sim.item_effects.loc[r, item])
            est_item.append(m.raters_bivector_items.loc[r, item])
    r_item, _ = pearsonr(true_item, est_item)
    true_thresh, est_thresh = [], []
    for r in m.rater_names:
        true_thresh.extend(sim.threshold_effects.loc[r].values.tolist())
        est_thresh.extend(m.raters_bivector_thresholds.loc[r].values.tolist())
    r_thresh, _ = pearsonr(true_thresh, est_thresh)
    with open(os.path.join(base, 'recovery.json'), 'w') as f:
        import json
        json.dump({'r_item': r_item, 'r_thresh': r_thresh}, f)
    # Consistency vs matrix
    np.random.seed(SIM_SEED)
    sim2 = MFRM_Sim_Bivector(
        no_of_items=N_ITEMS, no_of_persons=N_PERSONS,
        no_of_raters=N_RATERS, max_score=MAX_SCORE,
    )
    m_mat = rp.MFRM(sim2.responses, max_score=MAX_SCORE)
    m_mat.calibrate(model='matrix')
    r_diffs, _ = pearsonr(m.items.values, m_mat.items.values)
    with open(os.path.join(base, 'consistency.json'), 'w') as f:
        json.dump({'r_items_vs_matrix': r_diffs}, f)
    print(f'  Fixtures saved → {base}/')


def assert_bivector(sim, m, verbose=False):
    base = os.path.join(FIXTURE_DIR, 'bivector')

    assert_series('[Bivector] items', m.items,
                  os.path.join(base, 'items.csv'), verbose)
    assert_series('[Bivector] thresholds', pd.Series(m.thresholds),
               os.path.join(base, 'thresholds.csv'), verbose)
    assert_series('[Bivector] persons_bivector', m.persons_bivector,
                  os.path.join(base, 'persons_bivector.csv'), verbose)
    assert_series('[Bivector] item_infit_ms_bivector', m.item_infit_ms_bivector,
                  os.path.join(base, 'item_infit_ms_bivector.csv'), verbose)

    # Recovery
    true_item, est_item = [], []
    for r in m.rater_names:
        for item in m.item_names:
            true_item.append(sim.item_effects.loc[r, item])
            est_item.append(m.raters_bivector_items.loc[r, item])
    r_item, _ = pearsonr(true_item, est_item)
    check('[Bivector] item effect recovery r >= 0.85',
          r_item >= RECOVERY_R_MIN, f'r={r_item:.3f}', verbose)

    # Threshold recovery omitted here — needs larger sample (N>=300, I>=8)
    # See run_bivector_tests.py for the full threshold recovery check.

    # Zero-sum constraint: across raters per item and per threshold
    for item in m.item_names:
        s = sum(m.raters_bivector_items.loc[r, item] for r in m.rater_names)
        check(f'[Bivector] item cross-rater sum ~ 0 ({item})',
              abs(s) < 0.01, f'sum={s:.6f}', verbose)
    for k in range(MAX_SCORE):
        s = sum(m.raters_bivector_thresholds.loc[r, k] for r in m.rater_names)
        check(f'[Bivector] threshold cross-rater sum ~ 0 (k={k})',
              abs(s) < 0.01, f'sum={s:.6f}', verbose)

    # Consistency vs matrix
    np.random.seed(SIM_SEED)
    sim2 = MFRM_Sim_Bivector(
        no_of_items=N_ITEMS, no_of_persons=N_PERSONS,
        no_of_raters=N_RATERS, max_score=MAX_SCORE,
    )
    m_mat = rp.MFRM(sim2.responses, max_score=MAX_SCORE)
    m_mat.calibrate(model='matrix')
    r_diffs, _ = pearsonr(m.items.values, m_mat.items.values)
    check('[Bivector] item difficulties r vs matrix >= 0.95',
          r_diffs >= 0.95, f'r={r_diffs:.3f}', verbose)


def run_bivector(generate=False, verbose=False):
    section('MFRM — Bivector')
    try:
        sim, m = build_bivector()
    except Exception:
        check('[Bivector] build', False, traceback.format_exc())
        return
    if generate:
        generate_bivector(sim, m)
    else:
        assert_bivector(sim, m, verbose)

# ── Entry point ───────────────────────────────────────────────────────────────

RUNNERS = {
    'slm':      run_slm,
    'pcm':      run_pcm,
    'rsm':      run_rsm,
    'mfrm':     run_mfrm,
    'bivector': run_bivector,
}


def main():
    parser = argparse.ArgumentParser(description='RaschPy regression tests')
    parser.add_argument('--generate', action='store_true',
                        help='Generate known-good fixtures (run once, then commit)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print each passing check')
    parser.add_argument('--model', choices=list(RUNNERS), default=None,
                        help='Run only this model (default: all)')
    args = parser.parse_args()

    to_run = [args.model] if args.model else list(RUNNERS)

    print(f'\n{"=" * 60}')
    if args.generate:
        print(f'  Generating fixtures — {", ".join(r.upper() for r in to_run)}')
        print(f'  Output: {FIXTURE_DIR}/')
    else:
        print(f'  RaschPy regression tests — {", ".join(r.upper() for r in to_run)}')
    print(f'{"=" * 60}')

    os.makedirs(FIXTURE_DIR, exist_ok=True)

    for name in to_run:
        RUNNERS[name](generate=args.generate, verbose=args.verbose)

    if args.generate:
        print(f'\n{"=" * 60}')
        print('  Done. Commit regression_data/ to version control,')
        print('  then run without --generate to assert against fixtures.')
        print(f'{"=" * 60}\n')
        return

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
