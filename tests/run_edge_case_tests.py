"""
run_edge_case_tests.py
----------------------
Unified edge case tests for SLM, PCM, RSM, and MFRM (all four models).

Cases per model:
  1. All-missing person   - calibrate() should complete (warn+drop) or raise cleanly
  2. All-missing item     - same
  3. Extreme scores all-0 - ability via extreme score adjustment should be finite
  4a. Extreme scores all-max, constant=0.1 - additive smoothing active, should pass
  4b. Extreme scores all-max, constant=0   - should warn or raise cleanly
  5. Single item          - should warn or raise cleanly

MFRM additional cases:
  6. Single rater         - should warn and suggest RSM

xfail() is available for future known gaps — flip check() to xfail() if a case is
known to be broken, and back again when fixed.

Run with:
    python run_edge_case_tests.py                # all models
    python run_edge_case_tests.py -v             # verbose
    python run_edge_case_tests.py --model slm    # single model
    python run_edge_case_tests.py --model mfrm -v

Exit code 0 if all checks pass (xfails count as pass), 1 if any unexpected failures.
"""

import sys
import os
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

warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────────

SIM_SEED      = 42
N_PERSONS     = 100
N_ITEMS       = 6
N_RATERS      = 4
MAX_SCORE     = 3
MAX_SCORE_VEC = [3] * 6
RATER_RANGE   = 1.0

MFRM_MODELS = [
    ('Global',     'global',     MFRM_Sim_Global),
    ('Items',      'items',      MFRM_Sim_Items),
    ('Thresholds', 'thresholds', MFRM_Sim_Thresholds),
    ('Matrix',     'matrix',     MFRM_Sim_Matrix),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS_STR  = '\033[32mPASS\033[0m'
FAIL_STR  = '\033[31mFAIL\033[0m'
XFAIL_STR = '\033[33mXFAIL\033[0m'

results = []   # (name, status, msg)  status in {'pass', 'fail', 'xfail'}


def check(name, expr, msg='', verbose=False):
    if expr:
        results.append((name, 'pass', ''))
        if verbose:
            print(f'  {PASS_STR}   {name}')
    else:
        results.append((name, 'fail', msg))
        short = msg.strip().splitlines()[-1] if msg else ''
        print(f'  {FAIL_STR}   {name}' + (f' — {short}' if short else ''))


def xfail(name, reason, verbose=False):
    """Record an expected failure — counts as passing."""
    results.append((name, 'xfail', reason))
    if verbose:
        print(f'  {XFAIL_STR}  {name}')
        print(f'           {reason}')
    else:
        print(f'  {XFAIL_STR}  {name}')


def section(title):
    print(f'\n{"─" * 60}')
    print(f'  {title}')
    print(f'{"─" * 60}')


def calibrate_ok(m, **kwargs):
    try:
        m.calibrate(**kwargs)
        return True, None
    except Exception as e:
        return False, e


def is_clean_exception(e):
    return isinstance(e, (ValueError, TypeError, ArithmeticError,
                          NotImplementedError))


def outputs_finite(m, attr):
    try:
        m.item_stats_df() if not hasattr(m, 'items') else None
        df = getattr(m, attr, None)
        if not isinstance(df, pd.DataFrame):
            return False, f'{attr} not a DataFrame'
        num = df.select_dtypes(include='number')
        if num.isnull().all(axis=None):
            return False, f'all-NaN {attr}'
        return True, None
    except Exception as e:
        return False, str(e)


def outputs_finite_mfrm(m, model_name):
    try:
        m.item_stats_df(model=model_name)
        df = getattr(m, f'item_stats_{model_name}', None)
        if not isinstance(df, pd.DataFrame):
            return False, 'item_stats_df not a DataFrame'
        num = df.select_dtypes(include='number')
        if num.isnull().all(axis=None):
            return False, 'all-NaN item_stats'
        return True, None
    except Exception as e:
        return False, str(e)


def caught_any_warning(w):
    return len(w) > 0


def caught_rsm_warning(w):
    return any(
        'rater' in str(x.message).lower() or
        'rsm'   in str(x.message).lower() or
        'item'  in str(x.message).lower()
        for x in w
    )


# ── SLM ───────────────────────────────────────────────────────────────────────

def run_slm(verbose=False):

    def build():
        np.random.seed(SIM_SEED)
        return rp.SLM(SLM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS).responses)

    # Case 1 — all-missing person
    section('SLM — Case 1: all-missing person')
    try:
        data = build().responses.copy()
        data.iloc[0, :] = np.nan
        m = rp.SLM(data)
        ok, exc = calibrate_ok(m)
        if ok:
            m.item_stats_df()
            ok2, msg2 = outputs_finite(m, 'item_stats')
            check('SLM all-missing person — completes, outputs finite', ok2, msg2, verbose)
        else:
            check('SLM all-missing person — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check('SLM all-missing person — no unhandled crash', False,
              traceback.format_exc(), verbose)

    # Case 2 — all-missing item
    section('SLM — Case 2: all-missing item')
    try:
        data = build().responses.astype(float).copy()
        data.iloc[:, 0] = np.nan
        m = rp.SLM(data)
        ok, exc = calibrate_ok(m)
        if ok:
            m.item_stats_df()
            ok2, msg2 = outputs_finite(m, 'item_stats')
            check('SLM all-missing item — completes, outputs finite', ok2, msg2, verbose)
        else:
            check('SLM all-missing item — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check('SLM all-missing item — no unhandled crash', False,
              traceback.format_exc(), verbose)

    # Case 3 — extreme scores all-0
    section('SLM — Case 3: extreme scores all-0')
    try:
        np.random.seed(SIM_SEED)
        sim  = SLM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS)
        data = sim.responses.copy()
        data.iloc[0, :] = 0
        m = rp.SLM(data)
        ok, exc = calibrate_ok(m)
        if ok:
            m.person_estimates()
            abils  = m.persons
            person = data.index[0]
            val    = abils.loc[person] if abils is not None else None
            check('SLM extreme all-0 — ability finite',
                  val is not None and np.isfinite(val), f'got {val}', verbose)
        else:
            check('SLM extreme all-0 — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check('SLM extreme all-0 — no unhandled crash', False,
              traceback.format_exc(), verbose)

    # Case 4a — all-max, constant=0.1
    section('SLM — Case 4a: extreme scores all-max, constant=0.1')
    try:
        np.random.seed(SIM_SEED)
        sim  = SLM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS)
        data = sim.responses.copy()
        data.iloc[0, :] = 1   # SLM max score is 1
        m = rp.SLM(data)
        ok, exc = calibrate_ok(m, constant=0.1)
        if ok:
            m.item_stats_df()
            ok2, msg2 = outputs_finite(m, 'item_stats')
            check('SLM all-max constant=0.1 — outputs finite', ok2, msg2, verbose)
        else:
            check('SLM all-max constant=0.1 — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check('SLM all-max constant=0.1 — no unhandled crash', False,
              traceback.format_exc(), verbose)

    # Case 4b — all-max, constant=0
    section('SLM — Case 4b: extreme scores all-max, constant=0')
    try:
        np.random.seed(SIM_SEED)
        sim  = SLM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS)
        data = sim.responses.copy()
        data.iloc[:, :] = 1
        m = rp.SLM(data)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            ok, exc = calibrate_ok(m, constant=0)
        if ok:
            if caught_any_warning(w):
                check('SLM all-max constant=0 — warns', True, verbose=verbose)
            else:
                xfail('SLM all-max constant=0 — warns',
                      'KNOWN GAP: calibrate() completes silently with constant=0 '
                      'and all-max data. Should warn to drop item or use non-zero constant.',
                      verbose)
        else:
            check('SLM all-max constant=0 — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check('SLM all-max constant=0 — no unhandled crash', False,
              traceback.format_exc(), verbose)

    # Case 5 — single item
    section('SLM — Case 5: single item')
    try:
        np.random.seed(SIM_SEED)
        data = SLM_Sim(no_of_items=1, no_of_persons=N_PERSONS).responses
        m = rp.SLM(data)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            ok, exc = calibrate_ok(m)
        if ok:
            if caught_rsm_warning(w):
                check('SLM single item — warns', True, verbose=verbose)
            else:
                xfail('SLM single item — warns',
                      'KNOWN GAP: calibrate() does not warn when n_items=1.', verbose)
            check('SLM single item — no unhandled crash', True, verbose=verbose)
        else:
            check('SLM single item — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check('SLM single item — no unhandled crash', False,
              traceback.format_exc(), verbose)


# ── PCM ───────────────────────────────────────────────────────────────────────

def run_pcm(verbose=False):

    def build():
        np.random.seed(SIM_SEED)
        sim = PCM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS,
                      max_score_vector=MAX_SCORE_VEC)
        return rp.PCM(sim.responses, max_score_vector=MAX_SCORE_VEC)

    # Case 1 — all-missing person
    section('PCM — Case 1: all-missing person')
    try:
        np.random.seed(SIM_SEED)
        sim  = PCM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS,
                       max_score_vector=MAX_SCORE_VEC)
        data = sim.responses.copy()
        data.iloc[0, :] = np.nan
        m = rp.PCM(data, max_score_vector=MAX_SCORE_VEC)
        ok, exc = calibrate_ok(m)
        if ok:
            m.item_stats_df()
            ok2, msg2 = outputs_finite(m, 'item_stats')
            check('PCM all-missing person — completes, outputs finite', ok2, msg2, verbose)
        else:
            check('PCM all-missing person — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check('PCM all-missing person — no unhandled crash', False,
              traceback.format_exc(), verbose)

    # Case 2 — all-missing item
    section('PCM — Case 2: all-missing item')
    try:
        np.random.seed(SIM_SEED)
        sim  = PCM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS,
                       max_score_vector=MAX_SCORE_VEC)
        data = sim.responses.astype(float).copy()
        data.iloc[:, 0] = np.nan
        m = rp.PCM(data, max_score_vector=MAX_SCORE_VEC)
        ok, exc = calibrate_ok(m)
        if ok:
            m.item_stats_df()
            ok2, msg2 = outputs_finite(m, 'item_stats')
            check('PCM all-missing item — completes, outputs finite', ok2, msg2, verbose)
        else:
            check('PCM all-missing item — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check('PCM all-missing item — no unhandled crash', False,
              traceback.format_exc(), verbose)

    # Case 3 — extreme scores all-0
    section('PCM — Case 3: extreme scores all-0')
    try:
        np.random.seed(SIM_SEED)
        sim  = PCM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS,
                       max_score_vector=MAX_SCORE_VEC)
        data = sim.responses.copy()
        data.iloc[0, :] = 0
        m = rp.PCM(data, max_score_vector=MAX_SCORE_VEC)
        ok, exc = calibrate_ok(m)
        if ok:
            m.person_estimates()
            abils  = m.persons
            person = data.index[0]
            val    = abils.loc[person] if abils is not None else None
            check('PCM extreme all-0 — ability finite',
                  val is not None and np.isfinite(val), f'got {val}', verbose)
        else:
            check('PCM extreme all-0 — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check('PCM extreme all-0 — no unhandled crash', False,
              traceback.format_exc(), verbose)

    # Case 4a — all-max, constant=0.1
    section('PCM — Case 4a: extreme scores all-max, constant=0.1')
    try:
        np.random.seed(SIM_SEED)
        sim  = PCM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS,
                       max_score_vector=MAX_SCORE_VEC)
        data = sim.responses.copy()
        data.iloc[0, :] = MAX_SCORE
        m = rp.PCM(data, max_score_vector=MAX_SCORE_VEC)
        ok, exc = calibrate_ok(m, constant=0.1)
        if ok:
            m.item_stats_df()
            ok2, msg2 = outputs_finite(m, 'item_stats')
            check('PCM all-max constant=0.1 — outputs finite', ok2, msg2, verbose)
        else:
            check('PCM all-max constant=0.1 — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check('PCM all-max constant=0.1 — no unhandled crash', False,
              traceback.format_exc(), verbose)

    # Case 4b — all-max, constant=0
    section('PCM — Case 4b: extreme scores all-max, constant=0')
    try:
        np.random.seed(SIM_SEED)
        sim  = PCM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS,
                       max_score_vector=MAX_SCORE_VEC)
        data = sim.responses.copy()
        data.iloc[:, :] = MAX_SCORE
        m = rp.PCM(data, max_score_vector=MAX_SCORE_VEC)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            ok, exc = calibrate_ok(m, constant=0)
        if ok:
            if caught_any_warning(w):
                check('PCM all-max constant=0 — warns', True, verbose=verbose)
            else:
                xfail('PCM all-max constant=0 — warns',
                      'KNOWN GAP: calibrate() completes silently with constant=0 '
                      'and all-max data. Should warn to drop item or use non-zero constant.',
                      verbose)
        else:
            check('PCM all-max constant=0 — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check('PCM all-max constant=0 — no unhandled crash', False,
              traceback.format_exc(), verbose)

    # Case 5 — single item
    section('PCM — Case 5: single item')
    try:
        np.random.seed(SIM_SEED)
        sim  = PCM_Sim(no_of_items=1, no_of_persons=N_PERSONS,
                       max_score_vector=[MAX_SCORE])
        data = sim.responses
        m = rp.PCM(data, max_score_vector=[MAX_SCORE])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            ok, exc = calibrate_ok(m)
        if ok:
            if caught_rsm_warning(w):
                check('PCM single item — warns', True, verbose=verbose)
            else:
                xfail('PCM single item — warns',
                      'KNOWN GAP: calibrate() does not warn when n_items=1.', verbose)
            check('PCM single item — no unhandled crash', True, verbose=verbose)
        else:
            check('PCM single item — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check('PCM single item — no unhandled crash', False,
              traceback.format_exc(), verbose)


# ── RSM ───────────────────────────────────────────────────────────────────────

def run_rsm(verbose=False):

    # Case 1 — all-missing person
    section('RSM — Case 1: all-missing person')
    try:
        np.random.seed(SIM_SEED)
        sim  = RSM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS,
                       max_score=MAX_SCORE)
        data = sim.responses.copy()
        data.iloc[0, :] = np.nan
        m = rp.RSM(data, max_score=MAX_SCORE)
        ok, exc = calibrate_ok(m)
        if ok:
            m.item_stats_df()
            ok2, msg2 = outputs_finite(m, 'item_stats')
            check('RSM all-missing person — completes, outputs finite', ok2, msg2, verbose)
        else:
            check('RSM all-missing person — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check('RSM all-missing person — no unhandled crash', False,
              traceback.format_exc(), verbose)

    # Case 2 — all-missing item
    section('RSM — Case 2: all-missing item')
    try:
        np.random.seed(SIM_SEED)
        sim  = RSM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS,
                       max_score=MAX_SCORE)
        data = sim.responses.astype(float).copy()
        data.iloc[:, 0] = np.nan
        m = rp.RSM(data, max_score=MAX_SCORE)
        ok, exc = calibrate_ok(m)
        if ok:
            m.item_stats_df()
            ok2, msg2 = outputs_finite(m, 'item_stats')
            check('RSM all-missing item — completes, outputs finite', ok2, msg2, verbose)
        else:
            check('RSM all-missing item — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check('RSM all-missing item — no unhandled crash', False,
              traceback.format_exc(), verbose)

    # Case 3 — extreme scores all-0
    section('RSM — Case 3: extreme scores all-0')
    try:
        np.random.seed(SIM_SEED)
        sim  = RSM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS,
                       max_score=MAX_SCORE)
        data = sim.responses.copy()
        data.iloc[0, :] = 0
        m = rp.RSM(data, max_score=MAX_SCORE)
        ok, exc = calibrate_ok(m)
        if ok:
            m.person_estimates()
            abils  = m.persons
            person = data.index[0]
            val    = abils.loc[person] if abils is not None else None
            check('RSM extreme all-0 — ability finite',
                  val is not None and np.isfinite(val), f'got {val}', verbose)
        else:
            check('RSM extreme all-0 — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check('RSM extreme all-0 — no unhandled crash', False,
              traceback.format_exc(), verbose)

    # Case 4a — all-max, constant=0.1
    section('RSM — Case 4a: extreme scores all-max, constant=0.1')
    try:
        np.random.seed(SIM_SEED)
        sim  = RSM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS,
                       max_score=MAX_SCORE)
        data = sim.responses.copy()
        data.iloc[0, :] = MAX_SCORE
        m = rp.RSM(data, max_score=MAX_SCORE)
        ok, exc = calibrate_ok(m, constant=0.1)
        if ok:
            m.item_stats_df()
            ok2, msg2 = outputs_finite(m, 'item_stats')
            check('RSM all-max constant=0.1 — outputs finite', ok2, msg2, verbose)
        else:
            check('RSM all-max constant=0.1 — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check('RSM all-max constant=0.1 — no unhandled crash', False,
              traceback.format_exc(), verbose)

    # Case 4b — all-max, constant=0
    section('RSM — Case 4b: extreme scores all-max, constant=0')
    try:
        np.random.seed(SIM_SEED)
        sim  = RSM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS,
                       max_score=MAX_SCORE)
        data = sim.responses.copy()
        data.iloc[:, :] = MAX_SCORE
        m = rp.RSM(data, max_score=MAX_SCORE)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            ok, exc = calibrate_ok(m, constant=0)
        if ok:
            if caught_any_warning(w):
                check('RSM all-max constant=0 — warns', True, verbose=verbose)
            else:
                xfail('RSM all-max constant=0 — warns',
                      'KNOWN GAP: calibrate() completes silently with constant=0 '
                      'and all-max data. Should warn to drop item or use non-zero constant.',
                      verbose)
        else:
            check('RSM all-max constant=0 — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check('RSM all-max constant=0 — no unhandled crash', False,
              traceback.format_exc(), verbose)

    # Case 5 — single item
    section('RSM — Case 5: single item')
    try:
        np.random.seed(SIM_SEED)
        sim  = RSM_Sim(no_of_items=1, no_of_persons=N_PERSONS, max_score=MAX_SCORE)
        data = sim.responses
        m    = rp.RSM(data, max_score=MAX_SCORE)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            ok, exc = calibrate_ok(m)
        if ok:
            if caught_rsm_warning(w):
                check('RSM single item — warns', True, verbose=verbose)
            else:
                xfail('RSM single item — warns',
                      'KNOWN GAP: calibrate() does not warn when n_items=1.', verbose)
            check('RSM single item — no unhandled crash', True, verbose=verbose)
        else:
            check('RSM single item — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check('RSM single item — no unhandled crash', False,
              traceback.format_exc(), verbose)


# ── MFRM ─────────────────────────────────────────────────────────────────────

def _mfrm_null_person(data, person_id):
    """Null all rater rows for a given person in a (Rater, Person) MultiIndex df."""
    mask = data.index.get_level_values(-1) == person_id
    data = data.astype(float).copy()
    data.iloc[mask, :] = np.nan
    return data


def run_mfrm_model(label, model_name, sim_cls, verbose=False):

    def build(n_items=N_ITEMS, n_raters=N_RATERS):
        np.random.seed(SIM_SEED)
        sim = sim_cls(no_of_items=n_items, no_of_persons=N_PERSONS,
                      no_of_raters=n_raters, max_score=MAX_SCORE,
                      facet_range=RATER_RANGE)
        return sim.responses

    tag = f'MFRM/{label}'

    # Case 1 — all-missing person (all raters for that person)
    section(f'{tag} — Case 1: all-missing person')
    try:
        data = build()
        person_id = data.index.get_level_values(-1).unique()[0]
        data = _mfrm_null_person(data, person_id)
        m = rp.MFRM(data, max_score=MAX_SCORE)
        ok, exc = calibrate_ok(m, model=model_name)
        if ok:
            ok2, msg2 = outputs_finite_mfrm(m, model_name)
            check(f'{tag} all-missing person — completes, outputs finite',
                  ok2, msg2, verbose)
        else:
            check(f'{tag} all-missing person — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check(f'{tag} all-missing person — no unhandled crash', False,
              traceback.format_exc(), verbose)

    # Case 2 — all-missing item (across all raters)
    section(f'{tag} — Case 2: all-missing item')
    try:
        data = build().astype(float)
        first_item = data.columns.get_level_values(-1).unique()[0]
        mask = data.columns.get_level_values(-1) == first_item
        data.iloc[:, mask] = np.nan
        m = rp.MFRM(data, max_score=MAX_SCORE)
        ok, exc = calibrate_ok(m, model=model_name)
        if ok:
            ok2, msg2 = outputs_finite_mfrm(m, model_name)
            check(f'{tag} all-missing item — completes, outputs finite',
                  ok2, msg2, verbose)
        else:
            check(f'{tag} all-missing item — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check(f'{tag} all-missing item — no unhandled crash', False,
              traceback.format_exc(), verbose)

    # Case 3 — extreme scores all-0
    section(f'{tag} — Case 3: extreme scores all-0')
    try:
        data = build()
        person_id = data.index.get_level_values(-1).unique()[0]
        mask = data.index.get_level_values(-1) == person_id
        data = data.astype(float)
        data.iloc[mask, :] = 0
        m = rp.MFRM(data, max_score=MAX_SCORE)
        ok, exc = calibrate_ok(m, model=model_name)
        if ok:
            m.person_estimates(model=model_name)
            abils = getattr(m, f'persons_{model_name}', None)
            val   = abils.loc[person_id] if abils is not None else None
            check(f'{tag} extreme all-0 — ability finite',
                  val is not None and np.isfinite(val), f'got {val}', verbose)
        else:
            check(f'{tag} extreme all-0 — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check(f'{tag} extreme all-0 — no unhandled crash', False,
              traceback.format_exc(), verbose)

    # Case 4a — all-max, constant=0.1
    section(f'{tag} — Case 4a: extreme scores all-max, constant=0.1')
    try:
        data = build()
        person_id = data.index.get_level_values(-1).unique()[0]
        mask = data.index.get_level_values(-1) == person_id
        data = data.astype(float)
        data.iloc[mask, :] = MAX_SCORE
        m = rp.MFRM(data, max_score=MAX_SCORE)
        ok, exc = calibrate_ok(m, model=model_name, constant=0.1)
        if ok:
            ok2, msg2 = outputs_finite_mfrm(m, model_name)
            check(f'{tag} all-max constant=0.1 — outputs finite', ok2, msg2, verbose)
        else:
            check(f'{tag} all-max constant=0.1 — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check(f'{tag} all-max constant=0.1 — no unhandled crash', False,
              traceback.format_exc(), verbose)

    # Case 4b — all-max, constant=0
    section(f'{tag} — Case 4b: extreme scores all-max, constant=0')
    try:
        data = build().astype(float)
        data.iloc[:, :] = MAX_SCORE
        m = rp.MFRM(data, max_score=MAX_SCORE)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            ok, exc = calibrate_ok(m, model=model_name, constant=0)
        if ok:
            if caught_any_warning(w):
                check(f'{tag} all-max constant=0 — warns', True, verbose=verbose)
            else:
                xfail(f'{tag} all-max constant=0 — warns',
                      'KNOWN GAP: calibrate() completes silently with constant=0 '
                      'and all-max data. Should warn to drop item or use non-zero constant.',
                      verbose)
        else:
            check(f'{tag} all-max constant=0 — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check(f'{tag} all-max constant=0 — no unhandled crash', False,
              traceback.format_exc(), verbose)

    # Case 5 — single item
    section(f'{tag} — Case 5: single item')
    try:
        data = build(n_items=1)
        m = rp.MFRM(data, max_score=MAX_SCORE)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            ok, exc = calibrate_ok(m, model=model_name)
        if ok:
            if caught_rsm_warning(w):
                check(f'{tag} single item — warns', True, verbose=verbose)
            else:
                xfail(f'{tag} single item — warns',
                      'KNOWN GAP: calibrate() does not warn when n_items=1.', verbose)
            check(f'{tag} single item — no unhandled crash', True, verbose=verbose)
        else:
            check(f'{tag} single item — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check(f'{tag} single item — no unhandled crash', False,
              traceback.format_exc(), verbose)

    # Case 6 — single rater (MFRM only)
    section(f'{tag} — Case 6: single rater')
    try:
        data = build(n_raters=1)
        m = rp.MFRM(data, max_score=MAX_SCORE)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            ok, exc = calibrate_ok(m, model=model_name)
        if ok:
            if caught_rsm_warning(w):
                check(f'{tag} single rater — warns', True, verbose=verbose)
            else:
                xfail(f'{tag} single rater — warns',
                      'KNOWN GAP: calibrate() does not warn when n_raters=1. '
                      'Should suggest RSM as the appropriate model.', verbose)
            check(f'{tag} single rater — no unhandled crash', True, verbose=verbose)
        else:
            check(f'{tag} single rater — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check(f'{tag} single rater — no unhandled crash', False,
              traceback.format_exc(), verbose)


def run_mfrm(verbose=False):
    for label, model_name, sim_cls in MFRM_MODELS:
        run_mfrm_model(label, model_name, sim_cls, verbose=verbose)


def run_bivector(verbose=False):

    def build(n_items=N_ITEMS, n_raters=N_RATERS):
        np.random.seed(SIM_SEED)
        sim = MFRM_Sim_Bivector(no_of_items=n_items, no_of_persons=N_PERSONS,
                                no_of_raters=n_raters, max_score=MAX_SCORE)
        return sim.responses

    tag = 'MFRM/Bivector'

    # Case 1 — all-missing person
    section(f'{tag} — Case 1: all-missing person')
    try:
        data = build()
        person_id = data.index.get_level_values(-1).unique()[0]
        data = _mfrm_null_person(data, person_id)
        m = rp.MFRM(data, max_score=MAX_SCORE)
        ok, exc = calibrate_ok(m, model='bivector')
        if ok:
            ok2, msg2 = outputs_finite_mfrm(m, 'bivector')
            check(f'{tag} all-missing person — completes, outputs finite', ok2, msg2, verbose)
        else:
            check(f'{tag} all-missing person — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check(f'{tag} all-missing person — no unhandled crash', False,
              traceback.format_exc(), verbose)

    # Case 2 — all-missing item
    section(f'{tag} — Case 2: all-missing item')
    try:
        data = build().astype(float)
        first_item = data.columns.get_level_values(-1).unique()[0]
        mask = data.columns.get_level_values(-1) == first_item
        data.iloc[:, mask] = np.nan
        m = rp.MFRM(data, max_score=MAX_SCORE)
        ok, exc = calibrate_ok(m, model='bivector')
        if ok:
            ok2, msg2 = outputs_finite_mfrm(m, 'bivector')
            check(f'{tag} all-missing item — completes, outputs finite', ok2, msg2, verbose)
        else:
            check(f'{tag} all-missing item — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check(f'{tag} all-missing item — no unhandled crash', False,
              traceback.format_exc(), verbose)

    # Case 3 — extreme scores all-0
    section(f'{tag} — Case 3: extreme scores all-0')
    try:
        data = build()
        person_id = data.index.get_level_values(-1).unique()[0]
        mask = data.index.get_level_values(-1) == person_id
        data = data.astype(float)
        data.iloc[mask, :] = 0
        m = rp.MFRM(data, max_score=MAX_SCORE)
        ok, exc = calibrate_ok(m, model='bivector')
        if ok:
            m.person_estimates(model='bivector')
            abils = getattr(m, 'persons_bivector', None)
            val   = abils.loc[person_id] if abils is not None else None
            check(f'{tag} extreme all-0 — ability finite',
                  val is not None and np.isfinite(val), f'got {val}', verbose)
        else:
            check(f'{tag} extreme all-0 — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check(f'{tag} extreme all-0 — no unhandled crash', False,
              traceback.format_exc(), verbose)

    # Case 4a — all-max, constant=0.1
    section(f'{tag} — Case 4a: extreme scores all-max, constant=0.1')
    try:
        data = build()
        person_id = data.index.get_level_values(-1).unique()[0]
        mask = data.index.get_level_values(-1) == person_id
        data = data.astype(float)
        data.iloc[mask, :] = MAX_SCORE
        m = rp.MFRM(data, max_score=MAX_SCORE)
        ok, exc = calibrate_ok(m, model='bivector', constant=0.1)
        if ok:
            ok2, msg2 = outputs_finite_mfrm(m, 'bivector')
            check(f'{tag} all-max constant=0.1 — outputs finite', ok2, msg2, verbose)
        else:
            check(f'{tag} all-max constant=0.1 — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check(f'{tag} all-max constant=0.1 — no unhandled crash', False,
              traceback.format_exc(), verbose)

    # Case 4b — all-max, constant=0
    section(f'{tag} — Case 4b: extreme scores all-max, constant=0')
    try:
        data = build().astype(float)
        data.iloc[:, :] = MAX_SCORE
        m = rp.MFRM(data, max_score=MAX_SCORE)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            ok, exc = calibrate_ok(m, model='bivector', constant=0)
        if ok:
            if caught_any_warning(w):
                check(f'{tag} all-max constant=0 — warns', True, verbose=verbose)
            else:
                xfail(f'{tag} all-max constant=0 — warns',
                      'KNOWN GAP: calibrate() completes silently with constant=0 '
                      'and all-max data.', verbose)
        else:
            check(f'{tag} all-max constant=0 — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check(f'{tag} all-max constant=0 — no unhandled crash', False,
              traceback.format_exc(), verbose)

    # Case 5 — single item
    section(f'{tag} — Case 5: single item')
    try:
        data = build(n_items=1)
        m = rp.MFRM(data, max_score=MAX_SCORE)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            ok, exc = calibrate_ok(m, model='bivector')
        if ok:
            if caught_rsm_warning(w):
                check(f'{tag} single item — warns', True, verbose=verbose)
            else:
                xfail(f'{tag} single item — warns',
                      'KNOWN GAP: calibrate() does not warn when n_items=1.', verbose)
            check(f'{tag} single item — no unhandled crash', True, verbose=verbose)
        else:
            check(f'{tag} single item — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check(f'{tag} single item — no unhandled crash', False,
              traceback.format_exc(), verbose)

    # Case 6 — single rater
    section(f'{tag} — Case 6: single rater')
    try:
        data = build(n_raters=1)
        m = rp.MFRM(data, max_score=MAX_SCORE)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            ok, exc = calibrate_ok(m, model='bivector')
        if ok:
            if caught_rsm_warning(w):
                check(f'{tag} single rater — warns', True, verbose=verbose)
            else:
                xfail(f'{tag} single rater — warns',
                      'KNOWN GAP: calibrate() does not warn when n_raters=1.', verbose)
            check(f'{tag} single rater — no unhandled crash', True, verbose=verbose)
        else:
            check(f'{tag} single rater — clean exception',
                  is_clean_exception(exc), f'{type(exc).__name__}: {exc}', verbose)
    except Exception:
        check(f'{tag} single rater — no unhandled crash', False,
              traceback.format_exc(), verbose)

    # Case 7 — anchoring: antisymmetry and person shift
    section(f'{tag} — Case 7: anchoring')
    try:
        np.random.seed(SIM_SEED)
        sim = MFRM_Sim_Bivector(no_of_items=N_ITEMS, no_of_persons=N_PERSONS,
                                no_of_raters=N_RATERS, max_score=MAX_SCORE)
        m_free = rp.MFRM(sim.responses, max_score=MAX_SCORE)
        m_free.calibrate(model='bivector')
        m_free.person_estimates(model='bivector')

        anchor_raters = list(m_free.rater_names[:N_RATERS // 2])

        m_anc = rp.MFRM(sim.responses, max_score=MAX_SCORE)
        m_anc.calibrate(model='bivector')
        m_anc.calibrate_anchor(model='bivector', anchors=anchor_raters)
        m_anc.person_estimates(model='bivector', anchor=True)

        # Anchored and free estimates should differ
        common = m_free.persons_bivector.index.intersection(
            m_anc.anchor_persons_bivector.index)
        shift = (m_anc.anchor_persons_bivector.loc[common]
                 - m_free.persons_bivector.loc[common]).abs().mean()
        check(f'{tag} anchoring — person abilities shift (mean |shift| > 0)',
              shift > 0, f'mean |shift|={shift:.4f}', verbose)

        # anchor_items_bivector attribute exists
        check(f'{tag} anchoring — anchor_items_bivector set',
              hasattr(m_anc, 'anchor_items_bivector'), '', verbose)

    except Exception:
        check(f'{tag} anchoring — no unhandled crash', False,
              traceback.format_exc(), verbose)


# ── Entry point ───────────────────────────────────────────────────────────────

RUNNERS = {
    'slm':      run_slm,
    'pcm':      run_pcm,
    'rsm':      run_rsm,
    'mfrm':     run_mfrm,
    'bivector': run_bivector,
}


def main():
    parser = argparse.ArgumentParser(description='RaschPy edge case tests')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print each passing check')
    parser.add_argument('--model', choices=list(RUNNERS), default=None,
                        help='Run only this model (default: all)')
    args = parser.parse_args()

    to_run = [args.model] if args.model else list(RUNNERS)

    print(f'\n{"=" * 60}')
    print(f'  RaschPy edge case tests — {", ".join(r.upper() for r in to_run)}')
    print(f'{"=" * 60}')

    for name in to_run:
        RUNNERS[name](verbose=args.verbose)

    # Summary
    total   = len(results)
    passed  = sum(1 for _, s, _ in results if s == 'pass')
    xfailed = sum(1 for _, s, _ in results if s == 'xfail')
    failed  = sum(1 for _, s, _ in results if s == 'fail')

    print(f'\n{"=" * 60}')
    print(f'  {passed} passed, {xfailed} xfail, {failed} failed  ({total} total)')

    if failed:
        print('\n  Unexpected failures:')
        for name, status, msg in results:
            if status == 'fail':
                short = msg.strip().splitlines()[-1] if msg else ''
                print(f'    ✗  {name}' + (f'  [{short}]' if short else ''))
                if args.verbose and msg:
                    for line in msg.strip().splitlines():
                        print(f'         {line}')

    if xfailed:
        print('\n  Expected failures (flip to check() when fixed):')
        for name, status, msg in results:
            if status == 'xfail':
                print(f'    -  {name}')
                if args.verbose and msg:
                    print(f'       {msg}')

    print(f'{"=" * 60}\n')
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
