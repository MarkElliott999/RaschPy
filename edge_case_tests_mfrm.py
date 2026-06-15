"""
edge_case_tests_mfrm.py
-----------------------
Edge case tests for all four MFRM models.

Cases covered:
  1. All-missing person   - calibrate() should either complete gracefully (warn+drop)
                            or raise a clean, recognisable exception.
                            Currently expected to fail - regression point for when
                            cleaning is added to calibrate().
  2. All-missing item     - same as above.
  3. Extreme scores all-0 - ability estimated via extreme score adjustment.
                            KNOWN BUG: triggers IndexingError in class_intervals
                            (single-string item .loc indexing). Tracked as todo.
  4. Extreme scores all-max, constant=0.1 - additive smoothing active, should pass.
  4b. Extreme scores all-max, constant=0  - should warn/raise cleanly.
                            KNOWN GAP: currently completes silently. Tracked as todo.
  5. Single rater         - should warn and suggest RSM.
                            KNOWN GAP: no warning issued. Tracked as todo.
  6. Single item          - should warn and suggest RSM with raters-as-items.
                            KNOWN GAP: no warning issued. Tracked as todo.

Known failures are recorded as XFAIL (expected failure) - they PASS the test suite
but are clearly labelled. When the underlying gaps are fixed, flip them to normal checks.

Run with:
    python edge_case_tests_mfrm.py
    python edge_case_tests_mfrm.py -v     # verbose

Exit code 0 if all checks pass, 1 otherwise.
"""

import sys
import traceback
import argparse
import warnings
import numpy as np
import pandas as pd

from raschpy import MFRM
from raschpy.simulation import (
    MFRM_Sim_Global,
    MFRM_Sim_Items,
    MFRM_Sim_Thresholds,
    MFRM_Sim_Matrix,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIM_SEED    = 42
N_PERSONS   = 100
N_ITEMS     = 6
N_RATERS    = 4
MAX_SCORE   = 3
RATER_RANGE = 1.0

MODELS = [
    ("Global",     "global",     MFRM_Sim_Global),
    ("Items",      "items",      MFRM_Sim_Items),
    ("Thresholds", "thresholds", MFRM_Sim_Thresholds),
    ("Matrix",     "matrix",     MFRM_Sim_Matrix),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS_STR  = "\033[32mPASS\033[0m"
FAIL_STR  = "\033[31mFAIL\033[0m"
XFAIL_STR = "\033[33mXFAIL\033[0m"
results   = []


def check(name, expr, msg="", verbose=False):
    if expr:
        results.append((name, "pass", ""))
        if verbose:
            print(f"  {PASS_STR}   {name}")
    else:
        results.append((name, "fail", msg))
        short = msg.splitlines()[-1] if msg else ""
        print(f"  {FAIL_STR}   {name}" + (f" - {short}" if short else ""))


def xfail(name, reason, actual_exc=None, verbose=False):
    """Record an expected failure - counts as passing."""
    detail = f"{reason}"
    if actual_exc:
        detail += f" (got {type(actual_exc).__name__}: {actual_exc})"
    results.append((name, "xfail", detail))
    if verbose:
        print(f"  {XFAIL_STR}  {name} - {detail}")
    else:
        print(f"  {XFAIL_STR}  {name}")


def simulate(sim_cls, n_persons=N_PERSONS, n_items=N_ITEMS,
             n_raters=N_RATERS, max_score=MAX_SCORE):
    np.random.seed(SIM_SEED)
    sim = sim_cls(
        no_of_items=n_items,
        no_of_persons=n_persons,
        no_of_raters=n_raters,
        max_score=max_score,
        rater_range=RATER_RANGE,
    )
    return sim.scores


def calibrate_ok(m, model_name, **kwargs):
    try:
        m.calibrate(model=model_name, **kwargs)
        return True, None
    except Exception as e:
        return False, e


def is_clean_exception(e):
    return isinstance(e, (ValueError, TypeError, ArithmeticError,
                          NotImplementedError))


def outputs_finite(m, model_name):
    try:
        m.item_stats_df(model=model_name)
        df = getattr(m, f"item_stats_{model_name}", None)
        if not isinstance(df, pd.DataFrame):
            return False, "item_stats_df not a DataFrame"
        num = df.select_dtypes(include="number")
        if num.isnull().all(axis=None):
            return False, "all-NaN item_stats"
        return True, None
    except Exception as e:
        return False, str(e)


def section(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ---------------------------------------------------------------------------
# Case 1 & 2: all-missing person / item
# ---------------------------------------------------------------------------

def run_all_missing(label, model_name, sim_cls, verbose):
    section(f"Case 1 - all-missing person  [{label}]")
    try:
        data = simulate(sim_cls)
        data.iloc[0, :] = np.nan
        m = MFRM(data)
        ok, exc = calibrate_ok(m, model_name)
        if ok:
            finite_ok, finite_msg = outputs_finite(m, model_name)
            check(f"[{label}] all-missing person - completes and outputs finite",
                  finite_ok, finite_msg or "", verbose)
        else:
            check(f"[{label}] all-missing person - clean exception (not cryptic crash)",
                  is_clean_exception(exc),
                  f"got {type(exc).__name__}: {exc}", verbose)
    except Exception:
        check(f"[{label}] all-missing person - no unhandled crash",
              False, traceback.format_exc(), verbose)

    section(f"Case 2 - all-missing item  [{label}]")
    try:
        data = simulate(sim_cls)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            first_item = data.columns.get_level_values(-1).unique()[0]
            mask = data.columns.get_level_values(-1) == first_item
            data = data.astype(float)
            data.iloc[:, mask] = np.nan
        m = MFRM(data)
        ok, exc = calibrate_ok(m, model_name)
        if ok:
            finite_ok, finite_msg = outputs_finite(m, model_name)
            check(f"[{label}] all-missing item - completes and outputs finite",
                  finite_ok, finite_msg or "", verbose)
        else:
            check(f"[{label}] all-missing item - clean exception (not cryptic crash)",
                  is_clean_exception(exc),
                  f"got {type(exc).__name__}: {exc}", verbose)
    except Exception:
        check(f"[{label}] all-missing item - no unhandled crash",
              False, traceback.format_exc(), verbose)


# ---------------------------------------------------------------------------
# Case 3 & 4: extreme scores
# ---------------------------------------------------------------------------

def run_extreme_scores(label, model_name, sim_cls, verbose):

    # -- all zeros --
    section(f"Case 3 - extreme scores all-0  [{label}]")
    try:
        data = simulate(sim_cls)
        data.iloc[0, :] = 0
        m = MFRM(data)
        ok, exc = calibrate_ok(m, model_name)
        if ok:
            try:
                m.person_abils(model=model_name)
                abils = getattr(m, f"abils_{model_name}", None)
                person = data.index.get_level_values(-1)[0]
                abil_val = abils.loc[person] if abils is not None else None
                check(f"[{label}] extreme all-0 - ability estimate is finite",
                      abil_val is not None and np.isfinite(abil_val),
                      f"got {abil_val}", verbose)
            except Exception as e:
                check(f"[{label}] extreme all-0 - ability estimate is finite",
                      False, traceback.format_exc(), verbose)
        else:
            check(f"[{label}] extreme all-0 - clean exception",
                  is_clean_exception(exc),
                  f"got {type(exc).__name__}: {exc}", verbose)
    except Exception:
        check(f"[{label}] extreme all-0 - no unhandled crash",
              False, traceback.format_exc(), verbose)

    # -- all max, constant=0.1 --
    section(f"Case 4a - extreme scores all-max, constant=0.1  [{label}]")
    try:
        data = simulate(sim_cls)
        data.iloc[0, :] = MAX_SCORE
        m = MFRM(data)
        ok, exc = calibrate_ok(m, model_name, constant=0.1)
        if ok:
            finite_ok, finite_msg = outputs_finite(m, model_name)
            check(f"[{label}] extreme all-max constant=0.1 - outputs finite",
                  finite_ok, finite_msg or "", verbose)
        else:
            check(f"[{label}] extreme all-max constant=0.1 - clean exception",
                  is_clean_exception(exc),
                  f"got {type(exc).__name__}: {exc}", verbose)
    except Exception:
        check(f"[{label}] extreme all-max constant=0.1 - no unhandled crash",
              False, traceback.format_exc(), verbose)

    # -- all max, constant=0 --
    section(f"Case 4b - extreme scores all-max, constant=0  [{label}]")
    try:
        data = simulate(sim_cls)
        data.iloc[:, :] = MAX_SCORE
        m = MFRM(data)
        caught_warning = False
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ok, exc = calibrate_ok(m, model_name, constant=0)
            caught_warning = len(w) > 0

        if ok:
            if caught_warning:
                check(f"[{label}] extreme all-max constant=0 - warns",
                      True, verbose=verbose)
            else:
                # KNOWN GAP: no warning issued yet
                xfail(f"[{label}] extreme all-max constant=0 - warns",
                      "KNOWN GAP: calibrate() completes silently with constant=0 and "
                      "all-max data. Should warn that item needs dropping or "
                      "non-zero constant applied.",
                      verbose=verbose)
        else:
            check(f"[{label}] extreme all-max constant=0 - clean exception",
                  is_clean_exception(exc),
                  f"got {type(exc).__name__}: {exc}", verbose)
    except Exception:
        check(f"[{label}] extreme all-max constant=0 - no unhandled crash",
              False, traceback.format_exc(), verbose)


# ---------------------------------------------------------------------------
# Case 5: single rater
# ---------------------------------------------------------------------------

def run_single_rater(verbose):
    section("Case 5 - single rater  [Global]")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = simulate(MFRM_Sim_Global, n_raters=1)
        m = MFRM(data)
        caught_rsm_warning = False
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ok, exc = calibrate_ok(m, "global")
            caught_rsm_warning = any(
                "rater" in str(x.message).lower() or "rsm" in str(x.message).lower()
                for x in w
            )

        if ok:
            if caught_rsm_warning:
                check("single rater - warns about RSM", True, verbose=verbose)
            else:
                xfail("single rater - warns about RSM",
                      "KNOWN GAP: calibrate() does not warn when n_raters=1. "
                      "Should warn and suggest RSM as the appropriate model.",
                      verbose=verbose)
            check("single rater - no unhandled crash", True, verbose=verbose)
        else:
            check("single rater - clean exception",
                  is_clean_exception(exc),
                  f"got {type(exc).__name__}: {exc}", verbose)
    except Exception:
        check("single rater - no unhandled crash",
              False, traceback.format_exc(), verbose)


# ---------------------------------------------------------------------------
# Case 6: single item
# ---------------------------------------------------------------------------

def run_single_item(verbose):
    section("Case 6 - single item  [Global]")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = simulate(MFRM_Sim_Global, n_items=1)
        m = MFRM(data)
        caught_rsm_warning = False
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ok, exc = calibrate_ok(m, "global")
            caught_rsm_warning = any(
                "item" in str(x.message).lower() or "rsm" in str(x.message).lower()
                for x in w
            )

        if ok:
            if caught_rsm_warning:
                check("single item - warns about RSM", True, verbose=verbose)
            else:
                xfail("single item - warns about RSM",
                      "KNOWN GAP: calibrate() does not warn when n_items=1. "
                      "Should warn and suggest RSM with raters-as-items.",
                      verbose=verbose)
            check("single item - no unhandled crash", True, verbose=verbose)
        else:
            check("single item - clean exception",
                  is_clean_exception(exc),
                  f"got {type(exc).__name__}: {exc}", verbose)
    except Exception:
        check("single item - no unhandled crash",
              False, traceback.format_exc(), verbose)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MFRM edge case tests")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print each passing check")
    args = parser.parse_args()

    for label, model_name, sim_cls in MODELS:
        run_all_missing(label, model_name, sim_cls, args.verbose)
        run_extreme_scores(label, model_name, sim_cls, args.verbose)

    run_single_rater(args.verbose)
    run_single_item(args.verbose)

    print(f"\n{'=' * 60}")
    total   = len(results)
    passed  = sum(1 for _, s, _ in results if s == "pass")
    xfailed = sum(1 for _, s, _ in results if s == "xfail")
    failed  = sum(1 for _, s, _ in results if s == "fail")

    print(f"  {passed} passed, {xfailed} xfail, {failed} failed  ({total} total)")

    if failed:
        print("\n  Failed checks (unexpected):")
        for name, status, msg in results:
            if status == "fail":
                print(f"    - {name}")
                if msg and args.verbose:
                    for line in msg.strip().splitlines():
                        print(f"        {line}")

    if xfailed:
        print("\n  Expected failures (known gaps/bugs - flip to check() when fixed):")
        for name, status, msg in results:
            if status == "xfail":
                print(f"    - {name}")
                if msg:
                    print(f"        {msg}")

    print(f"\n{'=' * 60}\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
