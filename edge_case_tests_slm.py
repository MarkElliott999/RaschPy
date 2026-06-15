"""
edge_case_tests_slm.py
----------------------
Edge case tests for the SLM model.

Cases covered:
  1. All-missing person   - calibrate() should complete (warn+drop) or raise cleanly.
  2. All-missing item     - same.
  3. Extreme scores all-0 - ability estimated via extreme score adjustment.
  4. Extreme scores all-1 - all maximum (dichotomous), constant=0.1 should pass.
  4b. Extreme scores all-1, constant=0 - should warn or raise cleanly.
  5. Single item          - degenerate case, should warn or raise cleanly.

Run with:
    python edge_case_tests_slm.py
    python edge_case_tests_slm.py -v
"""

import sys
import traceback
import argparse
import warnings
import numpy as np
import pandas as pd

from raschpy import SLM
from raschpy.simulation import SLM_Sim

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIM_SEED  = 42
N_PERSONS = 100
N_ITEMS   = 6
MAX_SCORE = 1   # SLM is dichotomous

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
    detail = reason
    if actual_exc:
        detail += f" (got {type(actual_exc).__name__}: {actual_exc})"
    results.append((name, "xfail", detail))
    if verbose:
        print(f"  {XFAIL_STR}  {name} - {detail}")
    else:
        print(f"  {XFAIL_STR}  {name}")


def simulate():
    np.random.seed(SIM_SEED)
    sim = SLM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS)
    return sim.scores


def calibrate_ok(m, **kwargs):
    try:
        m.calibrate(**kwargs)
        return True, None
    except Exception as e:
        return False, e


def is_clean_exception(e):
    return isinstance(e, (ValueError, TypeError, ArithmeticError,
                          NotImplementedError))


def outputs_finite(m):
    try:
        m.item_stats_df()
        df = getattr(m, "item_stats", None)
        if not isinstance(df, pd.DataFrame):
            return False, "item_stats not a DataFrame"
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
# Cases
# ---------------------------------------------------------------------------

def run_all_missing(verbose):
    section("Case 1 - all-missing person")
    try:
        data = simulate()
        data.iloc[0, :] = np.nan
        m = SLM(data)
        ok, exc = calibrate_ok(m)
        if ok:
            finite_ok, finite_msg = outputs_finite(m)
            check("all-missing person - completes and outputs finite",
                  finite_ok, finite_msg or "", verbose)
        else:
            check("all-missing person - clean exception",
                  is_clean_exception(exc),
                  f"got {type(exc).__name__}: {exc}", verbose)
    except Exception:
        check("all-missing person - no unhandled crash",
              False, traceback.format_exc(), verbose)

    section("Case 2 - all-missing item")
    try:
        data = simulate()
        data = data.astype(float)
        data.iloc[:, 0] = np.nan
        m = SLM(data)
        ok, exc = calibrate_ok(m)
        if ok:
            finite_ok, finite_msg = outputs_finite(m)
            check("all-missing item - completes and outputs finite",
                  finite_ok, finite_msg or "", verbose)
        else:
            check("all-missing item - clean exception",
                  is_clean_exception(exc),
                  f"got {type(exc).__name__}: {exc}", verbose)
    except Exception:
        check("all-missing item - no unhandled crash",
              False, traceback.format_exc(), verbose)


def run_extreme_scores(verbose):
    section("Case 3 - extreme scores all-0")
    try:
        data = simulate()
        data.iloc[0, :] = 0
        m = SLM(data)
        ok, exc = calibrate_ok(m)
        if ok:
            try:
                m.person_abils()
                abils = getattr(m, "person_abilities", None)
                person = data.index[0] if not isinstance(
                    data.index, pd.MultiIndex) else data.index.get_level_values(-1)[0]
                abil_val = abils.loc[person] if abils is not None else None
                check("extreme all-0 - ability estimate is finite",
                      abil_val is not None and np.isfinite(abil_val),
                      f"got {abil_val}", verbose)
            except Exception as e:
                check("extreme all-0 - ability estimate is finite",
                      False, traceback.format_exc(), verbose)
        else:
            check("extreme all-0 - clean exception",
                  is_clean_exception(exc),
                  f"got {type(exc).__name__}: {exc}", verbose)
    except Exception:
        check("extreme all-0 - no unhandled crash",
              False, traceback.format_exc(), verbose)

    section("Case 4a - extreme scores all-max, constant=0.1")
    try:
        data = simulate()
        data.iloc[0, :] = MAX_SCORE
        m = SLM(data)
        ok, exc = calibrate_ok(m, constant=0.1)
        if ok:
            finite_ok, finite_msg = outputs_finite(m)
            check("extreme all-max constant=0.1 - outputs finite",
                  finite_ok, finite_msg or "", verbose)
        else:
            check("extreme all-max constant=0.1 - clean exception",
                  is_clean_exception(exc),
                  f"got {type(exc).__name__}: {exc}", verbose)
    except Exception:
        check("extreme all-max constant=0.1 - no unhandled crash",
              False, traceback.format_exc(), verbose)

    section("Case 4b - extreme scores all-max, constant=0")
    try:
        data = simulate()
        data.iloc[:, :] = MAX_SCORE
        m = SLM(data)
        caught_warning = False
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ok, exc = calibrate_ok(m, constant=0)
            caught_warning = len(w) > 0

        if ok:
            if caught_warning:
                check("extreme all-max constant=0 - warns", True, verbose=verbose)
            else:
                xfail("extreme all-max constant=0 - warns",
                      "KNOWN GAP: calibrate() completes silently with constant=0 "
                      "and all-max data. Note: SLM constructor does warn about "
                      "disconnected networks (a side-effect of all-max data), but "
                      "calibrate() should also specifically warn about constant=0 "
                      "and suggest dropping the item or using a non-zero constant.",
                      verbose=verbose)
        else:
            check("extreme all-max constant=0 - clean exception",
                  is_clean_exception(exc),
                  f"got {type(exc).__name__}: {exc}", verbose)
    except Exception:
        check("extreme all-max constant=0 - no unhandled crash",
              False, traceback.format_exc(), verbose)


def run_single_item(verbose):
    section("Case 5 - single item")
    try:
        np.random.seed(SIM_SEED)
        sim = SLM_Sim(no_of_items=1, no_of_persons=N_PERSONS)
        data = sim.scores
        m = SLM(data)
        caught_warning = False
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ok, exc = calibrate_ok(m)
            caught_warning = any(
                "item" in str(x.message).lower() or "rsm" in str(x.message).lower()
                for x in w
            )
        if ok:
            if caught_warning:
                check("single item - warns", True, verbose=verbose)
            else:
                xfail("single item - warns",
                      "KNOWN GAP: calibrate() does not warn when n_items=1.",
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
    parser = argparse.ArgumentParser(description="SLM edge case tests")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    run_all_missing(args.verbose)
    run_extreme_scores(args.verbose)
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
        print("\n  Expected failures (flip to check() when fixed):")
        for name, status, msg in results:
            if status == "xfail":
                print(f"    - {name}")
                if msg:
                    print(f"        {msg}")

    print(f"\n{'=' * 60}\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
