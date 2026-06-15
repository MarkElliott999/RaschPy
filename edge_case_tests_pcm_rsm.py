"""
edge_case_tests_pcm_rsm.py
--------------------------
Edge case tests for PCM and RSM models.

Cases covered (for each model):
  1. All-missing person   - should complete (warn+drop) or raise cleanly
  2. All-missing item     - same
  3. Extreme scores all-0 - ability via extreme score adjustment
  4a. Extreme scores all-max, constant=0.1 - additive smoothing active
  4b. Extreme scores all-max, constant=0   - should warn or raise cleanly
  5. Single item          - should warn or raise cleanly

Run with:
    python edge_case_tests_pcm_rsm.py
    python edge_case_tests_pcm_rsm.py -v
"""

import sys
import traceback
import argparse
import warnings
import numpy as np
import pandas as pd

from raschpy import PCM, RSM
from raschpy.simulation import PCM_Sim, RSM_Sim

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIM_SEED      = 42
N_PERSONS     = 100
N_ITEMS       = 6
MAX_SCORE     = 3
MAX_SCORE_VEC = [3, 3, 3, 3, 3, 3]

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


def calibrate_ok(m, **kwargs):
    try:
        m.calibrate(**kwargs)
        return True, None
    except Exception as e:
        return False, e


def is_clean_exception(e):
    return isinstance(e, (ValueError, TypeError, ArithmeticError,
                          NotImplementedError))


def outputs_finite(m, attr="item_stats"):
    try:
        m.item_stats_df()
        df = getattr(m, attr, None)
        if not isinstance(df, pd.DataFrame):
            return False, f"{attr} not a DataFrame"
        num = df.select_dtypes(include="number")
        if num.isnull().all(axis=None):
            return False, f"all-NaN {attr}"
        return True, None
    except Exception as e:
        return False, str(e)


def section(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ---------------------------------------------------------------------------
# Generic case runners (work for both PCM and RSM)
# ---------------------------------------------------------------------------

def run_all_missing(label, build_normal, build_single_item, verbose):
    section(f"Case 1 - all-missing person  [{label}]")
    try:
        data = build_normal()
        data.iloc[0, :] = np.nan
        m = (PCM(data, max_score_vector=MAX_SCORE_VEC)
             if label == "PCM" else RSM(data, max_score=MAX_SCORE))
        ok, exc = calibrate_ok(m)
        if ok:
            finite_ok, finite_msg = outputs_finite(m)
            check(f"[{label}] all-missing person - completes and outputs finite",
                  finite_ok, finite_msg or "", verbose)
        else:
            check(f"[{label}] all-missing person - clean exception",
                  is_clean_exception(exc),
                  f"got {type(exc).__name__}: {exc}", verbose)
    except Exception:
        check(f"[{label}] all-missing person - no unhandled crash",
              False, traceback.format_exc(), verbose)

    section(f"Case 2 - all-missing item  [{label}]")
    try:
        data = build_normal()
        data = data.astype(float)
        data.iloc[:, 0] = np.nan
        m = (PCM(data, max_score_vector=MAX_SCORE_VEC)
             if label == "PCM" else RSM(data, max_score=MAX_SCORE))
        ok, exc = calibrate_ok(m)
        if ok:
            finite_ok, finite_msg = outputs_finite(m)
            check(f"[{label}] all-missing item - completes and outputs finite",
                  finite_ok, finite_msg or "", verbose)
        else:
            check(f"[{label}] all-missing item - clean exception",
                  is_clean_exception(exc),
                  f"got {type(exc).__name__}: {exc}", verbose)
    except Exception:
        check(f"[{label}] all-missing item - no unhandled crash",
              False, traceback.format_exc(), verbose)


def run_extreme_scores(label, build_normal, max_score, verbose):
    section(f"Case 3 - extreme scores all-0  [{label}]")
    try:
        data = build_normal()
        data.iloc[0, :] = 0
        m = (PCM(data, max_score_vector=MAX_SCORE_VEC)
             if label == "PCM" else RSM(data, max_score=MAX_SCORE))
        ok, exc = calibrate_ok(m)
        if ok:
            try:
                m.person_abils()
                abils = getattr(m, "person_abilities", None)
                person = (data.index[0] if not isinstance(data.index, pd.MultiIndex)
                          else data.index.get_level_values(-1)[0])
                abil_val = abils.loc[person] if abils is not None else None
                check(f"[{label}] extreme all-0 - ability estimate is finite",
                      abil_val is not None and np.isfinite(abil_val),
                      f"got {abil_val}", verbose)
            except Exception:
                check(f"[{label}] extreme all-0 - ability estimate is finite",
                      False, traceback.format_exc(), verbose)
        else:
            check(f"[{label}] extreme all-0 - clean exception",
                  is_clean_exception(exc),
                  f"got {type(exc).__name__}: {exc}", verbose)
    except Exception:
        check(f"[{label}] extreme all-0 - no unhandled crash",
              False, traceback.format_exc(), verbose)

    section(f"Case 4a - extreme scores all-max, constant=0.1  [{label}]")
    try:
        data = build_normal()
        data.iloc[0, :] = max_score
        m = (PCM(data, max_score_vector=MAX_SCORE_VEC)
             if label == "PCM" else RSM(data, max_score=MAX_SCORE))
        ok, exc = calibrate_ok(m, constant=0.1)
        if ok:
            finite_ok, finite_msg = outputs_finite(m)
            check(f"[{label}] extreme all-max constant=0.1 - outputs finite",
                  finite_ok, finite_msg or "", verbose)
        else:
            check(f"[{label}] extreme all-max constant=0.1 - clean exception",
                  is_clean_exception(exc),
                  f"got {type(exc).__name__}: {exc}", verbose)
    except Exception:
        check(f"[{label}] extreme all-max constant=0.1 - no unhandled crash",
              False, traceback.format_exc(), verbose)

    section(f"Case 4b - extreme scores all-max, constant=0  [{label}]")
    try:
        data = build_normal()
        data.iloc[:, :] = max_score
        m = (PCM(data, max_score_vector=MAX_SCORE_VEC)
             if label == "PCM" else RSM(data, max_score=MAX_SCORE))
        caught_warning = False
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ok, exc = calibrate_ok(m, constant=0)
            caught_warning = len(w) > 0

        if ok:
            if caught_warning:
                check(f"[{label}] extreme all-max constant=0 - warns",
                      True, verbose=verbose)
            else:
                xfail(f"[{label}] extreme all-max constant=0 - warns",
                      "KNOWN GAP: calibrate() completes silently with constant=0 "
                      "and all-max data. Note: constructor may warn about disconnected "
                      "networks (side-effect of all-max data), but calibrate() should "
                      "also specifically warn about constant=0.",
                      verbose=verbose)
        else:
            check(f"[{label}] extreme all-max constant=0 - clean exception",
                  is_clean_exception(exc),
                  f"got {type(exc).__name__}: {exc}", verbose)
    except Exception:
        check(f"[{label}] extreme all-max constant=0 - no unhandled crash",
              False, traceback.format_exc(), verbose)


def run_single_item(label, verbose):
    section(f"Case 5 - single item  [{label}]")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            np.random.seed(SIM_SEED)
            if label == "PCM":
                sim = PCM_Sim(no_of_items=1, no_of_persons=N_PERSONS,
                              max_score_vector=[MAX_SCORE])
                data = sim.scores
                m = PCM(data, max_score_vector=[MAX_SCORE])
            else:
                sim = RSM_Sim(no_of_items=1, no_of_persons=N_PERSONS,
                              max_score=MAX_SCORE)
                data = sim.scores
                m = RSM(data, max_score=MAX_SCORE)

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
                check(f"[{label}] single item - warns", True, verbose=verbose)
            else:
                xfail(f"[{label}] single item - warns",
                      "KNOWN GAP: calibrate() does not warn when n_items=1.",
                      verbose=verbose)
            check(f"[{label}] single item - no unhandled crash", True, verbose=verbose)
        else:
            check(f"[{label}] single item - clean exception",
                  is_clean_exception(exc),
                  f"got {type(exc).__name__}: {exc}", verbose)
    except Exception:
        check(f"[{label}] single item - no unhandled crash",
              False, traceback.format_exc(), verbose)


# ---------------------------------------------------------------------------
# Build functions
# ---------------------------------------------------------------------------

def build_pcm():
    np.random.seed(SIM_SEED)
    sim = PCM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS,
                  max_score_vector=MAX_SCORE_VEC)
    return sim.scores


def build_rsm():
    np.random.seed(SIM_SEED)
    sim = RSM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS,
                  max_score=MAX_SCORE)
    return sim.scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PCM/RSM edge case tests")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    for label, build_fn, max_score in (
        ("PCM", build_pcm, MAX_SCORE),
        ("RSM", build_rsm, MAX_SCORE),
    ):
        run_all_missing(label, build_fn, None, args.verbose)
        run_extreme_scores(label, build_fn, max_score, args.verbose)
        run_single_item(label, args.verbose)

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
