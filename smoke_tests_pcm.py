"""
smoke_tests_pcm.py
------------------
Smoke tests for the PCM model.

Checks:
  1. Simulate -> instantiate -> calibrate (bail on failure)
  2. item_stats_df, threshold_stats_df, person_stats_df, test_stats_df
  3. person_abils — returns Series, no all-NaN
  4. std_errors — no crash
  5. All plot methods via Agg backend

Run with:
    python smoke_tests_pcm.py
    python smoke_tests_pcm.py -v
"""

import sys
import traceback
import argparse
import warnings
import numpy as np
import pandas as pd

from raschpy import PCM
from raschpy.simulation import PCM_Sim

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIM_SEED      = 42
N_PERSONS     = 200
N_ITEMS       = 6
MAX_SCORE_VEC = [3, 3, 3, 3, 3, 3]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS_STR = "\033[32mPASS\033[0m"
FAIL_STR = "\033[31mFAIL\033[0m"
results  = []


def check(name, expr, msg="", verbose=False):
    if expr:
        results.append((name, True, ""))
        if verbose:
            print(f"  {PASS_STR}  {name}")
    else:
        results.append((name, False, msg))
        short = msg.splitlines()[-1] if msg else ""
        print(f"  {FAIL_STR}  {name}" + (f" - {short}" if short else ""))


def assert_df(name, df, verbose=False):
    if not isinstance(df, pd.DataFrame):
        check(name, False, f"expected DataFrame, got {type(df).__name__}", verbose)
        return
    check(name + " - non-empty", len(df) >= 1, f"got {len(df)} rows", verbose)
    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    check(name + " - no all-NaN columns", len(all_nan_cols) == 0,
          f"all-NaN: {all_nan_cols}", verbose)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run(verbose=False):
    print(f"\n{'─' * 60}")
    print("  Model: PCM")
    print(f"{'─' * 60}")

    # 1. Simulate
    try:
        np.random.seed(SIM_SEED)
        sim = PCM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS,
                      max_score_vector=MAX_SCORE_VEC)
        data = sim.scores
        check("simulate", True, verbose=verbose)
    except Exception:
        check("simulate", False, traceback.format_exc())
        print("  Cannot continue without data.")
        return

    # 2. Instantiate
    try:
        m = PCM(data, max_score_vector=MAX_SCORE_VEC)
        check("instantiate", True, verbose=verbose)
    except Exception:
        check("instantiate", False, traceback.format_exc())
        print("  Cannot continue without model instance.")
        return

    # 3. Calibrate
    try:
        m.calibrate()
        check("calibrate", True, verbose=verbose)
    except Exception:
        check("calibrate", False, traceback.format_exc())
        print("  Cannot continue without calibration.")
        return

    # 4. stats_df methods
    for method_name, attr_name in (
        ("item_stats_df",   "item_stats"),
        ("person_stats_df", "person_stats"),
        ("test_stats_df",   "test_stats"),
    ):
        try:
            getattr(m, method_name)()
            df = getattr(m, attr_name, None)
            assert_df(f"{method_name}", df, verbose=verbose)
        except Exception:
            check(f"{method_name}", False, traceback.format_exc(), verbose)

    # threshold_stats_df stores to two attributes
    try:
        m.threshold_stats_df()
        for attr in ("threshold_stats_uncentred", "threshold_stats_centred"):
            df = getattr(m, attr, None)
            assert_df(f"threshold_stats_df - {attr}", df, verbose=verbose)
    except Exception:
        check("threshold_stats_df", False, traceback.format_exc(), verbose)

    # 5. Abilities
    try:
        m.person_abils()
        abils = getattr(m, "person_abilities", None)
        check("person_abils - returns Series",
              isinstance(abils, pd.Series), verbose=verbose)
        if isinstance(abils, pd.Series):
            check("person_abils - no all-NaN",
                  not abils.isna().all(),
                  f"{abils.isna().sum()} NaNs out of {len(abils)}", verbose)
    except Exception:
        check("person_abils", False, traceback.format_exc(), verbose)

    # 6. Standard errors
    try:
        m.std_errors(no_of_samples=10)
        check("std_errors", True, verbose=verbose)
    except Exception:
        check("std_errors", False, traceback.format_exc(), verbose)

    # 7. Plot methods
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    item = m.dataframe.columns[0]
    plot_calls = [
        ("icc",                lambda: m.icc(item=item)),
        ("crcs",               lambda: m.crcs(item=item)),
        ("threshold_ccs",      lambda: m.threshold_ccs(item=item)),
        ("iic",                lambda: m.iic(item=item)),
        ("tcc",                lambda: m.tcc()),
        ("test_info",          lambda: m.test_info()),
        ("test_csem",          lambda: m.test_csem()),
        ("std_residuals_plot", lambda: m.std_residuals_plot()),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for plot_name, call in plot_calls:
            try:
                call()
                plt.close("all")
                check(f"plot/{plot_name}", True, verbose=verbose)
            except Exception:
                plt.close("all")
                check(f"plot/{plot_name}", False,
                      traceback.format_exc(), verbose)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PCM smoke tests")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    run(verbose=args.verbose)

    print(f"\n{'=' * 60}")
    total  = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = total - passed

    print(f"  {passed}/{total} checks passed", end="")
    if failed:
        print(f"  ({failed} failed)\n")
        print("  Failed checks:")
        for name, ok, msg in results:
            if not ok:
                print(f"    - {name}")
                if msg and args.verbose:
                    for line in msg.strip().splitlines():
                        print(f"        {line}")
    else:
        print("  - all good.\n")

    print(f"{'=' * 60}\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
