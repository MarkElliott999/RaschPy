"""
smoke_tests_mfrm.py
-------------------
Smoke tests for all four MFRM models (global, items, thresholds, matrix).

Each test:
  1. Simulates a small dataset with the corresponding sim class
  2. Instantiates MFRM with that data
  3. Runs calibration
  4. Calls every major stats_df method and a representative set of plot methods
  5. Asserts stored attributes have the expected shape / no all-NaN columns

Run with:
    python smoke_tests_mfrm.py
    python smoke_tests_mfrm.py -v        # verbose - prints each check as it passes
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

# Simulation parameters

SIM_SEED    = 42
N_PERSONS   = 200
N_ITEMS     = 6
N_RATERS    = 4
MAX_SCORE   = 3     # 4 categories => max score 3
RATER_RANGE = 1.0

# Helpers

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

results = []   # list of (name, passed, message)


def check(name, expr, msg="", verbose=False):
    if expr:
        results.append((name, True, ""))
        if verbose:
            print(f"  {PASS}  {name}")
    else:
        results.append((name, False, msg))
        print(f"  {FAIL}  {name}" + (f" - {msg}" if msg else ""))


def assert_df(name, df, verbose=False):
    if not isinstance(df, pd.DataFrame):
        check(name, False, f"expected DataFrame, got {type(df).__name__}", verbose)
        return
    check(name + " - non-empty", len(df) >= 1, f"got {len(df)} rows", verbose)
    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    check(name + " - no all-NaN columns", len(all_nan_cols) == 0,
          f"all-NaN: {all_nan_cols}", verbose)


def simulate(sim_cls):
    np.random.seed(SIM_SEED)
    sim = sim_cls(
        no_of_items=N_ITEMS,
        no_of_persons=N_PERSONS,
        no_of_raters=N_RATERS,
        max_score=MAX_SCORE,
        rater_range=RATER_RANGE,
    )
    return sim.scores


def run_model(label, model_name, sim_cls, verbose=False):
    print(f"\n{'─' * 60}")
    print(f"  Model: {label}  (model='{model_name}')")
    print(f"{'─' * 60}")

    # 1. Simulate
    try:
        data = simulate(sim_cls)
        check(f"[{label}] simulate", True, verbose=verbose)
    except Exception:
        check(f"[{label}] simulate", False, traceback.format_exc())
        print(f"  Cannot continue without data - skipping {label}.")
        return

    # 2. Instantiate
    try:
        m = MFRM(data)
        check(f"[{label}] instantiate", True, verbose=verbose)
    except Exception:
        check(f"[{label}] instantiate", False, traceback.format_exc())
        print(f"  Cannot continue without model instance - skipping {label}.")
        return

    # 3. Calibrate
    try:
        m.calibrate(model=model_name)
        check(f"[{label}] calibrate", True, verbose=verbose)
    except Exception:
        check(f"[{label}] calibrate", False, traceback.format_exc())
        print(f"  Cannot continue without calibration - skipping {label}.")
        return

    # 4. stats_df methods - call method, then read stored attribute
    for method_name, attr_name in (
        ("item_stats_df",      f"item_stats_{model_name}"),
        ("threshold_stats_df", f"threshold_stats_{model_name}"),
        ("person_stats_df",    f"person_stats_{model_name}"),
        ("test_stats_df",      f"test_stats_{model_name}"),
    ):
        try:
            getattr(m, method_name)(model=model_name)
            df = getattr(m, attr_name, None)
            assert_df(f"[{label}] {method_name}", df, verbose=verbose)
        except Exception:
            check(f"[{label}] {method_name}", False, traceback.format_exc(), verbose)

    # rater_stats_df
    try:
        m.rater_stats_df(model=model_name, marginal=True)
        df_marginal = getattr(m, f"rater_stats_{model_name}", None)
        assert_df(f"[{label}] rater_stats_df(marginal=True)", df_marginal, verbose=verbose)
    except Exception:
        check(f"[{label}] rater_stats_df(marginal=True)", False,
              traceback.format_exc(), verbose)

    if model_name == "matrix":
        try:
            m.rater_stats_df(model=model_name, marginal=False)
            df_cell = getattr(m, f"rater_stats_{model_name}", None)
            assert_df(f"[{label}] rater_stats_df(marginal=False)", df_cell, verbose=verbose)
        except Exception:
            check(f"[{label}] rater_stats_df(marginal=False)", False,
                  traceback.format_exc(), verbose)

    # 5. Abilities
    try:
        m.person_abils(model=model_name)
        abils = getattr(m, f"abils_{model_name}", None)
        check(f"[{label}] person_abils - returns Series",
              isinstance(abils, pd.Series), verbose=verbose)
        if isinstance(abils, pd.Series):
            check(f"[{label}] person_abils - no all-NaN",
                  not abils.isna().all(),
                  f"{abils.isna().sum()} NaNs out of {len(abils)}", verbose)
    except Exception:
        check(f"[{label}] person_abils", False, traceback.format_exc(), verbose)

    # 6. Standard errors
    try:
        m.std_errors(model=model_name, no_of_samples=10)
        check(f"[{label}] std_errors", True, verbose=verbose)
    except Exception:
        check(f"[{label}] std_errors", False, traceback.format_exc(), verbose)

    # 7. Plot methods (Agg backend - no window)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    item  = m.items[0]
    rater = m.raters[0]

    plot_calls = [
        ("icc",               lambda: m.icc(item=item, rater=rater)),
        ("crcs",              lambda: m.crcs(item=item, rater=rater)),
        ("threshold_ccs",     lambda: m.threshold_ccs(item=item, rater=rater)),
        ("iic",               lambda: m.iic(item=item)),
        ("tcc",               lambda: m.tcc()),
        ("test_info",         lambda: m.test_info()),
        ("test_csem",         lambda: m.test_csem()),
        ("std_residuals_plot",lambda: m.std_residuals_plot()),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for plot_name, call in plot_calls:
            try:
                call()
                plt.close("all")
                check(f"[{label}] plot/{plot_name}", True, verbose=verbose)
            except Exception:
                plt.close("all")
                check(f"[{label}] plot/{plot_name}", False,
                      traceback.format_exc(), verbose)


MODELS = [
    ("Global",     "global",     MFRM_Sim_Global),
    ("Items",      "items",      MFRM_Sim_Items),
    ("Thresholds", "thresholds", MFRM_Sim_Thresholds),
    ("Matrix",     "matrix",     MFRM_Sim_Matrix),
]


def main():
    parser = argparse.ArgumentParser(description="MFRM smoke tests")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print each passing check")
    args = parser.parse_args()

    for label, model_name, sim_cls in MODELS:
        run_model(label, model_name, sim_cls, verbose=args.verbose)

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
