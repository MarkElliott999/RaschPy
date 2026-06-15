"""
regression_tests_mfrm.py
------------------------
Numerical regression tests for all four MFRM models.

First run - generate known-good fixtures:
    python regression_tests_mfrm.py --generate

Subsequent runs - assert against fixtures:
    python regression_tests_mfrm.py
    python regression_tests_mfrm.py -v     # verbose: print each passing check

Fixtures are saved to regression_data/ at the project root.
All numeric comparisons use atol=1e-6 via np.testing.assert_allclose.
"""

import sys
import os
import json
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
N_PERSONS   = 200
N_ITEMS     = 6
N_RATERS    = 4
MAX_SCORE   = 3
RATER_RANGE = 1.0
ATOL        = 1e-6

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "regression_data")

MODELS = [
    ("Global",     "global",     MFRM_Sim_Global),
    ("Items",      "items",      MFRM_Sim_Items),
    ("Thresholds", "thresholds", MFRM_Sim_Thresholds),
    ("Matrix",     "matrix",     MFRM_Sim_Matrix),
]

# ---------------------------------------------------------------------------
# Helpers: reporting
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


# ---------------------------------------------------------------------------
# Helpers: serialisation
# ---------------------------------------------------------------------------

def fixture_path(model_name, name):
    return os.path.join(FIXTURE_DIR, model_name, name)


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
    return np.load(path + ".npy")


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def flatten_columns(df):
    """Flatten MultiIndex columns to 'level0 | level1' strings for CSV round-trip."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [" | ".join(str(s) for s in col) for col in df.columns]
    return df


def save_plot_lines(ax_list, path):
    """Save line x/y data from a list of axes as a dict of numpy arrays."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {}
    for ax_idx, ax in enumerate(ax_list):
        for line_idx, line in enumerate(ax.lines):
            key = f"ax{ax_idx}_line{line_idx}"
            data[key] = {"x": line.get_xdata().tolist(),
                         "y": line.get_ydata().tolist()}
    save_json(data, path)


def load_plot_lines(path):
    return load_json(path)


# ---------------------------------------------------------------------------
# Simulation + calibration
# ---------------------------------------------------------------------------

def build_model(sim_cls, model_name):
    np.random.seed(SIM_SEED)
    sim = sim_cls(
        no_of_items=N_ITEMS,
        no_of_persons=N_PERSONS,
        no_of_raters=N_RATERS,
        max_score=MAX_SCORE,
        rater_range=RATER_RANGE,
    )
    m = MFRM(sim.scores)
    m.calibrate(model=model_name)
    m.item_stats_df(model=model_name)
    m.threshold_stats_df(model=model_name)
    m.person_stats_df(model=model_name)
    m.test_stats_df(model=model_name)
    m.rater_stats_df(model=model_name, marginal=True)
    if model_name == "matrix":
        m.rater_stats_df(model=model_name, marginal=False)
    m.person_abils(model=model_name)
    m.std_errors(model=model_name, no_of_samples=100)
    return m


# ---------------------------------------------------------------------------
# Generate fixtures
# ---------------------------------------------------------------------------

def generate_fixtures(m, model_name, label):
    base = os.path.join(FIXTURE_DIR, model_name)
    os.makedirs(base, exist_ok=True)

    # stats DataFrames
    for attr in (
        f"item_stats_{model_name}",
        f"threshold_stats_{model_name}",
        f"person_stats_{model_name}",
        f"test_stats_{model_name}",
        f"rater_stats_{model_name}",
    ):
        df = getattr(m, attr, None)
        if isinstance(df, pd.DataFrame):
            save_df(df, os.path.join(base, f"{attr}.csv"))

    # abilities
    abils = getattr(m, f"abils_{model_name}", None)
    if isinstance(abils, pd.Series):
        save_series(abils, os.path.join(base, f"abils_{model_name}.csv"))

    # item SE (Series, model-independent but save per model for isolation)
    if isinstance(getattr(m, "item_se", None), pd.Series):
        save_series(m.item_se, os.path.join(base, "item_se.csv"))

    # threshold SE (numpy array)
    thr_se = getattr(m, f"threshold_se_{model_name}", None)
    if isinstance(thr_se, np.ndarray):
        save_array(thr_se, os.path.join(base, f"threshold_se_{model_name}"))

    # rater SE - shape varies by model
    rater_se = getattr(m, f"rater_se_{model_name}", None)
    if rater_se is not None:
        _save_rater_se(rater_se, model_name, base)

    # plot line data
    _save_all_plot_lines(m, model_name, base)

    print(f"  Fixtures saved to regression_data/{model_name}/")


def _save_rater_se(rater_se, model_name, base):
    path = os.path.join(base, f"rater_se_{model_name}")
    if model_name == "global":
        # Series
        save_series(rater_se, path + ".csv")
    elif model_name == "items":
        # dict of Series keyed by rater
        os.makedirs(path, exist_ok=True)
        for rater, s in rater_se.items():
            save_series(s, os.path.join(path, f"{rater}.csv"))
    elif model_name == "thresholds":
        # dict of numpy arrays keyed by rater
        os.makedirs(path, exist_ok=True)
        for rater, arr in rater_se.items():
            save_array(arr, os.path.join(path, str(rater)))
    elif model_name == "matrix":
        # nested dict: rater -> item -> numpy array
        os.makedirs(path, exist_ok=True)
        for rater, item_dict in rater_se.items():
            rater_dir = os.path.join(path, str(rater))
            os.makedirs(rater_dir, exist_ok=True)
            for item, arr in item_dict.items():
                save_array(arr, os.path.join(rater_dir, str(item)))


def _save_all_plot_lines(m, model_name, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    item  = m.items[0]
    rater = m.raters[0]

    plot_calls = [
        ("icc",                lambda: m.icc(item=item, rater=rater)),
        ("crcs",               lambda: m.crcs(item=item, rater=rater)),
        ("threshold_ccs",      lambda: m.threshold_ccs(item=item, rater=rater)),
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
                axes = plt.gcf().get_axes()
                save_plot_lines(axes,
                    os.path.join(base, "plots", f"{plot_name}.json"))
                plt.close("all")
            except Exception:
                plt.close("all")
                print(f"  WARNING: could not save plot fixture for {plot_name}")


# ---------------------------------------------------------------------------
# Assert against fixtures
# ---------------------------------------------------------------------------

def assert_numeric_df(name, df, fixture_path_csv, verbose):
    """Load fixture CSV and compare numeric columns within atol."""
    try:
        ref = load_df(fixture_path_csv)
    except FileNotFoundError:
        check(name, False, f"fixture not found: {fixture_path_csv}", verbose)
        return

    # align columns and index
    try:
        df = flatten_columns(df)
        shared_cols = ref.columns.intersection(df.columns)
        num_cols = ref[shared_cols].select_dtypes(include="number").columns

        check(name + " - columns match",
              set(ref.columns) == set(df.columns),
              f"ref={set(ref.columns)} got={set(df.columns)}", verbose)

        check(name + " - index match",
              list(ref.index.astype(str)) == list(df.index.astype(str)),
              "index mismatch", verbose)

        np.testing.assert_allclose(
            df[num_cols].values.astype(float),
            ref[num_cols].values.astype(float),
            atol=ATOL, equal_nan=True,
        )
        check(name + " - numeric values within 1e-6", True, verbose=verbose)
    except AssertionError as e:
        check(name + " - numeric values within 1e-6", False, str(e), verbose)
    except Exception:
        check(name + " - numeric values within 1e-6", False,
              traceback.format_exc(), verbose)


def assert_numeric_series(name, s, fixture_path_csv, verbose):
    try:
        ref = load_series(fixture_path_csv)
    except FileNotFoundError:
        check(name, False, f"fixture not found: {fixture_path_csv}", verbose)
        return
    try:
        np.testing.assert_allclose(
            s.values.astype(float),
            ref.values.astype(float),
            atol=ATOL, equal_nan=True,
        )
        check(name + " - values within 1e-6", True, verbose=verbose)
    except AssertionError as e:
        check(name + " - values within 1e-6", False, str(e), verbose)


def assert_numeric_array(name, arr, fixture_npy, verbose):
    try:
        ref = load_array(fixture_npy)
    except FileNotFoundError:
        check(name, False, f"fixture not found: {fixture_npy}.npy", verbose)
        return
    try:
        np.testing.assert_allclose(arr, ref, atol=ATOL, equal_nan=True)
        check(name + " - values within 1e-6", True, verbose=verbose)
    except AssertionError as e:
        check(name + " - values within 1e-6", False, str(e), verbose)


def assert_rater_se(m, model_name, base, verbose):
    rater_se = getattr(m, f"rater_se_{model_name}", None)
    if rater_se is None:
        check(f"rater_se_{model_name} - exists", False,
              "attribute not found", verbose)
        return

    path = os.path.join(base, f"rater_se_{model_name}")

    if model_name == "global":
        assert_numeric_series(
            f"rater_se_{model_name}", rater_se, path + ".csv", verbose)

    elif model_name == "items":
        for rater, s in rater_se.items():
            assert_numeric_series(
                f"rater_se_{model_name}[{rater}]", s,
                os.path.join(path, f"{rater}.csv"), verbose)

    elif model_name == "thresholds":
        for rater, arr in rater_se.items():
            assert_numeric_array(
                f"rater_se_{model_name}[{rater}]", arr,
                os.path.join(path, str(rater)), verbose)

    elif model_name == "matrix":
        for rater, item_dict in rater_se.items():
            for item, arr in item_dict.items():
                assert_numeric_array(
                    f"rater_se_{model_name}[{rater}][{item}]", arr,
                    os.path.join(path, str(rater), str(item)), verbose)


def assert_plot_lines(m, model_name, base, verbose):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    item  = m.items[0]
    rater = m.raters[0]

    plot_calls = [
        ("icc",                lambda: m.icc(item=item, rater=rater)),
        ("crcs",               lambda: m.crcs(item=item, rater=rater)),
        ("threshold_ccs",      lambda: m.threshold_ccs(item=item, rater=rater)),
        ("iic",                lambda: m.iic(item=item)),
        ("tcc",                lambda: m.tcc()),
        ("test_info",          lambda: m.test_info()),
        ("test_csem",          lambda: m.test_csem()),
        ("std_residuals_plot", lambda: m.std_residuals_plot()),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for plot_name, call in plot_calls:
            fixture = os.path.join(base, "plots", f"{plot_name}.json")
            try:
                call()
                axes = plt.gcf().get_axes()
                plt.close("all")
            except Exception:
                plt.close("all")
                check(f"plot/{plot_name} - renders", False,
                      traceback.format_exc(), verbose)
                continue

            try:
                ref = load_plot_lines(fixture)
            except FileNotFoundError:
                check(f"plot/{plot_name}", False,
                      f"fixture not found: {fixture}", verbose)
                continue

            current = {}
            for ax_idx, ax in enumerate(axes):
                for line_idx, line in enumerate(ax.lines):
                    key = f"ax{ax_idx}_line{line_idx}"
                    current[key] = {"x": line.get_xdata().tolist(),
                                    "y": line.get_ydata().tolist()}

            line_keys_match = set(ref.keys()) == set(current.keys())
            check(f"plot/{plot_name} - line count",
                  line_keys_match,
                  f"ref keys={set(ref.keys())} got={set(current.keys())}",
                  verbose)

            if not line_keys_match:
                continue

            all_ok = True
            for key in ref:
                try:
                    np.testing.assert_allclose(
                        np.array(current[key]["x"]),
                        np.array(ref[key]["x"]),
                        atol=ATOL, equal_nan=True,
                    )
                    np.testing.assert_allclose(
                        np.array(current[key]["y"]),
                        np.array(ref[key]["y"]),
                        atol=ATOL, equal_nan=True,
                    )
                except AssertionError as e:
                    check(f"plot/{plot_name}/{key} - values within 1e-6",
                          False, str(e), verbose)
                    all_ok = False

            if all_ok:
                check(f"plot/{plot_name} - all lines within 1e-6",
                      True, verbose=verbose)


# ---------------------------------------------------------------------------
# Run one model
# ---------------------------------------------------------------------------

def run_model(label, model_name, sim_cls, generate, verbose):
    print(f"\n{'─' * 60}")
    print(f"  Model: {label}  (model='{model_name}')")
    print(f"{'─' * 60}")

    try:
        m = build_model(sim_cls, model_name)
    except Exception:
        check(f"[{label}] build", False, traceback.format_exc())
        print(f"  Cannot continue - skipping {label}.")
        return

    base = os.path.join(FIXTURE_DIR, model_name)

    if generate:
        generate_fixtures(m, model_name, label)
        return

    # stats DataFrames
    for attr in (
        f"item_stats_{model_name}",
        f"threshold_stats_{model_name}",
        f"person_stats_{model_name}",
        f"test_stats_{model_name}",
        f"rater_stats_{model_name}",
    ):
        df = getattr(m, attr, None)
        if isinstance(df, pd.DataFrame):
            assert_numeric_df(
                f"[{label}] {attr}", df,
                os.path.join(base, f"{attr}.csv"), verbose)

    # abilities
    abils = getattr(m, f"abils_{model_name}", None)
    if isinstance(abils, pd.Series):
        assert_numeric_series(
            f"[{label}] abils_{model_name}", abils,
            os.path.join(base, f"abils_{model_name}.csv"), verbose)

    # item SE
    item_se = getattr(m, "item_se", None)
    if isinstance(item_se, pd.Series):
        assert_numeric_series(
            f"[{label}] item_se", item_se,
            os.path.join(base, "item_se.csv"), verbose)

    # threshold SE
    thr_se = getattr(m, f"threshold_se_{model_name}", None)
    if isinstance(thr_se, np.ndarray):
        assert_numeric_array(
            f"[{label}] threshold_se_{model_name}", thr_se,
            os.path.join(base, f"threshold_se_{model_name}"), verbose)

    # rater SE
    assert_rater_se(m, model_name, base, verbose)

    # plots
    assert_plot_lines(m, model_name, base, verbose)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MFRM regression tests")
    parser.add_argument("--generate", action="store_true",
                        help="Generate known-good fixtures (run once)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print each passing check")
    args = parser.parse_args()

    if args.generate:
        os.makedirs(FIXTURE_DIR, exist_ok=True)
        print(f"Generating fixtures in {FIXTURE_DIR}/")

    for label, model_name, sim_cls in MODELS:
        run_model(label, model_name, sim_cls,
                  generate=args.generate, verbose=args.verbose)

    if args.generate:
        print("\nDone. Commit regression_data/ to version control,")
        print("then run without --generate to assert against fixtures.")
        return

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
