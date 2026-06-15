"""
regression_tests_slm_pcm_rsm.py
--------------------------------
Numerical regression tests for SLM, PCM, and RSM models.

First run - generate known-good fixtures:
    python regression_tests_slm_pcm_rsm.py --generate

Subsequent runs - assert against fixtures:
    python regression_tests_slm_pcm_rsm.py
    python regression_tests_slm_pcm_rsm.py -v     # verbose

Fixtures are saved to regression_data/ at the project root (same folder as
MFRM regression fixtures, in subfolders slm/, pcm/, rsm/).
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

from raschpy import SLM, PCM, RSM
from raschpy.simulation import SLM_Sim, PCM_Sim, RSM_Sim

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIM_SEED      = 42
N_PERSONS     = 200
N_ITEMS       = 6
MAX_SCORE     = 3                          # RSM and PCM uniform max score
MAX_SCORE_VEC = [3, 3, 3, 3, 3, 3]        # PCM per-item max scores
ATOL          = 1e-6

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "regression_data")

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
# Helpers: serialisation (mirrors regression_tests_mfrm.py)
# ---------------------------------------------------------------------------

def flatten_columns(df):
    """Flatten MultiIndex columns to 'level0 | level1' strings for CSV round-trip."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [" | ".join(str(s) for s in col) for col in df.columns]
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
    return np.load(path + ".npy")


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_plot_lines(ax_list, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {}
    for ax_idx, ax in enumerate(ax_list):
        for line_idx, line in enumerate(ax.lines):
            key = f"ax{ax_idx}_line{line_idx}"
            data[key] = {"x": line.get_xdata().tolist(),
                         "y": line.get_ydata().tolist()}
    save_json(data, path)


# ---------------------------------------------------------------------------
# Helpers: assertions
# ---------------------------------------------------------------------------

def assert_numeric_df(name, df, fixture_csv, verbose):
    try:
        ref = load_df(fixture_csv)
    except FileNotFoundError:
        check(name, False, f"fixture not found: {fixture_csv}", verbose)
        return
    try:
        df = flatten_columns(df)
        shared = ref.columns.intersection(df.columns)
        num_cols = ref[shared].select_dtypes(include="number").columns
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


def assert_numeric_series(name, s, fixture_csv, verbose):
    try:
        ref = load_series(fixture_csv)
    except FileNotFoundError:
        check(name, False, f"fixture not found: {fixture_csv}", verbose)
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


def assert_plot_lines(m, plot_calls, base, verbose):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
                ref = load_json(fixture)
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
            check(f"plot/{plot_name} - line count", line_keys_match,
                  f"ref={set(ref.keys())} got={set(current.keys())}", verbose)
            if not line_keys_match:
                continue

            all_ok = True
            for key in ref:
                try:
                    np.testing.assert_allclose(
                        np.array(current[key]["x"]),
                        np.array(ref[key]["x"]),
                        atol=ATOL, equal_nan=True)
                    np.testing.assert_allclose(
                        np.array(current[key]["y"]),
                        np.array(ref[key]["y"]),
                        atol=ATOL, equal_nan=True)
                except AssertionError as e:
                    check(f"plot/{plot_name}/{key} - values within 1e-6",
                          False, str(e), verbose)
                    all_ok = False
            if all_ok:
                check(f"plot/{plot_name} - all lines within 1e-6",
                      True, verbose=verbose)


# ---------------------------------------------------------------------------
# SLM
# ---------------------------------------------------------------------------

def build_slm():
    np.random.seed(SIM_SEED)
    sim = SLM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS)
    m = SLM(sim.scores)
    m.calibrate()
    m.item_stats_df()
    m.person_stats_df()
    m.test_stats_df()
    m.person_abils()
    m.std_errors(no_of_samples=100)
    return m


def slm_plot_calls(m):
    item = m.dataframe.columns[0]
    return [
        ("icc",               lambda: m.icc(item=item)),
        ("iic",               lambda: m.iic(item=item)),
        ("tcc",               lambda: m.tcc()),
        ("test_info",         lambda: m.test_info()),
        ("test_csem",         lambda: m.test_csem()),
        ("std_residuals_plot",lambda: m.std_residuals_plot()),
    ]


def generate_slm(m):
    base = os.path.join(FIXTURE_DIR, "slm")
    os.makedirs(base, exist_ok=True)
    save_df(m.item_stats,    os.path.join(base, "item_stats.csv"))
    save_df(m.person_stats,  os.path.join(base, "person_stats.csv"))
    save_df(m.test_stats,    os.path.join(base, "test_stats.csv"))
    save_series(m.person_abilities, os.path.join(base, "person_abilities.csv"))
    save_series(m.item_se,   os.path.join(base, "item_se.csv"))
    _save_plot_lines(m, slm_plot_calls(m), base)
    print("  Fixtures saved to regression_data/slm/")


def assert_slm(m, verbose):
    base = os.path.join(FIXTURE_DIR, "slm")
    assert_numeric_df("[SLM] item_stats",   m.item_stats,
                      os.path.join(base, "item_stats.csv"), verbose)
    assert_numeric_df("[SLM] person_stats", m.person_stats,
                      os.path.join(base, "person_stats.csv"), verbose)
    assert_numeric_df("[SLM] test_stats",   m.test_stats,
                      os.path.join(base, "test_stats.csv"), verbose)
    assert_numeric_series("[SLM] person_abilities", m.person_abilities,
                          os.path.join(base, "person_abilities.csv"), verbose)
    assert_numeric_series("[SLM] item_se", m.item_se,
                          os.path.join(base, "item_se.csv"), verbose)
    assert_plot_lines(m, slm_plot_calls(m), base, verbose)


# ---------------------------------------------------------------------------
# PCM
# ---------------------------------------------------------------------------

def build_pcm():
    np.random.seed(SIM_SEED)
    sim = PCM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS,
                  max_score_vector=MAX_SCORE_VEC)
    m = PCM(sim.scores, max_score_vector=MAX_SCORE_VEC)
    m.calibrate()
    m.item_stats_df()
    m.person_stats_df()
    m.test_stats_df()
    m.person_abils()
    m.std_errors(no_of_samples=100)
    return m


def pcm_plot_calls(m):
    item = m.dataframe.columns[0]
    return [
        ("icc",               lambda: m.icc(item=item)),
        ("crcs",              lambda: m.crcs(item=item)),
        ("iic",               lambda: m.iic(item=item)),
        ("tcc",               lambda: m.tcc()),
        ("test_info",         lambda: m.test_info()),
        ("test_csem",         lambda: m.test_csem()),
        ("std_residuals_plot",lambda: m.std_residuals_plot()),
    ]


def generate_pcm(m):
    base = os.path.join(FIXTURE_DIR, "pcm")
    os.makedirs(base, exist_ok=True)
    save_df(m.item_stats,    os.path.join(base, "item_stats.csv"))
    save_df(m.person_stats,  os.path.join(base, "person_stats.csv"))
    save_df(m.test_stats,    os.path.join(base, "test_stats.csv"))
    save_series(m.person_abilities, os.path.join(base, "person_abilities.csv"))
    save_series(m.item_se, os.path.join(base, "item_se.csv"))
    thr_se_dir = os.path.join(base, "threshold_se_by_item")
    os.makedirs(thr_se_dir, exist_ok=True)
    for item, arr in m.threshold_se.items():
        save_array(arr, os.path.join(thr_se_dir, str(item)))
    _save_plot_lines(m, pcm_plot_calls(m), base)
    print("  Fixtures saved to regression_data/pcm/")


def assert_pcm(m, verbose):
    base = os.path.join(FIXTURE_DIR, "pcm")
    assert_numeric_df("[PCM] item_stats",   m.item_stats,
                      os.path.join(base, "item_stats.csv"), verbose)
    assert_numeric_df("[PCM] person_stats", m.person_stats,
                      os.path.join(base, "person_stats.csv"), verbose)
    assert_numeric_df("[PCM] test_stats",   m.test_stats,
                      os.path.join(base, "test_stats.csv"), verbose)
    assert_numeric_series("[PCM] person_abilities", m.person_abilities,
                          os.path.join(base, "person_abilities.csv"), verbose)
    assert_numeric_series("[PCM] item_se", m.item_se,
                          os.path.join(base, "item_se.csv"), verbose)
    thr_se_dir = os.path.join(base, "threshold_se_by_item")
    for item, arr in m.threshold_se.items():
        assert_numeric_array(f"[PCM] threshold_se[{item}]", arr,
                             os.path.join(thr_se_dir, str(item)), verbose)
    assert_plot_lines(m, pcm_plot_calls(m), base, verbose)


# ---------------------------------------------------------------------------
# RSM
# ---------------------------------------------------------------------------

def build_rsm():
    np.random.seed(SIM_SEED)
    sim = RSM_Sim(no_of_items=N_ITEMS, no_of_persons=N_PERSONS,
                  max_score=MAX_SCORE)
    m = RSM(sim.scores, max_score=MAX_SCORE)
    m.calibrate()
    m.item_stats_df()
    m.person_stats_df()
    m.test_stats_df()
    m.person_abils()
    m.std_errors(no_of_samples=100)
    return m


def rsm_plot_calls(m):
    item = m.dataframe.columns[0]
    return [
        ("icc",               lambda: m.icc(item=item)),
        ("crcs",              lambda: m.crcs(item=item)),
        ("threshold_ccs",     lambda: m.threshold_ccs(item=item)),
        ("iic",               lambda: m.iic(item=item)),
        ("tcc",               lambda: m.tcc()),
        ("test_info",         lambda: m.test_info()),
        ("test_csem",         lambda: m.test_csem()),
        ("std_residuals_plot",lambda: m.std_residuals_plot()),
    ]


def generate_rsm(m):
    base = os.path.join(FIXTURE_DIR, "rsm")
    os.makedirs(base, exist_ok=True)
    save_df(m.item_stats,    os.path.join(base, "item_stats.csv"))
    save_df(m.person_stats,  os.path.join(base, "person_stats.csv"))
    save_df(m.test_stats,    os.path.join(base, "test_stats.csv"))
    save_series(m.person_abilities, os.path.join(base, "person_abilities.csv"))
    save_series(m.item_se,   os.path.join(base, "item_se.csv"))
    save_series(m.threshold_se if isinstance(m.threshold_se, pd.Series)
                else pd.Series(m.threshold_se),
                os.path.join(base, "threshold_se.csv"))
    # cat_width_se: Series (threshold -> float)
    save_series(m.cat_width_se, os.path.join(base, "cat_width_se.csv"))
    # threshold_se: dict of arrays by item
    thr_se_dir = os.path.join(base, "threshold_se_by_item")
    if isinstance(m.threshold_se, dict):
        os.makedirs(thr_se_dir, exist_ok=True)
        for item, arr in m.threshold_se.items():
            save_array(arr, os.path.join(thr_se_dir, str(item)))
    _save_plot_lines(m, rsm_plot_calls(m), base)
    print("  Fixtures saved to regression_data/rsm/")


def assert_rsm(m, verbose):
    base = os.path.join(FIXTURE_DIR, "rsm")
    assert_numeric_df("[RSM] item_stats",   m.item_stats,
                      os.path.join(base, "item_stats.csv"), verbose)
    assert_numeric_df("[RSM] person_stats", m.person_stats,
                      os.path.join(base, "person_stats.csv"), verbose)
    assert_numeric_df("[RSM] test_stats",   m.test_stats,
                      os.path.join(base, "test_stats.csv"), verbose)
    assert_numeric_series("[RSM] person_abilities", m.person_abilities,
                          os.path.join(base, "person_abilities.csv"), verbose)
    assert_numeric_series("[RSM] item_se", m.item_se,
                          os.path.join(base, "item_se.csv"), verbose)

    # threshold_se - Series or dict of arrays
    if isinstance(m.threshold_se, dict):
        thr_se_dir = os.path.join(base, "threshold_se_by_item")
        for item, arr in m.threshold_se.items():
            assert_numeric_array(f"[RSM] threshold_se[{item}]", arr,
                                 os.path.join(thr_se_dir, str(item)), verbose)
    else:
        assert_numeric_series("[RSM] threshold_se",
                              m.threshold_se if isinstance(m.threshold_se, pd.Series)
                              else pd.Series(m.threshold_se),
                              os.path.join(base, "threshold_se.csv"), verbose)

    # cat_width_se: Series (threshold -> float)
    assert_numeric_series("[RSM] cat_width_se", m.cat_width_se,
                          os.path.join(base, "cat_width_se.csv"), verbose)

    assert_plot_lines(m, rsm_plot_calls(m), base, verbose)


# ---------------------------------------------------------------------------
# Shared plot-line saving
# ---------------------------------------------------------------------------

def _save_plot_lines(m, plot_calls, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
# Main
# ---------------------------------------------------------------------------

MODELS = [
    ("SLM", build_slm, generate_slm, assert_slm),
    ("PCM", build_pcm, generate_pcm, assert_pcm),
    ("RSM", build_rsm, generate_rsm, assert_rsm),
]


def main():
    parser = argparse.ArgumentParser(description="SLM/PCM/RSM regression tests")
    parser.add_argument("--generate", action="store_true",
                        help="Generate known-good fixtures (run once)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print each passing check")
    args = parser.parse_args()

    if args.generate:
        os.makedirs(FIXTURE_DIR, exist_ok=True)
        print(f"Generating fixtures in {FIXTURE_DIR}/")

    for label, build_fn, generate_fn, assert_fn in MODELS:
        print(f"\n{'─' * 60}")
        print(f"  Model: {label}")
        print(f"{'─' * 60}")
        try:
            m = build_fn()
        except Exception:
            check(f"[{label}] build", False, traceback.format_exc())
            print(f"  Cannot continue - skipping {label}.")
            continue

        if args.generate:
            generate_fn(m)
        else:
            assert_fn(m, args.verbose)

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
ENDOFFILE