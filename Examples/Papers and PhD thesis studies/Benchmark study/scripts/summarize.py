"""
Summarise results.csv: failure rate per (model, algorithm) and per
(model, condition, algorithm), plus accuracy metrics conditional on
convergence (a failed run contributes to the failure-rate denominator but
not to the accuracy-metric mean/SD -- the two are deliberately reported
separately, not blended into one number, since "how often does it work"
and "how good is it when it does" are different claims).

Usage: python summarize.py <results_csv>
"""

import sys

import pandas as pd


def summarize(results_csv):
    df = pd.read_csv(results_csv)

    print("=" * 70)
    print("Failure rate by model x algorithm (across all conditions/reps)")
    print("=" * 70)
    overall = df.groupby(["model", "algorithm"]).agg(
        n_runs=("converged", "size"),
        n_failed=("converged", lambda s: (~s.astype(bool)).sum()),
    )
    overall["failure_rate"] = (overall["n_failed"] / overall["n_runs"]).round(4)
    print(overall.to_string())

    print()
    print("=" * 70)
    print("Failure rate by model x condition x algorithm")
    print("=" * 70)
    by_cond = df.groupby(["model", "condition", "algorithm"]).agg(
        n_runs=("converged", "size"),
        n_failed=("converged", lambda s: (~s.astype(bool)).sum()),
    )
    by_cond["failure_rate"] = (by_cond["n_failed"] / by_cond["n_runs"]).round(4)
    # Only show rows with at least one failure -- a clean 0% row for
    # every condition would just be noise here.
    print(by_cond[by_cond["n_failed"] > 0].to_string())

    print()
    print("=" * 70)
    print("Accuracy metrics, conditional on convergence (failed runs excluded)")
    print("Mean/std alone hide tail risk (e.g. a rare catastrophic outlier can")
    print("sit behind an unremarkable mean) -- percentiles are reported in")
    print("whichever direction is actually the 'bad' direction for that metric.")
    print("=" * 70)
    converged = df[df["converged"].astype(bool)]

    # One-tailed, bad = high (unbounded above, 0 is ideal): report the
    # upper tail, where rare catastrophic failures show up.
    for metric in ["RMSE", "parameter_estimation_residual"]:
        print(f"\n--- {metric} (upper tail = bad; ideal = 0) ---")
        tab = converged.groupby(["model", "algorithm"])[metric].agg(
            ["mean", "std", "median",
             lambda s: s.quantile(0.95), lambda s: s.quantile(0.99), "max"],
        )
        tab.columns = ["mean", "std", "median", "p95", "p99", "max"]
        print(tab.to_string())

    # One-tailed, bad = low (bounded above at 1, 1 is ideal): report the
    # lower tail.
    print("\n--- r_squared (lower tail = bad; ideal = 1) ---")
    tab = converged.groupby(["model", "algorithm"])["r_squared"].agg(
        ["mean", "std", "median",
         lambda s: s.quantile(0.05), lambda s: s.quantile(0.01), "min"],
    )
    tab.columns = ["mean", "std", "median", "p05", "p01", "min"]
    print(tab.to_string())

    # Two-tailed, bad = far from 1 in EITHER direction (shrinkage <1 or
    # inflation >1 both meaningful): report both tails.
    for metric in ["SD_ratio", "OLS_slope"]:
        print(f"\n--- {metric} (both tails = bad; ideal = 1) ---")
        tab = converged.groupby(["model", "algorithm"])[metric].agg(
            ["mean", "std", "median",
             lambda s: s.quantile(0.025), lambda s: s.quantile(0.975),
             lambda s: s.quantile(0.005), lambda s: s.quantile(0.995)],
        )
        tab.columns = ["mean", "std", "median", "p2.5", "p97.5", "p0.5", "p99.5"]
        print(tab.to_string())

    print()
    print("=" * 70)
    print("Wall-clock time per run, seconds (all attempts, converged or not)")
    print("=" * 70)
    timing = df.groupby(["model", "algorithm"])["time_seconds"].agg(["mean", "std", "median"])
    print(timing.to_string())


if __name__ == "__main__":
    summarize(sys.argv[1])
