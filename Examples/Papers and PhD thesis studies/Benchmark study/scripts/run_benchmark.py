"""
Item 14 benchmark orchestrator: RaschPy vs eRm (CML) / TAM (MML, JML,
MFR) / pairwise (Choppin), ground-truth recovery across the 33-condition
grid (see conditions.py and the project_raschpy_benchmark_scope memory
file for the full design).

Per (model, condition, rep):
  1. Simulate data at known generating parameters (RaschPy's own *_Sim
     classes, deterministic seed from condition_seed()).
  2. Calibrate with RaschPy itself (calibrate() directly -- NOT
     recovery_stats(), which auto-triggers an unused 500-resample
     bootstrap; see run_one()'s own comment for the ~11.5s/rep this cost
     before being caught), then compute RMSE/r/r_squared/SD_ratio/
     OLS_slope/parameter_estimation_residual directly via
     _parameter_recovery_table()/_parameter_estimation_residual().
  3. Write the response data to a scratch CSV, shell out to the
     appropriate R script(s) for this model, parse their item-location
     estimates.
  4. Re-centre every package's item locations to zero-mean before
     computing origin-sensitive metrics (RMSE/OLS_slope/SD_ratio) --
     neutralises cross-package identification-convention differences
     (confirmed eRm's own "sum0" default does NOT zero-mean the derived
     item-location summary the way RaschPy/pairwise do).
  5. Also compute the parameter-estimation residual for each R package's
     estimates, reusing RaschPy's own shared
     Rasch._parameter_estimation_residual() with the R package's
     item/threshold values substituted in for "estimated".

Results are NOT persisted per-replication (1650+ raw datasets would
bloat the repo pointlessly, and are trivially regenerable from
condition_seed()) -- only the aggregated per-condition-per-algorithm
metric rows are written to results.csv.

Raw simulated data is generated on the fly in a scratch temp dir, not
committed anywhere.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPTS_DIR, "..", "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)

import raschpy as rp  # noqa: E402
from raschpy.simulation import MFRM_Sim, PCM_Sim, RSM_Sim, SLM_Sim  # noqa: E402

from conditions import ALGORITHMS, CONDITIONS, REPS, condition_seed  # noqa: E402

_SIM_CLASSES = {"slm": SLM_Sim, "pcm": PCM_Sim, "rsm": RSM_Sim, "mfrm": MFRM_Sim}


def _recentre(series):
    return series - series.mean()


def _item_metrics(generating, estimated):
    """RMSE/r/r_squared/SD_ratio/OLS_slope on zero-mean-recentred item locations."""
    g = _recentre(generating.astype(float))
    e = _recentre(estimated.reindex(generating.index).astype(float))
    valid = g.notna() & e.notna()
    g, e = g[valid].values, e[valid].values
    if len(g) < 3 or np.std(g) < 0.05:
        return dict(RMSE=np.nan, r=np.nan, r_squared=np.nan, SD_ratio=np.nan, OLS_slope=np.nan)
    r = float(np.corrcoef(g, e)[0, 1])
    ols_slope, _ = np.polyfit(g, e, 1)
    return dict(
        RMSE=float(np.sqrt(np.mean((e - g) ** 2))), r=r, r_squared=r ** 2,
        SD_ratio=float(np.std(e) / np.std(g)), OLS_slope=float(ols_slope),
    )


def _vector_metrics(generating, estimated, prefix):
    """RMSE/r/SD_ratio/OLS_slope on two aligned flat arrays, both demeaned
    (a common origin shift is not identified across packages)."""
    g = np.asarray(generating, dtype=float)
    e = np.asarray(estimated, dtype=float)
    ok = np.isfinite(g) & np.isfinite(e)
    g, e = g[ok] - g[ok].mean(), e[ok] - e[ok].mean()
    out = {f"{prefix}_{k}": np.nan for k in ("RMSE", "r", "SD_ratio", "OLS_slope")}
    if len(g) < 2 or np.std(g) < 0.05:
        return out
    out[f"{prefix}_RMSE"] = float(np.sqrt(np.mean((e - g) ** 2)))
    out[f"{prefix}_SD_ratio"] = float(np.std(e) / np.std(g))
    if len(g) >= 3:
        out[f"{prefix}_r"] = float(np.corrcoef(g, e)[0, 1])
        out[f"{prefix}_OLS_slope"] = float(np.polyfit(g, e, 1)[0])
    return out


def _tau_from_abs(abs_df):
    """Shared centred thresholds from absolute item-category thresholds:
    each item's thresholds centred on their mean, averaged over items
    (exact for a rating scale fit; a pooled estimate for per-item fits)."""
    centred = abs_df.sub(abs_df.mean(axis=1), axis=0)
    tau = centred.mean(axis=0)
    return tau - tau.mean()


def _threshold_metrics(model, sim, est_abs=None, est_tau=None):
    """Threshold recovery: absolute item-category thresholds (PCM, RSM;
    demeaned jointly) and the shared centred thresholds tau (RSM, MFRM)."""
    out = {}
    if model == "pcm":
        gen = sim.thresholds_uncentred
        gen.columns = range(1, gen.shape[1] + 1)
        # A package that drops an unused category returns fewer threshold
        # columns; aligning by position would then compare the wrong
        # thresholds, so such fits are not comparable (metrics stay NaN).
        if est_abs is not None and est_abs.shape[1] == gen.shape[1]:
            est = est_abs.reindex(index=gen.index, columns=gen.columns)
            out.update(_vector_metrics(gen.values.ravel(), est.values.ravel(), "thr"))
    elif model == "rsm":
        tau_gen = np.asarray(sim.thresholds, dtype=float)
        tau_gen = tau_gen - tau_gen.mean()
        gen = pd.DataFrame(np.asarray(sim.items, dtype=float)[:, None] + tau_gen[None, :],
                           index=sim.items.index, columns=range(1, len(tau_gen) + 1))
        if est_abs is not None and est_abs.shape[1] == gen.shape[1]:
            est = est_abs.reindex(index=gen.index, columns=gen.columns)
            out.update(_vector_metrics(gen.values.ravel(), est.values.ravel(), "thr"))
            if est_tau is None:
                est_tau = _tau_from_abs(est)
        if est_tau is not None and len(est_tau) == len(tau_gen):
            out.update(_vector_metrics(tau_gen, np.asarray(est_tau, dtype=float), "tau"))
    elif model == "mfrm" and est_tau is not None and len(est_tau) == len(np.asarray(sim.thresholds)):
        tau_gen = np.asarray(sim.thresholds, dtype=float)
        out.update(_vector_metrics(tau_gen - tau_gen.mean(), np.asarray(est_tau, dtype=float), "tau"))
    return out


def _save_est(model, condition, rep, algorithm, items, abs_thr, tau):
    """Append raw estimates to the pickle named by SAVE_ESTIMATES (for agreement analyses)."""
    import pickle
    path = os.environ["SAVE_ESTIMATES"]
    store = pickle.load(open(path, "rb")) if os.path.exists(path) else []
    store.append(dict(model=model, condition=condition, rep=rep, algorithm=algorithm,
                      items=pd.Series(items).astype(float), abs_thr=abs_thr, tau=tau))
    pickle.dump(store, open(path, "wb"))


def _read_abs_thresholds(path, columns=None):
    df = pd.read_csv(path).set_index("item")
    df = df[columns] if columns is not None else df.select_dtypes("number")
    df.columns = range(1, df.shape[1] + 1)
    return df


def _person_locations(dist, n, sd, seed):
    """Non-normal person locations with mean 0 and SD sd (shape only differs)."""
    rng = np.random.default_rng(seed + 7)
    if dist == "fat_tail":     # Student t, 3 df (variance 3), rescaled
        x = rng.standard_t(3, n) / np.sqrt(3.0)
    elif dist == "uniform":
        x = rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), n)
    elif dist == "bimodal":    # equal mixture at +/-0.9, component SD sqrt(0.19)
        x = rng.choice([-0.9, 0.9], n) + rng.normal(0.0, np.sqrt(0.19), n)
    # stronger departures (added 2026-10-09), standardised analytically to mean 0, SD 1
    elif dist == "skew_strong":   # lognormal, sigma 1 (skewness ~6.2)
        m, v = np.exp(0.5), (np.e - 1.0) * np.e
        x = (rng.lognormal(0.0, 1.0, n) - m) / np.sqrt(v)
    elif dist == "bimodal_wide":  # equal mixture at +/-0.95, component SD sqrt(1 - 0.95^2) ~ 0.31
        x = rng.choice([-0.95, 0.95], n) + rng.normal(0.0, np.sqrt(1 - 0.95 ** 2), n)
    elif dist == "mixture_8020":  # 80% at -0.6, 20% at +2.4 (mean 0), component SD 0.4, rescaled
        x = (np.where(rng.random(n) < 0.8, -0.6, 2.4) + rng.normal(0.0, 0.4, n)) / np.sqrt(1.44 + 0.16)
    else:
        raise ValueError(f"Unknown person_dist: {dist}")
    return list(x * sd)


def _build_sim(model, cond, seed):
    kwargs = {
        k: v for k, v in cond.items()
        if k not in ("label", "missing_prop", "mixed_k", "person_dist")
    }
    if "person_dist" in cond:
        kwargs["manual_persons"] = _person_locations(
            cond["person_dist"], cond["no_of_persons"], 1.5, seed,
        )
    if model == "pcm" and "mixed_k" in cond:
        kwargs = dict(kwargs)
        kwargs["max_score_vector"] = cond["mixed_k"]
        kwargs.pop("max_score", None)
    elif model == "pcm":
        kwargs = dict(kwargs)
        kwargs["max_score_vector"] = [kwargs.pop("max_score")] * kwargs["no_of_items"]
    kwargs["item_range"] = 2
    kwargs["seed"] = seed
    return _SIM_CLASSES[model](**kwargs)


def _apply_missingness(sim, missing_prop, seed):
    """Mutates sim.responses in place -- matches the pattern already
    validated this session (passing the sim object directly to a model
    constructor auto-attaches .generating and reads max_score(_vector)
    correctly; a separately-tracked responses variable was NOT confirmed
    to propagate the same way, so this avoids relying on that)."""
    if missing_prop <= 0:
        return
    rng = np.random.default_rng(seed)
    mask = rng.random(sim.responses.shape) < missing_prop
    sim.responses = sim.responses.mask(mask)


def _make_model(model, sim):
    if model == "slm":
        m = rp.SLM(sim)
        kw = dict(x.split("=") for x in os.environ.get("RASCHPY_SLM_KW", "").split(",") if x)
        m.calibrate(**kw)
    elif model == "pcm":
        m = rp.PCM(sim)
        m.calibrate()
    elif model == "rsm":
        m = rp.RSM(sim)
        # Optional alternative settings (e.g. RASCHPY_RSM_KW="method=cos,threshold_method=cpat-info-weighted")
        kw = dict(x.split("=") for x in os.environ.get("RASCHPY_RSM_KW", "").split(",") if x)
        m.calibrate(**kw)
    elif model == "mfrm":
        m = rp.MFRM(sim)
        m.calibrate(model="global")
    return m


def _run_r_script(args_list):
    result = subprocess.run(
        ["Rscript"] + args_list, capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"R script failed: {' '.join(args_list)}\n{result.stderr}")


# In-R fitting time of the last external fit (the model-fitting call
# only, timed inside R by each fit_*.R script); NaN if not reported.
_LAST_FIT_SECONDS = np.nan


def _external_item_estimate(model, algorithm, responses_csv, max_score_list, tmp_dir):
    """Returns (item locations, absolute thresholds DataFrame or None,
    shared centred thresholds or None)."""
    prefix = os.path.join(tmp_dir, f"ext_{algorithm}")
    for f in os.listdir(tmp_dir):
        if f.startswith(f"ext_{algorithm}"):
            os.remove(os.path.join(tmp_dir, f))
    abs_thr, tau = None, None
    poly = model in ("pcm", "rsm")
    if algorithm == "erm_cml":
        _run_r_script([
            os.path.join(_SCRIPTS_DIR, "fit_erm.R"), model, responses_csv, prefix,
        ])
        if model == "slm":
            df = pd.read_csv(f"{prefix}_items.csv")
            return df.set_index("item")["location"], None, None
        df = pd.read_csv(f"{prefix}_items_thresholds.csv")
        cols = [c for c in df.columns if c.startswith("Threshold")]
        abs_thr = _read_abs_thresholds(f"{prefix}_items_thresholds.csv", cols)
        return df.set_index("item")["Location"], abs_thr, None
    elif algorithm in ("tam_mml", "tam_jml"):
        estimator = "mml" if algorithm == "tam_mml" else "jml"
        _run_r_script([
            os.path.join(_SCRIPTS_DIR, "fit_tam.R"), estimator, model, responses_csv, prefix,
        ])
        df = pd.read_csv(f"{prefix}_items.csv")
        if poly:
            abs_thr = _read_abs_thresholds(f"{prefix}_thresholds_abs.csv")
        return df.set_index("item")["location"], abs_thr, None
    elif algorithm == "pairwise":
        max_score_arg = ",".join(str(k) for k in max_score_list)
        _run_r_script([
            os.path.join(_SCRIPTS_DIR, "fit_pairwise.R"), model, responses_csv,
            max_score_arg, prefix,
        ])
        df = pd.read_csv(f"{prefix}_items.csv")
        if poly:
            abs_thr = _read_abs_thresholds(f"{prefix}_thresholds.csv")
        return df.set_index("item")["location"], abs_thr, None
    elif algorithm == "tam_mfr":
        _run_r_script([
            os.path.join(_SCRIPTS_DIR, "fit_tam.R"), "mfr", "global", responses_csv, prefix,
        ])
        df = pd.read_csv(f"{prefix}_items.csv")
        tau = pd.read_csv(f"{prefix}_tau.csv")["tau"].values
        return df.set_index("item")["location"], None, tau
    elif algorithm == "psychotools_cml":
        _run_r_script([
            os.path.join(_SCRIPTS_DIR, "fit_psychotools.R"), model, responses_csv, prefix,
        ])
        df = pd.read_csv(f"{prefix}_items.csv")
        if poly:
            abs_thr = _read_abs_thresholds(f"{prefix}_thresholds_abs.csv")
        return df.set_index("item")["location"], abs_thr, None
    elif algorithm == "mirt_mml":
        _run_r_script([
            os.path.join(_SCRIPTS_DIR, "fit_mirt.R"), model, responses_csv, prefix,
        ])
        df = pd.read_csv(f"{prefix}_items.csv")
        if poly:
            abs_thr = _read_abs_thresholds(f"{prefix}_thresholds_abs.csv")
        return df.set_index("item")["location"], abs_thr, None
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def run_one(model, cond, rep, tmp_dir, algorithms=None):
    seed = condition_seed(model, cond["label"], rep)
    sim = _build_sim(model, cond, seed)
    _apply_missingness(sim, cond.get("missing_prop", 0.0), seed)

    rows = []
    # Unused score categories (item-category cells for PCM, pooled
    # categories otherwise): thresholds next to them are not identified.
    x = sim.responses.values.astype(float)
    if model == "pcm":
        n_empty = int(sum((x[:, i] == k).sum() == 0
                          for i, mk in enumerate(sim.max_score_vector) for k in range(mk + 1)))
    elif model in ("rsm", "mfrm"):
        n_empty = int(sum((x == k).sum() == 0 for k in range(sim.max_score + 1)))
    else:
        n_empty = 0

    # RaschPy's own fit, wrapped in the SAME try/except pattern as every
    # external package below -- failure-rate comparison is a first-class
    # metric here (not just accuracy-when-successful), so RaschPy's own
    # path must be symmetric with the others, not silently assumed to
    # always succeed. Previously this call wasn't guarded at all, which
    # would have crashed the whole overnight run on a single pathological
    # seed rather than recording one failed row and continuing.
    #
    # Deliberately does NOT call recovery_stats() -- that convenience
    # wrapper auto-triggers a 500-resample bootstrap (std_errors() /
    # fit_statistics()) to populate the z/p/Flagged columns, which this
    # benchmark never uses (only RMSE/r/r_squared/SD_ratio/OLS_slope,
    # computed straight from the generating/estimated arrays, no SE
    # needed). Measured directly: calibrate() alone is ~0.01s on a
    # moderate PCM condition (closed-form PAIR/CPAT, as expected) but
    # the unused bootstrap cost ~11.5s on top of it -- almost the entire
    # per-rep runtime was this avoidable step, not RaschPy's own
    # estimation. _parameter_recovery_table/_parameter_estimation_residual
    # are called directly instead, se=None, bypassing the bootstrap
    # entirely.
    try:
        t0 = time.perf_counter()
        m = _make_model(model, sim)
        elapsed = time.perf_counter() - t0

        items_table, items_summary = m._parameter_recovery_table(
            sim.items, m.items, se=None,
        )
        raschpy_summary = items_summary["Overall"]
        pe_residual = m._parameter_estimation_residual(
            sim.persons,
            lambda theta, item: m.exp_score(theta, sim.items[item])
            if model == "slm" else np.nan,
            lambda theta, item: m.exp_score(theta, m.items[item])
            if model == "slm" else np.nan,
        ) if model == "slm" else np.nan

        if model == "pcm":
            rp_thr = _threshold_metrics(model, sim, est_abs=m.thresholds_uncentred.set_axis(
                range(1, m.thresholds_uncentred.shape[1] + 1), axis=1))
        elif model == "rsm":
            tau_est = np.asarray(m.thresholds, dtype=float)
            rp_abs = pd.DataFrame(m.items.reindex(sim.items.index).values[:, None] + tau_est[None, :],
                                  index=sim.items.index, columns=range(1, len(tau_est) + 1))
            rp_thr = _threshold_metrics(model, sim, est_abs=rp_abs, est_tau=tau_est)
        elif model == "mfrm":
            rp_thr = _threshold_metrics(model, sim, est_tau=np.asarray(m.thresholds, dtype=float))
        else:
            rp_thr = {}
        rp_thr["empty_categories"] = n_empty
        if os.environ.get("SAVE_ESTIMATES"):
            _save_est(model, cond["label"], rep, "raschpy", m.items,
                      rp_abs if model == "rsm" else None,
                      np.asarray(m.thresholds, dtype=float) if model in ("rsm", "mfrm") else None)
        rows.append(dict(
            model=model, condition=cond["label"], rep=rep, algorithm="raschpy",
            converged=True, error=None, time_seconds=elapsed, fit_seconds=elapsed,
            RMSE=raschpy_summary.get("RMSE"), r=raschpy_summary.get("r"),
            r_squared=raschpy_summary.get("r_squared"), SD_ratio=raschpy_summary.get("SD_ratio"),
            OLS_slope=raschpy_summary.get("OLS_slope"),
            parameter_estimation_residual=pe_residual, **rp_thr,
        ))
    except Exception as exc:  # noqa: BLE001
        rows.append(dict(
            model=model, condition=cond["label"], rep=rep, algorithm="raschpy",
            converged=False, error=str(exc)[:200], time_seconds=np.nan,
            RMSE=np.nan, r=np.nan, r_squared=np.nan, SD_ratio=np.nan,
            OLS_slope=np.nan, parameter_estimation_residual=np.nan,
        ))
        # RaschPy's own fit failing means there's no self.items to compare
        # external packages against either -- nothing further to do for
        # this (model, condition, rep).
        return rows

    responses_csv = os.path.join(tmp_dir, "responses.csv")
    if model == "mfrm":
        # MFRM's own responses is (rater, person)-MultiIndex-wide -- the
        # R side (fit_tam.R's mfr branch) expects long format instead:
        # one row per (rater, person), pid/rater columns plus one column
        # per item. Confirmed this exact layout against a toy MFRM_Sim
        # dataset before relying on it (see the R-side validation notes).
        long = sim.responses.reset_index()
        long.columns = ["rater", "pid"] + list(long.columns[2:])
        long.to_csv(responses_csv, index=False)
    else:
        sim.responses.to_csv(responses_csv)

    # Per-item max_score, as a list -- PCM allows a different max_score
    # per item (the whole point of the mixed-K condition), so a single
    # scalar is wrong in general; RSM/SLM just get a uniform list.
    # getattr(sim, "max_score", 1) was the original (buggy) approach --
    # PCM_Sim has no .max_score attribute at all (only
    # .max_score_vector), so it silently fell back to 1 for every PCM
    # condition, corrupting fit_pairwise.R's category count. Caught via
    # a real "R script failed" error on every PCM condition, not
    # anticipated in advance.
    if model == "pcm":
        max_score_list = list(sim.max_score_vector)
    elif model in ("rsm", "mfrm"):
        max_score_list = [sim.max_score] * sim.no_of_items
    else:
        max_score_list = [1] * sim.no_of_items

    for algorithm in (algorithms if algorithms is not None else ALGORITHMS[model]):
        try:
            t0 = time.perf_counter()
            global _LAST_FIT_SECONDS
            _LAST_FIT_SECONDS = np.nan
            est_items, est_abs, est_tau = _external_item_estimate(
                model, algorithm, responses_csv, max_score_list, tmp_dir,
            )
            fit_time_file = os.path.join(tmp_dir, f"ext_{algorithm}_fit_seconds.txt")
            if os.path.exists(fit_time_file):
                with open(fit_time_file) as fh:
                    _LAST_FIT_SECONDS = float(fh.read().strip())
            elapsed = time.perf_counter() - t0
            # A package can fail to converge WITHOUT raising (e.g. eRm's
            # RSM() hit a singular Hessian on one seed -- crashed outright
            # with se=TRUE, but even se=FALSE just silently returned
            # wildly divergent point estimates, thousands of logits off).
            # No real Rasch item location is ever remotely this large --
            # treat it as a non-convergent run (NaN), not a genuine
            # 2500-logit RMSE that would corrupt the aggregate stats for
            # this condition. Found via a real eRm crash on RSM baseline
            # rep 2, not anticipated in advance.
            if not np.isfinite(est_items.astype(float)).all() or est_items.astype(float).abs().max() > 50:
                raise RuntimeError(
                    f"{algorithm} produced non-finite or implausible "
                    f"(|value| > 50) estimates -- treated as non-convergent."
                )
        except Exception as exc:  # noqa: BLE001
            rows.append(dict(
                model=model, condition=cond["label"], rep=rep, algorithm=algorithm,
                converged=False, error=str(exc)[:200],
                time_seconds=time.perf_counter() - t0,
                RMSE=np.nan, r=np.nan, r_squared=np.nan, SD_ratio=np.nan,
                OLS_slope=np.nan, parameter_estimation_residual=np.nan,
            ))
            continue

        if os.environ.get("SAVE_ESTIMATES"):
            _save_est(model, cond["label"], rep, algorithm, est_items, est_abs,
                      est_tau if est_tau is not None else (_tau_from_abs(est_abs) if est_abs is not None and model == "rsm" else None))
        metrics = _item_metrics(sim.items, est_items)
        try:
            metrics.update(_threshold_metrics(model, sim, est_abs=est_abs, est_tau=est_tau))
        except Exception:  # noqa: BLE001 -- threshold metrics stay NaN
            pass
        metrics["empty_categories"] = n_empty

        # Parameter-estimation residual against this package's estimate,
        # reusing RaschPy's own shared helper with the external item
        # vector substituted in place of self.items. Item-location-only
        # (no threshold substitution) -- deliberately matches this
        # benchmark's item-location-focused scope; a separate
        # threshold-level comparison is not attempted here given each
        # package's own threshold output shape differs non-trivially.
        #
        # exp_score() is a NONLINEAR function of item location, unlike
        # the RMSE/SD_ratio/OLS_slope comparison above (which is origin-
        # shift-invariant by construction via symmetric de-meaning on
        # both sides) -- an uncorrected identification-convention offset
        # here would genuinely distort the residual, not just cancel
        # out. Re-centre this package's item locations onto the
        # generating scale's own actual mean (NOT assumed to be exactly
        # 0 -- confirmed empirically it isn't, for PCM's mixed-K
        # condition specifically) before using them.
        gen_items_ = sim.items
        est_recentred = est_items - est_items.mean() + gen_items_.mean()

        def _exp_score_est(theta, item, _est=est_recentred):
            loc = _est.get(item, np.nan)
            if pd.isna(loc):
                return np.nan
            if model == "slm":
                return m.exp_score(theta, loc)
            return np.nan  # threshold-dependent models: item-location-only proxy not meaningful

        if model == "slm":
            pe_residual = m._parameter_estimation_residual(
                sim.persons,
                lambda theta, item: m.exp_score(theta, gen_items_[item]),
                _exp_score_est,
            )
        else:
            pe_residual = np.nan

        rows.append(dict(
            model=model, condition=cond["label"], rep=rep, algorithm=algorithm,
            converged=True, error=None, time_seconds=elapsed,
            fit_seconds=_LAST_FIT_SECONDS,
            **metrics, parameter_estimation_residual=pe_residual,
        ))

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument(
        "--algorithms", nargs="+", default=None,
        help="Restrict to these external algorithms only (RaschPy's own "
             "row is always computed). Use when adding a single new "
             "algorithm to an existing results.csv without recomputing "
             "everything else already in it.",
    )
    parser.add_argument("--out", default=os.path.join(_SCRIPTS_DIR, "..", "results.csv"))
    args = parser.parse_args()

    all_rows = []
    with tempfile.TemporaryDirectory(prefix="raschpy_benchmark_") as tmp_dir:
        for model in args.models:
            conds = CONDITIONS[model]
            if args.conditions:
                conds = [c for c in conds if c["label"] in args.conditions]
            for cond in conds:
                for rep in range(1, args.reps + 1):
                    print(f"{model} / {cond['label']} / rep {rep}")
                    rows = run_one(model, cond, rep, tmp_dir, algorithms=args.algorithms)
                    all_rows.extend(rows)
                    pd.DataFrame(all_rows).to_csv(args.out, index=False)

    print(f"Done. Results written to {args.out}")


if __name__ == "__main__":
    main()
