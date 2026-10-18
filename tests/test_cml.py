"""
Tests for conditional maximum likelihood (estimation='cml').

External agreement uses stored psychotools::raschmodel fits (tests/external_data/cml_*_psychotools.csv,
made by cml_fit_psychotools.R), so no R is needed at run time. The stored fits stop at psychotools' own
optimiser tolerance (gradient ~1e-4), hence the tolerances below.
"""

import itertools
import os
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from raschpy import PCM, RSM, SLM, _esf

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "external_data")


def _slm_cml(name):
    X = pd.read_csv(os.path.join(DATA, f"cml_{name}.csv"))
    ref = pd.read_csv(os.path.join(DATA, f"cml_{name}_psychotools.csv"))
    m = SLM(X)
    m.calibrate(estimation="cml")
    return m, ref


@pytest.mark.parametrize("name", ["slm_complete", "slm_missing", "slm_booklet"])
def test_slm_cml_matches_psychotools(name):
    m, ref = _slm_cml(name)
    m.std_errors()
    np.testing.assert_allclose(m.items.reindex(ref["item"]).values, ref["psy"].values, atol=5e-5)
    np.testing.assert_allclose(m.item_se.reindex(ref["item"]).values, ref["psy_se"].values, atol=5e-6)
    assert m.convergence["log_likelihood"] == pytest.approx(ref["psy_ll"][0], abs=1e-6)
    assert m.convergence["converged"] and m.se_method == "hessian"


def test_slm_cml_is_the_conditional_maximum():
    m, _ = _slm_cml("slm_booklet")
    X = np.array(m.responses, dtype=float)
    _, g, _ = _esf.cond_loglik(X, [[v] for v in m.items.values])
    assert np.max(np.abs(g - g.mean())) < 1e-8
    assert abs(m.items.mean()) < 1e-12


def test_slm_cml_does_not_depend_on_pair_settings():
    m, _ = _slm_cml("slm_missing")
    m2 = SLM(m.responses)
    m2.calibrate(estimation="cml", method="cos", constant=1.0)
    np.testing.assert_allclose(m2.items.values, m.items.values, atol=1e-7)


def test_slm_cml_drops_items_without_estimates():
    X = pd.read_csv(os.path.join(DATA, "cml_slm_complete.csv"))
    X.iloc[:, 0] = 1  # an item everyone answers correctly
    m = SLM(X, validate=False)
    with pytest.warns(UserWarning, match="CML estimates do not exist for 1 item"):
        m.calibrate(estimation="cml")
    assert m.convergence["dropped_items"] == {"I1": "unused category 0"}
    assert list(m.cml_dropped_responses.columns) == ["I1"] and "I1" not in m.items.index
    ref = SLM(X.iloc[:, 1:], validate=False)
    ref.calibrate(estimation="cml")
    np.testing.assert_allclose(m.items.values, ref.items.values, atol=1e-10)
    m.std_errors()
    assert m.item_se.notna().all()


def test_se_method_rules():
    m, _ = _slm_cml("slm_complete")
    with pytest.raises(ValueError, match="not yet available for estimation='cml'"):
        m.std_errors(se_method="sandwich")
    p = SLM(m.responses)
    p.calibrate()
    with pytest.raises(ValueError, match="covariance of estimation='cml'"):
        p.std_errors(se_method="hessian")
    with pytest.raises(ValueError, match="estimation must be"):
        p.calibrate(estimation="jml")


def test_slm_cml_refits_use_cml():
    m, _ = _slm_cml("slm_complete")
    assert m._calibration_kw()["estimation"] == "cml"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.std_errors(se_method="bootstrap", no_of_samples=5, seed=1)
    assert m.se_method == "bootstrap" and m.item_bootstrap.shape[0] == 5


# ---------------------------------------------------------------- PCM

def _pcm_cml(name):
    X = pd.read_csv(os.path.join(DATA, f"cml_{name}.csv"))
    m = PCM(X)
    m.calibrate(estimation="cml")
    return m


def _stacked(m):
    return np.concatenate([m.thresholds_uncentred.loc[n].dropna().values for n in m.item_names])


@pytest.mark.parametrize("name", ["pcm_complete", "pcm_missing"])
def test_pcm_cml_matches_psychotools(name):
    m = _pcm_cml(name)
    ref = pd.read_csv(os.path.join(DATA, f"cml_{name}_psychotools.csv"))
    m.std_errors()
    np.testing.assert_allclose(_stacked(m), ref["psy"].values, atol=5e-5)
    np.testing.assert_allclose(np.concatenate([m.threshold_se[n] for n in m.item_names]), ref["psy_se"].values,
                               atol=5e-6)
    assert m.convergence["log_likelihood"] == pytest.approx(ref["psy_ll"][0], abs=1e-6)
    assert m.se_method == "hessian" and abs(_stacked(m).mean()) < 1e-12


def test_pcm_cml_booklet_matches_brute_force():
    # psychotools::pcmodel fails silently on this design (positive log-likelihood, code 0), so the reference is
    # brute-force enumeration of each booklet's response patterns
    m = _pcm_cml("pcm_booklet")
    X = np.array(m.responses, float)
    Ks = m.max_score_vector.values.astype(int)
    b = _stacked(m)
    off = np.concatenate([[0], np.cumsum(Ks)])
    le = [np.concatenate([[0.0], -np.cumsum(b[off[i]:off[i + 1]])]) for i in range(len(Ks))]
    obs = ~np.isnan(X)
    total = 0.0
    for pattern in {tuple(o) for o in obs}:
        items = [i for i in range(len(Ks)) if pattern[i]]
        rows = X[np.all(obs == np.array(pattern), axis=1)][:, items].astype(int)
        by_total = {}
        for cs in itertools.product(*[range(Ks[i] + 1) for i in items]):
            by_total.setdefault(sum(cs), []).append(sum(le[i][c] for i, c in zip(items, cs)))
        lg = {r: logsumexp(v) for r, v in by_total.items()}
        for row in rows:
            r = row.sum()
            if 0 < r < Ks[items].sum():
                total += sum(le[i][x] for i, x in zip(items, row)) - lg[r]
    assert m.convergence["log_likelihood"] == pytest.approx(total, abs=1e-8)
    _, g, _ = _esf.cond_loglik(X, [b[off[i]:off[i + 1]] for i in range(len(Ks))])
    assert np.max(np.abs(g - g.mean())) < 1e-8


def test_pcm_cml_drops_item_with_unused_category():
    X = pd.read_csv(os.path.join(DATA, "cml_pcm_complete.csv"))
    X.loc[X["I1"] == 2, "I1"] = 3  # category 2 of I1 unused
    m = PCM(X, validate=False)
    with pytest.warns(UserWarning, match="CML estimates do not exist for 1 item"):
        m.calibrate(estimation="cml")
    assert m.convergence["dropped_items"] == {"I1": "unused category 2"}
    assert "I1" not in m.items.index and "I1" not in m.max_score_vector.index
    m.std_errors()
    assert np.all(np.isfinite(m.item_se.values))



# ---------------------------------------------------------------- RSM

def _brute_cond_ll(X, b_rows):
    """Conditional log-likelihood by enumerating every response pattern of each missingness pattern."""
    Ks = [len(x) for x in b_rows]
    le = [np.concatenate([[0.0], -np.cumsum(x)]) for x in b_rows]
    obs = ~np.isnan(X)
    total = 0.0
    for pattern in {tuple(o) for o in obs}:
        items = [i for i in range(len(Ks)) if pattern[i]]
        rows = X[np.all(obs == np.array(pattern), axis=1)][:, items].astype(int)
        by_total = {}
        for cs in itertools.product(*[range(Ks[i] + 1) for i in items]):
            by_total.setdefault(sum(cs), []).append(sum(le[i][c] for i, c in zip(items, cs)))
        lg = {r: logsumexp(v) for r, v in by_total.items()}
        for row in rows:
            r = row.sum()
            if 0 < r < sum(Ks[i] for i in items):
                total += sum(le[i][x] for i, x in zip(items, row)) - lg[r]
    return total


def _rsm_cml(name):
    X = pd.read_csv(os.path.join(DATA, f"cml_{name}.csv"))
    m = RSM(X)
    m.calibrate(estimation="cml")
    return m


@pytest.mark.parametrize("name", ["rsm_complete", "rsm_missing"])
def test_rsm_cml_matches_psychotools(name):
    # psychotools' threshold standard errors (vcov of threshpar(relative = TRUE)) are not compared: in a
    # replicate simulation of rsm_complete they were about twice the sampling SD for all but the last
    # threshold, while RaschPy's matched it
    m = _rsm_cml(name)
    ref = pd.read_csv(os.path.join(DATA, f"cml_{name}_psychotools.csv"))
    items, thr = ref[ref["kind"] == "item"], ref[ref["kind"] == "threshold"]
    m.std_errors()
    np.testing.assert_allclose(m.items.values, items["psy"].values, atol=5e-5)
    np.testing.assert_allclose(m.thresholds.values, thr["psy"].values, atol=5e-5)
    np.testing.assert_allclose(m.item_se.values, items["psy_se"].values, atol=5e-6)
    assert m.convergence["log_likelihood"] == pytest.approx(ref["psy_ll"].iloc[0], abs=1e-6)
    assert m.se_method == "hessian" and abs(m.items.mean()) < 1e-12 and abs(m.thresholds.mean()) < 1e-12


def test_rsm_cml_booklet_matches_brute_force():
    # psychotools::rsmodel fails silently on this design (positive log-likelihood, code 0)
    m = _rsm_cml("rsm_booklet")
    X = np.array(m.responses, float)
    b = m.items.values[:, None] + m.thresholds.values[None, :]
    assert m.convergence["log_likelihood"] == pytest.approx(_brute_cond_ll(X, list(b)), abs=1e-8)
    A, _ = m._cml_basis()
    _, g, _ = _esf.cond_loglik(X, list(b))
    assert np.max(np.abs(A.T @ g)) < 1e-8


def test_rsm_cml_keeps_items_with_unused_categories():
    X = pd.read_csv(os.path.join(DATA, "cml_rsm_complete.csv"))
    X.loc[X["I1"] == 3, "I1"] = 2  # I1 never uses the top category: fine for RSM (shared thresholds)
    m = RSM(X, validate=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(estimation="cml")
    assert m.convergence["dropped_items"] == {} and "I1" in m.items.index


def test_rsm_cml_raises_when_a_category_is_unused_everywhere():
    X = pd.read_csv(os.path.join(DATA, "cml_rsm_complete.csv"))
    X = X.where(X != 2, 1)  # category 2 unused on every item
    m = RSM(X, validate=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="category 2 is unused"):
            m.calibrate(estimation="cml")


# ---------------------------------------------------------------- Andersen LR (global_fit)

@pytest.mark.parametrize("cls,name", [(SLM, "slm_complete"), (PCM, "pcm_complete"), (RSM, "rsm_complete")])
def test_andersen_lr_is_the_conditional_likelihood_ratio(cls, name):
    X = pd.read_csv(os.path.join(DATA, f"cml_{name}.csv"))
    m = cls(X)
    m.calibrate(estimation="cml")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.global_fit(method="andersen-lr", split="random", seed=3)
    groups = m._global_fit_groups("random", None, 3)
    items = [c for c in m.item_names if c not in m.global_fit_excluded]
    fits = [m._global_fit_subset(groups[n], items=items) for n in groups]
    pooled = m._global_fit_subset(groups["A"].append(groups["B"]), items=items)
    lr = 2 * (sum(f.convergence["log_likelihood"] for f in fits) - pooled.convergence["log_likelihood"])
    assert r["statistic"] == pytest.approx(lr, abs=1e-8) and r["statistic"] >= 0
    assert r["df"] == pooled._cml_free_parameters()


def test_andersen_lr_needs_cml():
    m = SLM(pd.read_csv(os.path.join(DATA, "cml_slm_complete.csv")))
    m.calibrate()
    with pytest.raises(ValueError, match="calibrate\\(estimation='cml'\\)"):
        m.global_fit(method="andersen-lr")


# ---------------------------------------------------------------- Martin-Lof (global_fit)

@pytest.mark.parametrize(
    "cls,name,lr,df",
    [
        (SLM, "slm_complete", 62.5736, 55),   # eRm::MLoef: 62.5736, df 55
        (PCM, "pcm_complete", 149.356, 224),  # eRm::MLoef: 149.356, df 224
        (RSM, "rsm_complete", 193.0884, 226),  # eRm::MLoef: 193.0884, df 224 (omits the K - 1 extra widths)
    ],
)
def test_martin_lof_matches_erm(cls, name, lr, df):
    X = pd.read_csv(os.path.join(DATA, f"cml_{name}.csv"))
    m = cls(X)
    m.calibrate(estimation="cml")
    r = m.global_fit(method="martin-lof", split=list(X.columns[: X.shape[1] // 2]))
    assert r["statistic"] == pytest.approx(lr, abs=1e-3) and r["df"] == df


def test_martin_lof_monte_carlo():
    X = pd.read_csv(os.path.join(DATA, "cml_slm_complete.csv"))
    m = SLM(X)
    m.calibrate(estimation="cml")
    r1 = m.global_fit(method="martin-lof", p_method="monte-carlo", no_of_samples=20, seed=4)
    r2 = m.global_fit(method="martin-lof", p_method="monte-carlo", no_of_samples=20, seed=4)
    assert r1["p-value"] == r2["p-value"] and 0 < r1["p-value"] <= 1
    assert r1["Monte Carlo samples"] + r1["Monte Carlo samples skipped"] == 20
    with pytest.raises(ValueError, match="p_method"):
        m.global_fit(method="martin-lof", p_method="exact")


# ---------------------------------------------------------------- solvers

@pytest.mark.parametrize("cls,name", [(SLM, "slm_missing"), (PCM, "pcm_missing"), (RSM, "rsm_missing")])
def test_cml_solvers_reach_the_same_maximum(cls, name):
    X = pd.read_csv(os.path.join(DATA, f"cml_{name}.csv"))
    fits = {}
    for solver in ("newton", "chord", "lbfgs"):
        m = cls(X)
        m.calibrate(estimation="cml", solver=solver)
        m.std_errors()
        fits[solver] = m
        assert m.convergence["converged"] and m.convergence["solver"] == solver
    ref = fits["newton"]
    for solver in ("chord", "lbfgs"):
        m = fits[solver]
        assert m.convergence["log_likelihood"] == pytest.approx(ref.convergence["log_likelihood"], abs=1e-6)
        np.testing.assert_allclose(m.items.values, ref.items.values, atol=1e-4)
        np.testing.assert_allclose(np.asarray(m.item_se), np.asarray(ref.item_se), atol=1e-5)
    assert fits["chord"].convergence["hessians"] == 1


def test_solver_rules():
    X = pd.read_csv(os.path.join(DATA, "cml_slm_complete.csv"))
    m = SLM(X)
    with pytest.raises(ValueError, match="solver must be"):
        m.calibrate(estimation="cml", solver="bfgs")
    with pytest.warns(UserWarning, match="applies to estimation='cml' only"):
        m.calibrate(solver="chord")
    m.calibrate(estimation="cml", solver="chord")
    assert m._calibration_kw()["solver"] == "chord"


# ---------------------------------------------------------------- conditional fit

@pytest.mark.parametrize("cls,name", [(SLM, "slm_complete"), (PCM, "pcm_complete")])
def test_conditional_fit_matches_iarm(cls, name):
    X = pd.read_csv(os.path.join(DATA, f"cml_{name}.csv"))
    m = cls(X)
    m.calibrate(estimation="cml")
    m.fit_statistics(conditional=True, se=False, conditional_samples=20, seed=1)
    fit = pd.read_csv(os.path.join(DATA, f"cml_{name}_iarm_fit.csv")).set_index("item")
    rest = pd.read_csv(os.path.join(DATA, f"cml_{name}_iarm_rest.csv")).set_index("item")
    # iarm uses eRm's estimates (optimiser tolerance ~1e-5) and prints restscore results to 3 digits
    np.testing.assert_allclose(m.item_outfit_ms.reindex(fit.index), fit["outfit"], atol=2e-5)
    np.testing.assert_allclose(m.item_infit_ms.reindex(fit.index), fit["infit"], atol=2e-5)
    V, M4 = m.info_df, m.kurtosis_df
    n = m.std_residual_df.notna().sum()
    q_out = (((M4 / V**2).sum() / n**2) - 1 / n) ** 0.5
    np.testing.assert_allclose(q_out.reindex(fit.index), fit["outfit_se"], atol=5e-6)
    t = m.item_restscore_table.reindex(rest.index)
    np.testing.assert_allclose(t["Observed gamma"], rest["observed"], atol=5e-4)
    np.testing.assert_allclose(t["Expected gamma"], rest["expected"], atol=5e-4)
    np.testing.assert_allclose(t["SE"], rest["se"], atol=5e-4)
    assert m.fit_conditional


def test_conditional_false_leaves_fit_statistics_unchanged():
    X = pd.read_csv(os.path.join(DATA, "cml_pcm_missing.csv"))
    a, b = PCM(X), PCM(X)
    a.calibrate()
    b.calibrate()
    a.fit_statistics(se=False)
    b.fit_statistics(se=False, conditional=False)
    np.testing.assert_array_equal(a.item_infit_ms.values, b.item_infit_ms.values)
    assert not b.fit_conditional and not hasattr(b, "item_restscore_table")


def test_q3_star():
    X = pd.read_csv(os.path.join(DATA, "cml_rsm_missing.csv"))
    m = RSM(X)
    m.calibrate(estimation="cml")
    m.fit_statistics(se=False, conditional=True, conditional_samples=20, seed=1)
    m.res_corr_analysis()
    q3 = m.residual_df.corr()
    off = ~np.eye(len(q3), dtype=bool)
    np.testing.assert_allclose(m.q3_star.values, q3.values - q3.values[off].mean(), atol=1e-12)
    assert m.q3_star_max == pytest.approx((q3.values - q3.values[off].mean())[off].max())


def test_conditional_fit_bootstrap():
    X = pd.read_csv(os.path.join(DATA, "cml_slm_complete.csv"))
    m = SLM(X)
    m.calibrate(estimation="cml")
    m.fit_statistics(se=False, conditional=True, conditional_samples=20, bootstrap_samples=60, seed=7)
    b1 = m.item_fit_bootstrap.copy()
    assert m.item_fit_bootstrap_info == dict(samples=60, skipped=0, seed=7)
    np.testing.assert_allclose(b1["Infit MS"].values, m.item_infit_ms.reindex(b1.index).values, atol=1e-12)
    assert (b1["Infit p (Westfall-Young)"] >= b1["Infit p"] - 1e-12).all()
    assert ((b1["Infit 95% low"] <= b1["Infit 95% high"])).all()
    m.fit_statistics(se=False, conditional=True, conditional_samples=20, bootstrap_samples=60, seed=7)
    pd.testing.assert_frame_equal(m.item_fit_bootstrap, b1)
    with pytest.raises(ValueError, match="needs conditional=True"):
        m.fit_statistics(se=False, bootstrap_samples=10)
