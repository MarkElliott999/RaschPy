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
    m.std_errors(se_method="hessian")  # psychotools reports Hessian SEs
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
    m.std_errors()
    hess = m.item_se.copy()
    m.std_errors(se_method="sandwich")
    assert m.se_method == "sandwich" and np.allclose(m.item_se, hess, rtol=0.2)
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
    m.std_errors(se_method="hessian")  # psychotools reports Hessian SEs
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
    m.std_errors(se_method="hessian")  # psychotools reports Hessian SEs
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
    q3 = m.q3_bootstrap
    assert len(q3) == 15 * 14 // 2 and (q3["p (Westfall-Young)"] >= q3["p"] - 1e-12).all()
    assert q3["Q3"].is_monotonic_decreasing
    r = m.residual_contrast_bootstrap
    assert r["first_contrast"] == pytest.approx(
        np.linalg.eigvalsh(m.std_residual_df.corr().values)[-1], abs=1e-10
    )
    assert 0 < r["p"] <= 1 and r["cutoff_95"] > r["expected"]
    rt = m.item_restscore_table
    assert (rt["p (bootstrap, Westfall-Young)"] >= rt["p (bootstrap)"] - 1e-12).all()
    with pytest.raises(ValueError, match="needs conditional=True"):
        m.fit_statistics(se=False, bootstrap_samples=10)


def test_mfrm_conditional_fit():
    from raschpy import MFRM
    from raschpy.simulation import MFRM_Sim

    sim = MFRM_Sim(no_of_items=4, no_of_persons=120, no_of_facet_elements=3, max_score=2, model="matrix", seed=3)
    m = MFRM(sim.responses)
    items, facets = list(m.item_names), list(m.facet_names)
    for model in ("global", "matrix", "bistretch"):
        m.fit_statistics(model=model, se=False)
        before = getattr(m, f"item_infit_ms_{model}").copy()
        m.fit_statistics(model=model, se=False, conditional=True)
        assert getattr(m, f"fit_conditional_{model}")
        # the grid read from the log numerators reproduces the engine's probabilities at another theta
        _, thr, fe = m._get_params(model)
        logn, _ = m._cat_probs_mfrm(np.zeros(1), items, facets, thr, model, fe, log_numerators=True)
        probs, _ = m._cat_probs_mfrm(np.array([0.7]), items, facets, thr, model, fe)
        for r in facets:
            le = logn[r][:, 0, :] - logn[r][0, 0, :]
            lp = np.arange(le.shape[0])[:, None] * 0.7 + le
            p = np.exp(lp - lp.max(0))
            np.testing.assert_allclose(p / p.sum(0), probs[r][:, 0, :], atol=1e-12)
        # conditional expectations sum to each person's total
        E = getattr(m, f"exp_score_df_{model}")
        X = m.responses.loc[E.index]
        np.testing.assert_allclose(
            E.groupby(level=1).sum().sum(1).values, X.groupby(level=1).sum().sum(1).values, atol=1e-9
        )
        m.fit_statistics(model=model, se=False)
        np.testing.assert_allclose(getattr(m, f"item_infit_ms_{model}").values, before.values, atol=1e-12)


def test_mfrm_conditional_extras_bootstrap_and_pca():
    from raschpy import MFRM
    from raschpy.simulation import MFRM_Sim

    sim = MFRM_Sim(no_of_items=4, no_of_persons=150, no_of_facet_elements=3, max_score=2, model="global", seed=6)
    m = MFRM(sim.responses)
    kw = dict(model="global", se=False, conditional=True, conditional_samples=30, bootstrap_samples=25, seed=2)
    m.fit_statistics(**kw)
    it = m.item_fit_bootstrap_global.copy()
    np.testing.assert_allclose(it["Infit MS"].values, m.item_infit_ms_global.reindex(it.index).values, atol=1e-12)
    for tab in (m.item_fit_bootstrap_global, m.rater_fit_bootstrap_global):
        assert (tab["Infit p (Westfall-Young)"] >= tab["Infit p"] - 1e-12).all()
    for tab in (m.item_restscore_table_global, m.rater_restscore_table_global):
        assert {"Observed gamma", "Expected gamma", "p (BH)", "p (bootstrap, Westfall-Young)"} <= set(tab.columns)
    assert len(m.q3_bootstrap_global) == 6 and m.person_conditional_p_global.notna().any()
    rc = m.residual_contrast_bootstrap_global
    first = np.linalg.eigvalsh(m.std_residual_df_global.astype(float).corr().values)[-1]
    assert rc["first_contrast"] == pytest.approx(first, abs=1e-10)
    assert 1 / 26 <= rc["p"] <= 1 and rc["cutoff_95"] >= rc["expected"] > 1
    m.fit_statistics(**kw)
    pd.testing.assert_frame_equal(m.item_fit_bootstrap_global, it)
    # residual analyses: eigenvalues of the correlation matrices; item correlations of the raw residuals; Q3*
    m._run_item_res_corr("global")
    m._run_facet_res_corr("global")
    R = m.std_residual_df_global.corr().values
    np.testing.assert_allclose(
        m.item_eigenvalues_global.values[:, 0], np.sort(np.linalg.eigvalsh(R))[::-1][:-1], atol=1e-10
    )
    q3 = m.residual_df_global.corr()
    np.testing.assert_allclose(m.item_residual_correlations_global.values, q3.values, atol=1e-12)
    off = ~np.eye(len(q3), dtype=bool)
    assert m.item_q3_star_max_global == pytest.approx((q3.values - q3.values[off].mean())[off].max())
    with pytest.raises(ValueError, match="needs conditional=True"):
        m.fit_statistics(model="global", se=False, bootstrap_samples=5)


def _anchor_model(name, seed=4):
    from raschpy.simulation import PCM_Sim, RSM_Sim, SLM_Sim

    Sim, Cls, kw = dict(
        SLM=(SLM_Sim, SLM, {}),
        PCM=(PCM_Sim, PCM, dict(max_score_vector=[2, 3] * 4)),
        RSM=(RSM_Sim, RSM, dict(max_score=3)),
    )[name]
    m = Cls(Sim(no_of_items=8, no_of_persons=400, seed=seed, **kw).responses)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(estimation="cml")
    return m


@pytest.mark.parametrize("name", ["SLM", "PCM", "RSM"])
def test_fixed_anchor_cml(name):
    m = _anchor_model(name)
    kw = dict(selection_method="none", plot=False)
    # anchors = CML estimates + a constant: the fixed fit is that shift exactly
    m.calibrate_anchor(m.items.iloc[:4] + 0.7, **kw)
    np.testing.assert_allclose(m.anchor_items, m.items + 0.7, atol=1e-8)
    assert list(m.anchor_cml.fixed_items) == list(m.items.index[:4])
    assert (m.anchor_cml.item_se.iloc[:4] == 0).all() and (m.anchor_cml.item_se.iloc[4:] > 0).all()
    assert m.anchor_summary["Anchors fixed (CML)"] == 4
    if name == "PCM":
        np.testing.assert_allclose(m.anchor_cml.thresholds.values, m.thresholds_uncentred.values + 0.7, atol=1e-8)
    if name == "RSM":
        np.testing.assert_allclose(m.anchor_cml.thresholds, m.thresholds, atol=1e-8)
    # perturbed anchors: kept at their values, the same maximum from independent solvers
    anch = m.items.iloc[:4] + 0.5 + np.array([0.3, -0.2, 0.1, -0.25])
    fits = []
    for solver in ("newton", "lbfgs"):
        m._cml_solver = solver
        m.calibrate_anchor(anch, **kw)
        fits.append(m.anchor_items.copy())
    np.testing.assert_allclose(fits[0][anch.index], anch, atol=1e-12)
    np.testing.assert_allclose(fits[0], fits[1], atol=1e-5)
    assert not np.allclose(fits[0].iloc[4:], m.items.iloc[4:] + m.anchor_adj, atol=1e-3)
    # translation on request; calibrate=True keeps CML
    m.calibrate_anchor(anch, fix_anchors=False, **kw)
    assert m.anchor_cml is None
    np.testing.assert_allclose(m.anchor_items.iloc[4:], m.items.iloc[4:] + m.anchor_adj)
    m.calibrate_anchor(anch, calibrate=True, **kw)
    assert m.estimation == "cml" and m.anchor_cml is not None


def test_fixed_anchor_needs_cml():
    from raschpy.simulation import SLM_Sim

    m = SLM(SLM_Sim(no_of_items=8, no_of_persons=300, seed=1).responses)
    m.calibrate()
    with pytest.raises(ValueError, match="fix_anchors=True needs CML"):
        m.calibrate_anchor(m.items.iloc[:3], selection_method="none", plot=False, fix_anchors=True)
    m.calibrate_anchor(m.items.iloc[:3], selection_method="none", plot=False)
    assert m.anchor_cml is None and "Anchors fixed (CML)" not in m.anchor_summary


def test_conditional_model_selection_rsm_pcm():
    from raschpy.simulation import PCM_Sim

    resp = PCM_Sim(no_of_items=6, no_of_persons=500, max_score_vector=[3] * 6, seed=5).responses
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, p = RSM(resp), PCM(resp)
        r.calibrate(estimation="cml")
        p.calibrate(estimation="cml")
        # at CML estimates the one step is null: the conditional maxima
        ll_r, ll_p = RSM._conditional_ll_rsm_pcm(r, p)
        assert ll_r == pytest.approx(r.convergence["log_likelihood"], abs=1e-8)
        assert ll_p == pytest.approx(p.convergence["log_likelihood"], abs=1e-8)
        # from the default estimates: close to the maxima, never above
        r0, p0 = RSM(resp), PCM(resp)
        r0.calibrate()
        p0.calibrate()
        l0r, l0p = RSM._conditional_ll_rsm_pcm(r0, p0)
        assert ll_r - 0.05 < l0r <= ll_r + 1e-8 and ll_p - 0.05 < l0p <= ll_p + 1e-8
        # both entry points agree; PCM-generated data
        a, b = PCM(resp), RSM(resp)
        a.model_selection(likelihood="conditional")
        b.model_selection(likelihood="conditional")
    assert a.model_comparison_rsm_pcm_aic == pytest.approx(b.model_comparison_rsm_pcm_aic)
    assert a.model_comparison_rsm_pcm_aic_preferred == "PCM"
    with pytest.raises(ValueError, match="likelihood must be"):
        a.model_selection(likelihood="marginal")


def test_conditional_model_selection_mfrm():
    from raschpy import MFRM, _rcl
    from raschpy.simulation import MFRM_Sim

    R, I, K = 4, 5, 3
    m = MFRM(MFRM_Sim(no_of_items=I, no_of_persons=300, no_of_facet_elements=R, max_score=K, model="bivector", seed=3).responses)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for mod in ("bivector", "bistretch"):
            m.calibrate(model=mod)
            Xv, thr, _ = m._virtual_items(mod)
            prep = _esf.prepare(Xv, [K] * (R * I))
            b0 = np.concatenate(thr)
            plug = _esf.cond_loglik(Xv, list(b0.reshape(-1, K)), prepared=prep)[0]
            one = m._conditional_ll_mfrm(mod, Xv, prep)
            assert one >= plug
            if mod == "bivector":
                top = _esf.fit_cml(Xv, _rcl.CellModel(mod, R, I, K).jacobian(), b0, [K] * (R * I), solver="newton")["ll"]
                assert top - 0.01 < one <= top + 1e-8
        m.model_selection(likelihood="conditional")
    assert m.model_comparison_mfrm_aic_preferred == "bivector"


def test_cml_dif_omnibus_is_conditional_lr():
    from raschpy.simulation import PCM_Sim, SLM_Sim

    s = SLM_Sim(no_of_items=8, no_of_persons=600, seed=7)
    g = pd.Series(np.where(np.arange(600) < 300, "A", "B"), index=s.responses.index)
    m = SLM(s.responses, exogenous=pd.DataFrame({"G": g}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(estimation="cml")
        m.dif_test("G", reference="A", seed=1)
        omni = m.dif_omnibus_table.iloc[0]
        # SLM, every item common: H1 is the two groups' own CML fits, so the LR is Andersen's
        m.global_fit(method="andersen-lr", split="exogenous", covariate="G")
    assert float(omni["LR"]) == pytest.approx(m.global_fit_summary["statistic"], rel=1e-6)
    assert int(omni["df"]) == m.global_fit_summary["df"] == 7
    # PCM with partly overlapping item sets: df = common items - 1 ('item'), thresholds - 1 ('full')
    p = PCM_Sim(no_of_items=6, no_of_persons=600, max_score_vector=[2, 3] * 3, seed=2)
    r = p.responses.copy()
    g = pd.Series(np.where(np.arange(600) < 300, "A", "B"), index=r.index)
    r.loc[g == "A", r.columns[5]] = np.nan
    r.loc[g == "B", r.columns[0]] = np.nan
    pm = PCM(r, exogenous=pd.DataFrame({"G": g}), max_score_vector=[2, 3] * 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pm.calibrate(estimation="cml")
        pm.dif_test("G", reference="A", seed=1)
        assert int(pm.dif_omnibus_table.iloc[0]["df"]) == 3
        pm.dif_test("G", reference="A", omnibus_scope="full", seed=1)
        assert int(pm.dif_omnibus_table.iloc[0]["df"]) == (3 + 2 + 3 + 2) - 1


@pytest.mark.parametrize("name", ["SLM", "PCM", "RSM"])
def test_cml_sandwich(name):
    m = _anchor_model(name, seed=9)
    X = np.array(m.responses.loc[:, list(m.item_names)], dtype=np.float64)
    thr = m._cml_threshold_list(list(m.item_names))
    # persons' scores sum to the gradient, which is ~0 at the CML estimate
    S = _esf.person_scores(X, thr)
    g = _esf.cond_loglik(X, thr)[1]
    np.testing.assert_allclose(S.sum(axis=0), g, atol=1e-10)
    m.std_errors(se_method="hessian")
    hess = m.item_se.copy()
    m.std_errors(se_method="sandwich")
    assert m.se_method == "sandwich"
    np.testing.assert_allclose(m.item_se, hess, rtol=0.25)
    assert not np.allclose(m.item_se, hess)


@pytest.mark.parametrize("name", ["SLM", "PCM", "RSM"])
def test_conditional_q1(name):
    m = _anchor_model(name, seed=11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit_statistics(conditional=True, conditional_samples=10)
    names = list(m.item_names)
    X = np.array(m.responses.loc[:, names], dtype=np.float64)
    mom = _esf.conditional_moments(X, m._cml_threshold_list(names))
    lab = m._q1_labels
    ok = lab >= 0
    # complete data: ability groups are groups of adjacent total scores
    tot = np.nansum(X, axis=1)
    for g in range(lab.max()):
        assert tot[ok & (lab == g)].max() < tot[ok & (lab == g + 1)].min()
    # Q1 by hand for the first item; df = groups - 1
    valid = np.isfinite(mom["V"][:, 0]) & (mom["V"][:, 0] > 0) & ok
    q1 = sum(
        (X[valid & (lab == g), 0].sum() - mom["E"][valid & (lab == g), 0].sum()) ** 2
        / mom["V"][valid & (lab == g), 0].sum()
        for g in range(lab.max() + 1)
    )
    assert m.item_q1_table["Q1"].iloc[0] == pytest.approx(q1, rel=1e-10)
    assert (m.item_q1_table["df"] == lab.max()).all()
    # at the CML estimates, each item's observed - expected sums to zero over the groups
    gq = m.item_q1_groups
    dev = ((gq["Observed mean"] - gq["Expected mean"]) * gq["Persons"]).groupby(level="Item").sum()
    np.testing.assert_allclose(dev.values, 0, atol=1e-6)
    # supplied groups
    sup = pd.Series(np.where(np.arange(len(m.responses)) % 2 == 0, "even", "odd"), index=m.responses.index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit_statistics(conditional=True, conditional_samples=10, q1_groups=sup)
    assert (m.item_q1_table["df"] == 1).all()
    assert set(m.item_q1_groups.index.get_level_values("Group")) == {"even", "odd"}


def test_andersen_item_tables():
    from raschpy.simulation import SLM_Sim

    s = SLM_Sim(no_of_items=8, no_of_persons=1000, seed=4)
    r = s.responses.copy()
    g = pd.Series(np.where(np.arange(1000) < 500, "A", "B"), index=r.index)
    th = s.persons.values[500:]
    p = 1 / (1 + np.exp(-(th - (float(s.items.iloc[2]) + 0.8))))
    r.loc[g == "B", r.columns[2]] = (np.random.default_rng(1).random(500) < p).astype(float)
    m = SLM(r, exogenous=pd.DataFrame({"G": g}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(estimation="cml")
        m.global_fit(method="andersen-lr", split="exogenous", covariate="G")
    tab, oe = m.global_fit_item_table, m.global_fit_obsexp
    assert tab["p (BH)"].idxmin() == r.columns[2] and tab.loc[r.columns[2], "p (BH)"] < 0.001
    assert (tab["df"] == 1).all()
    # pooled CML: observed - expected sums to zero over the two groups for every item
    dev = ((oe["Observed mean"] - oe["Expected mean"]) * oe["Persons"]).groupby(level="Item").sum()
    np.testing.assert_allclose(dev.values, 0, atol=1e-6)
    # with two groups and opposite deviations, the item chi-square is the square of either z
    np.testing.assert_allclose(oe["z"].groupby(level="Item").apply(lambda z: (z**2).sum()).values,
                               tab["Chi-square"].values, rtol=1e-10)


@pytest.mark.parametrize("name", ["SLM", "PCM", "RSM"])
def test_icc_cond_obs(name):
    import matplotlib

    matplotlib.use("Agg")
    m = _anchor_model(name, seed=12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit_statistics(conditional=True, conditional_samples=10, q1_groups=4)
    item = m.item_names[1]
    # same groups as Q1 with the same number of groups: identical observed / expected means
    lev, x, o, e, h = m._icc_overlay_points([item], 4, True)[item][0]
    q = m.item_q1_groups.loc[item]
    np.testing.assert_allclose(o, q["Observed mean"].values, atol=1e-12)
    np.testing.assert_allclose(e, q["Expected mean"].values, atol=1e-12)
    np.testing.assert_allclose(np.sign(o - e), np.sign(q["z"].values))
    assert np.all(h > 0) and np.all(np.diff(x) > 0)

    def markers(fig):
        return len(fig.axes[0].collections)

    plain = markers(m.icc(item, obs=True))
    cond = markers(m.icc(item, obs=True, cond_obs=True))
    nobars = markers(m.icc(item, obs=True, cond_obs=True, cond_obs_bars=False))
    assert cond > nobars > plain - 1  # observed + expected markers, plus bars
    assert markers(m.icc(item, obs=False, cond_obs=True)) == 0  # no-op without obs
    groups = pd.Series(np.where(np.arange(len(m.responses)) % 2 == 0, "a", "b"), index=m.responses.index)
    fig = m.icc(item, obs=True, cond_obs=True, obs_groups=groups)
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()][:2] == ["a", "b"]
    two = m.icc(list(m.item_names[:2]), obs=True, cond_obs=True, obs_groups=groups)  # groups ignored
    assert "a" not in [t.get_text() for t in two.axes[0].get_legend().get_texts()]


def test_local_dependence_gamma():
    from raschpy.simulation import SLM_Sim

    s = SLM_Sim(no_of_items=8, no_of_persons=800, seed=3)
    r = s.responses.copy()
    c = np.random.default_rng(0).random(800) < 0.35
    r.loc[c, "Item_5"] = r.loc[c, "Item_4"]
    m = SLM(r)
    tab = m.local_dependence_gamma()
    assert len(tab) == 8 * 7 and m.local_dependence_gamma_info["p_method"] == "permutation"
    top = tab.iloc[:2]
    assert set(top["Item"]) == {"Item_4", "Item_5"} and top["Flagged"].all()
    # only the planted pair is flagged positive (a strong pair can push one of its items negative
    # against others: its rest score includes the copy)
    assert not (tab.iloc[2:]["Flagged"] & (tab.iloc[2:]["Gamma"] > 0)).any()
    mc1 = m.local_dependence_gamma(p_method="monte-carlo", samples=50, seed=4)
    mc2 = m.local_dependence_gamma(p_method="monte-carlo", samples=50, seed=4)
    pd.testing.assert_frame_equal(mc1, mc2)
    assert mc1["p"].min() == pytest.approx(1 / 51)
    asy = m.local_dependence_gamma(p_method="asymptotic")
    np.testing.assert_allclose(asy.set_index(["Item", "Paired item"])["Gamma"].sort_index(),
                               tab.set_index(["Item", "Paired item"])["Gamma"].sort_index())
    with pytest.raises(ValueError, match="p_method must be"):
        m.local_dependence_gamma(p_method="exact")


@pytest.mark.parametrize("name", ["SLM", "PCM", "RSM"])
def test_targeting(name):
    m = _anchor_model(name, seed=13)
    out = m.targeting()
    assert abs(out["Population mean"]) < 0.4 and 0.8 < out["Population SD"] < 2.5
    assert 0 < out["Expected / maximum information"] <= 1
    for k in ("rho1", "rho3", "rho4", "rho5", "rho6"):
        assert 0 < out[k] < 1
    assert out["rho2"] == pytest.approx(out["rho1"], abs=1e-9)
    curves = m.targeting_curves
    assert curves["Information"].idxmax() == pytest.approx(out["Target"], abs=0.05)
    st = m.targeting_score_table
    assert st["Model proportion"].sum() == pytest.approx(1) and st["WLE"].is_monotonic_increasing
    assert len(m.targeting_items) == len(m.item_names)


@pytest.mark.parametrize("name", ["SLM", "PCM", "RSM"])
def test_r1c(name):
    m = _anchor_model(name, seed=14)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit_statistics(conditional=True, conditional_samples=10, q1_groups=4)
    t = m.item_q1_r1c
    I, G = len(m.item_names), 4
    assert t.loc["item scores", "Groups"] == G and t.loc["item scores", "Estimates"] == "CML"
    # item scores: each group's deviations sum to zero over items (its totals are fixed), and the
    # estimated item parameters use I - 1 more: df = (G - 1)(I - 1)
    assert t.loc["item scores", "df"] == (G - 1) * (I - 1)
    assert 0 <= t.loc["item scores", "p"] <= 1
    assert ("thresholds" in t.index) == (name != "SLM")
    # with the default estimates, one Newton step reaches (nearly) the CML statistic
    m2 = _anchor_model(name, seed=14)
    m2.calibrate(estimation="cp-full-rcl" if name == "RSM" else "pair")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m2.fit_statistics(conditional=True, conditional_samples=10, q1_groups=4)
    assert m2.item_q1_r1c.loc["item scores", "Estimates"] != "CML"
    assert m2.item_q1_r1c.loc["item scores", "Statistic"] == pytest.approx(
        t.loc["item scores", "Statistic"], rel=0.05
    )


def test_global_stats_df():
    from raschpy.simulation import SLM_Sim

    s = SLM_Sim(no_of_items=8, no_of_persons=600, seed=5)
    g = pd.Series(np.where(np.arange(600) < 300, "A", "B"), index=s.responses.index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # collect only: nothing is computed, everything listed as not run
        m = SLM(s.responses, exogenous=pd.DataFrame({"G": g}))
        m.calibrate()
        before = set(vars(m))
        tab = m.global_stats_df()
        assert set(vars(m)) - before == {"global_stats"}
        assert tab["Details"].str.startswith(("not run", "needs")).all()
        # run_all_tests without CML estimates: the CML-only tests are left out
        m.global_stats_df(run_all_tests=True)
        t = m.global_stats
        assert t.loc["Andersen LR", "Details"] == "needs calibrate(estimation='cml')"
        assert np.isfinite(t.loc["R1c (item scores)", "Value"]) and np.isfinite(t.loc["rho1", "Value"])
        # CML: a test already run is kept, not rerun with the defaults; options passed through
        c = SLM(s.responses, exogenous=pd.DataFrame({"G": g}))
        c.calibrate(estimation="cml")
        c.global_fit(method="andersen-lr", split="exogenous", covariate="G")
        c.global_stats_df(run_all_tests=True, test_kwargs={"martin-lof": {"p_method": "monte-carlo", "no_of_samples": 20, "seed": 1}})
    t = c.global_stats
    andersen = t.loc[["Andersen LR"]]
    assert len(andersen) == 1 and andersen["Details"].iloc[0].startswith("split=exogenous; covariate=G")
    assert "p_method=monte-carlo" in t.loc["Martin-Lof", "Details"]
    assert set(c.global_fit_results) == {("andersen-lr", "split=exogenous; covariate=G"),
                                         ("martin-lof", "split=median; p_method=monte-carlo")}


@pytest.mark.parametrize("name", ["SLM", "PCM", "RSM"])
def test_stats_tables_conditional_columns(name):
    m = _anchor_model(name, seed=15)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.item_stats_df()
        base = list(m.item_stats.columns)
        m.item_stats_df(full=True)  # also after a default call (the SLM interval bug)
        m.fit_statistics(conditional=True, conditional_samples=20, bootstrap_samples=10, seed=1)
        m.targeting()
        m.item_stats_df()
        added = [c for c in m.item_stats.columns if c not in base]
        assert added == ["Restscore p (BH)", "Q1 p (BH)", "Infit p (Westfall-Young)", "Outfit p (Westfall-Young)"]
        np.testing.assert_allclose(m.item_stats["Q1 p (BH)"], m.item_q1_table["p (BH)"].round(3))
        m.item_stats_df(full=True)
        for c in ("Observed gamma", "Expected gamma", "Q1", "Q1 df", "Target", "Max information"):
            assert c in m.item_stats.columns
        m.person_stats_df()
        assert m.person_stats["Conditional p"].notna().any()
        m.fit_statistics()  # back to person-based fit: no stale conditional columns
        m.item_stats_df()
        m.person_stats_df()
    assert list(m.item_stats.columns) == base and "Conditional p" not in m.person_stats.columns
