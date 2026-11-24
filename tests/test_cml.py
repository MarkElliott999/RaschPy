"""
Tests for conditional maximum likelihood (estimation='cml').

External agreement uses stored psychotools::raschmodel fits (tests/external_data/cml_*_psychotools.csv,
made by cml_fit_psychotools.R), so no R is needed at run time. The stored fits stop at psychotools' own
optimiser tolerance (gradient ~1e-4), hence the tolerances below.
"""

import functools
import itertools
import os
import warnings

import numpy as np
import scipy.stats as scipy_stats
import pandas as pd
import pytest
from scipy.special import logsumexp

from raschpy import PCM, RSM, SLM, _esf

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "external_data")


def _plain_cml(cls):
    """calibrate() arguments for plain (unpenalised) CML, which the external references fit: PCM's
    default CML ridge off."""
    return dict(estimation="cml", ridge=0) if cls is PCM else dict(estimation="cml")


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
    m.calibrate(estimation="cml", ridge=0)
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
        m.calibrate(estimation="cml", ridge=0)
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
            m.calibrate(estimation="cml", unused_categories="drop")


def test_rsm_cml_width_ridge_for_an_unused_category():
    """unused_categories='ridge' (the CML default): finite estimates, the penalty only on the widths next to
    the unused category, those parameters listed in no_data_parameters with posterior SEs ('nan' leaves
    them out), and no penalty when every category is used."""
    X = pd.read_csv(os.path.join(DATA, "cml_rsm_complete.csv"))
    m0 = RSM(X, validate=False)
    m0.calibrate(estimation="cml")
    assert m0._cml_penalty is None and m0.no_data_parameters == {}
    K = int(np.nanmax(X.to_numpy()))
    Xe = X.where(X != K, K - 1)  # top category unused on every item
    m = RSM(Xe, max_score=K, validate=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(estimation="cml")
    assert np.all(np.isfinite(m.thresholds)) and m.convergence["converged"]
    assert m.cml_unused_categories == {"all items": [K]}
    assert m.no_data_parameters == {"widths": {K - 1: "ridge"}, "thresholds": {K: "ridge"}}
    with pytest.warns(UserWarning, match="set by the width ridge"):
        m.std_errors()
    assert np.isfinite(m.cat_width_se[K - 1]) and np.isfinite(m.threshold_se[K - 1])
    with pytest.warns(UserWarning, match="no_data_se='nan'"):
        m.std_errors(no_data_se="nan")
    assert np.isnan(m.cat_width_se[K - 1]) and np.isnan(m.threshold_se[K - 1])


# ---------------------------------------------------------------- Andersen LR (global_fit)

@pytest.mark.parametrize("cls,name", [(SLM, "slm_complete"), (PCM, "pcm_complete"), (RSM, "rsm_complete")])
def test_andersen_lr_is_the_conditional_likelihood_ratio(cls, name):
    X = pd.read_csv(os.path.join(DATA, f"cml_{name}.csv"))
    m = cls(X)
    m.calibrate(estimation="cml")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.global_fit(method="andersen", split="random", seed=3)
    groups = m._global_fit_groups("random", None, 3)
    items = [c for c in m.item_names if c not in m.global_fit_excluded]
    fits = [m._global_fit_subset(groups[n], items=items) for n in groups]
    pooled = m._global_fit_subset(groups["A"].append(groups["B"]), items=items)
    lr = 2 * (sum(f.convergence["log_likelihood"] for f in fits) - pooled.convergence["log_likelihood"])
    assert r["statistic"] == pytest.approx(lr, abs=1e-8) and r["statistic"] >= 0
    assert r["df"] == pooled._cml_free_parameters()


def test_andersen_lr_needs_cml():
    """PAIR estimates do not maximise the likelihood the statistic evaluates (CML and 'cp-full-rcl' do)."""
    m = SLM(pd.read_csv(os.path.join(DATA, "cml_slm_complete.csv")))
    m.calibrate(estimation="pair")
    with pytest.raises(ValueError, match="calibrate\\(estimation='cml'\\)"):
        m.global_fit(method="andersen")



@pytest.mark.parametrize("name", ["SLM", "PCM", "RSM"])
def test_andersen_lr_three_groups(name):
    m = _anchor_model(name, seed=21)
    g = pd.Series(np.arange(len(m.responses)) % 3, index=m.responses.index).map({0: "a", 1: "b", 2: "c"})
    m.exogenous = pd.DataFrame({"G": g})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.global_fit(method="andersen", covariate="G")
    groups = m._global_fit_groups("exogenous", "G", None, max_groups=np.inf)
    items = [c for c in m.item_names if c not in m.global_fit_excluded]
    fits = [m._global_fit_subset(groups[n], items=items) for n in groups]
    pooled = m._global_fit_subset(m.responses.index, items=items)
    lr = 2 * (sum(f.convergence["log_likelihood"] for f in fits) - pooled.convergence["log_likelihood"])
    assert r["statistic"] == pytest.approx(lr, abs=1e-8)
    assert r["df"] == 2 * pooled._cml_free_parameters() and r["covariate"] == "G"
    assert "Difference" not in m.global_fit_item_table
    with pytest.raises(ValueError, match="exactly two values"):
        m.global_fit(method="wald", covariate="G")


def test_global_fit_covariate_needs_exogenous_split():
    m = _anchor_model("SLM", seed=22)
    m.exogenous = pd.DataFrame({"G": np.arange(len(m.responses)) % 2}, index=m.responses.index)
    with pytest.raises(ValueError, match="only used with split='exogenous'"):
        m.global_fit(method="andersen", split="score", covariate="G")
    with pytest.raises(ValueError, match="not used by method='martin-lof'"):
        m.global_fit(method="martin-lof", covariate="G")


def test_score_groups_split():
    # whole score groups, cut where the non-extreme persons divide most evenly (iarm's score_groups)
    X = pd.DataFrame(
        [[0, 0, 0]] * 3 + [[1, 0, 0]] * 4 + [[1, 1, 0]] * 5 + [[1, 1, 1]] * 2, columns=list("abc")
    )
    g = SLM(X)._score_group_split(X)
    assert list(g["low"]) == list(range(7)) and list(g["high"]) == list(range(7, 14))
    m = _anchor_model("SLM", seed=23)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.global_fit(method="andersen", split="score_groups")
    tot = m.responses.sum(axis=1)
    low, high = (tot[m._global_fit_groups("score_groups", None, None)[k]] for k in ("low", "high"))
    assert low.max() < high.min() and r["df"] > 0


def _unused_category_data():
    from raschpy.simulation import PCM_Sim

    X = PCM_Sim(no_of_items=6, no_of_persons=400, max_score_vector=[3] * 6, offset=-1.0, seed=31).responses
    g = pd.Series(np.where(np.arange(len(X)) < 200, "a", "b"), index=X.index)
    X.loc[g == "a", X.columns[0]] = X.loc[g == "a", X.columns[0]].replace(3, 2)  # category 3 unused in a
    X.loc[g == "a", X.columns[1]] = X.loc[g == "a", X.columns[1]].replace(1, 2)  # category 1 unused in a
    return X, g


def test_andersen_lr_unused_categories_keep():
    X, g = _unused_category_data()
    m = PCM(X, exogenous=pd.DataFrame({"G": g}))
    m.calibrate(estimation="cml", ridge=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        drop = m.global_fit(method="andersen", covariate="G")
        assert set(m.global_fit_excluded) == set(X.columns[:2])
        keep = m.global_fit(method="andersen", covariate="G", unused_categories="null")
    assert m.global_fit_excluded == []
    a, b = m.global_fit_groups["a"], m.global_fit_groups["b"]
    assert a.cml_unused_categories == {X.columns[0]: [3], X.columns[1]: [1]}
    assert a._cml_free_parameters() == 6 * 3 - 1 - 2 and b._cml_free_parameters() == 17
    ll = keep["log-likelihoods"]
    assert keep["statistic"] == pytest.approx(2 * (ll["a"] + ll["b"] - ll["pooled"]), abs=1e-9)
    assert keep["df"] == 17 + 15 - 17 and drop["df"] == 4 * 3 - 1
    # markers: an unused category's threshold is inf, the next one holds the sum either side of it
    t1, t0 = a.thresholds_uncentred.loc[X.columns[1]], a.thresholds_uncentred.loc[X.columns[0]]
    assert t1[1] == np.inf and np.isfinite(t1[[2, 3]]).all() and t0[3] == np.inf
    with pytest.raises(ValueError, match="method='andersen' only"):
        m.global_fit(method="wald", covariate="G", unused_categories="null")


def test_rsm_keep_interior_unused_category_matches_direct_cml():
    """RSM with a category unused by every item: 'null' maximises the conditional likelihood of the
    rating scale model with that category's probability zero (checked by a direct fit)."""
    from scipy.optimize import minimize
    from raschpy.simulation import RSM_Sim

    X = RSM_Sim(no_of_items=5, no_of_persons=300, max_score=3, seed=32).responses.replace(2, 1)
    m = RSM(X, max_score=3)
    m.calibrate(estimation="cml", unused_categories="null")
    cats, Xi = [0, 1, 3], X.to_numpy(int)
    I, tot = Xi.shape[1], Xi.sum(1)

    def nll(p):
        d = np.r_[p[: I - 1], -p[: I - 1].sum()]
        cum = dict(zip(cats, np.r_[0.0, p[I - 1 :]]))
        lw = np.array([[-(x * d[i] + cum[x]) for x in cats] for i in range(I)])
        G = np.zeros(3 * I + 1)
        G[0] = 1
        for i in range(I):
            new = np.zeros_like(G)
            for j, x in enumerate(cats):
                new[x:] += G[: len(G) - x] * np.exp(lw[i, j])
            G = new
        own = lw[np.arange(I)[None, :], np.searchsorted(cats, Xi)].sum(1)
        return -(own - np.log(G[tot])).sum()

    direct = -minimize(nll, np.zeros(I - 1 + 2), method="BFGS", options=dict(gtol=1e-10)).fun
    assert m.convergence["log_likelihood"] == pytest.approx(direct, abs=1e-6)
    assert m._cml_free_parameters() == I - 1 + 1
    assert m.thresholds[2] == np.inf and np.isfinite(m.thresholds[[1, 3]]).all()


def test_andersen_lr_monte_carlo():
    X, g = _unused_category_data()
    m = PCM(X, exogenous=pd.DataFrame({"G": g}))
    m.calibrate(estimation="cml", ridge=0)
    kw = dict(method="andersen", covariate="G", unused_categories="null")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chi = m.global_fit(**kw).copy()
        mc = m.global_fit(**kw, p_method="monte-carlo", no_of_samples=30, seed=5).copy()
        again = m.global_fit(**kw, p_method="monte-carlo", no_of_samples=30, seed=5)
        early = m.global_fit(**kw, p_method="monte-carlo", no_of_samples=30, seed=5, mc_exceedances=2)
    assert chi["p_method"] == "chi2" and chi["p-value"] == chi["p-value (chi-square)"]
    assert mc["statistic"] == chi["statistic"] and mc["p-value (chi-square)"] == chi["p-value"]
    assert mc["Monte Carlo samples"] + mc["Monte Carlo samples skipped"] == 30
    assert 1 / 31 <= mc["p-value"] <= 1 and again["p-value"] == mc["p-value"]
    assert early["Monte Carlo samples"] <= 30
    with pytest.raises(ValueError, match="p_method must be"):
        m.global_fit(method="andersen", covariate="G", p_method="permutation")


def _keep_pcm_data():
    from raschpy.simulation import PCM_Sim

    X = PCM_Sim(no_of_items=6, no_of_persons=500, max_score_vector=[3] * 6, seed=41).responses
    X[X.columns[0]] = X[X.columns[0]].replace(3, 2)  # top category unused
    X[X.columns[1]] = X[X.columns[1]].replace(0, 1)  # bottom category unused
    return X


def _keep_outputs(m):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.std_errors(se_method="hessian")
        m.person_estimates()
        m.fit_statistics(conditional=True)
    return dict(
        ll=m.convergence["log_likelihood"],
        items=(m.items - m.items.mean()).values,
        se=m.item_se.values,
        persons=(m.persons - m.items.mean()).values,
        infit=m.item_infit_ms.values,
        outfit=m.item_outfit_ms.values,
        q1_overall=m.item_q1_overall.loc["item scores", "Statistic"],
    )


def test_calibrate_keep_end_categories_equal_narrowed_range():
    """An unused top (bottom) category kept with probability zero gives the same model as the item with
    its score range narrowed (shifted down): every output agrees."""
    X = _keep_pcm_data()
    k = PCM(X, max_score_vector=[3] * 6)
    k.calibrate(estimation="cml", unused_categories="null", ridge=0)
    a, b = X.columns[0], X.columns[1]
    assert k.cml_unused_categories == {a: [3], b: [0]} and k._cml_free_parameters() == 6 * 3 - 1 - 2
    assert k.thresholds_uncentred.loc[a, 3] == np.inf and k.thresholds_uncentred.loc[b, 1] == -np.inf
    Y = X.copy()
    Y[b] = Y[b] - 1
    o = PCM(Y, max_score_vector=[2, 2, 3, 3, 3, 3])
    o.calibrate(estimation="cml", ridge=0)
    ka, oa = _keep_outputs(k), _keep_outputs(o)
    for key in ka:
        np.testing.assert_allclose(ka[key], oa[key], atol=1e-6, err_msg=key)
    lt = k.score_lookup_table()
    assert lt.index[0] == 1 and lt.index[-1] == 17  # possible totals only


def test_calibrate_keep_rsm_top_category_equals_lower_max_score():
    from raschpy.simulation import RSM_Sim

    X = RSM_Sim(no_of_items=6, no_of_persons=400, max_score=3, seed=42).responses.replace(3, 2)
    k = RSM(X, max_score=3)
    k.calibrate(estimation="cml", unused_categories="null")
    o = RSM(X, max_score=2)
    o.calibrate(estimation="cml")
    ka, oa = _keep_outputs(k), _keep_outputs(o)
    for key in ka:
        np.testing.assert_allclose(ka[key], oa[key], atol=1e-6, err_msg=key)


def test_calibrate_keep_interior_conditional_fit_direct():
    """Interior unused category: conditional infit/outfit equal a direct computation of the conditional
    distribution of each response given the total (the unused category with probability zero)."""
    X = _keep_pcm_data()
    X[X.columns[2]] = X[X.columns[2]].replace(2, 1)
    m = PCM(X, max_score_vector=[3] * 6)
    m.calibrate(estimation="cml", unused_categories="null")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.person_estimates()
        m.fit_statistics(conditional=True)
    W = [np.exp(_esf.log_category_weights(t)) for t in m._cml_threshold_list(list(X.columns))]
    Xa, I = X.to_numpy(int), X.shape[1]
    poly = lambda items: functools.reduce(np.convolve, [W[i] for i in items], np.array([1.0]))  # noqa: E731
    E, V = np.zeros(Xa.shape), np.zeros(Xa.shape)
    for i in range(I):
        Gm = poly([j for j in range(I) if j != i])
        for v, r in enumerate(Xa.sum(axis=1)):
            p = np.array([W[i][x] * (Gm[r - x] if 0 <= r - x < len(Gm) else 0) for x in range(4)])
            p /= p.sum()
            E[v, i] = p @ np.arange(4)
            V[v, i] = p @ (np.arange(4) - E[v, i]) ** 2
    pos = m.responses.index.get_indexer(m.residual_df.index)
    R, Vv = Xa[pos] - E[pos], np.where(V[pos] > 1e-10, V[pos], np.nan)
    np.testing.assert_allclose(np.nanmean(R**2 / Vv, axis=0), m.item_outfit_ms.values, atol=1e-10)
    infit = np.nansum(np.where(np.isnan(Vv), np.nan, R**2), axis=0) / np.nansum(Vv, axis=0)
    np.testing.assert_allclose(infit, m.item_infit_ms.values, atol=1e-10)


def test_calibrate_keep_needs_cml():
    with pytest.raises(ValueError, match="estimation='cml' only"):
        PCM(_keep_pcm_data(), max_score_vector=[3] * 6).calibrate(unused_categories="null")

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
    m.calibrate(**_plain_cml(cls))
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
    m.calibrate(**_plain_cml(cls))
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
    assert m.item_fit_bootstrap_info == dict(samples=60, skipped=0, seed=7, null_fit="own")
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


def test_conditional_model_selection_mfrm():
    from raschpy import MFRM
    from raschpy.simulation import MFRM_Sim

    R, I, K = 4, 5, 3
    resp = MFRM_Sim(no_of_items=I, no_of_persons=300, no_of_facet_elements=R, max_score=K, model="bivector", seed=3).responses
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MFRM(resp)
        m.model_selection(delta=0, stretch_delta=0)  # cp-full-rcl candidates: the composite route, no tolerance
        assert m.model_comparison_mfrm_info["route"] == "composite"
        assert m.model_comparison_mfrm_aic_preferred == "bivector"
        mc = MFRM(resp)
        mc.calibrate(model="global", estimation="cml")
        mc.model_selection(delta=0, stretch_delta=0)  # CML candidates: scored at their estimates, no tolerance
        assert mc.model_comparison_mfrm_info["route"] == "cml"
        for mod in ("bivector", "bistretch"):
            Xv, thr, _ = mc._virtual_items(mod)
            ll = _esf.cond_loglik(Xv, thr)[0]
            assert np.isclose(mc.model_comparison_mfrm_aic_summary.loc[mod, "LL"], ll)
    assert mc.model_comparison_mfrm_aic_preferred == "bivector"


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
        m.global_fit(method="andersen", split="exogenous", covariate="G")
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


_INFLUENCE_CASES = [
    ("SLM", {}),
    ("SLM", dict(method="cos")),
    ("SLM", dict(method="ls")),
    ("SLM", dict(method="evm")),
    ("SLM", dict(estimation="pair", matrix_power=2)),
    ("SLM", dict(estimation="cml", ridge=0.5)),
    ("PCM", dict(method="cos")),
    ("PCM", dict(estimation="pair")),
    ("PCM", dict(ridge=0.3)),
    ("PCM", dict(estimation="cml", ridge=0.3)),
    ("RSM", dict(estimation="pair-cpat")),
    ("RSM", dict(estimation="pair-cpat", method="cos", threshold_method="cpat-info-weighted")),
    ("RSM", dict(estimation="cpat-rcl")),
    ("RSM", {}),
    ("RSM", dict(estimation="cml")),
]


@pytest.mark.parametrize("name,kw", _INFLUENCE_CASES)
def test_threshold_influence(name, kw):
    """Each person's influence on the fitted thresholds equals half the change between the fits with the
    person added twice and left out (a central difference), to second order."""
    from raschpy.simulation import PCM_Sim, RSM_Sim, SLM_Sim

    Sim, Cls, skw = dict(
        SLM=(SLM_Sim, SLM, {}), PCM=(PCM_Sim, PCM, dict(max_score_vector=[2, 3] * 3)),
        RSM=(RSM_Sim, RSM, dict(max_score=3)),
    )[name]
    r = Sim(no_of_items=6, no_of_persons=300, seed=5, **skw).responses
    ckw = dict(kw)
    if not ckw.get("estimation") == "cml":
        ckw.setdefault("constant", 0.1)
        if name == "RSM":
            ckw.setdefault("threshold_constant", 0.03)

    def fit(df, **extra):
        m = Cls(df, validate=False, **({"max_score": 3} if name == "RSM" else {}))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.calibrate(**{**ckw, **extra})
        b = np.concatenate(m._cml_threshold_list(list(m.item_names)))
        return m, b - b.mean()

    m, b0 = fit(r)
    extra = {"ridge": float(m.ridge)} if name == "PCM" and getattr(m, "ridge", None) else {}
    X = np.array(m.responses.loc[:, list(m.item_names)], float)
    Phi = m._threshold_influence(X, list(m.item_names))[0]
    Phi = Phi - Phi.mean(1, keepdims=True)
    tot = np.nansum(X, 1)
    for n in [int(q) for q in np.flatnonzero((tot > 0) & (tot < np.nanmax(tot)))[:3]]:
        twice = pd.concat([r, r.iloc[[n]].rename(index=lambda i: f"copy {i}")])
        up, down = fit(twice, **extra)[1], fit(r.drop(r.index[n]), **extra)[1]
        np.testing.assert_allclose(Phi[n], (up - down) / 2, atol=5e-3 * np.abs(up - down).max())


@pytest.mark.parametrize("name", ["SLM", "PCM", "RSM"])
def test_q1_groups_z_allows_for_estimation(name):
    """Under CML with two groups, an item's deviations are opposite in the two groups; allowing for the
    estimation of its own parameter alone would give each the variance V_A V_B / (V_A + V_B) and z^2 = Q1,
    and allowing for the other items' too gives a smaller variance, so z^2 is at least Q1 (close to it for
    SLM and RSM, whose items have one location each; PCM's items have one parameter per threshold)."""
    m = _anchor_model(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit_statistics(conditional=True, conditional_samples=10, q1_groups=2, q1_overall_samples=None)
    z = m.item_q1_groups["z"].unstack("Group")
    np.testing.assert_allclose(z[1].values, -z[2].values, rtol=1e-6)
    q1 = m.item_q1_table.loc[z.index, "Q1"].values
    assert np.all(z[1].values ** 2 >= q1 * (1 - 1e-9))
    if name != "PCM":
        assert np.all(z[1].values ** 2 <= 1.1 * q1)


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
        m.global_fit(method="andersen", split="exogenous", covariate="G")
    tab, oe = m.global_fit_item_table, m.global_fit_obsexp
    assert tab["p (BH)"].idxmin() == r.columns[2] and tab.loc[r.columns[2], "p (BH)"] < 0.001
    assert (tab["df"] == 1).all()
    # pooled CML: observed - expected sums to zero over the two groups for every item
    dev = ((oe["Observed mean"] - oe["Expected mean"]) * oe["Persons"]).groupby(level="Item").sum()
    np.testing.assert_allclose(dev.values, 0, atol=1e-6)
    # with two groups the deviations are opposite and so is z, whose variance allows for the estimation
    # of the item parameters (Glas's correction under CML), so z^2 exceeds the uncorrected chi-square's half
    z = oe["z"].unstack("Group")
    np.testing.assert_allclose(z["A"].values, -z["B"].values, rtol=1e-8)
    assert (z["A"].values ** 2 >= tab.loc[z.index, "Chi-square"].values / 2 - 1e-10).all()


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
    np.testing.assert_allclose(mc1.set_index(["Item", "Paired item"])["Gamma"].sort_index(),
                               tab.set_index(["Item", "Paired item"])["Gamma"].sort_index())
    for bad in ("asymptotic", "exact"):
        with pytest.raises(ValueError, match="p_method must be"):
            m.local_dependence_gamma(p_method=bad)


@pytest.mark.parametrize("name", ["SLM", "PCM", "RSM"])
def test_targeting(name):
    m = _anchor_model(name, seed=13)
    out = m.targeting()
    assert abs(out["Population mean"]) < 0.4 and 0.8 < out["Population SD"] < 2.5
    assert 0 < out["Expected / maximum information"] <= 1
    for k in ("rho1", "rho3", "rho4", "rho5", "rho6"):
        assert 0 < out[k] < 1
    assert out["rho2"] == pytest.approx(out["rho1"], abs=1e-9)
    assert out["rho4"] == pytest.approx(out["rho3"], abs=1e-9)
    curves = m.targeting_curves
    assert curves["Information"].idxmax() == pytest.approx(out["Target"], abs=0.05)
    st = m.targeting_score_table
    assert st["Model proportion"].sum() == pytest.approx(1) and st["WLE"].is_monotonic_increasing
    assert len(m.targeting_items) == len(m.item_names)


@pytest.mark.parametrize("name", ["SLM", "PCM", "RSM"])
def test_q1_overall(name):
    m = _anchor_model(name, seed=14)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit_statistics(conditional=True, conditional_samples=10, q1_groups=4, seed=3)
    t = m.item_q1_overall
    I = len(m.item_names)
    assert (t["Groups"] == 4).all() and ("thresholds" in t.index) == (name != "SLM")
    # item scores: van den Wollenberg's overall Q1, (I - 1) / I x the sum of the items' Q1 (same groups)
    assert t.loc["item scores", "Statistic"] == pytest.approx((I - 1) / I * m.item_q1_table["Q1"].sum())
    assert ((t["p"] > 0) & (t["p"] <= 1)).all() and (t["Samples"] <= 99).all()
    # reproducible with seed=; q1_overall_samples=None leaves it out
    m2 = _anchor_model(name, seed=14)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m2.fit_statistics(conditional=True, conditional_samples=10, q1_groups=4, seed=3)
    pd.testing.assert_frame_equal(t, m2.item_q1_overall)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m2.fit_statistics(conditional=True, conditional_samples=10, q1_groups=4, q1_overall_samples=None)
    assert m2.item_q1_overall is None


@pytest.mark.parametrize("name", ["SLM", "PCM", "RSM"])
def test_q1_overall_at_the_fit(name):
    """With estimates other than CML the statistic is computed at those estimates, not at a CML step."""
    from raschpy import _esf

    m = _anchor_model(name, seed=14)
    m.calibrate(estimation="cp-full-rcl")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.person_estimates()
        m.fit_statistics(conditional=True, conditional_samples=10, q1_groups=4, q1_overall_samples=19)
    names = list(m.item_names)
    X = m.responses.loc[:, names].to_numpy(float)
    thr = m._cml_threshold_list(names)
    st = m._q1_overall_statistics(X, thr, m._q1_labels, _esf.prepare(X, [len(t) for t in thr]))
    for level, value in st.items():
        assert m.item_q1_overall.loc[level, "Statistic"] == pytest.approx(value)


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
        # run_all_tests with 'cp-full-rcl' estimates: Andersen's composite test runs, Martin-Lof's (CML only) not
        m.global_stats_df(run_all_tests=True)
        t = m.global_stats
        assert "p_method=monte-carlo" in t.loc["Andersen LR", "Details"]
        assert t.loc["Martin-Lof", "Details"] == "needs calibrate(estimation='cml')"
        assert np.isfinite(t.loc["Overall Q1 (item scores)", "Value"]) and np.isfinite(t.loc["rho1", "Value"])
        assert np.isfinite(t.loc["Overall Q1 (item scores)", "p"])
        # CML: a test already run is kept, not rerun with the defaults; options passed through
        c = SLM(s.responses, exogenous=pd.DataFrame({"G": g}))
        c.calibrate(estimation="cml")
        c.global_fit(method="andersen", split="exogenous", covariate="G")
        c.global_stats_df(run_all_tests=True, test_kwargs={"martin-lof": {"p_method": "monte-carlo", "no_of_samples": 20, "seed": 1}})
    t = c.global_stats
    andersen = t.loc[["Andersen LR"]]
    assert len(andersen) == 1 and andersen["Details"].iloc[0].startswith("split=exogenous; covariate=G")
    assert "p_method=monte-carlo" in t.loc["Martin-Lof", "Details"]
    assert set(c.global_fit_results) == {("andersen", "split=exogenous; covariate=G"),
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


# ---------------------------------------------------------------------------
# PCM CML ridge (calibrate(ridge=...))
# ---------------------------------------------------------------------------


def _ridge_pcm_data(Ks, n=150, seed=5):
    from raschpy.simulation import PCM_Sim

    return PCM_Sim(no_of_items=len(Ks), no_of_persons=n, max_score_vector=Ks, seed=seed).responses


def _direct_ridge_penalty(Ks, ridge, group_widths=0.3):
    """The ridge penalty built directly, item by item: each item's centred thresholds minus its max-score
    group's mean centred thresholds (weight ridge), each group's mean widths minus their overall mean (weight
    group_widths); as b' Pb b / 2."""
    p = sum(Ks)
    off = np.concatenate([[0], np.cumsum(Ks)])
    G, W = [], []
    for K in sorted(set(Ks)):
        idx = [i for i, k in enumerate(Ks) if k == K]
        C = np.eye(K) - 1.0 / K
        mean = np.zeros((K, p))
        for j in idx:
            mean[:, off[j] : off[j + 1]] = np.eye(K) / len(idx)
        W.append(np.diff(mean, axis=0))
        if len(idx) > 1:
            for i in idx:
                own = np.zeros((K, p))
                own[:, off[i] : off[i + 1]] = np.eye(K)
                G.append(C @ (own - mean))
    W = np.vstack(W)
    W = W - W.mean(axis=0)
    return ridge * np.vstack(G).T @ np.vstack(G) + group_widths * W.T @ W


@pytest.mark.parametrize("Ks", [[4] * 10, [2, 3, 4, 5] * 3])
def test_pcm_ridge_cml_maximises_penalised_likelihood(Ks):
    """The ridge estimates maximise the conditional log-likelihood minus the penalty built directly; the
    weight is the 'minimise_bias' rule applied to the PAIR thresholds; items with unused categories are kept."""
    from raschpy import _rcl

    X = _ridge_pcm_data(Ks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = PCM(X, max_score_vector=Ks)
        m.calibrate(estimation="cml")
        p = PCM(X, max_score_vector=Ks)
        p.calibrate(estimation="pair")
    names = list(X.columns)
    assert m.convergence["converged"] and not m.convergence["dropped_items"]
    b0 = np.concatenate([p.thresholds_uncentred.loc[n].dropna().values for n in names])
    off = np.concatenate([[0], np.cumsum(Ks)])
    S = np.mean([np.ptp(b0[off[i] : off[i + 1]]) for i in range(len(Ks))])
    c0, c1, c2 = PCM.RIDGE_RULE
    assert m.ridge == pytest.approx(np.exp(c0 + c1 * np.log(S) + c2 * np.log(np.mean(Ks))), rel=1e-12)
    A = _rcl.zero_sum_basis(sum(Ks))
    Q = A.T @ _direct_ridge_penalty(Ks, m.ridge) @ A
    fit = _esf.fit_cml(X.to_numpy(float), A, b0 - b0.mean(), Ks, solver="newton", penalty=Q)
    b = np.concatenate([m.thresholds_uncentred.loc[n].dropna().values for n in names])
    np.testing.assert_allclose(b, fit["b"], atol=1e-7)
    assert m.convergence["penalised_log_likelihood"] == pytest.approx(fit["objective"], abs=1e-7)


def test_pcm_ridge_zero_is_plain_cml():
    X = _ridge_pcm_data([3] * 8, n=400, seed=11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = PCM(X, max_score_vector=[3] * 8)
        m.calibrate(estimation="cml", ridge=0)
    from raschpy import _rcl

    names = list(m.item_names)
    Ks = [3] * len(names)
    b = np.concatenate([m.thresholds_uncentred.loc[n].dropna().values for n in names])
    fit = _esf.fit_cml(np.array(m.responses.loc[:, names], float), _rcl.zero_sum_basis(len(b)), b, Ks)
    assert m.ridge == 0.0 and m._cml_penalty is None
    assert m.convergence["log_likelihood"] == pytest.approx(fit["ll"], abs=1e-8)
    assert fit["iterations"] <= 2


def test_pcm_ridge_unused_categories_options():
    """With the ridge, an item with an unused category is kept ('ridge', the default); ridge=0 drops it by
    default; 'ridge' needs ridge > 0; 'null' works with the ridge."""
    X = _keep_pcm_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = PCM(X, max_score_vector=[3] * 6)
        r.calibrate(estimation="cml")
        z = PCM(X, max_score_vector=[3] * 6)
        z.calibrate(estimation="cml", ridge=0)
        n = PCM(X, max_score_vector=[3] * 6)
        n.calibrate(estimation="cml", unused_categories="null")
    assert r._cml_unused == "ridge" and not r.convergence["dropped_items"]
    assert np.isfinite(r.thresholds_uncentred.to_numpy(float)).all()
    assert set(z.convergence["dropped_items"]) == set(X.columns[:2])
    assert n.convergence["converged"] and n.cml_unused_categories and n.ridge > 0
    with pytest.raises(ValueError, match="needs ridge > 0"):
        PCM(X, max_score_vector=[3] * 6).calibrate(estimation="cml", ridge=0, unused_categories="ridge")
    with pytest.raises(ValueError, match="estimation='cml' only"):
        PCM(X, max_score_vector=[3] * 6).calibrate(estimation="pair", unused_categories="ridge")


def test_pcm_ridge_argument_checks():
    X = _ridge_pcm_data([3] * 6, n=200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = PCM(X, max_score_vector=[3] * 6)
        a.calibrate(estimation="cml", ridge="minimize_bias")
        b = PCM(X, max_score_vector=[3] * 6)
        b.calibrate(estimation="cml")
        c = PCM(X, max_score_vector=[3] * 6)
        c.calibrate(estimation="cml", ridge=0.7)
    assert a.ridge == b.ridge and c.ridge == 0.7
    for bad in ("minimise_error", -0.1, True, None):
        with pytest.raises(ValueError, match="ridge must be"):
            PCM(X, max_score_vector=[3] * 6).calibrate(estimation="cml", ridge=bad)
    with pytest.warns(UserWarning, match="applies to estimation='cml', 'cp-full-rcl' and 'cpat-rcl' only"):
        PCM(X, max_score_vector=[3] * 6).calibrate(estimation="pair", ridge=0.5)


def test_pcm_ridge_test_refits_use_fixed_weight():
    """global_fit()'s group refits use the full-data fit's ridge weight, not the rule re-applied per group."""
    X = _ridge_pcm_data([3] * 8, n=300, seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = PCM(X, max_score_vector=[3] * 8)
        m.calibrate(estimation="cml")
        sub = m._global_fit_subset(X.index[:120])
    assert sub.ridge == m.ridge and sub._ridge_setting == m.ridge


def test_andersen_lr_too_few_common_items_message():
    """Plain CML where dropping items in the groups leaves fewer than two in common: a clear ValueError, not
    a failure inside the group refit."""
    import zlib
    from raschpy.simulation import PCM_Sim

    seed = zlib.crc32(b"nullA-3-100-13") % 2**31
    X = PCM_Sim(no_of_items=10, no_of_persons=100, max_score_vector=[3] * 10, person_sd=1.5, seed=seed).responses
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = PCM(X, max_score_vector=[3] * 10)
        m.calibrate(estimation="cml", ridge=0)
        with pytest.raises(ValueError, match="fewer than two items have CML estimates in both groups"):
            m.global_fit(method="andersen", split="score", seed=13)


def test_andersen_default_p_method_follows_ridge():
    """global_fit(method='andersen') defaults to Monte Carlo p-values (stopping after 10 exceedances)
    with PCM's CML ridge, chi-square with ridge=0; martin-lof stays chi-square."""
    X = _ridge_pcm_data([3] * 6, n=200, seed=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = PCM(X, max_score_vector=[3] * 6)
        r.calibrate(estimation="cml")
        z = PCM(X, max_score_vector=[3] * 6)
        z.calibrate(estimation="cml", ridge=0, unused_categories="null")
        a = r.global_fit(method="andersen", seed=1, no_of_samples=50)
        b = z.global_fit(method="andersen")
        c = r.global_fit(method="andersen", p_method="chi2")
        d = r.global_fit(method="martin-lof")
    assert a["p_method"] == "monte-carlo" and a["Monte Carlo samples"] <= 50
    assert b["p_method"] == c["p_method"] == d["p_method"] == "chi2"


# ---------------------------------------------------------------------------
# PCM estimation='cp-full-rcl' (full-pattern pairwise conditional likelihood with the ridge)
# ---------------------------------------------------------------------------


def _brute_pcm_rcl_loglik(X, Ks, b):
    """The composite log-likelihood summed directly over item pairs and patterns (a, b) vs (a - 1, b + 1)."""
    off = np.concatenate([[0], np.cumsum(Ks)])
    ll = 0.0
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            for a in range(1, Ks[i] + 1):
                for c in range(Ks[j]):
                    t = np.sum((X[:, i] == a) & (X[:, j] == c))
                    u = np.sum((X[:, i] == a - 1) & (X[:, j] == c + 1))
                    eta = b[off[j] + c] - b[off[i] + a - 1]
                    ll += -t * np.logaddexp(0, -eta) - u * np.logaddexp(0, eta)
    return ll


def test_pcm_rcl_maximises_penalised_composite_likelihood():
    """cp-full-rcl's estimates maximise the directly summed composite log-likelihood minus the ridge (the
    'cp-full-rcl' rule and group-width weight), with missing responses and mixed maximum scores; the person
    scores sum to its gradient."""
    from raschpy import _rcl
    from raschpy.simulation import PCM_Sim

    Ks = [2, 3, 4, 3, 2, 5]
    X = PCM_Sim(no_of_items=6, no_of_persons=250, max_score_vector=Ks, missing=0.1, seed=1).responses
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = PCM(X, max_score_vector=Ks)
        m.calibrate(estimation="cp-full-rcl")
        p = PCM(X, max_score_vector=Ks)
        p.calibrate(estimation="pair")
    names = list(X.columns)
    b0 = np.concatenate([p.thresholds_uncentred.loc[n].dropna().values for n in names])
    off = np.concatenate([[0], np.cumsum(Ks)])
    S = np.mean([np.ptp(b0[off[i] : off[i + 1]]) for i in range(len(Ks))])
    c0, c1, c2 = PCM.RCL_RIDGE_RULE
    assert m.ridge == pytest.approx(np.exp(c0 + c1 * np.log(S) + c2 * np.log(np.mean(Ks))), rel=1e-12)
    b = np.concatenate([m.thresholds_uncentred.loc[n].dropna().values for n in names])
    Xa = X.to_numpy(float)
    Pb = PCM._ridge_penalty(Ks, m.ridge, group_widths=PCM.RCL_RIDGE_GROUP_WIDTHS)
    A = _rcl.zero_sum_basis(len(b))

    def obj(th):
        bb = A @ th
        return _brute_pcm_rcl_loglik(Xa, Ks, bb) - 0.5 * bb @ Pb @ bb

    th = A.T @ b
    e = 1e-5
    grad = [(obj(th + e * v) - obj(th - e * v)) / (2 * e) for v in np.eye(len(th))]
    assert np.max(np.abs(grad)) < 1e-4
    assert m.convergence["log_likelihood"] == pytest.approx(_brute_pcm_rcl_loglik(Xa, Ks, b), abs=1e-8)
    D, t, u = _rcl.pcm_comparisons(Xa, Ks)
    S_p = _rcl.pcm_person_scores(Xa, Ks, b)
    np.testing.assert_allclose(S_p.sum(axis=0), D.T @ (t - (t + u) / (1 + np.exp(-(D @ b)))), atol=1e-9)


def test_pcm_rcl_options_and_standard_errors():
    X = _keep_pcm_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = PCM(X, max_score_vector=[3] * 6)
        m.calibrate(estimation="cp-full-rcl")
        m.std_errors()
    assert m._cml_unused == "ridge" and not m.convergence["dropped_items"] and m.se_method == "sandwich"
    assert np.isfinite(m.item_se.to_numpy()).all()
    with pytest.raises(ValueError, match="'null' applies to estimation='cml' only"):
        PCM(X, max_score_vector=[3] * 6).calibrate(estimation="cp-full-rcl", unused_categories="null")
    with pytest.raises(ValueError, match="se_method='hessian'"):
        m.std_errors(se_method="hessian")


def test_pcm_cml_start_from_rcl_same_estimates():
    """Ridge CML starts from the cp-full-rcl estimates: the same maximum as starting from PAIR."""
    from raschpy import _rcl

    Ks = [2, 3, 4, 5] * 3
    X = _ridge_pcm_data(Ks, n=200, seed=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = PCM(X, max_score_vector=Ks)
        m.calibrate(estimation="cml")
        p = PCM(X, max_score_vector=Ks)
        p.calibrate(estimation="pair")
    names = list(X.columns)
    b0 = np.concatenate([p.thresholds_uncentred.loc[n].dropna().values for n in names])
    A = _rcl.zero_sum_basis(len(b0))
    Q = A.T @ PCM._ridge_penalty(Ks, m.ridge) @ A
    ref = _esf.fit_cml(X.to_numpy(float), A, b0 - b0.mean(), Ks, solver="newton", penalty=Q)
    b = np.concatenate([m.thresholds_uncentred.loc[n].dropna().values for n in names])
    np.testing.assert_allclose(b, ref["b"], atol=1e-6)


# ---------------------------------------------------------------------------
# std_errors(no_data_se=): parameters the data do not inform (no_data_parameters)
# ---------------------------------------------------------------------------


def _no_data_pcm():
    from raschpy.simulation import PCM_Sim

    X = PCM_Sim(no_of_items=6, no_of_persons=500, max_score_vector=[3] * 6, seed=41).responses
    X[X.columns[0]] = X[X.columns[0]].replace(3, 2)  # top category unused: location not identified
    X[X.columns[1]] = X[X.columns[1]].replace(2, 1)  # interior category unused: location identified
    return X


@pytest.mark.parametrize("estimation", ["cml", "cp-full-rcl"])
def test_pcm_no_data_se(estimation):
    X = _no_data_pcm()
    a, b = X.columns[0], X.columns[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = PCM(X, max_score_vector=[3] * 6)
        m.calibrate(estimation=estimation)
        m.std_errors(se_method="sandwich", no_data_se="nan")
        sand = (m.item_se.copy(), {k: np.array(v) for k, v in m.threshold_se.items()})
    assert m.no_data_parameters == {
        a: dict(thresholds=[3], widths=[2], location=True),
        b: dict(thresholds=[2, 3], widths=[1, 2], location=False),
    }
    assert np.isnan(m.item_se[a]) and np.isfinite(m.item_se[b])
    assert np.isnan(m.threshold_se[b][1:]).all() and np.isfinite(m.threshold_se[b][0])
    with pytest.warns(UserWarning, match="no_data_se='posterior'"):
        m.std_errors(no_data_se="posterior", interval=0.95)
    assert np.isfinite(m.item_se[a]) and m.item_se[b] == pytest.approx(sand[0][b])
    # the other parameters keep their sandwich SEs
    others = [c for c in X.columns if c not in (a, b)]
    np.testing.assert_allclose(m.item_se[others].values, sand[0][others].values)
    lo = m.threshold_low[a][2]
    # analytic intervals: estimate +/- t * SE, t on persons - 1
    q = scipy_stats.t.ppf(0.975, m.no_of_persons - 1)
    assert lo == pytest.approx(m.thresholds_uncentred.loc[a, 3] - q * m.threshold_se[a][2], abs=1e-10)
    with pytest.raises(ValueError, match="no_data_se must be"):
        m.std_errors(no_data_se="zero")


def test_slm_no_data_se():
    from raschpy.simulation import SLM_Sim

    X = SLM_Sim(no_of_items=8, no_of_persons=80, seed=3).responses
    X[X.columns[0]] = 1.0  # everyone correct
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SLM(X)
        m.calibrate(estimation="cml", ridge=0.3)
        m.std_errors(no_data_se="nan")
        nan_se = m.item_se.copy()
        m.std_errors(no_data_se="posterior")
        p = SLM(X)
        p.calibrate(estimation="cml")
    assert list(m.no_data_parameters) == [X.columns[0]] and not m.convergence["dropped_items"]
    assert np.isnan(nan_se.iloc[0]) and np.isfinite(m.item_se.iloc[0])
    np.testing.assert_allclose(m.item_se.iloc[1:].values, nan_se.iloc[1:].values)
    assert X.columns[0] in p.convergence["dropped_items"]  # ridge=0: dropped as before


def test_rsm_no_data_se():
    """RSM: an empty category's thresholds and widths are rule-set (NaN under either option); a width with no
    comparison patterns under cp-full-rcl is set by the width ridge ('posterior' finite, 'nan' NaN)."""
    from raschpy.simulation import RSM_Sim

    X = RSM_Sim(no_of_items=8, no_of_persons=300, max_score=4, seed=2).responses.replace(2, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = RSM(X, max_score=4)
        m.calibrate()
        m.std_errors(se_method="sandwich", no_data_se="posterior")
    assert m.no_data_parameters == {"widths": {1: "rule", 2: "rule", 3: "rule"}, "thresholds": {2: "rule", 3: "rule"}}
    assert np.isnan(m.cat_width_se.values).all() and np.isnan(m.threshold_se[1:3]).all()
    rng = np.random.default_rng(0)
    rows = []
    while len(rows) < 600:  # each person answers two items, never in the patterns informing width 2
        r = np.full(5, np.nan)
        i, j = rng.choice(5, 2, replace=False)
        a, b = rng.integers(0, 5, 2)
        if (a == 2 and b == 2) or sorted((a, b)) == [1, 3]:
            continue
        r[i], r[j] = a, b
        rows.append(r)
    Y = pd.DataFrame(rows, columns=[f"I{k}" for k in range(5)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n = RSM(Y, max_score=4)
        n.calibrate()
        n.std_errors(se_method="sandwich", no_data_se="posterior")
        post = n.cat_width_se.copy()
        n.std_errors(se_method="sandwich", no_data_se="nan")
    assert n.no_data_parameters == {"widths": {2: "ridge"}, "thresholds": {}}
    assert np.isfinite(post[2]) and np.isnan(n.cat_width_se[2])
    np.testing.assert_allclose(post[[1, 3]].values, n.cat_width_se[[1, 3]].values)


def test_mfrm_no_data_se():
    """MFRM: a facet element x item cell with no responses (set by the ridge) gets the posterior SE or NaN,
    not the sandwich's near-zero; the other cells keep theirs."""
    from raschpy import MFRM
    from raschpy.simulation import MFRM_Sim

    X = MFRM_Sim(no_of_items=5, no_of_persons=300, no_of_facet_elements=4, max_score=3, model="items", seed=3).responses
    r0 = X.index.get_level_values(0).unique()[0]
    X.loc[r0, X.columns[0]] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MFRM(X, max_score=3)
        m.calibrate(model="items")
        m.std_errors(model="items", no_data_se="posterior")
        post = m.rater_se_items.copy()
        m.std_errors(model="items", no_data_se="nan")
    cell = (post.index[0], post.columns[0])
    assert m.no_data_parameters["items"]["cells"] == [cell] and m.no_data_parameters["items"]["kind"] == "ridge"
    assert post.loc[cell] > 0.5 and np.isnan(m.rater_se_items.loc[cell])
    others = m.rater_se_items.drop(index=cell[0])
    np.testing.assert_allclose(others.values, post.drop(index=cell[0]).values)


@pytest.mark.parametrize(
    "model", ["global", "items", "thresholds", "matrix", "bivector", "centrality", "pseudo_halo", "bistretch"]
)
def test_mfrm_no_data_se_empty_category(model):
    """A score category no response used: the shared thresholds and widths set by the empty-category rule get NaN
    SEs under either option, sandwich or bootstrap, in every sub-model; a partly affected threshold set keeps the
    SEs of the thresholds away from the empty category."""
    from raschpy import MFRM
    from raschpy.simulation import MFRM_Sim

    X = MFRM_Sim(no_of_items=5, no_of_persons=300, no_of_facet_elements=4, max_score=4, model="bivector",
                 seed=3).responses
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MFRM(X.replace(2, 1), max_score=4)  # category 2 empty: every width rule-set (K = 4)
        m.calibrate(model=model)
        m.std_errors(model=model, se_method="sandwich")
    assert m.no_data_parameters[model]["shared"] == dict(widths=[1, 2, 3], thresholds=[1, 2, 3, 4])
    assert np.isnan(getattr(m, f"threshold_se_{model}")).all()
    assert np.isnan(list(getattr(m, f"cat_width_se_{model}").values())).all()
    if model == "global":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            n = MFRM(MFRM_Sim(no_of_items=5, no_of_persons=300, no_of_facet_elements=4, max_score=6,
                              model="global", seed=3).responses.replace(1, 0), max_score=6)
            n.calibrate(model=model)
            n.std_errors(model=model, se_method="bootstrap", no_of_samples=20, seed=1, no_data_se="posterior")
        sh = n.no_data_parameters[model]["shared"]
        assert sh == dict(widths=[1, 2], thresholds=[1, 2])
        se = n.threshold_se_global
        assert np.isnan(se[:2]).all() and np.isfinite(se[2:]).all()


def test_category_stats_disordered_nan_for_unused_category():
    """category_stats_df: Disordered is True / False for the identified widths and NaN for those next to a
    category kept with unused_categories='null' (not identified; the extreme case of disorder)."""
    from raschpy.simulation import RSM_Sim
    X = RSM_Sim(no_of_items=6, no_of_persons=300, max_score=4, seed=4).responses
    X = X.where(X != 2, 1)
    m = RSM(X, max_score=4, validate=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(estimation="cml", unused_categories="null")
        m.std_errors()
    d = m.category_stats_df()["Disordered"].tolist()
    assert pd.isna(d[0]) and pd.isna(d[1]) and d[2] is False


def test_rsm_andersen_monte_carlo_when_a_group_needs_the_width_ridge():
    """global_fit(method='andersen') with RSM's CML and p_method=None: Monte Carlo when a group's (or the
    pooled) fit needed the unused-category width ridge, chi-square otherwise."""
    import zlib
    from raschpy.simulation import RSM_Sim
    out = {}
    for rep in (0, 3):  # (score groups use every category, the low group leaves the top one unused)
        seed = zlib.crc32(f"andersen-wr|150|{rep}".encode())
        persons = np.random.default_rng(seed).normal(-1.5, 1.0, 150)
        sim = RSM_Sim(no_of_items=8, no_of_persons=150, max_score=4, manual_thresholds=[-1.5, -1.0, 0.0, 2.5],
                      manual_persons=persons, seed=seed)
        m = RSM(sim.responses, max_score=4, validate=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.calibrate(estimation="cml")
            m.global_fit(method="andersen", no_of_samples=30, seed=1, progress=False)
        used = any(g._cml_penalty is not None for g in m.global_fit_groups.values())
        out[rep] = (used, m.global_fit_summary["p_method"])
    assert out[0] == (False, "chi2") and out[3] == (True, "monte-carlo")


def test_q1_small_samples_keep_two_groups():
    """Fewer than 2 * Q1_MIN_GROUP persons still give 2 groups (merging stops at 2), so Q1 is computed; a
    single group (supplied, or every location tied) gives no Q1 statistics, NaN overall and a warning."""
    from raschpy.base import Rasch
    from raschpy.simulation import SLM_Sim

    loc = np.random.default_rng(1).normal(size=40).round(1)
    assert len(np.unique(Rasch._ability_groups(loc)[0])) == 2
    assert len(np.unique(Rasch._ability_groups(np.zeros(50))[0])) == 1
    m = SLM(SLM_Sim(no_of_items=10, no_of_persons=40, seed=3).responses, validate=False)
    m.calibrate(estimation="cml")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit_statistics(conditional=True, q1_overall_samples=20)
    assert (m.item_q1_table["df"] == 1).all() and m.item_q1_table["p"].notna().all()
    assert m.item_q1_overall.loc["item scores", "Groups"] == 2
    assert np.isfinite(m.item_q1_overall.loc["item scores", "p"])
    one = pd.Series(0, index=m.responses.index)
    with pytest.warns(UserWarning, match="form 1 group"):
        m.fit_statistics(conditional=True, q1_groups=one, q1_overall_samples=20)
    assert m.item_q1_table["Q1"].isna().all() and m.item_q1_table["p"].isna().all()
    assert np.isnan(m.item_q1_overall.loc["item scores", "p"])


def test_pcm_constant_none_with_mixed_maximum_scores():
    """calibrate(constant=None): self.constant is None with mixed maximum scores (only the per-cell
    constants are applied) and the applied constant with one maximum score."""
    import numpy as np
    import pandas as pd
    from raschpy import PCM

    rng = np.random.default_rng(3)
    theta = rng.normal(size=300)
    maxes = [2, 2, 3, 3, 4]
    cols = {}
    for i, k in enumerate(maxes):
        tau = np.linspace(-1, 1, k) + 0.3 * i - 0.6
        logits = np.cumsum(np.c_[np.zeros(300), theta[:, None] - tau[None, :]], axis=1)
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        cols[f"I{i}"] = [rng.choice(k + 1, p=row) for row in p]
    df = pd.DataFrame(cols)

    mixed = PCM(df, max_score_vector=maxes)
    mixed.calibrate(estimation="pair")
    assert mixed.constant is None
    assert mixed.cell_constants is not None and np.all(mixed.cell_constants > 0)

    same = PCM(df[["I2", "I3"]], max_score_vector=[3, 3])
    same.calibrate(estimation="pair")
    assert same.constant is not None and same.constant > 0
    assert np.allclose(same.cell_constants, same.constant)

    given = PCM(df, max_score_vector=maxes)
    given.calibrate(estimation="pair", constant=0.2)
    assert given.constant == 0.2 and given.cell_constants is None
