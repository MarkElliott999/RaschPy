"""RSM vs PCM model_selection() on the object's own estimation (v2.0.0): composite likelihood ratio test for
'cp-full-rcl', 'cpat-rcl' and PAIR (method='log-lik'), CML LR for 'cml', conditional parametric bootstrap for PAIR's
other methods; deprecated arguments; the thin / unused category warning; PCM's 'cpat-rcl' estimator."""
import warnings

import numpy as np
import pandas as pd
import pytest

import raschpy as rp
from raschpy import _rcl, _selection
from raschpy.simulation import PCM_Sim, RSM_Sim


def _rsm_data(N=300, I=8, K=3, seed=2):
    return RSM_Sim(no_of_items=I, no_of_persons=N, max_score=K, seed=seed).responses


def _quiet(f):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f()


@pytest.mark.parametrize("est", ["cp-full-rcl", "cpat-rcl", "pair", "cml"])
def test_entry_points_agree(est):
    X = _rsm_data()
    r = rp.RSM(X, max_score=3)
    p = rp.PCM(X, max_score_vector=[3] * 8)
    _quiet(lambda: r.calibrate(estimation="pair-cpat" if est == "pair" else est))
    _quiet(lambda: p.calibrate(estimation=est))
    for m in (r, p):
        _quiet(lambda: m.model_selection(test="LR"))
    assert r.model_comparison_rsm_pcm_info["route"] == p.model_comparison_rsm_pcm_info["route"]
    assert r.model_comparison_rsm_pcm_lr == pytest.approx(p.model_comparison_rsm_pcm_lr, rel=1e-8)
    assert r.model_comparison_rsm_pcm_p == pytest.approx(p.model_comparison_rsm_pcm_p, rel=1e-8)


@pytest.mark.parametrize("est", ["cp-full-rcl", "cpat-rcl"])
def test_composite_statistic_is_the_composite_likelihood_difference(est):
    X = _rsm_data()
    r = rp.RSM(X, max_score=3)
    _quiet(lambda: r.calibrate(estimation=est))
    p = rp.PCM(X, max_score_vector=[3] * 8)
    _quiet(lambda: p.calibrate(estimation=est))
    names = list(X.columns)
    b0 = rp.RSM._stacked(rsm=r, names=names, K=3)
    b1 = rp.RSM._stacked(pcm=p, names=names)
    cpat = est == "cpat-rcl"
    D, t, u = _rcl.pcm_comparisons(X.to_numpy(float), [3] * 8, cpat)

    def ll(b):
        e = D @ (b - b.mean())
        return float((-t * np.logaddexp(0, -e) - u * np.logaddexp(0, e)).sum())

    out = _selection.composite_lr(X, 3, b0, b1, cpat=cpat)
    _quiet(lambda: r.model_selection(test="LR"))
    assert r.model_comparison_rsm_pcm_info["route"] == "composite"
    assert r.model_comparison_rsm_pcm_lr == pytest.approx(out["statistic"], rel=1e-10)
    assert out["statistic"] == pytest.approx(max(2 * (ll(b1) - ll(b0)), 0), rel=1e-10)
    assert out["df"] == 7 * 2 and 1 < out["weight_min"] <= out["weight_max"]
    assert out["penalty_pcm"] > out["penalty_rsm"] > 0


def test_pair_objective_maximum_reproduces_pair():
    X = _rsm_data(seed=4)
    p = rp.PCM(X, max_score_vector=[3] * 8)
    _quiet(lambda: p.calibrate(estimation="pair"))
    b_pair = rp.PCM._stacked(pcm=p, names=list(X.columns))
    b_bt = _selection.bt_restricted_fit(p._pair_bt_matrix, _rcl.zero_sum_basis(24))
    assert np.abs((b_bt - b_bt.mean()) - (b_pair - b_pair.mean())).max() < 1e-3
    q = rp.PCM(X, max_score_vector=[3] * 8)
    _quiet(lambda: q.calibrate(estimation="pair", method="cos"))
    assert q._pair_bt_matrix is None


def test_composite_test_detects_pcm_and_keeps_rsm():
    keep = rp.RSM(_rsm_data(N=500, seed=7), max_score=3)
    _quiet(lambda: keep.calibrate())
    _quiet(lambda: keep.model_selection(test="LR"))
    assert keep.model_comparison_rsm_pcm_lr_preferred == "RSM"
    X = PCM_Sim(no_of_items=8, no_of_persons=800, max_score_vector=[3] * 8, seed=3).responses
    drop = rp.PCM(X, max_score_vector=[3] * 8)
    _quiet(lambda: drop.calibrate())
    _quiet(lambda: drop.model_selection(test="LR"))
    assert drop.model_comparison_rsm_pcm_lr_preferred == "PCM" and drop.model_comparison_rsm_pcm_p < 0.001
    for t in ("AIC", "BIC"):
        _quiet(lambda: drop.model_selection(test=t))
    assert drop.model_comparison_rsm_pcm_aic_summary.name.endswith("(composite)")


def test_bootstrap_route():
    X = _rsm_data(N=150, I=6)
    p = rp.PCM(X, max_score_vector=[3] * 6)
    _quiet(lambda: p.calibrate(estimation="pair", method="cos"))
    _quiet(lambda: p.model_selection(test="LR", no_of_samples=19, seed=3))
    a = p.model_comparison_rsm_pcm_lr_summary
    _quiet(lambda: p.model_selection(test="LR", no_of_samples=19, seed=3))
    assert p.model_comparison_rsm_pcm_info["route"] == "bootstrap" and a["Resamples"] == 19
    assert a["p-value"] == p.model_comparison_rsm_pcm_lr_summary["p-value"] and 0 < a["p-value"] <= 1
    with pytest.raises(ValueError, match="only the bootstrap test"):
        _quiet(lambda: p.model_selection(test="AIC"))


def test_deprecated_arguments_warn():
    r = rp.RSM(_rsm_data(), max_score=3)
    _quiet(lambda: r.calibrate())
    with pytest.warns(FutureWarning, match="likelihood, sampling no longer apply"):
        r.model_selection(likelihood="joint", sampling="auto")


def test_thin_category_warning():
    rng = np.random.default_rng(1)
    I, K, N = 8, 3, 150
    d = rng.uniform(-1, 1, I)
    tau = np.array([-1.5, 5.0, -3.5])  # category 2 almost never chosen
    th = rng.normal(0, 1.5, N)
    LE = np.concatenate([np.zeros((I, 1)), -np.cumsum(d[:, None] + tau[None, :], axis=1)], axis=1)
    lw = LE[None] + np.arange(K + 1)[None, None, :] * th[:, None, None]
    pr = np.exp(lw - lw.max(2, keepdims=True)); pr /= pr.sum(2, keepdims=True)
    X = pd.DataFrame((rng.random((N, I, 1)) > np.cumsum(pr, axis=2)).sum(2).astype(float),
                     columns=[f"Item_{i + 1}" for i in range(I)])
    r = rp.RSM(X, max_score=K)
    _quiet(lambda: r.calibrate())
    with pytest.warns(UserWarning, match="RSM vs PCM: .*not reliable"):
        r.model_selection(test="LR")
    ok = rp.RSM(_rsm_data(), max_score=3)
    _quiet(lambda: ok.calibrate())
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ok.model_selection(test="LR")
    assert not [x for x in w if "RSM vs PCM" in str(x.message)]


def test_composite_aic_gate_is_equal_size():
    """Under 'cp-full-rcl', AIC's significance test is the composite LR test at the size AIC's rule has with a
    likelihood: p is the relative likelihood of the chi-square(df) ratio with the composite test's tail probability."""
    from scipy.stats import chi2

    X = PCM_Sim(no_of_items=8, no_of_persons=400, max_score_vector=[3] * 8, seed=4).responses
    r = rp.RSM(X, max_score=3)
    _quiet(r.calibrate)
    _quiet(lambda: r.model_selection(test="AIC"))
    _quiet(lambda: r.model_selection(test="LR"))
    g = r.model_comparison_rsm_pcm_aic_gate
    aic = r.model_comparison_rsm_pcm_aic
    if aic["PCM"] < aic["RSM"]:
        assert g["tail p"] == pytest.approx(r.model_comparison_rsm_pcm_p, rel=1e-10)
        lr = chi2.isf(g["tail p"], g["df"])
        assert r.model_comparison_rsm_pcm_aic_p == pytest.approx(np.exp(-abs(lr - 2 * g["df"]) / 2), rel=1e-8)
    else:
        assert g is None
    level = chi2.sf(2 * g["df"] + 2 * np.log(20), g["df"]) if g else None
    assert (r.model_comparison_rsm_pcm_aic_preferred == "PCM") == bool(g and g["tail p"] <= level)


@pytest.mark.parametrize("cls", [rp.RSM, rp.PCM])
def test_default_is_the_lr_test(cls):
    """test='LR' is the default (v2.0.0): the default call gives the LR test's results, and it runs on PAIR's
    'cos' fits, whose only test is the bootstrap LR test."""
    X = _rsm_data(N=150, I=6)
    m = cls(X, max_score=3) if cls is rp.RSM else cls(X, max_score_vector=[3] * 6)
    _quiet(lambda: m.calibrate())
    _quiet(lambda: m.model_selection())
    a = m.model_comparison_rsm_pcm_lr_summary
    _quiet(lambda: m.model_selection(test="LR"))
    pd.testing.assert_series_equal(a, m.model_comparison_rsm_pcm_lr_summary)
    assert a["Preferred"] == ("PCM" if a["p-value"] <= 0.05 else "RSM")
    c = cls(X, max_score=3) if cls is rp.RSM else cls(X, max_score_vector=[3] * 6)
    _quiet(lambda: c.calibrate(estimation="pair-cpat" if cls is rp.RSM else "pair", method="cos"))
    _quiet(lambda: c.model_selection(no_of_samples=9, seed=1))
    assert c.model_comparison_rsm_pcm_info["route"] == "bootstrap"


@pytest.mark.parametrize("cls, kw", [(rp.PCM, dict(estimation="pair", method="cos")),
                                     (rp.RSM, dict(estimation="pair-cpat", method="ls"))])
def test_bootstrap_statistic_compares_the_fitted_models(cls, kw):
    """The bootstrap test's statistic is the sum of squared differences between PCM's and RSM's fitted thresholds,
    each item's taken relative to its central location, for the object's own fit and the other model fitted with
    the same estimator."""
    from raschpy.base import Rasch
    X = _rsm_data(N=150, I=6)
    names = list(X.columns)
    m = cls(X, max_score=3) if cls is rp.RSM else cls(X, max_score_vector=[3] * 6)
    _quiet(lambda: m.calibrate(**kw))
    _quiet(lambda: m.model_selection(no_of_samples=9, seed=1))
    if cls is rp.RSM:
        r = m
        p = rp.PCM(X, max_score_vector=[3] * 6)
        _quiet(lambda: p.calibrate(estimation="pair", method=kw["method"]))
    else:
        p = m
        r = rp.RSM(X, max_score=3)
        _quiet(lambda: r.calibrate(estimation="pair-cpat", method=kw["method"]))
    c0 = Rasch._stacked(rsm=r, names=names, K=3).reshape(6, 3)
    c1 = Rasch._stacked(pcm=p, names=names).reshape(6, 3)
    d = (c1 - c1.mean(1, keepdims=True)) - (c0 - c0.mean(1, keepdims=True))
    assert m.model_comparison_rsm_pcm_lr_summary["Statistic"] == pytest.approx(float((d ** 2).sum()), rel=1e-10)


def test_pcm_cpat_rcl_estimator():
    """PCM 'cpat-rcl': the maximum of the composite likelihood of CPAT's patterns (|b - (a - 1)| <= 1) with the
    ridge; the persons' scores sum to its gradient; sandwich standard errors; close to 'cp-full-rcl'."""
    X = PCM_Sim(no_of_items=8, no_of_persons=400, max_score_vector=[3] * 8, seed=5).responses
    m = rp.PCM(X, max_score_vector=[3] * 8)
    _quiet(lambda: m.calibrate(estimation="cpat-rcl"))
    assert m.estimation == "cpat-rcl" and m.convergence["converged"]
    D, t, u = _rcl.pcm_comparisons(X.to_numpy(float), [3] * 8, cpat=True)
    full = _rcl.pcm_comparisons(X.to_numpy(float), [3] * 8)[0]
    assert 0 < D.shape[0] < full.shape[0]
    rows, cols = D.nonzero()
    lab = np.arange(24) % 3  # threshold number - 1 of each column
    pair = np.zeros((D.shape[0], 2), int)
    for r_, c_ in zip(rows, cols):
        pair[r_, int(D[r_, c_] < 0)] = lab[c_]  # (b, a - 1)
    assert np.abs(pair[:, 0] - pair[:, 1]).max() <= 1
    b = m._rcl_state["b"]
    s = 1 / (1 + np.exp(-(D @ b)))
    grad = D.T @ (t - (t + u) * s)
    S = _rcl.pcm_person_scores(X.to_numpy(float), [3] * 8, b, cpat=True)
    np.testing.assert_allclose(S.sum(0), grad, atol=1e-8)
    A = m._rcl_state["A"]
    Pb = m._ridge_penalty([3] * 8, m.ridge, group_widths=m.RCL_RIDGE_GROUP_WIDTHS)
    np.testing.assert_allclose(A.T @ (grad - Pb @ b), 0, atol=1e-6)  # the penalised maximum
    _quiet(m.std_errors)
    assert np.all(np.isfinite(m.item_se)) and np.all(m.item_se > 0)
    f = rp.PCM(X, max_score_vector=[3] * 8)
    _quiet(lambda: f.calibrate(estimation="cp-full-rcl"))
    assert np.corrcoef(m.items, f.items)[0, 1] > 0.99


def test_pcm_cpat_rcl_unmatched_threshold_warns():
    X = PCM_Sim(no_of_items=6, no_of_persons=300, max_score_vector=[1, 1, 2, 2, 5, 3], seed=2).responses
    m = rp.PCM(X, max_score_vector=[1, 1, 2, 2, 5, 3])
    with pytest.warns(UserWarning, match="Item_5 threshold 5 is not in any of CPAT's patterns"):
        m.calibrate(estimation="cpat-rcl")
