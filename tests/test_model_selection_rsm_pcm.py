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
    with pytest.warns(FutureWarning, match="likelihood no longer applies"):
        r.model_selection(likelihood="joint")


@pytest.mark.parametrize("bad, match", [
    (dict(sampling="dynamic", delta=0.1), "cannot be combined"),
    (dict(sampling="every"), "sampling must be"),
    (dict(sampling=1), "at least 2"),
    (dict(sampling=100, delta=0.1), "cannot be combined"),
    (dict(test="BIC", sampling="power"), "applies to test='LR' only"),
    (dict(sampling="power", delta=0), "needs delta > 0"),
    (dict(delta_units="sd"), "delta_units must be"),
    (dict(delta_measure="item"), "delta_measure must be"),
])
def test_sampling_and_delta_argument_errors(bad, match):
    r = rp.RSM(_rsm_data(), max_score=3)
    _quiet(lambda: r.calibrate())
    with pytest.raises(ValueError, match=match):
        r.model_selection(**bad)


@pytest.mark.parametrize("est", ["cml", "cp-full-rcl", "cpat-rcl"])
def test_close_fit_power_and_delta_hat(est):
    """Delta-hat is the bias-corrected size of the departure from RSM; delta=0 is the plain test; the close-fit p
    rises with delta; sampling='power' gives the sample size and the projected decision."""
    from raschpy import _selection
    X = PCM_Sim(no_of_items=8, no_of_persons=600, max_score_vector=[3] * 8, seed=4).responses
    r = rp.RSM(X, max_score=3)
    _quiet(lambda: r.calibrate(estimation=est))
    _quiet(lambda: r.model_selection(test="LR", delta=0))
    s = r.model_comparison_rsm_pcm_lr_summary
    # information-weighted size: about 0.29-0.30 widths for all three estimations (0.37-0.39 unweighted)
    assert s["Units"] == "width" and 0.25 < s["Delta-hat"] < 0.35 and s["Preferred"] == "PCM"
    assert r.model_comparison_rsm_pcm_close_fit_p is None
    ps = []
    for d in (0.1, 0.3, 0.6):
        _quiet(lambda: r.model_selection(test="LR", delta=d))
        ps.append(r.model_comparison_rsm_pcm_close_fit_p)
    assert ps[0] <= ps[1] <= ps[2] and ps[0] <= 0.05 < ps[2]
    assert r.model_comparison_rsm_pcm_lr_preferred == "RSM"  # a departure of 0.3 widths is not significantly > 0.6
    _quiet(lambda: r.model_selection(test="LR", delta=0.4, delta_units="logits"))
    assert r.model_comparison_rsm_pcm_lr_summary["Units"] == "logits"
    _quiet(lambda: r.model_selection(sampling="power", delta=0.3))
    pw = r.model_comparison_rsm_pcm_power
    assert pw["decision"] == "projected" and 2 <= pw["n"] < pw["n_data"] and pw["projected_rejection"] > 0.5
    _quiet(lambda: r.model_selection(test="LR", delta=0, min_effect=1.0))
    assert r.model_comparison_rsm_pcm_lr_preferred == "RSM"  # Delta-hat about 0.3 widths


def _one_item_data(N=400, I=16, K=3, shift=0.5, seed=3):
    """RSM data except that item 1's thresholds depart from the shared ones by an RMS of shift category widths."""
    rng = np.random.default_rng(seed)
    d, tau = np.linspace(-1, 1, I), 1.2 * (np.arange(K) - (K - 1) / 2)
    B = d[:, None] + tau[None, :]
    B[0] += shift * 1.2 * np.array([1.0, -2.0, 1.0][:K]) / np.sqrt(2.0)
    th = rng.normal(0, 1.5, N)
    LE = np.concatenate([np.zeros((N, I, 1)), np.cumsum(th[:, None, None] - B[None], axis=2)], axis=2)
    p = np.exp(LE - LE.max(2, keepdims=True))
    p /= p.sum(2, keepdims=True)
    X = (rng.random((N, I, 1)) > np.cumsum(p, axis=2)[:, :, :K]).sum(2)
    return pd.DataFrame(X, columns=[f"Item_{i + 1}" for i in range(I)])


@pytest.mark.parametrize("est", ["cml", "cp-full-rcl", "cpat-rcl"])
def test_per_item_close_fit_finds_one_deviant_item(est):
    """One item departing by 0.5 widths among 16: the overall close-fit test (RMS over all items) does not reject
    at delta=0.1, the per-item test does, and names the item; delta_measure picks which tests decide."""
    r = rp.RSM(_one_item_data(), max_score=3)
    _quiet(lambda: r.calibrate(estimation=est))
    _quiet(lambda: r.model_selection(test="LR", delta=0.1))
    s, t = r.model_comparison_rsm_pcm_lr_summary, r.model_comparison_rsm_pcm_item_close_fit
    assert s["p-value (close fit)"] > 0.05 and s["p-value (close fit, per item)"] < 0.01
    assert s["Preferred"] == "PCM" and t["p-value"].idxmin() == "Item_1" and t["Departure"].idxmax() == "Item_1"
    assert s["p-value (close fit, per item)"] == pytest.approx(min(1.0, 16 * t["p-value"].min()))
    _quiet(lambda: r.model_selection(test="LR", delta=0.1, delta_measure="overall"))
    assert r.model_comparison_rsm_pcm_lr_preferred == "RSM" and r.model_comparison_rsm_pcm_item_close_fit is None
    _quiet(lambda: r.model_selection(test="LR", delta=0.1, delta_measure="largest"))
    assert r.model_comparison_rsm_pcm_lr_preferred == "PCM"
    _quiet(lambda: r.model_selection(test="LR", delta=0))  # the plain test, no close-fit tests
    assert r.model_comparison_rsm_pcm_close_fit_p_item is None


def test_power_sample_size_matches_the_chi_square_power():
    """With equal weights (a full likelihood), power_sample_size is the non-central chi-square's."""
    from scipy.stats import chi2, ncx2
    from raschpy import _selection
    lam, b = np.ones(6), np.full(6, 0.5)  # non-centrality 1.5 at n_obs = 100
    n = _selection.power_sample_size(lam, b, 1.0, 0.05, 0.8, 100)
    crit = chi2.isf(0.05, 6)
    assert ncx2.sf(crit, 6, 1.5 * n / 100) >= 0.8 > ncx2.sf(crit, 6, 1.5 * (n - 1) / 100)


def test_subsampling_refits_and_records_the_seed():
    X = PCM_Sim(no_of_items=6, no_of_persons=400, max_score_vector=[3] * 6, seed=4).responses
    r = rp.RSM(X, max_score=3)
    _quiet(r.calibrate)
    _quiet(lambda: r.model_selection(sampling=150, n_subsamples=3, seed=7))
    a = r.model_comparison_rsm_pcm_sampling
    assert a["seed"] == 7 and len(a["p"]) == 3 and a["share_rejecting"] in (0, 1 / 3, 2 / 3, 1)
    assert r.model_comparison_rsm_pcm_lr_summary["Subsample size"] == 150
    _quiet(lambda: r.model_selection(sampling=150, n_subsamples=3, seed=7))
    assert r.model_comparison_rsm_pcm_sampling["p"] == a["p"]
    _quiet(lambda: r.model_selection(sampling=150))
    assert isinstance(r.model_comparison_rsm_pcm_sampling["seed"], int)
    _quiet(lambda: r.model_selection(sampling=10_000))
    assert r.model_comparison_rsm_pcm_sampling["decision"].startswith("full data")


@pytest.mark.parametrize("cls", ["RSM", "PCM"])
def test_dynamic_sampling_refits_at_the_v140_size(cls):
    X = PCM_Sim(no_of_items=6, no_of_persons=400, max_score_vector=[3] * 6, seed=4).responses
    r = rp.RSM(X, max_score=3) if cls == "RSM" else rp.PCM(X, max_score_vector=[3] * 6)
    _quiet(r.calibrate)
    _quiet(lambda: r.model_selection(sampling="dynamic", seed=7))
    a = r.model_comparison_rsm_pcm_sampling
    assert a["dynamic"] is True and a["n"] == min(20 * 5 * 2, 1500) == 200 and len(a["p"]) == 1
    assert r.model_comparison_rsm_pcm_lr_summary["Subsample size"] == 200
    _quiet(lambda: r.model_selection(sampling=200, seed=7))
    assert r.model_comparison_rsm_pcm_sampling["p"] == a["p"]
    assert r.model_comparison_rsm_pcm_sampling["dynamic"] is False


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
    # relabelled outputs: a relative likelihood (v1.4.0's _aic_p kept as an alias) and the level of the test
    s = r.model_comparison_rsm_pcm_aic_summary
    assert "p-value" not in s.index
    assert s["Relative likelihood"] == r.model_comparison_rsm_pcm_aic_relative_likelihood == r.model_comparison_rsm_pcm_aic_p
    df = r.model_comparison_rsm_pcm_lr_summary["df"]
    assert s["Test level"] == r.model_comparison_rsm_pcm_aic_level == pytest.approx(
        chi2.sf(2 * df + 2 * np.log(20), df), rel=1e-12)
    assert s["Test level"] < 0.05


@pytest.mark.parametrize("cls", [rp.RSM, rp.PCM])
def test_default_is_aic_with_close_fit(cls):
    """test=None is the default (v2.0.0): AIC with the relative-likelihood check and the close-fit test at delta=0.1
    widths where there is a likelihood, and the bootstrap LR test on PAIR's 'cos' fits, whose only test it is."""
    X = _rsm_data(N=150, I=6)
    m = cls(X, max_score=3) if cls is rp.RSM else cls(X, max_score_vector=[3] * 6)
    _quiet(lambda: m.calibrate())
    _quiet(lambda: m.model_selection())
    a = m.model_comparison_rsm_pcm_aic_summary
    assert m.model_comparison_rsm_pcm_test == "AIC" and a["delta"] == 0.1 and "p-value (close fit)" in a
    _quiet(lambda: m.model_selection(test="AIC", aic_relative_likelihood=True, delta=0.1))
    pd.testing.assert_series_equal(a, m.model_comparison_rsm_pcm_aic_summary)
    c = cls(X, max_score=3) if cls is rp.RSM else cls(X, max_score_vector=[3] * 6)
    _quiet(lambda: c.calibrate(estimation="pair-cpat" if cls is rp.RSM else "pair", method="cos"))
    _quiet(lambda: c.model_selection(no_of_samples=9, seed=1))
    assert c.model_comparison_rsm_pcm_info["route"] == "bootstrap" and c.model_comparison_rsm_pcm_test == "LR"
    _quiet(lambda: m.model_selection(sampling=100, n_subsamples=1, seed=1))
    assert m.model_comparison_rsm_pcm_test == "LR"


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


def test_aic_sig_test_is_a_deprecated_alias():
    """aic_sig_test= (v1.4.0) is the old name of aic_relative_likelihood=: FutureWarning, and its value is used."""
    X = PCM_Sim(no_of_items=6, no_of_persons=300, max_score_vector=[3] * 6, seed=4).responses
    r = rp.RSM(X, max_score=3)
    _quiet(r.calibrate)
    with pytest.warns(FutureWarning, match="aic_relative_likelihood"):
        r.model_selection(test="AIC", aic_sig_test=False)
    assert "Relative likelihood" not in r.model_comparison_rsm_pcm_aic_summary.index
    _quiet(lambda: r.model_selection(test="AIC"))
    assert {"Relative likelihood", "Test level"} <= set(r.model_comparison_rsm_pcm_aic_summary.index)


def _pcm_with_departure(N, seed, I=8, K=3, size=0.4, missing=0.0):
    """PCM data whose thresholds depart from an RSM structure by a known pattern; returns (X, true thresholds)."""
    rng = np.random.default_rng(seed)
    d = np.linspace(-1.2, 1.2, I)
    tau = 1.2 * (np.arange(K) - (K - 1) / 2)
    dev = np.random.default_rng(99).normal(0, 1, (I, K))
    dev -= dev.mean(1, keepdims=True); dev -= dev.mean(0, keepdims=True)
    B = d[:, None] + tau[None, :] + dev * size / np.sqrt(np.mean(dev**2))
    B -= B.mean()
    th = rng.normal(0, 1.5, N)
    LE = np.concatenate([np.zeros((I, 1)), -np.cumsum(B, 1)], 1)
    lw = LE[None] + np.arange(K + 1)[None, None, :] * th[:, None, None]
    p = np.exp(lw - lw.max(2, keepdims=True)); p /= p.sum(2, keepdims=True)
    X = (rng.random((N, I, 1)) > np.cumsum(p, 2)).sum(2).astype(float)
    if missing:
        X[rng.random(X.shape) < missing] = np.nan
    return pd.DataFrame(X, columns=[f"Item_{i + 1}" for i in range(I)]), B


@pytest.mark.parametrize("est", ["cml", "cp-full-rcl", "cpat-rcl"])
def test_close_fit_null_matches_a_large_sample(est):
    """The shift computed from expected responses for the true thresholds equals the (composite) LR statistic's
    excess over its null mean (the sum of its weights) in a large sample from them."""
    X, B = _pcm_with_departure(40000, seed=1)
    r = rp.RSM(X, max_score=3)
    _quiet(lambda: r.calibrate(estimation=est))
    _quiet(lambda: r.model_selection())
    P = r._rsm_pcm_close_fit_parts
    nc, _ = P["exact"]._expected_fit(B.reshape(-1) - B.mean())
    lam = P["exact"].lam
    excess = P["stat"] - lam.sum()
    assert abs(excess - nc) < 3 * np.sqrt(2 * (lam**2).sum() + 4 * lam.max() * nc)


@pytest.mark.parametrize("missing", [0.0, 0.2])
@pytest.mark.parametrize("est", ["cml", "cp-full-rcl"])
def test_close_fit_null_delta_hat_p_and_power_agree(est, missing):
    """Delta-hat is the size whose non-centrality equals the statistic's excess over df; at delta = Delta-hat the
    close-fit p is about 0.5 (the statistic sits at its median); the power-based sample size gives the requested power.
    Holds with missing responses."""
    from scipy.stats import chi2, ncx2
    X, _ = _pcm_with_departure(800, seed=2, missing=missing)
    r = rp.RSM(X, max_score=3)
    _quiet(lambda: r.calibrate(estimation=est))
    _quiet(lambda: r.model_selection(delta=0, delta_units="logits"))
    P = r._rsm_pcm_close_fit_parts
    Z, stat = P["exact"], P["stat"]
    excess = stat - Z.lam.sum()
    dh = r.model_comparison_rsm_pcm_delta_hat
    assert dh > 0 and abs(Z.nc(dh) - excess) < 1e-3 * excess
    assert 0.35 < Z.p(stat, dh) < 0.65
    n = Z.sample_size(dh / 2, 0.05, 0.8, len(X))
    assert abs(Z.power(dh / 2, 0.05, n / len(X)) - 0.8) < 1e-6
    if est == "cml":
        df = P["df"]
        assert abs(ncx2.sf(chi2.isf(0.05, df), df, Z.nc(dh / 2) * n / len(X)) - 0.8) < 1e-6
    _quiet(lambda: r.model_selection(delta=dh / 2, delta_units="logits"))
    assert r.model_comparison_rsm_pcm_close_fit_p == pytest.approx(Z.p(stat, dh / 2))


@pytest.mark.parametrize("cls", ["RSM", "PCM"])
def test_model_selection_ridge_sets_pcm_ridge_for_the_comparison(cls):
    """model_selection(ridge=r) compares RSM with PCM fitted with ridge r: under CML the LR statistic is twice the
    difference between that PCM fit's and RSM's conditional log-likelihoods. A PCM object's own estimates and ridge
    are unchanged."""
    X, _ = _pcm_with_departure(400, seed=3)
    m = rp.RSM(X, max_score=3) if cls == "RSM" else rp.PCM(X, max_score_vector=[3] * 8)
    _quiet(lambda: m.calibrate(estimation="cml"))
    before = (getattr(m, "ridge", None), repr(m.thresholds_uncentred) if cls == "PCM" else None)
    _quiet(lambda: m.model_selection(test="LR", ridge=0.0))
    stat = m.model_comparison_rsm_pcm_lr_summary["LR statistic"]
    p0, r0 = rp.PCM(X, max_score_vector=[3] * 8), rp.RSM(X, max_score=3)
    _quiet(lambda: p0.calibrate(estimation="cml", ridge=0.0))
    _quiet(lambda: r0.calibrate(estimation="cml"))
    want = 2 * (p0.convergence["log_likelihood"] - r0.convergence["log_likelihood"])
    assert stat == pytest.approx(max(want, 0.0), rel=1e-6)
    if cls == "PCM":
        assert m.ridge == before[0] and repr(m.thresholds_uncentred) == before[1]


def test_model_selection_ridge_needs_a_likelihood():
    X, _ = _pcm_with_departure(300, seed=4)
    m = rp.RSM(X, max_score=3)
    _quiet(lambda: m.calibrate(estimation="pair-cpat"))
    with pytest.raises(ValueError, match="ridge"):
        m.model_selection(ridge=1.0)


def test_sampling_auto_is_none():
    r = rp.RSM(_rsm_data(), max_score=3)
    _quiet(lambda: r.calibrate())
    _quiet(lambda: r.model_selection(test="LR", delta=0))
    p = r.model_comparison_rsm_pcm_p
    _quiet(lambda: r.model_selection(test="LR", delta=0, sampling="auto"))
    assert r.model_comparison_rsm_pcm_p == p and r.model_comparison_rsm_pcm_sampling is None
