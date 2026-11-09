"""MFRM.model_selection(): the comparison follows the candidates' estimation -- composite LR / CL-AIC / CL-BIC for
'cp-full-rcl' (the full pattern set) and 'cpat-rcl' (CPAT's patterns), conditional log-likelihoods for 'cml', and a
conditional parametric bootstrap (test='LR' only) for 'pair-cpat'."""

import warnings

import numpy as np
import pandas as pd
import pytest

import raschpy as rp
from raschpy import _esf
from raschpy.simulation import MFRM_Sim

MODELS = ["global", "items", "bivector", "centrality", "bistretch"]


def _resp():
    return MFRM_Sim(no_of_items=5, no_of_persons=300, no_of_facet_elements=4, max_score=3, model="items",
                    seed=11).responses


def test_mfrm_selection_composite_frozen_values():
    m = rp.MFRM(_resp())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.model_selection(test="LR", candidate_models=MODELS)
    assert m.model_comparison_mfrm_info == {"route": "composite", "estimation": "cp-full-rcl"}
    t = m.model_comparison_mfrm_lr_summary
    assert list(t.columns) == ["LR", "df", "Scale", "df (Satterthwaite)", "p-value", "Effect", "Preferred"]
    expected = {
        "global vs items": (1582.3719824703803, 12, 3.7962495483470917, 2.20378253146278e-68),
        "items vs bivector": (14.471812913034228, 6, 5.372017430675335, 0.8037915221493723),
        "global vs centrality": (101.36613548859532, 3, 3.909606725694843, 1.131418431490544e-05),
        "centrality vs bistretch": (582.1688447489469, 3, 3.081213827454513, 5.780452217611148e-39),
    }
    for lab, (lr, df, scale, p) in expected.items():
        assert np.isclose(t.loc[lab, "LR"], lr, rtol=1e-6)
        assert t.loc[lab, "df"] == df
        assert np.isclose(t.loc[lab, "Scale"], scale, rtol=1e-6)
        assert np.isclose(t.loc[lab, "p-value"], p, rtol=1e-4)
    assert t.loc["items vs bivector", "Preferred"] == "items"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.model_selection(candidate_models=MODELS)
    pen = m.model_comparison_mfrm_aic_summary["Penalty"]
    assert np.isclose(pen["items"], 66.13772061414947, rtol=1e-6)
    assert np.isclose(pen["global"], 21.812508006576586, rtol=1e-6)
    assert m.model_comparison_mfrm_aic_preferred == "items"
    gate = m.model_comparison_mfrm_aic_gate
    assert gate["df"] == 12 and np.isclose(gate["tail p"], 2.20378253146278e-68, rtol=1e-4)


def test_mfrm_selection_equal_size_gate():
    from scipy.stats import chi2

    resp = MFRM_Sim(no_of_items=5, no_of_persons=300, no_of_facet_elements=4, max_score=3, model="global",
                    seed=12).responses
    m = rp.MFRM(resp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.model_selection()
    g = m.model_comparison_mfrm_aic_gate
    assert np.isclose(g["tail p"], 0.08710589913760669, rtol=1e-6)
    # p is the relative likelihood of the chi-square(df) ratio with the same tail probability
    lr = chi2.isf(g["tail p"], g["df"])
    assert np.isclose(m.model_comparison_mfrm_aic_p, np.exp(-abs(lr - 2 * g["df"]) / 2))
    assert np.isclose(m.model_comparison_mfrm_aic_p, 0.7536204456348881, rtol=1e-4)
    assert m.model_comparison_mfrm_aic_preferred == "global"


def test_mfrm_selection_cml_scores_the_fits():
    mc = rp.MFRM(_resp())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mc.calibrate(model="global", estimation="cml")
        mc.model_selection(test="BIC", candidate_models=["global", "items"])
    assert mc.model_comparison_mfrm_info["route"] == "cml"
    Xv, thr, _ = mc._virtual_items("items")
    assert np.isclose(mc.model_comparison_mfrm_bic_summary.loc["items", "LL"], _esf.cond_loglik(Xv, thr)[0])


def test_mfrm_selection_bootstrap_route():
    m = rp.MFRM(_resp())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(model="global", estimation="pair-cpat")
        with pytest.raises(ValueError, match="needs a likelihood"):
            m.model_selection(test="AIC", candidate_models=["global", "items"])
        m.model_selection(candidate_models=["global", "items"], no_of_samples=19, seed=1, progress=False)
    assert m.model_comparison_mfrm_info["route"] == "bootstrap"
    row = m.model_comparison_mfrm_lr_summary.loc["global vs items"]
    assert np.isclose(row["Statistic"], 12.178282852972398, rtol=1e-6)
    assert row["Resamples"] == 19 and np.isclose(row["p-value"], 0.05)
    assert row["Preferred"] == "items"


def test_mfrm_selection_cpat_rcl_composite_route():
    """'cpat-rcl' candidates are compared on the unsmoothed CPAT composite log-likelihood at each fit, and a 'mixed'
    candidate gets the same value as per_rater_model_selection()'s criteria for that fit."""
    from raschpy import _rcl, _selection

    m = rp.MFRM(_resp())
    rm = ["global", "items", "global", "items"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(model="mixed", rater_models=rm, anchors=["Rater_1"], estimation="cpat-rcl")
        m.calibrate(model="global", estimation="cpat-rcl")
        m.model_selection(candidate_models=["global", "items", "mixed"])
    assert m.model_comparison_mfrm_info == {"route": "composite", "estimation": "cpat-rcl"}
    t = m.model_comparison_mfrm_aic_summary
    R, I, K = m.no_of_facet_elements, m.no_of_items, m.max_score
    X = m.responses.to_numpy(dtype=np.float64).reshape(R, -1, I)
    D, tt, u = _rcl.comparisons(*_rcl.pattern_counts(X, K), R, I, K)
    b = np.concatenate(m._virtual_items("items")[1])
    assert np.isclose(t.loc["items", "CL"], _selection._ll(D, tt, u, b))
    ctx = m._mixed_context("cpat-rcl", None, {})
    design, fit, _ = m._mixed_fit_core(ctx, rm, [0])
    assert np.isclose(t.loc["mixed", "CL"], m._mixed_criteria(ctx, design, fit)["ll"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.model_selection(test="LR", candidate_models=["global", "items"])
    assert 0 <= m.model_comparison_mfrm_lr_summary.loc["global vs items", "p-value"] <= 1


def test_mfrm_selection_estimation_checks_and_warnings():
    resp = _resp()
    m = rp.MFRM(resp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(model="global", estimation="cml")
        m.calibrate(model="items", estimation="cp-full-rcl")
    with pytest.raises(ValueError, match="same estimation"):
        m.model_selection(candidate_models=["global", "items"])
    m2 = rp.MFRM(resp)
    with pytest.warns(FutureWarning, match="likelihood, sampling"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            m2.model_selection(candidate_models=["global", "items"], likelihood="conditional", sampling="dynamic")
    mw = rp.MFRM(resp.replace(2, 1), max_score=3, validate=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mw.model_selection(candidate_models=["global", "thresholds"])
    msg = [str(x.message) for x in w if "model_selection()" in str(x.message)]
    assert len(msg) == 1 and "score category 2 is not used" in msg[0] and "conservative" in msg[0]


def test_weighted_chi2_tail_matches_closed_forms():
    from scipy.stats import chi2
    from raschpy import _selection

    # equal weights: lam * chi-square(d), including the far tail and d = 1
    for lam, d, x in ((3.0, 5, 40.0), (6.5, 12, 200.0), (1.0, 1, 3.84), (1.0, 1, 20.0), (4.0, 2, 30.0), (5.0, 3, 2000.0)):
        assert np.isclose(_selection.weighted_chi2_sf(x, [lam] * d), chi2.sf(x / lam, d), rtol=1e-9, atol=0)
    # weights a, a, b, b: a sum of two exponentials with means 2a, 2b
    a, b = 2.0, 7.0
    for x in (10.0, 60.0, 150.0, 900.0):
        exact = (b * np.exp(-x / (2 * b)) - a * np.exp(-x / (2 * a))) / (b - a)
        assert np.isclose(_selection.weighted_chi2_sf(x, [a, a, b, b]), exact, rtol=1e-9, atol=0)


def test_weighted_chi2_tail_wide_weights_and_methods():
    from raschpy import _selection

    rng = np.random.default_rng(3)
    # very unequal weights (as in Andersen's test), including a near-zero one: against simulation
    w = np.concatenate([[2.5e-7], np.geomspace(0.045, 23, 77)])
    sims = (rng.chisquare(1, (400_000, len(w))) * w).sum(1)
    for q in (0.5, 0.95, 0.995):
        x = np.quantile(sims, q)
        assert abs(_selection.weighted_chi2_sf(x, w) - (1 - q)) < 4 * np.sqrt(q * (1 - q) / len(sims))
    # Ruben's series and Imhof's inversion agree away from the far tail (many weights)
    w = np.geomspace(2, 20, 30)
    x = w.sum() * 1.5
    assert np.isclose(_selection._imhof_sf(x, w)[0], _selection._ruben_sf(x, w), rtol=1e-7)
    # two weights: the integral against the chi-square case and simulation
    assert np.isclose(_selection.weighted_chi2_sf(9.0, [3.0, 3.0 * (1 + 1e-13)]), np.exp(-9.0 / 6.0), rtol=1e-9)
    s2 = (rng.chisquare(1, (400_000, 2)) * [1.0, 40.0]).sum(1)
    x = np.quantile(s2, 0.99)
    assert abs(_selection.weighted_chi2_sf(x, [1.0, 40.0]) - 0.01) < 4 * np.sqrt(0.0099 / len(s2))


def test_mfrm_models_is_a_deprecated_name_for_candidate_models():
    m = rp.MFRM(_resp())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(model="global")
        m.model_selection(candidate_models=["global", "items"])
    a = m.model_comparison_mfrm_aic_summary
    with pytest.warns(FutureWarning, match="candidate_models"):
        m.model_selection(models=["global", "items"])
    pd.testing.assert_frame_equal(a, m.model_comparison_mfrm_aic_summary)
    with pytest.raises(ValueError, match="candidate_models= only"):
        m.model_selection(models=["global"], candidate_models=["global", "items"])


def test_mfrm_bootstrap_statistic_compares_the_fitted_models():
    """The bootstrap test's statistic is the sum of squared differences between the larger and smaller models'
    fitted location grids, each relative to its own mean."""
    m = rp.MFRM(_resp())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(model="global", estimation="pair-cpat")
        m.model_selection(candidate_models=["global", "items"], no_of_samples=9, seed=1, progress=False)
    g = {mod: np.concatenate(m._virtual_items(mod)[1]) for mod in ("global", "items")}
    d = (g["items"] - g["items"].mean()) - (g["global"] - g["global"].mean())
    row = m.model_comparison_mfrm_lr_summary.loc["global vs items"]
    assert np.isclose(row["Statistic"], float(d @ d), rtol=1e-10) and row["Resamples"] == 9
