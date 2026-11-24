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
        m.model_selection(test="LR", candidate_models=MODELS, se_correction="kauermann-carroll")
    info = m.model_comparison_mfrm_info
    assert {k: info[k] for k in ("route", "estimation")} == {"route": "composite", "estimation": "cp-full-rcl"}
    # the simulated thresholds are disordered, so the default width-based tolerance falls back to 0
    assert info["delta"] == 0.0 and info["stretch_delta"] == 0.2
    t = m.model_comparison_mfrm_lr_summary
    assert list(t.columns) == ["LR", "df", "Scale", "df (Satterthwaite)", "p-value", "Departure",
                               "p-value (close fit)", "Effect", "Preferred"]
    expected = {
        "global vs items": (1582.3719824703803, 12, 3.812659175910171, 4.355602490674782e-68),
        "items vs bivector": (14.471812913034228, 6, 5.421885126140627, 0.8071002309233724),
        "global vs centrality": (101.36613548859168, 3, 3.9311025819874783, 1.2085076827505378e-05),
        "centrality vs bistretch": (582.1688447489505, 3, 3.0953232066364715, 8.661641216116338e-39),
    }
    for lab, (lr, df, scale, p) in expected.items():
        assert np.isclose(t.loc[lab, "LR"], lr, rtol=1e-6)
        assert t.loc[lab, "df"] == df
        assert np.isclose(t.loc[lab, "Scale"], scale, rtol=1e-6)
        assert np.isclose(t.loc[lab, "p-value"], p, rtol=1e-4)
    assert t.loc["items vs bivector", "Preferred"] == "items"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.model_selection(candidate_models=MODELS, se_correction="kauermann-carroll")
    pen = m.model_comparison_mfrm_aic_summary["Penalty"]
    assert np.isclose(pen["items"], 66.13772061414947, rtol=1e-6)
    assert np.isclose(pen["global"], 21.812508006576586, rtol=1e-6)
    assert m.model_comparison_mfrm_aic_preferred == "items"
    step = m.model_comparison_mfrm_aic_steps.loc["global > items"]
    assert step["df"] == 12 and np.isclose(step["tail p"], 4.355602490674782e-68, rtol=1e-4)
    assert step["Kept"] and step["p-value (close fit)"] <= 0.05


def test_mfrm_selection_equal_size_gate():
    """Each AIC step's relative likelihood is the one a likelihood ratio with the composite test's tail probability
    would give; on data with global effects only every step is refused."""
    from scipy.stats import chi2

    resp = MFRM_Sim(no_of_items=5, no_of_persons=300, no_of_facet_elements=4, max_score=3, model="global",
                    seed=12).responses
    m = rp.MFRM(resp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.model_selection()
    steps = m.model_comparison_mfrm_aic_steps
    for _, g in steps.iterrows():
        lr = chi2.isf(g["tail p"], g["df"])
        assert np.isclose(g["Relative likelihood"], np.exp(-max(lr - 2 * g["df"], 0) / 2))
    assert not steps["Kept"].any()
    assert m.model_comparison_mfrm_aic_preferred == "global"


def test_mfrm_selection_tolerance_one_deviant_rater():
    """One facet element of six with item effects moves the shared choice with no tolerance but not with the
    default; delta=0 gives the plain steps, and tolerances need a composite fit."""
    rng = np.random.default_rng(4)
    resp = MFRM_Sim(no_of_items=6, no_of_persons=800, no_of_facet_elements=6, max_score=3, model="global",
                    seed=21).responses
    # one rater's item effects: re-simulate that rater's responses with shifted item locations
    m0 = rp.MFRM(resp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m0.calibrate(model="global")
    X = resp.copy()
    r1 = X.index.get_level_values(0) == X.index.get_level_values(0)[0]
    shift = np.array([0.45, -0.45, 0.3, -0.3, 0.15, -0.15])
    pers = rng.normal(0, 1.5, r1.sum())
    tau = np.array([-1.2, 0.0, 1.2])
    for j, c in enumerate(X.columns):
        loc = np.linspace(-1, 1, 6)[j] + shift[j]
        le = np.concatenate([np.zeros((len(pers), 1)),
                             np.cumsum(pers[:, None] - loc - tau[None, :], axis=1)], axis=1)
        pr = np.exp(le - le.max(1, keepdims=True))
        pr /= pr.sum(1, keepdims=True)
        X.loc[r1, c] = np.where(X.loc[r1, c].isna(), np.nan,
                                (rng.random((len(pers), 1)) > np.cumsum(pr, 1)).sum(1).astype(float))
    m = rp.MFRM(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.model_selection(candidate_models=["global", "items"], delta=0, stretch_delta=0)
        plain = m.model_comparison_mfrm_aic_preferred
        m.model_selection(candidate_models=["global", "items"])
    assert plain == "items"
    assert m.model_comparison_mfrm_aic_preferred == "global"
    assert m.model_comparison_mfrm_info["delta"] == 0.2
    assert "p-value (close fit)" in m.model_comparison_mfrm_aic_steps.columns
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.model_selection(candidate_models=["global", "items"], estimation="cml")
    assert m.model_comparison_mfrm_info["delta"] == 0.2
    assert "p-value (close fit)" in m.model_comparison_mfrm_aic_steps.columns
    assert m.model_comparison_mfrm_aic_preferred == "global"


def test_mfrm_selection_cml_close_fit_plain_matches_lr():
    """'cml': with tolerances of 0 the close-fit p is the conditional LR test's chi-square p, at every nested pair."""
    m = rp.MFRM(_resp())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.model_selection(test="LR", candidate_models=MODELS + ["thresholds", "pseudo_halo"], estimation="cml",
                          delta=1e-9, delta_units="logits", stretch_delta=1e-9)
    t = m.model_comparison_mfrm_lr_summary
    assert m.model_comparison_mfrm_info["route"] == "cml"
    assert np.allclose(t["p-value (close fit)"].astype(float), t["p-value"].astype(float), rtol=1e-4, atol=1e-12)


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
        with pytest.raises(ValueError, match="tolerances need"):
            m.model_selection(test="LR", candidate_models=["global", "items"], delta=0.3)
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
    assert {k: m.model_comparison_mfrm_info[k] for k in ("route", "estimation")} == {
        "route": "composite", "estimation": "cpat-rcl"}
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
    # candidates calibrated with different estimations: the latest calibration's is used, with a warning
    with pytest.warns(UserWarning, match="'global' was calibrated with another estimation"):
        m.model_selection(candidate_models=["global", "items"])
    assert m.estimation_global == m.estimation_items == "cp-full-rcl"
    # an explicit estimation recalibrates without the warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.model_selection(candidate_models=["global", "items"], estimation="cpat-rcl")
    assert not [x for x in w if "another estimation" in str(x.message)]
    assert m.estimation_global == m.estimation_items == "cpat-rcl"
    with pytest.raises(ValueError, match="estimation must be"):
        m.model_selection(candidate_models=["global", "items"], estimation="pair")
    m2 = rp.MFRM(resp)
    with pytest.warns(FutureWarning, match="likelihood no longer applies"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            m2.model_selection(candidate_models=["global", "items"], likelihood="conditional")
    mw = rp.MFRM(resp.replace(2, 1), max_score=3, validate=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mw.model_selection(candidate_models=["global", "thresholds"])
    msg = [str(x.message) for x in w if "model_selection()" in str(x.message) and "not used" in str(x.message)]
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


def test_mfrm_selection_ridges_refit_the_candidates():
    """Ridge arguments recalibrate every candidate with them: a stronger interaction ridge changes 'matrix' (its
    effects and AIC) and leaves 'global' and 'bivector' as they were; pair-cpat has no ridges."""
    m = rp.MFRM(_resp(), max_score=3)
    cands = ["global", "bivector", "matrix"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.model_selection(test="AIC", candidate_models=cands, estimation="cp-full-rcl")
        a, e = dict(m.model_comparison_mfrm_aic), m.facet_effects_matrix.copy()
        m.model_selection(test="AIC", candidate_models=cands, estimation="cp-full-rcl", interaction_ridge=50.0)
        b = dict(m.model_comparison_mfrm_aic)
    assert b["global"] == pytest.approx(a["global"]) and b["bivector"] == pytest.approx(a["bivector"])
    assert b["matrix"] != pytest.approx(a["matrix"], abs=1e-6)
    assert not np.allclose(m.facet_effects_matrix.to_numpy(float), e.to_numpy(float))
    with pytest.raises(ValueError, match="ridge"):
        m.model_selection(test="LR", candidate_models=cands, estimation="pair-cpat", interaction_ridge=1.0)


def test_excluded_models_is_the_complement_of_candidate_models():
    """excluded_models= leaves the named models out of the eight; with candidate_models= too it must give the
    same set; aliases are resolved; unknown names and an empty result are errors."""
    m = rp.MFRM(_resp())
    allowed = m._MODELS + ("mixed",)
    rest = [x for x in m._MODELS if x not in ("matrix", "bistretch")]
    assert m._candidate_models(None, None, "model_selection", ["matrix", "bistretch"], allowed) == rest
    assert m._candidate_models(None, None, "model_selection", "matrix", allowed) == \
        [x for x in m._MODELS if x != "matrix"]
    assert "centrality" not in m._candidate_models(None, None, "model_selection", ["threshold_stretch"], allowed)
    assert m._candidate_models(rest[::-1], None, "model_selection", ["matrix", "bistretch"], allowed) == rest[::-1]
    with pytest.raises(ValueError, match="different sets of models"):
        m._candidate_models(["global", "items"], None, "model_selection", ["matrix"], allowed)
    with pytest.raises(ValueError, match="unrecognised"):
        m._candidate_models(None, None, "model_selection", ["matrx"], allowed)
    with pytest.raises(ValueError, match="leaves no models"):
        m._candidate_models(None, None, "per_rater_model_selection", list(m._MODELS))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.model_selection(test="AIC", excluded_models=["matrix", "bivector", "thresholds", "bistretch"])
    assert set(m.model_comparison_mfrm_aic) == {"global", "items", "centrality", "pseudo_halo"}
    with pytest.raises(ValueError, match="different sets of models"):
        m.per_rater_model_selection(candidate_models=["global", "items"], excluded_models=["matrix"])


@pytest.mark.parametrize("bad, match", [
    (dict(sampling="power"), "sampling must be"),
    (dict(sampling=1), "at least 2"),
    (dict(sampling=True), "at least 2"),
    (dict(sampling=100, delta=0.2), "cannot be combined"),
    (dict(sampling=100, stretch_delta=0.1), "cannot be combined"),
    (dict(n_subsamples=0), "n_subsamples"),
])
def test_mfrm_selection_sampling_argument_errors(bad, match):
    with pytest.raises(ValueError, match=match):
        rp.MFRM(_resp()).model_selection(candidate_models=["global", "items"], **bad)


@pytest.mark.parametrize("test", ["AIC", "BIC", "LR"])
def test_mfrm_selection_sampling_refits_subsamples(test):
    """sampling=n: every candidate refitted on n random persons, n_subsamples times; the decision is the subsamples'
    (AIC, BIC: the most common choice; LR: a pair's larger model if more than half reject the smaller), and the
    seed repeats the draws."""
    cands = ["global", "items", "bivector"]
    m = rp.MFRM(_resp())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.model_selection(test=test, candidate_models=cands, sampling=150, n_subsamples=3, seed=4)
    r = m.model_comparison_mfrm_sampling
    assert r["decision"] == "subsamples" and r["n"] == 150 and r["failed"] == 0 and r["seed"] == 4
    if test == "LR":
        t = m.model_comparison_mfrm_lr_summary
        assert "Share rejecting" in t.columns and list(t.columns)[-1] == "Preferred"
        for key, ps in r["p"].items():
            assert len(ps) == 3
            small, large = key.split(" vs ")
            share = np.mean(np.asarray(ps) <= 0.05)
            assert t.loc[key, "Preferred"] == (large if share > 0.5 else small)
        first = r["p"]
    else:
        lab = test.lower()
        assert len(r["choices"]) == 3
        counts = pd.Series(r["choices"]).value_counts()
        assert getattr(m, f"model_comparison_mfrm_{lab}_preferred") in counts[counts == counts.max()].index
        summ = getattr(m, f"model_comparison_mfrm_{lab}_summary")
        assert np.isclose(summ["Subsample share"].sum(), 1.0)
        first = r["choices"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.model_selection(test=test, candidate_models=cands, sampling=150, n_subsamples=3, seed=4)
    again = m.model_comparison_mfrm_sampling
    assert (again["p"] if test == "LR" else again["choices"]) == first


def test_mfrm_selection_sampling_dynamic_and_full_data():
    """'dynamic' sets n = min(20 R (I - 1) (K - 1), 1500); n at least the number of persons uses all the data; no
    sampling leaves the attribute None."""
    m = rp.MFRM(_resp())  # R 4, I 5, K 3: n = 640 > 300 persons
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.model_selection(candidate_models=["global", "items"], sampling="dynamic")
    r = m.model_comparison_mfrm_sampling
    assert r["n"] == 640 and r["dynamic"] and r["decision"].startswith("full data")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.model_selection(candidate_models=["global", "items"])
    assert m.model_comparison_mfrm_sampling is None


def test_mfrm_selection_sampling_auto_is_none():
    """sampling='auto' (v1.4.0's default) is accepted and means None."""
    m = rp.MFRM(_resp())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.model_selection(test="AIC", candidate_models=["global", "items"], sampling="auto")
    assert m.model_comparison_mfrm_sampling is None
