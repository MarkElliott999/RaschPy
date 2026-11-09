"""
calibrate(model='mixed', rater_models=..., anchors=...): one joint fit, each facet element (e.g. rater) on its own
representation, in the frame of the anchors (the anchors' average deviation from the instrument is zero).
"""

import warnings

import numpy as np
import pytest

import raschpy as rp
from raschpy.simulation import MFRM_Sim

warnings.filterwarnings("ignore")

I, R, K = 5, 4, 3


def _grid(m, model):
    if model == "mixed":
        return (m.items_mixed.to_numpy()[None, :, None] + m.thresholds_mixed.to_numpy()[None, None, :]
                + m.facet_effects_mixed.to_numpy().reshape(R, I, K))
    return np.concatenate(m._virtual_items(model)[1]).reshape(R, I, K)


@pytest.mark.parametrize("estimation", ["cp-full-rcl", "cpat-rcl", "cml"])
@pytest.mark.parametrize("model", ["global", "items", "matrix", "centrality", "bistretch"])
def test_every_rater_on_one_representation_reproduces_its_fit(estimation, model):
    s = MFRM_Sim(I, 300, R, K, model="mixed", rater_models=[model] * R, seed=11)
    single = rp.MFRM(s.responses, max_score=K, validate=False)
    single.calibrate(model=model, estimation=estimation)
    mixed = rp.MFRM(s.responses, max_score=K, validate=False)
    mixed.calibrate(model="mixed", rater_models=[model] * R, estimation=estimation)
    d = _grid(single, model) - _grid(mixed, "mixed")
    assert np.abs(d - d.mean()).max() < 1e-6


@pytest.mark.parametrize("estimation", ["cp-full-rcl", "cml"])
def test_the_frame_reads_the_labels(estimation):
    """B differs from C and D on the items. With C and D as anchors, B is an items rater; with no anchors (the
    average rater) the labels items/global/global force B's item effects to 0, and the fit is worse."""
    s = MFRM_Sim(I, 1500, 3, K, model="mixed", rater_models=["items", "global", "global"],
                 anchors=["Rater_2", "Rater_3"], neutral_anchors=True, seed=1)
    labels = ["items", "global", "global"]
    a = rp.MFRM(s.responses, max_score=K, validate=False)
    a.calibrate(model="mixed", rater_models=labels, anchors=["Rater_2", "Rater_3"], estimation=estimation)
    n = rp.MFRM(s.responses, max_score=K, validate=False)
    n.calibrate(model="mixed", rater_models=labels, estimation=estimation)
    true = s.rater_params["Rater_1"][1]
    est = a.rater_params_mixed["Rater_1"][1]
    assert np.abs((est - est.mean()) - (true - true.mean())).max() < 0.2
    b = n.rater_params_mixed["Rater_1"][1]
    assert np.abs(b - b.mean()).max() < 1e-8
    assert n.convergence_mixed["log_likelihood"] < a.convergence_mixed["log_likelihood"] - 50
    assert a.anchors_mixed == ["Rater_2", "Rater_3"] and n.anchors_mixed is None


def test_mixed_recovers_native_parameters_in_the_generating_frame():
    eight = ["global", "items", "thresholds", "bivector", "matrix", "centrality", "pseudo_halo", "bistretch"]
    s = MFRM_Sim(6, 3000, 8, 3, model="mixed", rater_models=eight, anchors=["Rater_1"], neutral_anchors=True, seed=5)
    m = rp.MFRM(s.responses, max_score=3, validate=False)
    m.calibrate(model="mixed", rater_models=eight, anchors=["Rater_1"], estimation="cml")
    assert m.rater_models.tolist() == eight
    for fe, (mod, tp) in s.rater_params.items():
        _, ep = m.rater_params_mixed[fe]
        if isinstance(tp, dict):
            err = max(np.abs(np.asarray(ep[k], float) - np.asarray(tp[k], float)).max() for k in tp)
        else:
            err = np.abs(np.asarray(ep, float) - np.asarray(tp, float)).max()
        assert err < (0.4 if mod == "matrix" else 0.12), (fe, mod, err)
    assert np.abs(m.items_mixed.to_numpy() - s.native_items.to_numpy()).max() < 0.06


def test_mixed_pair_cpat_is_derived_from_matrix():
    s = MFRM_Sim(I, 200, R, K, model="mixed", rater_models=["global", "items", "thresholds", "matrix"], seed=2)
    m = rp.MFRM(s.responses, max_score=K, validate=False)
    m.calibrate(model="mixed", rater_models=["global", "items", "thresholds", "matrix"], estimation="pair-cpat")
    assert m.estimation_mixed == "pair-cpat" and m.facet_effects_mixed.shape == (R * I, K)


def test_mixed_argument_errors():
    s = MFRM_Sim(I, 100, R, K, model="global", seed=3)
    m = rp.MFRM(s.responses, max_score=K, validate=False)
    with pytest.raises(ValueError, match="without rater_models"):
        m.calibrate(model="mixed", estimation="pair-cpat")
    with pytest.raises(ValueError, match="3 representations but there are 4"):
        m.calibrate(model="mixed", rater_models=["global"] * 3)
    with pytest.raises(ValueError, match="unknown representation"):
        m.calibrate(model="mixed", rater_models=["global", "items", "nope", "global"])
    with pytest.raises(ValueError, match="supports estimation"):
        m.calibrate(model="mixed", rater_models=["global"] * 4, estimation="jml")
    with pytest.raises(ValueError, match="model='mixed' only"):
        m.calibrate(model="items", anchors=["Rater_1"])


@pytest.mark.parametrize("estimation", ["cp-full-rcl", "cpat-rcl", "cml"])
@pytest.mark.parametrize("model", ["items", "centrality"])
def test_mixed_sandwich_se_matches_the_single_fit(estimation, model):
    s = MFRM_Sim(I, 300, R, K, model="mixed", rater_models=[model] * R, seed=12)
    single = rp.MFRM(s.responses, max_score=K, validate=False)
    single.calibrate(model=model, estimation=estimation)
    single.std_errors(model=model)
    mixed = rp.MFRM(s.responses, max_score=K, validate=False)
    mixed.calibrate(model="mixed", rater_models=[model] * R, estimation=estimation)
    mixed.std_errors(model="mixed", interval=0.95)
    np.testing.assert_allclose(mixed.item_se_mixed.to_numpy(), getattr(single, f"item_se_{model}").to_numpy(),
                               rtol=1e-4)
    assert np.all(np.isfinite(mixed.rater_se_mixed.to_numpy()))
    assert (mixed.item_low_mixed < mixed.items_mixed).all()


@pytest.mark.parametrize("scheme", ["iterate", "one-pass"])
def test_per_rater_selection_on_fits_in_the_anchors_frame(scheme):
    """Generated with C, D neutral: anchored on them the selection finds B items and C, D global; unanchored
    (the average rater) B's item effects show in C and D too, so global is not chosen for all three."""
    s = MFRM_Sim(I, 1500, 3, K, model="mixed", rater_models=["items", "global", "global"],
                 anchors=["Rater_2", "Rater_3"], neutral_anchors=True, seed=1)
    m = rp.MFRM(s.responses, max_score=K, validate=False)
    tab = m.per_rater_model_selection(anchors=["Rater_2", "Rater_3"], estimation="cp-full-rcl", scheme=scheme,
                                      candidate_models=["global", "items", "thresholds", "bivector"])
    assert tab["Representation"].tolist() == ["items", "global", "global"]
    assert m.rater_models.tolist() == ["items", "global", "global"] and m.anchors_mixed == ["Rater_2", "Rater_3"]
    assert m.per_rater_model_selection_info["frame"] == "anchors ['Rater_2', 'Rater_3']"
    n = rp.MFRM(s.responses, max_score=K, validate=False)
    n.calibrate(model="mixed", estimation="cp-full-rcl",
                selection=dict(scheme=scheme, candidate_models=["global", "items", "thresholds", "bivector"]))
    assert n.rater_models.tolist() == ["items", "items", "items"]
    assert n.per_rater_model_selection_info["frame"] == "average facet element (no anchors)"


def test_mixed_downstream_matches_the_single_fit():
    """With every rater on one representation, persons, fit statistics, the composite Andersen test, DIF and
    bootstrap SEs on 'mixed' equal those of the single fit (the same model), and 'mixed' runs in the plots and as
    a model_selection candidate."""
    import matplotlib
    import pandas as pd

    matplotlib.use("Agg")
    s = MFRM_Sim(I, 400, R, K, model="mixed", rater_models=["items"] * R, seed=8)
    grp = pd.DataFrame({"grp": pd.Series({p: ("a" if i % 2 else "b") for i, p in enumerate(s.persons.index)})})
    a = rp.MFRM(s.responses, max_score=K, validate=False, exogenous=grp)
    a.calibrate(model="items")
    b = rp.MFRM(s.responses, max_score=K, validate=False, exogenous=grp)
    b.calibrate(model="mixed", rater_models=["items"] * R)
    a.person_estimates(model="items"); b.person_estimates(model="mixed")
    assert np.abs(a.persons_items - b.persons_mixed).max() < 1e-5
    a.fit_statistics(model="items", conditional=True); b.fit_statistics(model="mixed", conditional=True)
    assert np.abs(np.asarray(a.item_infit_ms_items, float) - np.asarray(b.item_infit_ms_mixed, float)).max() < 1e-5
    ga = a.global_fit(model="items", seed=1, p_method="chi2"); gb = b.global_fit(model="mixed", seed=1, p_method="chi2")
    assert abs(ga["statistic"] - gb["statistic"]) < 1e-5
    a.dif_test("grp", model="items", no_of_samples=50, seed=1)
    b.dif_test("grp", model="mixed", no_of_samples=50, seed=1)
    da, db = a.dif_table, b.dif_table
    num = da.select_dtypes("number").columns
    assert np.nanmax(np.abs(da[num].to_numpy(float) - db[num].to_numpy(float))) < 1e-5
    b.std_errors(model="mixed")
    sand = b.item_se_mixed.copy()
    b.std_errors(model="mixed", se_method="bootstrap", no_of_samples=100, seed=1)
    assert 0.7 < float(np.median(b.item_se_mixed / sand)) < 1.3
    b.wright_map(model="mixed")
    b.model_selection(candidate_models=["global", "items", "mixed"], test="AIC")
    assert "mixed" in b.model_comparison_mfrm_aic_summary.index
    # DRF: global raters have a global effect test only
    b.calibrate(model="mixed", rater_models=["global"] * R)
    b.drf_test("grp", model="mixed")
    assert b.drf_table["p"].notna().all() and b.drf_item_table is None and b.drf_threshold_table is None
    with pytest.raises(ValueError, match="cannot be re-anchored"):
        b.calibrate_anchor("mixed", ["Rater_1"])


def test_mixed_recovery_stats_compares_each_representation():
    import matplotlib

    matplotlib.use("Agg")
    models = ["global", "items", "thresholds", "matrix", "centrality", "bistretch"]
    s = MFRM_Sim(5, 1000, 6, 3, model="mixed", rater_models=models, anchors=["Rater_1"], neutral_anchors=True, seed=3)
    m = rp.MFRM(s, max_score=3, validate=False)
    m.calibrate(model="mixed", rater_models=models, anchors=["Rater_1"])
    m.recovery_stats(model="mixed", plot=False)
    parts = m.recovery.raters.subvectors
    assert {"items", "thresholds", "matrix", "centrality omega", "bistretch omega_items"} <= set(parts)
    assert parts["items"].summary.loc["r", "Overall"] > 0.9


def _two_groups(params, kw, n, seeds):
    import pandas as pd

    g1 = MFRM_Sim(5, n, len(params), 3, model="mixed", rater_models=params, seed=seeds[0], **kw)
    g2 = MFRM_Sim(5, n, len(params), 3, model="mixed", rater_models=params, seed=seeds[1], **kw)
    r2 = g2.responses.copy()
    r2.index = r2.index.set_levels([f"F{p}" for p in r2.index.levels[1]], level=1)
    resp = pd.concat([g1.responses, r2])
    P = resp.index.get_level_values(1).unique()
    exo = pd.DataFrame({"grp": pd.Series(["focal" if str(p).startswith("F") else "ref" for p in P], index=P)})
    return rp.MFRM(resp, max_score=3, validate=False, exogenous=exo)


def test_mixed_drf_leaves_out_thin_cells_and_reports_widths():
    """A matrix rater that almost never gives 0 on one item: its thin cells are left out with one warning, the
    other raters are untouched, and the threshold table reports the bounded categories' width changes."""
    models = ["global", "items", "thresholds", "bivector", "matrix"]
    base = MFRM_Sim(5, 10, 5, 3, model="mixed", rater_models=models, anchors=["Rater_1"], neutral_anchors=True,
                    seed=5)
    params = dict(base.rater_params)
    M = np.asarray(params["Rater_5"][1], float).copy()
    M[1] -= 3.0  # Item_2: category 0 nearly unused
    params["Rater_5"] = ("matrix", M)
    kw = dict(manual_items=base.native_items.to_numpy(), manual_thresholds=base.native_thresholds.to_numpy())
    m = _two_groups(params, kw, 300, (1, 2))
    m.calibrate(model="mixed", rater_models=models, anchors=["Rater_1"])
    with pytest.warns(UserWarning, match="min_category_count=5"):
        m.drf_test("grp", model="mixed", reference="ref")
    t = m.drf_threshold_table
    assert t.loc["Rater_5", "Cells left out"] >= 1 and "Item_2 category 0" in t.loc["Rater_5", "Note"]
    assert (t.drop("Rater_5")["Cells left out"] == 0).all()
    assert {"Difference category 1", "Difference category 2"} <= set(t.columns)
    assert "Difference category 3" not in t.columns
    assert np.isfinite(m.drf_item_table.loc["Rater_5", "p"]) and "Difference Item_2" in m.drf_item_table
    inter, overall = m.drf_interaction_table, m.drf_overall_table
    assert inter["p"].notna().tolist() == [False, False, False, False, True]
    assert inter.loc["Rater_5", "df"] == 8 - inter.loc["Rater_5", "Cells left out"]
    assert overall.loc["Rater_1", "Note"].startswith("single component") and overall.loc["Rater_4", "df"] == 7
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        m.drf_test("grp", model="mixed", reference="ref", min_category_count=None)
    assert (m.drf_threshold_table["Cells left out"] == 0).all()


def test_mixed_drf_tests_global_effect_when_a_category_is_sparse_in_one_group():
    """Raters that gave 2 ratings of 3 in the focal group: their global effects (and a bivector rater's item
    effects) are tested on the remaining ratings, their threshold tests are not run, with notes and one warning."""
    models = ["global", "items", "thresholds", "thresholds", "bivector", "bivector"]
    base = MFRM_Sim(5, 10, 6, 3, model="mixed", rater_models=models, anchors=["Rater_1"], neutral_anchors=True,
                    seed=5)
    kw = dict(manual_items=base.native_items.to_numpy(), manual_thresholds=base.native_thresholds.to_numpy())
    m = _two_groups(dict(base.rater_params), kw, 300, (1, 2))
    resp = m.responses.copy()
    for r in ("Rater_4", "Rater_6"):
        rows = (resp.index.get_level_values(0) == r) & resp.index.get_level_values(1).str.startswith("F")
        x = resp.loc[rows]
        keep = np.argwhere((x == 3).to_numpy())[:2]
        y = x.mask(x == 3, 2).to_numpy().copy()
        y[tuple(keep.T)] = 3
        resp.loc[rows] = y
    m = rp.MFRM(resp, max_score=3, validate=False, exogenous=m.exogenous)
    m.calibrate(model="mixed", rater_models=models, anchors=["Rater_1"])
    with pytest.warns(UserWarning, match=r"Rater_4 \(score 3: 2 ratings in group focal\) and Rater_6 .* gave fewer "
                                         r"than min_category_count = 5 ratings in a score category in one group"):
        m.drf_test("grp", model="mixed", reference="ref", p_method="chi2")
    reason = "gave 2 ratings of 3 in group focal (fewer than min_category_count = 5)"
    sev, thr, it, ov = m.drf_table, m.drf_threshold_table, m.drf_item_table, m.drf_overall_table
    assert sev["p"].notna().all()
    assert sev.loc["Rater_4", "Note"] == f"Global effect tested without Threshold 3: Rater_4 {reason}."
    assert thr.loc[["Rater_4", "Rater_6"], "p"].isna().all() and thr.loc[["Rater_3", "Rater_5"], "p"].notna().all()
    assert thr.loc["Rater_4", "Note"] == (f"Not tested: Rater_4 {reason}, so Rater_4's Threshold 3 cannot be "
                                          f"compared between the groups.")
    assert np.isfinite(it.loc["Rater_6", "p"]) and it.loc["Rater_6", "df"] == 4
    assert ov.loc["Rater_6", "df"] == 5 and ov.loc["Rater_6", "Note"] == f"Thresholds left out: Rater_6 {reason}."
    assert ov.loc["Rater_4", "Note"].startswith("Thresholds left out: Rater_4") and np.isnan(ov.loc["Rater_4", "p"])
    assert 0 < m.drf_omnibus_table.loc["focal", "p"] <= 1


@pytest.mark.parametrize("model", ["items", "matrix", "bistretch"])
def test_drf_on_a_single_representation_matches_mixed(model):
    import pandas as pd

    s1 = MFRM_Sim(I, 300, R, K, model="mixed", rater_models=[model] * R, seed=3)
    s2 = MFRM_Sim(I, 300, R, K, model="mixed", rater_models=[model] * R, seed=4)
    r2 = s2.responses.copy()
    r2.index = r2.index.set_levels([f"F{p}" for p in r2.index.levels[1]], level=1)
    resp = pd.concat([s1.responses, r2])
    P = resp.index.get_level_values(1).unique()
    exo = pd.DataFrame({"grp": pd.Series(["focal" if str(p).startswith("F") else "ref" for p in P], index=P)})
    a = rp.MFRM(resp, max_score=K, validate=False, exogenous=exo)
    a.calibrate(model=model)
    a.drf_test("grp", model=model, reference="ref", seed=1)
    b = rp.MFRM(resp, max_score=K, validate=False, exogenous=exo)
    b.calibrate(model="mixed", rater_models=[model] * R)
    b.drf_test("grp", model="mixed", reference="ref", seed=1)
    for t in ("drf_table", "drf_item_table", "drf_threshold_table", "drf_interaction_table", "drf_overall_table"):
        if getattr(a, t) is None:
            assert getattr(b, t) is None
            continue
        pa, pb = getattr(a, t)["p"].astype(float), getattr(b, t)["p"].astype(float)
        assert (pa.isna() == pb.isna()).all() and np.nanmax(np.abs(pa - pb).fillna(0)) < 1e-8
    a.calibrate(model=model, estimation="pair-cpat")
    with pytest.raises(ValueError, match="estimation=.cp-full-rcl."):
        a.drf_test("grp", model=model, reference="ref", seed=1)


def test_drf_p_method():
    """p_method='chi2' gives the chi-squared p-values with the pool held fixed; 'monte-carlo' needs draws."""
    import pandas as pd
    from scipy.stats import chi2

    s1 = MFRM_Sim(I, 200, R, K, model="mixed", rater_models=["items"] * R, seed=3)
    s2 = MFRM_Sim(I, 200, R, K, model="mixed", rater_models=["items"] * R, seed=4)
    r2 = s2.responses.copy()
    r2.index = r2.index.set_levels([f"F{p}" for p in r2.index.levels[1]], level=1)
    resp = pd.concat([s1.responses, r2])
    P = resp.index.get_level_values(1).unique()
    exo = pd.DataFrame({"grp": pd.Series(["focal" if str(p).startswith("F") else "ref" for p in P], index=P)})
    m = rp.MFRM(resp, max_score=K, validate=False, exogenous=exo)
    m.calibrate(model="items")
    m.drf_test("grp", model="items", reference="ref", p_method="chi2")
    t = m.drf_item_table
    np.testing.assert_allclose(t["p"].astype(float), chi2.sf(t["T"].astype(float), t["df"].astype(float)))
    with pytest.raises(ValueError, match="p_method='chi2'"):
        m.drf_test("grp", model="items", reference="ref", no_of_samples=0)
    with pytest.raises(ValueError, match="p_method must be"):
        m.drf_test("grp", model="items", reference="ref", p_method="exact")


@pytest.mark.parametrize("model, df", [("global", R - 1), ("items", R * I - I), ("bistretch", 3 * R - 3)])
def test_drf_omnibus_every_model(model, df):
    """The omnibus test (any facet element differing) for every model: a Wald test of every facet element's
    differences on all its components, df the rank left by the frame's constraints (one per component)."""
    import pandas as pd

    base = MFRM_Sim(I, 10, R, K, model="mixed", rater_models=[model] * R, seed=3)
    kw = dict(manual_items=base.native_items.to_numpy(), manual_thresholds=base.native_thresholds.to_numpy())
    s1 = MFRM_Sim(I, 300, R, K, model="mixed", rater_models=dict(base.rater_params), seed=3, **kw)
    s2 = MFRM_Sim(I, 300, R, K, model="mixed", rater_models=dict(base.rater_params), seed=4, **kw)
    r2 = s2.responses.copy()
    r2.index = r2.index.set_levels([f"F{p}" for p in r2.index.levels[1]], level=1)
    resp = pd.concat([s1.responses, r2])
    P = resp.index.get_level_values(1).unique()
    exo = pd.DataFrame({"grp": pd.Series(["focal" if str(p).startswith("F") else "ref" for p in P], index=P)})
    m = rp.MFRM(resp, max_score=K, validate=False, exogenous=exo)
    m.calibrate(model=model)
    m.drf_test("grp", model=model, reference="ref", p_method="chi2", min_category_count=None)
    om = m.drf_omnibus_table
    assert om.loc["focal", "df"] == df and 0 < om.loc["focal", "p"] <= 1
    # Hotelling-type F reference: denominator df = persons in both groups - 2 - df + 1
    assert om.loc["focal", "Denominator df"] == 600 - 2 - df + 1


# ---------------------------------------------------------------------------------------------------------------
# Unused score categories (v2.0.0): as a single representation's fit -- a category no one used in the middle has its
# neighbouring widths set by RSM's CPAT threshold estimator on the pooled responses ('cp-full-rcl', 'cpat-rcl') or
# the unused-category width
# ridge ('cml'); an unused end category, and a category one facet element never used, are held by the ridges
# ---------------------------------------------------------------------------------------------------------------

def _unused_data(kind, N=300, seed=3):
    X = MFRM_Sim(5, N, 6, 4, model="bivector", seed=seed).responses.copy()
    if kind == "middle":
        X = X.replace(2, 3)
    elif kind == "top":
        X = X.replace(4, 3)
    elif kind == "bottom":
        X = X.replace(0, 1)
    elif kind == "rater":
        X.loc["Rater_1"] = X.loc["Rater_1"].replace(4, 3).to_numpy()
    return X


@pytest.mark.parametrize("estimation", ["cp-full-rcl", "cpat-rcl", "cml"])
@pytest.mark.parametrize("kind, models", [
    ("middle", ["global", "items", "thresholds", "bivector", "matrix", "centrality", "pseudo_halo", "bistretch"]),
    ("top", ["thresholds", "centrality"]),
    ("bottom", ["bivector", "pseudo_halo"]),
])
def test_unused_category_mixed_reproduces_the_single_fit(kind, models, estimation):
    X = _unused_data(kind)
    for model in models:
        single = rp.MFRM(X, max_score=4, validate=False)
        single.calibrate(model=model, estimation=estimation)
        mixed = rp.MFRM(X, max_score=4, validate=False)
        mixed.calibrate(model="mixed", rater_models=[model] * 6, estimation=estimation)
        assert mixed.convergence_mixed["converged"]
        np.testing.assert_allclose(mixed.items_mixed.to_numpy(), single.items.to_numpy(), atol=1e-5)
        np.testing.assert_allclose(mixed.thresholds_mixed.to_numpy(), np.asarray(single.thresholds, float),
                                   atol=1e-5)
    if kind == "middle" and estimation != "cml":
        fit = mixed._rcl_fits["mixed"]
        assert fit["split"] == [0, 1, 2] and len(fit["fixed"]) == 3  # every width touches category 2


@pytest.mark.parametrize("kind, estimation", [("top", "cml"), ("middle", "cml"), ("rater", "cp-full-rcl"),
                                              ("rater", "cpat-rcl"), ("rater", "cml")])
@pytest.mark.parametrize("model", ["thresholds", "matrix"])
def test_unused_category_mixed_se_and_no_data_match_the_single_fit(kind, estimation, model):
    """Sandwich SEs, the no-data entries (shared and per facet element) and their SEs, as the single fit's."""
    X = _unused_data(kind)
    single = rp.MFRM(X, max_score=4, validate=False)
    single.calibrate(model=model, estimation=estimation)
    single.std_errors(model=model, se_method="sandwich")
    mixed = rp.MFRM(X, max_score=4, validate=False)
    mixed.calibrate(model="mixed", rater_models=[model] * 6, estimation=estimation)
    mixed.std_errors(model="mixed")
    np.testing.assert_allclose(mixed.item_se_mixed.to_numpy(), single.item_se.to_numpy(), rtol=1e-6)
    np.testing.assert_allclose(mixed.threshold_se_mixed.to_numpy(),
                               np.asarray(getattr(single, f"threshold_se_{model}"), float), rtol=1e-6)
    a, b = single.no_data_parameters[model], mixed.no_data_parameters["mixed"]
    assert (a["cells"], a["thresholds"], a["shared"]) == (b["cells"], b["thresholds"], b["shared"])
    assert b["thresholds"]  # facet elements' thresholds next to a category they never used
    if model == "thresholds":
        r1 = mixed.rater_params_se_mixed["Rater_1"]
        np.testing.assert_allclose(r1.to_numpy(), single.rater_se_thresholds.loc["Rater_1"].to_numpy(float),
                                   rtol=1e-6)


def test_unused_middle_category_rcl_se_and_warning():
    """'cp-full-rcl' with an unused middle category: the widths next to it are set by the empty-category rule, so
    they and the thresholds they place have NaN SEs; one warning names them, and the effects the data inform have
    finite SEs."""
    mixed = rp.MFRM(_unused_data("middle"), max_score=4, validate=False)
    mixed.calibrate(model="mixed", rater_models=["thresholds", "bivector", "items", "matrix", "global", "centrality"],
                    anchors=["Rater_5"])
    with pytest.warns(UserWarning, match="empty-category rule"):
        mixed.std_errors(model="mixed")
    assert np.isnan(list(mixed.cat_width_se_mixed.values())).all()
    assert mixed.threshold_se_mixed.isna().all() and np.isfinite(mixed.item_se_mixed).all()
    assert mixed.no_data_parameters["mixed"]["shared"] == {"widths": [1, 2, 3], "thresholds": [1, 2, 3, 4]}
    assert np.isfinite(mixed.global_effect_se_mixed).all()


def test_unused_categories_argument_for_mixed():
    X = _unused_data("middle", N=150)
    m = rp.MFRM(X, max_score=4, validate=False)
    with pytest.raises(ValueError, match="needs rater_models="):
        m.calibrate(model="mixed", estimation="cml", unused_categories="null")
    with pytest.raises(ValueError, match="must be None, 'ridge' or 'null'"):
        m.calibrate(model="mixed", rater_models=["items"] * 6, estimation="cml", unused_categories="drop")
    with pytest.warns(UserWarning, match="applies to estimation='cml' only"):
        m.calibrate(model="mixed", rater_models=["items"] * 6, unused_categories="ridge")


@pytest.mark.parametrize("kind, df", [("middle", 1), ("top", 2)])
def test_drf_leaves_out_thresholds_next_to_a_category_no_one_used(kind, df):
    """The thresholds next to a category no facet element used in either group carry no data: they leave every
    facet element's fit, the threshold test is on the others (df = their number - 1), the widths next to the
    category are not tested (NaN), and every facet element still has its global effect test."""
    import pandas as pd

    X = _unused_data(kind, N=400, seed=1)
    P = X.index.get_level_values(1).unique()
    exo = pd.DataFrame({"grp": ["focal" if i % 2 else "ref" for i in range(len(P))]}, index=P)
    m = rp.MFRM(X, max_score=4, validate=False, exogenous=exo)
    m.calibrate(model="mixed", rater_models=["thresholds", "bivector", "items", "matrix", "global", "centrality"],
                anchors=["Rater_5"])
    m.drf_test("grp", model="mixed", reference="ref", p_method="chi2")
    t = m.drf_threshold_table
    assert m.drf_table["p"].notna().all()
    assert t.loc[["Rater_1", "Rater_2", "Rater_4"], "df"].tolist() == [df] * 3
    assert t.loc["Rater_6", "df"] == 1  # centrality: the change in its threshold stretch
    widths = t.loc["Rater_1", [f"Difference category {k}" for k in (1, 2, 3)]].to_numpy(float)
    gone = {"middle": [True, True, True], "top": [False, False, True]}[kind]
    assert np.array_equal(np.isnan(widths), gone)


def test_per_rater_selection_with_an_unused_middle_category():
    """The selection's fits keep the widths next to the unused category fixed: the composite penalty counts only
    the fitted coordinates."""
    X = _unused_data("middle", N=200)
    m = rp.MFRM(X, max_score=4, validate=False)
    tab = m.per_rater_model_selection(anchors=["Rater_5"], estimation="cp-full-rcl",
                                      candidate_models=["global", "items", "thresholds"])
    assert tab["Representation"].notna().all() and m._rcl_fits["mixed"]["split"] == [0, 1, 2]


def _null_data(kind, N=300, seed=3):
    X = MFRM_Sim(5, N, 6, 4, model="bivector", seed=seed).responses.copy()
    return {"middle": X.replace(2, 3), "run": X.replace(2, 3).replace(1, 0), "apart": X.replace(1, 0).replace(3, 4),
            "top": X.replace(4, 3)}[kind]


@pytest.mark.parametrize("kind", ["middle", "run", "apart", "top"])
@pytest.mark.parametrize("model", ["global", "items", "thresholds", "bivector", "matrix", "centrality", "pseudo_halo",
                                   "bistretch"])
def test_unused_categories_null_mixed_reproduces_the_single_fit(kind, model):
    """CML with unused_categories='null' (Wilson and Masters' markers): with every facet element on one
    representation, the cell thresholds (markers included), log-likelihood and sandwich SEs are the single fit's.
    One middle category unused, two adjacent ones, two non-adjacent ones, and the top one. With the top category
    unused, 'bivector' and 'matrix' differ by up to 0.001 logits: the ridge penalty also falls on combinations of
    parameters that the
    conditional likelihood does not depend on, which the single and mixed fits parameterise differently."""
    X = _null_data(kind)
    single = rp.MFRM(X, max_score=4, validate=False)
    single.calibrate(model=model, estimation="cml", unused_categories="null")
    mixed = rp.MFRM(X, max_score=4, validate=False)
    mixed.calibrate(model="mixed", rater_models=[model] * 6, estimation="cml", unused_categories="null")
    assert single.convergence["converged"] and mixed.convergence_mixed["converged"]
    tol = 2e-3 if kind == "top" and model in ("bivector", "matrix") else 1e-6
    assert abs(single.convergence["log_likelihood"] - mixed.convergence_mixed["log_likelihood"]) < max(tol, 1e-7)
    va, vb = np.array(single._virtual_items(model)[1]), np.array(mixed._virtual_items("mixed")[1])
    assert np.array_equal(va == np.inf, vb == np.inf) and np.array_equal(va == -np.inf, vb == -np.inf)
    fin = np.isfinite(va)
    assert np.abs(va[fin] - vb[fin]).max() < tol
    t = mixed.thresholds_mixed.to_numpy()
    assert np.array_equal(np.isinf(t), np.isinf(np.asarray(single.thresholds, float)))
    single.std_errors(model=model)
    mixed.std_errors(model="mixed")
    sa, sb = np.asarray(getattr(single, f"threshold_se_{model}"), float), mixed.threshold_se_mixed.to_numpy()
    assert np.array_equal(np.isnan(sa), np.isnan(sb)) and np.isnan(sb[np.isinf(t)]).all()
    np.testing.assert_allclose(sb, sa, atol=1e-5)
    np.testing.assert_allclose(mixed.item_se_mixed.to_numpy(), single.item_se.to_numpy(), atol=1e-5)


def test_unused_categories_null_mixed_downstream():
    """A mixed fit under the markers: persons and conditional fit statistics as the single fit's; with several
    representations, the markers' SEs are NaN (also where a facet element's effects have no data), and DRF runs."""
    import pandas as pd

    X = _null_data("middle", N=400, seed=1)
    P = X.index.get_level_values(1).unique()
    exo = pd.DataFrame({"grp": ["focal" if i % 2 else "ref" for i in range(len(P))]}, index=P)
    a = rp.MFRM(X, max_score=4, validate=False, exogenous=exo)
    a.calibrate(model="matrix", estimation="cml", unused_categories="null")
    m = rp.MFRM(X, max_score=4, validate=False, exogenous=exo)
    m.calibrate(model="mixed", rater_models=["matrix"] * 6, estimation="cml", unused_categories="null")
    a.person_estimates(model="matrix"); m.person_estimates(model="mixed")
    assert np.nanmax(np.abs(a.persons_matrix - m.persons_mixed)) < 1e-8
    a.fit_statistics(model="matrix", conditional=True); m.fit_statistics(model="mixed", conditional=True)
    assert np.nanmax(np.abs(np.asarray(a.item_infit_ms_matrix, float) - np.asarray(m.item_infit_ms_mixed, float))) < 1e-8
    m.calibrate(model="mixed", rater_models=["thresholds", "bivector", "items", "matrix", "global", "centrality"],
                anchors=["Rater_5"], estimation="cml", unused_categories="null")
    assert np.isposinf(m.thresholds_mixed[2]) and m.facet_effects_mixed[2].isna().all()
    m.std_errors(model="mixed")
    assert np.isnan(m.threshold_se_mixed[2]) and np.isnan(m.rater_params_se_mixed["Rater_1"][2])
    assert np.isnan(m.rater_params_se_mixed["Rater_2"]["threshold_effects"][2])
    assert np.isfinite(m.rater_params_se_mixed["Rater_1"][[1, 3, 4]]).all()
    m.drf_test("grp", model="mixed", reference="ref", p_method="chi2")
    assert m.drf_table["p"].notna().all() and m.drf_threshold_table.loc["Rater_1", "df"] == 1


def test_cml_null_fit_from_far_out_starting_thresholds():
    """Categories 1 and 2 unused: the CML search starts from the cp-full-rcl thresholds, which are far out next to
    the unused categories, so the likelihood is almost flat along one direction and Newton-Raphson's step there is
    enormous. No such step increases the likelihood, so none is taken; L-BFGS continues and Newton-Raphson finishes,
    and the fit reaches the maximum."""
    X = _null_data("run")
    single = rp.MFRM(X, max_score=4, validate=False)
    single.calibrate(model="global", estimation="cml", unused_categories="null")
    assert single.convergence["converged"] and single.convergence["solver"] == "lbfgs+newton"
    stretch = rp.MFRM(X, max_score=4, validate=False)
    stretch.calibrate(model="centrality", estimation="cml", unused_categories="null")
    # centrality nests global: its maximum is at least global's
    assert stretch.convergence["log_likelihood"] >= single.convergence["log_likelihood"] - 1e-6


def _unused_draws(K, unused, N, seed):
    """MFRM 'global' data (6 items, 4 raters) drawn with the unused categories' probabilities set to zero."""
    import pandas as pd

    s = MFRM_Sim(6, N, 4, K, model="global", seed=seed)
    P = np.stack([s.cat_probs[k].to_numpy() for k in range(K + 1)], axis=-1)
    P[..., unused] = 0.0
    P /= P.sum(-1, keepdims=True)
    u = np.random.default_rng(seed + 5).random(P.shape[:2])[..., None]
    return pd.DataFrame((u > np.cumsum(P, -1)).sum(-1).astype(float), index=s.cat_probs[0].index,
                        columns=s.cat_probs[0].columns), s


def test_cml_null_steps_that_do_not_exist_are_reported():
    """Categories 1 and 2 unused (maximum score 4): for these data the conditional likelihood is flat along the steps
    between the used categories where the fit stops, so the CML 'null' estimates of the steps do not exist; the
    warning says so (not 'did not converge') and points to 'ridge', which converges."""
    X, _ = _unused_draws(4, [1, 2], 150, 70000 + 4000 + 500 + 9)
    m = rp.MFRM(X, max_score=4, validate=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.calibrate(model="global", estimation="cml", unused_categories="null")
    msgs = " ".join(str(x.message) for x in w)
    assert "do not exist for these data" in msgs and "did not converge" not in msgs
    r = rp.MFRM(X, max_score=4, validate=False)
    r.calibrate(model="global", estimation="cml")
    assert r.convergence["converged"]


def test_cml_ridge_with_only_two_categories_used_converges():
    """Maximum score 3 with categories 1 and 2 unused: every width is next to an unused category; the ridge pulls each
    towards 0 (pulling them towards each other left their common size free), so the fit converges with finite
    thresholds and accurate item locations."""
    X, s = _unused_draws(3, [1, 2], 150, 70000 + 3000 + 500)
    m = rp.MFRM(X, max_score=3, validate=False)
    m.calibrate(model="global", estimation="cml")
    assert m.convergence["converged"] and np.isfinite(np.asarray(m.thresholds, float)).all()
    d, t = m.items - m.items.mean(), s.items - s.items.mean()
    assert np.sqrt(np.mean((d - t) ** 2)) < 0.15


def test_stats_tables_take_the_fits_own_anchors():
    """A joint mixed fit is already in its anchors' frame: its own anchors give the fit as it is (this raised a
    KeyError in the anchored standard errors), and other anchors raise a ValueError asking for a refit."""
    s = MFRM_Sim(I, 300, R, K, model="mixed", rater_models=["global", "global", "items", "global"], seed=5)
    m = rp.MFRM(s.responses, max_score=K, validate=False)
    anchors = list(m.facet_names[:2])
    m.calibrate(model="mixed", rater_models=["global", "global", "items", "global"], anchors=anchors)
    for f in ("rater_stats_df", "item_stats_df", "threshold_stats_df", "category_stats_df"):
        getattr(m, f)(model="mixed", anchors=anchors)
    with pytest.raises(ValueError):
        m.rater_stats_df(model="mixed", anchors=list(m.facet_names[1:3]))
