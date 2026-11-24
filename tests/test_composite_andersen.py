"""global_fit(method='andersen') with 'cp-full-rcl' estimates: Andersen's test on the pairwise conditional
(composite) likelihood, with the adjusted composite LR and a Monte Carlo p-value by default, or the composite LR
with the Satterthwaite p-value (p_method='chi2')."""
import warnings

import numpy as np
import pandas as pd
import pytest

import raschpy as rp
from raschpy.simulation import SLM_Sim, PCM_Sim, RSM_Sim

I = 6


def _data(kind, dif=0.0, N=400, groups=2, seed=1):
    """Responses with an exogenous grouping G; with dif, items 1 and 4 shifted by dif in the last group."""
    items = np.linspace(-1.2, 1.2, I)
    parts = []
    for g in range(groups):
        it = items.copy()
        if g == groups - 1:
            it[[0, 3]] += dif
        n = N // groups
        if kind == "slm":
            X = SLM_Sim(no_of_items=I, no_of_persons=n, manual_items=list(it), seed=seed + g).responses
        elif kind == "rsm":
            X = RSM_Sim(no_of_items=I, no_of_persons=n, max_score=3, manual_items=list(it),
                        manual_thresholds=[-1.0, 0.0, 1.0], seed=seed + g).responses
        else:
            X = PCM_Sim(no_of_items=I, no_of_persons=n, max_score_vector=[3] * I, manual_items=list(it),
                        manual_thresholds=[[-0.8, 0.0, 0.8]] * I, seed=seed + g).responses
        X.index = [f"g{g}_{x}" for x in X.index]
        parts.append(X)
    X = pd.concat(parts)
    exo = pd.DataFrame({"G": [x.split("_")[0] for x in X.index]}, index=X.index)
    return X, exo


def _model(kind, X, exo):
    if kind == "slm":
        m = rp.SLM(X, exogenous=exo, validate=False)
    elif kind == "rsm":
        m = rp.RSM(X, max_score=3, exogenous=exo, validate=False)
    else:
        m = rp.PCM(X, max_score_vector=[3] * I, exogenous=exo, validate=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate()
    assert m.estimation == "cp-full-rcl"
    return m


@pytest.mark.parametrize("kind", ["slm", "pcm", "rsm"])
def test_default_monte_carlo_and_chi2(kind):
    m = _model(kind, *_data(kind))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mc = m.global_fit(method="andersen", covariate="G", seed=3, no_of_samples=40, progress=False)
    assert mc["estimation"] == "cp-full-rcl" and mc["p_method"] == "monte-carlo"
    assert mc["statistic"] == mc["adjusted composite LR"]
    assert 0 < mc["p-value"] <= 1 and mc["reject"] == (mc["p-value"] <= 0.05)
    assert mc["Monte Carlo samples"] >= 1
    q = (I - 1) if kind == "slm" else (I - 1 + 2 if kind == "rsm" else 3 * I - 1)
    assert mc["df"] == q
    assert m.global_fit_item_table is None
    assert list(m.global_fit_obsexp.columns) == ["Persons", "Observed mean", "Expected mean", "z"]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ch = m.global_fit(method="andersen", covariate="G", p_method="chi2")
    liberal = [x for x in w if "liberal for PCM" in str(x.message)]
    assert len(liberal) == (1 if kind == "pcm" else 0)
    assert ch["statistic"] == ch["composite LR"] == mc["composite LR"]
    assert ch["p-value"] == ch["p-value (chi-square)"] == mc["p-value (chi-square)"]
    assert ch["Monte Carlo samples"] is None


def test_monte_carlo_reproducible_and_stops_early():
    m = _model("rsm", *_data("rsm"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = m.global_fit(method="andersen", covariate="G", seed=5, progress=False)
        b = m.global_fit(method="andersen", covariate="G", seed=5, progress=False)
    assert a["p-value"] == b["p-value"] and a["Monte Carlo samples"] == b["Monte Carlo samples"]
    if a["Monte Carlo stopped early"]:
        assert a["p-value"] == 10 / a["Monte Carlo samples"]


@pytest.mark.parametrize("kind", ["slm", "pcm", "rsm"])
def test_large_dif_rejects(kind):
    m = _model(kind, *_data(kind, dif=1.2, N=800))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mc = m.global_fit(method="andersen", covariate="G", seed=1, no_of_samples=40, progress=False)
        ch = m.global_fit(method="andersen", covariate="G", p_method="chi2")
    assert mc["reject"] and mc["p-value"] == 1 / 41
    assert ch["reject"]


def test_three_groups():
    m = _model("pcm", *_data("pcm", N=600, groups=3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.global_fit(method="andersen", covariate="G", seed=2, no_of_samples=20, progress=False)
    assert len(r["group sizes"]) == 3 and r["df"] == 2 * (3 * I - 1)
    assert set(r["composite log-likelihoods"]) == {"g0", "g1", "g2", "pooled"}


def test_frozen_values():
    """The statistics on one dataset, against the prototype's values (research code, 2026-10-29); the p-value is the
    exact weighted chi-square tail (2026-10-31)."""
    m = _model("pcm", *_data("pcm", seed=11))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.global_fit(method="andersen", covariate="G", p_method="chi2")
    expected = {"composite LR": 75.17248957381707, "adjusted composite LR": 21.80376957612665, "df": 17,
                "Satterthwaite scale": 3.993064089425095, "Satterthwaite df": 14.8823451072321,
                "p-value": 0.21270619474228478}  # the exact weighted chi-square tail (Satterthwaite: 0.21551817156806355)
    for k, v in expected.items():
        assert r[k] == pytest.approx(v, rel=1e-6), k


def test_invalid_options():
    m = _model("pcm", *_data("pcm"))
    with pytest.raises(ValueError, match="CML estimates only"):
        m.global_fit(method="andersen", covariate="G", unused_categories="null")
    with pytest.raises(ValueError, match="p_method must be"):
        m.global_fit(method="andersen", covariate="G", p_method="permutation")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(estimation="pair")
    with pytest.raises(ValueError, match="estimation='cp-full-rcl'"):
        m.global_fit(method="andersen", covariate="G")


def _mfrm(model, R=4, N=200, seed=11, responses=None, generating=None):
    from raschpy.simulation import MFRM_Sim

    if responses is None:
        responses = MFRM_Sim(I, N, R, 3, model=generating or model, seed=seed).responses
    m = rp.MFRM(responses, max_score=3, validate=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(model=model)
    return m


@pytest.mark.parametrize("model", ["global", "matrix"])
def test_mfrm_cell_influence(model):
    """Each person's influence on the fitted MFRM grid equals half the change between the fits with the person
    added twice and left out (a central difference), to second order."""
    m = _mfrm(model)
    r = m.responses
    Phi, _, b0 = m._cell_influence(model)
    R = m.no_of_facet_elements
    X3 = r.to_numpy(float).reshape(R, -1, I)
    tot = np.nansum(X3, axis=(0, 2))
    persons = list(r.index.get_level_values(1)[: X3.shape[1]])
    for n in [int(q) for q in np.flatnonzero((tot > 0) & (tot < 3 * I * R))[:2]]:
        rows = r[r.index.get_level_values(1) == persons[n]]
        copy = rows.rename(index=lambda p: f"copy {p}", level=1)
        twice = pd.concat([r, copy]).sort_index(level=0, sort_remaining=False)
        up = _mfrm(model, responses=twice)._cell_influence(model)[2]
        down = _mfrm(model, responses=r.drop(persons[n], level=1))._cell_influence(model)[2]
        np.testing.assert_allclose(Phi[n], (up - down) / 2, atol=5e-3 * np.abs(up - down).max())


def test_mfrm_obsexp_tables():
    """The three tables' observed and expected totals are the sums of their cells'; with two groups covering
    every person, a z near 0 in large samples; sets of single items reproduce the per-item variances."""
    m = _mfrm("global", N=300)
    persons = m.responses.index.get_level_values(1).unique()
    groups = {"A": persons[::2], "B": persons[1::2]}
    t = m._obsexp_tables(groups, ["A", "B"])
    lab = m.facet_label.capitalize()
    assert set(t) == {"cell", "item", "facet"}
    assert len(t["cell"]) == m.no_of_facet_elements * I * 2
    cell = t["cell"].assign(
        O=lambda d: d["Observed mean"] * d["Persons"], E=lambda d: d["Expected mean"] * d["Persons"]
    )
    for kind, by in (("item", ["Item", "Group"]), ("facet", [lab, "Group"])):
        s = cell.groupby(level=by)[["O", "E"]].sum()
        tab = t[kind]
        np.testing.assert_allclose((tab["Observed mean"] * tab["Persons"]).loc[s.index], s["O"], rtol=1e-10)
        np.testing.assert_allclose((tab["Expected mean"] * tab["Persons"]).loc[s.index], s["E"], rtol=1e-10)
    for tab in t.values():
        assert tab["z"].notna().all() and tab["z"].abs().max() < 4
    with pytest.raises(NotImplementedError):
        _mfrm("centrality", N=300, generating="global")._obsexp_tables(groups, ["A", "B"], model="centrality")


def test_corrected_group_variances_sets():
    from raschpy.base import Rasch

    rng = np.random.default_rng(0)
    X = (rng.random((200, 4)) < 0.5).astype(float)
    thr = [np.array([0.0]) for _ in range(4)]
    A = np.eye(4)[:, :3] - 0.25
    Phi = rng.normal(size=(200, 4)) * 0.01
    labels = np.arange(200) % 2
    one = Rasch._corrected_group_variances(X, thr, labels, 2, A, Phi)
    sets = Rasch._corrected_group_variances(X, thr, labels, 2, A, Phi, sets=[[i] for i in range(4)] + [[0, 1]])
    np.testing.assert_allclose(sets[:4], one)
    assert sets[4, 0] != one[0, 0] + one[1, 0]  # a set's variance includes its items' covariances


def _mfrm_exo(model="global", dif=0.0, groups=2, N=300, seed=2):
    """MFRM responses (4 facet elements, every person rated by all) with an exogenous grouping G; with dif,
    the ratings of item 1 in the last group raised by one category with probability dif."""
    from raschpy.simulation import MFRM_Sim

    r = MFRM_Sim(I, N, 4, 3, model=model, seed=seed).responses.copy()
    persons = r.index.get_level_values(1).unique()
    G = pd.Series([f"g{k % groups}" for k in range(len(persons))], index=persons)
    if dif:
        rng = np.random.default_rng(seed)
        last = r.index.get_level_values(1).isin(G.index[G == f"g{groups - 1}"])
        col = r.columns[0]
        up = last & (r[col] < 3).to_numpy() & (rng.random(len(r)) < dif)
        r.loc[up, col] += 1
    m = rp.MFRM(r, max_score=3, validate=False, exogenous=pd.DataFrame({"G": G}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(model=model)
    return m


def test_mfrm_global_fit_frozen_values():
    m = _mfrm_exo()
    s = m.global_fit(seed=3)
    assert s["model"] == "global" and s["split"] == "score" and s["p_method"] == "monte-carlo"
    frozen = dict(lr=54.16305056462079, adjusted=9.59431958792878, p_chi2=0.08069172696856913,
                  p=0.5882352941176471, used=17, df=10)
    got = dict(lr=s["composite LR"], adjusted=s["adjusted composite LR"], p_chi2=s["p-value (chi-square)"],
               p=s["p-value"], used=s["Monte Carlo samples"], df=s["df"])
    for k, v in frozen.items():
        assert got[k] == pytest.approx(v, rel=1e-6), k
    assert m.global_fit_obsexp.index.names == ["Item", "Group"]
    assert m.global_fit_obsexp_facets.shape[0] == 4 * 2 and m.global_fit_obsexp_cells.shape[0] == 4 * I * 2
    assert any(k[0] == "andersen" for k in m.global_fit_results)


def test_mfrm_global_fit_chi2_warns_and_dif_rejects():
    m = _mfrm_exo(dif=0.6)
    with pytest.warns(UserWarning, match="not dependable for MFRM"):
        s = m.global_fit(covariate="G", p_method="chi2")
    assert s["split"] == "exogenous" and s["p-value"] < 0.001 and s["reject"]
    z = m.global_fit_obsexp["z"].abs()
    assert z.idxmax()[0] == m.responses.columns[0]
    s = m.global_fit(covariate="G", seed=1, no_of_samples=99)
    assert s["p-value"] <= 0.05


def test_mfrm_global_fit_three_groups_and_stretch():
    m = _mfrm_exo(groups=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = m.global_fit(covariate="G", p_method="chi2")
    assert set(s["group sizes"]) == {"g0", "g1", "g2"}
    assert set(m.global_fit_obsexp.index.get_level_values("Group")) == {"g0", "g1", "g2"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(model="centrality")
        s = m.global_fit(model="centrality", split="random", seed=2, p_method="chi2")
    assert np.isfinite(s["statistic"]) and m.global_fit_obsexp is None and m.global_fit_obsexp_cells is None


def test_mfrm_global_fit_not_comparable_and_deprecated():
    m = _mfrm_exo()
    r = m.responses.copy()
    persons = m.exogenous.index[m.exogenous["G"] == "g1"]
    r.loc[(m.facet_names[0], persons), :] = np.nan  # the first facet element rates no one in g1
    m2 = rp.MFRM(r, max_score=3, validate=False, exogenous=m.exogenous)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m2.calibrate()
    with pytest.raises(ValueError, match="cannot be compared"):
        m2.global_fit(covariate="G", p_method="chi2")
    with pytest.warns(FutureWarning, match="andersen_lr_test\\(\\) is deprecated"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            s = m.andersen_lr_test_global(split_by="random")
    assert s["split"] == "random"


def test_mfrm_global_fit_cml_conditional_lr():
    """Under estimation='cml': the conditional LR of the group and pooled CML refits, with the composite
    route's df; the chi-square p-value is the default, penalised (the default ridges) or not, without a warning;
    'monte-carlo' is offered; with every penalty at 0, each log-likelihood is the fit's own."""
    m = _mfrm_exo()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_composite = m.global_fit(covariate="G", p_method="chi2")["df"]
        m.calibrate(estimation="cml")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = m.global_fit(covariate="G")
    assert s["estimation"] == "cml" and s["penalised"] and s["p_method"] == "chi2"
    assert s["df"] == df_composite and s["p-value"] == s["p-value (chi-square)"]
    ll = s["log-likelihoods"]
    assert s["statistic"] == pytest.approx(2 * (ll["g0"] + ll["g1"] - ll["pooled"]))
    assert m.global_fit_obsexp["z"].notna().all() and m.global_fit_obsexp_cells.shape[0] == 4 * I * 2
    mc = m.global_fit(covariate="G", seed=1, no_of_samples=19, p_method="monte-carlo")
    assert mc["p_method"] == "monte-carlo" and mc["statistic"] == s["statistic"] and 0 < mc["p-value"] <= 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(estimation="cml", global_ridge=0, item_ridge=0, threshold_ridge=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = m.global_fit(covariate="G")
    assert not s["penalised"] and s["p_method"] == "chi2" and s["df"] == df_composite
    for name, g in m.global_fit_groups.items():
        assert s["log-likelihoods"][name] == pytest.approx(g.convergence["log_likelihood"], rel=1e-8)


def test_mfrm_global_fit_cml_dif_rejects():
    m = _mfrm_exo(dif=0.6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(estimation="cml")
        s = m.global_fit(covariate="G", seed=1, no_of_samples=99)
    assert s["p-value"] <= 0.05 and s["reject"]
    assert m.global_fit_obsexp["z"].abs().idxmax()[0] == m.responses.columns[0]


@pytest.mark.parametrize("model", ["centrality", "mixed"])
def test_mfrm_global_fit_cml_and_cpat_rcl_other_models(model):
    m = _mfrm_exo()
    kw = {"rater_models": ["global", "items", "thresholds", "centrality"]} if model == "mixed" else {}
    for est in ("cml", "cpat-rcl"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.calibrate(model=model, estimation=est, **kw)
            s = m.global_fit(model=model, covariate="G", p_method="chi2")
        assert s["estimation"] == est and np.isfinite(s["statistic"]) and s["df"] > 0
        assert m.global_fit_obsexp is None


def test_mfrm_global_fit_cpat_rcl_and_pair_cpat():
    m = _mfrm_exo()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(estimation="cpat-rcl")
        s = m.global_fit(covariate="G", seed=1, no_of_samples=19)
    assert s["estimation"] == "cpat-rcl" and s["p_method"] == "monte-carlo" and "adjusted composite LR" in s
    assert m.global_fit_obsexp is not None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(estimation="pair-cpat")
    with pytest.raises(ValueError, match="do not maximise a likelihood"):
        m.global_fit(covariate="G")
