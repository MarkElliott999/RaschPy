"""MFRM penalties (v2.0.0): the three-level penalty (global_ridge, item_ridge, threshold_ridge,
interaction_ridge, stretch_ridge) for cp-full-rcl and estimation='cml', the deprecated v1.4 ridge=, the
unused-category width ridge, and CML standard errors."""
import json
import os
import warnings

import numpy as np
import pandas as pd
import pytest

import raschpy as rp
from raschpy import _esf, _rcl
from raschpy.simulation import MFRM_Sim

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "external_data")
MODELS = ("global", "items", "thresholds", "bivector", "matrix", "centrality", "pseudo_halo", "bistretch")
CELL = ("global", "items", "thresholds", "bivector", "matrix")


def _sim(seed=2, persons=150, missing=0.3):
    return MFRM_Sim(no_of_items=6, no_of_persons=persons, max_score=3, no_of_facet_elements=4,
                    model="matrix", seed=seed, missing=missing).responses


def _fit(resp, model, **kw):
    m = rp.MFRM(resp, max_score=3, validate=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(model=model, **kw)
    return m


def test_deprecated_ridge_reproduces_v14():
    """ridge= gives v1.4's estimates (stored from v1.4) with a FutureWarning, and cannot be combined with
    the new penalties or with estimation='cml'."""
    ref = json.load(open(os.path.join(DATA, "mfrm_v14_ridge_reference.json")))
    sp = ref["responses"]
    resp = pd.DataFrame(sp["data"], columns=sp["columns"]).set_index(sp["columns"][:2])
    resp.index.names = [None, None]
    for model in MODELS:
        m = rp.MFRM(resp, max_score=3, validate=False)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            m.calibrate(model=model, ridge=0.3)
        assert any(issubclass(w.category, FutureWarning) for w in rec)
        np.testing.assert_allclose(m.items.to_numpy(float), ref[model]["items"], atol=1e-8)
        np.testing.assert_allclose(np.asarray(m.thresholds, float), ref[model]["thresholds"], atol=1e-8)
    with pytest.raises(ValueError, match="cannot be combined"):
        rp.MFRM(resp, max_score=3, validate=False).calibrate(model="items", ridge=0.3, item_ridge=0.2)
    with pytest.raises(ValueError, match="does not apply to estimation='cml'"):
        rp.MFRM(resp, max_score=3, validate=False).calibrate(model="items", ridge=0.3, estimation="cml")


@pytest.mark.parametrize("estimation,defaults", [("cp-full-rcl", rp.MFRM._RCL_RIDGE), ("cml", rp.MFRM._CML_RIDGE)])
def test_penalty_defaults_are_stored(estimation, defaults):
    resp = _sim()
    for model in MODELS:
        m = _fit(resp, model, estimation=estimation)
        for name, models in rp.MFRM._RIDGE_APPLIES.items():
            d = defaults[name]
            expect = (d[model] if isinstance(d, dict) else d) if model in models else None
            assert getattr(m, name) == expect, (model, name)
        assert m.ridge is None


def test_penalty_not_applying_is_ignored_with_a_warning():
    resp = _sim()
    with pytest.warns(UserWarning, match="threshold_ridge=0.5 does not apply"):
        rp.MFRM(resp, max_score=3, validate=False).calibrate(model="items", threshold_ridge=0.5)


@pytest.mark.parametrize("model", CELL)
def test_level_penalty_value_gradient_and_matrix(model):
    """CellModel's three-level penalty: 0.5 theta' Q theta equals the value, the gradient matches finite
    differences, and Q equals the penalty built from the centring operators on the location grid."""
    R, I, K = 4, 5, 3
    lv = {"global": 0.7, "item": 0.3, "threshold": 0.2, "interaction": 5.0}
    cm = _rcl.CellModel(model, R, I, K, levels=lv)
    rng = np.random.default_rng(0)
    th = rng.normal(size=cm.p)
    v, g = cm.level_penalty(th)
    Q = cm.level_matrix()
    assert abs(v - 0.5 * th @ Q @ th) < 1e-9
    eps = 1e-6
    gn = np.array([(cm.level_penalty(th + eps * e)[0] - cm.level_penalty(th - eps * e)[0]) / (2 * eps)
                   for e in np.eye(cm.p)])
    np.testing.assert_allclose(g, gn, atol=1e-6)
    A = cm.jacobian()
    n = R * I * K
    C = np.kron(np.eye(R) - 1.0 / R, np.eye(I * K))
    L = np.kron(np.eye(R), np.full((1, I * K), 1.0 / (I * K)))
    D = np.eye(n) - np.kron(np.eye(R), np.full((I * K, I * K), 1.0 / (I * K)))
    cols = lambda names: np.where(np.isin(np.arange(cm.p), np.concatenate(
        [np.arange(cm.p)[cm.slices[k]] for k in names if k in cm.slices] or [np.array([], int)])), 1.0, 0.0)
    Ai, At, Ae = A * cols(["a"]), A * cols(["t", "t2"]), A * cols(["e"])
    Pl, Pi, Pt = L @ C @ A, D @ C @ Ai, D @ C @ At
    Qr = 0.7 * Pl.T @ Pl + 0.3 * Pi.T @ Pi + 0.2 * Pt.T @ Pt + 5.0 * Ae.T @ Ae
    np.testing.assert_allclose(Q, Qr, atol=1e-9)


@pytest.mark.parametrize("model", MODELS)
def test_cml_maximises_the_penalised_conditional_likelihood(model):
    """estimation='cml' converges, and its estimates maximise the penalised conditional log-likelihood of
    the virtual items: the objective falls in every coordinate direction."""
    m = _fit(_sim(), model, estimation="cml")
    assert m.convergence["converged"]
    assert getattr(m, f"estimation_{model}") == "cml"
    fit = m._rcl_fits[model]
    R, I, K = m.no_of_facet_elements, m.no_of_items, m.max_score
    Xv, Q, th = fit["X"], fit["Q"], fit["theta"]
    grid = (lambda t: fit["cm"].grid(t)) if fit["kind"] == "cml-cell" else (lambda t: fit["sm"].grid(t))
    prep = _esf.prepare(Xv, [K] * (R * I))

    def obj(t):
        return _esf.cond_loglik(Xv, list(grid(t).reshape(R * I, K)), prepared=prep)[0] - 0.5 * t @ Q @ t

    f0 = obj(th)
    assert abs(f0 - m.convergence["penalised_log_likelihood"]) < 1e-6
    for e in np.eye(len(th))[:: max(1, len(th) // 12)]:
        assert obj(th + 1e-3 * e) <= f0 + 1e-9 and obj(th - 1e-3 * e) <= f0 + 1e-9


def test_cml_width_ridge_for_an_unused_category():
    """With a score category unused by everyone, CML has estimates through the width ridge; the widths and
    thresholds it sets are in no_data_parameters, with posterior SEs (NaN with no_data_se='nan')."""
    resp = _sim(persons=200, missing=0.0)
    resp = resp.where(resp != 3, 2)  # top category unused
    m = rp.MFRM(resp, max_score=3, validate=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(model="global", estimation="cml")
    assert m.convergence["converged"] and np.all(np.isfinite(m.thresholds))
    assert m.cml_unused_categories == {"all items": [3]} and m._cml_ridge_widths == [2]
    with pytest.warns(UserWarning, match="the width ridge sets them"):
        m.std_errors(model="global")
    shared = m.no_data_parameters["global"]["shared"]
    assert shared == {"widths": [2], "thresholds": [3], "kind": "ridge"}
    assert np.isfinite(m.threshold_se_global[2]) and np.isfinite(m.cat_width_se_global[2])
    with pytest.warns(UserWarning, match="no_data_se='nan'"):
        m.std_errors(model="global", no_data_se="nan")
    assert np.isnan(m.threshold_se_global[2]) and np.isnan(m.cat_width_se_global[2])


@pytest.mark.parametrize("model", ("items", "matrix", "bistretch"))
def test_cml_standard_errors(model):
    """CML SEs: the sandwich by default, 'hessian' on request (close to it under the model), and the
    bootstrap refits with CML."""
    m = _fit(_sim(persons=250), model, estimation="cml")
    m.std_errors(model=model)
    assert getattr(m, f"se_method_{model}") == "sandwich"
    sw = np.asarray(m.item_se, float)
    m.std_errors(model=model, se_method="hessian")
    assert getattr(m, f"se_method_{model}") == "hessian"
    he = np.asarray(m.item_se, float)
    assert np.all(np.isfinite(sw)) and np.all(np.isfinite(he))
    assert np.all(np.abs(np.log(sw / he)) < 0.3)
    with pytest.raises(ValueError, match="se_method='hessian' needs"):
        _fit(_sim(), "items").std_errors(model="items", se_method="hessian")


def _null_data(case):
    resp = MFRM_Sim(no_of_items=6, no_of_persons=250, max_score=4, no_of_facet_elements=4, model="matrix",
                    seed=5, missing=0.2).responses
    if case == "interior":
        return resp.where(resp != 2, 1)
    if case == "top":
        return resp.where(resp != 4, 3)
    return resp.where(resp != 0, 1)


@pytest.mark.parametrize("case", ["interior", "top", "bottom"])
def test_null_single_facet_element_equals_rsm(case):
    """unused_categories='null' with one facet element is RSM's 'null' (Wilson and Masters; RSM's matches
    psychotools): same item locations, threshold markers and held sums."""
    from raschpy.simulation import RSM_Sim
    X = RSM_Sim(no_of_items=6, no_of_persons=300, max_score=4, seed=4).responses
    X = {"interior": X.where(X != 2, 1), "top": X.where(X != 4, 3), "bottom": X.where(X != 0, 1)}[case]
    r = rp.RSM(X, max_score=4, validate=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r.calibrate(estimation="cml", unused_categories="null")
        Xm = X.copy()
        Xm.index = pd.MultiIndex.from_product([["R1"], X.index])
        m = rp.MFRM(Xm, max_score=4, validate=False)
        m.calibrate(model="global", estimation="cml", unused_categories="null")
    np.testing.assert_allclose(m.items.reindex(r.items.index).to_numpy(float), r.items.to_numpy(float), atol=1e-7)
    np.testing.assert_allclose(np.asarray(m.thresholds, float), np.asarray(r.thresholds, float), atol=1e-7)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("case", ["interior", "bottom"])
def test_null_tables_reproduce_the_fit(model, case):
    """The stored items, thresholds and facet effects, read by the engine's rule (a threshold holding a sum
    carries the item location and item-level facet effects that many times), give the fitted cell thresholds
    (up to the persons' origin); the fit maximises the conditional likelihood with markers; unused categories
    get probability zero."""
    resp = _null_data(case)
    m = rp.MFRM(resp, max_score=4, validate=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(model=model, estimation="cml", unused_categories="null")
    assert m.convergence["converged"]
    fit = m._rcl_fits[model]
    Xv, thr, _ = m._virtual_items(model)
    B = np.concatenate(thr)
    bf = fit["b"] if fit["kind"] == "cml-cell" else fit["sm"].grid(fit["theta"]).ravel()
    fin = np.isfinite(bf)
    assert np.all(B[~fin] == bf[~fin])
    mc = np.tile(m._null_categories[model]["mult"], len(bf) // m.max_score)
    c = np.sum((B - bf)[fin] * mc[fin]) / np.sum(mc[fin] ** 2)
    np.testing.assert_allclose(B[fin], bf[fin] + c * mc[fin], atol=1e-9)
    ll = _esf.cond_loglik(Xv, list(B.reshape(-1, m.max_score)))[0]
    assert abs(ll - m.convergence["log_likelihood"]) < 1e-6
    probs, _ = m._cat_probs_mfrm(np.array([-1.0, 0.0, 1.5]), list(m.item_names), list(m.facet_names),
                                 m.thresholds, model, getattr(m, m._sev_attr(model)))
    unused = m.cml_unused_categories["all items"]
    for p in probs.values():
        np.testing.assert_allclose(p.sum(axis=0), 1.0)
        assert np.all(p[unused] == 0)


def test_null_standard_errors():
    """'null' SEs: finite for the item locations and free thresholds, NaN at the markers; close to the
    Hessian's under the model."""
    m = rp.MFRM(_null_data("interior"), max_score=4, validate=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(model="thresholds", estimation="cml", unused_categories="null")
        m.std_errors(model="thresholds")
    ts = np.asarray(m.threshold_se_thresholds, float)
    assert np.isnan(ts[1]) and np.all(np.isfinite(ts[[0, 2, 3]]))
    sw = np.asarray(m.item_se, float)
    assert np.all(np.isfinite(sw))
    rs = m.rater_se_thresholds
    assert rs[2].isna().all() and rs[[1, 3, 4]].notna().all().all()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.std_errors(model="thresholds", se_method="hessian")
    assert np.all(np.abs(np.log(sw / np.asarray(m.item_se, float))) < 0.3)
