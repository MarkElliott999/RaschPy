"""Kauermann-Carroll small-sample correction of MFRM's sandwich standard errors (std_errors(se_correction=)) and
of the tests built on the sandwich: the per-person information for any number of facet elements, the low-rank form
of the correction, and the API."""
import warnings

import numpy as np
import pytest
from scipy.stats import t as t_dist

from raschpy import MFRM, _rcl
from raschpy.base import Rasch
from raschpy.simulation import MFRM_Sim


def _dense_person_bread(b, X, K, drop, empty):
    R, P, I = X.shape
    t = _rcl.person_comparison_terms(R, I, K, drop, empty)
    Xp = np.transpose(X, (1, 0, 2)).reshape(P, R * I)
    bf = b.ravel()
    s = _rcl._sig(bf[t["gx"]] - bf[t["gy"]])
    w = s * (1 - s)
    E = ((Xp[:, t["first"]] == t["a1"]) & (Xp[:, t["second"]] == t["b1"])).astype(float) + (
        (Xp[:, t["first"]] == t["a2"]) & (Xp[:, t["second"]] == t["b2"])
    )
    n = R * I * K
    B = np.zeros((P, n, n))
    for k in range(len(w)):
        d = np.zeros(n)
        d[t["gx"][k]] += 1
        d[t["gy"][k]] -= 1
        B += (E[:, k] * w[k])[:, None, None] * np.outer(d, d)[None]
    return B


def _fd_person_bread(b, X, K, drop, empty):
    f = (lambda bb: _rcl.person_grid_scores(bb, X, K, drop)) if empty is None else (
        lambda bb: _rcl.full_person_grid_scores(bb, X, K, drop, empty))
    R, P, I = X.shape
    n, h = R * I * K, 1e-6
    out = np.zeros((P, n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        out[:, :, j] = -(f(b + e.reshape(b.shape)).reshape(P, -1) - f(b - e.reshape(b.shape)).reshape(P, -1)) / (2 * h)
    return out


@pytest.mark.parametrize("R,I,K", [(1, 4, 3), (3, 3, 3), (4, 2, 4)])
@pytest.mark.parametrize("drop,empty", [((), None), ((), ()), ((1,), None), ((), (2,))])
def test_person_terms_are_the_derivative_of_the_person_scores(R, I, K, drop, empty):
    rng = np.random.default_rng(R * 10 + I)
    X = rng.integers(0, K + 1, size=(R, 20, I)).astype(float)
    X[rng.random(X.shape) < 0.3] = np.nan
    b = np.sort(rng.normal(size=(R, I, K)), axis=2)
    np.testing.assert_allclose(_dense_person_bread(b, X, K, drop, empty), _fd_person_bread(b, X, K, drop, empty),
                               atol=1e-8)
    if R == 1:  # the single facet element helpers it generalises
        ref = (_rcl.person_grid_bread(b, X, K, slice(None), drop) if empty is None
               else _rcl.full_person_grid_bread(b, X, K, slice(None), drop, empty))
        np.testing.assert_allclose(_dense_person_bread(b, X, K, drop, empty), ref, atol=1e-14)


@pytest.mark.parametrize("R,I,K,empty", [(1, 5, 3, None), (3, 4, 3, ()), (5, 3, 2, None)])
def test_low_rank_correction_matches_the_dense_one(R, I, K, empty):
    rng = np.random.default_rng(3)
    X = rng.integers(0, K + 1, size=(R, 60, I)).astype(float)
    X[rng.random(X.shape) < 0.4] = np.nan
    b = np.sort(rng.normal(size=(R, I, K)), axis=2)
    n = R * I * K
    p = n - 3
    J = rng.normal(size=(n, p))
    D = _dense_person_bread(b, X, K, (), empty)
    A = np.linalg.pinv(np.einsum("ia,nij,jb->ab", J, D, J) + 0.5 * np.eye(p))
    U = rng.normal(size=(60, p))
    ref = Rasch._kc_scores(U, lambda sl: J.T @ D[sl] @ J, A)
    new = _rcl.kc_grid_scores(U, b, X, K, J, A, (), empty)
    np.testing.assert_allclose(new, ref, atol=1e-10)
    assert np.abs(new - U).max() > 1e-3  # the correction does something


@pytest.fixture(scope="module")
def sim():
    return MFRM_Sim(4, 60, 4, 3, seed=7)


@pytest.mark.parametrize("estimation", ["cp-full-rcl", "cpat-rcl", "cml"])
@pytest.mark.parametrize("model", ["global", "matrix", "bivector"])
def test_corrected_se_at_least_the_uncorrected(sim, estimation, model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MFRM(sim.responses, 3)
        m.calibrate(model=model, estimation=estimation)
        m.std_errors(model=model, se_correction=None)
        se0 = np.asarray(getattr(m, f"item_se_{model}"), float)
        assert m.se_correction is None
        m.std_errors(model=model, interval=0.95)
        se1 = np.asarray(getattr(m, f"item_se_{model}"), float)
    assert m.se_correction == getattr(m, f"se_correction_{model}") == "kauermann-carroll"
    assert np.all(se1 >= se0 - 1e-12) and np.any(se1 > se0 * 1.001)
    # intervals: estimate +/- t (persons - 1) SE
    tq = t_dist.ppf(0.975, m.no_of_persons - 1)
    items = np.asarray(getattr(m, f"items_{model}"), float)
    low = np.asarray(getattr(m, f"item_low_{model}"), float)
    np.testing.assert_allclose(items - low, tq * se1, rtol=1e-8)


def test_hessian_and_bootstrap_take_no_correction(sim):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MFRM(sim.responses, 3)
        m.calibrate(model="global", estimation="cml")
        m.std_errors(model="global", se_method="hessian")
        assert m.se_correction is None
        m.calibrate(model="global", estimation="pair-cpat")
        m.std_errors(model="global", no_of_samples=5, seed=1)
        assert m.se_correction is None


def test_invalid_se_correction(sim):
    m = MFRM(sim.responses, 3)
    with pytest.raises(ValueError, match="se_correction"):
        m.std_errors(model="global", se_correction="hc3")


def test_tests_follow_the_setting(sim):
    """The sandwich behind the tests (here the paired influences of divergence_test) uses the correction unless
    std_errors(se_correction=None) was the last call."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MFRM(sim.responses, 3)
        m.calibrate(model="matrix")
        on = m._grid_influence("matrix")
        m.std_errors(model="matrix", se_correction=None)
        off = m._grid_influence("matrix")
    v_on, v_off = (on ** 2).sum(0), (off ** 2).sum(0)
    assert np.all(v_on >= v_off - 1e-12) and np.any(v_on > v_off * 1.001)


def test_mixed_joint_fit(sim):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MFRM(sim.responses, 3)
        m.calibrate(model="mixed", rater_models=["global", "items", "thresholds", "global"])
        m.std_errors(model="mixed", se_correction=None)
        se0 = np.asarray(m.item_se_mixed, float)
        m.std_errors(model="mixed")
        se1 = np.asarray(m.item_se_mixed, float)
    assert m.se_correction_mixed == "kauermann-carroll"
    assert np.all(se1 >= se0 - 1e-12) and np.any(se1 > se0 * 1.001)


def _count_kc(monkeypatch):
    from raschpy import _rcl

    calls = []
    real = _rcl.kc_grid_scores

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(_rcl, "kc_grid_scores", spy)
    return calls


@pytest.mark.parametrize("method", ["per_rater_model_selection", "model_selection", "check_anchor_homogeneity"])
def test_selection_se_correction_argument(sim, method, monkeypatch):
    """se_correction= on the selection methods and the anchor check: the default (None) makes no correction,
    'kauermann-carroll' makes it, a bad value raises, and the object's own setting is unchanged afterwards."""
    kw = dict(per_rater_model_selection=dict(test="LR", progress=False),
              model_selection=dict(test="LR", candidate_models=["global", "items"], progress=False),
              check_anchor_homogeneity=dict(anchors=list(sim.responses.index.get_level_values(0).unique()[:3])))[method]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MFRM(sim.responses, 3)
        with pytest.raises(ValueError, match="se_correction"):
            getattr(m, method)(se_correction="hc3", **kw)
        calls = _count_kc(monkeypatch)
        getattr(m, method)(**kw)
        assert not calls
        assert "_se_correction_opt" not in m.__dict__
        getattr(m, method)(se_correction="kauermann-carroll", **kw)
        assert calls
        m.std_errors(model="global", se_correction=None)
        getattr(m, method)(se_correction="kauermann-carroll", **kw)
        assert m._se_correction_opt is None
