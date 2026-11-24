"""fit_statistics(conditional=True): the conditional fit bootstrap and item_q1_overall simulate their datasets from a
'cp-full-rcl' fit of the model (the object's own fit if it is 'cp-full-rcl' or 'cml'; the statistics and refits are
the object's), and a UserWarning names score categories that no person with a non-extreme total used when the fit
gives them a probability above zero."""

import warnings

import numpy as np
import pandas as pd
import pytest

import raschpy as rp
from raschpy import _esf
from raschpy.simulation import MFRM_Sim

N, I, K = 300, 6, 3


def _pcm_data(seed=1):
    rng = np.random.default_rng(seed)
    th, d, tau = rng.normal(size=N), np.linspace(-1, 1, I), np.array([-1.0, 0.0, 1.0])
    X = np.zeros((N, I))
    for i in range(I):
        num = np.concatenate([np.zeros((N, 1)), np.cumsum(th[:, None] - d[i] - tau, 1)], 1)
        p = np.exp(num - num.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        X[:, i] = (rng.random(N)[:, None] > p.cumsum(1)).sum(1)
    return pd.DataFrame(X, columns=[f"I{i}" for i in range(I)])


def _fit(df, cls=rp.PCM, **cal):
    m = cls(df, max_score_vector=[K] * I) if cls is rp.PCM else cls(df, max_score=K)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(**cal)
    return m


def _conditional(m, record=False, **kw):
    kw = dict(dict(se=False, conditional=True, conditional_samples=5, bootstrap_samples=5, q1_overall_samples=5,
                   seed=1, progress=False), **kw)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.fit_statistics(**kw)
    return [str(x.message) for x in w if "non-extreme total used" in str(x.message)
            or "own fit" in str(x.message)]


@pytest.mark.parametrize("cls,est", [(rp.PCM, "pair"), (rp.RSM, "cpat-rcl"), (rp.RSM, "pair-cpat")])
def test_simulated_from_cp_full_rcl_fit(cls, est, monkeypatch):
    df = _pcm_data()
    m = _fit(df, cls, estimation=est)
    names = list(m.item_names)
    target = _fit(df, cls, estimation="cp-full-rcl")._cml_threshold_list(names)
    used, orig = [], _esf.sample_given_totals

    def spy(X, thr, rng):
        used.append(thr)
        return orig(X, thr, rng)

    monkeypatch.setattr(_esf, "sample_given_totals", spy)
    _conditional(m)
    assert m.conditional_null_fit == "cp-full-rcl" and m.item_fit_bootstrap_info["null_fit"] == "cp-full-rcl"
    assert len(used) == 15  # 5 item-restscore draws + 5 bootstrap + 5 overall Q1 datasets
    for thr in used:
        for a, b in zip(thr, target):
            np.testing.assert_allclose(a, b, rtol=1e-10)
    # the statistics are still at the object's own estimates
    own = m._cml_threshold_list(names)
    assert not all(np.allclose(a, b) for a, b in zip(own, target))


@pytest.mark.parametrize("cls,est", [(rp.PCM, "cp-full-rcl"), (rp.PCM, "cml"), (rp.RSM, "cp-full-rcl")])
def test_own_fit_when_cp_full_rcl_or_cml(cls, est):
    m = _fit(_pcm_data(), cls, estimation=est)
    _conditional(m)
    assert m.conditional_null_fit == "own"


def test_slm_null():
    rng = np.random.default_rng(2)
    th, d = rng.normal(size=N), np.linspace(-1, 1, I)
    X = pd.DataFrame((rng.random((N, I)) < 1 / (1 + np.exp(-(th[:, None] - d)))).astype(float))
    m = rp.SLM(X)
    m.calibrate(estimation="pair", method="cos")
    _conditional(m)
    assert m.conditional_null_fit == "cp-full-rcl"


def test_fallback_to_own_fit(monkeypatch):
    m = _fit(_pcm_data(), estimation="pair")

    def fail(self, idx, items=None, cml_hessian=None):
        raise ValueError("no fit")

    monkeypatch.setattr(rp.PCM, "_global_fit_subset", fail)
    with pytest.warns(UserWarning, match="simulated from the object's fit"):
        m.fit_statistics(se=False, conditional=True, conditional_samples=5, bootstrap_samples=None,
                         q1_overall_samples=None, seed=1, progress=False)
    assert m.conditional_null_fit == "own"  # the item-restscore draws always simulate
    with pytest.warns(UserWarning, match="simulated from the object's fit"):
        assert m._conditional_null(list(m.item_names))[1] == "own"


def _extreme_only(df, item, k, K_max):
    """Category k of item used only by persons with an extreme total: everyone else moved off it, and one
    added person scoring k there with the maximum (k = K_max) or zero (k = 0) on every item."""
    df = df.copy()
    df.loc[df[item] == k, item] = k - 1 if k else 1
    extra = pd.DataFrame([[float(k)] * df.shape[1]], columns=df.columns, index=["extreme"])
    return pd.concat([df, extra])


def test_unused_category_warning_pcm():
    df = _pcm_data()
    df.loc[df["I1"] == 2, "I1"] = 1
    with pytest.warns(UserWarning, match=r"(?s)item I1, category 2: 0 responses.*conditional fit statistics"):
        m = _fit(df, estimation="pair")
    # named by the constructor's warning: not repeated
    assert not _conditional(m, bootstrap_samples=None, q1_overall_samples=None)
    # used only by a person with an extreme total: the constructor counts it, conditional fit does not
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = _fit(_extreme_only(_pcm_data(), "I1", 3, 3), estimation="pair")
    msgs = _conditional(m, bootstrap_samples=None, q1_overall_samples=None)
    assert len(msgs) == 1 and "I1 (3)" in msgs[0] and "unused_categories='null'" in msgs[0]
    # probability zero under unused_categories='null': nothing to warn about
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = _fit(_extreme_only(_pcm_data(), "I1", 3, 3), estimation="cml", unused_categories="null", ridge=0)
    assert not _conditional(m, bootstrap_samples=None, q1_overall_samples=None)


def test_unused_category_warning_rsm():
    df = _pcm_data()
    df.loc[df["I1"] == 2, "I1"] = 1  # used on the other items: the shared thresholds are informed
    assert not _conditional(_fit(df, rp.RSM, estimation="cpat-rcl"), bootstrap_samples=None,
                            q1_overall_samples=None)
    df[df == 2] = 1
    m = rp.RSM(df, max_score=K)
    with pytest.warns(UserWarning, match=r"(?s)Score category 2 has no responses.*conditional fit statistics"):
        m.calibrate(estimation="cpat-rcl")
    assert not _conditional(m, bootstrap_samples=None, q1_overall_samples=None)
    # used only by a person with an extreme total
    d = _pcm_data()
    for c in d:
        d.loc[d[c] == 3, c] = 2
    m = _fit(_extreme_only(d, "I0", 3, 3), rp.RSM, estimation="cpat-rcl")
    msgs = _conditional(m, bootstrap_samples=None, q1_overall_samples=None)
    assert len(msgs) == 1 and "score category 3 on any item" in msgs[0]


def _mfrm(resp, model, est):
    m = rp.MFRM(resp)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.calibrate(model=model, estimation=est)
        m.fit_statistics(model=model, se=False, conditional=True, conditional_samples=10, bootstrap_samples=5,
                         seed=1)
    msgs = [str(x.message) for x in w if "non-extreme total used" in str(x.message)]
    return m, msgs


def test_mfrm_null_and_unused_warning():
    resp = MFRM_Sim(no_of_items=4, no_of_persons=200, no_of_facet_elements=3, max_score=3, model="items",
                    seed=3).responses
    for est, expected in (("cpat-rcl", "cp-full-rcl"), ("pair-cpat", "cp-full-rcl"), ("cp-full-rcl", "own")):
        m, msgs = _mfrm(resp, "thresholds", est)
        assert m.fit_bootstrap_info_thresholds["null_fit"] == expected and not msgs
    first = resp.index.get_level_values(0).unique()[0]
    r = resp.copy()
    sub = r.loc[first]
    r.loc[first] = sub.mask(sub == 2, 1).values  # one facet element never uses category 2
    # named by calibrate()'s warning (with the conditional-fit consequence): not repeated
    m = rp.MFRM(r)
    with pytest.warns(UserWarning, match=rf"(?s){first} \(category 2: 0\).*conditional fit statistics"):
        m.calibrate(model="thresholds", estimation="cpat-rcl")
    assert not _mfrm(r, "thresholds", "cpat-rcl")[1]
    assert not _mfrm(r, "global", "cpat-rcl")[1]  # shared thresholds: informed by the other facet elements
    r2 = resp.mask(resp == 2, 1)
    assert not _mfrm(r2, "global", "cpat-rcl")[1]  # named by calibrate()'s empty-category warning
    # 'pair-cpat' 'global' gives no unused-category warning at calibrate(), so conditional fit names it
    _, msgs = _mfrm(r2, "global", "pair-cpat")
    assert len(msgs) == 1 and "score category 2 in any rating" in msgs[0]
