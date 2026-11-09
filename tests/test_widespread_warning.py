"""The widespread-difference warning of dif_test and drf_test (v2.0.0): one UserWarning when the anchor selection
keeps fewer than half of the units shared by the reference and a focal group, none on data without differences."""
import warnings

import numpy as np
import pandas as pd
import pytest

import raschpy as rp
from raschpy.simulation import MFRM_Sim, SLM_Sim

MSG = "widespread"


def _slm(shifts, seed=1, n=300):
    a = SLM_Sim(no_of_items=len(shifts), no_of_persons=2 * n, seed=seed)
    b = SLM_Sim(no_of_items=len(shifts), no_of_persons=2 * n, seed=seed + 100,
                manual_items=a.items.values + np.asarray(shifts, float), manual_persons=a.persons.values)
    r = pd.concat([a.responses.iloc[:n], b.responses.iloc[n:]])
    g = pd.Series(np.where(np.arange(2 * n) < n, "A", "B"), index=r.index)
    return rp.SLM(r, exogenous=pd.DataFrame({"G": g}))


def _mfrm(shifts, seed=1, n=300):
    a = MFRM_Sim(6, 2 * n, len(shifts), 3, model="global", seed=seed)
    b = MFRM_Sim(6, 2 * n, len(shifts), 3, model="global", seed=seed + 100,
                 manual_raters=a.facet_effects.values + np.asarray(shifts, float), manual_items=a.items.values,
                 manual_thresholds=a.thresholds.values, manual_persons=a.persons.values)
    in_b = a.responses.index.get_level_values(1).isin(list(a.person_names)[n:])
    resp = pd.concat([a.responses[~in_b], b.responses[b.responses.index.get_level_values(1).isin(list(a.person_names)[n:])]])
    resp = resp.reindex(a.responses.index)
    g = pd.Series(["A"] * n + ["B"] * n, index=list(a.person_names))
    return rp.MFRM(resp, max_score=3, exogenous=pd.DataFrame({"G": g}), validate=False)


def _messages(fn):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fn()
    return [str(w.message) for w in rec if MSG in str(w.message)]


SPREAD = [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0]


def test_dif_warns_when_dif_is_widespread():
    m = _slm(SPREAD + [0.0, 0.0])
    msgs = _messages(lambda: m.dif_test("G", reference="A", no_of_samples=30, seed=1))
    assert len(msgs) == 1 and "kept only" in msgs[0] and "of the 10 items" in msgs[0]


def test_dif_quiet_without_dif():
    m = _slm([0.0] * 10)
    assert _messages(lambda: m.dif_test("G", reference="A", no_of_samples=30, seed=1)) == []


def test_drf_warns_once_when_drf_is_widespread():
    m = _mfrm(SPREAD)
    msgs = _messages(lambda: m.drf_test("G", model="global", reference="A"))
    assert len(msgs) == 1 and "facet elements" in msgs[0] and "of the 8 facet elements" in msgs[0]


def test_drf_quiet_without_drf():
    m = _mfrm([0.0] * 8)
    assert _messages(lambda: m.drf_test("G", model="global", reference="A")) == []
