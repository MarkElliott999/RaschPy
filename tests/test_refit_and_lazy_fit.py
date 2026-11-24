"""Recalibrating with other settings refreshes what was derived from the earlier fit (v1.4.0 known
issue), and results and plotting methods calibrate an uncalibrated object first (SLM, PCM, RSM)."""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import raschpy as rp
from raschpy.simulation import PCM_Sim, RSM_Sim, SLM_Sim


def _make(kind):
    if kind == "slm":
        return lambda: rp.SLM(SLM_Sim(no_of_items=8, no_of_persons=200, seed=3).responses)
    if kind == "pcm":
        r = PCM_Sim(no_of_items=5, no_of_persons=200, max_score_vector=[2, 2, 3, 3, 2], seed=3).responses
        return lambda: rp.PCM(r, max_score_vector=[2, 2, 3, 3, 2])
    r = RSM_Sim(no_of_items=6, no_of_persons=200, max_score=3, seed=3).responses
    return lambda: rp.RSM(r, max_score=3)


# settings that change the estimates, standard errors and fit
_REFIT = {"slm": dict(constant=2.0), "pcm": dict(constant=2.0), "rsm": dict(constant=2.0, estimation="pair-cpat")}


@pytest.mark.parametrize("kind", ["slm", "pcm", "rsm"])
def test_recalibrating_refreshes_derived_results(kind):
    make = _make(kind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = make()
        m.calibrate()
        m.fit_statistics()
        m.item_stats_df()
        m.person_stats_df()
        before = m.item_stats.copy()
        m.calibrate(**_REFIT[kind])
        m.item_stats_df(seed=1) if kind == "rsm" else m.item_stats_df()
        m.person_stats_df()
        fresh = make()
        fresh.calibrate(**_REFIT[kind])
        fresh.item_stats_df(seed=1) if kind == "rsm" else fresh.item_stats_df()
        fresh.person_stats_df()
    assert not np.allclose(before["Infit MS"], m.item_stats["Infit MS"])
    for col in ("Estimate", "SE", "Infit MS", "Outfit MS"):
        assert np.allclose(m.item_stats[col], fresh.item_stats[col], equal_nan=True), col
    for col in ("Estimate", "CSEM", "Infit MS"):
        assert np.allclose(m.person_stats[col], fresh.person_stats[col], equal_nan=True), col
    # what the object held before its first fit is kept
    assert m.responses is not None and m.responses.equals(fresh.responses)


@pytest.mark.parametrize("kind", ["slm", "pcm", "rsm"])
@pytest.mark.parametrize(
    "call",
    ["person_estimates()", "score_lookup_table()", "icc('Item_1')", "tcc()", "test_info()",
     "test_csem()", "wright_map()", "std_residuals_plot()", "class intervals"],
)
def test_uncalibrated_object_calibrates_first(kind, call):
    if call == "class intervals":  # needs person estimates as well
        call = "class_intervals_thresholds('Item_1')" if kind == "rsm" else "class_intervals_cats('Item_1')"
    m = _make(kind)()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eval("m." + call)
    plt.close("all")
    assert hasattr(m, "items")
