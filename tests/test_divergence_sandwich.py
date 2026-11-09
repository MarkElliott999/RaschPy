"""
rater_stats_df(divergence_test=...) for the stretch models: the standard error of each free-minus-stretch difference
from the stacked sandwich of the two fits (each person's influence on both), against the paired bootstrap.
"""

import warnings

import numpy as np
import pytest

import raschpy as rp

warnings.filterwarnings("ignore")

I, K = 8, 3
ESTIMATIONS = ["cp-full-rcl", "cpat-rcl", "cml"]


@pytest.mark.parametrize("estimation", ESTIMATIONS)
def test_grid_influence_reproduces_the_sandwich(estimation):
    """Each person's influence on the effect grid: summed squares give the sandwich variances std_errors() reports."""
    s = rp.MFRM_Sim_Centrality(no_of_items=I, no_of_persons=300, no_of_facet_elements=5, max_score=K, seed=4)
    m = rp.MFRM(s)
    for model in ("thresholds", "centrality"):
        m.calibrate(model=model, estimation=estimation)
        m.std_errors(model=model)
        se = np.sqrt((m._grid_influence(model) ** 2).sum(axis=0)).reshape(5, I, K)[:, 0, :]
        assert np.abs(se - getattr(m, f"rater_se_{model}").to_numpy()).max() < 1e-8


def test_divergence_test_uses_the_sandwich_and_agrees_with_the_bootstrap():
    s = rp.MFRM_Sim_Bistretch(no_of_items=I, no_of_persons=400, no_of_facet_elements=5, max_score=K, seed=4)
    m = rp.MFRM(s)
    m.calibrate(model="bistretch")
    m.rater_stats_df(model="bistretch", divergence_test="wald")
    assert m.divergence_se_method["bistretch"] == "sandwich"
    sw = m._divergence_sandwich["bistretch"]
    b = rp.MFRM(s)
    b.calibrate(model="bistretch")
    b._divergence_influence = lambda model: None
    b.rater_stats_df(model="bistretch", divergence_test="wald", no_of_samples=200, seed=1)
    assert b.divergence_se_method["bistretch"] == "bootstrap"
    bs = b._get_divergence_inputs("bistretch", False)
    for axis in ("items", "thresholds"):
        ratio = sw[axis][2].to_numpy() / bs[axis][2].to_numpy()
        assert 0.9 < np.median(ratio) < 1.1
        np.testing.assert_allclose(sw[axis][0].to_numpy() - sw[axis][1].to_numpy(),
                                   bs[axis][0].to_numpy() - bs[axis][1].to_numpy(), atol=1e-8)
