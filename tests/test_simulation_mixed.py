"""
MFRM_Sim(model='mixed') -- each facet element on its own representation -- and the reporting frame (anchors=,
neutral_anchors=) added to every MFRM simulation class, plus the shared checks of manual facet element parameters.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from raschpy.simulation import (
    MFRM_Sim,
    MFRM_Sim_Bistretch,
    MFRM_Sim_Centrality,
    MFRM_Sim_PseudoHalo,
)

warnings.filterwarnings("ignore")

EIGHT = ["global", "items", "thresholds", "bivector", "matrix", "centrality", "pseudo_halo", "bistretch"]
I, R, K = 5, 4, 3


def _probs(sim):
    return np.stack([sim.cat_probs[c].to_numpy() for c in range(sim.max_score + 1)])


def _grid(sim):
    """Every facet element x item x threshold location minus the mean person location, from the reported values."""
    d, t, e = sim.items.to_numpy(), sim.thresholds.to_numpy(), np.asarray(sim.facet_effects, float)
    r = len(sim.facet_names)
    if e.ndim == 1:
        e = np.broadcast_to(e[:, None, None], (r, I, K))
    elif e.shape == (r, I):
        e = e[:, :, None] * np.ones(K)
    elif e.shape == (r, K):
        e = e[:, None, :] * np.ones((1, I, 1))
    else:
        e = e.reshape(r, I, K)
    return d[None, :, None] + t[None, None, :] + e - sim.persons.mean()


def test_mixed_list_gives_each_rater_its_representation():
    s = MFRM_Sim(6, 200, 8, 3, model="mixed", rater_models=EIGHT, seed=1)
    assert s.rater_models.tolist() == EIGHT
    assert s.responses.shape == (8 * 200, 6)
    for m in EIGHT:
        names = [f"facet_effects_mixed_{m}"] if m != "bivector" else [
            "facet_effects_mixed_bivector_items", "facet_effects_mixed_bivector_thresholds"]
        for n in names:
            assert n in s._mixed_tables
    assert list(s.facet_effects_mixed_bistretch.columns) == ["lambda", "omega_items", "omega_thresholds"]


def test_mixed_native_parameters_give_the_category_probabilities():
    s = MFRM_Sim(I, 100, R, K, model="mixed", rater_models=["matrix", "centrality", "pseudo_halo", "bistretch"], seed=2)
    delta, tau, theta = s.native_items.to_numpy(), s.native_thresholds.to_numpy(), s.native_persons.to_numpy()
    p = np.empty((K + 1, R * 100, I))
    for k, r in enumerate(s.facet_names):
        model, prm = s.rater_params[r]
        if model == "matrix":
            dev = prm.to_numpy()
        elif model == "centrality":
            dev = prm["lambda"] + (prm["omega"] - 1) * np.broadcast_to(tau, (I, K))
        elif model == "pseudo_halo":
            dev = prm["lambda"] + (prm["omega"] - 1) * np.broadcast_to(delta[:, None], (I, K))
        else:
            dev = prm["lambda"] + (prm["omega_items"] - 1) * delta[:, None] + (prm["omega_thresholds"] - 1) * tau[None]
        b = delta[:, None] + tau[None, :] + dev  # (I, K)
        lg = np.concatenate([np.zeros((100, I, 1)), np.cumsum(theta[:, None, None] - b[None], axis=2)], axis=2)
        e = np.exp(lg - lg.max(axis=2, keepdims=True))
        p[:, k * 100:(k + 1) * 100] = (e / e.sum(axis=2, keepdims=True)).transpose(2, 0, 1)
    np.testing.assert_allclose(_probs(s), p, atol=1e-12)


def test_mixed_rater_params_round_trip():
    s = MFRM_Sim(I, 80, R, K, model="mixed", rater_models=["bivector", "thresholds", "centrality", "global"], seed=3)
    t = MFRM_Sim(I, 80, R, K, model="mixed", rater_models=s.rater_params, manual_items=s.native_items.to_numpy(),
                 manual_thresholds=s.native_thresholds.to_numpy(), manual_persons=s.native_persons.to_numpy(), seed=9)
    np.testing.assert_allclose(_probs(s), _probs(t), atol=1e-12)


def test_mixed_partial_dict_and_random_assignment():
    manual = {"Rater_2": ("items", {f"Item_{i + 1}": 0.1 * i for i in range(I)}), "Rater_3": "centrality"}
    s = MFRM_Sim(I, 50, R, K, model="mixed", rater_models=manual, candidate_models=["global"], seed=4)
    assert s.rater_models.to_dict() == {"Rater_1": "global", "Rater_2": "items", "Rater_3": "centrality",
                                        "Rater_4": "global"}
    np.testing.assert_allclose(s.rater_params["Rater_2"][1].to_numpy(), 0.1 * np.arange(I))
    s = MFRM_Sim(I, 50, 6, K, model="mixed", candidate_models=["items", "matrix"], proportions={"items": 1, "matrix": 0},
                 seed=5)
    assert set(s.rater_models) == {"items"}


def test_mixed_list_of_wrong_length_raises():
    with pytest.raises(ValueError, match="3 representations but there are 4"):
        MFRM_Sim(I, 50, R, K, model="mixed", rater_models=["global", "items", "matrix"], seed=1)


def test_mixed_labelled_values_are_aligned_by_label():
    items = [f"Item_{i + 1}" for i in range(I)]
    vals = pd.Series(np.arange(I, dtype=float), index=items)[::-1]  # reversed order, aligned by label
    s = MFRM_Sim(I, 50, R, K, model="mixed", rater_models={"Rater_1": ("items", vals)}, seed=1)
    np.testing.assert_allclose(s.rater_params["Rater_1"][1].to_numpy(), np.arange(I))
    with pytest.raises(ValueError, match="unknown item label"):
        MFRM_Sim(I, 50, R, K, model="mixed", rater_models={"Rater_1": ("items", {"Item_9": 0.0})}, seed=1)
    with pytest.raises(ValueError, match="expected 3 values"):
        MFRM_Sim(I, 50, R, K, model="mixed", rater_models={"Rater_1": ("thresholds", [0.1, 0.2])}, seed=1)
    with pytest.raises(ValueError, match="finite"):
        MFRM_Sim(I, 50, R, K, model="mixed", rater_models={"Rater_1": ("global", np.nan)}, seed=1)


def test_mixed_stretch_bounds_allow_zero_and_negative_multipliers():
    s = MFRM_Sim(I, 50, 6, K, model="mixed", rater_models=["pseudo_halo"] * 6, stretch_range=(-0.5, 0.0), seed=6)
    om = s.facet_effects_mixed_pseudo_halo["omega"]
    assert ((om >= -0.5) & (om <= 0.0)).all()
    s = MFRM_Sim(I, 50, R, K, model="mixed", rater_models={"Rater_1": ("centrality", (0.2, 0.0))}, seed=6)
    assert s.rater_params["Rater_1"][1]["omega"] == 0.0


def test_manual_raters_retrofit_aligns_series_and_thresholds_dicts():
    items = [f"Item_{i + 1}" for i in range(I)]
    vals = pd.Series(np.arange(I, dtype=float), index=items)[::-1]
    s = MFRM_Sim(I, 50, 2, K, model="items", manual_raters={"A": vals, "B": vals * 0}, seed=1)
    np.testing.assert_allclose(s.facet_effects.loc["A"].to_numpy(), np.arange(I))
    s = MFRM_Sim(I, 50, 2, K, model="thresholds", manual_raters={"A": {3: 0.3, 1: 0.1, 2: 0.2}, "B": [0, 0, 0]},
                 seed=1)
    np.testing.assert_allclose(s.facet_effects.loc["A"].to_numpy(), [0.1, 0.2, 0.3])


_MAKERS = {
    "global": lambda **k: MFRM_Sim(I, 150, R, K, model="global", **k),
    "items": lambda **k: MFRM_Sim(I, 150, R, K, model="items", **k),
    "thresholds": lambda **k: MFRM_Sim(I, 150, R, K, model="thresholds", **k),
    "bivector": lambda **k: MFRM_Sim(I, 150, R, K, model="bivector", **k),
    "matrix": lambda **k: MFRM_Sim(I, 150, R, K, model="matrix", **k),
    "centrality": lambda **k: MFRM_Sim_Centrality(I, 150, K, R, **k),
    "pseudo_halo": lambda **k: MFRM_Sim_PseudoHalo(I, 150, K, R, **k),
    "bistretch": lambda **k: MFRM_Sim_Bistretch(I, 150, K, R, **k),
    "mixed": lambda **k: MFRM_Sim(I, 150, R, K, model="mixed",
                                  rater_models=["matrix", "centrality", "pseudo_halo", "items"], **k),
}


@pytest.mark.parametrize("name", list(_MAKERS))
def test_anchors_change_the_reported_frame_only(name):
    f = _MAKERS[name]
    plain, anch = f(seed=3), f(seed=3, anchors=["Rater_1", "Rater_2"])
    pd.testing.assert_frame_equal(plain.responses, anch.responses)
    np.testing.assert_allclose(_grid(plain), _grid(anch), atol=1e-12)
    assert f(seed=3, neutral_anchors=True).responses.equals(plain.responses)  # no-op without anchors
    neutral = f(seed=3, anchors=["Rater_1"], neutral_anchors=True)
    assert np.abs(np.asarray(neutral.facet_effects.loc[["Rater_1"]], float)).max() == 0.0


def test_anchored_truth_matches_calibrate_anchor():
    import raschpy as rp

    a = ["Rater_1", "Rater_2"]
    s = MFRM_Sim(6, 3000, 5, 3, model="items", seed=1, anchors=a)
    m = rp.MFRM(s.responses, max_score=3, validate=False)
    m.calibrate(model="items")
    m.calibrate_anchor("items", a)
    assert np.abs(m.anchor_facet_effects_items.to_numpy() - s.facet_effects.to_numpy()).max() < 0.15
    assert np.abs(m.anchor_items_items.to_numpy() - s.items.to_numpy()).max() < 0.1
