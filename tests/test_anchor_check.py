"""
Anchor sets with cut cells (calibrate(model='mixed', anchors=), per_rater_model_selection(anchors=)) and
check_anchor_homogeneity()'s check on joint fits: overall test, the global-effect search, the pattern test within the
suggested set, and the check object's actions.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import raschpy as rp
from raschpy import _rcl
from raschpy._anchor_check import corrected_reject
from raschpy._anchors import AnchorSet, parse_cell
from raschpy.simulation import MFRM_Sim

warnings.filterwarnings("ignore")

I, K = 8, 3
ANCHORS = [f"Rater_{r}" for r in range(1, 6)]
ITEMS = [f"Item_{i}" for i in range(1, I + 1)]
ESTIMATIONS = ["cp-full-rcl", "cpat-rcl", "cml"]


def _sim(deviations=None, n=800, seed=11):
    """5 anchors on 'global' (global effect 0) plus Rater_6 (global effect 0.3); deviations {rater: (representation,
    effects)}."""
    rm = {a: ("global", 0.0) for a in ANCHORS}
    rm["Rater_6"] = ("global", 0.3)
    rm.update(deviations or {})
    s = MFRM_Sim(no_of_items=I, no_of_persons=n, no_of_facet_elements=6, max_score=K, model="mixed",
                 rater_models=rm, seed=seed)
    return rp.MFRM(s.responses)


TWO_CELLS = {"Rater_3": ("items", 0.8 * np.eye(I)[4]), "Rater_4": ("thresholds", -0.8 * np.eye(K)[1])}


# ---- MixedDesign cuts ----------------------------------------------------------------------------------------

def test_design_cuts_add_one_free_shift_per_cell():
    d0 = _rcl.MixedDesign(["global"] * 4, 4, 3, anchors=[0, 1, 2])
    d1 = _rcl.MixedDesign(["global"] * 4, 4, 3, anchors=[0, 1, 2],
                          cuts={0: [("item", 1)], 1: [("threshold", 2), ("cell", 3, 0)]})
    assert d1.p == d0.p + 3
    th = np.random.default_rng(1).normal(size=d1.p)
    J = d1.grid_jacobian(th)
    e = 1e-6
    Jn = np.column_stack([(d1.grid(th + e * np.eye(d1.p)[k]) - d1.grid(th - e * np.eye(d1.p)[k])).ravel() / (2 * e)
                          for k in range(d1.p)])
    assert np.abs(J - Jn).max() < 1e-8
    assert np.linalg.matrix_rank(J) == d1.p
    assert abs(d1.unpack(th)["lam"][[0, 1, 2]].sum()) < 1e-12


@pytest.mark.parametrize("cuts, message", [
    ({3: [("item", 0)]}, "not an anchor"),
    ({0: [("item", i) for i in range(4)]}, "every item cut"),
    ({0: [("item", 0)], 1: [("item", 0)], 2: [("item", 0)]}, "every anchor is cut"),
])
def test_design_cuts_refuse_what_cannot_be_identified(cuts, message):
    with pytest.raises(ValueError, match=message):
        _rcl.MixedDesign(["global"] * 4, 4, 3, anchors=[0, 1, 2], cuts=cuts)


def test_design_cuts_need_an_anchor_fixing_each_part():
    """frame='part': the threshold part is fixed by the facet elements without it; with every other facet element
    on a representation with thresholds, cutting two of the only fixing anchor's three thresholds leaves their shifts
    undefined, while an anchor on 'items' (no threshold part) can fix them."""
    cuts = {2: [("threshold", 0), ("threshold", 1)]}
    with pytest.raises(ValueError, match="no remaining anchor fixes.*threshold 1; anchor 2 at threshold 2"):
        _rcl.MixedDesign(["thresholds", "thresholds", "global", "thresholds"], 4, 3, anchors=[0, 1, 2], cuts=cuts)
    d = _rcl.MixedDesign(["thresholds", "items", "global", "thresholds"], 4, 3, anchors=[0, 1, 2], cuts=cuts)
    assert d.reported[2] == "thresholds"


def test_design_cuts_need_global_anchors():
    with pytest.raises(ValueError, match="only for anchors on 'global'"):
        _rcl.MixedDesign(["items", "global", "global", "global"], 4, 3, anchors=[0, 1, 2], cuts={0: [("item", 0)]})


# ---- AnchorSet -----------------------------------------------------------------------------------------------

def test_cells_are_named_as_the_tables_print_them():
    assert parse_cell("Item_5", ITEMS, K) == ("item", "Item_5")
    assert parse_cell("threshold 2", ITEMS, K) == ("threshold", "Threshold 2")
    assert parse_cell("Threshold 2", ITEMS, K) == ("threshold", "Threshold 2")
    assert parse_cell(("Item_5", "THRESHOLD 3"), ITEMS, K) == ("cell", ("Item_5", "Threshold 3"))
    for bad in ("Item_99", "Threshold 4", ("Item_5", "Item_6")):
        with pytest.raises(ValueError):
            parse_cell(bad, ITEMS, K)
    with pytest.raises(ValueError, match="both an item name and a threshold label"):
        parse_cell("Threshold 1", ITEMS + ["Threshold 1"], K)


def test_anchor_sets_are_immutable_and_logged():
    a = AnchorSet(ANCHORS)
    b = a.cut("Rater_3", "Item_5", ITEMS, K)
    c = b.drop("Rater_4")
    assert a.cuts == {} and a.anchors == ANCHORS
    assert b.cuts == {"Rater_3": [("item", "Item_5")]}
    assert "Rater_4" not in c.anchors and c.adjusted
    assert [e["action"] for e in c.log] == ["cut", "drop"]
    d = c.restore("Rater_4").restore("Rater_3", "Item_5", ITEMS, K)
    assert set(d.anchors) == set(ANCHORS) and d.cuts == {} and not d.adjusted
    t = b.table(ITEMS, K)
    assert not t.loc["Rater_3", "Item_5"] and t.loc["Rater_3", "Item_4"] and t.loc["Rater_1"].all()


# ---- fits with cut cells ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("estimation", ESTIMATIONS)
def test_cut_cells_recover_the_shifts(estimation):
    m = _sim(TWO_CELLS)
    aset = AnchorSet(ANCHORS).cut("Rater_3", "Item_5", ITEMS, K).cut("Rater_4", "threshold 2", ITEMS, K)
    m.calibrate(model="mixed", rater_models={f: "global" for f in m.facet_names}, anchors=aset, estimation=estimation)
    shifts = dict(zip(m.anchor_cuts_mixed["Cell"], m.anchor_cuts_mixed["Shift"]))
    assert abs(shifts["Item_5"] - 0.8) < 0.25
    assert abs(shifts["Threshold 2"] + 0.8) < 0.25
    # the global effects stay defined by the uncut cells: the anchors near 0, Rater_6 near 0.3
    sev = m.global_effect_mixed
    assert sev[["Rater_1", "Rater_2", "Rater_5"]].abs().max() < 0.1 and abs(sev["Rater_6"] - 0.3) < 0.1
    assert m.anchor_set_mixed is aset
    # each cut anchor is reported under the representation its cuts imply: flat at its global effect off the cut
    assert m.rater_models["Rater_3"] == "items" and m.rater_models["Rater_4"] == "thresholds"
    it = m.facet_effects_mixed_items.loc["Rater_3"]
    assert it.drop("Item_5").abs().max() < 0.1 and np.allclose(it.drop("Item_5"), it.iloc[0])
    assert abs(it["Item_5"] - it.iloc[0] - shifts["Item_5"]) < 1e-9
    th = m.facet_effects_mixed_thresholds.loc["Rater_4"]
    assert abs(th[2] - th[1] - shifts["Threshold 2"]) < 1e-9 and abs(th[1] - th[3]) < 1e-9
    assert m.rater_params_mixed["Rater_3"][0] == "items"


def test_cut_anchor_reporting_and_refit():
    """Item and threshold cuts on one anchor imply 'bivector'; a refit with the reported rater_models gives the same
    fit, and the selection table reports the implied representation without selecting it."""
    m = _sim(TWO_CELLS)
    aset = AnchorSet(ANCHORS).cut("Rater_3", "Item_5", ITEMS, K).cut("Rater_3", "threshold 2", ITEMS, K)
    m.calibrate(model="mixed", rater_models={f: "global" for f in m.facet_names}, anchors=aset)
    assert m.rater_models["Rater_3"] == "bivector"
    grid = m.facet_effects_mixed.copy()
    m.calibrate(model="mixed", rater_models=m.rater_models, anchors=aset)
    assert np.allclose(m.facet_effects_mixed, grid, atol=1e-6)
    tab = m.per_rater_model_selection(anchors=aset)
    assert tab.loc["Rater_3", "Representation"] == "bivector"
    assert "cells cut" in tab.loc["Rater_3", "Note"]
    assert _rcl.cut_representation(["cell", "item"]) == "matrix"


def test_cut_cells_need_a_joint_fit_and_global_anchors():
    m = _sim()
    aset = AnchorSet(ANCHORS).cut("Rater_3", "Item_5", ITEMS, K)
    with pytest.raises(ValueError, match="implied representation"):
        m.calibrate(model="mixed", rater_models={f: ("thresholds" if f == "Rater_3" else "global")
                                                 for f in m.facet_names}, anchors=aset)
    with pytest.raises(ValueError, match="estimation="):
        m.calibrate(model="mixed", rater_models={f: "global" for f in m.facet_names}, anchors=aset,
                    estimation="pair-cpat")


@pytest.mark.parametrize("estimation", ESTIMATIONS)
def test_per_rater_selection_selects_the_anchors(estimation):
    """By default the anchors are selected like the other raters: the two patterned anchors get their generating
    representations, the rest stay on 'global'."""
    m = _sim(TWO_CELLS)
    m.per_rater_model_selection(anchors=ANCHORS, estimation=estimation)
    # Rater_3's two-item pattern is close to its item stretch: within the item tolerance of pseudo_halo, the
    # build-up stops there (as test='LR' does)
    got = m.rater_models[ANCHORS].tolist()
    assert got[2] in ("items", "pseudo_halo") and got[:2] + got[3:] == ["global", "global", "thresholds", "global"]
    assert m.per_rater_model_selection_info["force_global_anchors"] is False


@pytest.mark.parametrize("estimation", ESTIMATIONS)
def test_per_rater_selection_force_global_anchors(estimation):
    m = _sim(TWO_CELLS)
    m.per_rater_model_selection(anchors=ANCHORS, estimation=estimation, force_global_anchors=True)
    assert (m.rater_models[ANCHORS] == "global").all()
    assert m.per_rater_model_selection_table.loc[ANCHORS, "Note"].notna().all()


def test_per_rater_selection_keeps_an_anchor_without_each_part():
    """A sole anchor fixes the instrument in every part, so it stays on 'global'; the selection table shows the
    candidates it may not take as NaN."""
    m = _sim(TWO_CELLS)
    tab = m.per_rater_model_selection(anchors=["Rater_3"], estimation="cp-full-rcl")
    assert m.rater_models["Rater_3"] == "global"
    assert tab.loc["Rater_3", "AIC items"] != tab.loc["Rater_3", "AIC items"]  # NaN


@pytest.mark.parametrize("estimation", ESTIMATIONS)
def test_an_anchor_off_global_is_fitted_not_zeroed(estimation):
    """An anchor given a representation with a part the other anchors lack is free in that part (it used to be held
    at zero by the anchors' average, silently fitting 'global')."""
    m = _sim(TWO_CELLS)
    rm = {f: ("items" if f == "Rater_3" else "global") for f in m.facet_names}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(model="mixed", rater_models=rm, anchors=ANCHORS, estimation=estimation)
    est = np.asarray(m.rater_params_mixed["Rater_3"][1], float)
    true = 0.8 * np.eye(I)[4]
    assert np.abs((est - est.mean()) - (true - true.mean())).max() < 0.3


def test_design_frame_rules():
    """frame='part': an anchor with a part the others lack is free in it; frame='average': held at zero."""
    models = ["matrix", "global", "global", "global"]
    part = _rcl.MixedDesign(models, 4, 3, anchors=[0, 1, 2])
    avg = _rcl.MixedDesign(models, 4, 3, anchors=[0, 1, 2], frame="average")
    assert all(part.part_free.values()) and not any(avg.part_free.values())
    assert part.slices["E"].stop - part.slices["E"].start == 3 * 2
    assert avg.slices["E"].stop - avg.slices["E"].start == 0
    # every anchor with the part: the anchors' average either way
    both = _rcl.MixedDesign(["items", "items", "global"], 4, 3, anchors=[0, 1])
    assert not both.part_free["item"]


# ---- the check -----------------------------------------------------------------------------------------------

def test_corrections():
    p = {"a": 0.001, "b": 0.02, "c": 0.03, "d": 0.5}
    assert corrected_reject(p, 0.05, None) == {"a": True, "b": True, "c": True, "d": False}
    # Holm: 0.02 > 0.05 / 3, so a only
    assert corrected_reject(p, 0.05, "holm") == {"a": True, "b": False, "c": False, "d": False}
    # BH: 0.03 <= 3 * 0.05 / 4, so a, b and c
    assert corrected_reject(p, 0.05, "bh") == {"a": True, "b": True, "c": True, "d": False}


@pytest.mark.parametrize("estimation", ESTIMATIONS)
def test_check_on_agreeing_anchors_suggests_nothing(estimation):
    check = _sim(seed=3).check_anchor_homogeneity(anchors=ANCHORS, estimation=estimation)
    assert check.estimation == estimation
    assert check.suggestions.empty
    assert check.suggested.anchors == ANCHORS and not check.suggested.pattern_free
    assert not check.global_effects["Flagged"].any() and not check.pattern["Pattern free"].any()


CENTRAL = {"Rater_2": ("thresholds", np.array([-0.8, 0.0, 0.8]))}  # pattern only: no change in global effect


def test_an_anchor_differing_in_pattern_anchors_through_its_global_effect():
    m = _sim(CENTRAL)
    check = m.check_anchor_homogeneity(anchors=ANCHORS)
    assert m.anchor_check is check
    assert check.suggested.anchors == ANCHORS and check.suggested.pattern_free == ["Rater_2"]
    s = check.suggestions
    assert list(zip(s["rater"], s["Action"], s["Reason"])) == [("Rater_2", "free pattern", "pattern")]
    assert not check.anchors.pattern_free  # nothing applied by the check itself
    again = check.apply_suggested()
    assert again.anchors.pattern_free == ["Rater_2"]
    assert "changed: free pattern Rater_2" in repr(again)


def test_actions_return_new_checks():
    m = _sim(CENTRAL)
    check = m.check_anchor_homogeneity(anchors=ANCHORS)
    c2 = check.free_pattern("Rater_3")
    assert c2.anchors.pattern_free == ["Rater_3"] and not check.anchors.pattern_free
    assert c2.previous is check and m.anchor_check is c2
    assert not c2.hold_pattern("Rater_3").anchors.pattern_free
    c3 = c2.drop("Rater_3")
    assert "Rater_3" not in c3.anchors.anchors and not c3.anchors.pattern_free
    assert c3.restore("Rater_3").anchors.anchors[-1] == "Rater_3"
    applied = m.check_anchor_homogeneity(anchors=ANCHORS, action="apply")
    assert applied.anchors.pattern_free == ["Rater_2"]
    with pytest.raises(ValueError, match="action must be"):
        m.check_anchor_homogeneity(anchors=ANCHORS, action="drop")


def test_two_anchors_that_differ_in_pattern_cannot_be_told_apart():
    m = _sim({"Rater_1": ("items", 1.5 * (np.eye(I)[2] - np.eye(I)[5]))})
    check = m.check_anchor_homogeneity(anchors=["Rater_1", "Rater_2"])
    assert set(check.suggested.pattern_free) == {"Rater_1", "Rater_2"}
    assert any("can't say which one is off" in n for n in check.notes)


def test_a_cut_cell_is_left_out_of_the_pattern_test():
    """An anchor shifted at one item differs in pattern; with that item cut for it, it agrees."""
    m = _sim({"Rater_3": ("items", 2.0 * (np.eye(I)[4] - np.eye(I)[4].mean()))})
    assert m.check_anchor_homogeneity(anchors=ANCHORS).suggested.pattern_free == ["Rater_3"]
    w = AnchorSet(ANCHORS).cut("Rater_3", ITEMS[4], ITEMS, K)
    assert not m.check_anchor_homogeneity(anchors=w).suggested.pattern_free


def test_anchors_each_on_their_own_pattern_anchor_through_global_effects():
    """Every anchor on its own threshold pattern (same global effect): all kept, every pattern free, and a note says
    the zero point rests on their global effects alone."""
    rng = np.random.default_rng(5)
    pats = {a: ("thresholds", 1.2 * (lambda v: v - v.mean())(rng.standard_normal(K))) for a in ANCHORS}
    check = _sim(pats).check_anchor_homogeneity(anchors=ANCHORS)
    assert check.suggested.anchors == ANCHORS and set(check.suggested.pattern_free) == set(ANCHORS)
    assert any("rests on their global effects alone" in n for n in check.notes)


def test_cochran_test_stays_available():
    m = _sim()
    assert m.check_anchor_homogeneity(model="global", anchors=ANCHORS[:3], rater_homogeneity_test="cochran",
                                      no_of_samples=30, seed=1) is None
    assert m.anchor_homogeneity_test is not None
    with pytest.raises(TypeError, match="'cochran' test"):
        m.check_anchor_homogeneity(anchors=ANCHORS, no_of_samples=30)


def test_cochran_test_argument_checks():
    """Cochran: model= is required, an unknown argument is a TypeError, and screen-only arguments warn."""
    m = _sim()
    with pytest.raises(ValueError, match="needs model="):
        m.check_anchor_homogeneity(anchors=ANCHORS[:3], rater_homogeneity_test="cochran")
    for test in ("cochran", None):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            m.check_anchor_homogeneity(model="global", anchors=ANCHORS[:3], rater_homogeneity_test=test,
                                       candidate_models=["global"])
    with pytest.warns(UserWarning, match=r"ignores \['min_effect', 'delta'\]"):
        m.check_anchor_homogeneity(model="global", anchors=ANCHORS[:3], rater_homogeneity_test="cochran",
                                   min_effect=1.5, delta=0.2, no_of_samples=30, seed=1)


def test_rater_stats_mixed_with_anchors_after_another_model():
    """rater_stats_df(model='mixed', anchors=) after a different model's calibrate(): the selection's joint fit is
    used (it used to look for the 'pair-cpat' anchored attribute)."""
    m = _sim(n=300)
    m.calibrate(model="bivector", estimation="cp-full-rcl")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.rater_stats_df(model="mixed", anchors=ANCHORS)
    assert m.anchors_mixed == ANCHORS
    assert all(m.rater_models[a] == "global" for a in ANCHORS)
    assert list(m.rater_stats_mixed.index) == list(m.facet_names)


def test_rater_stats_mixed_with_an_anchor_set():
    """rater_stats_df(model='mixed', anchors=) after calibrate(model='mixed', anchors=) with an anchor set (cells
    cut) accepts the same anchor set (it used to fail iterating over the AnchorSet)."""
    from raschpy._anchors import AnchorSet

    m = _sim(n=300)
    aset = AnchorSet(ANCHORS).cut(ANCHORS[0], (m.responses.columns[0], "Threshold 1"), list(m.responses.columns),
                                  int(m.max_score))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(model="mixed", estimation="cp-full-rcl", anchors=aset,
                    rater_models={r: "global" for r in m.facet_names})
        m.rater_stats_df(model="mixed", anchors=aset)
    assert m.anchors_mixed == ANCHORS
    assert list(m.rater_stats_mixed.index) == list(m.facet_names)


# ---- global effect --------------------------------------------------------------------------------------------

def test_wald_tests_use_f_and_t_references_on_persons_minus_one():
    """The global-effect Wald test: Hotelling-type F on (df, n - df + 1), n = persons - 1; each anchor's z: t on
    persons - 1 df (delta=0: plain tests)."""
    from scipy.stats import f as fdist, t as tdist

    m = _sim(n=40)
    check = m.check_anchor_homogeneity(anchors=ANCHORS, delta=0)
    assert check.n_persons == 40
    W, df, p = check.overall.loc["global effect", ["Statistic", "df", "p"]].astype(float)
    n = 39
    assert abs(p - fdist.sf(W * (n - df + 1) / (df * n), df, n - df + 1)) < 1e-12
    sev = check.global_effects
    z = sev["Shift"].astype(float) / sev["SE"].astype(float)
    np.testing.assert_allclose(sev["p"].astype(float), 2 * tdist.sf(np.abs(z), n), rtol=1e-12)


def test_a_global_effect_outlier_is_suggested_for_dropping():
    m = _sim({"Rater_5": ("global", 0.7)})
    check = m.check_anchor_homogeneity(anchors=ANCHORS)
    assert check.overall.loc["global effect", "p"] < 0.001
    s = check.suggestions
    assert list(zip(s["rater"], s["Action"], s["Reason"])) == [("Rater_5", "drop", "global effect")]
    assert abs(s["Shift"].iloc[0] - 0.7) < 0.15
    assert abs(s["Pull on zero point"].iloc[0] - s["Shift"].iloc[0] / 5) < 1e-12


@pytest.mark.parametrize("estimation", ESTIMATIONS)
def test_a_pattern_does_not_pass_for_a_global_effect_difference(estimation):
    """Rater_2 differs in pattern only, Rater_5 in global effect only: Rater_5 is dropped, Rater_2's pattern freed."""
    m = _sim({**CENTRAL, "Rater_5": ("global", 0.7)})
    check = m.check_anchor_homogeneity(anchors=ANCHORS, estimation=estimation)
    s = check.suggestions
    assert set(zip(s["rater"], s["Action"], s["Reason"])) == {("Rater_2", "free pattern", "pattern"),
                                                             ("Rater_5", "drop", "global effect")}


def test_min_effect_applies_to_global_effect():
    m = _sim({"Rater_5": ("global", 0.15)}, n=2000)
    assert "Rater_5" in set(m.check_anchor_homogeneity(anchors=ANCHORS, delta=0).suggestions["rater"])
    assert m.check_anchor_homogeneity(anchors=ANCHORS, delta=0, min_effect=0.3).suggestions.empty


# ---- close-fit likelihood ratio ladder (test='LR', delta=) ---------------------------------------------------

@pytest.mark.parametrize("estimation", ["cp-full-rcl", "cml"])
def test_lr_ladder_recovers_the_representations(estimation):
    """Rater_3 differs on one item, Rater_4 on one threshold: the ladder finds them with a tolerance below their
    departures, and estimates the threshold departure's size (RMS over the cells, 0.38). Not with plain tests
    (delta=0): with every rater marking every person, the shift Rater_4 causes in the shared thresholds while it is
    on 'global' (about 0.16 logits at threshold 2) is significant, and the agreeing anchors are given threshold
    effects (a known limitation of v2.0.0)."""
    m = _sim(TWO_CELLS)
    want = ["global", "global", "items", "thresholds", "global", "global"]
    for delta in (0.1,):
        tab = m.per_rater_model_selection(anchors=ANCHORS, test="LR", delta=delta, delta_units="logits",
                                          estimation=estimation, progress=False)
        assert tab["Representation"].tolist() == want
    assert not any("items_centrality" in c or "thresholds_pseudo_halo" in c for c in tab.columns)
    assert abs(tab.loc["Rater_4", "Delta-hat thresholds>centrality"] - 0.377) < 0.08
    assert m.per_rater_model_selection_info["delta"] == 0.1


@pytest.mark.parametrize("estimation", ["cp-full-rcl", "cml"])
def test_lr_ladder_close_fit_p_at_delta_hat(estimation):
    """Delta-hat is the size at which the statistic's expected excess over its null mean equals the observed one
    (computed from the responses a departure of that size would give on average), so at delta = Delta-hat the
    close-fit p is about 0.5 (the statistic sits near its median)."""
    m = _sim(TWO_CELLS)
    kw = dict(anchors=ANCHORS, test="LR", delta_units="logits", estimation=estimation, progress=False)
    # Delta-hat from a run with a small tolerance (plain tests mis-select the agreeing anchors here; see above)
    dh = m.per_rater_model_selection(delta=0.1, **kw).loc["Rater_4", "Delta-hat thresholds>centrality"]
    assert dh > 0
    tab = m.per_rater_model_selection(delta=dh, **kw)
    assert 0.35 < tab.loc["Rater_4", "p thresholds>centrality"] < 0.65


def test_lr_ladder_ridges_reach_the_fits():
    """Ridge arguments are used by every candidate fit and recorded; a strong item ridge pulls Rater_3's item
    departure in, so its Delta-hat for items over the item stretch falls."""
    m = _sim(TWO_CELLS)
    kw = dict(anchors=ANCHORS, test="LR", delta_units="logits", progress=False)
    a = m.per_rater_model_selection(**kw).loc["Rater_3", "Delta-hat items>pseudo_halo"]
    assert m.per_rater_model_selection_info["ridges"] == {}
    b = m.per_rater_model_selection(item_ridge=50.0, **kw).loc["Rater_3", "Delta-hat items>pseudo_halo"]
    assert m.per_rater_model_selection_info["ridges"] == {"item_ridge": 50.0}
    assert b < a
    with pytest.raises(ValueError, match="ridge"):
        m.per_rater_model_selection(anchors=ANCHORS, estimation="pair-cpat", item_ridge=1.0)


def test_lr_ladder_tolerance_drops_small_departures():
    """A tolerance above both departures (RMS 0.28 and 0.38 logits, single deviations 0.8; the stretch rungs'
    tolerance, stretch_delta, above any stretch they could show) keeps every rater on 'global'; in category widths the same holds, and the
    unit is the median width of the starting fit."""
    m = _sim(TWO_CELLS)
    tab = m.per_rater_model_selection(anchors=ANCHORS, test="LR", delta=1.0, delta_max=2.0, delta_units="logits",
                                      stretch_delta=1.0, progress=False)
    assert set(tab["Representation"]) == {"global"}
    tab = m.per_rater_model_selection(anchors=ANCHORS, test="LR", delta=0.5, delta_max=1.0, delta_units="width",
                                      stretch_delta=1.0, progress=False)
    info = m.per_rater_model_selection_info
    assert set(tab["Representation"]) == {"global"} and info["median_category_width"] > 0
    assert info["delta_logits"] == pytest.approx(0.5 * info["median_category_width"])


def _stretch_rungs(m, estimation, **kw):
    """The two rungs pseudo_halo > global (adds the item stretch) and bivector > items (does not) for every
    non-anchor rater, the others on the starting representations."""
    return m._per_rater_selection_fitted(ANCHORS, "LR", ["global", "items", "bivector", "pseudo_halo"], True, 0.05,
                                         "one-pass", estimation, 1, None,
                                         _rungs=[("global", "pseudo_halo"), ("items", "bivector")], **kw)


@pytest.mark.parametrize("estimation", ["cp-full-rcl", "cml"])
def test_lr_ladder_stretch_delta_is_the_stretch_factor(estimation):
    """Rater_6 stretches the item locations by 30%: on the rung that adds the item stretch, Delta-hat is the
    stretch factor's change (about 0.3), stretch_delta = Delta-hat gives p about 0.5, and delta (widths) is not
    used there; the rung that adds no stretch keeps delta and ignores stretch_delta."""
    m = _sim({"Rater_6": ("pseudo_halo", (0.3, 1.3))})
    t0 = _stretch_rungs(m, estimation)
    dh = t0.loc["Rater_6", "Delta-hat pseudo_halo>global"]
    assert abs(dh - 0.3) < 0.1
    t1 = _stretch_rungs(m, estimation, stretch_delta=dh, delta=5.0)
    assert 0.35 < t1.loc["Rater_6", "p pseudo_halo>global"] < 0.65
    assert t1.loc["Rater_6", "Delta-hat pseudo_halo>global"] == pytest.approx(dh)
    t2 = _stretch_rungs(m, estimation, stretch_delta=dh)
    assert t2.loc["Rater_6", "p bivector>items"] == pytest.approx(t0.loc["Rater_6", "p bivector>items"])


def test_lr_ladder_stretch_delta_is_recorded():
    m = _sim()
    m.per_rater_model_selection(anchors=ANCHORS, test="LR", stretch_delta=0.1, progress=False)
    assert m.per_rater_model_selection_info["stretch_delta"] == 0.1


def test_lr_ladder_stretch_delta_per_side():
    """stretch_delta={'items': x, 'thresholds': y}: the item-stretch rung uses x only (the threshold tolerance
    leaves it as the plain test), and the per-side tolerances are recorded."""
    m = _sim({"Rater_6": ("pseudo_halo", (0.3, 1.3))})
    t0 = _stretch_rungs(m, "cp-full-rcl")
    dh = t0.loc["Rater_6", "Delta-hat pseudo_halo>global"]
    t_item = _stretch_rungs(m, "cp-full-rcl", stretch_delta={"items": dh})
    assert 0.35 < t_item.loc["Rater_6", "p pseudo_halo>global"] < 0.65
    t_thr = _stretch_rungs(m, "cp-full-rcl", stretch_delta={"thresholds": 0.5})
    assert t_thr.loc["Rater_6", "p pseudo_halo>global"] == pytest.approx(t0.loc["Rater_6", "p pseudo_halo>global"])
    m2 = _sim()
    m2.per_rater_model_selection(anchors=ANCHORS, test="LR", stretch_delta={"items": 0.05, "thresholds": 0.2},
                                 progress=False)
    info = m2.per_rater_model_selection_info
    assert (info["item_stretch_delta"], info["threshold_stretch_delta"]) == (0.05, 0.2)


def test_lr_ladder_argument_checks():
    m = _sim()
    with pytest.raises(ValueError, match="delta_measure must be"):
        m.per_rater_model_selection(anchors=ANCHORS, test="LR", delta_measure="mean")
    with pytest.raises(ValueError, match="delta_max must be >= 0"):
        m.per_rater_model_selection(anchors=ANCHORS, test="LR", delta_max=-0.1)
    with pytest.raises(ValueError, match="use min_effect"):
        m.per_rater_model_selection(anchors=ANCHORS, test="LR", delta_measure="either", estimation="pair-cpat")
    with pytest.raises(ValueError, match="use min_effect"):
        m.per_rater_model_selection(anchors=ANCHORS, test="LR", delta=0.1, estimation="pair-cpat")
    with pytest.raises(ValueError, match="delta_units"):
        m.per_rater_model_selection(anchors=ANCHORS, test="LR", delta=0.1, delta_units="sd")
    with pytest.raises(ValueError, match="stretch_delta must be >= 0"):
        m.per_rater_model_selection(anchors=ANCHORS, test="LR", stretch_delta=-0.1)
    with pytest.raises(ValueError, match="stretch_delta must be >= 0"):
        m.per_rater_model_selection(anchors=ANCHORS, test="LR", stretch_delta={"thresholds": -0.1})
    with pytest.raises(ValueError, match="unknown keys"):
        m.per_rater_model_selection(anchors=ANCHORS, test="LR", stretch_delta={"item": 0.1})
    with pytest.raises(ValueError, match="use min_effect"):
        m.per_rater_model_selection(anchors=ANCHORS, test="LR", stretch_delta={"items": 0.1}, estimation="pair-cpat")
    with pytest.raises(ValueError, match="use min_effect"):
        m.per_rater_model_selection(anchors=ANCHORS, test="LR", stretch_delta=0.1, estimation="pair-cpat")


# ---- likelihood ratio selection built up from 'global' (cp-full-rcl, cpat-rcl, cml) ----------------------------

def _sim_on_instrument(deviations):
    """As _sim, on the instrument of the all-global simulation with the same seed (so the generating thresholds are
    known for a stretch)."""
    base = {a: ("global", 0.0) for a in ANCHORS}
    s0 = MFRM_Sim(no_of_items=I, no_of_persons=800, no_of_facet_elements=6, max_score=K, model="mixed",
                  rater_models=dict(base, Rater_6=("global", 0.0)), seed=11)
    rm = dict(base, Rater_6=("global", 0.3))
    rm.update(deviations)
    s = MFRM_Sim(no_of_items=I, no_of_persons=800, no_of_facet_elements=6, max_score=K, model="mixed",
                 rater_models=rm, seed=11, manual_items=s0.native_items, manual_thresholds=s0.native_thresholds)
    tau = np.asarray(s0.native_thresholds, float)
    return rp.MFRM(s.responses), tau - tau.mean()


def test_lr_buildup_full_set_beside_stretch_is_reported_as_bivector():
    """Rater_6 has item effects and a threshold stretch (1.4): no offered representation has a full set on one side
    and a stretch on the other, so the build-up finds it with its internal representation and reports, fits and
    stores the free 'bivector'; the internal names never appear."""
    a = 0.8 * (np.eye(I)[4] - np.eye(I)[2])
    _, tc = _sim_on_instrument({})
    m, _ = _sim_on_instrument({"Rater_6": ("bivector", (a, 0.4 * tc))})
    tab = m.per_rater_model_selection(anchors=ANCHORS, test="LR", progress=False)
    assert tab["Representation"].tolist() == ["global"] * 5 + ["bivector"]
    assert m.rater_models["Rater_6"] == "bivector"
    hidden = {"items_centrality", "thresholds_pseudo_halo"}
    assert not hidden & set(tab["Representation"]) and not hidden & set(dict(m.rater_models).values())


def test_lr_buildup_finds_both_stretches():
    m, _ = _sim_on_instrument({"Rater_6": ("bistretch", (0.3, 1.3, 1.3))})
    tab = m.per_rater_model_selection(anchors=ANCHORS, test="LR", progress=False)
    assert tab["Representation"].tolist() == ["global"] * 5 + ["bistretch"]


# ---- the largest single deviation (delta_measure='largest' / 'either') and the AIC / BIC tolerances -------------------

@pytest.mark.parametrize("estimation", ["cp-full-rcl", "cml"])
def test_lr_both_flags_the_departing_item_and_threshold(estimation):
    """Rater_3 differs on Item_5 only, Rater_4 on threshold 2 only: with delta_measure='either' the same raters are
    found, the largest-deviation test names the departing item, and the decision p is the smaller of the two tests'
    p-values (each test at alpha)."""
    m = _sim(TWO_CELLS)
    tab = m.per_rater_model_selection(anchors=ANCHORS, test="LR", delta_measure="either", estimation=estimation,
                                      delta_units="logits", progress=False)
    assert tab["Representation"].tolist() == ["global", "global", "items", "thresholds", "global", "global"]
    row = tab.loc["Rater_3"]
    assert row["Flagged items>pseudo_halo"] == "Item_5"
    # Item_5 against the others' regression on the item locations (the smaller model has the item stretch): about
    # 0.6 logits of the 0.8, Item_5 being the highest item, where a stretch also reaches
    assert 0.4 < row["Max deviation items>pseudo_halo"] < 0.9
    assert row["p items>pseudo_halo"] == pytest.approx(
        min(row["p overall items>pseudo_halo"], row["p largest items>pseudo_halo"]))
    assert m.per_rater_model_selection_info["delta_measure"] == "either"


def test_lr_max_is_rms_on_stretch_rungs():
    """A rung that adds a stretch has one element, so the largest-deviation test does not apply: its p is the
    RMS test's, unchanged by delta_measure."""
    m = _sim(TWO_CELLS)
    a = m.per_rater_model_selection(anchors=ANCHORS, test="LR", progress=False)
    b = m.per_rater_model_selection(anchors=ANCHORS, test="LR", delta_measure="largest", progress=False)
    assert b.loc["Rater_1", "p pseudo_halo>global"] == pytest.approx(a.loc["Rater_1", "p pseudo_halo>global"])
    assert "p largest pseudo_halo>global" not in b.columns


@pytest.mark.parametrize("measure", ["overall", "largest", "either"])
def test_aic_tolerance_keeps_global(measure):
    """AIC's choices for the two departing raters go beyond 'global' with no tolerance, and stay on 'global' when the
    tolerances (1 logit, stretch 1) exceed their departures: the build-up refuses every step from 'global'."""
    m = _sim(TWO_CELLS)
    t0 = m.per_rater_model_selection(anchors=ANCHORS, test="AIC", delta=0.0, delta_max=0.0, stretch_delta=0.0, delta_measure="overall", progress=False)
    assert t0.loc["Rater_3", "Representation"] != "global"
    t1 = m.per_rater_model_selection(anchors=ANCHORS, test="AIC", delta=1.0, delta_max=1.0, delta_units="logits",
                                     stretch_delta=1.0, delta_measure=measure, progress=False)
    assert set(t1["Representation"]) == {"global"}
    kept = [c for c in t1.columns if c.startswith("Kept ") and c.endswith(">global")]
    assert kept and not t1.loc["Rater_3", kept].fillna(False).astype(bool).any()
    assert m.per_rater_model_selection_info["delta_max"] == 1.0


# ---- the tolerance defaults --------------------------------------------------------------------------------------

def test_tolerance_defaults_lr_and_aic():
    """With the proper fits, every test defaults to delta 0.1 and delta_max 0.2 median category widths, stretch_delta
    0.1 and 'either'; explicit values override."""
    m = _sim(TWO_CELLS)
    m.per_rater_model_selection(anchors=ANCHORS, test="LR", progress=False)
    info = m.per_rater_model_selection_info
    assert (info["delta"], info["delta_max"], info["delta_measure"]) == (0.1, 0.2, "either")
    m.per_rater_model_selection(anchors=ANCHORS, test="LR", delta=0.0, delta_measure="overall", progress=False)
    info = m.per_rater_model_selection_info
    assert (info["delta"], info["delta_measure"]) == (0.0, "overall")
    for test in ("AIC", "BIC"):
        m.per_rater_model_selection(anchors=ANCHORS, test=test, progress=False)
        info = m.per_rater_model_selection_info
        assert (info["delta"], info["delta_max"], info["delta_measure"]) == (0.1, 0.2, "either")


def test_lr_default_tolerances_fall_back_without_category_widths():
    """Dichotomous data have no category width: the default delta and delta_max fall back to the plain tests with a
    warning; a delta the user sets in widths still raises."""
    rng = np.random.default_rng(3)
    I_, R_, N_ = 6, 6, 500
    d = rng.uniform(-1, 1, I_)
    rows, idx = [], []
    for n, th in enumerate(rng.normal(0, 1.2, N_)):
        for r in rng.choice(R_, 2, replace=False):
            rows.append((rng.random(I_) < 1 / (1 + np.exp(-(th - d)))).astype(float))
            idx.append((f"Rater_{r + 1}", f"Person_{n + 1}"))
    X = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(idx), columns=[f"Item_{i + 1}" for i in range(I_)])
    m = rp.MFRM(X.sort_index(level=[0, 1], sort_remaining=False), max_score=1)
    with pytest.warns(UserWarning, match="no category width"):
        m.per_rater_model_selection(anchors=["Rater_1", "Rater_2"], test="LR", progress=False)
    info = m.per_rater_model_selection_info
    assert (info["delta"], info["delta_max"], info["delta_measure"]) == (0.0, 0.0, "overall")
    with pytest.raises(ValueError, match="delta_units='logits'"):
        m.per_rater_model_selection(anchors=["Rater_1", "Rater_2"], test="LR", delta=0.1, progress=False)


# ---- v2.0.0: global-effect-only sets, ties, report='full' -----------------------------------------------------

def test_a_non_anchor_can_decide_whether_cut_cells_are_identified():
    """Why per_rater_model_selection checks every facet element's candidates against the cuts (not only the
    anchors'): a non-anchor on 'global' fixes the instrument at the cut cells; on 'bivector' it does not."""
    cuts = {1: [("item", 1), ("threshold", 1)]}
    _rcl.MixedDesign(["matrix", "global", "global"], 4, 3, anchors=[0, 1], cuts=cuts)
    with pytest.raises(ValueError, match="no remaining anchor fixes"):
        _rcl.MixedDesign(["matrix", "global", "bivector"], 4, 3, anchors=[0, 1], cuts=cuts)


def test_global_effect_only_search_finds_tied_groups():
    """Two pairs of anchors at different global effects (-0.5 and +0.5 logits) and one between: the largest sets
    agreeing in global effect are the two pairs, a near tie (zero-point shifts of equal size)."""
    lv = {"Rater_1": -0.5, "Rater_2": -0.5, "Rater_3": 0.0, "Rater_4": 0.5, "Rater_5": 0.5}
    m = _sim({a: ("global", v) for a, v in lv.items()})
    with pytest.warns(UserWarning, match="tie as the largest"):
        check = m.check_anchor_homogeneity(anchors=ANCHORS)
    gc = check.global_core
    assert sorted(map(sorted, gc["sets"])) == [["Rater_1", "Rater_2"], ["Rater_4", "Rater_5"]]
    assert gc["near_tie"] and gc["exhaustive"]
    assert abs(abs(gc["zero_shift"][0]) - abs(gc["zero_shift"][1])) < 0.2
    assert len(check.ties) == len(check.pattern_ties) == 2
    assert check.apply_suggested(choice=1).anchors.anchors == check.ties[1].anchors


def test_apply_suggested_choice():
    m = _sim(CENTRAL)
    check = m.check_anchor_homogeneity(anchors=ANCHORS)
    assert check.apply_suggested(choice=0).anchors.anchors == check.suggested.anchors
    with pytest.raises(ValueError, match="choice must be"):
        check.apply_suggested(choice=len(check.ties))


def test_report_full_sets_profile_and_subsets():
    m = _sim(CENTRAL, n=400)
    final = m.check_anchor_homogeneity(anchors=ANCHORS)
    assert final.profile is None and final.subsets is None
    full = m.check_anchor_homogeneity(anchors=ANCHORS, report="full")
    assert list(full.profile.index) == [0.1, 0.15, 0.2, 0.3]
    assert full.profile.loc[0.1, "Anchors"] == list(final.suggested.anchors)
    assert full.profile.loc[0.1, "Pattern free"] == list(final.suggested.pattern_free)
    assert {"Subset", "Agrees", "Shift", "SE", "p", "Flagged"} <= set(full.subsets.columns)
    with pytest.raises(ValueError, match="report must be"):
        m.check_anchor_homogeneity(anchors=ANCHORS, report="all")


def test_calibrate_anchor_with_an_anchor_set():
    """Pattern-free anchors warn with a representation that anchors patterns; cut cells are for 'mixed' fits only."""
    m = _sim()
    m.calibrate(model="items", estimation="cp-full-rcl")
    with pytest.warns(UserWarning, match="global effects only"):
        m.calibrate_anchor("items", AnchorSet(ANCHORS, pattern_free=["Rater_1"]))
    m.calibrate(model="global", estimation="cp-full-rcl")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.calibrate_anchor("global", AnchorSet(ANCHORS, pattern_free=["Rater_1"]))
    assert not any("global effects only" in str(x.message) for x in w)
    with pytest.raises(ValueError, match="cut cells"):
        m.calibrate_anchor("items", AnchorSet(ANCHORS, {"Rater_1": [("item", ITEMS[0])]}))


def test_mixed_fits_with_pattern_free_anchors():
    """calibrate(model='mixed') warns when a pattern-free anchor is on 'global'; per_rater_model_selection selects a
    pattern-free anchor's representation even with force_global_anchors."""
    m = _sim(CENTRAL)
    aset = AnchorSet(ANCHORS, pattern_free=["Rater_2"])
    with pytest.warns(UserWarning, match="frees the patterns"):
        m.calibrate(model="mixed", estimation="cp-full-rcl", anchors=aset,
                    rater_models={r: "global" for r in m.facet_names})
    t = m.per_rater_model_selection(anchors=aset, force_global_anchors=True, progress=False)
    assert t.loc["Rater_2", "Representation"] != "global"
    assert set(t.loc[[a for a in ANCHORS if a != "Rater_2"], "Representation"]) == {"global"}
