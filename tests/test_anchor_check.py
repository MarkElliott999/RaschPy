"""
Anchor sets with cut cells (calibrate(model='mixed', anchors=), per_rater_model_selection(anchors=)) and
check_anchor_homogeneity()'s check on joint fits: overall test, screen of each anchor against the rest, Wald loop over
cells, refinement, and the check object's actions.
"""

import warnings

import numpy as np
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
    sev = m.facet_effects_mixed_global
    assert sev[ANCHORS].abs().max() < 0.1 and abs(sev["Rater_6"] - 0.3) < 0.1
    assert m.anchor_set_mixed is aset


def test_cut_cells_need_a_joint_fit_and_global_anchors():
    m = _sim()
    aset = AnchorSet(ANCHORS).cut("Rater_3", "Item_5", ITEMS, K)
    with pytest.raises(ValueError, match="needs them on 'global'"):
        m.calibrate(model="mixed", rater_models={f: ("items" if f == "Rater_3" else "global")
                                                 for f in m.facet_names}, anchors=aset)
    with pytest.raises(ValueError, match="estimation="):
        m.calibrate(model="mixed", rater_models={f: "global" for f in m.facet_names}, anchors=aset,
                    estimation="pair-cpat")


@pytest.mark.parametrize("estimation", ESTIMATIONS)
def test_per_rater_selection_holds_anchors_on_global(estimation):
    m = _sim(TWO_CELLS)
    m.per_rater_model_selection(anchors=ANCHORS, estimation=estimation)
    assert (m.rater_models[ANCHORS] == "global").all()
    assert m.per_rater_model_selection_table.loc[ANCHORS, "Note"].notna().all()


# ---- the check -----------------------------------------------------------------------------------------------

def test_corrections():
    p = {"a": 0.001, "b": 0.02, "c": 0.03, "d": 0.5}
    assert corrected_reject(p, 0.05, None) == {"a": True, "b": True, "c": True, "d": False}
    # Holm: 0.02 > 0.05 / 3, so a only
    assert corrected_reject(p, 0.05, "holm") == {"a": True, "b": False, "c": False, "d": False}
    # BH: 0.03 <= 3 * 0.05 / 4, so a, b and c
    assert corrected_reject(p, 0.05, "bh") == {"a": True, "b": True, "c": True, "d": False}


def test_check_finds_the_anchors_and_cells():
    m = _sim(TWO_CELLS)
    check = m.check_anchor_homogeneity(anchors=ANCHORS)
    assert m.anchor_check is check
    assert check.overall_p < 0.05
    s = check.suggestions
    assert set(zip(s["rater"], s["Action"], s["Cell"])) == {("Rater_3", "cut", "Item_5"),
                                                         ("Rater_4", "cut", "Threshold 2")}
    assert check.suggested.cuts == {"Rater_4": [("threshold", "Threshold 2")], "Rater_3": [("item", "Item_5")]}
    assert check.anchors.cuts == {}  # nothing applied by the check itself
    again = check.apply_suggested()
    assert set(again.suggestions["Action"]) == {"keep cut"}
    assert not any(t["Flagged"].any() for _, t in again.screen)
    assert "changed: cut Rater_4 at Threshold 2" in repr(again)


@pytest.mark.parametrize("estimation", ESTIMATIONS)
def test_check_on_agreeing_anchors_suggests_nothing(estimation):
    check = _sim(seed=3).check_anchor_homogeneity(anchors=ANCHORS, estimation=estimation)
    assert check.estimation == estimation
    assert check.suggestions.empty
    assert not any(t["Flagged"].any() for _, t in check.screen)


def test_actions_return_new_checks():
    m = _sim(TWO_CELLS)
    check = m.check_anchor_homogeneity(anchors=ANCHORS)
    c2 = check.cut("Rater_3", "Item_5")
    assert c2.anchors.cuts == {"Rater_3": [("item", "Item_5")]} and check.anchors.cuts == {}
    assert c2.previous is check and m.anchor_check is c2
    c3 = c2.drop("Rater_4")
    assert "Rater_4" not in c3.anchors.anchors
    assert c3.restore("Rater_4").anchors.anchors[-1] == "Rater_4"


def test_action_drop_and_min_effect():
    m = _sim(TWO_CELLS)
    dropped = m.check_anchor_homogeneity(anchors=ANCHORS, action="drop")
    assert set(dropped.anchors.anchors) == {"Rater_1", "Rater_2", "Rater_5"}
    # a pattern drop lists the cells behind it, with their estimates
    first = dropped.previous.suggestions
    assert set(first["Action"]) == {"drop"} and set(first["Reason"]) == {"pattern"}
    assert set(first["Cell"]) == {"Item_5", "Threshold 2"} and first["Shift"].notna().all()
    big = m.check_anchor_homogeneity(anchors=ANCHORS, min_effect=2.0)
    assert big.suggestions.empty or "cut" not in set(big.suggestions["Action"])


def test_two_anchors_that_disagree_cannot_be_told_apart():
    m = _sim({"Rater_1": ("items", 1.0 * np.eye(I)[2])})
    check = m.check_anchor_homogeneity(anchors=["Rater_1", "Rater_2"])
    assert any("can't say which one is wrong" in n for n in check.notes)
    assert check.suggestions.empty


def test_an_anchor_alone_at_a_cell_is_not_screened_there():
    """Every other anchor cut at a cell: the remaining one alone defines the zero point there, so it is not screened
    (it used to fail with an undefined zero point)."""
    m = _sim(TWO_CELLS)
    cell = (ITEMS[4], "Threshold 2")
    w = AnchorSet(ANCHORS[:3]).cut("Rater_1", cell, ITEMS, K).cut("Rater_2", cell, ITEMS, K)
    check = m.check_anchor_homogeneity(anchors=w)
    assert any("Rater_3: the only anchor not cut" in n for n in check.notes)


def test_cuts_leave_two_other_anchors_at_each_cell():
    """Three of five anchors shifted at the same cell: each cut leaves at least two other anchors uncut there, so
    every later screen keeps a zero point; the rest are dropped instead of cut."""
    shift = {f"Rater_{r}": ("matrix", 1.5 * np.outer(np.eye(I)[2], np.eye(K)[1])) for r in (1, 2, 3)}
    m = _sim(shift, n=600)
    check = m.check_anchor_homogeneity(anchors=ANCHORS)
    s = check.suggested
    held = np.zeros((I, K), int)
    for a in s.anchors:
        free = np.ones((I, K), bool)
        for kind, lab in s.cuts.get(a, []):
            free[check._cells(kind, lab)] = False
        held += free
    assert held.min() >= 1


def test_anchors_with_no_agreeing_core_get_no_suggestions():
    """Every anchor on its own threshold pattern: every anchor is flagged, so the check says the set has no agreeing
    core instead of dropping anchors until min_anchors stops it."""
    rng = np.random.default_rng(5)
    pats = {a: ("thresholds", 1.2 * (lambda v: v - v.mean())(rng.standard_normal(K))) for a in ANCHORS}
    m = _sim(pats)
    check = m.check_anchor_homogeneity(anchors=ANCHORS)
    assert any("no majority core" in n for n in check.notes)
    assert check.suggestions.empty
    assert check.suggested.anchors == ANCHORS


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
    with pytest.warns(UserWarning, match=r"ignores \['min_effect', 'max_share'\]"):
        m.check_anchor_homogeneity(model="global", anchors=ANCHORS[:3], rater_homogeneity_test="cochran",
                                   min_effect=1.5, max_share=0.4, no_of_samples=30, seed=1)


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


# ---- global effect --------------------------------------------------------------------------------------------

def test_wald_tests_use_f_and_t_references_on_persons_minus_one():
    """The global-effect Wald test: Hotelling-type F on (df, n - df + 1), n = persons - 1; each anchor's z: t on
    persons - 1 df."""
    from scipy.stats import f as fdist, t as tdist

    m = _sim(n=40)
    check = m.check_anchor_homogeneity(anchors=ANCHORS)
    assert check.n_persons == 40
    W, df, p = check.overall.loc["global effect", ["Statistic", "df", "p"]].astype(float)
    n = 39
    assert abs(p - fdist.sf(W * (n - df + 1) / (df * n), df, n - df + 1)) < 1e-12
    sev = [t for k, t in check.screen if k == "global effect"][0]
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
def test_a_shifted_cell_does_not_pass_for_a_global_effect_difference(estimation):
    """Pattern is resolved before global effect: Rater_3 and Rater_4 get cuts, only Rater_5 is dropped (for global
    effect)."""
    m = _sim({**TWO_CELLS, "Rater_5": ("global", 0.7)})
    s = m.check_anchor_homogeneity(anchors=ANCHORS, estimation=estimation).suggestions
    assert set(zip(s["rater"], s["Action"], s["Reason"], s["Cell"].map(str))) == {
        ("Rater_3", "cut", "pattern", "Item_5"), ("Rater_4", "cut", "pattern", "Threshold 2"),
        ("Rater_5", "drop", "global effect", "None")}


def test_min_effect_applies_to_global_effect():
    m = _sim({"Rater_5": ("global", 0.15)}, n=2000)
    assert "Rater_5" in set(m.check_anchor_homogeneity(anchors=ANCHORS).suggestions["rater"])
    assert m.check_anchor_homogeneity(anchors=ANCHORS, min_effect=0.3).suggestions.empty


def test_split_alpha_sets_the_screens_level():
    """split_alpha=True flags at alpha / 2: an anchor whose global effect p lies between alpha / 2 and alpha is flagged
    only without the split."""
    m = _sim({"Rater_5": ("global", 0.07)}, n=800, seed=5)
    split = m.check_anchor_homogeneity(anchors=ANCHORS)
    each = m.check_anchor_homogeneity(anchors=ANCHORS, split_alpha=False)
    assert split.settings["split_alpha"] and not each.settings["split_alpha"]
    sev = [t for k, t in split.screen if k == "global effect"][0]
    between = sev[(sev["p"] > 0.025) & (sev["p"] <= 0.05)]
    if len(between):
        assert not sev.loc[between.index, "Flagged"].any()
    assert len(each.suggestions) >= len(split.suggestions)

