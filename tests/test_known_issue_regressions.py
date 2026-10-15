"""
Regression tests for specific known issues that already bit RaschPy in
production once — see RaschPy_remediation_plan.docx item 2. Each test
guards one concrete, previously-real failure mode, so it can't silently
regress the way it already has at least once:

- #19/#32: MFRM row-order corruption (fixed, then silently reverted by
  an unrelated refactor commit and shipped in a released version,
  before being caught and re-fixed).
- #29: mfrm.py called scipy.stats.chi2/norm in several places without
  ever importing them -- those call paths raised a bare NameError the
  moment they actually ran, and nothing in the existing suite exercised
  them, so this had shipped undetected.
- #11: MFRM ICC/CRC/threshold-CC observed-overlay alignment in the
  non-anchor case -- investigated as a suspected regression, found NOT
  to be one (the curve-averaging mechanism was intact); guarded here so
  a real future regression in that mechanism is actually caught.
- #27/#36: model='mixed' support in fit_statistics()/rater_stats_df().
  #27 (crash) is fixed; #36 (rater_stats_df's table is missing its
  per-rater columns) is not, so that half is captured as a strict
  xfail that will loudly demand attention once it's fixed.

Uses raschpy's own simulation classes with an explicit seed= for
reproducibility -- see raschpy_architecture notes: np.random.seed()
has no effect on *_Sim output, only each sim's own seed= argument does.
"""

import warnings

import pandas as pd
import pytest

import raschpy as rp
from raschpy.simulation import MFRM_Sim_Global


def _make_mfrm(no_of_items=6, no_of_persons=250, no_of_raters=6,
                max_score=3, facet_range=2.0, seed=42):
    sim = MFRM_Sim_Global(
        no_of_items=no_of_items,
        no_of_persons=no_of_persons,
        no_of_facet_elements=no_of_raters,
        max_score=max_score,
        facet_range=facet_range,
        seed=seed,
    )
    return sim, rp.MFRM(sim.responses, max_score=max_score)


# ---------------------------------------------------------------------------
# #19 / #32 -- MFRM row-order corruption
# ---------------------------------------------------------------------------

def test_shuffled_rater_block_does_not_collapse_severity_spread():
    """
    Guards known-issues #19/#32.

    item_diffs() / _estimate_raters_*() read self.responses.values via a
    raw positional reshape(no_of_facet_elements, no_of_persons, -1) --
    only correct if every facet_element's block is laid out in identical
    person order. MFRM.__init__ must canonicalise this itself
    (reindex each facet_element's block to self.person_names) since
    nothing else guarantees it; a caller handing in a boolean-filtered,
    resampled, or otherwise reordered subset would otherwise silently
    collapse estimated rater severities toward zero while item estimates
    stayed fine (item_diffs never needs cross-block alignment, so it was
    accidentally immune -- which is exactly why this bug went unnoticed
    for as long as it did).

    This fix already shipped once, was then silently reverted by an
    unrelated large refactor commit, and shipped that way in a released
    version before being caught a second time -- so this test exists to
    make a third occurrence structurally impossible to miss.
    """
    sim, _ = _make_mfrm()
    responses = sim.responses

    # Reorder one rater's rows into a different person order than every
    # other rater -- the exact shape of input the canonical-grid fix
    # guards against (e.g. what a boolean-filtered or resampled subset
    # could look like before MFRM.__init__ gets a chance to fix it up).
    r0 = sim.rater_names[0]
    shuffled_block = responses.loc[r0].sample(frac=1, random_state=1)
    other_blocks = {r: responses.xs(r) for r in sim.rater_names[1:]}
    reordered = pd.concat({r0: shuffled_block, **other_blocks})

    m = rp.MFRM(reordered, max_score=3)
    m.calibrate(model="global")

    true_spread = sim.facet_effects.max() - sim.facet_effects.min()
    est_spread = m.facet_effects_global.max() - m.facet_effects_global.min()

    # Pre-fix symptom (documented in known-issues #19): a true spread of
    # ~1-2 logits collapsed to an estimated ~0.1-0.2 logits. Recovery
    # need not be exact -- just nowhere near that collapse.
    assert est_spread > 0.5 * true_spread, (
        f"rater severity spread collapsed under a reordered rater block: "
        f"true={true_spread:.3f}, estimated={est_spread:.3f} -- this is "
        f"the #19/#32 row-order corruption symptom"
    )


# ---------------------------------------------------------------------------
# #29 -- missing scipy.stats imports (chi2 / norm)
# ---------------------------------------------------------------------------

def test_check_anchor_homogeneity_runs_without_nameerror():
    """
    Guards known-issues #29.

    check_anchor_homogeneity()'s 'cochran' path calls chi2.sf/norm.cdf,
    which were never imported anywhere in mfrm.py -- this call path
    raised a bare NameError the instant it actually ran, undetected
    because nothing in the pre-existing suite exercised it (confirmed
    at the time: grepping tests/*.py for check_anchor_homogeneity
    returned nothing).
    """
    sim, m = _make_mfrm(no_of_raters=6, seed=11)
    anchors = sim.rater_names[:3]

    m.check_anchor_homogeneity(
        model="global", anchors=anchors, no_of_samples=50, seed=1,
    )

    # anchor_homogeneity_test is the single Cochran's Q summary (Q, df, p,
    # Flagged, ...) for this anchor set -- checking it holds real,
    # finite values (not NaN, and no exception on the way here) is what
    # actually demonstrates chi2.sf/norm.cdf ran successfully.
    summary = m.anchor_homogeneity_test
    assert summary is not None
    for field in ("Q", "df", "p", "Flagged"):
        assert field in summary.index
    assert pd.notna(summary["Q"])
    assert pd.notna(summary["p"])

    # anchor_homogeneity_per_rater is the per-rater breakdown (sequential
    # exclusion), one row per originally-supplied anchor rater.
    assert len(m.anchor_homogeneity_per_rater) == len(anchors)


def test_per_rater_model_selection_lr_runs_without_nameerror():
    """
    Guards known-issues #29's other affected call path:
    per_rater_model_selection(test='LR') also calls chi2.sf and hit the
    same missing-import NameError.
    """
    sim, m = _make_mfrm(no_of_raters=5, seed=12)

    m.per_rater_model_selection(test="LR")

    assert m.rater_models is not None
    assert set(m.rater_models.index) == set(sim.rater_names)


# ---------------------------------------------------------------------------
# check_anchor_homogeneity -- bistretch per-component breakdown
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("test", ["cochran", "robust"])
def test_check_anchor_homogeneity_bistretch_per_component(test):
    """
    check_anchor_homogeneity(model='bistretch') passed validation and ran
    the omnibus test, but its component list had no 'bistretch' case and
    fell through to an empty list -- so anchor_homogeneity_per_component
    was silently None even with per_component=True. bistretch's full
    vector has the same (rater, item) x threshold layout as bivector, so
    it should get the same one-row-per-(item, threshold)-cell breakdown.
    """
    sim, m = _make_mfrm(no_of_raters=5, seed=14)
    anchors = sim.rater_names[:4]

    m.check_anchor_homogeneity(
        model="bistretch", anchors=anchors, rater_homogeneity_test=test,
        per_component=True, no_of_samples=50, seed=1,
    )

    comp = m.anchor_homogeneity_per_component
    assert comp is not None
    assert len(comp) == len(m.item_names) * m.max_score


# ---------------------------------------------------------------------------
# #11 -- ICC/CRC/threshold-CC curve-averaging in the non-anchor case
# ---------------------------------------------------------------------------

def test_mean_facet_effects_averages_across_real_raters():
    """
    Guards known-issues #11.

    Investigated as a suspected regression (curves drawn against a
    dummy zero-severity rater instead of the mean of real raters) and
    found NOT to be one: icc/crcs/threshold_ccs's facet_element=None
    case draws the theoretical curve via _mean_facet_effects(), which
    sets the requested key to the mean across every other real rater --
    never a dummy zero severity. This test pins that mechanism down
    directly so a *real* future regression here is actually caught,
    unlike the first (false-alarm) investigation.
    """
    sim, m = _make_mfrm(no_of_raters=5, seed=13)
    m.calibrate(model="global")

    placeholder = sim.rater_names[0]
    result = m._mean_facet_effects("global", m.facet_effects_global, placeholder)

    assert result[placeholder] == pytest.approx(m.facet_effects_global.mean())


def test_icc_crcs_threshold_ccs_no_facet_element_do_not_crash():
    """
    Companion smoke check for #11: the actual plotting entry points run
    cleanly with facet_element=None (the non-anchor, averaged-curve
    case), across all three curve types.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sim, m = _make_mfrm(no_of_raters=4, seed=14)
    m.calibrate(model="global")
    m.fit_statistics(model="global")
    item0 = sim.item_names[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.icc(item=item0, model="global", facet_element=None, obs=True)
        plt.close("all")
        m.crcs(item=item0, model="global", facet_element=None)
        plt.close("all")
        m.threshold_ccs(item=item0, model="global", facet_element=None)
        plt.close("all")


# ---------------------------------------------------------------------------
# #27 / #36 -- model='mixed' support
# ---------------------------------------------------------------------------

def test_fit_statistics_mixed_does_not_crash():
    """
    Guards known-issues #27 (fixed): fit_statistics(model='mixed') used
    to crash immediately -- 'mixed' wasn't wired into std_errors()'s
    bootstrap loop, _cat_probs_mfrm(), or _diff_df_dict(). Also
    confirms the lazy-calibration path (auto-running
    per_rater_model_selection() when no rater_models assignment exists
    yet) still works, matching every other model's lazy-calibrate
    convention.
    """
    _, m = _make_mfrm(no_of_raters=5, seed=15)
    m.fit_statistics(model="mixed")
    assert hasattr(m, "rater_models")


def test_rater_stats_df_mixed_has_per_rater_columns():
    """
    Guards known-issues #36 (fixed, remediation-plan item 4, 2026-09-16):
    rater_stats_df(model='mixed') used to run cleanly but silently return
    only the 'Overall statistics' fit block -- no per-rater Estimate/SE
    columns for any item/threshold, and no indication of which
    parameterisation each rater was assigned. Fixed by adding a 'mixed'
    branch mirroring matrix's marginal=False (full cell table) path,
    plus a 'Rater model' column from self.rater_models. This was
    previously a strict xfail (by design, so it would start failing the
    moment the fix landed, forcing this exact update) -- now a real
    assertion.
    """
    sim, m = _make_mfrm(no_of_raters=5, seed=16)
    m.rater_stats_df(model="mixed")

    groups = set(m.rater_stats_mixed.columns.get_level_values(0))
    assert groups != {"Overall statistics"}, (
        "rater_stats_mixed only has the Overall statistics block -- "
        "per-rater Estimate/SE/Model columns are missing again (#36)"
    )
    assert "Rater model" in groups
    assert m.rater_stats_mixed.shape[0] == len(sim.rater_names)
    # Estimate/SE columns should be numeric, not tainted to object dtype
    # by the string-valued Rater model column sharing the same table.
    item0 = sim.item_names[0]
    assert m.rater_stats_mixed[(f"{item0}, Threshold 1", "Estimate")].dtype == float


@pytest.mark.parametrize("kind", ["slm", "rsm", "pcm", "mfrm"])
def test_warm_extreme_scores_finite_and_adjustment_free(kind):
    """
    Guards the one-step Warm correction bug (fixed 2026-10-02): person
    locations were ML + J/(2I^2) evaluated at the ML estimate. For extreme
    scores I is near zero there, so the correction overshot -- a smaller
    ext_score_adjustment pulled extreme persons towards the middle and then
    flipped their sign (e.g. RSM score 0 -> +40.9 logits at 0.01). Warm
    estimates are now the root of the weighted likelihood equation, which
    is finite for extreme scores, so they must not depend on
    ext_score_adjustment, must be ordered (0 below max), and person() must
    agree with score_lookup().
    """
    from raschpy.simulation import SLM_Sim, RSM_Sim, PCM_Sim, MFRM_Sim_Global
    import numpy as np
    kw = {}
    if kind == "slm":
        sim = SLM_Sim(no_of_items=8, no_of_persons=300, seed=21)
        d = sim.responses.copy(); top = 8
        m_cls = lambda d: rp.SLM(d)
    elif kind == "rsm":
        sim = RSM_Sim(no_of_items=6, no_of_persons=300, max_score=3, seed=22)
        d = sim.responses.copy(); top = 18
        m_cls = lambda d: rp.RSM(d, max_score=3)
    elif kind == "pcm":
        sim = PCM_Sim(no_of_items=6, no_of_persons=300, max_score_vector=[3] * 6, seed=23)
        d = sim.responses.copy(); top = 18
        m_cls = lambda d: rp.PCM(d, max_score_vector=[3] * 6)
    else:
        sim = MFRM_Sim_Global(no_of_items=4, no_of_persons=300, no_of_facet_elements=3,
                              max_score=3, seed=24)
        d = sim.responses.copy(); top = 36
        m_cls = lambda d: rp.MFRM(d, max_score=3)
        kw = {"model": "global"}

    if kind == "mfrm":
        persons = list(dict.fromkeys(d.index.get_level_values(1)))[:2]
        d.loc[pd.IndexSlice[:, persons[0]], :] = 0
        d.loc[pd.IndexSlice[:, persons[1]], :] = 3
    else:
        persons = list(d.index[:2])
        d.loc[persons[0]] = 0
        d.loc[persons[1]] = 1 if kind == "slm" else 3
    m = m_cls(d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(**kw)
        est = {a: m.person(persons, ext_score_adjustment=a, **kw) for a in (0.5, 0.1, 0.01)}
        lookup = (m.score_lookup(0, **kw), m.score_lookup(top, **kw))

    for a in (0.1, 0.01):
        assert np.allclose(est[a].values, est[0.5].values, atol=1e-4), (
            f"{kind}: Warm estimates of extreme scores depend on ext_score_adjustment again"
        )
    lo, hi = est[0.5].values
    assert np.isfinite([lo, hi]).all() and lo < -1 and hi > 1, f"{kind}: extreme estimates {lo}, {hi}"
    assert np.allclose([lo, hi], lookup, atol=1e-3), f"{kind}: person() {lo, hi} vs score_lookup() {lookup}"


def _rsm_sim(**kw):
    from raschpy.simulation import RSM_Sim
    return RSM_Sim(**kw)


def test_rsm_threshold_method_options():
    """
    RSM threshold_method (added 2026-10-02, estimation='pair-cpat'): 'cpat-cml' is the default,
    'cpat-info-weighted' reproduces the pre-v1.4.0 estimator, anything else
    raises, and both recover well-informed thresholds closely.
    """
    import numpy as np
    sim = _rsm_sim(no_of_items=10, no_of_persons=1500, max_score=4, seed=31)
    true = np.array(sim.thresholds, dtype=float); true -= true.mean()
    m = rp.RSM(sim.responses, max_score=4)
    m.calibrate(estimation="pair-cpat")
    assert m.threshold_method == "cpat-cml"
    cml = m.thresholds.values.copy()
    m.calibrate(threshold_method="cpat-info-weighted")
    info = m.thresholds.values.copy()
    for est in (cml, info):
        assert np.sqrt(np.mean((est - true) ** 2)) < 0.3
    assert not np.allclose(cml, info)
    with pytest.raises(ValueError):
        m.calibrate(threshold_method="cml")


def test_rsm_empty_middle_category_warns_and_stays_finite():
    """
    An empty middle category leaves the distances touching it unidentified.
    threshold_set() must warn (describing the problem and options), return
    finite thresholds, and keep the categories away from the gap close to
    the full-data estimates (targeted smoothing, not a flat constant that
    distorts the whole scale).
    """
    import numpy as np
    sim = _rsm_sim(no_of_items=10, no_of_persons=1500, max_score=5, seed=32)
    full = rp.RSM(sim.responses, max_score=5); full.calibrate()
    d = sim.responses.mask(sim.responses == 3, 2)
    m = rp.RSM(d, max_score=5, validate=False)
    with pytest.warns(UserWarning, match="Score category 3 has no responses"):
        m.calibrate()
    th = m.thresholds.values
    assert np.isfinite(th).all()
    # distance 1-2 does not touch category 3 (touches 0,1,2)
    assert abs((th[1] - th[0]) - (full.thresholds.values[1] - full.thresholds.values[0])) < 0.5


def test_rsm_empty_category_collapsed_width_fallback():
    """max_score=3 with category 1 empty: every distance touches it, so the
    smoothing is set from a collapsed fit. Must stay finite and ordered at
    the merged boundary's neighbours (no runaway disordering)."""
    import numpy as np
    sim = _rsm_sim(no_of_items=10, no_of_persons=1000, max_score=3, seed=33)
    d = sim.responses.mask(sim.responses == 1, 0)
    m = rp.RSM(d, max_score=3, validate=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate()
    assert np.isfinite(m.thresholds.values).all()
    assert np.ptp(m.thresholds.values) < 15


def test_rsm_no_information_placeholder_and_raise():
    """A distance with no comparison patterns at all gets a placeholder and
    a warning by default, or raises with no_info='raise'."""
    import numpy as np
    sim = _rsm_sim(no_of_items=3, no_of_persons=50, max_score=5, missing=0.5, seed=5)
    m = rp.RSM(sim.responses, max_score=5, validate=False)
    with pytest.warns(UserWarning, match="no information on the distance"):
        m.calibrate()
    assert np.isfinite(m.thresholds.values).all()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="could not be estimated"):
            rp.RSM(sim.responses, max_score=5, validate=False).calibrate(no_info="raise")


def test_rsm_thin_distance_warning():
    """A threshold distance observed fewer than 20 times triggers the
    poorly-informed warning (this path once raised a NameError while
    building the warning text) and still returns finite thresholds."""
    import numpy as np
    sim = _rsm_sim(no_of_items=6, no_of_persons=60, max_score=5, missing=0.3, seed=40)
    m = rp.RSM(sim.responses, max_score=5, validate=False)
    with pytest.warns(UserWarning, match="poorly informed"):
        m.calibrate()
    assert np.isfinite(m.thresholds.values).all()


def test_mfrm_no_information_threshold_placeholder():
    """After the CPAT same-item fix, an MFRM threshold distance with no
    comparison patterns raised a ValueError (7/50 benchmark runs at K=7,
    N=20). It now gets a placeholder with a warning, as in RSM."""
    import numpy as np
    sim = MFRM_Sim_Global(no_of_items=4, no_of_persons=20, no_of_facet_elements=5,
                          max_score=7, seed=1)
    m = rp.MFRM(sim.responses, max_score=7, validate=False)
    with pytest.warns(UserWarning, match="no information on the distance"):
        m.calibrate(model="global")
    assert np.isfinite(m.thresholds.values).all()


def test_rsm_sandwich_standard_errors():
    """RSM.std_errors() defaults to analytic sandwich SEs (stacked item and
    threshold estimating equations, persons as units), which should agree
    with the bootstrap on well-informed data, and falls back to the
    bootstrap when a category is empty."""
    import numpy as np
    sim = _rsm_sim(no_of_items=12, no_of_persons=800, max_score=4, missing=0.2, seed=8)
    m = rp.RSM(sim.responses, max_score=4); m.calibrate(); m.std_errors(interval=0.95)
    assert m.se_method == "sandwich" and m.item_bootstrap is None
    assert np.isfinite(m.item_se).all() and np.isfinite(m.threshold_se).all()
    assert np.all(m.threshold_low < m.thresholds.values) and np.all(m.threshold_high > m.thresholds.values)
    b = rp.RSM(sim.responses, max_score=4); b.calibrate(); b.std_errors(se_method="bootstrap", no_of_samples=300, seed=1)
    assert b.se_method == "bootstrap"
    assert 0.8 < np.median(b.item_se / m.item_se) < 1.25
    assert np.all((b.threshold_se / m.threshold_se > 0.75) & (b.threshold_se / m.threshold_se < 1.33))
    m.category_stats_df()   # disorder probability without bootstrap draws
    d = sim.responses.mask(sim.responses == 2, 1)
    e = rp.RSM(d, max_score=4, validate=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e.calibrate()
    with pytest.warns(UserWarning, match="bootstrap rather than analytically"):
        e.std_errors(no_of_samples=30, seed=1)
    assert e.se_method == "bootstrap"


def test_slm_pcm_sandwich_standard_errors():
    """SLM and PCM std_errors() default to analytic sandwich SEs (Bradley-
    Terry estimating equations of PAIR, persons as units), agreeing with the
    bootstrap on well-informed data; PCM falls back to the bootstrap when a
    category is unused, and dif_test still runs on bootstrap draws."""
    import numpy as np
    from raschpy.simulation import SLM_Sim, PCM_Sim
    s = SLM_Sim(no_of_items=15, no_of_persons=800, seed=12)
    m = rp.SLM(s.responses); m.calibrate(); m.std_errors(interval=0.95)
    assert m.se_method == "sandwich" and m.item_bootstrap is None
    b = rp.SLM(s.responses); b.calibrate(); b.std_errors(se_method="bootstrap", no_of_samples=300, seed=1)
    assert 0.85 < np.median(b.item_se / m.item_se) < 1.18
    m.fit_statistics()   # separation index without bootstrap draws
    assert np.isfinite(m.isi)

    p = PCM_Sim(no_of_items=8, no_of_persons=1500, max_score_vector=[3] * 8,
                manual_items=list(np.linspace(-1, 1, 8)), manual_thresholds=[[-1.0, 0.0, 1.0]] * 8, seed=13)
    q = rp.PCM(p.responses, max_score_vector=[3] * 8); q.calibrate(); q.std_errors(interval=0.95)
    assert q.se_method == "sandwich" and q.central_bootstrap is None
    qb = rp.PCM(p.responses, max_score_vector=[3] * 8); qb.calibrate(); qb.std_errors(se_method="bootstrap", no_of_samples=300, seed=1)
    ratio = np.concatenate([qb.threshold_se[it] / q.threshold_se[it] for it in q.item_names])
    assert 0.8 < np.median(ratio) < 1.25
    q.category_stats_df()   # disorder probability without bootstrap draws

    d = p.responses.copy(); d.iloc[:, 0] = d.iloc[:, 0].replace(3, 2)   # item 1 never uses category 3
    e = rp.PCM(d, max_score_vector=[3] * 8, validate=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e.calibrate()
    with pytest.warns(UserWarning, match="bootstrap rather than analytically"):
        e.std_errors(no_of_samples=30, seed=1)
    assert e.se_method == "bootstrap"


def test_global_fit_wald():
    """global_fit(): Wald test of parameter invariance across two person groups,
    for SLM, RSM and PCM. Under the null it should mostly not reject; with
    large DIF between exogenous groups it should reject; parameters next to a
    category unused in one group are excluded and reported; andersen_lr_test
    is deprecated, and method='andersen-lr' needs an exogenous split."""
    import numpy as np
    from raschpy.simulation import SLM_Sim, RSM_Sim, PCM_Sim
    I = 12; items = np.linspace(-1.5, 1.5, I); shifted = items.copy(); shifted[[2, 8]] += 1.0
    for kind in ("slm", "rsm", "pcm"):
        def sim(it, seed):
            if kind == "slm": return SLM_Sim(no_of_items=I, no_of_persons=600, manual_items=list(it), seed=seed).responses
            if kind == "rsm": return RSM_Sim(no_of_items=I, no_of_persons=600, max_score=3, manual_items=list(it), manual_thresholds=[-1.0, 0.0, 1.0], seed=seed).responses
            return PCM_Sim(no_of_items=I, no_of_persons=600, max_score_vector=[3] * I, manual_items=list(it), manual_thresholds=[[-0.8, 0.0, 0.8]] * I, seed=seed).responses
        a, b = sim(items, 41), sim(shifted, 42)
        a.index = ["A_" + x for x in a.index]; b.index = ["B_" + x for x in b.index]
        resp = pd.concat([a, b]); exo = pd.DataFrame({"G": [x[0] for x in resp.index]}, index=resp.index)
        kw = dict(exogenous=exo, validate=False)
        m = rp.SLM(resp, **kw) if kind == "slm" else rp.RSM(resp, max_score=3, **kw) if kind == "rsm" else rp.PCM(resp, max_score_vector=[3] * I, **kw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.calibrate()
            assert m.global_fit(split="exogenous", covariate="G")["reject"]
            r = m.global_fit(split="random", seed=3)
        assert r["method"] == "wald" and r["df"] > 0 and 0 <= r["p-value"] <= 1
        assert r["se_correction"] == "kauermann-carroll"
        assert set(m.global_fit_table.columns) >= {"Difference", "SE", "z"}
        # the small-sample correction only inflates the covariance, slightly
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r0 = m.global_fit(split="random", seed=3, se_correction=None)
        assert r0["df"] == r["df"]
        assert 1.0 <= r0["statistic"] / r["statistic"] < 1.2
    with pytest.raises(NotImplementedError):
        m.global_fit(method="andersen-lr", split="exogenous", covariate="G")
    with pytest.raises(ValueError):
        m.global_fit(se_correction="hc1")
    with pytest.warns(FutureWarning):
        try:
            m.andersen_lr_test(split_by="exogenous", covariate="G")
        except Exception:
            pass


def test_mfrm_empty_and_thin_rater_warnings_and_drop():
    """Raters with no responses are removed at construction with a warning;
    raters that rated fewer than min_persons persons are reported and can be
    removed with drop_facet_elements=; stretch models with no restriction warn."""
    import numpy as np
    sim = MFRM_Sim_Global(no_of_items=4, no_of_persons=60, no_of_facet_elements=5, max_score=3, seed=11)
    resp = sim.responses.copy()
    resp.loc["Rater_4"] = np.nan
    persons = resp.loc["Rater_5"].index
    resp.loc[("Rater_5", list(persons[2:])), :] = np.nan
    with pytest.warns(UserWarning) as rec:
        m = rp.MFRM(resp, max_score=3, validate=False)
    msgs = " ".join(str(w.message) for w in rec)
    assert "no responses and was removed" in msgs and "fewer than 3 persons" in msgs
    assert m.empty_raters == ["Rater_4"] and m.thin_raters == ["Rater_5"]
    assert m.dropped_raters == ["Rater_4"] and m.no_of_raters == 4 and m.rater_person_counts["Rater_5"] == 2
    with pytest.raises(ValueError):
        rp.MFRM(resp, max_score=3, validate=False, drop_facet_elements="empty")
    with pytest.warns(UserWarning, match="removed"):
        m = rp.MFRM(resp, max_score=3, validate=False, drop_facet_elements="thin")
    assert m.dropped_raters == ["Rater_4", "Rater_5"] and list(m.rater_names) == ["Rater_1", "Rater_2", "Rater_3"]
    with pytest.raises(ValueError):
        rp.MFRM(resp, max_score=3, validate=False, drop_facet_elements=["Rater_99"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        m = rp.MFRM(sim.responses, max_score=3, validate=False)
    assert m.empty_raters == [] and m.thin_raters == []
    two = MFRM_Sim_Global(no_of_items=4, no_of_persons=60, no_of_facet_elements=4, max_score=2, seed=12)
    m = rp.MFRM(two.responses, max_score=2, validate=False)
    with pytest.warns(UserWarning, match="imposes no restriction"):
        m.calibrate(model="centrality")


def test_mfrm_facet_label_and_sparse_category_warning():
    """Messages say 'facet element (e.g. rater)' unless a facet name is passed (aliases keep
    'rater' by default); threshold-varying sub-models warn about facet elements
    that used a category twice or less, other sub-models don't."""
    sim = rp.MFRM_Sim_Thresholds(no_of_items=6, no_of_persons=60, no_of_facet_elements=5, max_score=5, seed=21)
    with pytest.warns(UserWarning, match=r"facet elements? \(e\.g\. raters?\) used some score categories twice or less"):
        m = rp.MFRM(sim.responses, max_score=5, validate=False)
        m.calibrate(model="thresholds")
    assert hasattr(m, "rater_names") and m.facet_label == "facet element"
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        m.calibrate(model="items")
    with pytest.warns(UserWarning, match=r"judges? used some score categories"):
        mj = rp.MFRM(sim.responses, max_score=5, validate=False, facet="judge")
        mj.calibrate(model="thresholds")
    assert hasattr(mj, "judge_names") and mj.facets_label == "judges"


def test_mfrm_drop_facet_elements_method_and_alias():
    """drop_facet_elements() (alias drop_{facets}()) rebuilds the object
    without the named/thin facet elements, clears results, keeps renames and
    accumulates dropped_facet_elements."""
    sim = rp.MFRM_Sim_Global(no_of_items=6, no_of_persons=80, no_of_facet_elements=5, max_score=3, seed=3)
    r = sim.responses.copy()
    keep = r.index.get_level_values(1).unique()[:2]
    r[(r.index.get_level_values(0) == "Rater_5") & ~r.index.get_level_values(1).isin(keep)] = float("nan")
    with pytest.warns(UserWarning, match="drop_raters"):
        m = rp.MFRM(r, max_score=3, validate=False)
    m.calibrate(model="global")
    m.rename_facet_element("Rater_1", "Alice")
    with pytest.warns(UserWarning, match="removed"):
        m.drop_raters("thin")
    assert list(m.facet_names) == ["Alice", "Rater_2", "Rater_3", "Rater_4"]
    assert not hasattr(m, "facet_effects_global") and m.max_score == 3
    with pytest.warns(UserWarning, match="removed"):
        m.drop_facet_elements(["Alice"])
    assert m.dropped_raters == ["Rater_5", "Alice"]
    m.calibrate(model="global")
    assert len(m.facet_effects_global) == 3
    with pytest.raises(ValueError):
        m.drop_raters("bogus")


def test_mfrm_sim_drop_facet_elements():
    """MFRM_Sim.drop_facet_elements() (alias drop_{facets}()) removes the
    facet elements' responses and generating parameters (renames relabel the
    parameters too), leaving the other category probabilities unchanged."""
    for cls in (rp.MFRM_Sim_Global, rp.MFRM_Sim_Bivector, rp.MFRM_Sim_Bistretch):
        s = cls(no_of_items=5, no_of_persons=60, no_of_facet_elements=5, max_score=3, seed=4)
        before = s.cat_probs[1].loc["Rater_2"].to_numpy().copy()
        s.rename_rater("Rater_1", "Alice")
        assert list(dict.fromkeys(s.facet_effects.index.get_level_values(0)))[0] == "Alice"
        with pytest.warns(UserWarning, match="removed"):
            s.drop_raters(["Alice", "Rater_4"])
        assert s.facet_names == ["Rater_2", "Rater_3", "Rater_5"] and s.no_of_raters == 3
        assert list(dict.fromkeys(s.facet_effects.index.get_level_values(0))) == s.facet_names
        if hasattr(s, "omega_items"):
            assert list(s.omega_items.index) == s.facet_names
        assert s.dropped_raters == ["Alice", "Rater_4"]
        assert (s.cat_probs[1].loc["Rater_2"].to_numpy() == before).all()
        m = rp.MFRM(s, validate=False)
        assert list(m.facet_names) == s.facet_names
    with pytest.raises(ValueError):
        s.drop_raters(["Rater_99"])


@pytest.mark.parametrize("make, model_cls", [
    (lambda: rp.SLM_Sim(no_of_items=5, no_of_persons=150, seed=2), "SLM"),
    (lambda: rp.RSM_Sim(no_of_items=5, no_of_persons=150, max_score=3, seed=2), "RSM"),
    (lambda: rp.PCM_Sim(no_of_items=5, no_of_persons=150, max_score_vector=[2, 3, 3, 2, 3], seed=2), "PCM"),
    (lambda: rp.MFRM_Sim_Bivector(no_of_items=5, no_of_persons=150, no_of_facet_elements=4, max_score=3, seed=2), "MFRM"),
    (lambda: rp.MFRM_Sim_Bistretch(no_of_items=5, no_of_persons=150, no_of_facet_elements=4, max_score=3, seed=2), "MFRM"),
])
def test_sim_renames_relabel_generating_parameters(make, model_cls):
    """Sim item/person/facet renames relabel the generating parameters and
    probability caches too (previously only responses), so recovery_stats()
    lines up instead of raising KeyError; values are unchanged."""
    s = make()
    items_before = s.items.to_numpy().copy()
    s.rename_items_all([f"Q{i}" for i in range(1, 6)])
    s.rename_person(s.person_names[0], "Ann")
    if hasattr(s, "facet_names"):
        s.rename_rater("Rater_2", "Bob")
        assert "Bob" in set(s.facet_effects.index.get_level_values(0))
    assert list(s.items.index) == s.item_names and s.persons.index[0] == "Ann"
    assert (s.items.to_numpy() == items_before).all()
    m = getattr(rp, model_cls)(s, validate=False) if model_cls == "MFRM" else getattr(rp, model_cls)(s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(**({"model": s.model} if model_cls == "MFRM" else {}))
        m.recovery_stats(plot=False)
    assert not m.recovery.items.table["generating"].isna().any()
    assert not m.recovery.persons.table["generating"].isna().any()
    with pytest.warns(UserWarning, match="duplicate"):
        s.rename_item("Q1", "Q2")
    assert s.item_names[:2] == ["Q1", "Q2"]


def test_rsm_rcl_cpat_default_and_options():
    """RSM calibrates by RCL-CPAT by default (converging, beating the old
    two-stage items here), keeps 'pair-cpat' as an option that persists
    across refits, warns about ignored pair-cpat arguments, and its sandwich
    SEs agree with the bootstrap."""
    import numpy as np
    sim = _rsm_sim(no_of_items=12, no_of_persons=600, max_score=4, missing=0.1, seed=7)
    m = rp.RSM(sim.responses, max_score=4)
    m.calibrate()
    assert m.estimation == "cp-full-rcl" and m.convergence["converged"]
    true_items = sim.items.reindex(m.items.index)
    p = rp.RSM(sim.responses, max_score=4)
    p.calibrate(estimation="pair-cpat")
    assert np.sqrt(((m.items - true_items) ** 2).mean()) < np.sqrt(((p.items - true_items) ** 2).mean())
    with pytest.warns(UserWarning, match="ignored with estimation='cp-full-rcl'"):
        m.calibrate(method="cos")
    m.std_errors()
    assert m.se_method == "sandwich"
    b = rp.RSM(sim.responses, max_score=4)
    b.calibrate()
    b.std_errors(se_method="bootstrap", no_of_samples=200, seed=1)
    assert np.allclose(m.item_se, b.item_se, rtol=0.25)
    p.calibrate()
    assert p.estimation == "pair-cpat"


def test_mfrm_rcl_cpat_per_model_items_and_anchoring():
    """MFRM RCL-CPAT: each sub-model has its own item locations/thresholds,
    used by its own tables; regularisation arguments for another sub-model
    warn; stretch anchoring leaves every location unchanged but for one
    common shift."""
    import numpy as np
    sim = rp.MFRM_Sim_Bistretch(no_of_items=6, no_of_persons=200, no_of_facet_elements=5,
                                max_score=3, seed=2, global_range=2, stretch_range=1)
    m = rp.MFRM(sim.responses, max_score=3, validate=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(model="global")
        m.calibrate(model="bistretch")
    assert m.estimation == "cp-full-rcl" and m.convergence["converged"]
    assert not np.allclose(m.items_global, m.items_bistretch)
    assert np.allclose(m.item_stats_df(model="global", se=False)["Estimate"], m.items_global.round(3))
    with pytest.warns(UserWarning, match="stretch_ridge=2 does not apply"):
        m.calibrate(model="global", stretch_ridge=2)
    m.calibrate_anchor("bistretch", anchors=["Rater_1", "Rater_3"])

    def grid(anchor):
        items, thr, sev = m._get_params("bistretch", anchor=anchor)
        return items.to_numpy()[None, :, None] + thr.to_numpy()[None, None, :] + sev.to_numpy().reshape(5, 6, 3)
    shift = grid(True) - grid(False)
    assert np.ptp(shift) < 1e-10
    assert np.isclose(m.anchor_lambda_bistretch.loc[["Rater_1", "Rater_3"]].mean(), 0)
    assert np.isclose(m.anchor_omega_items_bistretch.loc[["Rater_1", "Rater_3"]].mean(), 1)


def test_mfrm_rcl_sandwich_standard_errors():
    """MFRM std_errors() defaults to sandwich SEs for RCL-CPAT sub-models,
    agreeing with the bootstrap; store_bootstrap=True, 'pair-cpat' and an
    empty category fall back to the bootstrap."""
    import numpy as np
    sim = rp.MFRM_Sim_Bistretch(no_of_items=8, no_of_persons=400, no_of_facet_elements=5,
                                max_score=3, seed=5, global_range=2, stretch_range=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = rp.MFRM(sim.responses, max_score=3, validate=False)
        m.calibrate(model="bistretch")
        m.std_errors(model="bistretch", interval=0.95)
        b = rp.MFRM(sim.responses, max_score=3, validate=False)
        b.calibrate(model="bistretch")
        b.std_errors(model="bistretch", se_method="bootstrap", no_of_samples=300, seed=1)
    assert m.se_method_bistretch == "sandwich" and b.se_method_bistretch == "bootstrap"
    for attr in ("item_se_bistretch", "rater_se_bistretch", "omega_items_se_bistretch"):
        ratio = np.asarray(getattr(m, attr), float) / np.asarray(getattr(b, attr), float)
        assert 0.85 < np.nanmedian(ratio) < 1.15
    assert np.allclose(m.item_high_bistretch - m.item_low_bistretch, 2 * 1.959964 * m.item_se_bistretch)
    assert set(m.stretch_param_cov_bistretch) == {"lambda_omega_items", "lambda_omega_thresholds",
                                                  "omega_items_omega_thresholds"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.std_errors(model="bistretch", store_bootstrap=True, no_of_samples=20, seed=1)
    assert m.se_method_bistretch == "bootstrap"
    with pytest.raises(ValueError):
        m.std_errors(model="bistretch", se_method="sandwich", store_bootstrap=True)


def test_mfrm_unlinked_facet_groups_and_isolation_warnings():
    """Facet elements in groups never compared with each other get one
    warning (both estimation methods), standard errors fall back to the
    bootstrap (NaN only for an explicit sandwich request), and pair-cpat's per-item zero-comparison
    findings are reported in a single warning."""
    import numpy as np
    sim = MFRM_Sim_Global(no_of_items=6, no_of_persons=200, no_of_facet_elements=4, max_score=3, seed=1)
    resp = sim.responses.copy()
    first = set(resp.index.get_level_values(1).unique()[:100])
    resp[np.array([(r in ("Rater_1", "Rater_2")) != (p in first) for r, p in resp.index])] = np.nan
    for est in ("cp-full-rcl", "cpat-rcl", "pair-cpat"):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            rp.MFRM(resp, max_score=3, validate=False).calibrate(model="items", estimation=est)
        assert sum("fall into 2 groups" in str(w.message) for w in rec) == 1
    m = rp.MFRM(resp, max_score=3, validate=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(model="centrality")
    with pytest.warns(UserWarning, match="computed by bootstrap rather than analytically"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*bootstrap resamples raised")
            m.std_errors(model="centrality", no_of_samples=20, seed=1)
    assert m.se_method_centrality == "bootstrap" and m.lambda_se_centrality.notna().all()
    with pytest.warns(UserWarning, match="standard errors are set to NaN"):
        m.std_errors(model="centrality", se_method="sandwich")
    assert m.lambda_se_centrality.isna().all() and m.item_se_centrality.notna().all()
    resp2 = sim.responses.copy()
    resp2.loc["Rater_4", ["Item_1", "Item_2"]] = np.nan
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        rp.MFRM(resp2, max_score=3, validate=False).calibrate(model="items", estimation="pair-cpat")
    iso = [str(w.message) for w in rec if "zero PAIR comparisons" in str(w.message)]
    assert len(iso) == 1 and "Rater_4 (Item_1, Item_2)" in iso[0]


def test_mfrm_model_selection_drops_equivalent_models():
    """With max_score=2 centrality and thresholds are the same model (with
    max_score=1 several more collapse): the later model in ladder order is
    left out with one warning, and every other comparison still runs."""
    sim = MFRM_Sim_Global(no_of_items=4, no_of_persons=300, no_of_facet_elements=4, max_score=2, seed=1)
    m = rp.MFRM(sim.responses, max_score=2, validate=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="^(?!.*parameterisations)")
        with pytest.warns(UserWarning, match="'centrality' is the same model as 'thresholds' and has been left out"):
            m.model_selection(test="LR", seed=1)
        pairs = m.model_comparison_mfrm_lr_summary.index
        assert not any("centrality" in p for p in pairs) and "global vs pseudo_halo" in pairs
        with pytest.warns(UserWarning, match="'centrality' is the same model as 'thresholds'"):
            m.per_rater_model_selection(seed=1)
        with pytest.raises(ValueError, match="at least 2 distinct models"):
            m.model_selection(models=["centrality", "thresholds"])
    sim1 = MFRM_Sim_Global(no_of_items=4, no_of_persons=300, no_of_facet_elements=4, max_score=1, seed=1)
    m1 = rp.MFRM(sim1.responses, max_score=1, validate=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="^(?!.*same model)")
        with pytest.warns(UserWarning, match="'thresholds' and 'centrality' are the same model as 'global'"):
            m1.model_selection(test="LR", seed=1)
    tab = m1.model_comparison_mfrm_lr_summary
    assert list(tab.index) == ["global vs items", "global vs pseudo_halo", "pseudo_halo vs items"]
    assert (tab["df"] > 0).all()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for test in ("AIC", "LR"):
            m1.per_rater_model_selection(test=test, seed=1)
            sel = m1.per_rater_model_selection_table["selected_model"]
            assert set(sel) <= {"global", "items", "pseudo_halo"}


def test_mfrm_anchored_per_rater_selection_uses_anchored_frame():
    """Anchored per_rater_model_selection must evaluate the anchored facet
    effects against the anchored item locations and thresholds; with the
    unanchored ones, the anchors' mean pattern made flat raters look like
    stretch models (here Rater_5/6 came out as pseudo_halo/bistretch)."""
    import numpy as np
    I, K = 6, 5
    i = np.arange(I)[:, None] - 2.5
    k = np.arange(K)[None, :] - 2.0
    pats = [np.zeros((I, K))] * 6 + [np.zeros((I, K)) + k + i] * 6
    names = [f"Rater_{j + 1}" for j in range(len(pats))]
    raters = {n: {f"Item_{a + 1}": list(p[a]) for a in range(I)} for n, p in zip(names, pats)}
    sim = rp.MFRM_Sim_Matrix(no_of_items=I, no_of_persons=1500, no_of_facet_elements=len(pats), max_score=K,
                             person_sd=2.5, manual_items=list(i.ravel()), manual_thresholds=list(2 * k.ravel()),
                             manual_raters=raters, seed=1)
    m = rp.MFRM(sim.responses, max_score=K, validate=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.per_rater_model_selection(anchors=names[:4], seed=1)
    sel = m.anchor_per_rater_model_selection_table["selected_model"]
    assert list(sel[["Rater_5", "Rater_6"]]) == ["global", "global"]
    assert (sel[names[6:]] == "bistretch").all()


def test_mfrm_rcl_stretch_divergence_test_fits_parent():
    """Under estimation='cpat-rcl' a stretch model is fitted directly, so its
    bootstrap resamples carry no parent (thresholds/items/bivector) fit; the
    divergence test must fit the parent itself (it raised AttributeError),
    anchored or not, without changing the object's latest calibration."""
    from raschpy import MFRM_Sim_Centrality, MFRM_Sim_Bistretch
    for cls, mod in ((MFRM_Sim_Centrality, "centrality"), (MFRM_Sim_Bistretch, "bistretch")):
        sim = cls(no_of_items=6, no_of_persons=150, no_of_facet_elements=5, max_score=3, seed=2)
        m = rp.MFRM(sim)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.calibrate(model=mod)
            items = m.items.copy()
            m.rater_stats_df(model=mod, divergence_test="wald", no_of_samples=20, seed=1)
            m.calibrate_anchor(mod, ["Rater_1", "Rater_2"])
            m.rater_stats_df(model=mod, divergence_test="wald", no_of_samples=20, seed=1,
                             anchors=["Rater_1", "Rater_2"])
        assert (m.items == items).all()


def test_mfrm_extreme_facet_elements():
    """A facet element that used only the lowest or only the highest category
    is reported once on creation, listed in extreme_facet_elements (alias
    extreme_raters) and removable with 'extreme'; a constant middle-category
    rater is not extreme; estimates stay finite."""
    import numpy as np
    sim = MFRM_Sim_Global(no_of_items=6, no_of_persons=150, no_of_facet_elements=5, max_score=3, seed=11)
    resp = sim.responses.copy().astype(float)
    resp.loc["Rater_3"] = 1.0
    resp.loc["Rater_4"] = 0.0
    resp.loc["Rater_5"] = 3.0
    with pytest.warns(UserWarning, match=r"Rater_4: lowest, Rater_5: highest"):
        m = rp.MFRM(resp, max_score=3, validate=False)
    assert m.extreme_facet_elements == m.extreme_raters == ["Rater_4", "Rater_5"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate(model="global")
        assert np.isfinite(m.facet_effects_global).all()
        m.drop_raters("extreme")
    assert list(m.rater_names) == ["Rater_1", "Rater_2", "Rater_3"]
    assert m.dropped_raters == ["Rater_4", "Rater_5"]
    sim.responses.loc["Rater_5"] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.drop_raters("extreme")
    assert sim.facet_names == ["Rater_1", "Rater_2", "Rater_3", "Rater_4"]


def test_rcl_sandwich_after_quasi_newton_fit(monkeypatch):
    """Large designs are fitted by L-BFGS, which stores no bread; the sandwich
    must form it on demand (it raised KeyError) and agree with the Newton-
    Raphson fit's, and fall back to the bootstrap beyond SANDWICH_LIMIT."""
    import numpy as np
    from raschpy import _rcl, MFRM_Sim_Bivector
    sim = MFRM_Sim_Bivector(no_of_items=6, no_of_persons=300, no_of_facet_elements=5, max_score=3, seed=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = rp.MFRM(sim); a.calibrate(model="bivector"); a.std_errors(model="bivector")
        monkeypatch.setattr(_rcl, "DENSE_LIMIT", 10)
        b = rp.MFRM(sim); b.calibrate(model="bivector"); b.std_errors(model="bivector")
    assert b._rcl_fits["bivector"]["info"]["optimiser"] == "l-bfgs-b"
    assert b.se_method_bivector == "sandwich"
    assert np.allclose(a.item_se_bivector, b.item_se_bivector, atol=1e-6)
    monkeypatch.setattr(_rcl, "SANDWICH_LIMIT", 10)
    c = rp.MFRM(sim)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c.calibrate(model="bivector")
    with pytest.warns(UserWarning, match="too large to form the analytic standard errors"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*bootstrap resamples raised")
            c.std_errors(model="bivector", no_of_samples=10, seed=1)
    assert c.se_method_bivector == "bootstrap"


def test_mfrm_global_fit_not_available():
    """MFRM inherits global_fit() but has no per-group machinery for it: it
    must raise a clear NotImplementedError, not an AttributeError."""
    m = rp.MFRM(MFRM_Sim_Global(no_of_items=5, no_of_persons=100, no_of_facet_elements=3, max_score=3, seed=1))
    with pytest.raises(NotImplementedError, match="not yet for MFRM"):
        m.global_fit()


def test_rcl_full_patterns():
    """estimation='cp-full-rcl' (default) fits all response-pattern pairs: it differs from 'cpat-rcl', converges and
    gives sandwich SEs for every sub-model kind."""
    import numpy as np
    sim = _rsm_sim(no_of_items=10, no_of_persons=400, max_score=4, seed=11)
    a = rp.RSM(sim.responses, max_score=4); a.calibrate()
    b = rp.RSM(sim.responses, max_score=4); b.calibrate(estimation="cpat-rcl")
    assert a.estimation == "cp-full-rcl" and a.convergence["converged"]
    assert not np.allclose(a.thresholds, b.thresholds)
    a.std_errors()
    assert a.se_method == "sandwich" and np.isfinite(a.threshold_se).all()
    from raschpy.simulation.mfrm_sim import MFRM_Sim_Bistretch
    s2 = MFRM_Sim_Bistretch(no_of_items=6, no_of_persons=300, max_score=3, no_of_facet_elements=4, seed=3)
    m = rp.MFRM(s2.responses, max_score=3, validate=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for model in ("global", "matrix", "bistretch"):
            m.calibrate(model=model)
            assert getattr(m, f"estimation_{model}") == "cp-full-rcl"
            m.std_errors(model=model)
            assert getattr(m, f"se_method_{model}") == "sandwich"


# ---------------------------------------------------------------------------
# calibrate_anchor(calibrate=True) passes estimation= (and the MFRM
# regularisation arguments) on to calibrate()
# ---------------------------------------------------------------------------

def test_calibrate_anchor_passes_estimation():
    sim = _rsm_sim(no_of_items=10, no_of_persons=400, max_score=3, seed=1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = rp.RSM(sim.responses)
        r.calibrate_anchor(sim.items.iloc[:8], calibrate=True, plot=False)
    assert r.estimation == "cp-full-rcl"
    assert not [x for x in w if "ignored" in str(x.message)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r.calibrate_anchor(sim.items.iloc[:8], calibrate=True, plot=False, estimation="pair-cpat")
    assert r.estimation == "pair-cpat"

    _, m = _make_mfrm()
    anchors = list(m.facet_names[:2])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.calibrate_anchor("items", anchors, calibrate=True)
    assert m.estimation_items == "cp-full-rcl"
    assert not [x for x in w if "ignored" in str(x.message)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.calibrate_anchor("matrix", anchors, calibrate=True, ridge=0.5, interaction_ridge=1.0)
        assert m.ridge == 0.5 and m.interaction_ridge == 1.0
        m.calibrate_anchor("items", anchors, calibrate=True, estimation="pair-cpat")
    assert m.estimation_items == "pair-cpat"
    assert hasattr(m, "anchor_items_items")


def test_mfrm_dif_issue17_warning_pair_cpat_only():
    """The known issue #17 DIF warning (compression from pair-cpat's staged estimation) is raised for
    representations at risk only when the groups are calibrated with estimation='pair-cpat'."""
    import numpy as np
    sim = MFRM_Sim_Global(no_of_items=5, no_of_persons=300, no_of_facet_elements=4, max_score=3, seed=3)
    persons = sim.responses.index.get_level_values(1).unique()
    exo = pd.DataFrame({"g": np.where(np.arange(len(persons)) % 2, "a", "b")}, index=persons)
    for est, expected in (("cp-full-rcl", False), ("pair-cpat", True)):
        m = rp.MFRM(sim.responses, max_score=3, exogenous=exo, validate=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m.calibrate(model="thresholds", estimation=est)
            m.dif_test("g", model="thresholds", no_of_samples=20, seed=1)
        assert any("risk of compressed item" in str(x.message) for x in w) == expected


def test_std_residuals_plot_dpi_and_deprecated_plot_density(tmp_path):
    """std_residuals_plot takes dpi= like every other plot; plot_density= still works, with a FutureWarning."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from raschpy.simulation import SLM_Sim, PCM_Sim
    sims = [
        (rp.SLM, SLM_Sim(no_of_items=6, no_of_persons=150, seed=1), {}),
        (rp.PCM, PCM_Sim(no_of_items=5, no_of_persons=150, max_score_vector=[2] * 5, seed=1), {}),
        (rp.RSM, _rsm_sim(no_of_items=5, no_of_persons=150, max_score=2, seed=1), {"max_score": 2}),
    ]
    _, mf = _make_mfrm()
    objs = []
    for cls, sim, kw in sims:
        objs.append(cls(sim.responses, **kw))
    objs.append(mf)
    for i, m in enumerate(objs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit_statistics()
            m.std_residuals_plot(filename=str(tmp_path / f"a{i}"), dpi=50)
        with pytest.warns(FutureWarning, match="plot_density"):
            m.std_residuals_plot(filename=str(tmp_path / f"b{i}"), plot_density=50)
        plt.close("all")
    assert len(list(tmp_path.iterdir())) == 8


def test_mfrm_sandwich_unused_extreme_category_and_extreme_facet_element_se():
    """MFRM falls back to the bootstrap when the lowest or highest category is unused (as for a middle
    one), and reports NaN standard errors for a facet element that used only one extreme category."""
    import numpy as np
    sim = MFRM_Sim_Global(no_of_items=6, no_of_persons=300, no_of_facet_elements=5, max_score=3, seed=2)
    r = sim.responses.copy()
    m = rp.MFRM(r.where(r != 3, 2), max_score=3, validate=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.calibrate(model="global")
        m.std_errors(model="global", no_of_samples=20, seed=1)
    assert m.se_method_global == "bootstrap"
    assert any("bootstrap rather than analytically" in str(x.message) for x in w)

    r = sim.responses.astype(float).copy()
    name = r.index.get_level_values(0).unique()[0]
    mask = r.index.get_level_values(0) == name
    vals = r.values.copy(); sub = vals[mask]; sub[~np.isnan(sub)] = 3; vals[mask] = sub
    r.loc[:, :] = vals
    for model in ("global", "items"):
        m = rp.MFRM(r, max_score=3, validate=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.calibrate(model=model)
            m.std_errors(model=model)
        se = getattr(m, f"rater_se_{model}")
        assert np.all(np.isnan(np.asarray(se.loc[name], float)))
        assert np.isfinite(np.asarray(se.drop(name), float)).all()
