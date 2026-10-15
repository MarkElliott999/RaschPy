"""
Property-based invariance tests for RaschPy (remediation-plan item 13).

Three properties, run with hypothesis rather than hand-written cases:

1. Permutation invariance -- shuffling person-row / item-column order in
   the input data must not change calibrated estimates once re-aligned by
   name. Generalises the one handcrafted #19/#32 regression test
   (test_known_issue_regressions.py) into a broadly-sampled property --
   that bug (ragged rater x person MFRM order dependence) is exactly the
   class of thing this would have caught structurally, without needing
   the specific failing case to be found and hand-written first.
2. Origin-translation invariance -- shifting the whole scale (item AND
   person locations together) by a constant must leave fit statistics
   (infit/outfit) unchanged, since they depend only on person-item
   location differences, not absolute origin.
3. Recovery grids -- calibrate() at known generating parameters across a
   small parameter grid stays within a fixed recovery-error bound.

Data is generated via RaschPy's own *_Sim classes with hypothesis-drawn
parameters, not raw random response matrices -- raw matrices risk
disconnection/degenerate cases unrelated to the property being tested.
Tolerance follows the existing regression-test convention (atol=1e-6,
np.testing.assert_allclose) -- see run_regression_tests.py.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import raschpy as rp
from raschpy.simulation import MFRM_Sim, PCM_Sim, RSM_Sim, SLM_Sim

warnings.filterwarnings("ignore")

ATOL = 1e-6

# Kept deliberately small -- these are closed-form PAIR/CPAT calibrations
# run twice per example (original + permuted), so example count matters
# for CI runtime more than it would for an iterative estimator.
_SETTINGS = settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

_small_n_items = st.integers(min_value=5, max_value=10)
_small_n_persons = st.integers(min_value=60, max_value=150)
_small_max_score = st.integers(min_value=2, max_value=5)
_seed = st.integers(min_value=1, max_value=10_000)


def _permuted(df, rng):
    """Shuffle row and column order of a response DataFrame."""
    shuffled_cols = rng.permutation(df.columns)
    shuffled_rows = rng.permutation(df.index)
    return df.loc[shuffled_rows, shuffled_cols]


def _assert_series_match(a, b, label):
    idx = a.index.union(b.index)
    np.testing.assert_allclose(
        a.reindex(idx).astype(float).values,
        b.reindex(idx).astype(float).values,
        atol=ATOL,
        equal_nan=True,
        err_msg=f"{label} differs after permutation",
    )


# ---------------------------------------------------------------------------
# 1. Permutation invariance
# ---------------------------------------------------------------------------

@_SETTINGS
@given(
    no_of_items=_small_n_items, no_of_persons=_small_n_persons, seed=_seed,
)
def test_permutation_invariance_slm(no_of_items, no_of_persons, seed):
    sim = SLM_Sim(
        no_of_items=no_of_items, no_of_persons=no_of_persons, item_range=2, seed=seed,
    )
    rng = np.random.default_rng(seed)

    original = rp.SLM(sim.responses.copy())
    original.calibrate()

    permuted = rp.SLM(_permuted(sim.responses, rng))
    permuted.calibrate()

    _assert_series_match(original.items, permuted.items, "SLM item locations")
    original.person_estimates()
    permuted.person_estimates()
    _assert_series_match(original.persons, permuted.persons, "SLM person locations")


@_SETTINGS
@given(
    no_of_items=_small_n_items, no_of_persons=_small_n_persons,
    max_score=_small_max_score, seed=_seed,
)
def test_permutation_invariance_pcm(no_of_items, no_of_persons, max_score, seed):
    sim = PCM_Sim(
        no_of_items=no_of_items, no_of_persons=no_of_persons,
        max_score_vector=[max_score] * no_of_items, item_range=2, seed=seed,
    )
    rng = np.random.default_rng(seed)

    original = rp.PCM(sim.responses.copy(), [max_score] * no_of_items)
    original.calibrate()

    permuted = rp.PCM(_permuted(sim.responses, rng), [max_score] * no_of_items)
    permuted.calibrate()

    _assert_series_match(original.items, permuted.items, "PCM item locations")


@_SETTINGS
@given(
    no_of_items=_small_n_items, no_of_persons=_small_n_persons,
    max_score=_small_max_score, seed=_seed,
)
def test_permutation_invariance_rsm(no_of_items, no_of_persons, max_score, seed):
    sim = RSM_Sim(
        no_of_items=no_of_items, no_of_persons=no_of_persons,
        max_score=max_score, item_range=2, seed=seed,
    )
    rng = np.random.default_rng(seed)

    original = rp.RSM(sim.responses.copy(), max_score)
    original.calibrate()

    permuted = rp.RSM(_permuted(sim.responses, rng), max_score)
    permuted.calibrate()

    _assert_series_match(original.items, permuted.items, "RSM item locations")
    _assert_series_match(original.thresholds, permuted.thresholds, "RSM thresholds")


@_SETTINGS
@given(
    no_of_items=st.integers(min_value=5, max_value=8),
    no_of_persons=st.integers(min_value=40, max_value=80),
    no_of_facet_elements=st.integers(min_value=3, max_value=6),
    max_score=_small_max_score, seed=_seed,
)
def test_permutation_invariance_mfrm_global(
    no_of_items, no_of_persons, no_of_facet_elements, max_score, seed,
):
    # This is the #19/#32 bug class specifically: MFRM's response grid is
    # indexed by (rater, person) tuples, not a plain person index, so a
    # ragged/shuffled row order previously risked silently collapsing
    # rater severity spread (see test_known_issue_regressions.py). Row
    # permutation here shuffles across BOTH levels jointly, which the
    # handwritten regression test's specific single-rater-shuffle case
    # doesn't exercise as broadly.
    sim = MFRM_Sim(
        no_of_items=no_of_items, no_of_persons=no_of_persons,
        no_of_facet_elements=no_of_facet_elements, max_score=max_score,
        model="global", item_range=2, seed=seed,
    )
    rng = np.random.default_rng(seed)

    original = rp.MFRM(sim.responses.copy(), max_score=max_score)
    original.calibrate(model="global")

    permuted = rp.MFRM(_permuted(sim.responses, rng), max_score=max_score)
    permuted.calibrate(model="global")

    _assert_series_match(original.items, permuted.items, "MFRM item locations")
    _assert_series_match(original.thresholds, permuted.thresholds, "MFRM thresholds")
    _assert_series_match(
        original.facet_effects_global, permuted.facet_effects_global,
        "MFRM global rater severities",
    )


# ---------------------------------------------------------------------------
# 2. Origin-translation invariance of fit statistics
# ---------------------------------------------------------------------------
#
# The Rasch model is identified only up to a joint additive constant across
# items and persons -- shifting the whole scale by a constant (item AND
# person locations together) leaves every response probability, and hence
# every fit statistic, exactly unchanged. This tests that property
# directly: inject a shifted item vector into a fresh probe (bypassing
# calibrate()), let person_estimates()/fit_statistics() naturally re-derive
# person locations against the shifted items on the SAME response data, and
# confirm fit statistics match the unshifted baseline. Deliberately doesn't
# route through calibrate_anchor() -- that method's own robust anchor-
# selection machinery (trimming, Wald tests) is a separate concern from the
# underlying mathematical invariance property being tested here.

@_SETTINGS
@given(
    no_of_items=_small_n_items, no_of_persons=_small_n_persons,
    shift=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    seed=_seed,
)
def test_origin_translation_invariance_slm(no_of_items, no_of_persons, shift, seed):
    sim = SLM_Sim(
        no_of_items=no_of_items, no_of_persons=no_of_persons, item_range=2, seed=seed,
    )
    m = rp.SLM(sim.responses.copy())
    m.calibrate()
    m.fit_statistics()
    baseline_item_infit = m.item_infit_ms.copy()
    baseline_item_outfit = m.item_outfit_ms.copy()
    baseline_person_infit = m.person_infit_ms.copy()
    baseline_person_outfit = m.person_outfit_ms.copy()

    probe = rp.SLM(sim.responses.copy())
    probe.items = m.items + shift
    probe.fit_statistics()

    _assert_series_match(
        baseline_item_infit, probe.item_infit_ms, "SLM item infit MS under origin shift",
    )
    _assert_series_match(
        baseline_item_outfit, probe.item_outfit_ms, "SLM item outfit MS under origin shift",
    )
    _assert_series_match(
        baseline_person_infit, probe.person_infit_ms, "SLM person infit MS under origin shift",
    )
    _assert_series_match(
        baseline_person_outfit, probe.person_outfit_ms, "SLM person outfit MS under origin shift",
    )


# ---------------------------------------------------------------------------
# 3. Recovery grids
# ---------------------------------------------------------------------------

@_SETTINGS
@given(
    no_of_items=st.integers(min_value=8, max_value=15),
    no_of_persons=st.integers(min_value=200, max_value=600),
    seed=_seed,
)
def test_recovery_grid_slm(no_of_items, no_of_persons, seed):
    sim = SLM_Sim(
        no_of_items=no_of_items, no_of_persons=no_of_persons, item_range=2, seed=seed,
    )
    m = rp.SLM(sim)
    m.calibrate()
    m.recovery_stats(plot=False)
    rmse = m.recovery.items.summary.loc["RMSE", "Overall"]
    assert rmse < 0.6, f"SLM item-location RMSE {rmse:.3f} exceeds bound at N={no_of_persons}"


@_SETTINGS
@given(
    no_of_items=st.integers(min_value=8, max_value=15),
    no_of_persons=st.integers(min_value=200, max_value=600),
    max_score=_small_max_score, seed=_seed,
)
def test_recovery_grid_pcm(no_of_items, no_of_persons, max_score, seed):
    sim = PCM_Sim(
        no_of_items=no_of_items, no_of_persons=no_of_persons,
        max_score_vector=[max_score] * no_of_items, item_range=2, seed=seed,
    )
    m = rp.PCM(sim)
    m.calibrate()
    m.recovery_stats(plot=False)
    rmse = m.recovery.items.summary.loc["RMSE", "Overall"]
    assert rmse < 0.6, f"PCM item-location RMSE {rmse:.3f} exceeds bound at N={no_of_persons}"


@_SETTINGS
@given(
    no_of_items=st.integers(min_value=8, max_value=15),
    no_of_persons=st.integers(min_value=200, max_value=600),
    max_score=_small_max_score, seed=_seed,
)
def test_recovery_grid_rsm(no_of_items, no_of_persons, max_score, seed):
    sim = RSM_Sim(
        no_of_items=no_of_items, no_of_persons=no_of_persons,
        max_score=max_score, item_range=2, seed=seed,
    )
    m = rp.RSM(sim)
    m.calibrate()
    m.recovery_stats(plot=False)
    rmse = m.recovery.items.summary.loc["RMSE", "Overall"]
    assert rmse < 0.6, f"RSM item-location RMSE {rmse:.3f} exceeds bound at N={no_of_persons}"
