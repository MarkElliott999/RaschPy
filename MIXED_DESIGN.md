# `mixed` as a full MFRM model — design for review (2026-10-25)

Scope: v2.0.0 item 6 (in scope since 2026-10-24). `calibrate(model='mixed')` fits one joint model in which each
facet element (rater) has its own representation, and `per_rater_model_selection` chooses those representations
from proper fits instead of deriving them from the matrix fit. Nothing built yet.

## Today

- `per_rater_model_selection` fits the **matrix** model once, then derives every rater's restricted candidates as
  **marginal means** of its matrix parameters (bivector = item and threshold marginals, items = item marginals, …,
  stretch models by a slope regression on them). There are no refits.
- Candidates are scored with the **joint** log-likelihood over each rater's own observations, at the matrix
  fit's person estimates, with person subsampling (`sampling='dynamic'`). They are compared by an LR ladder
  (matrix → bivector → items / thresholds / global, plus one targeted stretch test) or by minimum AIC / BIC.
- `mixed` is that derived surface (`facet_effects_mixed`), with no estimation of its own: no item or threshold
  estimates of its own, and no SEs except by bootstrapping the whole procedure.
- Known weaknesses (2026-10-09 studies): derived candidates sit below their own maxima, so the plug-in likelihood
  ratio inflates with N (the conditional version picked over-complex models 23–27% of the time). The one-step
  conditional fix helped (LR worst-case correct choice 0.53 → 0.65) but is the kind of "scored at parameters
  nobody estimated" step you parked on 2026-10-22.

## Proposed model

One location grid for every (rater, item, threshold) cell:

    b[r, i, h] = δ_i + τ_h + λ_r + (ωI_r − 1) δ_i + (ωT_r − 1) τ_h + a_ri + t_rh + e_rih

with a per-rater mask of the terms its representation allows:

| representation | terms |
|---|---|
| global | λ |
| items | λ, a |
| thresholds | λ, t |
| bivector | λ, a, t |
| matrix | λ, a, t, e |
| centrality | λ, ωT |
| pseudo_halo | λ, ωI |
| bistretch | λ, ωI, ωT |

- Within each rater, a is zero-sum over items, t over thresholds, and e over items and thresholds (so λ is
  the rater's severity throughout, as in the three-level penalty).
- **Identification, minimal:** λ is zero-sum over raters (plus the usual δ, τ zero-sums). A deviation type (a, t, e,
  ω) is centred over raters only if **every** rater carries it. Otherwise the raters without it pin δ / τ, and
  centring would be a binding constraint. Example: one matrix rater among global raters would lose its interactions.
- With every rater on the same representation this is exactly that representation's model, so `mixed`
  reproduces the eight existing fits. That becomes a test.

**Penalties:** the three-level penalty per rater, with the weights of that rater's representation:
- severity (`global_ridge`) for everyone;
- `item_ridge` / `threshold_ridge` at the item, thresholds, bivector or matrix value as appropriate;
- `interaction_ridge` for matrix raters;
- `stretch_ridge` for stretch raters.

CML and `cp-full-rcl` keep their own default weights. `cp-full-rcl` keeps its new smoothing (threshold constant,
item-location ridge, width ridge).

**Reporting frame (revised, user 2026-10-27):** each rater's cell effects are built from its own fitted parameters
relative to the fit's common structure δ_i + τ_h (defined by the raters without the corresponding terms, as in the
minimal identification): λ_r plus that rater's own item, threshold and interaction terms. A global rater shows its
severity in every cell; a matrix rater its severity plus its own cell effects; no rater's deviations pass into
another's row (the all-rater cell average would give every rater minus 1/R of the other raters' non-global terms).
Rebuilt the same way in every refit (bootstrap resamples, DIF/DRF group fits, anchored fits), so tables, SEs and
tests share one frame and each rater keeps its representation. With one representation for every rater this is that
uniform model's table (checked in V4). `facet_effects_mixed` (R·I × K) in this frame; `rater_parameters_mixed`
the native parameters; `rater_models` the representations.

## Fitting

One generic **grid design** for both estimators: grid(θ), its Jacobian, and (when any rater is a stretch rater)
the bilinear second-derivative term. This is the interface the MFRM CML stretch fit already uses.
- **CML:** the virtual items with `_esf.fit_cml` (no stretch raters, linear) or `fit_cml_stretch` (bilinear).
  Unused categories as now ('ridge' / 'null').
- **`cp-full-rcl`:** the full-pattern comparisons applied to the grid, as `fit_cell` does. This needs `fit_cell`
  generalised from `CellModel` to any design, plus the bilinear term for stretch raters. The current RCL stretch
  code (`StretchModel`) is a separate implementation, so this is the largest single piece.
- **Start:** the matrix fit (the same estimator), restricted per rater.

API: `calibrate(model='mixed', rater_models=None | dict | Series, ...)` and
`calibrate_anchor('mixed', anchors, rater_models=...)`.
- A dict or Series {rater: representation} fixes the representations, chosen by the user: no selection is run.
  Representation names are validated (aliases accepted); unknown rater names raise an error. Raters not named
  default to 'matrix', or raise an error (decision D6 below).
- `None` (default) runs `per_rater_model_selection()` first (with `anchors=` under `calibrate_anchor`, as
  `anchor_rater_models` is now), then the joint refit. `candidate_models=` (default None: all eight) is passed
  to its `models=`, restricting the representations considered; giving it together with an explicit
  `rater_models` raises an error. A stored selection is reused only if made with the same candidate set,
  anchors and estimation settings. Other selection settings (`test`, `alpha`, `aic_sig_test`, `min_effect`) are
  set by calling `per_rater_model_selection(...)` and passing its `rater_models`.
- `per_rater_model_selection(models=)` is renamed `candidate_models=` (user, 2026-10-27), `models=` deprecated
  (FutureWarning naming the new argument; both together raise an error); the same for
  `MFRM.model_selection(models=)`, which has the same meaning (user, 2026-10-27).
  `self.rater_models` (or `anchor_rater_models`) records the representations, and whether they were given or
  selected.
- `per_rater_model_selection()` stays a public method (D8): it reports which raters depart from the common
  structure and how, without fitting `mixed`, and stores and returns the `rater_models` Series, so
  `m.calibrate(model='mixed', rater_models=m.rater_models)` needs no conversion.

## `per_rater_model_selection` on proper fits

For rater r and candidate m: fit rater r at m with the other raters as set by the D2 candidate approach (jointly
for (a)–(c), held at the matrix fit for (d)), warm-started from the matrix fit. That gives R × 8 fits (less where a candidate
equals its parent, as now). Each rater's candidates are compared on the same data (all raters' observations,
since the fit is joint), by the rule in D3.

Cost: R × (number of candidates) fits.
- **`cp-full-rcl`:** about 0.02–0.1 s each, so trivial.
- **CML:** about 0.3–5 s each (matrix-sized fits are the slow ones). At R = 10 that is roughly 1–5 minutes; at R = 30,
  10–40 minutes. Progress reporting (built) covers it, with the long-run warning.

## Decisions for you

- **D1. Fit statistic for selection.**
  - (a) CML candidate fits scored by their conditional log-likelihood. This is a proper likelihood at its own
    penalised maximum, but slower.
  - (b) The object's estimator. Under `cp-full-rcl` that means scoring RCL fits with the conditional likelihood at
    the RCL estimates (plug-in): fast, but this is the plug-in drift seen before (estimates not at the
    likelihood's own maximum).
  - (c) As (b), but the composite LR test (Kent 1982, prototyped for RSM vs PCM, not yet reviewed).
  - Recommendation: (a) by default (`estimation='cml'` for the selection fits), with the object's estimator as an
    option. Note this departs from "refits use the object's own settings", deliberately and documented.
- **D2. Other raters while rater r's candidates are fitted.** Candidate approaches, all to be compared in V2
  (issues to be found as each is worked through):
  - (a) Joint refit with every other rater at matrix: one pass, every comparison nested in the same full model.
    Under the minimal identification, δ and τ are shared only by raters without the corresponding deviation
    terms, so with every other rater at matrix nobody shares them with rater r. 'global' for rater r then tests
    only whether r's own grid is additive, not whether r agrees with the others.
  - (b) Iterate: refit with the others at their current choices until no rater changes (coordinate-wise). This
    can find a better mixture, but costs more and its result can depend on order.
  - (c) Sequential, from restricted: start with every rater at the most restricted representation, so all raters
    define the common structure. Then relax one rater one level at a time (global → items / thresholds →
    bivector → matrix, and the stretch branch), refitting jointly at each step, as in the planned sequential misfit
    workflow. Each step is tested against the raters that still conform. The selection is data-dependent (the
    reference distribution must allow for it), and the result can depend on order.
  - (d) One pass, others fixed: hold every other rater at the matrix fit and fit only rater r's parameters, at
    each candidate's own conditional maximum, against a fixed reference structure (δ_i + τ_h from the matrix
    fit). 'Rater r is global' then means r's grid is the reference plus its severity, as in the uniform models. The
    others set the frame, so nothing beyond r's parameters needs identifying. All raters are tested against the
    same reference, so the result does not depend on order, and each fit is small. Issues:
    - The matrix fit's reference includes r with weight 1/R, which favours restrictions. A leave-one-out
      reference, built from the other raters' grids, avoids this, as the DRF anchors do.
    - The LR treats the fixed reference as exact.
    - The final `mixed` fit is still joint, under the minimal identification.
  - Recommendation: none until V2. Test (d) with a leave-one-out reference and with the full reference, (a), (c),
    and today's derived selection; (b) as an option.
- **D3. Decision rule:** AIC + significance test (the model_selection default), BIC, and the LR ladder, as now.
  The penalised fits make the LR's reference distribution approximate (as for Andersen under the ridge).
  Recommendation: keep all three; choose the default by the classification study (V2) and judge rules by
  worst-case correct-choice rate (your criterion).
- **D4. SEs for the joint mixed fit:** sandwich (analytic, as the other fits). The bootstrap refits the fixed
  mixture, so its SEs are conditional on the selection (documented); re-selecting in each resample is an
  option for later, as it multiplies the cost by R × 8.
- **D5. `sampling`, the person estimates and `warm_corr` / `tolerance` / `max_iters` / `ext_score_adjustment`**
  become unnecessary under the conditional likelihood. Recommendation: deprecate them (FutureWarning, ignored)
  rather than remove them.
- **D6. Raters missing from a `rater_models` dict:** default 'matrix', or raise an error. Recommendation: error,
  as explicit is safer.
- **D7. Breaking change:** `facet_effects_mixed` and `per_rater_model_selection` results change. This is fine for
  v2.0.0, with a CHANGELOG entry. Is the old derivation kept as an option (`method='derived'`) for a release?
  Recommendation: OPEN (user, 2026-10-27): for objects estimated with PAIR/CPAT the derived selection is the only
  route within their own estimator, so it may stay as a supported, documented-approximate option rather than be
  deprecated; alternative: CML refits for them too.

- **D8. `per_rater_model_selection`'s role:** a public method in its own right (a diagnostic of which raters
  depart from the common structure, and the input to `mixed`), or a helper of `calibrate(model='mixed')` only.
  Recommendation: public (user, 2026-10-27: agreed in discussion, with user-specified `rater_models` as the
  alternative to selection).

- **D9. Several selections and mixed fits on one object** (user, 2026-10-27). Today
  `per_rater_model_selection_runs` keys calls by `tuple(anchors)` only, so calls with the same anchors and
  different candidates overwrite each other, and a second `calibrate(model='mixed')` would overwrite
  `facet_effects_mixed`.
  - Each `per_rater_model_selection()` call returns and stores a result object (`rater_models`, `table`,
    `counts`, `settings`: anchors, candidate models, test, alpha, estimation) in
    `per_rater_model_selection_runs[run_name]`; `run_name=None` defaults to `run_n`.
  - Each `calibrate(model='mixed', ...)` / `calibrate_anchor('mixed', ...)` stores its fit in
    `mixed_fits[fit_name]`; `fit_name=None` defaults to `run_n`. With `store='full'` (default; user, 2026-10-27)
    the entry is a complete MFRM object: a snapshot of the object at the time of the fit, with its own copy of the
    data and without `mixed_fits` or selection runs of its own. Downstream work on that fit runs on the snapshot
    (`m.mixed_fits['panel A'].fit_statistics(model='mixed')`) and its results stay there; on the main object,
    `model='mixed'` means its latest fit. Sizes measured 2026-10-27: 0.4 MB calibrated / 2.0 MB with fit
    statistics (300 persons, 8 items, 8 raters), 4.6 / 35 MB (2,000 persons, 20 items, 10 raters), mostly
    category probabilities and residual matrices. `store='light'` keeps parameters only (plus SEs once computed,
    the `rater_models` used and the settings), for many fits or large data; how a light entry is used downstream
    (rebuilt into a full object on request, or parameters only) to be settled in the build.
  - Default names: n = number of stored entries + 1, moving to the next unused number if taken; reusing an
    explicit name replaces that entry with one UserWarning naming it.
  - `rater_models` also accepts a selection result object.
  - The top-level attributes (`rater_models`, `per_rater_model_selection_table`, `facet_effects_mixed`, ...)
    mirror the latest call, so existing code keeps working.
  - Open: whether the `anchor_*` attributes are absorbed into the stored settings (proposed: not in v2.0.0,
    kept compatible with the planned internal state with staleness tracking).

## Downstream

The probability engine already reads the full grid for 'mixed', so person estimates, fit statistics, the Wright map
and plots work once `facet_effects_mixed` is a real fit. To add or check:
- `std_errors(model='mixed')` (sandwich);
- `dif_test` / `drf_test` with 'mixed';
- `model_selection` including 'mixed' as a candidate against the uniform models: OUT of v2.0.0 (user, 2026-10-27);
  its own selection from the same data flatters its likelihood, which AIC / BIC do not count;
- `calibrate_anchor('mixed', anchors)`;
- recovery_stats against a mixed generator.

## Validation (before shipping)

- **V1. Recovery:** data from a known mixture. Compare parameter recovery for `mixed` with the true representations
  against the uniform matrix fit, across designs (crossed, double marking, crossed core). `mixed` should be at
  least as accurate and less variable.
- **V2. Classification:** per-rater correct-choice rate, worst case across generating representations, for D1 (a)
  against (b), the D2 candidates ((a), (c), (d) with a leave-one-out reference and with the full reference, and
  (b) where affordable), and each D3 rule, compared with today's derived selection, with over-complex
  and under-complex rates. Designs as the 2026-10-09 study (16 raters, a mixture of representations, s = 1 and
  0.5 effect sizes), plus double marking.
- **V3. Run time** for R = 5–30, CML and RCL.
- **V4. Equivalence tests:** a uniform `rater_models` reproduces each of the eight representation fits; the
  stretch / linear mixture with one rater of each type converges; SEs agree with the bootstrap.
- **Needs:** a mixture simulator. The agreed `manual_raters={rater: (model, params)}` format would go in research
  first; the simulation release is separate.

## Order of work

1. Grid design + penalties for a mixture.
2. CML fit, then the generalised RCL fit, including stretch raters.
3. V4 equivalence tests.
4. `calibrate(model='mixed')`.
5. `per_rater_model_selection` on proper fits.
6. SEs and downstream.
7. Validation V1–V3.
8. Docs.

Size: large, several weeks in the original estimate; the generic grid design makes it smaller than that, since
CML stretch and the SE machinery already work through such an interface.

## Decisions recorded (user, 2026-10-27)

- D4: sandwich SEs by default; bootstrap SEs conditional on the chosen mixture (documented); re-selecting in each
  resample later.
- D5: the person-estimation arguments of `per_rater_model_selection` (`sampling`, `warm_corr`, `tolerance`,
  `max_iters`, `ext_score_adjustment`) deprecated for the proper-fit routes; kept for the derived selection (D7).
- D6: raters missing from a user-given `rater_models` raise an error.
- D7: the derived selection stays as a supported option for objects estimated with PAIR/CPAT (their only route
  within their own estimator), documented as approximate (it scores derived, not fitted, parameters).
- Default `candidate_models`: all eight.
- `mixed` is not a `model_selection()` candidate in v2.0.0.
- Reporting frame: each rater relative to the common structure (see above); recentring only for reporting, one
  identification inside the fit.
- D1 follows the `model_selection` decisions (V2.0.0_SCOPE.md item 9), settled first on RSM vs PCM.
