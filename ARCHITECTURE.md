# RaschPy architecture

This is an orientation document for anyone working on RaschPy's source, not a
user-facing API reference (see `README.md` and the PDF manual for that). It
exists so a contributor — or a future session of Claude Code — doesn't have to
reconstruct the shape of the codebase from scratch each time.

## What this package is

RaschPy implements four Rasch measurement models — `SLM` (dichotomous),
`PCM` (partial credit), `RSM` (rating scale), and `MFRM` (many-facet, i.e.
rater-mediated) — plus a matching `*_Sim` simulation class for each, used for
methods research, teaching, and this project's own validation. All four model
classes follow the same workflow (see below) and share a common base class.

## Package layout

```
raschpy/
  __init__.py        public API surface (__all__), __version__
  base.py             Rasch (shared base class), _SimParams, _Namespace
  slm.py               SLM
  pcm.py               PCM
  rsm.py               RSM
  mfrm.py             MFRM  — by far the largest file; dominated by the
                       eight-way rater-parameterisation fan-out (see below)
  _rcl.py             RCL engine shared by RSM and MFRM ('cp-full-rcl' full pattern set, 'cpat-rcl'): CPAT pattern
                       counts, cell models (linear) and stretch models
                       (bilinear), penalised Newton-Raphson / L-BFGS fits,
                       sandwich covariances (see Estimation approach)
  loaders.py          CSV/Excel/JSON readers: loadup_slm, loadup_pcm,
                       loadup_rsm, loadup_mfrm_single, loadup_mfrm_xlsx_tabs,
                       loadup_mfrm_multiple, loadup_exogenous
  simulation/
    base_sim.py        Rasch_Sim (shared base class)
    slm_sim.py          SLM_Sim
    pcm_sim.py          PCM_Sim
    rsm_sim.py          RSM_Sim
    mfrm_sim.py         MFRM_Sim + 8 convenience subclasses
                         (MFRM_Sim_Global/Items/Thresholds/Matrix/Bivector/
                         Centrality/PseudoHalo/Bistretch)
tests/
  run_smoke_tests.py, run_regression_tests.py, run_edge_case_tests.py
                       the three original hand-rolled suites (see Testing
                       below for how they're actually invoked now)
  test_smoke.py, test_regression.py, test_edge_case.py
                       pytest wrappers around the three runners above,
                       parametrized per model
  test_known_issue_regressions.py
                       regression tests for specific bugs that have already
                       shipped broken once — see Known structural invariants
  regression_data/    committed fixtures the regression suite asserts against
  conftest.py         sys.path setup + shared test helpers
Examples/              worked notebooks: Model classes/ (one subfolder per
                       model class), Anchor calibration/, DIF testing/,
                       Model selection/, Simulations/, raschpy_addendum/,
                       Papers and PhD thesis studies/
```

`raschpy/` (lowercase) is the real, importable package. If you ever see a
capitalised `RaschPy/` directory, that's legacy and not part of the current
build.

## Estimation approach — this is the part that looks unusual

RaschPy does **not** use joint maximum likelihood (JML/JMLE), conditional
maximum likelihood (CML/CMLE), or marginal maximum likelihood (MML) for item
calibration. It uses **PAIR** (Choppin, 1968, 1985), the **eigenvector
method** (Garner & Engelhard, 2002, 2009), and **CPAT** (Elliott & Buttery,
2022a, 2022b) for thresholds.

Mechanically: `calibrate()` builds a directed pairwise-comparison matrix from
the raw response data (who scored above whom, on which item), resolves
structural zeroes via matrix powers, and extracts item/threshold locations
through a closed-form **priority-vector** computation
(`Rasch.priority_vector()`, with `method='log-lik'`/`'cos'`/`'ls'`/`'evm'`
choosing the extraction rule). This is linear algebra, not a converging
likelihood optimisation — item locations are never jointly iterated with
person abilities the way JMLE iterates them. Only person locations use
Newton–Raphson (`person()`), and that's a cheap, independent, per-person
1-D root-find, not a joint fit across the whole design.

This is *why* RaschPy is fast relative to JMLE or MCMC-based approaches — it
sidesteps joint (N+I)-dimensional iteration entirely by exploiting Rasch's
specific-objectivity pairwise identity (`matrix[j,i] / matrix[i,j]` is
directly a log-odds estimate of `b_i - b_j`), not by being a cleverer
iterative algorithm. Two consequences worth knowing before touching the
calibration code:

- **Don't describe PAIR/CPAT calibration as "iterative"** — it isn't, and this
  has been a recurring correction in past work on this repo. (RCL-CPAT, below,
  does use Newton–Raphson — but over the pairwise counts, not over persons.) The `constant`
  additive-smoothing parameter and `matrix_power` are finite-sample/structural
  bookkeeping, not iteration.
- **The stretch models (`centrality`/`pseudo_halo`/`bistretch`) are not more
  "closed-form" than the other five MFRM rater parameterisations** — every
  PAIR/CPAT model in this package is equally non-iterative. Don't single
  them out.
- Every `Rasch` subclass implements its own
  `_build_pairwise_matrix(self) -> (matrix, row_items)` — the raw,
  unsmoothed matrix `calibrate()` itself uses. `check_data_connectivity()`
  (on the shared base class) calls this to predict `calibrate()`'s real
  structural zeroes exactly, without duplicating each model's
  matrix-construction logic. This is the contract new model variants (or
  refactors) need to preserve.
- Standard errors are analytic by default where they can be: a sandwich
  (Huber–White, persons as clusters) for SLM and PCM, and for RSM and MFRM
  under RCL-CPAT. The bootstrap (`std_errors(se_method='bootstrap')`, or the
  automatic fallback with a warning: unused category, non-convergence,
  unidentified directions, designs too large for the dense bread, MFRM
  `pair-cpat`, `store_bootstrap=True`) is the only place a genuinely repeated
  resample-and-recalibrate computation happens, and is the slow path.

### RSM and MFRM: RCL (v1.4.0 default: 'cp-full-rcl', full pattern set; 'cpat-rcl' = CPAT patterns only)

`estimation='cp-full-rcl'` (default; sticky per object) and `'cpat-rcl'` fit item locations,
category widths and rater parameters **jointly**, as one penalised pairwise
composite likelihood over three kinds of CPAT comparison: item comparisons
within a rater, threshold patterns `(k,k)` vs `(k-1,k+1)` within a rater, and
rater comparisons within an item. Every comparison is a binomial in the
difference of two cells of the location grid `b[r, i, h]`, so the whole fit
works on pattern counts (`_rcl.pattern_counts`), never on persons. Cell models
(global, items, thresholds, bivector, matrix) are linear in orthonormal-basis
coordinates (`CellModel`; concave objective, Newton–Raphson, L-BFGS-B above
`DENSE_LIMIT`); stretch models are bilinear (`StretchModel`; Newton with Fisher
scoring fallback, two starts from the `pair-cpat` fit). Smoothing constants
are pseudo-counts spread over the cells; ridges are `ridge` (0.3),
`interaction_ridge` (3, matrix shrunk towards bivector) and `stretch_ridge` (1,
on `omega - 1`). Thin widths get a ridge towards the other widths; widths next
to an empty category are split out and set by the cpat-cml rule. RSM is the
case R = 1. `estimation='pair-cpat'` keeps the published separate stages
(items → thresholds → raters) and reproduces v1.3.1 for RSM. Full derivation:
`methods_writeup_v1.4.0.docx`, Part C.

## Class hierarchy

```
Rasch (base.py)              Rasch_Sim (simulation/base_sim.py)
 ├── SLM (slm.py)             ├── SLM_Sim
 ├── PCM (pcm.py)             ├── PCM_Sim
 ├── RSM (rsm.py)             ├── RSM_Sim
 └── MFRM (mfrm.py)           └── MFRM_Sim
                                   ├── MFRM_Sim_Global
                                   ├── MFRM_Sim_Items
                                   ├── MFRM_Sim_Thresholds
                                   ├── MFRM_Sim_Matrix
                                   ├── MFRM_Sim_Bivector
                                   ├── MFRM_Sim_Centrality
                                   ├── MFRM_Sim_PseudoHalo
                                   └── MFRM_Sim_Bistretch
```

`Rasch` provides data-connectivity validation, `priority_vector()`,
point-measure correlation, rename utilities, and the shared plotting
infrastructure (`_PLOT_STYLE_RC`, `std_residuals_hist`,
`plot_anchor_selection`, `wright_map`). It is not meant to be instantiated
directly.

`wright_map` deduplication is complete (2026-09-19, remediation-plan item
15): `SLM`/`PCM`/`RSM`/`MFRM` all now inherit one `Rasch.wright_map`. PCM's
and RSM's two copies were confirmed byte-identical executable code (already
written polymorphically via `isinstance(self.thresholds, pd.Series)`);
SLM (dichotomous, no threshold structure) folded in as a subtractive case,
guarded by an explicit `ValueError` when `item_level='thresholds'`/
`item_strip=True` is requested on a model with no `self.thresholds`. `MFRM`
(facet-element panel, per-`model=` dispatch, stretch panels) was folded in
last, using a `has_facets = hasattr(self, "facets")` discriminator (not an
`isinstance` check, so any future facet-element model class is picked up
automatically) that gates every facets-dependent computation — not just the
panel/axis creation the original scoping anticipated, but scattered
touchpoints throughout (plot-range/panel-sizing math, the threshold-matrix
`cat_prob`/`exp_score` dispatch, which needed a genuine three-way branch
since MFRM's own `cat_prob`/`exp_score` have an incompatible rater-aware
signature and keeps its own local closures). `model=None` is the shared
default; a facet-element model requires a real value, a non-facet model
(SLM/PCM/RSM) must leave it `None`, both enforced by explicit `ValueError`s
alongside similar guards on `facet_labels`/`overlay_facet`. See the
`raschpy_wright_map_dedup` project memory for the full scoping and
implementation record (all three phases, including the mid-merge design
pivot and a lost-guard bug caught by the smoke suite).

`calibrate_anchor` and `dif_test` deduplication (2026-09-20/21, remediation-
plan item 15's other two pieces) follow the same template-method pattern:
each subclass's own method stays, but the ~150-line shared block moves onto
`Rasch`. `_calibrate_anchor_core()`/`_snapshot_calibrate_anchor_run()`
extract the selection/translation-constant/`overwrite_anchors`/summary/plot
block common to `SLM`/`PCM`/`RSM` (the `robust=`/`threshold_constant=` kwarg
split is papered over by a `std_errors_kwargs` dict the caller builds).
`_dif_validate()`/`_dif_item_wald_block()`/`_dif_item_table()` extract the
argument validation, item-location leave-one-out purification/Wald test,
and `item_table` construction common to all four classes including `MFRM`
(which additionally builds its own `ref_boot`/`focal_boot` matrices before
calling the shared block, since it stores bootstrap resamples per rater
representation rather than under one fixed attribute name). In both cases
the parts that are genuinely model-specific — PCM's per-item threshold-
compatibility test, PCM/RSM/MFRM's own DCF block, the per-item LR test's H0
construction (a single reassigned location for SLM/RSM/MFRM vs. PCM
rebuilding the item's whole uncentred threshold row), and each class's own
omnibus test — stay in the subclass; only the byte-identical boilerplate
moved. Neither extraction changes output: both were verified against the
pinned regression fixtures before being trusted.

The eight `MFRM_Sim_*` subclasses are thin wrappers (~15 lines each) around
`MFRM_Sim` with `model=` pre-selected — they don't reimplement generation
logic, and most of them delegate their own docstring back to `MFRM_Sim`
("See MFRM_Sim for full parameter docs") rather than repeating it.

## The workflow (all four model classes)

```python
model = SLM(responses)      # or PCM/RSM/MFRM, or pass a *_Sim object directly
model.calibrate()           # item/threshold locations via PAIR/CPAT
model.fit_statistics()      # infit/outfit, person/item/test fit; auto-triggers
                             #   calibrate() and std_errors() if not already run
model.item_stats_df()       # → model.item_stats
model.person_stats_df()     # → model.person_stats
model.test_stats_df()       # → model.test_stats
model.icc(item=...)         # plots: icc, crcs, threshold_ccs, iic, tcc,
                             #   test_info, test_csem, std_residuals_plot,
                             #   wright_map
```

Every downstream method (`fit_statistics`, `*_stats_df`, `std_errors`, the
plotting methods) lazily auto-triggers whatever it depends on if it hasn't
been run yet — you never have to remember to call `calibrate()` before
`item_stats_df()`. This lazy-auto-calibrate convention is load-bearing: when
adding a new method, match it rather than requiring the caller to sequence
calls manually.

Passing a `*_Sim` object directly to a model constructor (`SLM(sim)`) attaches
the simulation's generating parameters as `model.generating`, which the
`recovery_stats()`/`_recovery_plot` machinery and the example notebooks use
to compare estimated vs. true values.

## MFRM's rater-parameterisation scheme

MFRM is the architecturally distinctive part of this package: rater severity
can be represented eight different ways, not just one, reflecting how much
structure a given rater's behaviour actually needs:

| model | rater severity is... | free params/rater |
|---|---|---|
| `global` | one scalar, applied uniformly | 1 |
| `centrality` (`threshold_stretch`) | a shift + stretch on `thresholds` | 2 |
| `pseudo_halo` (`item_stretch`) | a shift + stretch on `items` | 2 |
| `thresholds` | a full vector by threshold | K |
| `items` | a full vector by item | I |
| `bistretch` | a shift + item-stretch + threshold-stretch | 3 |
| `bivector` | independent item vector + threshold vector | I+K-1 |
| `matrix` | a full (item × threshold) cell grid | I×K |

`centrality`/`pseudo_halo` are restricted forms of `thresholds`/`items`;
`bistretch` is a restricted form of `bivector`. `model_selection()` compares
all eight for the whole rater pool at once; `per_rater_model_selection()`
assigns each rater individually (via AIC/BIC directly, or an LR testing
ladder), producing a **ninth**, derived representation: `model='mixed'`.

`'mixed'` is not a real calibration — it has no `calibrate(model='mixed')`
and no `calibrate_mixed()`. Every rater's `'mixed'` values are a restriction
of a single `'matrix'` calibration to whichever of the eight representations
`per_rater_model_selection()` assigned that specific rater
(`_apply_rater_models()`). This is why `'mixed'` needs special-casing
throughout `mfrm.py` (`_sync_mixed()`, `cal_model` substitution in
`std_errors()`/`anchor_std_errors()`) rather than being just another entry
in the five-model dispatch — it has no calibration of its own to lazily
trigger, and its per-rater assignment is not anchor-invariant the way
item/person locations are (a different anchor set can give a different, still
valid, per-rater assignment). `model_selection()` and
`per_rater_model_selection()` drop sub-models that the design makes identical
(`_drop_equivalent_models`: centrality = thresholds when `max_score=2`,
pseudo_halo = items with 2 items, and more for dichotomous data). If you're extending `'mixed'` support to a new
method, check `mfrm.py`'s existing `model in (...)` / `model ==` dispatch
sites first — several intentionally don't support it yet (see
`raschpy_known_issues.md`-equivalent notes / the remediation plan for the
current list).

## Known structural invariants — read before touching these

**Per-model storage (MFRM, v1.4.0).** Under RCL-CPAT each sub-model has its own
item locations and thresholds: `items_{model}`, `thresholds_{model}`,
`item_se_{model}` (anchored: `anchor_items_{model}`, ...). The unsuffixed
`self.items`/`self.thresholds` are only the *latest* calibration. Model-specific
code must read them via `_items_for(model, anchor=...)` /
`_thresholds_for(...)` / `_item_se_for(...)` — and anchored code must pass
`anchor=True`: anchoring moves the anchors' mean into the item locations and
thresholds, so mixing anchored facet effects with unanchored locations shifts
every rater by the anchors' mean pattern (this broke anchored
`per_rater_model_selection()` until v1.4.0). Refits on new objects (bootstrap
resamples, DIF groups) must carry the estimation settings via `_refit_kw(model)`.

**Don't keep large arrays on the object.** RCL fits (`_rcl_fits[model]`,
RSM `_rcl_fit`) store only the comparisons (`info["_lazy"]`); the dense
Jacobian and bread the sandwich needs grow with the square of the parameter
count, so `_rcl.cell_state()` builds them on demand and `_rcl.release_state()`
drops them after use. Bootstrap resamples are generated lazily
(`_bootstrap_samples` is a generator) and `_resamples()` strips each one's
responses and fit arrays once the loop moves on, unless `store_bootstrap=True`
(DIF/DRF and the divergence test refit on them). Code that reads resample
attributes after the bootstrap loop may only use estimates. Resamples are built
with `_keep_empty=True` (fixed facet element set) and set `_resample`, which
suppresses data-quality warnings that resampling itself creates.

**Numerical guards in the RCL engine.** Pattern counts use batched BLAS matrix
products (not `numpy.einsum`, ~30× slower) with float32 0/1 indicators, exact
while counts stay below 2**24 persons. The stretch fit's Newton step
(`_determined_step`) takes no step along directions the Hessian does not
determine (e.g. the shift between rater groups never compared): a plain solve
there returns arbitrary, possibly enormous steps. `null_directions()` uses the
same 1e-9 relative cut-off to set those estimates' standard errors to NaN.

**MFRM's canonical row grid.** `item_diffs()`/`_estimate_raters_*()` read
`self.responses.values` via a raw positional
`.reshape(no_of_facet_elements, no_of_persons, -1)`. This is only correct if
the physical row order is laid out as contiguous per-rater blocks, each in
*identical* person order — nothing else in the class enforces this. Because
of that, `MFRM.__init__` rebuilds `self.responses` immediately after
`person_names`/`facet_names` are established:

```python
self.responses = pd.concat(
    {fe: self.responses.xs(fe).reindex(self.person_names) for fe in self.facet_names}
)
```

This has been silently reverted once already by an unrelated large refactor
and shipped broken in a released version before being caught and re-fixed —
if you ever see a large mechanical diff touch `MFRM.__init__`, re-verify this
line survived. `tests/test_known_issue_regressions.py` has a regression test
for exactly this (shuffles one rater's row order and checks severity
recovery doesn't collapse).

**Anchor-run caching.** A second `calibrate_anchor()` /
`per_rater_model_selection()` call with a *different* anchor set must not
silently overwrite the first. SLM/PCM/RSM/MFRM each snapshot every anchored
run into a dict keyed by anchor identity
(`calibrate_anchor_runs[tuple(sorted(anchors...))]` on SLM/PCM/RSM;
`per_rater_model_selection_runs[tuple(anchors) or None]` on MFRM), so old
and new runs both stay retrievable. MFRM additionally has
`_anchors_stale(model, anchors)` — a read-only, side-effect-free check used
by `fit_statistics`/residual-correlation methods to invalidate their own
downstream caches correctly. `_ensure_calibrated()` itself is *not*
side-effect-free (it mutates `anchor_rater_names_{model}`), so don't call it
as a cheap pre-check before delegating to another method that also calls
it — the second call will see no mismatch and skip its own invalidation. Use
`_anchors_stale` for read-only checks instead.

**Simulation seeding.** Every `*_Sim` class takes its own `seed=None`
constructor argument and internally does
`self._rng = np.random.default_rng(seed)` — a local `Generator`, completely
independent of the global `numpy.random.seed()` state. Calling
`np.random.seed(N)` before constructing a sim has **no effect** on its
output. Always pass `seed=` directly for a reproducible run — this has
caused real cross-session confusion (results looking different between two
"identically seeded" runs that weren't actually seeded the same way).

**The editable-install footgun.** `import raschpy` resolves via whatever's
first on `sys.path`. Running `python tests/run_smoke_tests.py` (etc.)
directly inserts only the `tests/` directory onto `sys.path`, *not* the repo
root — so without an editable install (`pip install -e ".[dev]"`), it can
silently test a stale `pip`-installed copy instead of your local edits, and
report a clean pass while verifying nothing. This has bitten real sessions
more than once, including regressing back after being fixed once already.
CI (`.github/workflows/tests.yml`) always installs editable fresh, and
`tests/conftest.py` inserts the repo root onto `sys.path` as a local
fallback — but if you're running `python tests/run_*.py` directly rather
than through `pytest`, confirm `raschpy.__file__` points at this checkout
first. See `CONTRIBUTING.md`.

## Testing

Three suites exist, each covering all four model classes (plus MFRM's eight
rater representations where relevant):

- **Smoke** (`run_smoke_tests.py`) — simulate → instantiate → calibrate →
  every stats table → every plot method, checking for crashes and obviously
  malformed output (empty, all-NaN).
- **Regression** (`run_regression_tests.py`) — asserts numeric output against
  committed fixtures in `tests/regression_data/` (`pair-cpat`, plus SLM/PCM)
  and `tests/regression_data/rcl/`, `tests/regression_data/cpat-rcl/` (RCL for RSM and MFRM); the pytest
  wrapper is parametrised over (model, estimation). Regenerate with
  `python tests/run_regression_tests.py --generate [--estimation rcl|cpat-rcl|pair-cpat]`
  when a change intentionally alters output, then commit the new fixtures.
- **Edge case** (`run_edge_case_tests.py`) — deliberately awkward inputs
  (near-empty data, disconnected sub-networks, extreme scores). Reports
  `pass`/`fail`/`xfail` — an `xfail` is a known, currently-expected failure
  the runner deliberately records rather than hiding.

All three are wrapped for `pytest` (`tests/test_smoke.py`,
`test_regression.py`, `test_edge_case.py`), parametrized per model via each
runner's own `RUNNERS` dict, so `pytest` from the repo root runs everything
with no extra flags. `tests/test_known_issue_regressions.py` is a fourth,
separate file specifically guarding bugs that have already shipped broken at
least once (see Known structural invariants above) — when you fix a bug that
was silent/undetected the first time, add a test here, not just to smoke or
regression, so the *specific* failure mode is pinned down, not just "some
check somewhere still passes."

`RaschPy_remediation_plan.docx` (repo root) tracks known weaknesses and an
agreed priority order for addressing them, if you're looking for what's
already flagged as needing work.
