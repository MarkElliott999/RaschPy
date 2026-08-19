# Changelog

All notable changes to RaschPy are documented in this file.

## [1.2.1] - 2026-08-19

### Changed
- **Performance: vectorized PAIR/CPAT pairwise-matrix construction** across RSM, PCM, and MFRM. All routines vectorised as far as possible, replaced (often nested) raw Python for-loops with matrix multiplication, following the approach already in place for SLM. Applies to PAIR estimation for items and raters in `_build_pairwise_matrix()` (RSM/PCM/MFRM item calibration) and `_estimate_raters_global()`/`_pair_matrix()` (MFRM rater estimation), and also to CPAT threshold estimation `_threshold_distance()`. Also eliminated redundant dataframe build calls. No API or behaviour change — verified numerically identical to the prior implementation. Performance improvement on small-scale testing showed speeding up of ~1.2x (global, larger designs) to ~2.6x (bivector); benefits `std_errors()` bootstrap loops most, since this involves repeated calibration calls, so may be second or even minutes rather than milliseconds there.

## [1.2.0] - 2026-08-06

### Added
- **Three new MFRM rater-severity parameterisations**, each a closed-form restriction of an existing fully-free representation — no new estimation machinery, derived directly from an already-calibrated fit:
  - `model="centrality"` (alias `"threshold_stretch"`) — Jin & Wang (2018)-style rater centrality/extremity: a shift `beta_r` plus a stretch `alpha_r` applied to the reference thresholds, restricting the `"thresholds"` parameterisation. Sits strictly between `"global"` (1 free param/rater) and `"thresholds"` (K free params/rater) on the model-selection ladder.
  - `model="pseudo_halo"` (alias `"item_stretch"`) — the item-facet analogue: a shift `beta_r` plus a stretch `gamma_r` applied to reference item locations, restricting `"items"`. Named `pseudo_halo` deliberately — it tests compressed item-difficulty *perception* under an otherwise locally-independent response process, not classic halo effect's usual mechanism (elevated local item interdependence); pair with `item_res_corr_analysis`/`facet_res_corr_analysis` run per-rater to check for that separately.
  - `model="bistretch"` — combines both stretch axes at once: a shift `beta_r`, an item-difficulty-stretch `gamma_r`, and a threshold-stretch `alpha_r`, restricting `"bivector"`. Sits strictly between `"global"` and `"bivector"` on the model-selection ladder — the parsimonious middle ground when a rater shows genuine structure on both axes simultaneously.
  - All three integrate at full parity with the original five representations: `calibrate()`/`calibrate_anchor()`, `fit_statistics()`, `std_errors()` (bootstrap SEs), every `*_stats_df()` table, `model_selection()` (AIC/BIC/LR, with LR degrees of freedom against both the parent model and `"global"`), and all plotting methods (`icc`, `crcs`, `threshold_ccs`, `tcc`, `test_info`, `test_csem`, `std_residuals_plot`, etc.) — the MFRM rater-parameterisation family is now eight representations, not five.
  - Matching simulation classes `MFRM_Sim_Centrality`, `MFRM_Sim_PseudoHalo`, `MFRM_Sim_Bistretch` for genuine parameter recovery testing.
- **`item_stretch`/`threshold_stretch` aliases** for `pseudo_halo`/`centrality` — axis-based names matching the `"items"`/`"thresholds"` and `"bistretch"` naming convention. Work everywhere a model name is accepted: as the `model=`/`models=` value, and as the method-name suffix (e.g. `calibrate_threshold_stretch()`) for every method family with a `_centrality`/`_pseudo_halo` variant. Resulting attributes are still stored under the canonical `_centrality`/`_pseudo_halo` names regardless of which spelling was used.
- **`model_selection(models=...)`** — restrict the eight-way comparison to a chosen subset (e.g. a direct two-model test) without paying the cost of calibrating the rest.
- **`per_rater_model_selection()` extended to all eight representations, plus a new `models=` parameter.** Under `test='LR'`, the existing top-down ladder (`matrix → bivector → {items, thresholds} → global`) now runs one additional targeted stretch test depending on which of those five wins, so any rater can be assigned any of the eight representations rather than just the original five. `test='AIC'`/`'BIC'` (AIC is the default) instead pick the minimum-criterion model directly across all active representations, with no ladder involved. `models=` restricts which representations are eligible — for `test='LR'`, `{'global', 'items', 'thresholds', 'bivector'}` must stay included as the mandatory backbone the ladder's fork needs, while `'matrix'` and each stretch model are independently toggleable; for `test='AIC'`/`'BIC'` it is a direct filter over any non-empty subset.

### Fixed
- **MFRM row-order canonicalisation** (the fix v1.1.1's own changelog described) had been silently reverted by the refactor commit that produced that release, and shipped broken in v1.1.1 — any `MFRM` where facet_elements (raters) rate non-overlapping person sets could crash (`IndexError`) or silently corrupt facet/rater severity estimates via a mis-shaped positional reshape. Restored and reverified: `check_data_connectivity()` and `calibrate()` both now behave correctly on ragged rater×person designs.
- `MFRM_Sim_Centrality`/`MFRM_Sim_PseudoHalo`/`MFRM_Sim_Bistretch` raised `TypeError: got multiple values for keyword argument 'manual_items'` if a caller passed `manual_items=`/`manual_thresholds=`/`manual_persons=`/etc. directly to the constructor — fixed.

## [1.1.1] - 2026-07-23

### Added
- **Differential Item Functioning (DIF) testing**: `dif_test()` on `SLM`, `PCM`, and `RSM` — tests every item for DIF against a person-level covariate (`self.exogenous`), with anchor purification (robust-z or Wald-based outlier trimming) before testing, per-item Wald and/or likelihood-ratio tests, an omnibus joint test, BH/Bonferroni multiple-comparison correction, and optional ETS-style A/B/C severity categories (Zwick, Thayer & Mazzeo, 1997). `PCM`/`RSM` additionally report differential category functioning (DCF) via `threshold_dif_table` — tested via category *widths* rather than raw threshold locations, since a single genuine width change otherwise cascades into every downstream threshold.
- `calibrate_anchor()` added to `SLM`/`PCM`/`RSM` for item-bank anchoring (equating a new form's items onto a bank's known difficulties), sharing the same robust anchor-selection machinery as DIF purification.
- `andersen_lr_test(split_by='exogenous', covariate=...)` on `SLM`/`PCM`/`RSM` — Andersen's (1973) likelihood-ratio invariance test generalised to split by any exogenous person covariate, not just a median split on ability/score. (Not available on `MFRM`, where `andersen_lr_test` remains disabled for an unrelated reason — see Fixed/known issues.)
- `welch=True` option on `dif_test()` (all three models) for Welch–Satterthwaite t-tests instead of z-tests when group sizes are small or unequal; `size_adjust=True` for Tristán (2006) sample-size-adjusted standard errors.
- New worked example: `Examples/DIF testing/DIF testing example.ipynb`.

### Fixed
- Andersen/omnibus DIF tests were over-rejecting under the null (~1.3–1.7x nominal Type I error) because the alternative-hypothesis log-likelihoods used each subgroup's own separately-estimated person locations instead of the pooled model's — fixed across `SLM`/`PCM`/`RSM`.
- `PCM`'s omnibus DIF test additionally under-charged its degrees of freedom for per-item threshold/step-structure freedom, causing much larger false-positive inflation (~60% rejection under the null); fixed via a properly-scoped `omnibus_scope='item'|'full'` option.
- `MFRM.__init__` now canonicalises person row order on construction — a positional-reshape bug could otherwise silently collapse rater severity estimates toward zero when an `MFRM` was built from a non-canonically-ordered person subset.

### Changed
- `category_counts_df` (SLM, PCM, RSM, MFRM) rewritten to take `persons=`, `items=` (and, for MFRM, `facet_elements=`) filters plus a `counts_name=` option to store multiple named results side by side, instead of always computing across the full dataset. `category_counts_item` is folded into this method and removed as a separate call.
- Internal naming conventions standardised across `base.py`, `slm.py`, `pcm.py`, `rsm.py`, `mfrm.py`, `loaders.py` — `ability`/`abilities`/`difficulty`/`difficulties` renamed to `person_location(s)`/`item_location(s)` (and related parameter names) for consistency, fixing several naming-related bugs along the way.
- Package `__init__.py` consolidated under `raschpy/` (previously partly under a stray top-level `RaschPy/`), with an explicit documented `__all__` public API surface.

### Documentation
- PDF manual fully audited and updated: every model chapter (SLM/PCM/RSM/MFRM) and every simulation class checked method-by-method against the code, with documentation gaps fixed and duplicate LaTeX labels resolved.

### Removed
- Legacy `raschpy/old stable files with old names/` duplicate source files.

## [1.1.0] - 2026-07-12

### Added
- **MFRM model selection**: `model_selection()` (global comparison across all 5 rater-severity parameterisations) and `per_rater_model_selection()` (top-down per-rater assignment, with an anchored variant), plus full downstream support for the resulting "mixed" model in `fit_statistics`, `std_errors`, and `rater_stats_df`.
- **Anchoring**: `RSM.calibrate_anchor()` added (ported from SLM); `PCM.calibrate_anchor()` rewritten from an untested legacy implementation to the modern SLM/RSM design, with a per-item threshold-structure diagnostic; `MFRM.check_anchor_homogeneity()` added to test whether proposed anchor raters agree with each other.
- `manual_raters` parameter on `MFRM_Sim` for specifying known rater severities under any of the five parameterisations.

### Fixed
- Anchor-run caching: repeated anchor-calibration calls with different anchor sets no longer silently overwrite each other or serve stale results to downstream fit statistics (MFRM and SLM).
- A real statistical flaw in the naive Cochran's Q approach to anchor-rater homogeneity testing (collateral flagging when one rater is a severe outlier), fixed via sequential exclusion.

### Changed
- Simulation classes: `manual_abilities` → `manual_persons`, `manual_diffs` → `manual_items`; "abilities" terminology replaced with "locations" throughout (more neutral for non-educational applications, e.g. healthcare).

### Publishing
- Published to PyPI: https://pypi.org/project/raschpy/
- Replication script and PEP 8 (black + flake8) compliance added.

## [1.0.0] - 2026-06-25

First production/stable release.

## [0.1.0] - 2026-06-15

Initial public release.
