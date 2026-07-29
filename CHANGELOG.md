# Changelog

All notable changes to RaschPy are documented in this file.

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
