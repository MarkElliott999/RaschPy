# v1.2.0 TODO — bistretch MFRM model

**Renamed from `TODO_v1.1.2.md` 2026-08-06** — this release adds three
entirely new public MFRM rater parameterisations plus their sim classes,
new `models=` parameters on both selection methods, and a new alias
mechanism, all backward-compatible additions. Per semver that's a MINOR
bump, not a PATCH — see `CHANGELOG.md`'s `[1.2.0]` entry. `pyproject.toml`
and the manual title page updated to match; `meta.yaml` deliberately left
at 1.1.1 since it tracks the last actually-*published* PyPI release (its
`sha256` is tied to that real tarball) and should only move to 1.2.0 once
this version is actually published.

Status as of 2026-08-05. `centrality` and `pseudo_halo` are fully done
(core, sim, tests, notebooks, README, manual). `bistretch` core
estimation, full alias parity, and its sim class are done and verified;
the items below remain.

- [x] **Wire `bistretch` into `model_selection()`** — done 2026-08-05.
      Added `rater_k = 3*(R-1)`. Added LR nested-pairs/df entries:
      `bivector` vs `bistretch` (df = I+K-4), `centrality` vs `bistretch`
      (df = 1), `pseudo_halo` vs `bistretch` (df = 1), `global` vs
      `bistretch` (df = 2). Verified live on bistretch-generated data:
      AIC/BIC/LR all correctly select `bistretch`; smoke suite clean.

- [x] **Rework `per_rater_model_selection()`'s ladder** — done 2026-08-05.
      Implemented the agreed simplified design: existing ladder
      unchanged, then one targeted stretch test per branch (matrix→none,
      bivector→bistretch, thresholds→centrality, items→pseudo_halo,
      global→centrality AND pseudo_halo with a bistretch tie-break when
      both win). New `_derive_stretch_single` helper added for the
      per-rater numpy-array case. `_apply_rater_models` (used by bootstrap
      SEs for the "mixed" model) extended to reconstruct the 3 new
      per-rater assignments too. Verified against Global/Matrix/Items/
      Thresholds/Bistretch-generated data — each correctly recovers its
      own generating structure. Smoke (435/435) and edge-case (86/86)
      suites both clean.

- [x] **Test suite coverage for `bistretch`** — done 2026-08-05.
      Added `('Bistretch', 'bistretch', MFRM_Sim_Bistretch)` to the
      `MFRM_MODELS` list in `run_smoke_tests.py`, `run_regression_tests.py`,
      and `run_edge_case_tests.py` (7 entries total in this generic-loop
      list — `bivector` is deliberately excluded, tested separately via
      its own dedicated `run_bivector`/`build_bivector` functions with
      extra parameter-recovery/matrix-consistency checks, same as before
      this session). Generated `tests/regression_data/bistretch/`
      fixtures. Regenerating the full `--model mfrm` fixture set confirmed
      zero drift in the git-tracked `global`/`items`/`thresholds`/`matrix`
      fixtures (bit-for-bit identical `git diff --stat`). Full regression
      suite: 274/274 passed (up from 238 baseline, +36 new bistretch
      checks). `centrality`/`pseudo_halo`/`bistretch` regression fixture
      dirs are still untracked — nothing has been committed yet.

- [x] **Bistretch worked-example notebook** — done 2026-08-05.
      `Examples/Model classes/Extended MFRM bistretch/Bistretch MFRM
      example.ipynb` (99 cells, real executed output, zero errors).
      Primary data generated under `MFRM_Sim_Bivector` (bistretch's
      actual parent, matching centrality/pseudo_halo's own precedent),
      plus a bonus genuine-recovery section using `MFRM_Sim_Bistretch`
      (beta/alpha/gamma r=0.996/0.974/0.896 on this seed). Anchor section
      checks both item AND threshold locations (bistretch adjusts both
      axes, unlike its 2-param siblings).

- [x] **Three missing `MFRM_Sim_*` example notebooks** — done 2026-08-05,
      not originally on this list (caught mid-session: these were never
      built for centrality/pseudo_halo either). Added
      `Examples/Simulations/MFRM_Sim_Centrality.ipynb`,
      `MFRM_Sim_PseudoHalo.ipynb`, `MFRM_Sim_Bistretch.ipynb`, adapted
      from the `Thresholds`/`Items`/`Bivector` templates. Shipped as
      clean source with no baked-in output, matching every other
      notebook already in that folder (verified separately, via a
      throwaway execution pass, that all three run with zero errors).

- [x] **`per_rater_model_selection(models=...)`** — done 2026-08-05, not
      originally on this list (user asked after sanity-checking
      `model_selection()`'s own `models=` parameter). `{global, items,
      thresholds, bivector}` are a mandatory backbone for `test='LR'`
      (raises `ValueError` if missing); `matrix` and each stretch model
      are independently toggleable. AIC/BIC has no backbone requirement
      — direct filter over any non-empty subset. Verified across 6
      scenarios (full ladder, backbone-only, backbone+one-stretch-model,
      missing-backbone error, AIC without `global`, AIC restricted to
      just the two 2-param stretch models). Smoke (488/488) and
      edge-case (95/95) suites clean.

- [x] **Real bug fix (found in passing): `manual_items`/etc. broken for
      all three stretch-model sim classes** — done 2026-08-05.
      `MFRM_Sim_Centrality`/`PseudoHalo`/`Bistretch` all crashed with
      `TypeError: got multiple values for keyword argument 'manual_items'`
      if a caller passed `manual_items=`/`manual_thresholds=`/
      `manual_persons=`/etc. directly — these landed in `**kw` and
      collided with the same keyword already set explicitly in the
      internal `super().__init__()` call. Fixed in all three classes
      (`raschpy/simulation/mfrm_sim.py`) by popping the conflicting keys
      out of `kw` before that call. Verified fixed for all three; no
      regressions.

- [x] **README.md update** — done 2026-08-05.
      Added a `bistretch` section (identification explanation,
      calibration snippet, genuine-recovery sim example) matching the
      `centrality`/`pseudo_halo` sections' depth. Updated the top-level
      model count (7→8), the MFRM parameterisation table, the
      `model_selection()` section (LR df formulas + `models=` example),
      the `per_rater_model_selection()` description (new ladder-plus-
      extra-test design), and the simulation imports. Every new snippet
      verified to actually run.

- [x] **Manual update (the big job)** — done 2026-08-05.
      Manual lives at `~/Documents/RaschPy_manual/` (copied from
      `~/Downloads/` for sandbox access this session). Discovered the
      manual predates ALL THREE new models, not just bistretch (an
      earlier "centrality/pseudo_halo manual — done" note in this same
      TODO was wrong). Done so far, all in `mfrm.tex`/`mfrm_sim.tex`:
      - `calibrate` section: all 3 models added in full (Usage,
        Arguments, Returns, worked examples with cross-referenced
        beta/alpha/gamma numbers, calling examples).
      - `model_selection`/`per_rater_model_selection`: fully rewritten
        for 8 models, including the new `models=` parameter on both
        (with the LR backbone-requirement logic spelled out).
      - ~10 other model-generic spots swept (cat_prob/exp_score/
        variance/kurtosis, calibrate_anchor, various model= value
        lists, category_stats_df, rater_stats_df, save_stats).
      - Three full new `mfrm_sim.tex` sections: `MFRM_Sim_Centrality`,
        `MFRM_Sim_PseudoHalo`, `MFRM_Sim_Bistretch` (Generating +
        Customising subsections each, matching existing depth).
      - Structural balance (braces/tables/equations, duplicate labels)
        verified clean throughout.
      **Done since (2026-08-05, same day):** swept all 17 further
      sections found to need the same "5→8" treatment beyond the
      original 6 stats tables: `calibrate_anchor`, `std_errors` (correct
      rows only — see deferral below), `anchor_std_errors`,
      `person_estimates`, `score_lookup`, `score_lookup_table`, `csem`,
      `item_stats_df`, `threshold_stats_df`, `category_stats_df`,
      `rater_stats_df`, `person_stats_df`, `test_stats_df`,
      `item_res_corr_analysis`, `rater_res_corr_analysis`, `save_stats`,
      `save_residuals_items`/`save_residuals_raters`, `icc`, `crcs`,
      `threshold_ccs`, `iic`, `tcc`, `test_info`, `test_csem`,
      `std_residuals_plot`. Also fixed a genuine pre-existing gap in
      `rater_res_corr_analysis` (missing `bivector` in 4 of its 5
      Returns tables) found while extending it. Structural balance
      (115/115 tables, 115/115 tabular, braces 11794/11794, 0 duplicate
      labels) verified clean after the full sweep.
      **Done since (2026-08-05, same day):** fixed the `std_errors`/
      `anchor_std_errors` bootstrap-attribute documentation bug
      (pre-existing, predates v1.1.2, affects all models). Removed the
      confirmed-nonexistent `item_bootstrap`/`threshold_bootstrap`/
      `rater_bootstrap` rows; unsuffixed `item_se`/`item_low`/
      `item_high` (and their `anchor_` equivalents) to match the actual
      code (central item locations are shared across representations,
      not per-model); corrected `cat_width_bootstrap`/`cat_width_se`'s
      documented type from "pandas dataframe/series" to "dict" (matches
      code). Added a new "Retrieving the raw bootstrap replicas"
      paragraph at the end of `std_errors`'s Returns section explaining
      that `self._bootstrap_samples_{model}` (when `store_bootstrap=
      True`) is a list of fully-calibrated `MFRM` replicas, each
      carrying the full complement of that model's own attributes
      (`.items`, `.thresholds`, `.facet_effects_{model}`, model-specific
      `beta_`/`alpha_`/`gamma_` params, `anchor_`-prefixed if anchored)
      — giving direct access to every raw per-sample value with no
      separate bootstrap-array attribute needed. `anchor_std_errors`
      points back to this paragraph, since it mutates the same stored
      replica list in place on its fast path. Structural balance
      (114/114 tables, 114/114 tabular, braces 11689/11689, 0 duplicate
      labels) verified clean.
      **Done since (2026-08-05, same day):** fixed a structural bug
      found by the user — `MFRM_Sim_Centrality`/`PseudoHalo`/`Bistretch`
      had been added as `\subsection`s nested inside `mfrm_sim.tex`'s
      `MFRM_Sim_Bivector` `\section`, rendering as if they were part of
      the Bivector class rather than their own classes. Split each out
      into its own file — `mfrm_sim_centrality.tex`,
      `mfrm_sim_pseudo_halo.tex`, `mfrm_sim_bistretch.tex` — each a
      proper `subfiles`-package subfile with its own
      `\section{class MFRM_Sim_X}` wrapping the existing
      Generating/Customising subsections (content unchanged, just
      correctly promoted and relocated). `mfrm_sim.tex` trimmed back
      down to the original 5 classes (Global/Items/Thresholds/Matrix/
      Bivector). `main.tex` updated with 3 new `\subfile{}`+`\newpage`
      entries after `mfrm_sim`, in class order. Full-manual static
      re-check after the split: 0 environment mismatches, 0 unbalanced
      braces, 0 duplicate labels, 0 unresolved `\ref`/`\eqref` (543
      total), 0 missing `\cite` keys (478 total) across all 14 files.
      **User confirmed a real Overleaf compile of the pre-split manual
      was clean** — this was a structural/organisational fix caught by
      the user, not a compile error.
      **Still remaining:** none — the manual update task is complete.

- [x] **`item_stretch`/`threshold_stretch` aliases for `pseudo_halo`/`centrality`**
      — done 2026-08-05, requested after the manual/file-split work
      above. Axis-based names matching `items`/`thresholds` and
      `bistretch`'s own naming.
      - `raschpy/mfrm.py`: added `_MODEL_ALIASES = {"threshold_stretch":
        "centrality", "item_stretch": "pseudo_halo"}` next to `_MODELS`.
        Added a generic `__getattr__` fallback that resolves any
        `X_threshold_stretch`/`X_item_stretch` method-name lookup to the
        matching `X_centrality`/`X_pseudo_halo` method (covers all 46
        method families' dispatch aliases automatically, no per-method
        duplication — confirmed only triggers on attribute-lookup
        failure, so it can't shadow a real attribute or break
        deepcopy/pickle, which look for dunder methods that never match
        the suffix check). Inserted `model = self._MODEL_ALIASES.get(
        model, model)` as the first statement in all 33 public methods
        that take a `model=` parameter (`calibrate`, `calibrate_anchor`,
        `std_errors`, `anchor_std_errors`, `item_stats_df` and the other
        5 stats tables, `icc`/`crcs`/`threshold_ccs`/`iic`/`tcc`/
        `test_info`/`test_csem`/`std_residuals_plot`, `cat_prob`/
        `exp_score`/`variance`/`kurtosis`, `warm`, `csem`,
        `score_lookup`/`score_lookup_table`, `person`/`person_estimates`,
        `category_probability_dict`, `fit_statistics`,
        `andersen_lr_test`, `check_anchor_homogeneity`, `save_stats`),
        and the equivalent list-comprehension normalization in the 2
        methods taking `models=` (`model_selection`,
        `per_rater_model_selection`). Both insertion points identified
        via an AST pass (finds each method's first statement after its
        docstring) so normalization always lands before any existing
        `_MODELS` validation.
      - Verified: `mfrm.calibrate(model='threshold_stretch')`,
        `mfrm.calibrate_threshold_stretch()`,
        `mfrm.model_selection(models=['global', 'item_stretch'])`, and
        `mfrm.item_stats_df_threshold_stretch()` all work and produce
        results identical to the canonical `centrality`/`pseudo_halo`
        calls, still stored under the canonical `_centrality`/
        `_pseudo_halo`-suffixed attribute names. Full test suite clean
        after the change: smoke 488/488, edge-case 95/95, regression
        274/274.
      - `README.md`: model table rows now show the alias; new covering
        paragraph explaining the alias mechanics; one-sentence mention
        in the centrality/pseudo_halo section intro.
      - `mfrm.tex`: light touch per request — one paragraph in
        `calibrate` explaining the aliases (not a full duplicate
        Usage/Returns block), plus one-clause mentions added to the
        `models` argument row in both `model_selection` and
        `per_rater_model_selection`. Structural balance re-verified
        clean (114/114 tables, braces 11742/11742, 0 duplicate labels).
      - Notebooks: added a markdown+code cell pair to both `Centrality
        MFRM example.ipynb` and `Pseudo halo MFRM example.ipynb`
        (right after the existing `calibrate_centrality()`/
        `calibrate_pseudo_halo()` cell) demonstrating the alias method
        call and confirming its results are bit-identical to the
        canonical call. Both notebooks re-executed in full (two-pass
        placeholder-path technique, `python3` Jupyter kernel — the
        notebook's own `conda-base-py` kernel metadata doesn't exist in
        this environment) to bake real output; zero errors, 93 cells
        each (91 original + 2 new).

- [x] **`beta` → `lambda` rename** — done 2026-08-06, user's own request
      ("I use beta for person abilities... in line with the fact that I
      use lambda for severity parameters"). Scope: only the
      severity-shift parameter (`beta_r` → `lambda_r`) in
      `centrality`/`pseudo_halo`/`bistretch`; `alpha`/`gamma` untouched
      (gamma "deserves its own Greek letter").
      - `raschpy/mfrm.py`/`raschpy/simulation/mfrm_sim.py`: renamed all
        public attributes (`beta_centrality`→`lambda_centrality`, incl.
        `anchor_` variants and all 3 models), `manual_beta`→
        `manual_lambda` (sim constructor kwarg), sim classes' bare
        `self.beta`→`self.lambda_` (trailing underscore — `lambda` is a
        reserved keyword, `self.lambda` is invalid syntax), and internal
        scratch variables `beta`→`lam` in the shared
        `_derive_stretch_model`/`_derive_stretch_single` helpers and
        `per_rater_model_selection`'s ladder code (same keyword
        restriction). Verified via direct calibration + the
        `threshold_stretch`/`item_stretch` aliases (unaffected, since
        `__getattr__` works on any attribute name generically).
      - `README.md`, `mfrm.tex`, `mfrm_sim_centrality.tex`/
        `mfrm_sim_pseudo_halo.tex`/`mfrm_sim_bistretch.tex`: same rename.
        Checked first whether `\lambda_r` collided with the manual's
        pre-existing `\lambda_r`/`\lambda_{ri}`/`\lambda_{rk}` notation
        (already used chapter-wide for "rater's facet_effect term," incl.
        `global`'s own single-subscript `\lambda_r`) — turned out to be a
        natural fit, not a clash, since centrality/pseudo_halo/bistretch
        collapse to `global` at neutral stretch. Manual structural
        balance re-verified clean throughout.
      - Both "Model classes" worked-example notebooks (Centrality/Pseudo
        halo/Bistretch) re-executed in full (needed, not just a source
        edit — one cell builds a `pd.DataFrame` with a literal `'beta
        (severity)'` column-header string). Bistretch's
        recovery-correlation loop needed real restructuring (sim-side
        attribute `lambda_` vs mfrm-side `lambda_bistretch` don't share
        a naming pattern the old single-variable loop could drive
        anymore). Both "Simulations" notebooks (Centrality/PseudoHalo/
        Bistretch) source-only updated + verified via a throwaway
        execution pass (ships with zero baked output, per existing
        convention) — cleaned up the stray CSV files that pass left
        behind as a side effect.
      - Re-executing the three worked-example notebooks reintroduced the
        loose-file duplication into `output files/` fixed earlier this
        session (notebooks still don't `chdir` into `output files/` by
        design — user's own call: "I'm not worried about save paths...
        the output files stored are for reference only"). Redid the
        same move-into-`output-files` reorganization pass afterward.
      - Full test suite clean: smoke 488/488, edge-case 95/95,
        regression 274/274 (`--model mfrm`).

- [x] **Naming-convention note copied to manual** — done 2026-08-06.
      User had added a note to the PseudoHalo notebook (cell 39)
      explaining why `lambda` (additive) vs `omega`/`alpha` (multiplicative,
      distinct Greek letters) — asked for it to be copied to the manual.
      First attempt picked the wrong paragraph (the "why called
      pseudo_halo, not classic halo" one, added to `mfrm.tex` regardless
      since it's genuinely useful) — corrected once the real note was
      found. Added as a new paragraph in the `calibrate` section
      (`mfrm.tex` line ~824), right after the `threshold_stretch`/
      `item_stretch` alias paragraph, covering all three models generically
      (not just pseudo_halo) since the lambda/stretch-letter convention
      applies uniformly.

- [x] **`rater_stats_df` redesigned for `centrality`/`pseudo_halo`/
      `bistretch`** — done 2026-08-06, user's own request (unhappy with
      the existing full-vector-only format). `rater_stats_{model}` is now
      the model's own raw parameters (`lambda`, plus `alpha`/`omega`) with
      direct bootstrap SEs/CIs, one row per rater — not the reconstructed
      full severity vector. That reconstruction is preserved as a second
      table, `rater_stats_{model}_full_vector`, stored from the same call.
      - New `_store_stretch_param_se` helper in `raschpy/mfrm.py`,
        bootstrapping each raw parameter directly from
        `s.lambda_{model}`/`s.alpha_{model}`/`s.omega_{model}` across
        replicas (previously only the reconstructed-vector SE existed —
        the raw-parameter SE genuinely didn't exist anywhere in the
        codebase before this).
      - **Real pre-existing bug found and fixed along the way**:
        `bistretch`'s `rater_stats_df` had NO dispatch branch at all —
        `rater_stats_bistretch` silently contained only "Overall
        statistics" (Count/Infit MS/Outfit MS), missing all severity
        data entirely. Fixed as part of the same redesign (added the
        dual item+threshold reconstruction, mirroring `bivector`'s own
        branch).
      - `save_stats`/`save_stats_{model}` extended: for these 3 models, a
        sixth file/sheet (`_rater_stats_full_vector.csv` or a "Rater
        statistics (full vector)" sheet) is now also saved.
      - `__getattr__` extended to match the alias substring anywhere in
        an attribute name (not just as a trailing suffix), so
        `rater_stats_item_stretch_full_vector` correctly resolves to
        `rater_stats_pseudo_halo_full_vector` too.
      - README, manual (`rater_stats_df`/`save_stats` sections, plus a
        new worked example), and all three "Model classes" notebooks
        updated with a demo cell showing both tables.
      - Regression fixtures for `centrality`/`pseudo_halo`/`bistretch`
        regenerated (the only 3 affected — `rater_stats_{model}`'s shape
        genuinely changed by design); zero drift in every other
        git-tracked fixture.

- [x] **`gamma` → `omega` rename** — done 2026-08-06, user's own
      follow-up request: Jin & Wang (2018) use omega for their stretch
      factor, so `pseudo_halo`/`bistretch`'s item-difficulty-stretch
      parameter (previously `gamma`) is renamed to match, for consistency
      with `centrality`'s `alpha` (also from Jin & Wang's own convention).
      `alpha` itself untouched. No Python-keyword collision (unlike the
      earlier `beta`→`lambda` rename) — `omega` needed no trailing
      underscore or scratch-variable workaround anywhere.
      - Same scope as the `beta`→`lambda` rename: `raschpy/mfrm.py`,
        `raschpy/simulation/mfrm_sim.py` (`manual_gamma`→`manual_omega`,
        `self.gamma`→`self.omega`), `README.md`, `mfrm.tex` +
        `mfrm_sim_pseudo_halo.tex`/`mfrm_sim_bistretch.tex` (added a new
        paragraph explaining the Greek-letter convention, citing Jin &
        Wang for `alpha`), all 3 "Model classes" notebooks (re-executed;
        one cell needed real restructuring, not just text substitution —
        `Bistretch`'s recovery-correlation loop, since the sim-side
        attribute (`omega`, bare) and mfrm-side attribute
        (`omega_bistretch`) no longer share a naming pattern a single
        loop variable can drive), and 2 "Simulations" notebooks (source +
        throwaway verify pass; `MFRM_Sim_Centrality` untouched, no gamma
        there).
      - **Real mistake made and fixed during this pass**: re-executing
        the `Pseudo halo MFRM example.ipynb` notebook wrote 20 files into
        `~/Downloads/Jupyter notebook outputs/` — a leftover hardcoded
        `os.chdir(...)` path in that notebook's cell 1 (commented-out
        placeholder, real path underneath) from an earlier, unrelated
        session, never caught by the execution script's placeholder-match
        check. Files were clearly identifiable by name+today's-timestamp
        against the user's own genuinely older content in that folder
        (8 HTML files from June), removed cleanly; cell 1 fixed back to
        the standard placeholder; notebook re-executed correctly
        afterward. **Lesson: verify every notebook's chdir cell matches
        the exact expected placeholder string before treating "exit code
        0" as proof the execution went where intended** — a silent
        non-match means the script executes against whatever `os.chdir`
        was already set to, which could be anywhere.
      - Regression fixtures for `pseudo_halo`/`bistretch` regenerated
        (2 affected — column labels `gamma`→`omega`); zero drift
        elsewhere. Full suite clean: smoke 488/488, edge-case 95/95,
        regression 274/274.

## Pending for tomorrow (2026-08-07) — user reviewing notebooks/README/manual themselves
- [ ] **Update `Examples/Model selection example.ipynb`** to demonstrate
      the `models=` restriction argument on `model_selection()` (and
      probably `per_rater_model_selection()` too, since it gained the
      same parameter this session) — the notebook currently predates
      that argument entirely. Not started; flagged by the user, not
      yet scoped in detail (which cells to add, whether to also touch
      `per_rater_model_selection` examples in the same notebook or a
      separate one).
- [x] User's own review pass on notebooks, README, and the manual
      (`~/Documents/RaschPy_manual/`) — surfaced two follow-up requests,
      both completed 2026-08-07 (see below).

## `facet_effects_{model}` compact/full_vector swap + final alpha→omega unification (2026-08-07)
- [x] **`self.facet_effects_{model}` swapped to the compact parameter
      table for `centrality`/`pseudo_halo`/`bistretch`** (was: the full
      reconstructed severity matrix; user wanted it to hold the actual
      model parameters instead, e.g. a `lambda`/`omega` column per rater
      for `centrality`). The full reconstructed matrix moved to
      `self.facet_effects_{model}_full_vector`. New `_sev_attr(model,
      anchor=False)` helper added as the single choke point for "give me
      the full reconstructed severity matrix attribute name" — used by
      `_get_params`, `check_anchor_homogeneity`, `_store_rater_se`, and
      `rater_stats_df`'s top-level fetch, so every one of `cat_prob`/
      `exp_score`/`variance`/`kurtosis`/`person`/`warm`/`icc`/`crcs`/
      `tcc`/`iic`/`test_info`/`test_csem`/`std_residuals_plot`/
      `category_probability_dict` (all routed through `_get_params`)
      picks up the swap automatically. `_set_facet_aliases` reworked to
      source `raters_{model}`/`{facets}_{model}` aliases from the
      `_full_vector` attribute for these three models. Verified via full
      smoke/edge-case/regression suites, all clean, before moving on.
- [x] **Final `alpha`→`omega` unification.** User's call: Jin & Wang's
      own `omega` notation is for the *threshold*-stretch coefficient
      (i.e. what this package was calling `alpha` in `centrality`), not
      item-stretch — so `alpha` is confusing and eliminated entirely.
      `omega` is now used for every stretch parameter in every model:
      `centrality`'s `alpha`→`omega`; `pseudo_halo` already used `omega`
      (unchanged); `bistretch`'s two axes disambiguated as
      `omega_items_bistretch`/`omega_thresholds_bistretch` (previously
      `omega_bistretch`/`alpha_bistretch`). Touched: `mfrm.py` (estimate/
      calibrate_anchor methods, `_store_stretch_param_se` call sites,
      `rater_stats_df`'s `param_names` dict, docstrings), `mfrm_sim.py`
      (`MFRM_Sim_Centrality.manual_alpha`→`manual_omega`;
      `MFRM_Sim_Bistretch.manual_alpha`/`manual_omega`→
      `manual_omega_thresholds`/`manual_omega_items`), README, the
      manual (`mfrm.tex` + `mfrm_sim_centrality.tex` +
      `mfrm_sim_bistretch.tex`, including rewriting the Jin & Wang
      citation note that had misattributed which axis `omega` names),
      and all 4 affected notebooks (2 worked examples + 2 sim examples;
      re-executed with real output, zero errors, using the established
      chdir-verification discipline).
- [x] **New `global_`/`stretch_` parameter aliases**, per the same
      user request as the rename above: `self.global_{model}` is a
      synonym for `self.lambda_{model}`, `self.stretch_{model}` for
      `self.omega_{model}` (and `stretch_items_bistretch`/
      `stretch_thresholds_bistretch` for bistretch's two axes) — `global`
      because `omega_r == 1` collapses the model to `'global'`, matching
      `'global'`'s own single-subscript `lambda_r` notation. Implemented
      as a new `_PARAM_ALIASES` dict + `__getattr__` extension, mirroring
      the existing `_MODEL_ALIASES` mechanism (prefix match + `_{alias}_`
      infix match, both gated on `hasattr` so it never shadows a real
      attribute). Verified live for both single-axis and two-axis cases.
- [x] **Bug found and fixed while investigating a regression-suite
      failure**: `bistretch`'s `item_stats`/`threshold_stats`/
      `person_stats`/`test_stats`/SE fixtures were stale relative to the
      `facet_effects_{model}` swap above (fixtures dated 2026-08-06,
      predating some later refinement in that work) — confirmed via two
      independent fresh-process runs producing bit-for-bit identical
      live output that differed from the stored fixture, ruling out
      nondeterminism. Regenerated; `git status --porcelain` on
      already-tracked fixtures confirmed zero drift elsewhere (only the
      untracked `centrality`/`pseudo_halo`/`bistretch` dirs changed).
      Full regression suite: 274/274 clean afterward.
- [x] **Process note**: mid-diagnosis, briefly ran `git stash push` on
      `mfrm.py`/`mfrm_sim.py` intending to isolate just today's rename —
      this reverted to the last real commit, i.e. *all* uncommitted work
      across this entire multi-session feature (none of it has been
      committed yet), not just today's diff. Caught immediately and
      restored via `git stash pop` before any work was lost. Lesson:
      `git stash` on a file with a long uncommitted history is not a
      safe way to isolate "just this session's changes" — there is no
      commit boundary to stash back to short of the true last commit.

## Already done this session (2026-08-04/05)
- `_estimate_raters_bistretch` closed-form derivation, all dispatch
  sites (`calibrate`, `cat_prob`, `_cat_probs_mfrm`, anchor calibration,
  `_store_rater_se`).
- Full alias parity: 46/46 method families, verified via set-diff
  against `bivector`/`centrality`/`pseudo_halo` and an `ast`-based
  whole-class duplicate-name check (0 duplicates / 476 total methods).
- Fixed a real bug from the batch alias-generation script: 12 method
  families had silently duplicated, unrenamed `_centrality`/
  `_pseudo_halo` blocks appended after the correct `_bistretch` method.
  All 12 cleaned; confirmed via re-run of the duplicate check.
- `MFRM_Sim_Bistretch` added and exported from `raschpy/__init__.py`
  and `raschpy/simulation/__init__.py`. Genuine parameter recovery
  verified (N=300, I=20, R=10, K=4, seed=42): corr(true, est)
  beta=0.997, alpha=0.983, gamma=0.768.
- Full smoke (435/435) and edge-case (86/86) suites re-run clean after
  the duplicate-method fix — no regressions elsewhere.
