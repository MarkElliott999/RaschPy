# RaschPy
_RaschPy_ is a Python package for Rasch analysis which can estimate parameters for a variety of Rasch models, generate a range of model fit statistics, and output tables and graphical plots. _RaschPy_ also contains simulation functionality. _RaschPy_ is open source and free to download. The following is intended to highlight the main functionality of _RaschPy_ rather than serve as a comprehensive user manual; a full navigable manual is available in the GitHub repository. A basic Excel spreadsheet demonstrating the PAIR algorithm for dichotomous data is also available, following the example of the Moulton JMLE dichotomous demo available via https://www.rasch.org/moulton.htm (and using the same set of responses — the final results are compared to the Moulton JMLE output).

## Models
_RaschPy_ has a parent class `Rasch` for analysis, with the following child classes for different Rasch models:
- `SLM` for the simple logistic model (dichotomous Rasch model) (Rasch 1960)
- `PCM` for the partial credit model (Masters 1982)
- `RSM` for the rating scale model (Andrich 1978)
- `MFRM` for the many-facet Rasch model (rating scale model formulation) (Linacre 1994), including extended rater representations (Elliott and Buttery 2022a, Elliott 2025), with eight rater parameterisations (global, items, thresholds, bivector, matrix, centrality/threshold stretch, pseudo_halo/item stretch and bistretch formulations) — centrality and pseudo_halo are restricted forms of thresholds/items respectively (Jin & Wang 2018-style rater centrality/extremity, and an item-difficulty-compression model), with two parameters each; bistretch is a restricted form of bivector with three parameters combining both stretch axes at once.

## Analysis
To analyse data, create an object in the appropriate class, passing a pandas DataFrame of response data as an argument along with other arguments relevant to the chosen Rasch model, such as the maximum score for `RSM` or `MFRM`, or a vector of maximum scores for `PCM`. At the time of writing, the `RSM` and `MFRM` classes only support a single response group (i.e. all items must have the same threshold structure), and the `MFRM` class only supports one additional facet for rater severity. Parameter estimation uses variants of PAIR (Choppin 1968, 1985), the eigenvector method (Garner & Engelhard 2002, 2009) and CPAT (Elliott & Buttery 2022a, 2022b).

Each model follows the same workflow: instantiate → calibrate → fit statistics → output tables → plots. All major results are stored as attributes on the model object after each step.

## Examples

### Loading data

_RaschPy_ includes loaders for CSV, Excel, and JSON that validate scores and handle missing data:

```python
from raschpy.loaders import loadup_slm, loadup_pcm, loadup_rsm
from raschpy.loaders import loadup_mfrm_single, loadup_mfrm_xlsx_tabs, loadup_mfrm_multiple

# Wide-format file (persons as rows, items as columns)
responses, invalid = loadup_slm('my_data.csv')
responses, invalid = loadup_rsm('my_data.csv', max_score=4)
responses, invalid = loadup_pcm('my_data.csv', max_score_vector=[3, 3, 4, 4, 3])

# Long-format file (Person, Item, Score columns)
responses, invalid = loadup_rsm('my_data.csv', max_score=4, long=True)

# MFRM: one file with (Rater, Person) MultiIndex
responses, invalid = loadup_mfrm_single('my_mfrm_data.csv', max_score=3)

# MFRM: one Excel workbook with one sheet per rater
responses, invalid = loadup_mfrm_xlsx_tabs('my_mfrm_data.xlsx', max_score=3)

# MFRM: separate files per rater
responses, invalid = loadup_mfrm_multiple(
    {'Rater_A': 'rater_a.csv', 'Rater_B': 'rater_b.csv'}, max_score=3
)
```

Alternatively, pass any pandas DataFrame directly to the model constructor.

---

### Simple Logistic Model (SLM)

Dichotomous data (0/1). Each row is a person, each column an item.

```python
from raschpy import SLM
from raschpy.simulation import SLM_Sim

# Pass a simulation object directly — generating parameters are attached to m.generating
sim = SLM_Sim(no_of_items=10, no_of_persons=300)
slm = SLM(sim)

# Or pass a DataFrame as usual
slm = SLM(responses)
slm.calibrate()          # PAIR item location estimation
slm.fit_statistics()     # item, person, and test-level fit

slm.item_stats_df(full=True)
print(slm.item_stats)    # Estimate, SE, Infit MS, Outfit MS, ...

slm.person_stats_df()
print(slm.person_stats)  # Estimate, CSEM, Score, Infit MS, ...

slm.test_stats_df()
print(slm.test_stats)    # ISI, PSI, reliability

slm.icc(item='Item_1', obs=True)     # Item Characteristic Curve
slm.tcc(obs=True)                    # Test Characteristic Curve
slm.test_info()                      # Test Information Curve
slm.std_residuals_plot(normal=True)  # Standardised residuals histogram

slm.save_stats('slm_results', format='xlsx')
```

**Score lookup and person estimation**

```python
# Convert raw scores to person location estimates
model.score_lookup_table()
print(model.score_lookup)   # pandas Series indexed by raw score

# By default, missing responses are excluded from person estimates.
# To include them, treated as incorrect rather than missing:
model.person_estimates(missing_as_incorrect=True)
```

---

### Partial Credit Model (PCM)

Polytomous data where items may have different maximum scores.

```python
from raschpy import PCM
from raschpy.simulation import PCM_Sim

# Pass a simulation object directly — generating parameters are attached to m.generating
sim = PCM_Sim(no_of_items=5, no_of_persons=300, max_score_vector=[3, 3, 4, 4, 3])
pcm = PCM(sim)

# Or pass a DataFrame as usual
pcm = PCM(responses, max_score_vector=[3, 3, 4, 4, 3])
pcm.calibrate()
pcm.fit_statistics()

pcm.item_stats_df(full=True)
print(pcm.item_stats)                  # Central item locations

pcm.threshold_stats_df(full=True)
print(pcm.threshold_stats_uncentred)   # Uncentred threshold estimates
print(pcm.threshold_stats_centred)     # Centred threshold offsets

pcm.icc(item='Item_1', obs=True)             # Expected score curve
pcm.crcs(item='Item_1', obs='all')           # Category Response Curves
pcm.threshold_ccs(item='Item_1', obs='all')  # Threshold Characteristic Curves
pcm.tcc(obs=True)

pcm.score_lookup_table()
print(pcm.score_lookup)   # pandas Series indexed by raw score

pcm.save_stats('pcm_results', format='xlsx')
```

---

### Rating Scale Model (RSM)

Polytomous data where all items share the same rating scale structure.

```python
from raschpy import RSM
from raschpy.simulation import RSM_Sim

# Pass a simulation object directly — generating parameters are attached to m.generating
sim = RSM_Sim(no_of_items=8, no_of_persons=300, max_score=4)
rsm = RSM(sim)

# Or pass a DataFrame as usual
rsm = RSM(responses, max_score=4)
rsm.calibrate()
rsm.fit_statistics()

rsm.item_stats_df(full=True)
print(rsm.item_stats)      # Item locations

rsm.threshold_stats_df(full=True)
print(rsm.threshold_stats) # Shared Rasch-Andrich thresholds

rsm.icc(item='Item_1', obs=True)
rsm.crcs(obs='all')         # Pooled across all items
rsm.threshold_ccs(obs='all')
rsm.tcc(obs=True)

rsm.score_lookup_table()
print(rsm.score_lookup)   # pandas Series indexed by raw score

rsm.save_stats('rsm_results', format='xlsx')
```

---

### Many-Facet Rasch Model (MFRM)

Polytomous data with multiple raters. Data must be a DataFrame with a `(Rater, Person)` MultiIndex and items as columns. Eight rater parameterisations are available, selected at calibration time:

| `model=` | Rater severity structure |
|---|---|
| `'global'` | Single scalar severity per rater |
| `'items'` | Separate severity per rater × item |
| `'thresholds'` | Separate severity per rater × threshold |
| `'bivector'` | Separate severities per rater × item and per rater × threshold |
| `'matrix'` | Full severity matrix per rater × item × threshold |
| `'centrality'` (alias `'threshold_stretch'`) | 2 parameters per rater (shift + threshold-stretch), a closed-form restriction of `'thresholds'` |
| `'pseudo_halo'` (alias `'item_stretch'`) | 2 parameters per rater (shift + item-difficulty-stretch), a closed-form restriction of `'items'` |
| `'bistretch'` | 3 parameters per rater (shift + item-stretch + threshold-stretch), a closed-form restriction of `'bivector'` |

`'threshold_stretch'`/`'item_stretch'` are exact synonyms for `'centrality'`/`'pseudo_halo'` — same model, same estimation, just an axis-based name matching `'items'`/`'thresholds'` and `'bistretch'`'s own naming. They work everywhere a model name is accepted: as the `model=`/`models=` value (e.g. `mfrm.calibrate(model='threshold_stretch')`, `mfrm.model_selection(models=['global', 'item_stretch'])`), and as the method-name suffix (e.g. `mfrm.calibrate_threshold_stretch()`, `mfrm.item_stats_df_item_stretch()`) for every method family that has a `_centrality`/`_pseudo_halo` variant. Attributes are still stored under the canonical `_centrality`/`_pseudo_halo` names (e.g. `mfrm.lambda_centrality`) regardless of which spelling was used to calibrate.

```python
from raschpy import MFRM

mfrm = MFRM(responses)
mfrm.calibrate(model='global')
mfrm.fit_statistics(model='global')

mfrm.item_stats_df(model='global', full=True)
print(mfrm.item_stats_global)        # Item locations

mfrm.threshold_stats_df(model='global', full=True)
print(mfrm.threshold_stats_global)   # Shared thresholds

mfrm.rater_stats_df(model='global', full=True)
print(mfrm.rater_stats_global)       # Rater severities and fit

mfrm.person_stats_df(model='global')
mfrm.test_stats_df(model='global')

mfrm.icc(item='Item_1', model='global', obs=True)
mfrm.crcs(item='Item_1', model='global')
mfrm.tcc(model='global', obs=True)

mfrm.save_stats(model='global', filename='mfrm_results', format='xlsx')
```

The same object can hold calibrations for multiple parameterisations simultaneously:

```python
mfrm.calibrate(model='items')
mfrm.fit_statistics(model='items')
mfrm.rater_stats_df(model='items', full=True)
print(mfrm.rater_stats_items)  # Per-item severity table
```

**Extreme and invalid persons**

Persons with entirely missing data are always removed at instantiation and stored in `m.invalid_responses`. Persons with extreme scores (all-zero or perfect) are removed only when `extreme_persons=False` is passed to the constructor, and are stored in `m.extreme_persons`.

---

### Many-Facet Rasch Model — Bivector formulation

The bivector formulation represents rater severity as an additive combination of per-item leniency effects and per-threshold consistency effects, allowing rater behaviour to vary systematically across the latent continuum. It functions as rater-as-RSM, as opposed to the matrix formulation, which functions as rater-as-PCM.

```python
mfrm.calibrate(model='bivector')
mfrm.fit_statistics(model='bivector')

mfrm.rater_stats_df(model='bivector', full=True)
# rater_stats_bivector has MultiIndex columns:
# per-item marginal severities, then per-threshold marginal severities,
# plus overall fit statistics
print(mfrm.rater_stats_bivector)

mfrm.icc(item='Item_1', model='bivector', obs=True)
mfrm.tcc(model='bivector', obs=True)

mfrm.save_stats(model='bivector', filename='mfrm_bivector_results', format='xlsx')
```

---

### Many-Facet Rasch Model — centrality and pseudo_halo formulations

`'centrality'` and `'pseudo_halo'` are restricted forms of `'thresholds'` and `'items'` respectively, defined by two parameters: a severity shift `lambda_r` plus a stretch parameter `omega_r` (the same name for both models — `omega` is Jin & Wang's (2018) notation for a stretch coefficient for their centrality/extremity model, whish is `centrality` here); they sit between `'global'` (1 free parameter per rater) and their parent model on the `model_selection()` df ladder. `'threshold_stretch'`/`'item_stretch'` are exact aliases for `'centrality'`/`'pseudo_halo'` — e.g. `mfrm.calibrate(model='threshold_stretch')` or `mfrm.calibrate_item_stretch()` behave identically to the calls below. `lambda_{model}`/`omega_{model}` also have `global_{model}`/`stretch_{model}` as aliases (e.g. `mfrm.global_centrality` and `mfrm.stretch_centrality` are aliases for `mfrm.lambda_centrality` and `mfrm.omega_centrality`).

For `centrality`, `omega_r > 1` spreads the reference thresholds apart for that rater ("central" — a larger person-location gap is needed to traverse to a more extreme category); conversely, `0 < omega_r < 1` compresses thresholds ("extreme"); `omega_r == 1` is neutral (Jin & Wang 2018). For `pseudo_halo`, `omega_r < 1` compresses the spread of reference item locations toward zero (the pseudo-halo signature — the rater effectively ignores/reduces differences in items locations, increasing the likelihood of a flat score profile as in the halo effect but without requiring local item dependence, hence the name `pseudo_halo`); `omega_r > 1` exaggerates real item-difficulty differences instead. Both stretch parameters can in principle take zero or negative values, representing a disordered situation for the reference facet – this is flagged via `UserWarning` but not clipped).

`lambda_r`/`omega_r` are recovered per rater via a robust regression of the rater's full operational vector model values against the reference item/threshold values, controlled by `calibrate(..., regression='huber')` (default) or `regression='theil-sen'`. `'huber'` is an M-estimator that downweights, rather than discards, points with large residuals by switching from a loss function of squared error to one of MAE beyond a threshold; `'theil-sen'` is the median of pairwise slopes. If `'huber'`'s underlying M-estimator fit fails to converge for a given rater — not observed in testing — that rater automatically falls back to `'theil-sen'` with a `UserWarning` naming it; the rest of the raters are unaffected.

```python
mfrm.calibrate(model='centrality')
mfrm.fit_statistics(model='centrality')

print(mfrm.lambda_centrality)             # severity shift per rater
print(mfrm.omega_centrality)            # threshold-stretch per rater
print(mfrm.raters_centrality)           # reconstructed full (rater x threshold) severity table

mfrm.calibrate(model='pseudo_halo')
mfrm.fit_statistics(model='pseudo_halo')

print(mfrm.lambda_pseudo_halo)            # severity shift per rater
print(mfrm.omega_pseudo_halo)           # item-difficulty-stretch per rater
print(mfrm.raters_pseudo_halo)          # reconstructed full (rater x item) severity table
```

`rater_stats_df()` for these two models returns the two model parameters (with bootstrap SEs and, if `full=True`, 95% CIs) as the primary `rater_stats_{model}` table, rather than the full reconstructed vector. A single call also stores that reconstructed vector separately, as `rater_stats_{model}_full_vector`:

```python
mfrm.rater_stats_df(model='centrality', full=True)
print(mfrm.rater_stats_centrality)              # lambda/omega per rater, with SEs, 95% CIs, and fit stats
print(mfrm.rater_stats_centrality_full_vector)  # the reconstructed full (rater x threshold) table instead

mfrm.rater_stats_df(model='pseudo_halo', full=True)
print(mfrm.rater_stats_pseudo_halo)              # lambda/omega per rater, with SEs, 95% CIs, and fit stats
print(mfrm.rater_stats_pseudo_halo_full_vector)  # the reconstructed full (rater x item) table instead
```

Pass `alias=True` to relabel the primary table's parameter columns with a descriptive name consistent with the rest of the model family instead of `lambda`/`omega` — `'Global'` for `lambda` and `'Item stretch'`/`'Threshold stretch'` for `omega`, depending on which axis the model's stretch parameter operates on (`'Threshold stretch'` for `centrality`, `'Item stretch'` for `pseudo_halo`). This is purely cosmetic and does not affect values. `save_stats()` accepts the same `alias=` argument, forwarded straight through to its own internal `rater_stats_df()` call:

```python
mfrm.rater_stats_df(model='pseudo_halo', full=True, alias=True)
print(mfrm.rater_stats_pseudo_halo.columns.get_level_values(0).unique())  # ['Global', 'Item stretch', 'Overall statistics']

mfrm.save_stats(model='centrality', filename='mfrm_centrality_results', format='xlsx', alias=True)
```

Pass `divergence_test='wald'` or `'t'` to run an item-by-item (or threshold-by-threshold) divergence test comparing the free vector model's own per-element estimates (`'items'` for `pseudo_halo`, `'thresholds'` for `centrality`) against the reconstructed vector from the stretch model , flagging elements where the stretch representation diverges significantly from what the vector model estimates – a graded, localised complement to the aggregate `lambda`/`omega` summary and to the discrete `model_selection()`/`per_rater_model_selection()` ladder. The SE is derived from a single, paired bootstrap: each bootstrap resample produces estimates for both the stretch model and its corresponding vector model, so for a given resample, the free and stretch estimates are produced from the same resampled dataset. The difference is calculated directly per resample before taking the SD, rather than combining two independently-bootstrapped SEs via `sqrt(se1**2+se2**2)`; this correctly accounts for the correlation between the two sets of estimates. Off by default (`divergence_test=None`) since it requires `store_bootstrap=True`, a larger bootstrap-object footprint:

```python
mfrm.rater_stats_df(model='pseudo_halo', divergence_test='wald', correction='bh')
print(mfrm.rater_stats_pseudo_halo['Flagged items'])  # per-rater count of significantly divergent items

mfrm.rater_stats_df(model='centrality', full=True, divergence_test='wald')
print(mfrm.rater_stats_centrality_full_vector['Threshold 1'])  # now has a 'Flagged' column too
```

`'wald'` references the test statistic against the standard normal distribution. `'t'` instead uses a plain Student's t-distribution with `no_of_samples - 1` degrees of freedom for small numbers of bootstraps run for speed; this is not a paired t-test since this would require nested bootstrapping at high cost; since the t-test converges to the Wald result once `no_of_samples` is a few hundred (at `n=300`, p-values near `alpha=0.05` differ by ~2%), if speed is less important, run `'wald'` with a larger `no_of_samples` instead, which improves the SE estimate itself rather than just widening the reference distribution around it.

`full=True` gives two more ways to see which elements were flagged, alongside the `'Flagged items'`/`'Flagged categories'` count still shown in the primary table: one plain True/False column per item and/or threshold is added to the primary table itself, directly after that count, for scanning flagged status across all elements at a glance – and `rater_stats_{model}_full_vector` separately gets a per-element `'Flagged'` column added to each item/threshold's own `Estimate`/`SE`/CI block, for the same information alongside that element's own detail:

```python
mfrm.rater_stats_df(model='pseudo_halo', full=True, divergence_test='wald')
print(mfrm.rater_stats_pseudo_halo['Item_1'])  # 'Flagged' column, True/False per rater
```

Pass `correction='bh'` to apply Benjamini-Hochberg FDR correction within each rater's own family of tests (the rater's items thresholds, not pooled across raters); default `correction=None` compares raw p-values against `alpha` (default `0.05`) directly. Pass `plot=True` to also build and store `self.fit_plot_{model}` (e.g. `self.fit_plot_pseudo_halo`) – a scatter of the free vector model's estimates against the stretch model's reconstruction, flagged elements in a different colour, in the style of `self.anchor_plot`.

Since, as described above, `pseudo_halo` tests compression of item locations under an otherwise locally-independent response process, rather that classic halo effect's usual mechanism (elevated local item interdependence), pair `omega_pseudo_halo` with a per-rater residual-correlation to check for full halo effect:

```python
mfrm.item_res_corr_analysis_pseudo_halo()
print(mfrm.item_residual_correlations_pseudo_halo)
```

A rater near the group's typical residual-correlation level with `omega_r < 1` more plausibly reflects genuine compressed-perception pseudo_halo; a rater whose residual correlation is a clear outlier above the group more plausibly reflects true local-dependence halo effect, fairly independently of that rater's own `omega_r`. Note that since there is often LID due to rating scale interdependence, as opposed to rater behaviour, and it is not possible to disentangle these two effects, this is a comparative diagnostic across raters rather than an absolute one.

Both models also have dedicated simulation classes for genuine parameter recovery checks:

```python
from raschpy.simulation import MFRM_Sim_Centrality, MFRM_Sim_PseudoHalo

sim = MFRM_Sim_Centrality(no_of_items=10, no_of_persons=200, no_of_raters=8, max_score=4, seed=42)
print(sim.lambda_, sim.omega)   # true generating values

mfrm = MFRM(sim)
mfrm.calibrate(model='centrality')
print(mfrm.lambda_centrality, mfrm.omega_centrality)  # compare against sim.lambda_/sim.omega
```

---

### Many-Facet Rasch Model — bistretch formulation

`'bistretch'` combines both stretch axes at once: a restricted of `'bivector'` with three parameters. Each rater is defined by a severity shift `lambda_r`, an item-difficulty-stretch `omega_items_r`, and a threshold-stretch `omega_thresholds_r`, instead of `'bivector'`'s full `I+K-1` free parameters. `omega_items_{model}` and `omega_thresholds_{model}` also have `stretch_items_{model}`/`stretch_thresholds_{model}` as aliases, matching the `stretch_{model}` aliases for `centrality`/`pseudo_halo`:

```python
mfrm.calibrate(model='bistretch')
mfrm.fit_statistics(model='bistretch')

print(mfrm.lambda_bistretch)              # severity shift per rater
print(mfrm.omega_items_bistretch)       # item-difficulty-stretch per rater
print(mfrm.omega_thresholds_bistretch)  # threshold-stretch per rater
print(mfrm.raters_bistretch)            # reconstructed full (rater x item) x threshold severity table
```

As with `centrality`/`pseudo_halo`, `rater_stats_df()`'s primary `rater_stats_bistretch` table shows the 3 parameters (with SEs and, if `full=True`, 95% CIs); the reconstructed full (rater x item) x threshold vector is stored separately as `rater_stats_bistretch_full_vector`:

```python
mfrm.rater_stats_df(model='bistretch', full=True)
print(mfrm.rater_stats_bistretch)              # lambda/omega_items/omega_thresholds per rater, with SEs, 95% CIs, and fit stats
print(mfrm.rater_stats_bistretch_full_vector)  # the reconstructed full (rater x item) x threshold table instead
```

`alias=True` relabels all 3 columns — `'Global'`/`'Item stretch'`/`'Threshold stretch'` for `lambda`/`omega_items`/`omega_thresholds` respectively:

```python
mfrm.rater_stats_df(model='bistretch', full=True, alias=True)
print(mfrm.rater_stats_bistretch.columns.get_level_values(0).unique())  # ['Global', 'Item stretch', 'Threshold stretch', 'Overall statistics']
```

Since `bistretch` restricts both axes at once, `divergence_test` runs against `bivector`'s item vector, and against its threshold vector, producing separate `'Flagged items'` and `'Flagged categories'` columns:

```python
mfrm.rater_stats_df(model='bistretch', divergence_test='t', correction='bh')
print(mfrm.rater_stats_bistretch[['Flagged items', 'Flagged categories']])
```

As with `centrality`/`pseudo_halo`, a dedicated simulation class generates data with true `lambda_`/`omega_items`/`omega_thresholds` structure for genuine parameter recovery checks:

```python
from raschpy.simulation import MFRM_Sim_Bistretch

sim = MFRM_Sim_Bistretch(no_of_items=10, no_of_persons=200, no_of_raters=8, max_score=4, seed=42)
print(sim.lambda_, sim.omega_items, sim.omega_thresholds)   # true generating values

mfrm = MFRM(sim)
mfrm.calibrate(model='bistretch')
print(mfrm.lambda_bistretch, mfrm.omega_items_bistretch, mfrm.omega_thresholds_bistretch)  # compare against sim values
```

---

### Model comparison — RSM vs PCM threshold structure

RSM constrains every item to share the same threshold structure; PCM lets each item have its own. Since RSM is nested within PCM, `model_selection()` compares them directly via a likelihood-ratio test, AIC, or BIC (requires uniform max scores across items):

```python
rsm.model_selection(test='AIC')
print(rsm.model_comparison_rsm_pcm_aic_summary)    # ranked comparison table
print(rsm.model_comparison_rsm_pcm_aic_preferred)  # 'RSM' or 'PCM'

# Equivalently from the PCM side
pcm.model_selection(test='LR')
print(pcm.model_comparison_rsm_pcm_lr_summary)
```

---

### MFRM rater-parameterisation model selection

`model_selection()` calibrates all eight rater parameterisations (global, items, thresholds, bivector, matrix, centrality, pseudo_halo, bistretch) and ranks them by the chosen criterion — or pass `models=` to restrict the comparison to a subset (e.g. a direct two-model test) without paying the cost of calibrating the rest:

```python
mfrm.model_selection(test='AIC')
print(mfrm.model_comparison_mfrm_aic_summary)    # ranked comparison across all eight models
print(mfrm.model_comparison_mfrm_aic_preferred)  # e.g. 'items'

# LR degrees of freedom fall straight out of the parameter-count difference,
# e.g. for centrality: (K-2)*(R-1) against thresholds, (R-1) against global;
# for bistretch: (I+K-4)*(R-1) against bivector, (R-1) against centrality
# or pseudo_halo, 2*(R-1) against global
mfrm.model_selection(test='LR')
print(mfrm.model_comparison_mfrm_lr_summary)

# Restrict to a nested pair — only these two are calibrated
mfrm.model_selection(test='LR', models=['bivector', 'bistretch'])
```

**Mixed model: per-rater parameterisation assignment**

Rather than forcing every rater into the same parameterisation, `per_rater_model_selection()` assigns each rater the simplest adequate structure according to the chosen criterion individually. By default (`test='AIC'`) it picks the minimum-AIC representation directly per rater, guarded by a significance test against the simplest active baseline (`aic_sig_test`) and an optional absolute effect gate (`min_effect`). Under `test='LR'`, it instead runs a top-down ladder (matrix → bivector → items/thresholds → global), then one extra targeted stretch test depending on which of those five wins: `bivector` winner → tested against `bistretch`; `thresholds` winner → tested against `centrality`; `items` winner → tested against `pseudo_halo`; `global` winner → tested against both `centrality` and `pseudo_halo`, resolved via a further `bistretch` test if both beat `global`. Either way, the assigned model can be any of all eight parameterisations, not just the original five. Pass a list to `models=` to restrict which parameterisations are considered — for AIC/BIC this may be any subset; for LR, `{'global', 'items', 'thresholds', 'bivector'}` must be included since LR is strictly a test for nested models, while `'matrix'` and each stretch model are independently toggleable:

```python
mfrm.per_rater_model_selection(test='AIC')
print(mfrm.rater_models)                      # Series: rater -> assigned parameterisation
print(mfrm.per_rater_model_selection_table)   # full per-rater testing-ladder results
print(mfrm.per_rater_model_selection_counts)  # how many raters landed on each parameterisation

mfrm.fit_statistics(model='mixed')            # fit statistics using each rater's assigned model
mfrm.rater_stats_df(model='mixed', full=True)
print(mfrm.rater_stats_mixed)

# Exclude matrix and all three stretch models – back to the original 5-model ladder
mfrm.per_rater_model_selection(test='LR', models=['global', 'items', 'thresholds', 'bivector'])
```

---

### Category width statistics

For PCM, RSM, and MFRM, `category_stats_df()` reports each item's category *widths* (the gap between consecutive thresholds) rather than raw threshold locations — a width below 0 flags local category disordering:

```python
pcm.category_stats_df()
print(pcm.category_stats)   # Estimate (width), SE, Disordered, Prop disordered

mfrm.category_stats_df(model='global')
print(mfrm.category_stats_global)
```

---

### Differential Item Functioning (DIF) and invariance testing

`dif_test()` splits persons by an exogenous covariate (e.g. Gender, L1), calibrates each group independently, purifies them onto a common scale (so genuine DIF items can't distort the scale used to detect DIF), and tests every item for DIF against a chosen reference group — supporting covariates with any number of levels, each tested individually against the reference:

```python
import pandas as pd
from raschpy import SLM

exogenous = pd.DataFrame({'Gender': [...]}, index=responses.index)  # one row per person
slm = SLM(responses, exogenous=exogenous)

slm.dif_test(covariate='Gender')
print(slm.dif_table)          # per-item, per-focal-group DIF statistics
print(slm.dif_omnibus_table)  # Andersen-style joint test per focal group
```

For a simpler two-group check (a single Andersen likelihood-ratio test rather than the full anchor-purification workflow), use `andersen_lr_test(split_by='exogenous')` — note `split_by='person_location'`/`'score'` is disabled as a general model-fit test (found to have no power in simulation), so this is exogenous-covariate-only:

```python
slm.andersen_lr_test(split_by='exogenous', covariate='Gender')
print(slm.andersen_lr, slm.andersen_df, slm.andersen_p)
```

`dif_test()` is available on `SLM`, `PCM`, and `RSM` — it is not yet implemented for `MFRM`. `andersen_lr_test()` is available on all four, with per-parameterisation variants on `MFRM` (e.g. `andersen_lr_test_global`).

---

### Anchor calibration to an external item bank

To place a new calibration on the same scale as a previously-calibrated item bank, pass a `dict` or `pandas.Series` of externally-supplied item locations, keyed by item name. A subset of "anchor" items in common with the bank is used to compute a translation constant, with outlier anchors automatically down-weighted or excluded:

```python
bank_locations = {'Item_1': -0.8, 'Item_2': 0.3, 'Item_3': 1.1}

slm.calibrate_anchor(bank_locations)
print(slm.anchor_items)      # item locations shifted onto the bank scale
print(slm.anchor_summary)    # anchors supplied/selected/dropped, correlation, SD ratio, translation constant
print(slm.anchor_selection)  # per-anchor-item diagnostics (which were kept/dropped as outliers)
```

Available identically on `SLM`, `RSM`, and `PCM`. Diagnostics are plotted automatically (`plot=True` by default) via `plot_anchor_selection()`, which can also be called manually against any `anchor_selection`-shaped table.

---

### Simulation

_RaschPy_ includes simulation classes for generating synthetic data under each model. Simulation runs automatically on instantiation; generating parameters are stored as attributes on the simulation object (`sim.items`, `sim.persons`, `sim.thresholds`, and for MFRM models `sim.facet_effects`). When a simulation object is passed directly to a model constructor, the generating parameters are also attached to the model object under a `generating` namespace, making recovery comparisons straightforward:

```python
from raschpy.simulation import SLM_Sim, RSM_Sim, PCM_Sim
from raschpy.simulation import MFRM_Sim_Global, MFRM_Sim_Items, MFRM_Sim_Thresholds, MFRM_Sim_Bivector, MFRM_Sim_Matrix
from raschpy.simulation import MFRM_Sim_Centrality, MFRM_Sim_PseudoHalo, MFRM_Sim_Bistretch

sim = SLM_Sim(no_of_items=10, no_of_persons=300, item_range=3, person_sd=1.5)
data = sim.responses   # pandas DataFrame

sim = RSM_Sim(no_of_items=8, no_of_persons=300, max_score=4)
data = sim.responses   # pandas DataFrame

sim = PCM_Sim(no_of_items=6, no_of_persons=300, max_score_vector=[3, 3, 3, 4, 4, 4])
data = sim.responses   # pandas DataFrame

sim = MFRM_Sim_Global(no_of_items=6, no_of_persons=200, no_of_raters=4, max_score=3)
data = sim.responses   # (Rater, Person) MultiIndex DataFrame

# Pass the sim object directly to the model constructor to attach generating parameters
from raschpy import MFRM
mfrm = MFRM(sim)
mfrm.calibrate(model='global')

# Compare generating and estimated parameters
print(sim.items)                      # Generating item locations
print(mfrm.generating.items)          # Same, accessible on the model object
print(mfrm.items)                     # Estimated item locations

# Also for rater severities etc,
print(sim.facet_effects)              # Generating rater severities
print(mfrm.generating.facet_effects)  # Same, accessible on the model object
print(mfrm.raters_global)             # Estimated rater locations
```

---

### Bootstrap standard errors and confidence intervals

Standard errors are computed by bootstrap resampling. They are triggered automatically inside `fit_statistics()`, or can be run explicitly to request confidence intervals:

```python
model.std_errors(no_of_samples=200, interval=0.95)
model.item_stats_df(interval=0.95)  # adds 2.5% and 97.5% columns
```

---

### Anchor calibration — MFRM raters

To place an MFRM estimate within an anchored frame of reference, relative to the mean of a set of 'gold standard' raters, pass a list of anchor raters. A new set of anchored estimates will be generated. (For anchoring SLM/RSM/PCM item locations onto an external item bank instead, see [Anchor calibration to an external item bank](#anchor-calibration-to-an-external-item-bank) above.)

```python
mfrm.calibrate_global()
print(mfrm.raters_global)                    # Unanchored rater severities

anchors = ['Rater_1', 'Rater_3', 'Rater_6']
mfrm.calibrate_global_anchor(anchors)
print(mfrm.anchor_raters_global)             # Anchored rater severities
```

**Checking anchor rater homogeneity**

Anchoring only constrains the *mean* severity of the chosen anchor raters to zero — it has no way to notice whether that mean is a stable, shared reference point or an artefact of averaging over raters who don't behave alike. `check_anchor_homogeneity()` is a separate, opt-in diagnostic that tests whether the proposed anchor set actually agrees with itself before you rely on it:

```python
mfrm.check_anchor_homogeneity(model='global', anchors=anchors)
print(mfrm.anchor_homogeneity_test)        # omnibus Cochran's Q test
print(mfrm.anchor_homogeneity_per_rater)   # per-rater severity, SE, z, p, Flagged
```

## Usage and citation
_RaschPy_ is provided as freeware under an Apache 2.0 Licence (see LICENSE file in this repository for details). Users are free to use or modify the code for their own purposes, but should cite using the following format:

Elliott, M. (2026) _RaschPy_. Downloaded from: https://github.com/MarkElliott999/RaschPy

## References
Andrich, D. (1978). A rating formulation for ordered response categories. _Psychometrika_, _43_(4), 561–573.

Choppin, B. (1968). Item bank using sample-free calibration. _Nature_, _219_(5156), 870–872.

Choppin, B. (1985). A fully conditional estimation procedure for Rasch model parameters. _Evaluation in Education_, _9_(1), 29–42.

Elliott, M. (2025). Extended many-facet Rasch models: Accounting for rater effects in automated essay scoring systems (Doctoral dissertation, University of Cambridge).

Elliott, M., & Buttery, P. J. (2022a). Extended rater representations in the many-facet Rasch model. _Journal of Applied Measurement_, _22_(1), 133–160.

Elliott, M., & Buttery, P. J. (2022b). Non-iterative conditional pairwise estimation for the rating scale model. _Educational and Psychological Measurement_, _82_(5), 989–1019.

Garner, M., & Engelhard, G. (2002). An eigenvector method for estimating item parameters of the dichotomous and polytomous Rasch models. _Journal of Applied Measurement_, _3_(2), 107–128.

Garner, M., & Engelhard, G. (2009). Using paired comparison matrices to estimate parameters of the partial credit Rasch measurement model for rater-mediated assessments. _Journal of Applied Measurement_, _10_(1), 30–41.

Jin, K.-Y., & Wang, W.-C. (2018). A new facets model for rater's centrality/extremity response style. _Journal of Educational Measurement_, _55_(4), 543–563.

Linacre, J. M. (1994). _Many-Facet Rasch Measurement_. MESA Press.

Masters, G. N. (1982). A Rasch model for partial credit scoring. _Psychometrika_, _47_(2), 149–174.

Rasch, G. (1960). _Probabilistic models for some intelligence and attainment tests_. Danmarks Pædagogiske Institut.

## DISCLAIMER
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
