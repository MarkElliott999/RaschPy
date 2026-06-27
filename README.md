# RaschPy
_RaschPy_ is a Python package for Rasch analysis which can estimate parameters for a variety of Rasch models, generate a range of model fit statistics, and output tables and graphical plots. _RaschPy_ also contains simulation functionality. _RaschPy_ is open source and free to download. The following is intended to highlight the main functionality of _RaschPy_ rather than serve as a comprehensive user manual; a full navigable manual is available in the GitHub repository. A basic Excel spreadsheet demonstrating the PAIR algorithm for dichotomous data is also available, following the example of the Moulton JMLE dichotomous demo available via https://www.rasch.org/moulton.htm (and using the same set of responses — the final results are compared to the Moulton JMLE output).

## Models
_RaschPy_ has a parent class `Rasch` for analysis, with the following child classes for different Rasch models:
- `SLM` for the simple logistic model (dichotomous Rasch model) (Rasch 1960)
- `PCM` for the partial credit model (Masters 1982)
- `RSM` for the rating scale model (Andrich 1978)
- `MFRM` for the many-facet Rasch model (rating scale model formulation) (Linacre 1994), including extended rater representations (Elliott and Buttery 2022a, Elliott 2025), with five rater parameterisations (global, items, thresholds, bivector and matrix formulations)

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
slm.calibrate()          # PAIR item difficulty estimation
slm.fit_statistics()     # item, person, and test-level fit

slm.item_stats_df(full=True)
print(slm.item_stats)    # Estimate, SE, Infit MS, Outfit MS, ...

slm.person_stats_df()
print(slm.person_stats)  # Ability, CSEM, Score, Infit MS, ...

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
# Convert raw scores to ability estimates
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
print(pcm.item_stats)                  # Central item difficulties

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
print(rsm.item_stats)      # Item difficulties

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

Polytomous data with multiple raters. Data must be a DataFrame with a `(Rater, Person)` MultiIndex and items as columns. Five rater parameterisations are available, selected at calibration time:

| `model=` | Rater severity structure |
|---|---|
| `'global'` | Single scalar severity per rater |
| `'items'` | Separate severity per rater × item |
| `'thresholds'` | Separate severity per rater × threshold |
| `'bivector'` | Separate severities per rater × item and per rater × threshold |
| `'matrix'` | Full severity matrix per rater × item × threshold |

```python
from raschpy import MFRM

mfrm = MFRM(responses)
mfrm.calibrate(model='global')
mfrm.fit_statistics(model='global')

mfrm.item_stats_df(model='global', full=True)
print(mfrm.item_stats_global)        # Item difficulties

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

The bivector formulation represents rater severity as an additive combination of per-item leniency effects and per-threshold consistency effects, allowing rater behaviour to vary systematically across the latent continuum. It functions as rater-as-RSM, as opposed to the matrix formulation, which functions as rater-as-PCM

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

### Simulation

_RaschPy_ includes simulation classes for generating synthetic data under each model. Simulation runs automatically on instantiation; generating parameters are stored as attributes on the simulation object (`sim.items`, `sim.persons`, `sim.thresholds`, and for MFRM models `sim.facet_effects`). When a simulation object is passed directly to a model constructor, the generating parameters are also attached to the model object under a `generating` namespace, making recovery comparisons straightforward:

```python
from raschpy.simulation import SLM_Sim, RSM_Sim, PCM_Sim
from raschpy.simulation import MFRM_Sim_Global, MFRM_Sim_Items, MFRM_Sim_Thresholds, MFRM_Sim_Bivector, MFRM_Sim_Matrix

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

### Anchor calibration

To place MFRM estimate within an anchored frame of reference, relative to the mean of a set of 'gold standard' raters, pass a list of anchor raters. A new set of anchored estimates will be generated:

```python
mfrm.calibrate_global()
print(mfrm.raters_global)                    # Unanchored rater severities

anchors = ['Rater_1', 'Rater_3', 'Rater_6']
mfrm.calibrate_global_anchor(anchors)
print(mfrm.anchor_raters_global)             # Anchored rater severities
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

Linacre, J. M. (1994). _Many-Facet Rasch Measurement_. MESA Press.

Masters, G. N. (1982). A Rasch model for partial credit scoring. _Psychometrika_, _47_(2), 149–174.

Rasch, G. (1960). _Probabilistic models for some intelligence and attainment tests_. Danmarks Pædagogiske Institut.

## DISCLAIMER
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
