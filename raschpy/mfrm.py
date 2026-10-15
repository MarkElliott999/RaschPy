from math import log
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2, gaussian_kde, norm
from scipy.stats import t as t_dist
from scipy.optimize import brentq
from sklearn.decomposition import PCA
from sklearn.linear_model import HuberRegressor
from matplotlib import pyplot as plt
from matplotlib import colors as colors
from matplotlib import cm as cmx
from matplotlib.patches import Rectangle
import seaborn as sns
import matplotlib.patheffects as path_effects

from raschpy.base import Rasch, _BootstrapWarnings, _save_table


class MFRM(Rasch):
    """
    Many-Facet Rasch Model (Linacre 1994) with RSM (Andrich 1978) formulation,
        including extended facet_element representations (Elliott & Buttery 2022a).

        Supports five facet_element effect parameterisations:
          'global'     — scalar facet_effect λ_r per facet_element
          'items'      — vector λ_{r,i} per (facet_element, item)
          'thresholds' — vector λ_{r,k} per (facet_element, threshold)
          'bivector'   — additive λ_{r,i} + λ_{r,k} per (facet_element, item, threshold)
                         (facet_element as RSM; zero-sum threshold vector per facet_element)
          'matrix'     — full λ_{r,i,k} per (facet_element, item, threshold)
                         (facet_element as PCM)

    The log-numerator for person n, facet_element r, item i, category k is:
      global:     k*(θ_n − δ_i − λ_r) − Σ τ_k
          items:      k*(θ_n − δ_i − λ_{r,i}) − Σ τ_k
          thresholds: k*(θ_n − δ_i) − Σ(τ_k + λ_{r,k})
          bivector:   k*(θ_n − δ_i) − Σ(τ_k + λ_{r,i} + λ_{r,k})
                  where Σ_k λ_{r,k} = 0 for each r
          matrix:     k*(θ_n − δ_i) − Σ(τ_k + λ_{r,i,k})

    Data format: (Rater, Person) MultiIndex × Items DataFrame.
    """

    # ------------------------------------------------------------------
    # Model registry — maps model name to facet_effect attribute names
    # ------------------------------------------------------------------
    _MODELS = (
        "global", "items", "thresholds", "bivector", "matrix",
        "centrality", "pseudo_halo", "bistretch",
    )

    # Alternate, axis-based spellings for the two single-stretch models —
    # 'threshold_stretch' stretches the thresholds axis (like 'centrality'
    # stretches thresholds relative to 'thresholds'), 'item_stretch' stretches
    # the items axis (like 'pseudo_halo' stretches items relative to
    # 'items'), mirroring 'bistretch' (both axes) and the 'items'/'thresholds'
    # naming of the original models. Not members of _MODELS themselves —
    # always resolved to their canonical name before use.
    _MODEL_ALIASES = {
        "threshold_stretch": "centrality",
        "item_stretch": "pseudo_halo",
    }

    # Alternate spellings for the two raw stretch-model parameters --
    # 'global' for 'lambda' (the additive shift; == 1 stretch collapses
    # the model to 'global', and it matches 'global's own single-subscript
    # lambda_r notation), 'stretch' for 'omega' (the multiplicative
    # stretch coefficient, on whichever axis/axes the model has -- e.g.
    # self.stretch_centrality, or self.stretch_items_bistretch /
    # self.stretch_thresholds_bistretch for bistretch's two axes).
    _PARAM_ALIASES = {
        "global": "lambda",
        "stretch": "omega",
    }

    # Human-readable column labels for rater_stats_df(alias=True) -- same
    # 'global'/'stretch' distinction as _PARAM_ALIASES above, but spelled
    # out in full, and with the stretch axis named explicitly rather than
    # left generic, since a bare "Stretch" column wouldn't say which axis
    # it stretches once bistretch has two of them in the same table.
    #
    # NOTE for future closed-form raw-parameter models (e.g. a possible
    # Jin & Eckes-style addition): a new entry here is NOT automatic --
    # rater_stats_df's alias=True branch only fires for models listed in
    # its own `elif model in ("centrality", "pseudo_halo", "bistretch")`
    # check, so adding a model to _MODELS/_estimate_raters_* etc. does not
    # by itself give it alias support. Decide and add both: (1) this
    # dict's entry (raw param name -> display label) and (2) the new
    # model's name in that elif condition, or alias=True will silently do
    # nothing for it (no error, just no relabelling). save_stats' own
    # alias=False passthrough needs no separate change -- it forwards
    # unconditionally to rater_stats_df for every model, so it inherits
    # whatever rater_stats_df itself supports automatically.
    _RATER_PARAM_ALIAS_LABELS = {
        "centrality": {"lambda": "Global", "omega": "Threshold stretch"},
        "pseudo_halo": {"lambda": "Global", "omega": "Item stretch"},
        "bistretch": {
            "lambda": "Global",
            "omega_items": "Item stretch",
            "omega_thresholds": "Threshold stretch",
        },
    }

    def __getattr__(self, name):
        """
        Fallback for axis-based model-alias attribute/method names, e.g.
        self.calibrate_threshold_stretch(...) as a synonym for
        self.calibrate_centrality(...), or self.rater_stats_item_stretch_
        full_vector as a synonym for self.rater_stats_pseudo_halo_
        full_vector (the alias substring can appear anywhere in the name,
        not just at the end); and for raw stretch-parameter alternate
        spellings, e.g. self.global_centrality as a synonym for
        self.lambda_centrality, or self.stretch_items_bistretch as a
        synonym for self.omega_items_bistretch. Only triggers when normal
        attribute lookup fails, so it never shadows a real attribute.
        """
        for alias, canon in self._MODEL_ALIASES.items():
            suffix = f"_{alias}"
            if name.endswith(suffix):
                canon_name = name[: -len(suffix)] + f"_{canon}"
                if canon_name != name and hasattr(self, canon_name):
                    return getattr(self, canon_name)
            infix = f"_{alias}_"
            if infix in name:
                canon_name = name.replace(infix, f"_{canon}_")
                if canon_name != name and hasattr(self, canon_name):
                    return getattr(self, canon_name)
        for alias, canon in self._PARAM_ALIASES.items():
            prefix = f"{alias}_"
            if name.startswith(prefix):
                canon_name = f"{canon}_" + name[len(prefix):]
                if canon_name != name and hasattr(self, canon_name):
                    return getattr(self, canon_name)
            infix = f"_{alias}_"
            if infix in name:
                canon_name = name.replace(infix, f"_{canon}_")
                if canon_name != name and hasattr(self, canon_name):
                    return getattr(self, canon_name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def _attr(self, model, name, anchor=False):
        """Return the attribute name for a given model and statistic."""
        prefix = "anchor_" if anchor else ""
        suffix = f"_{model}"
        return f"{prefix}{name}{suffix}"

    def _sev_attr(self, model, anchor=False):
        """
        Attribute name holding the full reconstructed location matrix
        for the given model (same shape as facet_effects_items/
        facet_effects_thresholds -- one row per facet element (e.g. rater), one column per
        item/threshold/cell). For centrality/pseudo_halo/bistretch this
        is facet_effects_{model}_full_vector: facet_effects_{model}
        itself holds the compact (lambda, plus omega -- or omega_items
        and omega_thresholds for bistretch) parameter table instead, and
        is not usable directly for category probability computation.
        """
        prefix = "anchor_" if anchor else ""
        suffix = "_full_vector" if model in ("centrality", "pseudo_halo", "bistretch") else ""
        return f"{prefix}facet_effects_{model}{suffix}"

    def _get_params(self, model, anchor=False):
        """
        Return (item_locations, thresholds, facet_effects) for the requested model.
        Auto-triggers calibration if not yet run. facet_effects is always
        the full reconstructed location matrix -- see _sev_attr.
        """
        sev_attr = self._sev_attr(model, anchor=anchor)
        if anchor:
            diff_attr = f"anchor_items_{model}"
            thr_attr = f"anchor_thresholds_{model}"
            if not hasattr(self, diff_attr):
                if model == "mixed":
                    raise AttributeError(
                        "No anchor_facet_effects_mixed yet -- 'mixed' has "
                        "no calibrate_anchor() of its own (it's always "
                        f"derived by restricting a 'matrix' fit per {self.facet_label_eg}). "
                        "Run self.per_rater_model_selection(anchors=...) "
                        "first."
                    )
                raise AttributeError(
                    f"Anchor calibration required. "
                    f"Run self.calibrate_{model}_anchor()."
                )
        else:
            diff_attr = "items"
            thr_attr = "thresholds"
            if not hasattr(self, f"facet_effects_{model}"):
                if model == "mixed":
                    raise AttributeError(
                        "No facet_effects_mixed yet -- 'mixed' has no "
                        "calibration of its own (it's always derived by "
                        f"restricting a 'matrix' fit per {self.facet_label_eg}). Run "
                        "self.per_rater_model_selection() first."
                    )
                self.calibrate(model=model)
        if anchor:
            return (
                getattr(self, diff_attr),
                getattr(self, thr_attr),
                getattr(self, sev_attr),
            )
        return self._items_for(model), self._thresholds_for(model), getattr(self, sev_attr)

    def _items_for(self, model, anchor=False):
        """Item locations from the given sub-model's calibration. With
        estimation='cp-full-rcl' (or 'cpat-rcl') each sub-model's joint fit gives its own item
        locations (items_{model}); 'mixed' is derived from 'matrix'. Falls
        back to self.items (the latest calibration) when no per-model value
        is stored."""
        if anchor:
            return getattr(self, f"anchor_items_{model}")
        key = "matrix" if model == "mixed" else model
        return getattr(self, f"items_{key}", self.items)

    def _thresholds_for(self, model, anchor=False):
        """Thresholds from the given sub-model's calibration; see _items_for()."""
        if anchor:
            return getattr(self, f"anchor_thresholds_{model}")
        key = "matrix" if model == "mixed" else model
        return getattr(self, f"thresholds_{key}", self.thresholds)

    def _item_se_for(self, model, kind="se", anchor=False):
        """Item standard errors (kind='se') or interval bounds ('low'/'high')
        for the given sub-model; falls back to the unsuffixed (latest)
        attribute. None if absent."""
        prefix = "anchor_" if anchor else ""
        key = "matrix" if model == "mixed" else model
        return getattr(self, f"{prefix}item_{kind}_{key}", getattr(self, f"{prefix}item_{kind}", None))

    def _facet_n(self, n, count=True):
        """'{n} facet element(s) (e.g. rater(s))' for messages, at first mention."""
        label = self.facet_label_eg if n == 1 else self.facets_label_eg
        return f"{n} {label}" if count else label

    def _severity_grid(self, model):
        """
        Full (facet element (e.g. rater), item, category) location grid for `model`'s ACTUAL
        fitted facet_effects, broadcasting each model's natural
        representation (scalar/per-item/per-threshold/per-cell) up to a
        common shape. Used only for model_selection's min_effect gate.

        Every model's facet_effects (via _get_params/_sev_attr) are
        already stored in the same "raw per-category-step increment"
        convention that facet_effects_matrix uses -- see _cat_probs_mfrm,
        where a 'global' item_offset applied uniformly at every category
        k is mathematically identical to a constant per-step increment of
        that same size, cumsum'd; likewise 'items'/'thresholds' offsets.
        So a plain broadcast (no reconstruction) gives a directly
        comparable grid across every model pairing.
        """
        facet_effects = self._get_params(model)[2]
        R, I, m = self.no_of_facet_elements, self.no_of_items, self.max_score
        if model == "global":
            vals = facet_effects.reindex(self.facet_names).to_numpy()
            return np.broadcast_to(vals[:, None, None], (R, I, m))
        if model in ("items", "pseudo_halo"):
            vals = facet_effects.reindex(index=self.facet_names, columns=self.item_names).to_numpy()
            return np.broadcast_to(vals[:, :, None], (R, I, m))
        if model in ("thresholds", "centrality"):
            vals = facet_effects.reindex(index=self.facet_names, columns=range(1, m + 1)).to_numpy()
            return np.broadcast_to(vals[:, None, :], (R, I, m))
        # bivector, matrix, bistretch: MultiIndex (rater, item) x category
        mi = pd.MultiIndex.from_product([self.facet_names, self.item_names])
        return facet_effects.reindex(mi).to_numpy().reshape(R, I, m)

    def _generating_severity_grid(self, gen_model):
        """
        Full (facet element (e.g. rater), item, category) location grid for the TRUE generating
        facet effects under gen_model (self.generating.model), broadcasting
        up to a common shape exactly like _severity_grid does for the
        estimated side. Used by recovery_stats() so facet-effect
        recovery can compare across ANY pair of MFRM submodels (e.g.
        estimating 'items' against data generated under 'thresholds'),
        not just a matching pair -- self.generating.facet_effects already
        holds the right shape for every submodel (confirmed empirically:
        Series for global, (R,K) for centrality/thresholds-shaped models,
        (R,I) for pseudo_halo/items-shaped models, (R*I,K) MultiIndex for
        bivector/bistretch/matrix-shaped models), so this is a direct
        mirror of _severity_grid's own branching, just reading from
        self.generating instead of self._get_params(model).
        """
        facet_effects = self.generating.facet_effects
        R, I, m = self.no_of_facet_elements, self.no_of_items, self.max_score
        if gen_model == "global":
            vals = facet_effects.reindex(self.facet_names).to_numpy()
            return np.broadcast_to(vals[:, None, None], (R, I, m))
        if gen_model in ("items", "pseudo_halo"):
            vals = facet_effects.reindex(index=self.facet_names, columns=self.item_names).to_numpy()
            return np.broadcast_to(vals[:, :, None], (R, I, m))
        if gen_model in ("thresholds", "centrality"):
            vals = facet_effects.reindex(index=self.facet_names, columns=range(1, m + 1)).to_numpy()
            return np.broadcast_to(vals[:, None, :], (R, I, m))
        mi = pd.MultiIndex.from_product([self.facet_names, self.item_names])
        return facet_effects.reindex(mi).to_numpy().reshape(R, I, m)

    def _model_effect(self, simple, complex_):
        """max|location grid difference| between two nested models -- see _severity_grid."""
        return float(np.max(np.abs(self._severity_grid(complex_) - self._severity_grid(simple))))

    def _drop_equivalent_models(self, active_models, where):
        """
        Remove sub-models that, for this design, are the same model as another
        active sub-model, keeping the first in _MODELS (ladder) order:
        max_score = 1: thresholds, centrality == global; bivector, matrix ==
        items; bistretch == pseudo_halo. max_score = 2: centrality ==
        thresholds. 2 items: pseudo_halo == items, and with max_score = 2
        also bistretch == bivector. Warns once. Returns the same container type.
        """
        I, m = self.no_of_items, self.max_score
        same = {}
        if m == 1:
            same.update(thresholds="global", centrality="global", bivector="items", matrix="items",
                        bistretch="pseudo_halo")
        if m == 2:
            same["centrality"] = "thresholds"
        if I == 2:
            same["pseudo_halo"] = "items"
            if m == 2:
                same["bistretch"] = "bivector"

        def canon(mod):
            while mod in same:
                mod = same[mod]
            return mod
        groups = {}
        for mod in self._MODELS:
            if mod in active_models:
                groups.setdefault(canon(mod), []).append(mod)
        groups = [g for g in groups.values() if len(g) > 1]
        if not groups:
            return active_models
        design = " and ".join(([f"max_score={m}"] if m <= 2 else []) + ([f"{I} items"] if I == 2 else []))
        msg = "; ".join(" and ".join(f"'{d}'" for d in g[1:])
                        + (" is the same model as '{}' and has" if len(g) == 2
                           else " are the same model as '{}' and have").format(g[0])
                        + " been left out" for g in groups)
        warnings.warn(f"{where}: with {design}, {msg}. All other comparisons still run.",
                      UserWarning, stacklevel=3)
        gone = {d for g in groups for d in g[1:]}
        if isinstance(active_models, set):
            return active_models - gone
        return [mod for mod in active_models if mod not in gone]

    def _walk_back_effect(self, selected, active_set, min_effect):
        """
        Post-hoc min_effect walk-back for model_selection's AIC/BIC branches,
        restricted to the original five-model ladder (matrix -> bivector ->
        {items, thresholds} -> global) -- mirrors per_rater_model_selection's
        own AIC/BIC walk-back exactly, but only downgrades between two
        models that are both in active_set (models= may have excluded one
        side). centrality/pseudo_halo/bistretch selections are left
        untouched, matching per_rater_model_selection's documented scope.
        """
        if min_effect <= 0 or selected not in ("matrix", "bivector", "items", "thresholds"):
            return selected
        if selected == "matrix" and "bivector" in active_set:
            if self._model_effect("bivector", "matrix") < min_effect:
                selected = "bivector"
        if selected == "bivector" and "items" in active_set and "thresholds" in active_set:
            thresh_shape_ok = self._model_effect("items", "bivector") >= min_effect
            item_dev_ok = self._model_effect("thresholds", "bivector") >= min_effect
            if not thresh_shape_ok and not item_dev_ok:
                selected = "global"
            elif not thresh_shape_ok:
                selected = "items"
            elif not item_dev_ok:
                selected = "thresholds"
        if selected == "items" and "global" in active_set:
            if self._model_effect("global", "items") < min_effect:
                selected = "global"
        if selected == "thresholds" and "global" in active_set:
            if self._model_effect("global", "thresholds") < min_effect:
                selected = "global"
        return selected

    def _get_abils(self, model, anchor=False):
        """Return person_location estimates for the requested model. Auto-triggers if needed."""
        attr = f"anchor_persons_{model}" if anchor else f"persons_{model}"
        if not hasattr(self, attr):
            self.person_estimates(model=model, anchor=anchor)
        return getattr(self, attr)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(
        self,
        responses,
        max_score=0,
        extreme_persons=True,
        no_of_classes=5,
        facet=None,
        facet_plural=None,
        validate=True,
        exogenous=None,
        drop_facet_elements=None,
        min_persons=3,
        _keep_empty=False,
    ):
        """
        Initialise a Many-Facet Rasch Model (MFRM) object.

        The MFRM extends the RSM/PCM to include facet_element facets. Seven facet_element
        parameterisations are supported, selected at calibrate() time:
        'global' (single facet_effect per facet_element), 'items' (per-item facet_effects),
        'thresholds' (per-threshold facet_effects), 'matrix' (per-item-threshold),
        'bivector' (additive item + threshold vectors), 'centrality' (Jin & Wang
        2018-style 2-parameter shift+stretch restriction of 'thresholds'), and
        'pseudo_halo' (2-parameter shift+stretch restriction of 'items').

        Parameters
        ----------
        responses : pandas.DataFrame
            Response data with a (Rater, Person) MultiIndex and items as
            columns. Cell values should be non-negative integers from 0 to
            max_score; NaN for missing responses.
        max_score : int, default 0
            Maximum possible score per item. 0 means auto-detect from the
            data (np.nanmax). Supply explicitly to avoid issues when the
            maximum is never observed.
        extreme_persons : bool, default True
            If True, removes only persons with entirely missing data across
            all facet_elements. If False, additionally removes persons with all-zero
            or perfect total scores.
        no_of_classes : int, default 5
            Number of class intervals for observed-data overlays on plots.
        facet : str or None, default None
            Name of the facet (e.g. 'rater', 'judge', 'marker'). If given, it
            is used in messages and in attribute aliases (e.g. judge_names,
            judges_global). If None, messages say "facet element(s)", with
            "(e.g. rater)" at first mention, and the aliases use 'rater'
            (e.g. rater_names), as before.
        facet_plural : str or None, default None
            Plural of facet, if not facet + 's'.
        validate : bool, default True
            If True, checks whether the item response network is fully
            connected (i.e. all items are linked via common facet_element
            comparisons). Issues a UserWarning if the data is split into
            disconnected sub-networks, which makes item locations
            incomparable across sub-groups.
        exogenous : pandas.DataFrame or None, default None
            Optional person-level covariates (e.g. Gender, Nationality or
            Age) for differential item/facet_element functioning (DIF/DRF)
            analysis, indexed by person identifier. Values are kept as raw
            category labels. Persons in responses without a matching
            exogenous record (and vice versa) are allowed — such gaps are
            common when exogenous data is optional (e.g. for GDPR reasons)
            — and are reported via UserWarning plus the attributes
            no_exogenous_persons and exogenous_only_persons, rather than
            raising. Matches SLM/PCM/RSM's exogenous= exactly.
        drop_facet_elements : None, 'thin', 'extreme' or list, default None
            Facet elements (e.g. raters) to remove from the data before
            anything else is done. 'thin' removes those that rated fewer than
            min_persons persons; 'extreme' removes those that used only the
            lowest or only the highest category; a list removes the named
            ones. Facet elements with no responses are always removed, with a
            UserWarning. Thin and extreme facet elements that are kept are
            reported via UserWarning, with the call that would remove them
            (see also drop_facet_elements()).
        min_persons : int, default 3
            Facet elements that rated at least one but fewer than this many
            persons are reported as thin. In simulation (stretch sub-models,
            1,470 datasets), facet elements that rated 1-2 persons were the main
            source of badly wrong estimates; from 3 persons upwards the rate
            fell to a few percent.

        Attributes set
        --------------
        responses : pandas.DataFrame
            Filtered response data with (Rater, Person) MultiIndex.
        invalid_responses : pandas.DataFrame
            Rows removed based on the extreme_persons rule.
        max_score : int
            Maximum possible score per item.
        no_of_persons : int
            Number of unique persons after filtering.
        no_of_items : int
            Number of items (columns).
        no_of_raters : int
            Number of unique facet_elements.
        no_of_classes : int
            Number of class intervals for plots.
        facet_element_person_counts : pandas.Series
            Number of persons each facet element rated, before any removal.
        empty_facet_elements : list
            Facet elements with no responses (always removed).
        thin_facet_elements : list
            Facet elements that rated fewer than min_persons persons (but at
            least one), whether or not they were removed.
        extreme_facet_elements : list
            Facet elements that used only the lowest or only the highest
            score category, whether or not they were removed. Their location
            is not identified by the data (like a person with an extreme
            score); the reported value comes from the smoothing constants,
            and the other estimates are unaffected.
        dropped_facet_elements : list
            Facet elements removed: those with no responses, plus any
            removed via drop_facet_elements.
        items : pandas.Index
            Item identifiers (column names).
        facet_elements : pandas.Index
            Facet element identifiers.
        persons : pandas.Index
            Person identifiers.
        anchor_rater_names_{model} : list
            Empty list per model (global/items/thresholds/matrix) for
            anchor facet_element tracking.
        connectivity_status : dict
            Result of check_data_connectivity(), present only when
            validate=True. Always contains at least 'connected' (bool)
            and 'components_count' (int).
        """

        # 'which of the 4 top-level classes is this' -- unrelated to, and
        # never compared against, self.generating.model (which holds the
        # rater sub-parameterisation string, e.g. 'centrality', copied
        # verbatim from the sim object below -- see recovery_stats()).
        self.model = "MFRM"

        # Constructor settings, reused by drop_facet_elements() to rebuild.
        self._init_kwargs = dict(
            extreme_persons=extreme_persons, no_of_classes=no_of_classes,
            facet=facet, facet_plural=facet_plural, validate=validate,
            exogenous=exogenous, min_persons=min_persons,
        )

        # Sim-aware instantiation: store sim attributes in self.generating namespace
        from raschpy.simulation.mfrm_sim import MFRM_Sim, MFRM_Sim_Bivector
        from raschpy.base import _SimParams

        if isinstance(responses, (MFRM_Sim, MFRM_Sim_Bivector)):
            sim = responses
            self.generating = _SimParams()
            for attr, value in vars(sim).items():
                setattr(self.generating, attr, value)
            if max_score != 0 and max_score != sim.max_score:
                warnings.warn(
                    f"max_score={max_score} does not match sim.max_score={sim.max_score}. "
                    f"Using max_score={max_score}."
                )
                self.max_score = int(max_score)
            else:
                self.max_score = int(sim.max_score)
            responses = sim.responses
        else:
            self.max_score = (
                int(np.nanmax(responses)) if max_score == 0 else int(max_score)
            )

        # Validate max_score against observed data
        observed_max = int(np.nanmax(responses))
        if self.max_score < observed_max:
            raise ValueError(
                f"max_score={self.max_score} is less than the maximum observed score "
                f"({observed_max}) in the data."
            )
        if self.max_score > observed_max:
            warnings.warn(
                f"max_score={self.max_score} exceeds the maximum observed score "
                f"({observed_max}) in the data. Some score categories may be unobserved."
            )

        # Facet name: messages say "facet element(s)" unless a name was passed,
        # glossed "(e.g. rater)" at first mention; attribute aliases keep the
        # 'rater' names when none was (backward compatibility).
        if facet is None:
            facet = "rater"
            self.facet_label, self.facets_label = "facet element", "facet elements"
            self.facet_label_eg = "facet element (e.g. rater)"
            self.facets_label_eg = "facet elements (e.g. raters)"
        else:
            self.facet_label = facet
            self.facets_label = facet_plural if facet_plural is not None else facet + "s"
            self.facet_label_eg, self.facets_label_eg = self.facet_label, self.facets_label

        # Facet elements with no responses, or responses for very few persons:
        # their parameters are not (or barely) identified, and in the extended
        # sub-models they can distort other facet elements' estimates through
        # the identification constraints.
        _n_persons = responses.notna().any(axis=1).groupby(level=0, sort=False).sum()
        empty = _n_persons.index[_n_persons == 0].tolist()
        thin = _n_persons.index[(_n_persons > 0) & (_n_persons < min_persons)].tolist()
        # Facet elements that used only the lowest, or only the highest,
        # category: like a person with an extreme score, their location is not
        # identified by the data.
        _lo = responses.min(axis=1).groupby(level=0, sort=False).min()
        _hi = responses.max(axis=1).groupby(level=0, sort=False).max()
        extreme = [fe for fe in _n_persons.index
                   if _n_persons[fe] > 0 and (_hi[fe] == 0 or _lo[fe] == self.max_score)]
        # Facet elements with no responses are always removed: nothing can be
        # estimated for them and they distort the other estimates.
        if drop_facet_elements is None:
            requested = []
        elif isinstance(drop_facet_elements, str):
            if drop_facet_elements not in ("thin", "extreme"):
                raise ValueError("drop_facet_elements must be None, 'thin', 'extreme' or a list of names.")
            requested = list(thin if drop_facet_elements == "thin" else extreme)
        else:
            requested = list(drop_facet_elements)
            unknown = [fe for fe in requested if fe not in _n_persons.index]
            if unknown:
                raise ValueError(f"drop_facet_elements contains names not in the data: {unknown}.")
        self._resample = bool(_keep_empty)
        if _keep_empty:
            # internal (bootstrap resamples): keep the facet element set fixed
            empty = []
        requested = [fe for fe in requested if fe not in empty]
        dropped = empty + requested
        if dropped:
            responses = responses[~responses.index.get_level_values(0).isin(dropped)]
        if empty:
            warnings.warn(
                f"{self._facet_n(len(empty))} {'has' if len(empty) == 1 else 'have'} no responses and "
                f"{'was' if len(empty) == 1 else 'were'} removed: {empty}. Nothing can be "
                f"estimated for them, and keeping them would distort the other "
                f"{self.facets_label}' estimates.",
                UserWarning,
                stacklevel=2,
            )
        if requested:
            warnings.warn(
                f"{self._facet_n(len(requested))} removed: {requested}.", UserWarning, stacklevel=2
            )
        kept_thin = [fe for fe in thin if fe not in dropped]
        if kept_thin:
            counts = {fe: int(_n_persons[fe]) for fe in kept_thin}
            warnings.warn(
                f"{self._facet_n(len(kept_thin))} rated fewer than {min_persons} persons "
                f"(persons rated: {counts}). Their parameters are poorly determined "
                f"and, in the extended sub-models, can distort other {self.facets_label}' "
                f"estimates. To remove them, call drop_{self._facets_name(facet, facet_plural)}('thin') "
                f"(also removes {self.facets_label} with no responses) or pass a list of "
                f"names, or create the object with drop_facet_elements='thin'.",
                UserWarning,
                stacklevel=2,
            )
        kept_extreme = [fe for fe in extreme if fe not in dropped]
        if kept_extreme:
            which_end = {fe: "lowest" if _hi[fe] == 0 else "highest" for fe in kept_extreme}
            warnings.warn(
                f"{self._facet_n(len(kept_extreme))} used only the lowest or only the highest score "
                f"category ({', '.join(f'{fe}: {e}' for fe, e in which_end.items())}). Like a person "
                f"with an extreme score, {'its' if len(kept_extreme) == 1 else 'their'} location is not "
                f"identified by the data: the reported value comes from the smoothing constants. "
                f"The other estimates are unaffected. To remove "
                f"{'it' if len(kept_extreme) == 1 else 'them'}, call "
                f"drop_{self._facets_name(facet, facet_plural)}('extreme') or create the object with "
                f"drop_facet_elements='extreme'.",
                UserWarning,
                stacklevel=2,
            )
        self.facet_element_person_counts = _n_persons
        self.empty_facet_elements = empty
        self.thin_facet_elements = thin
        self.extreme_facet_elements = extreme
        self.dropped_facet_elements = dropped

        unstacked_df = responses.unstack(level=0)

        # Always remove all-NaN persons (truly invalid — no usable data across any rater)
        all_nan_mask = unstacked_df.isna().all(axis=1)
        invalid_idx = unstacked_df[all_nan_mask].index
        self.invalid_responses = responses[
            responses.index.get_level_values(1).isin(invalid_idx)
        ]
        valid_unstacked = unstacked_df[~all_nan_mask]

        if extreme_persons:
            extreme_idx = valid_unstacked.iloc[
                0:0
            ].index  # empty; no persons removed as extreme
        else:
            scores = valid_unstacked.sum(axis=1)
            max_scores = valid_unstacked.notna().sum(axis=1) * self.max_score
            extreme_mask = (scores == 0) | (scores == max_scores)
            extreme_idx = valid_unstacked[extreme_mask].index

        self.extreme_persons = responses[
            responses.index.get_level_values(1).isin(extreme_idx)
        ]
        self.responses = responses[
            ~responses.index.get_level_values(1).isin(invalid_idx.union(extreme_idx))
        ]

        self.no_of_persons = len(self.responses.index.get_level_values(1).unique())
        self.no_of_items = self.responses.shape[1]
        self.facet = facet
        self.facets = facet_plural if facet_plural is not None else facet + "s"
        self.no_of_facet_elements = len(self.responses.index.get_level_values(0).unique())
        self.no_of_raters = self.no_of_facet_elements  # alias; see facet naming
        setattr(self, f"no_of_{self.facets}", self.no_of_facet_elements)
        self.no_of_classes = no_of_classes
        self.item_names = self.responses.columns
        self.facet_names = self.responses.index.get_level_values(0).unique()
        self.rater_names = self.facet_names  # alias for default facet
        self.person_names = self.responses.index.get_level_values(1).unique()

        # Canonicalise physical row order into contiguous per-facet_element
        # blocks, each in the identical self.person_names order. item_diffs()
        # and _estimate_raters_{model}() read self.responses.values via a raw
        # positional reshape(no_of_facet_elements, no_of_persons, -1) that
        # silently misaligns persons across facet_element blocks whenever the
        # physical row order isn't exactly this (e.g. after any MultiIndex
        # .loc[(slice(None), idx), :] selection with a shuffled/resampled idx,
        # a boolean mask, or facet_elements that rate non-overlapping person
        # sets) — this has been observed to corrupt facet/rater severity
        # estimates (collapsing them toward zero, or raising IndexError)
        # while leaving item difficulty estimates unaffected, since only the
        # former needs cross-facet_element alignment for the same person.
        _index_names = self.responses.index.names
        self.responses = pd.concat(
            {fe: self.responses.xs(fe).reindex(self.person_names) for fe in self.facet_names}
        )
        self.responses.index.names = _index_names

        # Optional person-level covariates for DIF/DRF (e.g. Gender, Nationality, Age)
        if exogenous is not None:
            self.no_exogenous_persons = self.person_names[
                ~self.person_names.isin(exogenous.index)
            ]
            self.exogenous_only_persons = exogenous.index[
                ~exogenous.index.isin(self.person_names)
            ]
            self.exogenous = exogenous.reindex(self.person_names)

            if len(self.no_exogenous_persons) > 0:
                warnings.warn(
                    f"{len(self.no_exogenous_persons)} person(s) in the response data "
                    f"have no matching exogenous record (exogenous data is often "
                    f"optional, e.g. for GDPR reasons). See no_exogenous_persons for "
                    f"the full list. These persons will be excluded from any DIF/DRF "
                    f"grouping that relies on the missing covariate(s).",
                    UserWarning,
                    stacklevel=2,
                )

            if len(self.exogenous_only_persons) > 0:
                warnings.warn(
                    f"{len(self.exogenous_only_persons)} person(s) in the exogenous "
                    f"data are not present in the response data and will be ignored. "
                    f"See exogenous_only_persons for the full list.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            self.exogenous = None
            self.no_exogenous_persons = pd.Index([])
            self.exogenous_only_persons = pd.Index([])

        # Facet name aliases (e.g. self.judge_names, self.judges)
        setattr(self, f"{self.facet}_names", self.facet_names)
        setattr(self, self.facets, self.facet_names)
        setattr(self, f"empty_{self.facets}", self.empty_facet_elements)
        setattr(self, f"thin_{self.facets}", self.thin_facet_elements)
        setattr(self, f"extreme_{self.facets}", self.extreme_facet_elements)
        setattr(self, f"{self.facet}_person_counts", self.facet_element_person_counts)
        setattr(self, f"dropped_{self.facets}", self.dropped_facet_elements)
        if self.facets != "facet_elements":
            setattr(self, f"drop_{self.facets}", self.drop_facet_elements)

        # Anchor facet_element tracking per model
        for model in self._MODELS:
            setattr(self, f"anchor_rater_names_{model}", [])

        # Dynamic method aliases for facet-named stats and res_corr
        for model in self._MODELS:
            setattr(
                self,
                f"{self.facet}_stats_df_{model}",
                lambda m=model, **kw: self.rater_stats_df(model=m, **kw),
            )
            setattr(
                self,
                f"{self.facet}_res_corr_analysis_{model}",
                lambda m=model, **kw: self._run_facet_res_corr(m, **kw),
            )

        # RUN AUTOMATIC CONNECTION CHECK VALIDATION
        if validate:
            self.connectivity_status = self.check_data_connectivity()

            # check_data_connectivity() itself warns about disconnection,
            # structural zeros and weak connectivity (one warning per problem).

    # ------------------------------------------------------------------
    # Rename utilities
    # ------------------------------------------------------------------

    def _set_facet_aliases(self, model, anchor=False):
        """Set dynamic facet-named aliases for public facet_effect/SE attributes."""
        prefix = "anchor_" if anchor else ""
        # Facet effect estimates. For centrality/pseudo_halo/bistretch,
        # facet_effects_{model} itself is the compact parameter table, not
        # the full severity matrix -- {facets}_{model}/raters_{model} alias
        # the full matrix instead, stored as facet_effects_{model}_full_vector.
        sev_source = (
            f"facet_effects_{model}_full_vector"
            if model in ("centrality", "pseudo_halo", "bistretch")
            else f"facet_effects_{model}"
        )
        for attr in [
            sev_source,
            "facet_effects_bivector_items",
            "facet_effects_bivector_thresholds",
            "marginal_facet_effects_items",
            "marginal_facet_effects_thresholds",
        ]:
            if hasattr(self, f"{prefix}{attr}"):
                # Dynamic alias using actual facet name (e.g. judges_global),
                # stripping any _full_vector suffix from the alias itself.
                facet_alias = attr.replace("facet_effects", self.facets).replace(
                    "_full_vector", ""
                )
                setattr(
                    self, f"{prefix}{facet_alias}", getattr(self, f"{prefix}{attr}")
                )
                # rater_ alias for default-facet backward compatibility
                rater_alias = attr.replace("facet_effects", "raters").replace(
                    "_full_vector", ""
                )
                setattr(
                    self, f"{prefix}{rater_alias}", getattr(self, f"{prefix}{attr}")
                )
        # SE / CI attributes
        for suffix in [
            f"se_{model}",
            f"low_{model}",
            f"high_{model}",
            f"infit_ms_{model}",
            f"outfit_ms_{model}",
            f"infit_zstd_{model}",
            f"outfit_zstd_{model}",
            f"residual_correlations_{model}",
            f"loadings_{model}",
            "se_marginal_items",
            "se_marginal_thresholds",
        ]:
            canonical = f"{prefix}rater_{suffix}"
            if hasattr(self, canonical):
                setattr(
                    self, f"{prefix}{self.facet}_{suffix}", getattr(self, canonical)
                )
        # Stats table
        stats_attr = f"{prefix}rater_stats_{model}"
        if hasattr(self, stats_attr):
            setattr(
                self, f"{prefix}{self.facet}_stats_{model}", getattr(self, stats_attr)
            )

    @staticmethod
    def _facets_name(facet, facet_plural):
        """Plural facet name used in attribute/method aliases ('raters' by default)."""
        facet = "rater" if facet is None else facet
        return facet_plural if facet_plural is not None else facet + "s"

    def drop_facet_elements(self, which):
        """
        Remove facet elements (e.g. raters) from the object, e.g. after the
        warning on creation about facet elements that rated very few persons, or to exclude a facet element whose
        rating behaviour is erratic. Aliased as drop_{facet_plural}() (e.g.
        drop_raters()).

        The object is rebuilt from its current responses with the same
        constructor settings (max_score, extreme_persons, no_of_classes,
        facet, facet_plural, validate, exogenous, min_persons), so all
        calibrations and other results are cleared; self.generating (for
        objects created from a simulation) and renamed facet elements and
        persons are kept. Persons left with no responses are removed as
        invalid, as on creation.

        Parameters
        ----------
        which : 'thin', 'extreme' or list
            'thin' removes facet elements that rated fewer than min_persons
            persons; 'extreme' removes those that used only the lowest or
            only the highest score category; a list removes the named ones.

        Attributes set
        --------------
        dropped_facet_elements : list
            All facet elements removed so far, on creation and by this method.
        """
        if isinstance(which, str) and which not in ("thin", "extreme"):
            raise ValueError(
                f"which must be 'thin', 'extreme' or a list of {self.facet_label_eg} names."
            )
        if not isinstance(which, str):
            which = list(which)
            unknown = [fe for fe in which if fe not in self.facet_names]
            if unknown:
                raise ValueError(f"Names not in the data: {unknown}.")
        source = pd.concat([self.responses, self.extreme_persons])
        source = source[~source.index.duplicated()]
        kwargs = dict(self._init_kwargs, max_score=self.max_score)
        generating = getattr(self, "generating", None)
        previously_dropped = list(self.dropped_facet_elements)
        self.__dict__.clear()
        self.__init__(source, drop_facet_elements=which, **kwargs)
        if generating is not None:
            self.generating = generating
        self.dropped_facet_elements = previously_dropped + self.dropped_facet_elements
        setattr(self, f"dropped_{self.facets}", self.dropped_facet_elements)

    def rename_facet_element(self, old, new):
        """
        Rename a single facet_element in the responses.

        Validates the rename (no duplicates, no self-rename, must be a string)
        and updates self.facet_names. Prints a message rather than raising if
        validation fails.

        Parameters
        ----------
        old : str
            Current facet_element name.
        new : str
            Desired new facet_element name.
        """

        if old == new:
            warnings.warn(
                f"New {self.facet_label_eg} name is the same as the old {self.facet_label} name.",
                UserWarning,
                stacklevel=2,
            )
        elif new in self.facet_names:
            warnings.warn(
                f"New {self.facet_label_eg} name is a duplicate of an existing {self.facet_label} name.",
                UserWarning,
                stacklevel=2,
            )
        if old not in self.facet_names:
            warnings.warn(
                f"Old {self.facet_label_eg} name {old!r} not found in data.",
                UserWarning,
                stacklevel=2,
            )
        elif not isinstance(new, str):
            warnings.warn(f"{self.facet_label_eg.capitalize()} names must be strings.", UserWarning, stacklevel=2)
        else:
            new_names = [new if r == old else r for r in self.facet_names]
            self.rename_facet_elements_all(new_names)

    def rename_facet_elements_all(self, new_names):
        """
        Rename all facet_elements at once.

        Validates the new name list (correct length, no duplicates, all strings)
        and rebuilds the responses with the new facet_element index labels.

        Parameters
        ----------
        new_names : list of str
            New facet_element names in the same order as self.facet_names.
        """

        if len(new_names) != len(set(new_names)):
            warnings.warn(
                f"List of new {self.facet_label_eg} names contains duplicates.",
                UserWarning,
                stacklevel=2,
            )
        elif len(new_names) != self.no_of_facet_elements:
            warnings.warn(
                f"Incorrect number of {self.facet_label_eg} names: {len(new_names)} provided, "
                f"{self.no_of_facet_elements} {self.facets_label} in data.",
                UserWarning,
                stacklevel=2,
            )
        elif not all(isinstance(n, str) for n in new_names):
            warnings.warn(f"{self.facet_label_eg.capitalize()} names must be strings.", UserWarning, stacklevel=2)
        else:
            df_dict = {
                new: self.responses.xs(old)
                for old, new in zip(self.facet_names, new_names)
            }
            self.responses = pd.concat(df_dict.values(), keys=df_dict.keys())
            mapping = dict(zip(self.facet_names, new_names))
            self.extreme_persons = self.extreme_persons.rename(index=mapping, level=0)
            self.facet_names = self.responses.index.get_level_values(0).unique()
            self.rater_names = self.facet_names  # keep alias in sync
            setattr(self, f"{self.facet}_names", self.facet_names)
            setattr(self, self.facets, self.facet_names)

    def rename_person(self, old, new):
        """
        Rename a single person in the responses.

        Validates the rename and updates the level-1 (Person) index.
        Prints a message rather than raising if validation fails.

        Parameters
        ----------
        old : str
            Current person name.
        new : str
            Desired new person name.
        """

        if old == new:
            warnings.warn(
                "New person name is the same as the old person name.",
                UserWarning,
                stacklevel=2,
            )
        elif new in self.person_names:
            warnings.warn(
                "New person name is a duplicate of an existing person name.",
                UserWarning,
                stacklevel=2,
            )
        if old not in self.person_names:
            warnings.warn(
                f"Old person name {old!r} not found in data.", UserWarning, stacklevel=2
            )
        elif not isinstance(new, str):
            warnings.warn("Person names must be strings.", UserWarning, stacklevel=2)
        else:
            self.responses = self.responses.rename(index={old: new}, level=1)
            self.person_names = self.responses.index.get_level_values(1).unique()

    def rename_persons_all(self, new_names):
        """
        Rename all persons at once.

        Validates the new name list and rebuilds the level-1 (Person) index.

        Parameters
        ----------
        new_names : list of str
            New person names in the same order as self.person_names.
        """

        if len(new_names) != len(set(new_names)):
            warnings.warn(
                "List of new person names contains duplicates.",
                UserWarning,
                stacklevel=2,
            )
        elif len(new_names) != self.no_of_persons:
            warnings.warn(
                f"Incorrect number of person names: {len(new_names)} provided, "
                f"{self.no_of_persons} persons in data.",
                UserWarning,
                stacklevel=2,
            )
        elif not all(isinstance(n, str) for n in new_names):
            warnings.warn("Person names must be strings.", UserWarning, stacklevel=2)
        else:
            rename_map = dict(zip(self.person_names, new_names))
            self.responses = self.responses.rename(index=rename_map, level=1)
            self.person_names = self.responses.index.get_level_values(1).unique()

    # ------------------------------------------------------------------
    # Scalar probability functions (used in plots)
    # ------------------------------------------------------------------

    def cat_prob(
        self,
        person_location,
        item,
        item_locations,
        facet_element,
        facet_effects,
        category,
        thresholds,
        model="global",
    ):
        """
        Compute the probability of a response category for a single observation.

        Applies the MFRM log-numerator: k*(b - d_i) - cumsum(tau) - rater_facet_effect,
        where the facet_effect term depends on the model parameterisation.
        Numerically stabilised via log-sum-exp.

        Parameters
        ----------
        person_location : float
            Person person_location estimate on the logit scale.
        item : str
            Item identifier.
        item_locations : pandas.Series
            Item item_location estimates indexed by item name.
        facet_element : str
            Facet element (e.g. rater) identifier.
        facet_effects : Series or dict
            Facet element effect parameters. Structure depends on model:
            global — Series indexed by facet_element;
            items  — dict of Series {facet_element: Series(items)};
            thresholds — dict of arrays {facet_element: array(thresholds)};
            matrix — nested dict {facet_element: {item: array}}.
        category : int
            Response category (0 to max_score).
        thresholds : array-like
            Rasch-Andrich threshold vector, length max_score+1, thresholds[0]=0.
        model : str, default 'global'
            Facet element parameterisation: 'global', 'items', 'thresholds', or 'matrix'.

        Returns
        -------
        float
            Probability of the specified category, in [0, 1].
        """
        model = self._MODEL_ALIASES.get(model, model)
        cats = np.arange(len(thresholds) + 1, dtype=float)
        cumtau = np.concatenate([[0.0], np.cumsum(thresholds)])
        log_nums = cats * (person_location - item_locations.loc[item]) - cumtau
        # Apply facet_element effect
        if model == "global":
            log_nums -= cats * facet_effects.loc[facet_element]
        elif model in ("items", "pseudo_halo"):
            log_nums -= cats * facet_effects.loc[facet_element, item]
        elif model in ("thresholds", "centrality"):
            log_nums -= np.concatenate(
                [[0.0], np.cumsum(facet_effects.loc[facet_element].values)]
            )
        elif model in ("bivector", "matrix", "bistretch"):
            log_nums -= np.concatenate(
                [[0.0], np.cumsum(facet_effects.loc[facet_element, item].values)]
            )
        log_nums -= log_nums.max()
        nums = np.exp(log_nums)
        return nums[category] / nums.sum()

    def exp_score(
        self,
        person_location,
        item,
        item_locations,
        facet_element,
        facet_effects,
        thresholds,
        model="global",
    ):
        """
        Compute the expected score for a single person/facet_element/item combination.

        Calculates E[X | person_location, item, facet_element, model] = sum(k * P(X=k)).
        Used in scalar Newton-Raphson estimation and score_lookup().

        Parameters
        ----------
        person_location : float
            Person person_location estimate on the logit scale.
        item : str
            Item identifier.
        item_locations : pandas.Series
            Item item_location estimates.
        facet_element : str
            Facet element (e.g. rater) identifier.
        facet_effects : Series or dict
            Facet element effect parameters (structure depends on model).
        thresholds : array-like
            Rasch-Andrich threshold vector, length max_score+1.
        model : str, default 'global'
            Facet element parameterisation.

        Returns
        -------
        float
            Expected score in [0, max_score].
        """
        model = self._MODEL_ALIASES.get(model, model)
        cats = np.arange(len(thresholds) + 1, dtype=float)
        probs = np.array(
            [
                self.cat_prob(
                    person_location,
                    item,
                    item_locations,
                    facet_element,
                    facet_effects,
                    cat,
                    thresholds,
                    model,
                )
                for cat in range(len(thresholds) + 1)
            ]
        )
        return (cats * probs).sum()

    def variance(
        self,
        person_location,
        item,
        item_locations,
        facet_element,
        facet_effects,
        thresholds,
        model="global",
    ):
        """
        Compute item variance (Fisher information) for a single observation.

        Calculates Var[X | person_location, item, facet_element, model] = sum((k - E[X])^2 * P(X=k)).
        Used in scalar Newton-Raphson estimation and score_lookup().

        Parameters
        ----------
        person_location : float
            Person person_location estimate on the logit scale.
        item : str
            Item identifier.
        item_locations : pandas.Series
            Item item_location estimates.
        facet_element : str
            Facet element (e.g. rater) identifier.
        facet_effects : Series or dict
            Facet element effect parameters.
        thresholds : array-like
            Rasch-Andrich threshold vector.
        model : str, default 'global'
            Facet element parameterisation.

        Returns
        -------
        float
            Item variance / Fisher information. Always non-negative.
        """
        model = self._MODEL_ALIASES.get(model, model)
        cats = np.arange(len(thresholds) + 1, dtype=float)
        probs = np.array(
            [
                self.cat_prob(
                    person_location,
                    item,
                    item_locations,
                    facet_element,
                    facet_effects,
                    cat,
                    thresholds,
                    model,
                )
                for cat in range(len(thresholds) + 1)
            ]
        )
        exp = (cats * probs).sum()
        return ((cats - exp) ** 2 * probs).sum()

    def kurtosis(
        self,
        person_location,
        item,
        item_locations,
        facet_element,
        facet_effects,
        thresholds,
        model="global",
    ):
        """
        Compute the fourth central moment for a single person/facet_element/item.

        Calculates sum((k - E[X])^4 * P(X=k)). Used in Wilson-Hilferty
        approximation for standardised fit statistics.

        Parameters
        ----------
        person_location : float
            Person person_location estimate on the logit scale.
        item : str
            Item identifier.
        item_locations : pandas.Series
            Item item_location estimates.
        facet_element : str
            Facet element (e.g. rater) identifier.
        facet_effects : Series or dict
            Facet element effect parameters.
        thresholds : array-like
            Rasch-Andrich threshold vector.
        model : str, default 'global'
            Facet element parameterisation.

        Returns
        -------
        float
            Fourth central moment of the response distribution.
        """
        model = self._MODEL_ALIASES.get(model, model)
        cats = np.arange(len(thresholds) + 1, dtype=float)
        probs = np.array(
            [
                self.cat_prob(
                    person_location,
                    item,
                    item_locations,
                    facet_element,
                    facet_effects,
                    cat,
                    thresholds,
                    model,
                )
                for cat in range(len(thresholds) + 1)
            ]
        )
        exp = (cats * probs).sum()
        return ((cats - exp) ** 4 * probs).sum()

    # Backwards-compatible aliases for the four parameterisations
    def cat_prob_global(self, person_location, item, item_locations, facet_element, facet_effects, category, thresholds):
        """Alias for cat_prob(..., model='global'). See cat_prob for full documentation."""
        return self.cat_prob(person_location, item, item_locations, facet_element, facet_effects, category, thresholds, "global")

    def cat_prob_items(self, person_location, item, item_locations, facet_element, facet_effects, category, thresholds):
        """Alias for cat_prob(..., model='items'). See cat_prob for full documentation."""
        return self.cat_prob(person_location, item, item_locations, facet_element, facet_effects, category, thresholds, "items")

    def cat_prob_thresholds(self, person_location, item, item_locations, facet_element, facet_effects, category, thresholds):
        """Alias for cat_prob(..., model='thresholds'). See cat_prob for full documentation."""
        return self.cat_prob(person_location, item, item_locations, facet_element, facet_effects, category, thresholds, "thresholds")

    def cat_prob_matrix(self, person_location, item, item_locations, facet_element, facet_effects, category, thresholds):
        """Alias for cat_prob(..., model='matrix'). See cat_prob for full documentation."""
        return self.cat_prob(person_location, item, item_locations, facet_element, facet_effects, category, thresholds, "matrix")

    def _resolve_bivector_effects(self, facet_item_effects, facet_threshold_effects, kw):
        """
        Resolve facet_item_effects/facet_threshold_effects, also accepting via
        **kw: the dynamic {self.facet}_item_effects/{self.facet}_threshold_effects
        aliases (e.g. judge_item_effects if self.facet='judge'), and the
        rater_item_effects/rater_threshold_effects aliases, always accepted
        regardless of the configured facet name (matching self.rater_names).
        Raises TypeError if neither the canonical nor an aliased form is
        supplied, or if unrecognised keywords are passed.
        """
        if facet_item_effects is None:
            facet_item_effects = kw.pop(f"{self.facet}_item_effects", None)
        if facet_item_effects is None:
            facet_item_effects = kw.pop("rater_item_effects", None)
        if facet_threshold_effects is None:
            facet_threshold_effects = kw.pop(f"{self.facet}_threshold_effects", None)
        if facet_threshold_effects is None:
            facet_threshold_effects = kw.pop("rater_threshold_effects", None)
        if kw:
            raise TypeError(f"unexpected keyword argument(s): {list(kw)}")
        if facet_item_effects is None or facet_threshold_effects is None:
            raise TypeError(
                "facet_item_effects/facet_threshold_effects (or their "
                f"{self.facet}_item_effects/{self.facet}_threshold_effects, or "
                "rater_item_effects/rater_threshold_effects, aliases) are required."
            )
        return facet_item_effects, facet_threshold_effects

    def _bivector_severity(self, facet_item_effects, facet_threshold_effects, facet_element, item):
        """
        Combine decomposed (facet_item_effects, facet_threshold_effects) into the
        per-threshold location array for one (facet_element, item) pair:
        λ'_rik = λ_ri. + λ_r.k. Mirrors the reconstruction in _estimate_raters_bivector.
        """
        ie = (
            facet_item_effects.loc[facet_element, item]
            if isinstance(facet_item_effects, pd.DataFrame)
            else facet_item_effects[facet_element][item]
        )
        te = (
            facet_threshold_effects.loc[facet_element].values
            if isinstance(facet_threshold_effects, pd.DataFrame)
            else facet_threshold_effects[facet_element]
        )
        return np.asarray(te, dtype=float) + ie

    def cat_prob_bivector(self, person_location, item, item_locations, facet_element, facet_item_effects=None, facet_threshold_effects=None, category=None, thresholds=None, **kw):
        """
        Bivector-native category probability, taking the decomposed
        facet_item_effects/facet_threshold_effects representation directly (e.g.
        self.facet_effects_bivector_items/self.facet_effects_bivector_thresholds)
        instead of the reconstructed combined matrix. See cat_prob for the
        underlying formula.

        facet_item_effects/facet_threshold_effects may also be passed under
        their rater_item_effects/rater_threshold_effects aliases.
        """
        facet_item_effects, facet_threshold_effects = self._resolve_bivector_effects(
            facet_item_effects, facet_threshold_effects, kw
        )
        severities = self._bivector_severity(facet_item_effects, facet_threshold_effects, facet_element, item)
        cats = np.arange(len(thresholds) + 1, dtype=float)
        cumtau = np.concatenate([[0.0], np.cumsum(thresholds)])
        log_nums = cats * (person_location - item_locations.loc[item]) - cumtau
        log_nums -= np.concatenate([[0.0], np.cumsum(severities)])
        log_nums -= log_nums.max()
        nums = np.exp(log_nums)
        return nums[category] / nums.sum()
    @staticmethod
    def _stretch_value(lambda_r, scale_r, reference_value):
        """
        Shared closed-form stretch substitution: lambda_r + (scale_r - 1) *
        reference_value -- the same formula _derive_stretch_model uses to
        reconstruct centrality/pseudo_halo/bistretch's full location vector
        from their own raw (lambda_r, omega_r) parameters. Factored out here
        so the cat_prob_X/exp_score_X/variance_X/kurtosis_X convenience
        methods for these three models build the same reconstruction from a
        single canonical formula, rather than encoding it a second time.
        """
        return lambda_r + (scale_r - 1.0) * reference_value

    def _centrality_severity(self, facet_element, lambda_r, omega_r, thresholds):
        """centrality's thresholds-shaped location vector, reconstructed from its own (lambda_r, omega_r), ready for cat_prob(model='centrality')."""
        lam_rt = self._stretch_value(lambda_r, omega_r, np.asarray(thresholds, dtype=float))
        return pd.DataFrame([lam_rt], index=[facet_element])

    def _pseudo_halo_severity(self, facet_element, item, lambda_r, omega_r, item_locations):
        """pseudo_halo's items-shaped location value, reconstructed from its own (lambda_r, omega_r), ready for cat_prob(model='pseudo_halo')."""
        mu_ri = self._stretch_value(lambda_r, omega_r, item_locations.loc[item])
        return pd.DataFrame({item: [mu_ri]}, index=[facet_element])

    def _bistretch_severity(self, facet_element, item, lambda_r, omega_items_r, omega_thresholds_r, item_locations, thresholds):
        """bistretch's matrix-shaped location vector for one (facet element (e.g. rater), item), reconstructed from its own (lambda_r, omega_items_r, omega_thresholds_r), ready for cat_prob(model='bistretch')."""
        item_component = self._stretch_value(lambda_r, omega_items_r, item_locations.loc[item])
        threshold_component = self._stretch_value(0.0, omega_thresholds_r, np.asarray(thresholds, dtype=float))
        lam_rit = item_component + threshold_component
        mi = pd.MultiIndex.from_tuples([(facet_element, item)])
        return pd.DataFrame([lam_rit], index=mi)

    def cat_prob_centrality(self, person_location, item, item_locations, facet_element, lambda_r, omega_r, category, thresholds):
        """
        Category probability for the closed-form 'centrality' restriction,
        taking the facet element's (e.g. rater's) own raw parameters (location shift lambda_r,
        threshold-stretch omega_r) directly, rather than the reconstructed
        'thresholds'-shaped vector cat_prob(model='centrality') itself
        expects. See cat_prob for the underlying formula.
        """
        facet_effects = self._centrality_severity(facet_element, lambda_r, omega_r, thresholds)
        return self.cat_prob(person_location, item, item_locations, facet_element, facet_effects, category, thresholds, "centrality")
    def cat_prob_pseudo_halo(self, person_location, item, item_locations, facet_element, lambda_r, omega_r, category, thresholds):
        """
        Category probability for the closed-form 'pseudo_halo' restriction,
        taking the facet element's (e.g. rater's) own raw parameters (location shift lambda_r,
        item-difficulty-stretch omega_r) directly, rather than the
        reconstructed 'items'-shaped vector cat_prob(model='pseudo_halo')
        itself expects. See cat_prob for the underlying formula.
        """
        facet_effects = self._pseudo_halo_severity(facet_element, item, lambda_r, omega_r, item_locations)
        return self.cat_prob(person_location, item, item_locations, facet_element, facet_effects, category, thresholds, "pseudo_halo")
    def cat_prob_bistretch(self, person_location, item, item_locations, facet_element, lambda_r, omega_items_r, omega_thresholds_r, category, thresholds):
        """
        Category probability for the closed-form 'bistretch' restriction,
        taking the facet element's (e.g. rater's) own raw parameters (location shift lambda_r,
        item-difficulty-stretch omega_items_r, threshold-stretch
        omega_thresholds_r) directly, rather than the reconstructed
        'matrix'/'bivector'-shaped vector cat_prob(model='bistretch') itself
        expects. See cat_prob for the underlying formula.
        """
        facet_effects = self._bistretch_severity(facet_element, item, lambda_r, omega_items_r, omega_thresholds_r, item_locations, thresholds)
        return self.cat_prob(person_location, item, item_locations, facet_element, facet_effects, category, thresholds, "bistretch")

    def exp_score_global(self, person_location, item, item_locations, facet_element, facet_effects, thresholds):
        """Alias for exp_score(..., model='global'). See exp_score for full documentation."""
        return self.exp_score(person_location, item, item_locations, facet_element, facet_effects, thresholds, "global")

    def exp_score_items(self, person_location, item, item_locations, facet_element, facet_effects, thresholds):
        """Alias for exp_score(..., model='items'). See exp_score for full documentation."""
        return self.exp_score(person_location, item, item_locations, facet_element, facet_effects, thresholds, "items")

    def exp_score_thresholds(self, person_location, item, item_locations, facet_element, facet_effects, thresholds):
        """Alias for exp_score(..., model='thresholds'). See exp_score for full documentation."""
        return self.exp_score(person_location, item, item_locations, facet_element, facet_effects, thresholds, "thresholds")

    def exp_score_matrix(self, person_location, item, item_locations, facet_element, facet_effects, thresholds):
        """Alias for exp_score(..., model='matrix'). See exp_score for full documentation."""
        return self.exp_score(person_location, item, item_locations, facet_element, facet_effects, thresholds, "matrix")

    def exp_score_bivector(self, person_location, item, item_locations, facet_element, facet_item_effects=None, facet_threshold_effects=None, thresholds=None, **kw):
        """
        Bivector-native expected score, decomposed facet_item_effects/facet_threshold_effects
        form (also accepts the rater_item_effects/rater_threshold_effects aliases).
        See exp_score for full documentation.
        """
        facet_item_effects, facet_threshold_effects = self._resolve_bivector_effects(
            facet_item_effects, facet_threshold_effects, kw
        )
        cats = np.arange(len(thresholds) + 1, dtype=float)
        probs = np.array(
            [
                self.cat_prob_bivector(
                    person_location, item, item_locations, facet_element,
                    facet_item_effects, facet_threshold_effects, cat, thresholds,
                )
                for cat in range(len(thresholds) + 1)
            ]
        )
        return (cats * probs).sum()
    def exp_score_centrality(self, person_location, item, item_locations, facet_element, lambda_r, omega_r, thresholds):
        """Expected score for 'centrality', taking the facet element's (e.g. rater's) own raw (lambda_r, omega_r) directly. See cat_prob_centrality/exp_score for details."""
        facet_effects = self._centrality_severity(facet_element, lambda_r, omega_r, thresholds)
        return self.exp_score(person_location, item, item_locations, facet_element, facet_effects, thresholds, "centrality")
    def exp_score_pseudo_halo(self, person_location, item, item_locations, facet_element, lambda_r, omega_r, thresholds):
        """Expected score for 'pseudo_halo', taking the facet element's (e.g. rater's) own raw (lambda_r, omega_r) directly. See cat_prob_pseudo_halo/exp_score for details."""
        facet_effects = self._pseudo_halo_severity(facet_element, item, lambda_r, omega_r, item_locations)
        return self.exp_score(person_location, item, item_locations, facet_element, facet_effects, thresholds, "pseudo_halo")
    def exp_score_bistretch(self, person_location, item, item_locations, facet_element, lambda_r, omega_items_r, omega_thresholds_r, thresholds):
        """Expected score for 'bistretch', taking the facet element's (e.g. rater's) own raw (lambda_r, omega_items_r, omega_thresholds_r) directly. See cat_prob_bistretch/exp_score for details."""
        facet_effects = self._bistretch_severity(facet_element, item, lambda_r, omega_items_r, omega_thresholds_r, item_locations, thresholds)
        return self.exp_score(person_location, item, item_locations, facet_element, facet_effects, thresholds, "bistretch")

    def variance_global(self, person_location, item, item_locations, facet_element, facet_effects, thresholds):
        """Alias for variance(..., model='global'). See variance for full documentation."""
        return self.variance(person_location, item, item_locations, facet_element, facet_effects, thresholds, "global")

    def variance_items(self, person_location, item, item_locations, facet_element, facet_effects, thresholds):
        """Alias for variance(..., model='items'). See variance for full documentation."""
        return self.variance(person_location, item, item_locations, facet_element, facet_effects, thresholds, "items")

    def variance_thresholds(self, person_location, item, item_locations, facet_element, facet_effects, thresholds):
        """Alias for variance(..., model='thresholds'). See variance for full documentation."""
        return self.variance(person_location, item, item_locations, facet_element, facet_effects, thresholds, "thresholds")

    def variance_matrix(self, person_location, item, item_locations, facet_element, facet_effects, thresholds):
        """Alias for variance(..., model='matrix'). See variance for full documentation."""
        return self.variance(person_location, item, item_locations, facet_element, facet_effects, thresholds, "matrix")

    def variance_bivector(self, person_location, item, item_locations, facet_element, facet_item_effects=None, facet_threshold_effects=None, thresholds=None, **kw):
        """
        Bivector-native variance, decomposed facet_item_effects/facet_threshold_effects
        form (also accepts the rater_item_effects/rater_threshold_effects aliases).
        See variance for full documentation.
        """
        facet_item_effects, facet_threshold_effects = self._resolve_bivector_effects(
            facet_item_effects, facet_threshold_effects, kw
        )
        cats = np.arange(len(thresholds) + 1, dtype=float)
        probs = np.array(
            [
                self.cat_prob_bivector(
                    person_location, item, item_locations, facet_element,
                    facet_item_effects, facet_threshold_effects, cat, thresholds,
                )
                for cat in range(len(thresholds) + 1)
            ]
        )
        exp = (cats * probs).sum()
        return ((cats - exp) ** 2 * probs).sum()
    def variance_centrality(self, person_location, item, item_locations, facet_element, lambda_r, omega_r, thresholds):
        """Variance for 'centrality', taking the facet element's (e.g. rater's) own raw (lambda_r, omega_r) directly. See cat_prob_centrality/variance for details."""
        facet_effects = self._centrality_severity(facet_element, lambda_r, omega_r, thresholds)
        return self.variance(person_location, item, item_locations, facet_element, facet_effects, thresholds, "centrality")
    def variance_pseudo_halo(self, person_location, item, item_locations, facet_element, lambda_r, omega_r, thresholds):
        """Variance for 'pseudo_halo', taking the facet element's (e.g. rater's) own raw (lambda_r, omega_r) directly. See cat_prob_pseudo_halo/variance for details."""
        facet_effects = self._pseudo_halo_severity(facet_element, item, lambda_r, omega_r, item_locations)
        return self.variance(person_location, item, item_locations, facet_element, facet_effects, thresholds, "pseudo_halo")
    def variance_bistretch(self, person_location, item, item_locations, facet_element, lambda_r, omega_items_r, omega_thresholds_r, thresholds):
        """Variance for 'bistretch', taking the facet element's (e.g. rater's) own raw (lambda_r, omega_items_r, omega_thresholds_r) directly. See cat_prob_bistretch/variance for details."""
        facet_effects = self._bistretch_severity(facet_element, item, lambda_r, omega_items_r, omega_thresholds_r, item_locations, thresholds)
        return self.variance(person_location, item, item_locations, facet_element, facet_effects, thresholds, "bistretch")

    def kurtosis_global(self, person_location, item, item_locations, facet_element, facet_effects, thresholds):
        """Alias for kurtosis(..., model='global'). See kurtosis for full documentation."""
        return self.kurtosis(person_location, item, item_locations, facet_element, facet_effects, thresholds, "global")

    def kurtosis_items(self, person_location, item, item_locations, facet_element, facet_effects, thresholds):
        """Alias for kurtosis(..., model='items'). See kurtosis for full documentation."""
        return self.kurtosis(person_location, item, item_locations, facet_element, facet_effects, thresholds, "items")

    def kurtosis_thresholds(self, person_location, item, item_locations, facet_element, facet_effects, thresholds):
        """Alias for kurtosis(..., model='thresholds'). See kurtosis for full documentation."""
        return self.kurtosis(person_location, item, item_locations, facet_element, facet_effects, thresholds, "thresholds")

    def kurtosis_matrix(self, person_location, item, item_locations, facet_element, facet_effects, thresholds):
        """Alias for kurtosis(..., model='matrix'). See kurtosis for full documentation."""
        return self.kurtosis(person_location, item, item_locations, facet_element, facet_effects, thresholds, "matrix")

    def kurtosis_bivector(self, person_location, item, item_locations, facet_element, facet_item_effects=None, facet_threshold_effects=None, thresholds=None, **kw):
        """
        Bivector-native kurtosis, decomposed facet_item_effects/facet_threshold_effects
        form (also accepts the rater_item_effects/rater_threshold_effects aliases).
        See kurtosis for full documentation.
        """
        facet_item_effects, facet_threshold_effects = self._resolve_bivector_effects(
            facet_item_effects, facet_threshold_effects, kw
        )
        cats = np.arange(len(thresholds) + 1, dtype=float)
        probs = np.array(
            [
                self.cat_prob_bivector(
                    person_location, item, item_locations, facet_element,
                    facet_item_effects, facet_threshold_effects, cat, thresholds,
                )
                for cat in range(len(thresholds) + 1)
            ]
        )
        exp = (cats * probs).sum()
        return ((cats - exp) ** 4 * probs).sum()

    # ------------------------------------------------------------------
    # Vectorised probability engine
    # ------------------------------------------------------------------
    def kurtosis_centrality(self, person_location, item, item_locations, facet_element, lambda_r, omega_r, thresholds):
        """Kurtosis for 'centrality', taking the facet element's (e.g. rater's) own raw (lambda_r, omega_r) directly. See cat_prob_centrality/kurtosis for details."""
        facet_effects = self._centrality_severity(facet_element, lambda_r, omega_r, thresholds)
        return self.kurtosis(person_location, item, item_locations, facet_element, facet_effects, thresholds, "centrality")
    def kurtosis_pseudo_halo(self, person_location, item, item_locations, facet_element, lambda_r, omega_r, thresholds):
        """Kurtosis for 'pseudo_halo', taking the facet element's (e.g. rater's) own raw (lambda_r, omega_r) directly. See cat_prob_pseudo_halo/kurtosis for details."""
        facet_effects = self._pseudo_halo_severity(facet_element, item, lambda_r, omega_r, item_locations)
        return self.kurtosis(person_location, item, item_locations, facet_element, facet_effects, thresholds, "pseudo_halo")
    def kurtosis_bistretch(self, person_location, item, item_locations, facet_element, lambda_r, omega_items_r, omega_thresholds_r, thresholds):
        """Kurtosis for 'bistretch', taking the facet element's (e.g. rater's) own raw (lambda_r, omega_items_r, omega_thresholds_r) directly. See cat_prob_bistretch/kurtosis for details."""
        facet_effects = self._bistretch_severity(facet_element, item, lambda_r, omega_items_r, omega_thresholds_r, item_locations, thresholds)
        return self.kurtosis(person_location, item, item_locations, facet_element, facet_effects, thresholds, "bistretch")

    def _cat_probs_mfrm(
        self, person_locations, items, facet_elements, thresholds, model, facet_effects,
        item_locations=None,
    ):
        """
        Vectorised MFRM category probability engine.

        Returns dict {facet_element: ndarray (K+1, N, I)} and cats array (K+1,).

        The log-numerator for person n, facet_element r, item i, category k:
          global:     k*(θ_n − δ_i − λ_r) − Σ τ_k
          items:      k*(θ_n − δ_i − λ_{r,i}) − Σ τ_k
          thresholds: k*(θ_n − δ_i) − Σ(τ_k + λ_{r,k})
          matrix:     k*(θ_n − δ_i) − Σ(τ_k + λ_{r,i,k})

        item_locations : pandas.Series or None, default None
            Item-difficulty values to use instead of the sub-model's own
            (_items_for(model)) -- lets a
            caller evaluate this facet_element's/model's category
            probabilities against an EXTERNALLY supplied item scale (e.g.
            a pooled/combined-group model's own items), the same pattern
            `facet_effects` already supports by being passed in explicitly
            rather than read from self. Used by _log_likelihood's own
            items= override (see there for why -- axis-isolated DIF/DRF
            omnibus tests).
        """
        cats = np.arange(len(thresholds) + 1, dtype=float)  # (K+1,)
        cumtau = np.concatenate([[0.0], np.cumsum(thresholds)])  # (K+1,)
        ab = np.asarray(person_locations, dtype=float)  # (N,)
        diff_arr = (self._items_for(model) if item_locations is None else item_locations).loc[items].values  # (I,)
        n_items = len(items)

        result = {}
        for facet_element in facet_elements:
            if model == "global":
                # item_offset: scalar, same for all (i)
                item_offset = float(facet_effects.loc[facet_element])
                thresh_offset = np.zeros(len(thresholds) + 1)
            elif model in ("items", "pseudo_halo"):
                # item_offset: (I,) vector
                item_offset = facet_effects.loc[facet_element, items].values
                thresh_offset = np.zeros(len(thresholds) + 1)
            elif model in ("thresholds", "centrality"):
                item_offset = 0.0
                thresh_offset = np.concatenate(
                    [[0.0], np.cumsum(facet_effects.loc[facet_element].values)]
                )
            elif model in ("bivector", "matrix", "mixed", "bistretch"):
                item_offset = 0.0
                thresh_offset = None  # applied per-item below
            else:
                raise ValueError(f"Unknown model: {model}")

            if model in ("bivector", "matrix", "mixed", "bistretch"):
                # Build (K+1, N, I) tensor item by item
                log_num = np.zeros((len(thresholds) + 1, len(ab), n_items))
                for j, item in enumerate(items):
                    sev_rik = facet_effects.loc[facet_element, item].values
                    cumtau_total = cumtau + np.concatenate([[0.0], np.cumsum(sev_rik)])
                    log_num[:, :, j] = (
                        cats[:, None] * (ab[None, :] - diff_arr[j])
                        - cumtau_total[:, None]
                    )
            else:
                if isinstance(item_offset, np.ndarray):
                    io = item_offset[None, None, :]  # (1, 1, I)
                else:
                    io = float(item_offset)
                cumtau_total = cumtau + thresh_offset  # (K+1,)
                log_num = (
                    cats[:, None, None]
                    * (ab[None, :, None] - diff_arr[None, None, :] - io)
                    - cumtau_total[:, None, None]
                )  # (K+1, N, I)

            log_num -= log_num.max(axis=0, keepdims=True)
            probs = np.exp(log_num)
            probs /= probs.sum(axis=0, keepdims=True)
            result[facet_element] = probs

        return result, cats

    # ------------------------------------------------------------------
    # Calibration — shared components
    # ------------------------------------------------------------------

    def _remove_null_persons(self):
        """Vectorised null person removal."""
        _pd = self.responses.unstack(level=0)
        _null = _pd.isnull().all(axis=1)
        self.null_persons = _pd.index[_null].tolist()
        if self.null_persons:
            self.responses = self.responses.drop(self.null_persons, level=1)
            self.person_names = self.responses.index.get_level_values(1).unique()
        self.no_of_persons = len(self.person_names)

    def _build_pairwise_matrix(self):
        """
        Raw (unsmoothed) directed pairwise comparison matrix used by
        item_diffs() and check_data_connectivity(). Entry (i, j) counts
        persons who scored exactly one point higher on item i than item j,
        summed across facet_elements — but only comparisons made by the
        SAME facet_element count, since MFRM's PAIR estimation compares
        items within a single facet_element's ratings, not across
        facet_elements. This means two items can appear connected via
        different facet_elements individually while still being
        structurally disconnected for calibration purposes.

        Returns
        -------
        matrix : numpy.ndarray, shape (no_of_items, no_of_items)
        row_items : numpy.ndarray
            Item name for each row/column (identity mapping for MFRM).
        """
        data = (
            self.responses.values.reshape(
                self.no_of_facet_elements, self.no_of_persons, -1
            )
            .swapaxes(1, 2)
            .transpose((1, 0, 2))
        )  # (I, R, P)

        # Fold (R, P) into one axis so the sum over facet_elements and
        # persons collapses into a single matmul per category.
        flat = data.reshape(self.no_of_items, -1)
        matrix = np.zeros((self.no_of_items, self.no_of_items), dtype=np.float64)
        for category in range(self.max_score):
            higher = (flat == category + 1).astype(np.float64)
            lower = (flat == category).astype(np.float64)
            matrix += higher @ lower.T
        return matrix, np.array(self.item_names)

    def item_diffs(
        self, constant=0.1, method="log-lik", matrix_power=None, log_lik_tol=0.000001
    ):
        """PAIR item item_location estimation summing across facet_elements."""
        matrix, _ = self._build_pairwise_matrix()
        matrix_power = self._resolve_matrix_power(method, matrix_power)

        constant_matrix = ((matrix + matrix.T) > 0).astype(np.float64) * constant
        matrix += constant_matrix
        np.fill_diagonal(matrix, matrix.diagonal() + constant)

        if matrix_power == 0:
            # No powering: 'log-lik' (Bradley-Terry) consumes structural
            # zeroes directly. Fill remaining off-diagonal zeroes with
            # `constant`.
            mat = np.array(matrix, dtype=np.float64)
            if constant:
                off_diagonal_mask = ~np.eye(matrix.shape[0], dtype=bool)
                mat[off_diagonal_mask & (mat == 0)] = constant
        else:
            mat = np.linalg.matrix_power(matrix, matrix_power)
            mat_pow = matrix_power
            while 0 in mat:
                mat = mat @ matrix
                mat_pow += 1
                if mat_pow == matrix_power + 5:
                    mat += constant
                    break

        self.items = self.priority_vector(mat, method=method, log_lik_tol=log_lik_tol)

    def _threshold_distance(self, threshold, item_locations, constant=0.1):
        """
        CPAT threshold distance estimate for MFRM (estimation='pair-cpat';
        Elliott & Buttery, 2022; thesis eq. 4.54). For each pair of
        different items (i, j), the log-ratio of the persons a
        facet_element scored k on both items to those it scored k-1 on i
        and k+1 on j is taken for each facet_element with any such patterns
        (smoothing constant added per facet_element cell), and averaged over
        those facet_elements -- the facet_element terms cancel in the
        average, as in eq. 4.54 -- plus the item difference. The item pairs
        are then combined, weighted by their harmonic-mean information
        2 n d / (n + d) from the counts summed over facet_elements.

        Before v1.4.0 the counts were summed over facet_elements before
        taking the log -- the log of sums rather than the sum of logs that
        eq. 4.54 specifies -- which attenuates the distance when facet
        elements differ (logit non-collapsibility), compressing the
        threshold scale.

        An item pair with no adjacent-category co-occurrence for any
        facet_element carries no information and is dropped. At
        ``constant=0`` a facet_element cell with only one side observed is
        dropped from its pair's average (its log-ratio is undefined without
        smoothing). If every item pair is dropped, there is no information
        in the data to estimate the distance, and this raises rather than
        returning a silent NaN.
        """
        R, I = self.no_of_facet_elements, self.no_of_items
        X = self.responses.to_numpy(dtype=np.float64).reshape(R, -1, I)   # (R, P, I)
        at_k = np.nan_to_num(X == threshold).astype(np.float64)
        at_km1 = np.nan_to_num(X == threshold - 1).astype(np.float64)
        at_kp1 = np.nan_to_num(X == threshold + 1).astype(np.float64)
        num = np.matmul(at_k.transpose(0, 2, 1), at_k)        # (R, I, I); BLAS, not einsum
        den = np.matmul(at_km1.transpose(0, 2, 1), at_kp1)
        # CPAT compares two DIFFERENT items (i != j); same-item cells are not
        # comparisons and must not enter the sums.
        idx = np.arange(I)
        num[:, idx, idx] = 0.0
        den[:, idx, idx] = 0.0

        # per facet_element cell log-ratios, averaged over the facet_elements
        # with any patterns for the pair
        seen = (num + den) > 0
        usable = seen & (num + constant > 0) & (den + constant > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cell_lr = np.where(usable, np.log(np.where(usable, num + constant, 1.0))
                               - np.log(np.where(usable, den + constant, 1.0)), 0.0)
        n_used = usable.sum(axis=0)
        pair_lr = np.where(n_used > 0, cell_lr.sum(axis=0) / np.maximum(n_used, 1), 0.0)

        # pairs weighted by their information, from the summed counts
        N, D = num.sum(axis=0), den.sum(axis=0)
        weight = np.where(n_used > 0, 2.0 * (N + constant) * (D + constant) / (N + D + 2 * constant), 0.0)
        np.fill_diagonal(weight, 0.0)

        total_weight = weight.sum()
        if total_weight == 0:
            raise ValueError(
                f"CPAT threshold distance for threshold {threshold} could not be "
                f"estimated: no pair of different items has any person scoring "
                f"{threshold} on both, or {threshold - 1} on one and "
                f"{threshold + 1} on the other, so the data contain no "
                f"information on the distance between thresholds {threshold} and "
                f"{threshold + 1}. The smoothing constant cannot supply it; a larger "
                f"or better-targeted sample is needed."
            )
        diffs = item_locations.reindex(self.responses.columns).values.astype(float)
        diff_matrix = diffs[:, None] - diffs[None, :]
        return float((weight * (pair_lr + diff_matrix)).sum() / total_weight)

    def ra_thresholds(self, item_locations, constant=0.1):
        """CPAT threshold set estimation.

        A distance with no information at all (no pair of different items
        with either comparison pattern) is set to the mean of the other
        distances (0 if there are none), with a UserWarning, as RSM does by
        default.
        """
        distances, no_information = [], []
        for k in range(1, self.max_score):
            try:
                distances.append(self._threshold_distance(k, item_locations, constant))
            except ValueError:
                distances.append(np.nan)
                no_information.append(k)
        distances = np.array(distances, dtype=float)
        if no_information:
            ok = np.isfinite(distances)
            distances[~ok] = distances[ok].mean() if ok.any() else 0.0
            self._warn_no_information(no_information, stacklevel=4)
        thresholds = np.concatenate([[0.0], np.cumsum(distances)])
        thresholds -= thresholds.mean()
        return thresholds

    def _warn_no_information(self, no_information, stacklevel=3):
        """Warn that the distances in no_information (k: between thresholds k
        and k+1) have no comparison patterns and were set to a placeholder."""
        if no_information:
            many = len(no_information) > 1
            pairs = ", ".join(f"{k}-{k + 1}" for k in no_information)
            warnings.warn(
                f"The data contain no information on the distance{'s' if many else ''} "
                f"between thresholds {pairs}: no pair of different items has any person "
                f"scoring k on both, or k-1 on one and k+1 on the other. The "
                f"distance{'s have' if many else ' has'} been set to the mean of the "
                f"other distances (0 if there are none), as a placeholder; the thresholds "
                f"involved are not estimated by the data. Options: collapse "
                f"sparsely used categories, or collect a larger or better-"
                f"targeted sample.",
                UserWarning,
                stacklevel=stacklevel,
            )

    def _threshold_total_weights(self, constant=0.1):
        """
        CPAT's own total_weight (the harmonic-mean information weight
        _threshold_distance() sums over item pairs), for every threshold
        1..max_score-1 -- the MFRM analogue of RSM's
        _threshold_total_weights (MFRM shares RSM's single shared-vector
        threshold formulation), used as the effective sample size for
        dif_test's DCF dynamic SE cap (dcf_cap). Mirrors
        _threshold_distance's own weight computation exactly (not the
        distance/diff_matrix term), including its facet_element-aware
        reshape (weight is summed across facet_elements AND persons, same
        as the distance itself), so this is cheap relative to a full
        calibrate() and doesn't need item locations.

        Not yet separately simulation-calibrated for MFRM the way
        dcf_cap's default (80000, borrowed from RSM) was for RSM itself --
        MFRM's total_weight additionally pools across facet_elements, so
        its scale relative to RSM's at matched person N is not yet known.
        Treat dcf_cap as a reasonable starting point for MFRM, not a
        validated one, pending its own dynamic-N calibration study (see
        raschpy_dynamic_n_calibration project notes).

        Parameters
        ----------
        constant : float, default 0.1
            Additive smoothing constant, matching threshold_constant from
            calibrate()/std_errors() for internal consistency.

        Returns
        -------
        pandas.Series
            total_weight per threshold, indexed 1..max_score-1 (matching
            cat_width_se_{model}'s own indexing convention).
        """
        data = (
            self.responses.values.reshape(
                self.no_of_facet_elements, self.no_of_persons, -1
            )
            .swapaxes(1, 2)
            .transpose((1, 0, 2))
        )  # (I, R, P)
        flat = data.reshape(self.no_of_items, -1)
        weights = {}
        for threshold in range(1, self.max_score):
            at_k = (flat == threshold).astype(np.float64)
            at_km1 = (flat == threshold - 1).astype(np.float64)
            at_kp1 = (flat == threshold + 1).astype(np.float64)
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                num_matrix = at_k @ at_k.T
                den_matrix = at_km1 @ at_kp1.T
                # CPAT compares two DIFFERENT items (i != j); same-item cells are not
                # comparisons (den_ii is structurally 0) and must not enter the sums.
                np.fill_diagonal(num_matrix, 0.0)
                np.fill_diagonal(den_matrix, 0.0)
            valid = (num_matrix + den_matrix) > 0
            num_s = np.where(valid, num_matrix + constant, 0.0)
            den_s = np.where(valid, den_matrix + constant, 0.0)
            finite = valid & (num_s > 0) & (den_s > 0)
            weight_matrix = np.where(finite, 2.0 * num_s * den_s / (num_s + den_s), 0.0)
            weights[threshold] = weight_matrix.sum()
        return pd.Series(weights)

    # ------------------------------------------------------------------
    # Rater facet_effect estimation
    # ------------------------------------------------------------------

    def _warn_isolated_raters(self, matrix, where=None):
        """
        Record (or warn about) facet_elements with zero PAIR comparisons with
        any other facet_element in `matrix`, checked BEFORE smoothing is
        added. The log-lik (Bradley-Terry) update excludes the diagonal from
        both wins and comparisons, so a structurally zero off-diagonal
        row/column is unidentified regardless of `constant` — the smoothing
        constant only ever gets added where (matrix + matrix.T) > 0 already,
        so it cannot rescue a facet_element with no comparisons at all (worst
        at constant=0, where the estimate is entirely undefined for it).

        where labels the slice (an item, a threshold) the matrix belongs to,
        None for the whole data. During calibrate() the findings are
        collected across slices and reported in one warning
        (_emit_isolated_raters); otherwise this warns directly. Deliberately
        does not raise: 'log-lik' tolerates an incomplete comparison graph
        and still returns a usable answer for every other facet_element.
        """
        off_diag_row = matrix.sum(axis=1) - np.diagonal(matrix)
        off_diag_col = matrix.sum(axis=0) - np.diagonal(matrix)
        isolated = (off_diag_row == 0) & (off_diag_col == 0)
        if not isolated.any():
            return
        names = sorted(np.array(self.facet_names)[isolated].tolist())
        log = getattr(self, "_isolated_log", None)
        if log is not None:
            for name in names:
                log.setdefault(name, []).append(where)
            return
        self._isolated_log = {name: [where] for name in names}
        self._emit_isolated_raters(stacklevel=4)

    def _emit_isolated_raters(self, stacklevel=3):
        """One warning for every facet_element recorded by _warn_isolated_raters()."""
        log, self._isolated_log = getattr(self, "_isolated_log", None) or {}, None
        if not log:
            return
        overall = sorted(n for n, w in log.items() if None in w)
        partial = {n: [x for x in w if x is not None] for n, w in log.items() if None not in w}
        parts = []
        if overall:
            parts.append(
                f"{self._facet_n(len(overall))} {'has' if len(overall) == 1 else 'have'} zero PAIR "
                f"comparisons with any other {self.facet_label}: {overall}."
            )
        if partial:
            detail = "; ".join(f"{n} ({', '.join(map(str, w))})" for n, w in sorted(partial.items()))
            lead = self._facet_n(len(partial)) if not overall else str(len(partial))
            parts.append(
                f"{lead} {'has' if len(partial) == 1 else 'have'} zero PAIR comparisons with any other "
                f"{self.facet_label} for some items or thresholds: {detail}."
            )
        warnings.warn(
            " ".join(parts) + " Their estimates there are unidentified from the data "
            "(undefined at constant=0; otherwise driven entirely by the smoothing constant). "
            "Consider dropping them or gathering more responses.",
            UserWarning,
            stacklevel=stacklevel,
        )

    def _pair_matrix(self, data_2d, constant, where=None):
        """Build a PAIR pairwise matrix from (R, P) data and apply smoothing."""
        R = data_2d.shape[0]
        matrix = np.zeros((R, R), dtype=np.float64)
        for category in range(self.max_score):
            higher = (data_2d == category + 1).astype(np.float64)
            lower = (data_2d == category).astype(np.float64)
            matrix += higher @ lower.T
        self._warn_isolated_raters(matrix, where)
        constant_matrix = ((matrix + matrix.T) > 0).astype(np.float64) * constant
        matrix += constant_matrix
        np.fill_diagonal(matrix, matrix.diagonal() + constant)
        return matrix

    def _raise_matrix_power(self, matrix, matrix_power, constant, method="log-lik"):
        """
        Raise a matrix to a given power, incrementing until no zeros remain.

        Used internally during PAIR calibration to ensure full connectivity
        in the facet comparison matrix. If zeros persist after matrix_power + 5
        iterations, adds a smoothing constant and stops.

        Parameters
        ----------
        matrix : numpy.ndarray
            Square comparison count matrix.
        matrix_power : int or None
            Starting matrix power. ``None`` is resolved against ``method``
            (0 for ``'log-lik'``, 3 otherwise; see
            ``base.Rasch._resolve_matrix_power``). ``0`` skips powering
            entirely and fills only the off-diagonal structural zeroes with
            ``constant``.
        constant : float
            Smoothing constant added if zeros persist.
        method : str, default 'log-lik'
            Priority-vector method, used only to resolve ``matrix_power=None``.

        Returns
        -------
        numpy.ndarray
            Powered matrix with zeros resolved or smoothed.
        """
        matrix_power = self._resolve_matrix_power(method, matrix_power)
        if matrix_power == 0:
            # No powering: 'log-lik' (Bradley-Terry) consumes structural
            # zeroes directly. Fill remaining off-diagonal zeroes with
            # `constant`.
            mat = np.array(matrix, dtype=np.float64)
            if constant:
                off_diagonal_mask = ~np.eye(matrix.shape[0], dtype=bool)
                mat[off_diagonal_mask & (mat == 0)] = constant
            return mat
        mat = np.linalg.matrix_power(matrix, matrix_power)
        mat_pow = matrix_power
        while 0 in mat:
            mat = mat @ matrix
            mat_pow += 1
            if mat_pow == matrix_power + 5:
                mat += constant
                break
        return mat

    def _estimate_raters_global(
        self, constant=0.1, method="log-lik", matrix_power=None, log_lik_tol=0.000001
    ):
        """PAIR facet_element effect estimation — scalar per facet_element."""
        data = (
            self.responses.values.reshape(
                self.no_of_facet_elements, self.no_of_persons, -1
            )
            .swapaxes(1, 2)
            .transpose((1, 0, 2))
        )  # (I, R, P)

        # Fold (I, P) into one axis so the sum over items and persons
        # collapses into a single matmul per category.
        flat = data.transpose(1, 0, 2).reshape(self.no_of_facet_elements, -1)
        matrix = np.zeros(
            (self.no_of_facet_elements, self.no_of_facet_elements), dtype=np.float64
        )
        for category in range(self.max_score):
            higher = (flat == category + 1).astype(np.float64)
            lower = (flat == category).astype(np.float64)
            matrix += higher @ lower.T
        self._warn_isolated_raters(matrix)
        constant_matrix = ((matrix + matrix.T) > 0).astype(np.float64) * constant
        matrix += constant_matrix
        np.fill_diagonal(matrix, matrix.diagonal() + constant)

        mat = self._raise_matrix_power(matrix, matrix_power, constant, method)
        self.facet_effects_global = self.priority_vector(
            mat, method=method, log_lik_tol=log_lik_tol, raters=True
        )

    def _item_rater_element(
        self, item, data, constant=0.1, method="log-lik", matrix_power=None, log_lik_tol=0.000001
    ):
        """PAIR facet_element effect for a single item (items parameterisation)."""
        matrix = self._pair_matrix(data[item, :, :], constant, where=self.item_names[item])
        mat = self._raise_matrix_power(matrix, matrix_power, constant, method)
        return self.priority_vector(
            mat, method=method, log_lik_tol=log_lik_tol, raters=True
        )

    def _estimate_raters_items(
        self, constant=0.1, method="log-lik", matrix_power=None, log_lik_tol=0.000001
    ):
        """PAIR facet_element effect estimation — vector per (facet_element, item)."""
        data = (
            self.responses.values.reshape(
                self.no_of_facet_elements, self.no_of_persons, -1
            )
            .swapaxes(1, 2)
            .transpose((1, 0, 2))
        )  # (I, R, P)
        facet_elements = np.zeros((self.no_of_facet_elements, self.no_of_items))
        for i in range(self.no_of_items):
            facet_elements[:, i] = self._item_rater_element(
                i,
                data,
                constant=constant,
                method=method,
                matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
            )
        self.facet_effects_items = pd.DataFrame(
            facet_elements, index=self.facet_names, columns=self.responses.columns
        )

    def _threshold_rater_element(
        self, category, data, constant=0.1, method="log-lik", matrix_power=None, log_lik_tol=0.000001
    ):
        """PAIR facet_element effect for a single threshold (thresholds parameterisation)."""
        # Sum across items: count(X_{i,r1}==k+1 AND X_{i,r2}==k)
        matrix = np.zeros((self.no_of_facet_elements, self.no_of_facet_elements))
        for i in range(self.no_of_items):
            at_k = (data[i, :, :] == category + 1).astype(np.float64)  # (R, P)
            at_km1 = (data[i, :, :] == category).astype(np.float64)
            matrix += at_k @ at_km1.T

        matrix = matrix.astype(np.float64)
        self._warn_isolated_raters(matrix, where=f"threshold {category + 1}")
        constant_matrix = ((matrix + matrix.T) > 0).astype(np.float64) * constant
        matrix += constant_matrix
        np.fill_diagonal(matrix, matrix.diagonal() + constant)

        mat = self._raise_matrix_power(matrix, matrix_power, constant, method)
        return self.priority_vector(
            mat, method=method, log_lik_tol=log_lik_tol, raters=True
        )

    def _estimate_raters_thresholds(
        self, constant=0.1, method="log-lik", matrix_power=None, log_lik_tol=0.000001
    ):
        """PAIR facet_element effect estimation — vector per (facet_element, threshold)."""
        data = (
            self.responses.values.reshape(
                self.no_of_facet_elements, self.no_of_persons, -1
            )
            .swapaxes(1, 2)
            .transpose((1, 0, 2))
        )  # (I, R, P)
        facet_elements = np.zeros((self.no_of_facet_elements, self.max_score))
        for k in range(self.max_score):
            facet_elements[:, k] = self._threshold_rater_element(
                k,
                data,
                constant=constant,
                method=method,
                matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
            )
        self.facet_effects_thresholds = pd.DataFrame(
            facet_elements, index=self.facet_names, columns=range(1, self.max_score + 1)
        )

    def _theil_sen_fit(self, ref_arr, dev_arr):
        """
        Classical univariate Theil-Sen: slope = median of all pairwise
        slopes (dev_i - dev_j) / (ref_i - ref_j) over every pair of
        points; intercept = Theil's median-residual estimator,
        median_k(dev_k - slope * ref_k). Closed-form, no iteration --
        deliberately not sklearn.linear_model.TheilSenRegressor, whose
        generality (multivariate support, spatial-median/Weiszfeld
        iteration, subsampling for large inputs) is unneeded machinery
        for what is always a single-predictor fit here, and was
        empirically slower with no accuracy benefit (see
        raschpy_omega_regression_study memory note). Used both for
        regression='theil-sen' and as the automatic per-facet_element
        fallback when regression='huber' fails to converge.
        """
        n = len(dev_arr)
        if n < 2:
            return float(dev_arr.mean()) if n else 0.0, 0.0
        slopes = [
            (dev_arr[i] - dev_arr[j]) / (ref_arr[i] - ref_arr[j])
            for i in range(n)
            for j in range(i + 1, n)
            if ref_arr[i] != ref_arr[j]
        ]
        slope = float(np.median(slopes)) if slopes else 0.0
        lam = float(np.median(dev_arr - slope * ref_arr))
        return lam, slope

    def _derive_stretch_model(self, deviation_df, reference_values, regression="huber"):
        """
        Shared closed-form 2-parameter (shift lambda_r + stretch scale_r)
        reduction of a fully free per-facet-element deviation table (the K-length
        'thresholds' or I-length 'items' facet_effects), recovered directly
        from an already-calibrated fit rather than via new iterative
        estimation. Used by both 'centrality' (restricts
        'thresholds') and 'pseudo_halo' (restricts 'items') -- same underlying
        recipe, transplanted onto whichever facet is being restricted.

        deviation = lambda_r + (scale_r - 1) * reference_value, so
        scale_r == 1 is neutral (collapses to 'global': deviation == lambda_r
        for every element) and scale_r is NOT bounded to be positive --
        scale_r <= 0 reverses (scale_r < 0) or collapses (scale_r == 0) the
        reference facet's order for that facet element (e.g. rater), a disorder situation
        flagged by the caller via _warn_nonpositive_stretch, not clipped
        here.

        Both lambda_r and scale_r are recovered together, per facet_element,
        from a single robust regression of the raw deviation values against
        the raw reference values (with intercept) -- NOT the differenced/
        through-origin OLS this package used previously. Empirically
        (see raschpy_omega_regression_study memory note), robust regression
        on raw values recovered scale_r substantially more accurately than
        differenced OLS or differenced WLS across every tested condition
        (varying facet_element counts, item counts, category counts), at
        negligible per-facet_element cost (microseconds to low
        milliseconds) -- WLS's own bootstrap-derived weights were both far
        more expensive AND not reliably more accurate, so were dropped
        entirely rather than offered as a third option.

        Parameters
        ----------
        deviation_df : pandas.DataFrame, shape (facet elements, elements)
            e.g. self.facet_effects_thresholds or self.facet_effects_items.
        reference_values : pandas.Series, indexed by deviation_df.columns
            e.g. self.thresholds or self.items.
        regression : {'huber', 'theil-sen'}, default 'huber'
            'huber' uses sklearn.linear_model.HuberRegressor (an M-estimator
            that downweights, rather than discards, points with large
            residuals) -- the most accurate option in every tested
            condition. 'theil-sen' uses the closed-form median-of-pairwise-
            slopes estimator (_theil_sen_fit) -- close behind huber in
            accuracy, fully deterministic (no iterative fit to converge).
            If regression='huber' fails for a given facet_element (no
            failure was observed in extensive testing, but the fit is
            iterative and failure conditions can't be ruled out for every
            future dataset), that facet_element automatically falls back
            to 'theil-sen' with a UserWarning naming it -- the rest of the
            facet_elements are unaffected.

        Returns
        -------
        lam, scale : pandas.Series, indexed by deviation_df.index (facet elements)
        reconstructed : pandas.DataFrame, same shape/columns as deviation_df
        """
        if regression not in ("huber", "theil-sen"):
            raise ValueError(
                f"regression must be 'huber' or 'theil-sen', got {regression!r}."
            )

        ref = reference_values.reindex(deviation_df.columns).astype(float)
        ref_arr = ref.values
        ref_col = ref_arr.reshape(-1, 1)

        lam = pd.Series(index=deviation_df.index, dtype=float)
        scale = pd.Series(index=deviation_df.index, dtype=float)

        for facet_element in deviation_df.index:
            dev_arr = deviation_df.loc[facet_element].values.astype(float)
            if regression == "huber":
                try:
                    hr = HuberRegressor(epsilon=1.5, max_iter=200).fit(ref_col, dev_arr)
                    lam_r, slope_r = float(hr.intercept_), float(hr.coef_[0])
                except Exception as e:
                    warnings.warn(
                        f"Huber regression failed for {self.facet_label_eg} "
                        f"{facet_element!r} while deriving stretch parameters "
                        f"({type(e).__name__}: {e}) -- falling back to "
                        f"Theil-Sen for this {self.facet_label} only.",
                        UserWarning,
                        stacklevel=2,
                    )
                    lam_r, slope_r = self._theil_sen_fit(ref_arr, dev_arr)
            else:
                lam_r, slope_r = self._theil_sen_fit(ref_arr, dev_arr)
            lam[facet_element] = lam_r
            scale[facet_element] = 1.0 + slope_r

        reconstructed = pd.DataFrame(
            {
                col: lam + (scale - 1.0) * ref.loc[col]
                for col in deviation_df.columns
            },
            index=deviation_df.index,
        )[deviation_df.columns]

        return lam, scale, reconstructed

    def _derive_stretch_single(self, deviation_1d, reference_values, regression="huber"):
        """Single-facet-element convenience wrapper around _derive_stretch_model, for
        callers (e.g. per_rater_model_selection) working with one facet element's (e.g. rater's)
        plain numpy deviation vector rather than the usual (facet elements x
        elements) DataFrame. deviation_1d must already be ordered to match
        reference_values.index."""
        df = pd.DataFrame([deviation_1d], index=["_single"], columns=reference_values.index)
        lam, scale, recon = self._derive_stretch_model(df, reference_values, regression=regression)
        return float(lam.iloc[0]), float(scale.iloc[0]), recon.iloc[0].values

    def _warn_nonpositive_stretch(self, scale, param_name, model_name):
        """Flag (not clip) facet_elements whose threshold stretch is <= 0
        (centrality's omega, bistretch's omega_thresholds): their thresholds
        collapse to one point (0) or reverse order (< 0), so the rating
        scale functions almost dichotomously -- a valid result, but an
        extreme one worth checking. Item stretches (pseudo_halo's omega,
        bistretch's omega_items) are flagged only when < 0 (reversed item
        difficulty order): omega_items = 0 is the limiting case of the
        pseudo-halo effect itself (all items treated as equally
        difficult)."""
        if model_name == "pseudo_halo" or param_name == "omega_items":
            bad = scale[scale < 0]
            if len(bad):
                warnings.warn(
                    f"{self._facet_n(len(bad))} {'has' if len(bad) == 1 else 'have'} {param_name} < 0 under the "
                    f"'{model_name}' model: {list(bad.index)}. They reverse the "
                    f"order of item difficulty: items that are harder overall "
                    f"are easier for them. This goes beyond the pseudo-halo "
                    f"effect, whose limit is {param_name} = 0 (all items "
                    f"treated as equally difficult). This is a valid result, "
                    f"not an estimation failure, but worth checking.",
                    UserWarning,
                    stacklevel=2,
                )
            return
        bad = scale[scale <= 0]
        if len(bad):
            warnings.warn(
                f"{self._facet_n(len(bad))} {'has' if len(bad) == 1 else 'have'} {param_name} <= 0 under the "
                f"'{model_name}' model: {list(bad.index)}. Their thresholds "
                f"collapse to a single point ({param_name} = 0), where every "
                f"category is equally likely, or reverse order ({param_name} "
                f"< 0), where the middle categories are less likely still. "
                f"Either way the lowest or highest category is the most likely "
                f"response almost everywhere, so the rating scale functions "
                f"almost dichotomously -- an extreme form of extreme response "
                f"tendency. This is a valid result, not an estimation failure, "
                f"but worth checking.",
                UserWarning,
                stacklevel=2,
            )

    def _bh_correct(self, pvals, alpha=0.05):
        """Benjamini-Hochberg FDR correction, applied within the given
        1-D array of p-values (one 'family' of tests -- callers decide
        the family, e.g. one facet element's items). Returns a boolean array,
        same length/order as pvals."""
        pvals = np.asarray(pvals, dtype=float)
        m = len(pvals)
        if m == 0:
            return np.zeros(0, dtype=bool)
        order = np.argsort(pvals)
        ranked = pvals[order]
        below = ranked <= (np.arange(1, m + 1) / m) * alpha
        if not below.any():
            return np.zeros(m, dtype=bool)
        cutoff = ranked[np.max(np.where(below)[0])]
        return pvals <= cutoff

    def _axis_divergence_test(
        self, free_est, stretch_est, se_diff, test, correction, alpha, no_of_samples
    ):
        """
        Per-facet-element, per-element divergence test comparing a fully-free
        per-facet-element deviation table (free_est, e.g. self.facet_effects_items)
        against a stretch-restricted model's reconstruction of the same
        table (stretch_est, e.g. self.facet_effects_pseudo_halo_full_vector)
        -- an item-by-item (or threshold-by-threshold) indication of where
        the simpler stretch representation may be losing predictiveness,
        as a complement to the aggregate lambda_r/omega_r summary.

        se_diff is the SE of (free_est - stretch_est) itself, from a
        single PAIRED bootstrap (see _get_divergence_inputs) -- NOT
        sqrt(free_se**2 + stretch_se**2) combining two independently-
        bootstrapped SEs, which would assume free_est and stretch_est are
        independent. They are not: stretch_est is derived FROM free_est
        (a regression fit or marginal mean of it), so there is a real,
        positive mechanical correlation between them, and ignoring it
        would make the test conservative (understate power) -- confirmed
        empirically for both pseudo_halo-vs-items and bivector-vs-matrix,
        with comparable magnitude for realistic item/threshold counts
        (not a problem unique to one pair, despite an earlier, incorrect
        claim to that effect -- see raschpy memory notes).

        Empirically, this cleanly separates facet elements (e.g. raters) whose data genuinely
        supports the stretch restriction (near-zero elements flagged,
        even for facet elements with a true omega_r close to 1, where the
        aggregate R² diagnostic that was tried and dropped gave
        misleadingly low values) from facet elements/datasets with no true
        stretch structure at all (most elements flagged) -- including
        after Benjamini-Hochberg correction. A robust-z alternative
        (median/MAD, as used elsewhere in this package for anchor
        homogeneity/DIF flagging) was also tried and rejected: it looks
        for a minority of outlier elements against an otherwise-
        consistent set, but a genuinely misspecified stretch model
        typically fails fairly UNIFORMLY across elements, which robust z
        cannot detect (nothing looks unusual relative to its equally-bad
        peers). Wald/t, testing each element against its own paired
        uncertainty independent of the others, does not have this blind
        spot.

        Parameters
        ----------
        test : {'wald', 't'}
            'wald' references the test statistic against the standard
            normal distribution (asymptotic). 't' instead uses a plain
            Student's t-distribution with no_of_samples-1 degrees of
            freedom -- NOT a Welch-Satterthwaite correction (that applies
            specifically to combining two INDEPENDENT samples' variances,
            which no longer describes this situation now that se_diff
            comes from one paired bootstrap with a single variance
            source); this is the textbook one-sample/paired-t treatment
            of "the SE estimate itself is less certain with fewer
            bootstrap draws". In practice converges to the Wald result
            once no_of_samples is a few hundred (e.g. at n=300, p-values
            near the alpha=0.05 boundary differ by ~2%; even at a strong
            z=3.5 effect, by ~15%, and both are already far below any
            realistic alpha there) -- but is offered anyway, at zero
            extra computational cost (reuses se_diff exactly as computed
            for 'wald'), because it gives a real, if modest, correction
            for anyone who deliberately drops no_of_samples for speed
            (at no_of_samples=100, the same z=3.5 case differs by ~50%).
            A fully rigorous studentized/"bootstrap-t" correction (nested
            bootstrap, re-estimating each resample's own SE) was
            considered and rejected: it multiplies the bootstrap cost by
            the inner sample size (50-200x), for a benefit expected to be
            modest here, since free_est/stretch_est are themselves built
            from many persons' responses and so already reasonably
            well-approximated by the CLT -- a bad cost/benefit trade
            given this plain 't' correction is already free to keep.
        correction : {None, 'bh'}
            If 'bh', Benjamini-Hochberg FDR correction is applied WITHIN
            each facet element's own family of tests (that facet element's items, or that
            facet element's thresholds) -- not pooled across facet elements. If None
            (default), raw p-values are compared against alpha directly.
        alpha : float
            Significance threshold (post-correction, if any).
        no_of_samples : int
            The no_of_samples used for the paired bootstrap -- needed for
            test='t''s degrees of freedom, not otherwise used.

        Returns
        -------
        p : pandas.DataFrame, same shape as free_est
        flagged : pandas.DataFrame of bool, same shape as free_est
        summary : pandas.DataFrame, index=free_est.index (facet elements),
            columns ['Flagged', 'Total'] -- per-facet-element flagged element
            count and the total number of elements tested.
        """
        if test not in ("wald", "t"):
            raise ValueError(f"test must be 'wald' or 't', got {test!r}.")
        if correction not in (None, "bh"):
            raise ValueError(f"correction must be None or 'bh', got {correction!r}.")

        diffs = free_est - stretch_est
        with np.errstate(invalid="ignore", divide="ignore"):
            z = diffs / se_diff

        if test == "t":
            p = pd.DataFrame(
                2 * (1 - t_dist.cdf(np.abs(z.values), no_of_samples - 1)),
                index=z.index, columns=z.columns,
            )
        else:
            p = pd.DataFrame(
                2 * (1 - norm.cdf(np.abs(z.values))), index=z.index, columns=z.columns
            )

        if correction == "bh":
            flagged = pd.DataFrame(index=p.index, columns=p.columns, dtype=bool)
            for r in p.index:
                flagged.loc[r] = self._bh_correct(p.loc[r].values, alpha=alpha)
        else:
            flagged = p < alpha

        summary = pd.DataFrame(
            {"Flagged": flagged.sum(axis=1).astype(int), "Total": flagged.shape[1]},
            index=free_est.index,
        )
        return p, flagged, summary

    def _fit_divergence_parent(self, model, parent, anc, samples):
        """
        Fit the parent model for the divergence test when the stretch model
        was not derived from it (estimation='cp-full-rcl' (or 'cpat-rcl') fits a stretch model
        directly, so its resamples carry no parent calibration). The parent
        is calibrated on each stored resample, in place -- still paired with
        that resample's stretch fit -- and on a copy of self, which is
        returned, so self's latest-calibration attributes are unchanged.
        Anchored: the parent is anchor-calibrated on the stretch model's
        anchors, with the full-data anchor adjustment fixed across resamples.
        The data issues these refits would warn about were already reported
        by the stretch model's own calibration and bootstrap, so they are not
        repeated.
        """
        import copy
        kw = self._refit_kw(parent)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = copy.copy(self)
            ref.calibrate(**kw, model=parent)
            anchors = adj = None
            if anc:
                anchors = list(getattr(self, f"anchor_rater_names_{model}"))
                ref.calibrate_anchor(parent, anchors)
                adj = ref._extract_anchor_adj(parent, anchors)
            for s in samples:
                s.calibrate(**kw, model=parent)
                if anc:
                    s.calibrate_anchor(parent, anchors, adj=adj)
        return ref

    def _get_divergence_inputs(self, model, anc):
        """
        Return {axis: (free_est, stretch_est, se_diff)} for the given
        stretch model, comparing against its own parent/vector model:
        'thresholds' for centrality, 'items' for pseudo_halo, 'bivector'
        (both axes) for bistretch.

        se_diff comes from a SINGLE, PAIRED bootstrap, not two
        independent ones: every resample used to bootstrap the stretch
        model's own SE already calibrates its free/vector parent as a
        byproduct (_estimate_raters_centrality calls
        _estimate_raters_thresholds first, _estimate_raters_pseudo_halo
        calls _estimate_raters_items first, _estimate_raters_bistretch
        calls _estimate_raters_bivector first), so for a given resample s,
        free_s and stretch_s come from the exact same resampled dataset.
        diffs_s = free_s - stretch_s per resample, se_diff = SD of
        diffs_s across resamples -- this correctly incorporates
        Cov(free_s, stretch_s) (real and positive, since stretch_est is
        derived from free_est) without estimating it separately, and
        without a second bootstrap pass: self._bootstrap_samples_{model}
        is already guaranteed to exist here, since rater_stats_df
        requests store_bootstrap=True on its own _ensure_se(model, ...)
        call whenever divergence_test is set, before reaching this
        method. Some resamples may drop a facet element (e.g. rater) entirely (excluded by
        chance from that resample); these are filtered out per axis,
        matching the same shape-check convention used in
        _store_rater_se.

        anc controls whether the anchor-calibrated (anchor_facet_effects_*)
        or unanchored (facet_effects_*) attributes are read, both on self
        (for the actual, non-bootstrap point estimates) and on each
        resample (populated by std_errors()'s bootstrap loop for the
        unanchored case, or anchor_std_errors()'s calibrate_anchor() loop
        over the SAME stored samples for the anchored case).
        """
        prefix = "anchor_" if anc else ""
        n_rows = self.no_of_facet_elements
        samples = getattr(self, f"_bootstrap_samples_{model}")
        parent = {"centrality": "thresholds", "pseudo_halo": "items", "bistretch": "bivector"}[model]
        free_attrs = ([f"{prefix}facet_effects_bivector_items", f"{prefix}facet_effects_bivector_thresholds"]
                      if model == "bistretch" else [f"{prefix}facet_effects_{parent}"])
        ref = self
        if any(not hasattr(s, a) for s in samples for a in free_attrs):
            ref = self._fit_divergence_parent(model, parent, anc, samples)

        def _paired(free_attr, stretch_attr):
            free_est = getattr(ref, free_attr)
            stretch_est = getattr(self, stretch_attr)
            valid = [
                s for s in samples
                if getattr(s, free_attr).shape[0] == n_rows
                and getattr(s, stretch_attr).shape[0] == n_rows
            ]
            diffs = np.array(
                [
                    getattr(s, free_attr).values - getattr(s, stretch_attr).values
                    for s in valid
                ]
            )
            se_diff = pd.DataFrame(
                np.nanstd(diffs, axis=0), index=free_est.index, columns=free_est.columns
            )
            return free_est, stretch_est, se_diff

        if model == "centrality":
            return {
                "thresholds": _paired(
                    f"{prefix}facet_effects_thresholds",
                    f"{prefix}facet_effects_centrality_full_vector",
                )
            }

        if model == "pseudo_halo":
            return {
                "items": _paired(
                    f"{prefix}facet_effects_items",
                    f"{prefix}facet_effects_pseudo_halo_full_vector",
                )
            }

        # bistretch
        return {
            "items": _paired(
                f"{prefix}facet_effects_bivector_items",
                f"{prefix}facet_effects_bistretch_items",
            ),
            "thresholds": _paired(
                f"{prefix}facet_effects_bivector_thresholds",
                f"{prefix}facet_effects_bistretch_thresholds",
            ),
        }

    def _build_fit_plot(
        self,
        model,
        div_inputs,
        div_results,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        figsize=(8, 6),
        font="Times New Roman",
        title_font_size=15,
        axis_font_size=12,
        labelsize=12,
        title=None,
        filename=None,
        file_format="png",
        dpi=300,
    ):
        """
        Scatter of the free vector model's per-element estimates against
        the stretch model's reconstruction of them (pooling every facet element (e.g. rater)
        and every element across whichever axis/axes div_inputs covers --
        one axis for centrality/pseudo_halo, both for bistretch), flagged
        elements in a different colour, with a y=x identity line -- same
        style as plot_anchor_selection/self.anchor_plot. Called from
        rater_stats_df when divergence_test is set and plot=True, which
        stores the resulting figure as self.fit_plot_{model} (e.g.
        self.fit_plot_pseudo_halo) -- like self.anchor_plot, this is a
        stored attribute, not a separate callable method; the usual
        model-alias spellings (e.g. self.fit_plot_item_stretch for
        self.fit_plot_pseudo_halo) work automatically via __getattr__
        once set. Not meant to be called directly with hand-built
        div_inputs/div_results -- use rater_stats_df(..., plot=True)
        instead.
        """
        free_vals, stretch_vals, flagged_vals = [], [], []
        for axis in div_inputs:
            free_est, stretch_est, _ = div_inputs[axis]
            _, flagged, _ = div_results[axis]
            free_vals.append(free_est.values.flatten())
            stretch_vals.append(stretch_est.values.flatten())
            flagged_vals.append(flagged.values.flatten())
        free_vals = np.concatenate(free_vals)
        stretch_vals = np.concatenate(stretch_vals)
        flagged_vals = np.concatenate(flagged_vals)

        lo = min(free_vals.min(), stretch_vals.min())
        hi = max(free_vals.max(), stretch_vals.max())
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        xmin = lo - pad if xmin is None else xmin
        xmax = hi + pad if xmax is None else xmax
        ymin = lo - pad if ymin is None else ymin
        ymax = hi + pad if ymax is None else ymax

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")

        not_flagged = ~flagged_vals
        ax.scatter(
            free_vals[not_flagged], stretch_vals[not_flagged],
            s=40, alpha=0.7, edgecolors="k", label="Not flagged",
        )
        if flagged_vals.any():
            ax.scatter(
                free_vals[flagged_vals], stretch_vals[flagged_vals],
                s=40, alpha=0.7, edgecolors="k", label="Flagged",
            )

        xseq = np.linspace(xmin, xmax, 100)
        ax.plot(xseq, xseq, color="black", lw=1, label="Identity")

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(
            "Free vector model", fontsize=axis_font_size, fontweight="bold", fontname=font
        )
        ax.set_ylabel(
            "Stretch model (reconstructed)",
            fontsize=axis_font_size,
            fontweight="bold",
            fontname=font,
        )
        if title:
            ax.set_title(title, fontsize=title_font_size, fontweight="bold")
        ax.tick_params(axis="x", labelsize=labelsize)
        ax.tick_params(axis="y", labelsize=labelsize)
        ax.legend()

        if filename is not None:
            fig.savefig(f"{filename}.{file_format}", dpi=dpi)

        plt.show(block=False)
        plt.pause(0.001)

        return fig

    def _estimate_raters_centrality(
        self, constant=0.1, method="log-lik", matrix_power=None, log_lik_tol=0.000001,
        regression="huber",
    ):
        """
        Jin & Wang (2018)-style facet element (e.g. rater) centrality/extremity facet_element
        effect -- a 2-parameter (lambda_r shift, omega_r stretch) restriction
        of the unrestricted 'thresholds' model. See _derive_stretch_model
        for the shared closed-form recipe (including the regression=
        argument). omega_r is Jin & Wang's own notation for this
        threshold-stretch coefficient.

        omega_r > 1 spreads reference thresholds apart for that facet element
        ("central" -- a bigger person-location gap is needed to cross into
        a more extreme category, so extreme categories are under-used);
        0 < omega_r < 1 compresses them ("extreme" -- smaller gaps suffice,
        so extreme categories are reached too readily); omega_r == 1 is
        neutral and collapses to 'global'.
        """
        self._estimate_raters_thresholds(
            constant=constant,
            method=method,
            matrix_power=matrix_power,
            log_lik_tol=log_lik_tol,
        )
        lam, omega, facet_effects = self._derive_stretch_model(
            self.facet_effects_thresholds, self.thresholds, regression=regression
        )
        self.lambda_centrality = lam
        self.omega_centrality = omega
        self.facet_effects_centrality_full_vector = facet_effects
        self.facet_effects_centrality = pd.DataFrame({"lambda": lam, "omega": omega})
        self._warn_nonpositive_stretch(omega, "omega", "centrality")

    def _estimate_raters_pseudo_halo(
        self, constant=0.1, method="log-lik", matrix_power=None, log_lik_tol=0.000001,
        regression="huber",
    ):
        """
        (Pseudo-)halo-effect facet_element effect -- item-facet analogue
        of centrality/extremity, a 2-parameter (lambda_r shift, omega_r
        stretch) restriction of the unrestricted 'items' model. See
        _derive_stretch_model for the shared closed-form recipe (including
        the regression= argument).

        omega_r < 1 compresses the spread of reference item locations
        toward zero for that facet element (e.g. rater) -- items look more similar in
        difficulty to that facet element than they really are, the halo
        signature (ironing out real item-difficulty differences).
        omega_r > 1 exaggerates real item-difficulty differences instead.
        omega_r == 1 is neutral and collapses to 'global'.

        Tests only compressed item-difficulty *perception* under an
        otherwise locally-independent response process -- NOT classic
        halo effect's usual mechanism (elevated local item interdependence
        / residual correlation for that facet element). Pair with
        item_res_corr_analysis/facet_res_corr_analysis run per-facet-element to
        check for that separately; omega_r alone should not be read as
        "this facet element exhibits halo effect."
        """
        self._estimate_raters_items(
            constant=constant,
            method=method,
            matrix_power=matrix_power,
            log_lik_tol=log_lik_tol,
        )
        lam, omega, facet_effects = self._derive_stretch_model(
            self.facet_effects_items, self.items, regression=regression
        )
        self.lambda_pseudo_halo = lam
        self.omega_pseudo_halo = omega
        self.facet_effects_pseudo_halo_full_vector = facet_effects
        self.facet_effects_pseudo_halo = pd.DataFrame({"lambda": lam, "omega": omega})
        self._warn_nonpositive_stretch(omega, "omega", "pseudo_halo")

    def _matrix_rater_element(
        self,
        item,
        category,
        data,
        constant=0.1,
        method="log-lik",
        matrix_power=None,
        log_lik_tol=0.000001,
    ):
        """PAIR facet_element effect for a single (item, category) cell (matrix param)."""
        at_k = (data[item, :, :] == category + 1).astype(np.float64)  # (R, P)
        at_km1 = (data[item, :, :] == category).astype(np.float64)
        matrix = at_k @ at_km1.T

        matrix = matrix.astype(np.float64)
        constant_matrix = ((matrix + matrix.T) > 0).astype(np.float64) * constant
        matrix += constant_matrix
        np.fill_diagonal(matrix, matrix.diagonal() + constant)

        mat = self._raise_matrix_power(matrix, matrix_power, constant, method)
        return self.priority_vector(
            mat, method=method, log_lik_tol=log_lik_tol, raters=True
        )

    def _estimate_raters_matrix(
        self, constant=0.1, method="log-lik", matrix_power=None, log_lik_tol=0.000001
    ):
        """PAIR facet_element effect estimation — full (facet_element, item, threshold) matrix."""
        data = (
            self.responses.values.reshape(
                self.no_of_facet_elements, self.no_of_persons, -1
            )
            .swapaxes(1, 2)
            .transpose((1, 0, 2))
        )  # (I, R, P)
        facet_elements = np.zeros(
            (self.no_of_facet_elements, self.no_of_items, self.max_score)
        )
        for i in range(self.no_of_items):
            for k in range(self.max_score):
                facet_elements[:, i, k] = self._matrix_rater_element(
                    i,
                    k,
                    data,
                    constant=constant,
                    method=method,
                    matrix_power=matrix_power,
                    log_lik_tol=log_lik_tol,
                )

        # MultiIndex DataFrame: (facet_element, item) × threshold
        mi = pd.MultiIndex.from_product(
            [self.facet_names, self.responses.columns], names=[self.facet, "item"]
        )
        self.facet_effects_matrix = pd.DataFrame(
            facet_elements.reshape(-1, self.max_score), index=mi, columns=range(1, self.max_score + 1)
        )

        # Marginal facet_effects
        sev_arr = facet_elements  # (R, I, K) — no sentinel to skip
        item_marg = sev_arr.mean(axis=2)
        thr_marg_raw = sev_arr.mean(axis=1)
        # Recentre to zero-sum per facet_element, matching
        # _calibrate_anchor_matrix's identical correction and the
        # zero-sum property _estimate_raters_bivector's docstring claims
        # for this vector (item_marg needs no adjustment -- see there).
        thr_marg = thr_marg_raw - thr_marg_raw.mean(axis=1, keepdims=True)

        self.marginal_facet_effects_items = pd.DataFrame(
            item_marg, index=self.facet_names, columns=self.responses.columns
        )
        self.marginal_facet_effects_thresholds = pd.DataFrame(
            thr_marg, index=self.facet_names, columns=range(1, self.max_score + 1)
        )

    def _estimate_raters_bivector(self, matrix_marginals=True, **kw):
        """
        Bivector facet_element effect estimation.

        Uses the full matrix PAIR estimator as an intermediate step. Elliott &
        Buttery (2022a) find evidence that marginal means derived from matrix
        estimates produce more accurate bivector parameter recovery than direct
        estimation of the two vectors in almost all conditions, because the
        matrix estimator captures variability across both items and thresholds
        simultaneously and aggregation of the cell estimates reduces stochastic
        noise.

        The estimated matrix is decomposed into two additive marginal vectors
        per facet_element:

            λ'_rik = λ_ri. + λ_r.k

        where:
          λ_ri. — mean over thresholds of σ_{r,i,k} (item vector, free mean;
                  overall facet_element effect lives here)
          λ_r.k — mean over items of σ_{r,i,k}, zero-summed per facet_element
                  (threshold vector, shape only; Σ_k λ_r.k = 0)

        The reconstructed full matrix λ'_rik is stored as facet_effects_bivector
        in the same {facet_element: {item: array}} format as facet_effects_matrix, so all
        downstream probability, fit, and plot machinery operates on it without
        modification.

        The intermediate matrix estimates are available as facet_effects_matrix
        but should not be interpreted as a matrix model calibration.

        Public attributes set
        ---------------------
        facet_effects_bivector_items : dict
            {facet_element: pd.Series({item: float})} — per-(facet_element, item) marginal means.
        facet_effects_bivector_thresholds : dict
            {facet_element: pd.Series} of length max_score + 1 — per-(facet_element, threshold)
            marginal means, zero-summed per facet_element. Index 0 is always 0.0.
        facet_effects_bivector : dict
            {facet_element: {item: array}} — reconstructed full facet_effect matrix
            (item_effect + threshold_effect per cell). Used by all downstream
            machinery.
        """
        if matrix_marginals:
            # Marginal-means estimator: full matrix PAIR → marginal means per vector
            self._estimate_raters_matrix(**kw)
            self.facet_effects_bivector_items = self.marginal_facet_effects_items
            self.facet_effects_bivector_thresholds = self.marginal_facet_effects_thresholds
        else:
            # Direct pooled-PAIR estimator: each vector estimated from its own
            # pooled comparison matrix (items PAIR summed over thresholds;
            # thresholds PAIR summed over items, corrected for μ_r).
            self._estimate_raters_items(**kw)
            self._estimate_raters_thresholds(**kw)
            mu_r = self.facet_effects_items.mean(axis=1)
            thr = self.facet_effects_thresholds.subtract(mu_r, axis=0)
            thr = thr.subtract(thr.mean(axis=1), axis=0)
            self.facet_effects_bivector_items = self.facet_effects_items
            self.facet_effects_bivector_thresholds = thr

        # Reconstruct full matrix as sum of marginals (λ'_rik = λ_ri. + λ_r.k)
        mi = pd.MultiIndex.from_product(
            [self.facet_names, self.item_names], names=[self.facet, "item"]
        )
        items_arr = self.facet_effects_bivector_items.reindex(
            index=self.facet_names, columns=self.item_names
        ).to_numpy()  # (R, I)
        thresh_arr = self.facet_effects_bivector_thresholds.reindex(
            index=self.facet_names, columns=range(1, self.max_score + 1)
        ).to_numpy()  # (R, K)
        full = items_arr[:, :, None] + thresh_arr[:, None, :]  # (R, I, K)
        self.facet_effects_bivector = pd.DataFrame(
            full.reshape(-1, self.max_score), index=mi, columns=range(1, self.max_score + 1)
        )

    def _estimate_raters_bistretch(self, regression="huber", **kw):
        """
        Bivector-stretch ("bistretch") facet_element effect -- a
        3-parameter (lambda_r shift, omega_items_r item-stretch,
        omega_thresholds_r threshold-stretch) restriction of the
        fully-free 'bivector' model, derived closed-form from bivector's
        own item_effect/threshold_effect marginal vectors.

        Reuses bivector's own existing identification convention
        directly -- no new scheme needed. bivector's item vector already
        carries the facet element's (e.g. rater's) overall level (nonzero mean by construction),
        and its threshold vector is already zero-summed per facet element (pure
        shape, no level). Applies the exact same _derive_stretch_model
        recipe 'centrality'/'pseudo_halo' already use, once per axis:

          item axis:      _derive_stretch_model(facet_effects_bivector_items,
                           self.items) -> (lambda_r, omega_items_r)
          threshold axis:  _derive_stretch_model(facet_effects_bivector_thresholds,
                           self.thresholds) -> (~0, omega_thresholds_r)
                           -- the returned shift comes back ~0 for free since
                           bivector already zero-summed this vector, so only
                           omega_thresholds_r is kept from this call.

        No recalibration happens beyond what _estimate_raters_bivector
        already computes -- this is a pure post-hoc derivation from its
        marginal vectors, same "no refit below" property the rest of the
        stretch family already has. See _derive_stretch_model for the
        regression= argument, applied to both axes.

        omega_items_r < 1 compresses item-difficulty spread (pseudo_halo-
        style); omega_thresholds_r > 1 spreads reference thresholds apart
        (centrality-style). Both == 1 collapses to 'global'.
        """
        self._estimate_raters_bivector(**kw)
        lam, omega_items, item_recon = self._derive_stretch_model(
            self.facet_effects_bivector_items, self.items, regression=regression
        )
        _, omega_thresholds, thresh_recon = self._derive_stretch_model(
            self.facet_effects_bivector_thresholds, self.thresholds, regression=regression
        )
        self.lambda_bistretch = lam
        self.omega_items_bistretch = omega_items
        self.omega_thresholds_bistretch = omega_thresholds
        self.facet_effects_bistretch_items = item_recon
        self.facet_effects_bistretch_thresholds = thresh_recon

        mi = pd.MultiIndex.from_product(
            [self.facet_names, self.item_names], names=[self.facet, "item"]
        )
        items_arr = item_recon.reindex(
            index=self.facet_names, columns=self.item_names
        ).to_numpy()  # (R, I)
        thresh_arr = thresh_recon.reindex(
            index=self.facet_names, columns=range(1, self.max_score + 1)
        ).to_numpy()  # (R, K)
        full = items_arr[:, :, None] + thresh_arr[:, None, :]  # (R, I, K)
        self.facet_effects_bistretch_full_vector = pd.DataFrame(
            full.reshape(-1, self.max_score), index=mi, columns=range(1, self.max_score + 1)
        )
        self.facet_effects_bistretch = pd.DataFrame(
            {"lambda": lam, "omega_items": omega_items, "omega_thresholds": omega_thresholds}
        )

        self._warn_nonpositive_stretch(omega_items, "omega_items", "bistretch")
        self._warn_nonpositive_stretch(omega_thresholds, "omega_thresholds", "bistretch")

    # ------------------------------------------------------------------
    # Calibration — top-level methods
    # ------------------------------------------------------------------

    def _resolve_matrix_marginals(self, model, matrix_marginals, robust):
        """Resolve the bivector/bistretch ``matrix_marginals`` argument.
        See calibrate() for the resolution table."""
        if model == "bistretch":
            if matrix_marginals is False:
                warnings.warn(
                    "matrix_marginals=False for model='bistretch': direct "
                    "marginals bias the stretch parameters toward omega=1; "
                    "matrix marginals are recommended.",
                    UserWarning, stacklevel=3,
                )
                return False
            return True
        # bivector
        if matrix_marginals is None:
            return not robust
        if matrix_marginals and robust:
            warnings.warn(
                "robust=True with an explicit matrix_marginals=True for "
                "model='bivector': the explicit matrix_marginals wins and the "
                "direct-estimation robust route is not used.",
                UserWarning, stacklevel=3,
            )
        return bool(matrix_marginals)

    def _resolve_facet_constant(self, model, robust, mm):
        """Per-model default facet element (e.g. rater) stage smoothing constant,
        used when ``constant`` is left None. See calibrate() for the
        rationale and the simulation studies these values come from."""
        if model in ("items", "thresholds"):
            return 0.3
        if model in ("centrality", "pseudo_halo"):
            return 0.1
        if model == "bistretch":
            return 0.05 if robust else 0.1
        if model == "bivector":
            # matrix-marginal route: 0.3 on the underlying matrix cells;
            # direct route (robust): the direct estimator's own flat wide
            # basin, ~2.0.
            return 0.3 if mm else 2.0
        if model == "matrix":
            m = float(self.responses.isna().to_numpy().mean())
            R = self.no_of_facet_elements
            c = (0.65 if robust else (0.68 - 0.22 * m)) / (R ** 0.5)
            return float(min(0.5, max(0.15, c)))
        return 0.1  # 'global'

    def calibrate(
        self,
        model="global",
        constant=None,
        threshold_constant=None,
        method=None,
        matrix_power=None,
        log_lik_tol=None,
        matrix_marginals=None,
        regression="huber",
        robust=False,
        estimation=None,
        ridge=None,
        interaction_ridge=None,
        stretch_ridge=None,
    ):
        """
        Calibrate the MFRM for the specified facet_element parameterisation.

        estimation='cp-full-rcl' (default): regularised composite likelihood on all
        response-pattern pairs. For two items scored by the same facet
        element, or two facet elements scoring the same item, every pair of
        response patterns (a, b) and (a-1, b+1) with the same total is a binomial comparison in which the person cancels.
        Smoothing constants go on the item and facet element comparisons and
        on CPAT's threshold patterns only; the other patterns are left
        unsmoothed. estimation='cpat-rcl': the same, restricted to CPAT's
        patterns: odd totals give the item comparisons within each facet
        element and the facet element comparisons within each item, even
        totals the threshold patterns ((k, k) against (k-1, k+1) on two
        items, by the same facet element).

        Both RCL methods fit all their patterns jointly, as one
        composite likelihood, by Newton-Raphson: item locations, thresholds
        and the model's facet element parameters together, so each
        sub-model has its own item locations and thresholds (items_{model},
        thresholds_{model}). The facet element parameters carry ridge
        penalties (ridge, interaction_ridge, stretch_ridge), and the
        pattern counts additive smoothing constants (see constant). Thin
        category widths and widths next to an empty middle category are
        handled as in RSM.calibrate(). The stretch models are fitted from
        two starts, from a 'pair-cpat' fit: its stretch parameters, and
        omega = 1; the fit with the higher composite likelihood is kept.

        estimation='pair-cpat': the method used before v1.4.0 (Elliott &
        Buttery, 2022), three stages in one pass:
          1. item_diffs()       — PAIR item item_locations (shared across models)
          2. ra_thresholds()    — CPAT shared thresholds (shared across models)
          3. raters_{model}()   — PAIR facet_element effects (model-specific)

        Parameters
        ----------
        model : one of 'global', 'items', 'thresholds', 'matrix', 'bivector',
            'centrality', 'pseudo_halo', 'bistretch'
        constant : float or None, default None
            Additive smoothing constant. ``None`` (default) uses the
            simulation-tuned per-stage defaults (see ``robust`` and the note
            below). An explicit float overrides every stage -- item
            locations, thresholds and the facet element (e.g. rater) stage all use
            that value -- and takes precedence over ``robust`` and
            ``threshold_constant``. Pass ``constant=0.1`` to reproduce
            pre-1.3.0 output.
        threshold_constant : float or None, default None
            Overrides only the ``ra_thresholds`` (CPAT threshold-location)
            stage. ``None`` follows ``constant``: the tuned default (0.3)
            when ``constant`` is ``None``, else the explicit ``constant``.
        method : {'cos', 'log-lik', 'ls', 'evm'} or None, default None
            Priority-vector method. ``None`` resolves by estimand density:
            ``'log-lik'`` for the low-parameter representations -- 'global'
            (one location per facet element from a small dense
            facet-element x facet-element matrix) and the three stretch
            models ('centrality', 'pseudo_halo', 'bistretch', whose parent
            vector is re-pooled through a slope regression) -- and ``'cos'``
            for 'items', 'thresholds', 'matrix' and 'bivector', where the
            estimand is a thin per-item / per-threshold / per-cell /
            per-marginal slice with no downstream re-pooling and 'cos'
            degrades more gracefully on the sparser sub-matrices. An
            explicit value is used as-is for every stage.
        matrix_marginals : bool or None, default None
            Bivector/bistretch only. ``None`` resolves to ``not robust`` for
            'bivector' (matrix-marginal means by default; the direct
            pooled-PAIR estimator under ``robust=True``) and to ``True`` for
            'bistretch' (always matrix-marginal -- direct marginals bias its
            stretch parameters toward omega=1). Elliott & Buttery (2022a)
            find matrix-marginal means recover the true parameters more
            accurately than direct estimation in almost all conditions. An
            explicit ``True`` under ``robust=True`` wins, with a UserWarning;
            an explicit ``False`` for 'bistretch' warns.
        regression : {'huber', 'theil-sen'}, default 'huber'
            centrality/pseudo_halo/bistretch only. Robust regression method
            used to recover each facet_element's raw lambda_r/omega_r (or
            omega_items_r/omega_thresholds_r for bistretch) from its free
            deviation values. See _derive_stretch_model for the full
            rationale and the automatic huber->theil-sen fallback behaviour.
            Ignored for every other model.
        robust : bool, default False
            Selects the more conservative / tail-protective calibration for
            rough data, resolved per model. It changes:
            'matrix' -- the facet-element-cell constant switches from the
            missing-sensitive default ``clip((0.68 - 0.22 m)/sqrt(R), 0.15,
            0.5)`` to the R-only ``clip(0.65/sqrt(R), 0.15, 0.5)``, where
            ``m`` is the observed missing-data proportion and ``R`` the
            number of facet elements;
            'bivector' -- switches to the direct pooled-PAIR estimator and a
            larger facet-element-stage constant (~2.0);
            'bistretch' -- the facet element/stretch-stage constant drops from 0.1 to
            0.05.
            For every other model ``robust=True`` currently has no effect
            (accepted for forward compatibility). Ignored, with a
            UserWarning, if an explicit ``constant`` is also given.

        Attributes set
        --------------
        constant : float
            The item-location (item_diffs) smoothing constant used.
        threshold_constant : float
            The ra_thresholds (threshold-location) smoothing constant used.
        facet_constant : float
            The facet element stage smoothing constant used, after
            resolving ``None`` / ``robust`` / the per-model rule.

        Notes
        -----
        Default additive smoothing constants and ridge penalties were
        determined through simulation studies.

        estimation='cp-full-rcl' / 'cpat-rcl' arguments
        ---------------------------------------
        estimation : {'cp-full-rcl', 'cpat-rcl', 'pair-cpat'} or None, default None
            None resolves to the method of the object's last calibration,
            or 'cp-full-rcl' for the first one. method, matrix_power,
            log_lik_tol, matrix_marginals and robust apply to 'pair-cpat'
            only; other values passed with 'cp-full-rcl' or 'cpat-rcl' are ignored,
            with a UserWarning. regression sets the stretch-model start from the
            'pair-cpat' fit.
        constant : with 'cp-full-rcl' or 'cpat-rcl', None uses the data-adaptive item rule
            ``min(1, 0.548 * I ** (-0.2 * log10(n * K)))`` per item pair (n
            the number of (facet element, person) rows with a response),
            spread over the R * K (facet element, boundary) cells, and 0.1
            per facet element parameter for the facet element comparisons,
            spread over the cells informing each parameter. An explicit
            value replaces both.
        threshold_constant : with 'cp-full-rcl' or 'cpat-rcl', per CPAT threshold-pattern cell,
            divided by R; None gives 0.03 (or constant, if given).
        ridge : float or None, default None
            'global', 'items', 'thresholds', 'bivector' and 'matrix' (its
            bivector parts): ridge * sum(deviation^2) on the facet element
            deviations, in logits. None gives 0.3.
        interaction_ridge : float or None, default None
            'matrix' only: penalty on the item x threshold interactions
            beyond the bivector parts. None gives 3.
        stretch_ridge : float or None, default None
            'centrality', 'pseudo_halo', 'bistretch': penalty on
            (omega - 1)^2. None gives 1.
        A regularisation argument passed to a sub-model it does not apply
        to (or with 'pair-cpat') is ignored, with a UserWarning.
        """
        model = self._MODEL_ALIASES.get(model, model)
        if model not in self._MODELS:
            raise ValueError(f"model must be one of {self._MODELS}")
        if estimation is None:
            estimation = getattr(self, "estimation", "cp-full-rcl")
        if estimation not in ("cp-full-rcl", "cpat-rcl", "pair-cpat"):
            raise ValueError(
                f"estimation must be 'cp-full-rcl', 'cpat-rcl', 'pair-cpat' or None, got {estimation!r}."
            )
        self.estimation = estimation
        _applies = {
            "ridge": ("global", "items", "thresholds", "bivector", "matrix"),
            "interaction_ridge": ("matrix",),
            "stretch_ridge": ("centrality", "pseudo_halo", "bistretch"),
        }
        _reg = dict(ridge=ridge, interaction_ridge=interaction_ridge, stretch_ridge=stretch_ridge)
        ignored = [
            f"{name}={value!r}" for name, value in _reg.items()
            if value is not None and (estimation == "pair-cpat" or model not in _applies[name])
        ]
        if estimation in ("cp-full-rcl", "cpat-rcl"):
            ignored += [
                f"{name}={value!r}" for name, value, default in (
                    ("method", method, None), ("matrix_power", matrix_power, None),
                    ("log_lik_tol", log_lik_tol, 0.000001), ("matrix_marginals", matrix_marginals, None),
                    ("robust", robust, False),
                ) if value is not None and value != default
            ]
        if ignored:
            where = (
                "with estimation='pair-cpat'" if estimation == "pair-cpat"
                else f"with estimation='{estimation}' and model='{model}'"
            )
            warnings.warn(
                f"{', '.join(ignored)} {'does' if len(ignored) == 1 else 'do'} not apply {where} "
                f"and {'is' if len(ignored) == 1 else 'are'} ignored.",
                UserWarning,
                stacklevel=2,
            )
        if log_lik_tol is None:
            log_lik_tol = 0.000001

        _STRETCH = ("centrality", "pseudo_halo", "bistretch")
        # A stretch model with no restriction degrees of freedom per facet
        # element (a line through 2 points) is just its parent model.
        _restriction_df = {
            "centrality": (self.max_score - 2, "thresholds"),
            "pseudo_halo": (self.no_of_items - 2, "items"),
            "bistretch": (self.no_of_items + self.max_score - 4, "bivector"),
        }
        if model in _restriction_df and _restriction_df[model][0] <= 0:
            parent = _restriction_df[model][1]
            warnings.warn(
                f"With {self.no_of_items} items and max_score={self.max_score}, "
                f"model='{model}' imposes no restriction on model='{parent}': it "
                f"has the same number of free parameters per {self.facet_label_eg} and "
                f"fits the data identically. Use model='{parent}' instead.",
                UserWarning,
                stacklevel=2,
            )
        # Sub-models with rater-specific thresholds: a facet element that used a
        # category twice or less leaves the thresholds next to it poorly
        # determined for that facet element. Not raised for bootstrap
        # resamples: resampling persons often thins a category the data itself
        # uses enough, and any real shortfall is reported for the full data.
        if model in ("thresholds", "bivector", "matrix", "centrality", "bistretch") and not self._resample:
            sparse = {}
            for fe in self.facet_names:
                v = self.responses.loc[fe].to_numpy(dtype=float)
                cnt = np.bincount(v[np.isfinite(v)].astype(int), minlength=self.max_score + 1)
                low = {k: int(c) for k, c in enumerate(cnt) if c <= 2}
                if low:
                    sparse[fe] = low
            if sparse:
                detail = "; ".join(
                    f"{fe} (" + ", ".join(f"category {k}: {c}" for k, c in low.items()) + ")"
                    for fe, low in sparse.items()
                )
                warnings.warn(
                    f"{self._facet_n(len(sparse))} used some score "
                    f"categories twice or less: {detail}. Under model='{model}', "
                    f"each {self.facet_label}'s location is estimated separately "
                    f"at each threshold, so these {self.facets_label}' locations "
                    f"at the thresholds bounding those categories are poorly "
                    f"determined. See category_counts_df() for the full breakdown.",
                    UserWarning,
                    stacklevel=2,
                )
        # Low-parameter representations (one severity per facet element, or a
        # stretch pair re-pooled through a slope regression) default to
        # 'log-lik'; the per-slice representations to 'cos'. See calibrate's
        # `method` docstring.
        _LOGLIK_DEFAULT = ("global",) + _STRETCH

        if method is None:
            method = "log-lik" if model in _LOGLIK_DEFAULT else "cos"

        if constant is not None and robust and estimation == "pair-cpat":
            warnings.warn(
                f"Both `constant={constant}` and `robust=True` were passed. An "
                f"explicit `constant` takes precedence and is used for every "
                f"stage; the robust rule is not applied.",
                UserWarning,
                stacklevel=2,
            )

        # item-location (item_diffs) stage
        item_constant = 0.1 if constant is None else constant

        # threshold-location (ra_thresholds) stage
        if threshold_constant is not None:
            thr_constant = threshold_constant
        elif constant is not None:
            thr_constant = constant
        else:
            thr_constant = 0.3

        # matrix_marginals resolution (bivector / bistretch only)
        mm = None
        if model in ("bivector", "bistretch"):
            mm = self._resolve_matrix_marginals(model, matrix_marginals, robust)

        # facet_element (rater) stage constant
        if constant is not None:
            facet_constant = constant
        else:
            facet_constant = self._resolve_facet_constant(model, robust, mm)

        if item_constant == 0:
            # Unsmoothed PAIR: re-run the structural connectivity check on the
            # data as it stands now (not just at __init__(validate=True)) --
            # constant=0 removes the only thing masking a directionally
            # isolated item or disconnected component, both of which
            # check_data_connectivity() already detects and warns about
            # (does not raise; 'log-lik' tolerates an incomplete comparison
            # graph and still returns an answer for the rest of the items).
            # This covers the shared item-location PAIR stage only -- the
            # facet_element (rater) PAIR stage below has its own analogous
            # check, since raters have no counterpart in check_data_connectivity().
            self.check_data_connectivity()

        if len(self.facet_names) == 1:
            warnings.warn(
                f"Only one {self.facet_label_eg} detected. MFRM with a single {self.facet_label} reduces to RSM. "
                "Consider using RSM instead.",
                UserWarning,
                stacklevel=2,
            )

        if len(self.item_names) == 1:
            warnings.warn(
                "Only one item detected. MFRM with a single item reduces to RSM "
                f"with {self.facets_label_eg} as items. Consider reconfiguring and using RSM instead.",
                UserWarning,
                stacklevel=2,
            )

        self._remove_null_persons()
        self._warn_facet_groups(stacklevel=3)
        if estimation in ("cp-full-rcl", "cpat-rcl"):
            self._calibrate_rcl(model, constant, threshold_constant, regression, full_patterns=estimation == "cp-full-rcl", **_reg)
            self._set_facet_aliases(model)
            return

        self._remove_null_persons()
        self.item_diffs(
            constant=item_constant,
            method=method,
            matrix_power=matrix_power,
            log_lik_tol=log_lik_tol,
        )
        self.thresholds = pd.Series(
            self.ra_thresholds(self.items, constant=thr_constant),
            index=range(1, self.max_score + 1),
        )
        kw = dict(constant=facet_constant, method=method,
                  matrix_power=matrix_power, log_lik_tol=log_lik_tol)
        if model in ("bivector", "bistretch"):
            kw["matrix_marginals"] = mm
        if model in _STRETCH:
            kw["regression"] = regression
        self._isolated_log = {}
        try:
            getattr(self, f"_estimate_raters_{model}")(**kw)
        finally:
            self._emit_isolated_raters(stacklevel=3)
        self.constant = item_constant
        self.threshold_constant = thr_constant
        self.facet_constant = facet_constant
        self.convergence = None
        setattr(self, f"estimation_{model}", "pair-cpat")
        setattr(self, f"items_{model}", self.items)
        setattr(self, f"thresholds_{model}", self.thresholds)
        self._set_facet_aliases(model)

    # parameters informing each facet element parameter, for spreading the
    # facet element constant over cells: per-cell constant = c * factor
    def _facet_cell_factor(self, model):
        I, K = self.no_of_items, self.max_score
        return {
            "global": 1 / (I * K), "items": 1 / K, "thresholds": 1 / I,
            "bivector": (I + K) / (I * K), "matrix": 1.0,
            # stretch models: their parent cell model's factor
            "centrality": 1 / I, "pseudo_halo": 1 / K, "bistretch": (I + K) / (I * K),
        }[model]

    def _refit_kw(self, model):
        """estimation= and the regularisation arguments of the latest
        calibration, for refits on new objects (bootstrap resamples, group
        calibrations), keeping only those that apply to model."""
        kw = dict(estimation=getattr(self, "estimation", None))
        applies = dict(ridge=("global", "items", "thresholds", "bivector", "matrix"),
                       interaction_ridge=("matrix",),
                       stretch_ridge=("centrality", "pseudo_halo", "bistretch"))
        if kw["estimation"] in ("cp-full-rcl", "cpat-rcl"):
            for name, models in applies.items():
                value = getattr(self, name, None)
                if value is not None and model in models:
                    kw[name] = value
        return kw

    def _warn_facet_groups(self, stacklevel=3):
        """Warn if the facet elements fall into groups never compared with
        each other (no person scored one point apart on the same item by
        facet elements from different groups): how the groups' locations
        differ is then not identified by the data."""
        from raschpy import _rcl
        R, I, K = self.no_of_facet_elements, self.no_of_items, self.max_score
        if R < 2:
            return
        X = self.responses.to_numpy(dtype=np.float64).reshape(R, -1, I)
        A = np.zeros((R, R))
        for c in range(K):
            hi = np.nan_to_num(X == c + 1).astype(np.float64).reshape(R, -1)
            lo = np.nan_to_num(X == c).astype(np.float64).reshape(R, -1)
            A += hi @ lo.T
        comps = _rcl.facet_components((A + A.T)[None, None])
        if len(comps) > 1:
            groups = "; ".join(str([self.facet_names[r] for r in c]) for c in comps)
            warnings.warn(
                f"The {self.facets_label_eg} fall into {len(comps)} groups with no comparisons "
                f"between them (no person was scored one point apart on the same item by "
                f"{self.facets_label} from different groups): {groups}. How the groups' "
                f"locations differ is not identified by the data, so comparisons between "
                f"{self.facets_label} in different groups are not meaningful. Link the groups "
                f"(some persons rated by {self.facets_label} from more than one group) or "
                f"analyse them separately.",
                UserWarning,
                stacklevel=stacklevel,
            )

    def _shipped_widths(self, item_locations, threshold_constant):
        """Category widths from the RSM cpat-cml threshold rule (with its
        empty-category handling), pooling every facet element's responses,
        given item_locations."""
        from raschpy.rsm import RSM
        stacked = self.responses.copy()
        stacked.index = [f"{r}|{p}" for r, p in stacked.index]
        pooled = RSM(stacked, max_score=self.max_score, validate=False)
        thr = pooled.threshold_set(
            item_locations.reindex(stacked.columns), constant=threshold_constant,
            threshold_method="cpat-cml",
        )
        return np.diff(np.asarray(thr, dtype=float))

    def _calibrate_rcl(self, model, constant, threshold_constant, regression,
                       ridge=None, interaction_ridge=None, stretch_ridge=None, full_patterns=True):
        """RCL calibration of one sub-model -- full pattern set (estimation='cp-full-rcl') or CPAT's patterns
        ('cpat-rcl'); see calibrate()."""
        import copy
        from raschpy import _rcl

        R, I, K = self.no_of_facet_elements, self.no_of_items, self.max_score
        ridge = 0.3 if ridge is None else float(ridge)
        interaction_ridge = 3.0 if interaction_ridge is None else float(interaction_ridge)
        stretch_ridge = 1.0 if stretch_ridge is None else float(stretch_ridge)
        if constant is None:
            rows = int(self.responses.notna().any(axis=1).sum())
            item_constant = min(1.0, 0.548 * I ** (-0.2 * np.log10(max(rows * K, 1))))
            facet_constant = 0.1
        else:
            item_constant = facet_constant = float(constant)
        if threshold_constant is not None:
            thr_constant = float(threshold_constant)
        else:
            thr_constant = 0.03 if constant is None else float(constant)
        c_item = item_constant / (R * K)
        c_thr = thr_constant / R
        c_facet = facet_constant * self._facet_cell_factor(model)

        X = self.responses.to_numpy(dtype=np.float64).reshape(R, -1, I)
        counts = _rcl.pattern_counts(X, K)
        cat_counts = _rcl.category_counts(X, K)
        split = _rcl.empty_widths(cat_counts, K)
        full_arg = dict(X=X, empty=_rcl.empty_categories(cat_counts)) if full_patterns else None
        rho = _rcl.width_ridges(counts[1], counts[2], K, exclude=split)
        totals = (counts[1] + counts[2]).sum(axis=(0, 2, 3))
        self._warn_no_information(
            [k + 1 for k in range(K - 1) if totals[k] == 0 and k not in split], stacklevel=5)
        items_idx = self.responses.columns
        facets = self.facet_names

        if model in ("centrality", "pseudo_halo", "bistretch"):
            # starts from a 'pair-cpat' fit, run on a copy so none of its
            # attributes are left behind
            start = copy.copy(self)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                start.calibrate(model=model, estimation="pair-cpat", regression=regression)
            d0 = start.items.reindex(items_idx).to_numpy(dtype=float)
            t0 = np.asarray(start.thresholds, dtype=float)
            lam0 = getattr(start, f"lambda_{model}").reindex(facets).to_numpy(dtype=float)
            if model == "centrality":
                wI0 = np.ones(R)
                wT0 = start.omega_centrality.reindex(facets).to_numpy(dtype=float)
            elif model == "pseudo_halo":
                wI0 = start.omega_pseudo_halo.reindex(facets).to_numpy(dtype=float)
                wT0 = np.ones(R)
            else:
                wI0 = start.omega_items_bistretch.reindex(facets).to_numpy(dtype=float)
                wT0 = start.omega_thresholds_bistretch.reindex(facets).to_numpy(dtype=float)
            wI0, wT0 = np.nan_to_num(wI0, nan=1.0), np.nan_to_num(wT0, nan=1.0)
            sm = _rcl.StretchModel(model, counts, R, I, K, c_item, c_thr, c_facet, stretch_ridge,
                                   width_rho=rho, drop_widths=split, full=full_arg)

            def fit(fixed_widths):
                fixed = {sm.tau_cols.start + k: v for k, v in fixed_widths.items()}
                best = None
                for st in (sm.theta_from(d0, t0, lam0, wI0, wT0),
                           sm.theta_from(d0, t0, lam0, np.ones(R), np.ones(R))):
                    th, info = sm.fit(st, fixed=fixed)
                    if best is None or info["objective"] > best[1]["objective"]:
                        best = (th, info)
                return best

            fixed_widths = {}
            if split:
                w = self._shipped_widths(pd.Series(d0, index=items_idx), thr_constant)
                fixed_widths = {k: w[k] for k in split}
            theta, info = fit(fixed_widths)
            if split:
                # re-estimate the split-out widths given the joint fit's items, and refit with them
                d, *_ = sm.unpack(theta)
                w = self._shipped_widths(pd.Series(d, index=items_idx), thr_constant)
                fixed_widths = {k: w[k] for k in split}
                theta, info = fit(fixed_widths)
            self._rcl_fits = getattr(self, "_rcl_fits", {})
            self._rcl_fits[model] = dict(
                kind="stretch", sm=sm, theta=theta, split=split,
                fixed=[sm.tau_cols.start + k for k in fixed_widths])
            d, tau, lam, wI, wT = sm.unpack(theta)
            lam_s = pd.Series(lam, index=facets)
            item_dev = pd.DataFrame(lam[:, None] + (wI[:, None] - 1) * d[None, :], index=facets, columns=items_idx)
            thr_dev = pd.DataFrame((wT[:, None] - 1) * tau[None, :], index=facets, columns=range(1, K + 1))
            if model == "centrality":
                omega = pd.Series(wT, index=facets)
                self.lambda_centrality, self.omega_centrality = lam_s, omega
                self.facet_effects_centrality_full_vector = thr_dev.add(lam_s, axis=0)
                self.facet_effects_centrality = pd.DataFrame({"lambda": lam_s, "omega": omega})
                self._warn_nonpositive_stretch(omega, "omega", "centrality")
            elif model == "pseudo_halo":
                omega = pd.Series(wI, index=facets)
                self.lambda_pseudo_halo, self.omega_pseudo_halo = lam_s, omega
                self.facet_effects_pseudo_halo_full_vector = item_dev
                self.facet_effects_pseudo_halo = pd.DataFrame({"lambda": lam_s, "omega": omega})
                self._warn_nonpositive_stretch(omega, "omega", "pseudo_halo")
            else:
                om_i, om_t = pd.Series(wI, index=facets), pd.Series(wT, index=facets)
                self.lambda_bistretch = lam_s
                self.omega_items_bistretch, self.omega_thresholds_bistretch = om_i, om_t
                self.facet_effects_bistretch_items, self.facet_effects_bistretch_thresholds = item_dev, thr_dev
                full = item_dev.to_numpy()[:, :, None] + thr_dev.to_numpy()[:, None, :]
                mi = pd.MultiIndex.from_product([facets, self.item_names], names=[self.facet, "item"])
                self.facet_effects_bistretch_full_vector = pd.DataFrame(
                    full.reshape(-1, K), index=mi, columns=range(1, K + 1))
                self.facet_effects_bistretch = pd.DataFrame(
                    {"lambda": lam_s, "omega_items": om_i, "omega_thresholds": om_t})
                self._warn_nonpositive_stretch(om_i, "omega_items", "bistretch")
                self._warn_nonpositive_stretch(om_t, "omega_thresholds", "bistretch")
            widths = np.diff(tau)
        else:
            cm, theta, info = _rcl.fit_cell(
                model, counts, R, I, K, c_item, c_thr, c_facet, ridge=ridge,
                interaction_ridge=interaction_ridge, width_rho=rho,
                fixed_widths={k: 0.0 for k in split}, full=full_arg,
            )
            self._rcl_fits = getattr(self, "_rcl_fits", {})
            self._rcl_fits[model] = dict(kind="cell", cm=cm, theta=theta, info=info, split=split)
            parts = cm.parts(theta)
            d, s = parts["delta"], parts["s"]
            widths = np.diff(parts["tau"])
            if split:
                w = self._shipped_widths(pd.Series(d, index=items_idx), thr_constant)
                widths[split] = w[split]
            mi = pd.MultiIndex.from_product([facets, self.item_names], names=[self.facet, "item"])
            if model == "global":
                self.facet_effects_global = pd.Series(s[:, 0, 0], index=facets)
            elif model == "items":
                self.facet_effects_items = pd.DataFrame(s[:, :, 0], index=facets, columns=items_idx)
            elif model == "thresholds":
                self.facet_effects_thresholds = pd.DataFrame(s[:, 0, :], index=facets, columns=range(1, K + 1))
            elif model == "matrix":
                self.facet_effects_matrix = pd.DataFrame(s.reshape(-1, K), index=mi, columns=range(1, K + 1))
                thr_marg = s.mean(axis=1)
                self.marginal_facet_effects_items = pd.DataFrame(s.mean(axis=2), index=facets, columns=items_idx)
                self.marginal_facet_effects_thresholds = pd.DataFrame(
                    thr_marg - thr_marg.mean(axis=1, keepdims=True), index=facets, columns=range(1, K + 1))
            else:  # bivector
                # (no facet element parameters when there is a single facet element)
                a = parts["a"][:, :, 0] if "a" in parts else np.zeros((R, I))
                t2 = parts["t2"][:, 0, :] if "t2" in parts else np.zeros((R, K))
                self.facet_effects_bivector_items = pd.DataFrame(a, index=facets, columns=items_idx)
                self.facet_effects_bivector_thresholds = pd.DataFrame(t2, index=facets, columns=range(1, K + 1))
                self.facet_effects_bivector = pd.DataFrame(s.reshape(-1, K), index=mi, columns=range(1, K + 1))

        tau = np.concatenate([[0.0], np.cumsum(widths)])
        self.items = pd.Series(d - d.mean(), index=items_idx)
        self.thresholds = pd.Series(tau - tau.mean(), index=range(1, K + 1))
        setattr(self, f"items_{model}", self.items)
        setattr(self, f"thresholds_{model}", self.thresholds)
        self.constant = item_constant
        self.threshold_constant = thr_constant
        self.facet_constant = facet_constant
        self.ridge = ridge if model in ("global", "items", "thresholds", "bivector", "matrix") else None
        self.interaction_ridge = interaction_ridge if model == "matrix" else None
        self.stretch_ridge = stretch_ridge if model in ("centrality", "pseudo_halo", "bistretch") else None
        self.convergence = dict(iterations=info["iterations"], converged=info["converged"])
        setattr(self, f"convergence_{model}", self.convergence)
        setattr(self, f"estimation_{model}", "cp-full-rcl" if full_patterns else "cpat-rcl")
        if not info["converged"]:
            warnings.warn(
                f"RCL-CPAT calibration of model='{model}' did not converge in "
                f"{info['iterations']} iterations; the estimates may be unreliable.",
                UserWarning,
                stacklevel=3,
            )

    # Backwards-compatible aliases
    def calibrate_global(self, **kw):
        """Alias for calibrate(model='global'). See calibrate for full documentation."""
        self.calibrate(model="global", **kw)

    def calibrate_items(self, **kw):
        """Alias for calibrate(model='items'). See calibrate for full documentation."""
        self.calibrate(model="items", **kw)

    def calibrate_thresholds(self, **kw):
        """Alias for calibrate(model='thresholds'). See calibrate for full documentation."""
        self.calibrate(model="thresholds", **kw)

    def calibrate_matrix(self, **kw):
        """Alias for calibrate(model='matrix'). See calibrate for full documentation."""
        self.calibrate(model="matrix", **kw)

    def calibrate_bivector(self, **kw):
        """Alias for calibrate(model='bivector'). See calibrate for full documentation."""
        self.calibrate(model="bivector", **kw)

    def calibrate_centrality(self, **kw):
        """Alias for calibrate(model='centrality'). See calibrate for full documentation."""
        self.calibrate(model="centrality", **kw)

    def calibrate_pseudo_halo(self, **kw):
        """Alias for calibrate(model='pseudo_halo'). See calibrate for full documentation."""
        self.calibrate(model="pseudo_halo", **kw)

    def calibrate_bistretch(self, **kw):
        """Alias for calibrate(model='bistretch'). See calibrate for full documentation."""
        self.calibrate(model="bistretch", **kw)

    # ------------------------------------------------------------------
    # Anchor calibration
    # ------------------------------------------------------------------

    def calibrate_anchor(
        self,
        model,
        anchors,
        calibrate=False,
        constant=None,
        threshold_constant=None,
        method=None,
        robust=False,
        matrix_power=None,
        log_lik_tol=0.000001,
        adj=None,
        regression="huber",
        estimation=None,
        matrix_marginals=None,
        ridge=None,
        interaction_ridge=None,
        stretch_ridge=None,
    ):
        """
        Anchor calibration: set mean facet_effect of anchors to zero
        and adjust item item_locations and thresholds accordingly.

        calibrate : if True, runs calibrate() first with constant,
            threshold_constant, method, robust, matrix_power, log_lik_tol,
            regression, estimation, matrix_marginals, ridge,
            interaction_ridge and stretch_ridge (all as for calibrate()).

        adj : pre-computed anchor adjustment from _extract_anchor_adj().
            If provided, used as a fixed constant instead of re-estimating
            from self. Pass this in bootstrap loops to avoid inflating SEs
            with anchor facet element (e.g. rater) sampling variance.
        regression : {'huber', 'theil-sen'}, default 'huber'
            centrality/pseudo_halo/bistretch only -- see calibrate(). Ignored
            for every other model.
        """
        model = self._MODEL_ALIASES.get(model, model)
        if calibrate:
            self.calibrate(
                model=model,
                constant=constant,
                threshold_constant=threshold_constant,
                method=method,
                robust=robust,
                matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
                estimation=estimation,
                matrix_marginals=matrix_marginals,
                ridge=ridge,
                interaction_ridge=interaction_ridge,
                stretch_ridge=stretch_ridge,
                **({"regression": regression} if model in ("centrality", "pseudo_halo", "bistretch") else {}),
            )

        if model == "global":
            self._calibrate_anchor_global(anchors, adj=adj)
        elif model == "items":
            self._calibrate_anchor_items(anchors, adj=adj)
        elif model == "thresholds":
            self._calibrate_anchor_thresholds(anchors, adj=adj)
        elif model == "bivector":
            self._calibrate_anchor_bivector(anchors, adj=adj)
        elif model == "matrix":
            self._calibrate_anchor_matrix(anchors, adj=adj)
        elif model == "centrality":
            self._calibrate_anchor_centrality(anchors, adj=adj, regression=regression)
        elif model == "pseudo_halo":
            self._calibrate_anchor_pseudo_halo(anchors, adj=adj, regression=regression)
        elif model == "bistretch":
            self._calibrate_anchor_bistretch(anchors, adj=adj, regression=regression)

        setattr(self, f"anchor_rater_names_{model}", anchors)
        self._set_facet_aliases(model, anchor=True)

    def check_anchor_homogeneity(
        self,
        model,
        anchors,
        rater_homogeneity_test="cochran",
        per_component="auto",
        alpha=0.05,
        correction="bh",
        robust_z_threshold=3.5,
        no_of_samples=500,
        constant=None,
        threshold_constant=None,
        method=None,
        robust=False,
        matrix_power=None,
        log_lik_tol=0.000001,
        seed=None,
    ):
        """
        Test whether the facet elements (e.g. raters) proposed as `anchors` actually agree with
        each other, before calibrate_anchor() recentres their group mean
        to zero.

        calibrate_anchor() only ever imposes mean(anchor facet_effects) == 0 —
        it has no way to notice whether that mean is a stable, shared
        reference point or an artefact of averaging over facet elements who don't
        behave alike. If one or more anchor facet elements have a genuinely
        different facet_effect from the rest, the anchor-implied "zero" is
        unstable and the whole anchored scale inherits that instability.
        This is unrelated to per_rater_model_selection (which asks how
        COMPLEX a facet element's own facet_effect structure needs to be) — this asks
        whether the specific set of facet elements chosen to DEFINE the anchor
        actually agree with each other. A separate, opt-in diagnostic
        (not run automatically by calibrate_anchor) since the 'cochran'
        test needs bootstrap SEs, which aren't otherwise computed by
        default and would add unexpected cost to every calibrate_anchor
        call.

        Uses each facet element's own free (unanchored) facet_effects_{model} —
        deliberately not anchor_facet_effects_{model} — since the point is
        to check whether this candidate anchor SET is internally
        consistent before it gets used to define anything, using an
        unanchored bootstrap so the test isn't circularly biased toward
        finding homogeneity by construction.

        Parameters
        ----------
        model : str
            One of 'global', 'items', 'thresholds', 'matrix', 'bivector',
            'centrality', 'pseudo_halo', 'bistretch'.
        anchors : list
            Facet element identifiers to test — the same set you'd pass to
            calibrate_anchor(model, anchors). At least 2 required.
        rater_homogeneity_test : {'cochran', 'robust'}, default 'cochran'
            'cochran': Cochran's Q heterogeneity test (Cochran, 1954,
            "The combination of estimates from different experiments,"
            Biometrics 10:101-129 — the meta-analysis heterogeneity Q, not
            Cochran's 1950 test for k related binary samples).
            Q = sum(w_r*(facet_effect_r - weighted_mean)^2),
            weights w_r=1/SE_r^2 from bootstrap SEs, chi-squared with
            df=n_anchors-1, evaluated on the anchor set exactly as
            supplied. Which specific facet element(s) are responsible is then
            identified by sequential exclusion (standard meta-analysis
            outlier diagnostic): repeatedly drop the facet element with the
            largest |z| against the CURRENT remaining group's own
            weighted mean and retest, until Q is no longer significant. A
            single-pass z-test against the full group's mean isn't used
            for per-facet-element flagging because one severely deviant facet element
            drags that mean far enough that every OTHER facet element also looks
            significant relative to it — sequential exclusion re-centres
            on each remaining subgroup instead, so it isolates the true
            offender(s) rather than flagging the whole set.
            'robust': Iglewicz & Hoaglin modified z-score against the
            anchor group's own median/MAD (same convention as
            _robust_anchor_selection, applied within the group rather
            than against an external reference). Purely descriptive
            outlier flagging (threshold=robust_z_threshold) — there's no
            standard significance test/p-value for this statistic, so no
            omnibus result is produced.
        per_component : 'auto', True, or False, default 'auto'
            For every model except 'global' (facet_effect is a vector or
            surface, not one number per facet element), also test homogeneity
            component-by-component (per item, per threshold, or per
            (item, threshold) cell for matrix/bivector/bistretch -- for
            centrality/pseudo_halo/bistretch, components are cells of the
            reconstructed full vector, not lambda/omega) in addition
            to the scalar (whole-profile-mean) test. 'auto' turns this on
            for every model except 'global' (nothing to break out there).
            Can produce many rows for matrix/bivector/bistretch
            (no_of_items * max_score).
        alpha : float, default 0.05
            Significance level for all Cochran's Q tests here: the initial
            omnibus test, each step of the sequential-exclusion procedure,
            and the per-component tests (rater_homogeneity_test='cochran'
            only).
        correction : {'bh', 'bonferroni', None}, default 'bh'
            Multiple-comparison correction across the per-component tests
            (if run) — genuinely independent tests, one per component, so
            correcting across them is appropriate. Not used for per-facet-element
            flagging: sequential exclusion (above) already decides that
            via repeated omnibus tests, not a family of independent
            per-facet-element p-values, so a correction doesn't apply there.
            rater_homogeneity_test='cochran' only.
        robust_z_threshold : float, default 3.5
            Flagging threshold for rater_homogeneity_test='robust'
            (Iglewicz & Hoaglin's own recommended default).
        no_of_samples : int, default 500
            Bootstrap samples, only used if unanchored SEs for this model
            aren't already stored (rater_homogeneity_test='cochran' only).
        constant, method, matrix_power, log_lik_tol : floats
            Calibration/bootstrap kwargs, used only if calibration or SEs
            for this model aren't already computed.
        seed : int or None, default None
            Seed passed through to the internal std_errors() call (only
            used if unanchored SEs aren't already computed). None draws
            fresh entropy each call.

        Attributes set
        --------------
        anchor_homogeneity_test : pandas.Series or None
            Q, df, p, Flagged, N_dropped_for_homogeneity for the scalar
            (whole-profile) omnibus test, evaluated on the anchor set as
            supplied (N_dropped_for_homogeneity is how many facet elements the
            sequential-exclusion procedure needed to remove before Q
            stopped being significant). None if
            rater_homogeneity_test='robust' (no omnibus statistic exists
            for that method).
        anchor_homogeneity_per_rater : pandas.DataFrame
            One row per anchor facet element. rater_homogeneity_test='cochran':
            Facet effect, SE, z, p, Flagged — Flagged is whether
            sequential exclusion removed this facet element (z/p are relative to
            whichever group they were tested against: their own removal
            step if dropped, or the final retained group if not).
            rater_homogeneity_test='robust': Facet effect, Robust z, Flagged.
        anchor_homogeneity_per_component : pandas.DataFrame or None
            One row per component (item name / threshold number / (item,
            threshold) tuple) with that component's own omnibus result —
            Q, df, p, p_corrected, Flagged (cochran), or Flagged only
            (robust, meaning at least one facet element exceeded the threshold on
            that component). None if per_component resolves to False.
        """
        model = self._MODEL_ALIASES.get(model, model)
        if model not in self._MODELS:
            raise ValueError(f"model must be one of {self._MODELS}")
        if rater_homogeneity_test not in ("cochran", "robust"):
            raise ValueError("rater_homogeneity_test must be 'cochran' or 'robust'")
        if correction not in ("bh", "bonferroni", None):
            raise ValueError("correction must be 'bh', 'bonferroni', or None")
        if len(anchors) < 2:
            raise ValueError(
                f"anchors must contain at least 2 {self.facets_label_eg} to test homogeneity."
            )
        if per_component == "auto":
            per_component = model != "global"

        if not hasattr(self, f"facet_effects_{model}"):
            self.calibrate(
                model=model,
                constant=constant,
                threshold_constant=threshold_constant,
                method=method,
                robust=robust,
                matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
            )

        sev_attr = self._sev_attr(model)
        sev0 = getattr(self, sev_attr)

        def _get_component(sev, rater, component):
            if model == "global":
                return sev.loc[rater]
            if model in ("items", "thresholds", "centrality", "pseudo_halo"):
                return sev.loc[rater].mean() if component is None else sev.loc[rater, component]
            # matrix / bivector / bistretch: MultiIndex (facet_element, item) rows, columns=thresholds
            if component is None:
                return sev.loc[rater].values.mean()
            item, k = component
            return sev.loc[(rater, item), k]

        if model in ("items", "thresholds", "centrality", "pseudo_halo"):
            components = list(sev0.columns)
        elif model in ("matrix", "bivector", "bistretch"):
            components = [
                (item, k)
                for item in self.item_names
                for k in range(1, self.max_score + 1)
            ]
        else:
            components = []

        cochran = rater_homogeneity_test == "cochran"
        samples = None
        if cochran:
            if not getattr(self, f"_bootstrap_stored_{model}", False):
                self.std_errors(
                    model=model,
                    no_of_samples=no_of_samples,
                    store_bootstrap=True,
                    constant=constant,
                    threshold_constant=threshold_constant,
                    method=method,
                    robust=robust,
                    matrix_power=matrix_power,
                    log_lik_tol=log_lik_tol,
                    seed=seed,
                )
            samples = getattr(self, f"_bootstrap_samples_{model}")

        def _facet_effects_and_se(component):
            facet_effects = pd.Series(
                {r: _get_component(sev0, r, component) for r in anchors}
            )
            if not cochran:
                return facet_effects, None
            ests = np.array([
                [_get_component(getattr(s, sev_attr), r, component) for r in anchors]
                for s in samples
            ])
            se = pd.Series(np.nanstd(ests, axis=0), index=anchors)
            return facet_effects, se

        def _omnibus(facet_effects, se):
            weights = 1.0 / se**2
            weighted_mean = (weights * facet_effects).sum() / weights.sum()
            q = float((weights * (facet_effects - weighted_mean) ** 2).sum())
            df = len(facet_effects) - 1
            p = float(chi2.sf(q, df))
            z = (facet_effects - weighted_mean) / se
            return weighted_mean, q, df, p, z

        def _sequential_exclusion(facet_effects, se, min_group=2):
            """Iteratively drop the facet element (e.g. rater) with the largest |z| against the
            CURRENT group's weighted mean, recomputing Q each time, until Q
            is no longer significant or min_group facet elements remain.

            A single-pass Cochran's Q against the full group's weighted mean
            is not robust to one severely deviant facet element: that facet element alone
            drags the weighted mean far enough that every OTHER facet element also
            looks significantly different from it, flagging the whole group
            instead of just the true offender. This sequential-exclusion
            approach (standard in meta-analysis heterogeneity diagnostics)
            avoids that by re-centring on each remaining subgroup.
            """
            current = list(facet_effects.index)
            dropped_order = []
            while True:
                sub_sev = facet_effects.loc[current]
                sub_se = se.loc[current]
                weighted_mean, q, df, p, z = _omnibus(sub_sev, sub_se)
                if p >= alpha or len(current) <= min_group:
                    return current, dropped_order, weighted_mean, z
                worst = z.abs().idxmax()
                dropped_order.append((worst, float(z.loc[worst])))
                current.remove(worst)

        # -- Scalar (whole-profile) test --
        scalar_facet_effects, scalar_se = _facet_effects_and_se(None)
        if cochran:
            # Omnibus reflects the anchor set exactly as supplied (before any
            # removal) — this is the "does my proposed set look homogeneous"
            # answer. Sequential exclusion below attributes WHICH rater(s)
            # are responsible if it doesn't.
            _, q, df, p, _ = _omnibus(scalar_facet_effects, scalar_se)
            kept, dropped_order, kept_mean, kept_z = _sequential_exclusion(
                scalar_facet_effects, scalar_se
            )
            rows = {}
            for rater, z_val in dropped_order:
                p_val = float(2 * (1 - norm.cdf(abs(z_val))))
                rows[rater] = {
                    "Facet effect": scalar_facet_effects[rater],
                    "SE": scalar_se[rater],
                    "z": z_val,
                    "p": p_val,
                    "Flagged": True,
                }
            for rater in kept:
                z_val = float(kept_z[rater])
                p_val = float(2 * (1 - norm.cdf(abs(z_val))))
                rows[rater] = {
                    "Facet effect": scalar_facet_effects[rater],
                    "SE": scalar_se[rater],
                    "z": z_val,
                    "p": p_val,
                    "Flagged": False,
                }
            table = pd.DataFrame(rows).T.loc[list(anchors)]
            self.anchor_homogeneity_test = pd.Series(
                {
                    "Q": q,
                    "df": df,
                    "p": p,
                    "Flagged": p < alpha,
                    "N_dropped_for_homogeneity": len(dropped_order),
                },
                name="Anchor homogeneity test",
            )
        else:
            median_val = scalar_facet_effects.median()
            mad = (scalar_facet_effects - median_val).abs().median()
            robust_z = 0.6745 * (scalar_facet_effects - median_val) / mad
            table = pd.DataFrame({"Facet effect": scalar_facet_effects, "Robust z": robust_z})
            table["Flagged"] = table["Robust z"].abs() > robust_z_threshold
            self.anchor_homogeneity_test = None

        self.anchor_homogeneity_per_rater = table

        if table["Flagged"].any():
            offenders = list(table.index[table["Flagged"]])
            warnings.warn(
                f"Anchor {self._facet_n(len(offenders), count=False)} {offenders} {'does' if len(offenders) == 1 else 'do'} not appear homogeneous with "
                f"the rest of the anchor set ({rater_homogeneity_test} test; "
                "see anchor_homogeneity_per_rater). The anchor-defined zero "
                "point may be unstable — consider revising the anchor set.",
                UserWarning,
                stacklevel=2,
            )

        # -- Optional per-component breakdown --
        if per_component and components:
            rows = {}
            for comp in components:
                comp_facet_effects, comp_se = _facet_effects_and_se(comp)
                if cochran:
                    _, q_c, df_c, p_c, _ = _omnibus(comp_facet_effects, comp_se)
                    rows[comp] = {"Q": q_c, "df": df_c, "p": p_c}
                else:
                    median_val = comp_facet_effects.median()
                    mad = (comp_facet_effects - median_val).abs().median()
                    robust_z = 0.6745 * (comp_facet_effects - median_val) / mad
                    rows[comp] = {
                        "Flagged": bool((robust_z.abs() > robust_z_threshold).any())
                    }
            comp_table = pd.DataFrame(rows).T
            if cochran:
                if correction == "bh":
                    comp_table["p_corrected"] = self._bh_correction(comp_table["p"])
                elif correction == "bonferroni":
                    comp_table["p_corrected"] = (
                        comp_table["p"] * len(comp_table)
                    ).clip(upper=1)
                p_col = "p_corrected" if correction else "p"
                comp_table["Flagged"] = comp_table[p_col] < alpha
            self.anchor_homogeneity_per_component = comp_table
        else:
            self.anchor_homogeneity_per_component = None

    def _extract_anchor_adj(self, model, anchors):
        """Extract the anchor adjustment from the current (full-data) calibration."""
        if model == "global":
            return float(self.facet_effects_global.loc[anchors].mean())
        elif model == "items":
            return self.facet_effects_items.loc[anchors].mean(axis=0)
        elif model == "thresholds":
            return self.facet_effects_thresholds.loc[anchors].mean(axis=0)
        elif model == "matrix":
            sev_array = self.facet_effects_matrix.values.reshape(
                self.no_of_facet_elements, self.no_of_items, self.max_score
            )
            anchor_idx = [list(self.facet_names).index(a) for a in anchors]
            return sev_array[anchor_idx].mean(axis=0)  # (I, K)
        elif model == "bivector":
            item_adj = self.facet_effects_bivector_items.loc[anchors].mean(axis=0)
            thr_adj = self.facet_effects_bivector_thresholds.loc[anchors].mean(axis=0)
            return (item_adj, thr_adj)
        elif model in ("centrality", "pseudo_halo", "bistretch") and \
                getattr(self, f"estimation_{model}", "pair-cpat") in ("cp-full-rcl", "cpat-rcl"):
            return self._stretch_anchor_adj(model, anchors)
        elif model == "centrality":
            return self.facet_effects_thresholds.loc[anchors].mean(axis=0)
        elif model == "pseudo_halo":
            return self.facet_effects_items.loc[anchors].mean(axis=0)
        elif model == "bistretch":
            item_adj = self.facet_effects_bivector_items.loc[anchors].mean(axis=0)
            thr_adj = self.facet_effects_bivector_thresholds.loc[anchors].mean(axis=0)
            return (item_adj, thr_adj)

    def _calibrate_anchor_global(self, anchors, adj=None):
        """Anchor calibration for global parameterisation. Shifts all facet effects so anchor mean is zero."""
        self.anchor_items_global = self._items_for("global").copy()
        self.anchor_thresholds_global = self._thresholds_for("global").copy()
        self.anchor_facet_effects_global = self.facet_effects_global.copy()

        if adj is None:
            adj = float(self.facet_effects_global.loc[anchors].mean())
        self.anchor_facet_effects_global -= adj

    def _calibrate_anchor_items(self, anchors, adj=None):
        """Anchor calibration for items parameterisation. Adjusts per-item facet effects and absorbs mean into item item_locations."""
        self.anchor_items_items = self._items_for("items").copy()
        self.anchor_thresholds_items = self._thresholds_for("items").copy()

        sev_df = self.facet_effects_items.copy()  # already (R, I) DataFrame
        if adj is None:
            adj = sev_df.loc[anchors].mean(axis=0)

        self.anchor_items_items += adj
        sev_df -= adj
        self.anchor_facet_effects_items = sev_df
        self.anchor_items_items -= self.anchor_items_items.mean()

    def _calibrate_anchor_thresholds(self, anchors, adj=None):
        """Anchor calibration for thresholds parameterisation. Adjusts per-threshold facet effects and absorbs mean into thresholds."""
        self.anchor_items_thresholds = self._items_for("thresholds").copy()
        self.anchor_thresholds_thresholds = self._thresholds_for("thresholds").copy()

        sev_df = self.facet_effects_thresholds.copy()  # already (R, K+1) DataFrame
        if adj is None:
            adj = sev_df.loc[anchors].mean(axis=0)

        self.anchor_thresholds_thresholds += adj.values
        sev_df -= adj
        self.anchor_facet_effects_thresholds = sev_df
        self.anchor_thresholds_thresholds -= self.anchor_thresholds_thresholds.mean()

    def _stretch_params(self, model):
        """(lambda, omega_items, omega_thresholds) Series of a stretch model's
        calibration (omega fixed at 1 on the axis the model does not stretch)."""
        lam = getattr(self, f"lambda_{model}")
        ones = pd.Series(1.0, index=lam.index)
        if model == "centrality":
            return lam, ones, self.omega_centrality
        if model == "pseudo_halo":
            return lam, self.omega_pseudo_halo, ones
        return lam, self.omega_items_bistretch, self.omega_thresholds_bistretch

    def _stretch_anchor_adj(self, model, anchors):
        """Anchor adjustment of an RCL-CPAT stretch model: the anchors' mean
        shift and mean stretches (see _calibrate_anchor_stretch_rcl)."""
        lam, om_i, om_t = self._stretch_params(model)
        return (float(lam.loc[anchors].mean()), float(om_i.loc[anchors].mean()),
                float(om_t.loc[anchors].mean()))

    def _calibrate_anchor_stretch_rcl(self, model, anchors, adj=None):
        """
        Anchor calibration of a stretch model fitted with estimation=
        'cp-full-rcl' or 'cpat-rcl'. The anchors become the reference: their mean shift
        lambda is set to 0 and their mean stretches omega to 1, with item
        locations and thresholds rescaled to compensate, so that every
        location lambda_r + omega_items_r * delta_i + omega_thresholds_r *
        tau_h is unchanged apart from a common shift (the anchors' mean
        lambda, taken up by the persons):
            omega' = omega / mean(omega over anchors),
            delta' = delta * mean(omega_items over anchors),
            tau' = tau * mean(omega_thresholds over anchors),
            lambda' = lambda - mean(lambda over anchors).
        Item locations and thresholds stay centred.
        """
        lam, om_i, om_t = self._stretch_params(model)
        lam_a, wi_a, wt_a = self._stretch_anchor_adj(model, anchors) if adj is None else adj
        d = self._items_for(model) * wi_a
        tau = self._thresholds_for(model) * wt_a
        lam2, om_i2, om_t2 = lam - lam_a, om_i / wi_a, om_t / wt_a
        items_idx, K = self.responses.columns, self.max_score
        item_dev = pd.DataFrame(lam2.to_numpy()[:, None] + (om_i2.to_numpy()[:, None] - 1) * d.reindex(items_idx).to_numpy()[None, :],
                                index=lam.index, columns=items_idx)
        thr_dev = pd.DataFrame((om_t2.to_numpy()[:, None] - 1) * tau.to_numpy()[None, :],
                               index=lam.index, columns=range(1, K + 1))
        setattr(self, f"anchor_items_{model}", d)
        setattr(self, f"anchor_thresholds_{model}", tau)
        setattr(self, f"anchor_lambda_{model}", lam2)
        if model == "centrality":
            self.anchor_omega_centrality = om_t2
            self.anchor_facet_effects_centrality_full_vector = thr_dev.add(lam2, axis=0)
            self.anchor_facet_effects_centrality = pd.DataFrame({"lambda": lam2, "omega": om_t2})
        elif model == "pseudo_halo":
            self.anchor_omega_pseudo_halo = om_i2
            self.anchor_facet_effects_pseudo_halo_full_vector = item_dev
            self.anchor_facet_effects_pseudo_halo = pd.DataFrame({"lambda": lam2, "omega": om_i2})
        else:
            self.anchor_omega_items_bistretch, self.anchor_omega_thresholds_bistretch = om_i2, om_t2
            self.anchor_facet_effects_bistretch_items = item_dev
            self.anchor_facet_effects_bistretch_thresholds = thr_dev
            full = item_dev.to_numpy()[:, :, None] + thr_dev.to_numpy()[:, None, :]
            mi = pd.MultiIndex.from_product([self.facet_names, self.item_names], names=[self.facet, "item"])
            self.anchor_facet_effects_bistretch_full_vector = pd.DataFrame(
                full.reshape(-1, K), index=mi, columns=range(1, K + 1))
            self.anchor_facet_effects_bistretch = pd.DataFrame(
                {"lambda": lam2, "omega_items": om_i2, "omega_thresholds": om_t2})

    def _calibrate_anchor_centrality(self, anchors, adj=None, regression="huber"):
        """
        Anchor calibration for centrality. Delegates to the
        parent 'thresholds' anchor calibration (same anchor adjustment,
        since centrality is a closed-form reduction of an
        already-calibrated 'thresholds' fit), then re-derives lambda_r/omega_r
        from the anchored facet effects and anchored reference thresholds.
        See _derive_stretch_model for the regression= argument.
        """
        if getattr(self, "estimation_centrality", "pair-cpat") in ("cp-full-rcl", "cpat-rcl"):
            return self._calibrate_anchor_stretch_rcl("centrality", anchors, adj=adj)
        self._calibrate_anchor_thresholds(anchors, adj=adj)
        self.anchor_items_centrality = self.anchor_items_thresholds.copy()
        self.anchor_thresholds_centrality = (
            self.anchor_thresholds_thresholds.copy()
        )
        lam, omega, facet_effects = self._derive_stretch_model(
            self.anchor_facet_effects_thresholds,
            self.anchor_thresholds_thresholds,
            regression=regression,
        )
        self.anchor_lambda_centrality = lam
        self.anchor_omega_centrality = omega
        self.anchor_facet_effects_centrality_full_vector = facet_effects
        self.anchor_facet_effects_centrality = pd.DataFrame({"lambda": lam, "omega": omega})
        self._warn_nonpositive_stretch(omega, "omega", "centrality")

    def _calibrate_anchor_pseudo_halo(self, anchors, adj=None, regression="huber"):
        """
        Anchor calibration for pseudo_halo. Delegates to the parent 'items'
        anchor calibration (same anchor adjustment, since pseudo_halo is a
        closed-form reduction of an already-calibrated 'items' fit), then
        re-derives lambda_r/omega_r from the anchored facet effects and
        anchored reference item locations. See _derive_stretch_model for
        the regression= argument.
        """
        if getattr(self, "estimation_pseudo_halo", "pair-cpat") in ("cp-full-rcl", "cpat-rcl"):
            return self._calibrate_anchor_stretch_rcl("pseudo_halo", anchors, adj=adj)
        self._calibrate_anchor_items(anchors, adj=adj)
        self.anchor_items_pseudo_halo = self.anchor_items_items.copy()
        self.anchor_thresholds_pseudo_halo = self.anchor_thresholds_items.copy()
        lam, omega, facet_effects = self._derive_stretch_model(
            self.anchor_facet_effects_items,
            self.anchor_items_items,
            regression=regression,
        )
        self.anchor_lambda_pseudo_halo = lam
        self.anchor_omega_pseudo_halo = omega
        self.anchor_facet_effects_pseudo_halo_full_vector = facet_effects
        self.anchor_facet_effects_pseudo_halo = pd.DataFrame({"lambda": lam, "omega": omega})
        self._warn_nonpositive_stretch(omega, "omega", "pseudo_halo")

    def _calibrate_anchor_matrix(self, anchors, adj=None):
        """
        Anchor calibration for matrix parameterisation.
        Subtracts the mean anchor facet_element effect (per item, per threshold)
        from all facet_elements, and absorbs it into item item_locations and thresholds.
        """
        self.anchor_items_matrix = self._items_for("matrix").copy()
        self.anchor_thresholds_matrix = self._thresholds_for("matrix").copy()

        # (R, I, K+1) array from MultiIndex DataFrame
        sev_array = self.facet_effects_matrix.values.reshape(
            self.no_of_facet_elements, self.no_of_items, self.max_score
        )

        if adj is None:
            anchor_idx = [list(self.facet_names).index(a) for a in anchors]
            anchor_sev_array = sev_array[anchor_idx]  # (R_anchor, I, K)
            facet_effect_adjustments = anchor_sev_array.mean(axis=0)  # (I, K)
        else:
            facet_effect_adjustments = adj  # (I, K) pre-computed from full data
        diff_adjustments = facet_effect_adjustments.mean(axis=1)  # (I,)
        threshold_adjustments = facet_effect_adjustments.mean(axis=0)  # (K,)

        for i, item in enumerate(self.responses.columns):
            self.anchor_items_matrix[item] += diff_adjustments[i]
        self.anchor_thresholds_matrix += threshold_adjustments

        sev_adj = sev_array.copy()
        for r in range(self.no_of_facet_elements):
            sev_adj[r, :, :] -= facet_effect_adjustments

        self.anchor_items_matrix -= self.anchor_items_matrix.mean()
        self.anchor_thresholds_matrix -= self.anchor_thresholds_matrix.mean()

        mi = pd.MultiIndex.from_product(
            [self.facet_names, self.responses.columns], names=[self.facet, "item"]
        )
        self.anchor_facet_effects_matrix = pd.DataFrame(
            sev_adj.reshape(-1, self.max_score), index=mi, columns=range(1, self.max_score + 1)
        )

        # Marginal facet_effects (no sentinel to skip)
        self.anchor_marginal_facet_effects_items = pd.DataFrame(
            sev_adj.mean(axis=2), index=self.facet_names, columns=self.responses.columns
        )
        self.anchor_marginal_facet_effects_thresholds = pd.DataFrame(
            sev_adj.mean(axis=1), index=self.facet_names, columns=range(1, self.max_score + 1)
        )
        # Zero-sum per facet_element
        adj_thr = self.anchor_marginal_facet_effects_thresholds.mean(axis=1)
        self.anchor_marginal_facet_effects_thresholds = (
            self.anchor_marginal_facet_effects_thresholds.subtract(adj_thr, axis=0)
        )

    def _calibrate_anchor_bivector(self, anchors, adj=None):
        """
        Anchor calibration for the bivector parameterisation.

        Bivector-native anchoring: operates directly on the two marginal
        vectors rather than on the full matrix. Item vector adjustment is
        absorbed into diffs (as in the items model); threshold vector
        adjustment is absorbed into thresholds (as in the thresholds model).
        The anchored full matrix is then reconstructed from the anchored
        vectors.
        """
        self.anchor_items_bivector = self._items_for("bivector").copy()
        self.anchor_thresholds_bivector = self._thresholds_for("bivector").copy()

        # ---- Item vector adjustment --------------------------------------
        item_sev_df = self.facet_effects_bivector_items.copy()  # (R, I) DataFrame
        item_adj = item_sev_df.loc[anchors].mean(axis=0) if adj is None else adj[0]

        self.anchor_items_bivector += item_adj
        item_sev_df -= item_adj
        self.anchor_facet_effects_bivector_items = item_sev_df
        self.anchor_items_bivector -= self.anchor_items_bivector.mean()

        # ---- Threshold vector adjustment ---------------------------------
        thr_sev_df = self.facet_effects_bivector_thresholds.copy()  # (R, K+1) DataFrame
        thr_adj = thr_sev_df.loc[anchors].mean(axis=0) if adj is None else adj[1]

        self.anchor_thresholds_bivector += thr_adj.values
        thr_sev_df -= thr_adj.values
        self.anchor_facet_effects_bivector_thresholds = thr_sev_df
        self.anchor_thresholds_bivector -= self.anchor_thresholds_bivector.mean()

        # ---- Reconstruct anchored full matrix as MultiIndex DataFrame ----
        mi = pd.MultiIndex.from_product(
            [self.facet_names, self.item_names], names=[self.facet, "item"]
        )
        rows = []
        for facet_element in self.facet_names:
            for item in self.item_names:
                row = np.array(
                    [
                        self.anchor_facet_effects_bivector_items.loc[
                            facet_element, item
                        ]
                        + self.anchor_facet_effects_bivector_thresholds.loc[
                            facet_element, k
                        ]
                        for k in range(1, self.max_score + 1)
                    ]
                )
                rows.append(row)
        self.anchor_facet_effects_bivector = pd.DataFrame(rows, index=mi, columns=range(1, self.max_score + 1))

    def _calibrate_anchor_bistretch(self, anchors, adj=None, regression="huber"):
        """
        Anchor calibration for bistretch. Delegates to the parent
        'bivector' anchor calibration (same anchor adjustment, since
        bistretch is a closed-form reduction of an already-calibrated
        'bivector' fit), then re-derives lambda_r/omega_items_r/
        omega_thresholds_r from the anchored facet effects and anchored
        reference items/thresholds. See _derive_stretch_model for the
        regression= argument, applied to both axes.
        """
        if getattr(self, "estimation_bistretch", "pair-cpat") in ("cp-full-rcl", "cpat-rcl"):
            return self._calibrate_anchor_stretch_rcl("bistretch", anchors, adj=adj)
        self._calibrate_anchor_bivector(anchors, adj=adj)
        self.anchor_items_bistretch = self.anchor_items_bivector.copy()
        self.anchor_thresholds_bistretch = self.anchor_thresholds_bivector.copy()

        lam, omega_items, item_recon = self._derive_stretch_model(
            self.anchor_facet_effects_bivector_items,
            self.anchor_items_bivector,
            regression=regression,
        )
        _, omega_thresholds, thresh_recon = self._derive_stretch_model(
            self.anchor_facet_effects_bivector_thresholds,
            self.anchor_thresholds_bivector,
            regression=regression,
        )
        self.anchor_lambda_bistretch = lam
        self.anchor_omega_items_bistretch = omega_items
        self.anchor_omega_thresholds_bistretch = omega_thresholds
        self.anchor_facet_effects_bistretch_items = item_recon
        self.anchor_facet_effects_bistretch_thresholds = thresh_recon

        mi = pd.MultiIndex.from_product(
            [self.facet_names, self.item_names], names=[self.facet, "item"]
        )
        rows = []
        for facet_element in self.facet_names:
            for item in self.item_names:
                row = np.array(
                    [
                        item_recon.loc[facet_element, item]
                        + thresh_recon.loc[facet_element, k]
                        for k in range(1, self.max_score + 1)
                    ]
                )
                rows.append(row)
        self.anchor_facet_effects_bistretch_full_vector = pd.DataFrame(
            rows, index=mi, columns=range(1, self.max_score + 1)
        )
        self.anchor_facet_effects_bistretch = pd.DataFrame(
            {"lambda": lam, "omega_items": omega_items, "omega_thresholds": omega_thresholds}
        )

        self._warn_nonpositive_stretch(omega_items, "omega_items", "bistretch")
        self._warn_nonpositive_stretch(omega_thresholds, "omega_thresholds", "bistretch")

    # Backwards-compatible aliases
    def calibrate_global_anchor(self, anchors, **kw):
        """Alias for calibrate_anchor('global', anchors). See calibrate_anchor for full documentation."""
        self.calibrate_anchor("global", anchors, **kw)

    def calibrate_items_anchor(self, anchors, **kw):
        """Alias for calibrate_anchor('items', anchors). See calibrate_anchor for full documentation."""
        self.calibrate_anchor("items", anchors, **kw)

    def calibrate_thresholds_anchor(self, anchors, **kw):
        """Alias for calibrate_anchor('thresholds', anchors). See calibrate_anchor for full documentation."""
        self.calibrate_anchor("thresholds", anchors, **kw)

    def calibrate_matrix_anchor(self, anchors, **kw):
        """Alias for calibrate_anchor('matrix', anchors). See calibrate_anchor for full documentation."""
        self.calibrate_anchor("matrix", anchors, **kw)

    def calibrate_bivector_anchor(self, anchors, **kw):
        """Alias for calibrate_anchor('bivector', anchors). See calibrate_anchor for full documentation."""
        self.calibrate_anchor("bivector", anchors, **kw)

    def calibrate_centrality_anchor(self, anchors, **kw):
        """Alias for calibrate_anchor('centrality', anchors). See calibrate_anchor for full documentation."""
        self.calibrate_anchor("centrality", anchors, **kw)

    def calibrate_pseudo_halo_anchor(self, anchors, **kw):
        """Alias for calibrate_anchor('pseudo_halo', anchors). See calibrate_anchor for full documentation."""
        self.calibrate_anchor("pseudo_halo", anchors, **kw)

    def calibrate_bistretch_anchor(self, anchors, **kw):
        """Alias for calibrate_anchor('bistretch', anchors). See calibrate_anchor for full documentation."""
        self.calibrate_anchor("bistretch", anchors, **kw)

    # ------------------------------------------------------------------
    # Standard errors (bootstrap)
    # ------------------------------------------------------------------

    def _bootstrap_samples(self, no_of_samples, seed=None):
        """Generate bootstrap person samples preserving facet_element structure."""
        rng = np.random.default_rng(seed)
        picks = [
            self.responses.index.get_level_values(1)[
                rng.integers(0, self.no_of_persons, self.no_of_persons)
            ]
            for _ in range(no_of_samples)
        ]
        data_dict = {
            facet_element: self.responses.xs(facet_element)
            for facet_element in self.facet_names
        }
        # built one at a time (a generator), so only the resamples the caller
        # keeps are held in memory at once
        for pick in picks:
            sample_dict = {
                facet_element: data_dict[facet_element].loc[pick].reset_index(drop=True)
                for facet_element in self.facet_names
            }
            sample = pd.concat(sample_dict.values(), keys=sample_dict.keys())
            with warnings.catch_warnings():
                # resamples keep every facet element (an element can be missed by
                # chance), so their estimates line up with the original's
                warnings.simplefilter("ignore", UserWarning)
                yield MFRM(sample, self.max_score, validate=False,
                           facet=self.facet if self.facet_label != "facet element" else None,
                           facet_plural=self.facets if self.facet_label != "facet element" else None,
                           _keep_empty=True)

    def _resamples(self, no_of_samples, seed=None, keep=False):
        """
        Bootstrap resamples (see _bootstrap_samples()), one at a time. Once the
        caller's loop has moved on from a resample, its responses and fit
        working arrays are dropped -- they are most of its memory, and only
        its estimates are used afterwards -- unless keep=True (the resamples
        are stored for later refits, e.g. store_bootstrap=True).
        """
        for s in self._bootstrap_samples(no_of_samples, seed=seed):
            yield s
            if not keep:
                for name in ("responses", "_rcl_fits", "extreme_persons", "invalid_responses"):
                    s.__dict__.pop(name, None)

    def _se_from_bootstrap(self, ests_arr, labels, interval):
        """Compute SE and optional CI from a (B, N) bootstrap array."""
        se = np.nanstd(ests_arr, axis=0)
        if interval is not None:
            lo = np.percentile(ests_arr, 50 * (1 - interval), axis=0)
            hi = np.percentile(ests_arr, 50 * (1 + interval), axis=0)
        else:
            lo = hi = None
        return se, lo, hi

    def std_errors(
        self,
        model="global",
        anchors=None,
        interval=None,
        no_of_samples=500,
        constant=None,
        threshold_constant=None,
        method=None,
        robust=False,
        matrix_power=None,
        log_lik_tol=0.000001,
        store_bootstrap=False,
        seed=None,
        se_method=None,
    ):
        """
        Standard errors for item locations, thresholds, and facet_element
        effects for the specified model.

        se_method='sandwich' (sub-models calibrated with estimation=
        'cp-full-rcl' or 'cpat-rcl'): analytic standard errors. The covariance of the fitted
        parameters is a sandwich, with bread the negative Hessian of the
        penalised composite log-likelihood at the estimates and meat the
        sum over persons of their score outer products (persons as the
        independent units); the standard errors of every reported quantity
        follow by the delta method, and intervals are estimate +/- z * SE.
        se_method='bootstrap': no_of_samples bootstrap resamples of persons,
        each recalibrated, as before v1.4.0.
        se_method=None (default): the sandwich for an RCL ('cp-full-rcl' or 'cpat-rcl') sub-model,
        unless store_bootstrap=True (which needs the fitted resamples) or
        the sub-model is 'mixed', in which case the bootstrap; and the
        bootstrap, with a UserWarning giving the reason, whenever the
        sandwich does not apply: an unused score category (whose
        neighbouring widths are set by the cpat-cml rule), a calibration
        that did not converge, or estimates the data do not identify (whose
        bootstrap standard errors then reflect the penalties and starting
        values). So standard errors are always produced. The bootstrap for a
        'pair-cpat' sub-model. An explicit se_method='sandwich' gives NaN
        for estimates the data do not identify, with a UserWarning. self.se_method_{model} (and
        self.se_method) record the method used.

        Parameters
        ----------
        store_bootstrap : bool, default False
            If True, store the fitted bootstrap samples as
            self._bootstrap_samples_{model} and set
            self._bootstrap_stored_{model} = True. Allows
            anchor_std_errors() to reuse the same samples without
            rerunning the bootstrap. Memory cost: no_of_samples fitted
            MFRM objects.
        seed : int or None, default None
            Seed for the bootstrap resampling RNG. Pass an int for fully
            reproducible standard errors; None (default) draws fresh entropy.
        se_method : {None, 'sandwich', 'bootstrap'}, default None
            Standard error method; see above.
        """
        model = self._MODEL_ALIASES.get(model, model)
        if se_method not in (None, "sandwich", "bootstrap"):
            raise ValueError(
                f"se_method must be None, 'sandwich' or 'bootstrap', got {se_method!r}."
            )
        if model != "mixed" and not hasattr(self, f"facet_effects_{model}"):
            self.calibrate(model=model)
        rcl = model != "mixed" and getattr(self, f"estimation_{model}", None) in ("cp-full-rcl", "cpat-rcl")
        if se_method == "sandwich" and (not rcl or store_bootstrap):
            raise ValueError(
                "se_method='sandwich' needs a sub-model calibrated with estimation='cp-full-rcl' (or 'cpat-rcl') "
                "(not 'mixed'), and no stored bootstrap samples (store_bootstrap=False)."
            )
        if se_method is None and rcl and not store_bootstrap:
            obstacle = self._rcl_sandwich_obstacle(model)
            if obstacle is not None:
                warnings.warn(
                    f"Standard errors for model='{model}' computed by bootstrap rather than "
                    f"analytically (sandwich), because {obstacle}. Pass se_method= to choose "
                    f"the method explicitly.",
                    UserWarning,
                    stacklevel=2,
                )
                se_method = "bootstrap"
            else:
                se_method = "sandwich"
        if se_method == "sandwich":
            self.se_method = "sandwich"
            setattr(self, f"se_method_{model}", "sandwich")
            setattr(self, f"_bootstrap_stored_{model}", False)
            setattr(self, f"_bootstrap_interval_{model}", interval)
            self._rcl_sandwich_std_errors(model, anchors, interval)
            return
        self.se_method = "bootstrap"
        setattr(self, f"se_method_{model}", "bootstrap")
        if model == "mixed":
            self._sync_mixed(
                anchors,
                constant=constant,
                threshold_constant=threshold_constant,
                method=method,
                robust=robust,
                matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
                seed=seed,
            )

        # 'mixed' has no calibrate()/calibrate_anchor() of its own — each rater's
        # facet_effect is a marginal projection of the matrix calibration,
        # restricted to that rater's individually assigned model. Bootstrap it by
        # recalibrating 'matrix' on the resample (anchored, if anchors is given),
        # then re-applying the ORIGINAL (non-resampled) per-rater model
        # assignment, so bootstrap variance reflects resampling only, not
        # re-selection noise. The assignment used is anchor_rater_models when
        # bootstrapping in anchor mode (it can genuinely differ per anchor set),
        # else the unanchored rater_models.
        cal_model = "matrix" if model == "mixed" else model
        rater_models_for_bootstrap = (
            (self.anchor_rater_models if anchors is not None else self.rater_models)
            if model == "mixed"
            else None
        )

        # Pre-compute anchor adjustment from full-data calibration so each
        # bootstrap sample uses a fixed scale shift rather than re-estimating
        # adj from the resample (which would inflate SEs with anchor rater
        # sampling variance).
        adj_fixed = self._extract_anchor_adj(cal_model, anchors) if anchors is not None else None

        samples = []
        collector = _BootstrapWarnings()
        for s in self._resamples(no_of_samples, seed=seed, keep=store_bootstrap):
            samples.append(s)
            with collector.resample():
                if model == "mixed":
                    s.calibrate(
                        **self._refit_kw("matrix"),
                        model="matrix",
                        constant=constant,
                        threshold_constant=threshold_constant,
                        method=method,
                        robust=robust,
                        matrix_power=matrix_power,
                        log_lik_tol=log_lik_tol,
                    )
                    if anchors is not None:
                        s.calibrate_anchor(
                            "matrix",
                            anchors,
                            constant=constant,
                            threshold_constant=threshold_constant,
                            method=method,
                            robust=robust,
                            matrix_power=matrix_power,
                            log_lik_tol=log_lik_tol,
                            adj=adj_fixed,
                        )
                        s.anchor_facet_effects_mixed = s._apply_rater_models(
                            s.anchor_facet_effects_matrix, rater_models_for_bootstrap, anchor=True
                        )
                    else:
                        s.facet_effects_mixed = s._apply_rater_models(
                            s.facet_effects_matrix, rater_models_for_bootstrap
                        )
                    continue
                s.calibrate(
                    **self._refit_kw(model),
                    model=model,
                    constant=constant,
                    threshold_constant=threshold_constant,
                    method=method,
                    robust=robust,
                    matrix_power=matrix_power,
                    log_lik_tol=log_lik_tol,
                )
                if anchors is not None:
                    s.calibrate_anchor(
                        model,
                        anchors,
                        constant=constant,
                        threshold_constant=threshold_constant,
                        method=method,
                        robust=robust,
                        matrix_power=matrix_power,
                        log_lik_tol=log_lik_tol,
                        adj=adj_fixed,
                    )
        collector.emit()

        if store_bootstrap:
            setattr(self, f"_bootstrap_samples_{model}", samples)
            setattr(self, f"_bootstrap_stored_{model}", True)
        else:
            setattr(self, f"_bootstrap_stored_{model}", False)

        setattr(self, f"_bootstrap_interval_{model}", interval)

        anc = anchors is not None
        prefix = "anchor_" if anc else ""

        # Item estimates
        if anc:
            item_ests = np.array(
                [getattr(s, f"anchor_items_{cal_model}").values for s in samples]
            )
            thresh_ests = np.array(
                [getattr(s, f"anchor_thresholds_{cal_model}") for s in samples]
            )
        else:
            item_ests = np.array([s.items.values for s in samples])
            thresh_ests = np.array([s.thresholds.values for s in samples])

        item_se, item_lo, item_hi = self._se_from_bootstrap(
            item_ests, self.responses.columns, interval
        )
        # Item locations can differ between sub-models (estimation='cp-full-rcl' (or 'cpat-rcl')),
        # so their SEs are stored per sub-model; the unsuffixed attributes
        # hold the latest.
        for suffix in ("", f"_{model}"):
            setattr(self, f"{prefix}item_se{suffix}", pd.Series(item_se, index=self.responses.columns))
            if item_lo is not None:
                setattr(self, f"{prefix}item_low{suffix}", pd.Series(item_lo, index=self.responses.columns))
                setattr(self, f"{prefix}item_high{suffix}", pd.Series(item_hi, index=self.responses.columns))

        thr_se, thr_lo, thr_hi = self._se_from_bootstrap(thresh_ests, None, interval)
        setattr(self, f"{prefix}threshold_se_{model}", thr_se)
        setattr(self, f"{prefix}threshold_low_{model}", thr_lo)
        setattr(self, f"{prefix}threshold_high_{model}", thr_hi)

        # Category width SEs
        cat_widths = {
            k + 1: thresh_ests[:, k + 1] - thresh_ests[:, k]
            for k in range(self.max_score - 1)
        }
        setattr(self, f"{prefix}cat_width_bootstrap_{model}", cat_widths)
        setattr(
            self,
            f"{prefix}cat_width_se_{model}",
            {k: np.nanstd(v) for k, v in cat_widths.items()},
        )
        if interval is not None:
            setattr(
                self,
                f"{prefix}cat_width_low_{model}",
                {
                    k: np.percentile(v, 50 * (1 - interval))
                    for k, v in cat_widths.items()
                },
            )
            setattr(
                self,
                f"{prefix}cat_width_high_{model}",
                {
                    k: np.percentile(v, 50 * (1 + interval))
                    for k, v in cat_widths.items()
                },
            )

        # Rater SE — structure differs by model
        self._store_rater_se(model, samples, anc, interval, prefix)

    def _store_rater_se(self, model, samples, anchor, interval, prefix):
        self._store_rater_se_impl(model, samples, anchor, interval, prefix)
        self._nan_extreme_facet_se(model, prefix)

    def _store_stretch_param_se(self, model, samples, anchor, interval, prefix, param_names):
        self._store_stretch_param_se_impl(model, samples, anchor, interval, prefix, param_names)
        self._nan_extreme_facet_se(model, prefix)

    def _rcl_sandwich_std_errors(self, model, anchors, interval):
        self._rcl_sandwich_std_errors_impl(model, anchors, interval)
        self._nan_extreme_facet_se(model, "anchor_" if anchors is not None else "")

    def _nan_extreme_facet_se(self, model, prefix):
        """Set the standard errors (and interval bounds) of extreme facet elements to NaN: their
        locations are not identified by the data, so no standard error is meaningful."""
        ext = list(getattr(self, "extreme_facet_elements", None) or [])
        if not ext:
            return
        names = [f"{prefix}rater_{k}_{model}" for k in ("se", "low", "high")]
        names += [f"{prefix}rater_se_marginal_items", f"{prefix}rater_se_marginal_thresholds"]
        names += [f"{prefix}{p}_{k}_{model}" for p in ("lambda", "omega", "omega_items", "omega_thresholds")
                  for k in ("se", "low", "high")]
        for name in names:
            v = getattr(self, name, None)
            if isinstance(v, (pd.Series, pd.DataFrame)):
                idx = v.index
                mask = (idx.get_level_values(0) if isinstance(idx, pd.MultiIndex) else idx).isin(ext)
                if mask.any():
                    v = v.copy()
                    v.loc[mask] = np.nan
                    setattr(self, name, v)
            elif isinstance(v, dict):
                v = dict(v)
                for e in ext:
                    if e in v:
                        v[e] = np.nan * np.asarray(v[e], dtype=float) if np.ndim(v[e]) else np.nan
                setattr(self, name, v)

    def _rcl_sandwich_obstacle(self, model):
        """None if the sandwich applies to an RCL-CPAT sub-model; otherwise
        the reason the bootstrap is used instead: an unused score category
        (the widths next to it are set by the cpat-cml rule), a calibration
        that did not converge, or estimates the data do not identify."""
        from raschpy import _rcl
        fit = self._rcl_fits[model]
        x = self.responses.to_numpy(dtype=float)
        empty = [c for c in range(self.max_score + 1) if (x == c).sum() == 0]
        if empty:
            return (
                f"score categor{'ies' if len(empty) > 1 else 'y'} {', '.join(map(str, empty))} "
                f"{'have' if len(empty) > 1 else 'has'} no responses: the category widths next to "
                f"{'them' if len(empty) > 1 else 'it'} are set by a rule the analytic standard "
                f"errors do not account for"
            )
        if not getattr(self, f"convergence_{model}")["converged"]:
            return "the calibration did not converge, and the sandwich assumes it did"
        R, I, K = self.no_of_facet_elements, self.no_of_items, self.max_score
        X = self.responses.to_numpy(dtype=np.float64).reshape(R, -1, I)
        if fit["kind"] == "cell":
            try:
                null = _rcl.cell_null(fit["cm"], fit["theta"], fit["info"])
            except _rcl.SandwichTooLarge as ex:
                return (f"the design is too large to form the analytic standard errors' "
                        f"information matrix ({ex})")
        else:
            null = _rcl.stretch_null(fit["sm"], fit["theta"], fixed=fit["fixed"])
        if null.shape[1]:
            if fit["kind"] == "cell":
                _rcl.release_state(fit["info"])
            return (
                "the data do not identify some of the estimates (for example the locations of "
                f"{self.facets_label} never compared with the others, or of items not linked "
                "to the others); their bootstrap standard errors reflect the penalties and "
                "starting values rather than the data"
            )
        return None

    def _rcl_sandwich_std_errors_impl(self, model, anchors, interval):
        """Sandwich standard errors for a sub-model calibrated by RCL-CPAT
        (see std_errors()), stored under the same attribute names as the
        bootstrap, with intervals estimate +/- z * SE and the bootstrap
        attributes None."""
        from raschpy import _rcl

        fit = self._rcl_fits[model]
        R, I, K = self.no_of_facet_elements, self.no_of_items, self.max_score
        X = self.responses.to_numpy(dtype=np.float64).reshape(R, -1, I)
        anc = anchors is not None
        prefix = "anchor_" if anc else ""
        if anc:
            self.calibrate_anchor(model, anchors)
        n_s = R * I * K
        stretch_rows = None
        if fit["kind"] == "cell":
            cm, theta, info = fit["cm"], fit["theta"], fit["info"]
            V, free, null = _rcl.cell_sandwich(cm, theta, info, X, K)
            J = info["_state"]["J"]
            Z = np.zeros((I + K + n_s, cm.p))
            Z[:I, cm.slices["delta"]] = cm._NI
            Z[I:I + K, cm.slices["tau"]] = cm._T
            Zs = J.copy()
            Zs[:, cm.slices["delta"]] = 0.0
            Zs[:, cm.slices["tau"]] = 0.0
            Z[I + K:] = Zs
            Z = Z[:, free]
            del J, Zs
            _rcl.release_state(info)
            # anchoring with a fixed adjustment shifts estimates by constants
            C = Z @ V @ Z.T
            out_null = Z @ null
        else:
            sm, theta = fit["sm"], fit["theta"]
            V, free, null = _rcl.stretch_sandwich(sm, theta, X, K, fixed=fit["fixed"], drop_widths=fit["split"])
            Mf = sm.M[:, free]
            C_phi = Mf @ V @ Mf.T
            adj = self._stretch_anchor_adj(model, anchors) if anc else None

            def outputs(phi):
                oD, oT, oL, oI, oW = sm.off
                d, tau, lam = phi[oD:oD + I], phi[oT:oT + K], phi[oL:oL + R]
                wI, wT = phi[oI:oI + R], phi[oW:oW + R]
                if adj is not None:
                    lam_a, wi_a, wt_a = adj
                    d, tau, lam, wI, wT = d * wi_a, tau * wt_a, lam - lam_a, wI / wi_a, wT / wt_a
                s = lam[:, None, None] + (wI[:, None, None] - 1) * d[None, :, None] + (wT[:, None, None] - 1) * tau[None, None, :]
                return np.concatenate([d, tau, s.ravel(), lam, wI, wT])

            phi0 = sm.phi0 + sm.M @ theta
            h = 1e-6
            F = np.column_stack([(outputs(phi0 + h * e) - outputs(phi0 - h * e)) / (2 * h)
                                 for e in np.eye(len(phi0))])
            C = F @ C_phi @ F.T
            out_null = F @ Mf @ null
            stretch_rows = I + K + n_s

        if out_null.shape[1]:
            # directions the data do not identify: no standard error
            hit = np.abs(out_null).max(axis=1) > 1e-8
            C = C.copy()
            C[hit, :] = np.nan
            C[:, hit] = np.nan
            _rcl.warn_unidentified(stacklevel=4)
        var = np.clip(np.diag(C), 0, None)
        z = norm.ppf(0.5 + interval / 2) if interval else None
        items_idx, cols_k = self.responses.columns, range(1, K + 1)

        # items
        item_se = pd.Series(np.sqrt(var[:I]), index=items_idx)
        item_est = self._items_for(model, anchor=anc).reindex(items_idx)
        for suffix in ("", f"_{model}"):
            setattr(self, f"{prefix}item_se{suffix}", item_se)
            if z is not None:
                setattr(self, f"{prefix}item_low{suffix}", item_est - z * item_se)
                setattr(self, f"{prefix}item_high{suffix}", item_est + z * item_se)
        # thresholds and widths
        thr_se = np.sqrt(var[I:I + K])
        thr_est = np.asarray(self._thresholds_for(model, anchor=anc), dtype=float)
        setattr(self, f"{prefix}threshold_se_{model}", thr_se)
        setattr(self, f"{prefix}threshold_low_{model}", thr_est - z * thr_se if z is not None else None)
        setattr(self, f"{prefix}threshold_high_{model}", thr_est + z * thr_se if z is not None else None)
        Dm = np.diff(np.eye(K), axis=0)
        w_se = np.sqrt(np.clip(np.diag(Dm @ C[I:I + K, I:I + K] @ Dm.T), 0, None))
        widths = np.diff(thr_est)
        for k in fit["split"]:
            w_se[k] = np.nan                                  # set by the cpat-cml rule, not by the fit
        setattr(self, f"{prefix}cat_width_bootstrap_{model}", None)
        setattr(self, f"{prefix}cat_width_se_{model}", {k + 1: float(w_se[k]) for k in range(K - 1)})
        if z is not None:
            setattr(self, f"{prefix}cat_width_low_{model}", {k + 1: widths[k] - z * w_se[k] for k in range(K - 1)})
            setattr(self, f"{prefix}cat_width_high_{model}", {k + 1: widths[k] + z * w_se[k] for k in range(K - 1)})

        # facet element effects: covariance of the location-grid deviations s
        Cs = C[I + K:I + K + n_s, I + K:I + K + n_s].reshape(R, I, K, R, I, K)
        r_ix = np.arange(R)

        def se_cells():
            return np.sqrt(np.clip(np.diag(C)[I + K:I + K + n_s], 0, None)).reshape(R, I, K)

        def se_item_part():
            # mean over thresholds of s, per (facet element, item)
            v = np.array([[Cs[r, i, :, r, i, :].sum() / K ** 2 for i in range(I)] for r in r_ix])
            return np.sqrt(np.clip(v, 0, None))

        def se_threshold_part(centred_over_items):
            # thresholds part: s - mean over thresholds (first item), or the
            # matrix marginal: mean over items, centred over thresholds
            L = np.eye(K) - np.full((K, K), 1 / K)
            out = np.zeros((R, K))
            for r in r_ix:
                if centred_over_items:
                    Cr = Cs[r, :, :, r, :, :].reshape(I * K, I * K)
                    A = np.kron(np.full((1, I), 1 / I), L)
                    out[r] = np.diag(A @ Cr @ A.T)
                else:
                    out[r] = np.diag(L @ Cs[r, 0, :, r, 0, :] @ L.T)
            return np.sqrt(np.clip(out, 0, None))

        sev_est = getattr(self, self._sev_attr(model, anchor=anc))
        cells = se_cells()
        facets = self.facet_names

        def store_interval(se_obj, est_obj):
            if z is not None:
                setattr(self, f"{prefix}rater_low_{model}", est_obj - z * se_obj)
                setattr(self, f"{prefix}rater_high_{model}", est_obj + z * se_obj)

        if model == "global":
            se = pd.Series(cells[:, 0, 0], index=facets)
            setattr(self, f"{prefix}rater_se_{model}", se)
            store_interval(se, sev_est.reindex(facets))
        elif model in ("items", "pseudo_halo"):
            se = pd.DataFrame(cells[:, :, 0], index=facets, columns=items_idx)
            setattr(self, f"{prefix}rater_se_{model}", se)
            store_interval(se, sev_est.reindex(index=facets, columns=items_idx))
        elif model in ("thresholds", "centrality"):
            se = pd.DataFrame(cells[:, 0, :], index=facets, columns=cols_k)
            setattr(self, f"{prefix}rater_se_{model}", se)
            store_interval(se, sev_est.reindex(index=facets, columns=cols_k))
        elif model == "matrix":
            mi = pd.MultiIndex.from_product([facets, items_idx], names=[self.facet, "item"])
            se = pd.DataFrame(cells.reshape(-1, K), index=mi, columns=cols_k)
            setattr(self, f"{prefix}rater_se_{model}", se)
            setattr(self, f"{prefix}rater_se_marginal_items",
                    pd.DataFrame(se_item_part(), index=facets, columns=items_idx))
            setattr(self, f"{prefix}rater_se_marginal_thresholds",
                    pd.DataFrame(se_threshold_part(True), index=facets, columns=cols_k))
            store_interval(se, sev_est.reindex(index=mi, columns=cols_k))
        else:  # bivector, bistretch: item part (mean over thresholds) and threshold part
            se_items = pd.DataFrame(se_item_part(), index=facets, columns=items_idx)
            se_thresholds = pd.DataFrame(se_threshold_part(False), index=facets, columns=cols_k)
            setattr(self, f"{prefix}rater_se_marginal_items", se_items)
            setattr(self, f"{prefix}rater_se_marginal_thresholds", se_thresholds)
            setattr(self, f"{prefix}rater_se_{model}", se_items)

        if stretch_rows is not None:
            names = {"centrality": ["lambda", "omega"], "pseudo_halo": ["lambda", "omega"],
                     "bistretch": ["lambda", "omega_items", "omega_thresholds"]}[model]
            rows = {"lambda": stretch_rows, "omega_items": stretch_rows + R, "omega_thresholds": stretch_rows + 2 * R}
            rows["omega"] = rows["omega_thresholds"] if model == "centrality" else rows["omega_items"]
            for pname in names:
                r0 = rows[pname]
                se = pd.Series(np.sqrt(var[r0:r0 + R]), index=facets)
                setattr(self, f"{prefix}{pname}_se_{model}", se)
                if z is not None:
                    est = getattr(self, f"{prefix}{pname}_{model}").reindex(facets)
                    setattr(self, f"{prefix}{pname}_low_{model}", est - z * se)
                    setattr(self, f"{prefix}{pname}_high_{model}", est + z * se)
            cov = {}
            for a_i, p1 in enumerate(names):
                for p2 in names[a_i + 1:]:
                    cov[f"{p1}_{p2}"] = pd.Series(
                        [C[rows[p1] + r, rows[p2] + r] for r in r_ix], index=facets)
            setattr(self, f"{prefix}stretch_param_cov_{model}", cov)
        self._set_facet_aliases(model, anchor=anc)

    def _store_stretch_param_se_impl(self, model, samples, anchor, interval, prefix, param_names):
        """
        Bootstrap SE/CI for a stretch-restricted model's own raw shift/
        stretch parameters (lambda, omega -- or omega_items/
        omega_thresholds for bistretch), as opposed to the reconstructed
        full-vector SE _store_rater_se already computes.
        Stored as self.{prefix}{param}_se_{model}, and (if interval is
        given) self.{prefix}{param}_low_{model}/_high_{model}.

        Also stores self.{prefix}stretch_param_cov_{model}: a dict
        {"{p1}_{p2}": pandas.Series} of per-facet-element bootstrap COVARIANCE
        between every pair of param_names (e.g. "lambda_omega" for
        centrality/pseudo_halo; "lambda_omega_items",
        "lambda_omega_thresholds", "omega_items_omega_thresholds" for
        bistretch) -- needed for drf_test's joint (lambda, omega)
        covariance-aware test, since lambda and omega come from one
        shared regression per facet element (e.g. rater) and are NOT independent. Each pair's
        covariance is computed only from bootstrap samples where BOTH
        parameters were successfully estimated (their own valid subsets
        may differ slightly from each single parameter's own SE
        computation above).
        """
        lo_p = 50 * (1 - interval) if interval is not None else None
        hi_p = 50 * (1 + interval) if interval is not None else None
        for pname in param_names:
            attr = f"anchor_{pname}_{model}" if anchor else f"{pname}_{model}"
            ests = np.array([
                getattr(s, attr).values for s in samples
                if len(getattr(s, attr)) == self.no_of_facet_elements
            ])
            se = pd.Series(np.nanstd(ests, axis=0), index=self.facet_names)
            setattr(self, f"{prefix}{pname}_se_{model}", se)
            if interval is not None:
                setattr(
                    self,
                    f"{prefix}{pname}_low_{model}",
                    pd.Series(np.percentile(ests, lo_p, axis=0), index=self.facet_names),
                )
                setattr(
                    self,
                    f"{prefix}{pname}_high_{model}",
                    pd.Series(np.percentile(ests, hi_p, axis=0), index=self.facet_names),
                )

        if len(param_names) > 1:
            cov_dict = {}
            for i, p1 in enumerate(param_names):
                for p2 in param_names[i + 1:]:
                    attr1 = f"anchor_{p1}_{model}" if anchor else f"{p1}_{model}"
                    attr2 = f"anchor_{p2}_{model}" if anchor else f"{p2}_{model}"
                    valid = [
                        s for s in samples
                        if len(getattr(s, attr1)) == self.no_of_facet_elements
                        and len(getattr(s, attr2)) == self.no_of_facet_elements
                    ]
                    ests1 = np.array([getattr(s, attr1).values for s in valid])
                    ests2 = np.array([getattr(s, attr2).values for s in valid])
                    if len(valid) > 1:
                        cov_vals = np.array([
                            np.cov(ests1[:, r], ests2[:, r])[0, 1]
                            for r in range(ests1.shape[1])
                        ])
                    else:
                        cov_vals = np.full(self.no_of_facet_elements, np.nan)
                    cov_dict[f"{p1}_{p2}"] = pd.Series(cov_vals, index=self.facet_names)
            setattr(self, f"{prefix}stretch_param_cov_{model}", cov_dict)

    def _store_rater_se_impl(self, model, samples, anchor, interval, prefix):
        """Store facet_element SE attributes for the given model."""
        lo_p = 50 * (1 - interval) if interval is not None else None
        hi_p = 50 * (1 + interval) if interval is not None else None

        if model == "global":
            sev_attr = (
                "anchor_facet_effects_global" if anchor else "facet_effects_global"
            )
            rater_ests = np.array([
                getattr(s, sev_attr).values for s in samples
                if len(getattr(s, sev_attr)) == self.no_of_facet_elements
            ])
            se = pd.Series(np.nanstd(rater_ests, axis=0), index=self.facet_names)
            setattr(self, f"{prefix}rater_se_{model}", se)
            if interval is not None:
                setattr(
                    self,
                    f"{prefix}rater_low_{model}",
                    pd.Series(
                        np.percentile(rater_ests, lo_p, axis=0), index=self.facet_names
                    ),
                )
                setattr(
                    self,
                    f"{prefix}rater_high_{model}",
                    pd.Series(
                        np.percentile(rater_ests, hi_p, axis=0), index=self.facet_names
                    ),
                )

        elif model in ("items", "pseudo_halo"):
            sev_attr = self._sev_attr(model, anchor=anchor)
            # Each sample's facet_effects_items/facet_effects_pseudo_halo_full_vector is a (R, I) DataFrame
            rater_ests = np.array(
                [getattr(s, sev_attr).values for s in samples
                 if getattr(s, sev_attr).shape[0] == self.no_of_facet_elements]
            )  # (B, R, I)
            se = pd.DataFrame(
                np.nanstd(rater_ests, axis=0),
                index=self.facet_names,
                columns=self.responses.columns,
            )
            setattr(self, f"{prefix}rater_se_{model}", se)
            if interval is not None:
                setattr(
                    self,
                    f"{prefix}rater_low_{model}",
                    pd.DataFrame(
                        np.percentile(rater_ests, lo_p, axis=0),
                        index=self.facet_names,
                        columns=self.responses.columns,
                    ),
                )
                setattr(
                    self,
                    f"{prefix}rater_high_{model}",
                    pd.DataFrame(
                        np.percentile(rater_ests, hi_p, axis=0),
                        index=self.facet_names,
                        columns=self.responses.columns,
                    ),
                )
            if model == "pseudo_halo":
                self._store_stretch_param_se(
                    model, samples, anchor, interval, prefix, ["lambda", "omega"]
                )

        elif model in ("thresholds", "centrality"):
            sev_attr = self._sev_attr(model, anchor=anchor)
            # Each sample's facet_effects_thresholds/facet_effects_centrality_full_vector
            # is a (R, K) DataFrame
            rater_ests = np.array(
                [getattr(s, sev_attr).values for s in samples
                 if getattr(s, sev_attr).shape[0] == self.no_of_facet_elements]
            )  # (B, R, K+1)
            se = pd.DataFrame(np.nanstd(rater_ests, axis=0), index=self.facet_names, columns=range(1, self.max_score + 1))
            setattr(self, f"{prefix}rater_se_{model}", se)
            if interval is not None:
                setattr(
                    self,
                    f"{prefix}rater_low_{model}",
                    pd.DataFrame(
                        np.percentile(rater_ests, lo_p, axis=0), index=self.facet_names, columns=range(1, self.max_score + 1)
                    ),
                )
                setattr(
                    self,
                    f"{prefix}rater_high_{model}",
                    pd.DataFrame(
                        np.percentile(rater_ests, hi_p, axis=0), index=self.facet_names, columns=range(1, self.max_score + 1)
                    ),
                )
            if model == "centrality":
                self._store_stretch_param_se(
                    model, samples, anchor, interval, prefix, ["lambda", "omega"]
                )

        elif model == "bivector":
            sev_i_attr = (
                "anchor_facet_effects_bivector_items"
                if anchor
                else "facet_effects_bivector_items"
            )
            sev_t_attr = (
                "anchor_facet_effects_bivector_thresholds"
                if anchor
                else "facet_effects_bivector_thresholds"
            )
            # Both are (R, I) and (R, K+1) DataFrames
            valid = [s for s in samples
                     if getattr(s, sev_i_attr).shape[0] == self.no_of_facet_elements]
            item_ests = np.array(
                [getattr(s, sev_i_attr).values for s in valid]
            )  # (B, R, I)
            thr_ests = np.array(
                [getattr(s, sev_t_attr).values for s in valid]
            )  # (B, R, K+1)
            se_items = pd.DataFrame(
                np.nanstd(item_ests, axis=0),
                index=self.facet_names,
                columns=self.responses.columns,
            )
            se_thresholds = pd.DataFrame(
                np.nanstd(thr_ests, axis=0), index=self.facet_names, columns=range(1, self.max_score + 1)
            )
            setattr(self, f"{prefix}rater_se_marginal_items", se_items)
            setattr(self, f"{prefix}rater_se_marginal_thresholds", se_thresholds)
            setattr(self, f"{prefix}rater_se_{model}", se_items)

        elif model == "bistretch":
            sev_i_attr = (
                "anchor_facet_effects_bistretch_items"
                if anchor
                else "facet_effects_bistretch_items"
            )
            sev_t_attr = (
                "anchor_facet_effects_bistretch_thresholds"
                if anchor
                else "facet_effects_bistretch_thresholds"
            )
            # Both are (R, I) and (R, K) DataFrames, same shape as bivector's
            # own split marginal vectors -- reuse the identical approach.
            valid = [s for s in samples
                     if getattr(s, sev_i_attr).shape[0] == self.no_of_facet_elements]
            item_ests = np.array(
                [getattr(s, sev_i_attr).values for s in valid]
            )  # (B, R, I)
            thr_ests = np.array(
                [getattr(s, sev_t_attr).values for s in valid]
            )  # (B, R, K)
            se_items = pd.DataFrame(
                np.nanstd(item_ests, axis=0),
                index=self.facet_names,
                columns=self.responses.columns,
            )
            se_thresholds = pd.DataFrame(
                np.nanstd(thr_ests, axis=0), index=self.facet_names, columns=range(1, self.max_score + 1)
            )
            setattr(self, f"{prefix}rater_se_marginal_items", se_items)
            setattr(self, f"{prefix}rater_se_marginal_thresholds", se_thresholds)
            setattr(self, f"{prefix}rater_se_{model}", se_items)
            self._store_stretch_param_se(
                model, samples, anchor, interval, prefix, ["lambda", "omega_items", "omega_thresholds"]
            )

        elif model == "matrix":
            sev_attr = (
                "anchor_facet_effects_matrix" if anchor else "facet_effects_matrix"
            )
            # Each sample's facet_effects_matrix is a MultiIndex DataFrame (R×I, K)
            # Skip samples where a rater was dropped during bootstrap resampling
            expected_rows = self.no_of_facet_elements * self.no_of_items
            rater_ests = np.array(
                [
                    getattr(s, sev_attr).values
                    for s in samples
                    if getattr(s, sev_attr).shape[0] == expected_rows
                ]
            )  # (B, R*I, K)
            mi = pd.MultiIndex.from_product(
                [self.facet_names, self.responses.columns], names=[self.facet, "item"]
            )
            se = pd.DataFrame(np.nanstd(rater_ests, axis=0), index=mi, columns=range(1, self.max_score + 1))
            setattr(self, f"{prefix}rater_se_{model}", se)

            # Marginal SEs: item = mean over K, threshold = mean over I
            sev_4d = rater_ests.reshape(
                len(rater_ests),
                self.no_of_facet_elements,
                self.no_of_items,
                self.max_score,
            )
            se_marginal_items = pd.DataFrame(
                np.nanstd(sev_4d.mean(axis=3), axis=0),
                index=self.facet_names,
                columns=self.responses.columns,
            )
            thr_means = sev_4d.mean(axis=2)  # (B, R, K) — mean over I
            se_marginal_thresholds = pd.DataFrame(
                np.nanstd(thr_means, axis=0), index=self.facet_names, columns=range(1, self.max_score + 1)
            )
            setattr(self, f"{prefix}rater_se_marginal_items", se_marginal_items)
            setattr(
                self, f"{prefix}rater_se_marginal_thresholds", se_marginal_thresholds
            )

            if interval is not None:
                setattr(
                    self,
                    f"{prefix}rater_low_{model}",
                    pd.DataFrame(np.percentile(rater_ests, lo_p, axis=0), index=mi, columns=range(1, self.max_score + 1)),
                )
                setattr(
                    self,
                    f"{prefix}rater_high_{model}",
                    pd.DataFrame(np.percentile(rater_ests, hi_p, axis=0), index=mi, columns=range(1, self.max_score + 1)),
                )

        elif model == "mixed":
            # facet_effects_mixed has the same MultiIndex (R×I, K) shape as
            # facet_effects_matrix (each rater's row is a marginal projection of
            # the matrix calibration under that rater's assigned model) — reuse
            # the matrix branch's reshape/store logic unchanged.
            sev_attr = "anchor_facet_effects_mixed" if anchor else "facet_effects_mixed"
            expected_rows = self.no_of_facet_elements * self.no_of_items
            rater_ests = np.array(
                [
                    getattr(s, sev_attr).values
                    for s in samples
                    if getattr(s, sev_attr).shape[0] == expected_rows
                ]
            )  # (B, R*I, K)
            mi = pd.MultiIndex.from_product(
                [self.facet_names, self.responses.columns], names=[self.facet, "item"]
            )
            se = pd.DataFrame(np.nanstd(rater_ests, axis=0), index=mi, columns=range(1, self.max_score + 1))
            setattr(self, f"{prefix}rater_se_{model}", se)

            sev_4d = rater_ests.reshape(
                len(rater_ests),
                self.no_of_facet_elements,
                self.no_of_items,
                self.max_score,
            )
            se_marginal_items = pd.DataFrame(
                np.nanstd(sev_4d.mean(axis=3), axis=0),
                index=self.facet_names,
                columns=self.responses.columns,
            )
            thr_means = sev_4d.mean(axis=2)
            se_marginal_thresholds = pd.DataFrame(
                np.nanstd(thr_means, axis=0), index=self.facet_names, columns=range(1, self.max_score + 1)
            )
            setattr(self, f"{prefix}rater_se_marginal_items", se_marginal_items)
            setattr(
                self, f"{prefix}rater_se_marginal_thresholds", se_marginal_thresholds
            )

            if interval is not None:
                setattr(
                    self,
                    f"{prefix}rater_low_{model}",
                    pd.DataFrame(np.percentile(rater_ests, lo_p, axis=0), index=mi, columns=range(1, self.max_score + 1)),
                )
                setattr(
                    self,
                    f"{prefix}rater_high_{model}",
                    pd.DataFrame(np.percentile(rater_ests, hi_p, axis=0), index=mi, columns=range(1, self.max_score + 1)),
                )

        self._set_facet_aliases(model, anchor=(prefix == "anchor_"))

    # Backwards-compatible aliases
    def std_errors_global(self, anchors=None, **kw):
        """Alias for std_errors(model=\'global\'). See std_errors for full documentation."""
        self.std_errors(model="global", anchors=anchors, **kw)

    def std_errors_items(self, anchors=None, **kw):
        """Alias for std_errors(model=\'items\'). See std_errors for full documentation."""
        self.std_errors(model="items", anchors=anchors, **kw)

    def std_errors_thresholds(self, anchors=None, **kw):
        """Alias for std_errors(model=\'thresholds\'). See std_errors for full documentation."""
        self.std_errors(model="thresholds", anchors=anchors, **kw)

    def std_errors_matrix(self, anchors=None, **kw):
        """Alias for std_errors(model=\'matrix\'). See std_errors for full documentation."""
        self.std_errors(model="matrix", anchors=anchors, **kw)

    def std_errors_bivector(self, anchors=None, **kw):
        """Alias for std_errors(model=\'bivector\'). See std_errors for full documentation."""
        self.std_errors(model="bivector", anchors=anchors, **kw)

    def std_errors_centrality(self, anchors=None, **kw):
        """Alias for std_errors(model=\'centrality\'). See std_errors for full documentation."""
        self.std_errors(model="centrality", anchors=anchors, **kw)

    def std_errors_pseudo_halo(self, anchors=None, **kw):
        """Alias for std_errors(model=\'pseudo_halo\'). See std_errors for full documentation."""
        self.std_errors(model="pseudo_halo", anchors=anchors, **kw)

    def std_errors_bistretch(self, anchors=None, **kw):
        """Alias for std_errors(model=\'bistretch\'). See std_errors for full documentation."""
        self.std_errors(model="bistretch", anchors=anchors, **kw)

    def std_errors_global_anchor(self, anchors, **kw):
        """Alias for std_errors(model=\'global\', anchors=anchors). See std_errors for full documentation."""
        self.std_errors(model="global", anchors=anchors, **kw)

    def anchor_std_errors(
        self,
        model="global",
        anchors=None,
        interval=None,
        no_of_samples=500,
        constant=None,
        threshold_constant=None,
        method=None,
        robust=False,
        matrix_power=None,
        log_lik_tol=0.000001,
        seed=None,
    ):
        """
        Compute bootstrap standard errors for anchor-adjusted parameters.

        If std_errors() was previously called with store_bootstrap=True for
        this model, reuses the stored bootstrap samples — applying
        calibrate_anchor() to each — without resampling. Otherwise reruns
        the full bootstrap.

        interval is inherited from std_errors() if not explicitly provided,
        so anchor CIs are consistent with the unanchored CIs.

        Stores anchor_item_se, anchor_item_low / anchor_item_high (if
        interval is set), anchor_threshold_se_{model},
        anchor_rater_se_{model}, and the corresponding low/high attributes,
        mirroring the naming convention of std_errors().

        Parameters
        ----------
        model : str
            One of 'global', 'items', 'thresholds', 'matrix'.
        anchors : list or None
            Facet elements (e.g. raters) whose mean facet_effect is anchored to zero. If None,
            falls back to anchor_rater_names_{model} set by calibrate_anchor().
        interval : float or None
            If provided, store percentile CIs at this level (e.g. 0.95).
            If None, inherits the interval used in std_errors() for this
            model. Pass interval=0 to explicitly suppress CIs even if
            std_errors() used one.
        no_of_samples : int
            Number of bootstrap samples. Only used when stored samples are
            not available.
        seed : int or None, default None
            Seed for the bootstrap resampling RNG (slow path only -- the
            fast path reuses samples already drawn by std_errors()). Pass
            an int for fully reproducible results; None draws fresh entropy.
        """
        # Inherit interval from std_errors() if not explicitly provided
        model = self._MODEL_ALIASES.get(model, model)
        if interval is None:
            interval = getattr(self, f"_bootstrap_interval_{model}", None)
        if getattr(self, f"se_method_{model}", None) == "sandwich":
            if anchors is None:
                anchors = getattr(self, f"anchor_rater_names_{model}", None)
            if not anchors:
                raise ValueError(
                    f"anchors must be provided, or calibrate_anchor() must have been run "
                    f'for model="{model}".'
                )
            self._rcl_sandwich_std_errors(model, anchors, interval or None)
            return

        stored_flag = getattr(self, f"_bootstrap_stored_{model}", False)
        # 'mixed' has no calibrate_anchor() of its own -- everything is derived
        # from an anchor-calibrated 'matrix' model, restricted per rater via
        # self.anchor_rater_models (see std_errors()'s bootstrap loop for the
        # unanchored case).
        cal_model = "matrix" if model == "mixed" else model

        if stored_flag:
            # Fast path: reuse stored calibrated samples
            samples = getattr(self, f"_bootstrap_samples_{model}")
            anchor_raters_used = getattr(self, f"anchor_rater_names_{cal_model}", anchors)
            if anchor_raters_used is None:
                raise ValueError(
                    f"anchors must be provided, or calibrate_anchor() "
                    f'must have been run for model="{model}" so that '
                    f"anchor_rater_names_{cal_model} is available."
                )
            adj_fixed = self._extract_anchor_adj(cal_model, anchor_raters_used)
            collector = _BootstrapWarnings()
            for s in samples:
                with collector.resample():
                    s.calibrate_anchor(
                        cal_model,
                        anchor_raters_used,
                        constant=constant,
                        threshold_constant=threshold_constant,
                        method=method,
                        robust=robust,
                        matrix_power=matrix_power,
                        log_lik_tol=log_lik_tol,
                        adj=adj_fixed,
                    )
                    if model == "mixed":
                        s.anchor_facet_effects_mixed = s._apply_rater_models(
                            s.anchor_facet_effects_matrix, self.anchor_rater_models, anchor=True
                        )
            collector.emit()
        else:
            # Slow path: full bootstrap rerun
            if anchors is None:
                anchors = getattr(self, f"anchor_rater_names_{cal_model}", None)
            if anchors is None:
                raise ValueError(
                    f"anchors must be provided, or calibrate_anchor() "
                    f'must have been run for model="{model}" so that '
                    f"anchor_rater_names_{cal_model} is available."
                )
            adj_fixed = self._extract_anchor_adj(cal_model, anchors)
            samples = []
            collector = _BootstrapWarnings()
            for s in self._resamples(no_of_samples, seed=seed):
                samples.append(s)
                with collector.resample():
                    s.calibrate(
                        **self._refit_kw(cal_model),
                        model=cal_model,
                        constant=constant,
                        threshold_constant=threshold_constant,
                        method=method,
                        robust=robust,
                        matrix_power=matrix_power,
                        log_lik_tol=log_lik_tol,
                    )
                    s.calibrate_anchor(
                        cal_model,
                        anchors,
                        constant=constant,
                        threshold_constant=threshold_constant,
                        method=method,
                        robust=robust,
                        matrix_power=matrix_power,
                        log_lik_tol=log_lik_tol,
                        adj=adj_fixed,
                    )
                    if model == "mixed":
                        s.anchor_facet_effects_mixed = s._apply_rater_models(
                            s.anchor_facet_effects_matrix, self.anchor_rater_models, anchor=True
                        )
            collector.emit()

        # Item location SEs -- from anchor_items_{cal_model} ('matrix' for 'mixed',
        # which calibrates via matrix); stored per sub-model, unsuffixed = latest
        item_ests = np.array(
            [getattr(s, f"anchor_items_{cal_model}").values for s in samples]
        )
        item_se, item_lo, item_hi = self._se_from_bootstrap(
            item_ests, self.responses.columns, interval
        )
        for suffix in ("", f"_{model}"):
            setattr(self, f"anchor_item_se{suffix}", pd.Series(item_se, index=self.responses.columns))
            if item_lo is not None:
                setattr(self, f"anchor_item_low{suffix}", pd.Series(item_lo, index=self.responses.columns))
                setattr(self, f"anchor_item_high{suffix}", pd.Series(item_hi, index=self.responses.columns))

        # Threshold SEs
        thresh_ests = np.array(
            [getattr(s, f"anchor_thresholds_{cal_model}") for s in samples]
        )
        thr_se, thr_lo, thr_hi = self._se_from_bootstrap(thresh_ests, None, interval)
        setattr(self, f"anchor_threshold_se_{model}", thr_se)
        setattr(self, f"anchor_threshold_low_{model}", thr_lo)
        setattr(self, f"anchor_threshold_high_{model}", thr_hi)

        # Category width SEs
        cat_widths = {
            k + 1: thresh_ests[:, k + 1] - thresh_ests[:, k]
            for k in range(self.max_score - 1)
        }
        setattr(self, f"anchor_cat_width_bootstrap_{model}", cat_widths)
        setattr(
            self,
            f"anchor_cat_width_se_{model}",
            {k: np.nanstd(v) for k, v in cat_widths.items()},
        )
        if interval is not None:
            setattr(
                self,
                f"anchor_cat_width_low_{model}",
                {
                    k: np.percentile(v, 50 * (1 - interval))
                    for k, v in cat_widths.items()
                },
            )
            setattr(
                self,
                f"anchor_cat_width_high_{model}",
                {
                    k: np.percentile(v, 50 * (1 + interval))
                    for k, v in cat_widths.items()
                },
            )

        # Rater SEs
        self._store_rater_se(
            model, samples, anchor=True, interval=interval, prefix="anchor_"
        )

    def anchor_std_errors_global(self, anchors=None, **kw):
        """Alias for anchor_std_errors(model=\'global\'). See anchor_std_errors for full documentation."""
        self.anchor_std_errors(model="global", anchors=anchors, **kw)

    def anchor_std_errors_items(self, anchors=None, **kw):
        """Alias for anchor_std_errors(model=\'items\'). See anchor_std_errors for full documentation."""
        self.anchor_std_errors(model="items", anchors=anchors, **kw)

    def anchor_std_errors_thresholds(self, anchors=None, **kw):
        """Alias for anchor_std_errors(model=\'thresholds\'). See anchor_std_errors for full documentation."""
        self.anchor_std_errors(model="thresholds", anchors=anchors, **kw)

    def anchor_std_errors_matrix(self, anchors=None, **kw):
        """Alias for anchor_std_errors(model=\'matrix\'). See anchor_std_errors for full documentation."""
        self.anchor_std_errors(model="matrix", anchors=anchors, **kw)

    def anchor_std_errors_bivector(self, anchors=None, **kw):
        """Alias for anchor_std_errors(model=\'bivector\'). See anchor_std_errors for full documentation."""
        self.anchor_std_errors(model="bivector", anchors=anchors, **kw)

    # ------------------------------------------------------------------
    # Category probability dictionary
    # ------------------------------------------------------------------
    def anchor_std_errors_centrality(self, anchors=None, **kw):
        """Alias for anchor_std_errors(model=\'centrality\'). See anchor_std_errors for full documentation."""
        self.anchor_std_errors(model="centrality", anchors=anchors, **kw)
    def anchor_std_errors_pseudo_halo(self, anchors=None, **kw):
        """Alias for anchor_std_errors(model=\'pseudo_halo\'). See anchor_std_errors for full documentation."""
        self.anchor_std_errors(model="pseudo_halo", anchors=anchors, **kw)
    def anchor_std_errors_bistretch(self, anchors=None, **kw):
        """Alias for anchor_std_errors(model=\'bistretch\'). See anchor_std_errors for full documentation."""
        self.anchor_std_errors(model="bistretch", anchors=anchors, **kw)

    def category_probability_dict(
        self,
        model="global",
        anchor=False,
        warm_corr=True,
        ext_scores=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        method=None,
        constant=None,
        threshold_constant=None,
        robust=False,
        matrix_power=None,
        log_lik_tol=0.000001,
    ):
        """Build the (Rater, Person) × Items category probability DataFrames."""
        model = self._MODEL_ALIASES.get(model, model)
        item_locations, thresholds, facet_effects = self._get_params(model, anchor)

        if not hasattr(self, f'{"anchor_" if anchor else ""}persons_{model}'):
            self.person_estimates(
                model=model,
                anchor=anchor,
                warm_corr=warm_corr,
                tolerance=tolerance,
                max_iters=max_iters,
                ext_score_adjustment=ext_score_adjustment,
            )
        person_locations = getattr(self, f'{"anchor_" if anchor else ""}persons_{model}')

        person_filter = self.responses.notna().astype(float).replace(0, np.nan)

        if not ext_scores:
            scores = sum(
                person_filter.loc[r].sum(axis=1) * self.max_score
                for r in self.facet_names
            )
            total_scores = sum(
                self.responses.loc[r].sum(axis=1) for r in self.facet_names
            )
            person_locations = person_locations[(total_scores > 0) & (total_scores < scores)]
            person_filter = (
                self.responses.loc[(slice(None), person_locations.index), :]
                .notna()
                .astype(float)
                .replace(0, np.nan)
            )

        probs_dict, cats = self._cat_probs_mfrm(
            person_locations.values,
            list(self.item_names),
            list(self.facet_names),
            thresholds,
            model,
            facet_effects,
        )
        # Convert to per-category (Rater×Person, Items) DataFrames
        cat_prob_dict = {}
        for cat_idx in range(len(cats)):
            frames = {
                facet_element: pd.DataFrame(
                    probs_dict[facet_element][cat_idx, :, :],
                    index=person_locations.index,
                    columns=self.item_names,
                )
                for facet_element in self.facet_names
            }
            df_cat = pd.concat(frames.values(), keys=frames.keys())
            df_cat *= person_filter
            cat_prob_dict[cat_idx] = df_cat

        setattr(self, f"cat_prob_dict_{model}", cat_prob_dict)

    # Backwards-compatible aliases
    def category_probability_dict_global(self, **kw):
        """Alias for category_probability_dict(model=\'global\'). See category_probability_dict for full documentation."""
        self.category_probability_dict(model="global", **kw)

    def category_probability_dict_items(self, **kw):
        """Alias for category_probability_dict(model=\'items\'). See category_probability_dict for full documentation."""
        self.category_probability_dict(model="items", **kw)

    def category_probability_dict_thresholds(self, **kw):
        """Alias for category_probability_dict(model=\'thresholds\'). See category_probability_dict for full documentation."""
        self.category_probability_dict(model="thresholds", **kw)

    def category_probability_dict_matrix(self, **kw):
        """Alias for category_probability_dict(model=\'matrix\'). See category_probability_dict for full documentation."""
        self.category_probability_dict(model="matrix", **kw)

    def category_probability_dict_bivector(self, **kw):
        """Alias for category_probability_dict(model=\'bivector\'). See category_probability_dict for full documentation."""
        self.category_probability_dict(model="bivector", **kw)

    # ------------------------------------------------------------------
    # Person location estimation
    # ------------------------------------------------------------------
    def category_probability_dict_centrality(self, **kw):
        """Alias for category_probability_dict(model=\'centrality\'). See category_probability_dict for full documentation."""
        self.category_probability_dict(model="centrality", **kw)
    def category_probability_dict_pseudo_halo(self, **kw):
        """Alias for category_probability_dict(model=\'pseudo_halo\'). See category_probability_dict for full documentation."""
        self.category_probability_dict(model="pseudo_halo", **kw)
    def category_probability_dict_bistretch(self, **kw):
        """Alias for category_probability_dict(model=\'bistretch\'). See category_probability_dict for full documentation."""
        self.category_probability_dict(model="bistretch", **kw)

    def person(
        self,
        persons,
        model="global",
        anchor=False,
        items=None,
        facet_elements=None,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        missing_as_incorrect=False,
    ):
        """
        Newton-Raphson ML person_location estimation; with warm_corr=True
        (default) returns Warm's (1989) weighted likelihood estimate, solved
        directly and finite for extreme scores (ext_score_adjustment then
        only sets the starting value).

        The key difference between models is how the log-numerator is constructed
        per facet_element — handled entirely by _cat_probs_mfrm() so the NR loop is
        identical across all four parameterisations.
        """
        model = self._MODEL_ALIASES.get(model, model)
        if isinstance(persons, str):
            persons = self.person_names if persons == "all" else [persons]
        if persons is None:
            persons = self.person_names
        if isinstance(items, str):
            items = list(self.item_names) if items == "all" else [items]
        if items is None:
            items = list(self.item_names)
        if facet_elements is None:
            facet_elements = list(self.facet_names)
        elif isinstance(facet_elements, str):
            facet_elements = (
                list(self.facet_names) if facet_elements == "all" else [facet_elements]
            )
        if isinstance(facet_elements, pd.core.indexes.base.Index):
            facet_elements = facet_elements.tolist()

        item_locations, thresholds, facet_effects = self._get_params(model, anchor)
        item_locations = item_locations.loc[items]

        person_data = self.responses.loc[pd.IndexSlice[facet_elements, persons], items]
        person_filter = person_data.notna().astype(float).replace(0, np.nan)

        if missing_as_incorrect:
            # For scoring: NaN stays NaN (sum skipna=True gives 0 contribution — correct)
            # For NR loop: treat all items as observed (full filter)
            nr_filter = person_filter.fillna(1.0)
            ext_scores_vec_val = len(facet_elements) * len(items) * self.max_score
        else:
            nr_filter = person_filter
            ext_scores_vec_val = None  # computed per-person below

        scores = sum(person_data.loc[r].sum(axis=1) for r in facet_elements).astype(
            float
        )

        if missing_as_incorrect:
            ext_scores_vec = pd.Series(
                ext_scores_vec_val, index=scores.index, dtype=float
            )
        else:
            ext_scores_vec = (
                sum(person_filter.loc[r].sum(axis=1) for r in facet_elements)
                * self.max_score
            )

        scores[scores == 0] = ext_score_adjustment
        scores[scores == ext_scores_vec] -= ext_score_adjustment

        item_count = sum(person_filter.loc[r].sum(axis=1) for r in facet_elements)
        mean_diffs = (
            sum(
                (person_filter.loc[r] * item_locations.values).sum(axis=1)
                for r in facet_elements
            )
            / item_count
        )

        try:
            estimates = pd.Series(
                np.log(scores.values)
                - np.log((ext_scores_vec - scores).values)
                + mean_diffs.values,
                index=list(persons),
            )

            active = pd.Series(True, index=list(persons))
            iters = 0

            while active.any() and iters <= max_iters:
                active_idx = estimates.index[active]

                probs_dict, cats = self._cat_probs_mfrm(
                    estimates.loc[active_idx].values,
                    items,
                    facet_elements,
                    thresholds,
                    model,
                    facet_effects,
                )

                # Aggregate expected scores and info across facet_elements
                exp_sum = pd.Series(0.0, index=active_idx)
                info_sum = pd.Series(0.0, index=active_idx)

                for facet_element in facet_elements:
                    probs = probs_dict[facet_element]  # (K+1, N_active, I)
                    pf = nr_filter.loc[facet_element].loc[active_idx].values  # (N, I)

                    exp = (cats[:, None, None] * probs).sum(axis=0) * pf  # (N, I)
                    dev = cats[:, None, None] - exp[None, :, :]
                    inf = (dev**2 * probs).sum(axis=0) * pf  # (N, I)

                    exp_sum += np.nansum(exp, axis=1)
                    info_sum += np.nansum(inf, axis=1)

                changes = ((exp_sum - scores.loc[active_idx]) / info_sum).clip(-1, 1)
                estimates.loc[active_idx] -= changes
                active.loc[active_idx] = abs(changes) > tolerance
                iters += 1

            if iters >= max_iters and active.any():
                n_nc = int(active.sum())
                warnings.warn(
                    f"{n_nc} person(s) did not converge in person(model={model!r}) "
                    f"and will be set to NaN. Consider increasing max_iters.",
                    UserWarning,
                    stacklevel=2,
                )
                estimates[active] = np.nan

            if warm_corr:
                raw = sum(
                    person_data.loc[r].reindex(estimates.index).sum(axis=1)
                    for r in facet_elements
                ).values.astype(float)
                pf_by_facet = {
                    r: nr_filter.loc[r].reindex(estimates.index).values
                    for r in facet_elements
                }
                estimates = pd.Series(
                    self._solve_wle(
                        raw,
                        estimates.values,
                        lambda t, rows: self._wle_moments(
                            t, items, facet_elements, facet_effects, thresholds,
                            {r: pf[rows] for r, pf in pf_by_facet.items()}, model,
                        ),
                        tolerance,
                        max_iters,
                    ),
                    index=estimates.index,
                )

        except Exception as e:
            warnings.warn(
                f"person(model={model!r}) failed with exception: {e}. "
                "Returning NaN for all persons.",
                UserWarning,
                stacklevel=2,
            )
            estimates = pd.Series(np.nan, index=list(persons))

        return estimates

    def person_estimates(
        self,
        model="global",
        anchor=False,
        items=None,
        facet_elements=None,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        missing_as_incorrect=False,
    ):
        """Estimate person_locations for all persons; store as self.persons_{model}."""
        model = self._MODEL_ALIASES.get(model, model)
        estimates = self.person(
            self.person_names,
            model=model,
            anchor=anchor,
            items=items,
            facet_elements=facet_elements,
            warm_corr=warm_corr,
            tolerance=tolerance,
            max_iters=max_iters,
            ext_score_adjustment=ext_score_adjustment,
            missing_as_incorrect=missing_as_incorrect,
        )
        attr = f'{"anchor_" if anchor else ""}persons_{model}'
        setattr(self, attr, estimates)

    def person_estimates_global(
        self, anchor=False, items=None, facet_elements=None, **kw
    ):
        """Alias for person_estimates(model='global'). See person_estimates for full documentation."""
        self.person_estimates(
            model="global",
            anchor=anchor,
            items=items,
            facet_elements=facet_elements,
            **kw,
        )

    def person_estimates_items(
        self, anchor=False, items=None, facet_elements=None, **kw
    ):
        """Alias for person_estimates(model='items'). See person_estimates for full documentation."""
        self.person_estimates(
            model="items",
            anchor=anchor,
            items=items,
            facet_elements=facet_elements,
            **kw,
        )

    def person_estimates_thresholds(
        self, anchor=False, items=None, facet_elements=None, **kw
    ):
        """Alias for person_estimates(model='thresholds'). See person_estimates for full documentation."""
        self.person_estimates(
            model="thresholds",
            anchor=anchor,
            items=items,
            facet_elements=facet_elements,
            **kw,
        )

    def person_estimates_matrix(
        self, anchor=False, items=None, facet_elements=None, **kw
    ):
        """Alias for person_estimates(model='matrix'). See person_estimates for full documentation."""
        self.person_estimates(
            model="matrix",
            anchor=anchor,
            items=items,
            facet_elements=facet_elements,
            **kw,
        )

    def person_estimates_bivector(
        self, anchor=False, items=None, facet_elements=None, **kw
    ):
        """Alias for person_estimates(model='bivector'). See person_estimates for full documentation."""
        self.person_estimates(
            model="bivector",
            anchor=anchor,
            items=items,
            facet_elements=facet_elements,
            **kw,
        )

    # ------------------------------------------------------------------
    # Warm correction
    # ------------------------------------------------------------------
    def person_estimates_centrality(
        self, anchor=False, items=None, facet_elements=None, **kw
    ):
        """Alias for person_estimates(model='centrality'). See person_estimates for full documentation."""
        self.person_estimates(
            model="centrality",
            anchor=anchor,
            items=items,
            facet_elements=facet_elements,
            **kw,
        )
    def person_estimates_pseudo_halo(
        self, anchor=False, items=None, facet_elements=None, **kw
    ):
        """Alias for person_estimates(model='pseudo_halo'). See person_estimates for full documentation."""
        self.person_estimates(
            model="pseudo_halo",
            anchor=anchor,
            items=items,
            facet_elements=facet_elements,
            **kw,
        )
    def person_estimates_bistretch(
        self, anchor=False, items=None, facet_elements=None, **kw
    ):
        """Alias for person_estimates(model='bistretch'). See person_estimates for full documentation."""
        self.person_estimates(
            model="bistretch",
            anchor=anchor,
            items=items,
            facet_elements=facet_elements,
            **kw,
        )

    def _wle_moments(
        self, person_locations, items, facet_elements, facet_effects, thresholds,
        person_filters, model,
    ):
        """
        Expected score, information, and third and fourth cumulants of the
        total score (summed over facet elements (e.g. raters) and items) at the given
        person locations, for _solve_wle().

        person_filters : dict {facet_element: array (N, I)}, 1 = observed,
        0 or NaN = missing.
        """
        probs_dict, cats = self._cat_probs_mfrm(
            np.asarray(person_locations, dtype=float), items, facet_elements,
            thresholds, model, facet_effects,
        )
        totals = [0.0, 0.0, 0.0, 0.0]
        for facet_element in facet_elements:
            parts = self._polytomous_moments(
                probs_dict[facet_element], cats, person_filters[facet_element]
            )
            totals = [t + p for t, p in zip(totals, parts)]
        return tuple(totals)

    def warm(
        self,
        person_locations,
        items,
        facet_elements,
        facet_effects,
        thresholds,
        person_filter,
        model="global",
    ):
        """
        Apply Warm's (1989) weighted maximum likelihood bias correction.

        Computes the MFRM generalisation of the Warm correction, summing over
        all facet_elements and items. The correction is (J1 - J2 + J3) / (2 * I^2)
        where I is total Fisher information and J1, J2, J3 are cubic moment
        terms. Uses the vectorised _cat_probs_mfrm engine.

        Parameters
        ----------
        person_locations : pandas.Series
            Current person_location estimates, indexed by person.
        items : list
            Item subset to use.
        facet_elements : list
            Facet element (e.g. rater) subset to use.
        facet_effects : Series or dict
            Facet element effect parameters (structure depends on model).
        thresholds : array-like
            Rasch-Andrich threshold vector.
        person_filter : pandas.DataFrame
            Binary mask (1.0 = responded, NaN = missing), with (Rater, Person)
            MultiIndex and items as columns.
        model : str, default 'global'
            Facet element parameterisation.

        Returns
        -------
        pandas.Series
            Warm bias correction terms indexed by person, to add to ML estimates.
        """
        model = self._MODEL_ALIASES.get(model, model)
        probs_dict, cats = self._cat_probs_mfrm(
            person_locations.values, items, facet_elements, thresholds, model, facet_effects
        )

        part1 = pd.Series(0.0, index=person_locations.index)
        part2 = pd.Series(0.0, index=person_locations.index)
        part3 = pd.Series(0.0, index=person_locations.index)
        info_sum = pd.Series(0.0, index=person_locations.index)

        for facet_element in facet_elements:
            probs = probs_dict[facet_element]  # (K+1, N, I)
            if isinstance(person_filter.index, pd.MultiIndex):
                pf = person_filter.loc[facet_element].values
            else:
                pf = person_filter.values

            exp = (cats[:, None, None] * probs).sum(axis=0) * pf  # (N, I)
            dev = cats[:, None, None] - exp[None, :, :]
            info = (dev**2 * probs).sum(axis=0) * pf  # (N, I)
            masked_probs = probs * np.where(np.isnan(pf), 0, pf)[None, :, :]

            part1 += np.nansum(
                (cats[:, None, None] ** 3 * masked_probs).sum(axis=0), axis=1
            )
            part2 += 3 * np.nansum((info + exp**2) * exp, axis=1)
            part3 += 2 * np.nansum(exp**3, axis=1)
            info_sum += np.nansum(info, axis=1)

        den = 2 * info_sum**2
        warm_corr = (part1 - part2 + part3) / den
        return pd.Series(warm_corr.values, index=person_locations.index)

    # Backwards-compatible aliases
    def warm_global(self, person_locations, items, facet_elements, facet_effects, pf, **kw):
        """Alias for warm(..., model='global'). See warm for full documentation."""
        return self.warm(
            person_locations, items, facet_elements, facet_effects, self._thresholds_for("global"), pf, "global"
        )

    def warm_items(self, person_locations, items, facet_elements, facet_effects, pf, **kw):
        """Alias for warm(..., model='items'). See warm for full documentation."""
        return self.warm(
            person_locations, items, facet_elements, facet_effects, self._thresholds_for("items"), pf, "items"
        )

    def warm_thresholds(self, person_locations, items, facet_elements, facet_effects, pf, **kw):
        """Alias for warm(..., model='thresholds'). See warm for full documentation."""
        thr = kw.get("thresholds", self._thresholds_for("thresholds"))
        return self.warm(
            person_locations, items, facet_elements, facet_effects, thr, pf, "thresholds"
        )

    def warm_centrality(self, person_locations, items, facet_elements, facet_effects, pf, **kw):
        """Alias for warm(..., model='centrality'). See warm for full documentation."""
        thr = kw.get("thresholds", self._thresholds_for("centrality"))
        return self.warm(
            person_locations, items, facet_elements, facet_effects, thr, pf, "centrality"
        )

    def warm_pseudo_halo(self, person_locations, items, facet_elements, facet_effects, pf, **kw):
        """Alias for warm(..., model='pseudo_halo'). See warm for full documentation."""
        return self.warm(
            person_locations, items, facet_elements, facet_effects, self._thresholds_for("pseudo_halo"), pf, "pseudo_halo"
        )

    def warm_bistretch(self, person_locations, items, facet_elements, facet_effects, pf, **kw):
        """Alias for warm(..., model='bistretch'). See warm for full documentation."""
        return self.warm(
            person_locations, items, facet_elements, facet_effects, self._thresholds_for("bistretch"), pf, "bistretch"
        )

    def warm_matrix(self, person_locations, items, facet_elements, facet_effects, pf, **kw):
        """Alias for warm(..., model='matrix'). See warm for full documentation."""
        return self.warm(
            person_locations, items, facet_elements, facet_effects, self._thresholds_for("matrix"), pf, "matrix"
        )

    def warm_bivector(self, person_locations, items, facet_elements, facet_effects, pf, **kw):
        """Alias for warm(..., model='bivector'). See warm for full documentation."""
        return self.warm(
            person_locations,
            items,
            facet_elements,
            facet_effects,
            self._thresholds_for("bivector"),
            pf,
            "bivector",
        )

    # ------------------------------------------------------------------
    # CSEM
    # ------------------------------------------------------------------

    def csem(
        self,
        model="global",
        anchor=False,
        persons=None,
        person_locations=None,
        items=None,
        facet_elements=None,
    ):
        """
        Compute the conditional standard error of measurement.

        Calculates CSEM = 1 / sqrt(I) where I is total Fisher information
        summed across all observed facet_element-item combinations for each person.
        Uses the vectorised _cat_probs_mfrm engine.

        Parameters
        ----------
        model : str, default 'global'
            Facet element (e.g. rater) parameterisation.
        anchor : bool, default False
            If True, uses anchor-calibrated parameters.
        persons : list, str, or None, default None
            Subset of persons. Overrides person_locations if provided.
            None uses all persons.
        person_locations : pandas.Series, float, list, numpy.ndarray, or None, default None
            Person location estimates. If None, uses stored persons_{model}
            (or anchor_persons_{model} if anchor=True), auto-generated via
            person_estimates() if not already present. A raw float/list/array
            of locations (or any locations not indexed by a real person) is
            treated as hypothetical: since there is no observed response row
            to consult, all items in items are treated as answered by every
            facet_element.
        items : list or None, default None
            Item subset. None uses all items.
        facet_elements : list, str, or None, default None
            Facet element subset. None uses all facet_elements.

        Returns
        -------
        pandas.Series
            CSEM values indexed by person, in logits.
        """
        model = self._MODEL_ALIASES.get(model, model)
        item_locations, thresholds, facet_effects = self._get_params(model, anchor)

        person_locations_supplied = person_locations is not None

        if person_locations is None:
            person_locations = self._get_abils(model, anchor)
        if isinstance(person_locations, (int, float)):
            person_locations = pd.Series({"Location": float(person_locations)})
        if isinstance(person_locations, (list, np.ndarray)):
            person_locations = pd.Series({f"Location {a}": a for a in person_locations})
        if persons is not None:
            if not isinstance(persons, (list, pd.Index, np.ndarray)):
                persons = [persons]
            if person_locations_supplied:
                person_locations = person_locations.loc[persons]
            else:
                person_locations = self._get_abils(model, anchor).loc[persons]

        if items is None:
            items = list(self.item_names)
        if facet_elements is None:
            facet_elements = list(self.facet_names)
        elif not isinstance(facet_elements, (list, pd.Index, np.ndarray)):
            facet_elements = [facet_elements]

        persons = person_locations.index

        # BUG FIX (original): unconditionally indexed self.responses by
        # (facet_elements, persons), which failed for hypothetical locations
        # (raw floats/lists/arrays, or any person_locations index not
        # matching self.responses) with no matching row. Real persons are
        # still filtered by their actual missing-response pattern;
        # hypothetical locations are treated as fully answered by every
        # facet_element.
        is_real_person = persons.isin(self.responses.index.get_level_values(1))
        full_index = pd.MultiIndex.from_product(
            [facet_elements, persons], names=self.responses.index.names
        )
        person_data = self.responses.reindex(full_index)[items]
        person_filter = person_data.notna().astype(float)
        if not is_real_person.all():
            hypothetical_persons = persons[~is_real_person]
            person_filter.loc[(slice(None), hypothetical_persons), :] = 1.0
        person_filter = person_filter.replace(0, np.nan)

        probs_dict, cats = self._cat_probs_mfrm(
            person_locations.values, items, facet_elements, thresholds, model, facet_effects
        )

        info_sum = pd.Series(0.0, index=persons)
        for facet_element in facet_elements:
            probs = probs_dict[facet_element]
            pf = person_filter.loc[facet_element].values
            exp = (cats[:, None, None] * probs).sum(axis=0) * pf
            dev = cats[:, None, None] - exp[None, :, :]
            info = (dev**2 * probs).sum(axis=0) * pf
            info_sum += np.nansum(info, axis=1)

        return 1.0 / (info_sum**0.5)

    # Backwards-compatible aliases
    def csem_global(self, **kw):
        """Alias for csem(model='global'). See csem for full documentation."""
        return self.csem(model="global", **kw)

    def csem_items(self, **kw):
        """Alias for csem(model='items'). See csem for full documentation."""
        return self.csem(model="items", **kw)

    def csem_thresholds(self, **kw):
        """Alias for csem(model='thresholds'). See csem for full documentation."""
        return self.csem(model="thresholds", **kw)

    def csem_matrix(self, **kw):
        """Alias for csem(model='matrix'). See csem for full documentation."""
        return self.csem(model="matrix", **kw)

    def csem_bivector(self, **kw):
        """Alias for csem(model='bivector'). See csem for full documentation."""
        return self.csem(model="bivector", **kw)

    # ------------------------------------------------------------------
    # Score-to-person_location lookup
    # ------------------------------------------------------------------
    def csem_centrality(self, **kw):
        """Alias for csem(model='centrality'). See csem for full documentation."""
        return self.csem(model="centrality", **kw)
    def csem_pseudo_halo(self, **kw):
        """Alias for csem(model='pseudo_halo'). See csem for full documentation."""
        return self.csem(model="pseudo_halo", **kw)
    def csem_bistretch(self, **kw):
        """Alias for csem(model='bistretch'). See csem for full documentation."""
        return self.csem(model="bistretch", **kw)

    def score_lookup(
        self,
        score,
        model="global",
        anchor=False,
        items=None,
        facet_elements=None,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
    ):
        """
        Convert a raw total score to an person_location estimate via Newton-Raphson ML.

        Used internally to draw score lines on TCC plots. Sums expected scores
        and information across all specified facet_element-item combinations using
        scalar exp_score() and variance() methods.

        Parameters
        ----------
        score : int or float
            Raw total score to convert. Extreme scores adjusted by
            ext_score_adjustment.
        model : str, default 'global'
            Facet element (e.g. rater) parameterisation.
        anchor : bool, default False
            If True, uses anchor-calibrated parameters.
        items : list or None, default None
            Item subset. None uses all items.
        facet_elements : list or None, default None
            Facet element subset. None uses all facet_elements.
        warm_corr : bool, default True
            If True, applies Warm's (1989) bias correction.
        tolerance : float, default 0.00001
            Newton-Raphson convergence tolerance.
        max_iters : int, default 100
            Maximum Newton-Raphson iterations.
        ext_score_adjustment : float, default 0.5
            Adjustment applied to extreme scores of 0 or maximum.

        Returns
        -------
        float
            Person location estimate in logits.
        """
        model = self._MODEL_ALIASES.get(model, model)
        item_locations, thresholds, facet_effects = self._get_params(model, anchor)

        if items is None:
            items = list(self.item_names)
        elif isinstance(items, str):
            items = list(self.item_names) if items == "all" else [items]

        if facet_elements is None:
            facet_elements = list(self.facet_names)
        elif isinstance(facet_elements, str):
            facet_elements = (
                list(self.facet_names) if facet_elements == "all" else [facet_elements]
            )

        item_locations = item_locations.loc[items]
        ext_score = len(items) * len(facet_elements) * self.max_score
        used_score = float(score)
        if used_score == 0:
            used_score = ext_score_adjustment
        elif used_score == ext_score:
            used_score -= ext_score_adjustment

        estimate = (
            log(used_score) - log(ext_score - used_score) + float(item_locations.mean())
        )
        change, iters = 1.0, 0

        while abs(change) > tolerance and iters <= max_iters:
            result = sum(
                self.exp_score(
                    estimate,
                    item,
                    item_locations,
                    facet_element,
                    facet_effects,
                    thresholds,
                    model,
                )
                for item in items
                for facet_element in facet_elements
            )
            info = sum(
                self.variance(
                    estimate,
                    item,
                    item_locations,
                    facet_element,
                    facet_effects,
                    thresholds,
                    model,
                )
                for item in items
                for facet_element in facet_elements
            )
            change = max(-1.0, min(1.0, (result - used_score) / info))
            estimate -= change
            iters += 1

        if warm_corr:
            pf_by_facet = {r: np.ones((1, len(items))) for r in facet_elements}
            estimate = float(
                self._solve_wle(
                    [float(score)],
                    [estimate],
                    lambda t, rows: self._wle_moments(
                        t, items, facet_elements, facet_effects, thresholds,
                        {r: pf[rows] for r, pf in pf_by_facet.items()}, model,
                    ),
                    tolerance,
                    max_iters,
                )[0]
            )

        if iters >= max_iters:
            warnings.warn(
                "Maximum iterations reached before convergence in score_lookup(). "
                "Returned estimate may be inaccurate.",
                UserWarning,
                stacklevel=2,
            )
        return estimate

    # Backwards-compatible aliases
    def score_lookup_global(
        self, score, anchor=False, items=None, facet_elements=None, **kw
    ):
        """Alias for score_lookup(..., model='global'). See score_lookup for full documentation."""
        return self.score_lookup(score, "global", anchor, items, facet_elements, **kw)

    def score_lookup_items(
        self, score, anchor=False, items=None, facet_elements=None, **kw
    ):
        """Alias for score_lookup(..., model='items'). See score_lookup for full documentation."""
        return self.score_lookup(score, "items", anchor, items, facet_elements, **kw)

    def score_lookup_thresholds(
        self, score, anchor=False, items=None, facet_elements=None, **kw
    ):
        """Alias for score_lookup(..., model='thresholds'). See score_lookup for full documentation."""
        return self.score_lookup(
            score, "thresholds", anchor, items, facet_elements, **kw
        )

    def score_lookup_matrix(
        self, score, anchor=False, items=None, facet_elements=None, **kw
    ):
        """Alias for score_lookup(..., model='matrix'). See score_lookup for full documentation."""
        return self.score_lookup(score, "matrix", anchor, items, facet_elements, **kw)

    def score_lookup_bivector(
        self, score, anchor=False, items=None, facet_elements=None, **kw
    ):
        """Alias for score_lookup(..., model='bivector'). See score_lookup for full documentation."""
        return self.score_lookup(score, "bivector", anchor, items, facet_elements, **kw)
    def score_lookup_centrality(
        self, score, anchor=False, items=None, facet_elements=None, **kw
    ):
        """Alias for score_lookup(..., model='centrality'). See score_lookup for full documentation."""
        return self.score_lookup(score, "centrality", anchor, items, facet_elements, **kw)
    def score_lookup_pseudo_halo(
        self, score, anchor=False, items=None, facet_elements=None, **kw
    ):
        """Alias for score_lookup(..., model='pseudo_halo'). See score_lookup for full documentation."""
        return self.score_lookup(score, "pseudo_halo", anchor, items, facet_elements, **kw)
    def score_lookup_bistretch(
        self, score, anchor=False, items=None, facet_elements=None, **kw
    ):
        """Alias for score_lookup(..., model='bistretch'). See score_lookup for full documentation."""
        return self.score_lookup(score, "bistretch", anchor, items, facet_elements, **kw)

    def score_lookup_table(
        self,
        model="global",
        anchor=False,
        attribute=True,
        items=None,
        facet_elements=None,
        ext_scores=True,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        filename=None,
        file_format="csv",
    ):
        """
        Build a score-to-person_location lookup table for all possible raw scores.

        Estimates the person_location corresponding to every possible raw score across
        the specified facet_element-item combination using Newton-Raphson, and stores
        the result as self.score_table.

        Parameters
        ----------
        model : str, default 'global'
            Facet element (e.g. rater) parameterisation.
        anchor : bool, default False
            If True, uses anchor-calibrated parameters.
        attribute : bool, default True
            If True, stores result as self.score_table.
        items : list or None, default None
            Item subset. None uses all items.
        facet_elements : list or None, default None
            Facet element subset. None uses all facet_elements.
        ext_scores : bool, default True
            If True, includes extreme scores adjusted by ext_score_adjustment.
        warm_corr : bool, default True
            If True, applies Warm's (1989) bias correction.
        tolerance : float, default 0.00001
            Newton-Raphson convergence tolerance.
        max_iters : int, default 100
            Maximum Newton-Raphson iterations.
        ext_score_adjustment : float, default 0.5
            Adjustment for extreme scores.
        filename : str or None, default None
            If given, also saves the returned table to
            ``{filename}.{file_format}`` (path without extension, as for
            the plotting methods).
        file_format : {'csv', 'xlsx'}, default 'csv'
            File format used when filename is given.

        Attributes set (if attribute=True)
        -----------------------------------
        score_table_{model} : pandas.Series
            Person location estimate for each possible raw score, indexed by score.

        Returns
        -------
        pandas.Series
            The table, also stored as the attribute listed above.
        """
        model = self._MODEL_ALIASES.get(model, model)
        if items is None:
            items = list(self.item_names)
        if facet_elements is None:
            facet_elements = list(self.facet_names)

        ext_score = len(items) * len(facet_elements) * self.max_score
        if ext_scores:
            scores = np.arange(ext_score + 1)
            used_scores = scores.astype(float)
            used_scores[0] += ext_score_adjustment
            used_scores[-1] -= ext_score_adjustment
        else:
            scores = np.arange(1, ext_score)
            used_scores = scores.astype(float)

        table = pd.Series(
            {
                score: self.score_lookup(
                    used_score,
                    model=model,
                    anchor=anchor,
                    items=items,
                    facet_elements=facet_elements,
                    warm_corr=warm_corr,
                    tolerance=tolerance,
                    max_iters=max_iters,
                    ext_score_adjustment=ext_score_adjustment,
                )
                for score, used_score in zip(scores, used_scores)
            }
        )
        if attribute:
            setattr(self, f"score_table_{model}", table)
        _save_table(table, filename, file_format)
        return table

    # Backwards-compatible aliases
    def score_lookup_table_global(self, **kw):
        """Alias for score_lookup_table(model='global'). See score_lookup_table for full documentation."""
        return self.score_lookup_table(model="global", **kw)

    def score_lookup_table_items(self, **kw):
        """Alias for score_lookup_table(model='items'). See score_lookup_table for full documentation."""
        return self.score_lookup_table(model="items", **kw)

    def score_lookup_table_thresholds(self, **kw):
        """Alias for score_lookup_table(model='thresholds'). See score_lookup_table for full documentation."""
        return self.score_lookup_table(model="thresholds", **kw)

    def score_lookup_table_matrix(self, **kw):
        """Alias for score_lookup_table(model='matrix'). See score_lookup_table for full documentation."""
        return self.score_lookup_table(model="matrix", **kw)

    def score_lookup_table_bivector(self, **kw):
        """Alias for score_lookup_table(model='bivector'). See score_lookup_table for full documentation."""
        return self.score_lookup_table(model="bivector", **kw)

    # ------------------------------------------------------------------
    # Category counts
    # ------------------------------------------------------------------
    def score_lookup_table_centrality(self, **kw):
        """Alias for score_lookup_table(model='centrality'). See score_lookup_table for full documentation."""
        return self.score_lookup_table(model="centrality", **kw)
    def score_lookup_table_pseudo_halo(self, **kw):
        """Alias for score_lookup_table(model='pseudo_halo'). See score_lookup_table for full documentation."""
        return self.score_lookup_table(model="pseudo_halo", **kw)
    def score_lookup_table_bistretch(self, **kw):
        """Alias for score_lookup_table(model='bistretch'). See score_lookup_table for full documentation."""
        return self.score_lookup_table(model="bistretch", **kw)

    def category_counts_df(
        self, persons=None, items=None, facet_elements=None, counts_name=None,
        filename=None,
        file_format="csv",
    ):
        """
        Build response frequency tables for one or more persons, across
        one or more items, over one or more facet_elements: one table
        aggregated across the requested facet_elements, and one broken
        down by each individual facet_element within that same subset.

        Computes category counts (0 through max_score), total valid
        responses, and missing responses per item. Both tables get a
        Total row.

        Parameters
        ----------
        persons : str, list, or None, default None
            Person(s) to include (a single person name or a list of
            names) -- e.g. to compare a score-split or exogenous-variable
            group against the rest. None uses all persons.
        items : str, list, or None, default None
            Item(s) to include (as elsewhere in the package, a single item
            name or a list of item names). None uses all items.
        facet_elements : str, list, or None, default None
            Facet_element(s) to include (a single facet_element name or a
            list of names). None uses all facet_elements.
        counts_name : str, int, or None, default None
            If None, stores the aggregated table as
            self.category_counts_table and the per-facet_element
            breakdown as self.category_counts_facet_elements (also
            aliased as self.category_counts_{self.facets}) -- overwriting
            any previous call. If given, stores both instead under
            self.counts (created if it doesn't already exist): the
            aggregated table under self.counts[counts_name] (and
            self.counts.<counts_name> when counts_name is a valid Python
            identifier), the breakdown under
            self.counts[f'{counts_name}_facet_elements'] -- so a
            succession of tables (e.g. one per exogenous-variable group)
            can be kept side by side for comparison rather than
            overwriting each other.
        filename : str or None, default None
            If given, also saves the returned table to
            ``{filename}.{file_format}`` (path without extension, as for
            the plotting methods).
        file_format : {'csv', 'xlsx'}, default 'csv'
            File format used when filename is given.

        Returns
        -------
        pandas.DataFrame
            The aggregated (across the requested facet_elements) table --
            items as rows, response categories plus Total and Missing as
            columns, with a Total row appended, all values integers.
        """
        if items is None:
            items = list(self.item_names)
        elif isinstance(items, str):
            items = [items]

        if facet_elements is None:
            facet_elements = list(self.facet_names)
        elif isinstance(facet_elements, str):
            facet_elements = [facet_elements]

        if persons is None:
            persons = list(self.person_names)
        elif isinstance(persons, str):
            persons = [persons]

        subset = self.responses.loc[pd.IndexSlice[facet_elements, persons], items]

        cat_counts = {
            item: subset[item]
            .value_counts()
            .reindex(range(self.max_score + 1), fill_value=0)
            .astype(int)
            for item in items
        }
        df = pd.DataFrame(cat_counts).T.sort_index(axis=1)
        df["Total"] = subset.count()
        df["Missing"] = subset.shape[0] - df["Total"]
        df.loc["Total"] = df.sum()
        df = df.astype(int)

        # Per-facet_element breakdown, restricted to the same subset.
        rater_counts = {}
        for facet_element in facet_elements:
            element_data = subset.xs(facet_element)
            rater_dict = {
                item: element_data[item]
                .value_counts()
                .reindex(range(self.max_score + 1), fill_value=0)
                .astype(int)
                for item in items
            }
            rdf = pd.DataFrame(rater_dict).T.sort_index(axis=1)
            rdf["Total"] = element_data.count()
            rdf["Missing"] = element_data.shape[0] - rdf["Total"]
            rdf.loc["Total"] = rdf.sum()
            rater_counts[facet_element] = rdf

        facet_elements_df = pd.concat(
            rater_counts.values(), keys=rater_counts.keys()
        ).astype(int)

        if counts_name is None:
            self.category_counts_table = df
            self.category_counts_facet_elements = facet_elements_df
            setattr(self, f"category_counts_{self.facets}", facet_elements_df)
        else:
            if not hasattr(self, "counts"):
                from raschpy.base import _Namespace

                self.counts = _Namespace()
            self.counts[counts_name] = df
            self.counts[f"{counts_name}_facet_elements"] = facet_elements_df

        _save_table(df, filename, file_format)
        return df

    # ------------------------------------------------------------------
    # Fit matrices (shared engine)
    # ------------------------------------------------------------------

    def fit_matrices(self, cat_prob_dict):
        """
        Compute expected scores, info, kurtosis, residuals from cat_prob_dict.
        cat_prob_dict: {cat: (Facet element (e.g. rater)×Person, Items) DataFrame}
        """
        exp_score_df = sum(cat * df for cat, df in cat_prob_dict.items())
        info_df = sum(
            df * (cat - exp_score_df) ** 2 for cat, df in cat_prob_dict.items()
        )
        kurtosis_df = sum(
            df * (cat - exp_score_df) ** 4 for cat, df in cat_prob_dict.items()
        )
        residual_df = self.responses.loc[exp_score_df.index] - exp_score_df
        std_residual_df = residual_df / (info_df**0.5)
        return exp_score_df, info_df, kurtosis_df, residual_df, std_residual_df

    def _ensure_fit_matrices(self, model, **kw):
        """Ensure calibration, person_locations, cat_prob_dict and fit matrices exist."""
        calib_kw = {
            k: v
            for k, v in kw.items()
            if k in ("constant", "threshold_constant", "method", "robust", "matrix_power", "log_lik_tol")
        }
        abil_kw = {
            k: v
            for k, v in kw.items()
            if k in ("warm_corr", "tolerance", "max_iters", "ext_score_adjustment")
        }
        if model == "mixed":
            # 'mixed' has no calibrate() of its own -- guard here too rather than
            # relying on every caller having already run _sync_mixed first (they
            # currently do, but this shouldn't be load-bearing).
            self._sync_mixed(kw.get("anchors", None), seed=kw.get("seed", None), **calib_kw, **abil_kw)
        elif not hasattr(self, f"facet_effects_{model}"):
            self.calibrate(model=model, **calib_kw)
        if not hasattr(self, f"persons_{model}"):
            self.person_estimates(model=model, **abil_kw)
        cpd_attr = f"cat_prob_dict_{model}"
        exp_attr = f"exp_score_df_{model}"
        if not hasattr(self, cpd_attr):
            cpd_kw = {
                k: v
                for k, v in kw.items()
                if k
                in (
                    "warm_corr",
                    "ext_scores",
                    "tolerance",
                    "max_iters",
                    "ext_score_adjustment",
                    "method",
                    "constant",
                    "matrix_power",
                    "log_lik_tol",
                )
            }
            self.category_probability_dict(model=model, **cpd_kw)
        if not hasattr(self, exp_attr):
            cpd = getattr(self, cpd_attr)
            (exp, info, kur, res, std) = self.fit_matrices(cpd)
            setattr(self, f"exp_score_df_{model}", exp)
            setattr(self, f"info_df_{model}", info)
            setattr(self, f"kurtosis_df_{model}", kur)
            setattr(self, f"residual_df_{model}", res)
            setattr(self, f"std_residual_df_{model}", std)

    def fit_matrices_global(self, **kw):
        """Alias for fit_matrices(model='global'). See fit_matrices for full documentation."""
        self._ensure_fit_matrices("global", **kw)

    def fit_matrices_items(self, **kw):
        """Alias for fit_matrices(model='items'). See fit_matrices for full documentation."""
        self._ensure_fit_matrices("items", **kw)

    def fit_matrices_thresholds(self, **kw):
        """Alias for fit_matrices(model='thresholds'). See fit_matrices for full documentation."""
        self._ensure_fit_matrices("thresholds", **kw)

    def fit_matrices_matrix(self, **kw):
        """Alias for fit_matrices(model='matrix'). See fit_matrices for full documentation."""
        self._ensure_fit_matrices("matrix", **kw)

    def fit_matrices_bivector(self, **kw):
        """Alias for fit_matrices(model='bivector'). See fit_matrices for full documentation."""
        self._ensure_fit_matrices("bivector", **kw)

    # ------------------------------------------------------------------
    # Item fit statistics
    # ------------------------------------------------------------------
    def fit_matrices_centrality(self, **kw):
        """Alias for fit_matrices(model='centrality'). See fit_matrices for full documentation."""
        self._ensure_fit_matrices("centrality", **kw)
    def fit_matrices_pseudo_halo(self, **kw):
        """Alias for fit_matrices(model='pseudo_halo'). See fit_matrices for full documentation."""
        self._ensure_fit_matrices("pseudo_halo", **kw)
    def fit_matrices_bistretch(self, **kw):
        """Alias for fit_matrices(model='bistretch'). See fit_matrices for full documentation."""
        self._ensure_fit_matrices("bistretch", **kw)

    def item_fit_statistics(
        self,
        exp_score_df,
        info_df,
        kurtosis_df,
        residual_df,
        std_residual_df,
        person_locations,
    ):
        """Shared item fit statistics computation."""
        scores = self.responses.sum(axis=1)
        max_scores = self.responses.count(axis=1) * self.max_score
        item_count = self.responses[(scores > 0) & (scores < max_scores)].count(axis=0)
        self.response_counts = self.responses.count(axis=0)
        self.item_facilities = self.responses.mean(axis=0) / self.max_score

        item_outfit_ms = (std_residual_df**2).mean()
        item_outfit_zstd = ((item_outfit_ms ** (1 / 3)) - 1 + 2 / (9 * item_count)) / (
            2 / (9 * item_count)
        ) ** 0.5

        item_infit_ms = (residual_df**2).sum() / info_df.sum()
        item_infit_zstd = ((item_infit_ms ** (1 / 3)) - 1 + 2 / (9 * item_count)) / (
            2 / (9 * item_count)
        ) ** 0.5

        # Expand person_locations to (Rater×Person) MultiIndex
        estimates_by_rater = pd.concat(
            {facet_element: person_locations for facet_element in self.facet_names},
            keys=self.facet_names,
        )
        estimates_by_rater.index.names = self.responses.index.names
        pm, exp_pm = self.pt_meas(estimates_by_rater, exp_score_df, info_df)

        return (
            item_outfit_ms,
            item_outfit_zstd,
            item_infit_ms,
            item_infit_zstd,
            pm,
            exp_pm,
        )

    def _run_item_fit(self, model, **kw):
        """Internal dispatcher: ensure fit matrices then run item fit statistics for the given model."""
        self._ensure_fit_matrices(model, **kw)
        person_locations = getattr(self, f"persons_{model}")
        (outfit_ms, outfit_z, infit_ms, infit_z, pm, exp_pm) = self.item_fit_statistics(
            getattr(self, f"exp_score_df_{model}"),
            getattr(self, f"info_df_{model}"),
            getattr(self, f"kurtosis_df_{model}"),
            getattr(self, f"residual_df_{model}"),
            getattr(self, f"std_residual_df_{model}"),
            person_locations,
        )
        setattr(self, f"item_outfit_ms_{model}", outfit_ms)
        setattr(self, f"item_outfit_zstd_{model}", outfit_z)
        setattr(self, f"item_infit_ms_{model}", infit_ms)
        setattr(self, f"item_infit_zstd_{model}", infit_z)
        setattr(self, f"point_measure_{model}", pm)
        setattr(self, f"exp_point_measure_{model}", exp_pm)

    def item_fit_statistics_global(self, **kw):
        """Alias for item_fit_statistics(model='global'). See item_fit_statistics for full documentation."""
        self._run_item_fit("global", **kw)

    def item_fit_statistics_items(self, **kw):
        """Alias for item_fit_statistics(model='items'). See item_fit_statistics for full documentation."""
        self._run_item_fit("items", **kw)

    def item_fit_statistics_thresholds(self, **kw):
        """Alias for item_fit_statistics(model='thresholds'). See item_fit_statistics for full documentation."""
        self._run_item_fit("thresholds", **kw)

    def item_fit_statistics_matrix(self, **kw):
        """Alias for item_fit_statistics(model='matrix'). See item_fit_statistics for full documentation."""
        self._run_item_fit("matrix", **kw)

    def item_fit_statistics_bivector(self, **kw):
        """Alias for item_fit_statistics(model='bivector'). See item_fit_statistics for full documentation."""
        self._run_item_fit("bivector", **kw)

    # ------------------------------------------------------------------
    # Threshold fit statistics
    # ------------------------------------------------------------------
    def item_fit_statistics_centrality(self, **kw):
        """Alias for item_fit_statistics(model='centrality'). See item_fit_statistics for full documentation."""
        self._run_item_fit("centrality", **kw)
    def item_fit_statistics_pseudo_halo(self, **kw):
        """Alias for item_fit_statistics(model='pseudo_halo'). See item_fit_statistics for full documentation."""
        self._run_item_fit("pseudo_halo", **kw)
    def item_fit_statistics_bistretch(self, **kw):
        """Alias for item_fit_statistics(model='bistretch'). See item_fit_statistics for full documentation."""
        self._run_item_fit("bistretch", **kw)

    def threshold_fit_statistics(self, person_locations, diff_df_dict):
        """Shared threshold fit statistics (dichotomised ICC approach).
        Mirrors RSM threshold_fit_statistics but with (Rater, Person) MultiIndex
        and nz filter for extreme total scores.
        """
        # Build (Rater×Person, Items) person_location DataFrame
        basic_persons_df = pd.DataFrame(
            [
                [person_locations[person] for _ in self.responses.columns]
                for person in self.person_names
            ],
            index=self.person_names,
            columns=self.responses.columns,
        )
        abil_df = pd.concat(
            [basic_persons_df] * self.no_of_facet_elements, keys=list(self.facet_names)
        )
        abil_df.index.names = self.responses.index.names

        scores = self.responses.sum(axis=1)
        max_scores = self.responses.count(axis=1) * self.max_score
        nz = (scores > 0) & (scores < max_scores)

        dich = {}
        for t in range(self.max_score):
            d = self.responses.where(self.responses.isin([t, t + 1]), np.nan) - t
            d.index.names = self.responses.index.names
            dich[t + 1] = d

        # Count non-missing in raw dich (before nz) — matches RSM
        dich_cnt = {
            t + 1: dich[t + 1].notna().sum().sum() for t in range(self.max_score)
        }

        dich_exp = {}
        dich_var = {}
        dich_kur = {}
        dich_res = {}
        dich_std = {}

        for t in range(self.max_score):
            mm = (dich[t + 1] + 1) / (dich[t + 1] + 1)
            mm = mm.loc[nz]
            mm.index.names = self.responses.index.names

            p = 1.0 / (1.0 + np.exp(diff_df_dict[t + 1] - abil_df))
            p = p.loc[nz]
            p.index.names = self.responses.index.names
            p = p * mm

            v = p * (1 - p) * mm
            k = (((-p) ** 4) * (1 - p) + ((1 - p) ** 4) * p) * mm

            dich_exp[t + 1] = p
            dich_var[t + 1] = v
            dich_kur[t + 1] = k

            d_t = dich[t + 1].loc[nz]
            d_t.index.names = self.responses.index.names
            dich_res[t + 1] = d_t - p
            dich_std[t + 1] = dich_res[t + 1] / (v**0.5)

        def _series(fn):
            """Build a Series indexed by threshold number (1..max_score) from a per-threshold function."""
            return pd.Series({t + 1: fn(t) for t in range(self.max_score)})

        # Outfit MS: sum(std_res²) / count of valid dich responses (matching RSM)
        outfit_ms = _series(
            lambda t: (
                (dich_std[t + 1] ** 2).sum().sum() / dich[t + 1].loc[nz].count().sum()
            )
        )
        infit_ms = _series(
            lambda t: (
                (dich_res[t + 1] ** 2).sum().sum() / dich_var[t + 1].sum().sum()
                if dich_var[t + 1].sum().sum() > 0
                else np.nan
            )
        )

        outfit_q = (
            _series(
                lambda t: (
                    (dich_kur[t + 1] / dich_var[t + 1] ** 2).sum().sum()
                    / dich_cnt[t + 1] ** 2
                    - 1 / dich_cnt[t + 1]
                )
            )
            ** 0.5
        )
        infit_q = (
            _series(
                lambda t: (
                    (dich_kur[t + 1] - dich_var[t + 1] ** 2).sum().sum()
                    / dich_var[t + 1].sum().sum() ** 2
                )
            )
            ** 0.5
        )

        outfit_z = (outfit_ms ** (1 / 3) - 1) * (3 / outfit_q) + outfit_q / 3
        infit_z = (infit_ms ** (1 / 3) - 1) * (3 / infit_q) + infit_q / 3

        # Point-measure correlations
        abil_dev = pd.concat(
            [person_locations.loc[self.person_names] - person_locations.loc[self.person_names].mean()]
            * self.no_of_facet_elements,
            keys=list(self.facet_names),
        ).loc[nz]
        abil_dev.index.names = self.responses.index.names

        fac = {t + 1: dich[t + 1].loc[nz].mean() for t in range(self.max_score)}

        pm_num = _series(
            lambda t: (
                (dich[t + 1].loc[nz] - fac[t + 1])
                .mul(abil_dev.values, axis=0)
                .sum()
                .sum()
            )
        )
        pm_den = _series(
            lambda t: (
                ((dich[t + 1].loc[nz] - fac[t + 1]) ** 2).sum().sum()
                * float((abil_dev**2).sum())
            )
            ** 0.5
        )
        thresh_pm = pm_num / pm_den

        exp_pm_c = {
            t + 1: dich_exp[t + 1] - dich_exp[t + 1].mean()
            for t in range(self.max_score)
        }
        exp_pm_num = _series(
            lambda t: exp_pm_c[t + 1].mul(abil_dev.values, axis=0).sum().sum()
        )
        exp_pm_den = _series(
            lambda t: (
                ((exp_pm_c[t + 1] ** 2) + dich_var[t + 1]).sum().sum()
                * float((abil_dev**2).sum())
            )
            ** 0.5
        )
        thresh_exp_pm = exp_pm_num / exp_pm_den

        # Discrimination
        diff_dev = {}
        for t in range(self.max_score):
            dd = abil_df - diff_df_dict[t + 1]
            dd = dd.loc[nz]
            dd.index.names = self.responses.index.names
            diff_dev[t + 1] = dd

        disc_num = _series(lambda t: (diff_dev[t + 1] * dich_res[t + 1]).sum().sum())
        disc_den = _series(
            lambda t: (dich_var[t + 1] * diff_dev[t + 1] ** 2).sum().sum()
        )
        discrimination = 1 + disc_num / disc_den

        return (
            outfit_ms,
            outfit_z,
            infit_ms,
            infit_z,
            thresh_pm,
            thresh_exp_pm,
            discrimination,
        )

    def _diff_df_dict(self, model, item_locations, thresholds, facet_effects):
        """Build the threshold location DataFrame dict for threshold fit stats."""
        diff_df_dict = {}
        for t in range(self.max_score):
            thr_loc = thresholds[t + 1]
            rows = {}
            for facet_element in self.facet_names:
                if model == "global":
                    row = item_locations + thr_loc + float(facet_effects.loc[facet_element])
                elif model in ("items", "pseudo_halo"):
                    row = item_locations + thr_loc + facet_effects.loc[facet_element]
                elif model in ("thresholds", "centrality"):
                    row = item_locations + thr_loc + facet_effects.loc[facet_element, t + 1]
                elif model in ("bivector", "matrix", "mixed", "bistretch"):
                    row = (
                        item_locations
                        + thr_loc
                        + pd.Series(
                            facet_effects.loc[facet_element].iloc[:, t].values,
                            index=self.responses.columns,
                        )
                    )
                rows[facet_element] = pd.DataFrame(
                    np.tile(row.values[None, :], (self.no_of_persons, 1)),
                    index=self.person_names,
                    columns=self.responses.columns,
                )
            df_t = pd.concat(list(rows.values()), keys=list(rows.keys()))
            df_t.index.names = self.responses.index.names
            diff_df_dict[t + 1] = df_t
        return diff_df_dict

    def _run_threshold_fit(self, model, anchors=None, **kw):
        """Internal dispatcher: run threshold fit statistics for the given model."""
        if not hasattr(self, f"persons_{model}"):
            self.person_estimates(model=model)
        # Always use unanchored params for fit statistics — anchor is origin shift only
        item_locations, thresholds, facet_effects = self._get_params(model, anchor=False)
        person_locations = getattr(self, f"persons_{model}")
        ddd = self._diff_df_dict(model, item_locations, thresholds, facet_effects)
        results = self.threshold_fit_statistics(person_locations, ddd)
        names = [
            "threshold_outfit_ms",
            "threshold_outfit_zstd",
            "threshold_infit_ms",
            "threshold_infit_zstd",
            "threshold_point_measure",
            "threshold_exp_point_measure",
            "threshold_discrimination",
        ]
        for name, val in zip(names, results):
            setattr(self, f"{name}_{model}", val)

    def threshold_fit_statistics_global(self, **kw):
        """Alias for threshold_fit_statistics(model='global'). See threshold_fit_statistics for full documentation."""
        self._run_threshold_fit("global", **kw)

    def threshold_fit_statistics_items(self, **kw):
        """Alias for threshold_fit_statistics(model='items'). See threshold_fit_statistics for full documentation."""
        self._run_threshold_fit("items", **kw)

    def threshold_fit_statistics_thresholds(self, **kw):
        """Alias for threshold_fit_statistics(model='thresholds'). See threshold_fit_statistics for full documentation."""
        self._run_threshold_fit("thresholds", **kw)

    def threshold_fit_statistics_matrix(self, **kw):
        """Alias for threshold_fit_statistics(model='matrix'). See threshold_fit_statistics for full documentation."""
        self._run_threshold_fit("matrix", **kw)

    def threshold_fit_statistics_bivector(self, **kw):
        """Alias for threshold_fit_statistics(model='bivector'). See threshold_fit_statistics for full documentation."""
        self._run_threshold_fit("bivector", **kw)

    # ------------------------------------------------------------------
    # Rater fit statistics
    # ------------------------------------------------------------------
    def threshold_fit_statistics_centrality(self, **kw):
        """Alias for threshold_fit_statistics(model='centrality'). See threshold_fit_statistics for full documentation."""
        self._run_threshold_fit("centrality", **kw)
    def threshold_fit_statistics_pseudo_halo(self, **kw):
        """Alias for threshold_fit_statistics(model='pseudo_halo'). See threshold_fit_statistics for full documentation."""
        self._run_threshold_fit("pseudo_halo", **kw)
    def threshold_fit_statistics_bistretch(self, **kw):
        """Alias for threshold_fit_statistics(model='bistretch'). See threshold_fit_statistics for full documentation."""
        self._run_threshold_fit("bistretch", **kw)

    def facet_pivot(self, df):
        """Pivot (Facet element (e.g. rater)×Person, Items) DataFrame to (Person×Items, Facet elements)."""
        return pd.DataFrame(
            {
                facet_element: df.xs(facet_element).T.stack()
                for facet_element in self.facet_names
            }
        )

    def facet_fit_statistics(self, info_df, kurtosis_df, residual_df, std_residual_df):
        """Shared facet_element fit statistics."""
        scores = self.responses.sum(axis=1)
        max_scores = self.responses.count(axis=1) * self.max_score
        rater_count = pd.Series(
            {
                facet_element: self.responses[(scores > 0) & (scores < max_scores)]
                .xs(facet_element)
                .count()
                .sum()
                for facet_element in self.facet_names
            }
        )

        rater_outfit_ms = pd.Series(
            {
                facet_element: (
                    (std_residual_df**2).xs(facet_element).sum().sum()
                    / (std_residual_df**2).xs(facet_element).count().sum()
                )
                for facet_element in self.facet_names
            }
        )
        rater_infit_ms = pd.Series(
            {
                facet_element: (
                    (residual_df**2).xs(facet_element).sum().sum()
                    / info_df.xs(facet_element).sum().sum()
                )
                for facet_element in self.facet_names
            }
        )

        rater_outfit_q = (
            (self.facet_pivot(kurtosis_df) / (self.facet_pivot(info_df) ** 2))
            / (rater_count**2)
        ).sum() - 1 / rater_count
        rater_outfit_q = rater_outfit_q**0.5

        rater_outfit_zstd = ((rater_outfit_ms ** (1 / 3)) - 1) * (
            3 / rater_outfit_q
        ) + rater_outfit_q / 3

        rater_infit_q = (
            (self.facet_pivot(kurtosis_df) - self.facet_pivot(info_df) ** 2).sum()
            / (self.facet_pivot(info_df).sum() ** 2)
        ) ** 0.5
        rater_infit_zstd = ((rater_infit_ms ** (1 / 3)) - 1) * (
            3 / rater_infit_q
        ) + rater_infit_q / 3

        return rater_outfit_ms, rater_outfit_zstd, rater_infit_ms, rater_infit_zstd

    def _run_facet_fit(self, model, **kw):
        """Internal dispatcher: run facet/rater fit statistics for the given model."""
        self._ensure_fit_matrices(model, **kw)
        results = self.facet_fit_statistics(
            getattr(self, f"info_df_{model}"),
            getattr(self, f"kurtosis_df_{model}"),
            getattr(self, f"residual_df_{model}"),
            getattr(self, f"std_residual_df_{model}"),
        )
        for name, val in zip(
            [
                "rater_outfit_ms",
                "rater_outfit_zstd",
                "rater_infit_ms",
                "rater_infit_zstd",
            ],
            results,
        ):
            setattr(self, f"{name}_{model}", val)
        self._set_facet_aliases(model)

    def facet_fit_statistics_global(self, **kw):
        """Alias for facet_fit_statistics(model='global'). See facet_fit_statistics for full documentation."""
        self._run_facet_fit("global", **kw)

    def facet_fit_statistics_items(self, **kw):
        """Alias for facet_fit_statistics(model='items'). See facet_fit_statistics for full documentation."""
        self._run_facet_fit("items", **kw)

    def facet_fit_statistics_thresholds(self, **kw):
        """Alias for facet_fit_statistics(model='thresholds'). See facet_fit_statistics for full documentation."""
        self._run_facet_fit("thresholds", **kw)

    def facet_fit_statistics_matrix(self, **kw):
        """Alias for facet_fit_statistics(model='matrix'). See facet_fit_statistics for full documentation."""
        self._run_facet_fit("matrix", **kw)

    def facet_fit_statistics_bivector(self, **kw):
        """Alias for facet_fit_statistics(model='bivector'). See facet_fit_statistics for full documentation."""
        self._run_facet_fit("bivector", **kw)

    # ------------------------------------------------------------------
    # Person fit statistics
    # ------------------------------------------------------------------
    def facet_fit_statistics_centrality(self, **kw):
        """Alias for facet_fit_statistics(model='centrality'). See facet_fit_statistics for full documentation."""
        self._run_facet_fit("centrality", **kw)
    def facet_fit_statistics_pseudo_halo(self, **kw):
        """Alias for facet_fit_statistics(model='pseudo_halo'). See facet_fit_statistics for full documentation."""
        self._run_facet_fit("pseudo_halo", **kw)
    def facet_fit_statistics_bistretch(self, **kw):
        """Alias for facet_fit_statistics(model='bistretch'). See facet_fit_statistics for full documentation."""
        self._run_facet_fit("bistretch", **kw)

    def person_fit_statistics(
        self, info_df, kurtosis_df, residual_df, std_residual_df, person_locations, **kw
    ):
        """Shared person fit statistics."""
        csems = 1.0 / (info_df.unstack(level=0).sum(axis=1) ** 0.5)
        rsems = (
            (residual_df.unstack(level=0) ** 2).sum(axis=1)
        ) ** 0.5 / info_df.unstack(level=0).sum(axis=1)

        person_outfit_ms = (std_residual_df.unstack(level=0) ** 2).mean(axis=1)
        person_infit_ms = (residual_df.unstack(level=0) ** 2).sum(
            axis=1
        ) / info_df.unstack(level=0).sum(axis=1)

        scores = self.responses.sum(axis=1)
        max_scores = self.responses.count(axis=1) * self.max_score
        person_count = (
            self.responses[(scores > 0) & (scores < max_scores)]
            .unstack(level=0)
            .notna()
            .sum(axis=1)
        )

        base_df = kurtosis_df.unstack(level=0) / (info_df.unstack(level=0) ** 2)
        # Sum kurtosis/info² per person, divide by person_count²
        # Avoid the fragile transpose trick — align directly on person index
        base_df = base_df.loc[person_count.index]
        outfit_q_sq = (base_df.sum(axis=1) / (person_count**2)) - (1 / person_count)
        person_outfit_q = np.where(outfit_q_sq >= 0, outfit_q_sq**0.5, np.nan)
        person_outfit_q = pd.Series(person_outfit_q, index=person_count.index)
        person_outfit_zstd = ((person_outfit_ms ** (1 / 3)) - 1) * (
            3 / person_outfit_q
        ) + person_outfit_q / 3
        person_outfit_zstd = person_outfit_zstd[: self.no_of_persons].astype(float)

        infit_q_sq = (kurtosis_df.unstack(level=0) - info_df.unstack(level=0) ** 2).sum(
            axis=1
        ) / (info_df.unstack(level=0).sum(axis=1) ** 2)
        person_infit_q = np.where(infit_q_sq >= 0, infit_q_sq**0.5, np.nan)
        person_infit_q = pd.Series(person_infit_q, index=infit_q_sq.index)
        person_infit_zstd = (
            ((person_infit_ms ** (1 / 3)) - 1) * (3 / person_infit_q)
            + person_infit_q / 3
        ).astype(float)

        return (
            csems,
            rsems,
            person_outfit_ms,
            person_outfit_zstd,
            person_infit_ms,
            person_infit_zstd,
        )

    def _run_person_fit(self, model, **kw):
        """Internal dispatcher: run person fit statistics for the given model."""
        self._ensure_fit_matrices(model, **kw)
        person_locations = getattr(self, f"persons_{model}")
        results = self.person_fit_statistics(
            getattr(self, f"info_df_{model}"),
            getattr(self, f"kurtosis_df_{model}"),
            getattr(self, f"residual_df_{model}"),
            getattr(self, f"std_residual_df_{model}"),
            person_locations,
        )
        names = [
            "csem_vector",
            "rsem_vector",
            "person_outfit_ms",
            "person_outfit_zstd",
            "person_infit_ms",
            "person_infit_zstd",
        ]
        for name, val in zip(names, results):
            if isinstance(val, pd.Series):
                val = pd.to_numeric(val, errors="coerce")
            setattr(self, f"{name}_{model}", val)

    def person_fit_statistics_global(self, **kw):
        """Alias for person_fit_statistics(model='global'). See person_fit_statistics for full documentation."""
        self._run_person_fit("global", **kw)

    def person_fit_statistics_items(self, **kw):
        """Alias for person_fit_statistics(model='items'). See person_fit_statistics for full documentation."""
        self._run_person_fit("items", **kw)

    def person_fit_statistics_thresholds(self, **kw):
        """Alias for person_fit_statistics(model='thresholds'). See person_fit_statistics for full documentation."""
        self._run_person_fit("thresholds", **kw)

    def person_fit_statistics_matrix(self, **kw):
        """Alias for person_fit_statistics(model='matrix'). See person_fit_statistics for full documentation."""
        self._run_person_fit("matrix", **kw)

    def person_fit_statistics_bivector(self, **kw):
        """Alias for person_fit_statistics(model='bivector'). See person_fit_statistics for full documentation."""
        self._run_person_fit("bivector", **kw)

    # ------------------------------------------------------------------
    # Test-level fit statistics
    # ------------------------------------------------------------------
    def person_fit_statistics_centrality(self, **kw):
        """Alias for person_fit_statistics(model='centrality'). See person_fit_statistics for full documentation."""
        self._run_person_fit("centrality", **kw)
    def person_fit_statistics_pseudo_halo(self, **kw):
        """Alias for person_fit_statistics(model='pseudo_halo'). See person_fit_statistics for full documentation."""
        self._run_person_fit("pseudo_halo", **kw)
    def person_fit_statistics_bistretch(self, **kw):
        """Alias for person_fit_statistics(model='bistretch'). See person_fit_statistics for full documentation."""
        self._run_person_fit("bistretch", **kw)

    def test_fit_statistics(self, person_locations, rsems, item_se=None, model=None):
        """Shared test-level separation and reliability statistics. model
        selects the sub-model's own item locations and standard errors
        (default: the latest calibration)."""
        if item_se is None:
            item_se = self.item_se if model is None else self._item_se_for(model)
        item_locations = self.items if model is None else self._items_for(model)

        scores = self.responses.unstack(level=0).sum(axis=1)
        max_scores = self.responses.unstack(level=0).count(axis=1) * self.max_score
        person_locations = person_locations[(scores > 0) & (scores < max_scores)]

        isi = (item_locations.var() / (item_se**2).mean() - 1) ** 0.5
        item_strata = (4 * isi + 1) / 3
        item_reliability = isi**2 / (1 + isi**2)

        mean_rsem2 = (rsems**2).mean()
        psi = ((np.var(person_locations) - mean_rsem2) / mean_rsem2) ** 0.5
        person_strata = (4 * psi + 1) / 3
        person_reliability = psi**2 / (1 + psi**2)

        return (
            isi,
            item_strata,
            item_reliability,
            psi,
            person_strata,
            person_reliability,
        )

    def _run_test_fit(self, model, **kw):
        """Internal dispatcher: run test-level separation statistics for the given model."""
        if not hasattr(self, f"csem_vector_{model}"):
            self._run_person_fit(model, **kw)
        if not hasattr(self, f"threshold_se_{model}"):
            self.std_errors(model=model, **kw)
        person_locations = getattr(self, f"persons_{model}")
        rsems = getattr(self, f"rsem_vector_{model}")
        results = self.test_fit_statistics(person_locations, rsems, model=model)
        for name, val in zip(
            [
                "isi",
                "item_strata",
                "item_reliability",
                "psi",
                "person_strata",
                "person_reliability",
            ],
            results,
        ):
            setattr(self, f"{name}_{model}", val)

    def test_fit_statistics_global(self, **kw):
        """Alias for test_fit_statistics(model='global'). See test_fit_statistics for full documentation."""
        self._run_test_fit("global", **kw)

    def test_fit_statistics_items(self, **kw):
        """Alias for test_fit_statistics(model='items'). See test_fit_statistics for full documentation."""
        self._run_test_fit("items", **kw)

    def test_fit_statistics_thresholds(self, **kw):
        """Alias for test_fit_statistics(model='thresholds'). See test_fit_statistics for full documentation."""
        self._run_test_fit("thresholds", **kw)

    def test_fit_statistics_matrix(self, **kw):
        """Alias for test_fit_statistics(model='matrix'). See test_fit_statistics for full documentation."""
        self._run_test_fit("matrix", **kw)

    def test_fit_statistics_bivector(self, **kw):
        """Alias for test_fit_statistics(model='bivector'). See test_fit_statistics for full documentation."""
        self._run_test_fit("bivector", **kw)

    # ------------------------------------------------------------------
    # Top-level fit_statistics
    # ------------------------------------------------------------------
    def test_fit_statistics_centrality(self, **kw):
        """Alias for test_fit_statistics(model='centrality'). See test_fit_statistics for full documentation."""
        self._run_test_fit("centrality", **kw)
    def test_fit_statistics_pseudo_halo(self, **kw):
        """Alias for test_fit_statistics(model='pseudo_halo'). See test_fit_statistics for full documentation."""
        self._run_test_fit("pseudo_halo", **kw)
    def test_fit_statistics_bistretch(self, **kw):
        """Alias for test_fit_statistics(model='bistretch'). See test_fit_statistics for full documentation."""
        self._run_test_fit("bistretch", **kw)

    def _log_likelihood(
        self, model="global", responses=None, anchor=False, persons=None,
        items=None, thresholds=None, facet_effects=None,
    ):
        """
        items, thresholds, facet_effects : optional overrides, default None
            Evaluate this facet_element's/model's log-likelihood against
            EXTERNALLY supplied parameter values instead of self's own
            current calibration (self.items / self.thresholds /
            self.facet_effects_{model}) -- the same "plug in a fixed
            value instead of this object's own free estimate" pattern
            `persons=` already uses for pooled-abilities DIF/DRF omnibus
            tests. Lets a caller hold one or two parameter families fixed
            (e.g. at a pooled/combined-group model's own values) while
            evaluating this object's own value for the remaining one(s) --
            the mechanism behind axis-isolated DIF/DRF omnibus tests (pin
            facet_effects to test items in isolation, or pin items+
            thresholds to test facet_effects in isolation). None (default,
            for all three) reproduces the previous, unisolated behaviour
            exactly.
        """
        if responses is None:
            responses = self.responses
        _items, _thresholds, _facet_effects = self._get_params(model, anchor=anchor)
        if items is None:
            items = _items
        if thresholds is None:
            thresholds = _thresholds
        if facet_effects is None:
            facet_effects = _facet_effects
        total_scores = responses.groupby(level=1).sum().sum(axis=1)
        max_possible = responses.notna().groupby(level=1).sum().sum(axis=1) * self.max_score
        non_extreme = total_scores.index[(total_scores > 0) & (total_scores < max_possible)]
        if persons is None:
            _pfx = "anchor_" if anchor else ""
            persons = getattr(self, f"{_pfx}persons_{model}")
        persons = persons.reindex(non_extreme).dropna()
        persons = persons[persons.abs() <= 20]
        facet_names = responses.index.get_level_values(0).unique()
        probs_dict, cats = self._cat_probs_mfrm(
            persons.values,
            list(self.item_names),
            list(facet_names),
            thresholds,
            model,
            facet_effects,
            item_locations=items,
        )
        ll = 0.0
        for facet_element in facet_names:
            probs = probs_dict[facet_element]  # (K+1, N, I)
            obs_arr = responses.loc[facet_element].reindex(persons.index).values  # (N, I)
            valid = ~np.isnan(obs_arr)
            obs_int = np.where(valid, obs_arr, 0).astype(int)
            n_idx, i_idx = np.meshgrid(
                np.arange(obs_arr.shape[0]), np.arange(obs_arr.shape[1]), indexing="ij"
            )
            prob_obs = probs[obs_int, n_idx, i_idx]
            prob_obs[~valid] = np.nan
            ll += float(np.nansum(np.log(prob_obs)))
        return ll

    def fit_statistics(
        self,
        model="global",
        anchors=None,
        warm_corr=True,
        se=True,
        test_stats=True,
        ext_scores=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        method=None,
        constant=None,
        threshold_constant=None,
        robust=False,
        matrix_power=None,
        log_lik_tol=0.000001,
        no_of_samples=500,
        interval=None,
        seed=None,
    ):
        """
        Compute all item, threshold, facet_element, person, and test-level fit statistics.

        Top-level orchestrator that auto-triggers calibrate(), std_errors(),
        person_abils(), and category_probability_dict() as needed, then runs
        all fit statistic sub-routines for the specified model. Stores all
        results as model-suffixed attributes.

        Parameters
        ----------
        model : str, default 'global'
            Facet element (e.g. rater) parameterisation: 'global', 'items', 'thresholds', or 'matrix'.
        anchors : list or None, default None
            Facet element identifiers to treat as anchors for SE computation.
        warm_corr : bool, default True
            Warm bias correction for person_location estimates.
        se : bool, default True
            If True, computes bootstrap SEs. Required for test-level stats.
        test_stats : bool, default True
            If True, computes ISI, PSI, strata, and reliability.
        ext_scores : bool, default True
            If True, includes extreme scorers in category probability dict.
        tolerance : float, default 0.00001
            Newton-Raphson convergence tolerance.
        max_iters : int, default 100
            Maximum Newton-Raphson iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment.
        method : str, default 'log-lik'
            Priority vector extraction method for calibration.
        constant : float, default 0.1
            Additive smoothing constant for calibration.
        matrix_power : int or None, default None
            Matrix power for calibration. None resolves to 0 for
            method='log-lik' (no powering), else 5.
        log_lik_tol : float, default 0.000001
            Convergence tolerance for calibration.
        no_of_samples : int, default 500
            Bootstrap samples for SE estimation.
        interval : float or None, default None
            CI width for bootstrap estimates.
        seed : int or None, default None
            Seed passed through to the internal std_errors() call (only
            relevant the first time SEs are computed for this model).

        Attributes set (model-suffixed)
        --------------------------------
        exp_score_df_{model}, info_df_{model}, kurtosis_df_{model} : DataFrame
            Expected scores, Fisher information, fourth moments.
        residual_df_{model}, std_residual_df_{model} : DataFrame
            Raw and standardised residuals.
        item_infit_ms_{model}, item_outfit_ms_{model} : Series
            Item infit and outfit mean-square.
        item_infit_zstd_{model}, item_outfit_zstd_{model} : Series
            Item Z statistics.
        threshold_infit_ms_{model}, threshold_outfit_ms_{model} : Series
            Threshold infit and outfit mean-square.
        rater_infit_ms_{model}, rater_outfit_ms_{model} : Series
            Facet element infit and outfit mean-square.
        person_infit_ms_{model}, person_outfit_ms_{model} : Series
            Person infit and outfit mean-square.
        csem_vector_{model}, rsem_vector_{model} : Series
            Conditional and residual SEM per person.
        isi_{model}, item_strata_{model}, item_reliability_{model} : float
            Item separation index, strata, and reliability (if test_stats).
        psi_{model}, person_strata_{model}, person_reliability_{model} : float
            Person separation index, strata, and reliability (if test_stats).
        """
        model = self._MODEL_ALIASES.get(model, model)
        if model == "mixed":
            self._sync_mixed(
                anchors,
                constant=constant,
                threshold_constant=threshold_constant,
                method=method,
                robust=robust,
                matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
                warm_corr=warm_corr,
                tolerance=tolerance,
                max_iters=max_iters,
                ext_score_adjustment=ext_score_adjustment,
                seed=seed,
            )
        elif not hasattr(self, f"facet_effects_{model}"):
            self.calibrate(
                model=model,
                constant=constant,
                threshold_constant=threshold_constant,
                method=method,
                robust=robust,
                matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
            )
        if se and not hasattr(self, f"threshold_se_{model}"):
            self.std_errors(
                model=model,
                anchors=anchors,
                interval=interval,
                no_of_samples=no_of_samples,
                constant=constant,
                threshold_constant=threshold_constant,
                method=method,
                robust=robust,
                matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
                seed=seed,
            )
        if not hasattr(self, f"persons_{model}"):
            self.person_estimates(
                model=model,
                warm_corr=warm_corr,
                tolerance=tolerance,
                max_iters=max_iters,
                ext_score_adjustment=ext_score_adjustment,
            )
        if not se:
            test_stats = False

        self.category_probability_dict(
            model=model,
            warm_corr=warm_corr,
            ext_scores=ext_scores,
            tolerance=tolerance,
            max_iters=max_iters,
            ext_score_adjustment=ext_score_adjustment,
            method=method,
            constant=constant,
            threshold_constant=threshold_constant,
            robust=robust,
            matrix_power=matrix_power,
            log_lik_tol=log_lik_tol,
        )
        self._ensure_fit_matrices(model)
        self._run_item_fit(model)
        self._run_threshold_fit(model, anchors=anchors)
        self._run_facet_fit(model)
        self._run_person_fit(model)
        if test_stats:
            self._run_test_fit(model)

    # Backwards-compatible aliases
    def fit_statistics_global(self, **kw):
        """Alias for fit_statistics(model='global'). See fit_statistics for full documentation."""
        self.fit_statistics(model="global", **kw)

    def fit_statistics_items(self, **kw):
        """Alias for fit_statistics(model='items'). See fit_statistics for full documentation."""
        self.fit_statistics(model="items", **kw)

    def fit_statistics_thresholds(self, **kw):
        """Alias for fit_statistics(model='thresholds'). See fit_statistics for full documentation."""
        self.fit_statistics(model="thresholds", **kw)

    def fit_statistics_matrix(self, **kw):
        """Alias for fit_statistics(model='matrix'). See fit_statistics for full documentation."""
        self.fit_statistics(model="matrix", **kw)

    def fit_statistics_bivector(self, **kw):
        """Alias for fit_statistics(model='bivector'). See fit_statistics for full documentation."""
        self.fit_statistics(model="bivector", **kw)

    # ------------------------------------------------------------------
    # Residual correlation analysis
    # ------------------------------------------------------------------
    def fit_statistics_centrality(self, **kw):
        """Alias for fit_statistics(model='centrality'). See fit_statistics for full documentation."""
        self.fit_statistics(model="centrality", **kw)
    def fit_statistics_pseudo_halo(self, **kw):
        """Alias for fit_statistics(model='pseudo_halo'). See fit_statistics for full documentation."""
        self.fit_statistics(model="pseudo_halo", **kw)
    def fit_statistics_bistretch(self, **kw):
        """Alias for fit_statistics(model='bistretch'). See fit_statistics for full documentation."""
        self.fit_statistics(model="bistretch", **kw)

    def andersen_lr_test(
        self,
        model="global",
        split_by="person_location",
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        constant=None,
        threshold_constant=None,
        method=None,
        robust=False,
        matrix_power=None,
        log_lik_tol=0.000001,
    ):
        """
        Andersen (1973) likelihood ratio test of parameter invariance.

        Disabled (raises NotImplementedError).

        Splits persons into low and high groups by median person_location or total raw
        score, fits the model separately in each group, and tests whether item
        and facet parameters are invariant across groups.

        Parameters
        ----------
        model : str, default 'global'
            Facet element (e.g. rater) parameterisation: 'global', 'items', 'thresholds', 'matrix', or 'bivector'.
        split_by : str, default 'person_location'
            Split criterion: 'person_location' (ML person estimates) or 'score' (total raw scores
            summed across all facet elements and items).
        warm_corr : bool, default True
            Warm bias correction for person_location estimates.
        tolerance, max_iters, ext_score_adjustment : floats
            Person estimation kwargs passed to group models.
        constant, method, matrix_power, log_lik_tol : floats
            Calibration kwargs passed to group models.

        Attributes set
        --------------
        andersen_lr_{model} : float
            Likelihood ratio statistic. Each group's own log-likelihood
            (the H1 side) is computed by plugging in person_location estimates
            from the pooled (combined-group) model rather than the
            group's own separately-fit person_locations, so the comparison
            differs from the pooled model only in item/facet parameters —
            otherwise nuisance person_location parameters are re-optimised
            independently on each side, inflating the statistic beyond
            what df accounts for.
        andersen_df_{model} : int
            Degrees of freedom.
        andersen_p_{model} : float
            p-value from chi-squared distribution.
        andersen_groups_{model} : dict
            {'low': MFRM, 'high': MFRM} — fitted group models for inspection.
        andersen_summary_{model} : pandas.Series
            LR statistic, df, and p-value.
        """
        model = self._MODEL_ALIASES.get(model, model)
        raise NotImplementedError(
            "MFRM.andersen_lr_test() is disabled. A simulation study "
            "(I=5, K=3, R=3, N=100..4000, split_by='person_location' and 'score') found "
            "the LR statistic floors to 0 (p=1.0, 'no misfit') in 58-100% of "
            "replications depending on model, and this floor rate does NOT "
            "improve with sample size — it is flat or worse at N=4000 than at "
            "N=100 for every parameterisation except matrix. Root cause: PAIR "
            "(Choppin 1968) is a matrix-algebraic pairwise-comparison estimator, "
            "not a likelihood method of any kind (not even pseudo-likelihood) — "
            "it does not maximise the Rasch response likelihood that "
            "_log_likelihood() evaluates afterward. An LR test requires both "
            "the restricted and unrestricted models to be fit by maximising the "
            "same likelihood surface, so that the unrestricted model's "
            "log-likelihood is guaranteed >= the restricted model's. PAIR gives "
            "no such guarantee, and there is no reason to expect the gap to "
            "shrink with more data, since PAIR was never targeting the "
            "likelihood surface. This method will be re-enabled once MFRM "
            "supports a genuine (C)ML calibration path for use in this test. "
            f"For {self.facet_label_eg} parameterisation comparisons (not group-invariance "
            "testing), use model_selection() instead, which is unaffected."
        )

    def andersen_lr_test_global(self, **kw):
        """Alias for andersen_lr_test(model='global'). See andersen_lr_test for full documentation."""
        self.andersen_lr_test(model="global", **kw)

    def andersen_lr_test_items(self, **kw):
        """Alias for andersen_lr_test(model='items'). See andersen_lr_test for full documentation."""
        self.andersen_lr_test(model="items", **kw)

    def andersen_lr_test_thresholds(self, **kw):
        """Alias for andersen_lr_test(model='thresholds'). See andersen_lr_test for full documentation."""
        self.andersen_lr_test(model="thresholds", **kw)

    def andersen_lr_test_matrix(self, **kw):
        """Alias for andersen_lr_test(model='matrix'). See andersen_lr_test for full documentation."""
        self.andersen_lr_test(model="matrix", **kw)

    def andersen_lr_test_bivector(self, **kw):
        """Alias for andersen_lr_test(model='bivector'). See andersen_lr_test for full documentation."""
        self.andersen_lr_test(model="bivector", **kw)

    # ------------------------------------------------------------------
    # Model selection — rater parameterisation comparison
    # ------------------------------------------------------------------
    def andersen_lr_test_centrality(self, **kw):
        """Alias for andersen_lr_test(model='centrality'). See andersen_lr_test for full documentation."""
        self.andersen_lr_test(model="centrality", **kw)
    def andersen_lr_test_pseudo_halo(self, **kw):
        """Alias for andersen_lr_test(model='pseudo_halo'). See andersen_lr_test for full documentation."""
        self.andersen_lr_test(model="pseudo_halo", **kw)
    def andersen_lr_test_bistretch(self, **kw):
        """Alias for andersen_lr_test(model='bistretch'). See andersen_lr_test for full documentation."""
        self.andersen_lr_test(model="bistretch", **kw)

    # ------------------------------------------------------------------
    # Differential Item Functioning (DIF) / Differential Category
    # Functioning (DCF)
    # ------------------------------------------------------------------

    _DIF_RISK = {
        "thresholds": (
            "high: the facet element effects at each threshold lie along "
            "the same axis as the shared thresholds."
        ),
        "matrix": (
            "highest: the full (facet element, item, threshold) grid is the "
            "least constrained representation, with the sparsest cells."
        ),
        "bivector": (
            "high: its threshold part has the same risk as 'thresholds', "
            "and its item part a milder one."
        ),
        "bistretch": (
            "high: its two stretch parameters reduce, but do not remove, "
            "the risk of 'bivector'."
        ),
        "centrality": (
            "moderate: a two-parameter restriction of 'thresholds', with "
            "the same risk, reduced."
        ),
    }

    def global_fit(self, *args, **kwargs):
        """
        Not available for MFRM: the global invariance test (Section on
        global_fit() for SLM, PCM and RSM) needs a per-group sandwich for
        every sub-model, which is not implemented. Use dif_test() to test
        item, threshold or facet element (e.g. rater) invariance across
        person groups.
        """
        raise NotImplementedError(
            "global_fit() is available for SLM, PCM and RSM, not yet for MFRM. To test "
            "invariance across groups of persons, use dif_test()."
        )

    def dif_test(
        self,
        covariate,
        model="global",
        reference=None,
        selection_method="wald",
        corr_tol=0.95,
        sd_ratio_tol=1.1,
        min_anchors=6,
        wald_alpha=0.05,
        test="wald",
        omnibus=True,
        welch=False,
        size_adjust=False,
        reference_n=100,
        dcf_cap=2000,
        correction="bh",
        alpha=0.05,
        logit_threshold=0.43,
        category=False,
        category_thresholds=(0.43, 0.64),
        category_alpha=0.05,
        no_of_samples=500,
        plot=False,
        plot_kwargs=None,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        constant=None,
        threshold_constant=None,
        method=None,
        matrix_power=None,
        matrix_marginals=None,
        regression="huber",
        robust=False,
        log_lik_tol=0.000001,
        seed=None,
    ):
        """
        Differential Item Functioning (DIF) and Differential Category
        Functioning (DCF) test by an exogenous person variable (e.g.
        Gender, L1), under the specified facet element (e.g. rater) model.

        Persons (not facet elements) are split into reference and focal
        groups, and each group is calibrated as its own MFRM under `model`,
        with the object's latest estimation method. Item-location DIF is
        tested per item, after purifying the item-location scale (items
        only); DCF is tested per category width, using the single shared
        threshold vector.

        The per-item likelihood-ratio test (test='lr'/'both') compares, for
        each item, each group's log-likelihood with its own fitted item
        location against that with a single location pooled across groups
        (precision-weighted, on each group's own scale), re-estimating
        person locations under the pooled value and holding the facet
        effects at their fitted values; df = 1. It is mildly liberal;
        test='wald' (the default) is preferred.

        There are two omnibus likelihood-ratio tests, both evaluated at the
        pooled (combined-group) model's person locations:
        dif_omnibus_table tests items only (facet effects held at their
        pooled values in both groups, items and thresholds free per group);
        df_omnibus_table tests items and facet effects together (all free
        per group). Both have df = common items - 1.

        If method=None, calibrate() resolves it by model, so dif_test()
        under two different models can use different item-estimation
        methods; pass method= explicitly for directly comparable results.

        With estimation='pair-cpat', a UserWarning is raised for the six
        models at risk of compressed item locations and thresholds
        which can produce apparent DIF/DCF when the two
        groups' facet element panels differ; 'global', 'items' and
        'pseudo_halo' carry no warning.

        Parameters
        ----------
        covariate : str
            Column name in self.exogenous to group persons by.
        model : str, default 'global'
            One of self._MODELS: the facet element representation each
            group is calibrated under.
        reference : str or None, default None
            Value of `covariate` for the reference group. None uses the
            largest group.
        selection_method : {'wald', 'robust_z', 'none'}, default 'wald'
            Purification of the item-location scale: sequential Wald tests
            ('wald'), modified z-score exclusion ('robust_z'), or a mean
            shift over all common items ('none').
        corr_tol, sd_ratio_tol, min_anchors, wald_alpha : floats/int
            Settings of the active selection method.
        test : {'wald', 'lr', 'both'}, default 'wald'
            Per-item DIF test(s). 'lr'/'both' add the likelihood-ratio test
            (two person re-estimations per item per focal group) and a
            'Flagged_LR' column; 'Flagged' always reflects the Wald test.
        omnibus : bool, default True
            If True, also computes the two omnibus tests per focal group.
        welch : bool, default False
            If True, the item-location and DCF Wald tests use Welch-
            Satterthwaite degrees of freedom instead of the normal.
        size_adjust : bool, default False
            If True, inflates each group's standard errors at large N:
            item-location SEs by sqrt(n / reference_n) (Tristán, 2006),
            category-width SEs by max(1, sqrt(n / dcf_cap)), n being the
            group's number of persons.
        reference_n : int, default 100
            Reference sample size for the item-location SEs when
            size_adjust=True.
        dcf_cap : float, default 2000
            Cap, in persons, for the category-width SEs when
            size_adjust=True.
        correction : 'bh', 'bonferroni', or None, default 'bh'
            Multiple-comparison correction, applied separately to the
            item-location and DCF p-values.
        alpha : float, default 0.05
            Significance level for flagging.
        logit_threshold : float, default 0.43
            Minimum absolute difference, in logits, for flagging.
        category : bool, default False
            If True, adds an ETS 'Category' column to dif_table: 'A'
            (negligible), 'B+'/'B-' (slight to moderate) or 'C+'/'C-'
            (moderate to large) (Zwick, Thayer & Lewis, 1999).
        category_thresholds : (float, float), default (0.43, 0.64)
            B and C category boundaries, in logits.
        category_alpha : float, default 0.05
            Significance level for the category tests.
        no_of_samples : int, default 500
            Bootstrap resamples for each group's standard errors.
        plot, plot_kwargs :
            If plot=True, plot_anchor_selection() per focal group, with
            plot_kwargs passed to it.
        warm_corr, tolerance, max_iters, ext_score_adjustment :
            Person-estimation settings, used for the likelihood-ratio and
            omnibus tests.
        constant, threshold_constant, method, matrix_power,
        matrix_marginals, regression, robust, log_lik_tol :
            Passed to each group's calibrate() call. matrix_marginals and
            regression are not passed to the bootstrap.
        seed : int or None, default None
            Seed for each group's bootstrap.

        Attributes set
        --------------
        dif_model : str
            The `model` used.
        dif_table : pandas.DataFrame
            One row per (item, focal group) pair. Columns: 'Group',
            'Reference' / 'Focal' / 'Focal (purified)', 'Difference', 'SE',
            'z', 'p', 'p (corrected)', 'Selected', 'Flagged'; 'df' if
            welch=True; 'LR', 'p_LR', 'p_LR (corrected)', 'Flagged_LR' if
            test is 'lr' or 'both'; 'Category' if category=True.
        dif_omnibus_table : pandas.DataFrame or None
            One row per focal group: 'LR', 'df', 'p', 'Flagged', for the
            test of items only. None if omnibus=False.
        df_omnibus_table : pandas.DataFrame or None
            As dif_omnibus_table, for the joint test of items and facet
            effects. None if omnibus=False.
        threshold_dif_table : pandas.DataFrame
            DCF on category widths (thresholds[k+1] - thresholds[k]). One
            row per (category, focal group) pair: 'Group', 'Category',
            'Reference' / 'Focal', 'Difference', 'SE', 'z', 'p',
            'p (corrected)', 'Flagged'; 'df' if welch=True.
        dif_reference, dif_covariate : the reference group value and
            covariate column name used.
        dif_reference_model : MFRM
            Fitted model for the reference group.
        dif_focal_models : dict {focal_value: MFRM}
            Fitted models for each focal group.
        dif_tc : dict {focal_value: float}
            Item-scale translation constant applied to each focal group.
        dif_anchor_selection : dict {focal_value: pandas.DataFrame or None}
            Purification table for each focal group.
        dif_plots : dict {focal_value: matplotlib.figure.Figure or None}
        dif_group_sizes : dict {group_value: int}
            Non-extreme, non-missing-variable N used for each group.
        """
        from raschpy.mfrm import MFRM

        model = self._MODEL_ALIASES.get(model, model)
        if model not in self._MODELS:
            raise ValueError(f"model must be one of {self._MODELS}")
        # issue #17 arises from 'pair-cpat''s staged estimation; the groups are calibrated with the latest
        # estimation method (_refit_kw), 'cp-full-rcl' if none
        if model in self._DIF_RISK and (getattr(self, "estimation", None) or "cp-full-rcl") == "pair-cpat":
            warnings.warn(
                f"model='{model}' with estimation='pair-cpat': risk of compressed item "
                f"locations and thresholds: {self._DIF_RISK[model]} "
                f"If the reference and focal groups' {self.facet_label_eg} panels differ in "
                f"composition, this compression can produce apparent DIF/DCF "
                f"that is an artefact of the comparison, not a genuine group "
                f"difference.",
                UserWarning,
                stacklevel=2,
            )
        self._dif_validate(
            correction, selection_method, test, category_thresholds, reference_n,
            dcf_cap=dcf_cap,
        )

        # Non-extreme persons, aggregating across facet_elements (matches
        # model_selection()/andersen_lr_test()'s own convention).
        total_scores = self.responses.groupby(level=1).sum().sum(axis=1)
        max_possible = (
            self.responses.notna().groupby(level=1).sum().sum(axis=1) * self.max_score
        )
        non_extreme = total_scores.index[(total_scores > 0) & (total_scores < max_possible)]

        cov_values, reference, focal_levels, group_sizes, n_missing = self._dif_group_setup(
            non_extreme, covariate, reference
        )

        def _group_responses(idx):
            # Splitting by person can leave a facet_element (rater) with
            # zero actual data for this group's persons -- self.responses
            # is a full (facet_element, person) grid (every facet_element
            # has a row for every person, NaN where unrated -- see
            # __init__'s canonicalisation step), so a naive person-mask
            # selection would still carry that facet_element's now-all-NaN
            # rows into the new MFRM's constructor, which has no analogue
            # of _remove_null_persons() for facet_elements and would treat
            # it as a genuine (degenerate) rater. Drop it here instead.
            mask = self.responses.index.get_level_values(1).isin(idx)
            resp = self.responses[mask]
            nonnull_by_rater = resp.notna().groupby(level=0).sum().sum(axis=1)
            active_raters = nonnull_by_rater[nonnull_by_rater > 0].index
            return resp[resp.index.get_level_values(0).isin(active_raters)]

        pe_kw = dict(
            warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
            ext_score_adjustment=ext_score_adjustment,
        )
        cal_kw = dict(
            constant=constant, threshold_constant=threshold_constant, method=method,
            matrix_power=matrix_power, matrix_marginals=matrix_marginals,
            regression=regression, robust=robust, log_lik_tol=log_lik_tol,
            **self._refit_kw(model),
        )
        se_kw = dict(
            no_of_samples=no_of_samples, constant=constant,
            threshold_constant=threshold_constant, method=method, robust=robust,
            matrix_power=matrix_power, log_lik_tol=log_lik_tol, seed=seed,
            # store_bootstrap=True keeps the fitted bootstrap resamples
            # (self._bootstrap_samples_{model}) so item purification's
            # Var(tc) can be estimated directly from their empirical
            # distribution instead of an independence-assumption formula
            # -- see _purification_loo_tc.
            store_bootstrap=True,
        )

        needs_ll = test in ("lr", "both") or omnibus

        ref_idx = cov_values.index[cov_values == reference]
        ref_model = MFRM(_group_responses(ref_idx), max_score=self.max_score)
        ref_model.calibrate(model=model, **cal_kw)
        ref_model.std_errors(model=model, **se_kw)
        if needs_ll:
            ref_model.person_estimates(model=model, **pe_kw)
            ll_ref = ref_model._log_likelihood(model=model)

        all_item_rows = []
        all_threshold_rows = []
        focal_models = {}
        tc_dict = {}
        anchor_selection_dict = {}
        plot_dict = {}
        omnibus_rows = {}
        df_omnibus_rows = {}

        for focal in focal_levels:
            focal_idx = cov_values.index[cov_values == focal]
            focal_model = MFRM(_group_responses(focal_idx), max_score=self.max_score)
            focal_model.calibrate(model=model, **cal_kw)
            focal_model.std_errors(model=model, **se_kw)
            if needs_ll:
                focal_model.person_estimates(model=model, **pe_kw)
                ll_focal = focal_model._log_likelihood(model=model)

            # --- Item-location DIF ---
            common_items = ref_model.items.index.intersection(focal_model.items.index)
            ref_samples = getattr(ref_model, f"_bootstrap_samples_{model}", [])
            focal_samples = getattr(focal_model, f"_bootstrap_samples_{model}", [])
            ref_boot = self._bootstrap_value_matrix(ref_samples, "items", common_items)
            focal_boot = self._bootstrap_value_matrix(focal_samples, "items", common_items)

            wald_result = self._dif_item_wald_block(
                ref_model, focal_model, ref_boot, focal_boot,
                selection_method, corr_tol, sd_ratio_tol, min_anchors, wald_alpha,
                size_adjust, reference_n, welch, correction, alpha, logit_threshold,
                category, category_thresholds, category_alpha,
            )
            common_items = wald_result["common_items"]
            tc = wald_result["tc"]
            focal_shifted = wald_result["focal_shifted"]
            diff = wald_result["diff"]
            se = wald_result["se"]
            z = wald_result["z"]
            p = wald_result["p"]
            p_corrected = wald_result["p_corrected"]
            flagged = wald_result["flagged"]
            selected = wald_result["selected"]
            item_welch_df = wald_result["welch_df"]
            category_col = wald_result["category_col"]
            anchor_selection_dict[focal] = wald_result["anchor_table"]

            if test in ("lr", "both"):
                ll_h1 = ll_ref + ll_focal
                sev_attr = self._sev_attr(model)
                compact_attr = f"facet_effects_{model}"

                def _make_dif_scratch(base_model):
                    scratch = MFRM(base_model.responses, max_score=self.max_score)
                    scratch.thresholds = base_model.thresholds
                    setattr(scratch, compact_attr, getattr(base_model, compact_attr))
                    if sev_attr != compact_attr:
                        setattr(scratch, sev_attr, getattr(base_model, sev_attr))
                    return scratch

                ref_scratch = _make_dif_scratch(ref_model)
                focal_scratch = _make_dif_scratch(focal_model)
                lr_rows = {}
                for item in common_items:
                    w_ref = 1.0 / ref_model.item_se[item] ** 2
                    w_focal = 1.0 / focal_model.item_se[item] ** 2
                    pooled = (
                        w_ref * ref_model.items[item] + w_focal * focal_shifted[item]
                    ) / (w_ref + w_focal)
                    pooled_focal_scale = pooled - tc[item]

                    ref_scratch.items = ref_model.items.copy()
                    ref_scratch.items[item] = pooled
                    ref_scratch.person_estimates(model=model, **pe_kw)
                    ll_ref_h0 = ref_scratch._log_likelihood(model=model)

                    focal_scratch.items = focal_model.items.copy()
                    focal_scratch.items[item] = pooled_focal_scale
                    focal_scratch.person_estimates(model=model, **pe_kw)
                    ll_focal_h0 = focal_scratch._log_likelihood(model=model)

                    lr_i = max(0.0, 2 * (ll_h1 - (ll_ref_h0 + ll_focal_h0)))
                    lr_rows[item] = {"LR": lr_i, "p_LR": float(chi2.sf(lr_i, 1))}

                lr_table = self._dif_lr_table(
                    lr_rows, common_items, correction, alpha, diff, logit_threshold
                )

            if omnibus:
                combined_idx = ref_idx.append(focal_idx)
                m_full = MFRM(_group_responses(combined_idx), max_score=self.max_score)
                m_full.calibrate(model=model, **cal_kw)
                m_full.person_estimates(model=model, **pe_kw)
                pooled_items, pooled_thresholds, pooled_facet_effects = m_full._get_params(model)
                ll_full = m_full._log_likelihood(model=model)
                pooled_persons = getattr(m_full, f"persons_{model}")
                pooled_ref_persons = pooled_persons.reindex(ref_model.person_names)
                pooled_focal_persons = pooled_persons.reindex(focal_model.person_names)

                # Joint DF -- both axes (items/thresholds AND facet_effects)
                # free per group.
                # df undercounts the facet_effects axis on purpose here
                # (kept for backward comparability with the earlier
                # single-omnibus behaviour) -- prefer dif_omnibus_table
                # below for an axis-attributed, properly-scoped answer.
                ll_ref_df = ref_model._log_likelihood(model=model, persons=pooled_ref_persons)
                ll_focal_df = focal_model._log_likelihood(model=model, persons=pooled_focal_persons)
                lr_df = max(0.0, -2 * (ll_full - (ll_ref_df + ll_focal_df)))
                df_df = len(common_items) - 1
                p_df = float(chi2.sf(lr_df, df_df))
                df_omnibus_rows[focal] = {
                    "LR": lr_df, "df": df_df, "p": p_df, "Flagged": p_df < alpha,
                }

                # Item-isolated DIF omnibus -- facet_effects pinned at the
                # pooled (m_full) value on both sides, items/thresholds
                # free per group. Properly nested against the same H0
                # (m_full's own fit): the facet_effects axis is held EQUAL
                # to H0's own value rather than left free, so df=items-1
                # is correctly scoped.
                ll_ref_dif = ref_model._log_likelihood(
                    model=model, persons=pooled_ref_persons, facet_effects=pooled_facet_effects,
                )
                ll_focal_dif = focal_model._log_likelihood(
                    model=model, persons=pooled_focal_persons, facet_effects=pooled_facet_effects,
                )
                lr_dif = max(0.0, -2 * (ll_full - (ll_ref_dif + ll_focal_dif)))
                df_dif = len(common_items) - 1
                p_dif = float(chi2.sf(lr_dif, df_dif))
                omnibus_rows[focal] = {
                    "LR": lr_dif, "df": df_dif, "p": p_dif, "Flagged": p_dif < alpha,
                }

            item_table = self._dif_item_table(
                focal, ref_model, focal_model, common_items, focal_shifted,
                diff, se, z, p, p_corrected, selected, flagged,
                welch_df=item_welch_df if welch else None,
                category_col=category_col if category else None,
                lr_table=lr_table if test in ("lr", "both") else None,
            )

            # --- DCF: category widths, shared threshold vector, no
            # purification (same rationale as RSM's dif_test) ---
            ref_widths = ref_model.thresholds.diff().dropna()
            focal_widths = focal_model.thresholds.diff().dropna()
            ref_widths.index = focal_widths.index = range(1, len(ref_widths) + 1)
            thr_index = ref_widths.index
            thr_diff = focal_widths - ref_widths

            ref_se_thr = pd.Series(
                getattr(ref_model, f"cat_width_se_{model}")
            ).reindex(thr_index)
            focal_se_thr = pd.Series(
                getattr(focal_model, f"cat_width_se_{model}")
            ).reindex(thr_index)
            ref_n_thr = ref_model.no_of_persons
            focal_n_thr = focal_model.no_of_persons
            if size_adjust:
                # Raw person-count cap, NOT _threshold_total_weights --
                # deliberate, see dcf_cap's own docstring/known-issues for
                # why (total_weight scales with rater-panel size R without
                # a matching true-precision gain, an unresolved confound;
                # raw N is invariant to R by construction, at the cost of
                # not tracking per-category sparsity the way PCM/RSM's own
                # weight-based cap does).
                ref_eff = min(ref_n_thr, dcf_cap)
                focal_eff = min(focal_n_thr, dcf_cap)
                ref_se_thr = ref_se_thr * np.sqrt(ref_n_thr / ref_eff)
                focal_se_thr = focal_se_thr * np.sqrt(focal_n_thr / focal_eff)
                ref_n_thr = ref_eff
                focal_n_thr = focal_eff

            thr_se = pd.Series(
                np.sqrt(ref_se_thr ** 2 + focal_se_thr ** 2).values, index=thr_index
            )
            if welch:
                thr_z_vals, thr_df_vals, thr_p_vals = self._welch_satterthwaite(
                    thr_diff.values, ref_se_thr.values, ref_n_thr,
                    focal_se_thr.values, focal_n_thr,
                )
                thr_z = pd.Series(thr_z_vals, index=thr_index)
                thr_welch_df = pd.Series(thr_df_vals, index=thr_index)
                thr_p = pd.Series(thr_p_vals, index=thr_index)
            else:
                thr_z = thr_diff / thr_se
                thr_p = pd.Series(2 * norm.sf(np.abs(thr_z.values)), index=thr_index)

            if correction == "bonferroni":
                thr_p_corrected = (thr_p * len(thr_p)).clip(upper=1.0)
            elif correction == "bh":
                thr_p_corrected = self._bh_correction(thr_p)
            else:
                thr_p_corrected = thr_p

            thr_flagged = (thr_p_corrected < alpha) & (thr_diff.abs() >= logit_threshold)

            threshold_table = pd.DataFrame(
                {
                    "Group": focal,
                    "Category": thr_index,
                    "Reference": ref_widths,
                    "Focal": focal_widths,
                    "Difference": thr_diff,
                    "SE": thr_se,
                    "z": thr_z,
                    "p": thr_p,
                    "p (corrected)": thr_p_corrected,
                    "Flagged": thr_flagged,
                },
                index=thr_index,
            )
            if welch:
                threshold_table["df"] = thr_welch_df

            all_item_rows.append(item_table)
            all_threshold_rows.append(threshold_table)
            focal_models[focal] = focal_model
            tc_dict[focal] = tc

            if plot and anchor_selection_dict[focal] is not None:
                plot_dict[focal] = self.plot_anchor_selection(
                    anchor_selection_dict[focal], **(plot_kwargs or {})
                )
            else:
                plot_dict[focal] = None

        self.dif_model = model
        self.dif_table = pd.concat(all_item_rows)
        self.dif_omnibus_table = pd.DataFrame(omnibus_rows).T if omnibus else None
        self.df_omnibus_table = pd.DataFrame(df_omnibus_rows).T if omnibus else None
        self.threshold_dif_table = pd.concat(all_threshold_rows, ignore_index=True)
        self.dif_reference = reference
        self.dif_covariate = covariate
        self.dif_reference_model = ref_model
        self.dif_focal_models = focal_models
        self.dif_tc = tc_dict
        self.dif_anchor_selection = anchor_selection_dict
        self.dif_plots = plot_dict
        self.dif_group_sizes = group_sizes

    def _drf_element_table(
        self, focal, ref_df, focal_df, ref_se_df, focal_se_df,
        selection_method, corr_tol, sd_ratio_tol, min_anchors, wald_alpha,
        welch, correction, alpha, logit_threshold, ref_n, focal_n, row_label,
        ref_boot_elements=None, focal_boot_elements=None,
    ):
        """
        Per-element DRF Wald test for a (facet element (e.g. rater) x element) facet_effect
        table -- 'items' (element=item), 'thresholds' (element=category),
        or one track of 'bivector'/'matrix' (item-marginal or threshold-
        marginal). Shared by all four because the underlying statistical
        problem is identical once you're looking at one (facet element x element)
        table: purifies EACH ELEMENT'S OWN COLUMN independently (one
        translation constant per element, not one per focal group) --
        confirmed necessary directly from priority_vector()'s own zero-
        centring convention (each element's facet-element-deviation column is
        independently zero-mean across whichever facet elements are present for
        THAT element). Every element is tested (not just
        purification-selected ones), matching dif_test's own convention.

        Purification is LEAVE-ONE-OUT per facet element within each element's
        column (via _purification_loo_tc): a whole-set translation constant
        inflates Type I error when the set of common facet elements is
        small.

        ref_df, focal_df, ref_se_df, focal_se_df : pandas.DataFrame
            (facet element x element), already restricted to the eligible
            (common) facet element set.
        ref_boot_elements, focal_boot_elements : dict {element: DataFrame
            or None}, from _bootstrap_element_matrices, or None --
            when supplied, purification's variance is estimated directly
            from the bootstrap distribution rather than an independence-
            assumption formula (see _purification_loo_tc).

        Returns
        -------
        table : pandas.DataFrame
            One row per (facet element, element), index=facet element (non-unique -- one
            row per element), with a `row_label` column naming the
            element. Correction is a single flat BH/Bonferroni pool
            across the WHOLE table (every facet element x element row), matching
            dif_table's own convention -- not a separate pool per
            element or per facet element.
        tc_per_element : dict {element: pandas.Series}
            Each element's LEAVE-ONE-OUT translation constant, one value
            per facet element (not a single scalar per element any more).
        anchor_selection_per_element : dict {element: pandas.DataFrame or None}
            Whole-set (non-LOO) diagnostic only -- see _purification_loo_tc.
        """
        rows = []
        tc_per_element = {}
        anchor_selection_per_element = {}
        for element in ref_df.columns:
            ref_col = ref_df[element]
            focal_col = focal_df[element]
            focal_se_col = focal_se_df[element]
            ref_se_col = ref_se_df[element]
            ref_boot = (ref_boot_elements or {}).get(element)
            focal_boot = (focal_boot_elements or {}).get(element)
            tc, ref_var, focal_var, selected, anchor_table = self._purification_loo_tc(
                ref_col, focal_col, ref_se_col, focal_se_col,
                selection_method, corr_tol, sd_ratio_tol, min_anchors, wald_alpha,
                ref_boot=ref_boot, focal_boot=focal_boot,
            )
            tc_per_element[element] = tc
            anchor_selection_per_element[element] = anchor_table

            focal_shifted = focal_col + tc
            diff = focal_shifted - ref_col
            ref_se_eff = np.sqrt(ref_var)
            focal_se_eff = np.sqrt(focal_var)
            se = np.sqrt(ref_var + focal_var)
            if welch:
                z_vals, df_vals, p_vals = self._welch_satterthwaite(
                    diff.values, ref_se_eff.values, ref_n, focal_se_eff.values, focal_n,
                )
                z = pd.Series(z_vals, index=ref_col.index)
                w_df = pd.Series(df_vals, index=ref_col.index)
                p = pd.Series(p_vals, index=ref_col.index)
            else:
                z = diff / se
                p = pd.Series(2 * norm.sf(np.abs(z.values)), index=ref_col.index)

            tbl = pd.DataFrame(
                {
                    "Group": focal,
                    row_label: element,
                    "Reference": ref_col,
                    "Focal": focal_col,
                    "Focal (purified)": focal_shifted,
                    "Difference": diff,
                    "SE": se,
                    "z": z,
                    "p": p,
                    "Selected": ref_col.index.isin(selected),
                },
                index=ref_col.index,
            )
            if welch:
                tbl["df"] = w_df
            rows.append(tbl)

        table = pd.concat(rows)
        if correction == "bonferroni":
            table["p (corrected)"] = (table["p"] * len(table)).clip(upper=1.0)
        elif correction == "bh":
            table["p (corrected)"] = self._bh_correction(table["p"])
        else:
            table["p (corrected)"] = table["p"]
        table["Flagged"] = (
            (table["p (corrected)"] < alpha) & (table["Difference"].abs() >= logit_threshold)
        )
        table.index.name = self.facet.capitalize()
        return table, tc_per_element, anchor_selection_per_element

    _STRETCH_PARAMS = {
        "centrality": ["lambda", "omega"],
        "pseudo_halo": ["lambda", "omega"],
        "bistretch": ["lambda", "omega_items", "omega_thresholds"],
    }

    def _drf_stretch_test(
        self, focal, model, ref_model, focal_model, eligible,
        selection_method, corr_tol, sd_ratio_tol, min_anchors, wald_alpha,
        correction, alpha, logit_threshold,
    ):
        """
        Joint covariance-aware DRF test for the stretch models
        ('centrality', 'pseudo_halo', 'bistretch').

        lambda purifies exactly like 'global' DRF (common-facet-element
        translation constant via _wald_anchor_selection/
        _robust_anchor_selection); it inherits community-wide zero-centring for free, same mechanism
        as facet_effects_global. omega (and omega_items/omega_thresholds
        for bistretch) need NO purification at all -- confirmed directly
        in code (shifting the reference scale by a known constant left
        omega exactly unchanged, to machine precision).

        Because lambda and omega come from ONE shared regression per
        facet element (e.g. rater) (WLS of the deviation vector on the reference scale), they
        are NOT independent -- the primary test is a joint
        T = Delta^T Sigma^-1 Delta (df = number of stretch parameters for
        this model), using the (lambda, omega) covariance
        _store_stretch_param_se now captures from paired bootstrap draws,
        NOT two independent marginal Wald tests, which would ignore that
        correlation. Marginal z-tests are still reported per parameter as
        a secondary breakdown (own, independent correction pool), mirror-
        ing the aggregate-plus-component convention used elsewhere in
        this file (e.g. check_anchor_homogeneity, dif_table/
        threshold_dif_table).

        Returns
        -------
        table : pandas.DataFrame, one row per common facet element.
        tc : float (lambda's translation constant for this focal group).
        anchor_table : pandas.DataFrame or None.
        """
        params = self._STRETCH_PARAMS[model]

        ref_lambda = getattr(ref_model, f"lambda_{model}").loc[eligible]
        focal_lambda = getattr(focal_model, f"lambda_{model}").loc[eligible]
        ref_lambda_se = getattr(ref_model, f"lambda_se_{model}").loc[eligible]
        focal_lambda_se = getattr(focal_model, f"lambda_se_{model}").loc[eligible]

        ref_samples = getattr(ref_model, f"_bootstrap_samples_{model}", [])
        focal_samples = getattr(focal_model, f"_bootstrap_samples_{model}", [])
        ref_boot = self._bootstrap_value_matrix(ref_samples, f"lambda_{model}", eligible)
        focal_boot = self._bootstrap_value_matrix(focal_samples, f"lambda_{model}", eligible)

        # Leave-one-out per rater, with tc's own variance estimated
        # directly from the bootstrap distribution -- fixes known issue
        # #39 (an independence-assumption closed form under-covered
        # too, ~2x nominal; see _purification_loo_tc's own docstring).
        tc, ref_lambda_var, focal_lambda_var, selected, anchor_table = self._purification_loo_tc(
            ref_lambda, focal_lambda, ref_lambda_se, focal_lambda_se,
            selection_method, corr_tol, sd_ratio_tol, min_anchors, wald_alpha,
            ref_boot=ref_boot, focal_boot=focal_boot,
        )

        ref_vals, focal_vals, ref_se, focal_se = {}, {}, {}, {}
        ref_vals["lambda"], focal_vals["lambda"] = ref_lambda, focal_lambda + tc
        # Both sides now come from _purification_loo_tc's bootstrap-based variance
        # (falls back to the independence-assumption formula, still
        # split correctly by group, when bootstrap samples aren't
        # available) -- feeds both the marginal z-test and the joint
        # covariance matrix's lambda diagonal below.
        ref_se["lambda"] = np.sqrt(ref_lambda_var)
        focal_se["lambda"] = np.sqrt(focal_lambda_var)
        for p in params[1:]:
            ref_vals[p] = getattr(ref_model, f"{p}_{model}").loc[eligible]
            focal_vals[p] = getattr(focal_model, f"{p}_{model}").loc[eligible]
            ref_se[p] = getattr(ref_model, f"{p}_se_{model}").loc[eligible]
            focal_se[p] = getattr(focal_model, f"{p}_se_{model}").loc[eligible]

        diffs = {p: focal_vals[p] - ref_vals[p] for p in params}

        def _cov_matrices(model_obj, se_dict):
            cov_pairs = getattr(model_obj, f"stretch_param_cov_{model}", {})
            mats = {}
            for r in eligible:
                M = np.zeros((len(params), len(params)))
                for i, p in enumerate(params):
                    M[i, i] = se_dict[p].loc[r] ** 2
                for i, p1 in enumerate(params):
                    for j, p2 in enumerate(params[i + 1:], start=i + 1):
                        key = f"{p1}_{p2}"
                        val = cov_pairs[key].loc[r] if key in cov_pairs else 0.0
                        M[i, j] = M[j, i] = val
                mats[r] = M
            return mats

        sigma_ref = _cov_matrices(ref_model, ref_se)
        sigma_focal = _cov_matrices(focal_model, focal_se)

        n_params = len(params)
        T_vals, p_vals = {}, {}
        for r in eligible:
            sigma = sigma_ref[r] + sigma_focal[r]
            delta = np.array([diffs[p].loc[r] for p in params])
            try:
                T = float(delta @ np.linalg.solve(sigma, delta))
            except np.linalg.LinAlgError:
                T = np.nan
            T_vals[r] = T
            p_vals[r] = float(chi2.sf(T, n_params)) if np.isfinite(T) else np.nan

        T_series = pd.Series(T_vals)
        p_series = pd.Series(p_vals)
        if correction == "bonferroni":
            p_corrected = (p_series * len(p_series)).clip(upper=1.0)
        elif correction == "bh":
            p_corrected = self._bh_correction(p_series)
        else:
            p_corrected = p_series
        flagged = (p_corrected < alpha) & (
            pd.concat([diffs[p].abs() for p in params], axis=1).max(axis=1) >= logit_threshold
        )

        data = {"Group": focal}
        for p in params:
            data[f"Reference_{p}"] = ref_vals[p]
            data[f"Focal_{p}"] = focal_vals[p] if p == "lambda" else focal_vals[p]
            data[f"Difference_{p}"] = diffs[p]
        data["T"] = T_series
        data["df"] = n_params
        data["p"] = p_series
        data["p (corrected)"] = p_corrected
        data["Selected"] = eligible.isin(selected)
        data["Flagged"] = flagged

        for p in params:
            se_p = np.sqrt(ref_se[p] ** 2 + focal_se[p] ** 2)
            z_p = diffs[p] / se_p
            pval_p = pd.Series(2 * norm.sf(np.abs(z_p.values)), index=eligible)
            if correction == "bonferroni":
                pval_p_corr = (pval_p * len(pval_p)).clip(upper=1.0)
            elif correction == "bh":
                pval_p_corr = self._bh_correction(pval_p)
            else:
                pval_p_corr = pval_p
            data[f"z_{p}"] = z_p
            data[f"p_{p}"] = pval_p
            data[f"p_{p} (corrected)"] = pval_p_corr
            data[f"Flagged_{p}"] = (pval_p_corr < alpha) & (diffs[p].abs() >= logit_threshold)

        table = pd.DataFrame(data, index=eligible)
        table.index.name = self.facet.capitalize()
        return table, tc, anchor_table

    def drf_test(
        self,
        covariate,
        model="global",
        reference=None,
        selection_method="wald",
        corr_tol=0.95,
        sd_ratio_tol=1.1,
        min_anchors=2,
        wald_alpha=0.05,
        min_persons_per_group=5,
        omnibus=None,
        welch=False,
        correction="bh",
        alpha=0.05,
        logit_threshold=0.43,
        no_of_samples=500,
        plot=False,
        plot_kwargs=None,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        constant=None,
        threshold_constant=None,
        method=None,
        matrix_power=None,
        robust=False,
        log_lik_tol=0.000001,
        seed=None,
    ):
        """
        Differential Rater Functioning (DRF) test by an exogenous person
        variable: the facet element (e.g. rater) location analogue of DIF.
        The tests depend on the model:

        - 'global': one test per common facet element.
        - 'items'/'thresholds'/'bivector'/'matrix': a separate test of each
          facet element's effect on each item or threshold. 'bivector' and
          'matrix' produce two tables, drf_table (items) and
          drf_threshold_table (thresholds); 'matrix' uses the marginal
          item and threshold projections.
        - 'centrality'/'pseudo_halo'/'bistretch': one joint test per common
          facet element of its 2 (or 3, 'bistretch') parameters (lambda,
          omega; or omega_items/omega_thresholds): T = D' S^-1 D, with D
          the focal-minus-reference differences and S their covariance
          from paired bootstrap draws, referred to chi-squared with 2 (or
          3) df. Separate z-tests per parameter are also reported. lambda
          is purified as for 'global'; omega is not purified.

        Only common facet elements are tested: those that rated at least
        min_persons_per_group persons in each group (others are listed in
        drf_excluded_raters). The facet element location scale is purified
        with the same anchor-selection methods as dif_test; all common
        facet elements are tested. For 'global' there is one translation
        constant per focal group; for 'items'/'thresholds'/'bivector'/
        'matrix' there is one per item or threshold.

        The omnibus test is available for 'global' only, and is run by
        default. It holds items and thresholds at their pooled
        (combined-group) values and lets only the facet effects vary per
        group, with df = common facet elements - 1.

        Parameters
        ----------
        covariate : str
            Column name in self.exogenous to group persons by.
        model : str, default 'global'
            One of self._MODELS; determines the tests (see above).
        reference : str or None, default None
            Value of `covariate` for the reference group. None uses the
            largest group.
        selection_method : {'wald', 'robust_z', 'none'}, default 'wald'
            Purification of the facet element location scale: sequential
            Wald tests ('wald'), modified z-score exclusion ('robust_z'),
            or a mean shift over all common facet elements ('none').
        corr_tol, sd_ratio_tol, min_anchors, wald_alpha : floats/int
            Settings of the active selection method. min_anchors defaults
            to 2.
        min_persons_per_group : int, default 5
            Minimum number of persons a facet element must have rated in
            each group to be tested.
        omnibus : bool or None, default None
            Whether to run the omnibus test ('global' only; True with any
            other model raises ValueError). None runs it for 'global'.
        welch : bool, default False
            If True, the Wald tests use Welch-Satterthwaite degrees of
            freedom instead of the normal.
        correction : 'bh', 'bonferroni', or None, default 'bh'
            Multiple-comparison correction across each table's tests.
        alpha : float, default 0.05
            Significance level for flagging.
        logit_threshold : float, default 0.43
            Minimum absolute difference, in logits, for flagging.
        no_of_samples : int, default 500
            Bootstrap resamples for each group's standard errors.
        plot, plot_kwargs :
            If plot=True, plot_anchor_selection() per focal group, with
            plot_kwargs passed to it.
        warm_corr, tolerance, max_iters, ext_score_adjustment :
            Person-estimation settings, used for the omnibus test.
        constant, threshold_constant, method, matrix_power, robust,
        log_lik_tol :
            Passed to each group's calibrate() call.
        seed : int or None, default None
            Seed for each group's bootstrap.

        Attributes set
        --------------
        drf_model : str
            The `model` used.
        drf_table : pandas.DataFrame
            Common facet elements only. 'global': one row per (facet
            element, focal group), with 'Group', 'Reference' / 'Focal' /
            'Focal (purified)', 'Difference', 'SE', 'z', 'p',
            'p (corrected)', 'Selected', 'Flagged', and 'df' if welch=True.
            'items'/'thresholds'/'bivector'/'matrix': one row per (facet
            element, item or threshold, focal group), with the same columns
            plus 'Item' or 'Category'; one correction across the table;
            items only for 'bivector'/'matrix'.
            'centrality'/'pseudo_halo'/'bistretch': one row per (facet
            element, focal group), with 'Group'; 'Reference_{param}',
            'Focal_{param}', 'Difference_{param}' per parameter; 'T', 'df',
            'p', 'p (corrected)', 'Selected' (lambda purification),
            'Flagged' (joint test); and 'z_{param}', 'p_{param}',
            'p_{param} (corrected)', 'Flagged_{param}' per parameter, with
            a separate correction.
        drf_threshold_table : pandas.DataFrame or None
            'bivector'/'matrix' only: the threshold tests, same columns as
            drf_table plus 'Category', corrected separately.
        drf_omnibus_table : pandas.DataFrame or None
            One row per focal group: 'LR', 'df', 'p', 'Flagged'. None
            unless the omnibus test is run.
        drf_excluded_raters : dict {focal_value: list}
            Facet elements present in both groups but excluded for rating
            fewer than min_persons_per_group persons in at least one.
        drf_reference, drf_covariate : the reference group value and
            covariate column name used.
        drf_reference_model : MFRM
            Fitted model for the reference group.
        drf_focal_models : dict {focal_value: MFRM}
        drf_tc : dict
            Leave-one-out translation constants, one per common facet
            element. 'global', and lambda for 'centrality'/'pseudo_halo'/
            'bistretch': {focal_value: pandas.Series}. 'items'/
            'thresholds': {focal_value: {element: pandas.Series}}.
            'bivector'/'matrix': {focal_value: {'items': {element: Series},
            'thresholds': {element: Series}}}.
        drf_anchor_selection : dict {focal_value: pandas.DataFrame or None}
        drf_plots : dict {focal_value: matplotlib.figure.Figure or None}
        drf_group_sizes : dict {group_value: int}
        """
        from raschpy.mfrm import MFRM

        model = self._MODEL_ALIASES.get(model, model)
        if model not in self._MODELS:
            raise ValueError(f"model must be one of {self._MODELS}")
        if omnibus is None:
            # 'global' only, since that's the only model with a validated
            # omnibus construction -- default to the better-performing
            # option where one exists; False elsewhere, since no
            # omnibus construction exists there yet.
            omnibus = model == "global"
        if omnibus and model != "global":
            raise ValueError(
                "omnibus=True is only supported for model='global'. The "
                "omnibus LR construction's df (see the pooled-item/"
                "threshold-pinning mechanism above) has only been worked "
                f"out for global's single {self.facet_label_eg} location scalar -- for "
                "'items'/'thresholds'/'bivector'/'matrix' the correct df "
                f"would need to charge for the full per-element {self.facet_label} "
                "parameter count (mirroring model_selection()'s own "
                "rater_k table), which hasn't been done. Pass "
                "omnibus=False for these models."
            )
        if correction not in ("bh", "bonferroni", None):
            raise ValueError("correction must be 'bh', 'bonferroni', or None")
        if selection_method not in ("wald", "robust_z", "none"):
            raise ValueError("selection_method must be 'wald', 'robust_z', or 'none'")
        if min_persons_per_group < 1:
            raise ValueError("min_persons_per_group must be at least 1.")

        if getattr(self, "exogenous", None) is None:
            raise ValueError(
                "No exogenous data available. Pass exogenous= to the "
                "constructor before calling drf_test()."
            )
        if covariate not in self.exogenous.columns:
            raise ValueError(f"'{covariate}' is not a column in self.exogenous.")


        # Non-extreme persons, aggregating across facet_elements (same
        # convention as dif_test).
        total_scores = self.responses.groupby(level=1).sum().sum(axis=1)
        max_possible = (
            self.responses.notna().groupby(level=1).sum().sum(axis=1) * self.max_score
        )
        non_extreme = total_scores.index[(total_scores > 0) & (total_scores < max_possible)]

        cov_values = self.exogenous.loc[non_extreme, covariate].dropna()
        n_missing = len(non_extreme) - len(cov_values)

        levels = cov_values.value_counts()
        if len(levels) < 2:
            raise ValueError(
                f"'{covariate}' has fewer than 2 distinct non-null values "
                f"among non-extreme persons -- cannot run a DRF test."
            )

        if reference is None:
            reference = levels.index[0]
        elif reference not in levels.index:
            raise ValueError(f"reference='{reference}' is not a value of '{covariate}'.")

        focal_levels = [lvl for lvl in levels.index if lvl != reference]
        if len(focal_levels) > 1:
            warnings.warn(
                f"'{covariate}' has {len(levels)} levels. drf_test() "
                f"compares each focal level to the reference level "
                f"'{reference}' individually (reference-group design); "
                f"all-pairwise comparisons between non-reference levels "
                f"are not supported.",
                UserWarning,
                stacklevel=2,
            )

        group_sizes = {reference: int(levels.loc[reference])}
        for focal in focal_levels:
            group_sizes[focal] = int(levels.loc[focal])

        if n_missing > 0:
            warnings.warn(
                f"{n_missing} non-extreme person(s) have a missing value for "
                f"'{covariate}' and were excluded. Group sizes used: "
                f"{group_sizes}.",
                UserWarning,
                stacklevel=2,
            )

        def _group_responses(idx):
            mask = self.responses.index.get_level_values(1).isin(idx)
            resp = self.responses[mask]
            nonnull_by_rater = resp.notna().groupby(level=0).sum().sum(axis=1)
            active_raters = nonnull_by_rater[nonnull_by_rater > 0].index
            return resp[resp.index.get_level_values(0).isin(active_raters)]

        def _rated_person_counts(resp):
            return resp.notna().any(axis=1).groupby(level=0).sum()

        pe_kw = dict(
            warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
            ext_score_adjustment=ext_score_adjustment,
        )
        cal_kw = dict(
            constant=constant, threshold_constant=threshold_constant, method=method,
            matrix_power=matrix_power, robust=robust, log_lik_tol=log_lik_tol,
            **self._refit_kw(model),
        )
        se_kw = dict(
            no_of_samples=no_of_samples, constant=constant,
            threshold_constant=threshold_constant, method=method, robust=robust,
            matrix_power=matrix_power, log_lik_tol=log_lik_tol, seed=seed,
            # store_bootstrap=True keeps the fitted bootstrap resamples
            # (self._bootstrap_samples_{model}) so purification's Var(tc)
            # can be estimated directly from their empirical distribution
            # instead of an independence-assumption formula -- see
            # _purification_loo_tc.
            store_bootstrap=True,
        )

        ref_idx = cov_values.index[cov_values == reference]
        ref_responses = _group_responses(ref_idx)
        ref_model = MFRM(ref_responses, max_score=self.max_score)
        ref_model.calibrate(model=model, **cal_kw)
        ref_model.std_errors(model=model, **se_kw)
        ref_counts = _rated_person_counts(ref_responses)

        all_rows = []
        all_threshold_rows = []
        focal_models = {}
        tc_dict = {}
        anchor_selection_dict = {}
        plot_dict = {}
        excluded_dict = {}
        omnibus_rows = {}

        for focal in focal_levels:
            focal_idx = cov_values.index[cov_values == focal]
            focal_responses = _group_responses(focal_idx)
            focal_model = MFRM(focal_responses, max_score=self.max_score)
            focal_model.calibrate(model=model, **cal_kw)
            focal_model.std_errors(model=model, **se_kw)
            focal_counts = _rated_person_counts(focal_responses)

            common_raters = ref_model.facet_names.intersection(focal_model.facet_names)
            eligible = pd.Index([
                r for r in common_raters
                if ref_counts.get(r, 0) >= min_persons_per_group
                and focal_counts.get(r, 0) >= min_persons_per_group
            ])
            excluded_dict[focal] = [r for r in common_raters if r not in eligible]
            if len(eligible) < 2:
                raise ValueError(
                    f"Only {self._facet_n(len(eligible))} {'bridges' if len(eligible) == 1 else 'bridge'} the "
                    f"reference and '{focal}' groups with at least "
                    f"min_persons_per_group={min_persons_per_group} rated "
                    f"persons in each -- cannot run a DRF test for this "
                    f"focal group. Lower min_persons_per_group or check "
                    f"{self.facet_label} panel overlap between groups."
                )

            elem_kw = dict(
                selection_method=selection_method, corr_tol=corr_tol,
                sd_ratio_tol=sd_ratio_tol, min_anchors=min_anchors, wald_alpha=wald_alpha,
                welch=welch, correction=correction, alpha=alpha,
                logit_threshold=logit_threshold,
                ref_n=ref_model.no_of_persons, focal_n=focal_model.no_of_persons,
            )

            if model in ("items", "thresholds"):
                # Per-axis-slice purification: each item's/threshold's own
                # column is independently zero-mean across raters, so _drf_element_table
                # purifies per element, not once for the whole table.
                ref_df = getattr(ref_model, f"facet_effects_{model}").loc[eligible]
                focal_df = getattr(focal_model, f"facet_effects_{model}").loc[eligible]
                ref_se_df = getattr(ref_model, f"rater_se_{model}").loc[eligible]
                focal_se_df = getattr(focal_model, f"rater_se_{model}").loc[eligible]
                ref_samples = getattr(ref_model, f"_bootstrap_samples_{model}", [])
                focal_samples = getattr(focal_model, f"_bootstrap_samples_{model}", [])
                ref_boot_e = self._bootstrap_element_matrices(
                    ref_samples, f"facet_effects_{model}", eligible
                )
                focal_boot_e = self._bootstrap_element_matrices(
                    focal_samples, f"facet_effects_{model}", eligible
                )
                row_label = "Item" if model == "items" else "Category"
                table, tc_e, anchor_e = self._drf_element_table(
                    focal, ref_df, focal_df, ref_se_df, focal_se_df,
                    row_label=row_label,
                    ref_boot_elements=ref_boot_e, focal_boot_elements=focal_boot_e,
                    **elem_kw,
                )
                all_rows.append(table)
                tc_dict[focal] = tc_e
                anchor_selection_dict[focal] = anchor_e
                plot_dict[focal] = None  # per-element plotting not supported yet

            elif model in ("bivector", "matrix"):
                # 'matrix' deliberately uses the MARGINAL item/threshold
                # projections (marginal_facet_effects_items/_thresholds),
                # not a full per-(rater,item,threshold) table, whose cells
                # are too sparse among common raters for a reliable test. Both marginals
                # are confirmed zero-mean-across-raters per element (a
                # mean of zero-mean-per-cell quantities is itself zero-
                # mean), so the same per-axis-slice purification applies
                # unchanged.
                if model == "bivector":
                    items_attr, thr_attr = (
                        "facet_effects_bivector_items", "facet_effects_bivector_thresholds"
                    )
                else:
                    items_attr, thr_attr = (
                        "marginal_facet_effects_items", "marginal_facet_effects_thresholds"
                    )
                ref_items_df = getattr(ref_model, items_attr).loc[eligible]
                focal_items_df = getattr(focal_model, items_attr).loc[eligible]
                ref_thr_df = getattr(ref_model, thr_attr).loc[eligible]
                focal_thr_df = getattr(focal_model, thr_attr).loc[eligible]
                # rater_se_marginal_items/_thresholds are shared (non-model-
                # suffixed) attribute names, freshly set by the std_errors()
                # call just above for THIS model -- safe to read here.
                ref_se_items_df = ref_model.rater_se_marginal_items.loc[eligible]
                focal_se_items_df = focal_model.rater_se_marginal_items.loc[eligible]
                ref_se_thr_df = ref_model.rater_se_marginal_thresholds.loc[eligible]
                focal_se_thr_df = focal_model.rater_se_marginal_thresholds.loc[eligible]

                ref_samples = getattr(ref_model, f"_bootstrap_samples_{model}", [])
                focal_samples = getattr(focal_model, f"_bootstrap_samples_{model}", [])
                ref_boot_items = self._bootstrap_element_matrices(ref_samples, items_attr, eligible)
                focal_boot_items = self._bootstrap_element_matrices(focal_samples, items_attr, eligible)
                ref_boot_thr = self._bootstrap_element_matrices(ref_samples, thr_attr, eligible)
                focal_boot_thr = self._bootstrap_element_matrices(focal_samples, thr_attr, eligible)

                table_items, tc_items, anchor_items = self._drf_element_table(
                    focal, ref_items_df, focal_items_df, ref_se_items_df, focal_se_items_df,
                    row_label="Item",
                    ref_boot_elements=ref_boot_items, focal_boot_elements=focal_boot_items,
                    **elem_kw,
                )
                table_thr, tc_thr, anchor_thr = self._drf_element_table(
                    focal, ref_thr_df, focal_thr_df, ref_se_thr_df, focal_se_thr_df,
                    row_label="Category",
                    ref_boot_elements=ref_boot_thr, focal_boot_elements=focal_boot_thr,
                    **elem_kw,
                )
                all_rows.append(table_items)
                all_threshold_rows.append(table_thr)
                tc_dict[focal] = {"items": tc_items, "thresholds": tc_thr}
                anchor_selection_dict[focal] = {"items": anchor_items, "thresholds": anchor_thr}
                plot_dict[focal] = None  # per-element plotting not supported yet

            elif model in ("centrality", "pseudo_halo", "bistretch"):
                # Joint covariance-aware test -- lambda purified (common-
                # rater anchoring, like 'global'), omega(s) unpurified
                # (confirmed shift-invariant). See _drf_stretch_test's own
                # docstring.
                table, tc_lambda, anchor_lambda = self._drf_stretch_test(
                    focal, model, ref_model, focal_model, eligible,
                    selection_method=selection_method, corr_tol=corr_tol,
                    sd_ratio_tol=sd_ratio_tol, min_anchors=min_anchors,
                    wald_alpha=wald_alpha, correction=correction, alpha=alpha,
                    logit_threshold=logit_threshold,
                )
                all_rows.append(table)
                tc_dict[focal] = tc_lambda
                anchor_selection_dict[focal] = anchor_lambda
                plot_dict[focal] = None  # per-element plotting not supported yet

            else:  # model == "global"
                ref_sev = ref_model.facet_effects_global.loc[eligible]
                focal_sev = focal_model.facet_effects_global.loc[eligible]
                ref_se = ref_model.rater_se_global.loc[eligible]
                focal_se_all = focal_model.rater_se_global.loc[eligible]

                ref_samples = getattr(ref_model, "_bootstrap_samples_global", [])
                focal_samples = getattr(focal_model, "_bootstrap_samples_global", [])
                ref_boot = self._bootstrap_value_matrix(ref_samples, "facet_effects_global", eligible)
                focal_boot = self._bootstrap_value_matrix(focal_samples, "facet_effects_global", eligible)

                # Leave-one-out per rater, with tc's own variance
                # estimated from the bootstrap distribution (see
                # _purification_loo_tc).
                tc, ref_var, focal_var, selected, anchor_table = self._purification_loo_tc(
                    ref_sev, focal_sev, ref_se, focal_se_all,
                    selection_method, corr_tol, sd_ratio_tol, min_anchors, wald_alpha,
                    ref_boot=ref_boot, focal_boot=focal_boot,
                )
                anchor_selection_dict[focal] = anchor_table

                focal_shifted = focal_sev + tc
                diff = focal_shifted - ref_sev

                ref_se = np.sqrt(ref_var)
                focal_se = np.sqrt(focal_var)
                se = np.sqrt(ref_var + focal_var)
                if welch:
                    z_vals, df_vals, p_vals = self._welch_satterthwaite(
                        diff.values, ref_se.values, ref_model.no_of_persons,
                        focal_se.values, focal_model.no_of_persons,
                    )
                    z = pd.Series(z_vals, index=eligible)
                    rater_welch_df = pd.Series(df_vals, index=eligible)
                    p = pd.Series(p_vals, index=eligible)
                else:
                    z = diff / se
                    p = pd.Series(2 * norm.sf(np.abs(z.values)), index=eligible)

                if correction == "bonferroni":
                    p_corrected = (p * len(p)).clip(upper=1.0)
                elif correction == "bh":
                    p_corrected = self._bh_correction(p)
                else:
                    p_corrected = p

                flagged = (p_corrected < alpha) & (diff.abs() >= logit_threshold)
                tc_dict[focal] = tc

            if omnibus:
                combined_idx = ref_idx.append(focal_idx)
                m_full = MFRM(_group_responses(combined_idx), max_score=self.max_score)
                m_full.calibrate(model=model, **cal_kw)
                m_full.person_estimates(model=model, **pe_kw)
                pooled_items, pooled_thresholds, _ = m_full._get_params(model)
                ll_full = m_full._log_likelihood(model=model)
                pooled_persons = getattr(m_full, f"persons_{model}")
                pooled_ref_persons = pooled_persons.reindex(ref_model.person_names)
                pooled_focal_persons = pooled_persons.reindex(focal_model.person_names)
                # Rater-isolated DRF omnibus -- items/thresholds pinned at
                # the pooled (m_full) value on both sides, facet_effects
                # free per group. Properly nested against the same H0
                # (m_full's own fit): the item axis is held EQUAL to H0's
                # own value rather than left free, so df=eligible-1 is
                # rater-specific.
                ll_ref_omni = ref_model._log_likelihood(
                    model=model, persons=pooled_ref_persons,
                    items=pooled_items, thresholds=pooled_thresholds,
                )
                ll_focal_omni = focal_model._log_likelihood(
                    model=model, persons=pooled_focal_persons,
                    items=pooled_items, thresholds=pooled_thresholds,
                )
                lr_omni = max(0.0, -2 * (ll_full - (ll_ref_omni + ll_focal_omni)))
                df_omni = len(eligible) - 1
                p_omni = float(chi2.sf(lr_omni, df_omni))
                omnibus_rows[focal] = {
                    "LR": lr_omni, "df": df_omni, "p": p_omni,
                    "Flagged": p_omni < alpha,
                }

            if model == "global":
                table = pd.DataFrame(
                    {
                        "Group": focal,
                        "Reference": ref_sev,
                        "Focal": focal_sev,
                        "Focal (purified)": focal_shifted,
                        "Difference": diff,
                        "SE": se,
                        "z": z,
                        "p": p,
                        "p (corrected)": p_corrected,
                        "Selected": eligible.isin(selected),
                        "Flagged": flagged,
                    },
                    index=eligible,
                )
                table.index.name = self.facet.capitalize()
                if welch:
                    table["df"] = rater_welch_df
                all_rows.append(table)

                if plot and anchor_selection_dict[focal] is not None:
                    plot_dict[focal] = self.plot_anchor_selection(
                        anchor_selection_dict[focal], **(plot_kwargs or {})
                    )
                else:
                    plot_dict[focal] = None

            focal_models[focal] = focal_model

        self.drf_model = model
        self.drf_table = pd.concat(all_rows)
        self.drf_threshold_table = (
            pd.concat(all_threshold_rows) if all_threshold_rows else None
        )
        self.drf_omnibus_table = pd.DataFrame(omnibus_rows).T if omnibus else None
        self.drf_excluded_raters = excluded_dict
        self.drf_reference = reference
        self.drf_covariate = covariate
        self.drf_reference_model = ref_model
        self.drf_focal_models = focal_models
        self.drf_tc = tc_dict
        self.drf_anchor_selection = anchor_selection_dict
        self.drf_plots = plot_dict
        self.drf_group_sizes = group_sizes

    def model_selection(
        self,
        test="AIC",
        models=None,
        aic_sig_test=True,
        alpha=0.05,
        min_effect=0,
        sampling="dynamic",
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        constant=None,
        threshold_constant=None,
        method=None,
        robust=False,
        matrix_power=None,
        log_lik_tol=0.000001,
        seed=None,
    ):
        """
        Compare MFRM facet element (e.g. rater) parameterisations using AIC, BIC, or LR.

        Calibrates the active models (all eight -- global, items, thresholds,
        bivector, matrix, centrality, pseudo_halo, bistretch -- unless
        models= restricts this) if not already done, computes the
        log-likelihood for each, and ranks them by the chosen criterion.
        centrality and pseudo_halo are each 2-parameter restrictions of
        thresholds/items respectively (Jin & Wang 2018-style facet element
        centrality/extremity, and a pseudo-halo-effect item-difficulty-
        compression model); bistretch is a 3-parameter restriction of
        bivector, combining both stretch axes -- see calibrate()'s
        docstring.

        Parameters
        ----------
        test : str, default 'AIC'
            Criterion: 'AIC', 'BIC', or 'LR'. For 'LR', nested pairwise tests
            are computed (global vs each alternative, bivector vs matrix,
            centrality vs thresholds, pseudo_halo vs items, bistretch vs
            bivector/centrality/pseudo_halo/global).
        models : list/tuple of str, or None, default None
            Restrict the comparison to just these model names (each one of
            self._MODELS) instead of the full family of eight -- only these
            are calibrated, and AIC/BIC ranking and LR testing are both
            confined to this set. Useful for a direct two-model comparison
            (e.g. models=['centrality', 'thresholds']) without paying the
            cost of calibrating the other six. Can be longer than 2 as long
            as the models are genuinely nested against each other -- for
            test='LR', only pairs from the given set that are actually
            nested (per the fixed nested_pairs/df_map below) get a row; pairs
            that aren't nested against each other at all (e.g. 'centrality'
            and 'pseudo_halo', which restrict different parent models) are
            silently skipped for LR but still fine for AIC/BIC, which don't
            require nesting. None (default) uses all eight models.
        aic_sig_test : bool, default True
            When True and test='AIC': applies a significance test against a
            baseline model. The baseline is whichever active model has the
            fewest free parameters (ties broken by lowest AIC) -- 'global'
            whenever it is in the active set, since it always has the fewest
            parameters of all eight models, but otherwise the simplest model
            actually available (e.g. centrality/pseudo_halo if models=
            excludes 'global'). p = e^(-|Δ|/2) where Δ = AIC_baseline -
            AIC_best. A more complex model is preferred only if p < alpha;
            otherwise the baseline is retained.
        alpha : float, default 0.05
            Significance level for the AIC relative-likelihood test
            (used only when test='AIC' and aic_sig_test=True).
        min_effect : float, default 0
            Minimum effect size (logits) required to prefer a more complex
            model over a simpler one. The effect between two models is the
            max absolute difference between their fitted location surfaces
            (each model's facet_effects broadcast to a common (facet element, item,
            category) grid -- see _severity_grid -- same units as
            facet_effects_matrix for every model). For test='LR', an
            additional condition alongside the p-value for every reported
            pair: the summary table's 'Preferred' column names the complex
            model only if p < alpha AND effect >= min_effect, otherwise the
            simpler one. For AIC/BIC, applied as a post-hoc walk-back from
            the selected model toward 'global', restricted to the original
            five-model ladder (matrix -> bivector -> {items, thresholds} ->
            global) -- if AIC/BIC selects centrality/pseudo_halo/bistretch
            directly, min_effect does not walk it back further (same scope
            as per_rater_model_selection's own min_effect). Default 0
            disables (never gates). min_effect is optional and a matter of taste: in simulation the
            tests alone classify well, so the gate expresses a judgement of
            practical importance rather than improving accuracy. If wanted,
            0.43 and 0.64 mirror the ETS DIF category boundaries (moderate
            and large).
        sampling : None, 'dynamic', or int, default 'dynamic'
            Subsamples non-extreme persons before LL computation (parameters
            estimated on full data; only the LL evaluation is subsampled).
            'dynamic' uses T = min(20*R*(I-1)*(K-1), 1500), K = max_score
            (K - 1 taken as 1 for dichotomous data); an integer fixes T
            directly; None disables. n_ll (= T when triggered) is used in
            BIC's ln(n) term.
        warm_corr, tolerance, max_iters, ext_score_adjustment : floats
            Person estimation kwargs.
        constant, method, matrix_power, log_lik_tol : floats
            Calibration kwargs.
        seed : int or None, default None
            Seed for the person subsampling RNG (only used when sampling
            triggers). Pass an int for reproducible LL evaluation; None
            (default) draws fresh entropy.

        Attributes set (AIC with aic_sig_test=True)
        --------------------------------------------
        model_comparison_mfrm_aic : dict  {model: AIC value}
        model_comparison_mfrm_aic_baseline : str  model tested against (fewest free parameters in the active set)
        model_comparison_mfrm_aic_p : float  relative likelihood p-value vs baseline
        model_comparison_mfrm_aic_preferred : str  preferred model name
        model_comparison_mfrm_aic_summary : pd.DataFrame  ranked comparison table (ΔAIC is vs baseline)

        Attributes set (BIC)
        --------------------
        model_comparison_mfrm_bic : dict  {model: BIC value}
        model_comparison_mfrm_bic_preferred : str
        model_comparison_mfrm_bic_summary : pd.DataFrame

        Attributes set (LR)
        -------------------
        model_comparison_mfrm_lr : dict  {pair_label: {LR, df, p-value, Effect, Preferred}}
        model_comparison_mfrm_lr_summary : pd.DataFrame  Effect and Preferred columns
            reflect min_effect (see above); Preferred is per-pair, not an
            overall selected model.
        """
        models = None if models is None else [self._MODEL_ALIASES.get(m, m) for m in models]
        if test not in ("LR", "AIC", "BIC"):
            raise ValueError("test must be 'LR', 'AIC', or 'BIC'")
        if sampling is not None and sampling != "dynamic" and not isinstance(sampling, int):
            raise ValueError("sampling must be None, 'dynamic', or an integer")

        if models is None:
            active_models = list(self._MODELS)
        else:
            active_models = list(models)
            unknown = set(active_models) - set(self._MODELS)
            if unknown:
                raise ValueError(
                    f"models contains unrecognised model name(s) {sorted(unknown)}; "
                    f"must be a subset of {self._MODELS}."
                )
            if len(active_models) != len(set(active_models)):
                raise ValueError("models must not contain duplicate entries.")
            if len(active_models) < 2:
                raise ValueError("models must contain at least 2 model names.")
        active_models = self._drop_equivalent_models(active_models, "model_selection()")
        if len(active_models) < 2:
            raise ValueError("models must contain at least 2 distinct models for this design.")

        n_items = self.no_of_items
        m = self.max_score
        R = self.no_of_facet_elements

        # Free parameter counts for each model
        shared_k = (n_items - 1) + (m - 1)
        rater_k = {
            "global":     R - 1,
            "items":      n_items * (R - 1),
            "thresholds": m * (R - 1),
            "bivector":   (n_items + m - 1) * (R - 1),
            "matrix":     n_items * m * (R - 1),
            "centrality": 2 * (R - 1),
            "pseudo_halo":       2 * (R - 1),
            "bistretch":  3 * (R - 1),
        }
        k_model = {mod: shared_k + rater_k[mod] for mod in active_models}

        # Calibrate and person-estimate each active model
        cal_kw = dict(constant=constant, threshold_constant=threshold_constant, method=method, robust=robust,
                      matrix_power=matrix_power, log_lik_tol=log_lik_tol)
        pe_kw = dict(warm_corr=warm_corr, tolerance=tolerance,
                     max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)
        for mod in active_models:
            if not hasattr(self, f"facet_effects_{mod}"):
                self.calibrate(model=mod, **cal_kw)
            if not hasattr(self, f"persons_{mod}"):
                self.person_estimates(model=mod, **pe_kw)

        # Non-extreme persons (aggregate total score across all raters)
        total_scores = self.responses.groupby(level=1).sum().sum(axis=1)
        max_possible = (
            self.responses.notna().groupby(level=1).sum().sum(axis=1) * m
        )
        non_extreme_mask = (total_scores > 0) & (total_scores < max_possible)
        n_persons = int(non_extreme_mask.sum())

        # Determine LL responses (optional subsampling of persons)
        ll_responses = None
        n_ll = n_persons
        if sampling is not None:
            T = (
                min(20 * R * (n_items - 1) * max(m - 1, 1), 1500)
                if sampling == "dynamic"
                else int(sampling)
            )
            if n_persons > T:
                rng = np.random.default_rng(seed)
                non_extreme_persons = total_scores.index[non_extreme_mask]
                sampled_persons = rng.choice(non_extreme_persons, size=T, replace=False)
                ll_responses = self.responses.loc[
                    self.responses.index.get_level_values(1).isin(sampled_persons)
                ]
                n_ll = T

        # Compute LL for each active model
        ll = {mod: self._log_likelihood(model=mod, responses=ll_responses)
              for mod in active_models}

        if test == "AIC":
            aic = {mod: 2 * k_model[mod] - 2 * ll[mod] for mod in active_models}
            best_mod = min(aic, key=aic.__getitem__)
            self.model_comparison_mfrm_aic = aic

            summary_rows = {
                mod: {"LL": ll[mod], "k": k_model[mod], "AIC": aic[mod]}
                for mod in active_models
            }
            summary_df = (
                pd.DataFrame(summary_rows).T
                .sort_values("AIC")
                .rename_axis("Model")
            )

            active_set = set(active_models)
            if aic_sig_test:
                baseline = min(active_models, key=lambda mod: (k_model[mod], aic[mod]))
                delta = aic[baseline] - aic[best_mod]
                aic_p = float(np.exp(-abs(delta) / 2))
                preferred = best_mod if (delta > 0 and aic_p < alpha) else baseline
                preferred = self._walk_back_effect(preferred, active_set, min_effect)
                self.model_comparison_mfrm_aic_baseline = baseline
                self.model_comparison_mfrm_aic_p = aic_p
                self.model_comparison_mfrm_aic_preferred = preferred
                summary_df["ΔAIC"] = summary_df["AIC"] - aic[baseline]
                self.model_comparison_mfrm_aic_summary = summary_df
            else:
                preferred = self._walk_back_effect(best_mod, active_set, min_effect)
                self.model_comparison_mfrm_aic_preferred = preferred
                self.model_comparison_mfrm_aic_summary = summary_df

        elif test == "BIC":
            bic = {mod: k_model[mod] * np.log(n_ll) - 2 * ll[mod] for mod in active_models}
            best_mod = min(bic, key=bic.__getitem__)
            preferred = self._walk_back_effect(best_mod, set(active_models), min_effect)
            self.model_comparison_mfrm_bic = bic
            self.model_comparison_mfrm_bic_preferred = preferred
            self.model_comparison_mfrm_bic_summary = (
                pd.DataFrame(
                    {mod: {"LL": ll[mod], "k": k_model[mod], "BIC": bic[mod]}
                     for mod in active_models}
                ).T
                .sort_values("BIC")
                .rename_axis("Model")
            )

        elif test == "LR":
            # Nested pairs: global < items, global < thresholds, global < bivector,
            # global < matrix, items < bivector, thresholds < bivector, bivector < matrix
            nested_pairs = [
                ("global", "items"),
                ("global", "thresholds"),
                ("items", "bivector"),
                ("thresholds", "bivector"),
                ("bivector", "matrix"),
                ("global", "centrality"),
                ("centrality", "thresholds"),
                ("global", "pseudo_halo"),
                ("pseudo_halo", "items"),
                ("global", "bistretch"),
                ("centrality", "bistretch"),
                ("pseudo_halo", "bistretch"),
                ("bistretch", "bivector"),
            ]
            df_map = {
                ("global", "items"):        (n_items - 1) * (R - 1),
                ("global", "thresholds"):   (m - 1) * (R - 1),
                ("items", "bivector"):      (m - 1) * (R - 1),
                ("thresholds", "bivector"): (n_items - 1) * (R - 1),
                ("bivector", "matrix"):     (n_items - 1) * (m - 1) * (R - 1),
                ("global", "centrality"):        (R - 1),
                ("centrality", "thresholds"):    (m - 2) * (R - 1),
                ("global", "pseudo_halo"):                        (R - 1),
                ("pseudo_halo", "items"):                         (n_items - 2) * (R - 1),
                ("global", "bistretch"):          2 * (R - 1),
                ("centrality", "bistretch"):      (R - 1),
                ("pseudo_halo", "bistretch"):      (R - 1),
                ("bistretch", "bivector"):        (n_items + m - 4) * (R - 1),
            }
            active_set = set(active_models)
            nested_pairs = [
                (n, a) for n, a in nested_pairs if n in active_set and a in active_set
            ]
            if not nested_pairs:
                warnings.warn(
                    f"No pair within models={active_models} is nested against "
                    f"the other (per the fixed nested_pairs list), so there's "
                    f"nothing for test='LR' to report. AIC/BIC don't require "
                    f"nesting and would work for this set.",
                    UserWarning,
                    stacklevel=2,
                )
            lr_results = {}
            for null_mod, alt_mod in nested_pairs:
                lr_stat = -2 * (ll[null_mod] - ll[alt_mod])
                if lr_stat < 0:
                    warnings.warn(
                        f"LR statistic for {null_mod} vs {alt_mod} is negative "
                        f"(PAIR approximation) and has been floored at 0.",
                        UserWarning,
                    )
                    lr_stat = 0.0
                df_val = df_map[(null_mod, alt_mod)]
                p_val = float(chi2.sf(lr_stat, df_val))
                effect = self._model_effect(null_mod, alt_mod)
                preferred = alt_mod if (p_val < alpha and effect >= min_effect) else null_mod
                label = f"{null_mod} vs {alt_mod}"
                lr_results[label] = {
                    "LR": lr_stat, "df": df_val, "p-value": p_val,
                    "Effect": effect, "Preferred": preferred,
                }

            self.model_comparison_mfrm_lr = lr_results
            self.model_comparison_mfrm_lr_summary = (
                pd.DataFrame(lr_results).T.rename_axis("Comparison")
            )

    def _rater_ll_from_sev(self, sev_rik, persons, responses_r, anchor=False):
        """
        Log-likelihood contribution for one facet element (e.g. rater) given an (I, m) facet_effect array.

        sev_rik : ndarray shape (I, m)
            Per-threshold facet_effect increments in matrix format — same as the
            values stored in facet_effects_matrix for one facet element. cumsum is
            applied per item inside this function, matching _cat_probs_mfrm.
        persons : pd.Series
            Person person_location estimates, already filtered to non-extreme persons.
        responses_r : pd.DataFrame shape (N, I)
            All responses for this facet element.
        anchor : bool
            sev_rik is in the anchored frame of reference: use the anchored
            matrix item locations and thresholds, which absorb the anchors'
            mean effects.
        """
        thresholds = self._thresholds_for("matrix", anchor=anchor).values          # (m,)
        cumtau = np.concatenate([[0.0], np.cumsum(thresholds)])  # (m+1,)
        diff_arr = self._items_for("matrix", anchor=anchor).values                  # (I,)
        cats = np.arange(len(thresholds) + 1, dtype=float)  # (m+1,)
        ab = persons.values                           # (N,)
        obs = responses_r.reindex(persons.index).values  # (N, I)

        log_num = np.zeros((len(thresholds) + 1, len(ab), len(diff_arr)))
        for j in range(len(diff_arr)):
            ct = cumtau + np.concatenate([[0.0], np.cumsum(sev_rik[j])])
            log_num[:, :, j] = cats[:, None] * (ab[None, :] - diff_arr[j]) - ct[:, None]

        log_num -= log_num.max(axis=0, keepdims=True)
        probs = np.exp(log_num)
        probs /= probs.sum(axis=0, keepdims=True)  # (K+1, N, I)

        valid = ~np.isnan(obs)
        obs_int = np.where(valid, obs, 0).astype(int)
        n_idx, i_idx = np.meshgrid(
            np.arange(obs.shape[0]), np.arange(obs.shape[1]), indexing="ij"
        )
        prob_obs = probs[obs_int, n_idx, i_idx]
        prob_obs[~valid] = np.nan
        return float(np.nansum(np.log(prob_obs)))

    def per_rater_model_selection(
        self,
        anchors=None,
        test='AIC',
        models=None,
        aic_sig_test=True,
        sampling='dynamic',
        alpha=0.05,
        min_effect=0,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        constant=None,
        threshold_constant=None,
        method=None,
        robust=False,
        matrix_power=None,
        log_lik_tol=0.000001,
        seed=None,
    ):
        """
        Assign the simplest adequate facet element (e.g. rater) parameterisation to each facet element.

        Calibrates the matrix model once (full data), derives all restricted
        parameterisations as marginal means of each facet element's matrix parameters,
        then tests top-down per facet element — no refitting at any step.

        Each facet element's LL is evaluated over all observations made by that facet element
        (whether persons were single- or multi-marked), filtered to persons
        non-extreme on that facet element's own data. Person person_locations come from the
        full-data matrix-model estimates.

        Testing ladder (top-down):
          matrix → bivector  (df = (I-1)*(m-1) per facet element)
            if rejected  → assign matrix
            if accepted  → test both forks simultaneously:
              bivector → items       (df = m-1)
              bivector → thresholds  (df = I-1)
              both pass  → global
              items only → items
              thresh only → thresholds
              neither    → bivector
          Then, one extra targeted stretch test depending on which of the
          five models above won (the simplified design used in place of a
          full recursive lattice-walk over all eight models -- see
          raschpy_pseudo_extended_mfrm_method.md for the full rationale,
          including the two accepted "shortcuts"):
              matrix winner      → no extra test
              bivector winner    → bivector vs bistretch   (df = I+m-4)
              thresholds winner  → thresholds vs centrality (df = m-2)
              items winner       → items vs pseudo_halo     (df = I-2)
              global winner      → global vs centrality AND global vs
                                    pseudo_halo (df = 1 each); if either/
                                    both beat global, that winner is
                                    assigned, or (if both win) resolved via
                                    a further global vs bistretch test
                                    (df = 2), falling back to whichever
                                    single axis had the stronger evidence
                                    if bistretch itself isn't confirmed.
          A more complex model at each extra rung is kept only if it's both
          significant (p < alpha) and clears min_effect.

        Derivations from matrix params σ_{r,i,k} (all via marginal means):
          bivector    : λ_ri· + λ_r·k  where λ_ri· = mean_k σ_{r,i,k},
                        λ_r·k = mean_i σ_{r,i,k} − μ_r  (zero-summed)
          items       : λ_ri·  replicated across thresholds
          thresholds  : mean_i σ_{r,i,k}  replicated across items
          global      : μ_r = mean_{i,k} σ_{r,i,k}  (scalar, replicated)
          centrality  : 2-param stretch restriction of the thresholds
                        vector (λ_r + (ω_r-1)·τ_k), via _derive_stretch_single.
          pseudo_halo : 2-param stretch restriction of the items vector
                        (λ_r + (ω_r-1)·δ_i), via _derive_stretch_single.
          bistretch   : 3-param stretch restriction of bivector, reusing
                        bivector's own λ_ri·/λ_r·k split exactly as
                        _estimate_raters_bistretch does.

        Parameters
        ----------
        anchors : list or None, default None
            Facet element identifiers to use as the anchor frame of reference. When
            provided, the matrix model is calibrated in anchor mode (anchor
            facet elements' mean facet_effect fixed to zero) before testing. Different
            anchor sets give different — all valid — per-facet-element assignments,
            because the zero-sum constraint means a non-uniform facet element can
            induce apparent non-uniformity in others. All facet elements (anchor and
            non-anchor) are tested; results are stored as
            anchor_facet_effects_mixed and anchor_rater_models.
        test : {'AIC', 'BIC', 'LR'}, default 'AIC'
            Model selection criterion. AIC and BIC pick the minimum across all
            eight models directly. LR uses the top-down ladder (matrix→bivector,
            then fork to items/thresholds/global) plus the extra targeted
            stretch test described above, with significance level alpha.
        models : list/tuple of str, or None, default None
            Restrict the candidate models each facet element can be assigned instead
            of the full family of eight. For AIC/BIC this is a direct filter
            -- the minimum is taken only over the given models (any non-empty
            subset works; aic_sig_test=True tests the winner against
            whichever active model has the fewest free parameters, same as
            model_selection() -- see aic_sig_test below). For LR, {'global', 'items', 'thresholds',
            'bivector'} must all be included -- they form the mandatory
            backbone the fork can't work without -- while 'matrix' and each
            of 'centrality'/'pseudo_halo'/'bistretch' are independently
            optional: excluding 'matrix' skips the top rung entirely (starts
            straight at the fork, as if bivector had already won it);
            excluding a stretch model skips just its corresponding extra-rung
            test, so the ladder stops one rung earlier along that branch.
            None (default) uses all eight models, exactly as before this
            parameter existed.
        aic_sig_test : bool, default True
            When test='AIC', prefer a model more complex than the baseline
            (whichever active model has the fewest free parameters, ties
            broken by lowest AIC -- 'global' whenever it is active, since it
            always has the fewest parameters of all eight models) only if
            ΔAIC > 0 (vs baseline) AND p = exp(-ΔAIC/2) < alpha. Prevents
            spurious complexity at large n where ΔAIC differences are small.
        sampling : 'dynamic', int, or None, default 'dynamic'
            Per-facet-element subsampling for LL evaluation. 'dynamic' uses
            T = min(20*(I-1)*(m-1), 1500); an integer fixes T directly;
            None uses all of each facet element's persons. Parameters are always
            estimated on the full data; only LL evaluation is subsampled.
            n_r (= T when triggered) is used in BIC's ln(n) term.
        seed : int or None, default None
            Seed for the per-facet-element subsampling RNG (only used when sampling
            triggers). Pass an int for reproducible LL evaluation; None
            (default) draws fresh entropy. Note: previously this subsampling
            was hardcoded to random_state=0 (always deterministic); passing
            seed=0 restores that exact prior behaviour.
        alpha : float, default 0.05
            Significance level. For LR: threshold for accepting a restriction.
            For AIC with aic_sig_test=True: threshold for the relative likelihood.
        min_effect : float, default 0
            Minimum effect size (logits) required to retain a more complex model
            at each step. The effect is the max absolute difference between the
            complex and simple model's facet_effect surfaces at that rung:
              mat→biv: max|interaction terms|
              biv→items: max|threshold shape λ_r·k|
              biv→thresholds: max|item deviations λ_ri·|
              items→global: max|λ_ri·|
              thresholds→global: max|λ_r·k|
              biv→bistretch, thr→centrality, itm→pseudo_halo, glb→centrality,
              glb→pseudo_halo, glb→bistretch: max|complex_sev - simple_sev|
              at that extra rung.
            For AIC/BIC the gate is applied as a post-hoc walk-back from the
            selected model toward global (only along the original five-model
            ladder -- if AIC/BIC directly selects centrality/pseudo_halo/
            bistretch as the global best, min_effect does not walk it back
            further). For LR it is an additional condition at each ladder
            step (including the extra rung) alongside the p-value. Default 0
            disables. min_effect is optional and a matter of taste: in simulation the
            tests alone classify well, so the gate expresses a judgement of
            practical importance rather than improving accuracy. If wanted,
            0.43 and 0.64 mirror the ETS DIF category boundaries (moderate
            and large).
        warm_corr, tolerance, max_iters, ext_score_adjustment : floats
            Person estimation kwargs (for matrix model).
        constant, method, matrix_power, log_lik_tol : floats
            Calibration kwargs.

        Attributes set (anchors=None)
        -----------------------------
        rater_models : pd.Series
        facet_effects_mixed : pd.DataFrame
        per_rater_model_selection_table : pd.DataFrame
        per_rater_model_selection_counts : pd.Series

        Attributes set (anchors provided)
        ----------------------------------
        anchor_rater_models : pd.Series
        anchor_facet_effects_mixed : pd.DataFrame
        anchor_per_rater_model_selection_table : pd.DataFrame
        anchor_per_rater_model_selection_counts : pd.Series

        Attributes set (always)
        ------------------------
        per_rater_model_selection_runs : dict
            Every call's rater_models/facet_effects_mixed/table/counts, keyed
            by tuple(anchors) (or None for an unanchored call), so results
            from earlier anchors= calls survive later ones with different
            anchor sets. E.g. mfrm.per_rater_model_selection_runs[tuple(my_anchors_1)].table.
        """
        models = None if models is None else [self._MODEL_ALIASES.get(m, m) for m in models]
        n_items = self.no_of_items
        m = self.max_score
        anchored = anchors is not None

        # dichotomous data: only items and pseudo_halo vary by item (see _drop_equivalent_models)
        backbone = {"global", "items"} if m == 1 else {"global", "items", "thresholds", "bivector"}
        if models is None:
            active_models = set(self._MODELS)
        else:
            active_models = set(models)
            unknown = active_models - set(self._MODELS)
            if unknown:
                raise ValueError(
                    f"models contains unrecognised model name(s) {sorted(unknown)}; "
                    f"must be a subset of {self._MODELS}."
                )
            if not active_models:
                raise ValueError("models must not be empty.")
            if test == "LR" and not backbone.issubset(active_models):
                raise ValueError(
                    f"For test='LR', models= must include the full backbone "
                    f"{sorted(backbone)} -- the top-down fork can't work "
                    f"without them. The other models are independently optional. "
                    f"AIC/BIC don't have this restriction."
                )
        active_models = self._drop_equivalent_models(active_models, "per_rater_model_selection()")
        matrix_allowed = "matrix" in active_models
        centrality_allowed = "centrality" in active_models
        pseudo_halo_allowed = "pseudo_halo" in active_models
        bistretch_allowed = "bistretch" in active_models

        cal_kw = dict(constant=constant, threshold_constant=threshold_constant, method=method, robust=robust,
                      matrix_power=matrix_power, log_lik_tol=log_lik_tol)
        pe_kw = dict(warm_corr=warm_corr, tolerance=tolerance,
                     max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)

        if anchored:
            # Calibrate matrix model then apply anchor adjustment (one pass).
            # Re-run if the matrix model hasn't been anchor-calibrated yet, or
            # was calibrated against a different anchor set.
            stored = getattr(self, "anchor_rater_names_matrix", None)
            stale = stored is None or set(stored) != set(anchors)
            if stale:
                self.calibrate_anchor(
                    model="matrix", anchors=anchors, calibrate=True, **cal_kw
                )
                self.person_estimates(model="matrix", anchor=True, **pe_kw)
            elif not hasattr(self, "anchor_persons_matrix"):
                # anchor_rater_names_matrix matches, but person estimates were
                # never computed for it (e.g. calibrate_anchor called directly).
                self.person_estimates(model="matrix", anchor=True, **pe_kw)
            sev_source = self.anchor_facet_effects_matrix
            persons_all = self.anchor_persons_matrix
        else:
            if not hasattr(self, "facet_effects_matrix"):
                self.calibrate(model="matrix", **cal_kw)
            if not hasattr(self, "persons_matrix"):
                self.person_estimates(model="matrix", **pe_kw)
            sev_source = self.facet_effects_matrix
            persons_all = self.persons_matrix

        records = {}
        for rater in self.facet_names:
            # All observations made by this rater, single- or multi-marked
            responses_r = self.responses.loc[rater]  # (N_rater, I)

            # Non-extreme filter on this rater's own data
            total_r = responses_r.sum(axis=1)
            max_r = responses_r.notna().sum(axis=1) * m
            non_extreme_r = (total_r > 0) & (total_r < max_r)
            persons_r = persons_all.reindex(
                total_r.index[non_extreme_r]
            ).dropna()
            persons_r = persons_r[persons_r.abs() <= 20]

            # Dynamic subsampling for LL evaluation (parameters from full data)
            if sampling == 'dynamic':
                T = min(20 * (n_items - 1) * max(m - 1, 1), 1500)
            elif isinstance(sampling, int):
                T = sampling
            else:
                T = None
            if T is not None and len(persons_r) > T:
                persons_r = persons_r.sample(T, random_state=seed)

            if len(persons_r) == 0:
                records[rater] = {"selected_model": "items" if m == 1 else "matrix",
                                  "LL_matrix": np.nan, "LL_bivector": np.nan,
                                  "LL_items": np.nan, "LL_thresholds": np.nan,
                                  "LL_global": np.nan}
                continue

            # Matrix params for this rater: shape (I, m)
            sev_mat = sev_source.loc[rater].values

            # Marginal decomposition
            item_means = sev_mat.mean(axis=1)          # λ_ri·  (I,)
            thresh_means = sev_mat.mean(axis=0)         # mean_i σ_{r,i,k}  (m,)
            mu_r = sev_mat.mean()                       # overall scalar

            lambda_rk = thresh_means - mu_r                          # zero-summed threshold shape (m,)
            sev_biv = item_means[:, None] + lambda_rk[None, :]
            sev_items = np.tile(item_means[:, None], (1, m))
            sev_thresh = np.tile(thresh_means[None, :], (n_items, 1))
            sev_global = np.full((n_items, m), mu_r)

            # Step-wise effect sizes: max|complex_sev - simple_sev| at each rung
            eff_mat_biv = float(np.max(np.abs(sev_mat - sev_biv)))   # interaction
            eff_biv_itm = float(np.max(np.abs(lambda_rk)))            # threshold shape
            eff_biv_thr = float(np.max(np.abs(item_means - mu_r)))    # item deviations
            # items→global and thresholds→global reuse the same quantities
            eff_itm_glb = eff_biv_thr
            eff_thr_glb = eff_biv_itm

            # 2/3-param stretch restrictions -- closed-form from the same
            # marginal vectors, same recipe _estimate_raters_centrality/
            # _pseudo_halo/_bistretch use, via _derive_stretch_single (the
            # single-rater wrapper around _derive_stretch_model). centrality
            # restricts thresholds (deviation = thresh_means, the direct
            # "thresholds" model values); pseudo_halo restricts items
            # (deviation = item_means); bistretch restricts bivector, reusing
            # bivector's own item_means/lambda_rk split exactly as
            # _estimate_raters_bistretch does.
            _, omega_cen, cen_1d = self._derive_stretch_single(
                thresh_means, self._thresholds_for("matrix", anchor=anchored)
            )
            _, omega_pha, pha_1d = self._derive_stretch_single(
                item_means, self._items_for("matrix", anchor=anchored)
            )
            lam_bis, omega_items_bis, item_1d_bis = self._derive_stretch_single(
                item_means, self._items_for("matrix", anchor=anchored)
            )
            _, omega_thresholds_bis, thr_1d_bis = self._derive_stretch_single(
                lambda_rk, self._thresholds_for("matrix", anchor=anchored)
            )
            sev_centrality = np.tile(cen_1d[None, :], (n_items, 1))
            sev_pseudo_halo = np.tile(pha_1d[:, None], (1, m))
            sev_bistretch = item_1d_bis[:, None] + thr_1d_bis[None, :]

            eff_thr_cen = float(np.max(np.abs(sev_thresh - sev_centrality)))
            eff_itm_pha = float(np.max(np.abs(sev_items - sev_pseudo_halo)))
            eff_biv_bis = float(np.max(np.abs(sev_biv - sev_bistretch)))
            eff_glb_cen = float(np.max(np.abs(sev_global - sev_centrality)))
            eff_glb_pha = float(np.max(np.abs(sev_global - sev_pseudo_halo)))
            eff_glb_bis = float(np.max(np.abs(sev_global - sev_bistretch)))

            def _ll(sev, mod):
                return self._rater_ll_from_sev(sev, persons_r, responses_r, anchored)
            ll_mat = _ll(sev_mat, "matrix")
            ll_biv = _ll(sev_biv, "bivector")
            ll_itm = _ll(sev_items, "items")
            ll_thr = _ll(sev_thresh, "thresholds")
            ll_glb = _ll(sev_global, "global")
            ll_cen = _ll(sev_centrality, "centrality")
            ll_pha = _ll(sev_pseudo_halo, "pseudo_halo")
            ll_bis = _ll(sev_bistretch, "bistretch")

            rec = {
                "LL_matrix": ll_mat, "LL_bivector": ll_biv,
                "LL_items": ll_itm, "LL_thresholds": ll_thr, "LL_global": ll_glb,
                "LL_centrality": ll_cen, "LL_pseudo_halo": ll_pha, "LL_bistretch": ll_bis,
            }

            # Per-rater free parameter counts (rater side only)
            k_map = {"global": 1, "items": n_items, "thresholds": m,
                     "bivector": n_items + m - 1, "matrix": n_items * m,
                     "centrality": 2, "pseudo_halo": 2, "bistretch": 3}
            n_r = len(persons_r)  # subsampled n when sampling triggered

            if test in ("AIC", "BIC"):
                ll_map = {"global": ll_glb, "items": ll_itm, "thresholds": ll_thr,
                          "bivector": ll_biv, "matrix": ll_mat,
                          "centrality": ll_cen, "pseudo_halo": ll_pha, "bistretch": ll_bis}
                ll_map = {mod: ll for mod, ll in ll_map.items() if mod in active_models}
                if test == "AIC":
                    ic = {mod: 2 * k_map[mod] - 2 * ll for mod, ll in ll_map.items()}
                else:
                    ic = {mod: k_map[mod] * np.log(n_r) - 2 * ll
                          for mod, ll in ll_map.items()}
                rec.update({f"{test}_{mod}": v for mod, v in ic.items()})
                best = min(ic, key=ic.get)
                if test == "AIC" and aic_sig_test:
                    baseline = min(active_models, key=lambda mod: (k_map[mod], ic[mod]))
                    delta = ic[baseline] - ic[best]
                    p_rel = float(np.exp(-abs(delta) / 2))
                    rec["p_vs_baseline"] = p_rel
                    rec["baseline_model"] = baseline
                    selected = best if (delta > 0 and p_rel < alpha) else baseline
                else:
                    selected = best
                # Effect size walk-back: simplify one rung at a time if effect too small
                if min_effect > 0 and selected != "global":
                    if selected == "matrix" and eff_mat_biv < min_effect:
                        selected = "bivector"
                    if selected == "bivector":
                        # small threshold shape → simplify to items
                        # small item deviation → simplify to thresholds
                        thresh_shape_ok = eff_biv_itm >= min_effect
                        item_dev_ok = eff_biv_thr >= min_effect
                        if not thresh_shape_ok and not item_dev_ok:
                            selected = "global"
                        elif not thresh_shape_ok:
                            selected = "items"  # drop threshold variation
                        elif not item_dev_ok:
                            selected = "thresholds"  # drop item variation
                        # both ok → stay bivector
                    if selected == "items" and eff_itm_glb < min_effect:
                        selected = "global"
                    if selected == "thresholds" and eff_thr_glb < min_effect:
                        selected = "global"

            elif m == 1:  # LR, dichotomous: global -> items, then the pseudo_halo restriction
                lr_gi = max(0.0, -2.0 * (ll_glb - ll_itm))
                p_gi = float(chi2.sf(lr_gi, n_items - 1))
                rec.update({"LR_glb_vs_items": lr_gi, "p_glb_vs_items": p_gi})
                if p_gi < alpha and eff_itm_glb >= min_effect:
                    selected = "items"
                    if pseudo_halo_allowed:
                        lr_pha = max(0.0, -2.0 * (ll_pha - ll_itm))
                        p_pha = float(chi2.sf(lr_pha, n_items - 2))
                        rec.update({"LR_itm_vs_pha": lr_pha, "p_itm_vs_pha": p_pha})
                        if not (p_pha < alpha and eff_itm_pha >= min_effect):
                            selected = "pseudo_halo"
                else:
                    selected = "global"
                    if pseudo_halo_allowed:
                        lr_gh = max(0.0, -2.0 * (ll_glb - ll_pha))
                        p_gh = float(chi2.sf(lr_gh, 1))
                        rec.update({"LR_glb_vs_pha": lr_gh, "p_glb_vs_pha": p_gh})
                        if p_gh < alpha and eff_glb_pha >= min_effect:
                            selected = "pseudo_halo"

            else:  # LR — top-down ladder
                rec.update({"LR_biv_vs_matrix": np.nan, "p_biv_vs_matrix": np.nan,
                            "LR_items_vs_biv": np.nan, "p_items_vs_biv": np.nan,
                            "LR_thresh_vs_biv": np.nan, "p_thresh_vs_biv": np.nan})
                if matrix_allowed:
                    lr_biv = max(0.0, -2.0 * (ll_biv - ll_mat))
                    p_biv = float(chi2.sf(lr_biv, (n_items - 1) * (m - 1)))
                    rec.update({"LR_biv_vs_matrix": lr_biv, "p_biv_vs_matrix": p_biv})
                else:
                    p_biv = 1.0  # 'matrix' not offered -- always fall through to the fork
                if matrix_allowed and (p_biv < alpha or eff_mat_biv < min_effect):
                    selected = "matrix" if p_biv < alpha else "bivector"
                else:
                    lr_itm = max(0.0, -2.0 * (ll_itm - ll_biv))
                    lr_thr = max(0.0, -2.0 * (ll_thr - ll_biv))
                    p_itm = float(chi2.sf(lr_itm, m - 1))
                    p_thr = float(chi2.sf(lr_thr, n_items - 1))
                    rec.update({"LR_items_vs_biv": lr_itm, "p_items_vs_biv": p_itm,
                                "LR_thresh_vs_biv": lr_thr, "p_thresh_vs_biv": p_thr})
                    # items_ok: threshold shape not significant OR not substantial → items model adequate
                    # thresh_ok: item deviation not significant OR not substantial → thresholds model adequate
                    items_ok = p_itm >= alpha or eff_biv_itm < min_effect
                    thresh_ok = p_thr >= alpha or eff_biv_thr < min_effect
                    if items_ok and thresh_ok:
                        selected = "global"
                    elif items_ok:
                        # threshold shape trivial, item deviations present → items model
                        selected = "items"
                        if eff_itm_glb < min_effect:
                            selected = "global"
                    elif thresh_ok:
                        # item deviations trivial, threshold shape present → thresholds model
                        selected = "thresholds"
                        if eff_thr_glb < min_effect:
                            selected = "global"
                    else:
                        selected = "bivector"

                # One extra targeted stretch test per ladder-winner branch --
                # the simplified design agreed in place of a full recursive
                # lattice-walk (see raschpy_pseudo_extended_mfrm_method.md).
                # "matrix" winner gets no extra test (bistretch/centrality/
                # pseudo_halo are all restrictions of models matrix already
                # beat). A more complex model is retained only if it's both
                # significant AND clears min_effect.
                if selected == "bivector" and bistretch_allowed:
                    lr_bis = max(0.0, -2.0 * (ll_bis - ll_biv))
                    df_bis = (n_items + m - 1) - 3
                    p_bis = float(chi2.sf(lr_bis, df_bis))
                    rec.update({"LR_biv_vs_bis": lr_bis, "p_biv_vs_bis": p_bis})
                    if not (p_bis < alpha and eff_biv_bis >= min_effect):
                        selected = "bistretch"
                elif selected == "thresholds" and centrality_allowed:
                    lr_cen = max(0.0, -2.0 * (ll_cen - ll_thr))
                    df_cen = m - 2
                    p_cen = float(chi2.sf(lr_cen, df_cen))
                    rec.update({"LR_thr_vs_cen": lr_cen, "p_thr_vs_cen": p_cen})
                    if not (p_cen < alpha and eff_thr_cen >= min_effect):
                        selected = "centrality"
                elif selected == "items" and pseudo_halo_allowed:
                    lr_pha = max(0.0, -2.0 * (ll_pha - ll_itm))
                    df_pha = n_items - 2
                    p_pha = float(chi2.sf(lr_pha, df_pha))
                    rec.update({"LR_itm_vs_pha": lr_pha, "p_itm_vs_pha": p_pha})
                    if not (p_pha < alpha and eff_itm_pha >= min_effect):
                        selected = "pseudo_halo"
                elif selected == "global" and (centrality_allowed or pseudo_halo_allowed):
                    cen_wins = pha_wins = False
                    p_gc = p_gh = None
                    if centrality_allowed:
                        lr_gc = max(0.0, -2.0 * (ll_glb - ll_cen))
                        p_gc = float(chi2.sf(lr_gc, 1))
                        rec.update({"LR_glb_vs_cen": lr_gc, "p_glb_vs_cen": p_gc})
                        cen_wins = p_gc < alpha and eff_glb_cen >= min_effect
                    if pseudo_halo_allowed:
                        lr_gh = max(0.0, -2.0 * (ll_glb - ll_pha))
                        p_gh = float(chi2.sf(lr_gh, 1))
                        rec.update({"LR_glb_vs_pha": lr_gh, "p_glb_vs_pha": p_gh})
                        pha_wins = p_gh < alpha and eff_glb_pha >= min_effect
                    if cen_wins and pha_wins:
                        # Both stretch axes independently beat global --
                        # resolve via a further test against the combined
                        # 3-param bistretch model, if it's offered.
                        if bistretch_allowed:
                            lr_gb = max(0.0, -2.0 * (ll_glb - ll_bis))
                            p_gb = float(chi2.sf(lr_gb, 2))
                            rec.update({"LR_glb_vs_bis": lr_gb, "p_glb_vs_bis": p_gb})
                            if p_gb < alpha and eff_glb_bis >= min_effect:
                                selected = "bistretch"
                            else:
                                # Combined model not itself confirmed --
                                # fall back to whichever single axis had
                                # the stronger evidence.
                                selected = "centrality" if p_gc <= p_gh else "pseudo_halo"
                        else:
                            # bistretch not offered -- fall back directly
                            # to whichever single axis had the stronger
                            # evidence.
                            selected = "centrality" if p_gc <= p_gh else "pseudo_halo"
                    elif cen_wins:
                        selected = "centrality"
                    elif pha_wins:
                        selected = "pseudo_halo"
                    # else: neither axis beats global -- stay "global"

            rec["selected_model"] = selected
            records[rater] = rec

        # Build column order: selected_model first, then LLs, then test-specific
        ll_cols = ["LL_matrix", "LL_bivector", "LL_items", "LL_thresholds", "LL_global",
                   "LL_centrality", "LL_pseudo_halo", "LL_bistretch"]
        if test in ("AIC", "BIC"):
            ic_cols = [f"{test}_{m_}" for m_ in
                       ("matrix", "bivector", "items", "thresholds", "global",
                        "centrality", "pseudo_halo", "bistretch")]
            extra_cols = ic_cols + (["p_vs_baseline", "baseline_model"] if test == "AIC" and aic_sig_test else [])
        else:
            extra_cols = (["LR_glb_vs_items", "p_glb_vs_items"] if m == 1 else []) + [
                          "LR_biv_vs_matrix", "p_biv_vs_matrix",
                          "LR_items_vs_biv", "p_items_vs_biv",
                          "LR_thresh_vs_biv", "p_thresh_vs_biv",
                          "LR_biv_vs_bis", "p_biv_vs_bis",
                          "LR_thr_vs_cen", "p_thr_vs_cen",
                          "LR_itm_vs_pha", "p_itm_vs_pha",
                          "LR_glb_vs_cen", "p_glb_vs_cen",
                          "LR_glb_vs_pha", "p_glb_vs_pha",
                          "LR_glb_vs_bis", "p_glb_vs_bis"]
        cols = ["selected_model"] + ll_cols + extra_cols
        if m == 1:  # dichotomous: only the global/items/pseudo_halo columns are meaningful
            keep = ("global", "items", "pseudo_halo", "glb_vs_items", "itm_vs_pha", "glb_vs_pha",
                    "p_vs_baseline", "baseline_model", "selected_model")
            cols = [c for c in cols if c.endswith(keep)]
        table = pd.DataFrame(records).T.rename_axis("Rater")
        table = table.reindex(columns=cols)
        rater_models = table["selected_model"]

        # Build facet_effects_mixed in matrix format from assigned models
        mi = pd.MultiIndex.from_product(
            [self.facet_names, self.item_names], names=[self.facet, "item"]
        )
        rows = []
        for rater in self.facet_names:
            sev_mat = sev_source.loc[rater].values  # (I, m)
            item_means = sev_mat.mean(axis=1)
            thresh_means = sev_mat.mean(axis=0)
            mu_r = sev_mat.mean()
            lambda_rk = thresh_means - mu_r
            assigned = rater_models[rater]
            if assigned == "matrix":
                sev = sev_mat
            elif assigned == "bivector":
                sev = item_means[:, None] + lambda_rk[None, :]
            elif assigned == "centrality":
                _, _, cen_1d = self._derive_stretch_single(
                    thresh_means, self._thresholds_for("matrix", anchor=anchored)
                )
                sev = np.tile(cen_1d[None, :], (n_items, 1))
            elif assigned == "pseudo_halo":
                _, _, pha_1d = self._derive_stretch_single(
                    item_means, self._items_for("matrix", anchor=anchored)
                )
                sev = np.tile(pha_1d[:, None], (1, m))
            elif assigned == "bistretch":
                _, _, item_1d_bis = self._derive_stretch_single(
                    item_means, self._items_for("matrix", anchor=anchored)
                )
                _, _, thr_1d_bis = self._derive_stretch_single(
                    lambda_rk, self._thresholds_for("matrix", anchor=anchored)
                )
                sev = item_1d_bis[:, None] + thr_1d_bis[None, :]
            elif assigned == "items":
                sev = np.tile(item_means[:, None], (1, m))
            elif assigned == "thresholds":
                sev = np.tile(thresh_means[None, :], (n_items, 1))
            else:  # global
                sev = np.full((n_items, m), mu_r)
            rows.extend(sev)
        mixed_sev = pd.DataFrame(rows, index=mi, columns=range(1, m + 1))

        counts = rater_models.value_counts().rename("Count")

        if anchored:
            self.anchor_rater_models = rater_models
            self.anchor_facet_effects_mixed = mixed_sev
            setattr(self, f"anchor_{self.facets}_mixed", mixed_sev)
            self.anchor_per_rater_model_selection_table = table
            self.anchor_per_rater_model_selection_counts = counts
            # 'mixed' has no calibrate_mixed_anchor() of its own -- item/
            # threshold locations under anchoring are shared across every
            # model regardless (only facet_effects differ per model), same
            # as the unanchored case (_get_params always uses plain
            # self.items/self.thresholds there, no per-model lookup at
            # all). Mirror that here so _get_params('mixed', anchor=True)
            # -- and everything routed through it: plot_data/icc/crcs/etc.
            # -- has anchor_items_mixed/anchor_thresholds_mixed to find.
            self.anchor_items_mixed = self.anchor_items_matrix
            self.anchor_thresholds_mixed = self.anchor_thresholds_matrix
        else:
            self.rater_models = rater_models
            self.facet_effects_mixed = mixed_sev
            setattr(self, f"{self.facets}_mixed", mixed_sev)
            self.per_rater_model_selection_table = table
            self.per_rater_model_selection_counts = counts

        # Snapshot this run keyed by its anchor set (None for unanchored) so
        # results from different anchors= calls can be recovered later
        # instead of being overwritten by the next call.
        from types import SimpleNamespace
        if not hasattr(self, "per_rater_model_selection_runs"):
            self.per_rater_model_selection_runs = {}
        key = tuple(anchors) if anchored else None
        self.per_rater_model_selection_runs[key] = SimpleNamespace(
            rater_models=rater_models,
            facet_effects_mixed=mixed_sev,
            table=table,
            counts=counts,
        )

        return table

    def _apply_rater_models(self, facet_effects_matrix, rater_models, anchor=False):
        """Derive facet_effects_mixed by applying per-facet-element model restrictions to matrix estimates.

        Parameters
        ----------
        facet_effects_matrix : DataFrame
            MultiIndex (facet element (e.g. rater), item) × thresholds, as produced by calibrate(model='matrix').
        rater_models : Series
            Per-facet-element model assignment (index=facet element names, values in _MODELS).
        anchor : bool
            facet_effects_matrix is anchored: derive the stretch models
            against the anchored matrix item locations and thresholds.

        Returns
        -------
        DataFrame
            Same MultiIndex structure as facet_effects_matrix, with each facet element's
            facet_effect values replaced by the marginal-mean projection of their
            assigned model.
        """
        n_items = self.no_of_items
        m = self.max_score
        blocks = []
        for rater in self.facet_names:
            sev_mat = facet_effects_matrix.loc[rater].values  # (I, m)
            assigned = rater_models[rater]
            if assigned == "matrix":
                sev_r = sev_mat
            else:
                item_means = sev_mat.mean(axis=1)    # (I,)
                thresh_means = sev_mat.mean(axis=0)  # (m,)
                mu_r = sev_mat.mean()
                if assigned == "global":
                    sev_r = np.full((n_items, m), mu_r)
                elif assigned == "items":
                    sev_r = np.tile(item_means[:, None], (1, m))
                elif assigned == "thresholds":
                    sev_r = np.tile(thresh_means[None, :], (n_items, 1))
                elif assigned == "bivector":
                    lambda_rk = thresh_means - mu_r
                    sev_r = item_means[:, None] + lambda_rk[None, :]
                elif assigned == "centrality":
                    _, _, cen_1d = self._derive_stretch_single(
                        thresh_means, self._thresholds_for("matrix", anchor=anchor)
                    )
                    sev_r = np.tile(cen_1d[None, :], (n_items, 1))
                elif assigned == "pseudo_halo":
                    _, _, pha_1d = self._derive_stretch_single(
                        item_means, self._items_for("matrix", anchor=anchor)
                    )
                    sev_r = np.tile(pha_1d[:, None], (1, m))
                elif assigned == "bistretch":
                    lambda_rk = thresh_means - mu_r
                    _, _, item_1d_bis = self._derive_stretch_single(
                        item_means, self._items_for("matrix", anchor=anchor)
                    )
                    _, _, thr_1d_bis = self._derive_stretch_single(
                        lambda_rk, self._thresholds_for("matrix", anchor=anchor)
                    )
                    sev_r = item_1d_bis[:, None] + thr_1d_bis[None, :]
            blocks.append(
                pd.DataFrame(
                    sev_r,
                    index=self.responses.columns,
                    columns=range(1, m + 1),
                )
            )
        mi = pd.MultiIndex.from_product(
            [self.facet_names, self.responses.columns], names=[self.facet, "item"]
        )
        return pd.DataFrame(
            np.vstack([b.values for b in blocks]),
            index=mi,
            columns=range(1, m + 1),
        )

    # ------------------------------------------------------------------
    # Residual correlation analysis
    # ------------------------------------------------------------------

    def item_res_corr_analysis(self, std_residual_df):
        """
        Analyse item standardised residual correlations.

        Computes the inter-item correlation matrix of standardised residuals
        and performs PCA to detect violations of local item independence.

        Parameters
        ----------
        std_residual_df : pandas.DataFrame
            Standardised residuals with (Rater, Person) MultiIndex and
            items as columns.

        Returns
        -------
        tuple of (correlations, eigenvectors, eigenvalues, variance_explained, loadings)
            All are DataFrames (or None if PCA fails).
        """
        item_residual_correlations = std_residual_df.corr(numeric_only=False)
        pca = PCA()
        try:
            pca.fit(item_residual_correlations)
            n = (
                self.no_of_items - 1
            )  # rank of correlation matrix is n-1; drop zero eigenvalue
            pc_labels = [f"PC {pc + 1}" for pc in range(n)]
            eigvec_labels = [f"Eigenvector {pc + 1}" for pc in range(self.no_of_items)]
            eigenvectors = pd.DataFrame(
                pca.components_[:n, :], index=pc_labels, columns=eigvec_labels
            )
            eigenvalues = pd.DataFrame(
                pca.explained_variance_[:n], index=pc_labels, columns=["Eigenvalue"]
            )
            variance_explained = pd.DataFrame(
                pca.explained_variance_ratio_[:n],
                index=pc_labels,
                columns=["Variance explained"],
            )
            loadings = pd.DataFrame(
                eigenvectors.values.T * (pca.explained_variance_[:n] ** 0.5),
                index=self.responses.columns,
                columns=pc_labels,
            )
        except Exception:
            warnings.warn(
                "PCA of item standardised residuals failed. "
                "Eigenvectors and loadings set to None.",
                UserWarning,
                stacklevel=2,
            )
            eigenvectors = eigenvalues = variance_explained = loadings = None
        return (
            item_residual_correlations,
            eigenvectors,
            eigenvalues,
            variance_explained,
            loadings,
        )

    def facet_res_corr_analysis(self, residual_df, std_residual_df):
        """
        Analyse facet_element residual correlations.

        Pivots the residual DataFrame to (Person×Items, Facet elements (e.g. raters)) shape,
        computes the inter-facet_element correlation matrix, and performs PCA.
        A large first eigenvalue suggests systematic facet_element bias.

        Parameters
        ----------
        residual_df : pandas.DataFrame
            Raw residuals with (Rater, Person) MultiIndex.
        std_residual_df : pandas.DataFrame
            Standardised residuals with (Rater, Person) MultiIndex.

        Returns
        -------
        tuple of (correlations, eigenvectors, eigenvalues, variance_explained, loadings)
            All are DataFrames (or None if PCA fails).
        """
        rater_res = self.facet_pivot(residual_df)
        rater_std_res = self.facet_pivot(std_residual_df)
        correlations = rater_res.corr(numeric_only=False)
        pca = PCA()
        try:
            pca.fit(rater_std_res.corr(numeric_only=False))
            n = (
                self.no_of_facet_elements - 1
            )  # rank of correlation matrix is n-1; drop zero eigenvalue
            pc_labels = [f"PC {pc + 1}" for pc in range(n)]
            eigvec_labels = [
                f"Eigenvector {pc + 1}" for pc in range(self.no_of_facet_elements)
            ]
            eigenvectors = pd.DataFrame(
                pca.components_[:n, :], index=pc_labels, columns=eigvec_labels
            )
            eigenvalues = pd.DataFrame(
                pca.explained_variance_[:n], index=pc_labels, columns=["Eigenvalue"]
            )
            variance_explained = pd.DataFrame(
                pca.explained_variance_ratio_[:n],
                index=pc_labels,
                columns=["Variance explained"],
            )
            loadings = pd.DataFrame(
                eigenvectors.values.T * (pca.explained_variance_[:n] ** 0.5),
                index=self.facet_names,
                columns=pc_labels,
            )
        except Exception:
            warnings.warn(
                f"PCA of {self.facet_label_eg} standardised residuals failed. "
                "Eigenvectors and loadings set to None.",
                UserWarning,
                stacklevel=2,
            )
            eigenvectors = eigenvalues = variance_explained = loadings = None
        return (correlations, eigenvectors, eigenvalues, variance_explained, loadings)

    def _run_item_res_corr(self, model, **kw):
        """Internal dispatcher: run item residual correlation analysis for the given model."""
        if not hasattr(self, f"std_residual_df_{model}"):
            self.fit_statistics(model=model, **kw)
        results = self.item_res_corr_analysis(getattr(self, f"std_residual_df_{model}"))
        for name, val in zip(
            [
                "item_residual_correlations",
                "item_eigenvectors",
                "item_eigenvalues",
                "item_variance_explained",
                "item_loadings",
            ],
            results,
        ):
            setattr(self, f"{name}_{model}", val)

    def _run_facet_res_corr(self, model, **kw):
        """Internal dispatcher: run facet/rater residual correlation analysis for the given model."""
        if not hasattr(self, f"std_residual_df_{model}"):
            self.fit_statistics(model=model, **kw)
        results = self.facet_res_corr_analysis(
            getattr(self, f"residual_df_{model}"),
            getattr(self, f"std_residual_df_{model}"),
        )
        for name, val in zip(
            [
                "rater_residual_correlations",
                "rater_eigenvectors",
                "rater_eigenvalues",
                "rater_variance_explained",
                "rater_loadings",
            ],
            results,
        ):
            setattr(self, f"{name}_{model}", val)
        self._set_facet_aliases(model)

    def item_res_corr_analysis_global(self, **kw):
        """Alias for item_res_corr_analysis(model='global'). See item_res_corr_analysis for full documentation."""
        self._run_item_res_corr("global", **kw)

    def item_res_corr_analysis_items(self, **kw):
        """Alias for item_res_corr_analysis(model='items'). See item_res_corr_analysis for full documentation."""
        self._run_item_res_corr("items", **kw)

    def item_res_corr_analysis_thresholds(self, **kw):
        """Alias for item_res_corr_analysis(model='thresholds'). See item_res_corr_analysis for full documentation."""
        self._run_item_res_corr("thresholds", **kw)

    def item_res_corr_analysis_matrix(self, **kw):
        """Alias for item_res_corr_analysis(model='matrix'). See item_res_corr_analysis for full documentation."""
        self._run_item_res_corr("matrix", **kw)

    def item_res_corr_analysis_bivector(self, **kw):
        """Alias for item_res_corr_analysis(model='bivector'). See item_res_corr_analysis for full documentation."""
        self._run_item_res_corr("bivector", **kw)
    def item_res_corr_analysis_centrality(self, **kw):
        """Alias for item_res_corr_analysis(model='bivector'). See item_res_corr_analysis for full documentation."""
        self._run_item_res_corr("centrality", **kw)
    def item_res_corr_analysis_pseudo_halo(self, **kw):
        """Alias for item_res_corr_analysis(model='bivector'). See item_res_corr_analysis for full documentation."""
        self._run_item_res_corr("pseudo_halo", **kw)

    def item_res_corr_analysis_bistretch(self, **kw):
        """Alias for item_res_corr_analysis(model='bistretch'). See item_res_corr_analysis for full documentation."""
        self._run_item_res_corr("bistretch", **kw)

    def facet_res_corr_analysis_global(self, **kw):
        """Alias for facet_res_corr_analysis(model='global'). See facet_res_corr_analysis for full documentation."""
        self._run_facet_res_corr("global", **kw)

    def facet_res_corr_analysis_items(self, **kw):
        """Alias for facet_res_corr_analysis(model='items'). See facet_res_corr_analysis for full documentation."""
        self._run_facet_res_corr("items", **kw)

    def facet_res_corr_analysis_thresholds(self, **kw):
        """Alias for facet_res_corr_analysis(model='thresholds'). See facet_res_corr_analysis for full documentation."""
        self._run_facet_res_corr("thresholds", **kw)

    def facet_res_corr_analysis_matrix(self, **kw):
        """Alias for facet_res_corr_analysis(model='matrix'). See facet_res_corr_analysis for full documentation."""
        self._run_facet_res_corr("matrix", **kw)

    def facet_res_corr_analysis_bivector(self, **kw):
        """Alias for facet_res_corr_analysis(model='bivector'). See facet_res_corr_analysis for full documentation."""
        self._run_facet_res_corr("bivector", **kw)

    # rater_ aliases for facet_res_corr_analysis methods (default facet)
    def rater_res_corr_analysis_global(self, **kw):
        """Alias for rater_res_corr_analysis(model='global'). See facet_res_corr_analysis for full documentation."""
        self._run_facet_res_corr("global", **kw)
    def facet_res_corr_analysis_centrality(self, **kw):
        """Alias for facet_res_corr_analysis(model='centrality'). See facet_res_corr_analysis for full documentation."""
        self._run_facet_res_corr("centrality", **kw)
    def facet_res_corr_analysis_pseudo_halo(self, **kw):
        """Alias for facet_res_corr_analysis(model='pseudo_halo'). See facet_res_corr_analysis for full documentation."""
        self._run_facet_res_corr("pseudo_halo", **kw)
    def facet_res_corr_analysis_bistretch(self, **kw):
        """Alias for facet_res_corr_analysis(model='bistretch'). See facet_res_corr_analysis for full documentation."""
        self._run_facet_res_corr("bistretch", **kw)

    def rater_res_corr_analysis_items(self, **kw):
        """Alias for rater_res_corr_analysis(model='items'). See facet_res_corr_analysis for full documentation."""
        self._run_facet_res_corr("items", **kw)

    def rater_res_corr_analysis_thresholds(self, **kw):
        """Alias for rater_res_corr_analysis(model='thresholds'). See facet_res_corr_analysis for full documentation."""
        self._run_facet_res_corr("thresholds", **kw)

    def rater_res_corr_analysis_matrix(self, **kw):
        """Alias for rater_res_corr_analysis(model='matrix'). See facet_res_corr_analysis for full documentation."""
        self._run_facet_res_corr("matrix", **kw)

    def rater_res_corr_analysis_bivector(self, **kw):
        """Alias for rater_res_corr_analysis(model='bivector'). See facet_res_corr_analysis for full documentation."""
        self._run_facet_res_corr("bivector", **kw)

    # ------------------------------------------------------------------
    # Output tables
    # ------------------------------------------------------------------
    def rater_res_corr_analysis_centrality(self, **kw):
        """Alias for rater_res_corr_analysis(model='bivector'). See facet_res_corr_analysis for full documentation."""
        self._run_facet_res_corr("centrality", **kw)

    # ------------------------------------------------------------------
    # Output tables
    # ------------------------------------------------------------------
    def rater_res_corr_analysis_pseudo_halo(self, **kw):
        """Alias for rater_res_corr_analysis(model='bivector'). See facet_res_corr_analysis for full documentation."""
        self._run_facet_res_corr("pseudo_halo", **kw)

    def rater_res_corr_analysis_bistretch(self, **kw):
        """Alias for rater_res_corr_analysis(model='bistretch'). See facet_res_corr_analysis for full documentation."""
        self._run_facet_res_corr("bistretch", **kw)

    # ------------------------------------------------------------------
    # Output tables
    # ------------------------------------------------------------------

    def _anchors_stale(self, model, anchors):
        """Read-only check: True if `anchors` differs from the anchor set this
        model was last (anchor-)calibrated with, or if it's never been
        anchor-calibrated at all. False if anchors is None (unanchored).

        Side-effect-free by design — callers that also need to trigger a
        recalibration (e.g. _ensure_calibrated) must do that themselves based
        on the return value, rather than this method doing it implicitly.
        Deciding staleness by calling _ensure_calibrated as a pre-check would
        be wrong: it mutates anchor_rater_names_{model} as a side effect, so a
        second call (e.g. fit_statistics' own internal _ensure_calibrated)
        would then see no mismatch and skip its own cache invalidation.
        """
        if anchors is None:
            return False
        name_attr = "anchor_rater_names_matrix" if model == "mixed" else f"anchor_rater_names_{model}"
        stored = getattr(self, name_attr, None)
        return stored is None or set(stored) != set(anchors)

    def _sync_mixed(self, anchors, **selection_kw):
        """Ensure a working (unanchored-named) facet_effects_mixed/rater_models
        exists for 'mixed', which has no calibrate() of its own -- it's always
        derived via per_rater_model_selection(). Lazily runs
        per_rater_model_selection() (with default selection criteria, matching
        how other models lazily calibrate() with default settings) if it
        hasn't been run yet, or if the given anchor set doesn't match what was
        last run under anchors.

        Fit statistics and residuals are invariant to the anchor shift itself
        (a pure translation), so once a valid per-facet-element assignment exists we
        can compute everything from it directly. When anchors is given, this
        (re-)syncs from the anchor-derived assignment for that exact anchor
        set every time, so the existing anchor-unaware fit-stat machinery
        (_ensure_fit_matrices, person_estimates, category_probability_dict,
        etc.) operates on the correct (anchored) assignment.

        selection_kw : calibration/estimation kwargs (constant, method,
            matrix_power, log_lik_tol, warm_corr, tolerance, max_iters,
            ext_score_adjustment) forwarded to per_rater_model_selection() if
            it needs to be run. Selection-criterion args (test, alpha,
            min_effect, etc.) are left at their own defaults, same as any
            other lazy auto-calibration in this class.
        """
        if anchors is not None:
            if not hasattr(self, "anchor_facet_effects_mixed") or self._anchors_stale(
                "mixed", anchors
            ):
                self.per_rater_model_selection(anchors=anchors, **selection_kw)
            self.facet_effects_mixed = self.anchor_facet_effects_mixed
            self.rater_models = self.anchor_rater_models
        elif not hasattr(self, "facet_effects_mixed"):
            self.per_rater_model_selection(**selection_kw)

    def _ensure_calibrated(self, model, **kw):
        """Lazy-load calibration and person_locations. SE computation is handled
        separately by _ensure_se to avoid redundant bootstrap runs."""
        calib_kw = {
            k: v
            for k, v in kw.items()
            if k in ("constant", "threshold_constant", "method", "robust", "matrix_power", "log_lik_tol")
        }
        abil_kw = {
            k: v
            for k, v in kw.items()
            if k in ("warm_corr", "tolerance", "max_iters", "ext_score_adjustment")
        }
        anchors = kw.get("anchors", None)
        seed = kw.get("seed", None)

        if model == "mixed":
            self._sync_mixed(anchors, seed=seed, **calib_kw, **abil_kw)
        else:
            if not hasattr(self, f"facet_effects_{model}"):
                self.calibrate(model=model, **calib_kw)
            if anchors is not None:
                stored = getattr(self, f"anchor_rater_names_{model}", None)
                if not hasattr(self, f"anchor_rater_names_{model}") or set(stored) != set(anchors):
                    self.calibrate_anchor(model, anchors, **calib_kw)
        if not hasattr(self, f"persons_{model}"):
            self.person_estimates(model=model, **abil_kw)

    def _ensure_se(
        self,
        model,
        anchors,
        interval,
        no_of_samples,
        constant,
        method,
        matrix_power,
        log_lik_tol,
        seed=None,
        store_bootstrap=False,
        threshold_constant=None,
        robust=False,
    ):
        """Internal helper: compute standard errors (and optionally anchor SEs) if not yet done.

        store_bootstrap : bool, default False
            If True, forces std_errors() to run with store_bootstrap=True
            (even if SEs are already cached without stored samples) so
            self._bootstrap_samples_{model} is available afterwards --
            used by rater_stats_df's divergence_test to get a single,
            PAIRED bootstrap covering both the stretch model and (as a
            byproduct of its own calibration) its free/vector parent,
            rather than two independent bootstrap passes. Every other
            caller of _ensure_se leaves this False, so this has no effect
            on their behaviour."""
        anc = anchors is not None
        prefix = "anchor_" if anc else ""
        trigger = f"{prefix}threshold_se_{model}"
        # Re-run if SEs not computed, or if CIs requested but not yet stored,
        # or if stored bootstrap samples are needed but weren't kept.
        ci_attr = f"anchor_item_low_{model}" if anc else f"item_low_{model}"
        ci_missing = interval is not None and not hasattr(self, ci_attr)
        need_bootstrap_samples = store_bootstrap and not getattr(
            self, f"_bootstrap_stored_{model}", False
        )
        if not hasattr(self, trigger) or ci_missing or need_bootstrap_samples:
            if anc:
                # Ensure unanchored SEs exist first (anchor_std_errors depends on them)
                if not hasattr(self, f"threshold_se_{model}") or need_bootstrap_samples:
                    self.std_errors(
                        model=model,
                        interval=interval,
                        no_of_samples=no_of_samples,
                        constant=constant,
                        threshold_constant=threshold_constant,
                        method=method,
                        robust=robust,
                        matrix_power=matrix_power,
                        log_lik_tol=log_lik_tol,
                        seed=seed,
                        store_bootstrap=store_bootstrap,
                    )
                self.anchor_std_errors(model=model, anchors=anchors, seed=seed)
            else:
                self.std_errors(
                    model=model,
                    interval=interval,
                    no_of_samples=no_of_samples,
                    constant=constant,
                    threshold_constant=threshold_constant,
                    method=method,
                    robust=robust,
                    matrix_power=matrix_power,
                    log_lik_tol=log_lik_tol,
                    seed=seed,
                    store_bootstrap=store_bootstrap,
                )

    def item_stats_df(
        self,
        model="global",
        anchors=None,
        full=False,
        ext_scores=True,
        zstd=False,
        point_measure_corr=False,
        dp=3,
        se=True,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        method=None,
        constant=None,
        threshold_constant=None,
        robust=False,
        matrix_power=None,
        log_lik_tol=0.000001,
        no_of_samples=500,
        interval=None,
        seed=None,
        filename=None,
        file_format="csv",
    ):
        """
        Build and store the item statistics summary table.

        Auto-triggers the full calibration/SE/fit chain if not yet run.
        Stores result as self.item_stats_{model}.

        Parameters
        ----------
        model : str, default 'global'
            Facet element (e.g. rater) parameterisation.
        anchors : list or None, default None
            If provided, uses anchor-calibrated item item_locations.
        full : bool, default False
            If True, sets zstd=True, point_measure_corr=True, interval=0.95.
        ext_scores : bool, default True
            Include extreme scorers in fit computation.
        zstd : bool, default False
            If True, includes Infit Z and Outfit Z columns.
        point_measure_corr : bool, default False
            If True, includes point-measure correlation columns.
        dp : int, default 3
            Decimal places.
        se : bool, default True
            If True, computes and includes the SE column (and CI bound
            columns, if interval is set). If False, skips the bootstrap
            entirely -- useful when only Infit/Outfit MS are needed.
        warm_corr : bool, default True
            Warm bias correction.
        tolerance : float, default 0.00001
            Newton-Raphson convergence tolerance.
        max_iters : int, default 100
            Maximum iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment.
        method : str, default 'log-lik'
            Priority vector extraction method.
        constant : float, default 0.1
            Additive smoothing constant.
        matrix_power : int or None, default None
            Matrix power for calibration. None resolves to 0 for
            method='log-lik' (no powering), else 5.
        log_lik_tol : float, default 0.000001
            Calibration convergence tolerance.
        no_of_samples : int, default 500
            Bootstrap samples for SE estimation. Unused if se=False.
        interval : float or None, default None
            CI width; if provided, percentile bound columns included.
            Ignored if se=False.
        seed : int or None, default None
            Seed passed through to the internal std_errors() call (only
            used if not already computed). None draws fresh entropy.
        filename : str or None, default None
            If given, also saves the returned table to
            ``{filename}.{file_format}`` (path without extension, as for
            the plotting methods).
        file_format : {'csv', 'xlsx'}, default 'csv'
            File format used when filename is given.

        Attributes set
        --------------
        item_stats_{model} : pandas.DataFrame
            Item statistics with items as rows. Always contains Estimate,
            Count, Facility, Infit MS, Outfit MS. SE (and CI bound columns,
            if interval is set) included only when se=True.

        Returns
        -------
        pandas.DataFrame
            The table, also stored as the attribute listed above.
        """

        model = self._MODEL_ALIASES.get(model, model)
        if full:
            zstd = point_measure_corr = True
            interval = interval or 0.95

        if not se:
            interval = None

        self._ensure_calibrated(
            model,
            anchors=anchors,
            interval=interval,
            no_of_samples=no_of_samples,
            constant=constant,
            threshold_constant=threshold_constant,
            method=method,
            robust=robust,
            matrix_power=matrix_power,
            log_lik_tol=log_lik_tol,
            warm_corr=warm_corr,
            tolerance=tolerance,
            max_iters=max_iters,
            ext_score_adjustment=ext_score_adjustment,
            seed=seed,
        )
        if se:
            self._ensure_se(
                model,
                anchors,
                interval,
                no_of_samples,
                constant,
                method,
                matrix_power,
                log_lik_tol,
                threshold_constant=threshold_constant,
                robust=robust,
                seed=seed,
            )
        if not hasattr(self, f"item_outfit_ms_{model}"):
            self._run_item_fit(model)

        anc = anchors is not None
        item_locations = self._items_for(model, anchor=anc)

        stats = pd.DataFrame(index=self.responses.columns)
        stats["Estimate"] = item_locations.round(dp)
        if se:
            se_vals = self._item_se_for(model, "se", anchor=anc)
            if se_vals is None:
                se_vals = self._item_se_for(model, "se")
            low = self._item_se_for(model, "low", anchor=anc)
            high = self._item_se_for(model, "high", anchor=anc)
            if anc and low is None:
                low, high = self._item_se_for(model, "low"), self._item_se_for(model, "high")
            stats["SE"] = se_vals.round(dp)
            if interval is not None and low is not None:
                lo_lbl = f"{round((1 - interval) * 50, 1)}%"
                hi_lbl = f"{round((1 + interval) * 50, 1)}%"
                stats[lo_lbl] = low.round(dp)
                stats[hi_lbl] = high.round(dp)
        stats["Count"] = self.response_counts.astype(int)
        stats["Facility"] = self.item_facilities.round(dp)
        stats["Infit MS"] = getattr(self, f"item_infit_ms_{model}").round(dp)
        if zstd:
            stats["Infit Z"] = getattr(self, f"item_infit_zstd_{model}").round(dp)
        stats["Outfit MS"] = getattr(self, f"item_outfit_ms_{model}").round(dp)
        if zstd:
            stats["Outfit Z"] = getattr(self, f"item_outfit_zstd_{model}").round(dp)
        if point_measure_corr:
            stats["PM corr"] = getattr(self, f"point_measure_{model}").round(dp)
            stats["Exp PM corr"] = getattr(self, f"exp_point_measure_{model}").round(dp)

        setattr(self, f"item_stats_{model}", stats)
        table = getattr(self, f"item_stats_{model}")
        _save_table(table, filename, file_format)
        return table

    # Backwards-compatible aliases
    def item_stats_df_global(self, **kw):
        """Alias for item_stats_df(model='global'). See item_stats_df for full documentation."""
        return self.item_stats_df(model="global", **kw)

    def item_stats_df_items(self, **kw):
        """Alias for item_stats_df(model='items'). See item_stats_df for full documentation."""
        return self.item_stats_df(model="items", **kw)

    def item_stats_df_thresholds(self, **kw):
        """Alias for item_stats_df(model='thresholds'). See item_stats_df for full documentation."""
        return self.item_stats_df(model="thresholds", **kw)

    def item_stats_df_matrix(self, **kw):
        """Alias for item_stats_df(model='matrix'). See item_stats_df for full documentation."""
        return self.item_stats_df(model="matrix", **kw)

    def item_stats_df_bivector(self, **kw):
        """Alias for item_stats_df(model='bivector'). See item_stats_df for full documentation."""
        return self.item_stats_df(model="bivector", **kw)
    def item_stats_df_centrality(self, **kw):
        """Alias for item_stats_df(model='bivector'). See item_stats_df for full documentation."""
        return self.item_stats_df(model="centrality", **kw)
    def item_stats_df_pseudo_halo(self, **kw):
        """Alias for item_stats_df(model='bivector'). See item_stats_df for full documentation."""
        return self.item_stats_df(model="pseudo_halo", **kw)
    def item_stats_df_bistretch(self, **kw):
        """Alias for item_stats_df(model='bivector'). See item_stats_df for full documentation."""
        return self.item_stats_df(model="bistretch", **kw)

    def threshold_stats_df(
        self,
        model="global",
        anchors=None,
        full=False,
        zstd=False,
        disc=False,
        point_measure_corr=False,
        dp=3,
        se=True,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        method=None,
        constant=None,
        threshold_constant=None,
        robust=False,
        matrix_power=None,
        log_lik_tol=0.000001,
        no_of_samples=500,
        interval=None,
        seed=None,
        filename=None,
        file_format="csv",
    ):
        """
        Build and store the threshold statistics summary table.

        Auto-triggers the full calibration/SE/fit chain if not yet run.
        Stores result as self.threshold_stats_{model}.

        Parameters
        ----------
        model : str, default 'global'
            Facet element (e.g. rater) parameterisation.
        anchors : list or None, default None
            Anchor facet_elements for SE computation.
        full : bool, default False
            If True, sets zstd=True, disc=True, point_measure_corr=True, interval=0.95.
        zstd : bool, default False
            If True, includes Infit Z and Outfit Z columns.
        disc : bool, default False
            If True, includes threshold discrimination column.
        point_measure_corr : bool, default False
            If True, includes point-measure correlation columns.
        dp : int, default 3
            Decimal places.
        se : bool, default True
            If True, computes and includes the SE column (and CI bound
            columns, if interval is set). If False, skips the bootstrap
            entirely.
        warm_corr : bool, default True
            Warm bias correction.
        tolerance : float, default 0.00001
            Convergence tolerance.
        max_iters : int, default 100
            Maximum iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment.
        method : str, default 'log-lik'
            Priority vector extraction method.
        constant : float, default 0.1
            Smoothing constant.
        matrix_power : int or None, default None
            Matrix power. None resolves to 0 for method='log-lik'
            (no powering), else 5.
        log_lik_tol : float, default 0.000001
            Calibration convergence tolerance.
        no_of_samples : int, default 500
            Bootstrap samples. Unused if se=False.
        interval : float or None, default None
            CI width. Ignored if se=False.
        seed : int or None, default None
            Seed passed through to the internal std_errors() call (only
            used if not already computed). None draws fresh entropy.
        filename : str or None, default None
            If given, also saves the returned table to
            ``{filename}.{file_format}`` (path without extension, as for
            the plotting methods).
        file_format : {'csv', 'xlsx'}, default 'csv'
            File format used when filename is given.

        Attributes set
        --------------
        threshold_stats_{model} : pandas.DataFrame
            Threshold statistics, rows Threshold 1..Threshold max_score.

        Returns
        -------
        pandas.DataFrame
            The table, also stored as the attribute listed above.
        """

        model = self._MODEL_ALIASES.get(model, model)
        if full:
            zstd = disc = point_measure_corr = True
            interval = interval or 0.95

        if not se:
            interval = None

        self._ensure_calibrated(
            model,
            anchors=anchors,
            interval=interval,
            no_of_samples=no_of_samples,
            constant=constant,
            threshold_constant=threshold_constant,
            method=method,
            robust=robust,
            matrix_power=matrix_power,
            log_lik_tol=log_lik_tol,
            warm_corr=warm_corr,
            tolerance=tolerance,
            max_iters=max_iters,
            ext_score_adjustment=ext_score_adjustment,
            seed=seed,
        )
        if se:
            self._ensure_se(
                model,
                anchors,
                interval,
                no_of_samples,
                constant,
                method,
                matrix_power,
                log_lik_tol,
                threshold_constant=threshold_constant,
                robust=robust,
                seed=seed,
            )
        if not hasattr(self, f"threshold_outfit_ms_{model}"):
            self._run_threshold_fit(model, anchors=anchors)

        anc = anchors is not None
        thresholds = (
            self._thresholds_for(model, anchor=anc)
        )
        thr_se_attr = f"anchor_threshold_se_{model}" if anc else f"threshold_se_{model}"
        thr_se = getattr(self, thr_se_attr, None)
        thr_lo = getattr(
            self,
            f"anchor_threshold_low_{model}" if anc else f"threshold_low_{model}",
            None,
        )
        thr_hi = getattr(
            self,
            f"anchor_threshold_high_{model}" if anc else f"threshold_high_{model}",
            None,
        )

        idx = [f"Threshold {t + 1}" for t in range(self.max_score)]
        stats = pd.DataFrame(index=idx)
        stats["Estimate"] = thresholds.values.round(dp)
        if thr_se is not None:
            stats["SE"] = thr_se.round(dp)
        if interval is not None and thr_lo is not None:
            lo_lbl = f"{round((1 - interval) * 50, 1)}%"
            hi_lbl = f"{round((1 + interval) * 50, 1)}%"
            stats[lo_lbl] = thr_lo.round(dp)
            stats[hi_lbl] = thr_hi.round(dp)
        stats["Infit MS"] = getattr(self, f"threshold_infit_ms_{model}").values.round(
            dp
        )
        if zstd:
            stats["Infit Z"] = getattr(
                self, f"threshold_infit_zstd_{model}"
            ).values.round(dp)
        stats["Outfit MS"] = getattr(self, f"threshold_outfit_ms_{model}").values.round(
            dp
        )
        if zstd:
            stats["Outfit Z"] = getattr(
                self, f"threshold_outfit_zstd_{model}"
            ).values.round(dp)
        if disc:
            stats["Discrim"] = getattr(
                self, f"threshold_discrimination_{model}"
            ).values.round(dp)
        if point_measure_corr:
            stats["PM corr"] = getattr(
                self, f"threshold_point_measure_{model}"
            ).values.round(dp)
            stats["Exp PM corr"] = getattr(
                self, f"threshold_exp_point_measure_{model}"
            ).values.round(dp)

        setattr(self, f"threshold_stats_{model}", stats)
        table = getattr(self, f"threshold_stats_{model}")
        _save_table(table, filename, file_format)
        return table

    def threshold_stats_df_global(self, **kw):
        """Alias for threshold_stats_df(model='global'). See threshold_stats_df for full documentation."""
        return self.threshold_stats_df(model="global", **kw)

    def threshold_stats_df_items(self, **kw):
        """Alias for threshold_stats_df(model='items'). See threshold_stats_df for full documentation."""
        return self.threshold_stats_df(model="items", **kw)

    def threshold_stats_df_thresholds(self, **kw):
        """Alias for threshold_stats_df(model='thresholds'). See threshold_stats_df for full documentation."""
        return self.threshold_stats_df(model="thresholds", **kw)

    def threshold_stats_df_matrix(self, **kw):
        """Alias for threshold_stats_df(model='matrix'). See threshold_stats_df for full documentation."""
        return self.threshold_stats_df(model="matrix", **kw)

    def threshold_stats_df_bivector(self, **kw):
        """Alias for threshold_stats_df(model='bivector'). See threshold_stats_df for full documentation."""
        return self.threshold_stats_df(model="bivector", **kw)
    def threshold_stats_df_centrality(self, **kw):
        """Alias for threshold_stats_df(model='bivector'). See threshold_stats_df for full documentation."""
        return self.threshold_stats_df(model="centrality", **kw)
    def threshold_stats_df_pseudo_halo(self, **kw):
        """Alias for threshold_stats_df(model='bivector'). See threshold_stats_df for full documentation."""
        return self.threshold_stats_df(model="pseudo_halo", **kw)
    def threshold_stats_df_bistretch(self, **kw):
        """Alias for threshold_stats_df(model='bivector'). See threshold_stats_df for full documentation."""
        return self.threshold_stats_df(model="bistretch", **kw)

    def category_stats_df(
        self,
        model="global",
        anchors=None,
        dp=3,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        method=None,
        constant=None,
        threshold_constant=None,
        robust=False,
        matrix_power=None,
        log_lik_tol=0.000001,
        no_of_samples=500,
        interval=None,
        seed=None,
        filename=None,
        file_format="csv",
    ):
        """
        Build and store the category width statistics summary table.

        Reports category *widths* (thresholds[k+1] - thresholds[k]) rather
        than raw threshold locations — see threshold_stats_df for the full
        rationale: a zero-summed threshold vector has only max_score-1 true
        degrees of freedom, so the max_score per-threshold SEs threshold_
        stats_df reports are correlated, not independent, and understate
        the true uncertainty of the physically meaningful step-structure
        quantity. The widths are those of the sub-model given by model=
        (thresholds_{model}; with estimation='pair-cpat' the same for every
        sub-model, though their bootstrap SEs can differ, since std_errors()
        recalibrates the whole model, facet element facet_effects included,
        per resample). Reported *alongside*
        threshold_stats_df's output, not instead of it — threshold-level
        SEs remain the expected, standard report.

        Deliberately lighter than threshold_stats_df: no Infit/Outfit or
        other fit statistics, since those aren't naturally defined for a
        difference of two threshold locations. Auto-triggers calibration/
        SE computation via _ensure_calibrated/_ensure_se if not yet run
        (not the full, heavier fit_statistics()).

        Widths can be negative — a negative width at category k means
        thresholds k and k+1 are disordered (category k is never the most
        likely response at any person_location level). Prop disordered makes this
        a continuous diagnostic rather than a single point-estimate
        yes/no: the proportion of bootstrap resamples in which that
        category's width was negative — reasonably read as the
        probability that the true category is disordered.

        Parameters
        ----------
        model : str, default 'global'
            Facet element parameterisation. Affects the reported SE (via that
            model's own bootstrap) but not the Estimate column itself.
        anchors : list or None, default None
            Anchor facet_elements. If given, uses anchor_thresholds_{model}
            and the anchored bootstrap SEs.
        dp : int, default 3
            Decimal places.
        warm_corr, tolerance, max_iters, ext_score_adjustment : floats
            Person-estimation kwargs, used only if calibration hasn't
            already been run.
        method, constant, matrix_power, log_lik_tol : floats
            Calibration kwargs, used only if calibration hasn't already
            been run.
        no_of_samples : int, default 500
            Bootstrap samples, used only if std_errors() hasn't already
            been run.
        interval : float or None, default None
            CI width. If provided, lower/upper percentile columns are
            included — but only if std_errors() is triggered by this call
            or was already run with an interval; it is not retroactively
            added to an existing SE run made without one.
        seed : int or None, default None
            Seed for std_errors(), used only if it hasn't already been
            run.
        filename : str or None, default None
            If given, also saves the returned table to
            ``{filename}.{file_format}`` (path without extension, as for
            the plotting methods).
        file_format : {'csv', 'xlsx'}, default 'csv'
            File format used when filename is given.

        Attributes set
        --------------
        category_stats_{model} : pandas.DataFrame
            Rows Category 1..Category max_score-1 (one fewer row than
            threshold_stats_{model}). Columns: Estimate (the width
            itself, can be negative), SE (from cat_width_se_{model} —
            differenced *within* each bootstrap resample before taking
            the std, so it already reflects Cov(threshold_k,
            threshold_{k+1})), CI bounds if interval is not None,
            Disordered (Estimate < 0), and Prop disordered.

        Returns
        -------
        pandas.DataFrame
            The table, also stored as the attribute listed above.
        """
        model = self._MODEL_ALIASES.get(model, model)
        self._ensure_calibrated(
            model,
            anchors=anchors,
            interval=interval,
            no_of_samples=no_of_samples,
            constant=constant,
            threshold_constant=threshold_constant,
            method=method,
            robust=robust,
            matrix_power=matrix_power,
            log_lik_tol=log_lik_tol,
            warm_corr=warm_corr,
            tolerance=tolerance,
            max_iters=max_iters,
            ext_score_adjustment=ext_score_adjustment,
            seed=seed,
        )
        self._ensure_se(
            model,
            anchors,
            interval,
            no_of_samples,
            constant,
            method,
            matrix_power,
            log_lik_tol,
            threshold_constant=threshold_constant,
            robust=robust,
            seed=seed,
        )

        anc = anchors is not None
        prefix = "anchor_" if anc else ""

        thresholds = (
            self._thresholds_for(model, anchor=anc)
        )
        thresholds = pd.Series(np.asarray(thresholds), index=range(1, self.max_score + 1))
        cat_widths = thresholds.diff().dropna()
        cat_widths.index = range(1, self.max_score)

        cat_se_dict = getattr(self, f"{prefix}cat_width_se_{model}")
        cats = sorted(cat_se_dict.keys())
        cat_idx = [f"Category {k}" for k in cats]

        stats = pd.DataFrame(index=cat_idx)
        stats["Estimate"] = cat_widths.loc[cats].values.round(dp)
        stats["SE"] = np.array([cat_se_dict[k] for k in cats]).round(dp)

        lo_attr = f"{prefix}cat_width_low_{model}"
        hi_attr = f"{prefix}cat_width_high_{model}"
        if interval is not None and hasattr(self, lo_attr):
            lo_dict = getattr(self, lo_attr)
            hi_dict = getattr(self, hi_attr)
            stats[f"{round((1 - interval) * 50, 1)}%"] = np.array(
                [lo_dict[k] for k in cats]
            ).round(dp)
            stats[f"{round((1 + interval) * 50, 1)}%"] = np.array(
                [hi_dict[k] for k in cats]
            ).round(dp)

        stats["Disordered"] = cat_widths.loc[cats].values < 0

        bootstrap_dict = getattr(self, f"{prefix}cat_width_bootstrap_{model}")
        stats["Prop disordered"] = np.array(
            [(np.asarray(bootstrap_dict[k]) < 0).mean() for k in cats]
        ).round(dp)

        setattr(self, f"category_stats_{model}", stats)
        table = getattr(self, f"category_stats_{model}")
        _save_table(table, filename, file_format)
        return table

    def category_stats_df_global(self, **kw):
        """Alias for category_stats_df(model='global'). See category_stats_df for full documentation."""
        return self.category_stats_df(model="global", **kw)

    def category_stats_df_items(self, **kw):
        """Alias for category_stats_df(model='items'). See category_stats_df for full documentation."""
        return self.category_stats_df(model="items", **kw)

    def category_stats_df_thresholds(self, **kw):
        """Alias for category_stats_df(model='thresholds'). See category_stats_df for full documentation."""
        return self.category_stats_df(model="thresholds", **kw)

    def category_stats_df_matrix(self, **kw):
        """Alias for category_stats_df(model='matrix'). See category_stats_df for full documentation."""
        return self.category_stats_df(model="matrix", **kw)

    def category_stats_df_bivector(self, **kw):
        """Alias for category_stats_df(model='bivector'). See category_stats_df for full documentation."""
        return self.category_stats_df(model="bivector", **kw)
    def category_stats_df_centrality(self, **kw):
        """Alias for category_stats_df(model='bivector'). See category_stats_df for full documentation."""
        return self.category_stats_df(model="centrality", **kw)
    def category_stats_df_pseudo_halo(self, **kw):
        """Alias for category_stats_df(model='bivector'). See category_stats_df for full documentation."""
        return self.category_stats_df(model="pseudo_halo", **kw)
    def category_stats_df_bistretch(self, **kw):
        """Alias for category_stats_df(model='bivector'). See category_stats_df for full documentation."""
        return self.category_stats_df(model="bistretch", **kw)

    def person_stats_df(
        self,
        model="global",
        anchors=None,
        full=False,
        rsem=False,
        zstd=False,
        dp=3,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        method=None,
        constant=None,
        threshold_constant=None,
        robust=False,
        matrix_power=None,
        log_lik_tol=0.000001,
        interval=None,
        no_of_samples=500,
        filename=None,
        file_format="csv",
    ):
        """
        Build and store the person statistics summary table.

        Auto-triggers calibration and person person_location estimation if not yet run.
        Stores result as self.person_stats_{model}.

        Parameters
        ----------
        model : str, default 'global'
            Facet element (e.g. rater) parameterisation.
        anchors : list or None, default None
            If provided, uses anchor-calibrated person_locations.
        full : bool, default False
            If True, sets rsem=True, zstd=True.
        rsem : bool, default False
            If True, includes Residual SEM (RSEM) column.
        zstd : bool, default False
            If True, includes Infit Z and Outfit Z columns.
        dp : int, default 3
            Decimal places.
        warm_corr : bool, default True
            Warm bias correction.
        tolerance : float, default 0.00001
            Convergence tolerance.
        max_iters : int, default 100
            Maximum iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment.
        method : str, default 'log-lik'
            Priority vector extraction method.
        constant : float, default 0.1
            Smoothing constant.
        matrix_power : int or None, default None
            Matrix power. None resolves to 0 for method='log-lik'
            (no powering), else 5.
        log_lik_tol : float, default 0.000001
            Calibration convergence tolerance.
        interval : float or None, default None
            CI width (unused directly; passed to _ensure_calibrated).
        no_of_samples : int, default 500
            Bootstrap samples.
        filename : str or None, default None
            If given, also saves the returned table to
            ``{filename}.{file_format}`` (path without extension, as for
            the plotting methods).
        file_format : {'csv', 'xlsx'}, default 'csv'
            File format used when filename is given.

        Attributes set
        --------------
        person_stats_{model} : pandas.DataFrame
            Person statistics with persons as rows. Contains Estimate, CSEM,
            Score, Max score, p, Infit MS, Outfit MS. Optional: RSEM, Infit Z,
            Outfit Z.

        Returns
        -------
        pandas.DataFrame
            The table, also stored as the attribute listed above.
        """

        model = self._MODEL_ALIASES.get(model, model)
        self._ensure_calibrated(
            model,
            warm_corr=warm_corr,
            tolerance=tolerance,
            max_iters=max_iters,
            ext_score_adjustment=ext_score_adjustment,
            constant=constant,
            threshold_constant=threshold_constant,
            method=method,
            robust=robust,
            matrix_power=matrix_power,
            log_lik_tol=log_lik_tol,
        )
        if not hasattr(self, f"person_outfit_ms_{model}"):
            self._run_person_fit(model)
        if full:
            rsem = zstd = True

        anc = anchors is not None
        estimates = self._get_abils(model, anchor=anc)

        stats = pd.DataFrame(index=self.person_names)
        stats["Estimate"] = estimates.round(dp)
        stats["CSEM"] = getattr(self, f"csem_vector_{model}").round(dp)
        if rsem:
            stats["RSEM"] = getattr(self, f"rsem_vector_{model}").round(dp)

        unstacked = self.responses.unstack(level=0)
        stats["Score"] = unstacked.sum(axis=1).astype(int)
        stats["Max score"] = (unstacked.count(axis=1) * self.max_score).astype(int)
        stats["p"] = (unstacked.mean(axis=1) / self.max_score).round(dp)

        for col, src in [
            ("Infit MS", getattr(self, f"person_infit_ms_{model}")),
            ("Outfit MS", getattr(self, f"person_outfit_ms_{model}")),
        ]:
            stats[col] = np.nan
            stats.loc[src.index, col] = src.round(dp).values
        if zstd:
            for col, src in [
                ("Infit Z", getattr(self, f"person_infit_zstd_{model}")),
                ("Outfit Z", getattr(self, f"person_outfit_zstd_{model}")),
            ]:
                stats[col] = np.nan
                stats.loc[src.index, col] = src.round(dp).values

        setattr(self, f"person_stats_{model}", stats)
        table = getattr(self, f"person_stats_{model}")
        _save_table(table, filename, file_format)
        return table

    def person_stats_df_global(self, **kw):
        """Alias for person_stats_df(model='global'). See person_stats_df for full documentation."""
        return self.person_stats_df(model="global", **kw)

    def person_stats_df_items(self, **kw):
        """Alias for person_stats_df(model='items'). See person_stats_df for full documentation."""
        return self.person_stats_df(model="items", **kw)

    def person_stats_df_thresholds(self, **kw):
        """Alias for person_stats_df(model='thresholds'). See person_stats_df for full documentation."""
        return self.person_stats_df(model="thresholds", **kw)

    def person_stats_df_matrix(self, **kw):
        """Alias for person_stats_df(model='matrix'). See person_stats_df for full documentation."""
        return self.person_stats_df(model="matrix", **kw)

    def person_stats_df_bivector(self, **kw):
        """Alias for person_stats_df(model='bivector'). See person_stats_df for full documentation."""
        return self.person_stats_df(model="bivector", **kw)
    def person_stats_df_centrality(self, **kw):
        """Alias for person_stats_df(model='bivector'). See person_stats_df for full documentation."""
        return self.person_stats_df(model="centrality", **kw)
    def person_stats_df_pseudo_halo(self, **kw):
        """Alias for person_stats_df(model='bivector'). See person_stats_df for full documentation."""
        return self.person_stats_df(model="pseudo_halo", **kw)
    def person_stats_df_bistretch(self, **kw):
        """Alias for person_stats_df(model='bivector'). See person_stats_df for full documentation."""
        return self.person_stats_df(model="bistretch", **kw)

    def test_stats_df(
        self,
        model="global",
        dp=3,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        method=None,
        constant=None,
        threshold_constant=None,
        robust=False,
        matrix_power=None,
        log_lik_tol=0.000001,
        no_of_samples=500,
        seed=None,
        filename=None,
        file_format="csv",
    ):
        """
        Build and store the test-level summary statistics table.

        Auto-triggers calibration and test fit statistics if not yet run.
        Stores result as self.test_stats_{model}.

        Parameters
        ----------
        model : str, default 'global'
            Facet element (e.g. rater) parameterisation.
        dp : int, default 3
            Decimal places.
        warm_corr : bool, default True
            Warm bias correction.
        tolerance : float, default 0.00001
            Convergence tolerance.
        max_iters : int, default 100
            Maximum iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment.
        method : str, default 'log-lik'
            Priority vector extraction method.
        constant : float, default 0.1
            Smoothing constant.
        matrix_power : int or None, default None
            Matrix power. None resolves to 0 for method='log-lik'
            (no powering), else 5.
        log_lik_tol : float, default 0.000001
            Calibration convergence tolerance.
        no_of_samples : int, default 500
            Bootstrap samples.
        seed : int or None, default None
            Seed passed through to the internal std_errors() call (only
            used if not already computed). None draws fresh entropy.
        filename : str or None, default None
            If given, also saves the returned table to
            ``{filename}.{file_format}`` (path without extension, as for
            the plotting methods).
        file_format : {'csv', 'xlsx'}, default 'csv'
            File format used when filename is given.

        Attributes set
        --------------
        test_stats_{model} : pandas.DataFrame
            Two-column table (Items, Persons) with rows:
            Mean, SD, Separation ratio, Strata, Reliability.

        Returns
        -------
        pandas.DataFrame
            The table, also stored as the attribute listed above.
        """

        model = self._MODEL_ALIASES.get(model, model)
        self._ensure_calibrated(
            model,
            constant=constant,
            threshold_constant=threshold_constant,
            method=method,
            robust=robust,
            matrix_power=matrix_power,
            log_lik_tol=log_lik_tol,
            seed=seed,
        )
        if not hasattr(self, f"psi_{model}"):
            self._run_test_fit(model, seed=seed)

        stats = pd.DataFrame(
            {
                "Items": [
                    self._items_for(model).mean(),
                    self._items_for(model).std(),
                    getattr(self, f"isi_{model}"),
                    getattr(self, f"item_strata_{model}"),
                    getattr(self, f"item_reliability_{model}"),
                ],
                "Persons": [
                    getattr(self, f"persons_{model}").mean(),
                    getattr(self, f"persons_{model}").std(),
                    getattr(self, f"psi_{model}"),
                    getattr(self, f"person_strata_{model}"),
                    getattr(self, f"person_reliability_{model}"),
                ],
            },
            index=["Mean", "SD", "Separation ratio", "Strata", "Reliability"],
        )
        setattr(self, f"test_stats_{model}", stats.round(dp))
        table = getattr(self, f"test_stats_{model}")
        _save_table(table, filename, file_format)
        return table

    def test_stats_df_global(self, **kw):
        """Alias for test_stats_df(model='global'). See test_stats_df for full documentation."""
        return self.test_stats_df(model="global", **kw)

    def test_stats_df_items(self, **kw):
        """Alias for test_stats_df(model='items'). See test_stats_df for full documentation."""
        return self.test_stats_df(model="items", **kw)

    def test_stats_df_thresholds(self, **kw):
        """Alias for test_stats_df(model='thresholds'). See test_stats_df for full documentation."""
        return self.test_stats_df(model="thresholds", **kw)

    def test_stats_df_matrix(self, **kw):
        """Alias for test_stats_df(model='matrix'). See test_stats_df for full documentation."""
        return self.test_stats_df(model="matrix", **kw)

    def test_stats_df_bivector(self, **kw):
        """Alias for test_stats_df(model='bivector'). See test_stats_df for full documentation."""
        return self.test_stats_df(model="bivector", **kw)

    # ------------------------------------------------------------------
    # Rater stats table (most complex -- varies substantially by model)
    # ------------------------------------------------------------------
    def test_stats_df_centrality(self, **kw):
        """Alias for test_stats_df(model='bivector'). See test_stats_df for full documentation."""
        return self.test_stats_df(model="centrality", **kw)

    # ------------------------------------------------------------------
    # Rater stats table (most complex -- varies substantially by model)
    # ------------------------------------------------------------------
    def test_stats_df_pseudo_halo(self, **kw):
        """Alias for test_stats_df(model='bivector'). See test_stats_df for full documentation."""
        return self.test_stats_df(model="pseudo_halo", **kw)

    # ------------------------------------------------------------------
    # Rater stats table (most complex -- varies substantially by model)
    # ------------------------------------------------------------------
    def test_stats_df_bistretch(self, **kw):
        """Alias for test_stats_df(model='bivector'). See test_stats_df for full documentation."""
        return self.test_stats_df(model="bistretch", **kw)

    # ------------------------------------------------------------------
    # Rater stats table (most complex -- varies substantially by model)
    # ------------------------------------------------------------------

    def rater_stats_df(
        self,
        model="global",
        anchors=None,
        full=False,
        zstd=False,
        marginal=True,
        alias=False,
        divergence_test=None,
        correction=None,
        alpha=0.05,
        plot=False,
        dp=3,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        method=None,
        constant=None,
        threshold_constant=None,
        robust=False,
        matrix_power=None,
        log_lik_tol=0.000001,
        no_of_samples=500,
        interval=None,
        seed=None,
        filename=None,
        file_format="csv",
    ):
        """
        Build and store the facet_element statistics summary table.

        Output structure varies substantially by model:
          global     — one row per facet_element with scalar facet_effect estimate and fit stats.
          items      — MultiIndex columns (item, statistic), one row per facet_element.
          thresholds — MultiIndex columns (threshold, statistic), one row per facet_element.
          matrix     — marginal=True: twin-vector (per-item + per-threshold marginals
                       recentred to zero); marginal=False: full (item, threshold)
                       cell table.
          centrality, pseudo_halo, bistretch — the model's own raw shift/
                       stretch parameters (lambda, plus omega -- or
                       omega_items and omega_thresholds for bistretch)
                       with their direct bootstrap SEs, one row per
                       facet_element, MultiIndex columns (parameter,
                       statistic), plus (if divergence_test is set) a
                       'Flagged items'/'Flagged categories' column giving
                       each facet element's (e.g. rater's) count of items/thresholds where the
                       stretch reconstruction diverges significantly from
                       the free vector model (see divergence_test below).
                       The reconstructed full-length location vector (per
                       item for pseudo_halo, per threshold for centrality,
                       both for bistretch) that used to be this table's
                       own content is stored separately as
                       self.rater_stats_{model}_full_vector -- if
                       divergence_test is set AND full=True, that table
                       also gets a per-item/per-threshold boolean
                       'Flagged' row.

        Auto-triggers the full calibration/SE/fit chain if not yet run.
        Stores result as self.rater_stats_{model} (and, for centrality/
        pseudo_halo/bistretch, also self.rater_stats_{model}_full_vector).

        Parameters
        ----------
        model : str, default 'global'
            Facet element parameterisation.
        anchors : list or None, default None
            Anchor facet_elements for SE computation.
        full : bool, default False
            If True, sets zstd=True, interval=0.95.
        zstd : bool, default False
            If True, includes Infit Z and Outfit Z columns.
        marginal : bool, default True
            For matrix model only: if True returns marginal twin-vector
            representation; if False returns full (item, threshold) cell table.
        alias : bool, default False
            For centrality/pseudo_halo/bistretch only: if True, labels the
            primary table's parameter columns with their human-readable
            alias ('Global' for lambda; 'Item stretch' or 'Threshold
            stretch' for omega/omega_items/omega_thresholds, depending on
            which axis that model's stretch parameter acts on) instead of
            the raw parameter name. Purely cosmetic -- does not change
            which values are shown, or affect rater_stats_{model}_full_vector
            (which has no lambda/omega columns to alias). Ignored (no
            effect) for every other model.
        divergence_test : {None, 'wald', 't'}, default None
            For centrality/pseudo_halo/bistretch only: if set, runs an
            item-by-item (or threshold-by-threshold, or both for
            bistretch) test comparing the free vector model's own
            per-element estimates against the stretch model's
            reconstruction of them, flagging elements where the simpler
            representation diverges significantly -- see
            _axis_divergence_test for the full method (a single, PAIRED
            bootstrap -- see _get_divergence_inputs) and the reasoning
            behind offering 'wald'/'t' (and not the robust-z alternative
            used for DIF/anchor flagging elsewhere in this package, which
            does not work for this comparison). None (default) skips
            this entirely, since it requires forcing store_bootstrap=True
            on the stretch model's own bootstrap SE computation (a bigger
            bootstrap object footprint while it's kept, though not an
            extra bootstrap pass) -- opt in only when wanted. Adds
            'Flagged items' and/or 'Flagged categories' column(s) to the
            primary table (present regardless of full=). If full=True,
            also adds one column per item and/or threshold directly
            after that summary -- a plain True/False per element, for
            scanning flagged status across all elements at a glance --
            AND adds a per-element boolean 'Flagged' row to
            rater_stats_{model}_full_vector, alongside that element's own
            Estimate/SE/CI (the same information, positioned instead
            next to each element's own detail). Ignored (no effect) for
            every other model.
        correction : {None, 'bh'}, default None
            Only relevant when divergence_test is set. If 'bh',
            Benjamini-Hochberg FDR correction is applied within each
            facet element's own family of tests (that facet element's items, or that
            facet element's thresholds -- not pooled across facet elements). No-op when
            divergence_test=None.
        alpha : float, default 0.05
            Only relevant when divergence_test is set. Significance
            threshold for flagging (applied after correction, if any).
        plot : bool, default False
            Only relevant when divergence_test is set. If True, also
            builds and stores self.fit_plot_{model} -- a scatter of the
            free vector model's estimates against the stretch model's
            reconstruction, flagged elements in a different colour, in
            the same style as self.anchor_plot (see
            plot_anchor_selection). See fit_plot_centrality/
            fit_plot_pseudo_halo/fit_plot_bistretch.
        dp : int, default 3
            Decimal places.
        warm_corr : bool, default True
            Warm bias correction.
        tolerance : float, default 0.00001
            Convergence tolerance.
        max_iters : int, default 100
            Maximum iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment.
        method : str, default 'log-lik'
            Priority vector extraction method.
        constant : float, default 0.1
            Smoothing constant.
        matrix_power : int or None, default None
            Matrix power. None resolves to 0 for method='log-lik'
            (no powering), else 5.
        log_lik_tol : float, default 0.000001
            Calibration convergence tolerance.
        no_of_samples : int, default 500
            Bootstrap samples.
        interval : float or None, default None
            CI width.
        seed : int or None, default None
            Seed passed through to the internal std_errors() call (only
            used if not already computed). None draws fresh entropy.
        filename : str or None, default None
            If given, also saves the returned table to
            ``{filename}.{file_format}`` (path without extension, as for
            the plotting methods).
        file_format : {'csv', 'xlsx'}, default 'csv'
            File format used when filename is given.

        Attributes set
        --------------
        rater_stats_{model} : pandas.DataFrame
            Facet element statistics table. Structure depends on model (see above).

        Returns
        -------
        pandas.DataFrame
            The table, also stored as the attribute listed above.
        """

        model = self._MODEL_ALIASES.get(model, model)
        if full:
            zstd = True
            interval = interval or 0.95

        self._ensure_calibrated(
            model,
            anchors=anchors,
            interval=interval,
            no_of_samples=no_of_samples,
            constant=constant,
            threshold_constant=threshold_constant,
            method=method,
            robust=robust,
            matrix_power=matrix_power,
            log_lik_tol=log_lik_tol,
            warm_corr=warm_corr,
            tolerance=tolerance,
            max_iters=max_iters,
            ext_score_adjustment=ext_score_adjustment,
            seed=seed,
        )
        self._ensure_se(
            model,
            anchors,
            interval,
            no_of_samples,
            constant,
            method,
            matrix_power,
            log_lik_tol,
            threshold_constant=threshold_constant,
            robust=robust,
            seed=seed,
            store_bootstrap=(
                divergence_test is not None
                and model in ("centrality", "pseudo_halo", "bistretch")
            ),
        )
        if not hasattr(self, f"rater_outfit_ms_{model}"):
            self._run_facet_fit(model)

        anc = anchors is not None
        rse = getattr(self, f"rater_se_{model}", {})
        rlo = getattr(self, f"rater_low_{model}", None)
        rhi = getattr(self, f"rater_high_{model}", None)

        if model == "global":
            sev_attr = (
                f"anchor_facet_effects_{model}" if anc else f"facet_effects_{model}"
            )
            facet_effects = getattr(self, sev_attr)
            rse = getattr(
                self, f"anchor_rater_se_{model}" if anc else f"rater_se_{model}", {}
            )
            rlo = getattr(
                self, f"anchor_rater_low_{model}" if anc else f"rater_low_{model}", None
            )
            rhi = getattr(
                self,
                f"anchor_rater_high_{model}" if anc else f"rater_high_{model}",
                None,
            )
            stats = pd.DataFrame({"Estimate": facet_effects.round(dp)})
            if rse is not None:
                stats["SE"] = pd.Series(rse).round(dp)
            if interval is not None and rlo is not None:
                stats[f"{round((1-interval)*50, 1)}%"] = pd.Series(rlo).round(dp)
                stats[f"{round((1+interval)*50, 1)}%"] = pd.Series(rhi).round(dp)
            stats["Count"] = pd.Series(
                {r: self.responses.xs(r).count().sum() for r in self.facet_names}
            )
            stats["Infit MS"] = getattr(self, f"rater_infit_ms_{model}").round(dp)
            if zstd:
                stats["Infit Z"] = getattr(self, f"rater_infit_zstd_{model}").round(dp)
            stats["Outfit MS"] = getattr(self, f"rater_outfit_ms_{model}").round(dp)
            if zstd:
                stats["Outfit Z"] = getattr(self, f"rater_outfit_zstd_{model}").round(
                    dp
                )
            stats.index = self.facet_names
            setattr(self, f"rater_stats_{model}", stats)

        else:
            sev_attr = self._sev_attr(model, anchor=anc)
            facet_effects = getattr(self, sev_attr)
            se_attr = f"anchor_rater_se_{model}" if anc else f"rater_se_{model}"
            rse = getattr(self, se_attr, {})
            lo_attr = f"anchor_rater_low_{model}" if anc else f"rater_low_{model}"
            hi_attr = f"anchor_rater_high_{model}" if anc else f"rater_high_{model}"
            rlo = getattr(self, lo_attr, None)
            rhi = getattr(self, hi_attr, None)

            def _ov_stats():
                """Build the overall facet element (e.g. rater) fit statistics sub-table for the current model."""
                cols = (
                    ["Count", "Infit MS", "Infit Z", "Outfit MS", "Outfit Z"]
                    if zstd
                    else ["Count", "Infit MS", "Outfit MS"]
                )
                ov = pd.DataFrame(index=self.facet_names, columns=cols)
                ov["Count"] = pd.Series(
                    {r: self.responses.xs(r).count().sum() for r in self.facet_names}
                ).astype(int)
                ov["Infit MS"] = getattr(self, f"rater_infit_ms_{model}").round(dp)
                ov["Outfit MS"] = getattr(self, f"rater_outfit_ms_{model}").round(dp)
                if zstd:
                    ov["Infit Z"] = getattr(self, f"rater_infit_zstd_{model}").round(dp)
                    ov["Outfit Z"] = getattr(self, f"rater_outfit_zstd_{model}").round(
                        dp
                    )
                return ov.T

            result = {}
            flagged_col_keys = []  # (group, 'Flagged') columns needing an explicit bool cast below

            if model == "items":
                for item in self.item_names:
                    sub = pd.DataFrame(index=self.facet_names)
                    sub["Estimate"] = facet_effects[item].values.round(dp)
                    if rse is not None and not isinstance(rse, dict) and not rse.empty:
                        sub["SE"] = rse[item].values.round(dp)
                    if interval is not None and rlo is not None:
                        sub[f"{round((1-interval)*50, 1)}%"] = rlo[item].values.round(dp)
                        sub[f"{round((1+interval)*50, 1)}%"] = rhi[item].values.round(dp)
                    result[item] = sub.T

            elif model == "thresholds":
                for t in range(self.max_score):
                    key = f"Threshold {t+1}"
                    sub = pd.DataFrame(index=self.facet_names)
                    sub["Estimate"] = facet_effects.iloc[:, t].values.round(dp)
                    if rse is not None and not isinstance(rse, dict) and not rse.empty:
                        sub["SE"] = rse.iloc[:, t].values.round(dp)
                    if interval is not None and rlo is not None:
                        sub[f"{round((1-interval)*50, 1)}%"] = rlo.iloc[
                            :, t
                        ].values.round(dp)
                        sub[f"{round((1+interval)*50, 1)}%"] = rhi.iloc[
                            :, t
                        ].values.round(dp)
                    result[key] = sub.T

            elif model in ("centrality", "pseudo_halo", "bistretch"):
                # Primary table: the model's own raw shift/stretch parameters
                # (lambda, omega -- or omega_items/omega_thresholds for
                # bistretch), with their direct bootstrap SEs -- not the
                # reconstructed full-vector profile below.
                param_names = {
                    "centrality": ["lambda", "omega"],
                    "pseudo_halo": ["lambda", "omega"],
                    "bistretch": ["lambda", "omega_items", "omega_thresholds"],
                }[model]
                for pname in param_names:
                    est_attr = f"anchor_{pname}_{model}" if anc else f"{pname}_{model}"
                    se_attr = f"anchor_{pname}_se_{model}" if anc else f"{pname}_se_{model}"
                    lo_attr = f"anchor_{pname}_low_{model}" if anc else f"{pname}_low_{model}"
                    hi_attr = f"anchor_{pname}_high_{model}" if anc else f"{pname}_high_{model}"
                    p_est = getattr(self, est_attr)
                    p_se = getattr(self, se_attr, None)
                    sub = pd.DataFrame(index=self.facet_names)
                    sub["Estimate"] = p_est.round(dp)
                    if p_se is not None:
                        sub["SE"] = p_se.round(dp)
                    if interval is not None:
                        p_lo = getattr(self, lo_attr, None)
                        p_hi = getattr(self, hi_attr, None)
                        if p_lo is not None and p_hi is not None:
                            sub[f"{round((1-interval)*50, 1)}%"] = p_lo.round(dp)
                            sub[f"{round((1+interval)*50, 1)}%"] = p_hi.round(dp)
                    col_key = self._RATER_PARAM_ALIAS_LABELS[model][pname] if alias else pname
                    result[col_key] = sub.T

                # Divergence test (opt-in): free vector model vs stretch
                # reconstruction, item-by-item (or threshold-by-threshold,
                # or both for bistretch). See _axis_divergence_test.
                div_results = None
                if divergence_test is not None:
                    div_inputs = self._get_divergence_inputs(model, anc)
                    div_results = {
                        axis: self._axis_divergence_test(
                            *div_inputs[axis], test=divergence_test, correction=correction,
                            alpha=alpha, no_of_samples=no_of_samples,
                        )
                        for axis in div_inputs
                    }
                    if "items" in div_results:
                        result["Flagged items"] = pd.DataFrame(
                            {"Count": div_results["items"][2]["Flagged"]}, index=self.facet_names
                        ).T
                    if "thresholds" in div_results:
                        result["Flagged categories"] = pd.DataFrame(
                            {"Count": div_results["thresholds"][2]["Flagged"]}, index=self.facet_names
                        ).T
                    # full=True: one column per item/threshold, straight
                    # after the Flagged items/categories summary above --
                    # a plain True/False per element, for scanning flagged
                    # status across all elements at a glance, alongside
                    # (not instead of) the per-element 'Flagged' row folded
                    # into each element's own Estimate/SE/CI block in
                    # rater_stats_{model}_full_vector below.
                    if full:
                        if "items" in div_results:
                            item_flagged = div_results["items"][1]
                            for item in item_flagged.columns:
                                result[item] = pd.DataFrame(
                                    {"Flagged": item_flagged[item]}, index=self.facet_names
                                ).T
                                flagged_col_keys.append(item)
                        if "thresholds" in div_results:
                            thr_flagged = div_results["thresholds"][1]
                            for t in thr_flagged.columns:
                                key = f"Threshold {t}"
                                result[key] = pd.DataFrame(
                                    {"Flagged": thr_flagged[t]}, index=self.facet_names
                                ).T
                                flagged_col_keys.append(key)

                # Secondary table: the reconstructed full (item- or
                # threshold-length, or both for bistretch) severity vector,
                # stored separately as rater_stats_{model}_full_vector --
                # this is what rater_stats_{model} itself used to contain.
                fv_result = {}
                if model == "pseudo_halo":
                    flagged_items = div_results["items"][1] if div_results else None
                    for item in self.item_names:
                        sub = pd.DataFrame(index=self.facet_names)
                        sub["Estimate"] = facet_effects[item].values.round(dp)
                        if rse is not None and not isinstance(rse, dict) and not rse.empty:
                            sub["SE"] = rse[item].values.round(dp)
                        if interval is not None and rlo is not None:
                            sub[f"{round((1-interval)*50, 1)}%"] = rlo[item].values.round(dp)
                            sub[f"{round((1+interval)*50, 1)}%"] = rhi[item].values.round(dp)
                        if full and flagged_items is not None:
                            sub["Flagged"] = flagged_items[item].values
                        fv_result[item] = sub.T
                elif model == "centrality":
                    flagged_thresholds = div_results["thresholds"][1] if div_results else None
                    for t in range(self.max_score):
                        key = f"Threshold {t+1}"
                        sub = pd.DataFrame(index=self.facet_names)
                        sub["Estimate"] = facet_effects.iloc[:, t].values.round(dp)
                        if rse is not None and not isinstance(rse, dict) and not rse.empty:
                            sub["SE"] = rse.iloc[:, t].values.round(dp)
                        if interval is not None and rlo is not None:
                            sub[f"{round((1-interval)*50, 1)}%"] = rlo.iloc[:, t].values.round(dp)
                            sub[f"{round((1+interval)*50, 1)}%"] = rhi.iloc[:, t].values.round(dp)
                        if full and flagged_thresholds is not None:
                            sub["Flagged"] = flagged_thresholds.iloc[:, t].values
                        fv_result[key] = sub.T
                else:  # bistretch -- dual reconstruction, same pattern as bivector below
                    mg_i_attr = (
                        "anchor_facet_effects_bistretch_items" if anc else "facet_effects_bistretch_items"
                    )
                    mg_t_attr = (
                        "anchor_facet_effects_bistretch_thresholds" if anc else "facet_effects_bistretch_thresholds"
                    )
                    mg_items = getattr(self, mg_i_attr)
                    mg_thrs = getattr(self, mg_t_attr)
                    mg_se_i = getattr(
                        self, "anchor_rater_se_marginal_items" if anc else "rater_se_marginal_items", None
                    )
                    mg_se_t = getattr(
                        self, "anchor_rater_se_marginal_thresholds" if anc else "rater_se_marginal_thresholds", None
                    )
                    flagged_items = div_results["items"][1] if div_results else None
                    flagged_thresholds = div_results["thresholds"][1] if div_results else None
                    for item in self.item_names:
                        sub = pd.DataFrame(index=self.facet_names)
                        sub["Estimate"] = mg_items[item].values.round(dp)
                        if mg_se_i is not None:
                            sub["SE"] = mg_se_i[item].values.round(dp)
                        if full and flagged_items is not None:
                            sub["Flagged"] = flagged_items[item].values
                        fv_result[item] = sub.T
                    for t in range(self.max_score):
                        key = f"Threshold {t+1}"
                        sub = pd.DataFrame(index=self.facet_names)
                        sub["Estimate"] = mg_thrs.iloc[:, t].values.round(dp)
                        if mg_se_t is not None:
                            sub["SE"] = mg_se_t.iloc[:, t].values.round(dp)
                        if full and flagged_thresholds is not None:
                            sub["Flagged"] = flagged_thresholds.iloc[:, t].values
                        fv_result[key] = sub.T

                fv_result["Overall statistics"] = _ov_stats()
                fv_stats = pd.concat(fv_result.values(), keys=fv_result.keys()).T
                setattr(self, f"rater_stats_{model}_full_vector", fv_stats)

                if plot and div_results is not None:
                    setattr(
                        self, f"fit_plot_{model}",
                        self._build_fit_plot(model, div_inputs, div_results),
                    )

            elif model == "bivector":
                mg_i_attr = (
                    "anchor_facet_effects_bivector_items"
                    if anc
                    else "facet_effects_bivector_items"
                )
                mg_t_attr = (
                    "anchor_facet_effects_bivector_thresholds"
                    if anc
                    else "facet_effects_bivector_thresholds"
                )
                mg_items = getattr(self, mg_i_attr)  # (R, I) DataFrame
                mg_thrs = getattr(self, mg_t_attr)  # (R, K+1) DataFrame
                mg_se_i = getattr(
                    self,
                    (
                        "anchor_rater_se_marginal_items"
                        if anc
                        else "rater_se_marginal_items"
                    ),
                    None,
                )
                mg_se_t = getattr(
                    self,
                    (
                        "anchor_rater_se_marginal_thresholds"
                        if anc
                        else "rater_se_marginal_thresholds"
                    ),
                    None,
                )

                for item in self.item_names:
                    sub = pd.DataFrame(index=self.facet_names)
                    sub["Estimate"] = mg_items[item].values.round(dp)
                    if mg_se_i is not None:
                        sub["SE"] = mg_se_i[item].values.round(dp)
                    result[item] = sub.T

                for t in range(self.max_score):
                    key = f"Threshold {t+1}"
                    sub = pd.DataFrame(index=self.facet_names)
                    sub["Estimate"] = mg_thrs.iloc[:, t].values.round(dp)
                    if mg_se_t is not None:
                        sub["SE"] = mg_se_t.iloc[:, t].values.round(dp)
                    result[key] = sub.T

            elif model == "matrix":
                if marginal:
                    mg_i_attr = (
                        "anchor_marginal_facet_effects_items"
                        if anc
                        else "marginal_facet_effects_items"
                    )
                    mg_t_attr = (
                        "anchor_marginal_facet_effects_thresholds"
                        if anc
                        else "marginal_facet_effects_thresholds"
                    )
                    mg_items = getattr(self, mg_i_attr)  # (R, I) DataFrame
                    mg_thrs = getattr(self, mg_t_attr)  # (R, K+1) DataFrame
                    mg_se_i = getattr(
                        self,
                        (
                            "anchor_rater_se_marginal_items"
                            if anc
                            else "rater_se_marginal_items"
                        ),
                        None,
                    )
                    mg_se_t = getattr(
                        self,
                        (
                            "anchor_rater_se_marginal_thresholds"
                            if anc
                            else "rater_se_marginal_thresholds"
                        ),
                        None,
                    )

                    for item in self.item_names:
                        sub = pd.DataFrame(index=self.facet_names)
                        sub["Estimate"] = mg_items[item].values.round(dp)
                        if mg_se_i is not None:
                            sub["SE"] = mg_se_i[item].values.round(dp)
                        result[item] = sub.T

                    for t in range(self.max_score):
                        key = f"Threshold {t+1}"
                        sub = pd.DataFrame(index=self.facet_names)
                        sub["Estimate"] = mg_thrs.iloc[:, t].values.round(dp)
                        if mg_se_t is not None:
                            sub["SE"] = mg_se_t.iloc[:, t].values.round(dp)
                        result[key] = sub.T

                else:
                    for item in self.item_names:
                        for t in range(1, self.max_score + 1):
                            key = f"{item}, Threshold {t}"
                            sub = pd.DataFrame(index=self.facet_names)
                            sub["Estimate"] = facet_effects.loc[
                                (slice(None), item), t
                            ].values.round(dp)
                            if (
                                rse is not None
                                and not isinstance(rse, dict)
                                and not rse.empty
                            ):
                                sub["SE"] = rse.loc[
                                    (slice(None), item), t
                                ].values.round(dp)
                            if interval is not None and rlo is not None:
                                sub[f"{round((1-interval)*50, 1)}%"] = rlo.loc[
                                    (slice(None), item), t
                                ].values.round(dp)
                                sub[f"{round((1+interval)*50, 1)}%"] = rhi.loc[
                                    (slice(None), item), t
                                ].values.round(dp)
                            result[key] = sub.T

            elif model == "mixed":
                # facet_effects_mixed / anchor_facet_effects_mixed are
                # aliased (via _sync_mixed) to the same (rater, item) x
                # threshold cell shape as facet_effects_matrix -- so this
                # is a verbatim copy of the matrix model's marginal=False
                # (full cell table) path above. There is no marginal=True
                # representation for 'mixed': it blends a different
                # sub-parameterisation per rater (self.rater_models), so
                # there is no single restricted per-rater marginal to
                # derive the way there is for matrix -- see known-issues
                # #36. marginal is ignored here; always the full table.
                if marginal:
                    warnings.warn(
                        "rater_stats_df(model='mixed') has no marginal=True "
                        "representation -- 'mixed' assigns a different "
                        f"{self.facet_label_eg} parameterisation per {self.facet_label} (self.rater_models), "
                        f"so there is no single restricted per-{self.facet_label} marginal "
                        "to report, unlike 'matrix'. Returning the full "
                        "(item, threshold) cell table, as if marginal=False.",
                        UserWarning,
                        stacklevel=2,
                    )
                for item in self.item_names:
                    for t in range(1, self.max_score + 1):
                        key = f"{item}, Threshold {t}"
                        sub = pd.DataFrame(index=self.facet_names)
                        sub["Estimate"] = facet_effects.loc[
                            (slice(None), item), t
                        ].values.round(dp)
                        if (
                            rse is not None
                            and not isinstance(rse, dict)
                            and not rse.empty
                        ):
                            sub["SE"] = rse.loc[(slice(None), item), t].values.round(
                                dp
                            )
                        if interval is not None and rlo is not None:
                            sub[f"{round((1-interval)*50, 1)}%"] = rlo.loc[
                                (slice(None), item), t
                            ].values.round(dp)
                            sub[f"{round((1+interval)*50, 1)}%"] = rhi.loc[
                                (slice(None), item), t
                            ].values.round(dp)
                        result[key] = sub.T

                # The one thing genuinely specific to 'mixed': which
                # sub-parameterisation each rater was assigned.
                models_attr = "anchor_rater_models" if anc else "rater_models"
                rater_models = getattr(self, models_attr, None)
                if rater_models is not None:
                    sub = pd.DataFrame(index=self.facet_names)
                    sub["Model"] = rater_models.reindex(self.facet_names).values
                    result["Rater model"] = sub.T

            result["Overall statistics"] = _ov_stats()
            stats = pd.concat(result.values(), keys=result.keys()).T
            if flagged_col_keys:
                # The (group, 'Flagged') columns get upcast to float by the
                # concat+transpose above (their bool rows getting stacked
                # against every other group's float rows in the
                # intermediate, rater-columned representation forces a
                # common dtype per rater-column) -- cast back explicitly so
                # these read as True/False, not 1.0/0.0.
                flagged_cols = [(k, "Flagged") for k in flagged_col_keys]
                stats[flagged_cols] = stats[flagged_cols].astype(bool)
            if "Rater model" in result:
                # Same dtype-taint pattern as flagged_cols above, but from
                # mixing a genuinely non-numeric (string) group -- the
                # concat+transpose above forces every column to object
                # dtype once one row is text, not just the text one's own
                # column. Cast every other column back to float; leave
                # ('Rater model', 'Model') as the object/string dtype it
                # actually is.
                numeric_cols = [c for c in stats.columns if c[0] != "Rater model"]
                stats[numeric_cols] = stats[numeric_cols].astype(float)
            setattr(self, f"rater_stats_{model}", stats)
        self._set_facet_aliases(model)
        table = getattr(self, f"rater_stats_{model}")
        _save_table(table, filename, file_format)
        return table

    def rater_stats_df_global(self, **kw):
        """Alias for rater_stats_df(model='global'). See rater_stats_df for full documentation."""
        return self.rater_stats_df(model="global", **kw)

    def rater_stats_df_items(self, **kw):
        """Alias for rater_stats_df(model='items'). See rater_stats_df for full documentation."""
        return self.rater_stats_df(model="items", **kw)

    def rater_stats_df_thresholds(self, **kw):
        """Alias for rater_stats_df(model='thresholds'). See rater_stats_df for full documentation."""
        return self.rater_stats_df(model="thresholds", **kw)

    def rater_stats_df_matrix(self, **kw):
        """Alias for rater_stats_df(model='matrix'). See rater_stats_df for full documentation."""
        return self.rater_stats_df(model="matrix", **kw)

    def rater_stats_df_bivector(self, **kw):
        """Alias for rater_stats_df(model='bivector'). See rater_stats_df for full documentation."""
        return self.rater_stats_df(model="bivector", **kw)

    # ------------------------------------------------------------------
    # Save statistics
    # ------------------------------------------------------------------
    def rater_stats_df_centrality(self, **kw):
        """Alias for rater_stats_df(model='centrality'). See rater_stats_df for full documentation."""
        return self.rater_stats_df(model="centrality", **kw)
    def rater_stats_df_pseudo_halo(self, **kw):
        """Alias for rater_stats_df(model='pseudo_halo'). See rater_stats_df for full documentation."""
        return self.rater_stats_df(model="pseudo_halo", **kw)
    def rater_stats_df_bistretch(self, **kw):
        """Alias for rater_stats_df(model='bistretch'). See rater_stats_df for full documentation."""
        return self.rater_stats_df(model="bistretch", **kw)

    def save_stats(
        self,
        model="global",
        filename="",
        format="csv",
        alias=False,
        divergence_test=None,
        correction=None,
        alpha=0.05,
        plot=False,
        dp=3,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        method=None,
        constant=None,
        threshold_constant=None,
        robust=False,
        matrix_power=None,
        log_lik_tol=0.000001,
        no_of_samples=500,
        interval=None,
    ):
        """
        Export item, threshold, facet_element, person, and test statistics to file.

        Auto-triggers all stats_df methods if not yet run. Saves all five
        tables to either a single Excel workbook or separate CSV files. For
        model in ('centrality', 'pseudo_halo', 'bistretch'), also saves a
        sixth table/file -- rater_stats_{model}_full_vector, the
        reconstructed full-length location vector (see rater_stats_df).

        Parameters
        ----------
        model : str, default 'global'
            Facet element (e.g. rater) parameterisation.
        filename : str, default ''
            Output filename or path (without extension for CSV).
        format : str, default 'csv'
            'csv' saves five separate CSV files. 'xlsx' saves to a single
            workbook with separate sheets.
        alias : bool, default False
            For centrality/pseudo_halo/bistretch only: passed straight
            through to the internal rater_stats_df() call, labelling that
            table's parameter columns with their human-readable alias
            ('Global', 'Item stretch', 'Threshold stretch') instead of the
            raw parameter name. See rater_stats_df for details. Ignored
            (no effect) for every other model. Note this only takes effect
            if rater_stats_{model} hasn't already been computed -- like
            every other argument here, it's skipped if that table already
            exists (see Notes below).
        divergence_test : {None, 'wald', 't'}, default None
            For centrality/pseudo_halo/bistretch only: passed straight
            through to the internal rater_stats_df() call. See
            rater_stats_df for details. Note the per-element 'Flagged'
            breakdown in rater_stats_{model}_full_vector requires
            rater_stats_df's own full=True, which save_stats does not
            currently expose -- only the 'Flagged items'/'Flagged
            categories' summary column(s) in the primary table will
            appear here.
        correction : {None, 'bh'}, default None
            Passed straight through. No-op when divergence_test=None.
        alpha : float, default 0.05
            Passed straight through. Only relevant when divergence_test
            is set.
        plot : bool, default False
            Passed straight through. If True, also builds and stores
            self.fit_plot_{model}. Only relevant when divergence_test is
            set.
        dp : int, default 3
            Decimal places.
        warm_corr : bool, default True
            Warm bias correction.
        tolerance : float, default 0.00001
            Convergence tolerance.
        max_iters : int, default 100
            Maximum iterations.
        ext_score_adjustment : float, default 0.5
            Extreme score adjustment.
        method : str, default 'log-lik'
            Priority vector extraction method.
        constant : float, default 0.1
            Smoothing constant.
        matrix_power : int or None, default None
            Matrix power. None resolves to 0 for method='log-lik'
            (no powering), else 5.
        log_lik_tol : float, default 0.000001
            Calibration convergence tolerance.
        no_of_samples : int, default 500
            Bootstrap samples.
        interval : float or None, default None
            CI width for SEs.
        """

        model = self._MODEL_ALIASES.get(model, model)
        kw = dict(
            dp=dp,
            warm_corr=warm_corr,
            tolerance=tolerance,
            max_iters=max_iters,
            ext_score_adjustment=ext_score_adjustment,
            method=method,
            constant=constant,
            threshold_constant=threshold_constant,
            robust=robust,
            matrix_power=matrix_power,
            log_lik_tol=log_lik_tol,
        )

        for attr, method_name, extra in [
            (
                f"item_stats_{model}",
                "item_stats_df",
                dict(no_of_samples=no_of_samples, interval=interval),
            ),
            (
                f"threshold_stats_{model}",
                "threshold_stats_df",
                dict(no_of_samples=no_of_samples, interval=interval),
            ),
            (
                f"rater_stats_{model}",
                "rater_stats_df",
                dict(
                    no_of_samples=no_of_samples, interval=interval, alias=alias,
                    divergence_test=divergence_test, correction=correction,
                    alpha=alpha, plot=plot,
                ),
            ),
            (f"person_stats_{model}", "person_stats_df", {}),
            (f"test_stats_{model}", "test_stats_df", {}),
        ]:
            if not hasattr(self, attr):
                getattr(self, method_name)(model=model, **kw, **extra)

        if format == "xlsx":
            if not filename.endswith(".xlsx"):
                filename += ".xlsx"
            with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                getattr(self, f"item_stats_{model}").to_excel(
                    writer, sheet_name="Item statistics"
                )
                getattr(self, f"threshold_stats_{model}").to_excel(
                    writer, sheet_name="Threshold statistics"
                )
                getattr(self, f"rater_stats_{model}").to_excel(
                    writer, sheet_name="Rater statistics"
                )
                getattr(self, f"person_stats_{model}").to_excel(
                    writer, sheet_name="Person statistics"
                )
                getattr(self, f"test_stats_{model}").to_excel(
                    writer, sheet_name="Test statistics"
                )
                if model in ("centrality", "pseudo_halo", "bistretch"):
                    getattr(self, f"rater_stats_{model}_full_vector").to_excel(
                        writer, sheet_name="Rater statistics (full vector)"
                    )
        else:
            if filename.endswith(".csv"):
                filename = filename[:-4]
            getattr(self, f"item_stats_{model}").to_csv(f"{filename}_item_stats.csv")
            getattr(self, f"threshold_stats_{model}").to_csv(
                f"{filename}_threshold_stats.csv"
            )
            getattr(self, f"rater_stats_{model}").to_csv(f"{filename}_rater_stats.csv")
            if model in ("centrality", "pseudo_halo", "bistretch"):
                getattr(self, f"rater_stats_{model}_full_vector").to_csv(
                    f"{filename}_rater_stats_full_vector.csv"
                )
            getattr(self, f"person_stats_{model}").to_csv(
                f"{filename}_person_stats.csv"
            )
            getattr(self, f"test_stats_{model}").to_csv(f"{filename}_test_stats.csv")

    def save_stats_global(self, **kw):
        """Alias for save_stats(model='global'). See save_stats for full documentation."""
        self.save_stats(model="global", **kw)

    def save_stats_items(self, **kw):
        """Alias for save_stats(model='items'). See save_stats for full documentation."""
        self.save_stats(model="items", **kw)

    def save_stats_thresholds(self, **kw):
        """Alias for save_stats(model='thresholds'). See save_stats for full documentation."""
        self.save_stats(model="thresholds", **kw)

    def save_stats_matrix(self, **kw):
        """Alias for save_stats(model='matrix'). See save_stats for full documentation."""
        self.save_stats(model="matrix", **kw)

    def save_stats_bivector(self, **kw):
        """Alias for save_stats(model='bivector'). See save_stats for full documentation."""
        self.save_stats(model="bivector", **kw)
    def save_stats_centrality(self, **kw):
        """Alias for save_stats(model='centrality'). See save_stats for full documentation."""
        self.save_stats(model="centrality", **kw)
    def save_stats_pseudo_halo(self, **kw):
        """Alias for save_stats(model='pseudo_halo'). See save_stats for full documentation."""
        self.save_stats(model="pseudo_halo", **kw)
    def save_stats_bistretch(self, **kw):
        """Alias for save_stats(model='bistretch'). See save_stats for full documentation."""
        self.save_stats(model="bistretch", **kw)

    def save_residuals(
        self,
        eigenvectors,
        eigenvalues,
        variance_explained,
        loadings,
        fit_statistics_method,
        eigenvector_string,
        filename,
        format="csv",
        single=True,
        dp=3,
        **kw,
    ):
        """
        Export residual correlation analysis results to file.

        Low-level method called by save_residuals_items_* and
        save_residuals_raters_* aliases. Auto-triggers the fit statistics
        method if the eigenvectors attribute is not yet set.

        Parameters
        ----------
        eigenvectors : pandas.DataFrame or None
            PCA eigenvectors to save.
        eigenvalues : pandas.DataFrame or None
            PCA eigenvalues to save.
        variance_explained : pandas.DataFrame or None
            PCA variance explained proportions to save.
        loadings : pandas.DataFrame or None
            PCA loadings to save.
        fit_statistics_method : str
            Name of the method to call if eigenvectors are not yet computed
            (e.g. 'item_res_corr_analysis_global').
        eigenvector_string : str
            Attribute name to check for existence (e.g. 'item_eigenvectors_global').
        filename : str
            Output filename or path.
        format : str, default 'csv'
            'csv' or 'xlsx'.
        single : bool, default True
            If True, writes all tables to a single file/sheet.
            If False, writes each to a separate file/sheet.
        dp : int, default 3
            Decimal places.
        **kw
            Additional keyword arguments passed to the fit statistics method.
        """

        frames = [eigenvectors, eigenvalues, variance_explained, loadings]
        if not hasattr(self, eigenvector_string):
            getattr(self, fit_statistics_method)(**kw)

        if format == "xlsx":
            if not filename.endswith(".xlsx"):
                filename += ".xlsx"
            with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                if single:
                    row = 0
                    for frame in frames:
                        frame.round(dp).to_excel(
                            writer,
                            sheet_name="Residual analysis",
                            startrow=row,
                            startcol=0,
                        )
                        row += frame.shape[0] + 2
                else:
                    for frame, sheet in zip(
                        frames,
                        [
                            "Eigenvectors",
                            "Eigenvalues",
                            "Variance explained",
                            "Loadings",
                        ],
                    ):
                        frame.round(dp).to_excel(writer, sheet_name=sheet)
        else:
            if single:
                if not filename.endswith(".csv"):
                    filename += ".csv"
                with open(filename, "a") as f:
                    for frame in frames:
                        frame.round(dp).to_csv(f)
                        f.write("\n")
            else:
                if filename.endswith(".csv"):
                    filename = filename[:-4]
                for frame, suffix in zip(
                    frames,
                    [
                        "_eigenvectors",
                        "_eigenvalues",
                        "_variance_explained",
                        "_loadings",
                    ],
                ):
                    frame.round(dp).to_csv(f"{filename}{suffix}.csv")

    def _save_residuals_for(self, model, which, filename, **kw):
        """Shared implementation for save_residuals_items/facet_elements aliases."""
        attr = f"{which}_eigenvectors_{model}"
        if not hasattr(self, attr):
            runner = (
                self._run_item_res_corr if which == "item" else self._run_facet_res_corr
            )
            runner(model, **kw)
        self.save_residuals(
            getattr(self, f"{which}_eigenvectors_{model}"),
            getattr(self, f"{which}_eigenvalues_{model}"),
            getattr(self, f"{which}_variance_explained_{model}"),
            getattr(self, f"{which}_loadings_{model}"),
            f"{which}_res_corr_analysis_{model}",
            attr,
            filename,
            **kw,
        )

    def save_residuals_items_global(self, filename, **kw):
        """Alias for save_residuals_items(model='global'). See save_residuals_items for full documentation."""
        self._save_residuals_for("global", "item", filename, **kw)

    def save_residuals_items_items(self, filename, **kw):
        """Alias for save_residuals_items(model='items'). See save_residuals_items for full documentation."""
        self._save_residuals_for("items", "item", filename, **kw)

    def save_residuals_items_thresholds(self, filename, **kw):
        """Alias for save_residuals_items(model='thresholds'). See save_residuals_items for full documentation."""
        self._save_residuals_for("thresholds", "item", filename, **kw)

    def save_residuals_items_matrix(self, filename, **kw):
        """Alias for save_residuals_items(model='matrix'). See save_residuals_items for full documentation."""
        self._save_residuals_for("matrix", "item", filename, **kw)

    def save_residuals_items_bivector(self, filename, **kw):
        """Alias for save_residuals_items(model='bivector'). See save_residuals_items for full documentation."""
        self._save_residuals_for("bivector", "item", filename, **kw)
    def save_residuals_items_centrality(self, filename, **kw):
        """Alias for save_residuals_centrality(model='centrality'). See save_residuals_items for full documentation."""
        self._save_residuals_for("centrality", "item", filename, **kw)
    def save_residuals_items_pseudo_halo(self, filename, **kw):
        """Alias for save_residuals_pseudo_halo(model='pseudo_halo'). See save_residuals_items for full documentation."""
        self._save_residuals_for("pseudo_halo", "item", filename, **kw)
    def save_residuals_items_bistretch(self, filename, **kw):
        """Alias for save_residuals_bistretch(model='bistretch'). See save_residuals_items for full documentation."""
        self._save_residuals_for("bistretch", "item", filename, **kw)

    def save_residuals_raters_global(self, filename, **kw):
        """Alias for save_residuals_raters(model='global'). See save_residuals_raters for full documentation."""
        self._save_residuals_for("global", "rater", filename, **kw)

    def save_residuals_raters_items(self, filename, **kw):
        """Alias for save_residuals_raters(model='items'). See save_residuals_raters for full documentation."""
        self._save_residuals_for("items", "rater", filename, **kw)

    def save_residuals_raters_thresholds(self, filename, **kw):
        """Alias for save_residuals_raters(model='thresholds'). See save_residuals_raters for full documentation."""
        self._save_residuals_for("thresholds", "rater", filename, **kw)

    def save_residuals_raters_matrix(self, filename, **kw):
        """Alias for save_residuals_raters(model='matrix'). See save_residuals_raters for full documentation."""
        self._save_residuals_for("matrix", "rater", filename, **kw)

    def save_residuals_raters_bivector(self, filename, **kw):
        """Alias for save_residuals_raters(model='bivector'). See save_residuals_raters for full documentation."""
        self._save_residuals_for("bivector", "rater", filename, **kw)

    # ------------------------------------------------------------------
    # Class intervals
    # ------------------------------------------------------------------

    @staticmethod
    def _class_masks(estimates, no_of_classes):
        """Compute class interval index masks from person_location values."""
        class_groups = [f"class_{i + 1}" for i in range(no_of_classes)]
        q = estimates.quantile(
            [(i + 1) / no_of_classes for i in range(no_of_classes - 1)]
        )
        mask = {
            "class_1": estimates < q.values[0],
            f"class_{no_of_classes}": estimates >= q.values[-1],
            **{
                f"class_{i + 2}": (
                    (estimates >= q.values[i]) & (estimates < q.values[i + 1])
                )
                for i in range(no_of_classes - 2)
            },
        }
        return {cg: mask[cg][mask[cg]].index for cg in class_groups}
    def save_residuals_raters_centrality(self, filename, **kw):
        """Alias for save_residuals_raters(model='centrality'). See save_residuals_raters for full documentation."""
        self._save_residuals_for("centrality", "rater", filename, **kw)
    def save_residuals_raters_pseudo_halo(self, filename, **kw):
        """Alias for save_residuals_raters(model='pseudo_halo'). See save_residuals_raters for full documentation."""
        self._save_residuals_for("pseudo_halo", "rater", filename, **kw)
    def save_residuals_raters_bistretch(self, filename, **kw):
        """Alias for save_residuals_raters(model='bistretch'). See save_residuals_raters for full documentation."""
        self._save_residuals_for("bistretch", "rater", filename, **kw)

    def _facet_effect_item_offset(self, model, facet_effects, facet_element):
        """Return per-item facet_effect offset Series for a given facet_element and model."""
        if model == "global":
            return pd.Series(
                float(facet_effects.loc[facet_element]), index=self.item_names
            )
        elif model in ("items", "pseudo_halo"):
            return facet_effects.loc[facet_element]
        elif model in ("thresholds", "centrality"):
            return pd.Series(
                float(facet_effects.loc[facet_element].mean()), index=self.item_names
            )
        elif model in ("bivector", "matrix", "bistretch"):
            # facet_effects is MultiIndex (facet_element, item) × thresholds
            return facet_effects.loc[facet_element].mean(axis=1)

    def _zero_facet_effects(self, model, facet_effects):
        """Return facet_effects structure identical in shape but all values zero.
        Used to evaluate neutral (facet_effect=0) curves in plotting methods."""
        if model == "global":
            return pd.Series(0.0, index=facet_effects.index)
        elif model in ("items", "thresholds", "centrality", "pseudo_halo"):
            return pd.DataFrame(0.0, index=facet_effects.index, columns=facet_effects.columns)
        elif model in ("bivector", "matrix", "bistretch"):
            return pd.DataFrame(0.0, index=facet_effects.index, columns=facet_effects.columns)

    def _mean_facet_effects(self, model, facet_effects, facet_element):
        """Return a facet_effects structure where `facet_element` has the mean facet_effect
        across all facet_elements.  Used to plot curves that match obs averaged across
        the full facet_element pool."""
        if model == "global":
            result = facet_effects.copy()
            result[facet_element] = float(facet_effects.mean())
            return result
        elif model in ("items", "thresholds", "centrality", "pseudo_halo"):
            result = facet_effects.copy()
            result.loc[facet_element] = facet_effects.mean(axis=0)
            return result
        elif model in ("bivector", "matrix", "bistretch"):
            result = facet_effects.copy()
            result.loc[facet_element] = facet_effects.groupby(level=1).mean()
            return result

    def class_intervals(
        self, person_locations, items=None, facet_elements=None, shift=0, no_of_classes=5
    ):
        """Class intervals for TCC/ICC observed data overlay."""
        if isinstance(items, str) and items in ("all", "none"):
            items = None
        if isinstance(facet_elements, str):
            if facet_elements in ("none", "zero"):
                facet_elements = None
            elif facet_elements == "all":
                facet_elements = self.facet_names.tolist()
            else:
                facet_elements = [facet_elements]

        class_groups = [f"class_{i + 1}" for i in range(no_of_classes)]
        df = self.responses.copy()

        # Get person index (persons with non-missing data on relevant items)
        if items is None:
            abil_index = self.responses.unstack(level=0).dropna(how="any").index
        else:
            abil_index = self.responses[items].unstack(level=0).dropna(how="any").index

        estimates = person_locations.loc[abil_index]

        # Subset by facet_elements
        if isinstance(facet_elements, list):
            df = pd.concat({r: df.xs(r) for r in facet_elements}, keys=facet_elements)

        # Subset by items (after facet_element subsetting to preserve index structure)
        if items is not None:
            df = df[items]

        # Subset by person index — handle string vs list items separately
        # When items is a single string, df[items] is a Series; pd.IndexSlice
        # with three levels raises "Too many indexers" on a Series, so use
        # xs+loc instead.
        if isinstance(items, str):
            rater_list = (
                facet_elements
                if isinstance(facet_elements, list)
                else list(self.facet_names)
            )
            df = pd.concat(
                {r: df.xs(r).loc[abil_index] for r in rater_list}, keys=rater_list
            )
        elif isinstance(items, list):
            df = df.loc[pd.IndexSlice[:, abil_index], :]
        else:
            df = df.loc[pd.IndexSlice[:, abil_index], :]

        # Class quantile masks
        quantiles = estimates.quantile(
            [(i + 1) / no_of_classes for i in range(no_of_classes - 1)]
        )
        mask_dict = {
            "class_1": estimates < quantiles.values[0],
            f"class_{no_of_classes}": estimates >= quantiles.values[-1],
        }
        for i in range(no_of_classes - 2):
            mask_dict[f"class_{i + 2}"] = (estimates >= quantiles.values[i]) & (
                estimates < quantiles.values[i + 1]
            )

        # Expand masks to (Rater, Person) MultiIndex
        rater_list = (
            list(self.facet_names) if facet_elements is None else facet_elements
        )
        df_mask_dict = {}
        for cg in class_groups:
            expanded = pd.concat(
                {r: mask_dict[cg] for r in rater_list}, keys=rater_list
            )
            df_mask_dict[cg] = expanded[expanded].index

        mean_abilities = (
            pd.Series({cg: estimates[mask_dict[cg]].mean() for cg in class_groups})
            - shift
        )

        if facet_elements is None:
            obs = pd.Series(
                {cg: df.loc[df_mask_dict[cg]].mean().sum() for cg in class_groups}
            )
        else:
            obs = pd.Series(
                {
                    cg: sum(
                        (
                            df.xs(r)
                            .loc[
                                df_mask_dict[cg][
                                    df_mask_dict[cg].get_level_values(0) == r
                                ].get_level_values(1)
                            ]
                            .mean()
                            .sum()
                            if (df_mask_dict[cg].get_level_values(0) == r).any()
                            else 0.0
                        )
                        for r in facet_elements
                    )
                    for cg in class_groups
                }
            )

        return mean_abilities, obs

    def class_intervals_cats(
        self,
        person_locations,
        item_locations,
        thresholds,
        facet_effects,
        model="global",
        item=None,
        facet_element=None,
        shift=0,
        no_of_classes=5,
    ):
        """Class intervals for CRC observed data overlay."""
        if facet_element in ("none", "zero"):
            facet_element = None

        class_groups = [f"class_{i + 1}" for i in range(no_of_classes)]
        df = self.responses.copy()

        # Build person_location DataFrame: (Person, Items)
        abil_df = pd.DataFrame({it: person_locations for it in self.responses.columns})
        raw_abil_base = abil_df.copy()
        if item is None:
            for it in self.responses.columns:
                abil_df[it] -= float(item_locations[it])

        # Subtract facet_element effect from person_location
        abil_dict = {}
        for r in self.facet_names:
            a = abil_df.copy()
            if facet_element is None:
                sev = self._facet_effect_item_offset(model, facet_effects, r)
                for it in self.responses.columns:
                    a[it] -= float(sev[it])
            abil_dict[r] = a
        abil_df_full = pd.concat(abil_dict.values(), keys=abil_dict.keys())

        # Subset by item/facet_element
        if item is None and facet_element is None:
            pf = self.responses.notna().astype(float).replace(0, np.nan)
            abil_full = abil_df_full * pf
            mask_scores = df.unstack().unstack()
            mask_estimates = abil_full.unstack().unstack()
        elif item is None and facet_element is not None:
            df_r = df.xs(facet_element)
            pf = df_r.notna().astype(float).replace(0, np.nan)
            mask_scores = df_r.unstack()
            mask_estimates = (abil_df_full.xs(facet_element) * pf).unstack()
        elif item is not None and facet_element is None:
            df_i = df[item].unstack(level=0)
            pf = df_i.notna().astype(float).replace(0, np.nan)
            mask_scores = df_i.unstack()
            mask_estimates = (abil_df_full[item].unstack(level=0) * pf).unstack()
        else:
            df_ri = df.xs(facet_element)[item]
            pf = df_ri.notna().astype(float).replace(0, np.nan)
            mask_scores = df_ri
            mask_estimates = abil_df_full.xs(facet_element)[item] * pf

        masks = self._class_masks(mask_estimates, no_of_classes)
        if item is None and facet_element is None:
            raw_abil_full = pd.concat(
                {r: raw_abil_base for r in self.facet_names}, keys=self.facet_names
            )
            raw_for_x = (raw_abil_full * pf).unstack().unstack()
        elif item is None and facet_element is not None:
            raw_for_x = (raw_abil_base * pf).unstack()
        elif item is not None and facet_element is None:
            raw_frame = pd.DataFrame(
                {r: raw_abil_base[item] for r in self.facet_names}
            )
            raw_for_x = (raw_frame * pf).unstack()
        else:
            raw_for_x = raw_abil_base[item] * pf
        mean_abilities = np.array(
            [raw_for_x.loc[masks[cg]].mean() for cg in class_groups]
        )
        obs_props = np.array(
            [
                [
                    (mask_scores.loc[masks[cg]] == cat).sum() / len(masks[cg])
                    for cg in class_groups
                ]
                for cat in range(self.max_score + 1)
            ]
        )
        return mean_abilities, obs_props

    def class_intervals_thr(
        self,
        person_locations,
        item_locations,
        facet_effects,
        model="global",
        item=None,
        facet_element=None,
        shift=None,
        no_of_classes=5,
    ):
        """Class intervals for threshold CCC observed data overlay."""
        if item in ("none",):
            item = None
        if facet_element in ("none", "zero"):
            facet_element = None
        if shift is None:
            shift = 0

        class_groups = [f"class_{i + 1}" for i in range(no_of_classes)]
        df = self.responses.copy()

        abil_df = pd.DataFrame({it: person_locations for it in self.responses.columns})
        if item is None:
            for it in self.responses.columns:
                abil_df[it] -= float(item_locations[it])

        abil_dict = {}
        for r in self.facet_names:
            a = abil_df.copy()
            if facet_element is None:
                sev = self._facet_effect_item_offset(model, facet_effects, r)
                for it in self.responses.columns:
                    a[it] -= float(sev[it])
            abil_dict[r] = a
        abil_df_full = pd.concat(abil_dict.values(), keys=abil_dict.keys())
        abil_df_full.index.names = self.responses.index.names

        if item is not None:
            df = df[item]
            abil_df_full = abil_df_full[item]
        if facet_element is not None:
            df = df.xs(facet_element)
            abil_df_full = abil_df_full.xs(facet_element)

        mean_abilities_all, obs_props_all = [], []
        for t in range(self.max_score):
            cond_df = df[df.isin([t, t + 1])] - t
            cond_mask = cond_df.notna().astype(float).replace(0, np.nan)
            cond_estimates = abil_df_full * cond_mask

            if item is None:
                obs_data = pd.DataFrame(
                    {"person_location": cond_estimates.stack(), "score": cond_df.stack()}
                ).droplevel(level=1)
            else:
                obs_data = pd.DataFrame({"person_location": cond_estimates, "score": cond_df})

            masks = self._class_masks(obs_data["person_location"], no_of_classes)
            mean_abilities_all.append(
                [
                    obs_data.loc[masks[cg]]["person_location"].mean() + shift
                    for cg in class_groups
                ]
            )
            obs_props_all.append(
                [obs_data.loc[masks[cg]]["score"].mean() for cg in class_groups]
            )

        return np.array(mean_abilities_all), np.array(obs_props_all)

    # Backwards-compatible per-model aliases
    def class_intervals_cats_global(
        self, person_locations, item_locations, thresholds, facet_effects, **kw
    ):
        """Alias for class_intervals_cats(model='global'). See class_intervals_cats for full documentation."""
        return self.class_intervals_cats(
            person_locations, item_locations, thresholds, facet_effects, "global", **kw
        )

    def class_intervals_cats_items(
        self, person_locations, item_locations, thresholds, facet_effects, **kw
    ):
        """Alias for class_intervals_cats(model='items'). See class_intervals_cats for full documentation."""
        return self.class_intervals_cats(
            person_locations, item_locations, thresholds, facet_effects, "items", **kw
        )

    def class_intervals_cats_thresholds(
        self, person_locations, item_locations, thresholds, facet_effects, **kw
    ):
        """Alias for class_intervals_cats(model='thresholds'). See class_intervals_cats for full documentation."""
        return self.class_intervals_cats(
            person_locations, item_locations, thresholds, facet_effects, "thresholds", **kw
        )

    def class_intervals_cats_matrix(
        self, person_locations, item_locations, thresholds, facet_effects, **kw
    ):
        """Alias for class_intervals_cats(model='matrix'). See class_intervals_cats for full documentation."""
        return self.class_intervals_cats(
            person_locations, item_locations, thresholds, facet_effects, "matrix", **kw
        )

    def class_intervals_cats_bivector(
        self, person_locations, item_locations, thresholds, facet_effects, **kw
    ):
        """Alias for class_intervals_cats(model='bivector'). See class_intervals_cats for full documentation."""
        return self.class_intervals_cats(
            person_locations, item_locations, thresholds, facet_effects, "bivector", **kw
        )
    def class_intervals_cats_centrality(
        self, person_locations, item_locations, thresholds, facet_effects, **kw
    ):
        """Alias for class_intervals_cats(model='centrality'). See class_intervals_cats for full documentation."""
        return self.class_intervals_cats(
            person_locations, item_locations, thresholds, facet_effects, "centrality", **kw
        )
    def class_intervals_cats_pseudo_halo(
        self, person_locations, item_locations, thresholds, facet_effects, **kw
    ):
        """Alias for class_intervals_cats(model='pseudo_halo'). See class_intervals_cats for full documentation."""
        return self.class_intervals_cats(
            person_locations, item_locations, thresholds, facet_effects, "pseudo_halo", **kw
        )
    def class_intervals_cats_bistretch(
        self, person_locations, item_locations, thresholds, facet_effects, **kw
    ):
        """Alias for class_intervals_cats(model='bistretch'). See class_intervals_cats for full documentation."""
        return self.class_intervals_cats(
            person_locations, item_locations, thresholds, facet_effects, "bistretch", **kw
        )

    def class_intervals_thr_global(self, person_locations, item_locations, facet_effects, **kw):
        """Alias for class_intervals_thr(model='global'). See class_intervals_thr for full documentation."""
        return self.class_intervals_thr(
            person_locations, item_locations, facet_effects, "global", **kw
        )

    def class_intervals_thr_items(self, person_locations, item_locations, facet_effects, **kw):
        """Alias for class_intervals_thr(model='items'). See class_intervals_thr for full documentation."""
        return self.class_intervals_thr(
            person_locations, item_locations, facet_effects, "items", **kw
        )

    def class_intervals_thr_thresholds(self, person_locations, item_locations, facet_effects, **kw):
        """Alias for class_intervals_thr(model='thresholds'). See class_intervals_thr for full documentation."""
        return self.class_intervals_thr(
            person_locations, item_locations, facet_effects, "thresholds", **kw
        )

    def class_intervals_thr_matrix(self, person_locations, item_locations, facet_effects, **kw):
        """Alias for class_intervals_thr(model='matrix'). See class_intervals_thr for full documentation."""
        return self.class_intervals_thr(
            person_locations, item_locations, facet_effects, "matrix", **kw
        )

    def class_intervals_thr_bivector(self, person_locations, item_locations, facet_effects, **kw):
        """Alias for class_intervals_thr(model='bivector'). See class_intervals_thr for full documentation."""
        return self.class_intervals_thr(
            person_locations, item_locations, facet_effects, "bivector", **kw
        )

    def class_intervals_thr_centrality(self, person_locations, item_locations, facet_effects, **kw):
        """Alias for class_intervals_thr(model='centrality'). See class_intervals_thr for full documentation."""
        return self.class_intervals_thr(
            person_locations, item_locations, facet_effects, "centrality", **kw
        )

    def class_intervals_thr_pseudo_halo(self, person_locations, item_locations, facet_effects, **kw):
        """Alias for class_intervals_thr(model='pseudo_halo'). See class_intervals_thr for full documentation."""
        return self.class_intervals_thr(
            person_locations, item_locations, facet_effects, "pseudo_halo", **kw
        )

    def class_intervals_thr_bistretch(self, person_locations, item_locations, facet_effects, **kw):
        """Alias for class_intervals_thr(model='bistretch'). See class_intervals_thr for full documentation."""
        return self.class_intervals_thr(
            person_locations, item_locations, facet_effects, "bistretch", **kw
        )

    def _label_height_frac(self, font, axis_font_size, figsize):
        """
        Fraction of the axes' own pixel height that one line of label
        text occupies, at the given font/size/figure size -- measured
        directly via a probe render (not guessed), so callers can
        convert it into an exact data-unit label height for whatever
        y-range they end up using (that height scales linearly with the
        y-range, since the axes' pixel height itself doesn't depend on
        the data plotted). Used to size ymax and place stacked
        central_location labels with a small, guaranteed gap rather
        than a fixed fraction of y_max (see feedback_plot_layout_rigor
        -- measure, don't guess).
        """
        probe_fig = plt.figure(figsize=figsize)
        probe_ax = probe_fig.add_subplot(111)
        probe_txt = probe_ax.text(
            0, 0, "Item_1: 0.000", fontsize=axis_font_size, fontfamily=font
        )
        probe_fig.canvas.draw()
        renderer = probe_fig.canvas.get_renderer()
        label_h_px = probe_txt.get_window_extent(renderer=renderer).height
        ax_h_px = probe_ax.get_window_extent(renderer=renderer).height
        plt.close(probe_fig)
        return label_h_px / ax_h_px

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    def plot_data(
        self,
        x_data,
        y_data,
        model="global",
        anchor=False,
        items=None,
        curve_labels=None,
        legend_loc=None,
        facet_elements=None,
        obs=None,
        obs_by_column=False,
        obs_curve_index=None,
        marker_border=True,
        line_border=False,
        thresh_obs=None,
        x_obs_data=np.array([]),
        y_obs_data=np.array([]),
        thresh_lines=False,
        central_location=False,
        central_location_fit=False,
        score_lines_item=[None, None],
        score_lines_test=None,
        point_info_lines_item=[None, None],
        point_info_lines_test=None,
        point_csem_lines=None,
        score_labels=False,
        x_min=-5,
        x_max=5,
        y_max=0,
        warm=True,
        cat_highlight=None,
        graph_title="",
        y_label="",
        plot_style="white",
        palette="colorblind multi",
        black=False,
        figsize=(8, 6),
        font="Times New Roman",
        title_font_size=15,
        axis_font_size=12,
        labelsize=12,
        tex=True,
        plot_density=300,
        filename=None,
        file_format="png",
    ):
        """
        Core plotting function for person_location-function curves (MFRM).
        Shared across all eight facet_element parameterisations.
        """
        item_locations, thresholds, facet_effects = self._get_params(model, anchor)

        if isinstance(facet_elements, str):
            facet_elements = (
                None if facet_elements in ("none", "zero", "all") else [facet_elements]
            )
        if isinstance(items, str):
            items = None if items == "all" else items

        self._apply_plot_style(plot_style)

        palette_dict = {
            "dark blue": ["dark", "royalblue"],
            "light blue": ["light", "cornflowerblue"],
            "dark red": ["dark", "firebrick"],
            "light red": ["light", "indianred"],
            "dark green": ["dark", "forestgreen"],
            "light green": ["light", "mediumseagreen"],
            "dark grey": ["dark", "dimgrey"],
            "light grey": ["light", "darkgrey"],
            "dark multi": ["dark", "dark"],
            "light multi": ["light", "muted"],
            "colorblind multi": ["dark", "colorblind"],
        }
        shade, base_color = palette_dict[palette]
        if palette == "colorblind multi":
            # Okabe & Ito (2008) -- colour-vision-deficiency-safe
            # qualitative palette, the scientific-publishing standard.
            # The 8th (black) entry is swapped for white on dark
            # backgrounds, where it would otherwise be invisible.
            color_map = [
                "#E69F00", "#56B4E9", "#009E73", "#F0E442",
                "#0072B2", "#D55E00", "#CC79A7",
                "#FFFFFF" if plot_style in self._DARK_BACKGROUND_STYLES else "#000000",
            ]
        elif shade == "dark":
            color_map = (
                sns.color_palette("dark", as_cmap=True)
                if palette == "dark multi"
                else sns.dark_palette(base_color, reverse=True, as_cmap=True)
            )
        else:
            color_map = (
                sns.color_palette("muted", as_cmap=True)
                if palette == "light multi"
                else sns.light_palette(base_color, reverse=True, as_cmap=True)
            )

        with plt.rc_context({"font.family": font, "font.size": axis_font_size}):
            graph, ax = plt.subplots(figsize=figsize)
            no_of_plots = y_data.shape[1]
            cNorm = colors.Normalize(vmin=0, vmax=no_of_plots + 2)
            if "multi" not in palette:
                scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=color_map)

            # A thin contrasting stroke around each curve line, so it stays
            # legible on a background/gridline colour close to its own --
            # white on the darker plot_style schemes, black otherwise.
            line_fx = (
                [path_effects.withStroke(
                    linewidth=2.5,
                    foreground="white" if plot_style in self._DARK_BACKGROUND_STYLES else "black",
                )]
                if line_border
                else None
            )

            for i in range(no_of_plots):
                col = (
                    "black"
                    if black
                    else (
                        scalarMap.to_rgba(i) if "multi" not in palette else color_map[i]
                    )
                )
                label = curve_labels[i] if curve_labels is not None else i + 1
                ax.plot(
                    x_data, y_data[:, i], "", color=col, label=label,
                    path_effects=line_fx,
                )

            if curve_labels is not None:
                ax.legend(loc=legend_loc if legend_loc is not None else "best")

            if obs is not None:
                try:
                    if isinstance(y_obs_data, pd.Series):
                        col = (
                            scalarMap.to_rgba(0)
                            if "multi" not in palette
                            else color_map[0]
                        )
                        ax.plot(
                            x_obs_data, y_obs_data, "o", color=col,
                            markeredgecolor="k" if marker_border else "none", zorder=3,
                        )
                    elif obs_by_column:
                        # icc()'s own multi-curve convention: one column
                        # per curve (item and/or facet_element list),
                        # matching y_data's own column layout -- colour
                        # each to match its curve.
                        for j in range(y_obs_data.shape[1]):
                            col = (
                                scalarMap.to_rgba(j)
                                if "multi" not in palette
                                else color_map[j]
                            )
                            ax.plot(
                                x_obs_data, y_obs_data[:, j], "o", color=col,
                                markeredgecolor="k" if marker_border else "none", zorder=3,
                            )
                    else:
                        for j in range(y_obs_data.shape[0]):
                            k = obs_curve_index[j] if obs_curve_index is not None else j
                            col = (
                                scalarMap.to_rgba(k)
                                if "multi" not in palette
                                else color_map[k]
                            )
                            ax.plot(
                                x_obs_data, y_obs_data[j, :], "o", color=col,
                                markeredgecolor="k" if marker_border else "none", zorder=3,
                            )
                except Exception:
                    pass

            if thresh_obs is not None:
                try:
                    for j in range(x_obs_data.shape[0]):
                        k = obs_curve_index[j] if obs_curve_index is not None else j
                        col = (
                            scalarMap.to_rgba(k)
                            if "multi" not in palette
                            else color_map[k]
                        )
                        ax.plot(
                            x_obs_data[j, :], y_obs_data[j, :], "o", color=col,
                            markeredgecolor="k" if marker_border else "none", zorder=3,
                        )
                except Exception:
                    pass

            if thresh_lines:
                r_sc = (
                    facet_elements[0]
                    if isinstance(facet_elements, list)
                    else (
                        facet_elements
                        if facet_elements is not None
                        else list(self.facet_names)[0]
                    )
                )
                for t in range(self.max_score):
                    diff_val = 0.0 if items is None else float(item_locations[items])
                    # Per-threshold facet_effect for models that vary by threshold
                    if facet_elements is None:
                        sev_val = 0.0
                    elif model in ("thresholds", "centrality"):
                        sev_val = float(facet_effects.loc[r_sc, t + 1])
                    elif model in ("bivector", "matrix", "bistretch"):
                        item_key = (
                            items if items is not None else list(self.item_names)[0]
                        )
                        sev_val = float(facet_effects.loc[(r_sc, item_key), t + 1])
                    elif model in ("items", "pseudo_halo") and items is not None:
                        sev_val = float(
                            self._facet_effect_item_offset(model, facet_effects, r_sc)[items]
                        )
                    else:
                        sev_val = float(
                            self._facet_effect_item_offset(model, facet_effects, r_sc).mean()
                        )
                    xval = diff_val + thresholds[t + 1] + sev_val
                    ax.axvline(x=xval, color="black", linestyle="--")
                    # Staggered onto two rows near the bottom so that any
                    # two adjacent thresholds (the only ones close enough
                    # in x to clash) never share a row.
                    label_y = y_max * (0.05 if t % 2 == 0 else 0.11)
                    ax.text(
                        xval + (x_max - x_min) / 100,
                        label_y,
                        str(t + 1),
                        color="black",
                        va="bottom",
                        ha="left",
                    )

            if central_location:
                # items may be a list (one curve per item) -- draw each
                # item's own central location, always plain darkred
                # (colour isn't needed to disambiguate since each line
                # is labelled with its own item name).
                items_multi = isinstance(items, list)
                items_list = items if items_multi else [items]
                n_labels = len(items_list)
                if central_location_fit and items_multi and n_labels > 0:
                    # "Fit" style (iic): the curve's own peak sits right
                    # at each item's central location, so labels are
                    # stacked with a probe-measured height above that
                    # peak -- ymax is pre-widened by the caller to fit
                    # them exactly (measure, don't guess -- see
                    # feedback_plot_layout_rigor).
                    k = self._label_height_frac(font, axis_font_size, figsize)
                    label_h_data = k * y_max
                    gap = 0.3 * label_h_data
                    curve_peak = float(np.nanmax(y_data))
                for idx, it in enumerate(items_list):
                    xval = 0 if it is None else float(item_locations[it])
                    ax.axvline(x=xval, color="darkred", linestyle="--")
                    label = (
                        f"{it}: {round(xval, 2)}"
                        if items_multi and it is not None
                        else str(round(xval, 2))
                    )
                    if central_location_fit and items_multi:
                        label_y = curve_peak + gap + label_h_data * (n_labels - idx - 0.5)
                        label_x = xval - (x_max - x_min) / 100
                        ha = "right"
                    elif items_multi:
                        # "Top" style (icc): the curve is an ogive, near
                        # its ceiling by the time it nears max score, so
                        # labels just stack near the top of the fixed
                        # y_max instead -- placed to the *left* of each
                        # line (curves rise left-to-right, so that side
                        # stays clear of the curve even close to the
                        # ceiling) rather than boosting y_max past its
                        # own meaningful value.
                        label_y = y_max * 0.95 - idx * y_max * 0.05
                        label_x = xval - (x_max - x_min) / 100
                        ha = "right"
                    else:
                        label_y = y_max * 0.95
                        label_x = xval + (x_max - x_min) / 100
                        ha = "left"
                        label = f"Central location: {round(xval, 2)}"
                    ax.text(
                        label_x,
                        label_y,
                        label,
                        color="black",
                        va="center",
                        ha=ha,
                    )

            if score_lines_item[1] is not None:
                # ICC score line(s): invert each curve numerically by
                # finding the x value where that curve's own y_data column
                # is closest to s. y_data has one column per curve
                # (item and/or facet_element list), so this naturally
                # covers every combination without needing to know which
                # dimension(s) were lists -- each curve's own line(s) are
                # drawn in that curve's own colour (single-curve calls
                # keep the original plain black lines).
                n_curves = y_data.shape[1]
                multi_curves = n_curves > 1
                if all(s > 0 for s in score_lines_item[1]) and all(
                    s < self.max_score for s in score_lines_item[1]
                ):
                    for i in range(n_curves):
                        colorVal = (
                            "black"
                            if black or not multi_curves
                            else (
                                scalarMap.to_rgba(i)
                                if "multi" not in palette
                                else color_map[i]
                            )
                        )
                        for s in score_lines_item[1]:
                            idx = np.argmin(np.abs(y_data[:, i] - s))
                            estimate = x_data[idx]
                            ax.vlines(
                                x=estimate,
                                ymin=0,
                                ymax=s,
                                color=colorVal,
                                linestyles="dashed",
                            )
                            ax.hlines(
                                y=s,
                                xmin=x_min,
                                xmax=estimate,
                                color=colorVal,
                                linestyles="dashed",
                            )
                            if score_labels:
                                # Stagger each curve's estimate label a
                                # bit higher up than the last, so nearby
                                # curves' labels don't overwrite each
                                # other -- single-curve calls keep the
                                # original fixed height.
                                label_y = y_max / 50 + (
                                    i * y_max * 0.05 if multi_curves else 0
                                )
                                ax.text(
                                    estimate + (x_max - x_min) / 100,
                                    label_y,
                                    str(round(estimate, 2)),
                                    color=colorVal,
                                )
                                ax.text(
                                    x_min + (x_max - x_min) / 100, s + y_max / 50, str(s),
                                    color=colorVal,
                                )
                else:
                    warnings.warn(
                        "Invalid score for score line: values must be "
                        "strictly between 0 and the item maximum score.",
                        UserWarning,
                        stacklevel=2,
                    )

            if score_lines_test is not None:
                item_keys = (
                    list(self.item_names)
                    if items is None
                    else ([items] if isinstance(items, str) else items)
                )
                n_items = len(item_keys)
                n_raters = (
                    len(facet_elements)
                    if facet_elements is not None
                    else self.no_of_facet_elements
                )
                max_total = self.max_score * n_items * n_raters
                if all(0 < s < max_total for s in score_lines_test):
                    for s in score_lines_test:
                        estimate = self.score_lookup(
                            s,
                            model=model,
                            anchor=anchor,
                            items=item_keys,
                            facet_elements=facet_elements,
                            warm_corr=warm,
                        )
                        ax.vlines(
                            x=estimate,
                            ymin=0,
                            ymax=s,
                            color="black",
                            linestyles="dashed",
                        )
                        ax.hlines(
                            y=s,
                            xmin=x_min,
                            xmax=estimate,
                            color="black",
                            linestyles="dashed",
                        )
                        if score_labels:
                            ax.text(
                                estimate + (x_max - x_min) / 100,
                                y_max / 50,
                                str(round(estimate, 2)),
                            )
                            ax.text(
                                x_min + (x_max - x_min) / 100, s + y_max / 50, str(s)
                            )
                else:
                    warnings.warn(
                        "Invalid score for score line: values must be "
                        "strictly between 0 and the test maximum score.",
                        UserWarning,
                        stacklevel=2,
                    )

            if point_info_lines_item[1] is not None:
                # items and/or facet_elements may be a list (one curve
                # per combination) -- for a given location, every
                # curve's own information value is genuinely different
                # (that's the whole point of comparing them), so each
                # gets its own "curve: info" label near the y-axis,
                # nudged sideways per curve so close values don't
                # collide. Single-curve calls keep the original separate
                # location/info label pair. Reuses curve_labels (already
                # built by icc()/iic() for the legend) for the label
                # text when available, so it always matches.
                items_arg = point_info_lines_item[0]
                items_multi = isinstance(items_arg, list)
                raters_multi = isinstance(facet_elements, list)
                items_list = items_arg if items_multi else [items_arg]
                raters_list = facet_elements if raters_multi else [facet_elements]
                multi_curves = items_multi or raters_multi

                curve_i = 0
                for it in items_list:
                    for fe in raters_list:
                        colorVal = (
                            "black"
                            if black or not multi_curves
                            else (
                                scalarMap.to_rgba(curve_i)
                                if "multi" not in palette
                                else color_map[curve_i]
                            )
                        )
                        r = (
                            fe if fe is not None and fe not in ("none", "zero")
                            else list(self.facet_names)[0]
                        )
                        for estimate in point_info_lines_item[1]:
                            info = self.variance(
                                estimate, it, item_locations, r, facet_effects,
                                thresholds, model,
                            )
                            ax.vlines(
                                x=estimate, ymin=-100, ymax=info, color=colorVal,
                                linestyles="dashed",
                            )
                            ax.hlines(
                                y=info, xmin=-100, xmax=estimate, color=colorVal,
                                linestyles="dashed",
                            )
                            if score_labels:
                                if multi_curves:
                                    name = (
                                        curve_labels[curve_i]
                                        if curve_labels is not None
                                        else f"{it} / {r}"
                                    )
                                    label_x = (
                                        x_min
                                        + (x_max - x_min) / 100
                                        + curve_i * (x_max - x_min) * 0.03
                                    )
                                    ax.text(
                                        label_x, info + y_max / 50,
                                        f"{name}: {round(info, 3)}", color=colorVal,
                                    )
                                else:
                                    ax.text(
                                        estimate + (x_max - x_min) / 100,
                                        y_max / 50,
                                        str(round(estimate, 2)),
                                    )
                                    ax.text(
                                        x_min + (x_max - x_min) / 100,
                                        info + y_max / 50,
                                        str(round(info, 3)),
                                    )
                        curve_i += 1

            if point_info_lines_test is not None:
                item_keys = list(self.item_names) if items is None else items
                rater_list = (
                    list(self.facet_names) if facet_elements is None else facet_elements
                )
                for estimate in point_info_lines_test:
                    info = sum(
                        self.variance(
                            estimate, it, item_locations, r, facet_effects, thresholds, model
                        )
                        for it in item_keys
                        for r in rater_list
                    )
                    ax.vlines(
                        x=estimate,
                        ymin=-100,
                        ymax=info,
                        color="black",
                        linestyles="dashed",
                    )
                    ax.hlines(
                        y=info,
                        xmin=-100,
                        xmax=estimate,
                        color="black",
                        linestyles="dashed",
                    )
                    if score_labels:
                        ax.text(
                            estimate + (x_max - x_min) / 100,
                            y_max / 50,
                            str(round(estimate, 2)),
                        )
                        ax.text(
                            x_min + (x_max - x_min) / 100,
                            info + y_max / 50,
                            str(round(info, 3)),
                        )

            if point_csem_lines is not None:
                item_keys = list(self.item_names) if items is None else items
                rater_list = (
                    list(self.facet_names) if facet_elements is None else facet_elements
                )
                for estimate in point_csem_lines:
                    info = sum(
                        self.variance(
                            estimate, it, item_locations, r, facet_effects, thresholds, model
                        )
                        for it in item_keys
                        for r in rater_list
                    )
                    csem = 1.0 / (info**0.5)
                    ax.vlines(
                        x=estimate,
                        ymin=-100,
                        ymax=csem,
                        color="black",
                        linestyles="dashed",
                    )
                    ax.hlines(
                        y=csem,
                        xmin=-100,
                        xmax=estimate,
                        color="black",
                        linestyles="dashed",
                    )
                    if score_labels:
                        ax.text(
                            estimate + (x_max - x_min) / 100,
                            y_max / 50,
                            str(round(estimate, 2)),
                        )
                        ax.text(
                            x_min + (x_max - x_min) / 100,
                            csem + y_max / 50,
                            str(round(csem, 3)),
                        )

            if cat_highlight in range(self.max_score + 1):
                sev_shift = 0.0
                if facet_elements is not None:
                    # _facet_effect_item_offset expects a scalar facet_element
                    r_scalar = (
                        facet_elements[0]
                        if isinstance(facet_elements, list)
                        else facet_elements
                    )
                    sev_offset = self._facet_effect_item_offset(model, facet_effects, r_scalar)
                    if model in ("items", "pseudo_halo") and items is not None:
                        sev_shift = float(sev_offset[items])
                    else:
                        sev_shift = float(sev_offset.mean())
                diff_shift = 0.0 if items is None else float(item_locations[items])

                if cat_highlight == 0:
                    ax.axvspan(
                        -100,
                        diff_shift + thresholds[1] + sev_shift,
                        facecolor="blue",
                        alpha=0.2,
                    )
                elif cat_highlight == self.max_score:
                    ax.axvspan(
                        diff_shift + thresholds[self.max_score] + sev_shift,
                        100,
                        facecolor="blue",
                        alpha=0.2,
                    )
                else:
                    lo = diff_shift + thresholds[cat_highlight] + sev_shift
                    hi = diff_shift + thresholds[cat_highlight + 1] + sev_shift
                    if hi > lo:
                        ax.axvspan(lo, hi, facecolor="blue", alpha=0.2)

            if y_max <= 0:
                y_max = float(y_data.max()) * 1.1
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0, y_max)
            ax.set_xlabel("Person estimate", fontsize=axis_font_size, fontweight="bold")
            ax.set_ylabel(y_label, fontsize=axis_font_size, fontweight="bold")
            ax.set_title(graph_title, fontsize=title_font_size, fontweight="bold")
            ax.grid(True)
            ax.tick_params(axis="x", labelsize=labelsize)
            ax.tick_params(axis="y", labelsize=labelsize)

            if filename is not None:
                graph.savefig(f"{filename}.{file_format}", dpi=plot_density)
            plt.close(graph)

        return graph

    # Backwards-compatible plot_data aliases
    def plot_data_global(self, *args, **kw):
        """Alias for plot_data(model='global'). See plot_data for full documentation."""
        return self.plot_data(*args, model="global", **kw)

    def plot_data_items(self, *args, **kw):
        """Alias for plot_data(model='items'). See plot_data for full documentation."""
        return self.plot_data(*args, model="items", **kw)

    def plot_data_thresholds(self, *args, **kw):
        """Alias for plot_data(model='thresholds'). See plot_data for full documentation."""
        return self.plot_data(*args, model="thresholds", **kw)

    def plot_data_matrix(self, *args, **kw):
        """Alias for plot_data(model='matrix'). See plot_data for full documentation."""
        return self.plot_data(*args, model="matrix", **kw)

    def plot_data_bivector(self, *args, **kw):
        """Alias for plot_data(model='bivector'). See plot_data for full documentation."""
        return self.plot_data(*args, model="bivector", **kw)

    def plot_data_centrality(self, *args, **kw):
        """Alias for plot_data(model='centrality'). See plot_data for full documentation."""
        return self.plot_data(*args, model="centrality", **kw)

    def plot_data_pseudo_halo(self, *args, **kw):
        """Alias for plot_data(model='pseudo_halo'). See plot_data for full documentation."""
        return self.plot_data(*args, model="pseudo_halo", **kw)

    def plot_data_bistretch(self, *args, **kw):
        """Alias for plot_data(model='bistretch'). See plot_data for full documentation."""
        return self.plot_data(*args, model="bistretch", **kw)

    # ------------------------------------------------------------------
    # ICC, CRCS, Threshold CCS, IIC, TCC, Test info, Test CSEM, Residuals
    # ------------------------------------------------------------------
    def icc(
        self,
        item,
        model="global",
        anchor=False,
        facet_element=None,
        obs=None,
        warm=True,
        xmin=-5,
        xmax=5,
        no_of_classes=5,
        title=None,
        thresh_lines=False,
        score_lines=None,
        score_labels=False,
        central_location=False,
        cat_highlight=None,
        plot_style="white",
        palette="colorblind multi",
        black=False,
        marker_border=True,
        line_border=False,
        font="Times New Roman",
        title_font_size=15,
        axis_font_size=12,
        labelsize=12,
        filename=None,
        file_format="png",
        dpi=300,
    ):
        """
        Item Characteristic Curve for one item and/or facet element (e.g.
        facet element (e.g. rater)), or several of either (or both) overlaid on the same axes.

        item : str or list of str
            A single name draws one curve, as before. A list overlays one
            curve per item (in the given order), each in its own colour.
            thresh_lines/cat_highlight only make sense for one curve's
            own location, so they just silently no-op with several
            plotted at once rather than erroring. obs, central_location,
            and score_lines all work fine with item and/or facet_element
            as a list -- obs and score_lines each draw every curve's own
            point(s)/line(s) in that curve's own colour; central_location
            draws each item's own line, labelled with the value (and
            item name, when item is a list -- staggered downward from
            the top, one step per item, so nearby curves' labels don't
            overwrite each other), colour-matched too when facet_element
            isn't also a list (so each item maps to exactly one curve)
            and plain darkred otherwise (an item spans several curves
            there, with no single one to match).
        facet_element : str, list of str, or None
            A single name (or None, averaging across every facet
            element) draws one curve, as before. A list overlays one
            curve per named element instead (None isn't valid inside the
            list -- pass an explicit "average" entry isn't supported;
            use the scalar None case for that curve on its own).

            item and facet_element can both be lists at once, giving one
            curve per (item, facet element) combination -- legended
            "item / facet_element" -- but this is really a lot of curves
            on one plot at once and can get hard to read fast; one list
            and one fixed value (one item across several facet elements, or one
            facet element across several items) reads far more clearly. See the
            manual for an example of each.
        """
        model = self._MODEL_ALIASES.get(model, model)
        item_locations, thresholds, facet_effects = self._get_params(model, anchor)

        items_multi = isinstance(item, list)
        raters_multi = isinstance(facet_element, list)
        if raters_multi:
            facet_element = [
                (None if fe in ("none", "zero") else fe) for fe in facet_element
            ]
        elif facet_element in ("none", "zero"):
            facet_element = None

        if items_multi or raters_multi:
            # thresh_lines/cat_highlight each only make sense for a
            # single item's own location -- silently no-op with several
            # curves plotted at once rather than erroring, since they'd
            # just be visual clutter with no obvious single "right" curve
            # anyway.
            thresh_lines = False
            cat_highlight = None

        person_locations_arr = np.arange(-20, 20, 0.1)
        items_to_plot = item if items_multi else [item]
        raters_to_plot = facet_element if raters_multi else [facet_element]
        all_raters = list(self.facet_names)

        def _curve(it, fe):
            # None averages exp_score across every facet element, to
            # match the obs y-values (mean scores across the pool).
            if fe is None:
                return [
                    np.mean(
                        [
                            self.exp_score(
                                a, it, item_locations, r, facet_effects, thresholds, model
                            )
                            for r in all_raters
                        ]
                    )
                    for a in person_locations_arr
                ]
            return [
                self.exp_score(a, it, item_locations, fe, facet_effects, thresholds, model)
                for a in person_locations_arr
            ]

        y = np.column_stack(
            [_curve(it, fe) for it in items_to_plot for fe in raters_to_plot]
        )

        if items_multi and raters_multi:
            curve_labels = [
                f"{it} / {'Average' if fe is None else fe}"
                for it in items_to_plot
                for fe in raters_to_plot
            ]
        elif items_multi:
            curve_labels = items_to_plot
        elif raters_multi:
            curve_labels = ["Average" if fe is None else fe for fe in raters_to_plot]
        else:
            curve_labels = None

        obs_by_column = False
        if obs is not None:
            persons_attr = f'{"anchor_" if anchor else ""}persons_{model}'
            if not hasattr(self, persons_attr):
                self.person_estimates(model=model, anchor=anchor)
            person_estimates = getattr(self, persons_attr)
            if items_multi or raters_multi:
                # Same person-location class intervals for every curve
                # where possible (quantile groups come from the overall
                # ability estimate, not an item/rater-specific one, so
                # they mostly line up); the first curve's own x values
                # are reused for the rest. One observed-mean column per
                # curve, in that curve's own colour (obs_by_column=True
                # tells plot_data to read y_obs_data this way).
                obs_by_column = True
                xobsdata = None
                yobs_cols = []
                for it in items_to_plot:
                    for fe in raters_to_plot:
                        xd, yd = self.class_intervals(
                            person_estimates, items=it, facet_elements=fe,
                            no_of_classes=no_of_classes,
                        )
                        if xobsdata is None:
                            xobsdata = xd
                        yobs_cols.append(np.array(yd))
                yobsdata = np.column_stack(yobs_cols)
            else:
                xobsdata, yobsdata = self.class_intervals(
                    person_estimates,
                    items=item,
                    facet_elements=facet_element,
                    no_of_classes=no_of_classes,
                )
                # Keep yobsdata as a pd.Series so plot_data uses the scalar
                # obs branch (ax.plot(x, y, 'o')) rather than the row-iteration
                # branch, which mismatches shapes for the single-curve ICC case.
        else:
            xobsdata = yobsdata = np.array(np.nan)

        return self.plot_data(
            x_data=person_locations_arr,
            y_data=y,
            model=model,
            anchor=anchor,
            items=item,
            curve_labels=curve_labels,
            legend_loc="upper left" if score_lines is not None else "lower right",
            facet_elements=facet_element,
            obs=obs,
            obs_by_column=obs_by_column,
            marker_border=marker_border,
            line_border=line_border,
            warm=warm,
            x_obs_data=xobsdata,
            y_obs_data=yobsdata,
            x_min=xmin,
            x_max=xmax,
            y_max=self.max_score,
            thresh_lines=thresh_lines,
            graph_title=title or "",
            score_lines_item=[item, score_lines],
            score_labels=score_labels,
            central_location=central_location,
            cat_highlight=cat_highlight,
            y_label="Expected score",
            plot_style=plot_style,
            palette=palette,
            black=black,
            font=font,
            title_font_size=title_font_size,
            axis_font_size=axis_font_size,
            labelsize=labelsize,
            filename=filename,
            plot_density=dpi,
            file_format=file_format,
        )

    def icc_global(self, item, **kw):
        """Alias for icc(model='global'). See icc for full documentation."""
        return self.icc(item, model="global", **kw)

    def icc_items(self, item, **kw):
        """Alias for icc(model='items'). See icc for full documentation."""
        return self.icc(item, model="items", **kw)

    def icc_thresholds(self, item, **kw):
        """Alias for icc(model='thresholds'). See icc for full documentation."""
        return self.icc(item, model="thresholds", **kw)

    def icc_matrix(self, item, **kw):
        """Alias for icc(model='matrix'). See icc for full documentation."""
        return self.icc(item, model="matrix", **kw)

    def icc_bivector(self, item, **kw):
        """Alias for icc(model='bivector'). See icc for full documentation."""
        return self.icc(item, model="bivector", **kw)

    def icc_centrality(self, item, **kw):
        """Alias for icc(model='centrality'). See icc for full documentation."""
        return self.icc(item, model="centrality", **kw)

    def icc_pseudo_halo(self, item, **kw):
        """Alias for icc(model='pseudo_halo'). See icc for full documentation."""
        return self.icc(item, model="pseudo_halo", **kw)

    def icc_bistretch(self, item, **kw):
        """Alias for icc(model='bistretch'). See icc for full documentation."""
        return self.icc(item, model="bistretch", **kw)

    def crcs(
        self,
        model="global",
        anchor=False,
        item=None,
        facet_element=None,
        obs=None,
        no_of_classes=5,
        title=None,
        thresh_lines=False,
        central_location=False,
        cat_highlight=None,
        xmin=-5,
        xmax=5,
        plot_style="white",
        palette="colorblind multi",
        black=False,
        marker_border=True,
        line_border=False,
        font="Times New Roman",
        title_font_size=15,
        axis_font_size=12,
        labelsize=12,
        filename=None,
        file_format="png",
        dpi=300,
    ):
        """Category Response Curves."""
        model = self._MODEL_ALIASES.get(model, model)
        item_locations, thresholds, facet_effects = self._get_params(model, anchor)
        if item in ("none",):
            item = None
        if facet_element in ("none", "zero"):
            facet_element = None

        person_locations_arr = np.arange(-20, 20, 0.1)
        r_use = (
            facet_element[0]
            if isinstance(facet_element, list)
            else (
                facet_element
                if facet_element is not None
                else list(self.facet_names)[0]
            )
        )
        # Use mean facet_effects for the neutral curve so it represents a
        # typical facet_element, matching obs proportions averaged across all facet_elements.
        sev_for_curve = (
            facet_effects
            if facet_element is not None
            else self._mean_facet_effects(model, facet_effects, r_use)
        )

        if obs is not None:
            persons_attr = f'{"anchor_" if anchor else ""}persons_{model}'
            if not hasattr(self, persons_attr):
                self.person_estimates(model=model, anchor=anchor)
            person_locations = getattr(self, persons_attr)
            xobsdata, yobsdata = self.class_intervals_cats(
                person_locations,
                item_locations,
                thresholds,
                facet_effects,
                model=model,
                item=item,
                facet_element=facet_element,
                no_of_classes=no_of_classes,
            )
            if isinstance(obs, str) and obs == "all":
                obs = np.arange(self.max_score + 1)
            if not all(c in np.arange(self.max_score + 1) for c in obs):
                warnings.warn(
                    "Invalid 'obs' value. Valid values are None, 'all', "
                    "or a list of category indices.",
                    UserWarning,
                    stacklevel=2,
                )
                return
            yobsdata = yobsdata[obs, :]
        else:
            xobsdata = yobsdata = np.array(np.nan)
        y = np.array(
            [
                [
                    self.cat_prob(
                        a,
                        (item or list(self.item_names)[0]),
                        item_locations,
                        r_use,
                        sev_for_curve,
                        cat,
                        thresholds,
                        model,
                    )
                    for cat in range(self.max_score + 1)
                ]
                for a in person_locations_arr
            ]
        )

        return self.plot_data(
            x_data=person_locations_arr,
            y_data=y,
            model=model,
            anchor=anchor,
            items=item,
            facet_elements=facet_element,
            curve_labels=[f"Category {c}" for c in range(self.max_score + 1)],
            obs=obs,
            obs_curve_index=obs if obs is not None else None,
            marker_border=marker_border,
            line_border=line_border,
            x_obs_data=xobsdata,
            y_obs_data=yobsdata,
            x_min=xmin,
            x_max=xmax,
            y_max=1,
            thresh_lines=thresh_lines,
            central_location=central_location,
            cat_highlight=cat_highlight,
            graph_title=title or "",
            y_label="Probability",
            plot_style=plot_style,
            palette=palette,
            black=black,
            font=font,
            title_font_size=title_font_size,
            axis_font_size=axis_font_size,
            labelsize=labelsize,
            filename=filename,
            plot_density=dpi,
            file_format=file_format,
        )

    def crcs_global(self, item=None, **kw):
        """Alias for crcs(model='global'). See crcs for full documentation."""
        return self.crcs(model="global", item=item, **kw)

    def crcs_items(self, item=None, **kw):
        """Alias for crcs(model='items'). See crcs for full documentation."""
        return self.crcs(model="items", item=item, **kw)

    def crcs_thresholds(self, item=None, **kw):
        """Alias for crcs(model='thresholds'). See crcs for full documentation."""
        return self.crcs(model="thresholds", item=item, **kw)

    def crcs_matrix(self, item=None, **kw):
        """Alias for crcs(model='matrix'). See crcs for full documentation."""
        return self.crcs(model="matrix", item=item, **kw)

    def crcs_bivector(self, item=None, **kw):
        """Alias for crcs(model='bivector'). See crcs for full documentation."""
        return self.crcs(model="bivector", item=item, **kw)

    def crcs_centrality(self, item=None, **kw):
        """Alias for crcs(model='centrality'). See crcs for full documentation."""
        return self.crcs(model="centrality", item=item, **kw)

    def crcs_pseudo_halo(self, item=None, **kw):
        """Alias for crcs(model='pseudo_halo'). See crcs for full documentation."""
        return self.crcs(model="pseudo_halo", item=item, **kw)

    def crcs_bistretch(self, item=None, **kw):
        """Alias for crcs(model='bistretch'). See crcs for full documentation."""
        return self.crcs(model="bistretch", item=item, **kw)

    def threshold_ccs(
        self,
        model="global",
        anchor=False,
        item=None,
        facet_element=None,
        obs=None,
        no_of_classes=5,
        title=None,
        thresh_lines=False,
        central_location=False,
        cat_highlight=None,
        xmin=-5,
        xmax=5,
        plot_style="white",
        palette="colorblind multi",
        black=False,
        marker_border=True,
        line_border=False,
        font="Times New Roman",
        title_font_size=15,
        axis_font_size=12,
        labelsize=12,
        filename=None,
        file_format="png",
        dpi=300,
    ):
        """Threshold Characteristic Curves."""
        model = self._MODEL_ALIASES.get(model, model)
        item_locations, thresholds, facet_effects = self._get_params(model, anchor)
        if item in ("none",):
            item = None
        if facet_element in ("none", "zero"):
            facet_element = None

        person_locations_arr = np.arange(-20, 20, 0.1)
        r_use = (
            facet_element[0]
            if isinstance(facet_element, list)
            else (
                facet_element
                if facet_element is not None
                else list(self.facet_names)[0]
            )
        )
        diff_shift = 0.0 if item is None else float(item_locations[item])

        # Neutral threshold CCS: for a neutral plot (no specific facet_element
        # requested) use zero facet_effect so thresholds are placed at diff + tau.
        # For the thresholds model with a specific facet_element, each threshold has its
        # own facet_element effect — use per-threshold values rather than the mean.
        if facet_element is None:
            sev_thresh = np.zeros(self.max_score)
        elif model in ("thresholds", "centrality"):
            sev_thresh = facet_effects.loc[r_use].values.astype(float)
        elif model in ("bivector", "matrix", "bistretch"):
            item_key = item if item is not None else list(self.item_names)[0]
            sev_thresh = facet_effects.loc[(r_use, item_key)].values.astype(float)
        elif model in ("items", "pseudo_halo") and item is not None:
            sev_thresh = np.full(
                self.max_score,
                float(self._facet_effect_item_offset(model, facet_effects, r_use)[item]),
            )
        else:
            sev_thresh = np.full(
                self.max_score,
                float(self._facet_effect_item_offset(model, facet_effects, r_use).mean()),
            )
        abs_thresh = thresholds + diff_shift + sev_thresh

        xobsdata = yobsdata = np.array(np.nan)
        obs_curve_index = None
        if obs is not None:
            persons_attr = f'{"anchor_" if anchor else ""}persons_{model}'
            if not hasattr(self, persons_attr):
                self.person_estimates(model=model, anchor=anchor)
            person_locations = getattr(self, persons_attr)
            mean_abs, obs_props = self.class_intervals_thr(
                person_locations,
                item_locations,
                facet_effects,
                model=model,
                item=item,
                facet_element=facet_element,
                no_of_classes=no_of_classes,
            )
            # mean_abs frame depends on what class_intervals_thr subtracted:
            # - facet_element=None: facet_effect subtracted → need to add it back
            # - facet_element specified: nothing subtracted → no shift needed
            # - item=None: item_location also subtracted → add that back too
            if facet_element is None:
                # sev_thresh is zero for neutral plot — shift is diff only
                x_shift = float(item_locations.mean()) if item is None else 0.0
                xobsdata = mean_abs + x_shift
            else:
                # For facet_element-specific plot: mean_abs is raw person_location, no shift needed.
                # For thresholds/matrix model the per-threshold facet_effect is already
                # absorbed into abs_thresh for the curve; obs stay on raw person_location axis.
                xobsdata = mean_abs
            yobsdata = obs_props
            if obs != "all":
                if not all(c in np.arange(self.max_score) + 1 for c in obs):
                    warnings.warn(
                        "Invalid 'obs' value. Valid values are None, 'all', "
                        "or a list of threshold numbers.",
                        UserWarning,
                        stacklevel=2,
                    )
                    return
                obs_idx = [o - 1 for o in obs]
                xobsdata = xobsdata[obs_idx, :]
                yobsdata = yobsdata[obs_idx, :]
                obs_curve_index = obs_idx
        y = np.array(
            [
                [1.0 / (1.0 + np.exp(thr - a)) for thr in abs_thresh]
                for a in person_locations_arr
            ]
        )

        return self.plot_data(
            x_data=person_locations_arr,
            y_data=y,
            model=model,
            anchor=anchor,
            items=item,
            facet_elements=facet_element,
            curve_labels=[f"Threshold {t + 1}" for t in range(self.max_score)],
            obs=None,
            thresh_obs=obs,
            obs_curve_index=obs_curve_index,
            marker_border=marker_border,
            line_border=line_border,
            x_obs_data=xobsdata,
            y_obs_data=yobsdata,
            x_min=xmin,
            x_max=xmax,
            y_max=1,
            thresh_lines=thresh_lines,
            central_location=central_location,
            cat_highlight=cat_highlight,
            graph_title=title or "",
            y_label="Probability",
            plot_style=plot_style,
            palette=palette,
            black=black,
            font=font,
            title_font_size=title_font_size,
            axis_font_size=axis_font_size,
            labelsize=labelsize,
            filename=filename,
            file_format=file_format,
            plot_density=dpi,
        )

    def threshold_ccs_global(self, item=None, **kw):
        """Alias for threshold_ccs(model='global'). See threshold_ccs for full documentation."""
        return self.threshold_ccs(model="global", item=item, **kw)

    def threshold_ccs_items(self, item=None, **kw):
        """Alias for threshold_ccs(model='items'). See threshold_ccs for full documentation."""
        return self.threshold_ccs(model="items", item=item, **kw)

    def threshold_ccs_thresholds(self, item=None, **kw):
        """Alias for threshold_ccs(model='thresholds'). See threshold_ccs for full documentation."""
        return self.threshold_ccs(model="thresholds", item=item, **kw)

    def threshold_ccs_matrix(self, item=None, **kw):
        """Alias for threshold_ccs(model='matrix'). See threshold_ccs for full documentation."""
        return self.threshold_ccs(model="matrix", item=item, **kw)

    def threshold_ccs_bivector(self, item=None, **kw):
        """Alias for threshold_ccs(model='bivector'). See threshold_ccs for full documentation."""
        return self.threshold_ccs(model="bivector", item=item, **kw)

    def threshold_ccs_centrality(self, item=None, **kw):
        """Alias for threshold_ccs(model='centrality'). See threshold_ccs for full documentation."""
        return self.threshold_ccs(model="centrality", item=item, **kw)

    def threshold_ccs_pseudo_halo(self, item=None, **kw):
        """Alias for threshold_ccs(model='pseudo_halo'). See threshold_ccs for full documentation."""
        return self.threshold_ccs(model="pseudo_halo", item=item, **kw)

    def threshold_ccs_bistretch(self, item=None, **kw):
        """Alias for threshold_ccs(model='bistretch'). See threshold_ccs for full documentation."""
        return self.threshold_ccs(model="bistretch", item=item, **kw)

    def iic(
        self,
        item,
        model="global",
        anchor=False,
        facet_element=None,
        ymax=None,
        thresh_lines=False,
        central_location=False,
        point_info_lines=None,
        point_info_labels=False,
        cat_highlight=None,
        title=None,
        xmin=-5,
        xmax=5,
        plot_style="white",
        palette="colorblind multi",
        black=False,
        font="Times New Roman",
        title_font_size=15,
        axis_font_size=12,
        labelsize=12,
        filename=None,
        file_format="png",
        dpi=300,
    ):
        """
        Item Information Curve for one item and/or facet element (e.g.
        facet element (e.g. rater)), or several of either (or both) overlaid on the same axes.

        item : str or list of str
            A single name draws one curve, as before. A list overlays
            one curve per item (in the given order), each in its own
            colour. thresh_lines/cat_highlight only make sense for one
            curve's own location, so they just silently no-op with
            several plotted at once rather than erroring. central_location
            and point_info_lines both work fine with item and/or
            facet_element as a list -- central_location draws each
            item's own line, labelled with its value (and item name);
            point_info_lines draws every curve's own information value
            at each requested location, labelled "curve: info", since
            that comparison at a shared location is the point.
        facet_element : str, list of str, or None
            A single name draws one curve, as before -- None (or
            'none'/'zero') isn't an average here (unlike icc/other
            methods): it resolves to the first facet element, same as
            always. A list overlays one curve per named element instead.

            item and facet_element can both be lists at once, giving one
            curve per (item, facet element) combination -- legended
            "item / facet_element" -- but this is really a lot of curves
            on one plot at once and can get hard to read fast; one list
            and one fixed value reads far more clearly.
        """
        model = self._MODEL_ALIASES.get(model, model)
        item_locations, thresholds, facet_effects = self._get_params(model, anchor)

        items_multi = isinstance(item, list)
        raters_multi = isinstance(facet_element, list)

        if (items_multi or raters_multi):
            # thresh_lines/cat_highlight each only make sense for a
            # single curve's own location -- silently no-op with several
            # plotted at once rather than erroring, since they'd just be
            # visual clutter with no obvious single "right" curve anyway.
            thresh_lines = False
            cat_highlight = None

        estimates = np.arange(-20, 20, 0.1)
        items_to_plot = item if items_multi else [item]
        raters_to_plot = facet_element if raters_multi else [facet_element]

        def _r_use(fe):
            return (
                fe if fe is not None and fe not in ("none", "zero")
                else list(self.facet_names)[0]
            )

        def _curve(it, fe):
            r = _r_use(fe)
            return [
                self.variance(a, it, item_locations, r, facet_effects, thresholds, model)
                for a in estimates
            ]

        y = np.column_stack(
            [_curve(it, fe) for it in items_to_plot for fe in raters_to_plot]
        )

        if items_multi and raters_multi:
            curve_labels = [
                f"{it} / {_r_use(fe)}" for it in items_to_plot for fe in raters_to_plot
            ]
        elif items_multi:
            curve_labels = items_to_plot
        elif raters_multi:
            curve_labels = [_r_use(fe) for fe in raters_to_plot]
        else:
            curve_labels = None

        if ymax is None:
            ymax = float(y.max()) * 1.1
            if central_location and len(items_to_plot) > 0:
                # central_location's labels stack above the curve's own
                # peak (which sits right at each item's own central
                # location -- exactly where those labels are), so widen
                # ymax by exactly the measured label-stack height plus a
                # small gap, rather than guessing a fixed fraction (see
                # _label_height_frac / feedback_plot_layout_rigor --
                # measure, don't guess). Labels are then placed using
                # this same y.max()/k/gap_frac relationship in plot_data,
                # so the fit is exact, not approximate. (central_location's
                # own loop is item-only, independent of facet_element.)
                n_labels = len(items_to_plot)
                k = self._label_height_frac(font, axis_font_size, (8, 6))
                gap_frac = 0.3
                denom = max(1 - k * (n_labels + 2 * gap_frac), 0.1)
                ymax = max(ymax, float(y.max()) / denom)
        return self.plot_data(
            x_data=estimates,
            y_data=y,
            model=model,
            anchor=anchor,
            items=item,
            curve_labels=curve_labels,
            facet_elements=facet_element,
            x_min=xmin,
            x_max=xmax,
            y_max=ymax,
            thresh_lines=thresh_lines,
            central_location=central_location,
            central_location_fit=True,
            point_info_lines_item=[item, point_info_lines],
            score_labels=point_info_labels,
            cat_highlight=cat_highlight,
            graph_title=title or "",
            y_label="Fisher information",
            plot_style=plot_style,
            palette=palette,
            black=black,
            font=font,
            title_font_size=title_font_size,
            axis_font_size=axis_font_size,
            labelsize=labelsize,
            filename=filename,
            plot_density=dpi,
            file_format=file_format,
        )

    def iic_global(self, item, **kw):
        """Alias for iic(model='global'). See iic for full documentation."""
        return self.iic(item, model="global", **kw)

    def iic_items(self, item, **kw):
        """Alias for iic(model='items'). See iic for full documentation."""
        return self.iic(item, model="items", **kw)

    def iic_thresholds(self, item, **kw):
        """Alias for iic(model='thresholds'). See iic for full documentation."""
        return self.iic(item, model="thresholds", **kw)

    def iic_matrix(self, item, **kw):
        """Alias for iic(model='matrix'). See iic for full documentation."""
        return self.iic(item, model="matrix", **kw)

    def iic_bivector(self, item, **kw):
        """Alias for iic(model='bivector'). See iic for full documentation."""
        return self.iic(item, model="bivector", **kw)

    def iic_centrality(self, item, **kw):
        """Alias for iic(model='centrality'). See iic for full documentation."""
        return self.iic(item, model="centrality", **kw)

    def iic_pseudo_halo(self, item, **kw):
        """Alias for iic(model='pseudo_halo'). See iic for full documentation."""
        return self.iic(item, model="pseudo_halo", **kw)

    def iic_bistretch(self, item, **kw):
        """Alias for iic(model='bistretch'). See iic for full documentation."""
        return self.iic(item, model="bistretch", **kw)

    def tcc(
        self,
        model="global",
        anchor=False,
        items=None,
        facet_elements=None,
        obs=False,
        no_of_classes=5,
        title=None,
        score_lines=None,
        score_labels=False,
        xmin=-5,
        xmax=5,
        plot_style="white",
        palette="colorblind multi",
        black=False,
        marker_border=True,
        line_border=False,
        font="Times New Roman",
        title_font_size=15,
        axis_font_size=12,
        labelsize=12,
        filename=None,
        file_format="png",
        dpi=300,
    ):
        """Test Characteristic Curve."""
        model = self._MODEL_ALIASES.get(model, model)
        item_locations, thresholds, facet_effects = self._get_params(model, anchor)
        if isinstance(items, str) and items in ("all", "none"):
            items = None
        if isinstance(facet_elements, str) and facet_elements in (
            "all",
            "none",
            "zero",
        ):
            facet_elements = None

        xobsdata = yobsdata = np.array(np.nan)
        item_keys = (
            list(self.item_names)
            if items is None
            else ([items] if isinstance(items, str) else items)
        )
        rater_list = (
            list(self.facet_names)
            if facet_elements is None
            else (
                [facet_elements] if isinstance(facet_elements, str) else facet_elements
            )
        )

        if obs:
            persons_attr = f'{"anchor_" if anchor else ""}persons_{model}'
            if not hasattr(self, persons_attr):
                self.person_estimates(model=model, anchor=anchor)
            person_estimates = getattr(self, persons_attr)

            df_sub = (
                self.responses.loc[pd.IndexSlice[rater_list, :], item_keys]
                if item_keys != list(self.item_names)
                else self.responses.loc[pd.IndexSlice[rater_list, :], :]
            )

            # TCC obs: restrict to persons with complete data across all
            # facet_element×item combinations in scope, so all totals share the same
            # ceiling and are directly comparable.
            n_expected = len(rater_list) * len(item_keys)
            obs_counts = df_sub.notna().sum(axis=1).groupby(level=1).sum()
            complete_persons = obs_counts[obs_counts == n_expected].index
            n_complete = len(complete_persons)

            if n_complete == 0:
                warnings.warn(
                    "TCC observed score overlay suppressed: no persons have "
                    f"complete data across all {self.facet_label_eg} × item combinations in scope.",
                    UserWarning,
                    stacklevel=2,
                )
                xobsdata = yobsdata = np.array(np.nan)
            else:
                if n_complete < len(obs_counts):
                    warnings.warn(
                        f"TCC observed score overlay uses {n_complete} of "
                        f"{len(obs_counts)} persons with complete data across "
                        f"all {self.facet_label_eg} × item combinations in scope.",
                        UserWarning,
                        stacklevel=2,
                    )
                abil_index = person_estimates.index
                total_scores = df_sub.sum(axis=1).groupby(level=1).sum()
                total_scores = (
                    total_scores.reindex(complete_persons).reindex(abil_index).dropna()
                )
                estimates_aligned = person_estimates.reindex(total_scores.index)

                class_groups = [f"class_{i + 1}" for i in range(no_of_classes)]
                quantiles = estimates_aligned.quantile(
                    [(i + 1) / no_of_classes for i in range(no_of_classes - 1)]
                )
                mask_dict = {
                    "class_1": estimates_aligned < quantiles.values[0],
                    f"class_{no_of_classes}": estimates_aligned >= quantiles.values[-1],
                }
                for i in range(no_of_classes - 2):
                    mask_dict[f"class_{i + 2}"] = (
                        estimates_aligned >= quantiles.values[i]
                    ) & (estimates_aligned < quantiles.values[i + 1])

                xobsdata = pd.Series(
                    {cg: estimates_aligned[mask_dict[cg]].mean() for cg in class_groups}
                )
                yobsdata = pd.Series(
                    {cg: total_scores[mask_dict[cg]].mean() for cg in class_groups}
                )

        person_locations_arr = np.arange(-20, 20, 0.1)
        # Neutral TCC: observed points are mean raw scores on the raw person_location
        # axis with no facet_effect adjustment.  When no specific facet_elements are
        # requested, remove each facet_element's facet_effect contribution from the curve
        # so it also represents a zero-facet_effect baseline.
        # Curve: total expected score summed across all facet_element×item combinations.
        y = np.array(
            [
                sum(
                    self.exp_score(
                        a, it, item_locations, r, facet_effects, thresholds, model
                    )
                    for it in item_keys
                    for r in rater_list
                )
                for a in person_locations_arr
            ]
        ).reshape(-1, 1)
        y_max = self.max_score * len(item_keys) * len(rater_list)

        return self.plot_data(
            x_data=person_locations_arr,
            y_data=y,
            model=model,
            anchor=anchor,
            items=items,
            facet_elements=facet_elements,
            obs=obs,
            marker_border=marker_border,
            line_border=line_border,
            x_obs_data=xobsdata,
            y_obs_data=yobsdata,
            x_min=xmin,
            x_max=xmax,
            y_max=y_max,
            score_lines_test=score_lines,
            score_labels=score_labels,
            graph_title=title or "",
            y_label="Expected score",
            plot_style=plot_style,
            palette=palette,
            black=black,
            font=font,
            title_font_size=title_font_size,
            axis_font_size=axis_font_size,
            labelsize=labelsize,
            filename=filename,
            plot_density=dpi,
            file_format=file_format,
        )

    def tcc_global(self, **kw):
        """Alias for tcc(model='global'). See tcc for full documentation."""
        return self.tcc(model="global", **kw)

    def tcc_items(self, **kw):
        """Alias for tcc(model='items'). See tcc for full documentation."""
        return self.tcc(model="items", **kw)

    def tcc_thresholds(self, **kw):
        """Alias for tcc(model='thresholds'). See tcc for full documentation."""
        return self.tcc(model="thresholds", **kw)

    def tcc_matrix(self, **kw):
        """Alias for tcc(model='matrix'). See tcc for full documentation."""
        return self.tcc(model="matrix", **kw)

    def tcc_bivector(self, **kw):
        """Alias for tcc(model='bivector'). See tcc for full documentation."""
        return self.tcc(model="bivector", **kw)

    def tcc_centrality(self, **kw):
        """Alias for tcc(model='centrality'). See tcc for full documentation."""
        return self.tcc(model="centrality", **kw)

    def tcc_pseudo_halo(self, **kw):
        """Alias for tcc(model='pseudo_halo'). See tcc for full documentation."""
        return self.tcc(model="pseudo_halo", **kw)

    def tcc_bistretch(self, **kw):
        """Alias for tcc(model='bistretch'). See tcc for full documentation."""
        return self.tcc(model="bistretch", **kw)

    def test_info(
        self,
        model="global",
        anchor=False,
        items=None,
        facet_elements=None,
        point_info_lines=None,
        point_info_labels=False,
        xmin=-5,
        xmax=5,
        ymax=None,
        title=None,
        plot_style="white",
        palette="colorblind multi",
        black=False,
        font="Times New Roman",
        title_font_size=15,
        axis_font_size=12,
        labelsize=12,
        filename=None,
        file_format="png",
        dpi=300,
    ):
        """Test Information Curve."""
        model = self._MODEL_ALIASES.get(model, model)
        item_locations, thresholds, facet_effects = self._get_params(model, anchor)
        if isinstance(items, str) and items in ("all", "none"):
            items = None
        if isinstance(facet_elements, str) and facet_elements in (
            "all",
            "none",
            "zero",
        ):
            facet_elements = None
        if isinstance(items, str):
            items = None if items in ("all", "none") else [items]
        if isinstance(facet_elements, str):
            facet_elements = (
                None if facet_elements in ("all", "none", "zero") else [facet_elements]
            )
        item_keys = list(self.item_names) if items is None else items
        rater_list = (
            list(self.facet_names) if facet_elements is None else facet_elements
        )

        estimates = np.arange(-20, 20, 0.1)
        y = np.array(
            [
                sum(
                    self.variance(a, it, item_locations, r, facet_effects, thresholds, model)
                    for it in item_keys
                    for r in rater_list
                )
                for a in estimates
            ]
        ).reshape(-1, 1)
        if ymax is None:
            ymax = float(y.max()) * 1.1

        return self.plot_data(
            x_data=estimates,
            y_data=y,
            model=model,
            anchor=anchor,
            items=items,
            facet_elements=facet_elements,
            x_min=xmin,
            x_max=xmax,
            y_max=ymax,
            graph_title=title or "",
            point_info_lines_test=point_info_lines,
            score_labels=point_info_labels,
            y_label="Fisher information",
            plot_style=plot_style,
            palette=palette,
            black=black,
            font=font,
            title_font_size=title_font_size,
            axis_font_size=axis_font_size,
            labelsize=labelsize,
            filename=filename,
            plot_density=dpi,
            file_format=file_format,
        )

    def test_info_global(self, **kw):
        """Alias for test_info(model='global'). See test_info for full documentation."""
        return self.test_info(model="global", **kw)

    def test_info_items(self, **kw):
        """Alias for test_info(model='items'). See test_info for full documentation."""
        return self.test_info(model="items", **kw)

    def test_info_thresholds(self, **kw):
        """Alias for test_info(model='thresholds'). See test_info for full documentation."""
        return self.test_info(model="thresholds", **kw)

    def test_info_matrix(self, **kw):
        """Alias for test_info(model='matrix'). See test_info for full documentation."""
        return self.test_info(model="matrix", **kw)

    def test_info_bivector(self, **kw):
        """Alias for test_info(model='bivector'). See test_info for full documentation."""
        return self.test_info(model="bivector", **kw)

    def test_info_centrality(self, **kw):
        """Alias for test_info(model='centrality'). See test_info for full documentation."""
        return self.test_info(model="centrality", **kw)

    def test_info_pseudo_halo(self, **kw):
        """Alias for test_info(model='pseudo_halo'). See test_info for full documentation."""
        return self.test_info(model="pseudo_halo", **kw)

    def test_info_bistretch(self, **kw):
        """Alias for test_info(model='bistretch'). See test_info for full documentation."""
        return self.test_info(model="bistretch", **kw)

    def test_csem(
        self,
        model="global",
        anchor=False,
        items=None,
        facet_elements=None,
        point_csem_lines=None,
        point_csem_labels=False,
        xmin=-5,
        xmax=5,
        ymax=5,
        title=None,
        plot_style="white",
        palette="colorblind multi",
        black=False,
        font="Times New Roman",
        title_font_size=15,
        axis_font_size=12,
        labelsize=12,
        filename=None,
        file_format="png",
        dpi=300,
    ):
        """Test Conditional Standard Error of Measurement Curve."""
        model = self._MODEL_ALIASES.get(model, model)
        item_locations, thresholds, facet_effects = self._get_params(model, anchor)
        if isinstance(items, str) and items in ("all", "none"):
            items = None
        if isinstance(facet_elements, str) and facet_elements in (
            "all",
            "none",
            "zero",
        ):
            facet_elements = None
        if isinstance(items, str):
            items = None if items in ("all", "none") else [items]
        if isinstance(facet_elements, str):
            facet_elements = (
                None if facet_elements in ("all", "none", "zero") else [facet_elements]
            )
        item_keys = list(self.item_names) if items is None else items
        rater_list = (
            list(self.facet_names) if facet_elements is None else facet_elements
        )

        estimates = np.arange(-20, 20, 0.1)
        info = np.array(
            [
                sum(
                    self.variance(a, it, item_locations, r, facet_effects, thresholds, model)
                    for it in item_keys
                    for r in rater_list
                )
                for a in estimates
            ]
        )
        y = (1.0 / (info**0.5)).reshape(-1, 1)

        return self.plot_data(
            x_data=estimates,
            y_data=y,
            model=model,
            anchor=anchor,
            items=items,
            facet_elements=facet_elements,
            x_min=xmin,
            x_max=xmax,
            y_max=ymax,
            graph_title=title or "",
            point_csem_lines=point_csem_lines,
            score_labels=point_csem_labels,
            y_label="Conditional SEM",
            plot_style=plot_style,
            palette=palette,
            black=black,
            font=font,
            title_font_size=title_font_size,
            axis_font_size=axis_font_size,
            labelsize=labelsize,
            filename=filename,
            plot_density=dpi,
            file_format=file_format,
        )

    def test_csem_global(self, **kw):
        """Alias for test_csem(model='global'). See test_csem for full documentation."""
        return self.test_csem(model="global", **kw)

    def test_csem_items(self, **kw):
        """Alias for test_csem(model='items'). See test_csem for full documentation."""
        return self.test_csem(model="items", **kw)

    def test_csem_thresholds(self, **kw):
        """Alias for test_csem(model='thresholds'). See test_csem for full documentation."""
        return self.test_csem(model="thresholds", **kw)

    def test_csem_matrix(self, **kw):
        """Alias for test_csem(model='matrix'). See test_csem for full documentation."""
        return self.test_csem(model="matrix", **kw)

    def test_csem_bivector(self, **kw):
        """Alias for test_csem(model='bivector'). See test_csem for full documentation."""
        return self.test_csem(model="bivector", **kw)

    def test_csem_centrality(self, **kw):
        """Alias for test_csem(model='centrality'). See test_csem for full documentation."""
        return self.test_csem(model="centrality", **kw)

    def test_csem_pseudo_halo(self, **kw):
        """Alias for test_csem(model='pseudo_halo'). See test_csem for full documentation."""
        return self.test_csem(model="pseudo_halo", **kw)

    def test_csem_bistretch(self, **kw):
        """Alias for test_csem(model='bistretch'). See test_csem for full documentation."""
        return self.test_csem(model="bistretch", **kw)

    def std_residuals_plot(
        self,
        model="global",
        items=None,
        facet_elements=None,
        bin_width=0.5,
        x_min=-6,
        x_max=6,
        normal=False,
        title=None,
        plot_style="white",
        black=False,
        font="Times New Roman",
        title_font_size=15,
        axis_font_size=12,
        labelsize=12,
        filename=None,
        file_format="png",
        dpi=300,
        plot_density=None,
    ):
        """Standardised residuals histogram with optional item/facet_element subsetting."""
        if plot_density is not None:
            warnings.warn(
                "plot_density= is deprecated and will be removed in a future version; use dpi= instead.",
                FutureWarning, stacklevel=2,
            )
            dpi = plot_density
        model = self._MODEL_ALIASES.get(model, model)
        if not hasattr(self, f"std_residual_df_{model}"):
            self.fit_statistics(model=model)

        std_res = getattr(self, f"std_residual_df_{model}")

        # Normalise string arguments
        if isinstance(facet_elements, str):
            if facet_elements in ("all", "none"):
                facet_elements = None
            else:
                facet_elements = [facet_elements]
        if isinstance(items, str):
            if items in ("all", "none"):
                items = None
            else:
                items = [items]

        # Subset
        if items is None and facet_elements is None:
            residuals = pd.Series(std_res.values.flatten()).dropna()
        elif items is None:
            residuals = pd.Series(std_res.loc[facet_elements].values.flatten()).dropna()
        elif facet_elements is None:
            residuals = pd.Series(std_res[items].values.flatten()).dropna()
        else:
            residuals = pd.Series(
                std_res[items].loc[facet_elements].values.flatten()
            ).dropna()

        return self.std_residuals_hist(
            residuals,
            bin_width=bin_width,
            x_min=x_min,
            x_max=x_max,
            normal=normal,
            title=title,
            plot_style=plot_style,
            black=black,
            font=font,
            title_font_size=title_font_size,
            axis_font_size=axis_font_size,
            labelsize=labelsize,
            filename=filename,
            file_format=file_format,
            plot_density=dpi,
        )

    def std_residuals_plot_global(self, **kw):
        """Alias for std_residuals_plot(model='global'). See std_residuals_plot for full documentation."""
        return self.std_residuals_plot(model="global", **kw)

    def std_residuals_plot_items(self, **kw):
        """Alias for std_residuals_plot(model='items'). See std_residuals_plot for full documentation."""
        return self.std_residuals_plot(model="items", **kw)

    def std_residuals_plot_thresholds(self, **kw):
        """Alias for std_residuals_plot(model='thresholds'). See std_residuals_plot for full documentation."""
        return self.std_residuals_plot(model="thresholds", **kw)

    def std_residuals_plot_matrix(self, **kw):
        """Alias for std_residuals_plot(model='matrix'). See std_residuals_plot for full documentation."""
        return self.std_residuals_plot(model="matrix", **kw)

    def std_residuals_plot_bivector(self, **kw):
        """Alias for std_residuals_plot(model='bivector'). See std_residuals_plot for full documentation."""
        return self.std_residuals_plot(model="bivector", **kw)
    def std_residuals_plot_bistretch(self, **kw):
        """Alias for std_residuals_plot(model='bistretch'). See std_residuals_plot for full documentation."""
        return self.std_residuals_plot(model="bistretch", **kw)

    def std_residuals_plot_centrality(self, **kw):
        """Alias for std_residuals_plot(model='centrality'). See std_residuals_plot for full documentation."""
        return self.std_residuals_plot(model="centrality", **kw)

    def std_residuals_plot_pseudo_halo(self, **kw):
        """Alias for std_residuals_plot(model='pseudo_halo'). See std_residuals_plot for full documentation."""
        return self.std_residuals_plot(model="pseudo_halo", **kw)

    # Backwards-compatible aliases
    def wright_map_global(self, **kw):
        """Alias for wright_map(model='global'). See wright_map for full documentation."""
        return self.wright_map(model="global", **kw)

    def wright_map_items(self, **kw):
        """Alias for wright_map(model='items'). See wright_map for full documentation."""
        return self.wright_map(model="items", **kw)

    def wright_map_thresholds(self, **kw):
        """Alias for wright_map(model='thresholds'). See wright_map for full documentation."""
        return self.wright_map(model="thresholds", **kw)

    def wright_map_matrix(self, **kw):
        """Alias for wright_map(model='matrix'). See wright_map for full documentation."""
        return self.wright_map(model="matrix", **kw)

    def wright_map_bivector(self, **kw):
        """Alias for wright_map(model='bivector'). See wright_map for full documentation."""
        return self.wright_map(model="bivector", **kw)

    def wright_map_centrality(self, **kw):
        """Alias for wright_map(model='centrality'). See wright_map for full documentation."""
        return self.wright_map(model="centrality", **kw)

    def wright_map_pseudo_halo(self, **kw):
        """Alias for wright_map(model='pseudo_halo'). See wright_map for full documentation."""
        return self.wright_map(model="pseudo_halo", **kw)

    def wright_map_bistretch(self, **kw):
        """Alias for wright_map(model='bistretch'). See wright_map for full documentation."""
        return self.wright_map(model="bistretch", **kw)

    # ------------------------------------------------------------------
    # Simulation recovery diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def _category_widths(thresholds):
        """Category widths (tau_{k+1} - tau_k), relabelled 1..max_score-1."""
        widths = thresholds.diff().dropna()
        widths.index = range(1, len(thresholds))
        return widths

    def _rater_effect_subvectors(self, model, anchor=False):
        """
        (label, generating_series, estimated_series, se_series) tuples
        for model's own compact facet-effect representation --
        1 sub-vector for global/items/thresholds/matrix, 2 for
        bivector/centrality/pseudo_halo, 3 for bistretch. Only valid when
        model == self.generating.model (the matched case) -- see
        recovery_stats()'s cross-submodel fallback (_generating_severity_grid
        / _severity_grid) for the mismatched case, which has no analogous
        per-submodel SE.
        """
        prefix = "anchor_" if anchor else ""
        gen = self.generating
        if model in ("global", "items", "thresholds", "matrix"):
            est = getattr(self, f"{prefix}facet_effects_{model}")
            se = getattr(self, f"{prefix}rater_se_{model}")
            gen_vals = gen.facet_effects
            if isinstance(est, pd.Series):
                gen_flat, est_flat, se_flat = gen_vals.reindex(est.index), est, se
            else:
                gen_aligned = gen_vals.reindex(index=est.index, columns=est.columns)
                gen_flat, est_flat, se_flat = (
                    gen_aligned.stack(), est.stack(), se.stack(),
                )
            return [(model, gen_flat, est_flat, se_flat)]
        if model == "bivector":
            est_items = getattr(self, f"{prefix}facet_effects_bivector_items")
            est_thr = getattr(self, f"{prefix}facet_effects_bivector_thresholds")
            se_items = getattr(self, f"{prefix}rater_se_marginal_items")
            se_thr = getattr(self, f"{prefix}rater_se_marginal_thresholds")
            gen_items = gen.item_effects.reindex(index=est_items.index, columns=est_items.columns)
            gen_thr = gen.threshold_effects.reindex(index=est_thr.index, columns=est_thr.columns)
            return [
                ("items", gen_items.stack(), est_items.stack(), se_items.stack()),
                ("thresholds", gen_thr.stack(), est_thr.stack(), se_thr.stack()),
            ]
        if model in ("centrality", "pseudo_halo"):
            compact = getattr(self, f"{prefix}facet_effects_{model}")
            return [
                (
                    "lambda", gen.lambda_.reindex(compact.index),
                    compact["lambda"], getattr(self, f"{prefix}lambda_se_{model}"),
                ),
                (
                    "omega", gen.omega.reindex(compact.index),
                    compact["omega"], getattr(self, f"{prefix}omega_se_{model}"),
                ),
            ]
        if model == "bistretch":
            compact = getattr(self, f"{prefix}facet_effects_bistretch")
            return [
                (
                    "lambda", gen.lambda_.reindex(compact.index),
                    compact["lambda"], getattr(self, f"{prefix}lambda_se_bistretch"),
                ),
                (
                    "omega_items", gen.omega_items.reindex(compact.index),
                    compact["omega_items"], getattr(self, f"{prefix}omega_items_se_bistretch"),
                ),
                (
                    "omega_thresholds", gen.omega_thresholds.reindex(compact.index),
                    compact["omega_thresholds"], getattr(self, f"{prefix}omega_thresholds_se_bistretch"),
                ),
            ]
        raise ValueError(f"Unrecognised model {model!r}.")

    def recovery_stats(
        self, model=None, anchor=False, groups=None, rater_subset=None,
        separate_rater_plots=False, alpha=0.05, correction="bh",
        divergence_tol=0.1, plot=True, **plot_kwargs,
    ):
        """
        Compare recovered parameter estimates against the true generating
        values from the simulation this object was constructed from.

        No-op (warns, returns None) unless this object was built from a
        simulation (self.generating exists). Unlike SLM/PCM/RSM, there is
        NO further model-match requirement here: `model` defaults to
        self.generating.model ("recover whichever model was actually
        generated"), but an EXPLICIT model= that differs from
        self.generating.model is allowed through deliberately -- Elliott &
        Buttery (2022a) themselves used exactly this construction (fitting
        a simpler model to data generated under a richer one) to
        demonstrate the person-measurement cost of model misspecification,
        so a strict match would block MFRM's own primary use case for
        this method.

        Items/thresholds/category_widths are compared directly regardless
        of model vs self.generating.model, using the sub-model's own
        estimates (_items_for(model)/_thresholds_for(model)).

        Facet-effect recovery has two modes. When model ==
        self.generating.model (the matched case), it uses model's own
        compact representation with proper bootstrap SEs (1 sub-vector for
        global/items/thresholds/matrix, 2 for bivector/centrality/
        pseudo_halo, 3 for bistretch -- see _rater_effect_subvectors).
        When they differ (a deliberate misspecification study), it falls
        back to comparing the full broadcast (facet element, item, category)
        location grid (_severity_grid / _generating_severity_grid) with NO
        SE -- there's no single well-defined SE for a grid cell assembled
        from a coarser or differently-shaped representation, so z/p/
        Flagged are left NaN/False there; RMSE/MAE/bias/r/r_squared/SD_ratio/
        regression remain meaningful and are still computed.

        Person recovery does NOT compare the recovered person estimate
        directly against self.generating.persons -- that would conflate
        item/threshold/rater estimation and model-specification error
        with the irreducible stochastic noise of a single finite response
        pattern. Instead, following Elliott & Buttery (2022a), a pseudo-
        estimate is computed: the same Newton-Raphson person() procedure
        re-run on the same self.responses, but using the TRUE generating
        item/threshold/facet-effect parameters (under self.generating.model,
        NOT necessarily `model`) in place of the estimated ones. The
        persons table's 'error' column is (recovered estimate -
        pseudo-estimate), not (recovered estimate - true theta) -- both
        share the same actual response-pattern noise, so differencing
        cancels that shared stochastic component, leaving the COMBINED
        effect of parameter estimation error and model misspecification
        on person measurement (this method does not, and cannot, separate
        those two).

        groups='rater' (MFRM-specific) triggers a by-facet-element breakdown of
        person recovery: for each facet element R, both the recovered estimate and
        the pseudo-estimate are recomputed restricted to just R's own
        response row per person (person(facet_elements=[R])), isolating
        whether a SPECIFIC facet element's own location/calibration quality is
        dragging down person measurement, as distinct from the general
        (all-facet-elements-pooled) case. This is NOT a plain partition of the
        pooled result by group label (a person is typically rated by
        several facet elements, so "which group" doesn't apply the way a
        demographic covariate would) -- it's a genuinely separate
        restricted re-estimation per facet element. `rater_subset` (list of facet element
        names, default None = all of self.facet_names) restricts which
        facet elements get their own direct-comparison plot (see below); it does
        not affect the summary table, which always covers every facet element.
        A plain pandas.Series may also be passed to `groups` for ordinary
        person-level grouping (mirrors wright_map's group_by convention),
        applying to person recovery only.

        Parameters
        ----------
        model : str or None, default None
            Which facet element sub-model's estimates to recover. None defaults
            to self.generating.model.
        anchor : bool, default False
            If True, compare against model's anchored attributes
            (self.anchor_items_{model}, self.anchor_thresholds_{model},
            self.anchor_persons_{model}, self.anchor_facet_effects_{model})
            instead of the unanchored ones. Raises ValueError if
            calibrate_anchor() has not been run for this model.
        groups : pandas.Series, 'rater', or None, default None
            Person-level grouping for person recovery (see above).
        rater_subset : list or None, default None
            Restricts the by-facet-element direct-comparison plot (groups='rater'
            only) to these facet elements. None uses every facet element.
        separate_rater_plots : bool, default False
            For bivector/bistretch's multiple facet element sub-vectors, False
            (default) plots them combined on one axes (different colours,
            separate OLS/Huber fits); True plots them separately.
        alpha : float, default 0.05
            Significance threshold for the per-parameter z-test.
        correction : {'bh', 'bonferroni', None}, default 'bh'
            Multiple-comparison correction, applied within each parameter
            type's own family of tests.
        divergence_tol : float, default 0.1
            Threshold for the OLS-vs-Huber slope divergence flag.
        plot : bool, default True
            If True, also produce a recovery plot per parameter type.
        **plot_kwargs
            Forwarded to _recovery_plot().

        Returns
        -------
        None
            Results are stored in self.recovery: .items/.persons/
            .thresholds/.category_widths/.raters, each holding .table and
            .summary (see _parameter_recovery_table). Also sets
            self.recovery.parameter_estimation_residual (float): the
            Elliott & Buttery (2022) expected-score-space RMS residual
            between generating and estimated item location + shared
            thresholds at the true generating theta, deliberately
            facet-free (facet element effects excluded) -- see
            _parameter_estimation_residual.
        """
        from raschpy.base import _SimParams

        if not hasattr(self, "generating"):
            warnings.warn(
                "recovery_stats() is a no-op: this object was not built "
                "from a simulation (no self.generating found).",
                UserWarning,
                stacklevel=2,
            )
            return None

        gen_model = self.generating.model
        if model is None:
            model = gen_model
        model = self._MODEL_ALIASES.get(model, model)

        prefix = "anchor_" if anchor else ""
        if anchor:
            # No anchor_csem_vector_{model} requirement -- CSEM (Fisher-
            # information-based conditional SEM) is invariant to a pure
            # anchor shift (a constant translation changes neither the
            # residual nor information structure it's built from), unlike
            # the bootstrap SEs below which genuinely differ under
            # anchoring (they must account for anchor-rater sampling
            # variance) -- confirmed no anchor_csem_vector_{model}
            # attribute exists anywhere in the package. Person SE below
            # uses the plain csem_vector_{model} regardless of anchor.
            required = [
                f"anchor_items_{model}", f"anchor_thresholds_{model}",
                "anchor_item_se", f"anchor_threshold_se_{model}",
                f"anchor_cat_width_se_{model}", f"csem_vector_{model}",
            ]
            missing = [a for a in required if not hasattr(self, a)]
            if missing:
                raise ValueError(
                    f"anchor=True requires the full anchored analysis to "
                    f"have already been run for model={model!r} (missing: "
                    f"{missing}). Run calibrate_anchor(model={model!r}, ...) "
                    f"(or calibrate_{model}_anchor()) followed by "
                    f"anchor_std_errors(model={model!r}, ...), "
                    f"person_estimates(model={model!r}, anchor=True), and "
                    f"fit_statistics(model={model!r}) (for the plain, "
                    f"anchor-invariant csem_vector_{model}) first -- "
                    f"recovery_stats() does not lazily auto-trigger the "
                    f"anchored bootstrap (it needs an explicit anchors= "
                    f"list this method has no way to infer safely)."
                )
        else:
            if not hasattr(self, f"facet_effects_{model}"):
                self.calibrate(model=model)
            # interval=0.95 here (not strictly needed by recovery_stats()
            # itself, which only reads item_se/csem_vector_{model}) so a
            # later full=True call to rater_stats_df()/etc. doesn't crash:
            # those check hasattr(self, "item_se") to decide whether to
            # (re-)bootstrap, and would otherwise find item_se already set
            # but item_low/item_high still missing -- a real crash found
            # via testing the same issue in SLM/PCM/RSM, not a hypothetical.
            if not hasattr(self, f"csem_vector_{model}"):
                self.fit_statistics(model=model, interval=0.95)

        self.recovery = _SimParams()

        # the sub-model's own item locations / thresholds (anchored or not)
        est_items = self._items_for(model, anchor=anchor)
        items_table, items_summary = self._parameter_recovery_table(
            self.generating.items, est_items, se=self._item_se_for(model, anchor=anchor),
            alpha=alpha, correction=correction, divergence_tol=divergence_tol,
        )
        self.recovery.items = _SimParams()
        self.recovery.items.table = items_table
        self.recovery.items.summary = items_summary

        est_thresh = self._thresholds_for(model, anchor=anchor)
        threshold_se = pd.Series(
            getattr(self, f"{prefix}threshold_se_{model}"), index=range(1, self.max_score + 1)
        )
        thresholds_table, thresholds_summary = self._parameter_recovery_table(
            self.generating.thresholds, est_thresh, se=threshold_se, alpha=alpha,
            correction=correction, divergence_tol=divergence_tol,
        )
        self.recovery.thresholds = _SimParams()
        self.recovery.thresholds.table = thresholds_table
        self.recovery.thresholds.summary = thresholds_summary

        gen_widths = self._category_widths(self.generating.thresholds)
        est_widths = self._category_widths(est_thresh)
        cat_width_se = pd.Series(getattr(self, f"{prefix}cat_width_se_{model}"))
        widths_table, widths_summary = self._parameter_recovery_table(
            gen_widths, est_widths, se=cat_width_se, alpha=alpha,
            correction=correction, divergence_tol=divergence_tol,
        )
        self.recovery.category_widths = _SimParams()
        self.recovery.category_widths.table = widths_table
        self.recovery.category_widths.summary = widths_summary

        # Persons -- pseudo-estimate construction (Elliott & Buttery, 2022a).
        # Scratch instance, never mutates self (same convention as
        # RSM._threshold_structure_ll/std_errors/andersen_lr_test). Built
        # under self.generating.model, NOT `model` -- the pseudo-estimate
        # always reflects the TRUE generating structure.
        probe = MFRM(self.responses, max_score=self.generating.max_score)
        probe.items = self.generating.items
        probe.thresholds = self.generating.thresholds
        setattr(probe, probe._sev_attr(gen_model), self.generating.facet_effects)
        # _get_params' auto-calibrate gate checks hasattr(self,
        # f"facet_effects_{model}") -- the COMPACT name, not _sev_attr's
        # own name (which has a "_full_vector" suffix for centrality/
        # pseudo_halo/bistretch specifically). Without also setting the
        # compact name here, that gate would see "not yet calibrated" for
        # those three models and silently trigger a full independent
        # recalibration of the probe from the real data -- discovered
        # empirically: persons RMSE came out as exactly 0.0 for all three
        # stretch models before this fix, since the probe was just
        # reproducing self's own real fit rather than using the injected
        # generating severities at all. The compact attribute's VALUE is
        # never actually read for these three models (only its presence
        # matters for the gate -- _get_params returns the _full_vector
        # attribute's value, not this one), so any correctly-shaped
        # placeholder is fine.
        if not hasattr(probe, f"facet_effects_{gen_model}"):
            setattr(probe, f"facet_effects_{gen_model}", self.generating.facet_effects)

        if not hasattr(self, f"{prefix}persons_{model}"):
            self.person_estimates(model=model, anchor=anchor)
        est_persons = getattr(self, f"{prefix}persons_{model}")
        # csem_vector_{model} is never anchor-prefixed -- CSEM is
        # invariant to a pure anchor shift, see the anchor=True check
        # above.
        csem = getattr(self, f"csem_vector_{model}")

        if isinstance(groups, str) and groups == "rater":
            raters = list(self.facet_names)
            plot_raters = rater_subset if rater_subset is not None else raters
            real_by_rater = {}
            pseudo_by_rater = {}
            for r in raters:
                real_by_rater[r] = self.person(
                    self.person_names, model=model, anchor=anchor, facet_elements=[r],
                )
                pseudo_by_rater[r] = probe.person(
                    probe.person_names, model=gen_model, facet_elements=[r],
                )
            person_groups = pd.concat(
                {r: pd.Series(r, index=real_by_rater[r].index) for r in raters}
            ).droplevel(0)
            # Flatten to (rater, person) rows for the table/summary, plus
            # 'Overall' via the pooled pseudo/real estimate already
            # computed above.
            real_flat = pd.concat(
                {r: real_by_rater[r] for r in raters}
            )
            pseudo_flat = pd.concat(
                {r: pseudo_by_rater[r] for r in raters}
            )
            group_labels = pd.Series(
                [idx[0] for idx in real_flat.index], index=real_flat.index
            )
            persons_table, persons_summary = self._parameter_recovery_table(
                pseudo_flat, real_flat, se=None, alpha=alpha,
                correction=correction, divergence_tol=divergence_tol,
                groups=group_labels,
            )
            # Overall column reflects the pooled (all-rater) comparison,
            # not an aggregate of the per-rater ones -- overwrite it.
            probe.person_estimates(model=gen_model)
            pseudo_estimate = getattr(probe, f"persons_{gen_model}")
            _, pooled_summary = self._parameter_recovery_table(
                pseudo_estimate, est_persons, se=csem, alpha=alpha,
                correction=correction, divergence_tol=divergence_tol,
            )
            persons_summary["Overall"] = pooled_summary["Overall"]
        else:
            person_groups = groups if isinstance(groups, pd.Series) else None
            probe.person_estimates(model=gen_model)
            pseudo_estimate = getattr(probe, f"persons_{gen_model}")
            persons_table, persons_summary = self._parameter_recovery_table(
                pseudo_estimate, est_persons, se=csem, alpha=alpha,
                correction=correction, divergence_tol=divergence_tol,
                groups=person_groups,
            )
        self.recovery.persons = _SimParams()
        self.recovery.persons.table = persons_table
        self.recovery.persons.summary = persons_summary

        # Parameter-estimation residual (Elliott & Buttery, 2022): RMS
        # expected-score residual between (generating item location,
        # generating shared thresholds) and (estimated item location,
        # estimated shared thresholds), jointly -- at the true generating
        # theta throughout. Deliberately facet-free (ignores rater
        # effects entirely), matching wright_map's own established
        # facet-free convention for item/threshold-only point-position
        # math -- rater recovery has its own, separate comparison below
        # (self.recovery.raters), and folding 8 different rater-
        # representation shapes into this one scalar would need a design
        # decision this metric doesn't need to make.
        def _facet_free_exp_score(theta, item_location, thresh):
            thresh = np.asarray(thresh)
            cats = np.arange(len(thresh) + 1, dtype=float)
            cumsum = np.concatenate([[0.0], np.cumsum(thresh)])
            log_nums = cats * (theta - item_location) - cumsum
            log_nums -= log_nums.max()
            nums = np.exp(log_nums)
            probs = nums / nums.sum()
            return float((cats * probs).sum())

        gen_items_ = self.generating.items
        gen_thresh_ = self.generating.thresholds
        self.recovery.parameter_estimation_residual = self._parameter_estimation_residual(
            self.generating.persons,
            lambda theta, item: _facet_free_exp_score(theta, gen_items_[item], gen_thresh_),
            lambda theta, item: _facet_free_exp_score(theta, est_items[item], est_thresh),
        )

        # Rater/facet effects. subvectors is a plain dict (not a _SimParams
        # attribute namespace) since a sub-vector label can be 'global' --
        # a Python keyword, so `.subvectors.global` would be a SyntaxError
        # via dot access even though setattr/getattr would technically work.
        self.recovery.raters = _SimParams()
        if model == gen_model:
            subvectors = self._rater_effect_subvectors(model, anchor=anchor)
            raters_gen = {}
            raters_est = {}
            sub_tables = {}
            for label, g, e, se in subvectors:
                sub_table, sub_summary = self._parameter_recovery_table(
                    g, e, se=se, alpha=alpha, correction=correction,
                    divergence_tol=divergence_tol,
                )
                sub_tables[label] = _SimParams()
                sub_tables[label].table = sub_table
                sub_tables[label].summary = sub_summary
                raters_gen[label] = g
                raters_est[label] = e
            self.recovery.raters.subvectors = sub_tables
            self.recovery.raters.table = None
            self.recovery.raters.summary = None
        else:
            gen_grid = self._generating_severity_grid(gen_model).ravel()
            est_grid = self._severity_grid(model).ravel()
            mi = pd.MultiIndex.from_product(
                [self.facet_names, self.item_names, range(1, self.max_score + 1)]
            )
            gen_s = pd.Series(gen_grid, index=mi)
            est_s = pd.Series(est_grid, index=mi)
            raters_table, raters_summary = self._parameter_recovery_table(
                gen_s, est_s, se=None, alpha=alpha, correction=correction,
                divergence_tol=divergence_tol,
            )
            self.recovery.raters.table = raters_table
            self.recovery.raters.summary = raters_summary
            self.recovery.raters.subvectors = None
            raters_gen = {"raters": gen_s}
            raters_est = {"raters": est_s}

        if plot:
            self._recovery_plot(self.generating.items, est_items, "items", **plot_kwargs)
            self._recovery_plot(
                self.generating.thresholds, est_thresh, "thresholds", **plot_kwargs
            )
            self._recovery_plot(
                gen_widths, est_widths, "category widths", **plot_kwargs
            )
            self._recovery_plot(pseudo_estimate, est_persons, "persons", **plot_kwargs)
            if isinstance(groups, str) and groups == "rater":
                self._recovery_plot(
                    {r: pseudo_by_rater[r] for r in plot_raters},
                    {r: real_by_rater[r] for r in plot_raters},
                    "persons (by rater)", **plot_kwargs,
                )
            if separate_rater_plots:
                for label in raters_gen:
                    self._recovery_plot(
                        raters_gen[label], raters_est[label],
                        f"raters ({label})", **plot_kwargs,
                    )
            else:
                self._recovery_plot(raters_gen, raters_est, "raters", **plot_kwargs)

            if anchor:
                # Separate anchor-vs-non-anchor view -- a different visual
                # encoding purpose (which raters were anchored) from the
                # subvector colouring above, so plotted as its own figure
                # rather than combined into it. Flattens every sub-vector
                # together (rater is always level 0 of the index, whether
                # it's a plain rater-name Index for 'global'-shaped
                # sub-vectors or a MultiIndex with rater first for
                # everything else).
                anchor_names = set(getattr(self, f"anchor_rater_names_{model}"))
                gen_all = pd.concat(list(raters_gen.values()))
                est_all = pd.concat(list(raters_est.values()))
                rater_level = (
                    gen_all.index.get_level_values(0)
                    if isinstance(gen_all.index, pd.MultiIndex) else gen_all.index
                )
                is_anchor = pd.Series(rater_level, index=gen_all.index).isin(anchor_names)
                self._recovery_plot(
                    {
                        "Anchor raters": gen_all[is_anchor.values],
                        "Non-anchor raters": gen_all[~is_anchor.values],
                    },
                    {
                        "Anchor raters": est_all[is_anchor.values],
                        "Non-anchor raters": est_all[~is_anchor.values],
                    },
                    "raters (anchor split)",
                    colors={"Anchor raters": "black", "Non-anchor raters": "grey"},
                    markers={"Anchor raters": "^", "Non-anchor raters": "v"},
                    **plot_kwargs,
                )

        return None
