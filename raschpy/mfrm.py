from math import log
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib import colors as colors
from matplotlib import cm as cmx
import seaborn as sns

from raschpy.base import Rasch


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
    _MODELS = ("global", "items", "thresholds", "bivector", "matrix")

    def _attr(self, model, name, anchor=False):
        """Return the attribute name for a given model and statistic."""
        prefix = "anchor_" if anchor else ""
        suffix = f"_{model}"
        return f"{prefix}{name}{suffix}"

    def _get_params(self, model, anchor=False):
        """
        Return (item_locations, thresholds, facet_effects) for the requested model.
        Auto-triggers calibration if not yet run.
        """
        if anchor:
            diff_attr = f"anchor_items_{model}"
            thr_attr = f"anchor_thresholds_{model}"
            sev_attr = f"anchor_facet_effects_{model}"
            if not hasattr(self, diff_attr):
                raise AttributeError(
                    f"Anchor calibration required. "
                    f"Run self.calibrate_{model}_anchor()."
                )
        else:
            diff_attr = "items"
            thr_attr = "thresholds"
            sev_attr = f"facet_effects_{model}"
            if not hasattr(self, sev_attr):
                self.calibrate(model=model)
        return (
            getattr(self, diff_attr),
            getattr(self, thr_attr),
            getattr(self, sev_attr),
        )

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
        facet="rater",
        facet_plural=None,
        validate=True,
    ):
        """
        Initialise a Many-Facet Rasch Model (MFRM) object.

        The MFRM extends the RSM/PCM to include facet_element facets. Four facet_element
        parameterisations are supported, selected at calibrate() time:
        'global' (single facet_effect per facet_element), 'items' (per-item facet_effects),
        'thresholds' (per-threshold facet_effects), 'matrix' (per-item-threshold).

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
        validate : bool, default True
            If True, checks whether the item response network is fully
            connected (i.e. all items are linked via common facet_element
            comparisons). Issues a UserWarning if the data is split into
            disconnected sub-networks, which makes item locations
            incomparable across sub-groups.

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
        items : pandas.Index
            Item identifiers (column names).
        facet_elements : pandas.Index
            Rater identifiers.
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

        # Facet name aliases (e.g. self.judge_names, self.judges)
        setattr(self, f"{self.facet}_names", self.facet_names)
        setattr(self, self.facets, self.facet_names)

        # Anchor facet_element tracking per model
        for model in self._MODELS:
            setattr(self, f"anchor_rater_names_{model}", [])

        # Dynamic method aliases for facet-named stats and res_corr (Phase 4)
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

            # THROW SYSTEM WARNING WITH MATHEMATICAL DETAILS IF DISCONNECTED
            if not self.connectivity_status["connected"]:
                warnings.warn(
                    f"\n"
                    f"⚠️  CRITICAL DATA INTEGRITY WARNING: The response data is disconnected into "
                    f"{self.connectivity_status['components_count']} separate sub-networks.\n"
                    f"Item location estimates will be problematic because there are no empirical "
                    f"comparisons spanning across these isolated groups, the item parameter "
                    f"estimates for each independent subset will separately sum to zero. This means items "
                    f"belonging to different subsets cannot be compared or calibrated onto a single scale.",
                    category=UserWarning,
                    stacklevel=2,
                )

            # THROW SYSTEM WARNING FOR "FAKE CONNECTIVITY" — ITEMS THAT PASS THE
            # STANDARD CHECK BUT WILL STILL BREAK CALIBRATE()'S DIRECTED MATRIX
            directionally_isolated = self.connectivity_status.get(
                "directionally_isolated_items", []
            )
            if directionally_isolated:
                warnings.warn(
                    f"\n"
                    f"⚠️  DATA INTEGRITY WARNING: {len(directionally_isolated)} item(s) have a "
                    f"structurally unresolvable zero in calibrate()'s directed pairwise matrix: "
                    f"{directionally_isolated}.\n"
                    f"This can silently corrupt calibration (NaN/overflow) even when the item(s) "
                    f"otherwise appear connected. Consider dropping these items or gathering more "
                    f"responses before calibrating.",
                    category=UserWarning,
                    stacklevel=2,
                )

    # ------------------------------------------------------------------
    # Rename utilities
    # ------------------------------------------------------------------

    def _set_facet_aliases(self, model, anchor=False):
        """Set dynamic facet-named aliases for public facet_effect/SE attributes."""
        prefix = "anchor_" if anchor else ""
        # Facet effect estimates
        for attr in [
            f"facet_effects_{model}",
            "facet_effects_bivector_items",
            "facet_effects_bivector_thresholds",
            "marginal_facet_effects_items",
            "marginal_facet_effects_thresholds",
        ]:
            if hasattr(self, f"{prefix}{attr}"):
                # Dynamic alias using actual facet name (e.g. judges_global)
                facet_alias = attr.replace("facet_effects", self.facets)
                setattr(
                    self, f"{prefix}{facet_alias}", getattr(self, f"{prefix}{attr}")
                )
                # rater_ alias for default-facet backward compatibility
                rater_alias = attr.replace("facet_effects", "raters")
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
                "New facet_element name is the same as the old facet_element name.",
                UserWarning,
                stacklevel=2,
            )
        elif new in self.facet_names:
            warnings.warn(
                "New facet_element name is a duplicate of an existing facet_element name.",
                UserWarning,
                stacklevel=2,
            )
        if old not in self.facet_names:
            warnings.warn(
                f"Old facet_element name {old!r} not found in data.",
                UserWarning,
                stacklevel=2,
            )
        elif not isinstance(new, str):
            warnings.warn("Rater names must be strings.", UserWarning, stacklevel=2)
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
                "List of new facet_element names contains duplicates.",
                UserWarning,
                stacklevel=2,
            )
        elif len(new_names) != self.no_of_facet_elements:
            warnings.warn(
                f"Incorrect number of facet_element names: {len(new_names)} provided, "
                f"{self.no_of_facet_elements} facet_elements in data.",
                UserWarning,
                stacklevel=2,
            )
        elif not all(isinstance(n, str) for n in new_names):
            warnings.warn("Rater names must be strings.", UserWarning, stacklevel=2)
        else:
            df_dict = {
                new: self.responses.xs(old)
                for old, new in zip(self.facet_names, new_names)
            }
            self.responses = pd.concat(df_dict.values(), keys=df_dict.keys())
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
            Rater identifier.
        facet_effects : Series or dict
            Rater facet_effect parameters. Structure depends on model:
            global — Series indexed by facet_element;
            items  — dict of Series {facet_element: Series(items)};
            thresholds — dict of arrays {facet_element: array(thresholds)};
            matrix — nested dict {facet_element: {item: array}}.
        category : int
            Response category (0 to max_score).
        thresholds : array-like
            Rasch-Andrich threshold vector, length max_score+1, thresholds[0]=0.
        model : str, default 'global'
            Rater parameterisation: 'global', 'items', 'thresholds', or 'matrix'.

        Returns
        -------
        float
            Probability of the specified category, in [0, 1].
        """
        cats = np.arange(len(thresholds) + 1, dtype=float)
        cumtau = np.concatenate([[0.0], np.cumsum(thresholds)])
        log_nums = cats * (person_location - item_locations.loc[item]) - cumtau
        # Apply facet_element effect
        if model == "global":
            log_nums -= cats * facet_effects.loc[facet_element]
        elif model == "items":
            log_nums -= cats * facet_effects.loc[facet_element, item]
        elif model == "thresholds":
            log_nums -= np.concatenate(
                [[0.0], np.cumsum(facet_effects.loc[facet_element].values)]
            )
        elif model in ("bivector", "matrix"):
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
            Rater identifier.
        facet_effects : Series or dict
            Rater facet_effect parameters (structure depends on model).
        thresholds : array-like
            Rasch-Andrich threshold vector, length max_score+1.
        model : str, default 'global'
            Rater parameterisation.

        Returns
        -------
        float
            Expected score in [0, max_score].
        """
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
            Rater identifier.
        facet_effects : Series or dict
            Rater facet_effect parameters.
        thresholds : array-like
            Rasch-Andrich threshold vector.
        model : str, default 'global'
            Rater parameterisation.

        Returns
        -------
        float
            Item variance / Fisher information. Always non-negative.
        """
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
            Rater identifier.
        facet_effects : Series or dict
            Rater facet_effect parameters.
        thresholds : array-like
            Rasch-Andrich threshold vector.
        model : str, default 'global'
            Rater parameterisation.

        Returns
        -------
        float
            Fourth central moment of the response distribution.
        """
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
        per-threshold severity array for one (facet_element, item) pair:
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

    def _cat_probs_mfrm(
        self, person_locations, items, facet_elements, thresholds, model, facet_effects
    ):
        """
        Vectorised MFRM category probability engine.

        Returns dict {facet_element: ndarray (K+1, N, I)} and cats array (K+1,).

        The log-numerator for person n, facet_element r, item i, category k:
          global:     k*(θ_n − δ_i − λ_r) − Σ τ_k
          items:      k*(θ_n − δ_i − λ_{r,i}) − Σ τ_k
          thresholds: k*(θ_n − δ_i) − Σ(τ_k + λ_{r,k})
          matrix:     k*(θ_n − δ_i) − Σ(τ_k + λ_{r,i,k})
        """
        cats = np.arange(len(thresholds) + 1, dtype=float)  # (K+1,)
        cumtau = np.concatenate([[0.0], np.cumsum(thresholds)])  # (K+1,)
        ab = np.asarray(person_locations, dtype=float)  # (N,)
        diff_arr = self.items.loc[items].values  # (I,)
        n_items = len(items)

        result = {}
        for facet_element in facet_elements:
            if model == "global":
                # item_offset: scalar, same for all (i)
                item_offset = float(facet_effects.loc[facet_element])
                thresh_offset = np.zeros(len(thresholds) + 1)
            elif model == "items":
                # item_offset: (I,) vector
                item_offset = facet_effects.loc[facet_element, items].values
                thresh_offset = np.zeros(len(thresholds) + 1)
            elif model == "thresholds":
                item_offset = 0.0
                thresh_offset = np.concatenate(
                    [[0.0], np.cumsum(facet_effects.loc[facet_element].values)]
                )
            elif model == "bivector":
                item_offset = 0.0
                thresh_offset = None  # applied per-item below
            elif model in ("matrix", "mixed"):
                item_offset = 0.0
                thresh_offset = None  # applied per-item below
            else:
                raise ValueError(f"Unknown model: {model}")

            if model in ("bivector", "matrix", "mixed"):
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

        matrix = np.array(
            [
                [
                    sum(
                        np.count_nonzero(data[i, r, :] == data[j, r, :] + 1)
                        for r in range(self.no_of_facet_elements)
                    )
                    for j in range(self.no_of_items)
                ]
                for i in range(self.no_of_items)
            ],
            dtype=np.float64,
        )
        return matrix, np.array(self.item_names)

    def item_diffs(
        self, constant=0.1, method="cos", matrix_power=3, log_lik_tol=0.000001
    ):
        """PAIR item item_location estimation summing across facet_elements."""
        matrix, _ = self._build_pairwise_matrix()

        constant_matrix = ((matrix + matrix.T) > 0).astype(np.float64) * constant
        matrix += constant_matrix
        np.fill_diagonal(matrix, matrix.diagonal() + constant)

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
        CPAT threshold distance estimate for MFRM — sums counts across facet_elements.
        Vectorised via indicator matrix multiplication.
        """
        data = (
            self.responses.values.reshape(
                self.no_of_facet_elements, self.no_of_persons, -1
            )
            .swapaxes(1, 2)
            .transpose((1, 0, 2))
        )  # (I, R, P)

        # Sum count matrices across facet_elements
        num_matrix = np.zeros((self.no_of_items, self.no_of_items))
        den_matrix = np.zeros((self.no_of_items, self.no_of_items))
        for r in range(self.no_of_facet_elements):
            at_k = (data[:, r, :] == threshold).astype(np.float64)
            at_km1 = (data[:, r, :] == threshold - 1).astype(np.float64)
            at_kp1 = (data[:, r, :] == threshold + 1).astype(np.float64)
            num_matrix += at_k @ at_k.T
            den_matrix += at_km1 @ at_kp1.T

        valid = (num_matrix + den_matrix) > 0
        num_s = np.where(valid, num_matrix + constant, 0.0)
        den_s = np.where(valid, den_matrix + constant, 0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            weight_matrix = np.where(valid, 2.0 * num_s * den_s / (num_s + den_s), 0.0)

        diffs = item_locations.values
        diff_matrix = diffs[:, None] - diffs[None, :]

        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.where(valid, np.log(num_s) - np.log(den_s), 0.0)

        total_weight = weight_matrix.sum()
        if total_weight == 0:
            return np.nan
        return (weight_matrix * (log_ratio + diff_matrix)).sum() / total_weight

    def ra_thresholds(self, item_locations, constant=0.1):
        """CPAT threshold set estimation."""
        distances = [
            self._threshold_distance(k, item_locations, constant)
            for k in range(1, self.max_score)
        ]
        thresholds = np.array([sum(distances[:t]) for t in range(self.max_score)])
        thresholds -= thresholds.mean()
        return thresholds

    # ------------------------------------------------------------------
    # Rater facet_effect estimation
    # ------------------------------------------------------------------

    def _pair_matrix(self, data_2d, constant):
        """Build a PAIR pairwise matrix from (R, P) data and apply smoothing."""
        R = data_2d.shape[0]
        matrix = np.array(
            [
                [
                    np.count_nonzero(data_2d[r1, :] == data_2d[r2, :] + 1)
                    for r2 in range(R)
                ]
                for r1 in range(R)
            ],
            dtype=np.float64,
        )
        constant_matrix = ((matrix + matrix.T) > 0).astype(np.float64) * constant
        matrix += constant_matrix
        np.fill_diagonal(matrix, matrix.diagonal() + constant)
        return matrix

    def _raise_matrix_power(self, matrix, matrix_power, constant):
        """
        Raise a matrix to a given power, incrementing until no zeros remain.

        Used internally during PAIR calibration to ensure full connectivity
        in the facet comparison matrix. If zeros persist after matrix_power + 5
        iterations, adds a smoothing constant and stops.

        Parameters
        ----------
        matrix : numpy.ndarray
            Square comparison count matrix.
        matrix_power : int
            Starting matrix power.
        constant : float
            Smoothing constant added if zeros persist.

        Returns
        -------
        numpy.ndarray
            Powered matrix with zeros resolved or smoothed.
        """
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
        self, constant=0.1, method="cos", matrix_power=3, log_lik_tol=0.000001
    ):
        """PAIR facet_element effect estimation — scalar per facet_element."""
        data = (
            self.responses.values.reshape(
                self.no_of_facet_elements, self.no_of_persons, -1
            )
            .swapaxes(1, 2)
            .transpose((1, 0, 2))
        )  # (I, R, P)

        matrix = np.array(
            [
                [
                    sum(
                        np.count_nonzero(data[item, r1, :] == data[item, r2, :] + 1)
                        for item in range(self.no_of_items)
                    )
                    for r2 in range(self.no_of_facet_elements)
                ]
                for r1 in range(self.no_of_facet_elements)
            ],
            dtype=np.float64,
        )
        constant_matrix = ((matrix + matrix.T) > 0).astype(np.float64) * constant
        matrix += constant_matrix
        np.fill_diagonal(matrix, matrix.diagonal() + constant)

        mat = self._raise_matrix_power(matrix, matrix_power, constant)
        self.facet_effects_global = self.priority_vector(
            mat, method=method, log_lik_tol=log_lik_tol, raters=True
        )

    def _item_rater_element(
        self, item, constant=0.1, method="cos", matrix_power=3, log_lik_tol=0.000001
    ):
        """PAIR facet_element effect for a single item (items parameterisation)."""
        data = (
            self.responses.values.reshape(
                self.no_of_facet_elements, self.no_of_persons, -1
            )
            .swapaxes(1, 2)
            .transpose((1, 0, 2))
        )  # (I, R, P)
        matrix = self._pair_matrix(data[item, :, :], constant)
        mat = self._raise_matrix_power(matrix, matrix_power, constant)
        return self.priority_vector(
            mat, method=method, log_lik_tol=log_lik_tol, raters=True
        )

    def _estimate_raters_items(
        self, constant=0.1, method="cos", matrix_power=3, log_lik_tol=0.000001
    ):
        """PAIR facet_element effect estimation — vector per (facet_element, item)."""
        facet_elements = np.zeros((self.no_of_facet_elements, self.no_of_items))
        for i in range(self.no_of_items):
            facet_elements[:, i] = self._item_rater_element(
                i,
                constant=constant,
                method=method,
                matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
            )
        self.facet_effects_items = pd.DataFrame(
            facet_elements, index=self.facet_names, columns=self.responses.columns
        )

    def _threshold_rater_element(
        self, category, constant=0.1, method="cos", matrix_power=3, log_lik_tol=0.000001
    ):
        """PAIR facet_element effect for a single threshold (thresholds parameterisation)."""
        data = (
            self.responses.values.reshape(
                self.no_of_facet_elements, self.no_of_persons, -1
            )
            .swapaxes(1, 2)
            .transpose((1, 0, 2))
        )  # (I, R, P)

        # Sum across items: count(X_{i,r1}==k+1 AND X_{i,r2}==k)
        matrix = np.zeros((self.no_of_facet_elements, self.no_of_facet_elements))
        for i in range(self.no_of_items):
            at_k = (data[i, :, :] == category + 1).astype(np.float64)  # (R, P)
            at_km1 = (data[i, :, :] == category).astype(np.float64)
            matrix += at_k @ at_km1.T

        matrix = matrix.astype(np.float64)
        constant_matrix = ((matrix + matrix.T) > 0).astype(np.float64) * constant
        matrix += constant_matrix
        np.fill_diagonal(matrix, matrix.diagonal() + constant)

        mat = self._raise_matrix_power(matrix, matrix_power, constant)
        return self.priority_vector(
            mat, method=method, log_lik_tol=log_lik_tol, raters=True
        )

    def _estimate_raters_thresholds(
        self, constant=0.1, method="cos", matrix_power=3, log_lik_tol=0.000001
    ):
        """PAIR facet_element effect estimation — vector per (facet_element, threshold)."""
        facet_elements = np.zeros((self.no_of_facet_elements, self.max_score))
        for k in range(self.max_score):
            facet_elements[:, k] = self._threshold_rater_element(
                k,
                constant=constant,
                method=method,
                matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
            )
        self.facet_effects_thresholds = pd.DataFrame(
            facet_elements, index=self.facet_names, columns=range(1, self.max_score + 1)
        )

    def _matrix_rater_element(
        self,
        item,
        category,
        constant=0.1,
        method="cos",
        matrix_power=3,
        log_lik_tol=0.000001,
    ):
        """PAIR facet_element effect for a single (item, category) cell (matrix param)."""
        data = (
            self.responses.values.reshape(
                self.no_of_facet_elements, self.no_of_persons, -1
            )
            .swapaxes(1, 2)
            .transpose((1, 0, 2))
        )  # (I, R, P)

        at_k = (data[item, :, :] == category + 1).astype(np.float64)  # (R, P)
        at_km1 = (data[item, :, :] == category).astype(np.float64)
        matrix = at_k @ at_km1.T

        matrix = matrix.astype(np.float64)
        constant_matrix = ((matrix + matrix.T) > 0).astype(np.float64) * constant
        matrix += constant_matrix
        np.fill_diagonal(matrix, matrix.diagonal() + constant)

        mat = self._raise_matrix_power(matrix, matrix_power, constant)
        return self.priority_vector(
            mat, method=method, log_lik_tol=log_lik_tol, raters=True
        )

    def _estimate_raters_matrix(
        self, constant=0.1, method="cos", matrix_power=3, log_lik_tol=0.000001
    ):
        """PAIR facet_element effect estimation — full (facet_element, item, threshold) matrix."""
        facet_elements = np.zeros(
            (self.no_of_facet_elements, self.no_of_items, self.max_score)
        )
        for i in range(self.no_of_items):
            for k in range(self.max_score):
                facet_elements[:, i, k] = self._matrix_rater_element(
                    i,
                    k,
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
        self.marginal_facet_effects_items = pd.DataFrame(
            sev_arr.mean(axis=2), index=self.facet_names, columns=self.responses.columns
        )
        self.marginal_facet_effects_thresholds = pd.DataFrame(
            sev_arr.mean(axis=1), index=self.facet_names, columns=range(1, self.max_score + 1)
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
        rows = []
        for facet_element in self.facet_names:
            for item in self.item_names:
                row = np.array(
                    [
                        self.facet_effects_bivector_items.loc[facet_element, item]
                        + self.facet_effects_bivector_thresholds.loc[facet_element, k]
                        for k in range(1, self.max_score + 1)
                    ]
                )
                rows.append(row)
        self.facet_effects_bivector = pd.DataFrame(rows, index=mi, columns=range(1, self.max_score + 1))

    # ------------------------------------------------------------------
    # Calibration — top-level methods
    # ------------------------------------------------------------------

    def calibrate(
        self,
        model="global",
        constant=0.1,
        method="cos",
        matrix_power=3,
        log_lik_tol=0.000001,
        matrix_marginals=False,
    ):
        """
        Calibrate the MFRM for the specified facet_element parameterisation.

        Three-stage sequential estimation:
          1. item_diffs()       — PAIR item item_locations (shared across models)
          2. ra_thresholds()    — CPAT shared thresholds (shared across models)
          3. raters_{model}()   — PAIR facet_element effects (model-specific)

        Parameters
        ----------
        model : one of 'global', 'items', 'thresholds', 'matrix', 'bivector'
        matrix_marginals : bool, default False
            Bivector model only. If True (default), estimate item and threshold
            vectors as marginal means of the full matrix PAIR estimates. If
            False, estimate each vector directly using its own pooled PAIR
            (items PAIR summed across thresholds; thresholds PAIR summed across
            items, corrected for per-facet_element mean item effect).
        """
        if model not in self._MODELS:
            raise ValueError(f"model must be one of {self._MODELS}")

        if constant == 0:
            all_max_items = [
                item
                for item in self.item_names
                if (
                    self.responses.xs(item, level=-1, axis=1)
                    .dropna(how="all")
                    .eq(self.max_score)
                    .all(axis=None)
                )
            ]
            if all_max_items:
                warnings.warn(
                    f"Items with all-maximum scores detected with constant=0: "
                    f"{all_max_items}. Item estimation will fail. "
                    f"Either drop these items or use a non-zero constant.",
                    UserWarning,
                    stacklevel=2,
                )

        if len(self.facet_names) == 1:
            warnings.warn(
                "Only one facet_element detected. MFRM with a single facet_element reduces to RSM. "
                "Consider using RSM instead.",
                UserWarning,
                stacklevel=2,
            )

        if len(self.item_names) == 1:
            warnings.warn(
                "Only one item detected. MFRM with a single item reduces to RSM "
                "with facet_elements as items. Consider reconfiguring and using RSM instead.",
                UserWarning,
                stacklevel=2,
            )

        self._remove_null_persons()
        self.item_diffs(
            constant=constant,
            method=method,
            matrix_power=matrix_power,
            log_lik_tol=log_lik_tol,
        )
        self.thresholds = pd.Series(
            self.ra_thresholds(self.items, constant=constant),
            index=range(1, self.max_score + 1),
        )
        kw = dict(constant=constant, method=method,
                  matrix_power=matrix_power, log_lik_tol=log_lik_tol)
        if model == "bivector":
            kw["matrix_marginals"] = matrix_marginals
        getattr(self, f"_estimate_raters_{model}")(**kw)
        self._set_facet_aliases(model)

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

    # ------------------------------------------------------------------
    # Anchor calibration
    # ------------------------------------------------------------------

    def calibrate_anchor(
        self,
        model,
        anchors,
        calibrate=False,
        constant=0.1,
        method="cos",
        matrix_power=3,
        log_lik_tol=0.000001,
        adj=None,
    ):
        """
        Anchor calibration: set mean facet_effect of anchors to zero
        and adjust item item_locations and thresholds accordingly.

        adj : pre-computed anchor adjustment from _extract_anchor_adj().
            If provided, used as a fixed constant instead of re-estimating
            from self. Pass this in bootstrap loops to avoid inflating SEs
            with anchor rater sampling variance.
        """
        if calibrate:
            self.calibrate(
                model=model,
                constant=constant,
                method=method,
                matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
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
        constant=0.1,
        method="cos",
        matrix_power=3,
        log_lik_tol=0.000001,
        seed=None,
    ):
        """
        Test whether the raters proposed as `anchors` actually agree with
        each other, before calibrate_anchor() recentres their group mean
        to zero.

        calibrate_anchor() only ever imposes mean(anchor facet_effects) == 0 —
        it has no way to notice whether that mean is a stable, shared
        reference point or an artefact of averaging over raters who don't
        behave alike. If one or more anchor raters have a genuinely
        different facet_effect from the rest, the anchor-implied "zero" is
        unstable and the whole anchored scale inherits that instability.
        This is unrelated to per_rater_model_selection (which asks how
        COMPLEX a rater's own facet_effect structure needs to be) — this asks
        whether the specific set of raters chosen to DEFINE the anchor
        actually agree with each other. A separate, opt-in diagnostic
        (not run automatically by calibrate_anchor) since the 'cochran'
        test needs bootstrap SEs, which aren't otherwise computed by
        default and would add unexpected cost to every calibrate_anchor
        call.

        Uses each rater's own free (unanchored) facet_effects_{model} —
        deliberately not anchor_facet_effects_{model} — since the point is
        to check whether this candidate anchor SET is internally
        consistent before it gets used to define anything, using an
        unanchored bootstrap so the test isn't circularly biased toward
        finding homogeneity by construction.

        Parameters
        ----------
        model : str
            One of 'global', 'items', 'thresholds', 'matrix', 'bivector'.
        anchors : list
            Rater identifiers to test — the same set you'd pass to
            calibrate_anchor(model, anchors). At least 2 required.
        rater_homogeneity_test : {'cochran', 'robust'}, default 'cochran'
            'cochran': Cochran's Q heterogeneity test (Cochran, 1954,
            "The combination of estimates from different experiments,"
            Biometrics 10:101-129 — the meta-analysis heterogeneity Q, not
            Cochran's 1950 test for k related binary samples).
            Q = sum(w_r*(facet_effect_r - weighted_mean)^2),
            weights w_r=1/SE_r^2 from bootstrap SEs, chi-squared with
            df=n_anchors-1, evaluated on the anchor set exactly as
            supplied. Which specific rater(s) are responsible is then
            identified by sequential exclusion (standard meta-analysis
            outlier diagnostic): repeatedly drop the rater with the
            largest |z| against the CURRENT remaining group's own
            weighted mean and retest, until Q is no longer significant. A
            single-pass z-test against the full group's mean isn't used
            for per-rater flagging because one severely deviant rater
            drags that mean far enough that every OTHER rater also looks
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
            For items/thresholds/matrix/bivector models (facet_effect is a
            vector or surface, not one number per rater), also test
            homogeneity component-by-component (per item, per threshold,
            or per (item, threshold) cell for matrix/bivector) in addition
            to the scalar (whole-profile-mean) test. 'auto' turns this on
            for every model except 'global' (nothing to break out there).
            Can produce many rows for matrix/bivector
            (no_of_items * max_score).
        alpha : float, default 0.05
            Significance level for all Cochran's Q tests here: the initial
            omnibus test, each step of the sequential-exclusion procedure,
            and the per-component tests (rater_homogeneity_test='cochran'
            only).
        correction : {'bh', 'bonferroni', None}, default 'bh'
            Multiple-comparison correction across the per-component tests
            (if run) — genuinely independent tests, one per component, so
            correcting across them is appropriate. Not used for per-rater
            flagging: sequential exclusion (above) already decides that
            via repeated omnibus tests, not a family of independent
            per-rater p-values, so a correction doesn't apply there.
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
            supplied (N_dropped_for_homogeneity is how many raters the
            sequential-exclusion procedure needed to remove before Q
            stopped being significant). None if
            rater_homogeneity_test='robust' (no omnibus statistic exists
            for that method).
        anchor_homogeneity_per_rater : pandas.DataFrame
            One row per anchor rater. rater_homogeneity_test='cochran':
            Facet effect, SE, z, p, Flagged — Flagged is whether
            sequential exclusion removed this rater (z/p are relative to
            whichever group they were tested against: their own removal
            step if dropped, or the final retained group if not).
            rater_homogeneity_test='robust': Facet effect, Robust z, Flagged.
        anchor_homogeneity_per_component : pandas.DataFrame or None
            One row per component (item name / threshold number / (item,
            threshold) tuple) with that component's own omnibus result —
            Q, df, p, p_corrected, Flagged (cochran), or Flagged only
            (robust, meaning at least one rater exceeded the threshold on
            that component). None if per_component resolves to False.
        """
        if model not in self._MODELS:
            raise ValueError(f"model must be one of {self._MODELS}")
        if rater_homogeneity_test not in ("cochran", "robust"):
            raise ValueError("rater_homogeneity_test must be 'cochran' or 'robust'")
        if correction not in ("bh", "bonferroni", None):
            raise ValueError("correction must be 'bh', 'bonferroni', or None")
        if len(anchors) < 2:
            raise ValueError(
                "anchors must contain at least 2 raters to test homogeneity."
            )
        if per_component == "auto":
            per_component = model != "global"

        if not hasattr(self, f"facet_effects_{model}"):
            self.calibrate(
                model=model,
                constant=constant,
                method=method,
                matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
            )

        sev_attr = f"facet_effects_{model}"
        sev0 = getattr(self, sev_attr)

        def _get_component(sev, rater, component):
            if model == "global":
                return sev.loc[rater]
            if model in ("items", "thresholds"):
                return sev.loc[rater].mean() if component is None else sev.loc[rater, component]
            # matrix / bivector: MultiIndex (facet_element, item) rows, columns=thresholds
            if component is None:
                return sev.loc[rater].values.mean()
            item, k = component
            return sev.loc[(rater, item), k]

        if model == "items" or model == "thresholds":
            components = list(sev0.columns)
        elif model in ("matrix", "bivector"):
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
                    method=method,
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
            """Iteratively drop the rater with the largest |z| against the
            CURRENT group's weighted mean, recomputing Q each time, until Q
            is no longer significant or min_group raters remain.

            A single-pass Cochran's Q against the full group's weighted mean
            is not robust to one severely deviant rater: that rater alone
            drags the weighted mean far enough that every OTHER rater also
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
                f"Anchor rater(s) {offenders} do not appear homogeneous with "
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

    def _calibrate_anchor_global(self, anchors, adj=None):
        """Anchor calibration for global parameterisation. Shifts all facet effects so anchor mean is zero."""
        self.anchor_items_global = self.items.copy()
        self.anchor_thresholds_global = self.thresholds.copy()
        self.anchor_facet_effects_global = self.facet_effects_global.copy()

        if adj is None:
            adj = float(self.facet_effects_global.loc[anchors].mean())
        self.anchor_facet_effects_global -= adj

    def _calibrate_anchor_items(self, anchors, adj=None):
        """Anchor calibration for items parameterisation. Adjusts per-item facet effects and absorbs mean into item item_locations."""
        self.anchor_items_items = self.items.copy()
        self.anchor_thresholds_items = self.thresholds.copy()

        sev_df = self.facet_effects_items.copy()  # already (R, I) DataFrame
        if adj is None:
            adj = sev_df.loc[anchors].mean(axis=0)

        self.anchor_items_items += adj
        sev_df -= adj
        self.anchor_facet_effects_items = sev_df
        self.anchor_items_items -= self.anchor_items_items.mean()

    def _calibrate_anchor_thresholds(self, anchors, adj=None):
        """Anchor calibration for thresholds parameterisation. Adjusts per-threshold facet effects and absorbs mean into thresholds."""
        self.anchor_items_thresholds = self.items.copy()
        self.anchor_thresholds_thresholds = self.thresholds.copy()

        sev_df = self.facet_effects_thresholds.copy()  # already (R, K+1) DataFrame
        if adj is None:
            adj = sev_df.loc[anchors].mean(axis=0)

        self.anchor_thresholds_thresholds += adj.values
        sev_df -= adj
        self.anchor_facet_effects_thresholds = sev_df
        self.anchor_thresholds_thresholds -= self.anchor_thresholds_thresholds.mean()

    def _calibrate_anchor_matrix(self, anchors, adj=None):
        """
        Anchor calibration for matrix parameterisation.
        Subtracts the mean anchor facet_element effect (per item, per threshold)
        from all facet_elements, and absorbs it into item item_locations and thresholds.
        """
        self.anchor_items_matrix = self.items.copy()
        self.anchor_thresholds_matrix = self.thresholds.copy()

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
        self.anchor_items_bivector = self.items.copy()
        self.anchor_thresholds_bivector = self.thresholds.copy()

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
        samples = []
        for pick in picks:
            sample_dict = {
                facet_element: pd.DataFrame(
                    [data_dict[facet_element].loc[p] for p in pick]
                ).reset_index(drop=True)
                for facet_element in self.facet_names
            }
            samples.append(pd.concat(sample_dict.values(), keys=sample_dict.keys()))
        return [MFRM(s, self.max_score, validate=False) for s in samples]

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
        constant=0.1,
        method="cos",
        matrix_power=3,
        log_lik_tol=0.000001,
        store_bootstrap=False,
        seed=None,
    ):
        """
        Bootstrap standard errors for item item_locations, thresholds, and
        facet_element effects for the specified model.

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
        """
        if model == "mixed":
            self._sync_mixed(
                anchors,
                constant=constant,
                method=method,
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

        samples = self._bootstrap_samples(no_of_samples, seed=seed)
        for s in samples:
            if model == "mixed":
                s.calibrate(
                    model="matrix",
                    constant=constant,
                    method=method,
                    matrix_power=matrix_power,
                    log_lik_tol=log_lik_tol,
                )
                if anchors is not None:
                    s.calibrate_anchor(
                        "matrix",
                        anchors,
                        constant=constant,
                        method=method,
                        matrix_power=matrix_power,
                        log_lik_tol=log_lik_tol,
                        adj=adj_fixed,
                    )
                    s.anchor_facet_effects_mixed = s._apply_rater_models(
                        s.anchor_facet_effects_matrix, rater_models_for_bootstrap
                    )
                else:
                    s.facet_effects_mixed = s._apply_rater_models(
                        s.facet_effects_matrix, rater_models_for_bootstrap
                    )
                continue
            s.calibrate(
                model=model,
                constant=constant,
                method=method,
                matrix_power=matrix_power,
                log_lik_tol=log_lik_tol,
            )
            if anchors is not None:
                s.calibrate_anchor(
                    model,
                    anchors,
                    constant=constant,
                    method=method,
                    matrix_power=matrix_power,
                    log_lik_tol=log_lik_tol,
                    adj=adj_fixed,
                )

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
        setattr(self, f"{prefix}item_se", pd.Series(item_se, index=self.responses.columns))
        if item_lo is not None:
            setattr(self, f"{prefix}item_low", pd.Series(item_lo, index=self.responses.columns))
            setattr(self, f"{prefix}item_high", pd.Series(item_hi, index=self.responses.columns))

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

        elif model == "items":
            sev_attr = "anchor_facet_effects_items" if anchor else "facet_effects_items"
            # Each sample's facet_effects_items is now a (R, I) DataFrame
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

        elif model == "thresholds":
            sev_attr = (
                "anchor_facet_effects_thresholds"
                if anchor
                else "facet_effects_thresholds"
            )
            # Each sample's facet_effects_thresholds is now a (R, K+1) DataFrame
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

    def std_errors_global_anchor(self, anchors, **kw):
        """Alias for std_errors(model=\'global\', anchors=anchors). See std_errors for full documentation."""
        self.std_errors(model="global", anchors=anchors, **kw)

    def anchor_std_errors(
        self,
        model="global",
        anchors=None,
        interval=None,
        no_of_samples=500,
        constant=0.1,
        method="cos",
        matrix_power=3,
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
            Raters whose mean facet_effect is anchored to zero. If None,
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
        if interval is None:
            interval = getattr(self, f"_bootstrap_interval_{model}", None)

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
            for s in samples:
                s.calibrate_anchor(
                    cal_model,
                    anchor_raters_used,
                    constant=constant,
                    method=method,
                    matrix_power=matrix_power,
                    log_lik_tol=log_lik_tol,
                    adj=adj_fixed,
                )
                if model == "mixed":
                    s.anchor_facet_effects_mixed = s._apply_rater_models(
                        s.anchor_facet_effects_matrix, self.anchor_rater_models
                    )
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
            samples = self._bootstrap_samples(no_of_samples, seed=seed)
            for s in samples:
                s.calibrate(
                    model=cal_model,
                    constant=constant,
                    method=method,
                    matrix_power=matrix_power,
                    log_lik_tol=log_lik_tol,
                )
                s.calibrate_anchor(
                    cal_model,
                    anchors,
                    constant=constant,
                    method=method,
                    matrix_power=matrix_power,
                    log_lik_tol=log_lik_tol,
                    adj=adj_fixed,
                )
                if model == "mixed":
                    s.anchor_facet_effects_mixed = s._apply_rater_models(
                        s.anchor_facet_effects_matrix, self.anchor_rater_models
                    )

        # Item item_location SEs — from anchor_items_{cal_model} ('matrix' for 'mixed',
        # since item locations aren't model-specific and 'mixed' calibrates via matrix)
        item_ests = np.array(
            [getattr(s, f"anchor_items_{cal_model}").values for s in samples]
        )
        item_se, item_lo, item_hi = self._se_from_bootstrap(
            item_ests, self.responses.columns, interval
        )
        self.anchor_item_se = pd.Series(item_se, index=self.responses.columns)
        if item_lo is not None:
            self.anchor_item_low = pd.Series(item_lo, index=self.responses.columns)
            self.anchor_item_high = pd.Series(item_hi, index=self.responses.columns)

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

    def category_probability_dict(
        self,
        model="global",
        anchor=False,
        warm_corr=True,
        ext_scores=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        method="cos",
        constant=0.1,
        matrix_power=3,
        log_lik_tol=0.000001,
    ):
        """Build the (Rater, Person) × Items category probability DataFrames."""
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
        Newton-Raphson ML person_location estimation with optional Warm correction.

        The key difference between models is how the log-numerator is constructed
        per facet_element — handled entirely by _cat_probs_mfrm() so the NR loop is
        identical across all four parameterisations.
        """
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
                valid = estimates.notna()
                if valid.any():
                    valid_idx = estimates.index[valid]
                    valid_pf = nr_filter.loc[
                        pd.IndexSlice[facet_elements, valid_idx], :
                    ]
                    estimates[valid] += self.warm(
                        estimates[valid],
                        items,
                        facet_elements,
                        facet_effects,
                        thresholds,
                        valid_pf,
                        model,
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
            Rater subset to use.
        facet_effects : Series or dict
            Rater facet_effect parameters (structure depends on model).
        thresholds : array-like
            Rasch-Andrich threshold vector.
        person_filter : pandas.DataFrame
            Binary mask (1.0 = responded, NaN = missing), with (Rater, Person)
            MultiIndex and items as columns.
        model : str, default 'global'
            Rater parameterisation.

        Returns
        -------
        pandas.Series
            Warm bias correction terms indexed by person, to add to ML estimates.
        """
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
            person_locations, items, facet_elements, facet_effects, self.thresholds, pf, "global"
        )

    def warm_items(self, person_locations, items, facet_elements, facet_effects, pf, **kw):
        """Alias for warm(..., model='items'). See warm for full documentation."""
        return self.warm(
            person_locations, items, facet_elements, facet_effects, self.thresholds, pf, "items"
        )

    def warm_thresholds(self, person_locations, items, facet_elements, facet_effects, pf, **kw):
        """Alias for warm(..., model='thresholds'). See warm for full documentation."""
        thr = kw.get("thresholds", self.thresholds)
        return self.warm(
            person_locations, items, facet_elements, facet_effects, thr, pf, "thresholds"
        )

    def warm_matrix(self, person_locations, items, facet_elements, facet_effects, pf, **kw):
        """Alias for warm(..., model='matrix'). See warm for full documentation."""
        return self.warm(
            person_locations, items, facet_elements, facet_effects, self.thresholds, pf, "matrix"
        )

    def warm_bivector(self, person_locations, items, facet_elements, facet_effects, pf, **kw):
        """Alias for warm(..., model='bivector'). See warm for full documentation."""
        return self.warm(
            person_locations,
            items,
            facet_elements,
            facet_effects,
            self.thresholds,
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
            Rater parameterisation.
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
            Rater subset. None uses all facet_elements.

        Returns
        -------
        pandas.Series
            CSEM values indexed by person, in logits.
        """
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
            Rater parameterisation.
        anchor : bool, default False
            If True, uses anchor-calibrated parameters.
        items : list or None, default None
            Item subset. None uses all items.
        facet_elements : list or None, default None
            Rater subset. None uses all facet_elements.
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
            # Build a minimal single-person MultiIndex person_filter for warm()
            pf_mi = pd.DataFrame(
                1.0,
                index=pd.MultiIndex.from_product(
                    [facet_elements, ["_score_lookup_person_"]],
                    names=self.responses.index.names,
                ),
                columns=items,
            )
            estimate += float(
                self.warm(
                    pd.Series({"_score_lookup_person_": estimate}),
                    items,
                    facet_elements,
                    facet_effects,
                    thresholds,
                    pf_mi,
                    model,
                ).iloc[0]
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
    ):
        """
        Build a score-to-person_location lookup table for all possible raw scores.

        Estimates the person_location corresponding to every possible raw score across
        the specified facet_element-item combination using Newton-Raphson, and stores
        the result as self.score_table.

        Parameters
        ----------
        model : str, default 'global'
            Rater parameterisation.
        anchor : bool, default False
            If True, uses anchor-calibrated parameters.
        attribute : bool, default True
            If True, stores result as self.score_table.
        items : list or None, default None
            Item subset. None uses all items.
        facet_elements : list or None, default None
            Rater subset. None uses all facet_elements.
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

        Attributes set (if attribute=True)
        -----------------------------------
        score_table_{model} : pandas.Series
            Person location estimate for each possible raw score, indexed by score.
        """
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
        else:
            return table

    # Backwards-compatible aliases
    def score_lookup_table_global(self, **kw):
        """Alias for score_lookup_table(model='global'). See score_lookup_table for full documentation."""
        self.score_lookup_table(model="global", **kw)

    def score_lookup_table_items(self, **kw):
        """Alias for score_lookup_table(model='items'). See score_lookup_table for full documentation."""
        self.score_lookup_table(model="items", **kw)

    def score_lookup_table_thresholds(self, **kw):
        """Alias for score_lookup_table(model='thresholds'). See score_lookup_table for full documentation."""
        self.score_lookup_table(model="thresholds", **kw)

    def score_lookup_table_matrix(self, **kw):
        """Alias for score_lookup_table(model='matrix'). See score_lookup_table for full documentation."""
        self.score_lookup_table(model="matrix", **kw)

    def score_lookup_table_bivector(self, **kw):
        """Alias for score_lookup_table(model='bivector'). See score_lookup_table for full documentation."""
        self.score_lookup_table(model="bivector", **kw)

    # ------------------------------------------------------------------
    # Category counts
    # ------------------------------------------------------------------

    def category_counts_df(
        self, persons=None, items=None, facet_elements=None, counts_name=None
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

        return df

    # ------------------------------------------------------------------
    # Fit matrices (shared engine)
    # ------------------------------------------------------------------

    def fit_matrices(self, cat_prob_dict):
        """
        Compute expected scores, info, kurtosis, residuals from cat_prob_dict.
        cat_prob_dict: {cat: (Rater×Person, Items) DataFrame}
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
            if k in ("constant", "method", "matrix_power", "log_lik_tol")
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
                elif model == "items":
                    row = item_locations + thr_loc + facet_effects.loc[facet_element]
                elif model == "thresholds":
                    row = item_locations + thr_loc + facet_effects.loc[facet_element, t + 1]
                elif model in ("bivector", "matrix", "mixed"):
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

    def facet_pivot(self, df):
        """Pivot (Rater×Person, Items) DataFrame to (Person×Items, Raters)."""
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

    def test_fit_statistics(self, person_locations, rsems, item_se=None):
        """Shared test-level separation and reliability statistics."""
        if item_se is None:
            item_se = self.item_se

        scores = self.responses.unstack(level=0).sum(axis=1)
        max_scores = self.responses.unstack(level=0).count(axis=1) * self.max_score
        person_locations = person_locations[(scores > 0) & (scores < max_scores)]

        isi = (self.items.var() / (item_se**2).mean() - 1) ** 0.5
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
        if not hasattr(self, "item_se"):
            self.std_errors(model=model, **kw)
        person_locations = getattr(self, f"persons_{model}")
        rsems = getattr(self, f"rsem_vector_{model}")
        results = self.test_fit_statistics(person_locations, rsems)
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

    def _log_likelihood(self, model="global", responses=None, anchor=False, persons=None):
        if responses is None:
            responses = self.responses
        _, thresholds, facet_effects = self._get_params(model, anchor=anchor)
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
        method="cos",
        constant=0.1,
        matrix_power=3,
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
            Rater parameterisation: 'global', 'items', 'thresholds', or 'matrix'.
        anchors : list or None, default None
            Rater identifiers to treat as anchors for SE computation.
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
        method : str, default 'cos'
            Priority vector extraction method for calibration.
        constant : float, default 0.1
            Additive smoothing constant for calibration.
        matrix_power : int, default 3
            Matrix power for calibration.
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
            Rater infit and outfit mean-square.
        person_infit_ms_{model}, person_outfit_ms_{model} : Series
            Person infit and outfit mean-square.
        csem_vector_{model}, rsem_vector_{model} : Series
            Conditional and residual SEM per person.
        isi_{model}, item_strata_{model}, item_reliability_{model} : float
            Item separation index, strata, and reliability (if test_stats).
        psi_{model}, person_strata_{model}, person_reliability_{model} : float
            Person separation index, strata, and reliability (if test_stats).
        """
        if model == "mixed":
            self._sync_mixed(
                anchors,
                constant=constant,
                method=method,
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
                method=method,
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
                method=method,
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

    def andersen_lr_test(
        self,
        model="global",
        split_by="person_location",
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        constant=0.1,
        method="cos",
        matrix_power=3,
        log_lik_tol=0.000001,
    ):
        """
        Andersen (1973) likelihood ratio test of parameter invariance.

        DISABLED as of 2026-07-06 — see NotImplementedError raised below.

        Splits persons into low and high groups by median person_location or total raw
        score, fits the model separately in each group, and tests whether item
        and facet parameters are invariant across groups.

        Parameters
        ----------
        model : str, default 'global'
            Rater parameterisation: 'global', 'items', 'thresholds', 'matrix', or 'bivector'.
        split_by : str, default 'person_location'
            Split criterion: 'person_location' (ML person estimates) or 'score' (total raw scores
            summed across all raters and items).
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
        raise NotImplementedError(
            "MFRM.andersen_lr_test() is disabled. A 2026-07-06 simulation study "
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
            "For rater-parameterisation comparisons (not group-invariance "
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

    def model_selection(
        self,
        test="AIC",
        aic_sig_test=True,
        alpha=0.05,
        sampling="dynamic",
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        constant=0.1,
        method="cos",
        matrix_power=3,
        log_lik_tol=0.000001,
        seed=None,
    ):
        """
        Compare the five MFRM rater parameterisations using AIC, BIC, or LR.

        Calibrates all five models (global, items, thresholds, bivector, matrix)
        if not already done, computes the log-likelihood for each, and ranks
        them by the chosen criterion.

        Parameters
        ----------
        test : str, default 'AIC'
            Criterion: 'AIC', 'BIC', or 'LR'. For 'LR', nested pairwise tests
            are computed (global vs each alternative, bivector vs matrix).
        aic_sig_test : bool, default True
            When True and test='AIC': applies a significance test with global
            as the null. p = e^(-|Δ|/2) where Δ = AIC_global - AIC_best.
            A more complex model is preferred only if p < alpha; otherwise global
            is retained.
        alpha : float, default 0.05
            Significance level for the AIC relative-likelihood test
            (used only when test='AIC' and aic_sig_test=True).
        sampling : None, 'dynamic', or int, default 'dynamic'
            Subsamples non-extreme persons before LL computation (parameters
            estimated on full data; only the LL evaluation is subsampled).
            'dynamic' uses T = min(20*(I-1)*(R-1), 1500); an integer fixes T
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
        model_comparison_mfrm_aic_p : float  relative likelihood p-value vs global
        model_comparison_mfrm_aic_preferred : str  preferred model name
        model_comparison_mfrm_aic_summary : pd.DataFrame  ranked comparison table

        Attributes set (BIC)
        --------------------
        model_comparison_mfrm_bic : dict  {model: BIC value}
        model_comparison_mfrm_bic_preferred : str
        model_comparison_mfrm_bic_summary : pd.DataFrame

        Attributes set (LR)
        -------------------
        model_comparison_mfrm_lr : dict  {pair_label: (LR, df, p)}
        model_comparison_mfrm_lr_summary : pd.DataFrame
        """
        if test not in ("LR", "AIC", "BIC"):
            raise ValueError("test must be 'LR', 'AIC', or 'BIC'")
        if sampling is not None and sampling != "dynamic" and not isinstance(sampling, int):
            raise ValueError("sampling must be None, 'dynamic', or an integer")

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
        }
        k_model = {mod: shared_k + rater_k[mod] for mod in self._MODELS}

        # Calibrate and person-estimate each model
        cal_kw = dict(constant=constant, method=method,
                      matrix_power=matrix_power, log_lik_tol=log_lik_tol)
        pe_kw = dict(warm_corr=warm_corr, tolerance=tolerance,
                     max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)
        for mod in self._MODELS:
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
                min(20 * (n_items - 1) * (R - 1), 1500)
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

        # Compute LL for each model
        ll = {mod: self._log_likelihood(model=mod, responses=ll_responses)
              for mod in self._MODELS}

        if test == "AIC":
            aic = {mod: 2 * k_model[mod] - 2 * ll[mod] for mod in self._MODELS}
            best_mod = min(aic, key=aic.__getitem__)
            self.model_comparison_mfrm_aic = aic

            summary_rows = {
                mod: {"LL": ll[mod], "k": k_model[mod], "AIC": aic[mod]}
                for mod in self._MODELS
            }
            summary_df = (
                pd.DataFrame(summary_rows).T
                .sort_values("AIC")
                .rename_axis("Model")
            )

            if aic_sig_test:
                delta = aic["global"] - aic[best_mod]
                aic_p = float(np.exp(-abs(delta) / 2))
                preferred = best_mod if (delta > 0 and aic_p < alpha) else "global"
                self.model_comparison_mfrm_aic_p = aic_p
                self.model_comparison_mfrm_aic_preferred = preferred
                summary_df["ΔAIC"] = summary_df["AIC"] - aic["global"]
                self.model_comparison_mfrm_aic_summary = summary_df
            else:
                preferred = best_mod
                self.model_comparison_mfrm_aic_preferred = preferred
                self.model_comparison_mfrm_aic_summary = summary_df

        elif test == "BIC":
            bic = {mod: k_model[mod] * np.log(n_ll) - 2 * ll[mod] for mod in self._MODELS}
            best_mod = min(bic, key=bic.__getitem__)
            self.model_comparison_mfrm_bic = bic
            self.model_comparison_mfrm_bic_preferred = best_mod
            self.model_comparison_mfrm_bic_summary = (
                pd.DataFrame(
                    {mod: {"LL": ll[mod], "k": k_model[mod], "BIC": bic[mod]}
                     for mod in self._MODELS}
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
            ]
            df_map = {
                ("global", "items"):        (n_items - 1) * (R - 1),
                ("global", "thresholds"):   (m - 1) * (R - 1),
                ("items", "bivector"):      (m - 1) * (R - 1),
                ("thresholds", "bivector"): (n_items - 1) * (R - 1),
                ("bivector", "matrix"):     (n_items - 1) * (m - 1) * (R - 1),
            }
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
                label = f"{null_mod} vs {alt_mod}"
                lr_results[label] = {"LR": lr_stat, "df": df_val, "p-value": p_val}

            self.model_comparison_mfrm_lr = lr_results
            self.model_comparison_mfrm_lr_summary = (
                pd.DataFrame(lr_results).T.rename_axis("Comparison")
            )

    def _rater_ll_from_sev(self, sev_rik, persons, responses_r):
        """
        Log-likelihood contribution for one rater given an (I, m) facet_effect array.

        sev_rik : ndarray shape (I, m)
            Per-threshold facet_effect increments in matrix format — same as the
            values stored in facet_effects_matrix for one rater. cumsum is
            applied per item inside this function, matching _cat_probs_mfrm.
        persons : pd.Series
            Person person_location estimates, already filtered to non-extreme persons.
        responses_r : pd.DataFrame shape (N, I)
            All responses for this rater.
        """
        thresholds = self.thresholds.values          # (m,)
        cumtau = np.concatenate([[0.0], np.cumsum(thresholds)])  # (m+1,)
        diff_arr = self.items.values                  # (I,)
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
        aic_sig_test=True,
        sampling='dynamic',
        alpha=0.05,
        min_effect=0,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        constant=0.1,
        method="cos",
        matrix_power=3,
        log_lik_tol=0.000001,
        seed=None,
    ):
        """
        Assign the simplest adequate rater parameterisation to each rater.

        Calibrates the matrix model once (full data), derives all restricted
        parameterisations as marginal means of each rater's matrix parameters,
        then tests top-down per rater — no refitting at any step.

        Each rater's LL is evaluated over all observations made by that rater
        (whether persons were single- or multi-marked), filtered to persons
        non-extreme on that rater's own data. Person person_locations come from the
        full-data matrix-model estimates.

        Testing ladder (top-down):
          matrix → bivector  (df = (I-1)*(m-1) per rater)
            if rejected  → assign matrix
            if accepted  → test both forks simultaneously:
              bivector → items       (df = m-1)
              bivector → thresholds  (df = I-1)
              both pass  → global
              items only → items
              thresh only → thresholds
              neither    → bivector

        Derivations from matrix params σ_{r,i,k} (all via marginal means):
          bivector  : λ_ri· + λ_r·k  where λ_ri· = mean_k σ_{r,i,k},
                      λ_r·k = mean_i σ_{r,i,k} − μ_r  (zero-summed)
          items     : λ_ri·  replicated across thresholds
          thresholds: mean_i σ_{r,i,k}  replicated across items
          global    : μ_r = mean_{i,k} σ_{r,i,k}  (scalar, replicated)

        Parameters
        ----------
        anchors : list or None, default None
            Rater identifiers to use as the anchor frame of reference. When
            provided, the matrix model is calibrated in anchor mode (anchor
            raters' mean facet_effect fixed to zero) before testing. Different
            anchor sets give different — all valid — per-rater assignments,
            because the zero-sum constraint means a non-uniform rater can
            induce apparent non-uniformity in others. All raters (anchor and
            non-anchor) are tested; results are stored as
            anchor_facet_effects_mixed and anchor_rater_models.
        test : {'AIC', 'BIC', 'LR'}, default 'AIC'
            Model selection criterion. AIC and BIC pick the minimum across all
            five models directly. LR uses the top-down ladder (matrix→bivector,
            then fork to items/thresholds/global) with significance level alpha.
        aic_sig_test : bool, default True
            When test='AIC', prefer a model more complex than global only if
            ΔAIC > 0 (vs global) AND p = exp(-ΔAIC/2) < alpha. Prevents
            spurious complexity at large n where ΔAIC differences are small.
        sampling : 'dynamic', int, or None, default 'dynamic'
            Per-rater subsampling for LL evaluation. 'dynamic' uses
            T = min(20*(I-1)*(m-1), 1500); an integer fixes T directly;
            None uses all of each rater's persons. Parameters are always
            estimated on the full data; only LL evaluation is subsampled.
            n_r (= T when triggered) is used in BIC's ln(n) term.
        seed : int or None, default None
            Seed for the per-rater subsampling RNG (only used when sampling
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
            For AIC/BIC the gate is applied as a post-hoc walk-back from the
            selected model toward global. For LR it is an additional condition
            at each ladder step alongside the p-value. Default 0 disables.
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
        n_items = self.no_of_items
        m = self.max_score
        anchored = anchors is not None

        cal_kw = dict(constant=constant, method=method,
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
                T = min(20 * (n_items - 1) * (m - 1), 1500)
            elif isinstance(sampling, int):
                T = sampling
            else:
                T = None
            if T is not None and len(persons_r) > T:
                persons_r = persons_r.sample(T, random_state=seed)

            if len(persons_r) == 0:
                records[rater] = {"selected_model": "matrix",
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

            ll_mat = self._rater_ll_from_sev(sev_mat,    persons_r, responses_r)
            ll_biv = self._rater_ll_from_sev(sev_biv,    persons_r, responses_r)
            ll_itm = self._rater_ll_from_sev(sev_items,  persons_r, responses_r)
            ll_thr = self._rater_ll_from_sev(sev_thresh, persons_r, responses_r)
            ll_glb = self._rater_ll_from_sev(sev_global, persons_r, responses_r)

            rec = {
                "LL_matrix": ll_mat, "LL_bivector": ll_biv,
                "LL_items": ll_itm, "LL_thresholds": ll_thr, "LL_global": ll_glb,
            }

            # Per-rater free parameter counts (rater side only)
            k_map = {"global": 1, "items": n_items, "thresholds": m,
                     "bivector": n_items + m - 1, "matrix": n_items * m}
            n_r = len(persons_r)  # subsampled n when sampling triggered

            if test in ("AIC", "BIC"):
                ll_map = {"global": ll_glb, "items": ll_itm, "thresholds": ll_thr,
                          "bivector": ll_biv, "matrix": ll_mat}
                if test == "AIC":
                    ic = {mod: 2 * k_map[mod] - 2 * ll for mod, ll in ll_map.items()}
                else:
                    ic = {mod: k_map[mod] * np.log(n_r) - 2 * ll
                          for mod, ll in ll_map.items()}
                rec.update({f"{test}_{mod}": v for mod, v in ic.items()})
                best = min(ic, key=ic.get)
                if test == "AIC" and aic_sig_test:
                    delta = ic["global"] - ic[best]
                    p_rel = float(np.exp(-abs(delta) / 2))
                    rec["p_vs_global"] = p_rel
                    selected = best if (delta > 0 and p_rel < alpha) else "global"
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

            else:  # LR — top-down ladder
                lr_biv = max(0.0, -2.0 * (ll_biv - ll_mat))
                p_biv = float(chi2.sf(lr_biv, (n_items - 1) * (m - 1)))
                rec.update({"LR_biv_vs_matrix": lr_biv, "p_biv_vs_matrix": p_biv,
                            "LR_items_vs_biv": np.nan, "p_items_vs_biv": np.nan,
                            "LR_thresh_vs_biv": np.nan, "p_thresh_vs_biv": np.nan})
                if p_biv < alpha or eff_mat_biv < min_effect:
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

            rec["selected_model"] = selected
            records[rater] = rec

        # Build column order: selected_model first, then LLs, then test-specific
        ll_cols = ["LL_matrix", "LL_bivector", "LL_items", "LL_thresholds", "LL_global"]
        if test in ("AIC", "BIC"):
            ic_cols = [f"{test}_{m_}" for m_ in ("matrix", "bivector", "items", "thresholds", "global")]
            extra_cols = ic_cols + (["p_vs_global"] if test == "AIC" and aic_sig_test else [])
        else:
            extra_cols = ["LR_biv_vs_matrix", "p_biv_vs_matrix",
                          "LR_items_vs_biv", "p_items_vs_biv",
                          "LR_thresh_vs_biv", "p_thresh_vs_biv"]
        cols = ["selected_model"] + ll_cols + extra_cols
        table = pd.DataFrame(records).T.rename_axis("Rater")[cols]
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

    def _apply_rater_models(self, facet_effects_matrix, rater_models):
        """Derive facet_effects_mixed by applying per-rater model restrictions to matrix estimates.

        Parameters
        ----------
        facet_effects_matrix : DataFrame
            MultiIndex (rater, item) × thresholds, as produced by calibrate(model='matrix').
        rater_models : Series
            Per-rater model assignment (index=rater names, values in _MODELS).

        Returns
        -------
        DataFrame
            Same MultiIndex structure as facet_effects_matrix, with each rater's
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

        Pivots the residual DataFrame to (Person×Items, Raters) shape,
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
                "PCA of facet_element standardised residuals failed. "
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
        (a pure translation), so once a valid per-rater assignment exists we
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
            if k in ("constant", "method", "matrix_power", "log_lik_tol")
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
    ):
        """Internal helper: compute standard errors (and optionally anchor SEs) if not yet done."""
        anc = anchors is not None
        prefix = "anchor_" if anc else ""
        trigger = f"{prefix}threshold_se_{model}"
        # Re-run if SEs not computed, or if CIs requested but not yet stored.
        ci_attr = "anchor_item_low" if anc else "item_low"
        ci_missing = interval is not None and not hasattr(self, ci_attr)
        if not hasattr(self, trigger) or ci_missing:
            if anc:
                # Ensure unanchored SEs exist first (anchor_std_errors depends on them)
                if not hasattr(self, f"threshold_se_{model}"):
                    self.std_errors(
                        model=model,
                        interval=interval,
                        no_of_samples=no_of_samples,
                        constant=constant,
                        method=method,
                        matrix_power=matrix_power,
                        log_lik_tol=log_lik_tol,
                        seed=seed,
                    )
                self.anchor_std_errors(model=model, anchors=anchors, seed=seed)
            else:
                self.std_errors(
                    model=model,
                    interval=interval,
                    no_of_samples=no_of_samples,
                    constant=constant,
                    method=method,
                    matrix_power=matrix_power,
                    log_lik_tol=log_lik_tol,
                    seed=seed,
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
        method="cos",
        constant=0.1,
        matrix_power=3,
        log_lik_tol=0.000001,
        no_of_samples=500,
        interval=None,
        seed=None,
    ):
        """
        Build and store the item statistics summary table.

        Auto-triggers the full calibration/SE/fit chain if not yet run.
        Stores result as self.item_stats_{model}.

        Parameters
        ----------
        model : str, default 'global'
            Rater parameterisation.
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
        method : str, default 'cos'
            Priority vector extraction method.
        constant : float, default 0.1
            Additive smoothing constant.
        matrix_power : int, default 3
            Matrix power for calibration.
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

        Attributes set
        --------------
        item_stats_{model} : pandas.DataFrame
            Item statistics with items as rows. Always contains Estimate,
            Count, Facility, Infit MS, Outfit MS. SE (and CI bound columns,
            if interval is set) included only when se=True.
        """

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
            method=method,
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
                seed=seed,
            )
        if not hasattr(self, f"item_outfit_ms_{model}"):
            self._run_item_fit(model)

        anc = anchors is not None
        item_locations = getattr(self, f"anchor_items_{model}") if anc else self.items

        stats = pd.DataFrame(index=self.responses.columns)
        stats["Estimate"] = item_locations.round(dp)
        if se:
            se_vals = (
                self.anchor_item_se
                if (anc and hasattr(self, "anchor_item_se"))
                else self.item_se
            )
            low = (
                self.anchor_item_low
                if (anc and hasattr(self, "anchor_item_low"))
                else self.item_low if hasattr(self, "item_low") else None
            )
            high = (
                self.anchor_item_high
                if (anc and hasattr(self, "anchor_item_high"))
                else self.item_high if hasattr(self, "item_high") else None
            )
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

    # Backwards-compatible aliases
    def item_stats_df_global(self, **kw):
        """Alias for item_stats_df(model='global'). See item_stats_df for full documentation."""
        self.item_stats_df(model="global", **kw)

    def item_stats_df_items(self, **kw):
        """Alias for item_stats_df(model='items'). See item_stats_df for full documentation."""
        self.item_stats_df(model="items", **kw)

    def item_stats_df_thresholds(self, **kw):
        """Alias for item_stats_df(model='thresholds'). See item_stats_df for full documentation."""
        self.item_stats_df(model="thresholds", **kw)

    def item_stats_df_matrix(self, **kw):
        """Alias for item_stats_df(model='matrix'). See item_stats_df for full documentation."""
        self.item_stats_df(model="matrix", **kw)

    def item_stats_df_bivector(self, **kw):
        """Alias for item_stats_df(model='bivector'). See item_stats_df for full documentation."""
        self.item_stats_df(model="bivector", **kw)

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
        method="cos",
        constant=0.1,
        matrix_power=3,
        log_lik_tol=0.000001,
        no_of_samples=500,
        interval=None,
        seed=None,
    ):
        """
        Build and store the threshold statistics summary table.

        Auto-triggers the full calibration/SE/fit chain if not yet run.
        Stores result as self.threshold_stats_{model}.

        Parameters
        ----------
        model : str, default 'global'
            Rater parameterisation.
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
        method : str, default 'cos'
            Priority vector extraction method.
        constant : float, default 0.1
            Smoothing constant.
        matrix_power : int, default 3
            Matrix power.
        log_lik_tol : float, default 0.000001
            Calibration convergence tolerance.
        no_of_samples : int, default 500
            Bootstrap samples. Unused if se=False.
        interval : float or None, default None
            CI width. Ignored if se=False.
        seed : int or None, default None
            Seed passed through to the internal std_errors() call (only
            used if not already computed). None draws fresh entropy.

        Attributes set
        --------------
        threshold_stats_{model} : pandas.DataFrame
            Threshold statistics, rows Threshold 1..Threshold max_score.
        """

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
            method=method,
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
                seed=seed,
            )
        if not hasattr(self, f"threshold_outfit_ms_{model}"):
            self._run_threshold_fit(model, anchors=anchors)

        anc = anchors is not None
        thresholds = (
            getattr(self, f"anchor_thresholds_{model}") if anc else self.thresholds
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

    def threshold_stats_df_global(self, **kw):
        """Alias for threshold_stats_df(model='global'). See threshold_stats_df for full documentation."""
        self.threshold_stats_df(model="global", **kw)

    def threshold_stats_df_items(self, **kw):
        """Alias for threshold_stats_df(model='items'). See threshold_stats_df for full documentation."""
        self.threshold_stats_df(model="items", **kw)

    def threshold_stats_df_thresholds(self, **kw):
        """Alias for threshold_stats_df(model='thresholds'). See threshold_stats_df for full documentation."""
        self.threshold_stats_df(model="thresholds", **kw)

    def threshold_stats_df_matrix(self, **kw):
        """Alias for threshold_stats_df(model='matrix'). See threshold_stats_df for full documentation."""
        self.threshold_stats_df(model="matrix", **kw)

    def threshold_stats_df_bivector(self, **kw):
        """Alias for threshold_stats_df(model='bivector'). See threshold_stats_df for full documentation."""
        self.threshold_stats_df(model="bivector", **kw)

    def category_stats_df(
        self,
        model="global",
        anchors=None,
        dp=3,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        method="cos",
        constant=0.1,
        matrix_power=3,
        log_lik_tol=0.000001,
        no_of_samples=500,
        interval=None,
        seed=None,
    ):
        """
        Build and store the category width statistics summary table.

        Reports category *widths* (thresholds[k+1] - thresholds[k]) rather
        than raw threshold locations — see threshold_stats_df for the full
        rationale: a zero-summed threshold vector has only max_score-1 true
        degrees of freedom, so the max_score per-threshold SEs threshold_
        stats_df reports are correlated, not independent, and understate
        the true uncertainty of the physically meaningful step-structure
        quantity. self.thresholds (and its anchored counterpart) is shared
        across all five rater parameterisations, so the reported widths
        are the same regardless of model= — only their bootstrap SEs can
        differ, since std_errors() recalibrates the whole model (rater
        facet_effects included) per resample. Reported *alongside*
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
            Rater parameterisation. Affects the reported SE (via that
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
        """
        self._ensure_calibrated(
            model,
            anchors=anchors,
            interval=interval,
            no_of_samples=no_of_samples,
            constant=constant,
            method=method,
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
            seed=seed,
        )

        anc = anchors is not None
        prefix = "anchor_" if anc else ""

        thresholds = (
            getattr(self, f"anchor_thresholds_{model}") if anc else self.thresholds
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

    def category_stats_df_global(self, **kw):
        """Alias for category_stats_df(model='global'). See category_stats_df for full documentation."""
        self.category_stats_df(model="global", **kw)

    def category_stats_df_items(self, **kw):
        """Alias for category_stats_df(model='items'). See category_stats_df for full documentation."""
        self.category_stats_df(model="items", **kw)

    def category_stats_df_thresholds(self, **kw):
        """Alias for category_stats_df(model='thresholds'). See category_stats_df for full documentation."""
        self.category_stats_df(model="thresholds", **kw)

    def category_stats_df_matrix(self, **kw):
        """Alias for category_stats_df(model='matrix'). See category_stats_df for full documentation."""
        self.category_stats_df(model="matrix", **kw)

    def category_stats_df_bivector(self, **kw):
        """Alias for category_stats_df(model='bivector'). See category_stats_df for full documentation."""
        self.category_stats_df(model="bivector", **kw)

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
        method="cos",
        constant=0.1,
        matrix_power=3,
        log_lik_tol=0.000001,
        interval=None,
        no_of_samples=500,
    ):
        """
        Build and store the person statistics summary table.

        Auto-triggers calibration and person person_location estimation if not yet run.
        Stores result as self.person_stats_{model}.

        Parameters
        ----------
        model : str, default 'global'
            Rater parameterisation.
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
        method : str, default 'cos'
            Priority vector extraction method.
        constant : float, default 0.1
            Smoothing constant.
        matrix_power : int, default 3
            Matrix power.
        log_lik_tol : float, default 0.000001
            Calibration convergence tolerance.
        interval : float or None, default None
            CI width (unused directly; passed to _ensure_calibrated).
        no_of_samples : int, default 500
            Bootstrap samples.

        Attributes set
        --------------
        person_stats_{model} : pandas.DataFrame
            Person statistics with persons as rows. Contains Estimate, CSEM,
            Score, Max score, p, Infit MS, Outfit MS. Optional: RSEM, Infit Z,
            Outfit Z.
        """

        self._ensure_calibrated(
            model,
            warm_corr=warm_corr,
            tolerance=tolerance,
            max_iters=max_iters,
            ext_score_adjustment=ext_score_adjustment,
            constant=constant,
            method=method,
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

    def person_stats_df_global(self, **kw):
        """Alias for person_stats_df(model='global'). See person_stats_df for full documentation."""
        self.person_stats_df(model="global", **kw)

    def person_stats_df_items(self, **kw):
        """Alias for person_stats_df(model='items'). See person_stats_df for full documentation."""
        self.person_stats_df(model="items", **kw)

    def person_stats_df_thresholds(self, **kw):
        """Alias for person_stats_df(model='thresholds'). See person_stats_df for full documentation."""
        self.person_stats_df(model="thresholds", **kw)

    def person_stats_df_matrix(self, **kw):
        """Alias for person_stats_df(model='matrix'). See person_stats_df for full documentation."""
        self.person_stats_df(model="matrix", **kw)

    def person_stats_df_bivector(self, **kw):
        """Alias for person_stats_df(model='bivector'). See person_stats_df for full documentation."""
        self.person_stats_df(model="bivector", **kw)

    def test_stats_df(
        self,
        model="global",
        dp=3,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        method="cos",
        constant=0.1,
        matrix_power=3,
        log_lik_tol=0.000001,
        no_of_samples=500,
        seed=None,
    ):
        """
        Build and store the test-level summary statistics table.

        Auto-triggers calibration and test fit statistics if not yet run.
        Stores result as self.test_stats_{model}.

        Parameters
        ----------
        model : str, default 'global'
            Rater parameterisation.
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
        method : str, default 'cos'
            Priority vector extraction method.
        constant : float, default 0.1
            Smoothing constant.
        matrix_power : int, default 3
            Matrix power.
        log_lik_tol : float, default 0.000001
            Calibration convergence tolerance.
        no_of_samples : int, default 500
            Bootstrap samples.
        seed : int or None, default None
            Seed passed through to the internal std_errors() call (only
            used if not already computed). None draws fresh entropy.

        Attributes set
        --------------
        test_stats_{model} : pandas.DataFrame
            Two-column table (Items, Persons) with rows:
            Mean, SD, Separation ratio, Strata, Reliability.
        """

        self._ensure_calibrated(
            model,
            constant=constant,
            method=method,
            matrix_power=matrix_power,
            log_lik_tol=log_lik_tol,
            seed=seed,
        )
        if not hasattr(self, f"psi_{model}"):
            self._run_test_fit(model, seed=seed)

        stats = pd.DataFrame(
            {
                "Items": [
                    self.items.mean(),
                    self.items.std(),
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

    def test_stats_df_global(self, **kw):
        """Alias for test_stats_df(model='global'). See test_stats_df for full documentation."""
        self.test_stats_df(model="global", **kw)

    def test_stats_df_items(self, **kw):
        """Alias for test_stats_df(model='items'). See test_stats_df for full documentation."""
        self.test_stats_df(model="items", **kw)

    def test_stats_df_thresholds(self, **kw):
        """Alias for test_stats_df(model='thresholds'). See test_stats_df for full documentation."""
        self.test_stats_df(model="thresholds", **kw)

    def test_stats_df_matrix(self, **kw):
        """Alias for test_stats_df(model='matrix'). See test_stats_df for full documentation."""
        self.test_stats_df(model="matrix", **kw)

    def test_stats_df_bivector(self, **kw):
        """Alias for test_stats_df(model='bivector'). See test_stats_df for full documentation."""
        self.test_stats_df(model="bivector", **kw)

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
        dp=3,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        method="cos",
        constant=0.1,
        matrix_power=3,
        log_lik_tol=0.000001,
        no_of_samples=500,
        interval=None,
        seed=None,
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

        Auto-triggers the full calibration/SE/fit chain if not yet run.
        Stores result as self.rater_stats_{model}.

        Parameters
        ----------
        model : str, default 'global'
            Rater parameterisation.
        anchors : list or None, default None
            Anchor facet_elements for SE computation.
        full : bool, default False
            If True, sets zstd=True, interval=0.95.
        zstd : bool, default False
            If True, includes Infit Z and Outfit Z columns.
        marginal : bool, default True
            For matrix model only: if True returns marginal twin-vector
            representation; if False returns full (item, threshold) cell table.
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
        method : str, default 'cos'
            Priority vector extraction method.
        constant : float, default 0.1
            Smoothing constant.
        matrix_power : int, default 3
            Matrix power.
        log_lik_tol : float, default 0.000001
            Calibration convergence tolerance.
        no_of_samples : int, default 500
            Bootstrap samples.
        interval : float or None, default None
            CI width.
        seed : int or None, default None
            Seed passed through to the internal std_errors() call (only
            used if not already computed). None draws fresh entropy.

        Attributes set
        --------------
        rater_stats_{model} : pandas.DataFrame
            Rater statistics table. Structure depends on model (see above).
        """

        if full:
            zstd = True
            interval = interval or 0.95

        self._ensure_calibrated(
            model,
            anchors=anchors,
            interval=interval,
            no_of_samples=no_of_samples,
            constant=constant,
            method=method,
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
            seed=seed,
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
            sev_attr = (
                f"anchor_facet_effects_{model}" if anc else f"facet_effects_{model}"
            )
            facet_effects = getattr(self, sev_attr)
            se_attr = f"anchor_rater_se_{model}" if anc else f"rater_se_{model}"
            rse = getattr(self, se_attr, {})
            lo_attr = f"anchor_rater_low_{model}" if anc else f"rater_low_{model}"
            hi_attr = f"anchor_rater_high_{model}" if anc else f"rater_high_{model}"
            rlo = getattr(self, lo_attr, None)
            rhi = getattr(self, hi_attr, None)

            def _ov_stats():
                """Build the overall rater fit statistics sub-table for the current model."""
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

            result["Overall statistics"] = _ov_stats()
            stats = pd.concat(result.values(), keys=result.keys()).T
            setattr(self, f"rater_stats_{model}", stats)
        self._set_facet_aliases(model)

    def rater_stats_df_global(self, **kw):
        """Alias for rater_stats_df(model='global'). See rater_stats_df for full documentation."""
        self.rater_stats_df(model="global", **kw)

    def rater_stats_df_items(self, **kw):
        """Alias for rater_stats_df(model='items'). See rater_stats_df for full documentation."""
        self.rater_stats_df(model="items", **kw)

    def rater_stats_df_thresholds(self, **kw):
        """Alias for rater_stats_df(model='thresholds'). See rater_stats_df for full documentation."""
        self.rater_stats_df(model="thresholds", **kw)

    def rater_stats_df_matrix(self, **kw):
        """Alias for rater_stats_df(model='matrix'). See rater_stats_df for full documentation."""
        self.rater_stats_df(model="matrix", **kw)

    def rater_stats_df_bivector(self, **kw):
        """Alias for rater_stats_df(model='bivector'). See rater_stats_df for full documentation."""
        self.rater_stats_df(model="bivector", **kw)

    # ------------------------------------------------------------------
    # Save statistics
    # ------------------------------------------------------------------

    def save_stats(
        self,
        model="global",
        filename="",
        format="csv",
        dp=3,
        warm_corr=True,
        tolerance=0.00001,
        max_iters=100,
        ext_score_adjustment=0.5,
        method="cos",
        constant=0.1,
        matrix_power=3,
        log_lik_tol=0.000001,
        no_of_samples=500,
        interval=None,
    ):
        """
        Export item, threshold, facet_element, person, and test statistics to file.

        Auto-triggers all stats_df methods if not yet run. Saves all five
        tables to either a single Excel workbook or separate CSV files.

        Parameters
        ----------
        model : str, default 'global'
            Rater parameterisation.
        filename : str, default ''
            Output filename or path (without extension for CSV).
        format : str, default 'csv'
            'csv' saves five separate CSV files. 'xlsx' saves to a single
            workbook with separate sheets.
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
        method : str, default 'cos'
            Priority vector extraction method.
        constant : float, default 0.1
            Smoothing constant.
        matrix_power : int, default 3
            Matrix power.
        log_lik_tol : float, default 0.000001
            Calibration convergence tolerance.
        no_of_samples : int, default 500
            Bootstrap samples.
        interval : float or None, default None
            CI width for SEs.
        """

        kw = dict(
            dp=dp,
            warm_corr=warm_corr,
            tolerance=tolerance,
            max_iters=max_iters,
            ext_score_adjustment=ext_score_adjustment,
            method=method,
            constant=constant,
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
                dict(no_of_samples=no_of_samples, interval=interval),
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
        else:
            if filename.endswith(".csv"):
                filename = filename[:-4]
            getattr(self, f"item_stats_{model}").to_csv(f"{filename}_item_stats.csv")
            getattr(self, f"threshold_stats_{model}").to_csv(
                f"{filename}_threshold_stats.csv"
            )
            getattr(self, f"rater_stats_{model}").to_csv(f"{filename}_rater_stats.csv")
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

    def _facet_effect_item_offset(self, model, facet_effects, facet_element):
        """Return per-item facet_effect offset Series for a given facet_element and model."""
        if model == "global":
            return pd.Series(
                float(facet_effects.loc[facet_element]), index=self.item_names
            )
        elif model == "items":
            return facet_effects.loc[facet_element]
        elif model == "thresholds":
            return pd.Series(
                float(facet_effects.loc[facet_element].mean()), index=self.item_names
            )
        elif model in ("bivector", "matrix"):
            # facet_effects is MultiIndex (facet_element, item) × thresholds
            return facet_effects.loc[facet_element].mean(axis=1)

    def _zero_facet_effects(self, model, facet_effects):
        """Return facet_effects structure identical in shape but all values zero.
        Used to evaluate neutral (facet_effect=0) curves in plotting methods."""
        if model == "global":
            return pd.Series(0.0, index=facet_effects.index)
        elif model == "items":
            return pd.DataFrame(0.0, index=facet_effects.index, columns=facet_effects.columns)
        elif model == "thresholds":
            return pd.DataFrame(0.0, index=facet_effects.index, columns=facet_effects.columns)
        elif model in ("bivector", "matrix"):
            return pd.DataFrame(0.0, index=facet_effects.index, columns=facet_effects.columns)

    def _mean_facet_effects(self, model, facet_effects, facet_element):
        """Return a facet_effects structure where `facet_element` has the mean facet_effect
        across all facet_elements.  Used to plot curves that match obs averaged across
        the full facet_element pool."""
        if model == "global":
            result = facet_effects.copy()
            result[facet_element] = float(facet_effects.mean())
            return result
        elif model == "items":
            result = facet_effects.copy()
            result.loc[facet_element] = facet_effects.mean(axis=0)
            return result
        elif model == "thresholds":
            result = facet_effects.copy()
            result.loc[facet_element] = facet_effects.mean(axis=0)
            return result
        elif model in ("bivector", "matrix"):
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
        facet_elements=None,
        obs=None,
        thresh_obs=None,
        x_obs_data=np.array([]),
        y_obs_data=np.array([]),
        thresh_lines=False,
        central_location=False,
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
        palette="dark blue",
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
        Shared across all four facet_element parameterisations.
        """
        item_locations, thresholds, facet_effects = self._get_params(model, anchor)

        if isinstance(facet_elements, str):
            facet_elements = (
                None if facet_elements in ("none", "zero", "all") else [facet_elements]
            )
        if isinstance(items, str):
            items = None if items == "all" else items

        if plot_style == "dark":
            sns.set_style("darkgrid")
        else:
            sns.set_style("whitegrid")

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
        }
        shade, base_color = palette_dict[palette]
        if shade == "dark":
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

            for i in range(no_of_plots):
                col = (
                    "black"
                    if black
                    else (
                        scalarMap.to_rgba(i) if "multi" not in palette else color_map[i]
                    )
                )
                ax.plot(x_data, y_data[:, i], "", color=col, label=i + 1)

            if obs is not None:
                try:
                    if isinstance(y_obs_data, pd.Series):
                        col = (
                            scalarMap.to_rgba(0)
                            if "multi" not in palette
                            else color_map[0]
                        )
                        ax.plot(x_obs_data, y_obs_data, "o", color=col)
                    else:
                        for j in range(y_obs_data.shape[0]):
                            col = (
                                scalarMap.to_rgba(j)
                                if "multi" not in palette
                                else color_map[j]
                            )
                            ax.plot(x_obs_data, y_obs_data[j, :], "o", color=col)
                except Exception:
                    pass

            if thresh_obs is not None:
                try:
                    for j in range(x_obs_data.shape[0]):
                        col = (
                            scalarMap.to_rgba(j)
                            if "multi" not in palette
                            else color_map[j]
                        )
                        ax.plot(x_obs_data[j, :], y_obs_data[j, :], "o", color=col)
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
                    elif model == "thresholds":
                        sev_val = float(facet_effects.loc[r_sc, t + 1])
                    elif model in ("bivector", "matrix"):
                        item_key = (
                            items if items is not None else list(self.item_names)[0]
                        )
                        sev_val = float(facet_effects.loc[(r_sc, item_key), t + 1])
                    elif model == "items" and items is not None:
                        sev_val = float(
                            self._facet_effect_item_offset(model, facet_effects, r_sc)[items]
                        )
                    else:
                        sev_val = float(
                            self._facet_effect_item_offset(model, facet_effects, r_sc).mean()
                        )
                    xval = diff_val + thresholds[t + 1] + sev_val
                    ax.axvline(x=xval, color="black", linestyle="--")

            if central_location:
                if items is None:
                    ax.axvline(x=0, color="darkred", linestyle="--")
                else:
                    ax.axvline(
                        x=float(item_locations[items]), color="darkred", linestyle="--"
                    )

            if score_lines_item[1] is not None:
                item = score_lines_item[0]
                if all(s > 0 for s in score_lines_item[1]) and all(
                    s < self.max_score for s in score_lines_item[1]
                ):
                    # ICC score line: invert the curve numerically by finding
                    # the x value where y_data is closest to s
                    for s in score_lines_item[1]:
                        idx = np.argmin(np.abs(y_data[:, 0] - s))
                        estimate = x_data[idx]
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
                item = point_info_lines_item[0]
                r = (
                    facet_elements[0]
                    if isinstance(facet_elements, list)
                    else (
                        facet_elements
                        if facet_elements is not None
                        else list(self.facet_names)[0]
                    )
                )
                for estimate in point_info_lines_item[1]:
                    info = self.variance(
                        estimate, item, item_locations, r, facet_effects, thresholds, model
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
                    if model == "items" and items is not None:
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
        palette="dark blue",
        black=False,
        font="Times New Roman",
        title_font_size=15,
        axis_font_size=12,
        labelsize=12,
        filename=None,
        file_format="png",
        dpi=300,
    ):
        """Item Characteristic Curve."""
        item_locations, thresholds, facet_effects = self._get_params(model, anchor)
        if facet_element in ("none", "zero"):
            facet_element = None

        person_locations_arr = np.arange(-20, 20, 0.1)
        r_use = (
            facet_element if facet_element is not None else list(self.facet_names)[0]
        )
        # When no specific facet_element requested, average exp_score across all facet_elements
        # to match the obs y-values which are mean scores across the facet_element pool.
        if facet_element is None:
            all_raters = list(self.facet_names)
            y = np.array(
                [
                    np.mean(
                        [
                            self.exp_score(
                                a, item, item_locations, r, facet_effects, thresholds, model
                            )
                            for r in all_raters
                        ]
                    )
                    for a in person_locations_arr
                ]
            ).reshape(-1, 1)
        else:
            y = np.array(
                [
                    self.exp_score(
                        a, item, item_locations, r_use, facet_effects, thresholds, model
                    )
                    for a in person_locations_arr
                ]
            ).reshape(-1, 1)

        if obs is not None:
            persons_attr = f'{"anchor_" if anchor else ""}persons_{model}'
            if not hasattr(self, persons_attr):
                self.person_estimates(model=model, anchor=anchor)
            person_estimates = getattr(self, persons_attr)
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
            facet_elements=facet_element,
            obs=obs,
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
        palette="dark blue",
        black=False,
        font="Times New Roman",
        title_font_size=15,
        axis_font_size=12,
        labelsize=12,
        filename=None,
        file_format="png",
        dpi=300,
    ):
        """Category Response Curves."""
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
            obs=obs,
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
        palette="dark blue",
        black=False,
        font="Times New Roman",
        title_font_size=15,
        axis_font_size=12,
        labelsize=12,
        filename=None,
        file_format="png",
        dpi=300,
    ):
        """Threshold Characteristic Curves."""
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
        elif model == "thresholds":
            sev_thresh = facet_effects.loc[r_use].values.astype(float)
        elif model in ("bivector", "matrix"):
            item_key = item if item is not None else list(self.item_names)[0]
            sev_thresh = facet_effects.loc[(r_use, item_key)].values.astype(float)
        elif model == "items" and item is not None:
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
            obs=None,
            thresh_obs=obs,
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
        palette="dark blue",
        black=False,
        font="Times New Roman",
        title_font_size=15,
        axis_font_size=12,
        labelsize=12,
        filename=None,
        file_format="png",
        dpi=300,
    ):
        """Item Information Curve."""
        item_locations, thresholds, facet_effects = self._get_params(model, anchor)
        r_use = (
            facet_element[0]
            if isinstance(facet_element, list)
            else (
                facet_element
                if facet_element is not None and facet_element not in ("none", "zero")
                else list(self.facet_names)[0]
            )
        )
        estimates = np.arange(-20, 20, 0.1)
        y = np.array(
            [
                self.variance(
                    a, item, item_locations, r_use, facet_effects, thresholds, model
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
            items=item,
            facet_elements=facet_element,
            x_min=xmin,
            x_max=xmax,
            y_max=ymax,
            thresh_lines=thresh_lines,
            central_location=central_location,
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
        palette="dark blue",
        black=False,
        font="Times New Roman",
        title_font_size=15,
        axis_font_size=12,
        labelsize=12,
        filename=None,
        file_format="png",
        dpi=300,
    ):
        """Test Characteristic Curve."""
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
                    "complete data across all facet_element×item combinations in scope.",
                    UserWarning,
                    stacklevel=2,
                )
                xobsdata = yobsdata = np.array(np.nan)
            else:
                if n_complete < len(obs_counts):
                    warnings.warn(
                        f"TCC observed score overlay uses {n_complete} of "
                        f"{len(obs_counts)} persons with complete data across "
                        f"all facet_element×item combinations in scope.",
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
        palette="dark blue",
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
        palette="dark blue",
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
        font="Times New Roman",
        title_font_size=15,
        axis_font_size=12,
        labelsize=12,
        filename=None,
        file_format="png",
        plot_density=300,
    ):
        """Standardised residuals histogram with optional item/facet_element subsetting."""
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
            font=font,
            title_font_size=title_font_size,
            axis_font_size=axis_font_size,
            labelsize=labelsize,
            filename=filename,
            file_format=file_format,
            plot_density=plot_density,
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
