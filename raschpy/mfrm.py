from math import log
import warnings

import numpy as np
import pandas as pd
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

        Supports five facet_element severity parameterisations:
          'global'     — scalar severity λ_r per facet_element
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
    # Model registry — maps model name to severity attribute names
    # ------------------------------------------------------------------
    _MODELS = ("global", "items", "thresholds", "bivector", "matrix")

    def _attr(self, model, name, anchor=False):
        """Return the attribute name for a given model and statistic."""
        prefix = "anchor_" if anchor else ""
        suffix = f"_{model}"
        return f"{prefix}{name}{suffix}"

    def _get_params(self, model, anchor=False):
        """
        Return (difficulties, thresholds, severities) for the requested model.
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
        """Return ability estimates for the requested model. Auto-triggers if needed."""
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
    ):
        """
        Initialise a Many-Facet Rasch Model (MFRM) object.

        The MFRM extends the RSM/PCM to include facet_element facets. Four facet_element
        parameterisations are supported, selected at calibrate() time:
        'global' (single severity per facet_element), 'items' (per-item severities),
        'thresholds' (per-threshold severities), 'matrix' (per-item-threshold).

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

    # ------------------------------------------------------------------
    # Rename utilities
    # ------------------------------------------------------------------

    def _set_facet_aliases(self, model, anchor=False):
        """Set dynamic facet-named aliases for public severity/SE attributes."""
        prefix = "anchor_" if anchor else ""
        # Severity estimates
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
        ability,
        item,
        difficulties,
        facet_element,
        severities,
        category,
        thresholds,
        model="global",
    ):
        """
        Compute the probability of a response category for a single observation.

        Applies the MFRM log-numerator: k*(b - d_i) - cumsum(tau) - rater_severity,
        where the severity term depends on the model parameterisation.
        Numerically stabilised via log-sum-exp.

        Parameters
        ----------
        ability : float
            Person ability estimate on the logit scale.
        item : str
            Item identifier.
        difficulties : pandas.Series
            Item difficulty estimates indexed by item name.
        facet_element : str
            Rater identifier.
        severities : Series or dict
            Rater severity parameters. Structure depends on model:
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
        log_nums = cats * (ability - difficulties.loc[item]) - cumtau
        # Apply facet_element severity
        if model == "global":
            log_nums -= cats * severities.loc[facet_element]
        elif model == "items":
            log_nums -= cats * severities.loc[facet_element, item]
        elif model == "thresholds":
            log_nums -= np.concatenate(
                [[0.0], np.cumsum(severities.loc[facet_element].values)]
            )
        elif model in ("bivector", "matrix"):
            log_nums -= np.concatenate(
                [[0.0], np.cumsum(severities.loc[facet_element, item].values)]
            )
        log_nums -= log_nums.max()
        nums = np.exp(log_nums)
        return nums[category] / nums.sum()

    def exp_score(
        self,
        ability,
        item,
        difficulties,
        facet_element,
        severities,
        thresholds,
        model="global",
    ):
        """
        Compute the expected score for a single person/facet_element/item combination.

        Calculates E[X | ability, item, facet_element, model] = sum(k * P(X=k)).
        Used in scalar Newton-Raphson estimation and score_lookup().

        Parameters
        ----------
        ability : float
            Person ability estimate on the logit scale.
        item : str
            Item identifier.
        difficulties : pandas.Series
            Item difficulty estimates.
        facet_element : str
            Rater identifier.
        severities : Series or dict
            Rater severity parameters (structure depends on model).
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
                    ability,
                    item,
                    difficulties,
                    facet_element,
                    severities,
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
        ability,
        item,
        difficulties,
        facet_element,
        severities,
        thresholds,
        model="global",
    ):
        """
        Compute item variance (Fisher information) for a single observation.

        Calculates Var[X | ability, item, facet_element, model] = sum((k - E[X])^2 * P(X=k)).
        Used in scalar Newton-Raphson estimation and score_lookup().

        Parameters
        ----------
        ability : float
            Person ability estimate on the logit scale.
        item : str
            Item identifier.
        difficulties : pandas.Series
            Item difficulty estimates.
        facet_element : str
            Rater identifier.
        severities : Series or dict
            Rater severity parameters.
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
                    ability,
                    item,
                    difficulties,
                    facet_element,
                    severities,
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
        ability,
        item,
        difficulties,
        facet_element,
        severities,
        thresholds,
        model="global",
    ):
        """
        Compute the fourth central moment for a single person/facet_element/item.

        Calculates sum((k - E[X])^4 * P(X=k)). Used in Wilson-Hilferty
        approximation for standardised fit statistics.

        Parameters
        ----------
        ability : float
            Person ability estimate on the logit scale.
        item : str
            Item identifier.
        difficulties : pandas.Series
            Item difficulty estimates.
        facet_element : str
            Rater identifier.
        severities : Series or dict
            Rater severity parameters.
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
                    ability,
                    item,
                    difficulties,
                    facet_element,
                    severities,
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
    def cat_prob_global(self, a, i, d, r, s, c, t):
        """Alias for cat_prob(..., model='global'). See cat_prob for full documentation."""
        return self.cat_prob(a, i, d, r, s, c, t, "global")

    def cat_prob_items(self, a, i, d, r, s, c, t):
        """Alias for cat_prob(..., model='items'). See cat_prob for full documentation."""
        return self.cat_prob(a, i, d, r, s, c, t, "items")

    def cat_prob_thresholds(self, a, i, d, r, s, c, t):
        """Alias for cat_prob(..., model='thresholds'). See cat_prob for full documentation."""
        return self.cat_prob(a, i, d, r, s, c, t, "thresholds")

    def cat_prob_matrix(self, a, i, d, r, s, c, t):
        """Alias for cat_prob(..., model='matrix'). See cat_prob for full documentation."""
        return self.cat_prob(a, i, d, r, s, c, t, "matrix")

    def exp_score_global(self, a, i, d, r, s, t):
        """Alias for exp_score(..., model='global'). See exp_score for full documentation."""
        return self.exp_score(a, i, d, r, s, t, "global")

    def exp_score_items(self, a, i, d, r, s, t):
        """Alias for exp_score(..., model='items'). See exp_score for full documentation."""
        return self.exp_score(a, i, d, r, s, t, "items")

    def exp_score_thresholds(self, a, i, d, r, s, t):
        """Alias for exp_score(..., model='thresholds'). See exp_score for full documentation."""
        return self.exp_score(a, i, d, r, s, t, "thresholds")

    def exp_score_matrix(self, a, i, d, r, s, t):
        """Alias for exp_score(..., model='matrix'). See exp_score for full documentation."""
        return self.exp_score(a, i, d, r, s, t, "matrix")

    def variance_global(self, a, i, d, r, s, t):
        """Alias for variance(..., model='global'). See variance for full documentation."""
        return self.variance(a, i, d, r, s, t, "global")

    def variance_items(self, a, i, d, r, s, t):
        """Alias for variance(..., model='items'). See variance for full documentation."""
        return self.variance(a, i, d, r, s, t, "items")

    def variance_thresholds(self, a, i, d, r, s, t):
        """Alias for variance(..., model='thresholds'). See variance for full documentation."""
        return self.variance(a, i, d, r, s, t, "thresholds")

    def variance_matrix(self, a, i, d, r, s, t):
        """Alias for variance(..., model='matrix'). See variance for full documentation."""
        return self.variance(a, i, d, r, s, t, "matrix")

    def kurtosis_global(self, a, i, d, r, s, t):
        """Alias for kurtosis(..., model='global'). See kurtosis for full documentation."""
        return self.kurtosis(a, i, d, r, s, t, "global")

    def kurtosis_items(self, a, i, d, r, s, t):
        """Alias for kurtosis(..., model='items'). See kurtosis for full documentation."""
        return self.kurtosis(a, i, d, r, s, t, "items")

    def kurtosis_thresholds(self, a, i, d, r, s, t):
        """Alias for kurtosis(..., model='thresholds'). See kurtosis for full documentation."""
        return self.kurtosis(a, i, d, r, s, t, "thresholds")

    def kurtosis_matrix(self, a, i, d, r, s, t):
        """Alias for kurtosis(..., model='matrix'). See kurtosis for full documentation."""
        return self.kurtosis(a, i, d, r, s, t, "matrix")

    # ------------------------------------------------------------------
    # Vectorised probability engine
    # ------------------------------------------------------------------

    def _cat_probs_mfrm(
        self, abilities, items, facet_elements, thresholds, model, severities
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
        ab = np.asarray(abilities, dtype=float)  # (N,)
        diff_arr = self.items.loc[items].values  # (I,)
        n_items = len(items)

        result = {}
        for facet_element in facet_elements:
            if model == "global":
                # item_offset: scalar, same for all (i)
                item_offset = float(severities.loc[facet_element])
                thresh_offset = np.zeros(len(thresholds) + 1)
            elif model == "items":
                # item_offset: (I,) vector
                item_offset = severities.loc[facet_element, items].values
                thresh_offset = np.zeros(len(thresholds) + 1)
            elif model == "thresholds":
                item_offset = 0.0
                thresh_offset = np.concatenate(
                    [[0.0], np.cumsum(severities.loc[facet_element].values)]
                )
            elif model == "bivector":
                item_offset = 0.0
                thresh_offset = None  # applied per-item below
            elif model == "matrix":
                item_offset = 0.0
                thresh_offset = None  # applied per-item below
            else:
                raise ValueError(f"Unknown model: {model}")

            if model in ("bivector", "matrix"):
                # Build (K+1, N, I) tensor item by item
                log_num = np.zeros((len(thresholds) + 1, len(ab), n_items))
                for j, item in enumerate(items):
                    sev_rik = severities.loc[facet_element, item].values
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

    def item_diffs(
        self, constant=0.1, method="cos", matrix_power=3, log_lik_tol=0.000001
    ):
        """PAIR item difficulty estimation summing across facet_elements."""
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

    def _threshold_distance(self, threshold, difficulties, constant=0.1):
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

        diffs = difficulties.values
        diff_matrix = diffs[:, None] - diffs[None, :]

        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.where(valid, np.log(num_s) - np.log(den_s), 0.0)

        total_weight = weight_matrix.sum()
        if total_weight == 0:
            return np.nan
        return (weight_matrix * (log_ratio + diff_matrix)).sum() / total_weight

    def ra_thresholds(self, difficulties, constant=0.1):
        """CPAT threshold set estimation."""
        distances = [
            self._threshold_distance(k, difficulties, constant)
            for k in range(1, self.max_score)
        ]
        thresholds = np.array([sum(distances[:t]) for t in range(self.max_score)])
        thresholds -= thresholds.mean()
        return thresholds

    # ------------------------------------------------------------------
    # Rater severity estimation
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
        """PAIR facet_element severity estimation — scalar per facet_element."""
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
        """PAIR facet_element severity for a single item (items parameterisation)."""
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
        """PAIR facet_element severity estimation — vector per (facet_element, item)."""
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
        """PAIR facet_element severity for a single threshold (thresholds parameterisation)."""
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
        """PAIR facet_element severity estimation — vector per (facet_element, threshold)."""
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
        """PAIR facet_element severity for a single (item, category) cell (matrix param)."""
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
        """PAIR facet_element severity estimation — full (facet_element, item, threshold) matrix."""
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

        # Marginal severities
        sev_arr = facet_elements  # (R, I, K) — no sentinel to skip
        self.marginal_facet_effects_items = pd.DataFrame(
            sev_arr.mean(axis=2), index=self.facet_names, columns=self.responses.columns
        )
        self.marginal_facet_effects_thresholds = pd.DataFrame(
            sev_arr.mean(axis=1), index=self.facet_names, columns=range(1, self.max_score + 1)
        )

    def _estimate_raters_bivector(self, matrix_marginals=True, **kw):
        """
        Bivector facet_element severity estimation.

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
                  overall facet_element severity lives here)
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
            {facet_element: {item: array}} — reconstructed full severity matrix
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
          1. item_diffs()       — PAIR item difficulties (shared across models)
          2. ra_thresholds()    — CPAT shared thresholds (shared across models)
          3. raters_{model}()   — PAIR facet_element severities (model-specific)

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
        self.thresholds = pd.Series(self.ra_thresholds(self.items, constant=constant))
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
        Anchor calibration: set mean severity of anchors to zero
        and adjust item difficulties and thresholds accordingly.

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
        """Anchor calibration for items parameterisation. Adjusts per-item facet effects and absorbs mean into item difficulties."""
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
        Subtracts the mean anchor facet_element severity (per item, per threshold)
        from all facet_elements, and absorbs it into item difficulties and thresholds.
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
            severity_adjustments = anchor_sev_array.mean(axis=0)  # (I, K)
        else:
            severity_adjustments = adj  # (I, K) pre-computed from full data
        diff_adjustments = severity_adjustments.mean(axis=1)  # (I,)
        threshold_adjustments = severity_adjustments.mean(axis=0)  # (K,)

        for i, item in enumerate(self.responses.columns):
            self.anchor_items_matrix[item] += diff_adjustments[i]
        self.anchor_thresholds_matrix += threshold_adjustments

        sev_adj = sev_array.copy()
        for r in range(self.no_of_facet_elements):
            sev_adj[r, :, :] -= severity_adjustments

        self.anchor_items_matrix -= self.anchor_items_matrix.mean()
        self.anchor_thresholds_matrix -= self.anchor_thresholds_matrix.mean()

        mi = pd.MultiIndex.from_product(
            [self.facet_names, self.responses.columns], names=[self.facet, "item"]
        )
        self.anchor_facet_effects_matrix = pd.DataFrame(
            sev_adj.reshape(-1, self.max_score), index=mi, columns=range(1, self.max_score + 1)
        )

        # Marginal severities (no sentinel to skip)
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

    def _bootstrap_samples(self, no_of_samples):
        """Generate bootstrap person samples preserving facet_element structure."""
        picks = [
            self.responses.index.get_level_values(1)[
                np.random.randint(0, self.no_of_persons, self.no_of_persons)
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
        return [MFRM(s, self.max_score) for s in samples]

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
    ):
        """
        Bootstrap standard errors for item difficulties, thresholds, and
        facet_element severities for the specified model.

        Parameters
        ----------
        store_bootstrap : bool, default False
            If True, store the fitted bootstrap samples as
            self._bootstrap_samples_{model} and set
            self._bootstrap_stored_{model} = True. Allows
            anchor_std_errors() to reuse the same samples without
            rerunning the bootstrap. Memory cost: no_of_samples fitted
            MFRM objects.
        """
        # Pre-compute anchor adjustment from full-data calibration so each
        # bootstrap sample uses a fixed scale shift rather than re-estimating
        # adj from the resample (which would inflate SEs with anchor rater
        # sampling variance).
        adj_fixed = self._extract_anchor_adj(model, anchors) if anchors is not None else None

        samples = self._bootstrap_samples(no_of_samples)
        for s in samples:
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
                [getattr(s, f"anchor_items_{model}").values for s in samples]
            )
            thresh_ests = np.array(
                [getattr(s, f"anchor_thresholds_{model}") for s in samples]
            )
        else:
            item_ests = np.array([s.items.values for s in samples])
            thresh_ests = np.array([s.thresholds.values for s in samples])

        item_se, item_lo, item_hi = self._se_from_bootstrap(
            item_ests, self.responses.columns, interval
        )
        self.item_se = pd.Series(item_se, index=self.responses.columns)
        if item_lo is not None:
            self.item_low = pd.Series(item_lo, index=self.responses.columns)
            self.item_high = pd.Series(item_hi, index=self.responses.columns)

        thr_se, thr_lo, thr_hi = self._se_from_bootstrap(thresh_ests, None, interval)
        setattr(self, f"{prefix}threshold_se_{model}", thr_se)
        setattr(self, f"{prefix}threshold_low_{model}", thr_lo)
        setattr(self, f"{prefix}threshold_high_{model}", thr_hi)

        # Category width SEs
        cat_widths = {
            k + 1: thresh_ests[:, k + 1] - thresh_ests[:, k]
            for k in range(self.max_score - 1)
        }
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
            Raters whose mean severity is anchored to zero. If None,
            falls back to anchor_rater_names_{model} set by calibrate_anchor().
        interval : float or None
            If provided, store percentile CIs at this level (e.g. 0.95).
            If None, inherits the interval used in std_errors() for this
            model. Pass interval=0 to explicitly suppress CIs even if
            std_errors() used one.
        no_of_samples : int
            Number of bootstrap samples. Only used when stored samples are
            not available.
        """
        # Inherit interval from std_errors() if not explicitly provided
        if interval is None:
            interval = getattr(self, f"_bootstrap_interval_{model}", None)

        stored_flag = getattr(self, f"_bootstrap_stored_{model}", False)

        if stored_flag:
            # Fast path: reuse stored calibrated samples
            samples = getattr(self, f"_bootstrap_samples_{model}")
            anchor_raters_used = getattr(self, f"anchor_rater_names_{model}", anchors)
            if anchor_raters_used is None:
                raise ValueError(
                    f"anchors must be provided, or calibrate_anchor() "
                    f'must have been run for model="{model}" so that '
                    f"anchor_rater_names_{model} is available."
                )
            adj_fixed = self._extract_anchor_adj(model, anchor_raters_used)
            for s in samples:
                s.calibrate_anchor(
                    model,
                    anchor_raters_used,
                    constant=constant,
                    method=method,
                    matrix_power=matrix_power,
                    log_lik_tol=log_lik_tol,
                    adj=adj_fixed,
                )
        else:
            # Slow path: full bootstrap rerun
            if anchors is None:
                anchors = getattr(self, f"anchor_rater_names_{model}", None)
            if anchors is None:
                raise ValueError(
                    f"anchors must be provided, or calibrate_anchor() "
                    f'must have been run for model="{model}" so that '
                    f"anchor_rater_names_{model} is available."
                )
            adj_fixed = self._extract_anchor_adj(model, anchors)
            samples = self._bootstrap_samples(no_of_samples)
            for s in samples:
                s.calibrate(
                    model=model,
                    constant=constant,
                    method=method,
                    matrix_power=matrix_power,
                    log_lik_tol=log_lik_tol,
                )
                s.calibrate_anchor(
                    model,
                    anchors,
                    constant=constant,
                    method=method,
                    matrix_power=matrix_power,
                    log_lik_tol=log_lik_tol,
                    adj=adj_fixed,
                )

        # Item difficulty SEs — from anchor_items_{model}
        item_ests = np.array(
            [getattr(s, f"anchor_items_{model}").values for s in samples]
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
            [getattr(s, f"anchor_thresholds_{model}") for s in samples]
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
        difficulties, thresholds, severities = self._get_params(model, anchor)

        if not hasattr(self, f'{"anchor_" if anchor else ""}persons_{model}'):
            self.person_estimates(
                model=model,
                anchor=anchor,
                warm_corr=warm_corr,
                tolerance=tolerance,
                max_iters=max_iters,
                ext_score_adjustment=ext_score_adjustment,
            )
        abilities = getattr(self, f'{"anchor_" if anchor else ""}persons_{model}')

        person_filter = self.responses.notna().astype(float).replace(0, np.nan)

        if not ext_scores:
            scores = sum(
                person_filter.loc[r].sum(axis=1) * self.max_score
                for r in self.facet_names
            )
            total_scores = sum(
                self.responses.loc[r].sum(axis=1) for r in self.facet_names
            )
            abilities = abilities[(total_scores > 0) & (total_scores < scores)]
            person_filter = (
                self.responses.loc[(slice(None), abilities.index), :]
                .notna()
                .astype(float)
                .replace(0, np.nan)
            )

        probs_dict, cats = self._cat_probs_mfrm(
            abilities.values,
            list(self.item_names),
            list(self.facet_names),
            thresholds,
            model,
            severities,
        )
        # Convert to per-category (Rater×Person, Items) DataFrames
        cat_prob_dict = {}
        for cat_idx in range(len(cats)):
            frames = {
                facet_element: pd.DataFrame(
                    probs_dict[facet_element][cat_idx, :, :],
                    index=abilities.index,
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
    # Ability estimation
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
        Newton-Raphson ML ability estimation with optional Warm correction.

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

        difficulties, thresholds, severities = self._get_params(model, anchor)
        difficulties = difficulties.loc[items]

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
                (person_filter.loc[r] * difficulties.values).sum(axis=1)
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
                    severities,
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
                        severities,
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
        """Estimate abilities for all persons; store as self.persons_{model}."""
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

    # Backwards-compatible aliases
    def abil_global(self, persons, anchor=False, items=None, facet_elements=None, **kw):
        """Alias for person(..., model='global'). See person for full documentation."""
        return self.person(
            persons,
            model="global",
            anchor=anchor,
            items=items,
            facet_elements=facet_elements,
            **kw,
        )

    def abil_items(self, persons, anchor=False, items=None, facet_elements=None, **kw):
        """Alias for person(..., model='items'). See person for full documentation."""
        return self.person(
            persons,
            model="items",
            anchor=anchor,
            items=items,
            facet_elements=facet_elements,
            **kw,
        )

    def abil_thresholds(
        self, persons, anchor=False, items=None, facet_elements=None, **kw
    ):
        """Alias for person(..., model='thresholds'). See person for full documentation."""
        return self.person(
            persons,
            model="thresholds",
            anchor=anchor,
            items=items,
            facet_elements=facet_elements,
            **kw,
        )

    def abil_matrix(self, persons, anchor=False, items=None, facet_elements=None, **kw):
        """Alias for person(..., model='matrix'). See person for full documentation."""
        return self.person(
            persons,
            model="matrix",
            anchor=anchor,
            items=items,
            facet_elements=facet_elements,
            **kw,
        )

    def abil_bivector(
        self, persons, anchor=False, items=None, facet_elements=None, **kw
    ):
        """Alias for person(..., model='bivector'). See person for full documentation."""
        return self.person(
            persons,
            model="bivector",
            anchor=anchor,
            items=items,
            facet_elements=facet_elements,
            **kw,
        )

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
        abilities,
        items,
        facet_elements,
        severities,
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
        abilities : pandas.Series
            Current ability estimates, indexed by person.
        items : list
            Item subset to use.
        facet_elements : list
            Rater subset to use.
        severities : Series or dict
            Rater severity parameters (structure depends on model).
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
            abilities.values, items, facet_elements, thresholds, model, severities
        )

        part1 = pd.Series(0.0, index=abilities.index)
        part2 = pd.Series(0.0, index=abilities.index)
        part3 = pd.Series(0.0, index=abilities.index)
        info_sum = pd.Series(0.0, index=abilities.index)

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
        return pd.Series(warm_corr.values, index=abilities.index)

    # Backwards-compatible aliases
    def warm_global(self, abilities, items, facet_elements, severities, pf, **kw):
        """Alias for warm(..., model='global'). See warm for full documentation."""
        return self.warm(
            abilities, items, facet_elements, severities, self.thresholds, pf, "global"
        )

    def warm_items(self, abilities, items, facet_elements, severities, pf, **kw):
        """Alias for warm(..., model='items'). See warm for full documentation."""
        return self.warm(
            abilities, items, facet_elements, severities, self.thresholds, pf, "items"
        )

    def warm_thresholds(self, abilities, items, facet_elements, severities, pf, **kw):
        """Alias for warm(..., model='thresholds'). See warm for full documentation."""
        thr = kw.get("thresholds", self.thresholds)
        return self.warm(
            abilities, items, facet_elements, severities, thr, pf, "thresholds"
        )

    def warm_matrix(self, abilities, items, facet_elements, severities, pf, **kw):
        """Alias for warm(..., model='matrix'). See warm for full documentation."""
        return self.warm(
            abilities, items, facet_elements, severities, self.thresholds, pf, "matrix"
        )

    def warm_bivector(self, abilities, items, facet_elements, severities, pf, **kw):
        """Alias for warm(..., model='bivector'). See warm for full documentation."""
        return self.warm(
            abilities,
            items,
            facet_elements,
            severities,
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
        abilities=None,
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
        persons : list or None, default None
            Subset of persons. None uses all persons.
        abilities : pandas.Series or None, default None
            Ability estimates. If None, uses stored persons_{model}.
        items : list or None, default None
            Item subset. None uses all items.
        facet_elements : list or None, default None
            Rater subset. None uses all facet_elements.

        Returns
        -------
        pandas.Series
            CSEM values indexed by person, in logits.
        """
        difficulties, thresholds, severities = self._get_params(model, anchor)

        if abilities is None:
            abilities = self._get_abils(model, anchor)
        if persons is not None:
            abilities = abilities.loc[persons]
        if items is None:
            items = list(self.item_names)
        if facet_elements is None:
            facet_elements = list(self.facet_names)

        person_data = self.responses.loc[(facet_elements, abilities.index), items]
        person_filter = person_data.notna().astype(float).replace(0, np.nan)

        probs_dict, cats = self._cat_probs_mfrm(
            abilities.values, items, facet_elements, thresholds, model, severities
        )

        info_sum = pd.Series(0.0, index=abilities.index)
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
    # Score-to-ability lookup
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
        Convert a raw total score to an ability estimate via Newton-Raphson ML.

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
            Ability estimate in logits.
        """
        difficulties, thresholds, severities = self._get_params(model, anchor)

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

        difficulties = difficulties.loc[items]
        ext_score = len(items) * len(facet_elements) * self.max_score
        used_score = float(score)
        if used_score == 0:
            used_score = ext_score_adjustment
        elif used_score == ext_score:
            used_score -= ext_score_adjustment

        estimate = (
            log(used_score) - log(ext_score - used_score) + float(difficulties.mean())
        )
        change, iters = 1.0, 0

        while abs(change) > tolerance and iters <= max_iters:
            result = sum(
                self.exp_score(
                    estimate,
                    item,
                    difficulties,
                    facet_element,
                    severities,
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
                    difficulties,
                    facet_element,
                    severities,
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
                    severities,
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
        Build a score-to-ability lookup table for all possible raw scores.

        Estimates the ability corresponding to every possible raw score across
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
            Ability estimate for each possible raw score, indexed by score.
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

    def category_counts_item(self, item, facet_element=None):
        """
        Return response frequency counts for a single item.

        Parameters
        ----------
        item : str
            Item identifier (must be a column in self.responses).
        facet_element : str or None, default None
            If provided, returns counts for that facet_element only.
            If None, aggregates across all facet_elements.

        Returns
        -------
        pandas.Series
            Count of each response category (0 to max_score), indexed by
            category value. Returns None and prints a message if item or
            facet_element is invalid.
        """

        if item not in self.responses.columns:
            warnings.warn(
                f"Invalid item name: {item!r}. Returning None.",
                UserWarning,
                stacklevel=2,
            )
            return None
        if facet_element is None:
            return (
                self.responses[item]
                .value_counts()
                .reindex(range(self.max_score + 1), fill_value=0)
                .astype(int)
            )
        if facet_element not in self.facet_names:
            warnings.warn(
                f"Invalid facet_element name: {facet_element!r}. Returning None.",
                UserWarning,
                stacklevel=2,
            )
            return None
        return (
            self.responses.xs(facet_element)[item]
            .value_counts()
            .reindex(range(self.max_score + 1), fill_value=0)
            .astype(int)
        )

    def category_counts_df(self):
        """
        Build and store response frequency tables across all items.

        Computes two tables: an overall table aggregated across all facet_elements,
        and a per-facet_element breakdown. Both include category counts (0 through
        max_score), total valid responses, and missing responses per item.

        Attributes set
        --------------
        category_counts : pandas.DataFrame
            Overall (all-facet_element) frequency table with items as rows and
            response categories plus Total and Missing as columns.
            A Total row is appended. All values are integers.
        category_counts_raters : pandas.DataFrame
            Per-facet_element frequency table with a (Rater, Item) MultiIndex.
            Same column structure as category_counts.
        """

        # Overall category counts (across all facet_elements)
        cat_counts = {
            item: {
                score: int(self.category_counts_item(item).get(score, 0))
                for score in range(self.max_score + 1)
            }
            for item in self.item_names
        }
        df = pd.DataFrame(cat_counts).T.sort_index(axis=1)
        df["Total"] = self.responses.count()
        df["Missing"] = self.responses.shape[0] - df["Total"]
        df.loc["Total"] = df.sum()
        self.category_counts = df.astype(int)

        # Per-facet_element category counts
        rater_counts = {}
        for facet_element in self.facet_names:
            rater_dict = {
                item: {
                    score: int(
                        self.category_counts_item(item, facet_element).get(score, 0)
                    )
                    for score in range(self.max_score + 1)
                }
                for item in self.item_names
            }
            rdf = pd.DataFrame(rater_dict).T.sort_index(axis=1)
            rdf["Total"] = self.responses.xs(facet_element).count()
            rdf["Missing"] = len(self.responses.xs(facet_element).index) - rdf["Total"]
            rdf.loc["Total"] = rdf.sum()
            rater_counts[facet_element] = rdf

        self.category_counts_facet_elements = pd.concat(
            rater_counts.values(), keys=rater_counts.keys()
        ).astype(int)
        setattr(
            self, f"category_counts_{self.facets}", self.category_counts_facet_elements
        )

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
        """Ensure calibration, abilities, cat_prob_dict and fit matrices exist."""
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
        if not hasattr(self, f"facet_effects_{model}"):
            self.calibrate(model=model, **calib_kw)
        if not hasattr(self, f"persons_{model}"):
            self.person_estimates(model=model, **abil_kw)
        cpd_attr = f"cat_prob_dict_{model}"
        exp_attr = f"exp_score_df_{model}"
        if not hasattr(self, cpd_attr):
            self.category_probability_dict(model=model, **kw)
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
        abilities,
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

        # Expand abilities to (Rater×Person) MultiIndex
        estimates_by_rater = pd.concat(
            {facet_element: abilities for facet_element in self.facet_names},
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
        abilities = getattr(self, f"persons_{model}")
        (outfit_ms, outfit_z, infit_ms, infit_z, pm, exp_pm) = self.item_fit_statistics(
            getattr(self, f"exp_score_df_{model}"),
            getattr(self, f"info_df_{model}"),
            getattr(self, f"kurtosis_df_{model}"),
            getattr(self, f"residual_df_{model}"),
            getattr(self, f"std_residual_df_{model}"),
            abilities,
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

    def threshold_fit_statistics(self, abilities, diff_df_dict):
        """Shared threshold fit statistics (dichotomised ICC approach).
        Mirrors RSM threshold_fit_statistics but with (Rater, Person) MultiIndex
        and nz filter for extreme total scores.
        """
        # Build (Rater×Person, Items) ability DataFrame
        basic_persons_df = pd.DataFrame(
            [
                [abilities[person] for _ in self.responses.columns]
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
            [abilities.loc[self.person_names] - abilities.loc[self.person_names].mean()]
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

    def _diff_df_dict(self, model, difficulties, thresholds, severities):
        """Build the threshold location DataFrame dict for threshold fit stats."""
        diff_df_dict = {}
        for t in range(self.max_score):
            thr_loc = thresholds[t]
            rows = {}
            for facet_element in self.facet_names:
                if model == "global":
                    row = difficulties + thr_loc + float(severities.loc[facet_element])
                elif model == "items":
                    row = difficulties + thr_loc + severities.loc[facet_element]
                elif model == "thresholds":
                    row = difficulties + thr_loc + severities.loc[facet_element, t + 1]
                elif model in ("bivector", "matrix"):
                    row = (
                        difficulties
                        + thr_loc
                        + pd.Series(
                            severities.loc[facet_element].iloc[:, t].values,
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
        difficulties, thresholds, severities = self._get_params(model, anchor=False)
        abilities = getattr(self, f"persons_{model}")
        ddd = self._diff_df_dict(model, difficulties, thresholds, severities)
        results = self.threshold_fit_statistics(abilities, ddd)
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

    # ------------------------------------------------------------------
    # Person fit statistics
    # ------------------------------------------------------------------

    def person_fit_statistics(
        self, info_df, kurtosis_df, residual_df, std_residual_df, abilities, **kw
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
        abilities = getattr(self, f"persons_{model}")
        results = self.person_fit_statistics(
            getattr(self, f"info_df_{model}"),
            getattr(self, f"kurtosis_df_{model}"),
            getattr(self, f"residual_df_{model}"),
            getattr(self, f"std_residual_df_{model}"),
            abilities,
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

    # ------------------------------------------------------------------
    # Test-level fit statistics
    # ------------------------------------------------------------------

    def test_fit_statistics(self, abilities, rsems):
        """Shared test-level separation and reliability statistics."""
        scores = self.responses.unstack(level=0).sum(axis=1)
        max_scores = self.responses.unstack(level=0).count(axis=1) * self.max_score
        abilities = abilities[(scores > 0) & (scores < max_scores)]

        isi = (self.items.var() / (self.item_se**2).mean() - 1) ** 0.5
        item_strata = (4 * isi + 1) / 3
        item_reliability = isi**2 / (1 + isi**2)

        mean_rsem2 = (rsems**2).mean()
        psi = ((np.var(abilities) - mean_rsem2) / mean_rsem2) ** 0.5
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
        abilities = getattr(self, f"persons_{model}")
        rsems = getattr(self, f"rsem_vector_{model}")
        results = self.test_fit_statistics(abilities, rsems)
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

    # ------------------------------------------------------------------
    # Top-level fit_statistics
    # ------------------------------------------------------------------

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
            Warm bias correction for ability estimates.
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
        if not hasattr(self, f"facet_effects_{model}"):
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

    def _ensure_calibrated(self, model, **kw):
        """Lazy-load calibration and abilities. SE computation is handled
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
                    )
                self.anchor_std_errors(model=model, anchors=anchors)
            else:
                self.std_errors(
                    model=model,
                    interval=interval,
                    no_of_samples=no_of_samples,
                    constant=constant,
                    method=method,
                    matrix_power=matrix_power,
                    log_lik_tol=log_lik_tol,
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
        Build and store the item statistics summary table.

        Auto-triggers the full calibration/SE/fit chain if not yet run.
        Stores result as self.item_stats_{model}.

        Parameters
        ----------
        model : str, default 'global'
            Rater parameterisation.
        anchors : list or None, default None
            If provided, uses anchor-calibrated item difficulties.
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
            Bootstrap samples for SE estimation.
        interval : float or None, default None
            CI width; if provided, percentile bound columns included.

        Attributes set
        --------------
        item_stats_{model} : pandas.DataFrame
            Item statistics with items as rows. Always contains Estimate,
            SE, Count, Facility, Infit MS, Outfit MS.
        """

        if full:
            zstd = point_measure_corr = True
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
        )
        if not hasattr(self, f"item_outfit_ms_{model}"):
            self._run_item_fit(model)

        anc = anchors is not None
        difficulties = getattr(self, f"anchor_items_{model}") if anc else self.items
        se = (
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

        stats = pd.DataFrame(index=self.responses.columns)
        stats["Estimate"] = difficulties.round(dp)
        stats["SE"] = se.round(dp)
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

        Attributes set
        --------------
        threshold_stats_{model} : pandas.DataFrame
            Threshold statistics, rows Threshold 1..Threshold max_score.
        """

        if full:
            zstd = disc = point_measure_corr = True
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

        Auto-triggers calibration and person ability estimation if not yet run.
        Stores result as self.person_stats_{model}.

        Parameters
        ----------
        model : str, default 'global'
            Rater parameterisation.
        anchors : list or None, default None
            If provided, uses anchor-calibrated abilities.
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
        )
        if not hasattr(self, f"psi_{model}"):
            self._run_test_fit(model)

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
    ):
        """
        Build and store the facet_element statistics summary table.

        Output structure varies substantially by model:
          global     — one row per facet_element with scalar severity estimate and fit stats.
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
            severities = getattr(self, sev_attr)
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
            stats = pd.DataFrame({"Estimate": severities.round(dp)})
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
            severities = getattr(self, sev_attr)
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
                    sub["Estimate"] = severities[item].values.round(dp)
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
                    sub["Estimate"] = severities.iloc[:, t].values.round(dp)
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
                            sub["Estimate"] = severities.loc[
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
        """Compute class interval index masks from ability values."""
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

    def _severity_item_offset(self, model, severities, facet_element):
        """Return per-item severity offset Series for a given facet_element and model."""
        if model == "global":
            return pd.Series(
                float(severities.loc[facet_element]), index=self.item_names
            )
        elif model == "items":
            return severities.loc[facet_element]
        elif model == "thresholds":
            return pd.Series(
                float(severities.loc[facet_element].mean()), index=self.item_names
            )
        elif model in ("bivector", "matrix"):
            # severities is MultiIndex (facet_element, item) × thresholds
            return severities.loc[facet_element].mean(axis=1)

    def _zero_severities(self, model, severities):
        """Return severities structure identical in shape but all values zero.
        Used to evaluate neutral (severity=0) curves in plotting methods."""
        if model == "global":
            return pd.Series(0.0, index=severities.index)
        elif model == "items":
            return pd.DataFrame(0.0, index=severities.index, columns=severities.columns)
        elif model == "thresholds":
            return pd.DataFrame(0.0, index=severities.index, columns=severities.columns)
        elif model in ("bivector", "matrix"):
            return pd.DataFrame(0.0, index=severities.index, columns=severities.columns)

    def _mean_severities(self, model, severities, facet_element):
        """Return a severities structure where `facet_element` has the mean severity
        across all facet_elements.  Used to plot curves that match obs averaged across
        the full facet_element pool."""
        if model == "global":
            result = severities.copy()
            result[facet_element] = float(severities.mean())
            return result
        elif model == "items":
            result = severities.copy()
            result.loc[facet_element] = severities.mean(axis=0)
            return result
        elif model == "thresholds":
            result = severities.copy()
            result.loc[facet_element] = severities.mean(axis=0)
            return result
        elif model in ("bivector", "matrix"):
            result = severities.copy()
            result.loc[facet_element] = severities.groupby(level=1).mean()
            return result

    def class_intervals(
        self, abilities, items=None, facet_elements=None, shift=0, no_of_classes=5
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

        estimates = abilities.loc[abil_index]

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
        abilities,
        difficulties,
        thresholds,
        severities,
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

        # Build ability DataFrame: (Person, Items)
        abil_df = pd.DataFrame({it: abilities for it in self.responses.columns})
        raw_abil_base = abil_df.copy()
        if item is None:
            for it in self.responses.columns:
                abil_df[it] -= float(difficulties[it])

        # Subtract facet_element severity from ability
        abil_dict = {}
        for r in self.facet_names:
            a = abil_df.copy()
            if facet_element is None:
                sev = self._severity_item_offset(model, severities, r)
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
        abilities,
        difficulties,
        severities,
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

        abil_df = pd.DataFrame({it: abilities for it in self.responses.columns})
        if item is None:
            for it in self.responses.columns:
                abil_df[it] -= float(difficulties[it])

        abil_dict = {}
        for r in self.facet_names:
            a = abil_df.copy()
            if facet_element is None:
                sev = self._severity_item_offset(model, severities, r)
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
                    {"ability": cond_estimates.stack(), "score": cond_df.stack()}
                ).droplevel(level=1)
            else:
                obs_data = pd.DataFrame({"ability": cond_estimates, "score": cond_df})

            masks = self._class_masks(obs_data["ability"], no_of_classes)
            mean_abilities_all.append(
                [
                    obs_data.loc[masks[cg]]["ability"].mean() + shift
                    for cg in class_groups
                ]
            )
            obs_props_all.append(
                [obs_data.loc[masks[cg]]["score"].mean() for cg in class_groups]
            )

        return np.array(mean_abilities_all), np.array(obs_props_all)

    # Backwards-compatible per-model aliases
    def class_intervals_cats_global(
        self, abilities, difficulties, thresholds, severities, **kw
    ):
        """Alias for class_intervals_cats(model='global'). See class_intervals_cats for full documentation."""
        return self.class_intervals_cats(
            abilities, difficulties, thresholds, severities, "global", **kw
        )

    def class_intervals_cats_items(
        self, abilities, difficulties, thresholds, severities, **kw
    ):
        """Alias for class_intervals_cats(model='items'). See class_intervals_cats for full documentation."""
        return self.class_intervals_cats(
            abilities, difficulties, thresholds, severities, "items", **kw
        )

    def class_intervals_cats_thresholds(
        self, abilities, difficulties, thresholds, severities, **kw
    ):
        """Alias for class_intervals_cats(model='thresholds'). See class_intervals_cats for full documentation."""
        return self.class_intervals_cats(
            abilities, difficulties, thresholds, severities, "thresholds", **kw
        )

    def class_intervals_cats_matrix(
        self, abilities, difficulties, thresholds, severities, **kw
    ):
        """Alias for class_intervals_cats(model='matrix'). See class_intervals_cats for full documentation."""
        return self.class_intervals_cats(
            abilities, difficulties, thresholds, severities, "matrix", **kw
        )

    def class_intervals_cats_bivector(
        self, abilities, difficulties, thresholds, severities, **kw
    ):
        """Alias for class_intervals_cats(model='bivector'). See class_intervals_cats for full documentation."""
        return self.class_intervals_cats(
            abilities, difficulties, thresholds, severities, "bivector", **kw
        )

    def class_intervals_thr_global(self, abilities, difficulties, severities, **kw):
        """Alias for class_intervals_thr(model='global'). See class_intervals_thr for full documentation."""
        return self.class_intervals_thr(
            abilities, difficulties, severities, "global", **kw
        )

    def class_intervals_thr_items(self, abilities, difficulties, severities, **kw):
        """Alias for class_intervals_thr(model='items'). See class_intervals_thr for full documentation."""
        return self.class_intervals_thr(
            abilities, difficulties, severities, "items", **kw
        )

    def class_intervals_thr_thresholds(self, abilities, difficulties, severities, **kw):
        """Alias for class_intervals_thr(model='thresholds'). See class_intervals_thr for full documentation."""
        return self.class_intervals_thr(
            abilities, difficulties, severities, "thresholds", **kw
        )

    def class_intervals_thr_matrix(self, abilities, difficulties, severities, **kw):
        """Alias for class_intervals_thr(model='matrix'). See class_intervals_thr for full documentation."""
        return self.class_intervals_thr(
            abilities, difficulties, severities, "matrix", **kw
        )

    def class_intervals_thr_bivector(self, abilities, difficulties, severities, **kw):
        """Alias for class_intervals_thr(model='bivector'). See class_intervals_thr for full documentation."""
        return self.class_intervals_thr(
            abilities, difficulties, severities, "bivector", **kw
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
        central_diff=False,
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
        Core plotting function for ability-function curves (MFRM).
        Shared across all four facet_element parameterisations.
        """
        difficulties, thresholds, severities = self._get_params(model, anchor)

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
                    diff_val = 0.0 if items is None else float(difficulties[items])
                    # Per-threshold severity for models that vary by threshold
                    if facet_elements is None:
                        sev_val = 0.0
                    elif model == "thresholds":
                        sev_val = float(severities.loc[r_sc, t + 1])
                    elif model in ("bivector", "matrix"):
                        item_key = (
                            items if items is not None else list(self.item_names)[0]
                        )
                        sev_val = float(severities.loc[(r_sc, item_key), t + 1])
                    elif model == "items" and items is not None:
                        sev_val = float(
                            self._severity_item_offset(model, severities, r_sc)[items]
                        )
                    else:
                        sev_val = float(
                            self._severity_item_offset(model, severities, r_sc).mean()
                        )
                    xval = diff_val + thresholds[t] + sev_val
                    ax.axvline(x=xval, color="black", linestyle="--")

            if central_diff:
                if items is None:
                    ax.axvline(x=0, color="darkred", linestyle="--")
                else:
                    ax.axvline(
                        x=float(difficulties[items]), color="darkred", linestyle="--"
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
                        estimate, item, difficulties, r, severities, thresholds, model
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
                            estimate, it, difficulties, r, severities, thresholds, model
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
                            estimate, it, difficulties, r, severities, thresholds, model
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
                    # _severity_item_offset expects a scalar facet_element
                    r_scalar = (
                        facet_elements[0]
                        if isinstance(facet_elements, list)
                        else facet_elements
                    )
                    sev_offset = self._severity_item_offset(model, severities, r_scalar)
                    if model == "items" and items is not None:
                        sev_shift = float(sev_offset[items])
                    else:
                        sev_shift = float(sev_offset.mean())
                diff_shift = 0.0 if items is None else float(difficulties[items])

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
                    hi = diff_shift + thresholds[cat_highlight] + sev_shift
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
        central_diff=False,
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
        difficulties, thresholds, severities = self._get_params(model, anchor)
        if facet_element in ("none", "zero"):
            facet_element = None

        abilities_arr = np.arange(-20, 20, 0.1)
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
                                a, item, difficulties, r, severities, thresholds, model
                            )
                            for r in all_raters
                        ]
                    )
                    for a in abilities_arr
                ]
            ).reshape(-1, 1)
        else:
            y = np.array(
                [
                    self.exp_score(
                        a, item, difficulties, r_use, severities, thresholds, model
                    )
                    for a in abilities_arr
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
            x_data=abilities_arr,
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
            central_diff=central_diff,
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
        central_diff=False,
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
        difficulties, thresholds, severities = self._get_params(model, anchor)
        if item in ("none",):
            item = None
        if facet_element in ("none", "zero"):
            facet_element = None

        abilities_arr = np.arange(-20, 20, 0.1)
        r_use = (
            facet_element[0]
            if isinstance(facet_element, list)
            else (
                facet_element
                if facet_element is not None
                else list(self.facet_names)[0]
            )
        )
        # Use mean severities for the neutral curve so it represents a
        # typical facet_element, matching obs proportions averaged across all facet_elements.
        sev_for_curve = (
            severities
            if facet_element is not None
            else self._mean_severities(model, severities, r_use)
        )

        if obs is not None:
            persons_attr = f'{"anchor_" if anchor else ""}persons_{model}'
            if not hasattr(self, persons_attr):
                self.person_estimates(model=model, anchor=anchor)
            abilities = getattr(self, persons_attr)
            xobsdata, yobsdata = self.class_intervals_cats(
                abilities,
                difficulties,
                thresholds,
                severities,
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
                        difficulties,
                        r_use,
                        sev_for_curve,
                        cat,
                        thresholds,
                        model,
                    )
                    for cat in range(self.max_score + 1)
                ]
                for a in abilities_arr
            ]
        )

        return self.plot_data(
            x_data=abilities_arr,
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
            central_diff=central_diff,
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
        central_diff=False,
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
        difficulties, thresholds, severities = self._get_params(model, anchor)
        if item in ("none",):
            item = None
        if facet_element in ("none", "zero"):
            facet_element = None

        abilities_arr = np.arange(-20, 20, 0.1)
        r_use = (
            facet_element[0]
            if isinstance(facet_element, list)
            else (
                facet_element
                if facet_element is not None
                else list(self.facet_names)[0]
            )
        )
        diff_shift = 0.0 if item is None else float(difficulties[item])

        # Neutral threshold CCS: for a neutral plot (no specific facet_element
        # requested) use zero severity so thresholds are placed at diff + tau.
        # For the thresholds model with a specific facet_element, each threshold has its
        # own facet_element severity — use per-threshold values rather than the mean.
        if facet_element is None:
            sev_thresh = np.zeros(self.max_score)
        elif model == "thresholds":
            sev_thresh = severities.loc[r_use].values.astype(float)
        elif model in ("bivector", "matrix"):
            item_key = item if item is not None else list(self.item_names)[0]
            sev_thresh = severities.loc[(r_use, item_key)].values.astype(float)
        elif model == "items" and item is not None:
            sev_thresh = np.full(
                self.max_score,
                float(self._severity_item_offset(model, severities, r_use)[item]),
            )
        else:
            sev_thresh = np.full(
                self.max_score,
                float(self._severity_item_offset(model, severities, r_use).mean()),
            )
        abs_thresh = thresholds + diff_shift + sev_thresh

        xobsdata = yobsdata = np.array(np.nan)
        if obs is not None:
            persons_attr = f'{"anchor_" if anchor else ""}persons_{model}'
            if not hasattr(self, persons_attr):
                self.person_estimates(model=model, anchor=anchor)
            abilities = getattr(self, persons_attr)
            mean_abs, obs_props = self.class_intervals_thr(
                abilities,
                difficulties,
                severities,
                model=model,
                item=item,
                facet_element=facet_element,
                no_of_classes=no_of_classes,
            )
            # mean_abs frame depends on what class_intervals_thr subtracted:
            # - facet_element=None: severity subtracted → need to add it back
            # - facet_element specified: nothing subtracted → no shift needed
            # - item=None: difficulty also subtracted → add that back too
            if facet_element is None:
                # sev_thresh is zero for neutral plot — shift is diff only
                x_shift = float(difficulties.mean()) if item is None else 0.0
                xobsdata = mean_abs + x_shift
            else:
                # For facet_element-specific plot: mean_abs is raw ability, no shift needed.
                # For thresholds/matrix model the per-threshold severity is already
                # absorbed into abs_thresh for the curve; obs stay on raw ability axis.
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
                for a in abilities_arr
            ]
        )

        return self.plot_data(
            x_data=abilities_arr,
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
            central_diff=central_diff,
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
        central_diff=False,
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
        difficulties, thresholds, severities = self._get_params(model, anchor)
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
                    a, item, difficulties, r_use, severities, thresholds, model
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
            central_diff=central_diff,
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
        difficulties, thresholds, severities = self._get_params(model, anchor)
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

        abilities_arr = np.arange(-20, 20, 0.1)
        # Neutral TCC: observed points are mean raw scores on the raw ability
        # axis with no severity adjustment.  When no specific facet_elements are
        # requested, remove each facet_element's severity contribution from the curve
        # so it also represents a zero-severity baseline.
        # Curve: total expected score summed across all facet_element×item combinations.
        y = np.array(
            [
                sum(
                    self.exp_score(
                        a, it, difficulties, r, severities, thresholds, model
                    )
                    for it in item_keys
                    for r in rater_list
                )
                for a in abilities_arr
            ]
        ).reshape(-1, 1)
        y_max = self.max_score * len(item_keys) * len(rater_list)

        return self.plot_data(
            x_data=abilities_arr,
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
        difficulties, thresholds, severities = self._get_params(model, anchor)
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
                    self.variance(a, it, difficulties, r, severities, thresholds, model)
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
        difficulties, thresholds, severities = self._get_params(model, anchor)
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
                    self.variance(a, it, difficulties, r, severities, thresholds, model)
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
