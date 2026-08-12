import warnings

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from raschpy.simulation.base_sim import Rasch_Sim


class MFRM_Sim(Rasch_Sim):
    """
    Simulate polytomous response data for the Many-Facet Rasch Model (MFRM).

    Generates item locations, shared Rasch-Andrich thresholds, person
    locations, and facet_element effect parameters under one of five parameterisations,
    then computes category probabilities and samples scores for every
    facet_element-person-item combination. Simulation runs automatically on instantiation;
    access results via self.responses.

    Parameters
    ----------
    no_of_items : int
        Number of items to simulate.
    no_of_persons : int
        Number of persons to simulate.
    no_of_facet_elements : int
        Number of facet_elements to simulate.
    max_score : int
        Maximum possible score per item (number of categories minus 1).
    model : str, default 'global'
        Rater facet_effect parameterisation. One of:
        'global'     — single scalar facet_effect per facet_element;
        'items'      — separate facet_effect per (facet_element, item);
        'thresholds' — separate facet_effect per (facet_element, threshold);
        'matrix'     — full facet_effect per (facet_element, item, threshold);
        'bivector'   — additive (item effect + threshold effect) facet_effect
                       per (facet_element, item, threshold).
    item_range : float, default 2
        Total spread of item locations in logits.
    facet_range : float, default 2
        Total spread of facet_element effects in logits. Not used when
        model='bivector' — see item_facet_range/threshold_facet_range.
    item_facet_range : float, default 2
        model='bivector' only. Total spread of per-(facet_element, item)
        facet_effect effects across the full facet_element x item matrix in logits.
    threshold_facet_range : float, default 1
        model='bivector' only. Total spread of per-(facet_element, threshold)
        facet_effect effects across the full facet_element x threshold matrix in logits.
    category_base : float, default 1
        Base width of each rating category. Larger values produce wider,
        more ordered categories.
    person_sd : float, default 1.5
        Standard deviation of the person location distribution (normal).
    max_disorder : float, default 0
        Maximum threshold disorder. 0 produces perfectly ordered thresholds.
    offset : float, default 0
        Mean shift applied to person locations after centring.
    missing : float, default 0
        Proportion of responses to set as missing at random, in [0, 1).
    shared_missing : bool, default True
        If True, the same persons are missing across all facet_elements (correlated
        missingness). If False, missingness is independent across facet_elements.
    manual_persons : array-like or None, default None
        Custom person measures. Length must equal no_of_persons.
    manual_items : array-like or None, default None
        Custom item locations. Length must equal no_of_items.
    manual_thresholds : array-like or None, default None
        Custom threshold vector, length max_score. Must satisfy
        sum(thresholds) == 0.
    manual_raters : dict or array-like or None, default None
        Custom facet_element effect parameters. Structure must match the chosen model:
        global — array-like of length no_of_facet_elements;
        items  — {facet_element: {item: float}};
        thresholds — {facet_element: array of length max_score};
        matrix — {facet_element: {item: array of length max_score}}.
        Not supported for model='bivector' — use manual_item_effects/
        manual_threshold_effects instead.
    manual_item_effects : dict or None, default None
        model='bivector' only. Custom per-(facet_element, item) facet_effect
        effects. Structure: {facet_element: {item: float}}.
    manual_threshold_effects : dict or None, default None
        model='bivector' only. Custom per-(facet_element, threshold) facet_effect
        effects. Structure: {facet_element: array of length max_score}.
    manual_person_names : list of str or None, default None
        Custom person labels. If None, labels are 'Person_1', 'Person_2', etc.
    manual_item_names : list of str or None, default None
        Custom item labels. If None, labels are 'Item_1', 'Item_2', etc.
    manual_facet_names : list of str or None, default None
        Custom facet_element labels. If None, labels are '{Facet}_1', '{Facet}_2', etc. where Facet is the capitalised facet name.
    seed : int or None, default None
        Seed for the random number generator. Pass an int for a fully
        reproducible simulation; None (default) draws fresh entropy each run.

    Attributes set
    --------------
    responses : pandas.DataFrame
        Simulated response matrix with (Rater, Person) MultiIndex and items
        as columns. Values are integers in [0, max_score] or NaN (missing).
        This is the primary output — pass directly to MFRM(responses).
    persons : pandas.Series
        True person location parameters, indexed by person.
    items : pandas.Series
        True item location parameters, indexed by item.
    thresholds : numpy.ndarray
        True Rasch-Andrich threshold vector, length max_score,
        zero-sum.
    facet_effects : pandas.Series or dict
        True facet_element effect parameters. Structure depends on model.
    item_effects : pandas.DataFrame
        model='bivector' only. True per-(facet_element, item) facet_effect effects.
    threshold_effects : pandas.DataFrame
        model='bivector' only. True per-(facet_element, threshold) facet_effect
        effects, zero-mean per threshold across facet_elements.
    cat_probs : dict
        {cat: DataFrame} of category probabilities used for simulation.
    person_names : list of str
        Person labels.
    item_names : list of str
        Item labels.
    rater_names : list of str
        Rater labels.
    model : str
        Rater parameterisation used for simulation.
    """

    def __init__(
        self,
        no_of_items,
        no_of_persons,
        no_of_facet_elements,
        max_score,
        model="global",
        item_range=2,
        facet_range=2,
        item_facet_range=2,
        threshold_facet_range=1,
        category_base=1,
        person_sd=1.5,
        max_disorder=0,
        offset=0,
        missing=0,
        shared_missing=True,
        manual_persons=None,
        manual_items=None,
        manual_thresholds=None,
        manual_facet_effects=None,
        manual_raters=None,
        manual_item_effects=None,
        manual_threshold_effects=None,
        manual_person_names=None,
        manual_item_names=None,
        manual_facet_names=None,
        facet="rater",
        facet_plural=None,
        seed=None,
    ):
        """
        Instantiate and run an MFRM simulation.

        See class docstring for full parameter and attribute documentation.
        All simulation output is generated on instantiation and stored as
        instance attributes; see self.responses for the primary output.
        """

        self._rng = np.random.default_rng(seed)

        if model not in ("global", "items", "thresholds", "matrix", "bivector"):
            raise ValueError(
                "model must be one of 'global', 'items', 'thresholds', 'matrix', 'bivector'."
            )
        if model == "bivector" and manual_raters is not None:
            raise ValueError(
                "manual_raters is not supported for model='bivector'; "
                "use manual_item_effects/manual_threshold_effects instead."
            )

        self.model = model
        self.item_facet_range = item_facet_range
        self.threshold_facet_range = threshold_facet_range
        self.no_of_items = int(no_of_items)
        self.no_of_persons = int(no_of_persons)
        # manual_raters is a convenience alias: dict {name: facet_effect_array} → manual_facet_effects
        if manual_raters is not None:
            if manual_facet_effects is not None:
                raise ValueError("Pass manual_raters or manual_facet_effects, not both.")
            if manual_facet_names is None and hasattr(manual_raters, "keys"):
                manual_facet_names = list(manual_raters.keys())
            if no_of_facet_elements is None:
                no_of_facet_elements = len(manual_raters)
            # Conversion to manual_facet_effects is deferred until after item_names are set.

        self.no_of_facet_elements = int(no_of_facet_elements)
        self.facet = facet
        self.facets = facet_plural if facet_plural is not None else facet + "s"
        setattr(self, f"no_of_{self.facets}", self.no_of_facet_elements)
        self.max_score = max_score
        self.item_range = item_range
        self.facet_range = facet_range
        self.category_base = category_base
        self.person_sd = person_sd
        self.max_disorder = max_disorder
        self.offset = offset
        self.missing = missing
        self.shared_missing = shared_missing

        # ------------------------------------------------------------------
        # Persons
        # ------------------------------------------------------------------
        if manual_person_names is not None:
            assert (
                len(manual_person_names) == self.no_of_persons
            ), "Length of person names must match number of persons."
            self.person_names = manual_person_names
        else:
            self.person_names = [f"Person_{p + 1}" for p in range(self.no_of_persons)]

        if manual_persons is not None:
            assert (
                len(manual_persons) == self.no_of_persons
            ), "Length of manual persons must match number of persons."
            locations = np.array(manual_persons)
        else:
            locations = self._rng.normal(0, self.person_sd, self.no_of_persons)
            locations -= locations.mean()
            locations += self.offset

        self.persons = pd.Series(
            {person: loc for person, loc in zip(self.person_names, locations)}
        )

        # ------------------------------------------------------------------
        # Items
        # ------------------------------------------------------------------
        if manual_item_names is not None:
            assert (
                len(manual_item_names) == self.no_of_items
            ), "Length of item names must match number of items."
            self.item_names = manual_item_names
        else:
            self.item_names = [f"Item_{i + 1}" for i in range(self.no_of_items)]

        if manual_items is not None:
            assert (
                len(manual_items) == self.no_of_items
            ), "Length of manual locations must match number of items."
            items = np.array(manual_items)
        else:
            items = self._rng.uniform(0, 1, self.no_of_items)
            items *= self.item_range / (items.max() - items.min())
            items -= items.mean()

        self.items = pd.Series({item: d for item, d in zip(self.item_names, items)})

        # ------------------------------------------------------------------
        # Thresholds (shared RSM structure across all four models)
        # ------------------------------------------------------------------
        if manual_thresholds is not None:
            assert (
                len(manual_thresholds) == self.max_score
            ), "Number of manual thresholds must equal max_score."
            assert np.isclose(sum(manual_thresholds), 0), "Manual thresholds must sum to zero."
            self.thresholds = pd.Series(np.array(manual_thresholds), index=range(1, self.max_score + 1))
        else:
            cat_widths = self._rng.uniform(
                self.max_disorder,
                2 * self.category_base - self.max_disorder,
                self.max_score,
            )
            thresholds = np.array([cat_widths[:k].sum() for k in range(self.max_score)])
            thresholds -= thresholds.mean()
            self.thresholds = pd.Series(thresholds, index=range(1, self.max_score + 1))

        # ------------------------------------------------------------------
        # Raters
        # ------------------------------------------------------------------
        if manual_facet_names is not None:
            assert (
                len(manual_facet_names) == self.no_of_facet_elements
            ), "Length of facet_element names must match number of facet_elements."
            self.facet_names = manual_facet_names
        else:
            self.facet_names = [
                f"{self.facet.capitalize()}_{r + 1}"
                for r in range(self.no_of_facet_elements)
            ]
        self.rater_names = self.facet_names  # alias for default facet
        self.facet_ids = self.facet_names
        self.rater_ids = self.facet_names  # alias for default facet

        # ------------------------------------------------------------------
        # Facet effects (model-specific)
        # ------------------------------------------------------------------
        if manual_raters is not None:
            if self.model == "global":
                if hasattr(manual_raters, "items"):
                    manual_facet_effects = pd.Series(
                        {r: float(v) for r, v in manual_raters.items()}
                    )
                else:
                    manual_facet_effects = pd.Series(
                        np.asarray(manual_raters, dtype=float)
                    )
            elif self.model == "items":
                manual_facet_effects = pd.DataFrame(
                    {
                        r: (
                            np.array([v[item] for item in self.item_names], dtype=float)
                            if isinstance(v, dict)
                            else np.asarray(v, dtype=float)
                        )
                        for r, v in manual_raters.items()
                    },
                    index=self.item_names,
                ).T
            elif self.model == "thresholds":
                manual_facet_effects = pd.DataFrame(
                    {r: np.asarray(v) for r, v in manual_raters.items()},
                    index=range(1, self.max_score + 1),
                ).T
            elif self.model == "matrix":
                I, K = self.no_of_items, self.max_score
                blocks = []
                for r, v in manual_raters.items():
                    if isinstance(v, dict):
                        # {item_name: [threshold_vals]} → (I, K) array ordered by item_names
                        mat = np.array(
                            [v[item] for item in self.item_names], dtype=float
                        )
                    else:
                        arr = np.asarray(v, dtype=float)
                        if arr.ndim == 0:
                            # scalar → global: uniform (I, K)
                            mat = np.full((I, K), float(arr))
                        elif arr.ndim == 1 and len(arr) == I:
                            # (I,) → items: tile over thresholds
                            mat = np.tile(arr[:, None], (1, K))
                        elif arr.ndim == 1 and len(arr) == K:
                            # (K,) → thresholds: tile over items
                            mat = np.tile(arr[None, :], (I, 1))
                        elif arr.ndim == 2 and arr.shape == (I, K):
                            # (I, K) → matrix: use as-is
                            mat = arr
                        elif isinstance(v, (tuple, list)) and len(v) == 2:
                            # (item_vec, thresh_vec) → bivector
                            iv = np.asarray(v[0], dtype=float)
                            tv = np.asarray(v[1], dtype=float)
                            mat = iv[:, None] + tv[None, :]
                        else:
                            raise ValueError(
                                f"manual_raters['{r}']: cannot infer model from shape {arr.shape}. "
                                f"Pass a scalar, dict, (I,) array, (K,) array, (I,K) array, "
                                f"or (item_vec, thresh_vec) tuple."
                            )
                    blocks.append(
                        pd.DataFrame(mat, index=self.item_names, columns=range(1, K + 1))
                    )
                mi = pd.MultiIndex.from_product(
                    [list(manual_raters.keys()), self.item_names],
                    names=["facet_element", "item"],
                )
                manual_facet_effects = pd.DataFrame(
                    np.vstack([b.values for b in blocks]),
                    index=mi,
                    columns=range(1, K + 1),
                )
        self.facet_effects = self._generate_facet_effects(
            manual_facet_effects, manual_item_effects, manual_threshold_effects
        )
        setattr(self, f"{self.facets}_effects", self.facet_effects)

        # ------------------------------------------------------------------
        # Category probabilities
        # ------------------------------------------------------------------
        self.cat_probs = self._compute_cat_probs()

        # ------------------------------------------------------------------
        # Scores + missing data
        # ------------------------------------------------------------------
        scoring_randoms = {
            facet_element: pd.DataFrame(
                self.randoms(), columns=self.item_names, index=self.person_names
            )
            for facet_element in self.facet_names
        }
        scoring_randoms = pd.concat(
            scoring_randoms.values(), keys=scoring_randoms.keys()
        )

        self.responses = sum(
            scoring_randoms
            < sum(self.cat_probs[cat] for cat in range(c, self.max_score + 1))
            for c in range(1, self.max_score + 1)
        )

        if shared_missing:
            missing_randoms = pd.DataFrame(
                self.randoms(), columns=self.item_names, index=self.person_names
            )
            missing_randoms = pd.concat(
                {facet_element: missing_randoms for facet_element in self.facet_names},
                keys=self.facet_names,
            )
        else:
            missing_randoms = pd.concat(
                {
                    facet_element: pd.DataFrame(
                        self.randoms(), columns=self.item_names, index=self.person_names
                    )
                    for facet_element in self.facet_names
                },
                keys=self.facet_names,
            )

        self.responses[missing_randoms < self.missing] = np.nan

    # ------------------------------------------------------------------
    # Facet effect generation
    # ------------------------------------------------------------------

    def _generate_facet_effects(
        self, manual_facet_effects, manual_item_effects=None, manual_threshold_effects=None
    ):
        """Generate or validate facet_element effect parameters for the given model."""

        if self.model == "global":
            if manual_facet_effects is not None:
                assert (
                    len(manual_facet_effects) == self.no_of_facet_elements
                ), "Length of manual facet_effects must match number of facet_elements."
                sev = np.array(manual_facet_effects)
            else:
                sev = truncnorm.rvs(-1.96, 1.96, size=self.no_of_facet_elements, random_state=self._rng)
                sev *= self.facet_range / (sev.max() - sev.min())
                sev -= sev.mean()
            return pd.Series(
                {facet_element: s for facet_element, s in zip(self.facet_names, sev)}
            )

        elif self.model == "items":
            if manual_facet_effects is not None:
                assert (
                    len(manual_facet_effects) == self.no_of_facet_elements
                ), "Length of manual facet_effects must match number of facet_elements."
                return manual_facet_effects
            else:
                sev = np.array(
                    [
                        truncnorm.rvs(-1.96, 1.96, size=self.no_of_items, random_state=self._rng)
                        for _ in range(self.no_of_facet_elements)
                    ]
                )  # (R, I)
                sev *= self.facet_range / (sev.max() - sev.min())
                # Centre per item (column)
                for i in range(self.no_of_items):
                    sev[:, i] -= sev[:, i].mean()
                return pd.DataFrame(
                    sev, index=self.facet_names, columns=self.item_names
                )

        elif self.model == "thresholds":
            if manual_facet_effects is not None:
                assert (
                    len(manual_facet_effects) == self.no_of_facet_elements
                ), "Length of manual facet_effects must match number of facet_elements."
                return manual_facet_effects
            else:
                sev = np.array(
                    [
                        truncnorm.rvs(-1.96, 1.96, size=self.max_score, random_state=self._rng)
                        for _ in range(self.no_of_facet_elements)
                    ]
                )  # (R, K)
                sev *= self.facet_range / (sev.max() - sev.min())
                # Centre per threshold (column), matching _estimate_raters_thresholds'
                # per-threshold independent zero-centring across facet_elements —
                # otherwise a nonzero true per-column mean shows up as a constant
                # per-threshold bias against the (correctly zero-centred) estimates.
                for k in range(self.max_score):
                    sev[:, k] -= sev[:, k].mean()
                return pd.DataFrame(sev, index=self.facet_names, columns=range(1, self.max_score + 1))

        elif self.model == "matrix":
            if manual_facet_effects is not None:
                assert (
                    len(manual_facet_effects.index.get_level_values(0).unique())
                    == self.no_of_facet_elements
                ), "Length of manual facet_effects must match number of facet_elements."
                return manual_facet_effects
            else:
                sev = np.array(
                    [
                        [
                            truncnorm.rvs(-1.96, 1.96, size=self.max_score, random_state=self._rng)
                            for _ in range(self.no_of_items)
                        ]
                        for _ in range(self.no_of_facet_elements)
                    ]
                )  # (R, I, K)
                sev *= self.facet_range / (sev.max() - sev.min())
                # Centre per (item, threshold) cell across facet_elements, matching
                # _estimate_raters_matrix's per-cell independent zero-centring across
                # facet_elements — otherwise a nonzero true per-cell mean shows up as a
                # constant per-(item, threshold) bias against the (correctly
                # zero-centred) estimates.
                for i in range(self.no_of_items):
                    for k in range(self.max_score):
                        sev[:, i, k] -= sev[:, i, k].mean()
                mi = pd.MultiIndex.from_product(
                    [self.facet_names, self.item_names], names=["facet_element", "item"]
                )
                return pd.DataFrame(sev.reshape(-1, self.max_score), index=mi, columns=range(1, self.max_score + 1))

        elif self.model == "bivector":
            # Item effects — (R, I): free mean per facet_element, centred per item.
            if manual_item_effects is not None:
                assert (
                    len(manual_item_effects) == self.no_of_facet_elements
                ), "Length of manual item effects must match number of facet_elements."
                item_effects = manual_item_effects
            else:
                raw = np.array(
                    [
                        truncnorm.rvs(-1.96, 1.96, size=self.no_of_items, random_state=self._rng)
                        for _ in range(self.no_of_facet_elements)
                    ]
                )  # (R, I)
                raw *= self.item_facet_range / (raw.max() - raw.min())
                raw -= raw.mean(axis=0, keepdims=True)
                item_effects = pd.DataFrame(raw, index=self.facet_names, columns=self.item_names)

            # Threshold effects — (R, K): zero-sum per facet_element, centred per threshold
            # across facet_elements — matching the "matrix" branch's per-cell centring
            # above, otherwise a nonzero true per-threshold mean shows up as a constant
            # bias against the (correctly zero-centred) estimates.
            if manual_threshold_effects is not None:
                assert (
                    len(manual_threshold_effects) == self.no_of_facet_elements
                ), "Length of manual threshold effects must match number of facet_elements."
                threshold_effects = manual_threshold_effects
            else:
                raw = np.array(
                    [
                        truncnorm.rvs(-1.96, 1.96, size=self.max_score, random_state=self._rng)
                        for _ in range(self.no_of_facet_elements)
                    ]
                )  # (R, K)
                raw *= self.threshold_facet_range / (raw.max() - raw.min())
                raw -= raw.mean(axis=0, keepdims=True)
                threshold_effects = pd.DataFrame(
                    raw, index=self.facet_names, columns=range(1, self.max_score + 1)
                )

            self.item_effects = item_effects
            self.threshold_effects = threshold_effects

            # Reconstruct full facet_effect matrix, same (facet_element, item) x
            # threshold format as the "matrix" model, so cat_prob/exp_score/etc.
            # can treat bivector and matrix identically downstream.
            mi = pd.MultiIndex.from_product(
                [self.facet_names, self.item_names], names=["facet_element", "item"]
            )
            rows = []
            for facet_element in self.facet_names:
                for item in self.item_names:
                    ie = (
                        item_effects.loc[facet_element, item]
                        if isinstance(item_effects, pd.DataFrame)
                        else item_effects[facet_element][item]
                    )
                    te = (
                        threshold_effects.loc[facet_element].values
                        if isinstance(threshold_effects, pd.DataFrame)
                        else threshold_effects[facet_element]
                    )
                    rows.append(np.asarray(te, dtype=float) + ie)
            return pd.DataFrame(rows, index=mi, columns=range(1, self.max_score + 1))

    # ------------------------------------------------------------------
    # Category probability computation
    # ------------------------------------------------------------------

    def _compute_cat_probs(self):
        """Compute category probability DataFrames for all facet_elements and categories."""

        if self.model == "global":
            c_p_df = pd.DataFrame(
                {item: self.persons - self.items[item] for item in self.item_names}
            )
            cat_probs = {
                cat: {
                    facet_element: (
                        cat * (c_p_df - self.facet_effects.loc[facet_element])
                        - self.thresholds.iloc[:cat].sum()
                    )
                    for facet_element in self.facet_names
                }
                for cat in range(self.max_score + 1)
            }

        elif self.model == "items":
            c_p_df = {
                facet_element: pd.DataFrame(
                    {
                        item: self.persons
                        - self.items[item]
                        - self.facet_effects.loc[facet_element, item]
                        for item in self.item_names
                    }
                )
                for facet_element in self.facet_names
            }
            cat_probs = {
                cat: {
                    facet_element: (
                        cat * c_p_df[facet_element] - self.thresholds.iloc[:cat].sum()
                    )
                    for facet_element in self.facet_names
                }
                for cat in range(self.max_score + 1)
            }

        elif self.model == "thresholds":
            c_p_df = pd.DataFrame(
                {item: self.persons - self.items[item] for item in self.item_names}
            )
            cat_probs = {
                cat: {
                    facet_element: (
                        cat * c_p_df
                        - self.thresholds.iloc[:cat].sum()
                        - self.facet_effects.loc[facet_element].iloc[:cat].sum()
                    )
                    for facet_element in self.facet_names
                }
                for cat in range(self.max_score + 1)
            }

        elif self.model in ("matrix", "bivector"):
            c_p_df = pd.DataFrame(
                {item: self.persons - self.items[item] for item in self.item_names}
            )
            cat_probs = {
                cat: {
                    facet_element: (cat * c_p_df - self.thresholds.iloc[:cat].sum())
                    for facet_element in self.facet_names
                }
                for cat in range(self.max_score + 1)
            }
            # Apply per-(facet_element, item, threshold) facet_effect
            for cat in range(self.max_score + 1):
                for facet_element in self.facet_names:
                    for item in self.item_names:
                        cat_probs[cat][facet_element][item] -= (
                            self.facet_effects.loc[facet_element, item].iloc[:cat].sum()
                        )

        # Concatenate across facet_elements, exponentiate, normalise
        for cat in range(self.max_score + 1):
            cat_probs[cat] = pd.concat(
                cat_probs[cat].values(), keys=cat_probs[cat].keys()
            )
            cat_probs[cat] = np.exp(cat_probs[cat])

        den = sum(cat_probs[cat] for cat in range(self.max_score + 1))
        for cat in range(self.max_score + 1):
            cat_probs[cat] /= den

        return cat_probs

    # ------------------------------------------------------------------
    # Rename utilities
    # ------------------------------------------------------------------

    def rename_rater(self, old, new):
        """
        Rename a single facet_element in the simulated responses DataFrame.

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
            self.rename_raters_all(new_names)

    def rename_raters_all(self, new_names):
        """
        Rename all facet_elements at once.

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
        self.facet_names = self.responses.index.get_level_values(0).unique().tolist()
        self.rater_names = self.facet_names  # alias for default facet
        self.facet_ids = self.facet_names
        self.rater_ids = self.facet_names  # alias for default facet

    def rename_person(self, old, new):
        """
        Rename a single person in the simulated responses DataFrame.

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
        elif new in self.responses.index.get_level_values(1):
            warnings.warn(
                "New person name is a duplicate of an existing person name.",
                UserWarning,
                stacklevel=2,
            )
        if old not in self.responses.index.get_level_values(1):
            warnings.warn(
                f"Old person name {old!r} not found in data.", UserWarning, stacklevel=2
            )
        elif not isinstance(new, str):
            warnings.warn("Person names must be strings.", UserWarning, stacklevel=2)
        else:
            self.responses.rename(index={old: new}, inplace=True)
        self.person_names = self.responses.index.get_level_values(1).unique().tolist()

    def rename_persons_all(self, new_names):
        """
        Rename all persons at once.

        Parameters
        ----------
        new_names : list of str
            New person names in the same order as self.person_names.
        """

        old_names = self.responses.index.get_level_values(1)
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
            self.responses.rename(
                index={old: new for old, new in zip(old_names, new_names)}, inplace=True
            )
        self.person_names = self.responses.index.get_level_values(1).unique().tolist()


# ------------------------------------------------------------------
# Backwards-compatible subclass aliases
# ------------------------------------------------------------------


class MFRM_Sim_Global(MFRM_Sim):
    """
    MFRM simulation — global (scalar) facet_element effect parameterisation.

    Convenience subclass of MFRM_Sim with model='global' fixed.
    Each facet_element has a single scalar facet_effect estimate applied equally
    across all items and thresholds. See MFRM_Sim for full parameter docs.
    """

    def __init__(
        self,
        no_of_items,
        no_of_persons,
        max_score,
        no_of_facet_elements,
        **kw,
    ):
        """Convenience wrapper: MFRM_Sim with model='global' fixed. See MFRM_Sim for full documentation."""
        super().__init__(
            no_of_items,
            no_of_persons,
            no_of_facet_elements=no_of_facet_elements,
            max_score=max_score,
            model="global",
            **kw,
        )


class MFRM_Sim_Items(MFRM_Sim):
    """
    MFRM simulation — items (per facet_element×item) facet_effect parameterisation.

    Convenience subclass of MFRM_Sim with model='items' fixed.
    Each facet_element has a separate facet_effect for each item, constant across
    thresholds. See MFRM_Sim for full parameter docs.
    """

    def __init__(
        self,
        no_of_items,
        no_of_persons,
        max_score,
        no_of_facet_elements,
        **kw,
    ):
        """Convenience wrapper: MFRM_Sim with model='items' fixed. See MFRM_Sim for full documentation."""
        super().__init__(
            no_of_items,
            no_of_persons,
            no_of_facet_elements=no_of_facet_elements,
            max_score=max_score,
            model="items",
            **kw,
        )


class MFRM_Sim_Thresholds(MFRM_Sim):
    """
    MFRM simulation — thresholds (per facet_element×threshold) facet_effect parameterisation.

    Convenience subclass of MFRM_Sim with model='thresholds' fixed.
    Each facet_element has a separate facet_effect for each threshold, constant across
    items. See MFRM_Sim for full parameter docs.
    """

    def __init__(
        self,
        no_of_items,
        no_of_persons,
        max_score,
        no_of_facet_elements,
        **kw,
    ):
        """Convenience wrapper: MFRM_Sim with model='thresholds' fixed. See MFRM_Sim for full documentation."""
        super().__init__(
            no_of_items,
            no_of_persons,
            no_of_facet_elements=no_of_facet_elements,
            max_score=max_score,
            model="thresholds",
            **kw,
        )


class MFRM_Sim_Matrix(MFRM_Sim):
    """
    MFRM simulation — matrix (full facet_element×item×threshold tensor) parameterisation.

    Convenience subclass of MFRM_Sim with model='matrix' fixed.
    Each facet_element has a separate facet_effect for every (item, threshold) combination.
    See MFRM_Sim for full parameter docs.
    """

    def __init__(
        self,
        no_of_items,
        no_of_persons,
        max_score,
        no_of_facet_elements,
        **kw,
    ):
        """Convenience wrapper: MFRM_Sim with model='matrix' fixed. See MFRM_Sim for full documentation."""
        super().__init__(
            no_of_items,
            no_of_persons,
            no_of_facet_elements=no_of_facet_elements,
            max_score=max_score,
            model="matrix",
            **kw,
        )


class MFRM_Sim_Bivector(MFRM_Sim):
    """
    MFRM simulation — bivector (additive item + threshold facet_element effects)
    parameterisation.

    Convenience subclass of MFRM_Sim with model='bivector' fixed. The bivector
    model treats facet_element effect as the sum of two additive components: a
    per-(facet_element, item) item effect and a per-(facet_element, threshold)
    threshold effect. This is analogous to treating the facet_element as an RSM
    (rather than a PCM as in the matrix model) — each facet_element has a location
    profile across items and a shape profile across thresholds, but the two
    are independent and additive.

    True facet_effect for facet_element r, item i, threshold k is:

        facet_effect[r, i, k] = item_effect[r, i] + threshold_effect[r, k]

    Identification constraints:
    - item_effect: free mean per facet_element (overall facet_element effect lives here).
    - threshold_effect: zero-sum per facet_element across thresholds (shape only,
      no net location contribution).

    See MFRM_Sim for full parameter docs, including item_facet_range/
    threshold_facet_range and manual_item_effects/manual_threshold_effects
    (the bivector-specific parameters in place of MFRM_Sim's generic
    facet_range/manual_raters).
    """

    def __init__(
        self,
        no_of_items,
        no_of_persons,
        max_score,
        no_of_facet_elements,
        **kw,
    ):
        """Convenience wrapper: MFRM_Sim with model='bivector' fixed. See MFRM_Sim for full documentation."""
        super().__init__(
            no_of_items,
            no_of_persons,
            no_of_facet_elements=no_of_facet_elements,
            max_score=max_score,
            model="bivector",
            **kw,
        )
        self.model = "bivector"


class MFRM_Sim_Centrality(MFRM_Sim):
    """
    MFRM simulation -- Jin & Wang (2018)-style rater centrality/extremity
    parameterisation (2 true parameters per rater: severity shift lambda_r,
    threshold-stretch omega_r), restricting the fully-free 'thresholds'
    parameterisation. omega_r is Jin & Wang's own notation for this
    threshold-stretch coefficient.

    Thin wrapper: generates a preliminary MFRM_Sim(model='thresholds') to
    get realistic item locations/thresholds/persons, draws true
    lambda_r/omega_r per rater, computes the implied full (rater x threshold)
    severity table via lambda_rk = lambda_r + (omega_r - 1) * tau_k, then
    delegates to the full MFRM_Sim(model='thresholds', manual_raters=...)
    machinery (fixed to the same item/threshold/person values as the
    preliminary sim) for everything else -- category probabilities and
    response sampling. All the actual generative machinery is reused
    unchanged; only lambda_r/omega_r and their implied severity table are new.

    omega_r > 1 spreads reference thresholds apart for that rater
    ("central"); 0 < omega_r < 1 compresses them ("extreme"); omega_r == 1
    is neutral. See MFRM._derive_stretch_model / MFRM.calibrate_centrality
    for the corresponding closed-form estimation side.

    Parameters
    ----------
    global_range : float, default 2
        Total spread (maximum - minimum) of true lambda_r (severity
        shift) values across raters. Renamed from MFRM_Sim's own
        facet_range for this and the other two stretch models -- lambda_r
        is the parameter that collapses this representation to 'global'
        when every rater's omega_r == 1, and 'global_range' pairs
        directly with 'stretch_range' below, matching the global_/stretch
        attribute alias naming (see MFRM._PARAM_ALIASES).
    stretch_range : float, default 1
        Total spread of true omega_r values around 1 (analogous to
        global_range for lambda_r). E.g. stretch_range=1 typically gives
        omega_r roughly in [0.5, 1.5].
    manual_lambda : array-like or None, default None
        Custom true lambda_r values, length no_of_facet_elements. If None,
        drawn the same way MFRM_Sim's own facet_range scales severities.
    manual_omega : array-like or None, default None
        Custom true omega_r values, length no_of_facet_elements. If None,
        drawn as 1 + a stretch_range-scaled, zero-mean deviation.
    seed : int or None, default None
        Seed for both lambda_r/omega_r generation and everything else
        (persons/items/thresholds/response sampling, via the preliminary
        and final MFRM_Sim calls). Pass an int for full reproducibility.

    See MFRM_Sim for all other parameter docs (item_range, person_sd,
    missing, manual_items, manual_thresholds, manual_persons, etc. --
    all forwarded unchanged to both the preliminary and final sims, so
    supplying e.g. manual_items yourself works exactly as it would for
    MFRM_Sim_Thresholds).

    Attributes set (in addition to the usual MFRM_Sim attributes)
    ---------------------------------------------------------------
    lambda_ : pandas.Series
        True generating lambda_r (severity shift) per rater.
    omega : pandas.Series
        True generating omega_r (threshold stretch) per rater.
    """

    def __init__(
        self,
        no_of_items,
        no_of_persons,
        max_score,
        no_of_facet_elements,
        global_range=2,
        stretch_range=1,
        manual_lambda=None,
        manual_omega=None,
        seed=None,
        **kw,
    ):
        if "facet_range" in kw:
            raise TypeError(
                "MFRM_Sim_Centrality.__init__() got an unexpected keyword "
                "argument 'facet_range' -- renamed to 'global_range' (it "
                "controls the spread of true lambda_r values, not "
                "MFRM_Sim's own facet_range, which would otherwise silently "
                "absorb this into the discarded preliminary simulation)."
            )
        prelim = MFRM_Sim(
            no_of_items,
            no_of_persons,
            no_of_facet_elements=no_of_facet_elements,
            max_score=max_score,
            model="thresholds",
            seed=seed,
            **kw,
        )

        rng = np.random.default_rng(seed)
        R = prelim.no_of_facet_elements

        if manual_lambda is not None:
            assert (
                len(manual_lambda) == R
            ), "Length of manual_lambda must match number of facet_elements."
            lam = np.array(manual_lambda, dtype=float)
        else:
            lam = truncnorm.rvs(-1.96, 1.96, size=R, random_state=rng)
            lam *= global_range / (lam.max() - lam.min())
            lam -= lam.mean()

        if manual_omega is not None:
            assert (
                len(manual_omega) == R
            ), "Length of manual_omega must match number of facet_elements."
            omega = np.array(manual_omega, dtype=float)
        else:
            dev = truncnorm.rvs(-1.96, 1.96, size=R, random_state=rng)
            dev *= stretch_range / (dev.max() - dev.min())
            dev -= dev.mean()
            omega = 1.0 + dev

        tau = prelim.thresholds.values
        implied_raters = {
            r: lam[i] + (omega[i] - 1.0) * tau
            for i, r in enumerate(prelim.facet_names)
        }

        # Already captured into prelim (and re-supplied explicitly below) --
        # drop from kw so a caller who passed these directly doesn't hit a
        # "got multiple values for keyword argument" TypeError.
        for key in ("manual_items", "manual_thresholds", "manual_persons",
                    "manual_item_names", "manual_person_names",
                    "manual_facet_names", "manual_raters"):
            kw.pop(key, None)

        super().__init__(
            no_of_items,
            no_of_persons,
            no_of_facet_elements=prelim.no_of_facet_elements,
            max_score=prelim.max_score,
            model="thresholds",
            manual_items=prelim.items.values,
            manual_thresholds=prelim.thresholds.values,
            manual_persons=prelim.persons.values,
            manual_person_names=list(prelim.person_names),
            manual_item_names=list(prelim.item_names),
            manual_facet_names=list(prelim.facet_names),
            manual_raters=implied_raters,
            seed=seed,
            **kw,
        )
        self.model = "centrality"
        self.lambda_ = pd.Series(lam, index=self.facet_names)
        self.omega = pd.Series(omega, index=self.facet_names)

    def __getattr__(self, name):
        """
        Alternate spellings for lambda_/omega, matching MFRM's own
        global_{model}/stretch_{model} attribute aliases (see
        MFRM._PARAM_ALIASES) -- 'global_' for 'lambda_' (no model suffix
        needed here, since this sim class is already model-specific;
        trailing underscore because 'global', like 'lambda', is also a
        Python keyword), 'stretch' for 'omega'. Only triggers when normal
        attribute lookup fails, so it never shadows a real attribute.
        """
        if name == "global_":
            return self.lambda_
        if name == "stretch":
            return self.omega
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


class MFRM_Sim_PseudoHalo(MFRM_Sim):
    """
    MFRM simulation -- (pseudo-)halo-effect parameterisation, the
    item-facet analogue of MFRM_Sim_Centrality (2 true parameters per
    rater: severity shift lambda_r, item-difficulty stretch omega_r),
    restricting the fully-free 'items' parameterisation.

    Thin wrapper: generates a preliminary MFRM_Sim(model='items') to get
    realistic item locations/thresholds/persons, draws true
    lambda_r/omega_r per rater, computes the implied full (rater x item)
    severity table via mu_ri = lambda_r + (omega_r - 1) * delta_i, then
    delegates to the full MFRM_Sim(model='items', manual_raters=...)
    machinery (fixed to the same item/threshold/person values as the
    preliminary sim) for everything else. All the actual generative
    machinery is reused unchanged; only lambda_r/omega_r and their implied
    severity table are new.

    omega_r < 1 compresses the spread of reference item locations toward
    zero for that rater (the pseudo-halo signature -- items look more
    similar in difficulty to that rater than they really are); omega_r > 1
    exaggerates real item-difficulty differences; omega_r == 1 is neutral.
    See MFRM._derive_stretch_model / MFRM.calibrate_pseudo_halo for the
    corresponding closed-form estimation side.

    Parameters
    ----------
    global_range : float, default 2
        Total spread (maximum - minimum) of true lambda_r (severity
        shift) values across raters. Renamed from MFRM_Sim's own
        facet_range for this and the other two stretch models -- lambda_r
        is the parameter that collapses this representation to 'global'
        when every rater's omega_r == 1, and 'global_range' pairs
        directly with 'stretch_range' below, matching the global_/stretch
        attribute alias naming (see MFRM._PARAM_ALIASES).
    stretch_range : float, default 1
        Total spread of true omega_r values around 1 (analogous to
        global_range for lambda_r).
    manual_lambda : array-like or None, default None
        Custom true lambda_r values, length no_of_facet_elements. If None,
        drawn the same way MFRM_Sim's own facet_range scales severities.
    manual_omega : array-like or None, default None
        Custom true omega_r values, length no_of_facet_elements. If None,
        drawn as 1 + a stretch_range-scaled, zero-mean deviation.
    seed : int or None, default None
        Seed for both lambda_r/omega_r generation and everything else, via
        the preliminary and final MFRM_Sim calls. Pass an int for full
        reproducibility.

    See MFRM_Sim for all other parameter docs -- all forwarded unchanged
    to both the preliminary and final sims.

    Attributes set (in addition to the usual MFRM_Sim attributes)
    ---------------------------------------------------------------
    lambda_ : pandas.Series
        True generating lambda_r (severity shift) per rater.
    omega : pandas.Series
        True generating omega_r (item-difficulty stretch) per rater.
    """

    def __init__(
        self,
        no_of_items,
        no_of_persons,
        max_score,
        no_of_facet_elements,
        global_range=2,
        stretch_range=1,
        manual_lambda=None,
        manual_omega=None,
        seed=None,
        **kw,
    ):
        if "facet_range" in kw:
            raise TypeError(
                "MFRM_Sim_PseudoHalo.__init__() got an unexpected keyword "
                "argument 'facet_range' -- renamed to 'global_range' (it "
                "controls the spread of true lambda_r values, not "
                "MFRM_Sim's own facet_range, which would otherwise silently "
                "absorb this into the discarded preliminary simulation)."
            )
        prelim = MFRM_Sim(
            no_of_items,
            no_of_persons,
            no_of_facet_elements=no_of_facet_elements,
            max_score=max_score,
            model="items",
            seed=seed,
            **kw,
        )

        rng = np.random.default_rng(seed)
        R = prelim.no_of_facet_elements

        if manual_lambda is not None:
            assert (
                len(manual_lambda) == R
            ), "Length of manual_lambda must match number of facet_elements."
            lam = np.array(manual_lambda, dtype=float)
        else:
            lam = truncnorm.rvs(-1.96, 1.96, size=R, random_state=rng)
            lam *= global_range / (lam.max() - lam.min())
            lam -= lam.mean()

        if manual_omega is not None:
            assert (
                len(manual_omega) == R
            ), "Length of manual_omega must match number of facet_elements."
            omega = np.array(manual_omega, dtype=float)
        else:
            dev = truncnorm.rvs(-1.96, 1.96, size=R, random_state=rng)
            dev *= stretch_range / (dev.max() - dev.min())
            dev -= dev.mean()
            omega = 1.0 + dev

        delta = prelim.items.values
        implied_raters = {
            r: {
                item: float(lam[i] + (omega[i] - 1.0) * delta[j])
                for j, item in enumerate(prelim.item_names)
            }
            for i, r in enumerate(prelim.facet_names)
        }

        # Already captured into prelim (and re-supplied explicitly below) --
        # drop from kw so a caller who passed these directly doesn't hit a
        # "got multiple values for keyword argument" TypeError.
        for key in ("manual_items", "manual_thresholds", "manual_persons",
                    "manual_item_names", "manual_person_names",
                    "manual_facet_names", "manual_raters"):
            kw.pop(key, None)

        super().__init__(
            no_of_items,
            no_of_persons,
            no_of_facet_elements=prelim.no_of_facet_elements,
            max_score=prelim.max_score,
            model="items",
            manual_items=prelim.items.values,
            manual_thresholds=prelim.thresholds.values,
            manual_persons=prelim.persons.values,
            manual_person_names=list(prelim.person_names),
            manual_item_names=list(prelim.item_names),
            manual_facet_names=list(prelim.facet_names),
            manual_raters=implied_raters,
            seed=seed,
            **kw,
        )
        self.model = "pseudo_halo"
        self.lambda_ = pd.Series(lam, index=self.facet_names)
        self.omega = pd.Series(omega, index=self.facet_names)

    def __getattr__(self, name):
        """
        Alternate spellings for lambda_/omega, matching MFRM's own
        global_{model}/stretch_{model} attribute aliases (see
        MFRM._PARAM_ALIASES) -- 'global_' for 'lambda_' (no model suffix
        needed here, since this sim class is already model-specific;
        trailing underscore because 'global', like 'lambda', is also a
        Python keyword), 'stretch' for 'omega'. Only triggers when normal
        attribute lookup fails, so it never shadows a real attribute.
        """
        if name == "global_":
            return self.lambda_
        if name == "stretch":
            return self.omega
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


class MFRM_Sim_Bistretch(MFRM_Sim):
    """
    MFRM simulation -- "bistretch" parameterisation, a 3-parameter
    (severity shift lambda_r, item-stretch omega_items_r, threshold-stretch
    omega_thresholds_r) restriction of the fully-free 'bivector'
    parameterisation -- combines MFRM_Sim_PseudoHalo's item-stretch axis
    and MFRM_Sim_Centrality's threshold-stretch axis into a single
    facet_element effect.

    Thin wrapper: generates a preliminary MFRM_Sim(model='bivector') to
    get realistic item locations/thresholds/persons, draws true
    lambda_r/omega_items_r/omega_thresholds_r per rater, computes the
    implied item_effects table via mu_ri = lambda_r + (omega_items_r - 1)
    * delta_i (bivector's item axis carries the facet_element's overall
    level, exactly as in MFRM_Sim_PseudoHalo) and the implied
    threshold_effects table via lambda_rk = (omega_thresholds_r - 1) *
    tau_k (bivector's threshold axis is already zero-sum shape-only, so
    no lambda_r term here -- unlike MFRM_Sim_Centrality, where lambda_r
    carries the level because 'thresholds' has no separate item axis to
    carry it instead), then delegates to the full
    MFRM_Sim(model='bivector', manual_item_effects=...,
    manual_threshold_effects=...) machinery (fixed to the same
    item/threshold/person values as the preliminary sim) for everything
    else. All the actual generative machinery is reused unchanged; only
    lambda_r/omega_items_r/omega_thresholds_r and their implied effect
    tables are new.

    See MFRM._derive_stretch_model / MFRM.calibrate_bistretch for the
    corresponding closed-form estimation side.

    Parameters
    ----------
    global_range : float, default 2
        Total spread (maximum - minimum) of true lambda_r (severity
        shift) values across raters. Renamed from MFRM_Sim's own
        facet_range for this and the other two stretch models -- lambda_r
        is the parameter that collapses this representation to 'global'
        when every rater's omega_items_r == omega_thresholds_r == 1, and
        'global_range' pairs directly with 'stretch_range' below,
        matching the global_/stretch_items/stretch_thresholds attribute
        alias naming (see MFRM._PARAM_ALIASES).
    stretch_range : float, default 1
        Total spread of true omega_items_r and omega_thresholds_r values
        around 1 (analogous to global_range for lambda_r); used for both
        axes unless overridden individually.
    manual_lambda : array-like or None, default None
        Custom true lambda_r values, length no_of_facet_elements. If None,
        drawn the same way MFRM_Sim's own facet_range scales severities.
    manual_omega_items : array-like or None, default None
        Custom true omega_items_r values, length no_of_facet_elements. If
        None, drawn as 1 + a stretch_range-scaled, zero-mean deviation.
    manual_omega_thresholds : array-like or None, default None
        Custom true omega_thresholds_r values, length no_of_facet_elements.
        If None, drawn as 1 + a stretch_range-scaled, zero-mean deviation
        (independently of omega_items_r).
    seed : int or None, default None
        Seed for lambda_r/omega_items_r/omega_thresholds_r generation and
        everything else, via the preliminary and final MFRM_Sim calls.
        Pass an int for full reproducibility.

    See MFRM_Sim for all other parameter docs (item_range, person_sd,
    missing, manual_items, manual_thresholds, manual_persons, etc. --
    all forwarded unchanged to both the preliminary and final sims).

    Attributes set (in addition to the usual MFRM_Sim attributes)
    ---------------------------------------------------------------
    lambda_ : pandas.Series
        True generating lambda_r (severity shift) per rater.
    omega_items : pandas.Series
        True generating omega_items_r (item-difficulty stretch) per rater.
    omega_thresholds : pandas.Series
        True generating omega_thresholds_r (threshold stretch) per rater.
    """

    def __init__(
        self,
        no_of_items,
        no_of_persons,
        max_score,
        no_of_facet_elements,
        global_range=2,
        stretch_range=1,
        manual_lambda=None,
        manual_omega_items=None,
        manual_omega_thresholds=None,
        seed=None,
        **kw,
    ):
        if "facet_range" in kw:
            raise TypeError(
                "MFRM_Sim_Bistretch.__init__() got an unexpected keyword "
                "argument 'facet_range' -- renamed to 'global_range' (it "
                "controls the spread of true lambda_r values, not "
                "MFRM_Sim's own facet_range, which would otherwise silently "
                "absorb this into the discarded preliminary simulation)."
            )
        prelim = MFRM_Sim(
            no_of_items,
            no_of_persons,
            no_of_facet_elements=no_of_facet_elements,
            max_score=max_score,
            model="bivector",
            seed=seed,
            **kw,
        )

        rng = np.random.default_rng(seed)
        R = prelim.no_of_facet_elements

        if manual_lambda is not None:
            assert (
                len(manual_lambda) == R
            ), "Length of manual_lambda must match number of facet_elements."
            lam = np.array(manual_lambda, dtype=float)
        else:
            lam = truncnorm.rvs(-1.96, 1.96, size=R, random_state=rng)
            lam *= global_range / (lam.max() - lam.min())
            lam -= lam.mean()

        if manual_omega_items is not None:
            assert (
                len(manual_omega_items) == R
            ), "Length of manual_omega_items must match number of facet_elements."
            omega_items = np.array(manual_omega_items, dtype=float)
        else:
            dev = truncnorm.rvs(-1.96, 1.96, size=R, random_state=rng)
            dev *= stretch_range / (dev.max() - dev.min())
            dev -= dev.mean()
            omega_items = 1.0 + dev

        if manual_omega_thresholds is not None:
            assert (
                len(manual_omega_thresholds) == R
            ), "Length of manual_omega_thresholds must match number of facet_elements."
            omega_thresholds = np.array(manual_omega_thresholds, dtype=float)
        else:
            dev = truncnorm.rvs(-1.96, 1.96, size=R, random_state=rng)
            dev *= stretch_range / (dev.max() - dev.min())
            dev -= dev.mean()
            omega_thresholds = 1.0 + dev

        delta = prelim.items.values
        tau = prelim.thresholds.values

        implied_item_effects = pd.DataFrame(
            [
                lam[i] + (omega_items[i] - 1.0) * delta
                for i in range(R)
            ],
            index=prelim.facet_names,
            columns=prelim.item_names,
        )
        implied_threshold_effects = pd.DataFrame(
            [
                (omega_thresholds[i] - 1.0) * tau
                for i in range(R)
            ],
            index=prelim.facet_names,
            columns=range(1, prelim.max_score + 1),
        )

        # Already captured into prelim (and re-supplied explicitly below) --
        # drop from kw so a caller who passed these directly doesn't hit a
        # "got multiple values for keyword argument" TypeError.
        for key in ("manual_items", "manual_thresholds", "manual_persons",
                    "manual_item_names", "manual_person_names",
                    "manual_facet_names", "manual_item_effects",
                    "manual_threshold_effects"):
            kw.pop(key, None)

        super().__init__(
            no_of_items,
            no_of_persons,
            no_of_facet_elements=prelim.no_of_facet_elements,
            max_score=prelim.max_score,
            model="bivector",
            manual_items=prelim.items.values,
            manual_thresholds=prelim.thresholds.values,
            manual_persons=prelim.persons.values,
            manual_person_names=list(prelim.person_names),
            manual_item_names=list(prelim.item_names),
            manual_facet_names=list(prelim.facet_names),
            manual_item_effects=implied_item_effects,
            manual_threshold_effects=implied_threshold_effects,
            seed=seed,
            **kw,
        )
        self.model = "bistretch"
        self.lambda_ = pd.Series(lam, index=self.facet_names)
        self.omega_items = pd.Series(omega_items, index=self.facet_names)
        self.omega_thresholds = pd.Series(omega_thresholds, index=self.facet_names)

    def __getattr__(self, name):
        """
        Alternate spellings for lambda_/omega_items/omega_thresholds,
        matching MFRM's own global_{model}/stretch_items_{model}/
        stretch_thresholds_{model} attribute aliases (see
        MFRM._PARAM_ALIASES) -- 'global_' for 'lambda_' (no model suffix
        needed here, since this sim class is already model-specific;
        trailing underscore because 'global', like 'lambda', is also a
        Python keyword), 'stretch_items'/'stretch_thresholds' for
        'omega_items'/'omega_thresholds'. Only triggers when normal
        attribute lookup fails, so it never shadows a real attribute.
        """
        if name == "global_":
            return self.lambda_
        if name == "stretch_items":
            return self.omega_items
        if name == "stretch_thresholds":
            return self.omega_thresholds
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )
