import warnings

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from raschpy.simulation.base_sim import Rasch_Sim


class MFRM_Sim(Rasch_Sim):
    """
    Simulate polytomous response data for the Many-Facet Rasch Model (MFRM).

    Generates item difficulties, shared Rasch-Andrich thresholds, person
    abilities, and facet_element severity parameters under one of four parameterisations,
    then computes category probabilities and samples scores for every
    facet_element-person-item combination. Simulation runs automatically on instantiation;
    access results via self.responses.

    Parameters
    ----------
    no_of_items : int
        Number of items to simulate.
    no_of_persons : int
        Number of persons to simulate.
    no_of_raters : int
        Number of facet_elements to simulate.
    max_score : int
        Maximum possible score per item (number of categories minus 1).
    model : str, default 'global'
        Rater severity parameterisation. One of:
        'global'     — single scalar severity per facet_element;
        'items'      — separate severity per (facet_element, item);
        'thresholds' — separate severity per (facet_element, threshold);
        'matrix'     — full severity per (facet_element, item, threshold).
    item_range : float, default 2
        Total spread of item difficulties in logits.
    facet_range : float, default 2
        Total spread of facet_element severities in logits.
    category_base : float, default 1
        Base width of each rating category. Larger values produce wider,
        more ordered categories.
    person_sd : float, default 1.5
        Standard deviation of the person ability distribution (normal).
    max_disorder : float, default 0
        Maximum threshold disorder. 0 produces perfectly ordered thresholds.
    offset : float, default 0
        Mean shift applied to person abilities after centring.
    missing : float, default 0
        Proportion of responses to set as missing at random, in [0, 1).
    shared_missing : bool, default True
        If True, the same persons are missing across all facet_elements (correlated
        missingness). If False, missingness is independent across facet_elements.
    manual_abilities : array-like or None, default None
        Custom person abilities. Length must equal no_of_persons.
    manual_diffs : array-like or None, default None
        Custom item difficulties. Length must equal no_of_items.
    manual_thresholds : array-like or None, default None
        Custom threshold vector, length max_score + 1. Must satisfy
        thresholds[0] == 0 and sum(thresholds) == 0.
    manual_raters : dict or array-like or None, default None
        Custom facet_element severity parameters. Structure must match the chosen model:
        global — array-like of length no_of_raters;
        items  — {facet_element: {item: float}};
        thresholds — {facet_element: array of length max_score};
        matrix — {facet_element: {item: array of length max_score}}.
    manual_person_names : list of str or None, default None
        Custom person labels. If None, labels are 'Person_1', 'Person_2', etc.
    manual_item_names : list of str or None, default None
        Custom item labels. If None, labels are 'Item_1', 'Item_2', etc.
    manual_facet_names : list of str or None, default None
        Custom facet_element labels. If None, labels are '{Facet}_1', '{Facet}_2', etc. where Facet is the capitalised facet name.

    Attributes set
    --------------
    responses : pandas.DataFrame
        Simulated response matrix with (Rater, Person) MultiIndex and items
        as columns. Values are integers in [0, max_score] or NaN (missing).
        This is the primary output — pass directly to MFRM(responses).
    persons : pandas.Series
        True person ability parameters, indexed by person.
    items : pandas.Series
        True item difficulty parameters, indexed by item.
    thresholds : numpy.ndarray
        True Rasch-Andrich threshold vector, length max_score + 1,
        with thresholds[0] = 0.
    severities : pandas.Series or dict
        True facet_element severity parameters. Structure depends on model.
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
        no_of_raters=None,
        model="global",
        item_range=2,
        facet_range=2,
        category_base=1,
        person_sd=1.5,
        max_disorder=0,
        offset=0,
        missing=0,
        shared_missing=True,
        manual_abilities=None,
        manual_diffs=None,
        manual_thresholds=None,
        manual_facet_effects=None,
        manual_person_names=None,
        manual_item_names=None,
        manual_facet_names=None,
        facet="rater",
        facet_plural=None,
    ):
        """
        Instantiate and run an MFRM simulation.

        See class docstring for full parameter and attribute documentation.
        All simulation output is generated on instantiation and stored as
        instance attributes; see self.responses for the primary output.
        """

        if model not in ("global", "items", "thresholds", "matrix"):
            raise ValueError(
                "model must be one of 'global', 'items', 'thresholds', 'matrix'. "
                "For bivector simulation use MFRM_Sim_Bivector directly."
            )

        self.model = model
        self.no_of_items = int(no_of_items)
        self.no_of_persons = int(no_of_persons)
        # Resolve no_of_facet_elements / no_of_raters alias
        if no_of_raters is not None and no_of_facet_elements is None:
            no_of_facet_elements = no_of_raters
        elif no_of_raters is not None and no_of_facet_elements is not None:
            raise ValueError("Pass no_of_facet_elements or no_of_raters, not both.")
        self.no_of_facet_elements = int(no_of_facet_elements)
        self.no_of_raters = self.no_of_facet_elements  # alias
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

        if manual_abilities is not None:
            assert (
                len(manual_abilities) == self.no_of_persons
            ), "Length of manual abilities must match number of persons."
            abilities = np.array(manual_abilities)
        else:
            abilities = np.random.normal(0, self.person_sd, self.no_of_persons)
            abilities -= abilities.mean()
            abilities += self.offset

        self.persons = pd.Series(
            {person: ab for person, ab in zip(self.person_names, abilities)}
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

        if manual_diffs is not None:
            assert (
                len(manual_diffs) == self.no_of_items
            ), "Length of manual difficulties must match number of items."
            diffs = np.array(manual_diffs)
        else:
            diffs = np.random.uniform(0, 1, self.no_of_items)
            diffs *= self.item_range / (diffs.max() - diffs.min())
            diffs -= diffs.mean()

        self.items = pd.Series({item: d for item, d in zip(self.item_names, diffs)})

        # ------------------------------------------------------------------
        # Thresholds (shared RSM structure across all four models)
        # ------------------------------------------------------------------
        if manual_thresholds is not None:
            assert (
                len(manual_thresholds) == self.max_score + 1
            ), "Number of manual thresholds must be max score plus 1."
            assert sum(manual_thresholds) == 0, "Manual thresholds must sum to zero."
            self.thresholds = pd.Series(np.array(manual_thresholds))
        else:
            cat_widths = np.random.uniform(
                self.max_disorder,
                2 * self.category_base - self.max_disorder,
                self.max_score,
            )
            thresholds = np.array([cat_widths[:k].sum() for k in range(self.max_score)])
            thresholds -= thresholds.mean()
            self.thresholds = pd.Series(thresholds)

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

        # ------------------------------------------------------------------
        # Severities (model-specific)
        # ------------------------------------------------------------------
        self.facet_effects = self._generate_severities(manual_facet_effects)
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
    # Severity generation
    # ------------------------------------------------------------------

    def _generate_severities(self, manual_facet_effects):
        """Generate or validate facet_element severity parameters for the given model."""

        if self.model == "global":
            if manual_facet_effects is not None:
                assert (
                    len(manual_facet_effects) == self.no_of_facet_elements
                ), "Length of manual severities must match number of facet_elements."
                sev = np.array(manual_facet_effects)
            else:
                sev = truncnorm.rvs(-1.96, 1.96, size=self.no_of_facet_elements)
                sev *= self.facet_range / (sev.max() - sev.min())
                sev -= sev.mean()
            return pd.Series(
                {facet_element: s for facet_element, s in zip(self.facet_names, sev)}
            )

        elif self.model == "items":
            if manual_facet_effects is not None:
                assert (
                    len(manual_facet_effects) == self.no_of_facet_elements
                ), "Length of manual severities must match number of facet_elements."
                return manual_facet_effects
            else:
                sev = np.array(
                    [
                        truncnorm.rvs(-1.96, 1.96, size=self.no_of_items)
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
                ), "Length of manual severities must match number of facet_elements."
                return manual_facet_effects
            else:
                sev = np.array(
                    [
                        truncnorm.rvs(-1.96, 1.96, size=self.max_score)
                        for _ in range(self.no_of_facet_elements)
                    ]
                )  # (R, K)
                sev *= self.facet_range / (sev.max() - sev.min())
                sev -= sev.mean()
                return pd.DataFrame(sev, index=self.facet_names)

        elif self.model == "matrix":
            if manual_facet_effects is not None:
                assert (
                    len(manual_facet_effects.index.get_level_values(0).unique())
                    == self.no_of_facet_elements
                ), "Length of manual severities must match number of facet_elements."
                return manual_facet_effects
            else:
                sev = np.array(
                    [
                        [
                            truncnorm.rvs(-1.96, 1.96, size=self.max_score)
                            for _ in range(self.no_of_items)
                        ]
                        for _ in range(self.no_of_facet_elements)
                    ]
                )  # (R, I, K)
                sev *= self.facet_range / (sev.max() - sev.min())
                # Centre per (facet_element, item) cell
                for r in range(self.no_of_facet_elements):
                    for i in range(self.no_of_items):
                        sev[r, i, :] -= sev[r, i, :].mean()
                mi = pd.MultiIndex.from_product(
                    [self.facet_names, self.item_names], names=["facet_element", "item"]
                )
                return pd.DataFrame(sev.reshape(-1, self.max_score), index=mi)

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

        elif self.model == "matrix":
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
            # Apply per-(facet_element, item, threshold) severity
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
    MFRM simulation — global (scalar) facet_element severity parameterisation.

    Convenience subclass of MFRM_Sim with model='global' fixed.
    Each facet_element has a single scalar severity estimate applied equally
    across all items and thresholds. See MFRM_Sim for full parameter docs.
    """

    def __init__(
        self,
        no_of_items,
        no_of_persons,
        no_of_facet_elements=None,
        max_score=None,
        no_of_raters=None,
        **kw,
    ):
        """Convenience wrapper: MFRM_Sim with model='global' fixed. See MFRM_Sim for full documentation."""
        super().__init__(
            no_of_items,
            no_of_persons,
            no_of_facet_elements=no_of_facet_elements,
            max_score=max_score,
            no_of_raters=no_of_raters,
            model="global",
            **kw,
        )


class MFRM_Sim_Items(MFRM_Sim):
    """
    MFRM simulation — items (per facet_element×item) severity parameterisation.

    Convenience subclass of MFRM_Sim with model='items' fixed.
    Each facet_element has a separate severity for each item, constant across
    thresholds. See MFRM_Sim for full parameter docs.
    """

    def __init__(
        self,
        no_of_items,
        no_of_persons,
        no_of_facet_elements=None,
        max_score=None,
        no_of_raters=None,
        **kw,
    ):
        """Convenience wrapper: MFRM_Sim with model='items' fixed. See MFRM_Sim for full documentation."""
        super().__init__(
            no_of_items,
            no_of_persons,
            no_of_facet_elements=no_of_facet_elements,
            max_score=max_score,
            no_of_raters=no_of_raters,
            model="items",
            **kw,
        )


class MFRM_Sim_Thresholds(MFRM_Sim):
    """
    MFRM simulation — thresholds (per facet_element×threshold) severity parameterisation.

    Convenience subclass of MFRM_Sim with model='thresholds' fixed.
    Each facet_element has a separate severity for each threshold, constant across
    items. See MFRM_Sim for full parameter docs.
    """

    def __init__(
        self,
        no_of_items,
        no_of_persons,
        no_of_facet_elements=None,
        max_score=None,
        no_of_raters=None,
        **kw,
    ):
        """Convenience wrapper: MFRM_Sim with model='thresholds' fixed. See MFRM_Sim for full documentation."""
        super().__init__(
            no_of_items,
            no_of_persons,
            no_of_facet_elements=no_of_facet_elements,
            max_score=max_score,
            no_of_raters=no_of_raters,
            model="thresholds",
            **kw,
        )


class MFRM_Sim_Matrix(MFRM_Sim):
    """
    MFRM simulation — matrix (full facet_element×item×threshold tensor) parameterisation.

    Convenience subclass of MFRM_Sim with model='matrix' fixed.
    Each facet_element has a separate severity for every (item, threshold) combination.
    See MFRM_Sim for full parameter docs.
    """

    def __init__(
        self,
        no_of_items,
        no_of_persons,
        no_of_facet_elements=None,
        max_score=None,
        no_of_raters=None,
        **kw,
    ):
        """Convenience wrapper: MFRM_Sim with model='matrix' fixed. See MFRM_Sim for full documentation."""
        super().__init__(
            no_of_items,
            no_of_persons,
            no_of_facet_elements=no_of_facet_elements,
            max_score=max_score,
            no_of_raters=no_of_raters,
            model="matrix",
            **kw,
        )


class MFRM_Sim_Bivector:
    """
    Simulate polytomous response data for the Many-Facet Rasch Model (MFRM)
    under the bivector facet_element parameterisation.

    The bivector model treats facet_element severity as the sum of two additive
    components: a per-(facet_element, item) item effect and a per-(facet_element, threshold)
    threshold effect. This is analogous to treating the facet_element as an RSM
    (rather than a PCM as in the matrix model) — each facet_element has a location
    profile across items and a shape profile across thresholds, but the two
    are independent and additive.

    True severity for facet_element r, item i, threshold k is:

        severity[r, i, k] = item_effect[r, i] + threshold_effect[r, k]

    Identification constraints:
    - item_effect: free mean per facet_element (overall facet_element severity lives here).
    - threshold_effect: zero-sum per facet_element across thresholds (shape only,
      no net location contribution).

    Score sampling, missing data, category probabilities, and all rename
    utilities are delegated to MFRM_Sim_Matrix via the reconstructed full
    severity matrix. All attributes of MFRM_Sim_Matrix are available on
    this object, plus item_effects and threshold_effects.

    Parameters
    ----------
    no_of_items : int
        Number of items to simulate.
    no_of_persons : int
        Number of persons to simulate.
    no_of_raters : int
        Number of facet_elements to simulate.
    max_score : int
        Maximum possible score per item (number of categories minus 1).
    item_range : float, default 2
        Total spread of item difficulties in logits.
    item_facet_range : float, default 2
        Total spread of per-(facet_element, item) severity effects across the full
        facet_element x item matrix in logits.
    threshold_facet_range : float, default 1
        Total spread of per-(facet_element, threshold) severity effects across the
        full facet_element x threshold matrix in logits.
    category_base : float, default 1
        Base width of each rating category. Larger values produce wider,
        more ordered categories.
    person_sd : float, default 1.5
        Standard deviation of the person ability distribution (normal).
    max_disorder : float, default 0
        Maximum threshold disorder. 0 produces perfectly ordered thresholds.
    offset : float, default 0
        Mean shift applied to person abilities after centring.
    missing : float, default 0
        Proportion of responses to set as missing at random, in [0, 1).
    shared_missing : bool, default True
        If True, the same persons are missing across all facet_elements (correlated
        missingness). If False, missingness is independent across facet_elements.
    manual_abilities : array-like or None, default None
        Custom person abilities. Length must equal no_of_persons.
    manual_diffs : array-like or None, default None
        Custom item difficulties. Length must equal no_of_items.
    manual_thresholds : array-like or None, default None
        Custom threshold vector, length max_score + 1. Must satisfy
        thresholds[0] == 0 and sum(thresholds) == 0.
    manual_item_effects : dict or None, default None
        Custom per-(facet_element, item) severity effects.
        Structure: {facet_element: {item: float}}.
    manual_threshold_effects : dict or None, default None
        Custom per-(facet_element, threshold) severity effects.
        Structure: {facet_element: array of length max_score + 1}, with
        threshold_effects[facet_element][0] == 0 and
        sum(threshold_effects[facet_element]) == 0 for each facet_element.
    manual_person_names : list of str or None, default None
        Custom person labels.
    manual_item_names : list of str or None, default None
        Custom item labels.
    manual_facet_names : list of str or None, default None
        Custom facet_element labels.

    Attributes set
    --------------
    All attributes of MFRM_Sim_Matrix, plus:

    item_effects : dict
        True per-(facet_element, item) severity effects.
        Structure: {facet_element: {item: float}}.
    threshold_effects : dict
        True per-(facet_element, threshold) severity effects (zero-sum per facet_element).
        Structure: {facet_element: numpy.ndarray of length max_score + 1},
        with threshold_effects[facet_element][0] = 0 and
        sum(threshold_effects[facet_element]) = 0 for each facet_element.
    model : str
        Always 'bivector'.

    Note: self.facet_effects contains the reconstructed full severity matrix
    in {facet_element: {item: array}} format (item_effect + threshold_effect per
    cell), which is the format used internally by MFRM's probability
    machinery.
    """

    def __init__(
        self,
        no_of_items,
        no_of_persons,
        no_of_facet_elements=None,
        max_score=None,
        no_of_raters=None,
        item_range=2,
        item_facet_range=2,
        threshold_facet_range=1,
        category_base=1,
        person_sd=1.5,
        max_disorder=0,
        offset=0,
        missing=0,
        shared_missing=True,
        manual_abilities=None,
        manual_diffs=None,
        manual_thresholds=None,
        manual_item_effects=None,
        manual_threshold_effects=None,
        manual_person_names=None,
        manual_item_names=None,
        manual_facet_names=None,
        facet="rater",
        facet_plural=None,
    ):
        """
        Instantiate and run an MFRM bivector simulation.

        See class docstring for full parameter and attribute documentation.
        All simulation output is generated on instantiation and stored as
        instance attributes; see self.responses for the primary output.
        """

        # ------------------------------------------------------------------
        # Resolve no_of_facet_elements / no_of_raters alias
        # ------------------------------------------------------------------
        if no_of_raters is not None and no_of_facet_elements is None:
            no_of_facet_elements = no_of_raters
        elif no_of_raters is not None and no_of_facet_elements is not None:
            raise ValueError("Pass no_of_facet_elements or no_of_raters, not both.")
        # ------------------------------------------------------------------
        # Resolve names early so severity generation can use them
        # ------------------------------------------------------------------
        facet_elements = (
            manual_facet_names
            if manual_facet_names is not None
            else [f"{facet.capitalize()}_{r + 1}" for r in range(no_of_facet_elements)]
        )
        items = (
            manual_item_names
            if manual_item_names is not None
            else [f"Item_{i + 1}" for i in range(no_of_items)]
        )

        # ------------------------------------------------------------------
        # Item effects — (R, I) DataFrame
        # Free mean per facet_element; centred per item across facet_elements.
        # ------------------------------------------------------------------
        if manual_item_effects is not None:
            assert (
                len(manual_item_effects) == no_of_facet_elements
            ), "Length of manual item effects must match number of facet_elements."
            item_effects = manual_item_effects
        else:
            raw = np.array(
                [
                    truncnorm.rvs(-1.96, 1.96, size=no_of_items)
                    for _ in range(no_of_facet_elements)
                ]
            )  # (R, I)
            raw *= item_facet_range / (raw.max() - raw.min())
            raw -= raw.mean(axis=0, keepdims=True)
            item_effects = pd.DataFrame(raw, index=facet_elements, columns=items)

        # ------------------------------------------------------------------
        # Threshold effects — (R, K+1) DataFrame, zero-sum per facet_element
        # ------------------------------------------------------------------
        if manual_threshold_effects is not None:
            assert (
                len(manual_threshold_effects) == no_of_facet_elements
            ), "Length of manual threshold effects must match number of facet_elements."
            threshold_effects = manual_threshold_effects
        else:
            raw = np.array(
                [
                    truncnorm.rvs(-1.96, 1.96, size=max_score)
                    for _ in range(no_of_facet_elements)
                ]
            )  # (R, K)
            raw *= threshold_facet_range / (raw.max() - raw.min())
            raw -= raw.mean(axis=1, keepdims=True)
            threshold_effects = pd.DataFrame(raw, index=facet_elements)

        # ------------------------------------------------------------------
        # Reconstruct full severity matrix as MultiIndex DataFrame
        # ------------------------------------------------------------------
        mi = pd.MultiIndex.from_product(
            [facet_elements, items], names=["facet_element", "item"]
        )
        rows = []
        for facet_element in facet_elements:
            for item in items:
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
                row = np.array([ie + te[k] for k in range(max_score)])
                rows.append(row)
        manual_facet_effects = pd.DataFrame(rows, index=mi)

        # ------------------------------------------------------------------
        # Delegate all score sampling to MFRM_Sim_Matrix
        # ------------------------------------------------------------------
        sim = MFRM_Sim_Matrix(
            no_of_items=no_of_items,
            no_of_persons=no_of_persons,
            no_of_facet_elements=no_of_facet_elements,
            max_score=max_score,
            item_range=item_range,
            category_base=category_base,
            person_sd=person_sd,
            max_disorder=max_disorder,
            offset=offset,
            missing=missing,
            shared_missing=shared_missing,
            manual_abilities=manual_abilities,
            manual_diffs=manual_diffs,
            manual_thresholds=manual_thresholds,
            manual_facet_effects=manual_facet_effects,
            manual_person_names=manual_person_names,
            manual_item_names=manual_item_names,
            # Use the resolved facet_elements as facet names so they match
            # the keys in manual_facet_effects
            manual_facet_names=manual_facet_names or facet_elements,
            facet=facet,
            facet_plural=facet_plural,
        )

        # Copy all MFRM_Sim_Matrix attributes onto self
        self.__dict__.update(sim.__dict__)

        # Add bivector-specific attributes and correct the model label
        self.item_effects = item_effects
        self.threshold_effects = threshold_effects
        self.model = "bivector"
