import itertools
import warnings
from math import exp, log, sqrt, floor
import statistics

import numpy as np
import pandas as pd
from scipy.stats import truncnorm, norm

from raschpy.simulation.base_sim import Rasch_Sim
from raschpy.mfrm import MFRM


class MFRM_Sim(Rasch_Sim):
    """
    Simulate polytomous response data for the Many-Facet Rasch Model (MFRM).

    Generates item difficulties, shared Rasch-Andrich thresholds, person
    abilities, and rater severity parameters under one of four parameterisations,
    then computes category probabilities and samples scores for every
    rater-person-item combination. Simulation runs automatically on instantiation;
    access results via self.scores.

    Parameters
    ----------
    no_of_items : int
        Number of items to simulate.
    no_of_persons : int
        Number of persons to simulate.
    no_of_raters : int
        Number of raters to simulate.
    max_score : int
        Maximum possible score per item (number of categories minus 1).
    model : str, default 'global'
        Rater severity parameterisation. One of:
        'global'     — single scalar severity per rater;
        'items'      — separate severity per (rater, item);
        'thresholds' — separate severity per (rater, threshold);
        'matrix'     — full severity per (rater, item, threshold).
    item_range : float, default 2
        Total spread of item difficulties in logits.
    rater_range : float, default 2
        Total spread of rater severities in logits.
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
        If True, the same persons are missing across all raters (correlated
        missingness). If False, missingness is independent across raters.
    manual_abilities : array-like or None, default None
        Custom person abilities. Length must equal no_of_persons.
    manual_diffs : array-like or None, default None
        Custom item difficulties. Length must equal no_of_items.
    manual_thresholds : array-like or None, default None
        Custom threshold vector, length max_score + 1. Must satisfy
        thresholds[0] == 0 and sum(thresholds) == 0.
    manual_severities : dict or array-like or None, default None
        Custom rater severity parameters. Structure must match the chosen model:
        global — array-like of length no_of_raters;
        items  — {rater: {item: float}};
        thresholds — {rater: array of length max_score};
        matrix — {rater: {item: array of length max_score}}.
    manual_person_names : list of str or None, default None
        Custom person labels. If None, labels are 'Person_1', 'Person_2', etc.
    manual_item_names : list of str or None, default None
        Custom item labels. If None, labels are 'Item_1', 'Item_2', etc.
    manual_rater_names : list of str or None, default None
        Custom rater labels. If None, labels are 'Rater_1', 'Rater_2', etc.

    Attributes set
    --------------
    scores : pandas.DataFrame
        Simulated response matrix with (Rater, Person) MultiIndex and items
        as columns. Values are integers in [0, max_score] or NaN (missing).
        This is the primary output — pass directly to MFRM(scores).
    abilities : pandas.Series
        True person ability parameters, indexed by person.
    diffs : pandas.Series
        True item difficulty parameters, indexed by item.
    thresholds : numpy.ndarray
        True Rasch-Andrich threshold vector, length max_score + 1,
        with thresholds[0] = 0.
    severities : pandas.Series or dict
        True rater severity parameters. Structure depends on model.
    cat_probs : dict
        {cat: DataFrame} of category probabilities used for simulation.
    persons : list of str
        Person labels.
    items : list of str
        Item labels.
    raters : list of str
        Rater labels.
    model : str
        Rater parameterisation used for simulation.
    """

    def __init__(
        self,
        no_of_items,
        no_of_persons,
        no_of_raters,
        max_score,
        model="global",
        item_range=2,
        rater_range=2,
        category_base=1,
        person_sd=1.5,
        max_disorder=0,
        offset=0,
        missing=0,
        shared_missing=True,
        manual_abilities=None,
        manual_diffs=None,
        manual_thresholds=None,
        manual_severities=None,
        manual_person_names=None,
        manual_item_names=None,
        manual_rater_names=None,
    ):

        if model not in ("global", "items", "thresholds", "matrix"):
            raise ValueError(
                "model must be one of 'global', 'items', 'thresholds', 'matrix'. "
                "For bivector simulation use MFRM_Sim_Bivector directly."
            )

        self.model = model
        self.no_of_items = int(no_of_items)
        self.no_of_persons = int(no_of_persons)
        self.no_of_raters = int(no_of_raters)
        self.max_score = max_score
        self.item_range = item_range
        self.rater_range = rater_range
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
            self.persons = manual_person_names
        else:
            self.persons = [f"Person_{p + 1}" for p in range(self.no_of_persons)]

        if manual_abilities is not None:
            assert (
                len(manual_abilities) == self.no_of_persons
            ), "Length of manual abilities must match number of persons."
            abilities = np.array(manual_abilities)
        else:
            abilities = np.random.normal(0, self.person_sd, self.no_of_persons)
            abilities -= abilities.mean()
            abilities += self.offset

        self.abilities = pd.Series(
            {person: ab for person, ab in zip(self.persons, abilities)}
        )

        # ------------------------------------------------------------------
        # Items
        # ------------------------------------------------------------------
        if manual_item_names is not None:
            assert (
                len(manual_item_names) == self.no_of_items
            ), "Length of item names must match number of items."
            self.items = manual_item_names
        else:
            self.items = [f"Item_{i + 1}" for i in range(self.no_of_items)]

        if manual_diffs is not None:
            assert (
                len(manual_diffs) == self.no_of_items
            ), "Length of manual difficulties must match number of items."
            diffs = np.array(manual_diffs)
        else:
            diffs = np.random.uniform(0, 1, self.no_of_items)
            diffs *= self.item_range / (diffs.max() - diffs.min())
            diffs -= diffs.mean()

        self.diffs = pd.Series({item: d for item, d in zip(self.items, diffs)})

        # ------------------------------------------------------------------
        # Thresholds (shared RSM structure across all four models)
        # ------------------------------------------------------------------
        if manual_thresholds is not None:
            assert (
                len(manual_thresholds) == self.max_score + 1
            ), "Number of manual thresholds must be max score plus 1."
            assert manual_thresholds[0] == 0, "First threshold must be zero."
            assert sum(manual_thresholds) == 0, "Manual thresholds must sum to zero."
            self.thresholds = np.array(manual_thresholds)
        else:
            cat_widths = np.random.uniform(
                self.max_disorder,
                2 * self.category_base - self.max_disorder,
                self.max_score,
            )
            thresholds = np.array([cat_widths[:k].sum() for k in range(self.max_score)])
            thresholds -= thresholds.mean()
            self.thresholds = np.insert(thresholds, 0, 0.0)

        # ------------------------------------------------------------------
        # Raters
        # ------------------------------------------------------------------
        if manual_rater_names is not None:
            assert (
                len(manual_rater_names) == self.no_of_raters
            ), "Length of rater names must match number of raters."
            self.raters = manual_rater_names
        else:
            self.raters = [f"Rater_{r + 1}" for r in range(self.no_of_raters)]

        # ------------------------------------------------------------------
        # Severities (model-specific)
        # ------------------------------------------------------------------
        self.severities = self._generate_severities(manual_severities)

        # ------------------------------------------------------------------
        # Category probabilities
        # ------------------------------------------------------------------
        self.cat_probs = self._compute_cat_probs()

        # ------------------------------------------------------------------
        # Scores + missing data
        # ------------------------------------------------------------------
        scoring_randoms = {
            rater: pd.DataFrame(self.randoms(), columns=self.items, index=self.persons)
            for rater in self.raters
        }
        scoring_randoms = pd.concat(
            scoring_randoms.values(), keys=scoring_randoms.keys()
        )

        self.scores = sum(
            scoring_randoms
            < sum(self.cat_probs[cat] for cat in range(c, self.max_score + 1))
            for c in range(1, self.max_score + 1)
        )

        if shared_missing:
            missing_randoms = pd.DataFrame(
                self.randoms(), columns=self.items, index=self.persons
            )
            missing_randoms = pd.concat(
                {rater: missing_randoms for rater in self.raters}, keys=self.raters
            )
        else:
            missing_randoms = pd.concat(
                {
                    rater: pd.DataFrame(
                        self.randoms(), columns=self.items, index=self.persons
                    )
                    for rater in self.raters
                },
                keys=self.raters,
            )

        self.scores[missing_randoms < self.missing] = np.nan

    # ------------------------------------------------------------------
    # Severity generation
    # ------------------------------------------------------------------

    def _generate_severities(self, manual_severities):
        """Generate or validate rater severity parameters for the given model."""

        if self.model == "global":
            if manual_severities is not None:
                assert (
                    len(manual_severities) == self.no_of_raters
                ), "Length of manual severities must match number of raters."
                sev = np.array(manual_severities)
            else:
                sev = truncnorm.rvs(-1.96, 1.96, size=self.no_of_raters)
                sev *= self.rater_range / (sev.max() - sev.min())
                sev -= sev.mean()
            return pd.Series({rater: s for rater, s in zip(self.raters, sev)})

        elif self.model == "items":
            if manual_severities is not None:
                assert (
                    len(manual_severities) == self.no_of_raters
                ), "Length of manual severities must match number of raters."
                return manual_severities
            else:
                sev = np.array(
                    [
                        truncnorm.rvs(-1.96, 1.96, size=self.no_of_items)
                        for _ in range(self.no_of_raters)
                    ]
                )  # (R, I)
                sev *= self.rater_range / (sev.max() - sev.min())
                # Centre per item (column)
                for i in range(self.no_of_items):
                    sev[:, i] -= sev[:, i].mean()
                return {
                    rater: {item: sev[r, i] for i, item in enumerate(self.items)}
                    for r, rater in enumerate(self.raters)
                }

        elif self.model == "thresholds":
            if manual_severities is not None:
                assert (
                    len(manual_severities) == self.no_of_raters
                ), "Length of manual severities must match number of raters."
                return manual_severities
            else:
                sev = np.array(
                    [
                        truncnorm.rvs(-1.96, 1.96, size=self.max_score)
                        for _ in range(self.no_of_raters)
                    ]
                )  # (R, K)
                sev *= self.rater_range / (sev.max() - sev.min())
                sev -= sev.mean()
                sev = np.insert(sev, 0, 0.0, axis=1)  # prepend zero slot
                return {rater: sev[r, :] for r, rater in enumerate(self.raters)}

        elif self.model == "matrix":
            if manual_severities is not None:
                assert (
                    len(manual_severities) == self.no_of_raters
                ), "Length of manual severities must match number of raters."
                return manual_severities
            else:
                sev = np.array(
                    [
                        [
                            truncnorm.rvs(-1.96, 1.96, size=self.max_score)
                            for _ in range(self.no_of_items)
                        ]
                        for _ in range(self.no_of_raters)
                    ]
                )  # (R, I, K)
                sev *= self.rater_range / (sev.max() - sev.min())
                # Centre per (rater, item) cell
                for r in range(self.no_of_raters):
                    for i in range(self.no_of_items):
                        sev[r, i, :] -= sev[r, i, :].mean()
                sev = np.insert(sev, 0, 0.0, axis=2)  # prepend zero slot
                return {
                    rater: {item: sev[r, i, :] for i, item in enumerate(self.items)}
                    for r, rater in enumerate(self.raters)
                }

    # ------------------------------------------------------------------
    # Category probability computation
    # ------------------------------------------------------------------

    def _compute_cat_probs(self):
        """Compute category probability DataFrames for all raters and categories."""

        if self.model == "global":
            c_p_df = pd.DataFrame(
                {item: self.abilities - self.diffs[item] for item in self.items}
            )
            cat_probs = {
                cat: {
                    rater: (
                        cat * (c_p_df - self.severities[rater])
                        - self.thresholds[: cat + 1].sum()
                    )
                    for rater in self.raters
                }
                for cat in range(self.max_score + 1)
            }

        elif self.model == "items":
            c_p_df = {
                rater: pd.DataFrame(
                    {
                        item: self.abilities
                        - self.diffs[item]
                        - self.severities[rater][item]
                        for item in self.items
                    }
                )
                for rater in self.raters
            }
            cat_probs = {
                cat: {
                    rater: (cat * c_p_df[rater] - self.thresholds[: cat + 1].sum())
                    for rater in self.raters
                }
                for cat in range(self.max_score + 1)
            }

        elif self.model == "thresholds":
            c_p_df = pd.DataFrame(
                {item: self.abilities - self.diffs[item] for item in self.items}
            )
            cat_probs = {
                cat: {
                    rater: (
                        cat * c_p_df
                        - self.thresholds[: cat + 1].sum()
                        - self.severities[rater][: cat + 1].sum()
                    )
                    for rater in self.raters
                }
                for cat in range(self.max_score + 1)
            }

        elif self.model == "matrix":
            c_p_df = pd.DataFrame(
                {item: self.abilities - self.diffs[item] for item in self.items}
            )
            cat_probs = {
                cat: {
                    rater: (cat * c_p_df - self.thresholds[: cat + 1].sum())
                    for rater in self.raters
                }
                for cat in range(self.max_score + 1)
            }
            # Apply per-(rater, item, threshold) severity
            for cat in range(self.max_score + 1):
                for rater in self.raters:
                    for item in self.items:
                        cat_probs[cat][rater][item] -= self.severities[rater][item][
                            : cat + 1
                        ].sum()

        # Concatenate across raters, exponentiate, normalise
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
        Rename a single rater in the simulated scores DataFrame.

        Parameters
        ----------
        old : str
            Current rater name.
        new : str
            Desired new rater name.
        """

        if old == new:
            warnings.warn(
                "New rater name is the same as the old rater name.",
                UserWarning,
                stacklevel=2,
            )
        elif new in self.raters:
            warnings.warn(
                "New rater name is a duplicate of an existing rater name.",
                UserWarning,
                stacklevel=2,
            )
        if old not in self.raters:
            warnings.warn(
                f"Old rater name {old!r} not found in data.", UserWarning, stacklevel=2
            )
        elif not isinstance(new, str):
            warnings.warn("Rater names must be strings.", UserWarning, stacklevel=2)
        else:
            new_names = [new if r == old else r for r in self.raters]
            self.rename_raters_all(new_names)

    def rename_raters_all(self, new_names):
        """
        Rename all raters at once.

        Parameters
        ----------
        new_names : list of str
            New rater names in the same order as self.raters.
        """

        if len(new_names) != len(set(new_names)):
            warnings.warn(
                "List of new rater names contains duplicates.",
                UserWarning,
                stacklevel=2,
            )
        elif len(new_names) != self.no_of_raters:
            warnings.warn(
                f"Incorrect number of rater names: {len(new_names)} provided, "
                f"{self.no_of_raters} raters in data.",
                UserWarning,
                stacklevel=2,
            )
        elif not all(isinstance(n, str) for n in new_names):
            warnings.warn("Rater names must be strings.", UserWarning, stacklevel=2)
        else:
            df_dict = {
                new: self.scores.xs(old) for old, new in zip(self.raters, new_names)
            }
            self.scores = pd.concat(df_dict.values(), keys=df_dict.keys())
        self.raters = self.scores.index.get_level_values(0).unique().tolist()

    def rename_person(self, old, new):
        """
        Rename a single person in the simulated scores DataFrame.

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
        elif new in self.scores.index.get_level_values(1):
            warnings.warn(
                "New person name is a duplicate of an existing person name.",
                UserWarning,
                stacklevel=2,
            )
        if old not in self.scores.index.get_level_values(1):
            warnings.warn(
                f"Old person name {old!r} not found in data.", UserWarning, stacklevel=2
            )
        elif not isinstance(new, str):
            warnings.warn("Person names must be strings.", UserWarning, stacklevel=2)
        else:
            self.scores.rename(index={old: new}, inplace=True)
        self.persons = self.scores.index.get_level_values(1).unique().tolist()

    def rename_persons_all(self, new_names):
        """
        Rename all persons at once.

        Parameters
        ----------
        new_names : list of str
            New person names in the same order as self.persons.
        """

        old_names = self.scores.index.get_level_values(1)
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
            self.scores.rename(
                index={old: new for old, new in zip(old_names, new_names)}, inplace=True
            )
        self.persons = self.scores.index.get_level_values(1).unique().tolist()


# ------------------------------------------------------------------
# Backwards-compatible subclass aliases
# ------------------------------------------------------------------


class MFRM_Sim_Global(MFRM_Sim):
    """
    MFRM simulation — global (scalar) rater severity parameterisation.

    Convenience subclass of MFRM_Sim with model='global' fixed.
    Each rater has a single scalar severity estimate applied equally
    across all items and thresholds. See MFRM_Sim for full parameter docs.
    """

    def __init__(self, no_of_items, no_of_persons, no_of_raters, max_score, **kw):
        super().__init__(
            no_of_items, no_of_persons, no_of_raters, max_score, model="global", **kw
        )


class MFRM_Sim_Items(MFRM_Sim):
    """
    MFRM simulation — items (per rater×item) severity parameterisation.

    Convenience subclass of MFRM_Sim with model='items' fixed.
    Each rater has a separate severity for each item, constant across
    thresholds. See MFRM_Sim for full parameter docs.
    """

    def __init__(self, no_of_items, no_of_persons, no_of_raters, max_score, **kw):
        super().__init__(
            no_of_items, no_of_persons, no_of_raters, max_score, model="items", **kw
        )


class MFRM_Sim_Thresholds(MFRM_Sim):
    """
    MFRM simulation — thresholds (per rater×threshold) severity parameterisation.

    Convenience subclass of MFRM_Sim with model='thresholds' fixed.
    Each rater has a separate severity for each threshold, constant across
    items. See MFRM_Sim for full parameter docs.
    """

    def __init__(self, no_of_items, no_of_persons, no_of_raters, max_score, **kw):
        super().__init__(
            no_of_items,
            no_of_persons,
            no_of_raters,
            max_score,
            model="thresholds",
            **kw,
        )


class MFRM_Sim_Matrix(MFRM_Sim):
    """
    MFRM simulation — matrix (full rater×item×threshold tensor) parameterisation.

    Convenience subclass of MFRM_Sim with model='matrix' fixed.
    Each rater has a separate severity for every (item, threshold) combination.
    See MFRM_Sim for full parameter docs.
    """

    def __init__(self, no_of_items, no_of_persons, no_of_raters, max_score, **kw):
        super().__init__(
            no_of_items, no_of_persons, no_of_raters, max_score, model="matrix", **kw
        )


class MFRM_Sim_Bivector:
    """
    Simulate polytomous response data for the Many-Facet Rasch Model (MFRM)
    under the bivector rater parameterisation.

    The bivector model treats rater severity as the sum of two additive
    components: a per-(rater, item) item effect and a per-(rater, threshold)
    threshold effect. This is analogous to treating the rater as an RSM
    (rather than a PCM as in the matrix model) — each rater has a location
    profile across items and a shape profile across thresholds, but the two
    are independent and additive.

    True severity for rater r, item i, threshold k is:

        severity[r, i, k] = item_effect[r, i] + threshold_effect[r, k]

    Identification constraints:
    - item_effect: free mean per rater (overall rater severity lives here).
    - threshold_effect: zero-sum per rater across thresholds (shape only,
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
        Number of raters to simulate.
    max_score : int
        Maximum possible score per item (number of categories minus 1).
    item_range : float, default 2
        Total spread of item difficulties in logits.
    item_rater_range : float, default 2
        Total spread of per-(rater, item) severity effects across the full
        rater x item matrix in logits.
    threshold_rater_range : float, default 1
        Total spread of per-(rater, threshold) severity effects across the
        full rater x threshold matrix in logits.
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
        If True, the same persons are missing across all raters (correlated
        missingness). If False, missingness is independent across raters.
    manual_abilities : array-like or None, default None
        Custom person abilities. Length must equal no_of_persons.
    manual_diffs : array-like or None, default None
        Custom item difficulties. Length must equal no_of_items.
    manual_thresholds : array-like or None, default None
        Custom threshold vector, length max_score + 1. Must satisfy
        thresholds[0] == 0 and sum(thresholds) == 0.
    manual_item_effects : dict or None, default None
        Custom per-(rater, item) severity effects.
        Structure: {rater: {item: float}}.
    manual_threshold_effects : dict or None, default None
        Custom per-(rater, threshold) severity effects.
        Structure: {rater: array of length max_score + 1}, with
        threshold_effects[rater][0] == 0 and
        sum(threshold_effects[rater]) == 0 for each rater.
    manual_person_names : list of str or None, default None
        Custom person labels.
    manual_item_names : list of str or None, default None
        Custom item labels.
    manual_rater_names : list of str or None, default None
        Custom rater labels.

    Attributes set
    --------------
    All attributes of MFRM_Sim_Matrix, plus:

    item_effects : dict
        True per-(rater, item) severity effects.
        Structure: {rater: {item: float}}.
    threshold_effects : dict
        True per-(rater, threshold) severity effects (zero-sum per rater).
        Structure: {rater: numpy.ndarray of length max_score + 1},
        with threshold_effects[rater][0] = 0 and
        sum(threshold_effects[rater]) = 0 for each rater.
    model : str
        Always 'bivector'.

    Note: self.severities contains the reconstructed full severity matrix
    in {rater: {item: array}} format (item_effect + threshold_effect per
    cell), which is the format used internally by MFRM's probability
    machinery.
    """

    def __init__(
        self,
        no_of_items,
        no_of_persons,
        no_of_raters,
        max_score,
        item_range=2,
        item_rater_range=2,
        threshold_rater_range=1,
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
        manual_rater_names=None,
    ):

        # ------------------------------------------------------------------
        # Resolve names early so severity generation can use them
        # ------------------------------------------------------------------
        raters = (
            manual_rater_names
            if manual_rater_names is not None
            else [f"Rater_{r + 1}" for r in range(no_of_raters)]
        )
        items = (
            manual_item_names
            if manual_item_names is not None
            else [f"Item_{i + 1}" for i in range(no_of_items)]
        )

        # ------------------------------------------------------------------
        # Item effects  {rater: {item: float}}
        # Free mean per rater; centred per item across raters.
        # ------------------------------------------------------------------
        if manual_item_effects is not None:
            assert (
                len(manual_item_effects) == no_of_raters
            ), "Length of manual item effects must match number of raters."
            item_effects = manual_item_effects
        else:
            raw = np.array(
                [
                    truncnorm.rvs(-1.96, 1.96, size=no_of_items)
                    for _ in range(no_of_raters)
                ]
            )  # (R, I)
            raw *= item_rater_range / (raw.max() - raw.min())
            # Centre per item across raters so no item has a systematic
            # cross-rater bias, but leave per-rater means free
            raw -= raw.mean(axis=0, keepdims=True)
            item_effects = {
                rater: {item: raw[r, i] for i, item in enumerate(items)}
                for r, rater in enumerate(raters)
            }

        # ------------------------------------------------------------------
        # Threshold effects  {rater: array of length max_score + 1}
        # Zero-sum per rater; slot 0 is always 0.0 (placeholder).
        # ------------------------------------------------------------------
        if manual_threshold_effects is not None:
            assert (
                len(manual_threshold_effects) == no_of_raters
            ), "Length of manual threshold effects must match number of raters."
            for rater in raters:
                arr = np.array(manual_threshold_effects[rater])
                assert arr[0] == 0.0, f"threshold_effects[{rater!r}][0] must be 0."
                assert (
                    abs(arr.sum()) < 1e-9
                ), f"threshold_effects[{rater!r}] must sum to zero."
            threshold_effects = manual_threshold_effects
        else:
            raw = np.array(
                [
                    truncnorm.rvs(-1.96, 1.96, size=max_score)
                    for _ in range(no_of_raters)
                ]
            )  # (R, K)
            raw *= threshold_rater_range / (raw.max() - raw.min())
            # Zero-sum per rater (row-wise)
            raw -= raw.mean(axis=1, keepdims=True)
            # Prepend zero slot
            raw = np.insert(raw, 0, 0.0, axis=1)  # (R, K+1)
            threshold_effects = {rater: raw[r, :] for r, rater in enumerate(raters)}

        # ------------------------------------------------------------------
        # Reconstruct full severity matrix by summing the two vectors
        # ------------------------------------------------------------------
        manual_severities = {
            rater: {
                item: np.array(
                    [
                        item_effects[rater][item] + threshold_effects[rater][k]
                        for k in range(max_score + 1)
                    ]
                )
                for item in items
            }
            for rater in raters
        }

        # ------------------------------------------------------------------
        # Delegate all score sampling to MFRM_Sim_Matrix
        # ------------------------------------------------------------------
        sim = MFRM_Sim_Matrix(
            no_of_items=no_of_items,
            no_of_persons=no_of_persons,
            no_of_raters=no_of_raters,
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
            manual_severities=manual_severities,
            manual_person_names=manual_person_names,
            manual_item_names=manual_item_names,
            manual_rater_names=manual_rater_names,
        )

        # Copy all MFRM_Sim_Matrix attributes onto self
        self.__dict__.update(sim.__dict__)

        # Add bivector-specific attributes and correct the model label
        self.item_effects = item_effects
        self.threshold_effects = threshold_effects
        self.model = "bivector"
