import numpy as np
import pandas as pd

from raschpy.simulation.base_sim import Rasch_Sim


class RSM_Sim(Rasch_Sim):
    """
    Simulate polytomous response data according to the Rating Scale Model (RSM).

    Generates item difficulties, shared Rasch-Andrich thresholds, and person
    locations, then computes category probabilities and samples scores. All items
    share the same threshold structure. Simulation runs automatically on
    instantiation; access results via self.responses.

    Parameters
    ----------
    no_of_items : int
        Number of items to simulate.
    no_of_persons : int
        Number of persons to simulate.
    max_score : int
        Maximum possible score per item (number of categories minus 1).
    item_range : float, default 3
        Total spread of item difficulties in logits.
    category_base : float, default 1
        Base width of each rating category. Larger values produce wider,
        more ordered categories.
    person_sd : float, default 1.5
        Standard deviation of the person location distribution (normal).
    max_disorder : float, default 0
        Maximum threshold disorder allowed. 0 produces perfectly ordered
        thresholds; values > 0 introduce random disordering up to this limit.
    offset : float, default 0
        Mean shift applied to person locations after centring.
    missing : float, default 0
        Proportion of responses to set as missing at random, in [0, 1).
    manual_persons : array-like or None, default None
        Custom person measures. Length must equal no_of_persons.
    manual_items : array-like or None, default None
        Custom item difficulties. Length must equal no_of_items.
    manual_thresholds : array-like or None, default None
        Custom threshold vector of length max_score. Must satisfy:
        sum(thresholds) == 0.
    manual_person_names : list of str or None, default None
        Custom person labels. If None, labels are 'Person_1', 'Person_2', etc.
    manual_item_names : list of str or None, default None
        Custom item labels. If None, labels are 'Item_1', 'Item_2', etc.
    seed : int or None, default None
        Seed for the random number generator. Pass an int for a fully
        reproducible simulation; None (default) draws fresh entropy each run.

    Attributes set
    --------------
    responses : pandas.DataFrame
        Simulated response matrix, shape (no_of_persons, no_of_items).
        Values are integers in [0, max_score] or NaN (missing).
    persons : pandas.Series
        True person location parameters, indexed by person.
    items : pandas.Series
        True item difficulty parameters, indexed by item.
    thresholds : numpy.ndarray
        True Rasch-Andrich threshold vector, length max_score,
        zero-sum.
    cat_probs : dict
        {cat: DataFrame} of category probabilities used for simulation.
    person_names : list of str
        Person labels.
    item_names : list of str
        Item labels.
    no_of_items : int
        Number of items.
    no_of_persons : int
        Number of persons.
    max_score : int
        Maximum score per item.
    """

    def __init__(
        self,
        no_of_items,
        no_of_persons,
        max_score,
        item_range=3,
        category_base=1,
        person_sd=1.5,
        max_disorder=0,
        offset=0,
        missing=0,
        manual_persons=None,
        manual_items=None,
        manual_thresholds=None,
        manual_person_names=None,
        manual_item_names=None,
        seed=None,
    ):
        """
        Instantiate and run an RSM simulation.

        See class docstring for full parameter and attribute documentation.
        All simulation output is generated on instantiation and stored as
        instance attributes; see self.responses for the primary output.
        """

        self._rng = np.random.default_rng(seed)
        self.no_of_items = int(no_of_items)
        self.no_of_persons = int(no_of_persons)
        self.item_range = item_range
        self.max_score = max_score
        self.category_base = category_base
        self.person_sd = person_sd
        self.max_disorder = max_disorder
        self.offset = offset
        self.missing = missing
        self.persons = manual_persons
        self.items = manual_items
        self.thresholds = manual_thresholds
        self.person_names = manual_person_names
        self.item_names = manual_item_names
        self._dummy_df = pd.DataFrame([self.max_score])

        # Generate person, item, and threshold parameters

        if self.person_names is not None:
            assert (
                len(self.person_names) == self.no_of_persons
            ), "Length of person names must match number of persons."

        if self.item_names is not None:
            assert (
                len(self.item_names) == self.no_of_items
            ), "Length of item names must match number of items."

        if manual_person_names is not None:
            self.person_names = manual_person_names

        else:
            self.person_names = [
                f"Person_{person + 1}" for person in range(self.no_of_persons)
            ]

        if self.persons is None:
            self.persons = self._rng.normal(0, self.person_sd, self.no_of_persons)
            self.persons -= np.mean(self.persons)
            self.persons += self.offset

        else:
            assert (
                len(self.persons) == self.no_of_persons
            ), "Length of manual persons must match number of persons."
            self.persons = np.array(self.persons)

        self.persons = {
            person: location for person, location in zip(self.person_names, self.persons)
        }
        self.persons = pd.Series(self.persons)

        if manual_item_names is not None:
            self.item_names = manual_item_names

        else:
            self.item_names = [f"Item_{item + 1}" for item in range(self.no_of_items)]

        if self.items is None:
            self.items = self._rng.uniform(0, 1, self.no_of_items)
            self.items *= self.item_range / (np.max(self.items) - np.min(self.items))
            self.items -= np.mean(self.items)

        else:
            assert (
                len(self.items) == self.no_of_items
            ), "Length of manual difficulties must match number of items."
            self.items = np.array(self.items)

        self.items = {item: diff for item, diff in zip(self.item_names, self.items)}
        self.items = pd.Series(self.items)

        if self.thresholds is None:
            category_widths = self._rng.uniform(
                self.max_disorder,
                2 * self.category_base - self.max_disorder,
                self.max_score,
            )
            self.thresholds = [
                np.sum(category_widths[:category]) for category in range(self.max_score)
            ]
            self.thresholds = np.array(self.thresholds)
            self.thresholds -= np.mean(self.thresholds)
            self.thresholds = pd.Series(self.thresholds, index=range(1, self.max_score + 1))

        else:
            assert (
                len(self.thresholds) == self.max_score
            ), "Number of manual thresholds must be max score."
            assert np.isclose(sum(manual_thresholds), 0), "Manual thresholds must sum to zero."
            self.thresholds = pd.Series(np.array(self.thresholds), index=range(1, self.max_score + 1))

        # Calculate category probabilities for each person-item combination

        c_p_df = {item: self.persons - self.items[item] for item in self.item_names}
        c_p_df = pd.DataFrame(c_p_df)

        self.cat_probs = {
            cat: (cat * c_p_df) - sum(self.thresholds.iloc[:cat])
            for cat in range(self.max_score + 1)
        }
        for cat in range(self.max_score + 1):
            self.cat_probs[cat] = np.exp(self.cat_probs[cat])

        den = sum(self.cat_probs[cat] for cat in range(self.max_score + 1))

        for cat in range(self.max_score + 1):
            self.cat_probs[cat] /= den

        # Calculate scores and apply missing data

        scoring_randoms = pd.DataFrame(
            self.randoms(), columns=self.item_names, index=self.person_names
        )

        self.responses = sum(
            scoring_randoms
            < sum(
                self.cat_probs[category] for category in range(cat, self.max_score + 1)
            )
            for cat in range(1, self.max_score + 1)
        )

        missing_randoms = pd.DataFrame(
            self.randoms(), columns=self.item_names, index=self.person_names
        )
        self.responses[missing_randoms < self.missing] = np.nan
