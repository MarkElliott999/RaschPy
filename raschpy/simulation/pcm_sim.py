import itertools
import statistics

import numpy as np
import pandas as pd

from raschpy.simulation.base_sim import Rasch_Sim


class PCM_Sim(Rasch_Sim):
    """
    Simulate polytomous response data according to the Partial Credit Model (PCM).

    Generates item difficulties, per-item Rasch-Andrich thresholds, and person
    abilities, then computes category probabilities and samples scores. Unlike
    the RSM, each item has its own independent threshold structure. Simulation
    runs automatically on instantiation; access results via self.responses.

    Parameters
    ----------
    no_of_items : int
        Number of items to simulate.
    no_of_persons : int
        Number of persons to simulate.
    max_score_vector : list of int
        Maximum possible score for each item, in item order. Length must
        equal no_of_items. Items may have different maximum scores.
    item_range : float, default 3
        Total spread of item difficulties in logits.
    category_base : float, default 1
        Base width of each rating category per item. Larger values produce
        wider, more ordered categories.
    person_sd : float, default 1.5
        Standard deviation of the person ability distribution (normal).
    max_disorder : float, default 0
        Maximum threshold disorder per item. 0 produces perfectly ordered
        thresholds; values > 0 introduce random disordering.
    offset : float, default 0
        Mean shift applied to person abilities after centring.
    missing : float, default 0
        Proportion of responses to set as missing at random, in [0, 1).
    manual_abilities : array-like or None, default None
        Custom person abilities. Length must equal no_of_persons.
    manual_diffs : array-like or None, default None
        Custom item difficulties. Length must equal no_of_items.
    manual_thresholds : list of array-like or None, default None
        Custom per-item threshold vectors. Must be a list of no_of_items
        arrays, each of length max_score_vector[i] + 1, beginning with 0
        and summing to 0.
    manual_person_names : list of str or None, default None
        Custom person labels. If None, labels are 'Person_1', 'Person_2', etc.
    manual_item_names : list of str or None, default None
        Custom item labels. If None, labels are 'Item_1', 'Item_2', etc.

    Attributes set
    --------------
    responses : pandas.DataFrame
        Simulated response matrix, shape (no_of_persons, no_of_items).
        Values are integers in [0, max_score_vector[i]] or NaN (missing).
    persons : pandas.Series
        True person ability parameters, indexed by person.
    items : pandas.Series
        True item difficulty parameters (central difficulties), indexed by item.
    thresholds : dict
        {item: numpy.ndarray} of centred Rasch-Andrich threshold offsets per item,
        length max_score_vector[i] + 1, with index 0 = 0.
    thresholds_uncentred : dict
        {item: numpy.ndarray} of uncentred (absolute) thresholds per item,
        length max_score_vector[i].
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
    max_score_vector : list of int
        Maximum score per item.
    """

    def __init__(
        self,
        no_of_items,
        no_of_persons,
        max_score_vector,
        item_range=3,
        category_base=1,
        person_sd=1.5,
        max_disorder=0,
        offset=0,
        missing=0,
        manual_abilities=None,
        manual_diffs=None,
        manual_thresholds=None,
        manual_person_names=None,
        manual_item_names=None,
    ):
        """
        Instantiate and run a PCM simulation.

        See class docstring for full parameter and attribute documentation.
        All simulation output is generated on instantiation and stored as
        instance attributes; see self.responses for the primary output.
        """

        self.no_of_items = int(no_of_items)
        self.no_of_persons = int(no_of_persons)
        self.item_range = item_range
        self.max_score_vector = max_score_vector
        self.category_base = category_base
        self.person_sd = person_sd
        self.max_disorder = max_disorder
        self.offset = offset
        self.missing = missing
        self.persons = manual_abilities
        self.items = manual_diffs
        self.person_names = manual_person_names
        self.item_names = manual_item_names
        self._dummy_df = pd.DataFrame([1])

        # Generate person, item, and threshold parameters

        assert (
            len(self.max_score_vector) == self.no_of_items
        ), "Length of max score vector must match number of items."

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
            self.persons = np.random.normal(0, self.person_sd, self.no_of_persons)
            self.persons -= np.mean(self.persons)
            self.persons += self.offset

        else:
            assert (
                len(self.persons) == self.no_of_persons
            ), "Length of manual abilities must match number of persons."
            self.persons = np.array(self.persons)

        self.persons = {
            person: ability for person, ability in zip(self.person_names, self.persons)
        }
        self.persons = pd.Series(self.persons)

        if manual_item_names is not None:
            self.item_names = manual_item_names

        else:
            self.item_names = [f"Item_{item + 1}" for item in range(self.no_of_items)]

        if self.items is None:
            self.items = np.random.uniform(0, 1, self.no_of_items)
            self.items *= self.item_range / (np.max(self.items) - np.min(self.items))
            self.items -= np.mean(self.items)

        else:
            assert (
                len(self.items) == self.no_of_items
            ), "Length of manual difficulties must match number of items."
            self.items = np.array(self.items)

        self.items = {item: diff for item, diff in zip(self.item_names, self.items)}
        self.items = pd.Series(self.items)
        self.max_score_vector = pd.Series(
            {item: int(ms) for item, ms in zip(self.item_names, self.max_score_vector)}
        )

        if manual_thresholds is None:

            category_widths = {
                item: np.random.uniform(
                    self.max_disorder,
                    2 * self.category_base - self.max_disorder,
                    max_score - 1,
                )
                for item, max_score in zip(self.item_names, self.max_score_vector)
            }

            thr_dict = {
                item: np.array(
                    [
                        np.sum(category_widths[item][:category])
                        for category in range(max_score)
                    ]
                )
                for item, max_score in zip(self.item_names, self.max_score_vector)
            }

            for item in self.item_names:
                thr_dict[item] -= np.mean(thr_dict[item])

        else:
            assert (
                len(manual_thresholds) == self.no_of_items
            ), "No of sets of manual thresholds must match number of items."
            for item in range(no_of_items):
                assert len(manual_thresholds[item]) == self.max_score_vector[item], (
                    "All sets of item thresholds "
                    + "in manual thresholds must be max score vector for the corresponding item."
                )
            for item in range(no_of_items):
                assert sum(manual_thresholds[item]) == 0, (
                    "All sets of item thresholds in manual thresholds must "
                    + "sum to zero."
                )

            thr_dict = {
                item: np.array(thresholds)
                for item, thresholds in zip(self.item_names, manual_thresholds)
            }

        # thresholds as NaN-padded DataFrame (items × threshold indices)
        max_len = max(len(thr_dict[item]) for item in self.item_names)
        thr_rows = {}
        for item in self.item_names:
            arr = thr_dict[item]
            row = np.full(max_len, np.nan)
            row[: len(arr)] = arr
            thr_rows[item] = row
        self.thresholds = pd.DataFrame(thr_rows).T

        unc_dict = {item: thr_dict[item] + self.items[item] for item in self.item_names}

        threshold_list = itertools.chain.from_iterable(unc_dict.values())
        threshold_mean = statistics.mean(threshold_list)

        for item in self.item_names:
            self.items[item] -= threshold_mean
            unc_dict[item] -= threshold_mean

        # thresholds_uncentred as DataFrame (items × threshold indices)
        self.thresholds_uncentred = pd.DataFrame(
            {item: pd.Series(unc_dict[item]) for item in self.item_names}
        ).T

        # Calculate category probabilities for each person-item combination

        max_max_score = max(self.max_score_vector)

        self.cat_probs = {
            cat: pd.DataFrame(0, index=self.person_names, columns=self.item_names)
            for cat in range(max_max_score + 1)
        }

        for i, item in enumerate(self.item_names):
            for cat in range(self.max_score_vector[item] + 1):
                self.cat_probs[cat][item] = (self.persons * cat) - sum(
                    self.thresholds_uncentred.loc[item].dropna().iloc[:cat]
                )
                self.cat_probs[cat][item] = np.exp(self.cat_probs[cat][item])

            den_vector = sum(self.cat_probs.values())[item]

            for cat in range(self.max_score_vector[item] + 1):
                self.cat_probs[cat][item] /= den_vector

        # Calculate scores and apply missing data

        scoring_randoms = pd.DataFrame(
            self.randoms(), columns=self.item_names, index=self.person_names
        )

        cum_probs = {
            cat
            + 1: sum(
                self.cat_probs[category]
                for category in range(cat + 1, max_max_score + 1)
            )
            for cat in range(max_max_score)
        }

        self.responses = {
            cat + 1: scoring_randoms < cum_probs[cat + 1]
            for cat in range(max_max_score)
        }
        self.responses = sum(self.responses.values())

        missing_randoms = pd.DataFrame(
            self.randoms(), columns=self.item_names, index=self.person_names
        )
        self.responses[missing_randoms < self.missing] = np.nan
