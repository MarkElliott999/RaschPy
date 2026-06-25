import itertools
from math import exp, log, sqrt, floor
import statistics

import numpy as np
import pandas as pd
from scipy.stats import truncnorm, norm

from raschpy.simulation.base_sim import Rasch_Sim
from raschpy.slm import SLM


class SLM_Sim(Rasch_Sim):
    """
    Simulate dichotomous response data according to the Simple Logistic Model (SLM).

    Generates item difficulties from a uniform distribution scaled to item_range,
    person abilities from a normal distribution, and response scores by comparing
    uniform random draws against the SLM response probability. Simulation runs
    automatically on instantiation; access results via self.scores.

    Parameters
    ----------
    no_of_items : int
        Number of items to simulate.
    no_of_persons : int
        Number of persons to simulate.
    item_range : float, default 3
        Total spread of item difficulties in logits (max - min before centring).
    person_sd : float, default 1.5
        Standard deviation of the person ability distribution (normal).
    offset : float, default 0
        Mean shift applied to person abilities after centring. Positive values
        place persons above the items on average.
    missing : float, default 0
        Proportion of responses to set as missing at random, in [0, 1).
    manual_abilities : array-like or None, default None
        If provided, uses these values as person abilities instead of sampling.
        Length must equal no_of_persons.
    manual_diffs : array-like or None, default None
        If provided, uses these values as item difficulties instead of sampling.
        Length must equal no_of_items.
    manual_person_names : list of str or None, default None
        Custom person labels. If None, labels are 'Person_1', 'Person_2', etc.
    manual_item_names : list of str or None, default None
        Custom item labels. If None, labels are 'Item_1', 'Item_2', etc.

    Attributes set
    --------------
    scores : pandas.DataFrame
        Simulated response matrix, shape (no_of_persons, no_of_items).
        Values are 0, 1, or NaN (missing). This is the primary output.
    abilities : pandas.Series
        True person ability parameters used for simulation, indexed by person.
    diffs : pandas.Series
        True item difficulty parameters used for simulation, indexed by item.
    probs : pandas.DataFrame
        Probability of a correct response for each person-item combination.
    persons : list of str
        Person labels.
    items : list of str
        Item labels.
    no_of_items : int
        Number of items.
    no_of_persons : int
        Number of persons.
    """

    def __init__(
        self,
        no_of_items,
        no_of_persons,
        item_range=3,
        person_sd=1.5,
        offset=0,
        missing=0,
        manual_abilities=None,
        manual_diffs=None,
        manual_person_names=None,
        manual_item_names=None,
    ):

        self.no_of_items = int(no_of_items)
        self.no_of_persons = int(no_of_persons)
        self.item_range = item_range
        self.person_sd = person_sd
        self.offset = offset
        self.missing = missing
        self.abilities = manual_abilities
        self.diffs = manual_diffs
        self.persons = manual_person_names
        self.items = manual_item_names
        self.dataframe = pd.DataFrame([1])
        self.slm = SLM(self.dataframe)

        # Generate person and item parameters

        if self.persons is not None:
            assert (
                len(self.persons) == self.no_of_persons
            ), "Length of person names must match number of persons."

        if self.items is not None:
            assert (
                len(self.items) == self.no_of_items
            ), "Length of item names must match number of items."

        if manual_person_names is not None:
            self.persons = manual_person_names

        else:
            self.persons = [
                f"Person_{person + 1}" for person in range(self.no_of_persons)
            ]

        if self.abilities is None:
            self.abilities = np.random.normal(0, self.person_sd, self.no_of_persons)
            self.abilities -= np.mean(self.abilities)
            self.abilities += self.offset

        else:
            assert (
                len(self.abilities) == self.no_of_persons
            ), "Length of manual abilities must match number of persons."
            self.abilities = np.array(self.abilities)

        self.abilities = {
            person: ability for person, ability in zip(self.persons, self.abilities)
        }
        self.abilities = pd.Series(self.abilities)

        if manual_item_names is not None:
            self.items = manual_item_names

        else:
            self.items = [f"Item_{item + 1}" for item in range(self.no_of_items)]

        if self.diffs is None:
            self.diffs = np.random.uniform(0, 1, self.no_of_items)
            self.diffs *= self.item_range / (np.max(self.diffs) - np.min(self.diffs))
            self.diffs -= np.mean(self.diffs)

        else:
            assert (
                len(self.diffs) == self.no_of_items
            ), "Length of manual difficulties must match number of items."
            self.diffs = np.array(self.diffs)

        self.diffs = {item: diff for item, diff in zip(self.items, self.diffs)}
        self.diffs = pd.Series(self.diffs)

        # Calculate probability of a correct response for each person on each item

        self.probs = {item: self.diffs[item] - self.abilities for item in self.items}
        self.probs = pd.DataFrame(self.probs, columns=self.items, index=self.persons)
        self.probs = 1 / (1 + np.exp(self.probs))

        # Calculate scores and apply missing data

        scoring_randoms = pd.DataFrame(
            self.randoms(), columns=self.items, index=self.persons
        )
        self.scores = (scoring_randoms <= self.probs).astype(int)

        missing_randoms = pd.DataFrame(
            self.randoms(), columns=self.items, index=self.persons
        )
        self.scores[missing_randoms < self.missing] = np.nan
