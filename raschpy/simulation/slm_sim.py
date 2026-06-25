import numpy as np
import pandas as pd

from raschpy.simulation.base_sim import Rasch_Sim


class SLM_Sim(Rasch_Sim):
    """
    Simulate dichotomous response data according to the Simple Logistic Model (SLM).

    Generates item difficulties from a uniform distribution scaled to item_range,
    person abilities from a normal distribution, and response scores by comparing
    uniform random draws against the SLM response probability. Simulation runs
    automatically on instantiation; access results via self.responses.

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
    responses : pandas.DataFrame
        Simulated response matrix, shape (no_of_persons, no_of_items).
        Values are 0, 1, or NaN (missing). This is the primary output.
    persons : pandas.Series
        True person ability parameters used for simulation, indexed by person.
    items : pandas.Series
        True item difficulty parameters used for simulation, indexed by item.
    probs : pandas.DataFrame
        Probability of a correct response for each person-item combination.
    person_names : list of str
        Person labels.
    item_names : list of str
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
        """
        Instantiate and run an SLM simulation.

        See class docstring for full parameter and attribute documentation.
        All simulation output is generated on instantiation and stored as
        instance attributes; see self.responses for the primary output.
        """

        self.no_of_items = int(no_of_items)
        self.no_of_persons = int(no_of_persons)
        self.item_range = item_range
        self.person_sd = person_sd
        self.offset = offset
        self.missing = missing
        self.persons = manual_abilities
        self.items = manual_diffs
        self.person_names = manual_person_names
        self.item_names = manual_item_names
        self._dummy_df = pd.DataFrame([1])

        # Generate person and item parameters

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

        # Calculate probability of a correct response for each person on each item

        self.probs = {item: self.items[item] - self.persons for item in self.item_names}
        self.probs = pd.DataFrame(
            self.probs, columns=self.item_names, index=self.person_names
        )
        self.probs = 1 / (1 + np.exp(self.probs))

        # Calculate scores and apply missing data

        scoring_randoms = pd.DataFrame(
            self.randoms(), columns=self.item_names, index=self.person_names
        )
        self.responses = (scoring_randoms <= self.probs).astype(int)

        missing_randoms = pd.DataFrame(
            self.randoms(), columns=self.item_names, index=self.person_names
        )
        self.responses[missing_randoms < self.missing] = np.nan
