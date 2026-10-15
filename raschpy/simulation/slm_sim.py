import numpy as np
import pandas as pd

from raschpy.simulation.base_sim import Rasch_Sim


class SLM_Sim(Rasch_Sim):
    """
    Simulate dichotomous response data according to the Simple Logistic Model (SLM).

    Generates item difficulties from a uniform distribution scaled to item_range,
    person locations from a normal distribution, and response scores by comparing
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
        Standard deviation of the person location distribution (normal).
    offset : float, default 0
        Mean shift applied to person locations after centring. Positive values
        place persons above the items on average.
    missing : float, default 0
        Proportion of responses to set as missing at random, in [0, 1).
    manual_persons : array-like or None, default None
        Custom person measures. Length must equal no_of_persons.
    manual_items : array-like or None, default None
        Custom item difficulties. Length must equal no_of_items.
    manual_person_names : list of str or None, default None
        Custom person labels. If None, labels are 'Person_1', 'Person_2', etc.
    manual_item_names : list of str or None, default None
        Custom item labels. If None, labels are 'Item_1', 'Item_2', etc.
    seed : int or None, default None
        Seed for the random number generator. Pass an int for a fully
        reproducible simulation; None (default) draws fresh entropy each run.
        This seed only affects this simulation's own local random generator
        (numpy.random.default_rng(seed)), not the global numpy.random.seed()
        state -- a prior call to numpy.random.seed(N) has no effect on this
        simulation's output. Always pass seed= directly here for a
        reproducible run.
    missing_method : {'bernoulli', 'exact'}, default 'bernoulli'
        How `missing` is applied. 'bernoulli' makes each response missing
        independently with probability `missing`, so the realised proportion
        varies around it. 'exact' makes exactly round(missing * cells)
        responses missing. Both use the same random draws, so observed scores
        are identical and the missing patterns differ only in the few cells
        nearest the cut-off; seeded results are reproducible either way.

    Attributes set
    --------------
    responses : pandas.DataFrame
        Simulated response matrix, shape (no_of_persons, no_of_items).
        Values are 0, 1, or NaN (missing). This is the primary output.
    persons : pandas.Series
        True person location parameters used for simulation, indexed by person.
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
        manual_persons=None,
        manual_items=None,
        manual_person_names=None,
        manual_item_names=None,
        seed=None,
        missing_method="bernoulli",
    ):
        """
        Instantiate and run an SLM simulation.

        See class docstring for full parameter and attribute documentation.
        All simulation output is generated on instantiation and stored as
        instance attributes; see self.responses for the primary output.
        """

        self._rng = np.random.default_rng(seed)
        self.model = "SLM"
        self.no_of_items = int(no_of_items)
        self.no_of_persons = int(no_of_persons)
        self.item_range = item_range
        self.person_sd = person_sd
        self.offset = offset
        self.missing = missing
        self._check_missing_method(missing_method)
        self.persons = manual_persons
        self.items = manual_items
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

        # Calculate scores and apply missing data

        self._theta = self.persons.to_numpy()
        self._delta = self.items.to_numpy()
        self._prob_labels = (list(self.person_names), list(self.item_names))
        self._probs = None

        scores = np.empty((self.no_of_persons, self.no_of_items), dtype=np.int64)
        for start, stop in self._row_chunks(self.no_of_persons):
            u = self._rng.random((stop - start, self.no_of_items))
            scores[start:stop] = u <= self._prob_rows(start, stop)

        self.responses = self._responses_frame(
            scores, self._missing_mask(), self.person_names
        )

    def _prob_rows(self, start, stop):
        """Probability of a correct response for persons start:stop."""
        return 1 / (1 + np.exp(self._delta[None, :] - self._theta[start:stop, None]))

    @property
    def probs(self):
        """Probability of a correct response for each person-item combination (computed on first access)."""
        if self._probs is None:
            index, columns = self._prob_labels
            self._probs = pd.DataFrame(
                self._prob_rows(0, self.no_of_persons), index=index, columns=columns
            )
        return self._probs

    @probs.setter
    def probs(self, value):
        self._probs = value
