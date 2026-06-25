import warnings

import numpy as np
import pandas as pd


class Rasch_Sim:
    """
    Abstract base class for RaschPy simulation objects.

    Provides shared rename utilities (items, persons) and a random-number
    helper used by all concrete simulation subclasses (SLM_Sim, RSM_Sim,
    PCM_Sim, MFRM_Sim). Not intended to be instantiated directly.
    """

    def __init__(self):
        """Initialise the abstract base simulation object. Not intended for direct use."""
        pass

    def randoms(self):
        """
        Generate a (no_of_persons, no_of_items) array of uniform random numbers.

        Used internally by all simulation subclasses to sample response scores
        and apply missing data patterns.

        Returns
        -------
        numpy.ndarray
            Shape (no_of_persons, no_of_items), values in [0, 1).
        """
        return np.random.rand(self.no_of_persons, self.no_of_items)

    def rename_item(self, old, new):
        """
        Rename a single item in the simulated responses DataFrame.

        Parameters
        ----------
        old : str
            Current item name.
        new : str
            Desired new item name.

        Returns
        -------
        None
        """

        if old == new:
            warnings.warn(
                "New item name is the same as the old item name.",
                UserWarning,
                stacklevel=2,
            )
        elif new in self.responses.columns:
            warnings.warn(
                "New item name is a duplicate of an existing item name.",
                UserWarning,
                stacklevel=2,
            )
        if old not in self.responses.columns:
            warnings.warn(
                f"Old item name {old!r} not found in data.", UserWarning, stacklevel=2
            )
        if not isinstance(new, str):
            warnings.warn("Item names must be strings.", UserWarning, stacklevel=2)

        else:
            self.responses.rename(columns={old: new}, inplace=True)

        self.item_names = self.responses.columns.tolist()

    def rename_items_all(self, new_names):
        """
        Rename all items at once.

        Parameters
        ----------
        new_names : list of str
            New item names in the same order as self.items.

        Returns
        -------
        None
        """

        list_length = len(new_names)

        if len(new_names) != len(set(new_names)):
            warnings.warn(
                "List of new item names contains duplicates.", UserWarning, stacklevel=2
            )
        elif list_length != self.no_of_items:
            warnings.warn(
                f"Incorrect number of item names: {list_length} provided, "
                f"{self.no_of_items} items in data.",
                UserWarning,
                stacklevel=2,
            )
        if not all(isinstance(name, str) for name in new_names):
            warnings.warn("Item names must be strings.", UserWarning, stacklevel=2)

        else:
            self.responses.rename(
                columns={
                    old: new for old, new in zip(self.responses.columns, new_names)
                },
                inplace=True,
            )

        self.item_names = self.responses.columns.tolist()

    def rename_person(self, old, new):
        """
        Rename a single person in the simulated responses DataFrame.

        Parameters
        ----------
        old : str
            Current person name.
        new : str
            Desired new person name.

        Returns
        -------
        None
        """

        if old == new:
            warnings.warn(
                "New person name is the same as the old person name.",
                UserWarning,
                stacklevel=2,
            )
        elif new in self.responses.index:
            warnings.warn(
                "New person name is a duplicate of an existing person name.",
                UserWarning,
                stacklevel=2,
            )
        if old not in self.responses.index:
            warnings.warn(
                f"Old person name {old!r} not found in data.", UserWarning, stacklevel=2
            )
        if not isinstance(new, str):
            warnings.warn("Person names must be strings.", UserWarning, stacklevel=2)

        else:
            self.responses.rename(index={old: new}, inplace=True)

        self.person_names = self.responses.index.tolist()

    def rename_persons_all(self, new_names):
        """
        Rename all persons at once.

        Parameters
        ----------
        new_names : list of str
            New person names in the same order as self.persons.

        Returns
        -------
        None
        """

        list_length = len(new_names)

        if len(new_names) != len(set(new_names)):
            warnings.warn(
                "List of new person names contains duplicates.",
                UserWarning,
                stacklevel=2,
            )
        elif list_length != self.no_of_persons:
            warnings.warn(
                f"Incorrect number of person names: {list_length} provided, "
                f"{self.no_of_persons} persons in data.",
                UserWarning,
                stacklevel=2,
            )
        if not all(isinstance(name, str) for name in new_names):
            warnings.warn("Person names must be strings.", UserWarning, stacklevel=2)

        else:
            self.responses.rename(
                index={old: new for old, new in zip(self.responses.index, new_names)},
                inplace=True,
            )

        self.person_names = self.responses.index.tolist()

    def produce_df(self, rows, columns, row_names=None, column_names=None):
        """
        Build an empty MultiIndex DataFrame with the given row/column structure.

        Used internally by subclasses to construct structured output frames.

        Parameters
        ----------
        rows : list of iterables
            Factors for the row MultiIndex (passed to pd.MultiIndex.from_product).
        columns : list of iterables
            Factors for the column MultiIndex.
        row_names : list of str or None, default None
            Names for the row index levels.
        column_names : list of str or None, default None
            Names for the column index levels.

        Returns
        -------
        pandas.DataFrame
            Empty DataFrame with the specified MultiIndex structure.
        """

        row_index = pd.MultiIndex.from_product(rows, names=row_names)
        col_index = pd.MultiIndex.from_product(columns, names=column_names)

        return pd.DataFrame(index=row_index, columns=col_index)
