import itertools
from math import exp, log, sqrt, floor
import statistics

import numpy as np
import pandas as pd
from scipy.stats import truncnorm, norm

class Rasch_Sim:

    def __init__(self):

        pass

    def randoms(self):

        return np.random.rand(self.no_of_persons, self.no_of_items)

    def rename_item(self,
                    old,
                    new):
        
        if old == new:
            print('New item name is the same as old item name.')

        elif new in self.scores.columns:
            print('New item name is a duplicate of an existing item name')

        if old not in self.scores.columns:
            print(f'Old item name "{old}" not found in data. Please check')

        if not isinstance(new, str):
            print('Item names must be strings')

        else:
            self.scores.rename(columns={old: new}, inplace=True)

        self.items = self.scores.columns.tolist()

    def rename_items_all(self,
                         new_names):

        list_length = len(new_names)

        if len(new_names) != len(set(new_names)):
            print('List of new item names contains duplicates. Please ensure all items have unique names.')

        elif list_length != self.no_of_items:
            print(f'Incorrect number of item names. {list_length} in list, {self.no_of_items} items in data.')

        if not all(isinstance(name, str) for name in new_names):
            print('Item names must be strings')

        else:
            self.scores.rename(columns={old: new for old, new in zip(self.scores.columns, new_names)}, inplace=True)

        self.items = self.scores.columns.tolist()

    def rename_person(self,
                      old,
                      new):

        if old == new:
            print('New person name is the same as old person name.')

        elif new in self.scores.index:
            print('New person name is a duplicate of an existing person name.')

        if old not in self.scores.index:
            print(f'Old person name "{old}" not found in data. Please check.')

        if not isinstance(new, str):
            print('Item names must be strings')

        else:
            self.scores.rename(index={old: new}, inplace=True)

        self.persons = self.scores.index.tolist()

    def rename_persons_all(self,
                           new_names):

        list_length = len(new_names)

        if len(new_names) != len(set(new_names)):
            print('List of new person names contains duplicates. Please ensure all persons have unique names.')

        elif list_length != self.no_of_persons:
            print(f'Incorrect number of person names. {list_length} in list, {self.no_of_persons} persons in data.')

        if not all(isinstance(name, str) for name in new_names):
            print('Person names must be strings')

        else:
            self.scores.rename(index={old: new for old, new in zip(self.scores.index, new_names)}, inplace=True)

        self.persons = self.scores.index.tolist()

    def produce_df(self,
                   rows,
                   columns,
                   row_names=None,
                   column_names=None):

        '''
        Produces multi-index Pandas DataFrame with passed parameters.
        '''

        row_index = pd.MultiIndex.from_product(rows, names=row_names)
        col_index = pd.MultiIndex.from_product(columns, names=column_names)

        return pd.DataFrame(index=row_index, columns=col_index)