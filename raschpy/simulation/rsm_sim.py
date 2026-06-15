import itertools
from math import exp, log, sqrt, floor
import statistics

import numpy as np
import pandas as pd
from scipy.stats import truncnorm, norm

from raschpy.simulation.base_sim import Rasch_Sim
from raschpy.rsm import RSM

class RSM_Sim(Rasch_Sim):
    
    '''
    Generates simulated data accoding to the Rating Scale Model (RSM).
    '''

    def __init__(self,
                 no_of_items,
                 no_of_persons,
                 max_score,
                 item_range=3,
                 category_base=1,
                 person_sd=1.5,
                 max_disorder=0,
                 offset=0,
                 missing=0 ,
                 manual_abilities=None,
                 manual_diffs=None,
                 manual_thresholds=None,
                 manual_person_names=None,
                 manual_item_names=None):

        self.no_of_items = int(no_of_items)
        self.no_of_persons = int(no_of_persons)
        self.item_range = item_range
        self.max_score = max_score
        self.category_base = category_base
        self.person_sd = person_sd
        self.max_disorder = max_disorder
        self.offset = offset
        self.missing = missing
        self.abilities = manual_abilities
        self.diffs = manual_diffs
        self.thresholds = manual_thresholds
        self.persons = manual_person_names
        self.items = manual_item_names
        self.dataframe = pd.DataFrame([self.max_score])
        self.rsm = RSM(self.dataframe, self.max_score)

        '''
        Generates person, item and threshold parameters.
        '''

        if self.persons is not None:
            assert len(self.persons) == self.no_of_persons, 'Length of person names must match number of persons.'

        if self.items is not None:
            assert len(self.items) == self.no_of_items, 'Length of item names must match number of items.'

        if manual_person_names is not None:
            self.persons = manual_person_names

        else:
            self.persons = [f'Person_{person + 1}' for person in range(self.no_of_persons)]

        if self.abilities is None:
            self.abilities = np.random.normal(0, self.person_sd, self.no_of_persons)
            self.abilities -= np.mean(self.abilities)
            self.abilities += self.offset

        else:
            assert len(self.abilities) == self.no_of_persons, 'Length of manual abilities must match number of persons.'
            self.abilities = np.array(self.abilities)

        self.abilities = {person: ability for person, ability in zip(self.persons, self.abilities)}
        self.abilities = pd.Series(self.abilities)

        if manual_item_names is not None:
            self.items = manual_item_names

        else:
            self.items = [f'Item_{item + 1}' for item in range(self.no_of_items)]

        if self.diffs is None:
            self.diffs = np.random.uniform(0, 1, self.no_of_items)
            self.diffs *= (self.item_range / (np.max(self.diffs) - np.min(self.diffs)))
            self.diffs -= np.mean(self.diffs)

        else:
            assert len(self.diffs) == self.no_of_items, 'Length of manual difficulties must match number of items.'
            self.diffs = np.array(self.diffs)

        self.diffs = {item: diff for item, diff in zip(self.items, self.diffs)}
        self.diffs = pd.Series(self.diffs)

        if self.thresholds is None:
            category_widths = np.random.uniform(self.max_disorder,
                                                2 * self.category_base - self.max_disorder,
                                                self.max_score)       
            self.thresholds = [np.sum(category_widths[:category])
                               for category in range(self.max_score)]
            self.thresholds = np.array(self.thresholds)
            self.thresholds -= np.mean(self.thresholds)
            self.thresholds = np.insert(self.thresholds, 0, 0)

        else:
            assert len(self.thresholds) == self.max_score + 1, 'Number of manual thresholds must be max score plus 1.'
            assert self.thresholds[0] == 0, 'First threshold in manual thresholds must have value zero.'
            assert sum(manual_thresholds) == 0 , ('Manual thresholds must sum to zero.')
            self.thresholds = np.array(self.thresholds)

        '''
        Calculates probability of a response in each category for each person on each item
        '''

        c_p_df = {item: self.abilities - self.diffs[item]
                  for item in self.items}
        c_p_df = pd.DataFrame(c_p_df)

        self.cat_probs = {cat: (cat * c_p_df) - sum(self.thresholds[:cat + 1])
                          for cat in range(self.max_score + 1)}
        for cat in range(self.max_score + 1):
            self.cat_probs[cat] = np.exp(self.cat_probs[cat])

        den = sum(self.cat_probs[cat] for cat in range(self.max_score + 1))

        for cat in range(self.max_score + 1):
            self.cat_probs[cat] /= den

        '''
        Calculated scores and removes required amount of missing data
        '''

        scoring_randoms = pd.DataFrame(self.randoms(), columns=self.items, index=self.persons)

        self.scores = sum(scoring_randoms < sum(self.cat_probs[category]
                                                for category in range(cat, self.max_score + 1))
                          for cat in range(1, self.max_score + 1))

        missing_randoms = pd.DataFrame(self.randoms(), columns=self.items, index=self.persons)
        self.scores[missing_randoms < self.missing] = np.nan