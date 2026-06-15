import itertools
from math import exp, log, sqrt, floor
import statistics

import numpy as np
import pandas as pd
from scipy.stats import truncnorm, norm

from raschpy.simulation.base_sim import Rasch_Sim
from raschpy.slm import SLM

class SLM_Sim(Rasch_Sim):
    
    '''
    Generates simulated data accoding to the Simple Logistic Model (SLM).
    '''
    
    def __init__(self,
                 no_of_items,
                 no_of_persons,
                 item_range=3,
                 person_sd=1.5,
                 offset=0,
                 missing=0,
                 manual_abilities=None,
                 manual_diffs=None,
                 manual_person_names=None,
                 manual_item_names=None):
        
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
        
        '''
        Generates person and item parameters.
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

        '''
        Calculates probability of a correct response for each person on each item
        '''

        self.probs = {item: self.diffs[item] - self.abilities
                      for item in self.items}
        self.probs = pd.DataFrame(self.probs, columns=self.items, index=self.persons)
        self.probs = 1 / (1 + np.exp(self.probs))
        
        '''
        Calculates scores and removes required amount of missing data
        '''

        scoring_randoms = pd.DataFrame(self.randoms(), columns=self.items, index=self.persons)
        self.scores = (scoring_randoms <= self.probs).astype(int)

        missing_randoms = pd.DataFrame(self.randoms(), columns=self.items, index=self.persons)
        self.scores[missing_randoms < self.missing] = np.nan