import itertools
from math import exp, log, sqrt, floor
import statistics

import numpy as np
import pandas as pd
from scipy.stats import truncnorm, norm

from raschpy.simulation.base_sim import Rasch_Sim
from raschpy.pcm import PCM

class PCM_Sim(Rasch_Sim):
    
    '''
    Generates simulated data accoding to the Partial Credit Model (PCM).
    '''

    def __init__(self,
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
                 manual_item_names=None):

        self.no_of_items = int(no_of_items)
        self.no_of_persons = int(no_of_persons)
        self.item_range = item_range
        self.max_score_vector = max_score_vector
        self.category_base = category_base
        self.person_sd = person_sd
        self.max_disorder = max_disorder
        self.offset = offset
        self.missing = missing
        self.abilities = manual_abilities
        self.diffs = manual_diffs
        self.persons = manual_person_names
        self.items = manual_item_names
        self.dataframe = pd.DataFrame([1])
        self.pcm = PCM(self.dataframe, self.max_score_vector)
        
        '''
        Generates person, item and threshold parameters.
        '''

        assert len(self.max_score_vector) == self.no_of_items, 'Length of max score vector must match number of items.'

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

        if manual_thresholds is None:
            
            category_widths = {item:
                               np.random.uniform(self.max_disorder,
                                                 2 * self.category_base - self.max_disorder,
                                                 max_score - 1)
                               for item, max_score in zip(self.items, self.max_score_vector)}
            
            self.thresholds_centred = {item:
                                       np.array([np.sum(category_widths[item][:category])
                                                 for category in range(max_score)])
                                       for item, max_score in zip(self.items, self.max_score_vector)}
    
            for item in self.items:
    
                self.thresholds_centred[item] -= np.mean(self.thresholds_centred[item])
                self.thresholds_centred[item] = np.insert(self.thresholds_centred[item], 0, 0)

        else:
            assert len(manual_thresholds) == self.no_of_items, 'No of sets of manual thresholds must match number of items.'
            for item in range(no_of_items):
                assert len(manual_thresholds[item]) == self.max_score_vector[item] + 1, ('All sets of item thresholds ' +
                    'in manual thresholds must be max score vector plus one for the corresponding item, beginning zero.')
            for item in range(no_of_items):
                assert manual_thresholds[item][0] == 0 , ('All sets of item thresholds in manual thresholds must ' +
                    'be max score vector plus one for the corresponding item, beginning zero.')
            for item in range(no_of_items):
                assert sum(manual_thresholds[item]) == 0 , ('All sets of item thresholds in manual thresholds must ' +
                    'sum to zero.')

            self.thresholds_centred = {item: np.array(thresholds)
                                       for item, thresholds in zip(self.items, manual_thresholds)}

        self.thresholds_uncentred = {item:
                                     self.thresholds_centred[item][1:] + self.diffs[item]
                                     for item in self.items}

        threshold_list = itertools.chain.from_iterable(self.thresholds_uncentred.values())
        threshold_mean = statistics.mean(threshold_list)
        
        for item in self.items:
            self.diffs[item] -= threshold_mean
            self.thresholds_uncentred[item] -= threshold_mean
            
        '''
        Calculates probability of a response in each category for each person on each item
        '''

        max_max_score = max(self.max_score_vector)

        self.cat_probs = {cat: pd.DataFrame(0, index=self.persons, columns=self.items)
                          for cat in range(max_max_score + 1)}

        for i, item in enumerate(self.items):
            for cat in range(self.max_score_vector[i] + 1):
                self.cat_probs[cat][item] = (self.abilities * cat) - sum(self.thresholds_uncentred[item][:cat])
                self.cat_probs[cat][item] = np.exp(self.cat_probs[cat][item])

            den_vector = sum(self.cat_probs.values())[item]

            for cat in range(self.max_score_vector[i] + 1):
                self.cat_probs[cat][item] /= den_vector
        
        '''
        Calculates scores and removes required amount of missing data
        '''

        scoring_randoms = pd.DataFrame(self.randoms(), columns=self.items, index=self.persons)

        cum_probs = {cat + 1: sum(self.cat_probs[category] for category in range(cat + 1, max_max_score + 1))
                     for cat in range(max_max_score)}

        self.scores = {cat + 1: scoring_randoms < cum_probs[cat + 1]
                       for cat in range(max_max_score)}
        self.scores = sum(self.scores.values())

        missing_randoms = pd.DataFrame(self.randoms(), columns=self.items, index=self.persons)
        self.scores[missing_randoms < self.missing] = np.nan