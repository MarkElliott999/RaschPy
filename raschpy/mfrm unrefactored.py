from math import exp, log, sqrt, floor

import numpy as np
import pandas as pd
from scipy.stats import hmean, norm
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib import colors as colors
from matplotlib import cm as cmx
import seaborn as sns

from raschpy.base import Rasch

class MFRM(Rasch):

    def __init__(self,
                 dataframe,
                 max_score=0,
                 extreme_persons=True,
                 no_of_classes=5):

        '''
        Many-Facet Rasch Model (Linacre 1994); Rating Scale Model (Andrich 1978)
        formulation only (to add: MFRM Partial Credit Model functionality).
        '''

        if max_score == 0:
            self.max_score = int(np.nanmax(dataframe))
        else:
            self.max_score = max_score

        unstacked_df = dataframe.unstack(level=0)

        if extreme_persons:
            to_drop = unstacked_df[unstacked_df.isna().all(axis=1)].index
            
            self.invalid_responses = dataframe[dataframe.index.get_level_values(1).isin(to_drop)]
            self.dataframe = dataframe[~dataframe.index.get_level_values(1).isin(to_drop)]

        else:
            scores = unstacked_df.sum(axis=1)
            max_scores = (unstacked_df.notna() * self.max_score).sum(axis=1)
            to_drop = unstacked_df[(scores == 0) | (scores == max_scores)].index

            self.invalid_responses = dataframe[dataframe.index.get_level_values(1).isin(to_drop)]
            self.dataframe = dataframe[~dataframe.index.get_level_values(1).isin(to_drop)]
        
        self.no_of_persons = len(self.dataframe.index.levels[1])
        self.no_of_items = self.dataframe.shape[1]
        self.no_of_raters = len(self.dataframe.index.levels[0])
        self.no_of_classes = no_of_classes
        self.data = self.dataframe.to_numpy() # (item, rater, person)
        self.items = self.dataframe.columns
        self.raters = self.dataframe.index.get_level_values(0).unique()
        self.persons = self.dataframe.index.get_level_values(1).unique()
        self.anchor_raters_global = []
        self.anchor_raters_items = []
        self.anchor_raters_thresholds = []
        self.anchor_raters_matrix = []

    def rename_rater(self,
                     old,
                     new):

        if old == new:
            print('New rater name is the same as old rater name.')

        elif new in self.raters:
            print('New rater name is a duplicate of an existing rater name')

        if old not in self.raters:
            print(f'Old rater name "{old}" not found in data. Please check')

        if not isinstance(new, str):
            print('Rater names must be strings')

        else:
            new_names = [new if rater == old else rater for rater in self.raters]
            self.rename_raters_all(new_names)

    def rename_raters_all(self,
                          new_names):

        list_length = len(new_names)

        if len(new_names) != len(set(new_names)):
            print('List of new rater names contains duplicates. Please ensure all raters have unique names')

        elif list_length != self.no_of_raters:
            print(f'Incorrect number of rater names. {list_length} in list, {self.no_of_raters} raters in data.')

        if not all(isinstance(name, str) for name in new_names):
            print('Rater names must be strings')


        else:
            df_dict = {new: self.dataframe.xs(old) for old, new in zip(self.raters, new_names)}
            self.dataframe = pd.concat(df_dict.values(), keys = df_dict.keys())
            self.raters = self.dataframe.index.get_level_values(0).unique()

    def rename_person(self,
                      old,
                      new):

        if old == new:
            print('New person name is the same as old person name.')

        elif new in self.dataframe.index.get_level_values(1):
            print('New person name is a duplicate of an existing person name')

        if old not in self.dataframe.index.get_level_values(1):
            print(f'Old person name "{old}" not found in data. Please check')

        else:
            self.dataframe.rename(index={old: new},
                                  inplace=True)
            self.persons = self.dataframe.index.get_level_values(1).unique()

    def rename_persons_all(self,
                           new_names):

        list_length = len(new_names)
        old_names = self.dataframe.index.get_level_values(1)

        if len(new_names) != len(set(new_names)):
            print('List of new person names contains duplicates. Please ensure all persons have unique names')

        elif list_length != self.no_of_persons:
            print(f'Incorrect number of person names. {list_length} in list, {self.no_of_persons} persons in data.')

        else:
            self.dataframe.rename(index={old: new for old, new in zip(old_names, new_names)},
                                  inplace=True)
            self.persons = new_names

    def _cat_probs_mfrm(self, abilities, items, raters, thresholds,
                        item_offsets, thresh_offsets):
        """
        Vectorised MFRM category probability engine shared across all four
        rater parameterisations.

        The log-numerator for person n, rater r, item i, category k is:
            k * (theta_n - delta_i - item_offset[r,i]) - cumsum(tau + thresh_offset[r])[k]

        Parameters
        ----------
        abilities      : ndarray (N,)  -- person ability estimates
        items          : list of item names
        raters         : list of rater names
        thresholds     : ndarray (K+1,) -- shared RSM thresholds (tau_0=0 sentinel)
        item_offsets   : dict {rater: ndarray(I,)} or {rater: scalar}
                         Amount subtracted from (theta - delta) in the linear term.
                         global:     scalar severity_r
                         items:      per-item vector severity_ri
                         thresholds: zeros
                         matrix:     marginal item severity_ri
        thresh_offsets : dict {rater: ndarray(K+1,)} or None
                         Amount added to thresholds for each rater.
                         global/items: None (zeros)
                         thresholds:   per-threshold vector severity_rk
                         matrix:       per-item per-threshold array (averaged or full)

        Returns
        -------
        probs : dict {rater: ndarray (K+1, N, I)}
        cats  : ndarray (K+1,)
        """
        n_items = len(items)
        diff_arr = self.diffs.loc[items].values     # (I,)
        cats = np.arange(len(thresholds), dtype=float)
        cumsum_tau = np.cumsum(thresholds)          # (K+1,)
        ab = np.asarray(abilities, dtype=float)     # (N,)

        result = {}
        for rater in raters:
            # Item offset: shape (I,)
            if item_offsets is not None:
                io = item_offsets[rater]
                io = np.asarray([io] * n_items if np.isscalar(io) else io, dtype=float)
            else:
                io = np.zeros(n_items)

            # Threshold offset: shape (K+1,)
            if thresh_offsets is not None and thresh_offsets[rater] is not None:
                to = np.asarray(thresh_offsets[rater], dtype=float)
            else:
                to = np.zeros(len(thresholds))

            cumsum_total = cumsum_tau + to          # (K+1,)

            # log_num[k,n,i] = k*(ab[n] - delta_i - io[i]) - cumsum_total[k]
            log_num = (cats[:, None, None]
                       * (ab[None, :, None] - diff_arr[None, None, :] - io[None, None, :])
                       - cumsum_total[:, None, None])                # (K+1, N, I)

            # Numerically stable softmax
            log_num -= log_num.max(axis=0, keepdims=True)
            probs = np.exp(log_num)
            probs /= probs.sum(axis=0, keepdims=True)

            result[rater] = probs

        return result, cats


    def cat_prob_global(self,
                        ability,
                        item,
                        difficulties,
                        rater,
                        severities,
                        category,
                        thresholds):

        '''
        Calculates the probability of a score given ability, difficulty,
        set of thresholds and rater severity, basic MFRM.
        '''

        cat_prob_nums = [exp(cat * (ability - difficulties.loc[item] - severities[rater]) -
                             sum(thresholds[:cat + 1]))
                         for cat in range(self.max_score + 1)]

        return cat_prob_nums[category] / sum(cat_prob_nums)

    def cat_prob_items(self,
                       ability,
                       item,
                       difficulties,
                       rater,
                       severities,
                       category,
                       thresholds):

        '''
        Category response probability function for the RSM formulation
        of the extended vector-by-item form of the Many-Facet Rasch Model (MFRM).
        '''

        cat_prob_nums = [exp(cat * (ability - difficulties.loc[item] -
                                    severities[rater][item]) -
                             sum(thresholds[:cat + 1]))
                         for cat in range(self.max_score + 1)]

        return cat_prob_nums[category] / sum(cat_prob_nums)

    def cat_prob_thresholds(self,
                            ability,
                            item,
                            difficulties,
                            rater,
                            severities,
                            category,
                            thresholds):

        '''
        Category response probability function for the RSM formulation of the
        extended vector-by-threshold form of the Many-Facet Rasch Model (MFRM).
        '''

        cat_prob_nums = [exp(cat * (ability - difficulties.loc[item]) -
                             sum(thresholds[:cat + 1]) - sum(severities[rater][:cat + 1]))
                         for cat in range(self.max_score + 1)]

        return cat_prob_nums[category] / sum(cat_prob_nums)

    def cat_prob_matrix(self,
                        ability,
                        item,
                        difficulties,
                        rater,
                        severities,
                        category,
                        thresholds):

        '''
        Category response probability function for the RSM formulation of the
        extended matrix form of the Many-Facet Rasch Model (MFRM).
        '''

        cat_prob_nums = [exp(cat * (ability - difficulties.loc[item]) -
                             sum(thresholds[:cat + 1]) - sum(severities[rater][item][:cat + 1]))
                         for cat in range(self.max_score + 1)]

        return cat_prob_nums[category] / sum(cat_prob_nums)

    def exp_score_global(self,
                         ability,
                         item,
                         difficulties,
                         rater,
                         severities,
                         thresholds):

        '''
        Calculates the expected score given ability, difficulty,
        set of thresholds and rater severity, basic MFRM.
        '''

        cat_prob_nums = [exp(category * (ability - difficulties[item] - severities[rater]) -
                             sum(thresholds[:category + 1]))
                         for category in range(self.max_score + 1)]

        exp_score = (sum(category * prob for category, prob in enumerate(cat_prob_nums)) /
                     sum(prob for prob in cat_prob_nums))

        return exp_score

    def exp_score_items(self,
                        ability,
                        item,
                        difficulties,
                        rater,
                        severities,
                        thresholds):

        '''
        Expected score function for the RSM formulation of the extended
        vector-by-item form of the Many-Facet Rasch Model (MFRM).
        '''

        cat_prob_nums = [exp(category * (ability - difficulties[item] - severities[rater][item]) -
                             sum(thresholds[:category + 1]))
                         for category in range(self.max_score + 1)]

        exp_score = (sum(category * prob for category, prob in enumerate(cat_prob_nums)) /
                     sum(prob for prob in cat_prob_nums))

        return exp_score

    def exp_score_thresholds(self,
                             ability,
                             item,
                             difficulties,
                             rater,
                             severities,
                             thresholds):

        '''
        Expected score function for the RSM formulation of the extended
        vector-by-threshold form of the Many-Facet Rasch Model (MFRM).
        '''

        cat_prob_nums = [exp(category * (ability - difficulties[item]) -
                             sum(thresholds[:category + 1] + severities[rater][:category + 1]))
                         for category in range(self.max_score + 1)]

        exp_score = (sum(category * prob for category, prob in enumerate(cat_prob_nums)) /
                     sum(prob for prob in cat_prob_nums))

        return exp_score

    def exp_score_matrix(self,
                         ability,
                         item,
                         difficulties,
                         rater,
                         severities,
                         thresholds):

        '''
        Expected score function for the RSM formulation of the extended
        matrix form of the Many-Facet Rasch Model (MFRM).
        '''

        cat_prob_nums = [exp(category * (ability - difficulties[item]) -
                             sum(thresholds[:category + 1] + severities[rater][item][:category + 1]))
                         for category in range(self.max_score + 1)]

        exp_score = (sum(category * prob for category, prob in enumerate(cat_prob_nums)) /
                     sum(prob for prob in cat_prob_nums))

        return exp_score

    def variance_global(self,
                        ability,
                        item,
                        difficulties,
                        rater,
                        severities,
                        thresholds):

        '''
        Calculates the item information / variance given ability, difficulty,
        set of thresholds and rater severity, basic MFRM.
        '''

        expected = self.exp_score_global(ability, item, difficulties, rater, severities, thresholds)

        variance = sum(((category - expected) ** 2) *
                        self.cat_prob_global(ability, item, difficulties, rater, severities, category, thresholds)
                       for category in range(self.max_score + 1))

        return variance

    def variance_items(self,
                       ability,
                       item,
                       difficulties,
                       rater,
                       severities,
                       thresholds):

        '''
        Calculates the item information / variance given ability, difficulty,
        set of thresholds and rater severity, extended MFRM by item.
        '''

        expected = self.exp_score_items(ability, item, difficulties, rater, severities, thresholds)

        variance = sum(((category - expected) ** 2) *
                       self.cat_prob_items(ability, item, difficulties, rater, severities, category, thresholds)
                       for category in range(self.max_score + 1))

        return variance

    def variance_thresholds(self,
                            ability,
                            item,
                            difficulties,
                            rater,
                            severities,
                            thresholds):

        '''
        Calculates the item information / variance given ability, difficulty,
        set of thresholds and rater severity, extended MFRM by threshold.
        '''

        expected = self.exp_score_thresholds(ability, item, difficulties, rater, severities, thresholds)

        variance = sum(((category - expected) ** 2) *
                       self.cat_prob_thresholds(ability, item, difficulties, rater, severities, category, thresholds)
                       for category in range(self.max_score + 1))

        return variance

    def variance_matrix(self,
                        ability,
                        item,
                        difficulties,
                        rater,
                        severities,
                        thresholds):

        '''
        Calculates the item information / variance given ability, difficulty,
        set of thresholds and rater severity, basic MFRM.
        '''

        expected = self.exp_score_matrix(ability, item, difficulties, rater, severities, thresholds)

        variance = sum(((category - expected) ** 2) *
                       self.cat_prob_matrix(ability, item, difficulties, rater, severities, category, thresholds)
                       for category in range(self.max_score + 1))

        return variance

    def kurtosis_global(self,
                        ability,
                        item,
                        difficulties,
                        rater,
                        severities,
                        thresholds):

        '''
        Calculates the item kurtosis given ability, difficulty,
        set of thresholds and rater severity, basic MFRM.
        '''

        cat_probs = [self.cat_prob_global(ability, item, difficulties, rater, severities, category, thresholds)
                     for category in range(self.max_score + 1)]

        expected = self.exp_score_global(ability, item, difficulties, rater, severities, thresholds)

        kurtosis = sum(((category - expected) ** 4) * prob
                       for category, prob in enumerate(cat_probs))

        return kurtosis

    def kurtosis_items(self,
                       ability,
                       item,
                       difficulties,
                       rater,
                       severities,
                       thresholds):

        '''
        Calculates the item kurtosis given ability, difficulty,
        set of thresholds and rater severity, basic MFRM.
        '''

        cat_probs = [self.cat_prob_items(ability, item, difficulties, rater, severities, category, thresholds)
                     for category in range(self.max_score + 1)]

        expected = self.exp_score_items(ability, item, difficulties, rater, severities, thresholds)

        kurtosis = sum(((category - expected) ** 4) * prob
                       for category, prob in enumerate(cat_probs))

        return kurtosis

    def kurtosis_thresholds(self,
                            ability,
                            item,
                            difficulties,
                            rater,
                            severities,
                            thresholds):

        '''
        Calculates the item kurtosis given ability, difficulty,
        set of thresholds and rater severity, basic MFRM.
        '''

        cat_probs = [self.cat_prob_thresholds(ability, item, difficulties, rater, severities, category, thresholds)
                     for category in range(self.max_score + 1)]

        expected = self.exp_score_thresholds(ability, item, difficulties, rater, severities, thresholds)

        kurtosis = sum(((category - expected) ** 4) * prob
                       for category, prob in enumerate(cat_probs))

        return kurtosis

    def kurtosis_matrix(self,
                        ability,
                        item,
                        difficulties,
                        rater,
                        severities,
                        thresholds):

        '''
        Calculates the item kurtosis given ability, difficulty,
        set of thresholds and rater severity, basic MFRM.
        '''

        cat_probs = [self.cat_prob_matrix(ability, item, difficulties, rater, severities, category, thresholds)
                     for category in range(self.max_score + 1)]

        expected = self.exp_score_matrix(ability, item, difficulties, rater, severities, thresholds)

        kurtosis = sum(((category - expected) ** 4) * prob
                       for category, prob in enumerate(cat_probs))

        return kurtosis

    '''
    *** PAIR/CPAT ALGORITHM COMPONENTS ***
    '''

    def item_diffs(self,
                   constant=0.1,
                   method='cos',
                   matrix_power=3,
                   log_lik_tol=0.000001):

        '''
        Calculates PAIR item estimates (cosine similarity).
        '''

        data = self.dataframe.values.reshape(self.no_of_raters,
                                             self.no_of_persons, -1).swapaxes(1, 2)
        data = data.transpose((1, 0, 2))

        matrix = [[sum(np.count_nonzero(data[item_1, rater, :] ==
                                        data[item_2, rater, :] + 1)
                       for rater in range(self.no_of_raters))
                   for item_2 in range(self.no_of_items)]
                  for item_1 in range(self.no_of_items)]

        matrix = np.array(matrix).astype(np.float64)

        constant_matrix = (matrix + matrix.T > 0).astype(np.float64)
        constant_matrix *= constant
        matrix += constant_matrix
        matrix += (np.identity(self.no_of_items) * constant)

        mat = np.linalg.matrix_power(matrix, matrix_power)
        mat_pow = matrix_power

        while 0 in mat:

            mat = np.matmul(mat, matrix)
            mat_pow += 1

            if mat_pow == matrix_power + 5:
                mat += constant
                break

        self.diffs = self.priority_vector(mat, method=method, log_lik_tol=log_lik_tol)

    def raters_global(self,
                      constant=0.1,
                      method='cos',
                      matrix_power=3,
                      log_lik_tol=0.000001):

        '''
        Calculates (global) rater severity using PAIR (cosine similarity).
        '''

        data = self.dataframe.values.reshape(self.no_of_raters,
                                             self.no_of_persons, -1).swapaxes(1, 2)
        data = data.transpose((1, 0, 2))

        matrix = [[sum(np.count_nonzero(data[item, rater_1, :] ==
                                        data[item, rater_2, :] + 1)
                       for item in range(self.no_of_items))
                   for rater_2 in range(self.no_of_raters)]
                  for rater_1 in range(self.no_of_raters)]

        matrix = np.array(matrix).astype(np.float64)

        constant_matrix = (matrix + matrix.T > 0).astype(np.float64)
        constant_matrix *= constant
        matrix += constant_matrix
        matrix += (np.identity(self.no_of_raters) * constant)

        mat = np.linalg.matrix_power(matrix, matrix_power)
        mat_pow = matrix_power

        while 0 in mat:

            mat = np.matmul(mat, matrix)
            mat_pow += 1

            if mat_pow == matrix_power + 5:
                mat += constant
                break

        self.severities_global = self.priority_vector(mat, method=method, log_lik_tol=log_lik_tol, raters=True)

    def _item_rater_element(self,
                            item,
                            constant=0.1,
                            method='cos',
                            matrix_power=3,
                            log_lik_tol=0.000001):

        '''
        ** Private method **
        Mini-function for use in calculating rater severity (vector by item).
        '''

        data = self.dataframe.values.reshape(self.no_of_raters,
                                             self.no_of_persons, -1).swapaxes(1, 2)
        data = data.transpose((1, 0, 2))

        matrix = [[np.count_nonzero(data[item, rater_1, :] ==
                                    data[item, rater_2, :] + 1)
                   for rater_2 in range(self.no_of_raters)]
                  for rater_1 in range(self.no_of_raters)]

        matrix = np.array(matrix).astype(np.float64)

        constant_matrix = (matrix + matrix.T > 0).astype(np.float64)
        constant_matrix *= constant
        matrix += constant_matrix
        matrix += (np.identity(self.no_of_raters) * constant)

        mat = np.linalg.matrix_power(matrix, matrix_power)
        mat_pow = matrix_power

        while 0 in mat:

            mat = np.matmul(mat, matrix)
            mat_pow += 1

            if mat_pow == matrix_power + 5:
                mat += constant
                break
        rater_element = self.priority_vector(mat, method=method, log_lik_tol=log_lik_tol, raters=True)

        return rater_element

    def raters_items(self,
                     constant=0.1,
                     method='cos',
                     matrix_power=3,
                     log_lik_tol=0.000001):

        '''
        Calculates rater severity (vector by item).
        '''

        raters = np.zeros((self.no_of_raters, self.no_of_items))

        for item in range(self.no_of_items):
            raters[:, item] = self._item_rater_element(item, constant=constant, method=method,
                                                       matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        raters = pd.DataFrame(raters)
        raters.columns = self.dataframe.columns
        raters.index = self.raters
        raters = raters.T.to_dict()

        self.severities_items = raters

    def _threshold_rater_element(self,
                                 category,
                                 constant=0.1,
                                 method='cos',
                                 matrix_power=3,
                                 log_lik_tol=0.000001):

        '''
        ** Private method **
        Mini-function for use in calculating rater severity (vector by threshold).
        '''

        data = self.dataframe.values.reshape(self.no_of_raters,
                                             self.no_of_persons, -1).swapaxes(1, 2)
        data = data.transpose((1, 0, 2))

        matrix = [[np.sum([np.count_nonzero((data[item, rater_1, :] == category + 1) &
                                            (data[item, rater_2, :] == category))
                           for item in range(self.no_of_items)])
                   for rater_2 in range(self.no_of_raters)]
                  for rater_1 in range(self.no_of_raters)]

        matrix = np.array(matrix).astype(np.float64)

        constant_matrix = (matrix + matrix.T > 0).astype(np.float64)
        constant_matrix *= constant
        matrix += constant_matrix
        matrix += (np.identity(self.no_of_raters) * constant)

        mat = np.linalg.matrix_power(matrix, matrix_power)
        mat_pow = matrix_power

        while 0 in mat:

            mat = np.matmul(mat, matrix)
            mat_pow += 1

            if mat_pow == matrix_power + 5:
                mat += constant
                break

        raters = self.priority_vector(mat, method=method, log_lik_tol=log_lik_tol, raters=True)

        return raters

    def raters_thresholds(self,
                          constant=0.1,
                          method='cos',
                          matrix_power=3,
                          log_lik_tol=0.000001):

        '''
        Calculates rater severity (vector by threshold).
        '''

        raters = np.zeros((self.no_of_raters, self.max_score))

        for threshold in range(self.max_score):
            raters[:, threshold] = self._threshold_rater_element(threshold, constant=constant, method=method,
                                                                 matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        raters = np.insert(raters, 0, [0 for rater in range(self.no_of_raters)], axis=1)
        raters = {rater: severity for rater, severity in zip(self.raters, raters)}

        self.severities_thresholds = raters

    def _matrix_rater_element(self,
                              item,
                              category,
                              constant=0.1,
                              method='cos',
                              matrix_power=3,
                              log_lik_tol=0.000001):

        '''
        ** Private method **
        Mini-function for use in calculating rater severity (matrix).
        '''

        data = self.dataframe.values.reshape(self.no_of_raters,
                                             self.no_of_persons, -1).swapaxes(1, 2)
        data = data.transpose((1, 0, 2))

        matrix = [[np.count_nonzero((data[item, rater_1, :] == category + 1) &
                                    (data[item, rater_2, :] == category))
                   for rater_2 in range(self.no_of_raters)]
                  for rater_1 in range(self.no_of_raters)]

        matrix = np.array(matrix).astype(np.float64)

        constant_matrix = (matrix + matrix.T > 0).astype(np.float64)
        constant_matrix *= constant
        matrix += constant_matrix
        matrix += (np.identity(self.no_of_raters) * constant)

        mat = np.linalg.matrix_power(matrix, matrix_power)
        mat_pow = matrix_power

        while 0 in mat:

            mat = np.matmul(mat, matrix)
            mat_pow += 1

            if mat_pow == matrix_power + 5:
                mat += constant
                break

        raters = self.priority_vector(mat, method=method, log_lik_tol=log_lik_tol, raters=True)

        return raters

    def raters_matrix(self,
                      constant=0.1,
                      method='cos',
                      matrix_power=3,
                      log_lik_tol=0.000001):

        '''
        Calculates rater severity (matrix).
        '''

        raters = np.zeros((self.no_of_raters, self.no_of_items, self.max_score + 1))

        for item in range(self.no_of_items):
            for category in range(self.max_score):
                    raters[:, item, category + 1] = self._matrix_rater_element(item, category, constant=constant,
                                                                               method=method, matrix_power=matrix_power,
                                                                               log_lik_tol=log_lik_tol)

        rater_dict = {}

        for i, rater in enumerate(self.raters):

            rater_dict[rater] = raters[i, :, :]
            rater_dict[rater] = {item: rater_dict[rater][i, :] for i, item in enumerate(self.dataframe.columns)}

        marginal_items = {rater: raters[i, :, 1:].mean(axis=1) for i, rater in enumerate(self.raters)}
        for rater in self.raters:
            marginal_items[rater] = pd.Series({item: severity
                                               for item, severity in zip(self.dataframe.columns,
                                                                         marginal_items[rater])})

        marginal_thresholds = {rater: raters[i].mean(axis=0)
                               for i, rater in enumerate(self.raters)}
        for rater in self.raters:
            marginal_thresholds[rater][1:] -= marginal_thresholds[rater][1:].mean()

        self.severities_matrix = rater_dict
        self.marginal_severities_items = marginal_items
        self.marginal_severities_thresholds = marginal_thresholds

    def _threshold_distance(self,
                            threshold,
                            difficulties,
                            constant=0.1):

        '''
        ** Private method **
        Calculates difference between a pair of adjacent thresholds.
        '''

        data = self.dataframe.values.reshape(self.no_of_raters, self.no_of_persons, -1).swapaxes(1, 2)
        data = data.transpose((1, 0, 2))

        estimator = 0
        weights_sum = 0

        for item_1 in range(self.no_of_items):
            for item_2 in range(self.no_of_items):

                num = sum(np.count_nonzero((data[item_1, rater, :] == threshold) &
                                           (data[item_2, rater, :] == threshold))
                          for rater in range(self.no_of_raters))

                den = sum(np.count_nonzero((data[item_1, rater, :] == threshold - 1) &
                                           (data[item_2, rater, :] == threshold + 1))
                          for rater in range(self.no_of_raters))

                if num + den == 0:
                    pass

                else:
                    num += constant
                    den += constant

                    weight = hmean([num, den])

                    estimator += weight * (log(num) - log(den) +
                                           difficulties.iloc[item_1] - difficulties.iloc[item_2])
                    weights_sum += weight

        try:
            estimator /= weights_sum

        except Exception:
            estimator = np.nan

        return estimator

    def ra_thresholds(self,
                      difficulties,
                      constant=0.1):

        '''
        Calculates set of threshold estimates using CPAT
        '''

        distances = [self._threshold_distance(category, difficulties, constant)
                     for category in range(1, self.max_score)]

        thresholds = [sum(distances[:threshold])
                      for threshold in range(self.max_score)]

        thresholds = np.array(thresholds)

        np.add(thresholds, -np.mean(thresholds), out = thresholds, casting = 'unsafe')

        thresholds = np.insert(thresholds, 0, 0)

        return thresholds

    def calibrate_global(self,
                         constant=0.1,
                         method='cos',
                         matrix_power=3,
                         log_lik_tol=0.000001):

        '''
        Estimates items, thresholds and rater severities (global) in one method.
        '''

        # Vectorised null person removal (replaces O(N_null) per-person drop loop)
        _person_data = self.dataframe.unstack(level=0)
        _null_mask = _person_data.isnull().all(axis=1)
        self.null_persons = _person_data.index[_null_mask].tolist()
        if self.null_persons:
            self.dataframe = self.dataframe.drop(self.null_persons, level=1)
            self.persons = self.dataframe.index.get_level_values(1).unique()

        self.no_of_persons = len(self.persons)

        self.item_diffs(constant=constant, method=method, matrix_power=matrix_power, log_lik_tol=log_lik_tol)
        self.thresholds = self.ra_thresholds(self.diffs, constant)
        self.raters_global(constant=constant, method=method, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

    def calibrate_items(self,
                        constant=0.1,
                        method='cos',
                        matrix_power=3,
                        log_lik_tol=0.000001):

        '''
        Estimates items, thresholds and rater severities (items) in one method.
        '''

        # Vectorised null person removal (replaces O(N_null) per-person drop loop)
        _person_data = self.dataframe.unstack(level=0)
        _null_mask = _person_data.isnull().all(axis=1)
        self.null_persons = _person_data.index[_null_mask].tolist()
        if self.null_persons:
            self.dataframe = self.dataframe.drop(self.null_persons, level=1)
            self.persons = self.dataframe.index.get_level_values(1).unique()

        self.no_of_persons = len(self.persons)

        self.item_diffs(constant=constant, method=method, matrix_power=matrix_power, log_lik_tol=log_lik_tol)
        self.thresholds = self.ra_thresholds(self.diffs, constant)
        self.raters_items(constant=constant, method=method, matrix_power=matrix_power, log_lik_tol=log_lik_tol)


    def calibrate_thresholds(self,
                             constant=0.1,
                             method='cos',
                             matrix_power=3,
                             log_lik_tol=0.000001):

        '''
        Estimates items, thresholds and rater severities (thresholds) in one method.
        '''

        # Vectorised null person removal (replaces O(N_null) per-person drop loop)
        _person_data = self.dataframe.unstack(level=0)
        _null_mask = _person_data.isnull().all(axis=1)
        self.null_persons = _person_data.index[_null_mask].tolist()
        if self.null_persons:
            self.dataframe = self.dataframe.drop(self.null_persons, level=1)
            self.persons = self.dataframe.index.get_level_values(1).unique()

        self.no_of_persons = len(self.persons)

        self.item_diffs(constant=constant, method=method, matrix_power=matrix_power, log_lik_tol=log_lik_tol)
        self.thresholds = self.ra_thresholds(self.diffs, constant)
        self.raters_thresholds(constant=constant, method=method, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

    def calibrate_matrix(self,
                         constant=0.1,
                         method='cos',
                         matrix_power=3,
                         log_lik_tol=0.000001):

        '''
        Estimates items, thresholds and rater severities (matrix) in one method.
        '''

        # Vectorised null person removal (replaces O(N_null) per-person drop loop)
        _person_data = self.dataframe.unstack(level=0)
        _null_mask = _person_data.isnull().all(axis=1)
        self.null_persons = _person_data.index[_null_mask].tolist()
        if self.null_persons:
            self.dataframe = self.dataframe.drop(self.null_persons, level=1)
            self.persons = self.dataframe.index.get_level_values(1).unique()

        self.no_of_persons = len(self.persons)

        self.item_diffs(constant=constant, method=method, matrix_power=matrix_power, log_lik_tol=log_lik_tol)
        self.thresholds = self.ra_thresholds(self.diffs, constant)
        self.raters_matrix(constant=constant, method=method, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

    def std_errors_global(self,
                          anchor_raters=None,
                          interval=None,
                          no_of_samples=100,
                          constant=0.1,
                          method='cos',
                          matrix_power=3,
                          log_lik_tol=0.000001):

        '''
        Estimates items, thresholds and rater severities (matrix) in one method.
        '''

        samples = []

        picks = [np.random.randint(0, self.no_of_persons, self.no_of_persons)
                 for sample in range(no_of_samples)]
        picks = [self.dataframe.index.get_level_values(1)[pick] for pick in picks]

        data_dict = {rater: self.dataframe.xs(rater) for rater in self.raters}

        for sample in range(no_of_samples):
            sample_data_dict = {rater: pd.DataFrame([data_dict[rater].loc[pick]
                                                     for pick in picks[sample]]).reset_index(drop=True)
                                for rater in self.raters}

            samples.append(pd.concat(sample_data_dict.values(), keys=sample_data_dict.keys()))

        samples = [MFRM(sample, self.max_score) for sample in samples]

        for sample in samples:
            sample.calibrate_global(constant=constant, method=method,
                                    matrix_power=matrix_power, log_lik_tol=log_lik_tol)

            if anchor_raters is not None:
                sample.calibrate_global_anchor(anchor_raters, constant=constant, method=method,
                                               matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if anchor_raters is not None:
            item_ests = np.array([sample.anchor_diffs_global.values for sample in samples])
            threshold_ests = np.array([sample.anchor_thresholds_global for sample in samples])
            rater_ests = np.array([sample.anchor_severities_global.values for sample in samples])

        else:
            item_ests = np.array([sample.diffs.values for sample in samples])
            threshold_ests = np.array([sample.thresholds for sample in samples])
            rater_ests = np.array([sample.severities_global.values for sample in samples])

        item_se = {item: se for item, se in zip(self.dataframe.columns, np.nanstd(item_ests, axis=0))}
        item_se = pd.Series(item_se)

        if interval is not None:
            item_low = {item: low for item, low in zip(self.dataframe.columns,
                                                       np.percentile(item_ests,
                                                                     50 * (1 - interval), axis=0))}
            item_low = pd.Series(item_low)
            item_high = {item: high for item, high in zip(self.dataframe.columns,
                                                          np.percentile(item_ests,
                                                                        50 * (1 + interval), axis=0))}
            item_high = pd.Series(item_high)

        else:
            item_low = None
            item_high = None

        threshold_se = np.std(threshold_ests, axis=0)

        if interval is not None:
            threshold_low = np.percentile(threshold_ests, 50 * (1 - interval), axis=0)
            threshold_high = np.percentile(threshold_ests, 50 * (1 + interval), axis=0)

        else:
            threshold_low = None
            threshold_high = None

        cat_widths = {cat + 1: threshold_ests[:,cat + 2] - threshold_ests[:, cat + 1]
                      for cat in range(self.max_score - 1)}
        cat_width_se = {cat: np.nanstd(estimates)
                        for cat, estimates in cat_widths.items()}

        if interval is not None:
            cat_width_low = {cat: np.percentile(estimates, 50 * (1 - interval))
                            for cat, estimates in cat_widths.items()}
            cat_width_high = {cat: np.percentile(estimates, 50 * (1 + interval))
                            for cat, estimates in cat_widths.items()}

        else:
            cat_width_low = None
            cat_width_high = None

        rater_se = {rater: se for rater, se in zip(self.raters, np.std(rater_ests, axis=0))}
        rater_se = pd.Series(rater_se)

        if interval is not None:
            rater_low = {rater: percentile
                         for rater, percentile in zip(self.raters, np.percentile(rater_ests,
                                                                                 50 * (1 - interval), axis=0))}
            rater_low = pd.Series(rater_low)
            rater_high = {rater: percentile
                          for rater, percentile in zip(self.raters, np.percentile(rater_ests,
                                                                                  50 * (1 + interval), axis=0))}
            rater_high = pd.Series(rater_high)

        else:
            rater_low = None
            rater_high = None

        if anchor_raters is not None:
            self.anchor_item_bootstrap_global = item_ests
            self.anchor_item_se_global = item_se
            self.anchor_item_low_global = item_low
            self.anchor_item_high_global = item_high
            self.anchor_threshold_bootstrap_global = threshold_ests
            self.anchor_threshold_se_global = threshold_se
            self.anchor_threshold_low_global = threshold_low
            self.anchor_threshold_high_global = threshold_high
            self.anchor_cat_width_bootstrap_global = cat_widths
            self.anchor_cat_width_se_global = cat_width_se
            self.anchor_cat_width_low_global = cat_width_low
            self.anchor_cat_width_high_global = cat_width_high
            self.anchor_rater_bootstrap_global = rater_ests
            self.anchor_rater_se_global = rater_se
            self.anchor_rater_low_global = rater_low
            self.anchor_rater_high_global = rater_high

        else:
            self.item_bootstrap_global = item_ests
            self.item_se = item_se
            self.item_low = item_low
            self.item_high = item_high
            self.threshold_bootstrap_global = threshold_ests
            self.threshold_se_global = threshold_se
            self.threshold_low_global = threshold_low
            self.threshold_high_global = threshold_high
            self.cat_width_bootstrap_global = cat_widths
            self.cat_width_se_global = cat_width_se
            self.cat_width_low_global = cat_width_low
            self.cat_width_high_global = cat_width_high
            self.rater_bootstrap_global = rater_ests
            self.rater_se_global = rater_se
            self.rater_low_global = rater_low
            self.rater_high_global = rater_high

    def std_errors_items(self,
                         anchor_raters=None,
                         interval=None,
                         no_of_samples=100,
                         constant=0.1,
                         method='cos',
                         matrix_power=3,
                         log_lik_tol=0.000001):

        '''
        Estimates items, thresholds and rater severities (items) in one method.
        '''

        samples = []

        picks = [np.random.randint(0, self.no_of_persons, self.no_of_persons)
                 for sample in range(no_of_samples)]
        picks = [self.dataframe.index.get_level_values(1)[pick] for pick in picks]

        data_dict = {rater: self.dataframe.xs(rater) for rater in self.raters}

        for sample in range(no_of_samples):
            sample_data_dict = {rater: pd.DataFrame([data_dict[rater].loc[pick]
                                                     for pick in picks[sample]]).reset_index(drop=True)
                                for rater in self.raters}

            samples.append(pd.concat(sample_data_dict.values(), keys=sample_data_dict.keys()))

        samples = [MFRM(sample, self.max_score) for sample in samples]

        for sample in samples:
            sample.calibrate_items(constant=constant, method=method,
                                   matrix_power=matrix_power, log_lik_tol=log_lik_tol)

            if anchor_raters is not None:
                sample.calibrate_items_anchor(anchor_raters, constant=constant, method=method,
                                              matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if anchor_raters is not None:
            item_ests = np.array([sample.anchor_diffs_items.values for sample in samples])
            threshold_ests = np.array([sample.anchor_thresholds_items for sample in samples])

            rater_ests = {sample_no: pd.DataFrame.from_dict(sample.anchor_severities_items, orient='index')
                          for sample_no, sample in enumerate(samples)}
            rater_ests = pd.concat(rater_ests.values(), keys=rater_ests.keys())
            rater_ests = rater_ests.swaplevel(0, 1)

        else:
            item_ests = np.array([sample.diffs.values for sample in samples])
            threshold_ests = np.array([sample.thresholds for sample in samples])

            rater_ests = {sample_no: pd.DataFrame.from_dict(sample.severities_items, orient='index')
                          for sample_no, sample in enumerate(samples)}
            rater_ests = pd.concat(rater_ests.values(), keys=rater_ests.keys())
            rater_ests = rater_ests.swaplevel(0, 1)

        item_se = {item: se for item, se in zip(self.dataframe.columns, np.nanstd(item_ests, axis=0))}
        item_se = pd.Series(item_se)

        if interval is not None:
            item_low = {item: low for item, low in zip(self.dataframe.columns,
                                                       np.percentile(item_ests,
                                                                     50 * (1 - interval), axis=0))}
            item_low = pd.Series(item_low)
            item_high = {item: high for item, high in zip(self.dataframe.columns,
                                                          np.percentile(item_ests,
                                                                        50 * (1 + interval), axis=0))}
            item_high = pd.Series(item_high)

        else:
            item_low = None
            item_high = None

        threshold_se = np.std(threshold_ests, axis=0)

        if interval is not None:
            threshold_low = np.percentile(threshold_ests, 50 * (1 - interval), axis=0)
            threshold_high = np.percentile(threshold_ests, 50 * (1 + interval), axis=0)

        else:
            threshold_low = None
            threshold_high = None

        cat_widths = {cat + 1: threshold_ests[:,cat + 2] - threshold_ests[:,cat + 1]
                      for cat in range(self.max_score - 1)}
        cat_width_se = {cat: np.nanstd(estimates)
                        for cat, estimates in cat_widths.items()}

        if interval is not None:
            cat_width_low = {cat: np.percentile(estimates, 50 * (1 - interval))
                            for cat, estimates in cat_widths.items()}
            cat_width_high = {cat: np.percentile(estimates, 50 * (1 + interval))
                            for cat, estimates in cat_widths.items()}

        else:
            cat_width_low = None
            cat_width_high = None

        rater_se = {rater: np.std(rater_ests.xs(rater), axis=0) for rater in self.raters}

        if interval is not None:
            rater_low = {rater: {item: percentile
                                 for item, percentile in zip(self.dataframe.columns,
                                                             np.percentile(rater_ests.xs(rater),
                                                                           50 * (1 - interval), axis=0))}
                         for rater in self.raters}

            rater_high = {rater: {item: percentile
                                  for item, percentile in zip(self.dataframe.columns,
                                                              np.percentile(rater_ests.xs(rater),
                                                                            50 * (1 + interval), axis=0))}
                          for rater in self.raters}

        else:
            rater_low = None
            rater_high = None

        if anchor_raters is not None:
            self.anchor_item_bootstrap_items = item_ests
            self.anchor_item_se_items = item_se
            self.anchor_item_low_items = item_low
            self.anchor_item_high_items = item_high
            self.anchor_threshold_bootstrap_items = threshold_ests
            self.anchor_threshold_se_items = threshold_se
            self.anchor_threshold_low_items = threshold_low
            self.anchor_threshold_high_items = threshold_high
            self.anchor_cat_width_bootstrap_items = cat_widths
            self.anchor_cat_width_se_items = cat_width_se
            self.anchor_cat_width_low_items = cat_width_low
            self.anchor_cat_width_high_items = cat_width_high
            self.anchor_rater_bootstrap_items = rater_ests
            self.anchor_rater_se_items = rater_se
            self.anchor_rater_low_items = rater_low
            self.anchor_rater_high_items = rater_high

        else:
            self.item_bootstrap_items = item_ests
            self.item_se = item_se
            self.item_low = item_low
            self.item_high = item_high
            self.threshold_bootstrap_items = threshold_ests
            self.threshold_se_items = threshold_se
            self.threshold_low_items = threshold_low
            self.threshold_high_items = threshold_high
            self.cat_width_bootstrap_items = cat_widths
            self.cat_width_se_items = cat_width_se
            self.cat_width_low_items = cat_width_low
            self.cat_width_high_items = cat_width_high
            self.rater_bootstrap_items = rater_ests
            self.rater_se_items = rater_se
            self.rater_low_items = rater_low
            self.rater_high_items = rater_high

    def std_errors_thresholds(self,
                              anchor_raters=None,
                              interval=None,
                              no_of_samples=100,
                              constant=0.1,
                              method='cos',
                              matrix_power=3,
                              log_lik_tol=0.000001):

        '''
        Estimates items, thresholds and rater severities (thresholds) in one method.
        '''

        samples = []

        picks = [np.random.randint(0, self.no_of_persons, self.no_of_persons)
                 for sample in range(no_of_samples)]
        picks = [self.dataframe.index.get_level_values(1)[pick] for pick in picks]

        data_dict = {rater: self.dataframe.xs(rater) for rater in self.raters}

        for sample in range(no_of_samples):
            sample_data_dict = {rater: pd.DataFrame([data_dict[rater].loc[pick]
                                                     for pick in picks[sample]]).reset_index(drop=True)
                                for rater in self.raters}

            samples.append(pd.concat(sample_data_dict.values(), keys=sample_data_dict.keys()))

        samples = [MFRM(sample, self.max_score) for sample in samples]

        for sample in samples:
            sample.calibrate_thresholds(constant=constant, method=method,
                                        matrix_power=matrix_power, log_lik_tol=log_lik_tol)

            if anchor_raters is not None:
                sample.calibrate_thresholds_anchor(anchor_raters, constant=constant, method=method,
                                                   matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if anchor_raters is not None:
            item_ests = np.array([sample.anchor_diffs_thresholds.values for sample in samples])
            threshold_ests = np.array([sample.anchor_thresholds_thresholds for sample in samples])
            rater_ests = np.array([list(sample.anchor_severities_thresholds.values()) for sample in samples])

        else:
            item_ests = np.array([sample.diffs.values for sample in samples])
            threshold_ests = np.array([sample.thresholds for sample in samples])
            rater_ests = np.array([list(sample.severities_thresholds.values()) for sample in samples])

        item_se = {item: se for item, se in zip(self.dataframe.columns, np.nanstd(item_ests, axis=0))}
        item_se = pd.Series(item_se)

        if interval is not None:
            item_low = {item: low for item, low in zip(self.dataframe.columns,
                                                       np.percentile(item_ests,
                                                                     50 * (1 - interval), axis=0))}
            item_low = pd.Series(item_low)
            item_high = {item: high for item, high in zip(self.dataframe.columns,
                                                          np.percentile(item_ests,
                                                                        50 * (1 + interval), axis=0))}
            item_high = pd.Series(item_high)

        else:
            item_low = None
            item_high = None

        threshold_se = np.std(threshold_ests, axis=0)

        if interval is not None:
            threshold_low = np.percentile(threshold_ests, 50 * (1 - interval), axis=0)
            threshold_high = np.percentile(threshold_ests, 50 * (1 + interval), axis=0)

        else:
            threshold_low = None
            threshold_high = None

        cat_widths = {cat + 1: threshold_ests[:,cat + 2] - threshold_ests[:, cat + 1]
                      for cat in range(self.max_score - 1)}
        cat_width_se = {cat: np.nanstd(estimates)
                        for cat, estimates in cat_widths.items()}

        if interval is not None:
            cat_width_low = {cat: np.percentile(estimates, 50 * (1 - interval))
                            for cat, estimates in cat_widths.items()}
            cat_width_high = {cat: np.percentile(estimates, 50 * (1 + interval))
                            for cat, estimates in cat_widths.items()}

        else:
            cat_width_low = None
            cat_width_high = None

        rater_se = {rater: np.std(rater_ests[i, :], axis=0)
                    for i, rater in enumerate(self.raters)}

        if interval is not None:
            rater_low = {rater: np.percentile(rater_ests[i, :], 50 * (1 - interval), axis=0)
                         for i, rater in enumerate(self.raters)}

            rater_high = {rater: np.percentile(rater_ests[i, :], 50 * (1 + interval), axis=0)
                          for i, rater in enumerate(self.raters)}

        else:
            rater_low = None
            rater_high = None

        if anchor_raters is not None:
            self.anchor_item_bootstrap_thresholds = item_ests
            self.anchor_item_se_thresholds = item_se
            self.anchor_item_low_thresholds = item_low
            self.anchor_item_high_thresholds = item_high
            self.anchor_threshold_bootstrap_thresholds = threshold_ests
            self.anchor_threshold_se_thresholds = threshold_se
            self.anchor_threshold_low_thresholds = threshold_low
            self.anchor_threshold_high_thresholds = threshold_high
            self.anchor_cat_width_bootstrap_thresholds = cat_widths
            self.anchor_cat_width_se_thresholds = cat_width_se
            self.anchor_cat_width_low_thresholds = cat_width_low
            self.anchor_cat_width_high_thresholds = cat_width_high
            self.anchor_rater_bootstrap_thresholds = rater_ests
            self.anchor_rater_se_thresholds = rater_se
            self.anchor_rater_low_thresholds = rater_low
            self.anchor_rater_high_thresholds = rater_high

        else:
            self.item_bootstrap_thresholds = item_ests
            self.item_se = item_se
            self.item_low = item_low
            self.item_high = item_high
            self.threshold_bootstrap_thresholds = threshold_ests
            self.threshold_se_thresholds = threshold_se
            self.threshold_low_thresholds = threshold_low
            self.threshold_high_thresholds = threshold_high
            self.cat_width_bootstrap_thresholds = cat_widths
            self.cat_width_se_thresholds = cat_width_se
            self.cat_width_low_thresholds = cat_width_low
            self.cat_width_high_thresholds = cat_width_high
            self.rater_bootstrap_thresholds = rater_ests
            self.rater_se_thresholds = rater_se
            self.rater_low_thresholds = rater_low
            self.rater_high_thresholds = rater_high

    def std_errors_matrix(self,
                          anchor_raters=None,
                          interval=None,
                          no_of_samples=100,
                          constant=0.1,
                          method='cos',
                          matrix_power=3,
                          log_lik_tol=0.000001):

        '''
        Estimates items, thresholds and rater severities (matrix) in one method.
        '''

        samples = []

        picks = [np.random.randint(0, self.no_of_persons, self.no_of_persons)
                 for sample in range(no_of_samples)]
        picks = [self.dataframe.index.get_level_values(1)[pick] for pick in picks]

        data_dict = {rater: self.dataframe.xs(rater) for rater in self.raters}

        for sample in range(no_of_samples):
            sample_data_dict = {rater: pd.DataFrame([data_dict[rater].loc[pick]
                                                     for pick in picks[sample]]).reset_index(drop=True)
                                for rater in self.raters}

            samples.append(pd.concat(sample_data_dict.values(), keys=sample_data_dict.keys()))

        samples = [MFRM(sample, self.max_score) for sample in samples]

        for sample in samples:
            sample.calibrate_matrix(constant=constant, method=method,
                                    matrix_power=matrix_power, log_lik_tol=log_lik_tol)

            if anchor_raters is not None:
                sample.calibrate_matrix_anchor(anchor_raters, constant=constant, method=method,
                                               matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if anchor_raters is not None:
            item_ests = np.array([sample.anchor_diffs_matrix.values for sample in samples])
            threshold_ests = np.array([sample.anchor_thresholds_matrix for sample in samples])

            rater_ests = {i: sample.anchor_severities_matrix for i, sample in enumerate(samples)}
            rater_ests = {rater: {i: rater_ests[i][rater] for i, sample in enumerate(samples)}
                          for rater in self.raters}

        else:
            item_ests = np.array([sample.diffs.values for sample in samples])
            threshold_ests = np.array([sample.thresholds for sample in samples])

            rater_ests = {i: sample.severities_matrix for i, sample in enumerate(samples)}
            rater_ests = {rater: {i: rater_ests[i][rater] for i, sample in enumerate(samples)}
                          for rater in self.raters}

        for rater in self.raters:
            for i in range(no_of_samples):
                rater_ests[rater][i] = pd.DataFrame.from_dict(rater_ests[rater][i]).T

        if anchor_raters is not None:
            marginal_rater_ests_items = {i: sample.anchor_marginal_severities_items
                                         for i, sample in enumerate(samples)}
        else:
            marginal_rater_ests_items = {i: sample.marginal_severities_items
                                         for i, sample in enumerate(samples)}

        marginal_rater_ests_items = {rater: {i: marginal_rater_ests_items[i][rater]
                                             for i, sample in enumerate(samples)}
                                     for rater in self.raters}

        if anchor_raters is not None:
            marginal_rater_ests_thresholds = {i: sample.anchor_marginal_severities_thresholds
                                              for i, sample in enumerate(samples)}
        else:
            marginal_rater_ests_thresholds = {i: sample.marginal_severities_thresholds
                                              for i, sample in enumerate(samples)}

        marginal_rater_ests_thresholds = {rater: {i: marginal_rater_ests_thresholds[i][rater]
                                                  for i, sample in enumerate(samples)}
                                          for rater in self.raters}

        rater_ests_concat = {rater: pd.concat((rater_ests[rater][i] for i, sample in enumerate(samples)))
                             for rater in self.raters}
        by_row_index = {rater: rater_ests_concat[rater].groupby(rater_ests_concat[rater].index)
                        for rater in self.raters}

        item_se = {item: se for item, se in zip(self.dataframe.columns, np.nanstd(item_ests, axis=0))}
        item_se = pd.Series(item_se)

        if interval is not None:
            item_low = {item: low for item, low in zip(self.dataframe.columns,
                                                       np.percentile(item_ests,
                                                                     50 * (1 - interval), axis=0))}
            item_low = pd.Series(item_low)
            item_high = {item: high for item, high in zip(self.dataframe.columns,
                                                           np.percentile(item_ests,
                                                                         50 * (1 + interval), axis=0))}
            item_high = pd.Series(item_high)

        else:
            item_low = None
            item_high = None

        threshold_se = np.std(threshold_ests, axis=0)

        if interval is not None:
            threshold_low = np.percentile(threshold_ests, 50 * (1 - interval), axis=0)
            threshold_high = np.percentile(threshold_ests, 50 * (1 + interval), axis=0)

        else:
            threshold_low = None
            threshold_high = None

        cat_widths = {cat + 1: threshold_ests[:,cat + 2] - threshold_ests[:,cat + 1]
                      for cat in range(self.max_score - 1)}
        cat_width_se = {cat: np.nanstd(estimates)
                        for cat, estimates in cat_widths.items()}

        if interval is not None:
            cat_width_low = {cat: np.percentile(estimates, 50 * (1 - interval))
                            for cat, estimates in cat_widths.items()}
            cat_width_high = {cat: np.percentile(estimates, 50 * (1 + interval))
                            for cat, estimates in cat_widths.items()}

        else:
            cat_width_low = None
            cat_width_high = None

        rater_se = {rater: by_row_index[rater].std() for rater in self.raters}
        rater_se = {rater: {item: rater_se[rater].loc[item]
                            for item in self.dataframe.columns}
                    for rater in self.raters}

        rater_se_marginal_items = {rater: {item: pd.DataFrame(marginal_rater_ests_items[rater]).std(axis=1).iloc[i]
                                           for i, item in enumerate(self.dataframe.columns)}
                                   for rater in self.raters}

        rater_se_marginal_thresholds = {rater: np.array(pd.DataFrame(marginal_rater_ests_thresholds[rater]).std(axis=1))
                                        for rater in self.raters}

        if interval is not None:
            rater_low = {rater: by_row_index[rater].quantile((1 - interval) / 2)
                         for rater in self.raters}
            rater_low = {rater: {item: rater_low[rater].loc[item]
                                 for item in self.dataframe.columns}
                         for rater in self.raters}

            rater_low_marginal_items = {rater: {item:
                                                pd.DataFrame(marginal_rater_ests_items[rater]).quantile((1 - interval) / 2,
                                                                                                        axis=1).iloc[i]
                                                for i, item in enumerate(self.dataframe.columns)}
                                        for rater in self.raters}

            rater_low_marginal_thresholds = {rater:
                                             pd.DataFrame(marginal_rater_ests_thresholds[rater]).quantile((1 - interval) / 2,
                                                                                                          axis=1)
                                             for rater in self.raters}

            rater_high = {rater: by_row_index[rater].quantile((1 + interval) / 2)
                               for rater in self.raters}
            rater_high = {rater: {item: rater_high[rater].loc[item]
                                       for item in self.dataframe.columns}
                               for rater in self.raters}

            rater_high_marginal_items = {rater: {item:
                                                 pd.DataFrame(marginal_rater_ests_items[rater]).quantile((1 + interval) / 2,
                                                                                                         axis=1).iloc[i]
                                                 for i, item in enumerate(self.dataframe.columns)}
                                         for rater in self.raters}

            rater_high_marginal_thresholds = {rater:
                                              pd.DataFrame(marginal_rater_ests_thresholds[rater]).quantile((1 + interval) / 2,
                                                                                                           axis=1)
                                              for rater in self.raters}

        else:
            rater_low = None
            rater_low_marginal_items = None
            rater_low_marginal_thresholds = None
            rater_high = None
            rater_high_marginal_items = None
            rater_high_marginal_thresholds = None

        if anchor_raters is not None:
            self.anchor_item_bootstrap_matrix = item_ests
            self.anchor_item_se_matrix = item_se
            self.anchor_item_low_matrix = item_low
            self.anchor_item_high_matrix = item_high
            self.anchor_threshold_bootstrap_matrix = threshold_ests
            self.anchor_threshold_se_matrix = threshold_se
            self.anchor_threshold_low_matrix = threshold_low
            self.anchor_threshold_high_matrix = threshold_high
            self.anchor_cat_width_bootstrap_matrix = cat_widths
            self.anchor_cat_width_se_matrix = cat_width_se
            self.anchor_cat_width_low_matrix = cat_width_low
            self.anchor_cat_width_high_matrix = cat_width_high
            self.anchor_rater_bootstrap_matrix = rater_ests
            self.anchor_rater_se_matrix = rater_se
            self.anchor_rater_low_matrix = rater_low
            self.anchor_rater_high_matrix = rater_high
            self.anchor_rater_se_marginal_items = rater_se_marginal_items
            self.anchor_rater_low_marginal_items = rater_low_marginal_items
            self.anchor_rater_high_marginal_items = rater_high_marginal_items
            self.anchor_rater_se_marginal_thresholds = rater_se_marginal_thresholds
            self.anchor_rater_low_marginal_thresholds = rater_low_marginal_thresholds
            self.anchor_rater_high_marginal_thresholds = rater_high_marginal_thresholds

        else:
            self.item_bootstrap_matrix = item_ests
            self.item_se = item_se
            self.item_low = item_low
            self.item_high = item_high
            self.threshold_bootstrap_matrix = threshold_ests
            self.threshold_se_matrix = threshold_se
            self.threshold_low_matrix = threshold_low
            self.threshold_high_matrix = threshold_high
            self.cat_width_bootstrap_matrix = cat_widths
            self.cat_width_se_matrix = cat_width_se
            self.cat_width_low_matrix = cat_width_low
            self.cat_width_high_matrix = cat_width_high
            self.rater_bootstrap_matrix = rater_ests
            self.rater_se_matrix = rater_se
            self.rater_low_matrix = rater_low
            self.rater_high_matrix = rater_high
            self.rater_se_marginal_items = rater_se_marginal_items
            self.rater_low_marginal_items = rater_low_marginal_items
            self.rater_high_marginal_items = rater_high_marginal_items
            self.rater_se_marginal_thresholds = rater_se_marginal_thresholds
            self.rater_low_marginal_thresholds = rater_low_marginal_thresholds
            self.rater_high_marginal_thresholds = rater_high_marginal_thresholds

    def calibrate_global_anchor(self,
                                anchor_raters,
                                calibrate=False,
                                constant=0.1,
                                method='cos',
                                matrix_power=3,
                                log_lik_tol=0.000001):

        if calibrate:
            self.calibrate_global(constant=constant, method=method,
                                  matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        self.anchor_diffs_global = self.diffs.copy()
        self.anchor_thresholds_global = self.thresholds.copy()
        self.anchor_severities_global = self.severities_global.copy()

        anchor_severities = [self.severities_global[rater] for rater in anchor_raters]
        severity_adjustment = np.mean(anchor_severities)

        for rater in self.raters:
            self.anchor_severities_global[rater] -= severity_adjustment

        self.anchor_raters_global = anchor_raters

    def std_errors_global_anchor(self,
                                 anchor_raters,
                                 interval=None,
                                 no_of_samples=100,
                                 constant=0.1,
                                 method='cos',
                                 matrix_power=3,
                                 log_lik_tol=0.000001):

        '''
        Estimates SEs of anchored estimates
        '''

        samples = []

        picks = [np.random.randint(0, self.no_of_persons, self.no_of_persons)
                 for sample in range(no_of_samples)]
        picks = [self.dataframe.index.get_level_values(1)[pick] for pick in picks]

        data_dict = {rater: self.dataframe.xs(rater) for rater in self.raters}

        for sample in range(no_of_samples):
            sample_data_dict = {rater: pd.DataFrame([data_dict[rater].loc[pick]
                                                     for pick in picks[sample]]).reset_index(drop=True)
                                for rater in self.raters}

            samples.append(pd.concat(sample_data_dict.values(), keys=sample_data_dict.keys()))

        samples = [MFRM(sample, self.max_score) for sample in samples]

        for sample in samples:
            sample.calibrate_global_anchor(anchor_raters, interval=interval, calibrate=True, constant=constant,
                                           method=method, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        item_ests = np.array([sample.anchor_diffs_global.values for sample in samples])
        threshold_ests = np.array([sample.anchor_thresholds_global for sample in samples])
        rater_ests = np.array([sample.anchor_severities_global.values for sample in samples])

        item_se = {item: se for item, se in zip(self.dataframe.columns, np.nanstd(item_ests, axis=0))}
        item_se = pd.Series(item_se)

        if interval is not None:
            item_low = {item: low for item, low in zip(self.dataframe.columns,
                                                       np.percentile(item_ests,
                                                                     50 * (1 - interval), axis=0))}
            item_low = pd.Series(item_low)
            item_high = {item: high for item, high in zip(self.dataframe.columns,
                                                          np.percentile(item_ests,
                                                                        50 * (1 + interval), axis=0))}
            item_high = pd.Series(item_high)

        else:
            item_low = None
            item_high = None

        threshold_se = np.nanstd(threshold_ests, axis=0)

        if interval is not None:
            threshold_low = np.percentile(threshold_ests, 50 * (1 - interval), axis=0)
            threshold_high = np.percentile(threshold_ests, 50 * (1 + interval), axis=0)

        else:
            threshold_low = None
            threshold_high = None

        rater_se = {rater: se for rater, se in zip(self.raters, np.std(rater_ests, axis=0))}
        rater_se = pd.Series(rater_se)

        if interval is not None:
            rater_low = {rater: percentile
                         for rater, percentile in zip(self.raters, np.percentile(rater_ests,
                                                                                 50 * (1 - interval), axis=0))}
            rater_low = pd.Series(rater_low)
            rater_high = {rater: percentile
                          for rater, percentile in zip(self.raters, np.percentile(rater_ests,
                                                                                  50 * (1 + interval), axis=0))}
            rater_high = pd.Series(rater_high)

        else:
            rater_low = None
            rater_high = None

        self.anchor_item_se = item_se
        self.anchor_item_low = item_low
        self.anchor_item_high = item_high
        self.anchor_threshold_se_global = threshold_se
        self.anchor_threshold_low_global = threshold_low
        self.anchor_threshold_high_global = threshold_high
        self.anchor_rater_se_global = rater_se
        self.anchor_rater_low_global = rater_low
        self.anchor_rater_high_global = rater_high

    def calibrate_items_anchor(self,
                               anchor_raters,
                               calibrate=False,
                               constant=0.1,
                               method='cos',
                               matrix_power=3,
                               log_lik_tol=0.000001):

        if calibrate:
            self.calibrate_items(constant=constant, method=method,
                                 matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        self.anchor_diffs_items = self.diffs.copy()
        self.anchor_thresholds_items = self.thresholds.copy()

        severities_items_df = pd.DataFrame(self.severities_items).T

        anchor_severities_df = pd.DataFrame(self.severities_items).T
        anchor_severities_df = anchor_severities_df.loc[anchor_raters]

        severity_adjustments = anchor_severities_df.mean(axis=0)

        for i, item in enumerate(self.dataframe.columns):
            self.anchor_diffs_items[item] += severity_adjustments.iloc[i]

        for rater in self.raters:
            severities_items_df.loc[rater] -= severity_adjustments
        self.anchor_severities_items = {rater: {item: severities_items_df.loc[rater].iloc[i]
                                                for i, item in enumerate(self.dataframe.columns)}
                                        for rater in self.raters}

        diff_centraliser = self.anchor_diffs_items.mean()
        for item in self.dataframe.columns:
            self.anchor_diffs_items[item] -= diff_centraliser

        self.anchor_raters_items = anchor_raters

    def calibrate_thresholds_anchor(self,
                                    anchor_raters,
                                    calibrate=False,
                                    constant=0.1,
                                    method='cos',
                                    matrix_power=3,
                                    log_lik_tol=0.000001):

        if calibrate:
            self.calibrate_thresholds(constant=constant, method=method,
                                      matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        self.anchor_diffs_thresholds = self.diffs.copy()
        self.anchor_thresholds_thresholds = self.thresholds.copy()

        severities_thresholds_df = pd.DataFrame(self.severities_thresholds).T

        anchor_severities_df = pd.DataFrame(self.severities_thresholds).T
        anchor_severities_df = anchor_severities_df.loc[anchor_raters]

        severity_adjustments = anchor_severities_df.mean(axis=0)[1:]

        self.anchor_thresholds_thresholds[1:] += severity_adjustments

        for rater in self.raters:
            severities_thresholds_df.loc[rater, 1:] -= severity_adjustments
        self.anchor_severities_thresholds = {rater: severities_thresholds_df.loc[rater]
                                             for rater in self.raters}

        self.anchor_thresholds_thresholds[1:] -= self.anchor_thresholds_thresholds[1:].mean()

        self.anchor_raters_thresholds = anchor_raters

    def calibrate_matrix_anchor(self,
                                anchor_raters,
                                calibrate=False,
                                constant=0.1,
                                method='cos',
                                matrix_power=3,
                                log_lik_tol=0.000001):
        '''
        Joint anchoring for the matrix rater parameterisation.

        The matrix parameterisation has sigma_rik varying by rater r, item i,
        and threshold k simultaneously. Anchoring sets the mean severity of the
        anchor raters to zero and absorbs the adjustment into the item difficulties
        and thresholds.

        The adjustment matrix A[i,k] = mean_{r in anchor}(sigma_rik) is decomposed
        via the two-way ANOVA identity:
            A[i,k] = mu + alpha[i] + beta[k] + gamma[i,k]
        where mu is the grand mean, alpha[i] the item main effect, beta[k] the
        threshold main effect, and gamma[i,k] the interaction.

        Items absorb (mu + alpha[i]), thresholds absorb beta[k], and gamma[i,k]
        stays in the severity matrix (it cannot be absorbed into a single facet
        since it varies jointly -- this is precisely what the matrix parameterisation
        captures). The sequential approach (original) marginalised separately and
        double-counted the grand mean, leaving a constant offset.
        '''

        if calibrate:
            self.calibrate_matrix(constant=constant, method=method,
                                  matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        self.anchor_diffs_matrix      = self.diffs.copy()
        self.anchor_thresholds_matrix = self.thresholds.copy()

        # Build (R, I, K+1) severity array for all raters
        sev_array = np.array([
            [self.severities_matrix[rater][item]
             for item in self.dataframe.columns]
            for rater in self.raters
        ])  # shape (R, I, K+1); sev_array[r, i, 0] = 0 sentinel

        # Build (R_anchor, I, K+1) for anchor raters only
        anchor_idx = [list(self.raters).index(r) for r in anchor_raters]
        anchor_sev = sev_array[anchor_idx, :, :]   # (R_anchor, I, K+1)

        # A[i, k] = mean over anchor raters -- the (I, K+1) adjustment matrix
        # Operate only on thresholds 1..K (index 1 onward); index 0 is sentinel
        A = anchor_sev[:, :, 1:].mean(axis=0)       # (I, K)

        # Two-way ANOVA decomposition: A = mu + alpha + beta + gamma
        mu      = A.mean()                           # scalar grand mean
        alpha   = A.mean(axis=1) - mu               # (I,)  item main effect
        beta    = A.mean(axis=0) - mu               # (K,)  threshold main effect
        # gamma = A - mu - alpha[:,None] - beta[None,:] stays in severities

        # Absorb item component (mu + alpha[i]) into item difficulties
        for i, item in enumerate(self.dataframe.columns):
            self.anchor_diffs_matrix[item] += mu + alpha[i]

        # Absorb threshold main effect beta[k] into thresholds
        self.anchor_thresholds_matrix[1:] += beta

        # Remove full adjustment A[i,k] from all raters' severities
        # The residual interaction gamma[i,k] = A - mu - alpha - beta remains
        sev_array_adj = sev_array.copy()
        sev_array_adj[:, :, 1:] -= A[None, :, :]   # broadcast over raters

        # Re-centre: item difficulties and thresholds should have zero mean
        diff_centre = self.anchor_diffs_matrix.mean()
        self.anchor_diffs_matrix -= diff_centre
        # Compensate in severities: add diff_centre back as item offset per rater
        sev_array_adj[:, :, 1:] += diff_centre

        thresh_centre = self.anchor_thresholds_matrix[1:].mean()
        self.anchor_thresholds_matrix[1:] -= thresh_centre
        sev_array_adj[:, :, 1:] += thresh_centre

        # Rebuild severity dict from adjusted array
        self.anchor_severities_matrix = {
            rater: {
                item: sev_array_adj[r, i, :]
                for i, item in enumerate(self.dataframe.columns)
            }
            for r, rater in enumerate(self.raters)
        }

        # Compute marginal severities for downstream use
        # marginal by item: mean over thresholds (indices 1..K)
        self.anchor_marginal_severities_items = {
            rater: pd.Series({
                item: sev_array_adj[r, i, 1:].mean()
                for i, item in enumerate(self.dataframe.columns)
            })
            for r, rater in enumerate(self.raters)
        }

        # marginal by threshold: mean over items, with tau_0=0 sentinel
        self.anchor_marginal_severities_thresholds = {
            rater: pd.Series(
                np.concatenate([[0.0], sev_array_adj[r, :, 1:].mean(axis=0)])
            )
            for r, rater in enumerate(self.raters)
        }
        # Centre threshold marginals
        for rater in self.raters:
            r = list(self.raters).index(rater)
            adj = self.anchor_marginal_severities_thresholds[rater].iloc[1:].mean()
            self.anchor_marginal_severities_thresholds[rater].iloc[1:] -= adj

        self.anchor_raters_matrix = anchor_raters

    def abil_global(self,
                    persons,
                    anchor=False,
                    items=None,
                    raters=None,
                    warm_corr=True,
                    tolerance=0.00001,
                    max_iters=100,
                    ext_score_adjustment=0.5):

        '''
        Creates a raw score to ability estimate look-up table for a set
        of items using ML estimation (Newton-Raphson procedure) with
        optional Warm (1989) bias correction.
        '''

        if isinstance(persons, str):
            if persons == 'all':
                persons = self.persons

            else:
                persons = [persons]

        if persons is None:
            persons = self.persons

        if isinstance(items, str):
            if items == 'all':
                items = self.items.tolist()

            else:
                items = [items]
                
        if items is None:
            items = self.items
         
        if raters is None:
            raters = self.raters
         
        if isinstance(raters, str):
            if raters == 'all':
                raters = self.raters

        if isinstance(raters, pd.core.indexes.base.Index):
            raters = raters.tolist()

        if anchor:
            if hasattr(self, 'anchor_diffs_global'):
                difficulties = self.anchor_diffs_global.loc[items]
                thresholds = self.anchor_thresholds_global
                severities = self.anchor_severities_global.loc[raters]

            else:
                print('Anchor calibration required')
                return

        else:
            difficulties = self.diffs.loc[items]
            thresholds = self.thresholds
            severities = self.severities_global.loc[raters]
          
        person_data = self.dataframe.loc[(raters, persons), items]
        person_filter = person_data.notna().astype(float).replace(0, np.nan)

        scores = {rater: person_data.loc[rater].sum(axis=1).astype(float)
                  for rater in raters}
        scores = sum(scores.values())

        ext_scores = {rater: person_filter.loc[rater].sum(axis=1) * self.max_score
                      for rater in raters}
        ext_scores = sum(ext_scores.values())

        scores[scores == 0] = ext_score_adjustment
        scores[scores == ext_scores] -= ext_score_adjustment

        diff_df = pd.concat([difficulties for person in persons], axis=1).T
        diff_df.index = persons

        mean_diffs = {rater: diff_df * person_filter.loc[rater]
                      for rater in raters}
        mean_diffs = sum(mean_diffs[rater].sum(axis=1)
                         for rater in raters)

        item_count = {rater: person_filter.loc[rater]
                      for rater in raters}
        item_count = sum(item_count[rater].sum(axis=1)
                         for rater in raters)

        mean_diffs /= item_count

        try:
            estimates = np.log(scores) - np.log(ext_scores - scores) + mean_diffs
            changes = pd.Series({person: 1 for person in persons})
            iters = 0

            while (abs(changes).max() > tolerance) & (iters <= max_iters):

                c_p_df = {item: estimates - difficulties[item] for item in items}
                c_p_df = {rater: pd.DataFrame(c_p_df) - severities.loc[rater]
                          for rater in raters}
                c_p_df = pd.concat(c_p_df.values(), keys=c_p_df.keys())

                cat_prob_dict = {cat: (cat * c_p_df) - sum(thresholds[:cat + 1])
                                 for cat in range(self.max_score + 1)}

                for cat in range(self.max_score + 1):
                    cat_prob_dict[cat] = np.exp(cat_prob_dict[cat])

                den = sum(cat_prob_dict[cat] for cat in range(self.max_score + 1))

                for cat in range(self.max_score + 1):
                    cat_prob_dict[cat] /= den
                    cat_prob_dict[cat] *= person_filter

                exp_score_df = sum(cat * df for cat, df in cat_prob_dict.items())
                exp_score_df *= person_filter

                info_df = sum(df * (cat - exp_score_df) ** 2 for cat, df in cat_prob_dict.items())
                info_df *= person_filter

                result_list = sum(exp_score_df.loc[rater].sum(axis=1) for rater in raters)
                info_list = sum(info_df.loc[rater].sum(axis=1) for rater in raters)

                changes = (result_list - scores) / info_list
                changes = changes.clip(-1, 1)
                estimates -= changes
                iters += 1

            # Per-person convergence check: set non-converged persons to NaN
            # Prevents runaway +/-1 logit drift over max_iters for extreme scorers
            if iters >= max_iters:
                not_converged = abs(changes) > tolerance
                if not_converged.any():
                    print(f'Warning: {int(not_converged.sum())} person(s) did not converge '
                          f'in abil_global() and will be set to NaN.')
                    estimates[not_converged] = np.nan

            if warm_corr:
                valid = estimates.notna()
                if valid.any():
                    # Apply Warm correction only to converged persons
                    valid_idx = estimates.index[valid]
                    if isinstance(person_filter.index, pd.MultiIndex):
                        valid_pf = person_filter.loc[(slice(None), valid_idx), :]
                    else:
                        valid_pf = person_filter.loc[valid_idx]
                    estimates[valid] += self.warm_global(
                        estimates[valid], items, raters, severities, valid_pf
                    )

        except Exception as e:
            print(f'abil_global() failed: {e}')
            estimates = pd.Series(np.nan, index=list(persons))

        return estimates

    def person_abils_global(self,
                            anchor=False,
                            items=None,
                            raters=None,
                            warm_corr=True,
                            tolerance=0.00001,
                            max_iters=100,
                            ext_score_adjustment=0.5):

        '''
        Creates raw score to ability estimate look-up table. Newton-Raphson ML
        estimation, includes optional Warm (1989) bias correction.
        '''

        estimates = self.abil_global(persons=None, anchor=anchor, items=items, raters=raters,
                                     warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                     ext_score_adjustment=ext_score_adjustment)

        if anchor:
            self.anchor_abils_global = estimates

        else:
            self.abils_global = estimates

    def score_abil_global(self,
                          score,
                          anchor=False,
                          items=None,
                          raters=None,
                          warm_corr=True,
                          tolerance=0.00001,
                          max_iters=100,
                          ext_score_adjustment=0.5):

        if isinstance(items, str):
            if items == 'all':
                items = self.items
            else:
                items = [items]

        if items is None:
            items = self.items

        if isinstance(raters, str):
            if raters == 'all':
                raters = [rater for rater in self.raters]
            else:
                raters = [raters]

        if anchor:
            if hasattr(self, 'anchor_diffs_global'):
                difficulties = self.anchor_diffs_global.loc[items]
                thresholds = self.anchor_thresholds_global
                severities = self.anchor_severities_global

            else:
                print('Anchor calibration required')
                return

        else:
            difficulties = self.diffs.loc[items]
            thresholds = self.thresholds
            severities = self.severities_global

        if raters is None:
            severities = pd.Series({'dummy_rater': 0})
            raters = ['dummy_rater']

            if items is None:
                person_filter = np.array([1 for item in self.dataframe.columns])

            else:
                person_filter = np.array([1 for item in items])

        else:
            if items is None:
                person_filter = np.array([[1 for item in self.dataframe.columns]
                                          for rater in raters])

            else:
                person_filter = np.array([[1 for item in items]
                                          for rater in raters])

        severities = severities.loc[raters]

        ext_score = person_filter.sum() * self.max_score

        if score == 0:
            score = ext_score_adjustment

        elif score == ext_score:
            score -= ext_score_adjustment

        estimate = log(score) - log(ext_score - score) + difficulties.mean()
        change = 1
        iters = 0

        while (abs(change) > tolerance) & (iters <= max_iters):

            if raters is None:
                if items is None:
                    exp_list = [self.exp_score_global(estimate, item, difficulties, 'dummy_rater',
                                                      dummy_sevs, thresholds)
                                for item in self.dataframe.columns]
    
                    info_list = [self.variance_global(estimate, item, difficulties, 'dummy_rater',
                                                      dummy_sevs, thresholds)
                                 for item in self.dataframe.columns]
                    
                else:
                    exp_list = [self.exp_score_global(estimate, item, difficulties, 'dummy_rater',
                                                      dummy_sevs, thresholds)
                                for item in items]
    
                    info_list = [self.variance_global(estimate, item, difficulties, 'dummy_rater',
                                                      dummy_sevs, thresholds)
                                 for item in items]

            else:
                if items is None:
                    exp_list = [self.exp_score_global(estimate, item, difficulties, rater, severities, thresholds)
                                for item in self.dataframe.columns for rater in raters]
    
                    info_list = [self.variance_global(estimate, item, difficulties, rater, severities, thresholds)
                                 for item in self.dataframe.columns for rater in raters]
                    
                else:
                    exp_list = [self.exp_score_global(estimate, item, difficulties, rater, severities, thresholds)
                                for item in items for rater in raters]
    
                    info_list = [self.variance_global(estimate, item, difficulties, rater, severities, thresholds)
                                 for item in items for rater in raters]

            exp_list = np.array(exp_list)
            result = exp_list.sum()

            info_list = np.array(info_list)
            info = info_list.sum()

            change = max(-1, min(1, (result - score) / info))
            estimate -= change
            iters += 1

        if warm_corr:
            if raters is not None:
                sevs = severities[raters]
            else:
                sevs = severities
                
            estimate += self.warm_global(pd.Series({score: estimate}), items, raters, sevs, person_filter)

        if iters >= max_iters:
            print('Maximum iterations reached before convergence.')

        if isinstance(estimate, pd.Series):
            return estimate.iloc[0]

        else:
            return estimate

    def abil_lookup_table_global(self,
                                 anchor=False,
                                 attribute=True,
                                 items=None,
                                 raters=None,
                                 ext_scores=True,
                                 warm_corr=True,
                                 tolerance=0.00001,
                                 max_iters=100,
                                 ext_score_adjustment=0.5):

        if items is None:
            items = self.items

            if raters is None:
                person_filter = np.array([1 for item in self.items])

            else:
                person_filter = np.array([[1 for item in self.items]
                                          for rater in raters])

        elif isinstance(items, str):
            if items == 'all':
                if raters is None:
                    person_filter = np.array([1 for item in self.items])

                else:
                    person_filter = np.array([[1 for item in self.items]
                                              for rater in raters])

            else:
                if raters is None:
                    person_filter = np.array([1])

                else:
                    person_filter = np.array([1 for rater in raters])

        else:
            if raters is None:
                person_filter = np.array([1 for item in items])

            else:
                person_filter = np.array([[1 for item in items]
                                          for rater in raters])

        ext_score = person_filter.sum() * self.max_score

        if ext_scores:
            scores = np.array([score for score in range(ext_score + 1)])

            used_scores = scores.astype(float)
            used_scores[0] += ext_score_adjustment
            used_scores[-1] -= ext_score_adjustment

        else:
            scores = np.array([score + 1 for score in range(ext_score - 1)])
            used_scores = scores.astype(float)

        abil_table = {score: self.score_abil_global(used_score, anchor=anchor, items=items, raters=raters,
                                                    warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                                    ext_score_adjustment=ext_score_adjustment)
                      for score, used_score in zip(scores, used_scores)}

        if attribute:
            self.abil_table_global = pd.Series(abil_table)

        else:
            return pd.Series(abil_table)

    def warm_global(self,
                    abilities,
                    items,
                    raters,
                    severities,
                    person_filter,
                    anchor=False):

        '''
        Warm's (1989) bias correction for ML abiity estimates
        '''

        if isinstance(raters, str):
            if raters == 'all':
                raters = self.raters

        difficulties = self.diffs.loc[items]
        severities = severities.loc[raters]

        c_p_df = {item: abilities - difficulties[item] for item in items}
        c_p_df = {rater: pd.DataFrame(c_p_df) - severities.loc[rater]
                  for rater in raters}
        c_p_df = pd.concat(c_p_df.values(), keys=c_p_df.keys())

        cat_prob_dict = {cat: (cat * c_p_df) - sum(self.thresholds[:cat + 1])
                         for cat in range(self.max_score + 1)}

        for cat in range(self.max_score + 1):
            cat_prob_dict[cat] = np.exp(cat_prob_dict[cat])

        den = sum(cat_prob_dict[cat] for cat in range(self.max_score + 1))

        for cat in range(self.max_score + 1):
            cat_prob_dict[cat] /= den
            cat_prob_dict[cat] *= person_filter

        exp_score_df = sum(cat * df for cat, df in cat_prob_dict.items())
        exp_score_df *= person_filter

        info_df = sum(df * (cat - exp_score_df) ** 2 for cat, df in cat_prob_dict.items())
        info_df *= person_filter

        part_1 = sum((cat ** 3) * cat_prob_dict[cat].sum(axis=1)
                     for cat in range(self.max_score + 1))
        part_1 = sum(part_1.loc[rater] for rater in raters)

        part_2 = 3 * ((info_df + (exp_score_df ** 2)) * exp_score_df).sum(axis=1)
        part_2 = sum(part_2.loc[rater] for rater in raters)

        part_3 = (2 * (exp_score_df ** 3)).sum(axis=1)
        part_3 = sum(part_3.loc[rater] for rater in raters)

        den = 2 * (sum(info_df.loc[rater].sum(axis=1) for rater in raters) ** 2)

        warm_correction = (part_1 - part_2 + part_3) / den

        return warm_correction

    def csem_global(self,
                    persons=None,
                    abilities=None,
                    anchor=False,
                    items=None,
                    raters=None,
                    warm_corr=True,
                    tolerance=0.00001,
                    max_iters=100,
                    ext_score_adjustment=0.5):

        if items is None:
            items = self.dataframe.columns

        if raters is None:
            raters = self.raters

        if anchor:
            if hasattr(self, 'anchor_diffs_global'):
                difficulties = self.anchor_diffs_global
                thresholds = self.anchor_thresholds_global
                severities = self.anchor_severities_global

            else:
                print('Anchor calibration required')
                return

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_global

        difficulties = difficulties.loc[items]
        severities = severities[raters]

        if persons is not None:
            if anchor:
                abilities = self.anchor_abils_global.loc[persons]
            else:
                abilities = self.abils_global.loc[persons]

            person_data = self.dataframe.loc[persons, items]
            person_filter = person_data.notna().astype(float).replace(0, np.nan)

        if abilities is not None:
            abilities = {f'Ability_{abil}': abil for abil in abilities}
            abilities = pd.Series(abilities)
            person_filter = pd.DataFrame(1, index=abilities.index, columns=items)

        c_p_df = {item: abilities - difficulties[item] for item in items}
        c_p_df = {rater: pd.DataFrame(c_p_df) - severities.loc[rater]
                  for rater in raters}
        c_p_df = pd.concat(c_p_df.values(), keys=c_p_df.keys())

        cat_prob_dict = {cat: (cat * c_p_df) - sum(self.thresholds[:cat + 1])
                         for cat in range(self.max_score + 1)}

        for cat in range(self.max_score + 1):
            cat_prob_dict[cat] = np.exp(cat_prob_dict[cat])

        den = sum(cat_prob_dict[cat] for cat in range(self.max_score + 1))

        for cat in range(self.max_score + 1):
            cat_prob_dict[cat] /= den
            cat_prob_dict[cat] *= person_filter

        exp_score_df = sum(cat * df for cat, df in cat_prob_dict.items())
        exp_score_df *= person_filter

        info_df = sum(df * (cat - exp_score_df) ** 2 for cat, df in cat_prob_dict.items())
        info_df *= person_filter

        cond_sems = 1 / (info_df.sum(axis=1) ** 0.5)

        return cond_sems

    def abil_items(self,
                   persons,
                   anchor=False,
                   items=None,
                   raters=None,
                   warm_corr=True,
                   tolerance=0.00001,
                   max_iters=100,
                   ext_score_adjustment=0.5):

        '''
        Creates a raw score to ability estimate look-up table for a set
        of items using ML estimation (Newton-Raphson procedure) with
        optional Warm (1989) bias correction.
        '''

        if isinstance(persons, str):
            if persons == 'all':
                persons = self.persons

            else:
                persons = [persons]

        if persons is None:
            persons = self.persons

        if isinstance(items, str):
            if items == 'all':
                items = self.items.tolist()

            else:
                items = [items]
                
        if items is None:
            items = self.items
         
        if raters is None:
            raters = self.raters
         
        if isinstance(raters, str):
            if raters == 'all':
                raters = self.raters.tolist()
                
        if isinstance(raters, pd.core.indexes.base.Index):
            raters = raters.tolist()

        if anchor:
            if hasattr(self, 'anchor_diffs_items'):
                difficulties = self.anchor_diffs_items.loc[items]
                thresholds = self.anchor_thresholds_items
                severities = {rater: self.anchor_severities_items[rater]
                              for rater in raters}

            else:
                print('Anchor calibration required')
                return

        else:
            difficulties = self.diffs.loc[items]
            thresholds = self.thresholds
            severities = {rater: self.severities_items[rater]
                          for rater in raters}

        person_data = self.dataframe.loc[(raters, persons), items]
        person_filter = person_data.notna().astype(float).replace(0, np.nan)

        scores = {rater: person_data.loc[rater].sum(axis=1).astype(float)
                  for rater in raters}
        scores = sum(scores.values())

        ext_scores = {rater: person_filter.loc[rater].sum(axis=1) * self.max_score
                      for rater in raters}
        ext_scores = sum(ext_scores.values())

        scores[scores == 0] = ext_score_adjustment
        scores[scores == ext_scores] -= ext_score_adjustment

        diff_df = pd.concat([difficulties for person in persons], axis=1).T
        diff_df.index = persons

        mean_diffs = {rater: diff_df * person_filter.loc[rater]
                      for rater in raters}
        mean_diffs = sum(mean_diffs[rater].sum(axis=1)
                         for rater in raters)

        item_count = {rater: person_filter.loc[rater]
                      for rater in raters}
        item_count = sum(item_count[rater].sum(axis=1)
                         for rater in raters)

        mean_diffs /= item_count

        try:
            estimates = np.log(scores) - np.log(ext_scores - scores) + mean_diffs
            changes = pd.Series({person: 1 for person in persons})
            iters = 0

            while (abs(changes).max() > tolerance) & (iters <= max_iters):

                c_p_df = {rater: {item: estimates - difficulties[item] - severities[rater][item]
                                  for item in items}
                          for rater in raters}

                for rater in self.raters:
                    c_p_df[rater] = pd.DataFrame(c_p_df[rater])

                c_p_df = pd.concat(c_p_df.values(), keys=c_p_df.keys())

                cat_prob_dict = {cat: (cat * c_p_df) - sum(thresholds[:cat + 1])
                                 for cat in range(self.max_score + 1)}

                for cat in range(self.max_score + 1):
                    cat_prob_dict[cat] = np.exp(cat_prob_dict[cat])

                den = sum(cat_prob_dict[cat] for cat in range(self.max_score + 1))

                for cat in range(self.max_score + 1):
                    cat_prob_dict[cat] /= den
                    cat_prob_dict[cat] *= person_filter

                exp_score_df = sum(cat * df for cat, df in cat_prob_dict.items())
                exp_score_df *= person_filter

                info_df = sum(df * (cat - exp_score_df) ** 2 for cat, df in cat_prob_dict.items())
                info_df *= person_filter

                result_list = sum(exp_score_df.loc[rater].sum(axis=1) for rater in raters)
                info_list = sum(info_df.loc[rater].sum(axis=1) for rater in raters)

                changes = (result_list - scores) / info_list
                changes = changes.clip(-1, 1)
                estimates -= changes
                iters += 1

            # Per-person convergence check: set non-converged persons to NaN
            # Prevents runaway +/-1 logit drift over max_iters for extreme scorers
            if iters >= max_iters:
                not_converged = abs(changes) > tolerance
                if not_converged.any():
                    print(f'Warning: {int(not_converged.sum())} person(s) did not converge '
                          f'in abil_items() and will be set to NaN.')
                    estimates[not_converged] = np.nan

            if warm_corr:
                valid = estimates.notna()
                if valid.any():
                    # Apply Warm correction only to converged persons
                    valid_idx = estimates.index[valid]
                    if isinstance(person_filter.index, pd.MultiIndex):
                        valid_pf = person_filter.loc[(slice(None), valid_idx), :]
                    else:
                        valid_pf = person_filter.loc[valid_idx]
                    estimates[valid] += self.warm_items(
                        estimates[valid], items, raters, severities, valid_pf
                    )

        except Exception as e:
            print(f'abil_items() failed: {e}')
            estimates = pd.Series(np.nan, index=list(persons))

        return estimates

    def person_abils_items(self,
                           anchor=False,
                           items=None,
                           raters=None,
                           warm_corr=True,
                           tolerance=0.00001,
                           max_iters=100,
                           ext_score_adjustment=0.5):

        '''
        Creates raw score to ability estimate look-up table. Newton-Raphson ML
        estimation, includes optional Warm (1989) bias correction.
        '''

        '''
        Creates raw score to ability estimate look-up table. Newton-Raphson ML
        estimation, includes optional Warm (1989) bias correction.
        '''

        estimates = self.abil_items(persons=None, anchor=anchor, items=items, raters=raters,
                                    warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                    ext_score_adjustment=ext_score_adjustment)

        if anchor:
            self.anchor_abils_items = estimates

        else:
            self.abils_items = estimates

    def score_abil_items(self,
                         score,
                         anchor=False,
                         items=None,
                         raters=None,
                         warm_corr=True,
                         tolerance=0.00001,
                         max_iters=100,
                         ext_score_adjustment=0.5):
        
        if isinstance(items, str):
           if items == 'all':
               items = self.items
           else:
               items = [items]

        if items is None:
            items = self.items

        if isinstance(raters, str):
            if raters == 'all':
                raters = [rater for rater in self.raters]
            else:
                raters = [raters]
 
        if anchor:
            if hasattr(self, 'anchor_diffs_items'):
                difficulties = self.anchor_diffs_items.loc[items]
                thresholds = self.anchor_thresholds_items

                if raters is None:
                    severities = pd.Series({'dummy_rater': {item: 0 for item in self.dataframe.columns}})

                else:
                    severities = {rater: self.anchor_severities_items[rater] for rater in raters}
 
            else:
                print('Anchor calibration required')
                return
 
        else:
            difficulties = self.diffs.loc[items]
            thresholds = self.thresholds

            if raters is None:
                severities = pd.Series({'dummy_rater': {item: 0 for item in self.dataframe.columns}})

            else:
                severities = {rater: self.severities_items[rater] for rater in raters}

        if raters is None:
            if items is None:
                person_filter = np.array([1 for item in self.dataframe.columns])

            else:
                person_filter = np.array([1 for item in items])

        else:
            if items is None:
                person_filter = np.array([[1 for item in self.dataframe.columns]
                                          for rater in raters])

            else:
                person_filter = np.array([[1 for item in items]
                                          for rater in raters])

        if raters is None:
            raters = ['dummy_rater']

        ext_score = person_filter.sum() * self.max_score

        if score == 0:
            score = ext_score_adjustment

        elif score == ext_score:
            score -= ext_score_adjustment

        estimate = log(score) - log(ext_score - score) + difficulties.mean()
        change = 1
        iters = 0

        while (abs(change) > tolerance) & (iters <= max_iters):

            if raters is None:
                if items is None:
                    exp_list = [self.exp_score_items(estimate, item, difficulties, 'dummy_rater',
                                                      severities, thresholds)
                                for item in self.dataframe.columns]

                    info_list = [self.variance_items(estimate, item, difficulties, 'dummy_rater',
                                                      severities, thresholds)
                                 for item in self.dataframe.columns]

                else:
                    exp_list = [self.exp_score_items(estimate, item, difficulties, 'dummy_rater',
                                                      severities, thresholds)
                                for item in items]

                    info_list = [self.variance_items(estimate, item, difficulties, 'dummy_rater',
                                                      severities, thresholds)
                                 for item in items]

            else:
                if items is None:
                    exp_list = [self.exp_score_items(estimate, item, difficulties, rater, severities, thresholds)
                                for item in self.dataframe.columns for rater in raters]

                    info_list = [self.variance_items(estimate, item, difficulties, rater, severities, thresholds)
                                 for item in self.dataframe.columns for rater in raters]

                else:
                    exp_list = [self.exp_score_items(estimate, item, difficulties, rater, severities, thresholds)
                                for item in items for rater in raters]

                    info_list = [self.variance_items(estimate, item, difficulties, rater, severities, thresholds)
                                 for item in items for rater in raters]

            exp_list = np.array(exp_list)
            result = exp_list.sum()

            info_list = np.array(info_list)
            info = info_list.sum()

            change = max(-1, min(1, (result - score) / info))
            estimate -= change
            iters += 1

        if warm_corr:
            estimate += self.warm_items(pd.Series({score: estimate}), items, raters, severities, person_filter)

        if iters >= max_iters:
            print('Maximum iterations reached before convergence.')

        if isinstance(estimate, pd.Series):
            return estimate.iloc[0]

        else:
            return estimate

    def abil_lookup_table_items(self,
                                anchor=False,
                                attribute=True,
                                items=None,
                                raters=None,
                                ext_scores=True,
                                warm_corr=True,
                                tolerance=0.00001,
                                max_iters=100,
                                ext_score_adjustment=0.5):

        if items is None:
            items = self.items

        if items is None:
            if raters is None:
                person_filter = np.array([1 for item in self.items])

            else:
                person_filter = np.array([[1 for item in self.items]
                                          for rater in raters])

        elif isinstance(items, str):
            if items == 'all':
                if raters is None:
                    person_filter = np.array([1 for item in self.dataframe.columns])

                else:
                    person_filter = np.array([[1 for item in self.dataframe.columns]
                                              for rater in raters])

            else:
                if raters is None:
                    person_filter = np.array([1])

                else:
                    person_filter = np.array([1 for rater in raters])

        else:
            if raters is None:
                person_filter = np.array([1 for item in items])

            else:
                person_filter = np.array([[1 for item in items]
                                          for rater in raters])

        ext_score = person_filter.sum() * self.max_score

        if ext_scores:
            scores = np.array([score for score in range(ext_score + 1)])

            used_scores = scores.astype(float)
            used_scores[0] += ext_score_adjustment
            used_scores[-1] -= ext_score_adjustment

        else:
            scores = np.array([score + 1 for score in range(ext_score - 1)])
            used_scores = scores.astype(float)

        abil_table = {score: self.score_abil_items(used_score, anchor=anchor, items=items, raters=raters,
                                                   warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                                   ext_score_adjustment=ext_score_adjustment)
                      for score, used_score in zip(scores, used_scores)}

        if attribute:
            self.abil_table_items = pd.Series(abil_table)

        else:
            return pd.Series(abil_table)

    def warm_items(self,
                   abilities,
                   items,
                   raters,
                   severities,
                   person_filter,
                   anchor=False):

        '''
        Warm's (1989) bias correction for ML abiity estimates
        '''

        if isinstance(raters, str):
            if raters == 'all':
                raters = self.raters

        if anchor:
            difficulties = self.anchor_diffs_items.loc[items]
            thresholds = self.anchor_thresholds_items

        else:
            difficulties = self.diffs.loc[items]
            thresholds = self.thresholds

        severities = {rater: severities[rater] for rater in raters}

        if isinstance(abilities, (float, np.float64)):
            abilities = pd.Series({'dummy': abilities})

        c_p_df = {rater: {item: abilities - difficulties.loc[item] - severities[rater][item]
                          for item in items}
                  for rater in raters}
        c_p_df = {rater: pd.DataFrame(c_p_df[rater]) for rater in raters}
        c_p_df = pd.concat(c_p_df.values(), keys=c_p_df.keys())

        cat_prob_dict = {cat: (cat * c_p_df) - sum(thresholds[:cat + 1])
                         for cat in range(self.max_score + 1)}

        for cat in range(self.max_score + 1):
            cat_prob_dict[cat] = np.exp(cat_prob_dict[cat])

        den = sum(cat_prob_dict[cat] for cat in range(self.max_score + 1))

        for cat in range(self.max_score + 1):
            cat_prob_dict[cat] /= den
            cat_prob_dict[cat] *= person_filter

        exp_score_df = sum(cat * df for cat, df in cat_prob_dict.items())
        exp_score_df *= person_filter

        info_df = sum(df * (cat - exp_score_df) ** 2 for cat, df in cat_prob_dict.items())
        info_df *= person_filter

        part_1 = sum((cat ** 3) * cat_prob_dict[cat].sum(axis=1)
                     for cat in range(self.max_score + 1))
        part_1 = sum(part_1.loc[rater] for rater in raters)

        part_2 = 3 * ((info_df + (exp_score_df ** 2)) * exp_score_df).sum(axis=1)
        part_2 = sum(part_2.loc[rater] for rater in raters)

        part_3 = (2 * (exp_score_df ** 3)).sum(axis=1)
        part_3 = sum(part_3.loc[rater] for rater in raters)

        den = 2 * (sum(info_df.loc[rater].sum(axis=1) for rater in raters) ** 2)

        warm_correction = (part_1 - part_2 + part_3) / den

        return warm_correction

    def csem_items(self,
                   persons=None,
                   abilities=None,
                   anchor=False,
                   items=None,
                   raters=None,
                   warm_corr=True,
                   tolerance=0.00001,
                   max_iters=100,
                   ext_score_adjustment=0.5):

        if items is None:
            items = self.dataframe.columns

        if raters is None:
            raters = self.raters

        if anchor:
            if hasattr(self, 'anchor_diffs_items'):
                difficulties = self.anchor_diffs_items
                thresholds = self.anchor_thresholds_items
                severities = self.anchor_severities_items

            else:
                print('Anchor calibration required')
                return

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_items

        difficulties = difficulties.loc[items]
        severities = severities[raters]

        if persons is not None:
            if anchor:
                abilities = self.anchor_abils_items.loc[persons]
            else:
                abilities = self.abils_items.loc[persons]

            person_data = self.dataframe.loc[persons, items]
            person_filter = person_data.notna().astype(float).replace(0, np.nan)

        if abilities is not None:
            abilities = {f'Ability_{abil}': abil for abil in abilities}
            abilities = pd.Series(abilities)
            person_filter = pd.DataFrame(1, index=abilities.index, columns=items)

        c_p_df = {rater: {item: abilities - difficulties[item] - severities[rater][item]
                          for item in self.items}
                  for rater in self.raters}
        for rater in self.raters:
            c_p_df[rater] = pd.DataFrame(c_p_df[rater])

        c_p_df = pd.concat(c_p_df.values(), keys=c_p_df.keys())

        cat_prob_dict = {cat: (cat * c_p_df) - sum(self.thresholds[:cat + 1])
                         for cat in range(self.max_score + 1)}

        for cat in range(self.max_score + 1):
            cat_prob_dict[cat] = np.exp(cat_prob_dict[cat])

        den = sum(cat_prob_dict[cat] for cat in range(self.max_score + 1))

        for cat in range(self.max_score + 1):
            cat_prob_dict[cat] /= den
            cat_prob_dict[cat] *= person_filter

        exp_score_df = sum(cat * df for cat, df in cat_prob_dict.items())
        exp_score_df *= person_filter

        info_df = sum(df * (cat - exp_score_df) ** 2 for cat, df in cat_prob_dict.items())
        info_df *= person_filter

        cond_sems = 1 / (info_df.sum(axis=1) ** 0.5)

        return cond_sems

    def abil_thresholds(self,
                        persons,
                        anchor=False,
                        items=None,
                        raters=None,
                        warm_corr=True,
                        tolerance=0.00001,
                        max_iters=100,
                        ext_score_adjustment=0.5):

        '''
        Creates a raw score to ability estimate look-up table for a set
        of items using ML estimation (Newton-Raphson procedure) with
        optional Warm (1989) bias correction.
        '''

        if isinstance(persons, str):
            if persons == 'all':
                persons = self.persons

            else:
                persons = [persons]

        if persons is None:
            persons = self.persons

        if isinstance(items, str):
            if items == 'all':
                items = self.items.tolist()

            else:
                items = [items]

        if items is None:
            items = self.items

        if raters is None:
            raters = self.raters

        if isinstance(raters, str):
            if raters == 'all':
                raters = self.raters

        if isinstance(raters, pd.core.indexes.base.Index):
            raters = raters.tolist()

        if anchor:
            if hasattr(self, 'anchor_diffs_thresholds'):
                difficulties = self.anchor_diffs_thresholds.loc[items]
                thresholds = self.anchor_thresholds_thresholds
                severities = {rater: self.anchor_severities_thresholds[rater]
                              for rater in raters}

            else:
                print('Anchor calibration required')
                return

        else:
            difficulties = self.diffs.loc[items]
            thresholds = self.thresholds
            severities = {rater: self.severities_thresholds[rater]
                          for rater in raters}

        person_data = self.dataframe.loc[(raters, persons), items]
        person_filter = person_data.notna().astype(float).replace(0, np.nan)

        scores = {rater: person_data.loc[rater].sum(axis=1).astype(float)
                  for rater in raters}
        scores = sum(scores.values())

        ext_scores = {rater: person_filter.loc[rater].sum(axis=1) * self.max_score
                      for rater in raters}
        ext_scores = sum(ext_scores.values())

        scores[scores == 0] = ext_score_adjustment
        scores[scores == ext_scores] -= ext_score_adjustment

        diff_df = pd.concat([difficulties for person in persons], axis=1).T
        diff_df.index = persons

        mean_diffs = {rater: diff_df * person_filter.loc[rater]
                      for rater in raters}
        mean_diffs = sum(mean_diffs[rater].sum(axis=1)
                         for rater in raters)

        item_count = {rater: person_filter.loc[rater]
                      for rater in raters}
        item_count = sum(item_count[rater].sum(axis=1)
                         for rater in raters)

        mean_diffs /= item_count

        try:
            estimates = np.log(scores) - np.log(ext_scores - scores) + mean_diffs
            changes = pd.Series({person: 1 for person in persons})
            iters = 0

            while (abs(changes).max() > tolerance) & (iters <= max_iters):

                c_p_df = {item: estimates - difficulties.loc[item]
                          for item in items}
                c_p_df = pd.DataFrame(c_p_df)

                cat_probs = {cat: {rater: (cat * c_p_df - sum(thresholds[:cat + 1]) - sum(severities[rater][:cat + 1]))
                                   for rater in raters}
                             for cat in range(self.max_score + 1)}

                for cat in range(self.max_score + 1):
                    cat_probs[cat] = pd.concat(cat_probs[cat].values(), keys=cat_probs[cat].keys())
                    cat_probs[cat] = np.exp(cat_probs[cat])

                den = sum(cat_probs[cat] for cat in range(self.max_score + 1))

                for cat in range(self.max_score + 1):
                    cat_probs[cat] /= den
                    cat_probs[cat] *= person_filter

                exp_score_df = sum(cat * df for cat, df in cat_probs.items())
                exp_score_df *= person_filter

                info_df = sum(df * (cat - exp_score_df) ** 2 for cat, df in cat_probs.items())
                info_df *= person_filter

                result_list = sum(exp_score_df.loc[rater].sum(axis=1) for rater in raters)
                info_list = sum(info_df.loc[rater].sum(axis=1) for rater in raters)

                changes = (result_list - scores) / info_list
                changes = changes.clip(-1, 1)
                estimates -= changes
                iters += 1

            # Per-person convergence check
            if iters >= max_iters:
                not_converged = abs(changes) > tolerance
                if not_converged.any():
                    print(f'Warning: {int(not_converged.sum())} person(s) did not converge '
                          f'in abil_thresholds() and will be set to NaN.')
                    estimates[not_converged] = np.nan

            if warm_corr:
                valid = estimates.notna()
                if valid.any():
                    valid_idx = estimates.index[valid]
                    if isinstance(person_filter.index, pd.MultiIndex):
                        valid_pf = person_filter.loc[(slice(None), valid_idx), :]
                    else:
                        valid_pf = person_filter.loc[valid_idx]
                    estimates[valid] += self.warm_thresholds(
                        estimates[valid], items, raters, severities, valid_pf, anchor=anchor
                    )

        except Exception as e:
            print(f'abil_thresholds() failed: {e}')
            estimates = pd.Series(np.nan, index=list(persons))

        return estimates

    def person_abils_thresholds(self,
                                anchor=False,
                                items=None,
                                raters=None,
                                warm_corr=True,
                                tolerance=0.00001,
                                max_iters=100,
                                ext_score_adjustment=0.5):

        '''
        Creates raw score to ability estimate look-up table. Newton-Raphson ML
        estimation, includes optional Warm (1989) bias correction.
        '''

        estimates = self.abil_thresholds(persons=None, anchor=anchor, items=items, raters=raters,
                                         warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                         ext_score_adjustment=ext_score_adjustment)

        if anchor:
            self.anchor_abils_thresholds = estimates

        else:
            self.abils_thresholds = estimates

    def score_abil_thresholds(self,
                              score,
                              anchor=False,
                              items=None,
                              raters=None,
                              warm_corr=True,
                              tolerance=0.00001,
                              max_iters=100,
                              ext_score_adjustment=0.5):

        if isinstance(items, str):
            if items == 'all':
                items = self.items
            else:
                items = [items]

        if items is None:
            items = self.items

        if isinstance(raters, str):
            if raters == 'all':
                raters = [rater for rater in self.raters]
            else:
                raters = [raters]

        if anchor:
            if hasattr(self, 'anchor_diffs_matrix'):
                difficulties = self.anchor_diffs_thresholds.loc[items]
                thresholds = self.anchor_thresholds_thresholds

                if raters is None:
                    severities = {'dummy_rater': [0 for threshold in range(self.max_score + 1)]}

                else:
                    severities = {rater: self.anchor_severities_thresholds[rater] for rater in raters}

            else:
                print('Anchor calibration required')
                return

        else:
            difficulties = self.diffs.loc[items]
            thresholds = self.thresholds

            if raters is None:
                severities = {'dummy_rater': [0 for threshold in range(self.max_score + 1)]}

            else:
                severities = {rater: self.severities_thresholds[rater] for rater in raters}

        if raters is None:
            if items is None:
                person_filter = np.array([1 for item in self.dataframe.columns])

            else:
                person_filter = np.array([1 for item in items])

        else:
            if items is None:
                person_filter = np.array([[1 for item in self.dataframe.columns] for rater in raters])

            else:
                person_filter = np.array([[1 for item in items] for rater in raters])

        if raters is None:
            raters = ['dummy_rater']

        ext_score = person_filter.sum() * self.max_score

        if score == 0:
            score = ext_score_adjustment

        elif score == ext_score:
            score -= ext_score_adjustment

        estimate = log(score) - log(ext_score - score) + difficulties.mean()
        change = 1
        iters = 0

        while (abs(change) > tolerance) & (iters <= max_iters):

            if raters is None:
                if items is None:
                    exp_list = [self.exp_score_thresholds(estimate, item, difficulties, raters,
                                                          severities, thresholds)
                                for item in self.dataframe.columns]

                    info_list = [self.variance_thresholds(estimate, item, difficulties, raters,
                                                          severities, thresholds)
                                 for item in self.dataframe.columns]

                else:
                    exp_list = [self.exp_score_thresholds(estimate, item, difficulties, raters,
                                                          severities, thresholds)
                                for item in items]

                    info_list = [self.variance_thresholds(estimate, item, difficulties, raters,
                                                          severities, thresholds)
                                 for item in items]

            else:
                if items is None:
                    exp_list = [self.exp_score_thresholds(estimate, item, difficulties, rater, severities, thresholds)
                                for item in self.dataframe.columns for rater in raters]

                    info_list = [self.variance_thresholds(estimate, item, difficulties, rater, severities, thresholds)
                                 for item in self.dataframe.columns for rater in raters]

                else:
                    exp_list = [self.exp_score_thresholds(estimate, item, difficulties, rater, severities, thresholds)
                                for item in items for rater in raters]

                    info_list = [self.variance_thresholds(estimate, item, difficulties, rater, severities, thresholds)
                                 for item in items for rater in raters]

            exp_list = np.array(exp_list)
            result = exp_list.sum()

            info_list = np.array(info_list)
            info = info_list.sum()

            change = max(-1, min(1, (result - score) / info))
            estimate -= change
            iters += 1

        if warm_corr:
            severities = dict((rater, severities[rater]) for rater in raters)
            estimate += self.warm_thresholds(pd.Series({score: estimate}), items, raters, severities,
                                             person_filter, anchor=anchor)

        if iters >= max_iters:
            print('Maximum iterations reached before convergence.')

        if isinstance(estimate, pd.Series):
            return estimate.iloc[0]

        else:
            return estimate

    def abil_lookup_table_thresholds(self,
                                     anchor=False,
                                     items=None,
                                     raters=None,
                                     ext_scores=True,
                                     warm_corr=True,
                                     tolerance=0.00001,
                                     max_iters=100,
                                     ext_score_adjustment=0.5):

        if items is None:
            items = self.items

            if raters is None:
                person_filter = np.array([1 for item in self.dataframe.columns])

            else:
                person_filter = np.array([[1 for item in self.dataframe.columns]
                                          for rater in raters])

        elif isinstance(items, str):
            if items == 'all':
                if raters is None:
                    person_filter = np.array([1 for item in self.dataframe.columns])

                else:
                    person_filter = np.array([[1 for item in self.dataframe.columns]
                                              for rater in raters])

            else:
                if raters is None:
                    person_filter = np.array([1])

                else:
                    person_filter = np.array([1 for rater in raters])

        else:
            if raters is None:
                person_filter = np.array([1 for item in items])

            else:
                person_filter = np.array([[1 for item in items]
                                          for rater in raters])

        ext_score = person_filter.sum() * self.max_score

        if ext_scores:
            scores = np.array([score for score in range(ext_score + 1)])

            used_scores = scores.astype(float)
            used_scores[0] += ext_score_adjustment
            used_scores[-1] -= ext_score_adjustment

        else:
            scores = np.array([score + 1 for score in range(ext_score - 1)])
            used_scores = scores.astype(float)

        abil_table = {score: self.score_abil_thresholds(used_score, anchor=anchor, items=items, raters=raters,
                                                        warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                                        ext_score_adjustment=ext_score_adjustment)
                      for score, used_score in zip(scores, used_scores)}

        self.abil_table_thresholds = pd.Series(abil_table)

    def warm_thresholds(self,
                        abilities,
                        items,
                        raters,
                        severities,
                        person_filter,
                        anchor=False):

        '''
        Warm's (1989) bias correction for ML abiity estimates
        '''

        if isinstance(raters, str):
            if raters == 'all':
                raters = self.raters

            else:
                raters = [raters]

        if anchor:
            difficulties = self.anchor_diffs_thresholds.loc[items]
            thresholds = self.anchor_thresholds_thresholds

        else:
            difficulties = self.diffs.loc[items]
            thresholds = self.thresholds

        severities = {rater: severities[rater] for rater in raters}

        if isinstance(abilities, (float, np.float64)):
            abilities = pd.Series({'dummy': abilities})

        c_p_df = {item: abilities - difficulties.loc[item]
                  for item in items}

        c_p_df = pd.DataFrame(c_p_df)

        cat_probs = {cat: {rater: (cat * c_p_df - sum(thresholds[:cat + 1]) - sum(severities[rater][:cat + 1]))
                           for rater in raters}
                     for cat in range(self.max_score + 1)}

        for cat in range(self.max_score + 1):
            cat_probs[cat] = pd.concat(cat_probs[cat].values(), keys=cat_probs[cat].keys())
            cat_probs[cat] = np.exp(cat_probs[cat])

        den = sum(cat_probs[cat] for cat in range(self.max_score + 1))

        for cat in range(self.max_score + 1):
            cat_probs[cat] /= den
            cat_probs[cat] *= person_filter

        exp_score_df = sum(cat * df for cat, df in cat_probs.items())
        exp_score_df *= person_filter

        info_df = sum(df * (cat - exp_score_df) ** 2 for cat, df in cat_probs.items())
        info_df *= person_filter

        part_1 = sum((cat ** 3) * cat_probs[cat].sum(axis=1)
                     for cat in range(self.max_score + 1))
        part_1 = sum(part_1.loc[rater] for rater in raters)

        part_2 = 3 * ((info_df + (exp_score_df ** 2)) * exp_score_df).sum(axis=1)
        part_2 = sum(part_2.loc[rater] for rater in raters)

        part_3 = (2 * (exp_score_df ** 3)).sum(axis=1)
        part_3 = sum(part_3.loc[rater] for rater in raters)

        den = 2 * (sum(info_df.loc[rater].sum(axis=1) for rater in raters) ** 2)

        warm_correction = (part_1 - part_2 + part_3) / den

        return warm_correction

    def csem_thresholds(self,
                        persons=None,
                        abilities=None,
                        anchor=False,
                        items=None,
                        raters=None,
                        warm_corr=True,
                        tolerance=0.00001,
                        max_iters=100,
                        ext_score_adjustment=0.5):

        if items is None:
            items = self.dataframe.columns

        if raters is None:
            raters = self.raters

        if anchor:
            if hasattr(self, 'anchor_diffs_thresholds'):
                difficulties = self.anchor_diffs_thresholds
                thresholds = self.anchor_thresholds_thresholds
                severities = self.anchor_severities_thresholds

            else:
                print('Anchor calibration required')
                return

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_thresholds

        difficulties = difficulties.loc[items]
        severities = {rater: severities[rater] for rater in raters}

        if persons is not None:
            if anchor:
                abilities = self.anchor_abils_thresholds.loc[persons]
            else:
                abilities = self.abils_thresholds.loc[persons]

            person_data = self.dataframe.loc[persons, items]
            person_filter = person_data.notna().astype(float).replace(0, np.nan)

        if abilities is not None:
            abilities = {f'Ability_{abil}': abil for abil in abilities}
            abilities = pd.Series(abilities)
            person_filter = pd.DataFrame(1, index=abilities.index, columns=items)

        c_p_df = {item: estimates - difficulties.loc[item]
                  for item in items}
        c_p_df = pd.DataFrame(c_p_df)

        cat_probs = {cat: {rater: (cat * c_p_df - sum(thresholds[:cat + 1]) - sum(severities[rater][:cat + 1]))
                           for rater in raters}
                     for cat in range(self.max_score + 1)}

        for cat in range(self.max_score + 1):
            cat_probs[cat] = pd.concat(cat_probs[cat].values(), keys=cat_probs[cat].keys())
            cat_probs[cat] = np.exp(cat_probs[cat])

        den = sum(cat_probs[cat] for cat in range(self.max_score + 1))

        for cat in range(self.max_score + 1):
            cat_probs[cat] /= den
            cat_probs[cat] *= person_filter

        exp_score_df = sum(cat * df for cat, df in cat_probs.items())
        exp_score_df *= person_filter

        info_df = sum(df * (cat - exp_score_df) ** 2 for cat, df in cat_probs.items())
        info_df *= person_filter

        cond_sems = 1 / (info_df.sum(axis=1) ** 0.5)

        return cond_sems

    def abil_matrix(self,
                    persons,
                    anchor=False,
                    items=None,
                    raters=None,
                    warm_corr=True,
                    tolerance=0.00001,
                    max_iters=100,
                    ext_score_adjustment=0.5):

        '''
        Creates a raw score to ability estimate look-up table for a set
        of items using ML estimation (Newton-Raphson procedure) with
        optional Warm (1989) bias correction.
        '''

        if isinstance(persons, str):
            if persons == 'all':
                persons = self.persons

            else:
                persons = [persons]

        if persons is None:
            persons = self.persons

        if isinstance(items, str):
            if items == 'all':
                items = self.items.tolist()

            else:
                items = [items]

        if items is None:
            items = self.items

        if raters is None:
            raters = self.raters

        if isinstance(raters, str):
            if raters == 'all':
                raters = self.raters

        if isinstance(raters, pd.core.indexes.base.Index):
            raters = raters.tolist()

        if anchor:
            if hasattr(self, 'anchor_diffs_global'):
                difficulties = self.anchor_diffs_matrix.loc[items]
                thresholds = self.anchor_thresholds_matrix
                severities = {rater: self.anchor_severities_matrix[rater]
                              for rater in raters}

            else:
                print('Anchor calibration required')
                return

        else:
            difficulties = self.diffs.loc[items]
            thresholds = self.thresholds
            severities = {rater: self.severities_matrix[rater]
                          for rater in raters}

        person_data = self.dataframe.loc[(raters, persons), items]
        person_filter = person_data.notna().astype(float).replace(0, np.nan)

        scores = {rater: person_data.loc[rater].sum(axis=1).astype(float)
                  for rater in raters}
        scores = sum(scores.values())

        ext_scores = {rater: person_filter.loc[rater].sum(axis=1) * self.max_score
                      for rater in raters}
        ext_scores = sum(ext_scores.values())

        scores[scores == 0] = ext_score_adjustment
        scores[scores == ext_scores] -= ext_score_adjustment

        diff_df = pd.concat([difficulties for person in persons], axis=1).T
        diff_df.index = persons

        mean_diffs = {rater: diff_df * person_filter.loc[rater]
                      for rater in raters}
        mean_diffs = sum(mean_diffs[rater].sum(axis=1)
                         for rater in raters)

        item_count = {rater: person_filter.loc[rater]
                      for rater in raters}
        item_count = sum(item_count[rater].sum(axis=1)
                         for rater in raters)

        mean_diffs /= item_count

        try:
            estimates = np.log(scores) - np.log(ext_scores - scores) + mean_diffs
            changes = pd.Series({person: 1 for person in persons})
            iters = 0

            while (abs(changes).max() > tolerance) & (iters <= max_iters):

                c_p_df = {item: estimates - difficulties.loc[item]
                          for item in items}
                c_p_df = pd.DataFrame(c_p_df)

                cat_probs = {cat: {rater: (cat * c_p_df - sum(thresholds[:cat + 1]))
                                   for rater in raters}
                             for cat in range(self.max_score + 1)}

                for cat in range(self.max_score + 1):
                    for rater in raters:
                        for item in items:
                            cat_probs[cat][rater][item] -= sum(severities[rater][item][:cat + 1])

                for cat in range(self.max_score + 1):
                    cat_probs[cat] = pd.concat(cat_probs[cat].values(), keys=cat_probs[cat].keys())
                    cat_probs[cat] = np.exp(cat_probs[cat])

                den = sum(cat_probs[cat] for cat in range(self.max_score + 1))

                for cat in range(self.max_score + 1):
                    cat_probs[cat] /= den
                    cat_probs[cat] *= person_filter

                exp_score_df = sum(cat * df for cat, df in cat_probs.items())
                exp_score_df *= person_filter

                info_df = sum(df * (cat - exp_score_df) ** 2 for cat, df in cat_probs.items())
                info_df *= person_filter

                result_list = sum(exp_score_df.loc[rater].sum(axis=1) for rater in raters)
                info_list = sum(info_df.loc[rater].sum(axis=1) for rater in raters)

                changes = (result_list - scores) / info_list
                changes = changes.clip(-1, 1)
                estimates -= changes
                iters += 1

            # Per-person convergence check
            if iters >= max_iters:
                not_converged = abs(changes) > tolerance
                if not_converged.any():
                    print(f'Warning: {int(not_converged.sum())} person(s) did not converge '
                          f'in abil_matrix() and will be set to NaN.')
                    estimates[not_converged] = np.nan

            if warm_corr:
                valid = estimates.notna()
                if valid.any():
                    valid_idx = estimates.index[valid]
                    if isinstance(person_filter.index, pd.MultiIndex):
                        valid_pf = person_filter.loc[(slice(None), valid_idx), :]
                    else:
                        valid_pf = person_filter.loc[valid_idx]
                    estimates[valid] += self.warm_matrix(
                        estimates[valid], items, raters, severities, valid_pf, anchor=anchor
                    )

        except Exception as e:
            print(f'abil_matrix() failed: {e}')
            estimates = pd.Series(np.nan, index=list(persons))

        return estimates

    def person_abils_matrix(self,
                            anchor=False,
                            items=None,
                            raters=None,
                            warm_corr=True,
                            tolerance=0.00001,
                            max_iters=100,
                            ext_score_adjustment=0.5):

        '''
        Creates raw score to ability estimate look-up table. Newton-Raphson ML
        estimation, includes optional Warm (1989) bias correction.
        '''

        estimates = self.abil_matrix(persons=None, anchor=anchor, items=items, raters=raters,
                                     warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                     ext_score_adjustment=ext_score_adjustment)

        if anchor:
            self.anchor_abils_matrix = estimates

        else:
            self.abils_matrix = estimates

    def score_abil_matrix(self,
                          score,
                          anchor=False,
                          items=None,
                          raters=None,
                          warm_corr=True,
                          tolerance=0.00001,
                          max_iters=100,
                          ext_score_adjustment=0.5):

        if isinstance(items, str):
            if items == 'all':
                items = self.items
            else:
                items = [items]

        if items is None:
            items = self.items

        if isinstance(raters, str):
            if raters == 'all':
                raters = [rater for rater in self.raters]
            else:
                raters = [raters]

        if anchor:
            if hasattr(self, 'anchor_diffs_matrix'):
                difficulties = self.anchor_diffs_matrix.loc[items]
                thresholds = self.anchor_thresholds_matrix

                if raters is None:
                    severities = pd.Series({'dummy_rater': {item: [0 for threshold in range(self.max_score + 1)]
                                                            for item in self.dataframe.columns}})

                else:
                    severities = {rater: self.anchor_severities_matrix[rater] for rater in raters}

            else:
                print('Anchor calibration required')
                return

        else:
            difficulties = self.diffs.loc[items]
            thresholds = self.thresholds

            if raters is None:
                severities = pd.Series({'dummy_rater': {item: [0 for threshold in range(self.max_score + 1)]
                                                        for item in self.dataframe.columns}})

            else:
                severities = {rater: self.severities_matrix[rater] for rater in raters}

        if raters is None:
            raters = ['dummy_rater']

        if raters is None:
            if items is None:
                person_filter = np.array([1 for item in self.dataframe.columns])

            else:
                person_filter = np.array([1 for item in items])

        else:
            if items is None:
                person_filter = np.array([[1 for item in self.dataframe.columns] for rater in raters])

            else:
                person_filter = np.array([[1 for item in items] for rater in raters])

        ext_score = person_filter.sum() * self.max_score

        if score == 0:
            score = ext_score_adjustment

        elif score == ext_score:
            score -= ext_score_adjustment

        estimate = log(score) - log(ext_score - score)

        change = 1
        iters = 0

        while (abs(change) > tolerance) & (iters <= max_iters):

            if raters is None:
                if items is None:
                    exp_list = [self.exp_score_matrix(estimate, item, difficulties, 'dummy_rater',
                                                      dummy_sevs, thresholds)
                                for item in self.dataframe.columns]
    
                    info_list = [self.variance_matrix(estimate, item, difficulties, 'dummy_rater',
                                                      dummy_sevs, thresholds)
                                 for item in self.dataframe.columns]
                    
                else:
                    exp_list = [self.exp_score_matrix(estimate, item, difficulties, 'dummy_rater',
                                                      dummy_sevs, thresholds)
                                for item in items]
    
                    info_list = [self.variance_matrix(estimate, item, difficulties, 'dummy_rater',
                                                      dummy_sevs, thresholds)
                                 for item in items]

            else:
                if items is None:
                    exp_list = [self.exp_score_matrix(estimate, item, difficulties, rater, severities, thresholds)
                                for item in self.dataframe.columns for rater in raters]
    
                    info_list = [self.variance_matrix(estimate, item, difficulties, rater, severities, thresholds)
                                 for item in self.dataframe.columns for rater in raters]
                    
                else:
                    exp_list = [self.exp_score_matrix(estimate, item, difficulties, rater, severities, thresholds)
                                for item in items for rater in raters]
    
                    info_list = [self.variance_matrix(estimate, item, difficulties, rater, severities, thresholds)
                                 for item in items for rater in raters]

            exp_list = np.array(exp_list)
            result = exp_list.sum()

            info_list = np.array(info_list)
            info = info_list.sum()

            change = max(-1, min(1, (result - score) / info))
            estimate -= change
            iters += 1

        if warm_corr:
            sevs = dict((rater, severities[rater]) for rater in raters)
            estimate += self.warm_matrix(pd.Series({score: estimate}), items, raters, sevs,
                                         person_filter, anchor=anchor)

        if iters >= max_iters:
            print('Maximum iterations reached before convergence.')

        if isinstance(estimate, pd.Series):
            return estimate.iloc[0]

        else:
            return estimate

    def abil_lookup_table_matrix(self,
                                 anchor=False,
                                 items=None,
                                 raters=None,
                                 ext_scores=True,
                                 warm_corr=True,
                                 tolerance=0.00001,
                                 max_iters=100,
                                 ext_score_adjustment=0.5):

        if items is None:
            items = self.items

            if raters is None:
                person_filter = np.array([1 for item in self.dataframe.columns])

            else:
                person_filter = np.array([[1 for item in self.dataframe.columns]
                                          for rater in raters])

        elif isinstance(items, str):
            if items == 'all':
                if raters is None:
                    person_filter = np.array([1 for item in self.dataframe.columns])

                else:
                    person_filter = np.array([[1 for item in self.dataframe.columns]
                                              for rater in raters])

            else:
                if raters is None:
                    person_filter = np.array([1])

                else:
                    person_filter = np.array([1 for rater in raters])

        else:
            if raters is None:
                person_filter = np.array([1 for item in items])

            else:
                person_filter = np.array([[1 for item in items]
                                          for rater in raters])

        ext_score = person_filter.sum() * self.max_score

        if ext_scores:
            scores = np.array([score for score in range(ext_score + 1)])

            used_scores = scores.astype(float)
            used_scores[0] += ext_score_adjustment
            used_scores[-1] -= ext_score_adjustment

        else:
            scores = np.array([score + 1 for score in range(ext_score - 1)])
            used_scores = scores.astype(float)

        abil_table = {score: self.score_abil_matrix(used_score, anchor=anchor, items=items, raters=raters,
                                                    warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                                    ext_score_adjustment=ext_score_adjustment)
                      for score, used_score in zip(scores, used_scores)}

        self.abil_table_matrix = pd.Series(abil_table)

    def warm_matrix(self,
                    abilities,
                    items,
                    raters,
                    severities,
                    person_filter,
                    anchor=False):

        '''
        Warm's (1989) bias correction for ML abiity estimates
        '''

        if isinstance(raters, str):
            if raters == 'all':
                raters = self.raters

        if anchor:
            difficulties = self.anchor_diffs_matrix.loc[items]
            thresholds = self.anchor_thresholds_matrix

        else:
            difficulties = self.diffs.loc[items]
            thresholds = self.thresholds

        c_p_df = {item: abilities - difficulties.loc[item] for item in items}
        c_p_df = pd.DataFrame(c_p_df)

        cat_probs = {cat: {rater: (cat * c_p_df - sum(thresholds[:cat + 1]))
                           for rater in raters}
                     for cat in range(self.max_score + 1)}

        for cat in range(self.max_score + 1):
            for rater in raters:
                for item in items:
                    cat_probs[cat][rater][item] -= sum(severities[rater][item][:cat + 1])

        for cat in range(self.max_score + 1):
            cat_probs[cat] = pd.concat(cat_probs[cat].values(), keys=cat_probs[cat].keys())
            cat_probs[cat] = np.exp(cat_probs[cat])

        den = sum(cat_probs[cat] for cat in range(self.max_score + 1))

        for cat in range(self.max_score + 1):
            cat_probs[cat] /= den
            cat_probs[cat] *= person_filter

        exp_score_df = sum(cat * df for cat, df in cat_probs.items())
        exp_score_df *= person_filter

        info_df = sum(df * (cat - exp_score_df) ** 2 for cat, df in cat_probs.items())
        info_df *= person_filter

        part_1 = sum((cat ** 3) * cat_probs[cat].sum(axis=1)
                     for cat in range(self.max_score + 1))
        part_1 = sum(part_1.loc[rater] for rater in raters)

        part_2 = 3 * ((info_df + (exp_score_df ** 2)) * exp_score_df).sum(axis=1)
        part_2 = sum(part_2.loc[rater] for rater in raters)

        part_3 = (2 * (exp_score_df ** 3)).sum(axis=1)
        part_3 = sum(part_3.loc[rater] for rater in raters)

        den = 2 * (sum(info_df.loc[rater].sum(axis=1) for rater in raters) ** 2)

        warm_correction = (part_1 - part_2 + part_3) / den

        return warm_correction

    def csem_matrix(self,
                    persons=None,
                    abilities=None,
                    anchor=False,
                    items=None,
                    raters=None,
                    warm_corr=True,
                    tolerance=0.00001,
                    max_iters=100,
                    ext_score_adjustment=0.5):

        if items is None:
            items = self.dataframe.columns

        if raters is None:
            raters = self.raters

        if anchor:
            if hasattr(self, 'anchor_diffs_matrix'):
                difficulties = self.anchor_diffs_matrix
                thresholds = self.anchor_thresholds_matrix
                severities = self.anchor_severities_matrix

            else:
                print('Anchor calibration required')
                return

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_matrix

        difficulties = difficulties.loc[items]
        severities = {rater: severities[rater] for rater in raters}

        if persons is not None:
            if anchor:
                abilities = self.anchor_abils_matrix.loc[persons]
            else:
                abilities = self.abils_matrix.loc[persons]

            person_data = self.dataframe.loc[persons, items]
            person_filter = person_data.notna().astype(float).replace(0, np.nan)

        if abilities is not None:
            abilities = {f'Ability_{abil}': abil for abil in abilities}
            abilities = pd.Series(abilities)
            person_filter = pd.DataFrame(1, index=abilities.index, columns=items)

        c_p_df = {item: abilities - difficulties.loc[item] for item in items}
        c_p_df = pd.DataFrame(c_p_df)

        cat_prob_dict_matrix = {cat: {rater: (cat * c_p_df - sum(thresholds[:cat + 1]))
                                      for rater in raters}
                                for cat in range(self.max_score + 1)}

        for cat in range(self.max_score + 1):
            for rater in raters:
                for item in items:
                    cat_prob_dict_matrix[cat][rater][item] -= sum(severities[rater][item][:cat + 1])

        for cat in range(self.max_score + 1):
            cat_prob_dict_matrix[cat] = pd.concat(cat_prob_dict_matrix[cat].values(),
                                                  keys=cat_prob_dict_matrix[cat].keys())
            cat_prob_dict_matrix[cat] = np.exp(cat_prob_dict_matrix[cat])

        den = sum(cat_prob_dict_matrix[cat] for cat in range(self.max_score + 1))

        for cat in range(self.max_score + 1):
            cat_prob_dict_matrix[cat] /= den
            cat_prob_dict_matrix[cat] *= person_filter

        exp_score_df = sum(cat * df for cat, df in cat_prob_dict.items())
        exp_score_df *= person_filter

        info_df = sum(df * (cat - exp_score_df) ** 2 for cat, df in cat_prob_dict.items())
        info_df *= person_filter

        cond_sems = 1 / (info_df.sum(axis=1) ** 0.5)

        return csems

    def category_counts_item(self,
                             item,
                             rater=None):

        if item in self.dataframe.columns:

            if rater is None:
                return self.dataframe[item].value_counts().fillna(0).astype(int)

            else:
                if rater in self.raters:
                    return self.dataframe.xs(rater)[item].value_counts().fillna(0).astype(int)

                else:
                    print('Invalid rater name')

        else:
            print('Invalid item name')

    def category_counts_df(self):
        cat_counts_dict = {item: {} for item in self.items}

        for item in self.items:
            for score in range(self.max_score + 1):
                if score in self.category_counts_item(item).keys():
                    if self.category_counts_item(item)[score] == self.category_counts_item(item)[score]:
                        cat_counts_dict[item][score] = int(self.category_counts_item(item)[score])
                    else:
                        cat_counts_dict[item][score] = 0
                else:
                    cat_counts_dict[item][score] = 0

        category_counts_df = pd.DataFrame(cat_counts_dict).T

        category_counts_df['Total'] = self.dataframe.count()
        category_counts_df['Missing'] = self.dataframe.shape[0] - category_counts_df['Total']

        category_counts_df = category_counts_df.astype(int)

        category_counts_df.loc['Total'] = category_counts_df.sum()

        self.category_counts = category_counts_df

        self.category_counts_raters = {rater: {item: {} for item in self.items}
                                       for rater in self.raters}

        for rater in self.raters:
            for item in self.items:
                for score in range(self.max_score + 1):
                    if score in self.category_counts_item(item, rater).keys():
                        if (self.category_counts_item(item, rater)[score] ==
                            self.category_counts_item(item, rater)[score]):
                            self.category_counts_raters[rater][item][score] = int(self.category_counts_item(item, rater)[score])
                        else:
                            self.category_counts_raters[rater][item][score] = 0
                    else:
                        self.category_counts_raters[rater][item][score] = 0

            self.category_counts_raters[rater] = pd.DataFrame(self.category_counts_raters[rater]).T

            self.category_counts_raters[rater]['Total'] = self.dataframe.xs(rater).count()
            self.category_counts_raters[rater]['Missing'] = (len(self.dataframe.xs(rater).index) -
                                                             self.category_counts_raters[rater]['Total'])

            self.category_counts_raters[rater].loc['Total'] = self.category_counts_raters[rater].sum()

        self.category_counts_raters = pd.concat(self.category_counts_raters.values(),
                                                keys=self.category_counts_raters.keys())

        self.category_counts_raters = self.category_counts_raters.astype(int)

    def item_stats_df_global(self,
                             anchor_raters=None,
                             full=False,
                             ext_scores=True,
                             zstd=False,
                             point_measure_corr=False,
                             dp=3,
                             warm_corr=True,
                             tolerance=0.00001,
                             max_iters=100,
                             ext_score_adjustment=0.5,
                             method='cos',
                             constant=0.1,
                             matrix_power=3,
                             log_lik_tol=0.000001,
                             no_of_samples=100,
                             interval=None):

        if full:
            zstd = True
            point_measure_corr = True

            if interval is None:
                interval = 0.95

        if anchor_raters is not None:
            if (not hasattr(self, 'anchor_severites_global')) or (self.anchor_raters_global != anchor_raters):
                self.calibrate_global_anchor(anchor_raters, constant=constant, method=method, matrix_power=matrix_power,
                                             log_lik_tol=log_lik_tol)
                self.std_errors_global(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method, matrix_power=matrix_power,
                                       log_lik_tol=log_lik_tol)

            elif (self.anchor_item_low_global is None) and (interval is not None):
                self.std_errors_global(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method, matrix_power=matrix_power,
                                       log_lik_tol=log_lik_tol)

        else:
            if not hasattr(self, 'item_se_global'):
                self.std_errors_global(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method, matrix_power=matrix_power,
                                       log_lik_tol=log_lik_tol)

            elif (self.item_low is None) and (interval is not None):
                self.std_errors_global(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method, matrix_power=matrix_power,
                                       log_lik_tol=log_lik_tol)

        if not hasattr(self, 'item_outfit_ms_global'):
            self.item_fit_statistics_global(warm_corr=warm_corr, ext_scores=ext_scores, tolerance=tolerance,
                                            max_iters=max_iters, ext_score_adjustment=ext_score_adjustment,
                                            method=method, constant=constant, matrix_power=matrix_power,
                                            log_lik_tol=log_lik_tol)

        if anchor_raters is not None:
            difficulties = self.anchor_diffs_global
            std_errors = self.anchor_item_se_global
            low = self.anchor_item_low_global
            high = self.anchor_item_high_global

        else:
            difficulties = self.diffs
            std_errors = self.item_se
            low = self.item_low
            high = self.item_high

        self.item_stats_global = pd.DataFrame()

        self.item_stats_global['Estimate'] = difficulties.astype(float).round(dp)
        self.item_stats_global['SE'] = std_errors.astype(float).round(dp)

        if interval is not None:
            self.item_stats_global[f'{round((1 - interval) * 50, 1)}%'] = low.astype(float).round(dp)
            self.item_stats_global[f'{round((1 + interval) * 50, 1)}%'] = high.astype(float).round(dp)

        self.item_stats_global['Count'] = self.response_counts.astype(int)
        self.item_stats_global['Facility'] = self.item_facilities.astype(float).round(dp)

        self.item_stats_global['Infit MS'] = self.item_infit_ms_global.astype(float).round(dp)
        if zstd:
            self.item_stats_global['Infit Z'] = self.item_infit_zstd_global.astype(float).round(dp)

        self.item_stats_global['Outfit MS'] = self.item_outfit_ms_global.astype(float).round(dp)
        if zstd:
            self.item_stats_global['Outfit Z'] = self.item_outfit_zstd_global.astype(float).round(dp)

        if point_measure_corr:
            self.item_stats_global['PM corr'] = self.point_measure_global.astype(float).round(dp)
            self.item_stats_global['Exp PM corr'] = self.exp_point_measure_global.astype(float).round(dp)

        self.item_stats_global.index = self.dataframe.columns

    def threshold_stats_df_global(self,
                                  anchor_raters=None,
                                  full=False,
                                  zstd=False,
                                  disc=False,
                                  point_measure_corr=False,
                                  dp=3,
                                  warm_corr=True,
                                  tolerance=0.00001,
                                  max_iters=100,
                                  ext_score_adjustment=0.5,
                                  method='cos',
                                  constant=0.1,
                                  matrix_power=3,
                                  log_lik_tol=0.000001,
                                  no_of_samples=100,
                                  interval=None):

        if full:
            zstd = True
            disc = True
            point_measure_corr = True

            if interval is None:
                interval = 0.95

        if anchor_raters is not None:
            if (not hasattr(self, 'anchor_severites_global')) or (self.anchor_raters_global != anchor_raters):
                self.calibrate_global_anchor(anchor_raters, constant=constant, method=method, matrix_power=matrix_power,
                                             log_lik_tol=log_lik_tol)
                self.std_errors_global(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method, matrix_power=matrix_power,
                                       log_lik_tol=log_lik_tol)

            elif (self.anchor_item_low_global is None) and (interval is not None):
                self.std_errors_global(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method, matrix_power=matrix_power,
                                       log_lik_tol=log_lik_tol)
        else:
            if not hasattr(self, 'item_se_global'):
                self.std_errors_global(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method, matrix_power=matrix_power,
                                       log_lik_tol=log_lik_tol)

            elif (self.item_low is None) and (interval is not None):
                self.std_errors_global(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method, matrix_power=matrix_power,
                                       log_lik_tol=log_lik_tol)

        if (not hasattr(self, 'threshold_outfit_ms_global')) or (self.anchor_raters_global != anchor_raters):
            self.threshold_fit_statistics_global(anchor_raters=anchor_raters, warm_corr=warm_corr, tolerance=tolerance,
                                                 max_iters=max_iters, ext_score_adjustment=ext_score_adjustment,
                                                 method=method, constant=constant, matrix_power=matrix_power,
                                                 log_lik_tol=log_lik_tol)

        if anchor_raters is not None:
            thresholds = self.anchor_thresholds_global
        else:
            thresholds = self.thresholds

        self.threshold_stats_global = pd.DataFrame()

        self.threshold_stats_global['Estimate'] = thresholds[1:].round(dp)

        if anchor_raters is not None:
            self.threshold_stats_global['SE'] = self.anchor_threshold_se_global[1:].round(dp)

        else:
            self.threshold_stats_global['SE'] = self.threshold_se_global[1:].round(dp)

        if interval is not None:
            if anchor_raters is not None:
                self.threshold_stats_global[f'{round((1 - interval) * 50, 1)}%'] = self.anchor_threshold_low_global[1:].round(dp)
                self.threshold_stats_global[f'{round((1 + interval) * 50, 1)}%'] = self.anchor_threshold_high_global[1:].round(dp)

            else:
                self.threshold_stats_global[f'{round((1 - interval) * 50, 1)}%'] = self.threshold_low_global[1:].round(dp)
                self.threshold_stats_global[f'{round((1 + interval) * 50, 1)}%'] = self.threshold_high_global[1:].round(dp)

        infit_ms_vector = self.threshold_infit_ms_global.reset_index(drop=True)
        self.threshold_stats_global['Infit MS'] = infit_ms_vector.round(dp)
        
        if zstd:
            infit_z_vector = self.threshold_infit_zstd_global.reset_index(drop=True)
            self.threshold_stats_global['Infit Z'] = infit_z_vector.round(dp)
            
        outfit_ms_vector = self.threshold_outfit_ms_global.reset_index(drop=True)
        self.threshold_stats_global['Outfit MS'] = outfit_ms_vector.round(dp)
        
        if zstd:
            outfit_z_vector = self.threshold_outfit_zstd_global.reset_index(drop=True)
            self.threshold_stats_global['Outfit Z'] = outfit_z_vector.round(dp)

        if disc:
            disc_vector = self.threshold_discrimination_global.reset_index(drop=True)
            self.threshold_stats_global['Discrim'] = disc_vector.round(dp)

        if point_measure_corr:
            pm_vector = self.threshold_point_measure_global.reset_index(drop=True)
            exp_pm_vector = self.threshold_exp_point_measure_global.reset_index(drop=True)
            self.threshold_stats_global['PM corr'] = pm_vector.round(dp)
            self.threshold_stats_global['Exp PM corr'] = exp_pm_vector.round(dp)

        self.threshold_stats_global.index = [f'Threshold {threshold + 1}' for threshold in range(self.max_score)]

    def rater_stats_df_global(self,
                              anchor_raters=None,
                              full=False,
                              zstd=False,
                              dp=3,
                              warm_corr=True,
                              tolerance=0.00001,
                              max_iters=100,
                              ext_score_adjustment=0.5,
                              method='cos',
                              constant=0.1,
                              matrix_power=3,
                              log_lik_tol=0.000001,
                              no_of_samples=100,
                              interval=None):

        if full:
            zstd = True

            if interval is None:
                interval = 0.95

        if anchor_raters is not None:
            if (not hasattr(self, 'anchor_severites_global')) or (self.anchor_raters_global != anchor_raters):
                self.calibrate_global_anchor(anchor_raters, constant=constant, method=method)
                self.std_errors_global(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method, matrix_power=matrix_power,
                                       log_lik_tol=log_lik_tol)

        else:
            if not hasattr(self, 'item_se_global'):
                self.std_errors_global(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method, matrix_power=matrix_power,
                                       log_lik_tol=log_lik_tol)

        if (not hasattr(self, 'rater_outfit_ms_global')) or (self.anchor_raters_global != anchor_raters):
            self.rater_fit_statistics_global(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                             ext_score_adjustment=ext_score_adjustment, method=method,
                                             constant=constant, matrix_power=matrix_power, log_lik_tol=log_lik_tol,
                                             no_of_samples=no_of_samples, interval=interval)

        if anchor_raters is not None:
            severities = self.anchor_severities_global

            se = self.anchor_rater_se_global
            low = self.anchor_rater_low_global
            high = self.anchor_rater_high_global

        else:
            severities = self.severities_global

            se = self.rater_se_global
            low = self.rater_low_global
            high = self.rater_high_global

        self.rater_stats_global = pd.DataFrame()

        self.rater_stats_global['Estimate'] = severities.astype(float).round(dp)
        self.rater_stats_global['SE'] = se.astype(float).round(dp)

        if interval is not None:
            self.rater_stats_global[f'{round((1 - interval) * 50, 1)}%'] = low.astype(float).round(dp)
            self.rater_stats_global[f'{round((1 + interval) * 50, 1)}%'] = high.astype(float).round(dp)

        self.rater_stats_global['Count'] = np.array([self.dataframe.xs(rater).count().sum()
                                                     for rater in self.raters]).astype(int)

        self.rater_stats_global['Infit MS'] = self.rater_infit_ms_global.astype(float).round(dp)
        if zstd:
            self.rater_stats_global['Infit Z'] = self.rater_infit_zstd_global.astype(float).round(dp)
        self.rater_stats_global['Outfit MS'] = self.rater_outfit_ms_global.astype(float).round(dp)
        if zstd:
            self.rater_stats_global['Outfit Z'] = self.rater_outfit_zstd_global.astype(float).round(dp)

        self.rater_stats_global.index = self.raters

    def person_stats_df_global(self,
                               anchor_raters=None,
                               full=False,
                               rsem=False,
                               zstd=False,
                               interval=None,
                               no_of_samples=100,
                               dp=3,
                               warm_corr=True,
                               tolerance=0.00001,
                               max_iters=100,
                               ext_score_adjustment=0.5,
                               method='cos',
                               constant=0.1,
                               matrix_power=3,
                               log_lik_tol=0.000001,):

        '''
        Produces a person stats dataframe with raw score, ability estimate,
        CSEM and RSEM for each person.
        '''

        if anchor_raters is not None:
            if (not hasattr(self, 'anchor_severites_global')) or (self.anchor_raters_global != anchor_raters):
                self.calibrate_global_anchor(anchor_raters, constant=constant, method=method, matrix_power=matrix_power,
                                             log_lik_tol=log_lik_tol)
                self.std_errors_global(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method, matrix_power=matrix_power,
                                       log_lik_tol=log_lik_tol)

        if not hasattr(self, 'person_outfit_ms_global'):
            self.person_fit_statistics_global(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                              ext_score_adjustment=ext_score_adjustment, method=method,
                                              constant=constant, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if full:
            rsem = True
            zstd = True

        person_stats_df = pd.DataFrame()
        person_stats_df.index = self.dataframe.index.get_level_values(1).unique()

        if anchor_raters is None:
            person_stats_df['Estimate'] = self.abils_global.round(dp)

        else:
            person_stats_df['Estimate'] = self.anchor_abils_global.round(dp)

        person_stats_df['CSEM'] = self.csem_vector_global.round(dp)
        if rsem:
            person_stats_df['RSEM'] = self.rsem_vector_global.round(dp)

        person_stats_df['Score'] = [np.nan for person in self.persons]
        person_stats_df['Score'] = self.dataframe.unstack(level=0).sum(axis=1)
        person_stats_df['Score'] = person_stats_df['Score'].astype(int)

        person_stats_df['Max score'] = [np.nan for person in self.persons]
        person_stats_df.update({'Max score': self.dataframe.unstack(level=0).count(axis=1) * self.max_score})
        person_stats_df['Max score'] = person_stats_df['Max score'].astype(int)

        person_stats_df['p'] = [np.nan for person in self.persons]
        p_vector = self.dataframe.unstack(level=0).mean(axis=1) / self.max_score
        person_stats_df.update({'p': p_vector.astype(float)})
        person_stats_df['p'] = person_stats_df['p'].round(dp)

        person_stats_df['Infit MS'] = [np.nan for person in self.persons]
        infit_vector = self.person_infit_ms_global.astype(float)
        person_stats_df.update({'Infit MS': infit_vector.round(dp)})

        if zstd:
            person_stats_df['Infit Z'] = [np.nan for person in self.persons]
            infit_z_vector = self.person_infit_zstd_global.astype(float)
            person_stats_df.update({'Infit Z': infit_z_vector.round(dp)})

        person_stats_df['Outfit MS'] = [np.nan for person in self.persons]
        outfit_vector = self.person_outfit_ms_global.astype(float)
        person_stats_df.update({'Outfit MS': outfit_vector.round(dp)})

        if zstd:
            person_stats_df['Outfit Z'] = [np.nan for person in self.persons]
            outfit_z_vector = self.person_outfit_zstd_global.astype(float)
            person_stats_df.update({'Outfit Z': outfit_z_vector.round(dp)})

        self.person_stats_global = person_stats_df

    def test_stats_df_global(self,
                             dp=3,
                             warm_corr=True,
                             tolerance=0.00001,
                             max_iters=100,
                             ext_score_adjustment=0.5,
                             method='cos',
                             constant=0.1,
                             matrix_power=3,
                             log_lik_tol=0.000001):

        if not hasattr(self, 'psi_global'):
            self.test_fit_statistics_global(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                            ext_score_adjustment=ext_score_adjustment, method=method,
                                            constant=constant, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        self.test_stats_global = pd.DataFrame()

        self.test_stats_global['Items'] = [self.diffs.mean(),
                                           self.diffs.std(),
                                           self.isi_global,
                                           self.item_strata_global,
                                           self.item_reliability_global]

        self.test_stats_global['Persons'] = [self.abils_global.mean(),
                                             self.abils_global.std(),
                                             self.psi_global,
                                             self.person_strata_global,
                                             self.person_reliability_global]

        self.test_stats_global.index = ['Mean', 'SD', 'Separation ratio', 'Strata', 'Reliability']
        self.test_stats_global = round(self.test_stats_global, dp)

    def save_stats_global(self,
                          filename,
                          format='csv',
                          dp=3,
                          warm_corr=True,
                          tolerance=0.00001,
                          max_iters=100,
                          ext_score_adjustment=0.5,
                          method='cos',
                          constant=0.1,
                          matrix_power=3,
                          log_lik_tol=0.000001,
                          no_of_samples=100,
                          interval=None):

        if not hasattr(self, 'item_stats_global'):
            self.item_stats_df_global(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                      ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                      matrix_power=matrix_power, log_lik_tol=log_lik_tol,
                                      no_of_samples=no_of_samples, interval=interval)

        if not hasattr(self, 'threshold_stats_global'):
            self.threshold_stats_df_global(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                           ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                           matrix_power=matrix_power, log_lik_tol=log_lik_tol,
                                           no_of_samples=no_of_samples, interval=interval)

        if not hasattr(self, 'rater_stats_global'):
            self.rater_stats_df_global(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                       ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                       matrix_power=matrix_power, log_lik_tol=log_lik_tol,
                                       no_of_samples=no_of_samples, interval=interval)

        if not hasattr(self, 'person_stats_global'):
            self.person_stats_df_global(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                        ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                        matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if not hasattr(self, 'test_stats_global'):
            self.test_stats_df_global(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                      ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                      matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if format == 'xlsx':

            if filename[-5:] != '.xlsx':
                filename += '.xlsx'

            writer = pd.ExcelWriter(filename, engine='xlsxwriter')

            self.item_stats_global.to_excel(writer, sheet_name='Item statistics')
            self.threshold_stats_global.to_excel(writer, sheet_name='Threshold statistics')
            self.rater_stats_global.to_excel(writer, sheet_name='Rater statistics')
            self.person_stats_global.to_excel(writer, sheet_name='Person statistics')
            self.test_stats_global.to_excel(writer, sheet_name='Test statistics')

            writer.close()

        else:
            if filename[-4:] == '.csv':
                filename = filename[:-4]

            self.item_stats_global.to_csv(f'{filename}_item_stats.csv')
            self.threshold_stats_global.to_csv(f'{filename}_threshold_stats.csv')
            self.rater_stats_global.to_csv(f'{filename}_rater_stats.csv')
            self.person_stats_global.to_csv(f'{filename}_person_stats.csv')
            self.test_stats_global.to_csv(f'{filename}_test_stats.csv')

    def item_stats_df_items(self,
                            anchor_raters=None,
                            full=False,
                            zstd=False,
                            point_measure_corr=False,
                            dp=3,
                            warm_corr=True,
                            tolerance=0.00001,
                            max_iters=100,
                            ext_score_adjustment=0.5,
                            method='cos',
                            constant=0.1,
                            matrix_power=3,
                            log_lik_tol=0.000001,
                            no_of_samples=100,
                            interval=None):

        if full:
            zstd=True
            point_measure_corr = True

            if interval is None:
                interval = 0.95

        if anchor_raters is not None:
            if (not hasattr(self, 'anchor_severites_items')) or (self.anchor_raters_items != anchor_raters):
                self.calibrate_items_anchor(anchor_raters, constant=constant, method=method)
                self.std_errors_items(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                      constant=constant, method=method, matrix_power=matrix_power,
                                      log_lik_tol=log_lik_tol)

            elif (self.anchor_item_low_items is None) and (interval is not None):
                self.std_errors_items(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                      constant=constant, method=method, matrix_power=matrix_power,
                                      log_lik_tol=log_lik_tol)

        else:
            if not hasattr(self, 'item_se_items'):
                self.std_errors_items(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                      constant=constant, method=method, matrix_power=matrix_power,
                                      log_lik_tol=log_lik_tol)

            elif (self.item_low is None) and (interval is not None):
                self.std_errors_items(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                      constant=constant, method=method, matrix_power=matrix_power,
                                      log_lik_tol=log_lik_tol)

        if not hasattr(self, 'item_outfit_ms_items'):
            self.item_fit_statistics_items(warm_corr=warm_corr, tolerance=tolerance,
                                           max_iters=max_iters, ext_score_adjustment=ext_score_adjustment, method=method,
                                           constant=constant, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if anchor_raters is not None:
            difficulties = self.anchor_diffs_items
            std_errors = self.anchor_item_se_items
            low = self.anchor_item_low_items
            high = self.anchor_item_high_items
        else:
            difficulties = self.diffs
            std_errors = self.item_se
            low = self.item_low
            high = self.item_high

        self.item_stats_items = pd.DataFrame()

        self.item_stats_items['Estimate'] = difficulties.astype(float).round(dp)
        self.item_stats_items['SE'] = std_errors.astype(float).round(dp)

        if interval is not None:
            self.item_stats_items[f'{round((1 - interval) * 50, 1)}%'] = low.astype(float).round(dp)
            self.item_stats_items[f'{round((1 + interval) * 50, 1)}%'] = high.astype(float).round(dp)

        self.item_stats_items['Count'] = self.response_counts.astype(int)
        self.item_stats_items['Facility'] = self.item_facilities.astype(float).round(dp)

        self.item_stats_items['Infit MS'] = self.item_infit_ms_items.astype(float).round(dp)
        if zstd:
            self.item_stats_items['Infit Z'] = self.item_infit_zstd_items.astype(float).round(dp)

        self.item_stats_items['Outfit MS'] = self.item_outfit_ms_items.astype(float).round(dp)
        if zstd:
            self.item_stats_items['Outfit Z'] = self.item_outfit_zstd_items.astype(float).round(dp)

        if point_measure_corr:
            self.item_stats_items['PM corr'] = self.point_measure_items.astype(float).round(dp)
            self.item_stats_items['Exp PM corr'] = self.exp_point_measure_items.astype(float).round(dp)

        self.item_stats_items.index = self.dataframe.columns

    def threshold_stats_df_items(self,
                                 anchor_raters=None,
                                 full=False,
                                 zstd=False,
                                 disc=False,
                                 point_measure_corr=False,
                                 dp=3,
                                 warm_corr=True,
                                 tolerance=0.00001,
                                 max_iters=100,
                                 ext_score_adjustment=0.5,
                                 method='cos',
                                 constant=0.1,
                                 matrix_power=3,
                                 log_lik_tol=0.000001,
                                 no_of_samples=100,
                                 interval=None):

        if full:
            zstd = True
            disc = True
            point_measure_corr = True

            if interval is None:
                interval = 0.95

        if anchor_raters is not None:
            if (not hasattr(self, 'anchor_severites_items')) or (self.anchor_raters_items != anchor_raters):
                self.calibrate_items_anchor(anchor_raters, constant=constant, method=method, matrix_power=matrix_power,
                                            log_lik_tol=log_lik_tol)
                self.std_errors_items(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                      constant=constant, method=method, matrix_power=matrix_power,
                                      log_lik_tol=log_lik_tol)

            elif (self.anchor_item_low_items is None) and (interval is not None):
                self.std_errors_items(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                      constant=constant, method=method, matrix_power=matrix_power,
                                      log_lik_tol=log_lik_tol)

        else:
            if not hasattr(self, 'item_se_items'):
                self.std_errors_items(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                      constant=constant, method=method, matrix_power=matrix_power,
                                      log_lik_tol=log_lik_tol)

            elif (self.item_low is None) and (interval is not None):
                self.std_errors_items(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                      constant=constant, method=method, matrix_power=matrix_power,
                                      log_lik_tol=log_lik_tol)

        if not hasattr(self, 'threshold_outfit_ms_items'):
            self.threshold_fit_statistics_items(anchor_raters=anchor_raters, warm_corr=warm_corr, tolerance=tolerance,
                                                max_iters=max_iters, ext_score_adjustment=ext_score_adjustment,
                                                method=method, constant=constant)

        if anchor_raters is not None:
            thresholds = self.anchor_thresholds_items
        else:
            thresholds = self.thresholds

        self.threshold_stats_items = pd.DataFrame()

        self.threshold_stats_items['Estimate'] = thresholds[1:].round(dp)

        if anchor_raters is not None:
            self.threshold_stats_items['SE'] = self.anchor_threshold_se_items[1:].round(dp)

        else:
            self.threshold_stats_items['SE'] = self.threshold_se_items[1:].round(dp)

        if interval is not None:
            if anchor_raters is not None:
                self.threshold_stats_items[f'{round((1 - interval) * 50, 1)}%'] = self.anchor_threshold_low_items[1:].round(dp)
                self.threshold_stats_items[f'{round((1 + interval) * 50, 1)}%'] = self.anchor_threshold_high_items[1:].round(dp)

            else:
                self.threshold_stats_items[f'{round((1 - interval) * 50, 1)}%'] = self.threshold_low_items[1:].round(dp)
                self.threshold_stats_items[f'{round((1 + interval) * 50, 1)}%'] = self.threshold_high_items[1:].round(dp)

        infit_ms_vector = self.threshold_infit_ms_items.reset_index(drop=True)
        self.threshold_stats_items['Infit MS'] = infit_ms_vector.round(dp)
        
        if zstd:
            infit_z_vector = self.threshold_infit_zstd_items.reset_index(drop=True)
            self.threshold_stats_items['Infit Z'] = infit_z_vector.round(dp)
            
        outfit_ms_vector = self.threshold_outfit_ms_items.reset_index(drop=True)
        self.threshold_stats_items['Outfit MS'] = outfit_ms_vector.round(dp)
        
        if zstd:
            outfit_z_vector = self.threshold_outfit_zstd_items.reset_index(drop=True)
            self.threshold_stats_items['Outfit Z'] = outfit_z_vector.round(dp)

        if disc:
            disc_vector = self.threshold_discrimination_items.reset_index(drop=True)
            self.threshold_stats_items['Discrim'] = disc_vector.round(dp)

        if point_measure_corr:
            pm_vector = self.threshold_point_measure_items.reset_index(drop=True)
            exp_pm_vector = self.threshold_exp_point_measure_items.reset_index(drop=True)
            self.threshold_stats_items['PM corr'] = pm_vector.round(dp)
            self.threshold_stats_items['Exp PM corr'] = exp_pm_vector.round(dp)

        self.threshold_stats_items.index = [f'Threshold {threshold + 1}' for threshold in range(self.max_score)]

    def rater_stats_df_items(self,
                             anchor_raters=None,
                             full=False,
                             zstd=False,
                             dp=3,
                             warm_corr=True,
                             tolerance=0.00001,
                             max_iters=100,
                             ext_score_adjustment=0.5,
                             method='cos',
                             constant=0.1,
                             matrix_power=3,
                             log_lik_tol=0.000001,
                             no_of_samples=100,
                             interval=None):

        if full:
            zstd = True

            if interval is None:
                interval = 0.95

        if anchor_raters is not None:
            if (not hasattr(self, 'anchor_severites_items')) or (self.anchor_raters_items != anchor_raters):
                self.calibrate_items_anchor(anchor_raters, constant=constant, method=method)
                self.std_errors_items(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                      constant=constant, method=method, matrix_power=matrix_power,
                                      log_lik_tol=log_lik_tol)

        else:
            if not hasattr(self, 'item_se_items'):
                self.std_errors_items(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                      constant=constant, method=method, matrix_power=matrix_power,
                                      log_lik_tol=log_lik_tol)

        if not hasattr(self, 'rater_outfit_ms_items'):
            self.rater_fit_statistics_items(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                            ext_score_adjustment=ext_score_adjustment, method=method,
                                            constant=constant, matrix_power=matrix_power, log_lik_tol=log_lik_tol,
                                            no_of_samples=no_of_samples, interval=interval)

        if anchor_raters is not None:
            severities = self.anchor_severities_items

            se = self.anchor_rater_se_items
            low = self.anchor_rater_low_items
            high = self.anchor_rater_high_items

        else:
            severities = self.severities_items

            se = self.rater_se_items
            low = self.rater_low_items
            high = self.rater_high_items

        self.rater_stats_items = {}

        for item in self.dataframe.columns:

            item_stats = pd.DataFrame()

            item_stats['Estimate'] = np.array([severities[rater][item] for rater in self.raters]).astype(float).round(dp)
            item_stats['SE'] = np.array([se[rater][item] for rater in self.raters]).astype(float).round(dp)

            if interval is not None:
                item_stats[ f'{round((1 - interval) * 50, 1)}%'] = np.array([low[rater][item]
                                                                             for rater in self.raters]).astype(float).round(dp)
                item_stats[f'{round((1 + interval) * 50, 1)}%'] = np.array([high[rater][item]
                                                                            for rater in self.raters]).astype(float).round(dp)

            item_stats.index = self.raters
            self.rater_stats_items[item] = item_stats.T

            if zstd:
                ov_stats_df = pd.DataFrame(index=self.raters,
                                           columns=['Count', 'Infit MS', 'Infit Z', 'Outfit MS', 'Outfit Z'])
    
                count_vector = {rater: self.dataframe.xs(rater).count().sum() for rater in self.raters}
                ov_stats_df.update({'Count': count_vector})
                ov_stats_df['Count'] = ov_stats_df['Count'].astype(int)
                            
                infit_ms_vector = self.rater_infit_ms_items.astype(float)
                ov_stats_df.update({'Infit MS': infit_ms_vector.round(dp)})
                
                infit_z_vector = self.rater_infit_zstd_items.astype(float)
                ov_stats_df.update({'Infit Z': infit_z_vector.round(dp)})
                            
                outfit_ms_vector = self.rater_outfit_ms_items.astype(float)
                ov_stats_df.update({'Outfit MS': outfit_ms_vector.round(dp)})
                
                outfit_z_vector = self.rater_outfit_zstd_items.astype(float)
                ov_stats_df.update({'Outfit Z': outfit_z_vector.round(dp)})
                
            else:
                ov_stats_df = pd.DataFrame(index=self.raters,
                                           columns=['Count', 'Infit MS', 'Outfit MS'])
        
                count_vector = {rater: self.dataframe.xs(rater).count().sum() for rater in self.raters}
                ov_stats_df.update({'Count': count_vector})
                ov_stats_df['Count'] = ov_stats_df['Count'].astype(int)
                            
                infit_ms_vector = self.rater_infit_ms_items.astype(float)
                ov_stats_df.update({'Infit MS': infit_ms_vector.round(dp)})
                            
                outfit_ms_vector = self.rater_outfit_ms_items.astype(float)
                ov_stats_df.update({'Outfit MS': outfit_ms_vector.round(dp)})
                
        self.rater_stats_items['Overall statistics'] = ov_stats_df.T
        self.rater_stats_items = pd.concat(self.rater_stats_items.values(), keys=self.rater_stats_items.keys()).T

    def person_stats_df_items(self,
                              anchor_raters=None,
                              full=False,
                              rsem=False,
                              zstd=False,
                              dp=3,
                              warm_corr=True,
                              tolerance=0.00001,
                              max_iters=100,
                              ext_score_adjustment=0.5,
                              method='cos',
                              constant=0.1,
                              matrix_power=3,
                              log_lik_tol=0.000001,):

        '''
        Produces a person stats dataframe with raw score, ability estimate,
        CSEM and RSEM for each person.
        '''

        if not hasattr(self, 'person_outfit_ms_items'):
            self.person_fit_statistics_items(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                             ext_score_adjustment=ext_score_adjustment, method=method,
                                             constant=constant, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if full:
            rsem = True
            zstd = True

        person_stats_df = pd.DataFrame()
        person_stats_df.index = self.dataframe.index.get_level_values(1).unique()

        if anchor_raters is None:
            person_stats_df['Estimate'] = self.abils_items.round(dp)

        else:
            person_stats_df['Estimate'] = self.anchor_abils_items.round(dp)

        person_stats_df['CSEM'] = self.csem_vector_items.round(dp)
        if rsem:
            person_stats_df['RSEM'] = self.rsem_vector_items.round(dp)

        person_stats_df['Score'] = [np.nan for person in self.persons]
        score_vector = self.dataframe.unstack(level=0).sum(axis=1)
        person_stats_df.update({'Score': score_vector})
        person_stats_df['Score'] = person_stats_df['Score'].astype(int)

        person_stats_df['Max score'] = [np.nan for person in self.persons]
        max_score_vector = self.dataframe.unstack(level=0).count(axis=1) * self.max_score
        person_stats_df.update({'Max score': max_score_vector})
        person_stats_df['Max score'] = person_stats_df['Max score'].astype(int)

        person_stats_df['p'] = [np.nan for person in self.persons]
        p_vector = self.dataframe.unstack(level=0).mean(axis=1) / self.max_score
        person_stats_df.update({'p': p_vector.astype(float)})
        person_stats_df['p'] = person_stats_df['p'].round(dp)

        person_stats_df['Infit MS'] = [np.nan for person in self.persons]
        infit_ms_vector = self.person_infit_ms_items.astype(float)
        person_stats_df.update({'Infit MS': infit_ms_vector.round(dp)})

        if zstd:
            person_stats_df['Infit Z'] = [np.nan for person in self.persons]
            infit_z_vector = self.person_infit_zstd_items.astype(float)
            person_stats_df.update({'Infit Z': infit_z_vector.round(dp)})

        person_stats_df['Outfit MS'] = [np.nan for person in self.persons]
        outfit_ms_vector = self.person_outfit_ms_items.astype(float)
        person_stats_df.update({'Outfit MS': outfit_ms_vector.round(dp)})

        if zstd:
            person_stats_df['Outfit Z'] = [np.nan for person in self.persons]
            outfit_z_vector = self.person_outfit_zstd_items.astype(float)
            person_stats_df.update({'Outfit Z': outfit_z_vector.round(dp)})

        self.person_stats_items = person_stats_df

    def test_stats_df_items(self,
                            dp=3,
                            warm_corr=True,
                            tolerance=0.00001,
                            max_iters=100,
                            ext_score_adjustment=0.5,
                            method='cos',
                            constant=0.1,
                            matrix_power=3,
                            log_lik_tol=0.000001):

        if not hasattr(self, 'psi_items'):
            self.test_fit_statistics_items(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                           ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                           matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        self.test_stats_items = pd.DataFrame()

        self.test_stats_items['Items'] = [self.diffs.mean(),
                                          self.diffs.std(),
                                          self.isi_items,
                                          self.item_strata_items,
                                          self.item_reliability_items]

        self.test_stats_items['Persons'] = [self.abils_items.mean(),
                                            self.abils_items.std(),
                                            self.psi_items,
                                            self.person_strata_items,
                                            self.person_reliability_items]

        self.test_stats_items.index = ['Mean', 'SD', 'Separation ratio', 'Strata', 'Reliability']
        self.test_stats_items = round(self.test_stats_items, dp)

    def save_stats_items(self,
                         filename,
                         format='csv',
                         dp=3,
                         warm_corr=True,
                         tolerance=0.00001,
                         max_iters=100,
                         ext_score_adjustment=0.5,
                         method='cos',
                         constant=0.1,
                         matrix_power=3,
                         log_lik_tol=0.000001,
                         no_of_samples=100,
                         interval=None):

        if not hasattr(self, 'item_stats_items'):
            self.item_stats_df_items(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                     ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                     matrix_power=matrix_power, log_lik_tol=log_lik_tol,
                                     no_of_samples=no_of_samples, interval=interval)

        if not hasattr(self, 'threshold_stats_items'):
            self.threshold_stats_df_items(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                          ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                          matrix_power=matrix_power, log_lik_tol=log_lik_tol,
                                          no_of_samples=no_of_samples, interval=interval)

        if not hasattr(self, 'rater_stats_items'):
            self.rater_stats_df_items(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                      ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                      matrix_power=matrix_power, log_lik_tol=log_lik_tol,
                                      no_of_samples=no_of_samples, interval=interval)

        if not hasattr(self, 'person_stats_items'):
            self.person_stats_df_items(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                       ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                       matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if not hasattr(self, 'test_stats_items'):
            self.test_stats_df_items(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                     ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                     matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if format == 'xlsx':

            if filename[-5:] != '.xlsx':
                filename += '.xlsx'

            writer = pd.ExcelWriter(filename, engine='xlsxwriter')

            self.item_stats_items.to_excel(writer, sheet_name='Item statistics')
            self.threshold_stats_items.to_excel(writer, sheet_name='Threshold statistics')
            self.rater_stats_items.to_excel(writer, sheet_name='Rater statistics')
            self.person_stats_items.to_excel(writer, sheet_name='Person statistics')
            self.test_stats_items.to_excel(writer, sheet_name='Test statistics')

            writer.close()

        else:
            if filename[-4:] == '.csv':
                filename = filename[:-4]

            self.item_stats_items.to_csv(f'{filename}_item_stats.csv')
            self.threshold_stats_items.to_csv(f'{filename}_threshold_stats.csv')
            self.rater_stats_items.to_csv(f'{filename}_rater_stats.csv')
            self.person_stats_items.to_csv(f'{filename}_person_stats.csv')
            self.test_stats_items.to_csv(f'{filename}_test_stats.csv')

    def item_stats_df_thresholds(self,
                                 anchor_raters=None,
                                 full=False,
                                 zstd=False,
                                 point_measure_corr=False,
                                 dp=3,
                                 warm_corr=True,
                                 tolerance=0.00001,
                                 max_iters=100,
                                 ext_score_adjustment=0.5,
                                 method='cos',
                                 constant=0.1,
                                 matrix_power=3,
                                 log_lik_tol=0.000001,
                                 no_of_samples=100,
                                 interval=None):

        if full:
            zstd = True
            point_measure_corr = True

            if interval is None:
                interval = 0.95

        if anchor_raters is not None:
            if ((not hasattr(self, 'anchor_severites_thresholds')) or
                (self.anchor_raters_thresholds != anchor_raters)):

                self.calibrate_thresholds_anchor(anchor_raters, constant=constant, method=method,
                                                 matrix_power=matrix_power, log_lik_tol=log_lik_tol)
                self.std_errors_thresholds(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                           constant=constant, method=method, matrix_power=matrix_power,
                                           log_lik_tol=log_lik_tol)

            elif (self.anchor_item_low_thresholds is None) and (interval is not None):
                self.std_errors_thresholds(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                           constant=constant, method=method, matrix_power=matrix_power,
                                           log_lik_tol=log_lik_tol)

        else:
            if not hasattr(self, 'item_se_thresholds'):
                self.std_errors_thresholds(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                           constant=constant, method=method, matrix_power=matrix_power,
                                           log_lik_tol=log_lik_tol)

            elif (self.item_low is None) and (interval is not None):
                self.std_errors_thresholds(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                           constant=constant, method=method, matrix_power=matrix_power,
                                           log_lik_tol=log_lik_tol)

        if not hasattr(self, 'item_outfit_ms_thresholds'):
            self.item_fit_statistics_thresholds(warm_corr=warm_corr, tolerance=tolerance,
                                                max_iters=max_iters, ext_score_adjustment=ext_score_adjustment,
                                                method=method, constant=constant, matrix_power=matrix_power,
                                                log_lik_tol=log_lik_tol)

        if anchor_raters is not None:
            difficulties = self.anchor_diffs_thresholds
            std_errors = self.anchor_item_se_thresholds
            low = self.anchor_item_low_thresholds
            high = self.anchor_item_high_thresholds

        else:
            difficulties = self.diffs
            std_errors = self.item_se
            low = self.item_low
            high = self.item_high

        self.item_stats_thresholds = pd.DataFrame()

        self.item_stats_thresholds['Estimate'] = difficulties.astype(float).round(dp)
        self.item_stats_thresholds['SE'] = std_errors.astype(float).round(dp)

        if interval is not None:
            self.item_stats_thresholds[f'{round((1 - interval) * 50, 1)}%'] = low.astype(float).round(dp)
            self.item_stats_thresholds[f'{round((1 + interval) * 50, 1)}%'] = high.astype(float).round(dp)

        self.item_stats_thresholds['Count'] = self.response_counts.astype(int)
        self.item_stats_thresholds['Facility'] = self.item_facilities.astype(float).round(dp)

        self.item_stats_thresholds['Infit MS'] = self.item_infit_ms_thresholds.astype(float).round(dp)
        if zstd:
            self.item_stats_thresholds['Infit Z'] = self.item_infit_zstd_thresholds.astype(float).round(dp)

        self.item_stats_thresholds['Outfit MS'] = self.item_outfit_ms_thresholds.astype(float).round(dp)
        if zstd:
            self.item_stats_thresholds['Outfit Z'] = self.item_outfit_zstd_thresholds.astype(float).round(dp)

        if point_measure_corr:
            self.item_stats_thresholds['PM corr'] = self.point_measure_thresholds.astype(float).round(dp)
            self.item_stats_thresholds['Exp PM corr'] = self.exp_point_measure_thresholds.astype(float).round(dp)

        self.item_stats_thresholds.index = self.dataframe.columns

    def threshold_stats_df_thresholds(self,
                                      anchor_raters=None,
                                      full=False,
                                      zstd=False,
                                      disc=False,
                                      point_measure_corr=False,
                                      dp=3,
                                      warm_corr=True,
                                      tolerance=0.00001,
                                      max_iters=100,
                                      ext_score_adjustment=0.5,
                                      method='cos',
                                      constant=0.1,
                                      matrix_power=3,
                                      log_lik_tol=0.000001,
                                      no_of_samples=100,
                                      interval=None):

        if full:
            zstd = True
            disc = True
            point_measure_corr = True

            if interval is None:
                interval = 0.95

        if anchor_raters is not None:
            if (not hasattr(self, 'anchor_severites_thresholds')) or (self.anchor_raters_thresholds != anchor_raters):
                self.calibrate_thresholds_anchor(anchor_raters, constant=constant, method=method, matrix_power=matrix_power,
                                                 log_lik_tol=log_lik_tol)
                self.std_errors_thresholds(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                           constant=constant, method=method, matrix_power=matrix_power,
                                           log_lik_tol=log_lik_tol)

            elif (self.anchor_item_low_thresholds is None) and (interval is not None):
                self.std_errors_thresholds(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                           constant=constant, method=method, matrix_power=matrix_power,
                                           log_lik_tol=log_lik_tol)
                
        else:
            if not hasattr(self, 'item_se_thresholds'):
                self.std_errors_thresholds(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                           constant=constant, method=method, matrix_power=matrix_power,
                                           log_lik_tol=log_lik_tol)

            elif (self.item_low is None) and (interval is not None):
                self.std_errors_thresholds(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                           constant=constant, method=method, matrix_power=matrix_power,
                                           log_lik_tol=log_lik_tol)

        if not hasattr(self, 'threshold_outfit_ms_thresholds'):
            self.threshold_fit_statistics_thresholds(anchor_raters=anchor_raters, warm_corr=warm_corr, tolerance=tolerance,
                                                     max_iters=max_iters, ext_score_adjustment=ext_score_adjustment,
                                                     method=method, constant=constant)

        if anchor_raters is not None:
            thresholds = self.anchor_thresholds_thresholds
        else:
            thresholds = self.thresholds

        self.threshold_stats_thresholds = pd.DataFrame()

        self.threshold_stats_thresholds['Estimate'] = thresholds[1:].round(dp)

        if anchor_raters is not None:
            self.threshold_stats_thresholds['SE'] = self.anchor_threshold_se_thresholds[1:].round(dp)

        else:
            self.threshold_stats_thresholds['SE'] = self.threshold_se_thresholds[1:].round(dp)

        if interval is not None:
            if anchor_raters is not None:
                self.threshold_stats_thresholds[f'{round((1 - interval) * 50, 1)}%'] = self.anchor_threshold_low_thresholds[1:].round(dp)
                self.threshold_stats_thresholds[f'{round((1 + interval) * 50, 1)}%'] = self.anchor_threshold_high_thresholds[1:].round(dp)

            else:
                self.threshold_stats_thresholds[f'{round((1 - interval) * 50, 1)}%'] = self.threshold_low_thresholds[1:].round(dp)
                self.threshold_stats_thresholds[f'{round((1 + interval) * 50, 1)}%'] = self.threshold_high_thresholds[1:].round(dp)

        infit_ms_vector = self.threshold_infit_ms_thresholds.reset_index(drop=True)
        self.threshold_stats_thresholds['Infit MS'] = infit_ms_vector.round(dp)
        
        if zstd:
            infit_z_vector = self.threshold_infit_zstd_thresholds.reset_index(drop=True)
            self.threshold_stats_thresholds['Infit Z'] = infit_z_vector.round(dp)
            
        outfit_ms_vector = self.threshold_outfit_ms_thresholds.reset_index(drop=True)
        self.threshold_stats_thresholds['Outfit MS'] = outfit_ms_vector.round(dp)
        
        if zstd:
            outfit_z_vector = self.threshold_outfit_zstd_thresholds.reset_index(drop=True)
            self.threshold_stats_thresholds['Outfit Z'] = outfit_z_vector.round(dp)

        if disc:
            disc_vector = self.threshold_discrimination_thresholds.reset_index(drop=True)
            self.threshold_stats_thresholds['Discrim'] = disc_vector.round(dp)

        if point_measure_corr:
            pm_vector = self.threshold_point_measure_thresholds.reset_index(drop=True)
            exp_pm_vector = self.threshold_exp_point_measure_thresholds.reset_index(drop=True)
            self.threshold_stats_thresholds['PM corr'] = pm_vector.round(dp)
            self.threshold_stats_thresholds['Exp PM corr'] = exp_pm_vector.round(dp)

        self.threshold_stats_thresholds.index = [f'Threshold {threshold + 1}' for threshold in range(self.max_score)]

    def rater_stats_df_thresholds(self,
                                  anchor_raters=None,
                                  full=False,
                                  zstd=True,
                                  dp=3,
                                  warm_corr=True,
                                  tolerance=0.00001,
                                  max_iters=100,
                                  ext_score_adjustment=0.5,
                                  method='cos',
                                  constant=0.1,
                                  matrix_power=3,
                                  log_lik_tol=0.000001,
                                  no_of_samples=100,
                                  interval=None):

        if full:
            zstd=True

            if interval is None:
                interval = 0.95

        if anchor_raters is not None:
            if ((not hasattr(self, 'anchor_severites_thresholds')) or
                (self.anchor_raters_thresholds != anchor_raters)):

                self.calibrate_thresholds_anchor(anchor_raters, constant=constant, method=method,
                                                 matrix_power=matrix_power, log_lik_tol=log_lik_tol)
                self.std_errors_thresholds(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                           constant=constant, method=method, matrix_power=matrix_power,
                                           log_lik_tol=log_lik_tol)

        else:
            if not hasattr(self, 'item_se_thresholds'):
                self.std_errors_thresholds(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                           constant=constant, method=method, matrix_power=matrix_power,
                                           log_lik_tol=log_lik_tol)

        if not hasattr(self, 'rater_outfit_ms_thresholds'):
            self.rater_fit_statistics_thresholds(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                                 ext_score_adjustment=ext_score_adjustment, method=method,
                                                 constant=constant, matrix_power=matrix_power, log_lik_tol=log_lik_tol,
                                                 no_of_samples=no_of_samples, interval=interval)

        if anchor_raters is not None:
            severities = self.anchor_severities_thresholds

            se = self.anchor_rater_se_thresholds
            low = self.anchor_rater_low_thresholds
            high = self.anchor_rater_high_thresholds

        else:
            severities = self.severities_thresholds

            se = self.rater_se_thresholds
            low = self.rater_low_thresholds
            high = self.rater_high_thresholds

        self.rater_stats_thresholds = {}

        for threshold in range(self.max_score):

            threshold_stats = pd.DataFrame()

            threshold_stats['Estimate'] = np.array([severities[rater][threshold + 1] for rater in self.raters]).astype(float).round(dp)
            threshold_stats['SE'] = np.array([se[rater][threshold + 1] for rater in self.raters]).astype(float).round(dp)

            if interval is not None:
                threshold_stats[ f'{round((1 - interval) * 50, 1)}%'] = np.array([low[rater][threshold + 1]
                                                                                  for rater in self.raters]).astype(float).round(dp)
                threshold_stats[f'{round((1 + interval) * 50, 1)}%'] = np.array([high[rater][threshold + 1]
                                                                                 for rater in self.raters]).astype(float).round(dp)

            threshold_stats.index = self.raters
            self.rater_stats_thresholds[f'Threshold {threshold + 1}'] = threshold_stats.T

            if zstd:
                ov_stats_df = pd.DataFrame(index=self.raters,
                                           columns=['Count', 'Infit MS', 'Infit Z', 'Outfit MS', 'Outfit Z'])
    
                count_vector = {rater: self.dataframe.xs(rater).count().sum() for rater in self.raters}
                ov_stats_df.update({'Count': count_vector})
                ov_stats_df['Count'] = ov_stats_df['Count'].astype(int)
                            
                infit_ms_vector = self.rater_infit_ms_thresholds.astype(float)
                ov_stats_df.update({'Infit MS': infit_ms_vector.round(dp)})
                
                infit_z_vector = self.rater_infit_zstd_thresholds.astype(float)
                ov_stats_df.update({'Infit Z': infit_z_vector.round(dp)})
                            
                outfit_ms_vector = self.rater_outfit_ms_thresholds.astype(float)
                ov_stats_df.update({'Outfit MS': outfit_ms_vector.round(dp)})
                
                outfit_z_vector = self.rater_outfit_zstd_thresholds.astype(float)
                ov_stats_df.update({'Outfit Z': outfit_z_vector.round(dp)})
                
            else:
                ov_stats_df = pd.DataFrame(index=self.raters,
                                           columns=['Count', 'Infit MS', 'Outfit MS'])
        
                count_vector = {rater: self.dataframe.xs(rater).count().sum() for rater in self.raters}
                ov_stats_df.update({'Count': count_vector})
                ov_stats_df['Count'] = ov_stats_df['Count'].astype(int)
                            
                infit_ms_vector = self.rater_infit_ms_thresholds.astype(float)
                ov_stats_df.update({'Infit MS': infit_ms_vector.round(dp)})
                            
                outfit_ms_vector = self.rater_outfit_ms_thresholds.astype(float)
                ov_stats_df.update({'Outfit MS': outfit_ms_vector.round(dp)})
                
        self.rater_stats_thresholds['Overall statistics'] = ov_stats_df.T
        self.rater_stats_thresholds = pd.concat(self.rater_stats_thresholds.values(),
                                                keys=self.rater_stats_thresholds.keys()).T

    def person_stats_df_thresholds(self,
                                   anchor_raters=None,
                                   full=False,
                                   rsem=False,
                                   zstd=False,
                                   dp=3,
                                   warm_corr=True,
                                   tolerance=0.00001,
                                   max_iters=100,
                                   ext_score_adjustment=0.5,
                                   method='cos',
                                   constant=0.1,
                                   matrix_power=3,
                                   log_lik_tol=0.000001):

        '''
        Produces a person stats dataframe with raw score, ability estimate,
        CSEM and RSEM for each person.
        '''

        if not hasattr(self, 'person_outfit_ms_thresholds'):
            self.person_fit_statistics_thresholds(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                                  ext_score_adjustment=ext_score_adjustment, method=method,
                                                  constant=constant, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if full:
            rsem = True
            zstd = True

        person_stats_df = pd.DataFrame()
        person_stats_df.index = self.dataframe.index.get_level_values(1).unique()

        if anchor_raters is None:
            person_stats_df['Estimate'] = self.abils_thresholds.round(dp)

        else:
            person_stats_df['Estimate'] = self.anchor_abils_thresholds.round(dp)

        person_stats_df['CSEM'] = self.csem_vector_thresholds.round(dp)
        if rsem:
            person_stats_df['RSEM'] = self.rsem_vector_thresholds.round(dp)

        person_stats_df['Score'] = [np.nan for person in self.persons]
        score_vector = self.dataframe.unstack(level=0).sum(axis=1)
        person_stats_df.update({'Score': score_vector})
        person_stats_df['Score'] = person_stats_df['Score'].astype(int)

        person_stats_df['Max score'] = [np.nan for person in self.persons]
        max_score_vector = self.dataframe.unstack(level=0).count(axis=1) * self.max_score
        person_stats_df.update({'Max score': max_score_vector})
        person_stats_df['Max score'] = person_stats_df['Max score'].astype(int)

        person_stats_df['p'] = [np.nan for person in self.persons]
        p_vector = self.dataframe.unstack(level=0).mean(axis=1) / self.max_score
        person_stats_df.update({'p': p_vector.astype(float)})
        person_stats_df['p'] = person_stats_df['p'].round(dp)

        person_stats_df['Infit MS'] = [np.nan for person in self.persons]
        infit_ms_vector = self.person_infit_ms_thresholds.astype(float)
        person_stats_df.update({'Infit MS': infit_ms_vector.round(dp)})

        if zstd:
            person_stats_df['Infit Z'] = [np.nan for person in self.persons]
            infit_z_vector = self.person_infit_zstd_thresholds.astype(float)
            person_stats_df.update({'Infit Z': infit_z_vector.round(dp)})

        person_stats_df['Outfit MS'] = [np.nan for person in self.persons]
        outfit_ms_vector = self.person_outfit_ms_thresholds.astype(float)
        person_stats_df.update({'Outfit MS': outfit_ms_vector.round(dp)})

        if zstd:
            person_stats_df['Outfit Z'] = [np.nan for person in self.persons]
            outfit_z_vector = self.person_outfit_zstd_thresholds.astype(float)
            person_stats_df.update({'Outfit Z': outfit_z_vector.round(dp)})

        self.person_stats_thresholds = person_stats_df

    def test_stats_df_thresholds(self,
                                 dp=3,
                                 warm_corr=True,
                                 tolerance=0.00001,
                                 max_iters=100,
                                 ext_score_adjustment=0.5,
                                 method='cos',
                                 constant=0.1,
                                 matrix_power=3,
                                 log_lik_tol=0.000001):

        if not hasattr(self, 'psi_thresholds'):
            self.test_fit_statistics_thresholds(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                                ext_score_adjustment=ext_score_adjustment, method=method,
                                                constant=constant, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        self.test_stats_thresholds = pd.DataFrame()

        self.test_stats_thresholds['Items'] = [self.diffs.mean(),
                                               self.diffs.std(),
                                               self.isi_thresholds,
                                               self.item_strata_thresholds,
                                               self.item_reliability_thresholds]

        self.test_stats_thresholds['Persons'] = [self.abils_thresholds.mean(),
                                                 self.abils_thresholds.std(),
                                                 self.psi_thresholds,
                                                 self.person_strata_thresholds,
                                                 self.person_reliability_thresholds]

        self.test_stats_thresholds.index = ['Mean', 'SD', 'Separation ratio', 'Strata', 'Reliability']
        self.test_stats_thresholds = round(self.test_stats_thresholds, dp)

    def save_stats_thresholds(self,
                              filename,
                              format='csv',
                              dp=3,
                              warm_corr=True,
                              tolerance=0.00001,
                              max_iters=100,
                              ext_score_adjustment=0.5,
                              method='cos',
                              constant=0.1,
                              matrix_power=3,
                              log_lik_tol=0.000001,
                              no_of_samples=100,
                              interval=None):

        if not hasattr(self, 'item_stats_thresholds'):
            self.item_stats_df_thresholds(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                          ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                          matrix_power=matrix_power, log_lik_tol=log_lik_tol,
                                          no_of_samples=no_of_samples, interval=interval)

        if not hasattr(self, 'threshold_stats_thresholds'):
            self.threshold_stats_df_thresholds(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                               ext_score_adjustment=ext_score_adjustment, method=method,
                                               constant=constant, matrix_power=matrix_power,
                                       log_lik_tol=log_lik_tol, no_of_samples=no_of_samples, interval=interval)

        if not hasattr(self, 'rater_stats_thresholds'):
            self.rater_stats_df_thresholds(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                           ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                           matrix_power=matrix_power, log_lik_tol=log_lik_tol,
                                           no_of_samples=no_of_samples, interval=interval)

        if not hasattr(self, 'person_stats_thresholds'):
            self.person_stats_df_thresholds(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                            ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                            matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if not hasattr(self, 'test_stats_thresholds'):
            self.test_stats_df_thresholds(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                          ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                          matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if format == 'xlsx':

            if filename[-5:] != '.xlsx':
                filename += '.xlsx'

            writer = pd.ExcelWriter(filename, engine='xlsxwriter')

            self.item_stats_thresholds.to_excel(writer, sheet_name='Item statistics')
            self.threshold_stats_thresholds.to_excel(writer, sheet_name='Threshold statistics')
            self.rater_stats_thresholds.to_excel(writer, sheet_name='Rater statistics')
            self.person_stats_thresholds.to_excel(writer, sheet_name='Person statistics')
            self.test_stats_thresholds.to_excel(writer, sheet_name='Test statistics')

            writer.close()

        else:
            if filename[-4:] == '.csv':
                filename = filename[:-4]

            self.item_stats_thresholds.to_csv(f'{filename}_item_stats.csv')
            self.threshold_stats_thresholds.to_csv(f'{filename}_threshold_stats.csv')
            self.rater_stats_thresholds.to_csv(f'{filename}_rater_stats.csv')
            self.person_stats_thresholds.to_csv(f'{filename}_person_stats.csv')
            self.test_stats_thresholds.to_csv(f'{filename}_test_stats.csv')

    def item_stats_df_matrix(self,
                             anchor_raters=None,
                             full=False,
                             zstd=False,
                             point_measure_corr=False,
                             dp=3,
                             warm_corr=True,
                             tolerance=0.00001,
                             max_iters=100,
                             ext_score_adjustment=0.5,
                             method='cos',
                             constant=0.1,
                             matrix_power=3,
                             log_lik_tol=0.000001,
                             no_of_samples=100,
                             interval=None):

        if full:
            zstd = True
            point_measure_corr = True

            if interval is None:
                interval = 0.95

        if anchor_raters is not None:
            if (not hasattr(self, 'anchor_severites_matrix')) or (self.anchor_raters_matrix != anchor_raters):
                self.calibrate_matrix_anchor(anchor_raters, constant=constant, method=method, matrix_power=matrix_power,
                                             log_lik_tol=log_lik_tol)
                self.std_errors_matrix(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method, matrix_power=matrix_power,
                                       log_lik_tol=log_lik_tol)

            elif (self.anchor_item_low_matrix is None) and (interval is not None):
                self.std_errors_matrix(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method, matrix_power=matrix_power,
                                       log_lik_tol=log_lik_tol)

        else:
            if not hasattr(self, 'item_se_matrix'):
                self.std_errors_matrix(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method, matrix_power=matrix_power,
                                       log_lik_tol=log_lik_tol)

            elif (self.item_low is None) and (interval is not None):
                self.std_errors_matrix(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method, matrix_power=matrix_power,
                                       log_lik_tol=log_lik_tol)

        if not hasattr(self, 'item_outfit_ms_matrix'):
            self.item_fit_statistics_matrix(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                            ext_score_adjustment=ext_score_adjustment, method=method,
                                            constant=constant, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if anchor_raters is not None:
            difficulties = self.anchor_diffs_matrix
            std_errors = self.anchor_item_se_matrix
            low = self.anchor_item_low_matrix
            high = self.anchor_item_high_matrix

        else:
            difficulties = self.diffs
            std_errors = self.item_se
            low = self.item_low
            high = self.item_high

        self.item_stats_matrix = pd.DataFrame()

        self.item_stats_matrix['Estimate'] = difficulties.astype(float).round(dp)
        self.item_stats_matrix['SE'] = std_errors.astype(float).round(dp)

        if interval is not None:
            self.item_stats_matrix[f'{round((1 - interval) * 50, 1)}%'] = low.astype(float).round(dp)
            self.item_stats_matrix[f'{round((1 + interval) * 50, 1)}%'] = high.astype(float).round(dp)

        self.item_stats_matrix['Count'] = self.response_counts.astype(int)
        self.item_stats_matrix['Facility'] = self.item_facilities.astype(float).round(dp)

        self.item_stats_matrix['Infit MS'] = self.item_infit_ms_matrix.astype(float).round(dp)
        if zstd:
            self.item_stats_matrix['Infit Z'] = self.item_infit_zstd_matrix.astype(float).round(dp)

        self.item_stats_matrix['Outfit MS'] = self.item_outfit_ms_matrix.astype(float).round(dp)
        if zstd:
            self.item_stats_matrix['Outfit Z'] = self.item_outfit_zstd_matrix.astype(float).round(dp)

        if point_measure_corr:
            self.item_stats_matrix['PM corr'] = self.point_measure_matrix.astype(float).round(dp)
            self.item_stats_matrix['Exp PM corr'] = self.exp_point_measure_matrix.astype(float).round(dp)

        self.item_stats_matrix.index = self.dataframe.columns

    def threshold_stats_df_matrix(self,
                                  anchor_raters=None,
                                  full=False,
                                  zstd=False,
                                  disc=False,
                                  point_measure_corr=False,
                                  dp=3,
                                  warm_corr=True,
                                  tolerance=0.00001,
                                  max_iters=100,
                                  ext_score_adjustment=0.5,
                                  method='cos',
                                  constant=0.1,
                                  matrix_power=3,
                                  log_lik_tol=0.000001,
                                  no_of_samples=100,
                                  interval=None):

        if full:
            zstd = True
            disc = True
            point_measure_corr = True

            if interval is None:
                interval = 0.95

        if anchor_raters is not None:
            if (not hasattr(self, 'anchor_severites_matrix')) or (self.anchor_raters_matrix != anchor_raters):
                self.calibrate_matrix_anchor(anchor_raters, constant=constant, method=method, matrix_power=matrix_power,
                                             log_lik_tol=log_lik_tol)
                self.std_errors_matrix(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method, matrix_power=matrix_power,
                                       log_lik_tol=log_lik_tol)

            elif (self.anchor_item_low_matrix is None) and (interval is not None):
                self.std_errors_global(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method, matrix_power=matrix_power,
                                       log_lik_tol=log_lik_tol)

        else:
            if not hasattr(self, 'item_se_matrix'):
                self.std_errors_matrix(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method, matrix_power=matrix_power,
                                       log_lik_tol=log_lik_tol)

            elif (self.item_low is None) and (interval is not None):
                self.std_errors_matrix(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method, matrix_power=matrix_power,
                                       log_lik_tol=log_lik_tol)

        if not hasattr(self, 'threshold_outfit_ms_matrix'):
            self.threshold_fit_statistics_matrix(anchor_raters=anchor_raters, warm_corr=warm_corr, tolerance=tolerance,
                                                 max_iters=max_iters, ext_score_adjustment=ext_score_adjustment,
                                                 method=method, constant=constant, matrix_power=matrix_power,
                                                 log_lik_tol=log_lik_tol)

        if anchor_raters is not None:
            thresholds = self.anchor_thresholds_matrix
        else:
            thresholds = self.thresholds

        self.threshold_stats_matrix = pd.DataFrame()

        self.threshold_stats_matrix['Estimate'] = thresholds[1:].round(dp)

        if anchor_raters is not None:
            self.threshold_stats_matrix['SE'] = self.anchor_threshold_se_matrix[1:].round(dp)

        else:
            self.threshold_stats_matrix['SE'] = self.threshold_se_matrix[1:].round(dp)

        if interval is not None:
            if anchor_raters is not None:
                self.threshold_stats_matrix[f'{round((1 - interval) * 50, 1)}%'] = self.anchor_threshold_low_matrix[1:].round(dp)
                self.threshold_stats_matrix[f'{round((1 + interval) * 50, 1)}%'] = self.anchor_threshold_high_matrix[1:].round(dp)

            else:
                self.threshold_stats_matrix[f'{round((1 - interval) * 50, 1)}%'] = self.threshold_low_matrix[1:].round(dp)
                self.threshold_stats_matrix[f'{round((1 + interval) * 50, 1)}%'] = self.threshold_high_matrix[1:].round(dp)

        infit_ms_vector = self.threshold_infit_ms_matrix.reset_index(drop=True)
        self.threshold_stats_matrix['Infit MS'] = infit_ms_vector.round(dp)
        
        if zstd:
            infit_z_vector = self.threshold_infit_zstd_matrix.reset_index(drop=True)
            self.threshold_stats_matrix['Infit Z'] = infit_z_vector.round(dp)
            
        outfit_ms_vector = self.threshold_outfit_ms_matrix.reset_index(drop=True)
        self.threshold_stats_matrix['Outfit MS'] = outfit_ms_vector.round(dp)
        
        if zstd:
            outfit_z_vector = self.threshold_outfit_zstd_matrix.reset_index(drop=True)
            self.threshold_stats_matrix['Outfit Z'] = outfit_z_vector.round(dp)

        if disc:
            disc_vector = self.threshold_discrimination_matrix.reset_index(drop=True)
            self.threshold_stats_matrix['Discrim'] = disc_vector.round(dp)

        if point_measure_corr:
            pm_vector = self.threshold_point_measure_matrix.reset_index(drop=True)
            exp_pm_vector = self.threshold_exp_point_measure_matrix.reset_index(drop=True)
            self.threshold_stats_matrix['PM corr'] = pm_vector.round(dp)
            self.threshold_stats_matrix['Exp PM corr'] = exp_pm_vector.round(dp)
            
        self.threshold_stats_matrix.index = [f'Threshold {threshold + 1}' for threshold in range(self.max_score)]

    def rater_stats_df_matrix(self,
                              anchor_raters=None,
                              full=False,
                              zstd=False,
                              marginal=True,
                              dp=3,
                              warm_corr=True,
                              tolerance=0.00001,
                              max_iters=100,
                              ext_score_adjustment=0.5,
                              method='cos',
                              constant=0.1,
                              matrix_power=3,
                              log_lik_tol=0.000001,
                              no_of_samples=100,
                              interval=None):

        if full:
            zstd = True

            if interval is None:
                interval = 0.95

        if anchor_raters is not None:
            if (not hasattr(self, 'anchor_severites_matrix')) or (self.anchor_raters_matrix != anchor_raters):
                self.calibrate_matrix_anchor(anchor_raters, constant=constant, method=method, matrix_power=matrix_power,
                                             log_lik_tol=log_lik_tol)
                self.std_errors_matrix(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method, matrix_power=matrix_power,
                                       log_lik_tol=log_lik_tol)

        else:
            if not hasattr(self, 'item_se_matrix'):
                self.std_errors_matrix(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method, matrix_power=matrix_power,
                                       log_lik_tol=log_lik_tol)

        if not hasattr(self, 'rater_outfit_ms_matrix'):
            self.rater_fit_statistics_matrix(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                             ext_score_adjustment=ext_score_adjustment, method=method,
                                             constant=constant, matrix_power=matrix_power, log_lik_tol=log_lik_tol,
                                             no_of_samples=no_of_samples, interval=interval)

        if anchor_raters is not None:
            severities = self.anchor_severities_matrix
            marginal_items = self.anchor_marginal_severities_items
            marginal_thresholds = self.anchor_marginal_severities_thresholds

            se = self.anchor_rater_se_matrix
            low = self.anchor_rater_low_matrix
            high = self.anchor_rater_high_matrix

        else:
            severities = self.severities_matrix
            marginal_items = self.marginal_severities_items
            marginal_thresholds = self.marginal_severities_thresholds

            se = self.rater_se_matrix
            low = self.rater_low_matrix
            high = self.rater_high_matrix

        if marginal:
            self.rater_stats_matrix = {}

            for item in self.dataframe.columns:

                item_stats = pd.DataFrame()

                item_stats['Estimate'] = np.array([marginal_items[rater][item]
                                                   for rater in self.raters]).astype(float).round(dp)

                if anchor_raters is not None:
                    item_stats['SE'] = np.array([self.anchor_rater_se_marginal_items[rater][item]
                                                for rater in self.raters]).astype(float).round(dp)
                else:
                    item_stats['SE'] = np.array([self.rater_se_marginal_items[rater][item]
                                                for rater in self.raters]).astype(float).round(dp)

                if interval is not None:
                    if anchor_raters is not None:
                        item_stats[f'{round((1 - interval) * 50, 1)}%'] = np.array([self.anchor_rater_low_marginal_items[rater][item]
                                                                                    for rater in self.raters]).astype(float).round(dp)
                        item_stats[f'{round((1 + interval) * 50, 1)}%'] = np.array([self.anchor_rater_high_marginal_items[rater][item]
                                                                                    for rater in self.raters]).astype(float).round(dp)

                    else:
                        item_stats[f'{round((1 - interval) * 50, 1)}%'] = np.array([self.rater_low_marginal_items[rater][item]
                                                                                    for rater in self.raters]).astype(float).round(dp)
                        item_stats[f'{round((1 + interval) * 50, 1)}%'] = np.array([self.rater_high_marginal_items[rater][item]
                                                                                    for rater in self.raters]).astype(float).round(dp)

                item_stats.index = self.raters
                self.rater_stats_matrix[item] = item_stats.T

            for threshold in range(self.max_score):

                threshold_stats = pd.DataFrame()

                threshold_stats['Estimate'] = np.array([marginal_thresholds[rater][threshold + 1]
                                                        for rater in self.raters]).astype(float).round(dp)

                if anchor_raters is not None:
                    threshold_stats['SE'] = np.array([self.anchor_rater_se_marginal_thresholds[rater][threshold + 1]
                                                      for rater in self.raters]).astype(float).round(dp)
                else:
                    threshold_stats['SE'] = np.array([self.rater_se_marginal_thresholds[rater][threshold + 1]
                                                      for rater in self.raters]).astype(float).round(dp)

                if interval is not None:
                    if anchor_raters is not None:
                        threshold_stats[ f'{round((1 - interval) * 50, 1)}%'] = np.array([self.anchor_rater_low_marginal_thresholds[rater][threshold + 1]
                                                                                          for rater in self.raters]).astype(float).round(dp)
                        threshold_stats[f'{round((1 + interval) * 50, 1)}%'] = np.array([self.anchor_rater_high_marginal_thresholds[rater][threshold + 1]
                                                                                         for rater in self.raters]).astype(float).round(dp)

                    else:
                        threshold_stats[ f'{round((1 - interval) * 50, 1)}%'] = np.array([self.rater_low_marginal_thresholds[rater][threshold + 1]
                                                                                          for rater in self.raters]).astype(float).round(dp)
                        threshold_stats[f'{round((1 + interval) * 50, 1)}%'] = np.array([self.rater_high_marginal_thresholds[rater][threshold + 1]
                                                                                         for rater in self.raters]).astype(float).round(dp)

                threshold_stats.index = self.raters
                self.rater_stats_matrix[f'Threshold {threshold + 1}'] = threshold_stats.T

            if zstd:
                ov_stats_df = pd.DataFrame(index=self.raters,
                                           columns=['Count', 'Infit MS', 'Infit Z', 'Outfit MS', 'Outfit Z'])
    
                count_vector = {rater: self.dataframe.xs(rater).count().sum() for rater in self.raters}
                ov_stats_df.update({'Count': count_vector})
                ov_stats_df['Count'] = ov_stats_df['Count'].astype(int)
                            
                infit_ms_vector = self.rater_infit_ms_matrix.astype(float)
                ov_stats_df.update({'Infit MS': infit_ms_vector.round(dp)})
                
                infit_z_vector = self.rater_infit_zstd_matrix.astype(float)
                ov_stats_df.update({'Infit Z': infit_z_vector.round(dp)})
                            
                outfit_ms_vector = self.rater_outfit_ms_matrix.astype(float)
                ov_stats_df.update({'Outfit MS': outfit_ms_vector.round(dp)})
                
                outfit_z_vector = self.rater_outfit_zstd_matrix.astype(float)
                ov_stats_df.update({'Outfit Z': outfit_z_vector.round(dp)})
                
            else:
                ov_stats_df = pd.DataFrame(index=self.raters,
                                           columns=['Count', 'Infit MS', 'Outfit MS'])
        
                count_vector = {rater: self.dataframe.xs(rater).count().sum() for rater in self.raters}
                ov_stats_df.update({'Count': count_vector})
                ov_stats_df['Count'] = ov_stats_df['Count'].astype(int)
                
                infit_ms_vector = self.rater_infit_ms_matrix.astype(float)
                ov_stats_df.update({'Infit MS': infit_ms_vector.round(dp)})
                            
                outfit_ms_vector = self.rater_outfit_ms_matrix.astype(float)
                ov_stats_df.update({'Outfit MS': outfit_ms_vector.round(dp)})
        
            self.rater_stats_matrix['Overall statistics'] = ov_stats_df.T
            self.rater_stats_matrix = pd.concat(self.rater_stats_matrix.values(),
                                                keys=self.rater_stats_matrix.keys()).T

        else:
            self.rater_stats_matrix = {}

            for item in self.dataframe.columns:
                for threshold in range(self.max_score):

                    item_stats = pd.DataFrame()

                    item_stats['Estimate'] = np.array([severities[rater][item][threshold + 1]
                                                       for rater in self.raters]).round(dp)
                    item_stats['SE'] = np.array([se[rater][item][threshold + 1]
                                                 for rater in self.raters]).round(dp)

                    if interval is not None:
                        item_stats[f'{round((1 - interval) * 50, 1)}%'] = np.array([low[rater][item][threshold + 1]
                                                                                    for rater in self.raters]).round(dp)
                        item_stats[f'{round((1 + interval) * 50, 1)}%'] = np.array([high[rater][item][threshold + 1]
                                                                                    for rater in self.raters]).round(dp)

                    item_stats.index = self.raters
                    self.rater_stats_matrix[f'{item}, Threshold {threshold + 1}'] = item_stats.T

            if zstd:
                ov_stats_df = pd.DataFrame(index=self.raters,
                                           columns=['Count', 'Infit MS', 'Infit Z', 'Outfit MS', 'Outfit Z'])
    
                count_vector = {rater: self.dataframe.xs(rater).count().sum() for rater in self.raters}
                ov_stats_df.update({'Count': count_vector})
                ov_stats_df['Count'] = ov_stats_df['Count'].astype(int)
                            
                infit_ms_vector = self.rater_infit_ms_matrix.astype(float)
                ov_stats_df.update({'Infit MS': infit_ms_vector.round(dp)})
                
                infit_z_vector = self.rater_infit_zstd_matrix.astype(float)
                ov_stats_df.update({'Infit Z': infit_z_vector.round(dp)})
                            
                outfit_ms_vector = self.rater_outfit_ms_matrix.astype(float)
                ov_stats_df.update({'Outfit MS': outfit_ms_vector.round(dp)})
                
                outfit_z_vector = self.rater_outfit_zstd_matrix.astype(float)
                ov_stats_df.update({'Outfit Z': outfit_z_vector.round(dp)})
                
            else:
                ov_stats_df = pd.DataFrame(index=self.raters,
                                           columns=['Count', 'Infit MS', 'Outfit MS'])
        
                count_vector = {rater: self.dataframe.xs(rater).count().sum() for rater in self.raters}
                ov_stats_df.update({'Count': count_vector})
                ov_stats_df['Count'] = ov_stats_df['Count'].astype(int)
                
                infit_ms_vector = self.rater_infit_ms_matrix.astype(float)
                ov_stats_df.update({'Infit MS': infit_ms_vector.round(dp)})
                            
                outfit_ms_vector = self.rater_outfit_ms_matrix.astype(float)
                ov_stats_df.update({'Outfit MS': outfit_ms_vector.round(dp)})
        
            self.rater_stats_matrix['Overall statistics'] = ov_stats_df.T
            self.rater_stats_matrix = pd.concat(self.rater_stats_matrix.values(),
                                                keys=self.rater_stats_matrix.keys()).T

    def person_stats_df_matrix(self,
                               anchor_raters=None,
                               full=False,
                               rsem=False,
                               zstd=False,
                               dp=3,
                               warm_corr=True,
                               tolerance=0.00001,
                               max_iters=100,
                               ext_score_adjustment=0.5,
                               method='cos',
                               constant=0.1,
                               matrix_power=3,
                               log_lik_tol=0.000001):

        '''
        Produces a person stats dataframe with raw score, ability estimate,
        CSEM and RSEM for each person.
        '''

        if not hasattr(self, 'person_outfit_ms_matrix'):
            self.person_fit_statistics_matrix(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                              ext_score_adjustment=ext_score_adjustment, method=method,
                                              constant=constant, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if full:
            rsem = True
            zstd = True

        person_stats_df = pd.DataFrame()
        person_stats_df.index = self.dataframe.index.get_level_values(1).unique()

        if anchor_raters is None:
            person_stats_df['Estimate'] = self.abils_matrix.round(dp)

        else:
            person_stats_df['Estimate'] = self.anchor_abils_matrix.round(dp)

        person_stats_df['CSEM'] = self.csem_vector_matrix.round(dp)
        if rsem:
            person_stats_df['RSEM'] = self.rsem_vector_matrix.round(dp)

        person_stats_df['Score'] = [np.nan for person in self.persons]
        person_stats_df.update({'Score': self.dataframe.unstack(level=0).sum(axis=1)})
        person_stats_df['Score'] = person_stats_df['Score'].astype(int)

        person_stats_df['Max score'] = [np.nan for person in self.persons]
        person_stats_df.update({'Max score': self.dataframe.unstack(level=0).count(axis=1) * self.max_score})
        person_stats_df['Max score'] = person_stats_df['Max score'].astype(int)

        person_stats_df['p'] = [np.nan for person in self.persons]
        p_vector = self.dataframe.unstack(level=0).mean(axis=1) / self.max_score
        person_stats_df.update({'p': p_vector.astype(float)})
        person_stats_df['p'] = person_stats_df['p'].round(dp)

        person_stats_df['Infit MS'] = [np.nan for person in self.persons]
        infit_vector = self.person_infit_ms_matrix.astype(float)
        person_stats_df.update({'Infit MS': infit_vector.round(dp)})

        if zstd:
            person_stats_df['Infit Z'] = [np.nan for person in self.persons]
            infit_z_vector = self.person_infit_zstd_matrix.astype(float)
            person_stats_df.update({'Infit Z': infit_z_vector.round(dp)})

        person_stats_df['Outfit MS'] = [np.nan for person in self.persons]
        outfit_vector = self.person_outfit_ms_matrix.astype(float)
        person_stats_df.update({'Outfit MS': outfit_vector.round(dp)})

        if zstd:
            person_stats_df['Outfit Z'] = [np.nan for person in self.persons]
            outfit_z_vector = self.person_outfit_zstd_matrix.astype(float)
            person_stats_df.update({'Outfit Z': outfit_z_vector.round(dp)})

        self.person_stats_matrix = person_stats_df

    def test_stats_df_matrix(self,
                             dp=3,
                             warm_corr=True,
                             tolerance=0.00001,
                             max_iters=100,
                             ext_score_adjustment=0.5,
                             method='cos',
                             constant=0.1,
                             matrix_power=3,
                             log_lik_tol=0.000001):

        '''
        Produces a test statistics dataframe with raw score, ability estimate,
        CSEM and RSEM for each person.
        '''

        if not hasattr(self, 'psi_matrix'):
            self.test_fit_statistics_matrix(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                            ext_score_adjustment=ext_score_adjustment, method=method,
                                            constant=constant, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        self.test_stats_matrix = pd.DataFrame()

        self.test_stats_matrix['Items'] = [self.diffs.mean(),
                                           self.diffs.std(),
                                           self.isi_matrix,
                                           self.item_strata_matrix,
                                           self.item_reliability_matrix]

        self.test_stats_matrix['Persons'] = [self.abils_matrix.mean(),
                                             self.abils_matrix.std(),

                                             self.psi_matrix,
                                             self.person_strata_matrix,
                                             self.person_reliability_matrix]

        self.test_stats_matrix.index = ['Mean', 'SD', 'Separation ratio', 'Strata', 'Reliability']
        self.test_stats_matrix = round(self.test_stats_matrix, dp)

    def save_stats_matrix(self,
                          filename,
                          format='csv',
                          dp=3,
                          warm_corr=True,
                          tolerance=0.00001,
                          max_iters=100,
                          ext_score_adjustment=0.5,
                          method='cos',
                          constant=0.1,
                          matrix_power=3,
                          log_lik_tol=0.000001,
                          no_of_samples=100,
                          interval=None):

        if not hasattr(self, 'item_stats_matrix'):
            self.item_stats_df_matrix(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                      ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                      matrix_power=matrix_power, log_lik_tol=log_lik_tol,
                                      no_of_samples=no_of_samples, interval=interval)

        if not hasattr(self, 'threshold_stats_matrix'):
            self.threshold_stats_df_matrix(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                           ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                           matrix_power=matrix_power, log_lik_tol=log_lik_tol,
                                           no_of_samples=no_of_samples, interval=interval)

        if not hasattr(self, 'rater_stats_matrix'):
            self.rater_stats_df_matrix(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                       ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                       matrix_power=matrix_power, log_lik_tol=log_lik_tol,
                                       no_of_samples=no_of_samples, interval=interval)

        if not hasattr(self, 'person_stats_matrix'):
            self.person_stats_df_matrix(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                        ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                        matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if not hasattr(self, 'test_stats_matrix'):
            self.test_stats_df_matrix(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                      ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                      matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if format == 'xlsx':

            if filename[-5:] != '.xlsx':
                filename += '.xlsx'

            writer = pd.ExcelWriter(filename, engine='xlsxwriter')

            self.item_stats_matrix.to_excel(writer, sheet_name='Item statistics')
            self.threshold_stats_matrix.to_excel(writer, sheet_name='Threshold statistics')
            self.rater_stats_matrix.to_excel(writer, sheet_name='Rater statistics')
            self.person_stats_matrix.to_excel(writer, sheet_name='Person statistics')
            self.test_stats_matrix.to_excel(writer, sheet_name='Test statistics')

            writer.close()

        else:
            if filename[-4:] == '.csv':
                filename = filename[:-4]

            self.item_stats_matrix.to_csv(f'{filename}_item_stats.csv')
            self.threshold_stats_matrix.to_csv(f'{filename}_threshold_stats.csv')
            self.rater_stats_matrix.to_csv(f'{filename}_rater_stats.csv')
            self.person_stats_matrix.to_csv(f'{filename}_person_stats.csv')
            self.test_stats_matrix.to_csv(f'{filename}_test_stats.csv')

    def category_probability_dict_global(self,
                                         warm_corr=True,
                                         ext_scores=True,
                                         tolerance=0.00001,
                                         max_iters=100,
                                         ext_score_adjustment=0.5,
                                         method='cos',
                                         constant=0.1,
                                         matrix_power=3,
                                         log_lik_tol=0.000001,):

        if not hasattr(self, 'thresholds'):
            self.calibrate_global(constant=constant, method=method, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if not hasattr(self, 'abils_global'):
            self.person_abils_global(anchor=False, items=None, raters=None, warm_corr=warm_corr, tolerance=tolerance,
                                     max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)

        difficulties = self.diffs
        severities = self.severities_global

        scores = {rater: self.dataframe.loc[rater].sum(axis=1).astype(float)
                  for rater in self.raters}
        scores = sum(scores.values())

        person_filter = self.dataframe.notna().astype(float).replace(0, np.nan)

        if ext_scores:
            abilities = self.abils_global
            df = self.dataframe

        else:
            ext_scores = {rater: person_filter.loc[rater].sum(axis=1) * self.max_score
                          for rater in self.raters}
            ext_scores = sum(ext_scores.values())

            abilities = self.abils_global[scores > 0]
            abilities = abilities[scores < ext_scores]

            df = self.dataframe.loc[(slice(None), abilities.index), :]

        person_filter = df.notna().astype(float).replace(0, np.nan)

        c_p_df = {item: abilities - difficulties[item] for item in self.items}
        c_p_df = {rater: pd.DataFrame(c_p_df) - severities.loc[rater]
                  for rater in self.raters}
        c_p_df = pd.concat(c_p_df.values(), keys=c_p_df.keys())

        cat_prob_dict = {cat: (cat * c_p_df) - sum(self.thresholds[:cat + 1])
                         for cat in range(self.max_score + 1)}

        for cat in range(self.max_score + 1):
            cat_prob_dict[cat] = np.exp(cat_prob_dict[cat])

        den = sum(cat_prob_dict[cat] for cat in range(self.max_score + 1))

        for cat in range(self.max_score + 1):
            cat_prob_dict[cat] /= den
            cat_prob_dict[cat] *= person_filter

        self.cat_prob_dict_global = cat_prob_dict

    def category_probability_dict_items(self,
                                        warm_corr=True,
                                        ext_scores=True,
                                        tolerance=0.00001,
                                        max_iters=100,
                                        ext_score_adjustment=0.5,
                                        method='cos',
                                        constant=0.1,
                                        matrix_power=3,
                                        log_lik_tol=0.000001):

        if not hasattr(self, 'thresholds_items'):
            self.calibrate_items(constant=constant, method=method, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if not hasattr(self, 'abils_items'):
            self.person_abils_items(anchor=False, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                    ext_score_adjustment=ext_score_adjustment)

        difficulties = self.diffs
        severities = self.severities_items

        scores = {rater: self.dataframe.loc[rater].sum(axis=1).astype(float)
                  for rater in self.raters}
        scores = sum(scores.values())

        person_filter = self.dataframe.notna().astype(float).replace(0, np.nan)

        if ext_scores:
            abilities = self.abils_items
            df = self.dataframe

        else:
            ext_scores = {rater: person_filter.loc[rater].sum(axis=1) * self.max_score
                          for rater in self.raters}
            ext_scores = sum(ext_scores.values())

            abilities = self.abils_global[scores > 0]
            abilities = abilities[scores < ext_scores]

            df = self.dataframe.loc[(slice(None), abilities.index), :]

        person_filter = df.notna().astype(float).replace(0, np.nan)

        c_p_df = {rater: {item: abilities - difficulties[item] - severities[rater][item]
                          for item in self.items}
                  for rater in self.raters}
        for rater in self.raters:
            c_p_df[rater] = pd.DataFrame(c_p_df[rater])

        c_p_df = pd.concat(c_p_df.values(), keys=c_p_df.keys())

        cat_prob_dict = {cat: (cat * c_p_df) - sum(self.thresholds[:cat + 1])
                         for cat in range(self.max_score + 1)}

        for cat in range(self.max_score + 1):
            cat_prob_dict[cat] = np.exp(cat_prob_dict[cat])

        den = sum(cat_prob_dict[cat] for cat in range(self.max_score + 1))

        for cat in range(self.max_score + 1):
            cat_prob_dict[cat] /= den
            cat_prob_dict[cat] *= person_filter

        self.cat_prob_dict_items = cat_prob_dict

    def category_probability_dict_thresholds(self,
                                             warm_corr=True,
                                             ext_scores=False,
                                             tolerance=0.00001,
                                             max_iters=100,
                                             ext_score_adjustment=0.5,
                                             method='cos',
                                             constant=0.1,
                                             matrix_power=3,
                                             log_lik_tol=0.000001):

        if not hasattr(self, 'thresholds_thresholds'):
            self.calibrate_thresholds(constant=constant, method=method, matrix_power=matrix_power,
                                      log_lik_tol=log_lik_tol)

        if not hasattr(self, 'abils_thresholds'):
            self.person_abils_thresholds(anchor=False, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                         ext_score_adjustment=ext_score_adjustment)

        difficulties = self.diffs
        severities = self.severities_thresholds

        scores = {rater: self.dataframe.loc[rater].sum(axis=1).astype(float)
                  for rater in self.raters}
        scores = sum(scores.values())

        person_filter = self.dataframe.notna().astype(float).replace(0, np.nan)

        if ext_scores:
            abilities = self.abils_thresholds
            df = self.dataframe

        else:
            ext_scores = {rater: person_filter.loc[rater].sum(axis=1) * self.max_score
                          for rater in self.raters}
            ext_scores = sum(ext_scores.values())

            abilities = self.abils_thresholds[scores > 0]
            abilities = abilities[scores < ext_scores]

            df = self.dataframe.loc[(slice(None), abilities.index), :]

        person_filter = df.notna().astype(float).replace(0, np.nan)

        c_p_df = {item: abilities - difficulties.loc[item]
                  for item in self.items}
        c_p_df = pd.DataFrame(c_p_df)

        cat_probs = {cat: {rater: (cat * c_p_df - sum(self.thresholds[:cat + 1]) - sum(severities[rater][:cat + 1]))
                           for rater in self.raters}
                     for cat in range(self.max_score + 1)}

        for cat in range(self.max_score + 1):
            cat_probs[cat] = pd.concat(cat_probs[cat].values(), keys=cat_probs[cat].keys())
            cat_probs[cat] = np.exp(cat_probs[cat])

        den = sum(cat_probs[cat] for cat in range(self.max_score + 1))

        for cat in range(self.max_score + 1):
            cat_probs[cat] /= den
            cat_probs[cat] *= person_filter

            self.cat_prob_dict_thresholds = cat_probs

    def category_probability_dict_matrix(self,
                                         warm_corr=True,
                                         ext_scores=True,
                                         tolerance=0.00001,
                                         max_iters=100,
                                         ext_score_adjustment=0.5,
                                         method='cos',
                                         constant=0.1,
                                         matrix_power=3,
                                         log_lik_tol=0.000001):

        if not hasattr(self, 'thresholds_matrix'):
            self.calibrate_matrix(constant=constant, method=method, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if not hasattr(self, 'abils_matrix'):
            self.person_abils_matrix(anchor=False, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                     ext_score_adjustment=ext_score_adjustment)

        difficulties = self.diffs
        severities = self.severities_matrix

        scores = {rater: self.dataframe.loc[rater].sum(axis=1).astype(float)
                  for rater in self.raters}
        scores = sum(scores.values())

        person_filter = self.dataframe.notna().astype(float).replace(0, np.nan)

        if ext_scores:
            abilities = self.abils_matrix
            df = self.dataframe

        else:
            ext_scores = {rater: person_filter.loc[rater].sum(axis=1) * self.max_score
                          for rater in self.raters}
            ext_scores = sum(ext_scores.values())

            abilities = self.abils_matrix[scores > 0]
            abilities = abilities[scores < ext_scores]

            df = self.dataframe.loc[(slice(None), abilities.index), :]

        person_filter = df.notna().astype(float).replace(0, np.nan)

        c_p_df = {item: abilities - difficulties.loc[item]
                  for item in self.items}
        c_p_df = pd.DataFrame(c_p_df)

        self.cat_prob_dict_matrix = {cat: {rater: (cat * c_p_df -
                                                   sum(self.thresholds[:cat + 1]))
                                           for rater in self.raters}
                                     for cat in range(self.max_score + 1)}

        for cat in range(self.max_score + 1):
            for rater in self.raters:
                for item in self.items:
                    self.cat_prob_dict_matrix[cat][rater][item] -= sum(severities[rater][item][:cat + 1])

        for cat in range(self.max_score + 1):
            self.cat_prob_dict_matrix[cat] = pd.concat(self.cat_prob_dict_matrix[cat].values(),
                                                       keys=self.cat_prob_dict_matrix[cat].keys())
            self.cat_prob_dict_matrix[cat] = np.exp(self.cat_prob_dict_matrix[cat])

        den = sum(self.cat_prob_dict_matrix[cat] for cat in range(self.max_score + 1))

        for cat in range(self.max_score + 1):
            self.cat_prob_dict_matrix[cat] /= den
            self.cat_prob_dict_matrix[cat] *= person_filter

    def fit_matrices(self,
                     cat_prob_dict):

        '''
        Create matrices of expected scores, variances, kurtosis,
        residuals etc. to generate fit statistics
        '''

        exp_score_df = sum(cat * df for cat, df in cat_prob_dict.items())
        info_df = sum(df * (cat - exp_score_df) ** 2 for cat, df in cat_prob_dict.items())
        kurtosis_df = sum(cat_df * (cat - exp_score_df) ** 4 for cat, cat_df in cat_prob_dict.items())

        residual_df = self.dataframe.loc[exp_score_df.index] - exp_score_df
        std_residual_df = residual_df / (info_df ** 0.5)

        return exp_score_df, info_df, kurtosis_df, residual_df, std_residual_df

    def fit_matrices_global(self,
                            warm_corr=True,
                            ext_scores=True,
                            tolerance=0.00001,
                            max_iters=100,
                            ext_score_adjustment=0.5,
                            method='cos',
                            constant=0.1,
                            matrix_power=3,
                            log_lik_tol=0.000001):

        if not hasattr(self, 'cat_prob_dict_global'):
            self.category_probability_dict_global(warm_corr=warm_corr, ext_scores=ext_scores, tolerance=tolerance,
                                                  max_iters=max_iters, ext_score_adjustment=ext_score_adjustment,
                                                  method=method, constant=constant, matrix_power=matrix_power,
                                                  log_lik_tol=log_lik_tol)

        (self.exp_score_df_global,
         self.info_df_global,
         self.kurtosis_df_global,
         self.residual_df_global,
         self.std_residual_df_global) = self.fit_matrices(self.cat_prob_dict_global)

    def fit_matrices_items(self,
                           warm_corr=True,
                           ext_scores=True,
                           tolerance=0.00001,
                           max_iters=100,
                           ext_score_adjustment=0.5,
                           method='cos',
                           constant=0.1,
                           matrix_power=3,
                           log_lik_tol=0.000001):

        if not hasattr(self, 'cat_prob_dict_items'):
            self.category_probability_dict_items(warm_corr=warm_corr, ext_scores=ext_scores, tolerance=tolerance,
                                                 max_iters=max_iters, ext_score_adjustment=ext_score_adjustment,
                                                 method=method, constant=constant, matrix_power=matrix_power,
                                                 log_lik_tol=log_lik_tol)

        (self.exp_score_df_items,
         self.info_df_items,
         self.kurtosis_df_items,
         self.residual_df_items,
         self.std_residual_df_items) = self.fit_matrices(self.cat_prob_dict_items)

    def fit_matrices_thresholds(self,
                                warm_corr=True,
                                ext_scores=True,
                                tolerance=0.00001,
                                max_iters=100,
                                ext_score_adjustment=0.5,
                                method='cos',
                                constant=0.1,
                                matrix_power=3,
                                log_lik_tol=0.000001):

        if not hasattr(self, 'cat_prob_dict_thresholds'):
            self.category_probability_dict_thresholds(warm_corr=warm_corr, ext_scores=ext_scores, tolerance=tolerance,
                                                      max_iters=max_iters, ext_score_adjustment=ext_score_adjustment,
                                                      method=method, constant=constant, matrix_power=matrix_power,
                                                      log_lik_tol=log_lik_tol)

        (self.exp_score_df_thresholds,
         self.info_df_thresholds,
         self.kurtosis_df_thresholds,
         self.residual_df_thresholds,
         self.std_residual_df_thresholds) = self.fit_matrices(self.cat_prob_dict_thresholds)

    def fit_matrices_matrix(self,
                            warm_corr=True,
                            ext_scores=True,
                            tolerance=0.00001,
                            max_iters=100,
                            ext_score_adjustment=0.5,
                            method='cos',
                            constant=0.1,
                            matrix_power=3,
                            log_lik_tol=0.000001):

        if not hasattr(self, 'cat_prob_dict_matrix'):
            self.category_probability_dict_matrix(warm_corr=warm_corr, ext_scores=ext_scores, tolerance=tolerance,
                                                  max_iters=max_iters, ext_score_adjustment=ext_score_adjustment,
                                                  method=method, constant=constant, matrix_power=matrix_power,
                                                  log_lik_tol=log_lik_tol)

        (self.exp_score_df_matrix,
         self.info_df_matrix,
         self.kurtosis_df_matrix,
         self.residual_df_matrix,
         self.std_residual_df_matrix) = self.fit_matrices(self.cat_prob_dict_matrix)

    def item_fit_statistics(self,
                            exp_score_df,
                            info_df,
                            kurtosis_df,
                            residual_df,
                            std_residual_df,
                            abilities):

        '''
        Item fit statistics
        '''

        scores = self.dataframe.sum(axis=1)
        max_scores = self.dataframe.count(axis=1) * self.max_score
        item_count = self.dataframe[(scores > 0) & (scores < max_scores)].count(axis=0)
        self.response_counts = self.dataframe.count(axis=0)

        item_outfit_ms = (std_residual_df ** 2).mean()
        item_outfit_zstd = ((item_outfit_ms ** (1/3)) - 1 + (2 / (9 * item_count))) / ((2 / (9 * item_count)) ** 0.5)

        item_infit_ms = (residual_df ** 2).sum() / info_df.sum()
        item_infit_zstd = ((item_infit_ms ** (1/3)) - 1 + (2 / (9 * item_count))) / ((2 / (9 * item_count)) ** 0.5)

        self.item_facilities = self.dataframe.mean(axis=0) / self.max_score

        #abils_by_rater = pd.DataFrame()
        #for rater in self.raters:
            #abils_by_rater[rater] = abilities
        abils_by_rater = {rater: abilities for rater in self.raters}
        #abils_by_rater = pd.DataFrame(abils_by_rater)
        abils_by_rater = pd.concat(abils_by_rater.values(), keys=abils_by_rater.keys())
        abils_by_rater.index.names = self.dataframe.index.names
        #self.abils_by_rater = abils_by_rater

        item_point_measure, item_exp_point_measure = self.pt_meas(abils_by_rater, exp_score_df, info_df)

        return (item_outfit_ms,
                item_outfit_zstd,
                item_infit_ms,
                item_infit_zstd,
                item_point_measure,
                item_exp_point_measure)

    def item_fit_statistics_global(self,
                                   warm_corr=True,
                                   ext_scores=True,
                                   tolerance=0.00001,
                                   max_iters=100,
                                   ext_score_adjustment=0.5,
                                   method='cos',
                                   constant=0.1,
                                   matrix_power=3,
                                   log_lik_tol=0.000001):

        if not hasattr(self, 'exp_score_df_global'):
            self.fit_matrices_global(warm_corr=warm_corr, ext_scores=ext_scores, tolerance=tolerance,
                                     max_iters=max_iters, ext_score_adjustment=ext_score_adjustment, method=method,
                                     constant=constant, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        (self.item_outfit_ms_global,
         self.item_outfit_zstd_global,
         self.item_infit_ms_global,
         self.item_infit_zstd_global,
         self.point_measure_global,
         self.exp_point_measure_global) = self.item_fit_statistics(self.exp_score_df_global, self.info_df_global,
                                                                   self.kurtosis_df_global, self.residual_df_global,
                                                                   self.std_residual_df_global, self.abils_global)

    def item_fit_statistics_items(self,
                                  warm_corr=True,
                                  tolerance=0.00001,
                                  max_iters=100,
                                  ext_score_adjustment=0.5,
                                  method='cos',
                                  constant=0.1,
                                  matrix_power=3,
                                  log_lik_tol=0.000001):

        if not hasattr(self, 'exp_score_df_items'):
            self.fit_matrices_items(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                    ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                    matrix_power=matrix_power, log_lik_tol=log_lik_tol)

            (self.item_outfit_ms_items,
             self.item_outfit_zstd_items,
             self.item_infit_ms_items,
             self.item_infit_zstd_items,
             self.point_measure_items,
             self.exp_point_measure_items) = self.item_fit_statistics(self.exp_score_df_items, self.info_df_items,
                                                                      self.kurtosis_df_items, self.residual_df_items,
                                                                      self.std_residual_df_items, self.abils_items)

    def item_fit_statistics_thresholds(self,
                                       warm_corr=True,
                                       tolerance=0.00001,
                                       max_iters=100,
                                       ext_score_adjustment=0.5,
                                       method='cos',
                                       constant=0.1,
                                       matrix_power=3,
                                       log_lik_tol=0.000001):

        if not hasattr(self, 'exp_score_df_thresholds'):
            self.fit_matrices_thresholds(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                         ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                         matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        (self.item_outfit_ms_thresholds,
         self.item_outfit_zstd_thresholds,
         self.item_infit_ms_thresholds,
         self.item_infit_zstd_thresholds,
         self.point_measure_thresholds,
         self.exp_point_measure_thresholds) = self.item_fit_statistics(self.exp_score_df_thresholds,
                                                                       self.info_df_thresholds,
                                                                       self.kurtosis_df_thresholds,
                                                                       self.residual_df_thresholds,
                                                                       self.std_residual_df_thresholds,
                                                                       self.abils_thresholds)

    def item_fit_statistics_matrix(self,
                                   warm_corr=True,
                                   tolerance=0.00001,
                                   max_iters=100,
                                   ext_score_adjustment=0.5,
                                   method='cos',
                                   constant=0.1,
                                   matrix_power=3,
                                   log_lik_tol=0.000001):

        if not hasattr(self, 'exp_score_df_matrix'):
            self.fit_matrices_matrix(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                     ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                     matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        (self.item_outfit_ms_matrix,
         self.item_outfit_zstd_matrix,
         self.item_infit_ms_matrix,
         self.item_infit_zstd_matrix,
         self.point_measure_matrix,
         self.exp_point_measure_matrix) = self.item_fit_statistics(self.exp_score_df_matrix, self.info_df_matrix,
                                                                   self.kurtosis_df_matrix, self.residual_df_matrix,
                                                                   self.std_residual_df_matrix, self.abils_matrix)

    def item_res_corr_analysis(self,
                               std_residual_df):

        item_residual_correlations = std_residual_df.corr(numeric_only=False)

        pca = PCA()

        try:
            pca.fit(item_residual_correlations)

            item_eigenvectors = pd.DataFrame(pca.components_)
            item_eigenvectors.columns = [f'Eigenvector {pc + 1}' for pc in range(self.no_of_items)]

            item_eigenvalues = pca.explained_variance_
            item_eigenvalues = pd.DataFrame(item_eigenvalues)
            item_eigenvalues.index = [f'PC {pc + 1}' for pc in range(self.no_of_items)]
            item_eigenvalues.columns = ['Eigenvalue']

            item_variance_explained = pd.DataFrame(pca.explained_variance_ratio_)
            item_variance_explained.index = [f'PC {pc + 1}' for pc in range(self.no_of_items)]
            item_variance_explained.columns = ['Variance explained']

            item_loadings = item_eigenvectors.T * (pca.explained_variance_ ** 0.5)
            item_loadings = pd.DataFrame(item_loadings)
            item_loadings.columns = [f'PC {pc + 1}' for pc in range(self.no_of_items)]
            item_loadings.index = [item for item in self.dataframe.columns]

        except Exception:
            print('PCA of item residuals failed')

            item_eigenvectors = None
            item_eigenvalues = None
            item_variance_explained = None
            item_loadings = None

        return (item_residual_correlations,
                item_eigenvectors,
                item_eigenvalues,
                item_variance_explained,
                item_loadings)

    def item_res_corr_analysis_global(self,
                                      warm_corr=True,
                                      tolerance=0.00001,
                                      max_iters=100,
                                      ext_score_adjustment=0.5,
                                      constant=0.1,
                                      method='cos',
                                      matrix_power=3,
                                      log_lik_tol=0.000001,
                                      no_of_samples=100,
                                      interval=None):

        if not hasattr(self, 'std_residual_df_global'):
            self.fit_statistics_global(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                       ext_score_adjustment=ext_score_adjustment, constant=constant, method=method,
                                       matrix_power=matrix_power, log_lik_tol=log_lik_tol, no_of_samples=no_of_samples,
                                       interval=interval)

        (self.item_residual_correlations_global,
         self.item_eigenvectors_global,
         self.item_eigenvalues_global,
         self.item_variance_explained_global,
         self.item_loadings_global) = self.item_res_corr_analysis(self.std_residual_df_global)

    def item_res_corr_analysis_items(self,
                                     warm_corr=True,
                                     tolerance=0.00001,
                                     max_iters=100,
                                     ext_score_adjustment=0.5,
                                     constant=0.1,
                                     method='cos',
                                     matrix_power=3,
                                     log_lik_tol=0.000001,
                                     no_of_samples=100,
                                     interval=None):

        if not hasattr(self, 'std_residual_df_items'):
            self.fit_statistics_items(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                      ext_score_adjustment=ext_score_adjustment, constant=constant, method=method,
                                      matrix_power=matrix_power, log_lik_tol=log_lik_tol, no_of_samples=no_of_samples,
                                      interval=interval)

        (self.item_residual_correlations_items,
         self.item_eigenvectors_items,
         self.item_eigenvalues_items,
         self.item_variance_explained_items,
         self.item_loadings_items) = self.item_res_corr_analysis(self.std_residual_df_items)

    def item_res_corr_analysis_thresholds(self,
                                          warm_corr=True,
                                          tolerance=0.00001,
                                          max_iters=100,
                                          ext_score_adjustment=0.5,
                                          constant=0.1,
                                          method='cos',
                                          matrix_power=3,
                                          log_lik_tol=0.000001,
                                          no_of_samples=100,
                                          interval=None):

        if not hasattr(self, 'std_residual_df_thresholds'):
            self.fit_statistics_thresholds(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                           ext_score_adjustment=ext_score_adjustment, constant=constant, method=method,
                                           matrix_power=matrix_power, log_lik_tol=log_lik_tol,
                                           no_of_samples=no_of_samples, interval=interval)

        (self.item_residual_correlations_thresholds,
         self.item_eigenvectors_thresholds,
         self.item_eigenvalues_thresholds,
         self.item_variance_explained_thresholds,
         self.item_loadings_thresholds) = self.item_res_corr_analysis(self.std_residual_df_thresholds)

    def item_res_corr_analysis_matrix(self,
                                      warm_corr=True,
                                      tolerance=0.00001,
                                      max_iters=100,
                                      ext_score_adjustment=0.5,
                                      constant=0.1,
                                      method='cos',
                                      matrix_power=3,
                                      log_lik_tol=0.000001,
                                      no_of_samples=100,
                                      interval=None):

        if not hasattr(self, 'std_residual_df_matrix'):
            self.fit_statistics_matrix(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                       ext_score_adjustment=ext_score_adjustment, constant=constant, method=method,
                                       matrix_power=matrix_power, log_lik_tol=log_lik_tol, no_of_samples=no_of_samples,
                                       interval=interval)

        (self.item_residual_correlations_matrix,
         self.item_eigenvectors_matrix,
         self.item_eigenvalues_matrix,
         self.item_variance_explained_matrix,
         self.item_loadings_matrix) = self.item_res_corr_analysis(self.std_residual_df_matrix)

    def threshold_fit_statistics(self,
                                 abilities,
                                 diff_df_dict):

        '''
        Threshold fit statistics
        '''

        basic_abils_df = [[abilities[person] for item in self.dataframe.columns]
                           for person in self.persons]
        basic_abils_df = pd.DataFrame(basic_abils_df)
        basic_abils_df.index = self.persons
        basic_abils_df.columns = self.dataframe.columns

        abil_df = {rater: basic_abils_df for rater in self.raters}
        abil_df = pd.concat(abil_df.values(), keys=abil_df.keys())

        dich_thresh = {}
        for threshold in range(self.max_score):
            dich_thresh[threshold + 1] = self.dataframe.where(self.dataframe.isin([threshold, threshold + 1]), np.nan)
            dich_thresh[threshold + 1] -= threshold
            dich_thresh[threshold + 1].index.names = self.dataframe.index.names

        dich_thresh_exp = {}
        dich_thresh_var = {}
        dich_thresh_kur = {}
        dich_residuals = {}
        dich_std_residuals = {}

        dich_thresh_count = {threshold + 1: (dich_thresh[threshold + 1] == dich_thresh[threshold + 1]).sum().sum()
                             for threshold in range(self.max_score)}

        scores = self.dataframe.sum(axis=1)
        max_scores = self.dataframe.count(axis=1) * self.max_score

        for threshold in range(self.max_score):
            missing_mask = dich_thresh[threshold + 1].notna().astype(float).replace(0, np.nan)
            missing_mask = missing_mask.loc[(scores > 0) & (scores < max_scores)]
            missing_mask.index.names = self.dataframe.index.names

            dich_thresh_exp[threshold + 1] = 1 / (1 + np.exp(diff_df_dict[threshold + 1] - abil_df))
            dich_thresh_exp[threshold + 1] = dich_thresh_exp[threshold + 1].loc[(scores > 0) & (scores < max_scores)]
            dich_thresh_exp[threshold + 1].index.names = self.dataframe.index.names
            dich_thresh_exp[threshold + 1] *= missing_mask

            dich_thresh_var[threshold + 1] = dich_thresh_exp[threshold + 1] * (1 - dich_thresh_exp[threshold + 1])
            dich_thresh_var[threshold + 1] = dich_thresh_var[threshold + 1].loc[(scores > 0) & (scores < max_scores)]
            dich_thresh_var[threshold + 1].index.names = self.dataframe.index.names
            dich_thresh_var[threshold + 1] *= missing_mask

            dich_thresh_kur[threshold + 1] = (
                    ((-dich_thresh_exp[threshold + 1]) ** 4) * (1 - dich_thresh_exp[threshold + 1]) +
                    ((1 - dich_thresh_exp[threshold + 1]) ** 4) * dich_thresh_exp[threshold + 1])
            dich_thresh_kur[threshold + 1] = dich_thresh_kur[threshold + 1].loc[(scores > 0) & (scores < max_scores)]
            dich_thresh_kur[threshold + 1].index.names = self.dataframe.index.names
            dich_thresh_kur[threshold + 1] *= missing_mask

            dich_thresh[threshold + 1] = dich_thresh[threshold + 1].loc[(scores > 0) & (scores < max_scores)]

            dich_residuals[threshold + 1] = dich_thresh[threshold + 1] - dich_thresh_exp[threshold + 1]
            dich_residuals[threshold + 1] = dich_residuals[threshold + 1].loc[(scores > 0) & (scores < max_scores)]

            dich_std_residuals[threshold + 1] = (dich_residuals[threshold + 1] /
                                                 (dich_thresh_var[threshold + 1]) ** 0.5)

        threshold_outfit_ms = {threshold + 1: ((dich_std_residuals[threshold + 1] ** 2).sum().sum() /
                                               dich_thresh[threshold + 1].count().sum())
                               for threshold in range(self.max_score)}
        threshold_outfit_ms = pd.Series(threshold_outfit_ms)

        threshold_infit_ms = {threshold + 1: (dich_residuals[threshold + 1] ** 2).sum().sum() /
                                             dich_thresh_var[threshold + 1].sum().sum()
                              for threshold in range(self.max_score)}
        threshold_infit_ms = pd.Series(threshold_infit_ms)

        threshold_outfit_q = {threshold + 1: (((dich_thresh_kur[threshold + 1] /
                                                (dich_thresh_var[threshold + 1] ** 2)) /
                                               (dich_thresh_count[threshold + 1] ** 2)).sum().sum() -
                                              (1 / dich_thresh_count[threshold + 1]))
                              for threshold in range(self.max_score)}
        threshold_outfit_q = pd.Series(threshold_outfit_q)
        threshold_outfit_q = threshold_outfit_q ** 0.5

        threshold_outfit_zstd = (((threshold_outfit_ms ** (1/3)) - 1) *
                                 (3 / threshold_outfit_q) + (threshold_outfit_q / 3))

        threshold_infit_q = {threshold + 1: ((dich_thresh_kur[threshold + 1] -
                                              dich_thresh_var[threshold + 1] ** 2).sum().sum() /
                                             (dich_thresh_var[threshold + 1].sum().sum() ** 2))
                             for threshold in range(self.max_score)}
        threshold_infit_q = pd.Series(threshold_infit_q)
        threshold_infit_q = threshold_infit_q ** 0.5

        threshold_infit_zstd = (((threshold_infit_ms ** (1/3)) - 1) *
                                (3 / threshold_infit_q) + (threshold_infit_q / 3))

        dich_facilities = {threshold + 1: dich_thresh[threshold + 1].copy().mean(axis=0)
                           for threshold in range(self.max_score)}

        abil_deviation = abilities.copy() - abilities.mean()
        abil_deviation = {rater: abil_deviation for rater in self.raters}
        abil_deviation = pd.concat(abil_deviation.values(), keys=abil_deviation.keys())
        abil_deviation = abil_deviation.loc[(scores > 0) & (scores < max_scores)]

        point_measure_dict = {threshold + 1: dich_thresh[threshold + 1].copy()
                              for threshold in range(self.max_score)}

        for threshold in range(self.max_score):
            for item in self.dataframe.columns:
                point_measure_dict[threshold + 1][item] -= dich_facilities[threshold + 1][item]

        point_measure_nums = {}
        for threshold in range(self.max_score):
            pm_num = point_measure_dict[threshold + 1].copy()

            for item in self.dataframe.columns:
                pm_num[item] *= abil_deviation.values

            point_measure_nums[threshold + 1] = pm_num.sum().sum()

        point_measure_nums = pd.Series(point_measure_nums)

        point_measure_dens = {threshold + 1: ((point_measure_dict[threshold + 1] ** 2).sum().sum() *
                                              (abil_deviation ** 2).sum()) ** 0.5
                              for threshold in range(self.max_score)}
        point_measure_dens = pd.Series(point_measure_dens)

        threshold_point_measure = point_measure_nums / point_measure_dens

        threshold_exp_pm_dict = {threshold + 1: dich_thresh_exp[threshold + 1] -
                                                (dich_thresh_exp[threshold + 1].sum().sum() /
                                                 dich_thresh_exp[threshold + 1].count().sum())
                                 for threshold in range(self.max_score)}

        threshold_exp_pm_num = {threshold + 1: threshold_exp_pm_dict[threshold + 1].copy()
                                for threshold in range(self.max_score)}

        for threshold in range(self.max_score):
            for item in self.dataframe.columns:
                threshold_exp_pm_num[threshold + 1][item] *= abil_deviation.values

        threshold_exp_pm_num = {threshold + 1: threshold_exp_pm_num[threshold + 1].sum().sum()
                                for threshold in range(self.max_score)}

        threshold_exp_pm_num = pd.Series(threshold_exp_pm_num)

        threshold_exp_pm_den = {threshold + 1: ((threshold_exp_pm_dict[threshold + 1] ** 2) +
                                                dich_thresh_var[threshold + 1]).sum().sum()
                                for threshold in range(self.max_score)}
        threshold_exp_pm_den = pd.Series(threshold_exp_pm_den)
        threshold_exp_pm_den *= (abil_deviation ** 2).sum()
        threshold_exp_pm_den = threshold_exp_pm_den ** 0.5

        threshold_exp_point_measure = threshold_exp_pm_num / threshold_exp_pm_den

        threshold_rmsr = {threshold + 1: (((dich_residuals[threshold + 1] ** 2).sum().sum() /
                                          dich_residuals[threshold + 1].count().sum())) ** 0.5
                          for threshold in range(self.max_score)}
        threshold_rmsr = pd.Series(threshold_rmsr)

        differences = {threshold + 1: abil_df - diff_df_dict[threshold + 1]
                       for threshold in range(self.max_score)}

        for threshold in range(self.max_score):
            differences[threshold + 1] = differences[threshold + 1].filter(items=dich_residuals[threshold + 1].index,
                                                                           axis=0)
            differences[threshold + 1].index.names = self.dataframe.index.names

        nums = {threshold + 1: (differences[threshold + 1] * dich_residuals[threshold + 1]).sum().sum()
                for threshold in range(self.max_score)}
        nums = pd.Series(nums)

        dens = {threshold + 1: (dich_thresh_var[threshold + 1] * (differences[threshold + 1] ** 2)).sum().sum()
                for threshold in range(self.max_score)}
        dens = pd.Series(dens)

        threshold_discrimination = 1 + nums / dens

        return (threshold_outfit_ms,
                threshold_outfit_zstd,
                threshold_infit_ms,
                threshold_infit_zstd,
                threshold_point_measure,
                threshold_exp_point_measure,
                threshold_discrimination)

    def threshold_fit_statistics_global(self,
                                        anchor_raters=None,
                                        warm_corr=True,
                                        tolerance=0.00001,
                                        max_iters=100,
                                        ext_score_adjustment=0.5,
                                        method='cos',
                                        constant=0.1,
                                        matrix_power=3,
                                        log_lik_tol=0.000001):

        if not hasattr(self, 'abils_global'):
            self.person_abils_global(anchor=False, warm_corr=warm_corr, tolerance=tolerance,
                                     max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)

        if anchor_raters is not None:
            if not hasattr(self, 'anchor_thresholds_global'):
                self.calibrate_global_anchor(anchor_raters, constant=constant, method=method, matrix_power=matrix_power,
                                             log_lik_tol=log_lik_tol)

                self.person_abils_global(anchor=False, warm_corr=warm_corr, tolerance=tolerance,
                                         max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)

        if anchor_raters is not None:
            difficulties = self.anchor_diffs_global
            thresholds = self.anchor_thresholds_global
            severities = self.anchor_severities_global

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_global

        diff_df_dict = {}

        for threshold in range(self.max_score):
            diff_df_dict[threshold + 1] = {rater: [difficulties + thresholds[threshold + 1] + severities[rater]
                                                   for person in self.persons]
                                           for rater in self.raters}

            for rater in self.raters:
                diff_df_dict[threshold + 1][rater] = pd.DataFrame(diff_df_dict[threshold + 1][rater])
                diff_df_dict[threshold + 1][rater].index = self.persons
                diff_df_dict[threshold + 1][rater].columns = self.dataframe.columns

            diff_df_dict[threshold + 1] = pd.concat(diff_df_dict[threshold + 1].values(),
                                                    keys=diff_df_dict[threshold + 1].keys())

        (self.threshold_outfit_ms_global,
         self.threshold_outfit_zstd_global,
         self.threshold_infit_ms_global,
         self.threshold_infit_zstd_global,
         self.threshold_point_measure_global,
         self.threshold_exp_point_measure_global,
         self.threshold_discrimination_global) = self.threshold_fit_statistics(self.abils_global, diff_df_dict)

    def threshold_fit_statistics_items(self,
                                       anchor_raters=None,
                                       warm_corr=True,
                                       tolerance=0.00001,
                                       max_iters=100,
                                       ext_score_adjustment=0.5,
                                       method='cos',
                                       constant=0.1,
                                       matrix_power=3,
                                       log_lik_tol=0.000001):

        if not hasattr(self, 'abils_items'):
            self.person_abils_items(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                    ext_score_adjustment=ext_score_adjustment)

        if anchor_raters is not None:
            if not hasattr(self, 'anchor_thresholds_items'):
                self.calibrateitems_anchor(anchor_raters, constant=constant, method=method, matrix_power=matrix_power,
                                           log_lik_tol=log_lik_tol)

                self.person_abils_items(anchor_raters=anchor_raters, warm_corr=warm_corr, tolerance=tolerance,
                                        max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)

        if anchor_raters is not None:
            difficulties = self.anchor_diffs_items
            thresholds = self.anchor_thresholds_items
            severities = self.anchor_severities_items

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_items

        diff_df_dict = {}

        for threshold in range(self.max_score):
            diff_df_dict[threshold + 1] = {rater: [difficulties + thresholds[threshold + 1] +
                                                   pd.Series(severities[rater])
                                                   for person in self.persons]
                                           for rater in self.raters}

            for rater in self.raters:
                diff_df_dict[threshold + 1][rater] = pd.DataFrame(diff_df_dict[threshold + 1][rater])
                diff_df_dict[threshold + 1][rater].index = self.persons
                diff_df_dict[threshold + 1][rater].columns = self.dataframe.columns

            diff_df_dict[threshold + 1] = pd.concat(diff_df_dict[threshold + 1].values(),
                                                    keys=diff_df_dict[threshold + 1].keys())

        (self.threshold_outfit_ms_items,
         self.threshold_outfit_zstd_items,
         self.threshold_infit_ms_items,
         self.threshold_infit_zstd_items,
         self.threshold_point_measure_items,
         self.threshold_exp_point_measure_items,
         self.threshold_discrimination_items) = self.threshold_fit_statistics(self.abils_items, diff_df_dict)

    def threshold_fit_statistics_thresholds(self,
                                            anchor_raters=None,
                                            warm_corr=True,
                                            tolerance=0.00001,
                                            max_iters=100,
                                            ext_score_adjustment=0.5,
                                            method='cos',
                                            constant=0.1,
                                            matrix_power=3,
                                            log_lik_tol=0.000001):

        if anchor_raters is not None:
            if ((not hasattr(self, 'anchor_thresholds_thresholds')) or
                (self.anchor_raters_thresholds != anchor_raters)):
                self.calibrate_thresholds_anchor(anchor_raters, constant=constant, method=method,
                                                 matrix_power=matrix_power, log_lik_tol=log_lik_tol)

                self.person_abils_thresholds(anchor=True, items=None, raters=None,warm_corr=warm_corr,
                                             tolerance=tolerance, max_iters=max_iters,
                                             ext_score_adjustment=ext_score_adjustment)

        else:
            if not hasattr(self, 'thresholds_thresholds'):
                self.person_abils_thresholds(anchor=False, items=None, raters=None,warm_corr=warm_corr,
                                             tolerance=tolerance, max_iters=max_iters,
                                             ext_score_adjustment=ext_score_adjustment)

        if anchor_raters is not None:
            difficulties = self.anchor_diffs_thresholds
            thresholds = self.anchor_thresholds_thresholds
            severities = self.anchor_severities_thresholds

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_thresholds

        diff_df_dict = {}

        for threshold in range(self.max_score):
            diff_df_dict[threshold + 1] = {rater: [difficulties + thresholds[threshold + 1] +
                                                   severities[rater][threshold + 1]
                                                   for person in self.persons]
                                           for rater in self.raters}

            for rater in self.raters:
                diff_df_dict[threshold + 1][rater] = pd.DataFrame(diff_df_dict[threshold + 1][rater])
                diff_df_dict[threshold + 1][rater].index = self.persons
                diff_df_dict[threshold + 1][rater].columns = self.dataframe.columns

            diff_df_dict[threshold + 1] = pd.concat(diff_df_dict[threshold + 1].values(),
                                                    keys=diff_df_dict[threshold + 1].keys())

        (self.threshold_outfit_ms_thresholds,
         self.threshold_outfit_zstd_thresholds,
         self.threshold_infit_ms_thresholds,
         self.threshold_infit_zstd_thresholds,
         self.threshold_point_measure_thresholds,
         self.threshold_exp_point_measure_thresholds,
         self.threshold_discrimination_thresholds) = self.threshold_fit_statistics(self.abils_thresholds,
                                                                                   diff_df_dict)

    def threshold_fit_statistics_matrix(self,
                                        anchor_raters=None,
                                        warm_corr=True,
                                        tolerance=0.00001,
                                        max_iters=100,
                                        ext_score_adjustment=0.5,
                                        method='cos',
                                        constant=0.1,
                                        matrix_power=3,
                                        log_lik_tol=0.000001):

        if not hasattr(self, 'abils_matrix'):
            self.person_abils_matrix(anchor_raters=anchor_raters, warm_corr=warm_corr, tolerance=tolerance,
                                     max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)

        if anchor_raters is not None:
            if not hasattr(self, 'anchor_thresholds_matrix'):
                self.calibrate_matrix_anchor(anchor_raters, constant=constant, method=method, matrix_power=matrix_power,
                                             log_lik_tol=log_lik_tol)

                self.person_abils_matrix(anchor_raters=anchor_raters, warm_corr=warm_corr, tolerance=tolerance,
                                         max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)

        if anchor_raters is not None:
            difficulties = self.anchor_diffs_matrix
            thresholds = self.anchor_thresholds_matrix
            severities = self.anchor_severities_matrix

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_matrix

        diff_df_dict = {}

        for threshold in range(self.max_score):
            diff_df_dict[threshold + 1] = {rater: [difficulties + thresholds[threshold + 1] +
                                                   pd.Series({item: severities[rater][item][threshold + 1]
                                                              for item in self.dataframe.columns})
                                                  for person in self.persons]
                                           for rater in self.raters}

            for rater in self.raters:
                diff_df_dict[threshold + 1][rater] = pd.DataFrame(diff_df_dict[threshold + 1][rater])
                diff_df_dict[threshold + 1][rater].index = self.persons
                diff_df_dict[threshold + 1][rater].columns = self.dataframe.columns

            diff_df_dict[threshold + 1] = pd.concat(diff_df_dict[threshold + 1].values(),
                                                    keys=diff_df_dict[threshold + 1].keys())

        (self.threshold_outfit_ms_matrix,
         self.threshold_outfit_zstd_matrix,
         self.threshold_infit_ms_matrix,
         self.threshold_infit_zstd_matrix,
         self.threshold_point_measure_matrix,
         self.threshold_exp_point_measure_matrix,
         self.threshold_discrimination_matrix) = self.threshold_fit_statistics(self.abils_matrix, diff_df_dict)

    def rater_pivot(self,
                    df):

        pivoted = pd.DataFrame()
        for rater in self.raters:
            pivoted[rater] = df.xs(rater).T.stack()

        return pivoted

    def rater_fit_statistics(self,
                             info_df,
                             kurtosis_df,
                             residual_df,
                             std_residual_df):

        '''
        Rater fit statistics
        '''

        scores = self.dataframe.sum(axis=1)
        max_scores = self.dataframe.count(axis=1) * self.max_score

        rater_count = pd.Series({rater: self.dataframe[(scores > 0) & (scores < max_scores)].xs(rater).count().sum()
                                 for rater in self.raters})

        rater_outfit_ms = {rater: (std_residual_df ** 2).xs(rater).sum().sum() /
                                  (std_residual_df ** 2).xs(rater).count().sum()
                           for rater in self.raters}
        rater_outfit_ms = pd.Series(rater_outfit_ms)

        rater_infit_ms = {rater: (residual_df ** 2).xs(rater).sum().sum() /
                                  info_df.xs(rater).sum().sum()
                          for rater in self.raters}
        rater_infit_ms = pd.Series(rater_infit_ms)

        rater_outfit_q = ((self.rater_pivot(kurtosis_df) / (self.rater_pivot(info_df) ** 2)) /
                          (rater_count ** 2)).sum() - (1 / rater_count)
        rater_outfit_q = rater_outfit_q ** 0.5
        rater_outfit_zstd = (((rater_outfit_ms ** (1/3)) - 1) *
                             (3 / rater_outfit_q)) + (rater_outfit_q / 3)

        rater_infit_q = ((self.rater_pivot(kurtosis_df) - self.rater_pivot(info_df) ** 2).sum() /
                         (self.rater_pivot(info_df).sum() ** 2))
        rater_infit_q = rater_infit_q ** 0.5
        rater_infit_zstd = (((rater_infit_ms ** (1/3)) - 1) *
                            (3 / rater_infit_q)) + (rater_infit_q / 3)

        return (rater_outfit_ms, rater_outfit_zstd, rater_infit_ms, rater_infit_zstd)

    def rater_fit_statistics_global(self,
                                    warm_corr=True,
                                    tolerance=0.00001,
                                    max_iters=100,
                                    ext_score_adjustment=0.5,
                                    method='cos',
                                    constant=0.1,
                                    matrix_power=3,
                                    log_lik_tol=0.000001,
                                    no_of_samples=100,
                                    interval=None):

        if not hasattr(self, 'exp_score_df_global'):
            self.fit_matrices_global(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                     ext_score_adjustment=ext_score_adjustment, method=method,
                                     constant=constant, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        (self.rater_outfit_ms_global,
         self.rater_outfit_zstd_global,
         self.rater_infit_ms_global,
         self.rater_infit_zstd_global) = self.rater_fit_statistics(self.info_df_global, self.kurtosis_df_global,
                                                                   self.residual_df_global, self.std_residual_df_global)

    def rater_fit_statistics_items(self,
                                   warm_corr=True,
                                   tolerance=0.00001,
                                   max_iters=100,
                                   ext_score_adjustment=0.5,
                                   method='cos',
                                   constant=0.1,
                                   matrix_power=3,
                                   log_lik_tol=0.000001,
                                   no_of_samples=100,
                                   interval=None):

        if not hasattr(self, 'exp_score_df_items'):
            self.fit_matrices_items(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                    ext_score_adjustment=ext_score_adjustment, method=method,
                                    constant=constant, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        (self.rater_outfit_ms_items,
         self.rater_outfit_zstd_items,
         self.rater_infit_ms_items,
         self.rater_infit_zstd_items) = self.rater_fit_statistics(self.info_df_items, self.kurtosis_df_items,
                                                                  self.residual_df_items, self.std_residual_df_items)

    def rater_fit_statistics_thresholds(self,
                                        warm_corr=True,
                                        tolerance=0.00001,
                                        max_iters=100,
                                        ext_score_adjustment=0.5,
                                        method='cos',
                                        constant=0.1,
                                        matrix_power=3,
                                        log_lik_tol=0.000001,
                                        no_of_samples=100,
                                        interval=None):

        if not hasattr(self, 'exp_score_df_thresholds'):
            self.fit_matrices_thresholds(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                         ext_score_adjustment=ext_score_adjustment, method=method,
                                         constant=constant, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        (self.rater_outfit_ms_thresholds,
         self.rater_outfit_zstd_thresholds,
         self.rater_infit_ms_thresholds,
         self.rater_infit_zstd_thresholds) = self.rater_fit_statistics(self.info_df_thresholds,
                                                                       self.kurtosis_df_thresholds,
                                                                       self.residual_df_thresholds,
                                                                      self.std_residual_df_thresholds)

    def rater_fit_statistics_matrix(self,
                                    warm_corr=True,
                                    tolerance=0.00001,
                                    max_iters=100,
                                    ext_score_adjustment=0.5,
                                    method='cos',
                                    constant=0.1,
                                    matrix_power=3,
                                    log_lik_tol=0.000001,
                                    no_of_samples=100,
                                    interval=None):

        if not hasattr(self, 'exp_score_df_matrix'):
            self.fit_matrices_matrix(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                     ext_score_adjustment=ext_score_adjustment, method=method,
                                     constant=constant, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        (self.rater_outfit_ms_matrix,
         self.rater_outfit_zstd_matrix,
         self.rater_infit_ms_matrix,
         self.rater_infit_zstd_matrix) = self.rater_fit_statistics(self.info_df_matrix, self.kurtosis_df_matrix,
                                                                   self.residual_df_matrix, self.std_residual_df_matrix)

    def rater_res_corr_analysis(self,
                                residual_df,
                                std_residual_df):

        rater_residual_df = self.rater_pivot(residual_df)
        rater_std_residual_df = self.rater_pivot(std_residual_df)
        rater_residual_correlations = rater_residual_df.corr(numeric_only=False)

        pca = PCA()

        try:
            pca.fit(rater_std_residual_df.corr(numeric_only=False))

            rater_eigenvectors = pd.DataFrame(pca.components_)
            rater_eigenvectors.columns = [f'Eigenvector {pc + 1}' for pc in range(self.no_of_raters)]

            rater_eigenvalues = pca.explained_variance_
            rater_eigenvalues = pd.DataFrame(rater_eigenvalues)
            rater_eigenvalues.index = [f'PC {pc + 1}' for pc in range(self.no_of_raters)]
            rater_eigenvalues.columns = ['Eigenvalue']

            rater_variance_explained = pd.DataFrame(pca.explained_variance_ratio_)
            rater_variance_explained.index = [f'PC {pc + 1}' for pc in range(self.no_of_raters)]
            rater_variance_explained.columns = ['Variance explained']

            rater_loadings = rater_eigenvectors.T * (pca.explained_variance_ ** 0.5)
            rater_loadings = pd.DataFrame(rater_loadings)
            rater_loadings.columns = [f'PC {pc + 1}' for pc in range(self.no_of_raters)]
            rater_loadings.index = [rater for rater in self.raters]

        except Exception:
            print('PCA of rater residuals failed')

            rater_eigenvectors = None
            rater_eigenvalues = None
            rater_variance_explained = None
            rater_loadings = None

        return (rater_residual_correlations,
                rater_eigenvectors,
                rater_eigenvalues,
                rater_variance_explained,
                rater_loadings)

    def rater_res_corr_analysis_global(self):

        (self.rater_residual_correlations_global,
         self.rater_eigenvectors_global,
         self.rater_eigenvalues_global,
         self.rater_variance_explained_global,
         self.rater_loadings_global) = self.rater_res_corr_analysis(self.residual_df_global,
                                                                    self.std_residual_df_global)

    def rater_res_corr_analysis_items(self):

        (self.rater_residual_correlations_items,
         self.rater_eigenvectors_items,
         self.rater_eigenvalues_items,
         self.rater_variance_explained_items,
         self.rater_loadings_items) = self.rater_res_corr_analysis(self.residual_df_items,
                                                                   self.std_residual_df_items)

    def rater_res_corr_analysis_thresholds(self):

        (self.rater_residual_correlations_thresholds,
         self.rater_eigenvectors_thresholds,
         self.rater_eigenvalues_thresholds,
         self.rater_variance_explained_thresholds,
         self.rater_loadings_thresholds) = self.rater_res_corr_analysis(self.residual_df_thresholds,
                                                                        self.std_residual_df_thresholds)

    def rater_res_corr_analysis_matrix(self):

        (self.rater_residual_correlations_matrix,
         self.rater_eigenvectors_matrix,
         self.rater_eigenvalues_matrix,
         self.rater_variance_explained_matrix,
         self.rater_loadings_matrix) = self.rater_res_corr_analysis(self.residual_df_matrix,
                                                                    self.std_residual_df_matrix)

    def person_fit_statistics(self,
                              info_df,
                              kurtosis_df,
                              residual_df,
                              std_residual_df,
                              abilities,
                              warm_corr=True,
                              tolerance=0.00001,
                              max_iters=100,
                              ext_score_adjustment=0.5,
                              method='cos',
                              constant=0.1,
                              matrix_power=3,
                              log_lik_tol=0.000001,
                              no_of_samples=100,
                              interval=None):

        '''
        Person fit statistics
        '''

        csems = 1 / (info_df.unstack(level=0).sum(axis=1) ** 0.5)

        rsems = (((residual_df.unstack(level=0) ** 2).sum(axis=1)) ** 0.5 /
                 info_df.unstack(level=0).sum(axis=1))

        person_outfit_ms = (std_residual_df.unstack(level=0) ** 2).mean(axis=1)
        person_infit_ms = ((residual_df.unstack(level=0) ** 2).sum(axis=1) /
        info_df.unstack(level=0).sum(axis=1))

        scores = self.dataframe.sum(axis=1)
        max_scores = self.dataframe.count(axis=1) * self.max_score

        person_count = (self.dataframe[(scores > 0) & (scores < max_scores)].unstack(level=0) ==
        self.dataframe[(scores > 0) & (scores < max_scores)].unstack(level=0)).sum(axis=1)

        base_df = kurtosis_df.unstack(level=0) / (info_df.unstack(level=0) ** 2)
        base_df = base_df.transpose().reset_index().transpose()
        for column in base_df.columns:
            base_df[column] /= (person_count ** 2)
        person_outfit_q = base_df.sum(axis=1) - (1 / person_count)
        person_outfit_q = person_outfit_q ** 0.5
        person_outfit_zstd = (((person_outfit_ms ** (1/3)) - 1) * (3 / person_outfit_q)) + (person_outfit_q / 3)
        person_outfit_zstd = person_outfit_zstd[:self.no_of_persons]

        person_infit_q = ((kurtosis_df.unstack(level=0) - info_df.unstack(level=0) ** 2).sum(axis=1) /
                          (info_df.unstack(level=0).sum(axis=1) ** 2))
        person_infit_q = person_infit_q ** 0.5
        person_infit_zstd = (((person_infit_ms ** (1/3)) - 1) * (3 / person_infit_q)) + (person_infit_q / 3)

        return csems, rsems, person_outfit_ms, person_outfit_zstd, person_infit_ms, person_infit_zstd


    def person_fit_statistics_global(self,
                                     warm_corr=True,
                                     tolerance=0.00001,
                                     max_iters=100,
                                     ext_score_adjustment=0.5,
                                     method='cos',
                                     constant=0.1,
                                     matrix_power=3,
                                     log_lik_tol=0.000001):

        if not hasattr(self, 'exp_score_df_global'):
            self.fit_matrices_global(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                     ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                     matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        (self.csem_vector_global,
         self.rsem_vector_global,
         self.person_outfit_ms_global,
         self.person_outfit_zstd_global,
         self.person_infit_ms_global,
         self.person_infit_zstd_global) = self.person_fit_statistics(self.info_df_global,
                                                                     self.kurtosis_df_global,
                                                                     self.residual_df_global,
                                                                     self.std_residual_df_global,
                                                                     self.abils_global, warm_corr=warm_corr,
                                                                     tolerance=tolerance,
                                                                     max_iters=max_iters,
                                                                     ext_score_adjustment=ext_score_adjustment,
                                                                     method=method, constant=constant,
                                                                     matrix_power=matrix_power, log_lik_tol=log_lik_tol)

    def person_fit_statistics_items(self,
                                    warm_corr=True,
                                    tolerance=0.00001,
                                    max_iters=100,
                                    ext_score_adjustment=0.5,
                                    method='cos',
                                    constant=0.1,
                                    matrix_power=3,
                                    log_lik_tol=0.000001):

        if not hasattr(self, 'exp_score_df_items'):
            self.fit_matrices_items(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                    ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                    matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        (self.csem_vector_items,
         self.rsem_vector_items,
         self.person_outfit_ms_items,
         self.person_outfit_zstd_items,
         self.person_infit_ms_items,
         self.person_infit_zstd_items) = self.person_fit_statistics(self.info_df_items,
                                                                    self.kurtosis_df_items,
                                                                    self.residual_df_items, self.std_residual_df_items,
                                                                    self.abils_items, warm_corr=warm_corr,
                                                                    tolerance=tolerance,
                                                                    max_iters=max_iters,
                                                                    ext_score_adjustment=ext_score_adjustment,
                                                                    method=method, constant=constant,
                                                                    matrix_power=matrix_power, log_lik_tol=log_lik_tol)

    def person_fit_statistics_thresholds(self,
                                         warm_corr=True,
                                         tolerance=0.00001,
                                         max_iters=100,
                                         ext_score_adjustment=0.5,
                                         method='cos',
                                         constant=0.1,
                                         matrix_power=3,
                                         log_lik_tol=0.000001):

        if not hasattr(self, 'exp_score_df_thresholds'):
            self.fit_matrices_thresholds(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                         ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                         matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        (self.csem_vector_thresholds,
         self.rsem_vector_thresholds,
         self.person_outfit_ms_thresholds,
         self.person_outfit_zstd_thresholds,
         self.person_infit_ms_thresholds,
         self.person_infit_zstd_thresholds) = self.person_fit_statistics(self.info_df_thresholds,
                                                                         self.kurtosis_df_thresholds,
                                                                         self.residual_df_thresholds,
                                                                         self.std_residual_df_thresholds,
                                                                         self.abils_thresholds, warm_corr=warm_corr,
                                                                         tolerance=tolerance,
                                                                         max_iters=max_iters,
                                                                         ext_score_adjustment=ext_score_adjustment,
                                                                         method=method, constant=constant,
                                                                         matrix_power=matrix_power,
                                                                         log_lik_tol=log_lik_tol)

    def person_fit_statistics_matrix(self,
                                     warm_corr=True,
                                     tolerance=0.00001,
                                     max_iters=100,
                                     ext_score_adjustment=0.5,
                                     method='cos',
                                     constant=0.1,
                                     matrix_power=3,
                                     log_lik_tol=0.000001):

        if not hasattr(self, 'exp_score_df_matrix'):
            self.fit_matrices_matrix(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                     ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                     matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        (self.csem_vector_matrix,
         self.rsem_vector_matrix,
         self.person_outfit_ms_matrix,
         self.person_outfit_zstd_matrix,
         self.person_infit_ms_matrix,
         self.person_infit_zstd_matrix) = self.person_fit_statistics(self.info_df_matrix,
                                                                     self.kurtosis_df_matrix,
                                                                     self.residual_df_matrix,
                                                                     self.std_residual_df_matrix,
                                                                     self.abils_matrix, warm_corr=warm_corr,
                                                                     tolerance=tolerance,
                                                                     max_iters=max_iters,
                                                                     ext_score_adjustment=ext_score_adjustment,
                                                                     method=method, constant=constant)

    def test_fit_statistics(self,
                            abilities,
                            rsems):
        '''
        Test-level fit statistics
        '''

        scores = self.dataframe.unstack(level=0).sum(axis=1)
        max_scores = self.dataframe.unstack(level=0).count(axis=1) * self.max_score

        abilities = abilities.copy()[(scores > 0) & (scores < max_scores)]

        isi = (self.diffs.var() / (self.item_se ** 2).mean() - 1) ** 0.5
        item_strata = (4 * isi + 1) / 3
        item_reliability = (isi ** 2) / (1 + isi ** 2)

        psi = ((np.var(abilities) -  (rsems ** 2).mean()) ** 0.5) / ((rsems ** 2).mean() ** 0.5)
        person_strata = (4 * psi + 1) / 3
        person_reliability = (psi ** 2) / (1 + (psi ** 2))

        return isi, item_strata, item_reliability, psi, person_strata, person_reliability


    def test_fit_statistics_global(self,
                                   warm_corr=True,
                                   tolerance=0.00001,
                                   max_iters=100,
                                   ext_score_adjustment=0.5,
                                   method='cos',
                                   constant=0.1,
                                   matrix_power=3,
                                   log_lik_tol=0.000001,
                                   no_of_samples=100):

        if not hasattr(self, 'item_se'):
            self.std_errors_global(anchor_raters=None, interval=None, no_of_samples=no_of_samples, constant=constant,
                                   method=method, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if not hasattr(self, 'csem_vector_global'):
            self.person_fit_statistics_global(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                              ext_score_adjustment=ext_score_adjustment, method=method,
                                              constant=constant, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        (self.isi_global,
         self.item_strata_global,
         self.item_reliability_global,
         self.psi_global,
         self.person_strata_global,
         self.person_reliability_global) = self.test_fit_statistics(self.abils_global, self.rsem_vector_global)


    def test_fit_statistics_items(self,
                                  warm_corr=True,
                                  tolerance=0.00001,
                                  max_iters=100,
                                  ext_score_adjustment=0.5,
                                  method='cos',
                                  constant=0.1,
                                  matrix_power=3,
                                  log_lik_tol=0.000001,
                                  no_of_samples=100):

        if not hasattr(self, 'item_se'):
            self.std_errors_items(anchor_raters=None, interval=None, no_of_samples=no_of_samples, constant=constant,
                                  method=method, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if not hasattr(self, 'csem_vector_items'):
            self.person_fit_statistics_items(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                             ext_score_adjustment=ext_score_adjustment, method=method,
                                             constant=constant, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        (self.isi_items,
         self.item_strata_items,
         self.item_reliability_items,
         self.psi_items,
         self.person_strata_items,
         self.person_reliability_items) = self.test_fit_statistics(self.abils_items, self.rsem_vector_items)


    def test_fit_statistics_thresholds(self,
                                       warm_corr=True,
                                       tolerance=0.00001,
                                       max_iters=100,
                                       ext_score_adjustment=0.5,
                                       method='cos',
                                       constant=0.1,
                                       matrix_power=3,
                                       log_lik_tol=0.000001,
                                       no_of_samples=100):

        if not hasattr(self, 'item_se'):
            self.std_errors_thresholds(anchor_raters=None, interval=None, no_of_samples=no_of_samples,
                                       constant=constant, method=method, matrix_power=matrix_power,
                                       log_lik_tol=log_lik_tol)

        if not hasattr(self, 'csem_vector_thresholds'):
            self.person_fit_statistics_thresholds(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                                  ext_score_adjustment=ext_score_adjustment, method=method,
                                                  constant=constant, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        (self.isi_thresholds,
         self.item_strata_thresholds,
         self.item_reliability_thresholds,
         self.psi_thresholds,
         self.person_strata_thresholds,
         self.person_reliability_thresholds) = self.test_fit_statistics(self.abils_thresholds,
                                                                        self.rsem_vector_thresholds)


    def test_fit_statistics_matrix(self,
                                   warm_corr=True,
                                   tolerance=0.00001,
                                   max_iters=100,
                                   ext_score_adjustment=0.5,
                                   method='cos',
                                   constant=0.1,
                                   matrix_power=3,
                                   log_lik_tol=0.000001,
                                   no_of_samples=100):

        if not hasattr(self, 'item_se'):
            self.std_errors_matrix(anchor_raters=None, interval=None, no_of_samples=no_of_samples, constant=constant,
                                   method=method, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if not hasattr(self, 'csem_vector_matrix'):
            self.person_fit_statistics_matrix(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                              ext_score_adjustment=ext_score_adjustment, method=method,
                                              constant=constant, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        (self.isi_matrix,
         self.item_strata_matrix,
         self.item_reliability_matrix,
         self.psi_matrix,
         self.person_strata_matrix,
         self.person_reliability_matrix) = self.test_fit_statistics(self.abils_matrix, self.rsem_vector_matrix)

    def fit_statistics_global(self,
                              warm_corr=True,
                              se=True,
                              test_stats=True,
                              ext_scores=True,
                              tolerance=0.00001,
                              max_iters=100,
                              ext_score_adjustment=0.5,
                              method='cos',
                              constant=0.1,
                              matrix_power=3,
                              log_lik_tol=0.000001,
                              no_of_samples=100,
                              interval=None):

        '''
        All fit statistics
        '''

        if not hasattr(self, 'thresholds_global'):
            self.calibrate_global(constant=constant, method=method,
                                  matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if se:
            if not hasattr(self, 'threshold_se_global'):
                self.std_errors_global(interval=interval, no_of_samples=no_of_samples, constant=constant, method=method,
                                       matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if not hasattr(self, 'abils_global'):
            self.person_abils_global(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                     ext_score_adjustment=ext_score_adjustment)

        self.category_probability_dict_global(warm_corr=warm_corr, ext_scores=ext_scores,
                                              tolerance=tolerance, max_iters=max_iters,
                                              ext_score_adjustment=ext_score_adjustment, method=method,
                                              constant=constant, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if se == False:
            test_stats = False

        self.fit_matrices_global()
        self.item_fit_statistics_global()
        self.threshold_fit_statistics_global()
        self.rater_fit_statistics_global()
        self.person_fit_statistics_global()
        if test_stats:
            self.test_fit_statistics_global()


    def fit_statistics_items(self,
                             warm_corr=True,
                             se=True,
                             test_stats=True,
                             ext_scores=True,
                             tolerance=0.00001,
                             max_iters=100,
                             ext_score_adjustment=0.5,
                             method='cos',
                             constant=0.1,
                             matrix_power=3,
                             log_lik_tol=0.000001,
                             no_of_samples=100,
                             interval=None):
        '''
        All fit statistics
        '''

        if not hasattr(self, 'thresholds_items'):
            self.calibrate_items(constant=constant, method=method,
                                 matrix_power=matrix_power, log_lik_tol=log_lik_tol)
            
        if se:
            if not hasattr(self, 'threshold_se_items'):
                self.std_errors_items(interval=interval, no_of_samples=no_of_samples, constant=constant, method=method,
                                      matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if not hasattr(self, 'abils_items'):
            self.person_abils_items(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                    ext_score_adjustment=ext_score_adjustment)

        self.category_probability_dict_items(warm_corr=warm_corr, ext_scores=ext_scores,
                                             tolerance=tolerance, max_iters=max_iters,
                                              ext_score_adjustment=ext_score_adjustment, method=method,
                                              constant=constant, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if se == False:
            test_stats = False

        self.fit_matrices_items()
        self.item_fit_statistics_items()
        self.threshold_fit_statistics_items()
        self.rater_fit_statistics_items()
        self.person_fit_statistics_items()
        if test_stats:
            self.test_fit_statistics_items()


    def fit_statistics_thresholds(self,
                                  warm_corr=True,
                                  se=True,
                                  test_stats=True,
                                  ext_scores=True,
                                  tolerance=0.00001,
                                  max_iters=100,
                                  ext_score_adjustment=0.5,
                                  method='cos',
                                  constant=0.1,
                                  matrix_power=3,
                                  log_lik_tol=0.000001,
                                  no_of_samples=100,
                                  interval=None):
        '''
        All fit statistics
        '''

        if not hasattr(self, 'thresholds_thresholds'):
            self.calibrate_thresholds(constant=constant, method=method,
                                      matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if se:
            if not hasattr(self, 'threshold_se_thresholds'):
                self.std_errors_thresholds(interval=interval, no_of_samples=no_of_samples, constant=constant, method=method,
                                           matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if not hasattr(self, 'abils_thresholds'):
            self.person_abils_thresholds(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                         ext_score_adjustment=ext_score_adjustment)

        self.category_probability_dict_thresholds(warm_corr=warm_corr, ext_scores=ext_scores,
                                                  tolerance=tolerance, max_iters=max_iters,
                                                  ext_score_adjustment=ext_score_adjustment, method=method,
                                                  constant=constant)

        if se == False:
            test_stats = False

        self.fit_matrices_thresholds()
        self.item_fit_statistics_thresholds()
        self.threshold_fit_statistics_thresholds()
        self.rater_fit_statistics_thresholds()
        self.person_fit_statistics_thresholds()
        if test_stats:
            self.test_fit_statistics_thresholds()


    def fit_statistics_matrix(self,
                              warm_corr=True,
                              se=True,
                              test_stats=True,
                              ext_scores=True,
                              tolerance=0.00001,
                              max_iters=100,
                              ext_score_adjustment=0.5,
                              method='cos',
                              constant=0.1,
                              matrix_power=3,
                              log_lik_tol=0.000001,
                              no_of_samples=100,
                              interval=None):
        '''
        All fit statistics
        '''

        if not hasattr(self, 'thresholds_matrix'):
            self.calibrate_matrix(constant=constant, method=method,
                                  matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if se:
            if not hasattr(self, 'threshold_se_matrix'):
                self.std_errors_matrix(interval=interval, no_of_samples=no_of_samples, constant=constant, method=method,
                                       matrix_power=matrix_power, log_lik_tol=log_lik_tol)
            
        if not hasattr(self, 'abils_matrix'):
            self.person_abils_matrix(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                     ext_score_adjustment=ext_score_adjustment)

        self.category_probability_dict_matrix(warm_corr=warm_corr, ext_scores=ext_scores,
                                              tolerance=tolerance, max_iters=max_iters,
                                              ext_score_adjustment=ext_score_adjustment, method=method,
                                              constant=constant)

        if se == False:
            test_stats = False

        self.fit_matrices_matrix()
        self.item_fit_statistics_matrix()
        self.threshold_fit_statistics_matrix()
        self.rater_fit_statistics_matrix()
        self.person_fit_statistics_matrix()
        if test_stats:
            self.test_fit_statistics_matrix()

    def save_residuals(self,
                       eigenvectors,
                       eigenvalues,
                       variance_explained,
                       loadings,
                       fit_statistics_method,
                       eigenvector_string,
                       filename,
                       format='csv',
                       single=True,
                       dp=3,
                       warm_corr=True,
                       tolerance=0.00001,
                       max_iters=100,
                       ext_score_adjustment=0.5,
                       method='cos',
                       constant=0.1):

        if not hasattr(self, eigenvector_string):
            fit_statistics_method(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                  ext_score_adjustment=ext_score_adjustment, method=method,
                                  constant=constant)

        if eigenvector_string[:4] == 'item':
            residual_type = 'item'

        else:
            residual_type = 'rater'

        if single:
            if format == 'xlsx':

                if filename[-5:] != '.xlsx':
                    filename += '.xlsx'

                writer = pd.ExcelWriter(filename, engine='xlsxwriter')
                row = 0

                if eigenvectors is None:

                    workbook = xlsxwriter.Workbook(filename)
                    worksheet = workbook.add_worksheet()
                    worksheet.write(0, 0, f'PCA of {residual_type} residuals failed.')
                    workbook.close()

                else:
                    eigenvectors.round(dp).to_excel(writer, sheet_name='Rater residual analysis',
                                                    startrow=row, startcol=0)
                    row += (eigenvectors.shape[0] + 2)

                    eigenvalues.round(dp).to_excel(writer, sheet_name='Rater residual analysis',
                                                   startrow=row, startcol=0)
                    row += (eigenvalues.shape[0] + 2)

                    variance_explained.round(dp).to_excel(writer, sheet_name='Rater residual analysis',
                                                          startrow=row, startcol=0)
                    row += (variance_explained.shape[0] + 2)

                    loadings.round(dp).to_excel(writer, sheet_name='Rater residual analysis',
                                                startrow=row, startcol=0)

                    writer.close()

            else:
                if filename[-4:] != '.csv':
                    filename += '.csv'

                if eigenvectors is None:
                    with open(filename, 'a') as f:
                        f.write(f'PCA of {residual_type} residuals failed.')

                else:
                    with open(filename, 'a') as f:
                        eigenvectors.round(dp).to_csv(f)
                        f.write("\n")
                        eigenvalues.round(dp).to_csv(f)
                        f.write("\n")
                        variance_explained.round(dp).to_csv(f)
                        f.write("\n")
                        loadings.round(dp).to_csv(f)

        else:
            if format == 'xlsx':

                if filename[-5:] != '.xlsx':
                    filename += '.xlsx'

                writer = pd.ExcelWriter(filename, engine='xlsxwriter')

                if eigenvectors is None:
                    workbook = xlsxwriter.Workbook(filename)
                    worksheet = workbook.add_worksheet()
                    worksheet.write(0, 0, f'PCA of {residual_type} residuals failed.')
                    workbook.close()

                else:
                    eigenvectors.round(dp).to_excel(writer, sheet_name='Eigenvectors')
                    eigenvalues.round(dp).to_excel(writer, sheet_name='Eigenvalues')
                    variance_explained.round(dp).to_excel(writer, sheet_name='Variance explained')
                    loadings.round(dp).to_excel(writer, sheet_name='Principal Component loadings')

                    writer.close()

            else:
                if filename[-4:] == '.csv':
                    filename = filename[:-4]

                if eigenvectors is None:
                    with open(filename, 'a') as f:
                        f.write(f'PCA of {residual_type} residuals failed.')

                else:
                    eigenvectors.round(dp).to_csv(f'{filename}_eigenvectors.csv')
                    eigenvalues.round(dp).to_csv(f'{filename}_eigenvalues.csv')
                    variance_explained.round(dp).to_csv(f'{filename}_variance_explained.csv')
                    loadings.round(dp).to_csv(f'{filename}_principal_component_loadings.csv')

    def save_residuals_items_global(self,
                                    filename,
                                    format='csv',
                                    single=True,
                                    dp=3,
                                    warm_corr=True,
                                    tolerance=0.00001,
                                    max_iters=100,
                                    ext_score_adjustment=0.5,
                                    method='cos',
                                    constant=0.1):

    	self.save_residuals(self.item_eigenvectors_global, self.item_eigenvalues_global,
    						self.item_variance_explained_global, self.item_loadings_global,
    						self.item_fit_statistics_global, 'item_eigenvectors_global',
    						filename, format=format, single=single, dp=dp, warm_corr=warm_corr,
    						tolerance=tolerance, max_iters=max_iters,
    						ext_score_adjustment=ext_score_adjustment, method=method,
    						constant=constant)


    def save_residuals_items_items(self,
                                   filename,
                                   format='csv',
                                   single=True,
                                   dp=3,
                                   warm_corr=True,
                                   tolerance=0.00001,
                                   max_iters=100,
                                   ext_score_adjustment=0.5,
                                   method='cos',
                                   constant=0.1):

    	self.save_residuals(self.item_eigenvectors_items, self.item_eigenvalues_items,
    						self.item_variance_explained_items, self.item_loadings_items,
    						self.item_fit_statistics_items, 'item_eigenvectors_items',
    						filename, format=format, single=single, dp=dp, warm_corr=warm_corr,
    						tolerance=tolerance, max_iters=max_iters, ext_score_adjustment=ext_score_adjustment,
                            method=method, constant=constant)

    def save_residuals_items_thresholds(self,
                                     	filename,
                                     	format='csv',
                                     	single=True,
                                     	dp=3,
                                     	warm_corr=True,
                                     	tolerance=0.00001,
                                     	max_iters=100,
                                     	ext_score_adjustment=0.5,
                                     	method='cos',
                                        constant=0.1):

    	self.save_residuals(self.item_eigenvectors_thresholds, self.item_eigenvalues_thresholds,
    						self.item_variance_explained_thresholds, self.item_loadings_thresholds,
    						self.item_fit_statistics_thresholds, 'item_eigenvectors_thresholds',
    						filename, format=format, single=single, dp=dp, warm_corr=warm_corr,
    						tolerance=tolerance, max_iters=max_iters, ext_score_adjustment=ext_score_adjustment,
                            method=method, constant=constant)

    def save_residuals_items_matrix(self,
                                    filename,
                                    format='csv',
                                    single=True,
                                    dp=3,
                                    warm_corr=True,
                                    tolerance=0.00001,
                                    max_iters=100,
                                    ext_score_adjustment=0.5,
                                    method='cos',
                                    constant=0.1):

    	self.save_residuals(self.item_eigenvectors_matrix, self.item_eigenvalues_matrix,
    						self.item_variance_explained_matrix, self.item_loadings_matrix,
    						self.item_fit_statistics_matrix, 'item_eigenvectors_matrix',
    						filename, format=format, single=single, dp=dp, warm_corr=warm_corr,
    						tolerance=tolerance, max_iters=max_iters, ext_score_adjustment=ext_score_adjustment,
                            method=method, constant=constant)

    def save_residuals_raters_global(self,
                                     filename,
                                     format='csv',
                                     single=True,
                                     dp=3,
                                     warm_corr=True,
                                     tolerance=0.00001,
                                     max_iters=100,
                                     ext_score_adjustment=0.5,
                                     method='cos',
                                     constant=0.1):

        self.save_residuals(self.rater_eigenvectors_global, self.rater_eigenvalues_global,
                            self.rater_variance_explained_global, self.rater_loadings_global,
                            self.rater_fit_statistics_global, 'rater_eigenvectors_global',
                            filename, format=format, single=single, dp=dp, warm_corr=warm_corr,
                            tolerance=tolerance, max_iters=max_iters, ext_score_adjustment=ext_score_adjustment,
                            method=method, constant=constant)

    def save_residuals_raters_items(self,
                                    filename,
                                    format='csv',
                                    single=True,
                                    dp=3,
                                    warm_corr=True,
                                    tolerance=0.00001,
                                    max_iters=100,
                                    ext_score_adjustment=0.5,
                                    method='cos',
                                    constant=0.1):

        self.save_residuals(self.rater_eigenvectors_items, self.rater_eigenvalues_items,
                            self.rater_variance_explained_items, self.rater_loadings_items,
                            self.rater_fit_statistics_items, 'rater_eigenvectors_items',
                            filename, format=format, single=single, dp=dp, warm_corr=warm_corr,
                            tolerance=tolerance, max_iters=max_iters, ext_score_adjustment=ext_score_adjustment,
                            method=method, constant=constant)

    def save_residuals_raters_thresholds(self,
                                         filename,
                                         format='csv',
                                         single=True,
                                         dp=3,
                                         warm_corr=True,
                                         tolerance=0.00001,
                                         max_iters=100,
                                         ext_score_adjustment=0.5,
                                         method='cos',
                                         constant=0.1):

        self.save_residuals(self.rater_eigenvectors_thresholds, self.rater_eigenvalues_thresholds,
                            self.rater_variance_explained_thresholds, self.rater_loadings_thresholds,
                            self.rater_fit_statistics_thresholds, 'rater_eigenvectors_thresholds',
                            filename, format=format, single=single, dp=dp, warm_corr=warm_corr,
                            tolerance=tolerance, max_iters=max_iters, ext_score_adjustment=ext_score_adjustment,
                            method=method, constant=constant)

    def save_residuals_raters_matrix(self,
                                     filename,
                                     format='csv',
                                     single=True,
                                     dp=3,
                                     warm_corr=True,
                                     tolerance=0.00001,
                                     max_iters=100,
                                     ext_score_adjustment=0.5,
                                     method='cos',
                                     constant=0.1):

        self.save_residuals(self.rater_eigenvectors_matrix, self.rater_eigenvalues_matrix,
                            self.rater_variance_explained_matrix, self.rater_loadings_matrix,
                            self.rater_fit_statistics_matrix, 'rater_eigenvectors_matrix',
                            filename, format=format, single=single, dp=dp, warm_corr=warm_corr,
                            tolerance=tolerance, max_iters=max_iters, ext_score_adjustment=ext_score_adjustment,
                            method=method, constant=constant)

    def class_intervals(self,
                        abilities,
                        items=None,
                        raters=None,
                        shift=0,
                        no_of_classes=5):

        if isinstance(items, str):
            if items == 'all':
                items = None

            elif items == 'none':
                items = None

        if isinstance(raters, str):
            if raters == 'all':
                raters = self.raters.tolist()

            elif raters == 'none':
                raters = None

            elif raters == 'zero':
                raters = None

            else:
                raters = [raters]

        class_groups = [f'class_{class_no + 1}' for class_no in range(no_of_classes)]

        df = self.dataframe.copy()

        if items is None:
            abil_index = self.dataframe.unstack(level=0).dropna(how='any').index

        else:
            abil_index = self.dataframe[items].unstack(level=0).dropna(how='any').index

        abils = abilities.loc[abil_index]

        if isinstance(raters, list):
            df = {rater: df.xs(rater) for rater in raters}
            df = pd.concat(df.values(), keys=df.keys())

        if items is not None:
            df = df[items]

        if isinstance(items, list):
            df = df.loc[pd.IndexSlice[:, abil_index], :]

        if isinstance(items, str):
            df = df.loc[pd.IndexSlice[:, abil_index]]

        if items is None:
            df = df.loc[pd.IndexSlice[:, abil_index], :]

        quantiles = (abils.quantile([(i + 1) / no_of_classes
                                     for i in range(no_of_classes - 1)]))

        mask_dict = {}

        mask_dict['class_1'] = (abils < quantiles.values[0])
        mask_dict[f'class_{no_of_classes}'] = (abils >= quantiles.values[no_of_classes - 2])

        for class_no in range(no_of_classes - 2):
            mask_dict[f'class_{class_no + 2}'] = ((abils >= quantiles.values[class_no]) &
                                                  (abils < quantiles.values[class_no + 1]))

        df_mask_dict = {}

        if raters is None:
            for class_no in range(no_of_classes):
                class_name = f'class_{class_no + 1}'
                df_mask_dict[class_name] = {rater: mask_dict[class_name] for rater in self.raters.tolist()}
                df_mask_dict[class_name] = pd.concat(df_mask_dict[class_name].values(),
                                                     keys=df_mask_dict[class_name].keys())
                df_mask_dict[class_name] = df_mask_dict[class_name][df_mask_dict[class_name]].index

        if (isinstance(raters, list)):
            for class_no in range(no_of_classes):
                class_name = f'class_{class_no + 1}'
                df_mask_dict[class_name] = {rater: mask_dict[class_name] for rater in raters}
                df_mask_dict[class_name] = pd.concat(df_mask_dict[class_name].values(),
                                                     keys=df_mask_dict[class_name].keys())
                df_mask_dict[class_name] = df_mask_dict[class_name][df_mask_dict[class_name]].index

        mean_abilities = {class_group: abils[mask_dict[class_group]].mean()
                          for class_group in class_groups}
        mean_abilities = pd.Series(mean_abilities) - shift

        if raters is None:
            obs = {class_group: df.loc[df_mask_dict[class_group]].mean().sum()
                   for class_group in class_groups}

        else:
            # BUG FIX: df is already filtered to requested raters via pd.concat earlier.
            # The original .xs(rater) raised KeyError when a rater had no responses
            # in a particular class group. .mean().sum() correctly aggregates all raters.
            obs = {class_group: df.loc[df_mask_dict[class_group]].mean().sum()
                   for class_group in class_groups}

        obs = pd.Series(obs)

        return mean_abilities, obs

    def class_intervals_cats_global(self,
                                    abilities,
                                    difficulties,
                                    thresholds,
                                    severities,
                                    item=None,
                                    rater=None,
                                    shift=0,
                                    no_of_classes=5):

        if rater == 'none':
            rater = None

        if rater == 'zero':
            rater = None

        class_groups = [f'class_{class_no + 1}' for class_no in range(no_of_classes)]

        df = self.dataframe.copy()

        abil_df = pd.DataFrame()

        for item_ in self.dataframe.columns:
            abil_df[item_] = abilities

        if item is None:
            for item_ in self.dataframe.columns:
                abil_df.loc[:, item_] -= difficulties[item_]

        abil_dict = {}

        for rater_ in self.raters:
            abil_dict[rater_] = abil_df.copy()

            if rater is None:
                abil_dict[rater_] -= severities[rater_]

        abil_df = pd.concat(abil_dict.values(), keys=abil_dict.keys())

        if item is None:
            if rater is None:
                df = df
                df_mask = df.notna().astype(float).replace(0, np.nan)
                abil_df = abil_df * df_mask

                mask_scores = df.unstack().unstack()
                mask_abils = abil_df.unstack().unstack()

            else:
                df = df.xs(rater)
                df_mask = df.notna().astype(float).replace(0, np.nan)
                abil_df = abil_df.xs(rater) * df_mask

                mask_scores = df.unstack()
                mask_abils = abil_df.unstack()

        else:
            if rater is None:
                df = df[item].unstack(level=0)
                df_mask = df.notna().astype(float).replace(0, np.nan)
                abil_df = abil_df[item].unstack(level=0) * df_mask

                mask_scores = df.unstack()
                mask_abils = abil_df.unstack()

            else:
                df = df[item].xs(rater)
                df_mask = df.notna().astype(float).replace(0, np.nan)
                abil_df = abil_df[item].xs(rater) * df_mask

                mask_scores = df
                mask_abils = abil_df

        def class_masks(abils):
            mask_dict = {}

            quantiles = (abils.quantile([(i + 1) / no_of_classes
                                         for i in range(no_of_classes - 1)]))

            mask_dict['class_1'] = (abils < quantiles.values[0])
            mask_dict[f'class_{no_of_classes}'] = (abils >= quantiles.values[no_of_classes - 2])
            for class_group in range(no_of_classes - 2):
                mask_dict[f'class_{class_group + 2}'] = ((abils >= quantiles.values[class_group]) &
                                                         (abils < quantiles.values[class_group + 1]))

            for class_group in class_groups:
                mask_dict[class_group] = mask_dict[class_group][mask_dict[class_group]].index

            return mask_dict

        mask = class_masks(mask_abils)
        mean_abilities = [mask_abils.loc[mask[class_group]].mean()
                          for class_group in class_groups]
        mean_abilities = np.array(mean_abilities)

        obs_props = []

        for category in range(self.max_score + 1):
            obs_props_cat = [len(mask_scores.loc[mask[class_group]][mask_scores == category]) / len(
                mask_scores.loc[mask[class_group]])
                             for class_group in class_groups]
            obs_props.append(obs_props_cat)

        obs_props = np.array(obs_props)

        return mean_abilities, obs_props

    def class_intervals_cats_items(self,
                                   abilities,
                                   difficulties,
                                   thresholds,
                                   severities,
                                   item=None,
                                   rater=None,
                                   shift=0,
                                   no_of_classes=5):

        if rater == 'none':
            rater = None

        if rater == 'zero':
            rater = None

        class_groups = [f'class_{class_no + 1}' for class_no in range(no_of_classes)]

        df = self.dataframe.copy()

        abil_df = pd.DataFrame()

        for item_ in self.dataframe.columns:
            abil_df[item_] = abilities

        if item is None:
            for item_ in self.dataframe.columns:
                abil_df.loc[:, item_] -= difficulties[item_]

        abil_dict = {}

        for rater_ in self.raters:
            abil_dict[rater_] = abil_df.copy()

            if rater is None:
                for item_ in self.dataframe.columns:
                    abil_dict[rater_].loc[:, item_] -= severities[rater_][item_]

        abil_df = pd.concat(abil_dict.values(), keys=abil_dict.keys())

        if item is None:
            if rater is None:
                df = df
                df_mask = df.notna().astype(float).replace(0, np.nan)
                abil_df = abil_df * df_mask

                mask_scores = df.unstack().unstack()
                mask_abils = abil_df.unstack().unstack()

            else:
                df = df.xs(rater)
                df_mask = df.notna().astype(float).replace(0, np.nan)
                abil_df = abil_df.xs(rater) * df_mask

                mask_scores = df.unstack()
                mask_abils = abil_df.unstack()

        else:
            if rater is None:
                df = df[item].unstack(level=0)
                df_mask = df.notna().astype(float).replace(0, np.nan)
                abil_df = abil_df[item].unstack(level=0) * df_mask

                mask_scores = df.unstack()
                mask_abils = abil_df.unstack()

            else:
                df = df[item].xs(rater)
                df_mask = df.notna().astype(float).replace(0, np.nan)
                abil_df = abil_df[item].xs(rater) * df_mask

                mask_scores = df
                mask_abils = abil_df

        def class_masks(abils):
            mask_dict = {}

            quantiles = (abils.quantile([(i + 1) / no_of_classes
                                         for i in range(no_of_classes - 1)]))

            mask_dict['class_1'] = (abils < quantiles.values[0])
            mask_dict[f'class_{no_of_classes}'] = (abils >= quantiles.values[no_of_classes - 2])
            for class_group in range(no_of_classes - 2):
                mask_dict[f'class_{class_group + 2}'] = ((abils >= quantiles.values[class_group]) &
                                                         (abils < quantiles.values[class_group + 1]))

            for class_group in class_groups:
                mask_dict[class_group] = mask_dict[class_group][mask_dict[class_group]].index

            return mask_dict

        mask = class_masks(mask_abils)
        mean_abilities = [mask_abils.loc[mask[class_group]].mean()
                          for class_group in class_groups]
        mean_abilities = np.array(mean_abilities) - shift

        obs_props = []

        for category in range(self.max_score + 1):
            obs_props_cat = [len(mask_scores.loc[mask[class_group]][mask_scores == category]) / len(
                mask_scores.loc[mask[class_group]])
                             for class_group in class_groups]
            obs_props.append(obs_props_cat)

        obs_props = np.array(obs_props)

        return mean_abilities, obs_props

    def class_intervals_cats_thresholds(self,
                                        abilities,
                                        difficulties,
                                        thresholds,
                                        severities,
                                        item=None,
                                        rater=None,
                                        shift=0,
                                        no_of_classes=5):

        if rater == 'none':
            rater = None

        if rater == 'zero':
            rater = None

        class_groups = [f'class_{class_no + 1}' for class_no in range(no_of_classes)]

        df = self.dataframe.copy()

        abil_df = pd.DataFrame()

        for item_ in self.dataframe.columns:
            abil_df[item_] = abilities

        if item is None:
            for item_ in self.dataframe.columns:
                abil_df.loc[:, item_] -= difficulties[item_]

        abil_dict = {}

        for rater_ in self.raters:
            abil_dict[rater_] = abil_df.copy()

            if rater is None:
                abil_dict[rater_] -= severities[rater_][1:].mean()

        abil_df = pd.concat(abil_dict.values(), keys=abil_dict.keys())

        if item is None:
            if rater is None:
                df = df
                df_mask = df.notna().astype(float).replace(0, np.nan)
                abil_df = abil_df * df_mask

                mask_scores = df.unstack().unstack()
                mask_abils = abil_df.unstack().unstack()

            else:
                df = df.xs(rater)
                df_mask = df.notna().astype(float).replace(0, np.nan)
                abil_df = abil_df.xs(rater) * df_mask

                mask_scores = df.unstack()
                mask_abils = abil_df.unstack()

        else:
            if rater is None:
                df = df[item].unstack(level=0)
                df_mask = df.notna().astype(float).replace(0, np.nan)
                abil_df = abil_df[item].unstack(level=0) * df_mask

                mask_scores = df.unstack()
                mask_abils = abil_df.unstack()

            else:
                df = df[item].xs(rater)
                df_mask = df.notna().astype(float).replace(0, np.nan)
                abil_df = abil_df[item].xs(rater) * df_mask

                mask_scores = df
                mask_abils = abil_df

        def class_masks(abils):
            mask_dict = {}

            quantiles = (abils.quantile([(i + 1) / no_of_classes
                                         for i in range(no_of_classes - 1)]))

            mask_dict['class_1'] = (abils < quantiles.values[0])
            mask_dict[f'class_{no_of_classes}'] = (abils >= quantiles.values[no_of_classes - 2])
            for class_group in range(no_of_classes - 2):
                mask_dict[f'class_{class_group + 2}'] = ((abils >= quantiles.values[class_group]) &
                                                         (abils < quantiles.values[class_group + 1]))

            for class_group in class_groups:
                mask_dict[class_group] = mask_dict[class_group][mask_dict[class_group]].index

            return mask_dict

        mask = class_masks(mask_abils)
        mean_abilities = [mask_abils.loc[mask[class_group]].mean()
                          for class_group in class_groups]
        mean_abilities = np.array(mean_abilities)

        obs_props = []

        for category in range(self.max_score + 1):
            obs_props_cat = [len(mask_scores.loc[mask[class_group]][mask_scores == category]) / len(
                mask_scores.loc[mask[class_group]])
                             for class_group in class_groups]
            obs_props.append(obs_props_cat)

        obs_props = np.array(obs_props)

        return mean_abilities, obs_props

    def class_intervals_cats_matrix(self,
                                    abilities,
                                    difficulties,
                                    thresholds,
                                    severities,
                                    item=None,
                                    rater=None,
                                    shift=0,
                                    no_of_classes=5):

        if rater == 'none':
            rater = None

        if rater == 'zero':
            rater = None

        class_groups = [f'class_{class_no + 1}' for class_no in range(no_of_classes)]

        df = self.dataframe.copy()

        abil_df = pd.DataFrame()

        for item_ in self.dataframe.columns:
            abil_df[item_] = abilities

        if item is None:
            for item_ in self.dataframe.columns:
                abil_df.loc[:, item_] -= difficulties[item_]

        abil_dict = {}

        for rater_ in self.raters:
            abil_dict[rater_] = abil_df.copy()

            if rater is None:
                for item_ in self.dataframe.columns:
                    abil_dict[rater_][item_] -= severities[rater_][item_][1:].mean()

        abil_df = pd.concat(abil_dict.values(), keys=abil_dict.keys())

        if item is None:
            if rater is None:
                df = df
                df_mask = df.notna().astype(float).replace(0, np.nan)
                abil_df = abil_df * df_mask

                mask_scores = df.unstack().unstack()
                mask_abils = abil_df.unstack().unstack()

            else:
                df = df.xs(rater)
                df_mask = df.notna().astype(float).replace(0, np.nan)
                abil_df = abil_df.xs(rater) * df_mask

                mask_scores = df.unstack()
                mask_abils = abil_df.unstack()

        else:
            if rater is None:
                df = df[item].unstack(level=0)
                df_mask = df.notna().astype(float).replace(0, np.nan)
                abil_df = abil_df[item].unstack(level=0) * df_mask

                mask_scores = df.unstack()
                mask_abils = abil_df.unstack()

            else:
                df = df[item].xs(rater)
                df_mask = df.notna().astype(float).replace(0, np.nan)
                abil_df = abil_df[item].xs(rater) * df_mask

                mask_scores = df
                mask_abils = abil_df

        def class_masks(abils):
            mask_dict = {}

            quantiles = (abils.quantile([(i + 1) / no_of_classes
                                         for i in range(no_of_classes - 1)]))

            mask_dict['class_1'] = (abils < quantiles.values[0])
            mask_dict[f'class_{no_of_classes}'] = (abils >= quantiles.values[no_of_classes - 2])
            for class_group in range(no_of_classes - 2):
                mask_dict[f'class_{class_group + 2}'] = ((abils >= quantiles.values[class_group]) &
                                                         (abils < quantiles.values[class_group + 1]))

            for class_group in class_groups:
                mask_dict[class_group] = mask_dict[class_group][mask_dict[class_group]].index

            return mask_dict

        mask = class_masks(mask_abils)
        mean_abilities = [mask_abils.loc[mask[class_group]].mean()
                          for class_group in class_groups]
        mean_abilities = np.array(mean_abilities) - shift

        obs_props = []

        for category in range(self.max_score + 1):
            obs_props_cat = [len(mask_scores.loc[mask[class_group]][mask_scores == category]) / len(
                mask_scores.loc[mask[class_group]])
                             for class_group in class_groups]
            obs_props.append(obs_props_cat)

        obs_props = np.array(obs_props)

        return mean_abilities, obs_props

    def class_intervals_thr_global(self,
                                   abilities,
                                   difficulties,
                                   severities,
                                   item=None,
                                   rater=None,
                                   shift=None,
                                   no_of_classes=5):
        
        if item == 'none':
            item = None

        if rater == 'none':
            rater = None

        if rater == 'zero':
            rater = None

        if shift is None:
            shift = 0

        class_groups = [f'class_{class_no + 1}' for class_no in range(no_of_classes)]

        df = self.dataframe.copy()

        abil_df = pd.DataFrame()

        for item_ in self.dataframe.columns:
            abil_df[item_] = abilities

        if item is None:
            for item_ in self.dataframe.columns:
                abil_df.loc[:, item_] -= difficulties[item_]

        abil_dict = {}

        for rater_ in self.raters:
            abil_dict[rater_] = abil_df.copy()

            if rater is None:
                abil_dict[rater_] -= severities[rater_]

        abil_df = pd.concat(abil_dict.values(), keys=abil_dict.keys())

        if item is None:
            if rater is None:
                df = df
                abil_df = abil_df
    
            else:
                df = df.xs(rater)
                abil_df = abil_df.xs(rater)
                
        else:
            if rater is None:
                df = df[item].unstack(level=0)
                abil_df = abil_df[item].unstack(level=0)
    
            else:
                df = df[item].xs(rater)
                abil_df = abil_df[item].xs(rater)

        def class_masks(abils):

            quantiles = (abils.quantile([(i + 1) / no_of_classes
                                         for i in range(no_of_classes - 1)]))

            mask_dict = {}

            mask_dict['class_1'] = (abils < quantiles.values[0])
            mask_dict[f'class_{no_of_classes}'] = (abils >= quantiles.values[no_of_classes - 2])
            for class_group in range(no_of_classes - 2):
                mask_dict[f'class_{class_group + 2}'] = ((abils >= quantiles.values[class_group]) &
                                                         (abils < quantiles.values[class_group + 1]))

            for class_group in class_groups:
                mask_dict[class_group] = mask_dict[class_group][mask_dict[class_group]].index

            return mask_dict

        mean_abilities = []
        obs_props = []

        for threshold in range(self.max_score):
            cond_df = df[df.isin([threshold, threshold + 1])]
            cond_df -= threshold

            cond_df_mask = cond_df.notna().astype(float).replace(0, np.nan)

            cond_abils = abil_df * cond_df_mask

            obs_data_df = pd.DataFrame()

            if item is None:
                if rater is None:
                    obs_data_df['ability'] = cond_abils.stack()
                    obs_data_df['score'] = cond_df.stack()
                    obs_data_df = obs_data_df.droplevel(level=[0, 2])

                else:
                    obs_data_df['ability'] = cond_abils.stack()
                    obs_data_df['score'] = cond_df.stack()
                    obs_data_df = obs_data_df.droplevel(level=1)

            else:
                if rater is None:
                    obs_data_df['ability'] = cond_abils.stack()
                    obs_data_df['score'] = cond_df.stack()
                    obs_data_df = obs_data_df.droplevel(level=1)

                else:
                    obs_data_df['ability'] = cond_abils
                    obs_data_df['score'] = cond_df

            mask_dict = class_masks(obs_data_df['ability'])

            mean_abilities.append([obs_data_df.loc[mask_dict[class_group]]['ability'].mean() + shift
                                   for class_group in class_groups])

            obs_props.append([obs_data_df.loc[mask_dict[class_group]]['score'].mean()
                              for class_group in class_groups])

        mean_abilities = np.array(mean_abilities)
        obs_props = np.array(obs_props)

        return mean_abilities, obs_props

    def class_intervals_thr_items(self,
                                  abilities,
                                  difficulties,
                                  severities,
                                  item=None,
                                  rater=None,
                                  shift=None,
                                  no_of_classes=5):
        
        if item == 'none':
            item = None

        if rater == 'none':
            rater = None

        if rater == 'zero':
            rater = None

        if shift is None:
            shift = 0

        class_groups = [f'class_{class_no + 1}' for class_no in range(no_of_classes)]

        df = self.dataframe.copy()

        abil_df = pd.DataFrame()

        for item_ in self.dataframe.columns:
            abil_df[item_] = abilities

        if item is None:
            for item_ in self.dataframe.columns:
                abil_df.loc[:, item_] -= difficulties[item_]

        abil_dict = {}

        for rater_ in self.raters:
            abil_dict[rater_] = abil_df.copy()

            if rater is None:
                for item_ in self.dataframe.columns:
                    abil_dict[rater_].loc[:, item_] -= severities[rater_][item_]

        abil_df = pd.concat(abil_dict.values(), keys=abil_dict.keys())

        if item is None:
            if rater is None:
                df = df
                abil_df = abil_df

            else:
                df = df.xs(rater)
                abil_df = abil_df.xs(rater)

        else:
            if rater is None:
                df = df[item].unstack(level=0)
                abil_df = abil_df[item].unstack(level=0)

            else:
                df = df[item].xs(rater)
                abil_df = abil_df[item].xs(rater)

        def class_masks(abils):
            mask_dict = {}

            quantiles = (abils.quantile([(i + 1) / no_of_classes
                                         for i in range(no_of_classes - 1)]))

            mask_dict['class_1'] = (abils < quantiles.values[0])
            mask_dict[f'class_{no_of_classes}'] = (abils >= quantiles.values[no_of_classes - 2])
            for class_group in range(no_of_classes - 2):
                mask_dict[f'class_{class_group + 2}'] = ((abils >= quantiles.values[class_group]) &
                                                         (abils < quantiles.values[class_group + 1]))

            for class_group in class_groups:
                mask_dict[class_group] = mask_dict[class_group][mask_dict[class_group]].index

            return mask_dict

        mean_abilities = []
        obs_props = []

        for threshold in range(self.max_score):
            cond_df = df[df.isin([threshold, threshold + 1])]
            cond_df -= threshold

            cond_df_mask = cond_df.notna().astype(float).replace(0, np.nan)

            cond_abils = abil_df * cond_df_mask

            obs_data_df = pd.DataFrame()

            if item is None:
                if rater is None:
                    obs_data_df['ability'] = cond_abils.stack()
                    obs_data_df['score'] = cond_df.stack()
                    obs_data_df = obs_data_df.droplevel(level=[0, 2])

                else:
                    obs_data_df['ability'] = cond_abils.stack()
                    obs_data_df['score'] = cond_df.stack()
                    obs_data_df = obs_data_df.droplevel(level=1)

            else:
                if rater is None:
                    obs_data_df['ability'] = cond_abils.stack()
                    obs_data_df['score'] = cond_df.stack()
                    obs_data_df = obs_data_df.droplevel(level=1)

                else:
                    obs_data_df['ability'] = cond_abils
                    obs_data_df['score'] = cond_df

            mask_dict = class_masks(obs_data_df['ability'])

            mean_abilities.append([obs_data_df.loc[mask_dict[class_group]]['ability'].mean() + shift
                                   for class_group in class_groups])

            obs_props.append([obs_data_df.loc[mask_dict[class_group]]['score'].mean()
                              for class_group in class_groups])

        mean_abilities = np.array(mean_abilities)
        obs_props = np.array(obs_props)

        return mean_abilities, obs_props

    def class_intervals_thr_thresholds(self,
                                       abilities,
                                       difficulties,
                                       severities,
                                       item=None,
                                       rater=None,
                                       shifts=None,
                                       no_of_classes=5):
        
        if item == 'none':
            item = None

        if rater == 'none':
            rater = None

        if rater == 'zero':
            rater = None

        if shifts is None:
            shifts = np.zeros(self.max_score)

        class_groups = [f'class_{class_no + 1}' for class_no in range(no_of_classes)]

        df = self.dataframe.copy()

        abil_df = pd.DataFrame()

        for item_ in self.dataframe.columns:
            abil_df[item_] = abilities

        if item is None:
            for item_ in self.dataframe.columns:
                abil_df.loc[:, item_] -= difficulties[item_]

        abil_dict = {}

        for rater_ in self.raters:
            abil_dict[rater_] = abil_df.copy()

            if rater is None:
                abil_dict[rater_] -= severities[rater_][1:].mean()

        abil_df = pd.concat(abil_dict.values(), keys=abil_dict.keys())

        if item is None:
            if rater is None:
                df = df
                abil_df = abil_df

            else:
                df = df.xs(rater)
                abil_df = abil_df.xs(rater)

        else:
            if rater is None:
                df = df[item].unstack(level=0)
                abil_df = abil_df[item].unstack(level=0)

            else:
                df = df[item].xs(rater)
                abil_df = abil_df[item].xs(rater)

        def class_masks(abils):
            mask_dict = {}

            quantiles = (abils.quantile([(i + 1) / no_of_classes
                                         for i in range(no_of_classes - 1)]))

            mask_dict['class_1'] = (abils < quantiles.values[0])
            mask_dict[f'class_{no_of_classes}'] = (abils >= quantiles.values[no_of_classes - 2])
            for class_group in range(no_of_classes - 2):
                mask_dict[f'class_{class_group + 2}'] = ((abils >= quantiles.values[class_group]) &
                                                         (abils < quantiles.values[class_group + 1]))

            for class_group in class_groups:
                mask_dict[class_group] = mask_dict[class_group][mask_dict[class_group]].index

            return mask_dict

        mean_abilities = []
        obs_props = []

        for threshold in range(self.max_score):
            cond_df = df[df.isin([threshold, threshold + 1])]
            cond_df -= threshold

            cond_df_mask = cond_df.notna().astype(float).replace(0, np.nan)

            cond_abils = abil_df * cond_df_mask

            obs_data_df = pd.DataFrame()

            if item is None:
                if rater is None:
                    obs_data_df['ability'] = cond_abils.stack()
                    obs_data_df['score'] = cond_df.stack()
                    obs_data_df = obs_data_df.droplevel(level=[0, 2])

                else:
                    obs_data_df['ability'] = cond_abils.stack()
                    obs_data_df['score'] = cond_df.stack()
                    obs_data_df = obs_data_df.droplevel(level=1)

            else:
                if rater is None:
                    obs_data_df['ability'] = cond_abils.stack()
                    obs_data_df['score'] = cond_df.stack()
                    obs_data_df = obs_data_df.droplevel(level=1)

                else:
                    obs_data_df['ability'] = cond_abils
                    obs_data_df['score'] = cond_df

            mask_dict = class_masks(obs_data_df['ability'])

            mean_abilities.append([obs_data_df.loc[mask_dict[class_group]]['ability'].mean() - shifts[threshold + 1]
                                   for class_group in class_groups])

            obs_props.append([obs_data_df.loc[mask_dict[class_group]]['score'].mean()
                              for class_group in class_groups])

        mean_abilities = np.array(mean_abilities)
        obs_props = np.array(obs_props)

        return mean_abilities, obs_props

    def class_intervals_thr_matrix(self,
                                   abilities,
                                   difficulties,
                                   severities,
                                   item=None,
                                   rater=None,
                                   shifts=None,
                                   no_of_classes=5):
        
        if item == 'none':
            item = None

        if rater == 'none':
            rater = None

        if rater == 'zero':
            rater = None
            
        if shifts is None:
            shifts = np.zeros(self.max_score)

        class_groups = [f'class_{class_no + 1}' for class_no in range(no_of_classes)]

        df = self.dataframe.copy()

        abil_df = pd.DataFrame()

        for item_ in self.dataframe.columns:
            abil_df[item_] = abilities

        if item is None:
            for item_ in self.dataframe.columns:
                abil_df.loc[:, item_] -= difficulties[item_]

        abil_dict = {}

        for rater_ in self.raters:
            abil_dict[rater_] = abil_df.copy()

            if rater is None:
                for item_ in self.dataframe.columns:
                    abil_dict[rater_][item_] -= severities[rater_][item_][1:].mean()

        abil_df = pd.concat(abil_dict.values(), keys=abil_dict.keys())

        if item is None:
            if rater is None:
                df = df
                abil_df = abil_df

            else:
                df = df.xs(rater)
                abil_df = abil_df.xs(rater)

        else:
            if rater is None:
                df = df[item].unstack(level=0)
                abil_df = abil_df[item].unstack(level=0)

            else:
                df = df[item].xs(rater)
                abil_df = abil_df[item].xs(rater)

        def class_masks(abils):

            quantiles = (abils.quantile([(i + 1) / no_of_classes
                                         for i in range(no_of_classes - 1)]))

            mask_dict = {}

            mask_dict['class_1'] = (abils < quantiles.values[0])
            mask_dict[f'class_{no_of_classes}'] = (abils >= quantiles.values[no_of_classes - 2])
            for class_group in range(no_of_classes - 2):
                mask_dict[f'class_{class_group + 2}'] = ((abils >= quantiles.values[class_group]) &
                                                         (abils < quantiles.values[class_group + 1]))

            for class_group in class_groups:
                mask_dict[class_group] = mask_dict[class_group][mask_dict[class_group]].index

            return mask_dict

        mean_abilities = []
        obs_props = []

        for threshold in range(self.max_score):
            cond_df = df[df.isin([threshold, threshold + 1])]
            cond_df -= threshold

            cond_df_mask = cond_df.notna().astype(float).replace(0, np.nan)

            cond_abils = abil_df * cond_df_mask

            obs_data_df = pd.DataFrame()

            if item is None:
                if rater is None:
                    obs_data_df['ability'] = cond_abils.stack()
                    obs_data_df['score'] = cond_df.stack()
                    obs_data_df = obs_data_df.droplevel(level=[0, 2])

                else:
                    obs_data_df['ability'] = cond_abils.stack()
                    obs_data_df['score'] = cond_df.stack()
                    obs_data_df = obs_data_df.droplevel(level=1)

            else:
                if rater is None:
                    obs_data_df['ability'] = cond_abils.stack()
                    obs_data_df['score'] = cond_df.stack()
                    obs_data_df = obs_data_df.droplevel(level=1)

                else:
                    obs_data_df['ability'] = cond_abils
                    obs_data_df['score'] = cond_df

            mask_dict = class_masks(obs_data_df['ability'])

            mean_abilities.append([obs_data_df.loc[mask_dict[class_group]]['ability'].mean() - shifts[threshold + 1]
                                   for class_group in class_groups])

            obs_props.append([obs_data_df.loc[mask_dict[class_group]]['score'].mean()
                              for class_group in class_groups])

        mean_abilities = np.array(mean_abilities)
        obs_props = np.array(obs_props)

        return mean_abilities, obs_props

    '''
    *** PLOTS ***
    '''

    def plot_data_global(self,
                         x_data,
                         y_data,
                         anchor=False,
                         items=None,
                         raters=None,
                         obs=None,
                         thresh_obs=None,
                         x_obs_data=np.array([]),
                         y_obs_data=np.array([]),
                         thresh_lines=False,
                         central_diff=False,
                         cat_highlight=None,
                         score_lines_item=[None, None],
                         score_lines_test=None,
                         point_info_lines_item=[None, None],
                         point_info_lines_test=None,
                         point_csem_lines=None,
                         score_labels=False,
                         x_min=-5,
                         x_max=5,
                         y_max=0,
                         warm=True,
                         graph_title='',
                         y_label='',
                         plot_style='white',
                         palette='dark blue',
                         black=False,
                         figsize=(8, 6),
                         font='Times New Roman',
                         title_font_size=15,
                         axis_font_size=12,
                         labelsize=12,
                         tex=True,
                         plot_density=300,
                         filename=None,
                         file_format='png'):

        '''
        Basic plotting function to be called when plotting specific functions
        of person ability for RSM.
        '''

        if anchor:
            difficulties = self.anchor_diffs_global
            thresholds = self.anchor_thresholds_global
            severities = self.anchor_severities_global

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_global

        if isinstance(raters, str):
            if raters == 'none':
                raters = None

            elif raters == 'all':
                raters = self.raters.tolist()

        if isinstance(items, str):
            if items == 'all':
                items = None

            elif items == 'none':
                items = None

        dummy_sevs = pd.Series({'dummy_rater': 0})

        if plot_style == 'dark':
            sns.set_style('darkgrid')

        else:
            sns.set_style('whitegrid')

        palette_dict = {'dark blue': ['dark', 'royalblue'],
                        'light blue': ['light', 'cornflowerblue'],
                        'dark red': ['dark', 'firebrick'],
                        'light red': ['light', 'indianred'],
                        'dark green': ['dark', 'forestgreen'],
                        'light green': ['light', 'mediumseagreen'],
                        'dark grey': ['dark', 'dimgrey'],
                        'light grey': ['light', 'darkgrey'],
                        'dark multi': ['dark', 'dark'],
                        'light multi': ['light', 'muted']}

        if palette_dict[palette][0] == 'dark':
            if palette == 'dark multi':
                color_map = sns.color_palette('dark', as_cmap=True)
            else:
                color_map = sns.dark_palette(palette_dict[palette][1], reverse=True, as_cmap=True)

        if palette_dict[palette][0] == 'light':
            if palette == 'light multi':
                color_map = sns.color_palette('muted', as_cmap=True)
            else:
                color_map = sns.light_palette(palette_dict[palette][1], reverse=True, as_cmap=True)

        graph, ax = plt.subplots(figsize=figsize)

        no_of_plots = y_data.shape[1]

        cNorm = colors.Normalize(vmin=0, vmax=no_of_plots + 2)

        if 'multi' not in palette:
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=color_map)

        if black:
            for i in range(no_of_plots):
                ax.plot(x_data, y_data[:, i], '', label=i+1, color='black')

        else:
            for i in range(no_of_plots):
                if 'multi' not in palette:
                    colorVal = scalarMap.to_rgba(i)
                else:
                    colorVal = color_map[i]

                ax.plot(x_data, y_data[:, i], '', color=colorVal, label=i+1)

        if obs is not None:
            try:
                if isinstance(y_obs_data, pd.Series):
                    if 'multi' not in palette:
                        colorVal = scalarMap.to_rgba(0)
                    else:
                        colorVal = color_map[0]

                    ax.plot(x_obs_data, y_obs_data, 'o', color=colorVal)

                else:
                    no_of_observed_cats = y_obs_data.shape[1]
                    for j in range (no_of_observed_cats):
                        if 'multi' not in palette:
                            colorVal = scalarMap.to_rgba(j)
                        else:
                            colorVal = color_map[j]

                        ax.plot(x_obs_data, y_obs_data[:, j], 'o', color=colorVal)

            except:
                pass

        if thresh_obs is not None:
            if thresh_obs == 'all':
                thresh_obs = np.arange(self.max_score + 1)
            try:
                for ob in thresh_obs:
                    if 'multi' not in palette:
                        colorVal = scalarMap.to_rgba(ob)
                    else:
                        colorVal = color_map[ob]

                    ax.plot(x_obs_data[ob - 1, :], y_obs_data[ob - 1, :], 'o', color=colorVal)

            except:
                pass

        if thresh_lines:
            if items is None:
                if raters is None:
                    for threshold in range(self.max_score):
                        plt.axvline(x=thresholds[threshold + 1],
                                    color='black', linestyle='--')

                else:
                    for threshold in range(self.max_score):
                        plt.axvline(x=thresholds[threshold + 1] + severities[raters],
                                    color='black', linestyle='--')

            else:
                if raters is None:
                    for threshold in range(self.max_score):
                        plt.axvline(x=thresholds[threshold + 1] + difficulties[items],
                                    color='black', linestyle='--')

                else:
                    for threshold in range(self.max_score):
                        plt.axvline(x=thresholds[threshold + 1] + difficulties[items] + severities[raters],
                                    color='black', linestyle='--')

        if central_diff:
            if items is None:
                if raters is None:
                    plt.axvline(x=0, color='darkred', linestyle='--')

                else:
                    plt.axvline(x=severities[raters], color='darkred', linestyle='--')

            else:
                if raters is None:
                    plt.axvline(x=difficulties[items], color='darkred', linestyle='--')

                else:
                    plt.axvline(x=difficulties[items] + severities[raters], color='darkred', linestyle='--')

        if score_lines_item[1] is not None:

            if (all(x > 0 for x in score_lines_item[1]) &
                all(x < self.max_score for x in score_lines_item[1])):

                abils_set = [self.score_abil_global(score, anchor=anchor, items=items, raters=raters, warm_corr=False)
                             for score in score_lines_item[1]]

                for thresh, abil in zip(score_lines_item[1], abils_set):
                    plt.vlines(x=abil, ymin=-100, ymax=thresh, color='black', linestyles='dashed')
                    if score_labels:
                        plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                    plt.hlines(y=thresh, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                    if score_labels:
                        plt.text(x_min + (x_max - x_min) / 100, thresh + y_max / 50, str(thresh))

            else:
                print('Invalid score for score line.')

        if score_lines_test is not None:

            if isinstance(raters, list):
                no_of_raters = len(raters)

            else:
                no_of_raters = 1

            if items is None:
                no_of_items = self.no_of_items

            else:
                if isinstance(items, list):
                    no_of_items = len(items)

                else:
                    no_of_items = 1

            if (all(x > 0 for x in score_lines_test) &
                all(x < self.max_score * no_of_items * no_of_raters for x in score_lines_test)):

                abils_set = [self.score_abil_global(score, anchor=anchor, items=items, raters=raters, warm_corr=False)
                             for score in score_lines_test]

                for thresh, abil in zip(score_lines_test, abils_set):
                    plt.vlines(x=abil, ymin=-100, ymax=thresh, color='black', linestyles='dashed')
                    if score_labels:
                        plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                    plt.hlines(y=thresh, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                    if score_labels:
                        plt.text(x_min + (x_max - x_min) / 100, thresh + y_max / 50, str(thresh))

            else:
                print('Invalid score for score line.')

        if point_info_lines_item[1] is not None:

            item = point_info_lines_item[0]

            if raters is None:
                info_set = [self.variance_global(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                            for ability in point_info_lines_item[1]]

            else:
                info_set = [self.variance_global(ability, item, difficulties, raters, severities, thresholds)
                            for ability in point_info_lines_item[1]]

            for abil, info in zip(point_info_lines_item[1], info_set):
                plt.vlines(x=abil, ymin=-100, ymax=info, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=info, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(x_min + (x_max - x_min) / 100, info + y_max / 50, str(round(info, 3)))

        if point_info_lines_test is not None:

            if items is None:
                if raters is None:
                    info_set = [sum(self.variance_global(ability, item, difficulties, 'dummy_rater', dummy_sevs,
                                                         thresholds)
                                    for item in self.dataframe.columns)
                                for ability in point_info_lines_test]

                else:
                    info_set = [sum(self.variance_global(ability, item, difficulties, rater, severities, thresholds)
                                    for item in self.dataframe.columns for rater in raters)
                                for ability in point_info_lines_test]
                    
            else:
                if raters is None:
                    info_set = [sum(self.variance_global(ability, item, difficulties, 'dummy_rater', dummy_sevs,
                                                         thresholds)
                                    for item in items)
                                for ability in point_info_lines_test]

                else:
                    info_set = [sum(self.variance_global(ability, item, difficulties, rater, severities, thresholds)
                                    for item in items for rater in raters)
                                for ability in point_info_lines_test]

            for abil, info in zip(point_info_lines_test, info_set):
                plt.vlines(x=abil, ymin=-100, ymax=info, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=info, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(x_min + (x_max - x_min) / 100, info + y_max / 50, str(round(info, 3)))

        if point_csem_lines is not None:

            if items is None:
                if raters is None:
                    info_set = [sum(self.variance_global(ability, item, difficulties, 'dummy_rater', dummy_sevs,
                                                         thresholds)
                                    for item in self.dataframe.columns)
                                for ability in point_csem_lines]

                else:
                    info_set = [sum(self.variance_global(ability, item, difficulties, rater, severities, thresholds)
                                    for item in self.dataframe.columns for rater in raters)
                                for ability in point_csem_lines]
                    
            else:
                if raters is None:
                    info_set = [sum(self.variance_global(ability, item, difficulties, 'dummy_rater', dummy_sevs,
                                                         thresholds)
                                    for item in items)
                                for ability in point_csem_lines]

                else:
                    info_set = [sum(self.variance_global(ability, item, difficulties, rater, severities, thresholds)
                                    for item in items for rater in raters)
                                for ability in point_csem_lines]
            
            info_set = np.array(info_set)
            csem_set = 1 / (info_set ** 0.5)

            for abil, csem in zip(point_csem_lines, csem_set):
                plt.vlines(x=abil, ymin=-100, ymax=csem, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=csem, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(x_min + (x_max - x_min) / 100, csem + y_max / 50, str(round(csem, 3)))

        if cat_highlight in range(self.max_score + 1):

            if cat_highlight == 0:
                if items is None:
                    if raters is None:
                        plt.axvspan(-100, thresholds[1],
                                    facecolor='blue', alpha=0.2)

                    else:
                        plt.axvspan(-100, thresholds[1] + severities[raters],
                                    facecolor='blue', alpha=0.2)

                else:
                    if raters is None:
                        plt.axvspan(-100, difficulties[items] + thresholds[1],
                                    facecolor='blue', alpha=0.2)

                    else:
                        plt.axvspan(-100, difficulties[items] + thresholds[1] + severities[raters],
                                    facecolor='blue', alpha=0.2)

            elif cat_highlight == self.max_score:
                if items is None:
                    if raters is None:
                        plt.axvspan(thresholds[self.max_score],
                                    100, facecolor='blue', alpha=0.2)

                    else:
                        plt.axvspan(thresholds[self.max_score] + severities[raters],
                                    100, facecolor='blue', alpha=0.2)

                else:
                    if raters is None:
                        plt.axvspan(difficulties[items] + thresholds[self.max_score],
                                    100, facecolor='blue', alpha=0.2)

                    else:
                        plt.axvspan(difficulties[items] + thresholds[self.max_score] + severities[raters],
                                    100, facecolor='blue', alpha=0.2)

            else:
                if (thresholds[cat_highlight + 1] >
                    thresholds[cat_highlight]):
                    if items is None:
                        if raters is None:
                            plt.axvspan(thresholds[cat_highlight],
                                        thresholds[cat_highlight + 1],
                                        facecolor='blue', alpha=0.2)

                        else:
                            plt.axvspan(thresholds[cat_highlight] + severities[raters],
                                        thresholds[cat_highlight + 1] + severities[raters],
                                        facecolor='blue', alpha=0.2)
                    else:
                        if raters is None:
                            plt.axvspan(difficulties[items] + thresholds[cat_highlight],
                                        difficulties[items] + thresholds[cat_highlight + 1],
                                        facecolor='blue', alpha=0.2)

                        else:
                            plt.axvspan(difficulties[items] + thresholds[cat_highlight] + severities[raters],
                                        difficulties[items] + thresholds[cat_highlight + 1] + severities[raters],
                                        facecolor='blue', alpha=0.2)

        if y_max <= 0:
            y_max = y_data.max() * 1.1

        plt.xlim(x_min, x_max)
        plt.ylim(0, y_max)

        plt.xlabel('Ability', fontsize=axis_font_size, fontweight='bold')
        plt.ylabel(y_label, fontsize=axis_font_size, fontweight='bold')
        plt.title(graph_title, fontsize=title_font_size, fontweight='bold')

        plt.grid(True)

        plt.tick_params(axis="x", labelsize=labelsize)
        plt.tick_params(axis="y", labelsize=labelsize)

        if filename is not None:
            plt.savefig(f'{filename}.{file_format}', dpi=plot_density)

        # Close before returning -- Jupyter auto-displays any live figure
        # returned from a cell, which combined with plt.show() gives two plots.
        # Closing here means Jupyter gets a closed figure object which it will
        # not auto-display, so only one plot appears.
        plt.close(graph)

        return graph

    def plot_data_items(self,
                        x_data,
                        y_data,
                        anchor=False,
                        items=None,
                        raters=None,
                        item=None,
                        obs=None,
                        thresh_obs=None,
                        x_obs_data=np.array([]),
                        y_obs_data=np.array([]),
                        thresh_lines=False,
                        central_diff=False,
                        cat_highlight=None,
                        score_lines_item=[None, None],
                        score_lines_test=None,
                        point_info_lines_item=[None, None],
                        point_info_lines_test=None,
                        point_csem_lines=None,
                        score_labels=False,
                        x_min=-5,
                        x_max=5,
                        y_max=0,
                        warm=True,
                        graph_title='',
                        y_label='',
                        plot_style='white',
                        palette='dark blue',
                        black=False,
                        figsize=(8, 6),
                        font='Times New Roman',
                        title_font_size=15,
                        axis_font_size=12,
                        labelsize=12,
                        tex=True,
                        plot_density=300,
                        filename=None,
                        file_format='png'):

        '''
        Basic plotting function to be called when plotting specific functions
        of person ability for RSM.
        '''

        if anchor:
            difficulties = self.anchor_diffs_items
            thresholds = self.anchor_thresholds_items
            severities = self.anchor_severities_items

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_items

        if isinstance(raters, str):
            if raters == 'none':
                raters = None

            elif raters == 'all':
                raters = self.raters.tolist()

        if isinstance(items, str):
            if items == 'all':
                items = None

            elif items == 'none':
                items = None

        dummy_sevs = {'dummy_rater': {item: 0 for item in self.dataframe.columns}}

        if plot_style == 'dark':
            sns.set_style('darkgrid')

        else:
            sns.set_style('whitegrid')

        palette_dict = {'dark blue': ['dark', 'royalblue'],
                        'light blue': ['light', 'cornflowerblue'],
                        'dark red': ['dark', 'firebrick'],
                        'light red': ['light', 'indianred'],
                        'dark green': ['dark', 'forestgreen'],
                        'light green': ['light', 'mediumseagreen'],
                        'dark grey': ['dark', 'dimgrey'],
                        'light grey': ['light', 'darkgrey'],
                        'dark multi': ['dark', 'dark'],
                        'light multi': ['light', 'muted']}

        if palette_dict[palette][0] == 'dark':
            if palette == 'dark multi':
                color_map = sns.color_palette('dark', as_cmap=True)
            else:
                color_map = sns.dark_palette(palette_dict[palette][1], reverse=True, as_cmap=True)

        if palette_dict[palette][0] == 'light':
            if palette == 'light multi':
                color_map = sns.color_palette('muted', as_cmap=True)
            else:
                color_map = sns.light_palette(palette_dict[palette][1], reverse=True, as_cmap=True)

        graph, ax = plt.subplots(figsize=figsize)

        no_of_plots = y_data.shape[1]

        cNorm = colors.Normalize(vmin=0, vmax=no_of_plots + 2)

        if 'multi' not in palette:
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=color_map)

        if black:
            for i in range(no_of_plots):
                ax.plot(x_data, y_data[:, i], '', label=i+1, color='black')

        else:
            for i in range(no_of_plots):
                if 'multi' not in palette:
                    colorVal = scalarMap.to_rgba(i)
                else:
                    colorVal = color_map[i]

                ax.plot(x_data, y_data[:, i], '', color=colorVal, label=i+1)

        if obs is not None:
            try:
                if isinstance(y_obs_data, pd.Series):
                    if 'multi' not in palette:
                        colorVal = scalarMap.to_rgba(0)
                    else:
                        colorVal = color_map[0]

                    ax.plot(x_obs_data, y_obs_data, 'o', color=colorVal)

                else:
                    no_of_observed_cats = y_obs_data.shape[1]
                    for j in range (no_of_observed_cats):
                        if 'multi' not in palette:
                            colorVal = scalarMap.to_rgba(j)
                        else:
                            colorVal = color_map[j]

                        ax.plot(x_obs_data, y_obs_data[:, j], 'o', color=colorVal)

            except:
                pass

        if thresh_obs is not None:
            if thresh_obs == 'all':
                thresh_obs = np.arange(self.max_score + 1)
            try:
                for ob in thresh_obs:
                    if 'multi' not in palette:
                        colorVal = scalarMap.to_rgba(ob)
                    else:
                        colorVal = color_map[ob]

                    ax.plot(x_obs_data[ob - 1, :], y_obs_data[ob - 1, :], 'o', color=colorVal)

            except:
                pass

        if thresh_lines:
            if items is None:
                if raters is None:
                    for threshold in range(self.max_score):
                        plt.axvline(x=thresholds[threshold + 1],
                                    color='black', linestyle='--')

                else:
                    mean_sevs = pd.DataFrame(severities).mean()
                    for threshold in range(self.max_score):
                        plt.axvline(x=thresholds[threshold + 1] + mean_sevs[raters],
                                    color='black', linestyle='--')

            else:
                if raters is None:
                    for threshold in range(self.max_score):
                        plt.axvline(x=difficulties[items] + thresholds[threshold + 1],
                                    color='black', linestyle='--')

                else:
                    for threshold in range(self.max_score):
                        plt.axvline(x=difficulties[items] + thresholds[threshold + 1] + severities[raters][items],
                                    color='black', linestyle='--')

        if central_diff:
            if items is None:
                if raters is None:
                    plt.axvline(x=0, color='darkred', linestyle='--')

                else:
                    mean_sevs = pd.DataFrame(severities).mean()
                    plt.axvline(x=mean_sevs[raters], color='darkred', linestyle='--')

            else:
                if raters is None:
                    plt.axvline(x = difficulties[items], color='darkred', linestyle='--')

                else:
                    plt.axvline(x = difficulties[items] + severities[raters][items], color='darkred', linestyle='--')


        if score_lines_item[1] is not None:
            if (all(x > 0 for x in score_lines_item[1]) &
                all(x < self.max_score for x in score_lines_item[1])):

                abils_set = [self.score_abil_items(score, anchor=anchor, items=items, raters=raters, warm_corr=False)
                                 for score in score_lines_item[1]]

                for thresh, abil in zip(score_lines_item[1], abils_set):
                    plt.vlines(x=abil, ymin=-100, ymax=thresh, color='black', linestyles='dashed')
                    if score_labels:
                        plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                    plt.hlines(y=thresh, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                    if score_labels:
                        plt.text(x_min + (x_max - x_min) / 100, thresh + y_max / 50, str(thresh))

            else:
                print('Invalid score for score line.')

        if score_lines_test is not None:

            if isinstance(raters, list):
                no_of_raters = len(raters)

            else:
                no_of_raters = 1

            if items is None:
                no_of_items = self.no_of_items

            else:
                if isinstance(items, list):
                    no_of_items = len(items)

                else:
                    no_of_items = 1

            if (all(x > 0 for x in score_lines_test) &
                all(x < self.max_score * no_of_items * no_of_raters for x in score_lines_test)):

                abils_set = [self.score_abil_items(score, anchor=anchor, items=items, raters=raters, warm_corr=False)
                             for score in score_lines_test]

                for thresh, abil in zip(score_lines_test, abils_set):
                    plt.vlines(x=abil, ymin=-100, ymax=thresh, color='black', linestyles='dashed')
                    if score_labels:
                        plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                    plt.hlines(y=thresh, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                    if score_labels:
                        plt.text(x_min + (x_max - x_min) / 100, thresh + y_max / 50, str(thresh))

            else:
                print('Invalid score for score line.')

        if point_info_lines_item[1] is not None:

            item = point_info_lines_item[0]

            if raters is None:
                info_set = [self.variance_items(ability, item, difficulties, 'dummy_rater',
                                                dummy_sevs, thresholds)
                            for ability in point_info_lines_item[1]]

            else:
                info_set = [self.variance_items(ability, item, difficulties, raters, severities, thresholds)
                            for ability in point_info_lines_item[1]]

            for abil, info in zip(point_info_lines_item[1], info_set):
                plt.vlines(x=abil, ymin=-100, ymax=info, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=info, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(x_min + (x_max - x_min) / 100, info + y_max / 50, str(round(info, 3)))

        if point_info_lines_test is not None:

            if items is None:
                if raters is None:
                    info_set = [sum(self.variance_items(ability, item, difficulties, 'dummy_rater',
                                                        dummy_sevs, thresholds)
                                    for item in self.dataframe.columns)
                                for ability in point_info_lines_test]

                else:
                    info_set = [sum(self.variance_items(ability, item, difficulties, rater, severities, thresholds)
                                    for item in self.dataframe.columns for rater in raters)
                                for ability in point_info_lines_test]

            else:
                if raters is None:
                    info_set = [sum(self.variance_items(ability, item, difficulties, 'dummy_rater',
                                                        dummy_sevs, thresholds)
                                    for item in items)
                                for ability in point_info_lines_test]

                else:
                    info_set = [sum(self.variance_items(ability, item, difficulties, rater, severities, thresholds)
                                    for item in items for rater in raters)
                                for ability in point_info_lines_test]

            for abil, info in zip(point_info_lines_test, info_set):
                plt.vlines(x=abil, ymin=-100, ymax=info, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=info, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(x_min + (x_max - x_min) / 100, info + y_max / 50, str(round(info, 3)))

        if point_csem_lines is not None:

            if items is None:
                if raters is None:
                    info_set = [sum(self.variance_items(ability, item, difficulties, 'dummy_rater',
                                                        dummy_sevs, thresholds)
                                    for item in self.dataframe.columns)
                                for ability in point_csem_lines]

                else:
                    info_set = [sum(self.variance_items(ability, item, difficulties, rater, severities, thresholds)
                                    for item in self.dataframe.columns for rater in raters)
                                for ability in point_csem_lines]

            else:
                if raters is None:
                    info_set = [sum(self.variance_items(ability, item, difficulties, 'dummy_rater',
                                                         dummy_sevs, thresholds)
                                    for item in items)
                                for ability in point_csem_lines]

                else:
                    info_set = [sum(self.variance_items(ability, item, difficulties, rater, severities, thresholds)
                                    for item in items for rater in raters)
                                for ability in point_csem_lines]

            info_set = np.array(info_set)
            csem_set = 1 / (info_set ** 0.5)

            for abil, csem in zip(point_csem_lines, csem_set):
                plt.vlines(x=abil, ymin=-100, ymax=csem, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=csem, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(x_min + (x_max - x_min) / 100, csem + y_max / 50, str(round(csem, 3)))

        if cat_highlight in range(self.max_score + 1):

            if cat_highlight == 0:
                if items is None:
                    if raters is None:
                        plt.axvspan(-100, thresholds[1],
                                    facecolor='blue', alpha=0.2)

                    else:
                        plt.axvspan(-100, thresholds[1] + pd.Series(severities[raters]).mean(),
                                    facecolor='blue', alpha=0.2)

                else:
                    if raters is None:
                        plt.axvspan(-100, difficulties[items] + thresholds[1],
                                    facecolor='blue', alpha=0.2)

                    else:
                        plt.axvspan(-100, difficulties[items] + thresholds[1] + severities[raters][items],
                                    facecolor='blue', alpha=0.2)

            elif cat_highlight == self.max_score:
                if items is None:
                    if raters is None:
                        plt.axvspan(thresholds[self.max_score],
                                    100, facecolor='blue', alpha=0.2)

                    else:
                        plt.axvspan(thresholds[self.max_score] + severities[raters],
                                    100, facecolor='blue', alpha=0.2)

                else:
                    if raters is None:
                        plt.axvspan(difficulties[items] + thresholds[self.max_score],
                                    100, facecolor='blue', alpha=0.2)

                    else:
                        plt.axvspan(difficulties[items] + thresholds[self.max_score] + severities[raters],
                                    100, facecolor='blue', alpha=0.2)

            else:
                if thresholds[cat_highlight + 1] > thresholds[cat_highlight]:
                    if items is None:
                        if raters is None:
                            plt.axvspan(thresholds[cat_highlight],
                                        thresholds[cat_highlight + 1],
                                        facecolor='blue', alpha=0.2)

                        else:
                            mean_sev = np.mean([severities[raters][item] for item in self.dataframe.columns])

                            plt.axvspan(thresholds[cat_highlight] + mean_sev,
                                        thresholds[cat_highlight + 1] + mean_sev,
                                        facecolor='blue', alpha=0.2)
                    else:
                        if raters is None:
                            plt.axvspan(difficulties[items] + thresholds[cat_highlight],
                                        difficulties[items] + thresholds[cat_highlight + 1],
                                        facecolor='blue', alpha=0.2)

                        else:
                            plt.axvspan(difficulties[items] + thresholds[cat_highlight] + severities[raters][items],
                                        difficulties[items] + thresholds[cat_highlight + 1] + severities[raters][items],
                                        facecolor='blue', alpha=0.2)

        if y_max <= 0:
            y_max = y_data.max() * 1.1

        plt.xlim(x_min, x_max)
        plt.ylim(0, y_max)

        plt.xlabel('Ability', fontsize=axis_font_size, fontweight='bold')
        plt.ylabel(y_label, fontsize=axis_font_size, fontweight='bold')
        plt.title(graph_title, fontsize=title_font_size, fontweight='bold')

        plt.grid(True)

        plt.tick_params(axis="x", labelsize=labelsize)
        plt.tick_params(axis="y", labelsize=labelsize)

        if filename is not None:
            plt.savefig(f'{filename}.{file_format}', dpi=plot_density)

        # Close before returning -- Jupyter auto-displays any live figure
        # returned from a cell, giving two plots. Closing here prevents that.
        plt.close(graph)

        return graph

    def plot_data_thresholds(self,
                             x_data,
                             y_data,
                             anchor=False,
                             items=None,
                             raters=None,
                             obs=None,
                             thresh_obs=None,
                             x_obs_data=np.array([]),
                             y_obs_data=np.array([]),
                             thresh_lines=False,
                             central_diff=False,
                             cat_highlight=None,
                             score_lines_item=[None, None],
                             score_lines_test=None,
                             point_info_lines_item=[None, None],
                             point_info_lines_test=None,
                             point_csem_lines=None,
                             score_labels=False,
                             x_min=-5,
                             x_max=5,
                             y_max=0,
                             warm=True,
                             graph_title='',
                             y_label='',
                             plot_style='white',
                             palette='dark blue',
                             black=False,
                             figsize=(8, 6),
                             font='Times New Roman',
                             title_font_size=15,
                             axis_font_size=12,
                             labelsize=12,
                             tex=True,
                             plot_density=300,
                             filename=None,
                             file_format='png'):

        '''
        Basic plotting function to be called when plotting specific functions
        of person ability for RSM.
        '''

        if anchor:
            difficulties = self.anchor_diffs_thresholds
            thresholds = self.anchor_thresholds_thresholds
            severities = self.anchor_severities_thresholds

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_thresholds

        if isinstance(raters, str):
            if raters == 'none':
                raters = None

            elif raters == 'all':
                raters = self.raters.tolist()

        if isinstance(items, str):
            if items == 'all':
                items = None

            elif items == 'none':
                items = None

        dummy_sevs = {'dummy_rater': np.zeros(self.max_score + 1)}

        if plot_style == 'dark':
            sns.set_style('darkgrid')

        else:
            sns.set_style('whitegrid')

        palette_dict = {'dark blue': ['dark', 'royalblue'],
                        'light blue': ['light', 'cornflowerblue'],
                        'dark red': ['dark', 'firebrick'],
                        'light red': ['light', 'indianred'],
                        'dark green': ['dark', 'forestgreen'],
                        'light green': ['light', 'mediumseagreen'],
                        'dark grey': ['dark', 'dimgrey'],
                        'light grey': ['light', 'darkgrey'],
                        'dark multi': ['dark', 'dark'],
                        'light multi': ['light', 'muted']}

        if palette_dict[palette][0] == 'dark':
            if palette == 'dark multi':
                color_map = sns.color_palette('dark', as_cmap=True)
            else:
                color_map = sns.dark_palette(palette_dict[palette][1], reverse=True, as_cmap=True)

        if palette_dict[palette][0] == 'light':
            if palette == 'light multi':
                color_map = sns.color_palette('muted', as_cmap=True)
            else:
                color_map = sns.light_palette(palette_dict[palette][1], reverse=True, as_cmap=True)

        graph, ax = plt.subplots(figsize=figsize)

        no_of_plots = y_data.shape[1]

        cNorm = colors.Normalize(vmin=0, vmax=no_of_plots + 2)

        if 'multi' not in palette:
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=color_map)

        if black:
            for i in range(no_of_plots):
                ax.plot(x_data, y_data[:, i], '', label=i+1, color='black')

        else:
            for i in range(no_of_plots):
                if 'multi' not in palette:
                    colorVal = scalarMap.to_rgba(i)
                else:
                    colorVal = color_map[i]

                ax.plot(x_data, y_data[:, i], '', color=colorVal, label=i+1)

        if obs is not None:
            try:
                if isinstance(y_obs_data, pd.Series):
                    if 'multi' not in palette:
                        colorVal = scalarMap.to_rgba(0)
                    else:
                        colorVal = color_map[0]

                    ax.plot(x_obs_data, y_obs_data, 'o', color=colorVal)

                else:
                    no_of_observed_cats = y_obs_data.shape[1]
                    for j in range (no_of_observed_cats):
                        if 'multi' not in palette:
                            colorVal = scalarMap.to_rgba(j)
                        else:
                            colorVal = color_map[j]

                        ax.plot(x_obs_data, y_obs_data[:, j], 'o', color=colorVal)

            except:
                pass

        if thresh_obs is not None:
            if thresh_obs == 'all':
                thresh_obs = np.arange(self.max_score + 1)
            try:
                for ob in thresh_obs:
                    if 'multi' not in palette:
                        colorVal = scalarMap.to_rgba(ob)
                    else:
                        colorVal = color_map[ob]

                    ax.plot(x_obs_data[ob - 1, :], y_obs_data[ob - 1, :], 'o', color=colorVal)

            except:
                pass

        if thresh_lines:
            if items is None:
                if raters is None:
                    for threshold in range(self.max_score):
                        plt.axvline(x=thresholds[threshold + 1],
                                    color='black', linestyle='--')

                else:
                    for threshold in range(self.max_score):
                        plt.axvline(x=thresholds[threshold + 1] + severities[raters][threshold + 1],
                                    color='black', linestyle='--')

            else:
                if raters is None:
                    for threshold in range(self.max_score):
                        plt.axvline(x=difficulties[items] + thresholds[threshold + 1],
                                    color='black', linestyle='--')

                else:
                    for threshold in range(self.max_score):
                        plt.axvline(x=(difficulties[items] + thresholds[threshold + 1] +
                                       severities[raters][threshold + 1]),
                                    color='black', linestyle='--')

        if central_diff:
            if items is None:
                if raters is None:
                    plt.axvline(x=0, color='darkred', linestyle='--')

                else:
                    plt.axvline(x=severities[raters][1:].mean(), color='darkred', linestyle='--')

            else:
                if raters is None:
                    plt.axvline(x = difficulties[items], color='darkred', linestyle='--')

                else:
                    plt.axvline(x = difficulties[items] + severities[raters][1:].mean(),
                                color='darkred', linestyle='--')

        if score_lines_item[1] is not None:

            if (all(x > 0 for x in score_lines_item[1]) &
                all(x < self.max_score for x in score_lines_item[1])):

                abils_set = [self.score_abil_thresholds(score, anchor=anchor, items=items, raters=raters,
                                                        warm_corr=False)
                             for score in score_lines_item[1]]

                for thresh, abil in zip(score_lines_item[1], abils_set):
                    plt.vlines(x=abil, ymin=-100, ymax=thresh, color='black', linestyles='dashed')
                    if score_labels:
                        plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                    plt.hlines(y=thresh, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                    if score_labels:
                        plt.text(x_min + (x_max - x_min) / 100, thresh + y_max / 50, str(thresh))

            else:
                print('Invalid score for score line.')

        if score_lines_test is not None:

            if isinstance(raters, list):
                no_of_raters = len(raters)

            else:
                no_of_raters = 1

            if items is None:
                no_of_items = self.no_of_items

            else:
                if isinstance(items, list):
                    no_of_items = len(items)

                else:
                    no_of_items = 1

            if (all(x > 0 for x in score_lines_test) &
                all(x < self.max_score * no_of_items * no_of_raters for x in score_lines_test)):

                abils_set = [self.score_abil_thresholds(score, anchor=anchor, items=items, raters=raters,
                                                        warm_corr=False)
                             for score in score_lines_test]

                for thresh, abil in zip(score_lines_test, abils_set):
                    plt.vlines(x=abil, ymin=-100, ymax=thresh, color='black', linestyles='dashed')
                    if score_labels:
                        plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                    plt.hlines(y=thresh, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                    if score_labels:
                        plt.text(x_min + (x_max - x_min) / 100, thresh + y_max / 50, str(thresh))

            else:
                print('Invalid score for score line.')

        if point_info_lines_item[1] is not None:

            item = point_info_lines_item[0]

            if raters is None:
                info_set = [self.variance_thresholds(ability, item, difficulties, 'dummy_rater',
                                                     dummy_sevs, thresholds)
                            for ability in point_info_lines_item[1]]

            else:
                info_set = [self.variance_thresholds(ability, item, difficulties, raters, severities, thresholds)
                            for ability in point_info_lines_item[1]]

            for abil, info in zip(point_info_lines_item[1], info_set):
                plt.vlines(x=abil, ymin=-100, ymax=info, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=info, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(x_min + (x_max - x_min) / 100, info + y_max / 50, str(round(info, 3)))

        if point_info_lines_test is not None:

            if items is None:
                if raters is None:
                    info_set = [sum(self.variance_thresholds(ability, item, difficulties, 'dummy_rater',
                                                             dummy_sevs, thresholds)
                                    for item in self.dataframe.columns)
                                for ability in point_info_lines_test]

                else:
                    info_set = [sum(self.variance_thresholds(ability, item, difficulties, rater, severities, thresholds)
                                    for item in self.dataframe.columns for rater in raters)
                                for ability in point_info_lines_test]

            else:
                if raters is None:
                    info_set = [sum(self.variance_thresholds(ability, item, difficulties, 'dummy_rater',
                                                             dummy_sevs, thresholds)
                                    for item in items)
                                for ability in point_info_lines_test]

                else:
                    info_set = [sum(self.variance_thresholds(ability, item, difficulties, rater, severities, thresholds)
                                    for item in items for rater in raters)
                                for ability in point_info_lines_test]

            for abil, info in zip(point_info_lines_test, info_set):
                plt.vlines(x=abil, ymin=-100, ymax=info, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=info, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(x_min + (x_max - x_min) / 100, info + y_max / 50, str(round(info, 3)))

        if point_csem_lines is not None:

            if items is None:
                if raters is None:
                    info_set = [sum(self.variance_thresholds(ability, item, difficulties, 'dummy_rater', dummy_sevs,
                                                             thresholds)
                                    for item in self.dataframe.columns)
                                for ability in point_csem_lines]

                else:
                    info_set = [sum(self.variance_thresholds(ability, item, difficulties, rater, severities, thresholds)
                                    for item in self.dataframe.columns for rater in raters)
                                for ability in point_csem_lines]

            else:
                if raters is None:
                    info_set = [sum(self.variance_thresholds(ability, item, difficulties, 'dummy_rater', dummy_sevs,
                                                             thresholds)
                                    for item in items)
                                for ability in point_csem_lines]

                else:
                    info_set = [sum(self.variance_thresholds(ability, item, difficulties, rater, severities, thresholds)
                                    for item in items for rater in raters)
                                for ability in point_csem_lines]

            info_set = np.array(info_set)
            csem_set = 1 / (info_set ** 0.5)

            for abil, csem in zip(point_csem_lines, csem_set):
                plt.vlines(x=abil, ymin=-100, ymax=csem, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=csem, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(x_min + (x_max - x_min) / 100, csem + y_max / 50, str(round(csem, 3)))

        if cat_highlight in range(self.max_score + 1):

            if cat_highlight == 0:
                if items is None:
                    if raters is None:
                        plt.axvspan(-100, thresholds[1],
                                    facecolor='blue', alpha=0.2)

                    else:
                        plt.axvspan(-100, thresholds[1] + severities[raters][1],
                                    facecolor='blue', alpha=0.2)

                else:
                    if raters is None:
                        plt.axvspan(-100, difficulties[items] + thresholds[1],
                                    facecolor='blue', alpha=0.2)

                    else:
                        plt.axvspan(-100, difficulties[items] + thresholds[1] + severities[raters][1],
                                    facecolor='blue', alpha=0.2)

            elif cat_highlight == self.max_score:
                if items is None:
                    if raters is None:
                        plt.axvspan(thresholds[self.max_score],
                                    100, facecolor='blue', alpha=0.2)

                    else:
                        plt.axvspan(thresholds[self.max_score] + severities[raters][self.max_score],
                                    100, facecolor='blue', alpha=0.2)

                else:
                    if raters is None:
                        plt.axvspan(difficulties[items] + thresholds[self.max_score],
                                    100, facecolor='blue', alpha=0.2)

                    else:
                        plt.axvspan((difficulties[items] + thresholds[self.max_score] +
                                     severities[raters][self.max_score]),
                                    100, facecolor='blue', alpha=0.2)

            else:
                if (thresholds[cat_highlight + 1] >
                    thresholds[cat_highlight]):
                    if items is None:
                        if raters is None:
                            plt.axvspan(thresholds[cat_highlight],
                                        thresholds[cat_highlight + 1],
                                        facecolor='blue', alpha=0.2)

                        else:
                            plt.axvspan(thresholds[cat_highlight] + severities[raters][cat_highlight],
                                        thresholds[cat_highlight + 1] + severities[raters][cat_highlight + 1],
                                        facecolor='blue', alpha=0.2)
                    else:
                        if raters is None:
                            plt.axvspan(difficulties[items] + thresholds[cat_highlight],
                                        difficulties[items] + thresholds[cat_highlight + 1],
                                        facecolor='blue', alpha=0.2)

                        else:
                            plt.axvspan((difficulties[items] + thresholds[cat_highlight] +
                                         severities[raters][cat_highlight]),
                                        (difficulties[items] + thresholds[cat_highlight + 1] +
                                         severities[raters][cat_highlight + 1]),
                                        facecolor='blue', alpha=0.2)

        if y_max <= 0:
            y_max = y_data.max() * 1.1

        plt.xlim(x_min, x_max)
        plt.ylim(0, y_max)

        plt.xlabel('Ability', fontsize=axis_font_size, fontweight='bold')
        plt.ylabel(y_label, fontsize=axis_font_size, fontweight='bold')
        plt.title(graph_title, fontsize=title_font_size, fontweight='bold')

        plt.grid(True)

        plt.tick_params(axis="x", labelsize=labelsize)
        plt.tick_params(axis="y", labelsize=labelsize)

        if filename is not None:
            plt.savefig(f'{filename}.{file_format}', dpi=plot_density)

        # Close before returning -- Jupyter auto-displays any live figure
        # returned from a cell, giving two plots. Closing here prevents that.
        plt.close(graph)

        return graph

    def plot_data_matrix(self,
                         x_data,
                         y_data,
                         anchor=False,
                         items=None,
                         raters=None,
                         obs=None,
                         warm=True,
                         thresh_obs=None,
                         x_obs_data=np.array([]),
                         y_obs_data=np.array([]),
                         thresh_lines=False,
                         central_diff=False,
                         cat_highlight=None,
                         score_lines_item=[None, None],
                         score_lines_test=None,
                         point_info_lines_item=[None, None],
                         point_info_lines_test=None,
                         point_csem_lines=None,
                         score_labels=False,
                         x_min=-5,
                         x_max=5,
                         y_max=0,
                         graph_title='',
                         y_label='',
                         plot_style='white',
                         palette='dark blue',
                         black=False,
                         figsize=(8, 6),
                         font='Times New Roman',
                         title_font_size=15,
                         axis_font_size=12,
                         labelsize=12,
                         graph_name='plot',
                         tex=True,
                         plot_density=300,
                         filename=None,
                         file_format='png'):

        '''
        Basic plotting function to be called when plotting specific functions
        of person ability for RSM.
        '''

        if anchor:
            if not hasattr(self, 'anchor_thresholds_matrix'):
                print('Anchor calibration required')
                print('Run self.calibrate_matrix_anchor()')
                return

        if anchor:
            difficulties = self.anchor_diffs_matrix
            thresholds = self.anchor_thresholds_matrix
            severities = self.anchor_severities_matrix
            marginal_items = self.anchor_marginal_severities_items
            marginal_thresholds = self.anchor_marginal_severities_thresholds

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_matrix
            marginal_items = self.marginal_severities_items
            marginal_thresholds = self.marginal_severities_thresholds

        if isinstance(items, str):
            if items == 'all':
                items = None

            elif items == 'none':
                items = None

        if isinstance(raters, str):
            if raters == 'all':
                raters = self.raters

            elif raters == 'none':
                raters = None

        dummy_sevs = {'dummy_rater': {item: np.zeros(self.max_score + 1)
                                      for item in self.dataframe.columns}}

        if plot_style == 'dark':
            sns.set_style('darkgrid')

        else:
            sns.set_style('whitegrid')

        palette_dict = {'dark blue': ['dark', 'royalblue'],
                        'light blue': ['light', 'cornflowerblue'],
                        'dark red': ['dark', 'firebrick'],
                        'light red': ['light', 'indianred'],
                        'dark green': ['dark', 'forestgreen'],
                        'light green': ['light', 'mediumseagreen'],
                        'dark grey': ['dark', 'dimgrey'],
                        'light grey': ['light', 'darkgrey'],
                        'dark multi': ['dark', 'dark'],
                        'light multi': ['light', 'muted']}

        if palette_dict[palette][0] == 'dark':
            if palette == 'dark multi':
                color_map = sns.color_palette('dark', as_cmap=True)
            else:
                color_map = sns.dark_palette(palette_dict[palette][1], reverse=True, as_cmap=True)

        if palette_dict[palette][0] == 'light':
            if palette == 'light multi':
                color_map = sns.color_palette('muted', as_cmap=True)
            else:
                color_map = sns.light_palette(palette_dict[palette][1], reverse=True, as_cmap=True)

        graph, ax = plt.subplots(figsize=figsize)

        no_of_plots = y_data.shape[1]

        cNorm = colors.Normalize(vmin=0, vmax=no_of_plots + 2)

        if 'multi' not in palette:
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=color_map)

        if black:
            for i in range(no_of_plots):
                ax.plot(x_data, y_data[:, i], '', label=i+1, color='black')

        else:
            for i in range(no_of_plots):
                if 'multi' not in palette:
                    colorVal = scalarMap.to_rgba(i)
                else:
                    colorVal = color_map[i]

                ax.plot(x_data, y_data[:, i], '', color=colorVal, label=i+1)

        if obs is not None:
            try:
                if isinstance(y_obs_data, pd.Series):
                    if 'multi' not in palette:
                        colorVal = scalarMap.to_rgba(0)
                    else:
                        colorVal = color_map[0]

                    ax.plot(x_obs_data, y_obs_data, 'o', color=colorVal)

                else:
                    no_of_observed_cats = y_obs_data.shape[1]
                    for j in range (no_of_observed_cats):
                        if 'multi' not in palette:
                            colorVal = scalarMap.to_rgba(j)
                        else:
                            colorVal = color_map[j]

                        ax.plot(x_obs_data, y_obs_data[:, j], 'o', color=colorVal)

            except:
                pass

        if thresh_obs is not None:
            if thresh_obs == 'all':
                thresh_obs = np.arange(self.max_score + 1)
            try:
                for ob in thresh_obs:
                    if 'multi' not in palette:
                        colorVal = scalarMap.to_rgba(ob)
                    else:
                        colorVal = color_map[ob]

                    ax.plot(x_obs_data[ob - 1, :], y_obs_data[ob - 1, :], 'o', color=colorVal)

            except:
                pass

        if thresh_lines:
            if items is None:
                if raters is None:
                    for threshold in range(self.max_score):
                        plt.axvline(x=thresholds[threshold + 1],
                                    color='black', linestyle='--')

                else:
                    for threshold in range(self.max_score):
                        plt.axvline(x=(thresholds[threshold + 1] + marginal_items[raters].mean() +
                                       marginal_thresholds[raters][threshold + 1]),
                                    color='black', linestyle='--')

            else:
                if raters is None:
                    for threshold in range(self.max_score):
                        plt.axvline(x=difficulties[items] + thresholds[threshold + 1],
                                    color='black', linestyle='--')

                else:
                    for threshold in range(self.max_score):
                        plt.axvline(x=(difficulties[items] + thresholds[threshold + 1] +
                                       severities[raters][items][[threshold + 1]]),
                                    color='black', linestyle='--')

        if central_diff:
            if items is None:
                if raters is None:
                    plt.axvline(x=0, color='darkred', linestyle='--')

                else:
                    plt.axvline(x=marginal_items[raters].mean(), color='darkred', linestyle='--')

            else:
                if raters is None:
                    plt.axvline(x=difficulties[items], color='darkred', linestyle='--')

                else:
                    plt.axvline(x=difficulties[items] + severities[raters][items][1:].mean(),
                                color='darkred', linestyle='--')

        if score_lines_item[1] is not None:

            if (all(x > 0 for x in score_lines_item[1]) & all(x < self.max_score for x in score_lines_item[1])):

                abils_set = [self.score_abil_matrix(score, anchor=anchor, items=items, raters=raters, warm_corr=False)
                             for score in score_lines_item[1]]

                for thresh, abil in zip(score_lines_item[1], abils_set):
                    plt.vlines(x=abil, ymin=-100, ymax=thresh, color='black', linestyles='dashed')
                    if score_labels:
                        plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                    plt.hlines(y=thresh, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                    if score_labels:
                        plt.text(x_min + (x_max - x_min) / 100, thresh + y_max / 50, str(thresh))

            else:
                print('Invalid score for score line.')

        if score_lines_test is not None:

            if isinstance(raters, list):
                no_of_raters = len(raters)

            else:
                no_of_raters = 1

            if items is None:
                no_of_items = self.no_of_items

            else:
                if isinstance(items, list):
                    no_of_items = len(items)

                else:
                    no_of_items = 1

            if (all(x > 0 for x in score_lines_test) &
                all(x < self.max_score * no_of_items * no_of_raters for x in score_lines_test)):

                abils_set = [self.score_abil_matrix(score, anchor=anchor, items=items, raters=raters, warm_corr=False)
                             for score in score_lines_test]

                for thresh, abil in zip(score_lines_test, abils_set):
                    plt.vlines(x=abil, ymin=-100, ymax=thresh, color='black', linestyles='dashed')
                    if score_labels:
                        plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                    plt.hlines(y=thresh, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                    if score_labels:
                        plt.text(x_min + (x_max - x_min) / 100, thresh + y_max / 50, str(thresh))

            else:
                print('Invalid score for score line.')

        if point_info_lines_item[1] is not None:

            item = point_info_lines_item[0]

            if raters is None:
                info_set = [self.variance_matrix(ability, item, difficulties, 'dummy_rater',
                                                 dummy_sevs, thresholds)
                            for ability in point_info_lines_item[1]]

            else:
                info_set = [self.variance_matrix(ability, item, difficulties, raters, severities, thresholds)
                            for ability in point_info_lines_item[1]]

            for abil, info in zip(point_info_lines_item[1], info_set):
                plt.vlines(x=abil, ymin=-100, ymax=info, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=info, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(x_min + (x_max - x_min) / 100, info + y_max / 50, str(round(info, 3)))

        if point_info_lines_test is not None:

            if items is None:
                if raters is None:
                    info_set = [sum(self.variance_matrix(ability, item, difficulties, 'dummy_rater', dummy_sevs,
                                                         thresholds)
                                    for item in self.dataframe.columns)
                                for ability in point_info_lines_test]

                else:
                    info_set = [sum(self.variance_matrix(ability, item, difficulties, rater, severities, thresholds)
                                    for item in self.dataframe.columns for rater in raters)
                                for ability in point_info_lines_test]

            else:
                if raters is None:
                    info_set = [sum(self.variance_matrix(ability, item, difficulties, 'dummy_rater', dummy_sevs,
                                                         thresholds)
                                    for item in items)
                                for ability in point_info_lines_test]

                else:
                    info_set = [sum(self.variance_matrix(ability, item, difficulties, rater, severities, thresholds)
                                    for item in items for rater in raters)
                                for ability in point_info_lines_test]

            for abil, info in zip(point_info_lines_test, info_set):
                plt.vlines(x=abil, ymin=-100, ymax=info, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=info, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(x_min + (x_max - x_min) / 100, info + y_max / 50, str(round(info, 3)))

        if point_csem_lines is not None:

            if items is None:
                if raters is None:
                    info_set = [sum(self.variance_matrix(ability, item, difficulties, 'dummy_rater',
                                                         dummy_sevs, thresholds)
                                    for item in self.dataframe.columns)
                                for ability in point_csem_lines]

                else:
                    info_set = [sum(self.variance_matrix(ability, item, difficulties, rater, severities, thresholds)
                                    for item in self.dataframe.columns for rater in raters)
                                for ability in point_csem_lines]

            else:
                if raters is None:
                    info_set = [sum(self.variance_matrix(ability, item, difficulties, 'dummy_rater',
                                                         dummy_sevs, thresholds)
                                    for item in items)
                                for ability in point_csem_lines]

                else:
                    info_set = [sum(self.variance_matrix(ability, item, difficulties, rater, severities, thresholds)
                                    for item in items for rater in raters)
                                for ability in point_csem_lines]

            info_set = np.array(info_set)
            csem_set = 1 / (info_set ** 0.5)

            for abil, csem in zip(point_csem_lines, csem_set):
                plt.vlines(x=abil, ymin=-100, ymax=csem, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=csem, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(x_min + (x_max - x_min) / 100, csem + y_max / 50, str(round(csem, 3)))

        if cat_highlight in range(self.max_score + 1):

            if cat_highlight == 0:
                if items is None:
                    if raters is None:
                        plt.axvspan(-100, thresholds[1],
                                    facecolor='blue', alpha=0.2)

                    else:
                        plt.axvspan(-100, (thresholds[1] + marginal_items[raters].mean() +
                                           marginal_thresholds[raters][1]),
                                    facecolor='blue', alpha=0.2)

                else:
                    if raters is None:
                        plt.axvspan(-100, difficulties[items] + thresholds[1],
                                    facecolor='blue', alpha=0.2)

                    else:
                        plt.axvspan(-100, difficulties[items] + thresholds[1] + severities[raters][items][1],
                                    facecolor='blue', alpha=0.2)

            elif cat_highlight == self.max_score:
                if items is None:
                    if raters is None:
                        plt.axvspan(thresholds[self.max_score],
                                    100, facecolor='blue', alpha=0.2)

                    else:
                        plt.axvspan((thresholds[self.max_score] + marginal_items[raters].mean() +
                                     marginal_thresholds[raters][self.max_score]),
                                    100, facecolor='blue', alpha=0.2)

                else:
                    if raters is None:
                        plt.axvspan(difficulties[items] + thresholds[self.max_score],
                                    100, facecolor='blue', alpha=0.2)

                    else:
                        plt.axvspan((difficulties[items] + thresholds[self.max_score] +
                                     severities[raters][item][self.max_score]),
                                    100, facecolor='blue', alpha=0.2)

            else:
                if (thresholds[cat_highlight + 1] > thresholds[cat_highlight]):
                    if items is None:
                        if raters is None:
                            plt.axvspan(thresholds[cat_highlight],
                                        thresholds[cat_highlight + 1],
                                        facecolor='blue', alpha=0.2)

                        else:
                            plt.axvspan((thresholds[cat_highlight] + marginal_items[raters].mean() +
                                         marginal_thresholds[raters][cat_highlight]),
                                        (thresholds[cat_highlight] + marginal_items[raters].mean() +
                                         marginal_thresholds[raters][cat_highlight + 1]),
                                        facecolor='blue', alpha=0.2)
                    else:
                        if raters is None:
                            plt.axvspan(difficulties[items] + thresholds[cat_highlight],
                                        difficulties[items] + thresholds[cat_highlight + 1],
                                        facecolor='blue', alpha=0.2)

                        else:
                            plt.axvspan((difficulties[items] + thresholds[cat_highlight] +
                                         severities[raters][items][cat_highlight]),
                                        (difficulties[items] + thresholds[cat_highlight + 1] +
                                         severities[raters][items][cat_highlight + 1]),
                                        facecolor='blue', alpha=0.2)

        if y_max <= 0:
            y_max = y_data.max() * 1.1

        plt.xlim(x_min, x_max)
        plt.ylim(0, y_max)

        plt.xlabel('Ability', fontsize=axis_font_size, fontweight='bold')
        plt.ylabel(y_label, fontsize=axis_font_size, fontweight='bold')
        plt.title(graph_title, fontsize=title_font_size, fontweight='bold')

        plt.grid(True)
        plt.tick_params(axis="x", labelsize=labelsize)
        plt.tick_params(axis="y", labelsize=labelsize)

        if filename is not None:
            plt.savefig(f'{filename}.{file_format}', dpi=plot_density)

        # Close before returning -- Jupyter auto-displays any live figure
        # returned from a cell, giving two plots. Closing here prevents that.
        plt.close(graph)

        return graph

    def icc_global(self,
                   item,
                   anchor=False,
                   rater=None,
                   obs=None,
                   warm=True,
                   xmin=-5,
                   xmax=5,
                   no_of_classes=5,
                   title=None,
                   thresh_lines=False,
                   score_lines=None,
                   score_labels=False,
                   central_diff=False,
                   cat_highlight=None,
                   plot_style='white',
                   palette='dark blue',
                   black=False,
                   font='Times New Roman',
                   title_font_size=15,
                   axis_font_size=12,
                   labelsize=12,
                   filename=None,
                   file_format='png',
                   dpi=300):

        '''
        Plots Item Characteristic Curve, with optional overplotting of observed data,
        threshold lines and expected score threshold lines.
        '''
        
        if rater == 'none':
            rater = None

        if anchor:
            if not hasattr(self, 'anchor_thresholds_global'):
                print('Anchor calibration required')
                print('Run self.calibrate_global_anchor()')
                return

        if anchor:
            difficulties = self.anchor_diffs_global
            thresholds = self.anchor_thresholds_global
            severities = self.anchor_severities_global

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_global

        dummy_sevs = pd.Series({'dummy_rater': 0})
        
        shift = 0

        if anchor:
            if rater is None:
                shift = self.severities_global[self.anchor_raters_global].mean()

        if obs:
            if anchor:
                if not hasattr(self, 'anchor_abils_global'):
                    self.person_abils_global(anchor=True)
                abilities = self.anchor_abils_global

            else:
                if not hasattr(self, 'abils_global'):
                    self.person_abils_global()
                abilities = self.abils_global

            xobsdata, yobsdata = self.class_intervals(abilities, items=item, raters=rater, shift=shift,
                                                      no_of_classes=no_of_classes)

            yobsdata = yobsdata.values.reshape((-1, 1))

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)

        if rater is None:
            y = [self.exp_score_global(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                 for ability in abilities]

        else:
            y = [self.exp_score_global(ability, item, difficulties, rater, severities, thresholds)
                 for ability in abilities]

        y = np.array(y).reshape([len(abilities), 1])

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Expected score'

        plot = self.plot_data_global(x_data=abilities, y_data=y, anchor=anchor, raters=rater, x_min=xmin, x_max=xmax,
                                     y_max=self.max_score, items=item, obs=obs, warm=warm, x_obs_data=xobsdata,
                                     y_obs_data=yobsdata, thresh_lines=thresh_lines, graph_title=graphtitle,
                                     score_lines_item=[item, score_lines], score_labels=score_labels,
                                     central_diff=central_diff, cat_highlight=cat_highlight, y_label=ylabel,
                                     plot_style=plot_style, palette=palette, black=black, font=font,
                                     title_font_size=title_font_size, axis_font_size=axis_font_size,
                                     labelsize=labelsize, filename=filename, plot_density=dpi, file_format=file_format)

        return plot
        return plot

    def icc_items(self,
                  item,
                  anchor=False,
                  rater=None,
                  obs=None,
                  warm=True,
                  xmin=-5,
                  xmax=5,
                  no_of_classes=5,
                  title=None,
                  thresh_lines=False,
                  score_lines=None,
                  score_labels=False,
                  central_diff=False,
                  cat_highlight=None,
                  plot_style='white',
                  palette='dark blue',
                  black=False,
                  font='Times New Roman',
                  title_font_size=15,
                  axis_font_size=12,
                  labelsize=12,
                  filename=None,
                  file_format='png',
                  dpi=300):

        '''
        Plots Item Characteristic Curve, with optional overplotting
        of observed data, threshold lines and expected score threshold lines.
        '''
        
        if rater == 'none':
            rater = None

        if anchor:
            if not hasattr(self, 'anchor_thresholds_items'):
                print('Anchor calibration required')
                print('Run self.calibrate_items_anchor()')
                return

        if anchor:
            difficulties = self.anchor_diffs_items
            thresholds = self.anchor_thresholds_items
            severities = self.anchor_severities_items

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_items

        dummy_sevs = {'dummy_rater': {item: 0 for item in self.dataframe.columns}}
        
        shift = 0

        if anchor:
            if rater is None:
                shift = pd.DataFrame(self.anchor_severities_items).loc[item].mean()

        if obs:
            if anchor:
                if not hasattr(self, 'anchor_abils_items'):
                    self.person_abils_items(anchor=True)
                abilities = self.anchor_abils_items

            else:
                if not hasattr(self, 'abils_items'):
                    self.person_abils_items()
                abilities = self.abils_items

            xobsdata, yobsdata = self.class_intervals(abilities, items=item, raters=rater, shift=shift,
                                                      no_of_classes=no_of_classes)

            yobsdata = yobsdata.values.reshape((-1, 1))

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)

        if rater is None:
            y = [self.exp_score_items(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                 for ability in abilities]

        else:
            y = [self.exp_score_items(ability, item, difficulties, rater, severities, thresholds)
                 for ability in abilities]

        y = np.array(y).reshape([len(abilities), 1])

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Expected score'

        plot = self.plot_data_items(x_data=abilities, y_data=y, anchor=anchor, raters=rater, x_min=xmin, x_max=xmax,
                                    y_max=self.max_score, items=item, obs=obs,  warm=warm, x_obs_data=xobsdata,
                                    y_obs_data=yobsdata, thresh_lines=thresh_lines, graph_title=graphtitle,
                                    score_lines_item=[item, score_lines], score_labels=score_labels,
                                    central_diff=central_diff, cat_highlight=cat_highlight, y_label=ylabel,
                                    plot_style=plot_style, palette=palette, black=black, font=font,
                                    title_font_size=title_font_size, axis_font_size=axis_font_size,
                                    labelsize=labelsize, filename=filename, plot_density=dpi, file_format=file_format)

        return plot

    def icc_thresholds(self,
                       item,
                       anchor=False,
                       rater=None,
                       obs=None,
                       warm=True,
                       xmin=-5,
                       xmax=5,
                       no_of_classes=5,
                       title=None,
                       thresh_lines=False,
                       score_lines=None,
                       score_labels=False,
                       central_diff=False,
                       cat_highlight=None,
                       plot_style='white',
                       palette='dark blue',
                       black=False,
                       font='Times New Roman',
                       title_font_size=15,
                       axis_font_size=12,
                       labelsize=12,
                       filename=None,
                       file_format='png',
                       dpi=300):

        '''
        Plots Item Characteristic Curve, with optional overplotting
        of observed data, threshold lines and expected score threshold lines.
        '''
        
        if rater == 'none':
            rater = None

        if anchor:
            if not hasattr(self, 'anchor_thresholds_thresholds'):
                print('Anchor calibration required')
                print('Run self.calibrate_thresholds_anchor()')
                return

        if anchor:
            difficulties = self.anchor_diffs_thresholds
            thresholds = self.anchor_thresholds_thresholds
            severities = self.anchor_severities_thresholds

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_thresholds

        dummy_sevs = {'dummy_rater': np.zeros(self.max_score + 1)}
        
        shift = 0

        if anchor:
            if rater is None:
                shift = pd.DataFrame(self.anchor_severities_thresholds).iloc[1:].mean().mean()

        if obs:
            if anchor:
                if not hasattr(self, 'anchor_abils_thresholds'):
                    self.person_abils_thresholds(anchor=True)
                abilities = self.anchor_abils_thresholds

            else:
                if not hasattr(self, 'abils_thresholds'):
                    self.person_abils_thresholds()
                abilities = self.abils_thresholds

            xobsdata, yobsdata = self.class_intervals(abilities, items=item, raters=rater, shift=shift,
                                                      no_of_classes=no_of_classes)

            yobsdata = np.array(yobsdata).reshape((-1, 1))

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)

        if rater is None:
            y = [self.exp_score_thresholds(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                 for ability in abilities]

        else:
            y = [self.exp_score_thresholds(ability, item, difficulties, rater, severities, thresholds)
                 for ability in abilities]

        y = np.array(y).reshape([len(abilities), 1])

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Expected score'

        plot = self.plot_data_thresholds(x_data=abilities, y_data=y, anchor=anchor, raters=rater, x_min=xmin,
                                         x_max=xmax, y_max=self.max_score, items=item, obs=obs,  warm=warm,
                                         x_obs_data=xobsdata, y_obs_data=yobsdata, thresh_lines=thresh_lines,
                                         graph_title=graphtitle, score_lines_item=[item, score_lines],
                                         score_labels=score_labels, central_diff=central_diff,
                                         cat_highlight=cat_highlight, y_label=ylabel, plot_style=plot_style,
                                         palette=palette, black=black, font=font, title_font_size=title_font_size,
                                         axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                                         plot_density=dpi, file_format=file_format)


        return plot

    def icc_matrix(self,
                   item,
                   anchor=False,
                   rater=None,
                   obs=None,
                   warm=True,
                   xmin=-5,
                   xmax=5,
                   no_of_classes=5,
                   title=None,
                   thresh_lines=False,
                   score_lines=None,
                   score_labels=False,
                   central_diff=False,
                   cat_highlight=None,
                   plot_style='white',
                   palette='dark blue',
                   black=False,
                   font='Times New Roman',
                   title_font_size=15,
                   axis_font_size=12,
                   labelsize=12,
                   filename=None,
                   file_format='png',
                   dpi=300):

        '''
        Plots Item Characteristic Curve, with optional overplotting
        of observed data, threshold lines and expected score threshold lines.
        '''
        
        if rater == 'none':
            rater = None

        if anchor:
            if not hasattr(self, 'anchor_thresholds_matrix'):
                print('Anchor calibration required')
                print('Run self.calibrate_matrix_anchor()')
                return

        if anchor:
            difficulties = self.anchor_diffs_matrix
            thresholds = self.anchor_thresholds_matrix
            severities = self.anchor_severities_matrix

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_matrix

        dummy_sevs = {'dummy_rater': {item: np.zeros(self.max_score + 1)}}

        shift = 0

        if anchor:
            if rater is None:
                shift = pd.DataFrame(self.anchor_marginal_severities_items).loc[item].mean()

        if obs:
            if anchor:
                if not hasattr(self, 'anchor_abils_matrix'):
                    self.person_abils_matrix(anchor=True)
                abilities = self.anchor_abils_matrix

            else:
                if not hasattr(self, 'abils_matrix'):
                    self.person_abils_matrix()
                abilities = self.abils_matrix

            xobsdata, yobsdata = self.class_intervals(abilities, items=item, raters=rater, shift=shift,
                                                      no_of_classes=no_of_classes)

            yobsdata = np.array(yobsdata).reshape((-1, 1))

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)

        if rater is None:
            y = [self.exp_score_matrix(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                 for ability in abilities]

        else:
            y = [self.exp_score_matrix(ability, item, difficulties, rater, severities, thresholds)
                 for ability in abilities]

        y = np.array(y).reshape([len(abilities), 1])

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Expected score'

        plot = self.plot_data_matrix(x_data=abilities, y_data=y, anchor=anchor, raters=rater, x_min=xmin, x_max=xmax,
                                     y_max=self.max_score, items=item, obs=obs,  warm=warm, x_obs_data=xobsdata,
                                     y_obs_data=yobsdata, thresh_lines=thresh_lines, graph_title=graphtitle,
                                     score_lines_item=[item, score_lines], score_labels=score_labels,
                                     central_diff=central_diff, cat_highlight=cat_highlight, y_label=ylabel,
                                     plot_style=plot_style, palette=palette, black=black, font=font,
                                     title_font_size=title_font_size, axis_font_size=axis_font_size,
                                     labelsize=labelsize, filename=filename, plot_density=dpi, file_format=file_format)

        return plot

    def crcs_global(self,
                    item=None,
                    anchor=False,
                    rater=None,
                    obs=None,
                    xmin=-5,
                    xmax=5,
                    no_of_classes=5,
                    title=None,
                    thresh_lines=False,
                    central_diff=False,
                    cat_highlight=None,
                    plot_style='white',
                    palette='dark blue',
                    black=False,
                    font='Times New Roman',
                    title_font_size=15,
                    axis_font_size=12,
                    labelsize=12,
                    filename=None,
                    file_format='png',
                    dpi=300):

        '''
        Plots Category Response Curves, with optional overplotting
        of observed data and threshold lines.
        '''

        if item == 'none':
            item = None

        if rater == 'none':
            rater = None

        if anchor:
            if not hasattr(self, 'anchor_thresholds_global'):
                print('Anchor calibration required')
                print('Run self.calibrate_global_anchor()')
                return

        if anchor:
            difficulties = self.anchor_diffs_global
            thresholds = self.anchor_thresholds_global
            severities = self.anchor_severities_global

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_global

        dummy_sevs = pd.Series({'dummy_rater': 0})
            
        if obs == 'none':
            obs = None

        if obs == 'all':
            obs = np.arange(self.max_score + 1)

        if obs is not None:
            if anchor:
                if not hasattr(self, 'anchor_abils_global'):
                    self.person_abils_global(anchor=True)
                abilities = self.anchor_abils_global

            else:
                if not hasattr(self, 'abils_global'):
                    self.person_abils_global()
                abilities = self.abils_global

            xobsdata, yobsdata = self.class_intervals_cats_global(abilities, difficulties, thresholds, severities,
                                                                  item=item, rater=rater, no_of_classes=no_of_classes)

            yobsdata = yobsdata[obs].T

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)

        if rater is None:
            if item is None:
                y = [[self.cat_prob_global(ability + difficulties.values[0], difficulties.index[0],
                                           difficulties, 'dummy_rater', dummy_sevs, category, thresholds)
                      for category in range(self.max_score + 1)]
                     for ability in abilities]


            else:
                y = [[self.cat_prob_global(ability, item, difficulties, 'dummy_rater', dummy_sevs,
                                           category, thresholds)
                      for category in range(self.max_score + 1)]
                     for ability in abilities]

        else:
            if item is None:
                y = [[self.cat_prob_global(ability + difficulties.values[0], difficulties.index[0],
                                           difficulties, rater, severities, category, thresholds)
                      for category in range(self.max_score + 1)]
                     for ability in abilities]


            else:
                y = [[self.cat_prob_global(ability, item, difficulties, rater, severities, category,
                                           thresholds)
                      for category in range(self.max_score + 1)]
                     for ability in abilities]

        y = np.array(y)

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Probability'

        plot = self.plot_data_global(x_data=abilities, y_data=y, anchor=anchor, items=item, raters=rater, x_min=xmin,
                                     x_max=xmax, y_max=1, x_obs_data=xobsdata, y_obs_data=yobsdata, y_label=ylabel,
                                     graph_title=graphtitle, obs=obs, thresh_lines=thresh_lines, plot_style=plot_style,
                                     palette=palette, central_diff=central_diff, cat_highlight=cat_highlight,
                                     black=black, font=font, title_font_size=title_font_size, labelsize=labelsize,
                                     axis_font_size=axis_font_size, filename=filename, plot_density=dpi,
                                     file_format=file_format)

        return plot

    def crcs_items(self,
                   item=None,
                   anchor=False,
                   rater=None,
                   obs=None,
                   xmin=-5,
                   xmax=5,
                   no_of_classes=5,
                   title=None,
                   thresh_lines=False,
                   central_diff=False,
                   cat_highlight=None,
                   plot_style='white',
                   palette='dark blue',
                   black=False,
                   font='Times New Roman',
                   title_font_size=15,
                   axis_font_size=12,
                   labelsize=12,
                   filename=None,
                   file_format='png',
                   dpi=300):

        '''
        Plots Category Response Curves, with optional overplotting
        of observed data and threshold lines.
        '''

        if item == 'none':
            item = None

        if rater == 'none':
            rater = None

        if anchor:
            if not hasattr(self, 'anchor_thresholds_items'):
                print('Anchor calibration required')
                print('Run self.calibrate_items_anchor()')
                return

        if anchor:
            difficulties = self.anchor_diffs_items
            thresholds = self.anchor_thresholds_items
            severities = self.anchor_severities_items

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_items

        dummy_sevs = {'dummy_rater': {item: 0 for item in self.dataframe.columns}}
        
        shift = 0

        if rater is None:
            if anchor:
                shift = pd.DataFrame(self.anchor_severities_items).mean().mean()

            else:
                shift = pd.DataFrame(self.severities_items).mean().mean()

        if obs == 'none':
            obs = None

        if obs == 'all':
            obs = np.arange(self.max_score + 1)

        if obs is not None:
            if anchor:
                if not hasattr(self, 'anchor_abils_items'):
                    self.person_abils_items(anchor=True)
                abilities = self.anchor_abils_items

            else:
                if not hasattr(self, 'abils_items'):
                    self.person_abils_items()
                abilities = self.abils_items

            xobsdata, yobsdata = self.class_intervals_cats_items(abilities, difficulties, thresholds, severities,
                                                                 item=item, rater=rater, shift=shift,
                                                                 no_of_classes=no_of_classes)

            yobsdata = yobsdata[obs].T

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)

        if rater is None:
            if item is None:
                y = [[self.cat_prob_items(ability + difficulties.values[0], difficulties.index[0],
                                          difficulties, 'dummy_rater', dummy_sevs, category, thresholds)
                      for category in range(self.max_score + 1)]
                     for ability in abilities]


            else:
                y = [[self.cat_prob_items(ability, item, difficulties, 'dummy_rater', dummy_sevs,
                                          category, thresholds)
                      for category in range(self.max_score + 1)]
                     for ability in abilities]

        else:
            if item is None:
                y = [[self.cat_prob_items(ability + difficulties.values[0], difficulties.index[0],
                                          difficulties, rater, severities, category, thresholds)
                      for category in range(self.max_score + 1)]
                     for ability in abilities]


            else:
                y = [[self.cat_prob_items(ability, item, difficulties, rater, severities, category,
                                          thresholds)
                      for category in range(self.max_score + 1)]
                     for ability in abilities]

        y = np.array(y)

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Probability'

        plot = self.plot_data_items(x_data=abilities, y_data=y, anchor=anchor, items=item, raters=rater, x_min=xmin,
                                    x_max=xmax, y_max=1, x_obs_data=xobsdata, y_obs_data=yobsdata,y_label=ylabel,
                                    graph_title=graphtitle, obs=obs, thresh_lines=thresh_lines, plot_style=plot_style,
                                    palette=palette, central_diff=central_diff, cat_highlight=cat_highlight,
                                    black=black, font=font, title_font_size=title_font_size, labelsize=labelsize,
                                    axis_font_size=axis_font_size, filename=filename, plot_density=dpi,
                                    file_format=file_format)

        return plot

    def crcs_thresholds(self,
                        item=None,
                        anchor=False,
                        rater=None,
                        obs=None,
                        xmin=-5,
                        xmax=5,
                        no_of_classes=5,
                        title=None,
                        thresh_lines=False,
                        central_diff=False,
                        cat_highlight=None,
                        plot_style='white',
                        palette='dark blue',
                        black=False,
                        font='Times New Roman',
                        title_font_size=15,
                        axis_font_size=12,
                        labelsize=12,
                        filename=None,
                        file_format='png',
                        dpi=300):

        '''
        Plots Category Response Curves, with optional overplotting
        of observed data and threshold lines.
        '''

        if item == 'none':
            item = None

        if rater == 'none':
            rater = None

        if anchor:
            if not hasattr(self, 'anchor_thresholds_thresholds'):
                print('Anchor calibration required')
                print('Run self.calibrate_thresholds_anchor()')
                return

        if anchor:
            difficulties = self.anchor_diffs_thresholds
            thresholds = self.anchor_thresholds_thresholds
            severities = self.anchor_severities_thresholds

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_thresholds

        dummy_sevs = {'dummy_rater': np.zeros(self.max_score + 1)}

        if obs == 'none':
            obs = None

        if obs == 'all':
            obs = np.arange(self.max_score + 1)

        if obs is not None:
            if anchor:
                if not hasattr(self, 'anchor_abils_thresholds'):
                    self.person_abils_thresholds(anchor=True)
                abilities = self.anchor_abils_thresholds

            else:
                if not hasattr(self, 'abils_thresholds'):
                    self.person_abils_thresholds()
                abilities = self.abils_thresholds

            xobsdata, yobsdata = self.class_intervals_cats_thresholds(abilities, difficulties, thresholds,
                                                                      severities, item=item, rater=rater,
                                                                      no_of_classes=no_of_classes)

            yobsdata = yobsdata[obs].T

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)

        if rater is None:
            if item is None:
                y = [[self.cat_prob_thresholds(ability + difficulties.values[0], difficulties.index[0],
                                               difficulties, 'dummy_rater', dummy_sevs, category, thresholds)
                      for category in range(self.max_score + 1)]
                     for ability in abilities]


            else:
                y = [[self.cat_prob_thresholds(ability, item, difficulties, 'dummy_rater', dummy_sevs,
                                               category, thresholds)
                      for category in range(self.max_score + 1)]
                     for ability in abilities]

        else:
            if item is None:
                y = [[self.cat_prob_thresholds(ability + difficulties.values[0], difficulties.index[0],
                                               difficulties, rater, severities, category, thresholds)
                      for category in range(self.max_score + 1)]
                     for ability in abilities]


            else:
                y = [[self.cat_prob_thresholds(ability, item, difficulties, rater, severities, category,
                                               thresholds)
                      for category in range(self.max_score + 1)]
                     for ability in abilities]

        y = np.array(y)

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Probability'

        plot = self.plot_data_thresholds(x_data=abilities, y_data=y, anchor=anchor, items=item, raters=rater,
                                         x_min=xmin, x_max=xmax, y_max=1, x_obs_data=xobsdata, y_obs_data=yobsdata,
                                         y_label=ylabel, graph_title=graphtitle, obs=obs, thresh_lines=thresh_lines,
                                         plot_style=plot_style, palette=palette, central_diff=central_diff,
                                         cat_highlight=cat_highlight, black=black, font=font,
                                         title_font_size=title_font_size, labelsize=labelsize,
                                         axis_font_size=axis_font_size, filename=filename, plot_density=dpi,
                                         file_format=file_format)

        return plot

    def crcs_matrix(self,
                    item=None,
                    anchor=False,
                    rater=None,
                    obs=None,
                    xmin=-5,
                    xmax=5,
                    no_of_classes=5,
                    title=None,
                    thresh_lines=False,
                    central_diff=False,
                    cat_highlight=None,
                    plot_style='white',
                    palette='dark blue',
                    black=False,
                    font='Times New Roman',
                    title_font_size=15,
                    axis_font_size=12,
                    labelsize=12,
                    filename=None,
                    file_format='png',
                    dpi=300):

        '''
        Plots Category Response Curves, with optional overplotting
        of observed data and threshold lines.
        '''

        if item == 'none':
            item = None

        if rater == 'none':
            rater = None

        if anchor:
            if not hasattr(self, 'anchor_thresholds_matrix'):
                print('Anchor calibration required')
                print('Run self.calibrate_matrix_anchor()')
                return

        if anchor:
            difficulties = self.anchor_diffs_matrix
            thresholds = self.anchor_thresholds_matrix
            severities = self.anchor_severities_matrix

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_matrix
        
        
        dummy_sevs = {'dummy_rater': {item: np.zeros(self.max_score + 1)
                                      for item in self.dataframe.columns}}
        
        shift = 0

        if rater is None:
            if anchor:
                shift = pd.DataFrame(self.anchor_marginal_severities_items).mean().mean()

            else:
                shift = pd.DataFrame(self.marginal_severities_items).mean().mean()

        if obs == 'none':
            obs = None

        if obs == 'all':
            obs = np.arange(self.max_score + 1)

        if obs is not None:
            if anchor:
                if not hasattr(self, 'anchor_abils_matrix'):
                    self.person_abils_matrix(anchor=True)
                abilities = self.anchor_abils_matrix

            else:
                if not hasattr(self, 'abils_matrix'):
                    self.person_abils_matrix()
                abilities = self.abils_matrix

            xobsdata, yobsdata = self.class_intervals_cats_matrix(abilities, difficulties, thresholds, severities,
                                                                  item=item, rater=rater, shift=shift,
                                                                  no_of_classes=no_of_classes)

            yobsdata = yobsdata[obs].T

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)

        if rater is None:
            if item is None:
                y = [[self.cat_prob_matrix(ability + difficulties.values[0], difficulties.index[0], difficulties,
                                           'dummy_rater', dummy_sevs, category,
                                           thresholds)
                      for category in range(self.max_score + 1)]
                     for ability in abilities]

            else:
                y = [[self.cat_prob_matrix(ability, item, difficulties, 'dummy_rater', dummy_sevs, category, thresholds)
                      for category in range(self.max_score + 1)]
                     for ability in abilities]

        else:
            if item is None:
                y = [[self.cat_prob_matrix(ability + difficulties.values[0], difficulties.index[0], difficulties,
                                           rater, severities, category, thresholds)
                      for category in range(self.max_score + 1)]
                     for ability in abilities]


            else:
                y = [[self.cat_prob_matrix(ability, item, difficulties, rater, severities, category, thresholds)
                      for category in range(self.max_score + 1)]
                     for ability in abilities]

        y = np.array(y)

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Probability'

        plot = self.plot_data_matrix(x_data=abilities, y_data=y, anchor=anchor, items=item, raters=rater, x_min=xmin,
                                     x_max=xmax, y_max=1, x_obs_data=xobsdata, y_obs_data=yobsdata, y_label=ylabel,
                                     graph_title=graphtitle, obs=obs, thresh_lines=thresh_lines, plot_style=plot_style,
                                     palette=palette, central_diff=central_diff, cat_highlight=cat_highlight,
                                     black=black, font=font, title_font_size=title_font_size, labelsize=labelsize,
                                     axis_font_size=axis_font_size, filename=filename, plot_density=dpi,
                                     file_format=file_format)

        return plot

    def threshold_ccs_global(self,
                             item=None,
                             anchor=False,
                             rater=None,
                             obs=None,
                             warm=True,
                             xmin=-5,
                             xmax=5,
                             no_of_classes=5,
                             title=None,
                             thresh_lines=False,
                             central_diff=False,
                             cat_highlight=None,
                             plot_style='white',
                             palette='dark blue',
                             black=False,
                             font='Times New Roman',
                             title_font_size=15,
                             axis_font_size=12,
                             labelsize=12,
                             filename=None,
                             file_format='png',
                             dpi=300):

        '''
        Plots Threshold Characteristic Curves, with optional
        overplotting of observed data and threshold lines.
        '''

        if item == 'none':
            item = None

        if (rater == 'none') or (rater == 'zero'):
            rater = None

        if anchor:
            if not hasattr(self, 'anchor_thresholds_global'):
                print('Anchor calibration required')
                print('Run self.calibrate_global_anchor()')
                return

        if anchor:
            difficulties = self.anchor_diffs_global
            thresholds = self.anchor_thresholds_global
            severities = self.anchor_severities_global

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_global

        shift = 0

        if obs is not None:
            if anchor:
                if not hasattr(self, 'anchor_abils_global'):
                    self.person_abils_global(anchor=True)
                abilities = self.anchor_abils_global

            else:
                if not hasattr(self, 'abils_global'):
                    self.person_abils_global()
                abilities = self.abils_global

            xobsdata, yobsdata = self.class_intervals_thr_global(abilities, difficulties, severities, item=item,
                                                                 rater=rater, shift=shift, no_of_classes=no_of_classes)

            if obs == 'all':
                obs = [i + 1 for i in range(self.max_score)]

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)

        if item is None:
            if rater is None:
                adj_thresholds = thresholds[1:]

            else:
                adj_thresholds = thresholds[1:] + severities[rater]

        else:
            if rater is None:
                adj_thresholds = difficulties[item] + thresholds[1:]

            else:
                adj_thresholds = difficulties[item] + thresholds[1:] + severities[rater]

        y = np.array([[1 / (1 + np.exp(threshold - ability))
                       for threshold in adj_thresholds]
                      for ability in abilities])

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Probability'

        plot = self.plot_data_global(x_data=abilities, y_data=y, anchor=anchor, items=item, raters=rater, y_max=1,
                                     x_min=xmin, x_max=xmax, warm=warm, x_obs_data=xobsdata, y_obs_data=yobsdata,
                                     graph_title=graphtitle, y_label=ylabel, thresh_obs=obs, thresh_lines=thresh_lines,
                                     central_diff=central_diff, cat_highlight=cat_highlight, plot_style=plot_style,
                                     black=black, font=font, title_font_size=title_font_size,
                                     axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                                     file_format=file_format, plot_density=dpi)

        return plot

    def threshold_ccs_items(self,
                            item=None,
                            anchor=False,
                            rater=None,
                            obs=None,
                            warm=True,
                            xmin=-5,
                            xmax=5,
                            no_of_classes=5,
                            title=None,
                            thresh_lines=False,
                            central_diff=False,
                            cat_highlight=None,
                            plot_style='white',
                            palette='dark blue',
                            black=False,
                            font='Times New Roman',
                            title_font_size=15,
                            axis_font_size=12,
                            labelsize=12,
                            filename=None,
                            file_format='png',
                            dpi=300):

        '''
        Plots Threshold Characteristic Curves, with optional
        overplotting of observed data and threshold lines.
        '''

        if item == 'none':
            item = None

        if (rater == 'none') or (rater == 'zero'):
            rater = None

        if anchor:
            if not hasattr(self, 'anchor_thresholds_items'):
                print('Anchor calibration required')
                print('Run self.calibrate_items_anchor()')
                return

        if anchor:
            difficulties = self.anchor_diffs_items
            thresholds = self.anchor_thresholds_items
            severities = self.anchor_severities_items

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_items

        shift = 0

        if obs is not None:
            if anchor:
                if not hasattr(self, 'anchor_abils_items'):
                    self.person_abils_items(anchor=True)
                abilities = self.anchor_abils_items

            else:
                if not hasattr(self, 'abils_items'):
                    self.person_abils_items()
                abilities = self.abils_items

            xobsdata, yobsdata = self.class_intervals_thr_items(abilities, difficulties, severities, item=item,
                                                                rater=rater, shift=shift, no_of_classes=no_of_classes)


            if obs == 'all':
                obs = [i + 1 for i in range(self.max_score)]

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)

        if item is None:
            if rater is None:
                adj_thresholds = thresholds[1:]

            else:
                adj_thresholds = thresholds[1:] + np.mean([severities[rater][item]
                                                           for item in self.dataframe.columns])

        else:
            if rater is None:
                adj_thresholds = difficulties[item] + thresholds[1:]

            else:
                adj_thresholds = difficulties[item] + thresholds[1:] + severities[rater][item]

        y = np.array([[1 / (1 + np.exp(threshold - ability))
                       for threshold in adj_thresholds]
                      for ability in abilities])

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Probability'

        plot = self.plot_data_items(x_data=abilities, y_data=y, anchor=anchor, items=item, raters=rater, y_max=1,
                                    x_min=xmin, x_max=xmax, warm=warm, x_obs_data=xobsdata, y_obs_data=yobsdata,
                                    graph_title=graphtitle, y_label=ylabel, thresh_obs=obs, thresh_lines=thresh_lines,
                                    central_diff=central_diff, cat_highlight=cat_highlight, plot_style=plot_style,
                                    palette=palette, black=black, font=font, title_font_size=title_font_size,
                                    labelsize=labelsize, axis_font_size=axis_font_size, filename=filename,
                                    file_format=file_format, plot_density=dpi)

        return plot

    def threshold_ccs_thresholds(self,
                                 item=None,
                                 anchor=False,
                                 rater=None,
                                 obs=None,
                                 warm=True,
                                 xmin=-5,
                                 xmax=5,
                                 no_of_classes=5,
                                 title=None,
                                 thresh_lines=False,
                                 central_diff=False,
                                 cat_highlight=None,
                                 plot_style='white',
                                 palette='dark blue',
                                 black=False,
                                 font='Times New Roman',
                                 title_font_size=15,
                                 axis_font_size=12,
                                 labelsize=12,
                                 filename=None,
                                 file_format='png',
                                 dpi=300):

        '''
        Plots Threshold Characteristic Curves, with optional
        overplotting of observed data and threshold lines.
        '''

        if item == 'none':
            item = None

        if (rater == 'none') or (rater == 'zero'):
            rater = None

        if anchor:
            if not hasattr(self, 'anchor_thresholds_thresholds'):
                print('Anchor calibration required')
                print('Run self.calibrate_thresholds_anchor()')
                return

        if anchor:
            difficulties = self.anchor_diffs_thresholds
            thresholds = self.anchor_thresholds_thresholds
            severities = self.anchor_severities_thresholds

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_thresholds

        shifts = np.zeros(self.max_score)
        shifts = pd.Series(shifts)
        shifts.index = [thr + 1 for thr in range(self.max_score)]

        if rater is None:
            if anchor:
                if item is None:
                    sev_df = pd.DataFrame(self.anchor_severities_thresholds)

                else:
                    sev_df = pd.DataFrame(self.severities_thresholds)

                shifts = sev_df.iloc[1:].mean(axis=1)

        if obs is not None:
            if anchor:
                if not hasattr(self, 'anchor_abils_thresholds'):
                    self.person_abils_thresholds(anchor=True)
                abilities = self.anchor_abils_thresholds

            else:
                if not hasattr(self, 'abils_thresholds'):
                    self.person_abils_thresholds()
                abilities = self.abils_thresholds

            xobsdata, yobsdata = self.class_intervals_thr_thresholds(abilities, difficulties, severities, item=item,
                                                                     rater=rater, shifts=shifts,
                                                                     no_of_classes=no_of_classes)

            if obs == 'all':
                obs = [i + 1 for i in range(self.max_score)]

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)

        if item is None:
            if rater is None:
                adj_thresholds = thresholds[1:]

            else:
                adj_thresholds = thresholds[1:] + severities[rater][1:]

        else:
            if rater is None:
                adj_thresholds = difficulties[item] + thresholds[1:]

            else:
                adj_thresholds = difficulties[item] + thresholds[1:] + severities[rater][1:]

        y = np.array([[1 / (1 + np.exp(threshold - ability))
                       for threshold in adj_thresholds]
                      for ability in abilities])

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Probability'

        plot = self.plot_data_thresholds(x_data=abilities, y_data=y, anchor=anchor, raters=rater, y_max=1, x_min=xmin,
                                         x_max=xmax, items=item, warm=warm, x_obs_data=xobsdata, y_obs_data=yobsdata,
                                         graph_title=graphtitle, y_label=ylabel, thresh_obs=obs,
                                         thresh_lines=thresh_lines, central_diff=central_diff,
                                         cat_highlight=cat_highlight, plot_style=plot_style, palette=palette,
                                         black=black, font=font, title_font_size=title_font_size, labelsize=labelsize,
                                         axis_font_size=axis_font_size, filename=filename, file_format=file_format,
                                         plot_density=dpi)

        return plot

    def threshold_ccs_matrix(self,
                             item=None,
                             anchor=False,
                             rater=None,
                             obs=None,
                             warm=True,
                             xmin=-5,
                             xmax=5,
                             no_of_classes=5,
                             title=None,
                             thresh_lines=False,
                             central_diff=False,
                             cat_highlight=None,
                             plot_style='white',
                             palette='dark blue',
                             black=False,
                             font='Times New Roman',
                             title_font_size=15,
                             axis_font_size=12,
                             labelsize=12,
                             filename=None,
                             file_format='png',
                             dpi=300):

        '''
        Plots Threshold Characteristic Curves for RSM, with optional
        overplotting of observed data and threshold lines.
        '''

        if item == 'none':
            item = None

        if (rater == 'none') or (rater == 'zero'):
            rater = None

        if anchor:
            if not hasattr(self, 'anchor_thresholds_matrix'):
                print('Anchor calibration required')
                print('Run self.calibrate_matrix_anchor()')
                return

        if anchor:
            difficulties = self.anchor_diffs_matrix
            thresholds = self.anchor_thresholds_matrix
            severities = self.anchor_severities_matrix

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_matrix

        shifts = np.zeros(self.max_score)
        shifts = pd.Series(shifts)
        shifts.index = [thr + 1 for thr in range(self.max_score)]

        if rater is None:
            if anchor:
                if item is None:
                    sev_df = pd.DataFrame(self.anchor_marginal_severities_thresholds)

                else:
                    sev_df = pd.DataFrame(self.marginal_severities_thresholds)

                shifts = sev_df.iloc[1:].mean(axis=1)

        if obs is not None:
            if anchor:
                if not hasattr(self, 'anchor_abils_matrix'):
                    self.person_abils_matrix(anchor=True)
                abilities = self.anchor_abils_matrix

            else:
                if not hasattr(self, 'abils_matrix'):
                    self.person_abils_matrix()
                abilities = self.abils_matrix

            xobsdata, yobsdata = self.class_intervals_thr_matrix(abilities, difficulties, severities, item=item,
                                                                 rater=rater, shifts=shifts,
                                                                 no_of_classes=no_of_classes)

            if obs == 'all':
                obs = [i + 1 for i in range(self.max_score)]

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)

        if item is None:
            if rater is None:
                adj_thresholds = thresholds[1:]

            else:
                marginal_severities = [np.mean([severities[rater][item][score]
                                                for item in self.dataframe.columns])
                                       for score in range(self.max_score + 1)]
                adj_thresholds = thresholds[1:] + marginal_severities[1:]

        else:
            if rater is None:
                adj_thresholds = difficulties[item] + thresholds[1:]

            else:
                adj_thresholds = difficulties[item] + thresholds[1:] + severities[rater][item][1:]

        y = np.array([[1 / (1 + np.exp(threshold - ability))
                       for threshold in adj_thresholds]
                      for ability in abilities])

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Probability'

        plot = self.plot_data_matrix(x_data=abilities, y_data=y, anchor=anchor, raters=rater, y_max=1, x_min=xmin,
                                     x_max=xmax, items=item, warm=warm, x_obs_data=xobsdata, y_obs_data=yobsdata,
                                     graph_title=graphtitle, y_label=ylabel, thresh_obs=obs, thresh_lines=thresh_lines,
                                     central_diff=central_diff, cat_highlight=cat_highlight, plot_style=plot_style,
                                     palette=palette, black=black, font=font, title_font_size=title_font_size,
                                     labelsize=labelsize, axis_font_size=axis_font_size, filename=filename,
                                     file_format=file_format, plot_density=dpi)

        return plot

    def iic_global(self,
                   item,
                   anchor=False,
                   rater=None,
                   xmin=-5,
                   xmax=5,
                   central_diff=False,
                   thresh_lines=False,
                   point_info_lines=None,
                   point_info_labels=False,
                   cat_highlight=None,
                   title=None,
                   plot_style='white',
                   palette='dark blue',
                   black=False,
                   font='Times New Roman',
                   title_font_size=15,
                   axis_font_size=12,
                   labelsize=12,
                   filename=None,
                   file_format='png',
                   dpi=300):

        '''
        Plots Item Information Curves.
        '''

        if anchor:
            if not hasattr(self, 'anchor_thresholds_global'):
                print('Anchor calibration required')
                print('Run self.calibrate_global_anchor()')
                return

        if anchor:
            difficulties = self.anchor_diffs_global
            thresholds = self.anchor_thresholds_global
            severities = self.anchor_severities_global

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_global

        dummy_sevs = pd.Series({'dummy_rater': 0})

        abilities = np.arange(-20, 20, 0.1)
        
        if rater is None:
            y = [self.variance_global(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                 for ability in abilities]
            
        else:
            y = [self.variance_global(ability, item, difficulties, rater, severities, thresholds)
                 for ability in abilities]

        y = np.array(y).reshape(len(abilities), 1)
        
        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Fisher information'

        plot = self.plot_data_global(x_data=abilities, y_data=y, anchor=anchor, raters=rater, x_min=xmin, x_max=xmax,
                                     y_max=max(y) * 1.1, items=item, thresh_lines=thresh_lines, plot_style=plot_style,
                                     palette=palette, point_info_lines_item=[item, point_info_lines],
                                     score_labels=point_info_labels, cat_highlight=cat_highlight,
                                     central_diff=central_diff, graph_title=graphtitle, y_label=ylabel, black=black,
                                     font=font, title_font_size=title_font_size, axis_font_size=axis_font_size,
                                     labelsize=labelsize, filename=filename, plot_density=dpi, file_format=file_format)

        return plot

    def iic_items(self,
                   item,
                   anchor=False,
                   rater=None,
                   xmin=-5,
                   xmax=5,
                   central_diff=False,
                   thresh_lines=False,
                   point_info_lines=None,
                   point_info_labels=False,
                   cat_highlight=None,
                   title=None,
                   plot_style='white',
                   palette='dark blue',
                   black=False,
                   font='Times New Roman',
                   title_font_size=15,
                   axis_font_size=12,
                   labelsize=12,
                   filename=None,
                   file_format='png',
                   dpi=300):

        '''
        Plots Item Information Curves.
        '''

        if anchor:
            if not hasattr(self, 'anchor_thresholds_items'):
                print('Anchor calibration required')
                print('Run self.calibrate_items_anchor()')
                return

        if anchor:
            difficulties = self.anchor_diffs_items
            thresholds = self.anchor_thresholds_items
            severities = self.anchor_severities_items

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_items

        dummy_sevs = {'dummy_rater': {item: 0}}

        abilities = np.arange(-20, 20, 0.1)
        
        if rater is None:
            y = [self.variance_items(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                 for ability in abilities]
            
        else:
            y = [self.variance_items(ability, item, difficulties, rater, severities, thresholds)
                 for ability in abilities]
        
        y = np.array(y).reshape(len(abilities), 1)

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Fisher information'

        plot = self.plot_data_items(x_data=abilities, y_data=y, anchor=anchor, raters=rater, x_min=xmin, x_max=xmax,
                                    y_max=max(y) * 1.1, items=item, thresh_lines=thresh_lines, plot_style=plot_style,
                                    palette=palette, point_info_lines_item=[item, point_info_lines],
                                    score_labels=point_info_labels, cat_highlight=cat_highlight,
                                    central_diff=central_diff, graph_title=graphtitle, y_label=ylabel, black=black,
                                    font=font, title_font_size=title_font_size, axis_font_size=axis_font_size,
                                    labelsize=labelsize, filename=filename, plot_density=dpi, file_format=file_format)

        return plot

    def iic_thresholds(self,
                       item,
                       anchor=False,
                       rater=None,
                       xmin=-5,
                       xmax=5,
                       central_diff=False,
                       thresh_lines=False,
                       point_info_lines=None,
                       point_info_labels=False,
                       cat_highlight=None,
                       title=None,
                       plot_style='white',
                       palette='dark blue',
                       black=False,
                       font='Times New Roman',
                       title_font_size=15,
                       axis_font_size=12,
                       labelsize=12,
                       filename=None,
                       file_format='png',
                       dpi=300):

        '''
        Plots Item Information Curves.
        '''

        if anchor:
            if not hasattr(self, 'anchor_thresholds_thresholds'):
                print('Anchor calibration required')
                print('Run self.calibrate_thresholds_anchor()')
                return

        if anchor:
            difficulties = self.anchor_diffs_thresholds
            thresholds = self.anchor_thresholds_thresholds
            severities = self.anchor_severities_thresholds

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_thresholds

        dummy_sevs = {'dummy_rater': np.zeros(self.max_score + 1)}

        abilities = np.arange(-20, 20, 0.1)
        
        if rater is None:
            y = [self.variance_thresholds(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                 for ability in abilities]
            
        else:
            y = [self.variance_thresholds(ability, item, difficulties, rater, severities, thresholds)
                 for ability in abilities]
        
        y = np.array(y).reshape(len(abilities), 1)

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Fisher information'

        plot = self.plot_data_thresholds(x_data=abilities, y_data=y, anchor=anchor, raters=rater, x_min=xmin,
                                         x_max=xmax, y_max=max(y) * 1.1, items=item, thresh_lines=thresh_lines,
                                         plot_style=plot_style,  palette=palette,
                                         point_info_lines_item=[item, point_info_lines], score_labels=point_info_labels,
                                         cat_highlight=cat_highlight, central_diff=central_diff, graph_title=graphtitle,
                                         y_label=ylabel, black=black, font=font, title_font_size=title_font_size,
                                         axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                                         plot_density=dpi, file_format=file_format)

        return plot

    def iic_matrix(self,
                   item,
                   anchor=False,
                   rater=None,
                   xmin=-5,
                   xmax=5,
                   central_diff=False,
                   thresh_lines=False,
                   point_info_lines=None,
                   point_info_labels=False,
                   cat_highlight=None,
                   title=None,
                   plot_style='white',
                   palette='dark blue',
                   black=False,
                   font='Times New Roman',
                   title_font_size=15,
                   axis_font_size=12,
                   labelsize=12,
                   filename=None,
                   file_format='png',
                   dpi=300):

        '''
        Plots Item Information Curves.
        '''

        if anchor:
            if not hasattr(self, 'anchor_thresholds_matrix'):
                print('Anchor calibration required')
                print('Run self.calibrate_matrix_anchor()')
                return

        if anchor:
            difficulties = self.anchor_diffs_matrix
            thresholds = self.anchor_thresholds_matrix
            severities = self.anchor_severities_matrix

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_matrix

        dummy_sevs = {'dummy_rater': {item: np.zeros(self.max_score + 1)
                                      for item in self.dataframe.columns}}

        abilities = np.arange(-20, 20, 0.1)
        
        if rater is None:
            y = [self.variance_matrix(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                 for ability in abilities]
            
        else:
            y = [self.variance_matrix(ability, item, difficulties, rater, severities, thresholds)
                 for ability in abilities]

        y = np.array(y).reshape(len(abilities), 1)
        
        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Fisher information'

        plot = self.plot_data_matrix(x_data=abilities, y_data=y, anchor=anchor, raters=rater, x_min=xmin, x_max=xmax,
                                     y_max=max(y) * 1.1, items=item, thresh_lines=thresh_lines, plot_style=plot_style,
                                     palette=palette, point_info_lines_item=[item, point_info_lines],
                                     score_labels=point_info_labels, cat_highlight=cat_highlight,
                                     central_diff=central_diff, graph_title=graphtitle, y_label=ylabel, black=black,
                                     font=font, title_font_size=title_font_size, axis_font_size=axis_font_size,
                                     labelsize=labelsize, filename=filename, plot_density=dpi, file_format=file_format)

        return plot

    def tcc_global(self,
                   anchor=False,
                   items='all',
                   raters='zero',
                   obs=False,
                   xmin=-5,
                   xmax=5,
                   no_of_classes=5,
                   title=None,
                   score_lines=None,
                   score_labels=False,
                   plot_style='white',
                   palette='dark blue',
                   black=False,
                   font='Times New Roman',
                   title_font_size=15,
                   axis_font_size=12,
                   labelsize=12,
                   filename=None,
                   file_format='png',
                   dpi=300):

        '''
        Plots Test Characteristic Curve, with optional overplotting
        of observed data, threshold lines and expected score threshold lines.
        '''

        if isinstance(items, str):
            if (items == 'all') | (items == 'none'):
                items = None

        if isinstance(raters, str):
            if raters == 'all':
                raters = self.raters.tolist()

            elif raters == 'none':
                raters = None

            elif raters == 'zero':
                raters = None

        if anchor:
            if not hasattr(self, 'anchor_thresholds_global'):
                print('Anchor calibration required')
                print('Run self.calibrate_global_anchor()')
                return

            else:
                difficulties = self.anchor_diffs_global
                thresholds = self.anchor_thresholds_global
                severities = self.anchor_severities_global

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_global

        dummy_sevs = pd.Series({'dummy_rater': 0})

        if obs:
            if anchor:
                if not hasattr(self, 'anchor_abils_global'):
                    self.person_abils_global(anchor=True)
                abilities = self.anchor_abils_global

            else:
                if not hasattr(self, 'abils_global'):
                    self.person_abils_global()
                abilities = self.abils_global

            xobsdata, yobsdata = self.class_intervals(abilities, items=items, raters=raters,
                                                      no_of_classes=no_of_classes)

            yobsdata = np.array(yobsdata).reshape((-1, 1))

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)

        if items is None:
            if raters is None:
                y = [sum(self.exp_score_global(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                         for item in self.dataframe.columns)
                     for ability in abilities]

            elif isinstance(raters, list):
                y = [sum(self.exp_score_global(ability, item, difficulties, rater, severities, thresholds)
                         for item in self.dataframe.columns for rater in raters)
                     for ability in abilities]

            else:
                y = [sum(self.exp_score_global(ability, item, difficulties, raters, severities, thresholds)
                         for item in self.dataframe.columns)
                     for ability in abilities]

        if isinstance(items, list):
            if raters is None:
                y = [sum(self.exp_score_global(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                         for item in items)
                     for ability in abilities]

            elif isinstance(raters, list):
                y = [sum(self.exp_score_global(ability, item, difficulties, rater, severities, thresholds)
                         for item in items for rater in raters)
                     for ability in abilities]

            else:
                y = [sum(self.exp_score_global(ability, item, difficulties, raters, severities, thresholds)
                         for item in items)
                     for ability in abilities]

        if isinstance(items, str):
            if raters is None:
                y = [self.exp_score_global(ability, items, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                     for ability in abilities]

            elif isinstance(raters, list):
                y = [sum(self.exp_score_global(ability, items, difficulties, rater, severities, thresholds)
                         for rater in raters)
                     for ability in abilities]

            else:
                y = [self.exp_score_global(ability, items, difficulties, raters, severities, thresholds)
                     for ability in abilities]

        y = np.array(y).reshape(len(abilities), 1)

        if items is None:
            no_of_items = self.no_of_items

        elif isinstance(items, str):
            no_of_items = 1

        else:
            no_of_items = len(items)

        if (raters is None) or (isinstance(raters, str)):
            no_of_raters = 1

        else:
            no_of_raters = len(raters)

        y_max = self.max_score * no_of_items * no_of_raters

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Expected score'

        plot = self.plot_data_global(x_data=abilities, y_data=y, anchor=anchor,  items=items, raters=raters,
                                     x_obs_data=xobsdata, y_obs_data=yobsdata, x_min=xmin, x_max=xmax, y_max=y_max,
                                     score_lines_test=score_lines, score_labels=score_labels, graph_title=graphtitle,
                                     y_label=ylabel, obs=obs, plot_style=plot_style, palette=palette, black=black,
                                     font=font, title_font_size=title_font_size, axis_font_size=axis_font_size,
                                     labelsize=labelsize, filename=filename, plot_density=dpi, file_format=file_format)

        return plot

    def tcc_items(self,
                  anchor=False,
                  items='all',
                  raters='zero',
                  obs=False,
                  xmin=-5,
                  xmax=5,
                  no_of_classes=5,
                  title=None,
                  score_lines=None,
                  score_labels=False,
                  plot_style='white',
                  palette='dark blue',
                  black=False,
                  font='Times New Roman',
                  title_font_size=15,
                  axis_font_size=12,
                  labelsize=12,
                  filename=None,
                  file_format='png',
                  dpi=300):

        '''
        Plots Test Characteristic Curve.
        '''

        if isinstance(items, str):
            if (items == 'all') | (items == 'none'):
                items = None

        if isinstance(raters, str):
            if raters == 'all':
                raters = self.raters.tolist()

            elif raters == 'none':
                raters = None

            elif raters == 'zero':
                raters = None

        if anchor:
            if not hasattr(self, 'anchor_thresholds_items'):
                print('Anchor calibration required')
                print('Run self.calibrate_items_anchor()')
                return

        if anchor:
            difficulties = self.anchor_diffs_items
            thresholds = self.anchor_thresholds_items
            severities = self.anchor_severities_items

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_items

        dummy_sevs = {'dummy_rater': {item: 0 for item in self.dataframe.columns}}

        if obs:
            if anchor:
                if not hasattr(self, 'anchor_abils_items'):
                    self.person_abils_items(anchor=True)
                abilities = self.anchor_abils_items

            else:
                if not hasattr(self, 'abils_items'):
                    self.person_abils_items()
                abilities = self.abils_items

            xobsdata, yobsdata = self.class_intervals(abilities, items=items, raters=raters,
                                                      no_of_classes=no_of_classes)

            yobsdata = np.array(yobsdata).reshape((-1, 1))

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)

        if items is None:
            if raters is None:
                y = [sum(self.exp_score_items(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                         for item in self.dataframe.columns)
                     for ability in abilities]

            elif isinstance(raters, list):
                y = [sum(self.exp_score_items(ability, item, difficulties, rater, severities, thresholds)
                         for item in self.dataframe.columns for rater in raters)
                     for ability in abilities]

            else:
                y = [sum(self.exp_score_items(ability, item, difficulties, raters, severities, thresholds)
                         for item in self.dataframe.columns)
                     for ability in abilities]

        if isinstance(items, list):
            if raters is None:
                y = [sum(self.exp_score_items(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                         for item in items)
                     for ability in abilities]

            elif isinstance(raters, list):
                y = [sum(self.exp_score_items(ability, item, difficulties, rater, severities, thresholds)
                         for item in items for rater in raters)
                     for ability in abilities]

            else:
                y = [sum(self.exp_score_items(ability, item, difficulties, raters, severities, thresholds)
                         for item in items)
                     for ability in abilities]

        if isinstance(items, str):
            if raters is None:
                y = [self.exp_score_items(ability, items, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                     for ability in abilities]

            elif isinstance(raters, list):
                y = [sum(self.exp_score_items(ability, items, difficulties, rater, severities, thresholds)
                         for rater in raters)
                     for ability in abilities]

            else:
                y = [self.exp_score_items(ability, items, difficulties, raters, severities, thresholds)
                     for ability in abilities]

        y = np.array(y).reshape(len(abilities), 1)

        if items is None:
            no_of_items = self.no_of_items

        elif isinstance(items, str):
            no_of_items = 1

        else:
            no_of_items = len(items)

        if (raters is None) or (isinstance(raters, str)):
            no_of_raters = 1

        else:
            no_of_raters = len(raters)

        y_max = self.max_score * no_of_items * no_of_raters

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Expected score'

        plot = self.plot_data_items(x_data=abilities, y_data=y, anchor=anchor,  items=items, raters=raters,
                                    x_obs_data=xobsdata, y_obs_data=yobsdata, x_min=xmin, x_max=xmax, y_max=y_max,
                                    score_lines_test=score_lines, score_labels=score_labels, graph_title=graphtitle,
                                    y_label=ylabel, obs=obs, plot_style=plot_style, palette=palette, black=black,
                                    font=font, title_font_size=title_font_size, axis_font_size=axis_font_size,
                                    labelsize=labelsize, filename=filename, plot_density=dpi, file_format=file_format)
        return plot

    def tcc_thresholds(self,
                       anchor=False,
                       items='all',
                       raters='zero',
                       obs=False,
                       xmin=-5,
                       xmax=5,
                       no_of_classes=5,
                       title=None,
                       score_lines=None,
                       score_labels=False,
                       plot_style='white',
                       palette='dark blue',
                       black=False,
                       font='Times New Roman',
                       title_font_size=15,
                       axis_font_size=12,
                       labelsize=12,
                       filename=None,
                       file_format='png',
                       dpi=300):

        '''
        Plots Test Characteristic Curve.
        '''

        if isinstance(items, str):
            if (items == 'all') | (items == 'none'):
                items = None

        if isinstance(raters, str):
            if raters == 'all':
                raters = self.raters.tolist()

            elif raters == 'none':
                raters = None

            elif raters == 'zero':
                raters = None

        if anchor:
            if not hasattr(self, 'anchor_thresholds_thresholds'):
                print('Anchor calibration required')
                print('Run self.calibrate_thresholds_anchor()')
                return

        if anchor:
            difficulties = self.anchor_diffs_thresholds
            thresholds = self.anchor_thresholds_thresholds
            severities = self.anchor_severities_thresholds

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_thresholds

        dummy_sevs = {'dummy_rater': np.zeros(self.max_score + 1)}

        if obs:
            if anchor:
                if not hasattr(self, 'anchor_abils_thresholds'):
                    self.person_abils_thresholds(anchor=True)
                abilities = self.anchor_abils_thresholds

            else:
                if not hasattr(self, 'abils_thresholds'):
                    self.person_abils_thresholds()
                abilities = self.abils_thresholds

            xobsdata, yobsdata = self.class_intervals(abilities, items=items, raters=raters,
                                                      no_of_classes=no_of_classes)

            yobsdata = np.array(yobsdata).reshape((-1, 1))

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)

        if items is None:
            if raters is None:
                y = [sum(self.exp_score_thresholds(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                         for item in self.dataframe.columns)
                     for ability in abilities]

            elif isinstance(raters, list):
                y = [sum(self.exp_score_thresholds(ability, item, difficulties, rater, severities, thresholds)
                         for item in self.dataframe.columns for rater in raters)
                     for ability in abilities]

            else:
                y = [sum(self.exp_score_thresholds(ability, item, difficulties, raters, severities, thresholds)
                         for item in self.dataframe.columns)
                     for ability in abilities]

        if isinstance(items, list):
            if raters is None:
                y = [sum(self.exp_score_thresholds(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                         for item in items)
                     for ability in abilities]

            elif isinstance(raters, list):
                y = [sum(self.exp_score_thresholds(ability, item, difficulties, rater, severities, thresholds)
                         for item in items for rater in raters)
                     for ability in abilities]

            else:
                y = [sum(self.exp_score_thresholds(ability, item, difficulties, raters, severities, thresholds)
                         for item in items)
                     for ability in abilities]

        if isinstance(items, str):
            if raters is None:
                y = [self.exp_score_thresholds(ability, items, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                     for ability in abilities]

            elif isinstance(raters, list):
                y = [sum(self.exp_score_thresholds(ability, items, difficulties, rater, severities, thresholds)
                         for rater in raters)
                     for ability in abilities]

            else:
                y = [self.exp_score_thresholds(ability, items, difficulties, raters, severities, thresholds)
                     for ability in abilities]

        y = np.array(y).reshape(len(abilities), 1)

        if items is None:
            no_of_items = self.no_of_items

        elif isinstance(items, str):
            no_of_items = 1

        else:
            no_of_items = len(items)

        if (raters is None) or (isinstance(raters, str)):
            no_of_raters = 1

        else:
            no_of_raters = len(raters)

        y_max = self.max_score * no_of_items * no_of_raters

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Expected score'

        plot = self.plot_data_thresholds(x_data=abilities, y_data=y, anchor=anchor,  items=items, raters=raters,
                                         x_obs_data=xobsdata, y_obs_data=yobsdata, x_min=xmin, x_max=xmax, y_max=y_max,
                                         score_lines_test=score_lines, score_labels=score_labels,
                                         graph_title=graphtitle, y_label=ylabel, obs=obs, plot_style=plot_style,
                                         palette=palette, black=black, font=font, title_font_size=title_font_size,
                                         axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                                         plot_density=dpi, file_format=file_format)
        return plot

    def tcc_matrix(self,
                   anchor=False,
                   items='all',
                   raters='zero',
                   obs=False,
                   xmin=-5,
                   xmax=5,
                   no_of_classes=5,
                   title=None,
                   score_lines=None,
                   score_labels=False,
                   plot_style='white',
                   palette='dark blue',
                   black=False,
                   font='Times New Roman',
                   title_font_size=15,
                   axis_font_size=12,
                   labelsize=12,
                   filename=None,
                   file_format='png',
                   dpi=300):

        '''
        Plots Test Characteristic Curve.
        '''

        if isinstance(items, str):
            if (items == 'all') | (items == 'none'):
                items = None

        if isinstance(raters, str):
            if raters == 'all':
                raters = self.raters.tolist()

            elif raters == 'none':
                raters = None

            elif raters == 'zero':
                raters = None

        if anchor:
            if not hasattr(self, 'anchor_thresholds_matrix'):
                print('Anchor calibration required')
                print('Run self.calibrate_matrix_anchor()')
                return

        if anchor:
            difficulties = self.anchor_diffs_matrix
            thresholds = self.anchor_thresholds_matrix
            severities = self.anchor_severities_matrix

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_matrix

        dummy_sevs = {'dummy_rater': {item: np.zeros(self.max_score + 1)
                                      for item in self.dataframe.columns}}

        if obs:
            if anchor:
                if not hasattr(self, 'anchor_abils_matrix'):
                    self.person_abils_matrix(anchor=True)
                abilities = self.anchor_abils_matrix

            else:
                if not hasattr(self, 'abils_matrix'):
                    self.person_abils_matrix()
                abilities = self.abils_matrix

            xobsdata, yobsdata = self.class_intervals(abilities, items=items, raters=raters,
                                                      no_of_classes=no_of_classes)

            yobsdata = np.array(yobsdata).reshape((-1, 1))

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)

        if items is None:
            if raters is None:
                y = [sum(self.exp_score_matrix(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                         for item in self.dataframe.columns)
                     for ability in abilities]

            elif isinstance(raters, list):
                y = [sum(self.exp_score_matrix(ability, item, difficulties, rater, severities, thresholds)
                         for item in self.dataframe.columns for rater in raters)
                     for ability in abilities]

            else:
                y = [sum(self.exp_score_matrix(ability, item, difficulties, raters, severities, thresholds)
                         for item in self.dataframe.columns)
                     for ability in abilities]

        if isinstance(items, list):
            if raters is None:
                y = [sum(self.exp_score_matrix(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                         for item in items)
                     for ability in abilities]

            elif isinstance(raters, list):
                y = [sum(self.exp_score_matrix(ability, item, difficulties, rater, severities, thresholds)
                         for item in items for rater in raters)
                     for ability in abilities]

            else:
                y = [sum(self.exp_score_matrix(ability, item, difficulties, raters, severities, thresholds)
                         for item in items)
                     for ability in abilities]

        if isinstance(items, str):
            if raters is None:
                y = [self.exp_score_matrix(ability, items, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                     for ability in abilities]

            elif isinstance(raters, list):
                y = [sum(self.exp_score_matrix(ability, items, difficulties, rater, severities, thresholds)
                         for rater in raters)
                     for ability in abilities]

            else:
                y = [self.exp_score_matrix(ability, items, difficulties, raters, severities, thresholds)
                     for ability in abilities]

        y = np.array(y).reshape(len(abilities), 1)

        if items is None:
            no_of_items = self.no_of_items

        elif isinstance(items, str):
            no_of_items = 1

        else:
            no_of_items = len(items)

        if (raters is None) or (isinstance(raters, str)):
            no_of_raters = 1

        else:
            no_of_raters = len(raters)

        y_max = self.max_score * no_of_items * no_of_raters

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Expected score'

        plot = self.plot_data_matrix(x_data=abilities, y_data=y, anchor=anchor,  items=items, raters=raters,
                                     x_obs_data=xobsdata, y_obs_data=yobsdata, x_min=xmin, x_max=xmax, y_max=y_max,
                                     score_lines_test=score_lines, score_labels=score_labels, graph_title=graphtitle,
                                     y_label=ylabel, obs=obs, plot_style=plot_style, palette=palette, black=black,
                                     font=font, title_font_size=title_font_size, axis_font_size=axis_font_size,
                                     labelsize=labelsize, filename=filename, plot_density=dpi, file_format=file_format)
        return plot

    def test_info_global(self,
                         anchor=False,
                         items=None,
                         raters=None,
                         point_info_lines=None,
                         point_info_labels=False,
                         xmin=-5,
                         xmax=5,
                         ymax=None,
                         title=None,
                         plot_style='white',
                         palette='dark blue',
                         black=False,
                         font='Times New Roman',
                         title_font_size=15,
                         axis_font_size=12,
                         labelsize=12,
                         filename=None,
                         file_format='png',
                         dpi=300):

        '''
        Plots Test Information Curve for global MFRM.
        '''

        if anchor:
            if not hasattr(self, 'anchor_thresholds_global'):
                print('Anchor calibration required')
                print('Run self.calibrate_global_anchor()')
                return

        if anchor:
            difficulties = self.anchor_diffs_global
            thresholds = self.anchor_thresholds_global
            severities = self.anchor_severities_global

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_global

        if isinstance(items, str):
            if (items == 'all') | (items == 'none'):
                items = None

            else:
                items = [items]

        if isinstance(raters, str):
            if raters == 'all':
                raters = self.raters.tolist()

            elif raters == 'none':
                raters = None

            elif raters == 'zero':
                raters = None

            else:
                raters = [raters]

        dummy_sevs = pd.Series({'dummy_rater': 0})

        abilities = np.arange(-20, 20, 0.1)

        if items is None:
            if raters is None:
                y = [sum(self.variance_global(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                         for item in self.dataframe.columns)
                     for ability in abilities]
    
            else:
                y = [sum(self.variance_global(ability, item, difficulties, rater, severities, thresholds)
                         for item in self.dataframe.columns for rater in raters)
                     for ability in abilities]
                
        else:
            if raters is None:
                y = [sum(self.variance_global(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                         for item in items)
                     for ability in abilities]
    
            else:
                y = [sum(self.variance_global(ability, item, difficulties, rater, severities, thresholds)
                         for item in items for rater in raters)
                     for ability in abilities]

        y = np.array(y).reshape(len(abilities), 1)

        if ymax is None:
            ymax = max(y) * 1.1

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Fisher information'

        plot = self.plot_data_global(x_data=abilities, y_data=y, anchor=anchor, items=items, raters=raters, x_min=xmin,
                                     x_max=xmax, y_max=ymax, point_info_lines_test=point_info_lines,
                                     score_labels=point_info_labels, graph_title=graphtitle, y_label=ylabel,
                                     plot_style=plot_style, palette=palette, black=black, font=font,
                                     title_font_size=title_font_size, axis_font_size=axis_font_size,
                                     labelsize=labelsize, filename=filename, plot_density=dpi, file_format=file_format)

        return plot

    def test_info_items(self,
                        anchor=False,
                        items=None,
                        raters=None,
                        point_info_lines=None,
                        point_info_labels=False,
                        xmin=-5,
                        xmax=5,
                        ymax=None,
                        title=None,
                        plot_style='white',
                        palette='dark blue',
                        black=False,
                        font='Times New Roman',
                        title_font_size=15,
                        axis_font_size=12,
                        labelsize=12,
                        filename=None,
                        file_format='png',
                        dpi=300):

        '''
        Plots Test Information Curve.
        '''

        if anchor:
            if not hasattr(self, 'anchor_thresholds_items'):
                print('Anchor calibration required')
                print('Run self.calibrate_global_items()')
                return

        if anchor:
            difficulties = self.anchor_diffs_items
            thresholds = self.anchor_thresholds_items
            severities = self.anchor_severities_items

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_items

        if isinstance(items, str):
            if (items == 'all') | (items == 'none'):
                items = None

            else:
                items = [items]

        if isinstance(raters, str):
            if raters == 'all':
                raters = self.raters.tolist()

            elif raters == 'none':
                raters = None

            elif raters == 'zero':
                raters = None

            else:
                raters = [raters]

        dummy_sevs = {'dummy_rater': {item: 0 for item in self.dataframe.columns}}

        abilities = np.arange(-20, 20, 0.1)

        if items is None:
            if raters is None:
                y = [sum(self.variance_items(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                         for item in self.dataframe.columns)
                     for ability in abilities]

            else:
                y = [sum(self.variance_items(ability, item, difficulties, rater, severities, thresholds)
                         for item in self.dataframe.columns for rater in raters)
                     for ability in abilities]

        else:
            if raters is None:
                y = [sum(self.variance_items(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                         for item in items)
                     for ability in abilities]

            else:
                y = [sum(self.variance_items(ability, item, difficulties, rater, severities, thresholds)
                         for item in items for rater in raters)
                     for ability in abilities]

        y = np.array(y).reshape(len(abilities), 1)

        if ymax is None:
            ymax = max(y) * 1.1

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Fisher information'

        plot = self.plot_data_items(x_data=abilities, y_data=y, anchor=anchor, items=items, raters=raters, x_min=xmin,
                                    x_max=xmax, y_max=ymax, point_info_lines_test=point_info_lines,
                                    score_labels=point_info_labels, graph_title=graphtitle, y_label=ylabel,
                                    plot_style=plot_style, palette=palette, black=black, font=font,
                                    title_font_size=title_font_size, axis_font_size=axis_font_size,
                                    labelsize=labelsize, filename=filename, plot_density=dpi, file_format=file_format)

        return plot

    def test_info_thresholds(self,
                             anchor=False,
                             items=None,
                             raters=None,
                             point_info_lines=None,
                             point_info_labels=False,
                             xmin=-5,
                             xmax=5,
                             ymax=None,
                             title=None,
                             plot_style='white',
                             palette='dark blue',
                             black=False,
                             font='Times New Roman',
                             title_font_size=15,
                             axis_font_size=12,
                             labelsize=12,
                             filename=None,
                             file_format='png',
                             dpi=300):

        '''
        Plots Test Information Curve.
        '''

        if anchor:
            if not hasattr(self, 'anchor_thresholds_thresholds'):
                print('Anchor calibration required')
                print('Run self.calibrate_thresholds_anchor()')
                return

        if anchor:
            difficulties = self.anchor_diffs_thresholds
            thresholds = self.anchor_thresholds_thresholds
            severities = self.anchor_severities_thresholds

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_thresholds

        if isinstance(items, str):
            if (items == 'all') | (items == 'none'):
                items = None

            else:
                items = [items]

        if isinstance(raters, str):
            if raters == 'all':
                raters = self.raters.tolist()

            elif raters == 'none':
                raters = None

            elif raters == 'zero':
                raters = None

            else:
                raters = [raters]

        dummy_sevs = {'dummy_rater': np.zeros(self.max_score + 1)}

        abilities = np.arange(-20, 20, 0.1)

        if items is None:
            if raters is None:
                y = [sum(self.variance_thresholds(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                         for item in self.dataframe.columns)
                     for ability in abilities]

            else:
                y = [sum(self.variance_thresholds(ability, item, difficulties, rater, severities, thresholds)
                         for item in self.dataframe.columns for rater in raters)
                     for ability in abilities]

        else:
            if raters is None:
                y = [sum(self.variance_thresholds(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                         for item in items)
                     for ability in abilities]

            else:
                y = [sum(self.variance_thresholds(ability, item, difficulties, rater, severities, thresholds)
                         for item in items for rater in raters)
                     for ability in abilities]

        y = np.array(y).reshape(len(abilities), 1)

        if ymax is None:
            ymax = max(y) * 1.1

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Fisher information'

        plot = self.plot_data_thresholds(x_data=abilities, y_data=y, anchor=anchor, items=items, raters=raters,
                                         x_min=xmin, x_max=xmax, y_max=ymax, point_info_lines_test=point_info_lines,
                                         score_labels=point_info_labels, graph_title=graphtitle, y_label=ylabel,
                                         plot_style=plot_style, palette=palette, black=black, font=font,
                                         title_font_size=title_font_size, axis_font_size=axis_font_size,
                                         labelsize=labelsize, filename=filename, plot_density=dpi, file_format=file_format)

        return plot

    def test_info_matrix(self,
                         anchor=False,
                         items=None,
                         raters=None,
                         point_info_lines=None,
                         point_info_labels=False,
                         xmin=-5,
                         xmax=5,
                         ymax=None,
                         title=None,
                         plot_style='white',
                         palette='dark blue',
                         black=False,
                         font='Times New Roman',
                         title_font_size=15,
                         axis_font_size=12,
                         labelsize=12,
                         filename=None,
                         file_format='png',
                         dpi=300):

        '''
        Plots Test Information Curve.
        '''

        if anchor:
            if not hasattr(self, 'anchor_thresholds_matrix'):
                print('Anchor calibration required')
                print('Run self.calibrate_matrix_anchor()')
                return

        if anchor:
            difficulties = self.anchor_diffs_matrix
            thresholds = self.anchor_thresholds_matrix
            severities = self.anchor_severities_matrix

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_matrix

        if isinstance(items, str):
            if (items == 'all') | (items == 'none'):
                items = None

            else:
                items = [items]

        if isinstance(raters, str):
            if raters == 'all':
                raters = self.raters.tolist()

            elif raters == 'none':
                raters = None

            elif raters == 'zero':
                raters = None

            else:
                raters = [raters]

        dummy_sevs = {'dummy_rater': {item: np.zeros(self.max_score + 1)
                                      for item in self.dataframe.columns}}

        abilities = np.arange(-20, 20, 0.1)

        if items is None:
            if raters is None:
                y = [sum(self.variance_matrix(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                         for item in self.dataframe.columns)
                     for ability in abilities]

            else:
                y = [sum(self.variance_matrix(ability, item, difficulties, rater, severities, thresholds)
                         for item in self.dataframe.columns for rater in raters)
                     for ability in abilities]

        else:
            if raters is None:
                y = [sum(self.variance_matrix(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                         for item in items)
                     for ability in abilities]

            else:
                y = [sum(self.variance_matrix(ability, item, difficulties, rater, severities, thresholds)
                         for item in items for rater in raters)
                     for ability in abilities]

        y = np.array(y).reshape(len(abilities), 1)

        if ymax is None:
            ymax = max(y) * 1.1

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Fisher information'

        plot = self.plot_data_matrix(x_data=abilities, y_data=y, anchor=anchor, items=items, raters=raters, x_min=xmin,
                                     x_max=xmax, y_max=ymax, point_info_lines_test=point_info_lines,
                                     score_labels=point_info_labels, graph_title=graphtitle, y_label=ylabel,
                                     plot_style=plot_style, palette=palette, black=black, font=font,
                                     title_font_size=title_font_size, axis_font_size=axis_font_size,
                                     labelsize=labelsize, filename=filename, plot_density=dpi, file_format=file_format)

        return plot

    def test_csem_global(self,
                         anchor=False,
                         items=None,
                         raters=None,
                         point_csem_lines=None,
                         point_csem_labels=False,
                         xmin=-5,
                         xmax=5,
                         ymax=5,
                         title=None,
                         plot_style='white',
                         palette='dark blue',
                         black=False,
                         font='Times New Roman',
                         title_font_size=15,
                         axis_font_size=12,
                         labelsize=12,
                         filename=None,
                         file_format='png',
                         dpi=300):

        '''
        Plots Test Conditional Standard Error of Measurement Curve for RSM.
        '''

        if anchor:
            if not hasattr(self, 'anchor_thresholds_global'):
                print('Anchor calibration required')
                print('Run self.calibrate_global_anchor()')
                return

        if anchor:
            difficulties = self.anchor_diffs_global
            thresholds = self.anchor_thresholds_global
            severities = self.anchor_severities_global

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_global

        if isinstance(items, str):
            if (items == 'all') | (items == 'none'):
                items = None

            else:
                items = [items]

        if isinstance(raters, str):
            if raters == 'all':
                raters = self.raters.tolist()

            elif raters == 'none':
                raters = None

            elif raters == 'zero':
                raters = None

            else:
                raters = [raters]

        abilities = np.arange(-20, 20, 0.1)

        if items is None:
            if raters is None:
                y = [sum(self.variance_global(ability, item, difficulties, 'dummy_rater',
                                              pd.Series({'dummy_rater': 0}), thresholds)
                         for item in self.dataframe.columns)
                     for ability in abilities]

            else:
                y = [sum(self.variance_global(ability, item, difficulties, rater, severities, thresholds)
                         for item in self.dataframe.columns for rater in raters)
                     for ability in abilities]

        else:
            if raters is None:
                y = [sum(self.variance_global(ability, item, difficulties, 'dummy_rater',
                                              pd.Series({'dummy_rater': 0}), thresholds)
                         for item in items)
                     for ability in abilities]

            else:
                y = [sum(self.variance_global(ability, item, difficulties, rater, severities, thresholds)
                         for item in items for rater in raters)
                     for ability in abilities]

        y = np.array(y)
        y = 1 / (y ** 0.5)
        y = y.reshape(len(abilities), 1)

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Conditional SEM'

        plot = self.plot_data_global(x_data=abilities, y_data=y, anchor=anchor, items=items, raters=raters,
                                     x_min=xmin, x_max=xmax, y_max=ymax, point_csem_lines=point_csem_lines,
                                     score_labels=point_csem_labels, graph_title=graphtitle, y_label=ylabel,
                                     plot_style=plot_style, palette=palette, black=black, font=font,
                                     title_font_size=title_font_size, axis_font_size=axis_font_size,
                                     labelsize=labelsize, filename=filename, plot_density=dpi, file_format=file_format)

        return plot

    def test_csem_items(self,
                        anchor=False,
                        items=None,
                        raters=None,
                        point_csem_lines=None,
                        point_csem_labels=False,
                        xmin=-5,
                        xmax=5,
                        ymax=5,
                        title=None,
                        plot_style='white',
                        palette='dark blue',
                        black=False,
                        font='Times New Roman',
                        title_font_size=15,
                        axis_font_size=12,
                        labelsize=12,
                        filename=None,
                        file_format='png',
                        dpi=300):

        '''
        Plots Test Conditional Standard Error of Measurement Curve for RSM.
        '''

        if anchor:
            if not hasattr(self, 'anchor_thresholds_items'):
                print('Anchor calibration required')
                print('Run self.calibrate_global_items()')
                return

        if anchor:
            difficulties = self.anchor_diffs_items
            thresholds = self.anchor_thresholds_items
            severities = self.anchor_severities_items

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_items

        if isinstance(items, str):
            if (items == 'all') | (items == 'none'):
                items = None

            else:
                items = [items]

        if isinstance(raters, str):
            if raters == 'all':
                raters = self.raters.tolist()

            elif raters == 'none':
                raters = None

            elif raters == 'zero':
                raters = None

            else:
                raters = [raters]

        dummy_sevs = {'dummy_rater': {item: 0 for item in self.dataframe.columns}}

        abilities = np.arange(-20, 20, 0.1)

        if items is None:
            if raters is None:
                y = [sum(self.variance_items(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                         for item in self.dataframe.columns)
                     for ability in abilities]

            else:
                y = [sum(self.variance_items(ability, item, difficulties, rater, severities, thresholds)
                         for item in self.dataframe.columns for rater in raters)
                     for ability in abilities]

        else:
            if raters is None:
                y = [sum(self.variance_items(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                         for item in items)
                     for ability in abilities]

            else:
                y = [sum(self.variance_items(ability, item, difficulties, rater, severities, thresholds)
                         for item in items for rater in raters)
                     for ability in abilities]

        y = np.array(y)
        y = 1 / (y ** 0.5)
        y = y.reshape(len(abilities), 1)

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Conditional SEM'

        plot = self.plot_data_items(x_data=abilities, y_data=y, anchor=anchor, items=items, raters=raters,
                                    x_min=xmin, x_max=xmax, y_max=ymax, point_csem_lines=point_csem_lines,
                                    score_labels=point_csem_labels, graph_title=graphtitle, y_label=ylabel,
                                    plot_style=plot_style, palette=palette, black=black, font=font,
                                    title_font_size=title_font_size, axis_font_size=axis_font_size, labelsize=labelsize,
                                    filename=filename, plot_density=dpi, file_format=file_format)

        return plot

    def test_csem_thresholds(self,
                             anchor=False,
                             items=None,
                             raters=None,
                             point_csem_lines=None,
                             point_csem_labels=False,
                             xmin=-5,
                             xmax=5,
                             ymax=5,
                             title=None,
                             plot_style='white',
                             palette='dark blue',
                             black=False,
                             font='Times New Roman',
                             title_font_size=15,
                             axis_font_size=12,
                             labelsize=12,
                             filename=None,
                             file_format='png',
                             dpi=300):

        '''
        Plots Test Conditional Standard Error of Measurement Curve for RSM.
        '''

        if anchor:
            if not hasattr(self, 'anchor_thresholds_thresholds'):
                print('Anchor calibration required')
                print('Run self.calibrate_thresholds_anchor()')
                return

        if anchor:
            difficulties = self.anchor_diffs_thresholds
            thresholds = self.anchor_thresholds_thresholds
            severities = self.anchor_severities_thresholds

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_thresholds

        if isinstance(items, str):
            if (items == 'all') | (items == 'none'):
                items = None

            else:
                items = [items]

        if isinstance(raters, str):
            if raters == 'all':
                raters = self.raters.tolist()

            elif raters == 'none':
                raters = None

            elif raters == 'zero':
                raters = None

            else:
                raters = [raters]

        dummy_sevs = {'dummy_rater': np.zeros(self.max_score + 1)}

        abilities = np.arange(-20, 20, 0.1)

        if items is None:
            if raters is None:
                y = [sum(self.variance_thresholds(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                         for item in self.dataframe.columns)
                     for ability in abilities]

            else:
                y = [sum(self.variance_thresholds(ability, item, difficulties, rater, severities, thresholds)
                         for item in self.dataframe.columns for rater in raters)
                     for ability in abilities]

        else:
            if raters is None:
                y = [sum(self.variance_thresholds(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                         for item in items)
                     for ability in abilities]

            else:
                y = [sum(self.variance_thresholds(ability, item, difficulties, rater, severities, thresholds)
                         for item in items for rater in raters)
                     for ability in abilities]

        y = np.array(y)
        y = 1 / (y ** 0.5)
        y = y.reshape(len(abilities), 1)

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Conditional SEM'

        plot = self.plot_data_thresholds(x_data=abilities, y_data=y, anchor=anchor, items=items, raters=raters,
                                         x_min=xmin, x_max=xmax, y_max=ymax, point_csem_lines=point_csem_lines,
                                         score_labels=point_csem_labels, graph_title=graphtitle, y_label=ylabel,
                                         plot_style=plot_style, palette=palette, black=black, font=font,
                                         title_font_size=title_font_size, axis_font_size=axis_font_size,
                                         labelsize=labelsize, filename=filename, plot_density=dpi,
                                         file_format=file_format)

        return plot

    def test_csem_matrix(self,
                         anchor=False,
                         items=None,
                         raters=None,
                         point_csem_lines=None,
                         point_csem_labels=False,
                         xmin=-5,
                         xmax=5,
                         ymax=5,
                         title=None,
                         plot_style='white',
                         palette='dark blue',
                         black=False,
                         font='Times New Roman',
                         title_font_size=15,
                         axis_font_size=12,
                         labelsize=12,
                         filename=None,
                         file_format='png',
                         dpi=300):

        '''
        Plots Test Conditional Standard Error of Measurement Curve for RSM.
        '''

        if anchor:
            if not hasattr(self, 'anchor_thresholds_matrix'):
                print('Anchor calibration required')
                print('Run self.calibrate_matrix_anchor()')
                return

        if anchor:
            difficulties = self.anchor_diffs_matrix
            thresholds = self.anchor_thresholds_matrix
            severities = self.anchor_severities_matrix

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_matrix

        if isinstance(items, str):
            if (items == 'all') | (items == 'none'):
                items = None

            else:
                items = [items]

        if isinstance(raters, str):
            if raters == 'all':
                raters = self.raters.tolist()

            elif raters == 'none':
                raters = None

            elif raters == 'zero':
                raters = None

            else:
                raters = [raters]

        dummy_sevs = {'dummy_rater': {item: np.zeros(self.max_score + 1)
                                      for item in self.dataframe.columns}}
        
        abilities = np.arange(-20, 20, 0.1)

        if items is None:
            if raters is None:
                y = [sum(self.variance_matrix(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                         for item in self.dataframe.columns)
                     for ability in abilities]

            else:
                y = [sum(self.variance_matrix(ability, item, difficulties, rater, severities, thresholds)
                         for item in self.dataframe.columns for rater in raters)
                     for ability in abilities]

        else:
            if raters is None:
                y = [sum(self.variance_matrix(ability, item, difficulties, 'dummy_rater', dummy_sevs, thresholds)
                         for item in items)
                     for ability in abilities]

            else:
                y = [sum(self.variance_matrix(ability, item, difficulties, rater, severities, thresholds)
                         for item in items for rater in raters)
                     for ability in abilities]

        y = np.array(y)
        y = 1 / (y ** 0.5)
        y = y.reshape(len(abilities), 1)

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Conditional SEM'

        plot = self.plot_data_matrix(x_data=abilities, y_data=y, anchor=anchor, items=items, raters=raters,
                                     x_min=xmin, x_max=xmax, y_max=ymax, point_csem_lines=point_csem_lines,
                                     score_labels=point_csem_labels, graph_title=graphtitle, y_label=ylabel,
                                     plot_style=plot_style, palette=palette, black=black, font=font,
                                     title_font_size=title_font_size, axis_font_size=axis_font_size,
                                     labelsize=labelsize, filename=filename, plot_density=dpi, file_format=file_format)

        return plot

    def std_residuals_plot_global(self,
                                  items=None,
                                  raters=None,
                                  bin_width=0.5,
                                  x_min=-6,
                                  x_max=6,
                                  normal=False,
                                  title=None,
                                  plot_style='white',
                                  font='Times New Roman',
                                  title_font_size=15,
                                  axis_font_size=12,
                                  labelsize=12,
                                  filename=None,
                                  file_format='png',
                                  plot_density=300):

        '''
        Plots histogram of standardised residuals for SLM, with optional overplotting of Standard Normal Distribution.
        '''

        if isinstance(raters, str):
            if raters == 'all':
                raters = self.raters

            elif raters == 'none':
                raters = None

        if isinstance(items, str):
            if items == 'all':
                items = None

            elif items == 'none':
                items = None

        if items is None:
            if raters is None:
                std_residual_list = self.std_residual_df_global.unstack().unstack()

            else:
                std_residual_list = self.std_residual_df_global.loc[raters].unstack().unstack()

        else:
            if raters is None:
                std_residual_list = self.std_residual_df_global[items].unstack().unstack()

            else:
                if isinstance(raters, str):
                    if isinstance(items, str):
                        std_residual_list = self.std_residual_df_global[items].loc[raters]

                    else:
                        std_residual_list = self.std_residual_df_global[items].loc[raters].unstack()

                else:
                    std_residual_list = self.std_residual_df_global[items].loc[raters].unstack().unstack()

        plot = self.std_residuals_hist(std_residual_list, bin_width=bin_width, x_min=x_min, x_max=x_max, normal=normal,
                                       title=title, plot_style=plot_style, font=font, title_font_size=title_font_size,
                                       axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                                       file_format=file_format, plot_density=plot_density)

        return plot

    def std_residuals_plot_items(self,
                                 items=None,
                                 raters=None,
                                 bin_width=0.5,
                                 x_min=-6,
                                 x_max=6,
                                 normal=False,
                                 title=None,
                                 plot_style='white',
                                 font='Times New Roman',
                                 title_font_size=15,
                                 axis_font_size=12,
                                 labelsize=12,
                                 filename=None,
                                 file_format='png',
                                 plot_density=300):

        '''
        Plots histogram of standardised residuals for SLM, with optional overplotting of Standard Normal Distribution.
        '''

        if isinstance(raters, str):
            if raters == 'all':
                raters = self.raters

            elif raters == 'none':
                raters = None

        if isinstance(items, str):
            if items == 'all':
                items = None

            elif items == 'none':
                items = None

        if items is None:
            if raters is None:
                std_residual_list = self.std_residual_df_items.unstack().unstack()

            else:
                std_residual_list = self.std_residual_df_items.loc[raters].unstack().unstack()

        else:
            if raters is None:
                std_residual_list = self.std_residual_df_items[items].unstack().unstack()

            else:
                if isinstance(raters, str):
                    if isinstance(items, str):
                        std_residual_list = self.std_residual_df_items[items].loc[raters]

                    else:
                        std_residual_list = self.std_residual_df_items[items].loc[raters].unstack()

                else:
                    std_residual_list = self.std_residual_df_items[items].loc[raters].unstack().unstack()
                    
        plot = self.std_residuals_hist(std_residual_list, bin_width=bin_width, x_min=x_min, x_max=x_max, normal=normal,
                                       title=title, plot_style=plot_style, font=font, title_font_size=title_font_size,
                                       axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                                       file_format=file_format, plot_density=plot_density)

        return plot

    def std_residuals_plot_thresholds(self,
                                      items=None,
                                      raters=None,
                                      bin_width=0.5,
                                      x_min=-6,
                                      x_max=6,
                                      normal=False,
                                      title=None,
                                      plot_style='white',
                                      font='Times New Roman',
                                      title_font_size=15,
                                      axis_font_size=12,
                                      labelsize=12,
                                      filename=None,
                                      file_format='png',
                                      plot_density=300):

        if isinstance(raters, str):
            if raters == 'all':
                raters = self.raters

            elif raters == 'none':
                raters = None

        if isinstance(items, str):
            if items == 'all':
                items = None

            elif items == 'none':
                items = None

        if items is None:
            if raters is None:
                std_residual_list = self.std_residual_df_thresholds.unstack().unstack()

            else:
                std_residual_list = self.std_residual_df_thresholds.loc[raters].unstack().unstack()

        else:
            if raters is None:
                std_residual_list = self.std_residual_df_thresholds[items].unstack().unstack()

            else:
                if isinstance(raters, str):
                    if isinstance(items, str):
                        std_residual_list = self.std_residual_df_thresholds[items].loc[raters]

                    else:
                        std_residual_list = self.std_residual_df_thresholds[items].loc[raters].unstack()

                else:
                    std_residual_list = self.std_residual_df_thresholds[items].loc[raters].unstack().unstack()

        plot = self.std_residuals_hist(std_residual_list, bin_width=bin_width, x_min=x_min, x_max=x_max, normal=normal,
                                       title=title, plot_style=plot_style, font=font, title_font_size=title_font_size,
                                       axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                                       file_format=file_format, plot_density=plot_density)

        return plot

    def std_residuals_plot_matrix(self,
                                  items=None,
                                  raters=None,
                                  bin_width=0.5,
                                  x_min=-6,
                                  x_max=6,
                                  normal=False,
                                  title=None,
                                  plot_style='white',
                                  font='Times New Roman',
                                  title_font_size=15,
                                  axis_font_size=12,
                                  labelsize=12,
                                  filename=None,
                                  file_format='png',
                                  plot_density=300):

        if isinstance(raters, str):
            if raters == 'all':
                raters = self.raters

            elif raters == 'none':
                raters = None

        if isinstance(items, str):
            if items == 'all':
                items = None

            elif items == 'none':
                items = None

        if items is None:
            if raters is None:
                std_residual_list = self.std_residual_df_matrix.unstack().unstack()

            else:
                std_residual_list = self.std_residual_df_matrix.loc[raters].unstack().unstack()

        else:
            if raters is None:
                std_residual_list = self.std_residual_df_matrix[items].unstack().unstack()

            else:
                if isinstance(raters, str):
                    if isinstance(items, str):
                        std_residual_list = self.std_residual_df_matrix[items].loc[raters]

                    else:
                        std_residual_list = self.std_residual_df_matrix[items].loc[raters].unstack()

                else:
                    std_residual_list = self.std_residual_df_matrix[items].loc[raters].unstack().unstack()

        plot = self.std_residuals_hist(std_residual_list, bin_width=bin_width, x_min=x_min, x_max=x_max, normal=normal,
                                       title=title, plot_style=plot_style, font=font, title_font_size=title_font_size,
                                       axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                                       file_format=file_format, plot_density=plot_density)

        return plot