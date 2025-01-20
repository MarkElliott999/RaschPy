#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Fri Jan 31 13:50:17 2020

@author: Mark Elliott

Consolidated Rasch analysis with simulation functionality
'''

import pandas as pd
import numpy as np

import itertools

import math
from math import exp, log, sqrt, ceil
import statistics

from sklearn.decomposition import PCA
from scipy.stats import gmean, hmean, norm, truncnorm

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def loadup_SLM(filename):
    
    '''
    Cleans the data and produces variables for later use
    '''

    responses = pd.DataFrame(pd.read_csv(filename,
                                         header = None))

    scores = [0, 1]
    
    responses.columns = [f'Item_{item + 1}'
                         for item in range(responses.shape[0])]
    
    np.where(responses in scores, responses, np.nan)
    
    return responses

class Rasch:

    def __init__(self,
                 dataframe,
                 max_score=None,
                 no_of_classes=5):

        self.dataframe = dataframe
        self.no_of_items = dataframe.shape[1]
        self.no_of_persons = dataframe.shape[0]
        if max_score is None:
            self.max_score = int(np.nanmax(dataframe))
        else:
            self.max_score = max_score
        self.no_of_classes = no_of_classes

    '''
    
    ***PAIR AND CPAT ALGORITHMS***
    
    Steps in PAIR and CPAT estimaiton procedure for the RSM Rasch model
    (Adrich 1978).
    
    '''

    def item_diffs(self,
                   constant=0):
        
        '''
    
        Produces central item difficuty estimates (or difficulties for SLM)
    
        '''

        df_array = np.array(self.dataframe)

        matrix = [[np.count_nonzero((df_array[:, item_1]) ==
                                    (df_array[:, item_2] + 1)) + constant
                   for item_2 in range(self.no_of_items)]
                  for item_1 in range(self.no_of_items)]
        
        matrix = np.array(matrix)

        mat = np.linalg.matrix_power(matrix, 25)
        #mat_pow = 3

        '''
        while 0 in mat:

            mat = np.matmul(mat, matrix)
            mat_pow += 1

            if mat_pow == 5:

                matrix = matrix.astype(np.float128) + 0.1
                mat = matrix ** 3

                break
        '''

        recip_matrix = np.divide(mat.T, mat)
        
        '''

        normaliser = np.linalg.norm(recip_matrix, axis=0)

        normalised_matrix = recip_matrix.T / normaliser[:, None]
        
        '''

        weights = sum(recip_matrix)

        diffs_cos = np.log(weights)
        diffs_cos -= np.mean(diffs_cos)

        self.diffs_cos = diffs_cos

    def item_std_errors(self,
                        no_of_samples=100):
        
        '''
        
        Bootstraped standard error estimates for item difficulties.
        
        '''

        samples = [Rasch(self.dataframe.sample(frac=1, replace=True))
                   for sample in range(no_of_samples)]

        calibrations = [samples[sample].item_diffs()
                        for sample in range(no_of_samples)]

        item_ests = np.concatenate([calibrations[sample].reshape((self.no_of_items, 1)).T
                                    for sample in range(no_of_samples)])

        self.item_standard_errors = np.std(item_ests, axis=0)

        self.item_2_5_pc = np.percentile(item_ests, 2.5, axis=0)
        self.item_97_5_pc = np.percentile(item_ests, 97.5, axis=0)

    def score_vector(self):
        
        '''
        
        Function to prodecude vector of total raw scores for persons.
        
        '''

        person_scores = self.dataframe.T.sum()

        self.scores = np.array(person_scores)

    def missing_vector(self):
        
        '''
        
        Creates a vector of numerical indicators of patterns of missing data
        for the puprposes of producing item ft statistics (missing responses
        are not included in ability estimates for this purpose, unlike when
        calculating test results).
        
        Missing item indicators for an calculated according to item position by 2^i-1
        for the ith item; missing pattern indicators are calculated as the sum
        of the missing item indicators for the missing pattern (this is a one-to-one
        mapping).
        
        '''

        missing = []

        for candidate in range(self.no_of_persons):
            missing_indicator = 0

            for item in range(self.no_of_items):

                if math.isnan(self.dataframe.iloc[candidate, item]):
                    missing_indicator += (2 ** item)

            missing.append(missing_indicator)

        missing = np.array(missing)
        missing_patterns = np.unique(missing)

        self.missing = missing
        self.missing_patterns = missing_patterns

    def missing_subsets(self):
        
        '''
        
        Creates subset dataframes with the same missing patterns
        from the response dataframe.
        
        '''
        
        new_df = self.dataframe.copy()
        self.score_vector()

        new_df['Score'] = self.scores
        new_df['Missing_pattern'] = self.missing

        max_poss = [self.max_score * self.no_of_items
                    for row in range(self.no_of_persons)]
        max_poss = np.array(max_poss)
        max_poss -= (self.max_score *
                     np.array(self.dataframe.isnull().sum(axis=1).tolist()))
        new_df['Max_score'] = max_poss

        new_df = new_df[new_df['Score'] > 0]
        new_df = new_df[new_df['Score'] < new_df['Max_score']]
        new_df = new_df.drop(['Max_score'], axis=1)

        subsets = [new_df[new_df['Missing_pattern'] == pattern]
                   for pattern in self.missing_patterns]

        self.subsets = subsets

    def subset_abils(self):
        
        '''
        
        Calclate ability estimates for all candidates in subsets,
        ignoring missing data.
        
        Create new responses dataframe with appended score,
        missing pattern code and ability estimate for each candidate.
        
        '''

        subset_estimates = []

        for i, sub in enumerate(self.subsets):

            subset = sub.copy()

            if subset.shape[0] == 0:

                pass

            else:

                delete = (subset.iloc[0, :self.no_of_items] !=
                          subset.iloc[0, :self.no_of_items])

                subset_diffs = self.diffs.copy()
                subset_diffs = subset_diffs[delete == False]

                score_list = subset['Score']
                score_list = np.array(score_list)
                score_list = np.unique(score_list)

                subset_ests = self.ability_lookups(subset_diffs)

                subset_ests = pd.DataFrame(subset_ests)
                subset_ests = pd.concat([pd.DataFrame(score_list),
                                         subset_ests['Estimate']],
                                        axis=1)

                subset_ests.columns = ['Score', 'Ability_estimate']

                subset_estimates.append(subset_ests)

                subset['Ability_estimate'] = subset['Score'].map(subset_ests.set_index('Score')['Ability_estimate'])

                self.subsets[i] = subset

        new_responses = pd.concat([subset for subset in self.subsets],
                                  axis=0)

        new_responses.sort_index(inplace=True)

        self.new_responses = new_responses

    def residuals(self,
                  new_responses,
                  difficulties):
        
        '''
        
        Calculate residuals and standardised residuals for analysis of fit.
        
        '''

        expected, info = self.exp_info_matrix(new_responses, difficulties)
        scores = new_responses.copy().drop(['Score',
                                            'Missing_pattern',
                                            'Ability_estimate'],
                                           axis=1)

        residuals = scores.reset_index(drop = True).sub(expected.reset_index(drop = True))
        std_residuals = residuals.div(np.sqrt(info))
        std_residuals_list = np.array(std_residuals).reshape(residuals.shape[0]
                                                             * residuals.shape[1])
        std_residuals_list = std_residuals_list[~np.isnan(std_residuals_list)]

        return residuals, std_residuals, std_residuals_list

    def outfit_ms(self,
                  std_residuals_df):
        
        '''
        
        Calculates vector of item outfit statistics.
        
        '''

        squared_residuals = np.power(std_residuals_df, 2)
        outfit_ms = np.mean(squared_residuals)

        outfit_ms_vector = pd.DataFrame(outfit_ms)
        outfit_ms_vector.columns = ['Outfit mean square']
        outfit_ms_vector.index = [f'Item {i + 1}'
                                  for i in range(self.no_of_items)]

        return outfit_ms_vector

    def infit_ms(self,
                 std_residuals_df,
                 info_df):
        
        '''
        
        Calculates vector of item infit statistics
        
        '''

        sq_residuals = np.power(std_residuals_df, 2)
        weighted_sq_residuals = np.multiply(sq_residuals, info_df)
        weighted_sq_residuals_sum = np.sum(weighted_sq_residuals)
        info_sum = np.sum(info_df)

        infit_ms = weighted_sq_residuals_sum / info_sum

        infit_ms_vector = pd.DataFrame(infit_ms)
        infit_ms_vector.columns = ['Infit mean square']
        infit_ms_vector.index = [f'Item {i + 1}'
                                 for i in range(self.no_of_items)]

        return infit_ms_vector

    def kurtosis_matrix(self,
                        df,
                        item_diffs,
                        thresholds=[0, 0]):
        
        '''
        
        Calculates kurtosis matrix for use in converting infit/outfit
        mean square values into z-scores.
        
        '''

        new_abils = np.unique(df['Ability_estimate'])

        kurtosis_lkps = [[self.kurtosis(ability,
                                        item_diffs[item],
                                        thresholds)
                          for item in range(self.no_of_items)]
                         for ability in new_abils]

        kurtosis_lkps = pd.DataFrame(kurtosis_lkps)

        kurtosis_lkps.columns = [f'Item_{item + 1}'
                                 for item in range(self.no_of_items)]
        kurtosis_lkps.insert(0,
                             'Ability_estimate',
                             new_abils)

        kurtosis = df['Ability_estimate']
        kurtosis = pd.DataFrame(kurtosis)

        for item in range(self.no_of_items):

            kurtosis.insert(item + 1,
                            f'Item_{item + 1}',
                            kurtosis['Ability_estimate'].map(kurtosis_lkps.set_index
                                                             ('Ability_estimate')
                                                             [f'Item_{item + 1}']))

        kurtosis_matrix = kurtosis.drop(['Ability_estimate'], axis=1)

        return kurtosis_matrix

    def q_values(self,
                 new_responses,
                 kurtosis_matrix,
                 info_matrix):
        
        '''
        
        Calculates matrix of q-values to convert infit/outfit ms values into z-scores.
        
        '''

        q_matrix = kurtosis_matrix - np.power(info_matrix, 2)

        q_vector = [np.sum(q_matrix[new_responses[f'Item_{item + 1}'] ==
                                    new_responses[f'Item_{item + 1}']].iloc[:, item])
                    for item in range(self.no_of_items)]
        q_vector = np.array(q_vector)

        var_vector = [np.sum(info_matrix[new_responses[f'Item_{item + 1}'] ==
                                         new_responses[f'Item_{item + 1}']].iloc[:, item])
                      for item in range(self.no_of_items)]

        var_vector = np.array(var_vector)

        q_vector /= np.power(var_vector, 2)

        q_vector = np.sqrt(q_vector)

        return q_vector

    def z_vector(self,
                 ms_vector,
                 q_vector):
        
        '''
        
        Calculates z score  vectorfrom infit/outfit mean square vector.
        
        '''

        q_vector = q_vector.reshape(len(q_vector), 1)

        z = np.array(ms_vector)
        z = (np.power(z, 1 / 3) - 1) * 3
        z /= q_vector
        z += q_vector / 3

        return z

    def residuals_analysis(self,
                           std_residuals_df):
        
        '''
        
        Calculates matrix of pairwise correlations of standardised residuals
        for test of local independence and for further PCA for test of
        unidimensionality.
        
        First specify mini-function to create pairs of columns of standardised
        residuals with missing data rows removed.
        
        PCA of residual correlations to identify violations of 
        unidimensionality  and returns the eigenvectors, eigenvalues,
        variance explained and item loadings.
        
        '''

        residual_correlations = std_residuals_df.corr()

        pca = PCA()
        pca.fit(residual_correlations)

        eigenvectors = pd.DataFrame(pca.components_)

        eigenvectors.columns = ['Eigenvector {}'.format(item + 1)
                                for item in range(self.no_of_items)]

        eigenvalues = pca.explained_variance_
        eigenvalues = pd.DataFrame(eigenvalues)

        eigenvalues.insert(loc=0,
                           column='',
                           value=[f'Eigenvalue for PC {item + 1}'
                                  for item in range(self.no_of_items)])

        explained_variance = pd.DataFrame(pca.explained_variance_ratio_)

        explained_variance.insert(loc=0,
                                  column='',
                                  value=[f'Variance explained by PC {item + 1}'
                                         for item in range(self.no_of_items)])

        loadings = eigenvectors.T * np.sqrt(pca.explained_variance_)
        loadings = pd.DataFrame(loadings)

        loadings.columns = ['PC {}'.format(n + 1)
                            for n in range(self.no_of_items)]
        loadings.index = [f'Item {item + 1}'
                          for item in range(self.no_of_items)]

        return (residual_correlations,
                eigenvectors,
                eigenvalues,
                explained_variance,
                loadings)

class RSM(Rasch):

    '''
    Rating Scale Model (Andrich 1978) formulation of the polytomous Rasch model.
    '''

    def cat_prob(self,
                 ability,
                 difficulty,
                 category,
                 thresholds):
        
        '''
        Calculates category probability for given person ability,
        item difficulty and set of Rasch-Andrich thresholds.
        '''

        cat_prob_nums = [exp(cat * (ability - difficulty) -
                         sum(thresholds[:cat + 1]))
                         for cat in range(self.max_score + 1)]

        return cat_prob_nums[category] / sum(cat_prob_nums)
    
    def exp_score(self,
                  ability,
                  difficulty,
                  thresholds):
        
        '''
        Calculates expected score for given person ability,
        item difficulty and set of Rasch-Andrich thresholds.
        '''

        cat_prob_nums = [exp(cat * (ability - difficulty) -
                         sum(thresholds[:cat + 1]))
                         for cat in range(self.max_score + 1)]

        exp_score = (sum(category * prob
                         for category, prob in enumerate(cat_prob_nums)) /
                     sum(cat_prob_nums))

        return exp_score

    def variance(self,
                 ability,
                 difficulty,
                 thresholds):
        
        '''
        Calculates the item (Fisher) information from an item given the person
        ability, item difficulty and set of Rasch-Andrich thresholds. This is
        also the variance of the expected score function.
        '''

        cat_prob_nums = [exp(cat * (ability - difficulty) -
                         sum(thresholds[:cat + 1]))
                         for cat in range(self.max_score + 1)]

        expected = self.exp_score(ability,
                                  difficulty,
                                  thresholds)

        variance = (sum(((category - expected) ** 2) * cat_prob
                        for category, cat_prob in enumerate(cat_prob_nums)) /
                    sum(cat_prob_nums))

        return variance
    
    def kurtosis(self,
                 ability,
                 difficulty,
                 thresholds):
        
        '''
        Kurtosis function which calculates the kurtosis for an item
        given the person ability, item difficulty and set of
        Rasch-Andrich thresholds.
        '''

        cat_prob_nums = [exp(cat * (ability - difficulty) -
                         sum(thresholds[:cat + 1]))
                         for cat in range(self.max_score + 1)]

        expected = self.exp_score(ability,
                                  difficulty,
                                  thresholds)

        kurtosis = (sum(((category - expected) ** 4) * cat_prob
                        for category, cat_prob in enumerate(cat_prob_nums)) /
                    sum(cat_prob_nums))

        return kurtosis

    def _threshold_distance_no_diffs_unweighted(self,
                                                threshold,
                                                difficulties,
                                                constant=0):
        
        '''
        ** Private method **
        Estimates the distance between adjacent Rasch-Andrich thresholds.
        Unweighted CPAT method without using item difficulties.
        '''
        
        df_array = np.array(self.dataframe)
        
        estimator = 0
        count = 0
        
        for item_1 in range(self.no_of_items):
            
            for item_2 in range(self.no_of_items):
                
                num = np.count_nonzero((df_array[:, item_1] == threshold) &
                                       (df_array[:, item_2] == threshold)) + constant
                
                den_1 = np.count_nonzero((df_array[:, item_1] == threshold - 1) &
                                         (df_array[:, item_2] == threshold + 1)) + constant
                
                den_2 = np.count_nonzero((df_array[:, item_1] == threshold + 1) &
                                         (df_array[:, item_2] == threshold - 1)) + constant
    
                if 0 in [num, den_1, den_2]:    
                    pass
                
                else:   
                    count += 1
                    
                    estimator += (2 * log(num) - log(den_1) - log(den_2)) / 2
    
        try:
            estimator /= count
            
        except:
            estimator = np.nan
        
        return estimator

    def _threshold_distance_no_diffs_weighted(self,
                                              threshold,
                                              difficulties,
                                              constant=0):
        
        '''
        ** Private method **
        Estimates the distance between adjacent Rasch-Andrich thresholds.
        Weighted CPAT method without using item difficulties.
        '''
        
        df_array = np.array(self.dataframe)
        
        estimator = 0
        weight_sum = 0
        
        for item_1 in range(self.no_of_items):
            
            for item_2 in range(self.no_of_items):
                
                num = np.count_nonzero((df_array[:, item_1] == threshold) &
                                       (df_array[:, item_2] == threshold)) + constant
                
                den_1 = np.count_nonzero((df_array[:, item_1] == threshold - 1) &
                                         (df_array[:, item_2] == threshold + 1)) + constant
                
                den_2 = np.count_nonzero((df_array[:, item_1] == threshold + 1) &
                                         (df_array[:, item_2] == threshold - 1)) + constant
    
                if 0 in [num, den_1, den_2]:
                    pass
                
                else:   
                    weight = hmean([num, num, den_1, den_2])
                    
                    estimator += weight * (2 * log(num) -
                                           log(den_1) -
                                           log(den_2)) / 2
                    
                    weight_sum += weight
    
        try:
            estimator /= weight_sum
            
        except:
            estimator = np.nan
        
        return estimator

    def _threshold_distance_unweighted(self,
                                       threshold,
                                       difficulties,
                                       constant=0):
        
        '''
        ** Private method **
        Estimates the distance between adjacent Rasch-Andrich thresholds.
        Unweighted CPAT method.
        '''
        
        df_array = np.array(self.dataframe)
        
        estimator = 0
        count = 0
        
        for item_1 in range(self.no_of_items):
            
            for item_2 in range(self.no_of_items):
                
                num = np.count_nonzero((df_array[:, item_1] == threshold) &
                                       (df_array[:, item_2] == threshold)) + constant
                
                den = np.count_nonzero((df_array[:, item_1] == threshold - 1) &
                                       (df_array[:, item_2] == threshold + 1)) + constant

                if 0 in [num, den]:  
                    pass
                
                else:
                    count += 1         
                    estimator += (log(num) -
                                  log(den) +
                                  difficulties[item_1] -
                                  difficulties[item_2])
            
        try:    
            estimator /= count
            
        except:   
            estimator = np.nan
        
        return estimator
    
    def _threshold_distance(self,
                            threshold,
                            difficulties,
                            constant=0):
        
        '''
        ** Private method **
        Estimates the distance between adjacent Rasch-Andrich thresholds.
        Weighted CPAT method (this is the best of the options here).
        '''
        
        df_array = np.array(self.dataframe)
        
        estimator = 0
        weights_sum = 0
        
        for item_1 in range(self.no_of_items):
            
            for item_2 in range(self.no_of_items):
                
                num = np.count_nonzero((df_array[:, item_1] == threshold) &
                                       (df_array[:, item_2] == threshold)) + constant
                
                den = np.count_nonzero((df_array[:, item_1] == threshold - 1) &
                                       (df_array[:, item_2] == threshold + 1)) + constant
    
                if 0 in [num, den]:
                    pass
                
                else:
                    weight = hmean([num, den])
                    
                    estimator += weight * (log(num) -
                                           log(den) +
                                           difficulties[item_1] -
                                           difficulties[item_2])             
                    weights_sum += weight
    
        try:
            estimator /= weights_sum
            
        except:
            estimator = np.nan
        
        return estimator
    
    def threshold_set_no_diffs_unweighted(self,
                                     difficulties,
                                     constant=0):
        
        '''
        Calculates set of Rasch-Andrich threshold estimates.
        Uneighted CPAT method without using item difficulties.
        '''

        thresh_distances = [self._threshold_distance_no_diffs_unweighted(threshold+1,
                                                                    difficulties,
                                                                    constant)
                            for threshold in range(self.max_score - 1)]

        thresholds = [sum(thresh_distances[:threshold])
                      for threshold in range(self.max_score)]

        thresholds = np.array(thresholds)

        np.add(thresholds,
               -np.mean(thresholds),
               out = thresholds,
               casting = 'unsafe')

        thresholds = np.insert(thresholds, 0, 0)

        return thresholds

    def threshold_set_no_diffs_weighted(self,
                                   difficulties,
                                   constant=0):
        
        '''
        Calculates set of Rasch-Andrich threshold estimates.
        Weighted CPAT method without using item difficulties.
        '''

        thresh_distances = [self._threshold_distance_no_diffs_weighted(threshold + 1,
                                                                  difficulties,
                                                                  constant)
                            for threshold in range(self.max_score - 1)]

        thresholds = [sum(thresh_distances[:threshold])
                      for threshold in range(self.max_score)]

        thresholds = np.array(thresholds)

        np.add(thresholds,
               -np.mean(thresholds),
               out = thresholds,
               casting = 'unsafe')

        thresholds = np.insert(thresholds, 0, 0)

        return thresholds
    
    def threshold_set_unweighted(self,
                                 difficulties,
                                 constant=0):
        
        '''
        Calculates set of Rasch-Andrich threshold estimates.
        Unweighted CPAT method.
        '''

        thresh_distances = [self._threshold_distance_unweighted(threshold+1,
                                                               difficulties,
                                                               constant)
                            for threshold in range(self.max_score - 1)]

        thresholds = [sum(thresh_distances[:threshold])
                      for threshold in range(self.max_score)]

        thresholds = np.array(thresholds)

        np.add(thresholds,
               -np.mean(thresholds),
               out = thresholds,
               casting = 'unsafe')

        thresholds = np.insert(thresholds, 0, 0)

        return thresholds
    
    def threshold_set(self,
                      difficulties,
                      constant=0):
        
        '''
        Calculates set of Rasch-Andrich threshold estimates.
        Weighted CPAT method.
        '''

        thresh_distances = [self._threshold_distance(threshold + 1,
                                                     difficulties,
                                                     constant)
                            for threshold in range(self.max_score - 1)]

        thresholds = [sum(thresh_distances[:threshold])
                      for threshold in range(self.max_score)]

        thresholds = np.array(thresholds)

        np.add(thresholds,
               -np.mean(thresholds),
               out = thresholds,
               casting = 'unsafe')

        thresholds = np.insert(thresholds, 0, 0)

        return thresholds
    
    def _subset_pairs(self,
                      subset,
                      item,
                      constant=0):
        
        '''
        ** Private method **
        Mini-method for use in EVM threshold calculation
        '''
        
        item_scores = subset[:, item]
        
        counts = [np.count_nonzero(item_scores == score) + constant
                   for score in range(self.max_score + 1)]
        
        subset_pairs = [[counts[i + 1] * counts[j]
                         for j in range(self.max_score)]
                        for i in range(self.max_score)]
        subset_pairs = np.array(subset_pairs)
    
        return subset_pairs
    
    def thresholds_evm(self,
                       constant=0):
        
        '''
        Calculates set of Rasch-Andrich threshold estimates.
        Eigenvector method (Garner and Engelhard 2002).
        '''

        data = np.array(self.dataframe)
    
        missing_indicators = 1 - (data == data).astype(int)
        
        for item in range(self.no_of_items):
            
            missing_indicators[:, item] *= 2**item
        
        missing_vector = np.nansum(missing_indicators, axis = 1)
        
        score_vector = np.nansum(data, axis = 1)
    
        matrix = np.zeros((self.max_score, self.max_score))
    
        max_vector = self.max_score * np.count_nonzero(~np.isnan(data), axis = 1)
    
        responses_2 = data[score_vector > 0]
        max_vector = max_vector[score_vector > 0]
        missing_vector = missing_vector[score_vector > 0]
        score_vector = score_vector[score_vector > 0]
        responses_2 = responses_2[score_vector < max_vector]
        missing_vector = missing_vector[score_vector < max_vector]
        score_vector = score_vector[score_vector < max_vector]

        missing_patterns = np.unique(missing_vector)
        no_of_patterns = len(missing_patterns)
    
        subsets_complete = [responses_2[missing_vector == 
                                        missing_patterns[pattern]]
                            for pattern in range(no_of_patterns)]
    
        subset_score_vectors_complete = [score_vector[missing_vector == 
                                                      missing_patterns[pattern]]
                                         for pattern in range(no_of_patterns)]

        subsets = [subsets_complete[pattern]
                   for pattern in range(no_of_patterns)
                   if subsets_complete[pattern].shape[0] > 1]

        subset_score_vectors = [subset_score_vectors_complete[pattern]
                                for pattern in range(no_of_patterns)
                                if subsets_complete[pattern].shape[0] > 1]

        subsets = [[subset[subset_score_vectors[i] == score]
                    for score in np.unique(subset_score_vectors[i].astype(int))]
                   for i, subset in enumerate(subsets)]

        subsets = [subset
                   for subset_pattern in subsets
                   for subset in subset_pattern]

        subsets = [subset for subset in subsets if subset.shape[0] > 1]

        for subset in range(len(subsets)):
    
            matrix += sum(self._subset_pairs(subsets[subset], item, constant)
                          for item in range(subsets[subset].shape[1]))

        for score in range(self.max_score):
            matrix[score, score] = 0

        matrix = np.array(matrix).astype(np.float64)

        mat = np.linalg.matrix_power(matrix, 3)
        mat_pow = 3
        
        while 0 in mat:
            
            mat = np.matmul(mat, matrix)
            mat_pow += 1
            
            if mat_pow == 5:
                
                break
        
        recip_matrix = np.divide(mat.T, mat)
        
        pca = PCA()
        
        try:
            pca.fit(recip_matrix)
            eigenvectors = np.array(pca.components_)
            
            thresholds = -np.log(abs(eigenvectors[0]))
            thresholds -= np.mean(thresholds)
            thresholds = thresholds.real
        
            thresholds = np.insert(thresholds, 0, 0)
            
        except:
            thresholds = np.array([np.nan for i in range(self.max_score + 1)])
        
        return thresholds
    
    def _modified_evm_estimator(self,
                                array,
                                item_1,
                                item_2,
                                threshold_1,
                                threshold_2,
                                constant=0):
        
        '''
        ** Private method **
        Mini-method for use in modified EVM threshold calculation
        to create an estimator for a given item pair + threshold pair combination.
        '''

        num_1 = np.count_nonzero((array[:, item_1] == threshold_1 + 1) &
                                 (array[:, item_2] == threshold_2)) + constant

        den_1 = np.count_nonzero((array[:, item_1] == threshold_1) &
                                 (array[:, item_2] == threshold_2 + 1)) + constant

        num_2 = np.count_nonzero((array[:, item_1] == threshold_2) &
                                 (array[:, item_2] == threshold_1 + 1)) + constant
        
        den_2 = np.count_nonzero((array[:, item_1] == threshold_2 + 1) &
                                 (array[:, item_2] == threshold_1)) + constant

        
        try:
            estimator = - (log(num_1) - log(den_1) + log(num_2) - log(den_2)) / 2
        
        except:
            estimator = 0
        
        return estimator

    def _modified_evm_threshold_pair(self,
                                     array,
                                     threshold_1,
                                     threshold_2,
                                     constant=0):
        
        '''
        ** Private method **
        Mini-method for use in modified EVM threshold calculation
        to aggreagate _modified_evm_estimator() values accross item pairs
        to create an estimator for a given threshold pair.
        '''
        
        threshold_pair = sum(sum(self._modified_evm_estimator(array,
                                                              item_1,
                                                              item_2,
                                                              threshold_1,
                                                              threshold_2,
                                                              constant)
                                 for item_2 in range(item_1 + 1,
                                                     self.no_of_items))
                              for item_1 in range(self.no_of_items - 1))
    
        return threshold_pair
    
    def thresholds_modified_evm(self,
                                constant=0):

        '''
        Calculates set of modified EVM thresholds.
        '''
        
        array = np.array(self.dataframe)
        
        count = self.no_of_items * (self.no_of_items - 1) / 2
        
        t_matrix = [[self._modified_evm_threshold_pair(array,
                                                       threshold_1,
                                                       threshold_2,
                                                       constant) / count
                     for threshold_2 in range(self.max_score)]
                    for threshold_1 in range(self.max_score)]
        
        t_matrix = np.array(t_matrix)
            
        thresholds = [np.mean(t_matrix[row, :])
                      for row in range(self.max_score)]
        
        thresholds = np.array(thresholds)
        thresholds -= np.mean(thresholds)
        thresholds = np.insert(thresholds, 0, 0)
    
        return thresholds
    
    def item_diffs(self,
                   constant=0):
        
        '''
        Creates set of weighted CPAT threshold estimates plus
        PAIR item difficulty estimation (cosine similarity).
        '''
        
        df_array = np.array(self.dataframe)
    
        matrix = [[np.count_nonzero((df_array[:, item_1]) ==
                                    (df_array[:, item_2] + 1)) + constant
                   for item_2 in range(self.no_of_items)]
                  for item_1 in range(self.no_of_items)]
        
        matrix = np.array(matrix).astype(np.float64)

        mat = np.linalg.matrix_power(matrix, 3)
        mat_pow = 3
        
        while 0 in mat:
            
            mat = np.matmul(mat, matrix)
            mat_pow += 1
            
            if mat_pow == 5:
                
                matrix = matrix.astype(np.float128) + constant
                mat = matrix ** 3
                
                break
        
        recip_matrix = np.divide(mat.T, mat)
        
        normaliser = np.linalg.norm(recip_matrix, axis = 0)
        
        normalised_matrix = recip_matrix.T / normaliser[:, None]

        weights = sum(normalised_matrix)
        
        diffs_cos = np.log(weights)
        diffs_cos -= np.mean(diffs_cos)
    
        thresh_distances = [self._threshold_distance(threshold + 1,
                                                     diffs_cos,
                                                     constant)
                            for threshold in range(self.max_score - 1)]
    
        thresholds = [sum(thresh_distances[:threshold])
                      for threshold in range(self.max_score)]
    
        thresholds = np.array(thresholds)
        
        np.add(thresholds,
               -np.mean(thresholds),
               out = thresholds,
               casting = 'unsafe')
        
        thresholds = np.insert(thresholds, 0, 0)
        
        self.diffs = diffs_cos
        self.thresholds = thresholds
    
    def item_diffs_ls(self,
                      constant=0):
        
        '''
        Creates set of weighted CPAT threshold estimates plus
        PAIR item difficulty estimation (least squares).
        '''
        
        diffs_ls = self.items_ls(constant)
    
        thresh_distances = [self._threshold_distance(threshold + 1,
                                                     diffs_ls,
                                                     constant)
                            for threshold in range(self.max_score - 1)]
    
        thresholds = [sum(thresh_distances[:threshold])
                      for threshold in range(self.max_score)]
    
        thresholds = np.array(thresholds)
        
        np.add(thresholds,
               -np.mean(thresholds),
               out = thresholds,
               casting = 'unsafe')
        
        thresholds = np.insert(thresholds, 0, 0)
        
        return diffs_ls, thresholds
    
    def items_cos(self,
                  constant=0):
        
        '''
        PAIR item ddifficulty estimates, cosine similarity (Kou and Lin 2014).
        '''
        
        df_array = np.array(self.dataframe)
    
        matrix = [[np.count_nonzero((df_array[:, item_1]) ==
                                    (df_array[:, item_2] + 1)) + constant
                   for item_2 in range(self.no_of_items)]
                  for item_1 in range(self.no_of_items)]
        
        matrix = np.array(matrix).astype(np.float64)

        mat = np.linalg.matrix_power(matrix, 3)
        mat_pow = 3
        
        while 0 in mat:
            
            mat = np.matmul(mat, matrix)
            mat_pow += 1
            
            if mat_pow == 5:
                
                break
        
        recip_matrix = np.divide(mat.T, mat)
        
        normaliser = np.linalg.norm(recip_matrix, axis = 0)
        
        normalised_matrix = recip_matrix.T / normaliser[:, None]

        weights = sum(normalised_matrix)
        
        diffs_cos = np.log(weights)
        diffs_cos -= np.mean(diffs_cos)
    
        return diffs_cos

    def items_ls(self,
                 constant=0):
        
        '''
        PAIR item ddifficulty estimates, least squares (Choppin 1985).
        '''
        
        df_array = np.array(self.dataframe)
    
        matrix = [[np.count_nonzero((df_array[:, item_1]) ==
                                    (df_array[:, item_2] + 1)) + constant
                   for item_2 in range(self.no_of_items)]
                  for item_1 in range(self.no_of_items)]
        
        matrix = np.array(matrix).astype(np.float64)

        mat = np.linalg.matrix_power(matrix, 3)
        mat_pow = 3
        
        while 0 in mat:
            
            mat = np.matmul(mat, matrix)
            mat_pow += 1
            
            if mat_pow == 5:
                
                break
        
        recip_matrix = np.divide(mat.T, mat)

        log_matrix = np.log(recip_matrix)
        
        diffs_ls = np.mean(log_matrix, axis=1)
        diffs_ls -= np.mean(diffs_ls)
    
        return diffs_ls

    def items_gmean(self,
                    constant=0):

        '''
        PAIR item ddifficulty estimates, least squares (Choppin 1985).
        '''

        df_array = np.array(self.dataframe)

        matrix = [[gmean([np.count_nonzero((df_array[:, item_1] == threshold + 1) &
                                           (df_array[:, item_2] == threshold)) + constant
                          for threshold in range(self.max_score)])
                   for item_2 in range(self.no_of_items)]
                  for item_1 in range(self.no_of_items)]

        matrix = np.array(matrix).astype(np.float64)

        mat = np.linalg.matrix_power(matrix, 3)
        mat_pow = 3

        while 0 in mat:

            mat = np.matmul(mat, matrix)
            mat_pow += 1

            if mat_pow == 5:
                break

        recip_matrix = np.divide(mat.T, mat)

        log_matrix = np.log(recip_matrix)

        diffs_ls = np.mean(log_matrix, axis=1)
        diffs_ls -= np.mean(diffs_ls)

        return diffs_ls

    def items_evm(self,
                  constant=0):
        
        '''
        PAIR item ddifficulty estimates, Eigenvector method.
        '''
        
        df_array = np.array(self.dataframe)
    
        matrix = [[np.count_nonzero((df_array[:, item_1]) ==
                                    (df_array[:, item_2] + 1)) + constant
                   for item_2 in range(self.no_of_items)]
                  for item_1 in range(self.no_of_items)]
        
        matrix = np.array(matrix).astype(np.float64)

        mat = np.linalg.matrix_power(matrix, 3)
        mat_pow = 3
        
        while 0 in mat:
            
            mat = np.matmul(mat, matrix)
            mat_pow += 1
            
            if mat_pow == 5:
                
                break
        
        recip_matrix = np.divide(mat.T, mat)
        
        pca = PCA()
        
        try:
            pca.fit(recip_matrix)
            eigenvectors = np.array(pca.components_)
            
            diffs_evm = -np.log(abs(eigenvectors[0]))
            diffs_evm -= np.mean(diffs_evm)
            diffs_evm = diffs_evm.real
            
        except:
            diffs_evm = np.array([np.nan for i in range(self.no_of_items)])
    
        return diffs_evm

    def item_std_errors(self,
                        no_of_samples=100):
        
        '''
        Bootstraped standard error estimates for item and threshold estimates.
        '''

        samples = [RSM(self.dataframe.sample(frac=1, replace=True))
                   for sample in range(no_of_samples)]

        for sample in range(no_of_samples):
            samples[sample].item_diffs()

        item_ests = np.concatenate([samples[sample].diffs.
                                    reshape((self.no_of_items, 1)).T
                                    for sample in range(no_of_samples)])

        self.item_standard_errors = np.std(item_ests, axis=0)

        self.item_2_5_pc = np.percentile(item_ests, 2.5, axis=0)
        self.item_97_5_pc = np.percentile(item_ests, 97.5, axis=0)
        
        thresh_ests = np.concatenate([samples[sample].thresholds.
                                      reshape((self.max_score + 1, 1)).T
                                    for sample in range(no_of_samples)])
        
        self.threshold_standard_errors = np.std(thresh_ests, axis=0)

        self.threshold_2_5_pc = np.percentile(thresh_ests, 2.5, axis=0)
        self.threshold_97_5_pc = np.percentile(thresh_ests, 97.5, axis=0)
    
    def ability_lookups(self,
                        diffs=None,
                        scores=None,
                        warm_corr=True,
                        tolerance=0.0000001,
                        ext_score_adjustment=0.5):
        
        '''
        Creates a raw score to ability estimate look-up table for a set
        of items using ML estimation (Newton-Raphson procedure) with
        optional Warm (1989) bias correction.
        '''
        
        if diffs is None:
            diffs = self.diffs

        items = len(diffs)
        ext_score = self.max_score * self.no_of_items
        
        if scores is None:
            scores = np.arange(ext_score + 1)

        estimates = []
        iterations = []

        for score in scores:

            if score == 0:
                used_score = ext_score_adjustment

            elif score == ext_score:
                used_score = score - ext_score_adjustment

            else:
                used_score = score

            estimate = (log(used_score) - log(ext_score - used_score)
                        + np.mean(diffs[:]))
            change = 1
            rounds = 0

            while abs(change) > tolerance:

                result = sum(self.exp_score(estimate,
                                            diffs[item],
                                            self.thresholds)
                             for item in range(items))

                info = sum(self.variance(estimate,
                                         diffs[item],
                                         self.thresholds)
                           for item in range(items))

                change = max(-1, min(1, (result - used_score) / info))
                estimate -= change
                rounds += 1

            if warm_corr:
                estimate += self.warm(estimate, diffs)

            estimates.append(estimate)
            iterations.append(rounds)
            
        abils = pd.DataFrame()
        abils['Score'] = [score for score in scores]
        abils['Estimate'] = estimates
        abils['Iterations'] = iterations

        return abils

    def warm(self,
             ability,
             diffs):
        
        '''
        Warm's (1989) bias correction for ML abiity estimates
        '''
        
        no_of_items = len(diffs)

        exp_scores = [self.exp_score(ability,
                                     difficulty,
                                     self.thresholds)
                      for difficulty in diffs]

        variances = [self.variance(ability,
                                   difficulty,
                                   self.thresholds)
                     for difficulty in diffs]

        part_1 = sum(sum((cat ** 3) * self.cat_prob(ability,
                                                    difficulty,
                                                    cat,
                                                    self.thresholds)
                         for cat in range(self.max_score + 1))
                     for difficulty in diffs)

        part_2 = 3 * sum((variances[item] + (exp_scores[item] ** 2)) *
                         exp_scores[item]
                         for item in range(no_of_items))

        part_3 = sum(2 * (exp_scores[item] ** 3)
                     for item in range(no_of_items))

        warm_correction = ((part_1 - part_2 + part_3) /
                           (2 * (sum(variances[item]
                                     for item in range(no_of_items)) **
                                 2)))

        return warm_correction

    def basic_abils(self,
                    warm_corr=True,
                    tolerance=0.0000001,
                    ext_score_adjustment=0.5):

        '''
        Creates abils vector for all possible raw scores for all items.
        '''

        self.item_diffs()
        
        abils = self.ability_lookups(self.diffs,
                                     warm_corr=warm_corr,
                                     tolerance=tolerance,
                                     ext_score_adjustment=ext_score_adjustment)
        
        self.abils = abils['Estimate']

    def residuals(self):
        
        '''
        Creates multiple matrices for use in fit analysis.
        '''
        
        max_score_vector = self.dataframe.count(axis = 1) * self.max_score
        
        missing = [sum(np.power(2, item)
                       for item in range(self.no_of_items)
                       if math.isnan(self.dataframe.iloc[row, item]))
                   for row in range(self.dataframe.shape[0])]
        missing = np.array(missing)
        
        missing_patterns = np.unique(missing)

        self.missing = missing
        self.missing_patterns = missing_patterns
        
        '''
        Creates subset dataframes with the same missing patterns
        from the response dataframe.
        '''
        
        new_df = self.dataframe.copy()
        self.score_vector()

        new_df['Score'] = self.scores
        new_df['Missing_pattern'] = self.missing
        new_df['Max_score'] = max_score_vector


        new_df = new_df[new_df['Score'] > 0]
        new_df = new_df[new_df['Score'] < new_df['Max_score']]
        new_df = new_df.drop(['Max_score'], axis=1)

        subsets = [new_df[new_df['Missing_pattern'] == pattern]
                   for pattern in self.missing_patterns]

        '''
        Calculates ability estimates for all candidates in subsets,
        ignoring missing data.
        
        Create new responses dataframe with appended score,
        missing pattern code and ability estimate for each candidate.
        '''
    
        subset_estimates = []
        
        for i, sub in enumerate(subsets):
            
            subset = sub.copy()
            
            if subset.shape[0] == 0:
                
                pass
            
            else:
        
                delete = (subset.iloc[0, :self.no_of_items] !=
                          subset.iloc[0, :self.no_of_items])
                delete = np.array(delete)
        
                subset_diffs = self.diffs.copy()
                subset_diffs = subset_diffs[delete[:] == False]
        
                score_list = subset['Score'].copy()
                score_list = np.array(score_list)
                score_list = np.unique(score_list)
                
                subset_ests = pd.DataFrame()
                
                subset_ests['Score'] = score_list.copy()
        
                ests = self.ability_lookups(subset_diffs,
                                            score_list,
                                            warm_corr=True)
        
                subset_ests['Ability_estimate'] = ests['Estimate']
                
                subset_estimates.append(subset_ests)
        
                subset['Ability_estimate'] = subset['Score'].map(subset_ests.
                                                                 set_index('Score')['Ability_estimate'])
        
                subsets[i] = subset
        
        new_responses = pd.concat([subset for subset in subsets],
                                  axis = 0)
            
        self.new_responses = new_responses
        self.new_responses.sort_index(inplace = True)
        self.subsets = subsets

        '''
        Calculates expected score matrix for residual analysis.
        '''

        new_abils = np.unique(self.new_responses['Ability_estimate'])

        exp_score_lkps = [[self.exp_score(ability,
                                          difficulty,
                                          self.thresholds)
                           for difficulty in self.diffs]
                          for ability in new_abils]

        exp_score_lkps = pd.DataFrame(exp_score_lkps)
        exp_score_lkps.columns = [f'Item_{item + 1}'
                                  for item in range(self.no_of_items)]
        exp_score_lkps.insert(0,
                              'Ability_estimate',
                              new_abils)

        exp_scores = pd.DataFrame()

        for item in range(self.no_of_items):

            exp_scores[f'Item_{item + 1}'] = (self.new_responses['Ability_estimate'].
                                              map(exp_score_lkps.set_index('Ability_estimate')
                                                  [f'Item_{item + 1}']))
        
        self.exp_score_matrix = exp_scores
        
        '''
        Calculates information matrix for residual analysis.
        '''

        info_lkps = [[self.variance(ability,
                                    difficulty,
                                    self.thresholds)
                           for difficulty in self.diffs]
                          for ability in new_abils]

        info_lkps = pd.DataFrame(info_lkps)
        info_lkps.columns = [f'Item_{item + 1}'
                             for item in range(self.no_of_items)]
        info_lkps.insert(0,
                         'Ability_estimate',
                         new_abils)

        information = pd.DataFrame()

        for item in range(self.no_of_items):

            information[f'Item_{item + 1}'] = (self.new_responses['Ability_estimate'].
                                               map(info_lkps.set_index('Ability_estimate')
                                                   [f'Item_{item + 1}']))
        
        self.info_matrix = information

        '''
        Calculates kurtosis matrix for residual analysis.
        '''

        kurtosis_lkps = [[self.kurtosis(ability,
                                        difficulty,
                                        self.thresholds)
                          for difficulty in self.diffs]
                         for ability in new_abils]

        kurtosis_lkps = pd.DataFrame(kurtosis_lkps)
        kurtosis_lkps.columns = [f'Item_{item + 1}'
                                 for item in range(self.no_of_items)]
        kurtosis_lkps.insert(0,
                             'Ability_estimate',
                             new_abils)

        kurtosis = pd.DataFrame()

        for item in range(self.no_of_items):

            kurtosis[f'Item_{item + 1}'] = (self.new_responses['Ability_estimate'].
                                            map(kurtosis_lkps.set_index('Ability_estimate')
                                                [f'Item_{item + 1}']))
        
        self.kurtosis_matrix = kurtosis

        '''
        Calculate residuals (difference between observed and expected
        scores)and standardised residuals (residuals divided by SD).
        '''
        
        scores = self.new_responses.copy().iloc[:, :self.no_of_items]
        scores.columns = [f'Item_{item + 1}'
                          for item in range(self.no_of_items)]
        
        self.residual_matrix = scores - self.exp_score_matrix
        
        self.std_residual_matrix = (self.residual_matrix /
                                    np.sqrt(self.info_matrix))
        
        row_count = self.std_residual_matrix.shape[0]
        column_count = self.std_residual_matrix.shape[1]
        
        std_res_list = np.array(self.std_residual_matrix).reshape(row_count *
                                                                  column_count)
        std_res_list = std_res_list[~np.isnan(std_res_list)]
        self.std_residual_list = std_res_list
        
        self.nan_mask = np.isfinite(self.residual_matrix)
        masked_variance = self.info_matrix[self.nan_mask]
        
        self.outfit_ms_vector = ((self.std_residual_matrix ** 2).sum(axis = 0) /
                                 self.std_residual_matrix.count())
        
        self.infit_ms_vector = ((self.residual_matrix ** 2).sum(axis = 0) /
                                masked_variance.sum(axis = 0))
        
        self.rsem_vector = (np.sqrt((self.residual_matrix ** 2).sum(axis = 1)) /
                            masked_variance.sum(axis = 1))

        '''
        Calculates matrix of pairwise correlations of standardised residuals
        for test of local independence and for further PCA for test of
        unidimensionality.
        
        First specify mini-function to create pairs of columns of standardised
        residuals with missing data rows removed.
        
        PCA of residual correlations to identify violations of unidimensionality 
        and returns the eigenvectors, eigenvalues, variance explained 
        and item loadings.
        '''
        
        residual_correlations = self.std_residual_matrix.corr()
        
        no_of_items = residual_correlations.shape[0]
    
        pca = PCA()
        pca.fit(residual_correlations)
        
        eigenvectors = pd.DataFrame(pca.components_)
        
        eigenvectors.columns = ['Eigenvector {}'.format(item + 1)
                                for item in range (no_of_items)]
        
        eigenvalues = pca.explained_variance_
        eigenvalues = pd.DataFrame(eigenvalues)
        
        eigenvalues.insert(loc = 0,
                           column = '',
                           value = [f'Eigenvalue for PC {item + 1}'
                                    for item in range(no_of_items)])
        
        explained_variance = pd.DataFrame(pca.explained_variance_ratio_)
        
        explained_variance.insert(loc = 0,
                                  column = '',
                                  value = [f'Variance explained by PC {item + 1}'
                                           for item in range(no_of_items)])
        
        loadings = eigenvectors.T * np.sqrt(pca.explained_variance_)
        loadings = pd.DataFrame(loadings)
        
        loadings.columns = ['PC {}'.format(n + 1)
                            for n in range(no_of_items)]
        loadings.index = [f'Item {item + 1}'
                          for item in range(no_of_items)]
        
        self.residual_correlations = residual_correlations
        self.eigenvectors = eigenvectors
        self.eigenvalues = eigenvalues
        self.explained_variance = explained_variance
        self.loadings = loadings
        
    def class_intervals(self):
        
        '''
        Calculates class interval membership by quantiles defined by specified
        number of class intervals.
        
        Calculates mean abilities, observed mean scores and proportions
        in each score category for each class interval.
        '''

        self.quantiles = (self.results['Ability estimate'].
                          quantile([(i + 1) / self.no_of_classes
                                    for i in range(self.no_of_classes - 1)]))
        
        mask_dict = {f'mask_{class_no + 1}': []
                     for class_no in range(self.no_of_classes)}
        
        mask_dict['mask_1'] = (self.new_responses['Ability_estimate'] <
                               self.quantiles.values[0])
        
        mask_dict[f'mask_{self.no_of_classes}'] = (self.new_responses['Ability_estimate'] >=
                                                  self.quantiles.values[self.no_of_classes - 2])
        
        for class_no in range(self.no_of_classes - 2):
            
            mask_dict[f'mask_{class_no + 2}'] = ((self.new_responses['Ability_estimate'] >=
                                                  self.quantiles.values[class_no]) &
                                                 (self.new_responses['Ability_estimate'] <
                                                  self.quantiles.values[class_no + 1]))
        
        self.class_sizes = [sum(mask_dict[f'mask_{class_no + 1}'])
                                for class_no in range(self.no_of_classes)]
        
        self.responses_classes = {f'class_{class_no + 1}':
                                  self.new_responses[mask_dict[f'mask_{class_no + 1}']]
                                  for class_no in range(self.no_of_classes)}
        
        self.class_mean_abilities = [self.responses_classes[f'class_{class_no + 1}']['Ability_estimate'].mean()
                                     for class_no in range(self.no_of_classes)]
            
        self.class_observed_means = {f'Item_{item + 1}':
                                     np.array([self.responses_classes[f'class_{class_no + 1}'].iloc[:, item].mean()
                                               for class_no in range(self.no_of_classes)])
                                     for item in range(self.no_of_items)}
            
        self.class_observed_categories = {f'Item_{item + 1}':
                                          np.array([[sum(self.responses_classes[f'class_{class_no + 1}'][item] == category) /
                                                     self.responses_classes[f'class_{class_no + 1}'][item].count()
                                                     for category in range(self.max_score + 1)]
                                                    for class_no in range(self.no_of_classes)])
                                          for item in range(self.no_of_items)}
            
        category_counts = {f'Item_{item + 1}':
                           np.array([[sum(self.responses_classes[f'class_{class_no + 1}'][item] == category)
                                      for category in range(self.max_score + 1)]
                                     for class_no in range(self.no_of_classes)])
                           for item in range(self.no_of_items)}

        self.category_proportions = {f'Item_{item + 1}':
                                     (category_counts[f'Item_{item + 1}'] /
                                      self.responses_classes[f'class_{class_no + 1}'][item].count())
                                     for item in range(self.no_of_items)}
            
        cond_cats = {f'Item_{item + 1}':
                     [[category_counts[f'Item_{item + 1}'][class_no, category + 1] /
                       (category_counts[f'Item_{item + 1}'][class_no, category] +
                        category_counts[f'Item_{item + 1}'][class_no, category + 1])
                      if (category_counts[f'Item_{item + 1}'][class_no, category] +
                          category_counts[f'Item_{item + 1}'][class_no, category + 1] > 0)
                      else np.nan
                      for class_no in range(self.no_of_classes)]
                     for category in range(self.max_score)]
                     for item in range(self.no_of_items)}
                     
        self.threshold_conditional_categories = {key: np.array(value).T
                                                 for key, value in cond_cats.items()}

    def csem(self,
             ability,
             diffs=None):
        
        '''
        Calculates conditional standard error of measurement for an ability.
        '''
        
        if diffs is None:
            diffs = self.diffs
        
        total_info = np.sum(self.variance(ability,
                                          diff,
                                          self.thresholds)
                            for diff in diffs)
        
        cond_sem = 1 / np.sqrt(total_info)
        
        return cond_sem

    def item_stats_df(self,
                      constant,
                      no_of_samples=100,
                      percentiles=False,
                      rmsr=False,
                      discrim=False):

        '''
        Creates dataframes of item and threshold stats
        '''
        
        self.item_diffs(constant)
        self.item_std_errors(no_of_samples)
        self.residuals()
        
        self.item_estimates = pd.DataFrame()
        
        self.item_estimates['Difficulty'] = self.diffs
        self.item_estimates['SE'] = self.item_standard_errors
        self.item_estimates['Mean score'] = self.dataframe.mean(axis = 0).values
        self.item_estimates['Facility'] = (self.dataframe.mean(axis = 0).values /
                                             self.max_score)
        self.item_estimates['Infit MS'] = self.infit_ms_vector.values
        self.item_estimates['Outifit MS'] = self.outfit_ms_vector.values

        if rmsr:
            sq_residuals = self.residual_matrix ** 2
            self.item_estimates['RMSR'] = np.sqrt(sq_residuals.mean(axis = 0).
                                                  values)
        
        '''
        if discrim:
            differences =  pd.concat([self.new_responses['Ability_estimate'] -
                                      difficulty
                                      for difficulty in self.diffs_cos],
                                     axis = 1)
            differences.columns = [f'Item_{item+1}'
                                   for item in range(self.no_of_items)]
            num = (differences * self.residual_matrix).sum(axis=0)
            den = (self.info_matrix * (differences ** 2)).sum(axis=0)
            self.discrim_vector = 1 + num / den
            self.item_stats['Discrimination'] = self.discrim_vector.values
            '''
        
        self.threshold_estimates = pd.DataFrame()
        
        self.threshold_estimates['Rasch-Andrich threshold'] = self.thresholds[1:]
        self.threshold_estimates['SE'] = self.threshold_standard_errors[1:]
        
        if percentiles:
            
            self.item_estimates['2.5%'] = self.item_2_5_pc
            self.threshold_estimates['2.5%'] = self.threshold_2_5_pc[1:]
            
            self.item_estimates['97.5%'] = self.item_97_5_pc
            self.threshold_estimates['97.5%'] = self.threshold_97_5_pc[1:]
            
        self.item_estimates.index = [f'Item {item}'
                                     for item in range(self.no_of_items)]
        self.threshold_estimates.index = [f'Threshold {threshold + 1}'
                                          for threshold in range(self.max_score)]

    def results_df(self,
                   csem=True,
                   rsem=False):
        
        '''
        Produces a results dataframe with raw score, ability estimate,
        CSEM and RSEM for each person.
        '''
        
        self.basic_abils()
        
        raw_scores = np.sum(self.dataframe.T).astype(int)
        
        abil_vector = [self.abils[int(raw_scores[person])]
                       for person in range(self.no_of_persons)]
        
        results_df = pd.DataFrame()
        
        results_df['Raw score'] = raw_scores
        results_df['Ability estimate'] = abil_vector
        
        if csem:
            csem_vector = [self.csem(abil_vector[person])
                           for person in range(self.no_of_persons)]
            results_df['CSEM'] = csem_vector
        
        if rsem:
            results_df['RSEM'] = self.rsem_vector
            
        results_df.index = [f'Person {person+1}'
                            for person in range(self.no_of_persons)]
        
        self.rsem_beta = np.sqrt(np.mean(self.rsem_vector ** 2))
        
        sd_beta = np.std(abil_vector)
        self.rsep_beta = np.sqrt(sd_beta ** 2 - self.rsem_beta ** 2) / self.rsem_beta
        self.rel_beta = self.rsep_beta ** 2 / (1 + self.rsep_beta ** 2)
        
        self.psi = (1 - (np.sum(self.rsem_vector ** 2) /
                         (self.no_of_persons - 1)) / sd_beta ** 2)
    
        self.results = results_df