# -*- coding: utf-8 -*-
#!/usr/bin/env python3
'''
RaschPy
Created on Fri Jan 31 13:50:17 2020
@author: Mark Elliott
Consolidated Rasch analysis with simulation functionality
'''

import itertools
from math import exp, log, sqrt, floor
import statistics
import string

import numpy as np
import pandas as pd
from scipy.stats import hmean, truncnorm, norm
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import xlsxwriter

def loadup_slm(filename,
               item_names=True,
               person_names=True,
               long=False):

    '''
    Cleans the data and creates responses file in the correct format
    '''

    if item_names:
        header = 0
    else:
        header = None

    if person_names:
        index_col = 0
    else:
        index_col = None

    if filename.split('.')[1] == 'xlsx':
        responses = pd.read_excel(filename, sheet_name=0, header=header, index_col=index_col)

    else:
        responses = pd.read_csv(filename, header=header, index_col=index_col)

    if long:
        responses = pd.pivot_table(responses, values='Score', index='Person',
                                   columns=['Item'], aggfunc=min, dropna=False)

    no_of_items = responses.shape[1]
    no_of_persons = responses.shape[0]

    if item_names == False:
        responses.columns = [f'Item_{item + 1}' for item in range(no_of_items)]

    if person_names == False:
        responses.index = [f'Person_{person + 1}' for person in range(no_of_persons)]

    for column in responses.columns:
        responses[column] = pd.to_numeric(responses[column], errors='coerce').astype('Int64')

    responses = responses.where(responses.isin([0, 1]), np.nan)
    responses = responses.astype(float)

    invalid_responses = responses[responses.isnull().all(axis=1)].copy()
    for col in invalid_responses.columns:
        invalid_responses[col] = np.where(invalid_responses[col].isna(), np.nan, invalid_responses[col])
        
    responses = responses[~responses.isnull().all(axis=1)].copy()
    for col in responses.columns:
        responses[col] = np.where(responses[col].isna(), np.nan, responses[col])

    return responses, invalid_responses


def loadup_pcm(filename,
               max_score_vector=None,
               item_names=True,
               person_names=True,
               long=False):
    '''
    Cleans the data and creates responses file in the correct format
    '''

    if item_names:
        header = 0
    else:
        header = None

    if person_names:
        index_col = 0
    else:
        index_col = None

    if filename.split('.')[1] == 'xlsx':
        responses = pd.read_excel(filename, sheet_name=0, header=header, index_col=index_col)

    else:
        responses = pd.read_csv(filename, header=header, index_col=index_col)

    if long:
        responses = pd.pivot_table(responses, values='Score', index='Person',
                                   columns=['Item'], aggfunc=min, dropna=False)

    no_of_items = responses.shape[1]
    no_of_persons = responses.shape[0]

    if max_score_vector is None:
        max_score_vector = responses.max()
    scores = [np.arange(max_score_vector[item] + 1) for item in range(no_of_items)]

    if item_names == False:
        responses.columns = [f'Item_{item + 1}' for item in range(no_of_items)]

    if person_names == False:
        responses.index = [f'Person{person + 1}' for person in range(no_of_persons)]

    for i, col in enumerate(responses.columns):
        responses[col] = pd.to_numeric(responses[col], errors='coerce').astype('Int64')
        responses.loc[:, col] = responses.loc[:, col].where(responses.loc[:, col].isin(scores[i]), np.nan)

    responses = responses.astype(float)

    invalid_responses = responses[responses.isnull().all(axis=1)].copy()
    for col in invalid_responses.columns:
        invalid_responses[col] = np.where(invalid_responses[col].isna(), np.nan, invalid_responses[col])

    responses = responses[~responses.isnull().all(axis=1)].copy()
    for col in responses.columns:
        responses[col] = np.where(responses[col].isna(), np.nan, responses[col])

    return responses, invalid_responses

def loadup_rsm(filename,
               max_score=None,
               item_names=True,
               person_names=True,
               long=False):
    '''
    Cleans the data and creates responses file in the correct format
    '''

    if item_names:
        header = 0
    else:
        header = None

    if person_names:
        index_col = 0
    else:
        index_col = None

    if filename.split('.')[1] == 'xlsx':
        responses = pd.read_excel(filename, sheet_name=0, header=header, index_col=index_col)

    else:
        responses = pd.read_csv(filename, header=header, index_col=index_col)

    if long:
        responses = pd.pivot_table(responses, values='Score', index='Person',
                                   columns=['Item'], aggfunc=min, dropna=False)

    no_of_items = responses.shape[1]
    no_of_persons = responses.shape[0]

    if max_score is None:
        max_score = responses.max().max()
    scores = np.arange(max_score + 1)

    if item_names == False:
        responses.columns = [f'Item_{item + 1}' for item in range(no_of_items)]

    if person_names == False:
        responses.index = [f'Person{person + 1}' for person in range(no_of_persons)]

    for col in responses.columns:
        responses[col] = pd.to_numeric(responses[col], errors='coerce').astype('Int64')
        responses.loc[:, col] = responses.loc[:, col].where(responses.loc[:, col].isin(scores), np.nan)

    responses = responses.astype(float)

    invalid_responses = responses[responses.isnull().all(axis=1)].copy()
    for col in invalid_responses.columns:
        invalid_responses[col] = np.where(invalid_responses[col].isna(), np.nan, invalid_responses[col])

    responses = responses[~responses.isnull().all(axis=1)].copy()
    for col in responses.columns:
        responses[col] = np.where(responses[col].isna(), np.nan, responses[col])

    return responses, invalid_responses

def loadup_mfrm_single(filename,
                       max_score=None,
                       item_names=True,
                       long=False):
    '''
    Cleans the data and creates responses file in the correct format
    '''

    if item_names:
        header = 0
    else:
        header = None

    if filename.split('.')[1] == 'xlsx':
        responses = pd.read_excel(filename, sheet_name=0, header=header, index_col=[0, 1], na_values=np.nan)

    else:
        responses = pd.read_csv(filename, header=header, index_col=[0, 1], na_values=np.nan)

    if long:
        responses = pd.pivot_table(responses, values='Score', index=['Rater', 'Person'],
                                   columns=['Item'], aggfunc=min, dropna=False)
    
    if max_score is None:
        max_score = responses.max().max()
    scores = np.arange(max_score + 1)

    if not item_names:
        responses.columns = [f'Item_{item + 1}' for item in range(no_of_items)]

    rater_names = responses.index.get_level_values(0).unique()
    no_of_items = responses.shape[1]

    persons = responses.index.get_level_values(1).unique()
    no_of_persons = len(persons)

    data = np.empty((no_of_persons, no_of_items))
    data[:] = np.nan
    data = {rater: pd.DataFrame(data.copy()) for rater in rater_names}

    for rater, df in data.items():
        df.columns = responses.columns
        df.index = persons

        for person in responses.xs(rater).index:
            df.loc[person] = responses.xs(rater).loc[person]

    for df in data.values():
        for col in responses.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            df.loc[:, col] = df.loc[:, col].where(df.loc[:, col].isin(scores), np.nan)

    responses = pd.concat(data.values(), keys=data.keys())
    responses = responses.where(responses.isin(scores), np.nan)
    responses = responses.astype(float)

    responses.index.names = ['Rater', 'Person']

    transformed_responses = {item: responses.unstack()[item] for item in responses.columns}
    transformed_responses = pd.concat(transformed_responses.values(), keys=transformed_responses.keys()).T

    valid = transformed_responses[~transformed_responses.isnull().all(axis=1)]
    invalid = transformed_responses[transformed_responses.isnull().all(axis=1)]

    invalid_responses = invalid.stack(dropna=False).swaplevel().sort_index(level=[0, 1])
    for col in invalid_responses.columns:
        invalid_responses[col] = np.where(invalid_responses[col].isna(), np.nan, invalid_responses[col])

    responses = valid.stack(dropna=False).swaplevel().sort_index(level=[0, 1])
    for col in invalid_responses.columns:
        responses[col] = np.where(responses[col].isna(), np.nan, responses[col])

    return responses, invalid_responses

def loadup_mfrm_xlsx_tabs(filename,
                          max_score,
                          item_names=True,
                          missing=None,
                          long=False):
    '''
    Cleans the data and creates responses file in the correct format, from multiple tabs of .xlsx workbook
    '''

    if item_names:
        header = 0
    else:
        header = None

    responses = pd.read_excel(filename, sheet_name=None, header=header, index_col=0)

    if long:
        responses = pd.concat(responses.values(), keys=responses.keys())
        responses.index.names = ['Rater', 'Person']
        responses = pd.pivot_table(responses, values='Score', index=['Rater', 'Person'],
                                   columns=['Item'], aggfunc=min, dropna=False)
        responses = {rater: responses.xs(rater) for rater in responses.index.get_level_values(0)}

    rater_names = list(responses.keys())
    no_of_items = responses[rater_names[0]].shape[1]
    scores = np.arange(max_score + 1)

    if item_names == False:
        item_ids = [f'Item_{item + 1}' for item in range(no_of_items)]
        for df in responses.values():
            df.columns = item_ids
    
    else:
        item_ids = responses[next(iter(responses))].columns

    all_person_names = []
    for df in responses.values():
        all_person_names += list(df.index)
    all_person_names = np.array(all_person_names)

    unique_persons = np.unique(all_person_names)
    no_of_unique_persons = len(unique_persons)

    data = np.empty((no_of_unique_persons, no_of_items))
    data[:] = np.nan
    data = pd.DataFrame(data)
    data.columns = item_ids
    data.index = unique_persons

    data = {rater: pd.DataFrame(data.copy()) for rater in rater_names}

    for rater, df in data.items():
        for person in responses[rater].index:
            df.loc[person] = responses[rater].loc[person]

    for df in data.values():
        for column in df.columns:
            df[column] = np.floor(pd.to_numeric(df[column], errors='coerce')).astype('Int64')

    responses = pd.concat(data.values(), keys=data.keys())
    responses = responses.where(responses.isin(scores), np.nan)
    responses = responses.astype(float)

    responses.index.names = ['Rater', 'Person']

    transformed_responses = {item: responses.unstack()[item] for item in responses.columns}
    transformed_responses = pd.concat(transformed_responses.values(), keys=transformed_responses.keys()).T

    valid = transformed_responses[~transformed_responses.isnull().all(axis=1)]
    invalid = transformed_responses[transformed_responses.isnull().all(axis=1)]

    invalid_responses = invalid.stack(dropna=False).swaplevel().sort_index(level=[0, 1])
    for col in invalid_responses.columns:
        invalid_responses[col] = np.where(invalid_responses[col].isna(), np.nan, invalid_responses[col])

    responses = valid.stack(dropna=False).swaplevel().sort_index(level=[0, 1])
    for col in invalid_responses.columns:
        responses[col] = np.where(responses[col].isna(), np.nan, responses[col])

    return responses, invalid_responses

def loadup_mfrm_multiple(filename_dict,
                         max_score,
                         item_names=True,
                         missing=None,
                         long=False):
    '''
    Cleans the data and creates responses file in the correct format
    '''

    if item_names:
        header = 0
    else:
        header = None

    responses = {}

    for rater, filename in filename_dict.items():
        if filename.split('.')[1] == 'xlsx':
            responses[rater] = pd.read_excel(filename, sheet_name=0, header=header, index_col=0)

        else:
            responses[rater] = pd.read_csv(filename, header=header, index_col=0)

    if long:
        responses = pd.concat(responses.values(), keys=responses.keys())
        responses.index.names = ['Rater', 'Person']
        responses = pd.pivot_table(responses, values='Score', index=['Rater', 'Person'],
                                   columns=['Item'], aggfunc=min, dropna=False)
        responses = {rater: responses.xs(rater) for rater in responses.index.get_level_values(0)}

    rater_names = list(responses.keys())
    no_of_items = responses[rater_names[0]].shape[1]
    scores = np.arange(max_score + 1)

    if item_names == False:
        item_ids = [f'Item_{item + 1}' for item in range(no_of_items)]
        for df in responses.values():
            df.columns = item_ids

    else:
        item_ids = item_names

    all_person_names = []

    for df in responses.values():
        all_person_names += list(df.index)

    for rater, df in responses.items():
        df.index.name = None
        df.columns.name = None

    all_person_names = np.array(all_person_names)
    unique_persons = np.unique(all_person_names)
    no_of_unique_persons = len(unique_persons)

    data = np.empty((no_of_unique_persons, no_of_items))
    data[:] = np.nan
    data = {rater: pd.DataFrame(data.copy()) for rater in rater_names}

    for rater, df in data.items():
        df.columns = responses[next(iter(filename_dict))].columns
        df.index = unique_persons

        for person in responses[rater].index:
            df.loc[person] = responses[rater].loc[person]

    for df in data.values():
        if missing is not None:
            if type(missing) == str:
                missing = [missing]

            for missing_value in missing:
                df = df.replace(missing_value, 99)

            for column in responses.columns:
                df[column] = np.floor(pd.to_numeric(df[column], errors='coerce')).fillna(0).astype('Int64')

            df = df.where(df.isin(np.append(scores, 99)), 0)
            df = df.replace(99, np.nan)

        else:
            for column in df.columns:
                df[column] = np.floor(pd.to_numeric(df[column], errors='coerce')).astype('Int64')

    responses = pd.concat(data.values(), keys=data.keys())
    responses = responses.where(responses.isin(scores), np.nan)
    responses = responses.astype(float)

    responses.index.names = ['Rater', 'Person']

    transformed_responses = {item: responses.unstack()[item] for item in responses.columns}
    transformed_responses = pd.concat(transformed_responses.values(), keys=transformed_responses.keys()).T

    valid = transformed_responses[~transformed_responses.isnull().all(axis=1)]
    invalid = transformed_responses[transformed_responses.isnull().all(axis=1)]

    invalid_responses = invalid.stack(dropna=False).swaplevel().sort_index(level=[0, 1])
    for col in invalid_responses.columns:
        invalid_responses[col] = np.where(invalid_responses[col].isna(), np.nan, invalid_responses[col])

    responses = valid.stack(dropna=False).swaplevel().sort_index(level=[0, 1])
    for col in invalid_responses.columns:
        responses[col] = np.where(responses[col].isna(), np.nan, responses[col])

    return responses, invalid_responses

class Rasch:

    def __init__():

        pass

    def rename_item(self,
                    old,
                    new):

        if old == new:
            print('New item name is the same as old item name.')

        elif new in self.dataframe.columns:
            print('New item name is a duplicate of an existing item name')

        if old not in self.dataframe.columns:
            print(f'Old item name "{old}" not found in data. Please check')

        if isinstance(new, str) == False:
            print('Item names must be strings')

        else:
            self.dataframe.rename(columns={old: new},
                                  inplace=True)

    def rename_items_all(self,
                         new_names):

        list_length = len(new_names)

        if len(new_names) != len(set(new_names)):
            print('List of new item names contains duplicates. Please ensure all items have unique names')

        elif list_length != self.no_of_items:
            print(f'Incorrect number of item names. {list_length} in list, {self.no_of_items} items in data.')

        else:
            self.dataframe.rename(columns={old: new for old, new in zip(self.dataframe.columns, new_names)},
                                  inplace=True)

    def rename_person(self,
                      old,
                      new):

        if old == new:
            print('New person name is the same as old person name.')

        elif new in self.dataframe.index:
            print('New person name is a duplicate of an existing person name')

        if old not in self.dataframe.index:
            print(f'Old person name "{old}" not found in data. Please check')

        if isinstance(new, str) == False:
            print('Person names must be strings')

        else:
            self.dataframe.rename(index={old: new},
                                  inplace=True)

    def rename_persons_all(self,
                           new_names):

        list_length = len(new_names)

        if len(new_names) != len(set(new_names)):
            print('List of new person names contains duplicates. Please ensure all raters have unique names')

        elif list_length != self.no_of_persons:
            print(f'Incorrect number of person names. {list_length} in list, {self.no_of_persons} raters in data.')

        if all(isinstance(name, str) for name in new_names) == False:
            print('Person names must be strings')

        else:
            self.dataframe.rename(index={old: new
                                        for old, new in zip(self.dataframe.index, new_names)},
                                  inplace=True)

    def priority_vector(self,
                        matrix,
                        method='cos',
                        log_lik_tol=0.000001,
                        pcm=False,
                        raters=False):

        if pcm:
            names = []
            for i, item in enumerate(self.dataframe.columns):
                for j in range(self.max_score_vector[i]):
                    names.append(f'{str(item)}_{str(j + 1)}')

        else:
            if raters:
                names = self.raters

            else:
                names = self.dataframe.columns

        recip_matrix = np.divide(matrix.T, matrix)

        if method == 'evm':

            pca = PCA()

            try:
                pca.fit(recip_matrix)
                eigenvectors = np.array(pca.components_)

                measures = -np.log(abs(eigenvectors[0]))
                measures -= np.mean(measures)
                measures = measures.real

                measures = {item: measure for item, measure in zip(names, measures)}
                measures = pd.Series(measures)

            except:
                print('EVM method failed. Try another method.')

        elif method == 'log-lik':

            wins = matrix.sum(axis=1)
            change = 1

            weights = wins / wins.sum()
            weights = np.array(weights)

            while change > log_lik_tol:

                new_weights = []
                for item in range(self.no_of_items):
                    adjustment = 0

                    for item_2 in range(self.no_of_items):
                        if item_2 != item:
                            adjustment += ((matrix[item, item_2] + matrix[item_2, item]) /
                                           (weights[item] + weights[item_2]))

                    new_weights.append(wins[item] / adjustment)

                new_weights = np.array(new_weights)
                new_weights /= new_weights.sum()
                change = max(abs(weights - new_weights))
                weights = new_weights

            measures = -np.log(weights)
            measures -= np.mean(measures)

            measures = {name: measure for name, measure in zip(names, measures)}
            measures = pd.Series(measures)

        else:
            if method == 'ls':
                weights = np.mean(recip_matrix, axis=1)

            else:
                normaliser = np.linalg.norm(recip_matrix, axis=0)
                normalised_matrix = recip_matrix.T / normaliser[:, None]
                weights = sum(normalised_matrix)

            measures = np.log(weights)
            measures -= np.mean(measures)

            measures = {name: measure for name, measure in zip(names, measures)}
            measures = pd.Series(measures)

        return measures

    def std_residuals_hist(self,
                           std_residual_list,
                           bin_width=0.5,
                           x_min=-6,
                           x_max=6,
                           normal=False,
                           title=None,
                           plot_style='colorblind',
                           black=False,
                           font='Times',
                           title_font_size=15,
                           axis_font_size=12,
                           labelsize=12,
                           filename=None,
                           file_format='png',
                           plot_density=300):

        '''
        Plots histogram of standardised residuals for SLM, with optional
        overplotting of Standard Normal Distribution.
        '''

        plt.rcParams["text.latex.preamble"].join([r"\usepackage{dashbox}", r"\setmainfont{xcolor}", ])

        plt.style.use('seaborn-v0_8-' + plot_style)

        if black:
            histogram = plt.hist(std_residual_list, floor((std_residual_list.max() - std_residual_list.min()) / bin_width),
                                 range=(x_min, x_max), density=True, facecolor='gray', alpha=0.5,
                                 edgecolor='black', linewidth=1)

        else:
            histogram = plt.hist(std_residual_list, floor((std_residual_list.max() - std_residual_list.min()) / bin_width),
                                 range=(x_min, x_max), density=True, facecolor='steelblue', alpha=0.5,
                                 edgecolor='black', linewidth=1)

        if normal:
            x_norm = np.linspace(x_min, x_max, 200)
            y_norm = norm.pdf(x_norm, 0, 1)

            if black:
                plt.plot(x_norm, y_norm, '', color='black')

            else:
                plt.plot(x_norm, y_norm, '', color='maroon')

        plt.xlabel('Standardised residual', fontname=font, fontsize=axis_font_size, fontweight='bold')
        plt.ylabel('Density', fontname=font, fontsize=axis_font_size, fontweight='bold')

        if title is not None:
            plt.title(title, fontname=font, fontsize=title_font_size, fontweight='bold')

        plt.tick_params(axis="x", labelsize=labelsize)
        plt.tick_params(axis="y", labelsize=labelsize)

        if filename is not None:
            plt.savefig(filename + f'.{file_format}', dpi=plot_density)

        plt.show()
        plt.close()

class SLM(Rasch):

    def __init__(self,
                 dataframe,
                 extreme_persons=True,
                 no_of_classes=5):

        if extreme_persons:
            self.invalid_responses = dataframe[dataframe.isna().all(axis=1)]
            self.dataframe = dataframe[~dataframe.isna().all(axis=1)]

        else:
            zero_scores = dataframe[dataframe.sum(axis=1) == 0]
            all_correct = dataframe[dataframe.sum(axis=1) ==
                                    dataframe.count(axis=1)]
            self.invalid_responses = pd.concat([zero_scores, all_correct], axis=0)
            self.dataframe = dataframe[~dataframe.index.isin(self.invalid_responses.index)]

        self.no_of_items = self.dataframe.shape[1]
        self.items = self.dataframe.columns
        self.no_of_persons = self.dataframe.shape[0]
        self.persons = self.dataframe.index
        self.no_of_classes = no_of_classes
        self.max_score = 1
    
    def exp_score(self,
                  ability,
                  difficulty):
        
        '''
        Expected score function (also probability of correct response).
        '''
    
        exp_score = 1 / (1 + exp(difficulty - ability))
        
        return exp_score
    
    def cat_prob(self,
                 ability,
                 difficulty,
                 category):

        '''
        Category probability function which calculates the probability
        of scoring 0 or 1 from person ability and item difficulty.
        '''
        
        p = self.exp_score(ability, difficulty)
        
        cat_prob_nums = [1 - p, p] 
        
        return (cat_prob_nums[category] / sum(cat_prob_nums))
    
    def variance(self,
                 ability,
                 difficulty):
        
        '''
        Calculates Fisher information function from  person ability
        and item difficulty. Also the variance and differential
        of the expected score function.
        '''
        
        exponent = exp(difficulty - ability)
        
        expected = 1 / (1 + exponent)
        
        variance = expected * (1 - expected)
    
        return variance
    
    def kurtosis(self,
                 ability,
                 difficulty):
        
        '''
        Calculates kurtosis given person ability and item difficulty.
        '''
        
        expected = self.exp_score(ability, difficulty)
        
        cat_probs = [1 - expected, expected]
        
        kurtosis = sum(((category - expected) ** 4) * cat_prob
                       for category, cat_prob in enumerate(cat_probs))
        
        return kurtosis

    def calibrate(self,
                  constant=0.1,
                  method='cos',
                  matrix_power=3,
                  log_lik_tol=0.000001):

        '''
        Produces central item difficuty estimates (or difficulties for SLM)
        '''

        self.null_persons =  self.dataframe.index[self.dataframe.isnull().all(1)]
        self.dataframe = self.dataframe.drop(self.null_persons)
        self.no_of_persons = self.dataframe.shape[0]

        df_array = np.array(self.dataframe)

        matrix = [[np.count_nonzero((df_array[:, item_1]) ==
                                    (df_array[:, item_2] + 1))
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
                break

        self.diffs = self.priority_vector(mat, method=method, log_lik_tol=log_lik_tol)

    def std_errors(self,
                   interval=None,
                   no_of_samples=100,
                   constant=0.1,
                   method='cos',
                   matrix_power=3,
                   log_lik_tol=0.000001):
        
        '''
        Bootstraped standard error estimates for item difficulties.
        '''

        samples = [SLM(self.dataframe.sample(frac=1, replace=True))
                   for sample in range(no_of_samples)]

        for sample in samples:
            sample.calibrate(constant=constant, method=method, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        item_ests = np.array([sample.diffs.values for sample in samples])

        self.item_se = {item: se for item, se in zip(self.dataframe.columns,
                                                     np.nanstd(item_ests, axis=0))}
        self.item_se = pd.Series(self.item_se)

        if interval is not None:
            self.item_low = {item: low for item, low in zip(self.dataframe.columns,
                                                            np.nanpercentile(item_ests, (1 - interval) * 50, axis=0))}
            self.item_low = pd.Series(self.item_low)

            self.item_high = {item: high for item, high in zip(self.dataframe.columns,
                                                               np.nanpercentile(item_ests, (1 + interval) * 50, axis=0))}
            self.item_high = pd.Series(self.item_high)

        else:
            self.item_low = None
            self.item_high = None

        self.item_bootstrap = pd.DataFrame(item_ests)
        self.item_bootstrap.columns = self.dataframe.columns
        self.item_bootstrap.index = [f'Sample {i + 1}' for i in range (no_of_samples)]

    def abil(self,
             person,
             items=None,
             warm_corr=True,
             tolerance=0.0000001,
             max_iters=100,
             ext_score_adjustment=0.5):

        '''
        Creates raw score to ability estimate look-up. Uses
        Newton-Raphson for ML with optional Warm (1989) bias correction.
        '''

        if items is None:
            items = self.dataframe.columns

        difficulties = self.diffs.loc[items]

        person_data = self.dataframe.loc[person, items]
        person_filter = (person_data + 1) / (person_data + 1)
        score = np.nansum(person_data)

        ext_score = np.nansum(person_filter)

        if score == 0:
            score = ext_score_adjustment

        elif score == ext_score:
            score -= ext_score_adjustment

        try:
            estimate = (log(score) -
                        log(ext_score - score) +
                        difficulties[person_filter == 1].mean())
            change = 1
            iters = 0

            while (abs(change) > tolerance) & (iters <= max_iters):

                person_exp_list = [self.exp_score(estimate, difficulty)
                                   for flag, difficulty in zip(person_filter, difficulties)
                                   if flag == 1]
                result = sum(person_exp_list)

                person_info_list = [self.variance(estimate, difficulty)
                                    for flag, difficulty in zip(person_filter, difficulties)
                                    if flag == 1]
                info = sum(person_info_list)

                change = max(-1, min(1, (result - score) / info))
                estimate -= change

            if warm_corr:
                estimate += self.warm(estimate, difficulties, person_filter)

            if iters >= max_iters:
                print('Maximum iterations reached before convergence.')

        except:
            estimate = np.nan

        return estimate

    def person_abils(self,
                     items=None,
                     warm_corr=True,
                     tolerance=0.0000001,
                     max_iters=100,
                     ext_score_adjustment=0.5):

        '''
        Creates raw score to ability estimate look-up table. Newton-Raphson ML
        estimation, includes optional Warm (1989) bias correction.
        '''

        if items is None:
            items = self.dataframe.columns

        estimates = {person: self.abil(person, items, warm_corr=warm_corr, tolerance=tolerance,
                                       max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)
                     for person in self.dataframe.index}

        self.person_abilities = pd.Series(estimates)

    def score_abil(self,
                   score,
                   items=None,
                   warm_corr=True,
                   tolerance=0.0000001,
                   max_iters=100,
                   ext_score_adjustment=0.5):

        '''
        Creates raw score to ability estimate look-up. Uses
        Newton-Raphson for ML with optional Warm (1989) bias correction.
        '''

        if items is None:
            items = self.dataframe.columns

        difficulties = self.diffs.loc[items]

        person_filter = [True for item in items]
        max_score = len(difficulties)

        if score == 0:
            score = ext_score_adjustment

        elif score == max_score:
            score -= ext_score_adjustment

        estimate = (log(score) - log(max_score - score) + statistics.mean(difficulties))
        change = 1
        iters = 0

        while (abs(change) > tolerance) & (iters <= max_iters):

            person_exp_list = [self.exp_score(estimate, difficulty)
                               for difficulty in difficulties]
            result = sum(person_exp_list)

            person_info_list = [self.variance(estimate, difficulty)
                                for difficulty in difficulties]
            info = sum(person_info_list)

            change = max(-1, min(1, (result - score) / info))
            estimate -= change

        if warm_corr:
            estimate += self.warm(estimate, difficulties, person_filter)

        if iters >= max_iters:
            print('Maximum iterations reached before convergence.')

        return estimate

    def abil_lookup_table(self,
                          items=None,
                          ext_scores=True,
                          warm_corr=True,
                          tolerance=0.0000001,
                          max_iters=100,
                          ext_score_adjustment=0.5):

        if items is None:
            items = self.dataframe.columns
            
        if ext_scores:
            score_range = range(len(items) + 1)
            
        else:
            score_range = range(1, len(items))

        abil_table = {score: self.score_abil(score, items=items, warm_corr=warm_corr, tolerance=tolerance,
                                             max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)
                      for score in score_range}
        abil_table = pd.Series(abil_table)

        self.abil_table = abil_table

    def warm(self,
             ability,
             difficulties,
             person_filter):

        '''
        Warm's (1989) bias correction for ML abiity estimates.
        '''

        j_list = [self.variance(ability, difficulty) * (1 - 2 * self.exp_score(ability, difficulty))
                  for flag, difficulty in zip(person_filter, difficulties)
                  if flag == flag]
        j = sum(j_list)

        i_list = [self.variance(ability, difficulty)
                  for flag, difficulty in zip(person_filter, difficulties)
                  if flag == flag]
        i = sum(i_list)

        return j / (2 * i ** 2)

    def csem(self,
             person,
             abilities=None,
             items=None):

        '''
        Calculates conditional standard error of measurement for an ability.
        '''

        if items is None:
            items = self.dataframe.columns

        difficulties = self.diffs.loc[items]

        if abilities is None:
            if hasattr(self, 'person_abilities') == False:
                self.person_abils()

            abilities = self.person_abilities

        person_data = self.dataframe.loc[person, items]
        person_filter = (person_data + 1) / (person_data + 1)

        info_list = [self.variance(abilities[person], difficulty)
                     for flag, difficulty in zip(person_filter, difficulties)
                     if flag == flag]
        total_info = sum(info_list)

        return 1 / sqrt(total_info)

    def category_counts_item(self,
                             item):

        if item in self.dataframe.columns:
            return self.dataframe.apply(pd.value_counts)[item].fillna(0).astype(int)

        else:
            print('Invalid item name')

    def category_counts_df(self):

        category_counts_df = pd.DataFrame(0, index=self.dataframe.columns, columns=[0, 1])

        for item in self.dataframe.columns:
            for score, count in self.category_counts_item(item).items():
                category_counts_df.loc[item].iloc[int(score)] = count

        category_counts_df['Total'] = self.dataframe.count()
        category_counts_df['Missing'] = self.no_of_persons - category_counts_df['Total']

        category_counts_df = category_counts_df.astype(int)

        category_counts_df.loc['Total']= category_counts_df.sum()

        self.category_counts = category_counts_df

    def fit_statistics(self,
                       warm_corr=True,
                       tolerance=0.0000001,
                       max_iters=100,
                       ext_score_adjustment=0.5,
                       constant=0.1,
                       method='cos',
                       matrix_power=3,
                       log_lik_tol=0.000001,
                       no_of_samples=100,
                       interval=None):

        if hasattr(self, 'diffs') == False:
            self.calibrate(constant=constant, method=method, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if hasattr(self, 'item_se') == False:
            self.std_errors(interval=interval, no_of_samples=no_of_samples, constant=constant, method=method,
                            matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if hasattr(self, 'person_abilities') == False:
            self.person_abils(warm_corr=warm_corr, tolerance=tolerance,
                              max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)

        '''
        Create matrices of expected scores, variances, kurtosis, residuals etc. to generate fit statistics
        '''

        abil_matrix = [[self.person_abilities[person] for item in self.dataframe.columns]
                       for person in self.dataframe.index]
        abil_df = pd.DataFrame(abil_matrix)
        abil_df.index = self.dataframe.index
        abil_df.columns = self.dataframe.columns

        diff_matrix = [self.diffs for person in self.dataframe.index]
        diff_df = pd.DataFrame(diff_matrix)
        diff_df.index = self.dataframe.index
        diff_df.columns = self.dataframe.columns

        missing_mask = (self.dataframe + 1) / (self.dataframe + 1)
        item_count = (self.dataframe == self.dataframe).sum(axis=0)
        person_count = (self.dataframe == self.dataframe).sum(axis=1)

        self.exp_score_df = 1 / (1 + np.exp(diff_df - abil_df))
        self.exp_score_df *= missing_mask

        self.info_df = self.exp_score_df * (1 - self.exp_score_df)
        self.info_df *= missing_mask

        self.kurtosis_df = (((-self.exp_score_df) ** 4) * (1 - self.exp_score_df) +
                            ((1 - self.exp_score_df) ** 4) * self.exp_score_df)
        self.kurtosis_df *= missing_mask

        self.residual_df = self.dataframe - self.exp_score_df
        self.std_residual_df = self.residual_df / np.sqrt(self.info_df.astype(float))

        scores = self.dataframe.sum(axis=1)
        max_scores = self.dataframe.count(axis=1)

        self.exp_score_df = self.exp_score_df[(scores > 0) & (scores < max_scores)]
        self.info_df = self.info_df[(scores > 0) & (scores < max_scores)]
        self.kurtosis_df = self.kurtosis_df[(scores > 0) & (scores < max_scores)]
        self.residual_df = self.residual_df[(scores > 0) & (scores < max_scores)]
        self.std_residual_df = self.std_residual_df[(scores > 0) & (scores < max_scores)]

        '''
        Item fit statistics
        '''

        self.item_outfit_ms = (self.std_residual_df ** 2).mean()
        self.item_infit_ms = (self.residual_df ** 2).sum() / self.info_df.sum()

        item_outfit_q = ((self.kurtosis_df / (self.info_df ** 2)) / (item_count ** 2)).sum() - (1 / item_count)
        item_outfit_q = np.sqrt(item_outfit_q)
        self.item_outfit_zstd = (np.cbrt(self.item_outfit_ms) - 1) * (3 / item_outfit_q) + (item_outfit_q / 3)

        item_infit_q = (self.kurtosis_df - self.info_df ** 2).sum() / (self.info_df.sum() ** 2)
        item_infit_q = np.sqrt(item_infit_q)
        self.item_infit_zstd = (np.cbrt(self.item_infit_ms) - 1) * (3 / item_infit_q) + (item_infit_q / 3)

        self.response_counts = self.dataframe.count(axis=0)
        self.item_facilities = self.dataframe.mean(axis=0)

        self.point_measure = {item: self.dataframe[item].corr(self.person_abilities)
                              for item in self.dataframe.columns}
        self.point_measure = pd.Series(self.point_measure)

        item_abil_dev = {item: self.person_abilities[self.dataframe[item] == self.dataframe[item]]
                         for item in self.dataframe.columns}
        for item in self.dataframe.columns:
            item_abil_dev[item] -= item_abil_dev[item].mean()

        exp_point_measure_df = self.exp_score_df - self.dataframe.mean()

        exp_point_measure_num = [(exp_point_measure_df[item] * item_abil_dev[item]).sum()
                                 for item in self.dataframe.columns]
        exp_point_measure_num = pd.Series(exp_point_measure_num)
        exp_point_measure_num.index = self.dataframe.columns

        exp_point_measure_den = exp_point_measure_df ** 2 + self.info_df
        exp_point_measure_den = exp_point_measure_den.sum()

        for item in self.dataframe.columns:
            exp_point_measure_den[item] *= (item_abil_dev[item] ** 2).sum()

        exp_point_measure_den = np.sqrt(exp_point_measure_den)

        self.exp_point_measure = exp_point_measure_num / exp_point_measure_den

        '''
        Person fit statistics
        '''

        self.csem_vector = 1 / np.sqrt(self.info_df.sum(axis=1))
        self.rsem_vector = np.sqrt((self.residual_df ** 2).sum(axis=1)) / self.info_df.sum(axis=1)

        self.person_outfit_ms = (self.std_residual_df ** 2).mean(axis=1)
        self.person_outfit_ms.name = 'Outfit MS'
        self.person_infit_ms = (self.residual_df ** 2).sum(axis=1) / self.info_df.sum(axis=1)
        self.person_infit_ms.name = 'Infit MS'

        base_df = self.kurtosis_df / (self.info_df ** 2)
        for column in self.dataframe.columns:
            base_df[column] /= (person_count ** 2)
        person_outfit_q = base_df.sum(axis=1) -  (1 / person_count)
        person_outfit_q = np.sqrt(person_outfit_q)
        self.person_outfit_zstd = (np.cbrt(self.person_outfit_ms) - 1) * (3 / person_outfit_q) + (person_outfit_q / 3)
        self.person_outfit_zstd.name = 'Outfit Z'

        person_infit_q = (self.kurtosis_df - self.info_df ** 2).sum(axis=1) / (self.info_df.sum(axis=1) ** 2)
        person_infit_q = np.sqrt(person_infit_q)
        self.person_infit_zstd = (np.cbrt(self.person_infit_ms) - 1) * (3 / person_infit_q) + (person_infit_q / 3)
        self.person_infit_zstd.name = 'Infit Z'

        differences = pd.DataFrame()
        for item in self.dataframe.columns:
            differences[item] = self.person_abilities - self.diffs[item]
        num = (differences * self.residual_df).sum(axis=0)
        den = (self.info_df * (differences ** 2)).sum(axis=0)
        self.discrimination = 1 + num / den

        '''
        Test-level fit statistics
        '''

        self.isi = np.sqrt(self.diffs.var() / (self.item_se ** 2).mean() - 1)
        self.item_strata = (4 * self.isi + 1) / 3
        self.item_reliability = self.isi ** 2 / (1 + self.isi ** 2)

        self.psi = (np.sqrt(np.var(self.person_abilities) - (self.rsem_vector ** 2).mean()) /
                     np.sqrt((self.rsem_vector ** 2).mean()))
        self.person_strata = (4 * self.psi + 1) / 3
        self.person_reliability = (self.psi ** 2) / (1 + (self.psi ** 2))

        res_list = []
        diff_list = []
        for person in self.dataframe.index:
            res_list += self.dataframe.loc[person].tolist()
            diff_list += list(self.diffs.values)

        self.item_residual_corr = self.std_residual_df.corrwith(self.diffs, axis=1)
        self.person_residual_corr = self.std_residual_df.corrwith(self.person_abilities, axis=0)

    def res_corr_analysis(self,
                          warm_corr=True,
                          tolerance=0.0000001,
                          max_iters=100,
                          ext_score_adjustment=0.5,
                          constant=0.1,
                          method='cos',
                          matrix_power=3,
                          log_lik_tol=0.000001):

        '''
        Analysis of correlations of standardised residuals for violations of local item interdependence
        and unidimensionality
        '''

        if hasattr(self, 'std_residual_df') == False:
            self.fit_statistics(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                ext_score_adjustment=ext_score_adjustment, constant=constant, method=method,
                                matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        self.residual_correlations = self.std_residual_df.corr(numeric_only=False)

        pca = PCA()
        try:
            pca.fit(self.std_residual_df.corr())

            self.eigenvectors = pd.DataFrame(pca.components_)
            self.eigenvectors.columns = [f'Eigenvector {pc + 1}' for pc in range(self.no_of_items)]

            self.eigenvalues = pca.explained_variance_
            self.eigenvalues = pd.DataFrame(self.eigenvalues)
            self.eigenvalues.index = [f'PC {pc + 1}' for pc in range(self.no_of_items)]
            self.eigenvalues.columns = ['Eigenvalue']

            self.variance_explained = pd.DataFrame(pca.explained_variance_ratio_)
            self.variance_explained.index = [f'PC {pc + 1}' for pc in range(self.no_of_items)]
            self.variance_explained.columns = ['Variance explained']

            self.loadings = self.eigenvectors.T * np.sqrt(pca.explained_variance_)
            self.loadings = pd.DataFrame(self.loadings)
            self.loadings.columns = [f'PC {pc + 1}' for pc in range(self.no_of_items)]
            self.loadings.index = [item for item in self.dataframe.columns]

        except:
            self.pca_fail = True
            print('PCA of residuals failed')

            self.eigenvectors = None
            self.eigenvalues = None
            self.variance_explained = None
            self.loadings = None

    def item_stats_df(self,
                      full=False,
                      zstd=False,
                      disc=False,
                      point_measure_corr=False,
                      dp=3,
                      warm_corr=True,
                      tolerance=0.0000001,
                      max_iters=100,
                      ext_score_adjustment=0.5,
                      method='cos',
                      constant=0.1,
                      no_of_samples=100,
                      interval=None):

        if full:
            zstd = True
            disc=True
            point_measure_corr = True

            if interval is None:
                interval = 0.95

        if (hasattr(self, 'item_low') == False) and (interval is not None):
            self.std_errors(interval=interval, no_of_samples=no_of_samples, constant=constant, method=method)

        if hasattr(self, 'item_infit_ms') == False:
            self.fit_statistics(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                ext_score_adjustment=ext_score_adjustment, method=method,
                                constant=constant, no_of_samples=no_of_samples, interval=interval)

        self.item_stats = pd.DataFrame()

        self.item_stats['Estimate'] = self.diffs.to_numpy().round(dp)
        self.item_stats['SE'] = self.item_se.to_numpy().round(dp)

        if interval is not None:
            self.item_stats[f'{round((1 - interval) * 50, 1)}%'] = self.item_low.to_numpy().round(dp)
            self.item_stats[f'{round((1 + interval) * 50, 1)}%'] = self.item_high.to_numpy().round(dp)

        self.item_stats['Count'] = self.response_counts.to_numpy().astype(int)
        self.item_stats['Facility'] = self.item_facilities.to_numpy().round(dp)

        self.item_stats['Infit MS'] = self.item_infit_ms.to_numpy().round(dp)
        if zstd:
            self.item_stats['Infit Z'] = self.item_infit_zstd.to_numpy().round(dp)
        self.item_stats['Outfit MS'] = self.item_outfit_ms.to_numpy().round(dp)
        if zstd:
            self.item_stats['Outfit Z'] = self.item_outfit_zstd.to_numpy().round(dp)

        if disc:
            self.item_stats['Discrim'] = self.discrimination.to_numpy().round(dp)

        if point_measure_corr:
            self.item_stats['PM corr'] = self.point_measure.to_numpy().round(dp)
            self.item_stats['Exp PM corr'] = self.exp_point_measure.to_numpy().round(dp)

        self.item_stats.index = self.dataframe.columns

    def person_stats_df(self,
                        full=False,
                        rsem=False,
                        dp=3,
                        warm_corr=True,
                        tolerance=0.0000001,
                        max_iters=100,
                        ext_score_adjustment=0.5,
                        method='cos',
                        constant=0.1):
        '''

        Produces a person stats dataframe with raw score, ability estimate,
        CSEM and RSEM for each person.

        '''

        if hasattr(self, 'person_infit_ms') == False:
            self.fit_statistics(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                ext_score_adjustment=ext_score_adjustment, method=method, constant=constant)

        if full:
            rsem = True

        person_stats_df = pd.DataFrame()
        person_stats_df.index = self.dataframe.index

        person_stats_df['Estimate'] = self.person_abilities.round(dp)

        person_stats_df['CSEM'] = self.csem_vector.round(dp)
        if rsem:
            person_stats_df['RSEM'] = self.rsem_vector.round(dp)

        person_stats_df['Score'] = self.dataframe.sum(axis=1).astype(int)
        person_stats_df['Max score'] = self.dataframe.count(axis=1).astype(int)
        person_stats_df['p'] = self.dataframe.mean(axis=1).round(dp)

        person_stats_df['Infit MS'] = [np.nan for person in self.dataframe.index]
        person_stats_df['Infit Z'] = [np.nan for person in self.dataframe.index]
        person_stats_df['Outfit MS'] = [np.nan for person in self.dataframe.index]
        person_stats_df['Outfit Z'] = [np.nan for person in self.dataframe.index]

        person_stats_df.update(self.person_infit_ms.round(dp))
        person_stats_df.update(self.person_infit_zstd.round(dp))
        person_stats_df.update(self.person_outfit_ms.round(dp))
        person_stats_df.update(self.person_outfit_zstd.round(dp))

        self.person_stats = person_stats_df
        
    def test_stats_df(self,
                      dp=3,
                      warm_corr=True,
                      tolerance=0.0000001,
                      max_iters=100,
                      ext_score_adjustment=0.5,
                      method='cos',
                      constant=0.1):

        if hasattr(self, 'psi') == False:
            self.fit_statistics(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                ext_score_adjustment=ext_score_adjustment, method=method,
                                constant=constant)

        self.test_stats = pd.DataFrame()

        self.test_stats['Items'] = [self.diffs.mean(),
                                    self.diffs.std(),
                                    self.isi,
                                    self.item_strata,
                                    self.item_reliability]

        self.test_stats['Persons'] = [self.person_abilities.mean(),
                                      self.person_abilities.std(),
                                      self.psi,
                                      self.person_strata,
                                      self.person_reliability]

        self.test_stats.index = ['Mean', 'SD', 'Separation ratio', 'Strata', 'Reliability']
        self.test_stats = round(self.test_stats, dp)

    def save_stats(self,
                   filename,
                   format='csv',
                   dp=3,
                   warm_corr=True,
                   tolerance=0.0000001,
                   max_iters=100,
                   ext_score_adjustment=0.5,
                   method='cos',
                   constant=0.1,
                   no_of_samples=100,
                   interval=None):

        if hasattr(self, 'item_stats') == False:
            self.item_stats_df(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                               ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                               no_of_samples=no_of_samples, interval=interval)

        if hasattr(self, 'person_stats') == False:
            self.person_stats_df(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                 ext_score_adjustment=ext_score_adjustment, method=method, constant=constant)

        if hasattr(self, 'test_stats') == False:
            self.test_stats_df(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                               ext_score_adjustment=ext_score_adjustment, method=method, constant=constant)

        if format == 'xlsx':

            if filename[-5:] != '.xlsx':
                filename += '.xlsx'

            writer = pd.ExcelWriter(filename, engine='xlsxwriter')

            self.item_stats.to_excel(writer, sheet_name='Item statistics')
            self.person_stats.to_excel(writer, sheet_name='Person statistics')
            self.test_stats.to_excel(writer, sheet_name='Test statistics')

            writer.save()

        else:
            if filename[-4:] == '.csv':
                filename = filename[:-4]

            self.item_stats.to_csv(f'{filename}_item_stats.csv')
            self.person_stats.to_csv(f'{filename}_person_stats.csv')
            self.test_stats.to_csv(f'{filename}_test_stats.csv')

    def save_residuals(self,
                       filename,
                       format='csv',
                       single=True,
                       dp=3,
                       warm_corr=True,
                       tolerance=0.0000001,
                       max_iters=100,
                       ext_score_adjustment=0.5,
                       method='cos',
                       constant=0.1):

        if hasattr(self, 'eigenvectors') == False:
            self.fit_statistics(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                ext_score_adjustment=ext_score_adjustment, method=method, constant=constant)

        if single:
            if format == 'xlsx':

                if filename[-5:] != '.xlsx':
                    filename += '.xlsx'

                writer = pd.ExcelWriter(filename, engine='xlsxwriter')
                row = 0

                self.eigenvectors.round(dp).to_excel(writer, sheet_name='Item residual analysis',
                                                     startrow=row, startcol=0)
                row += (self.eigenvectors.shape[0] + 2)

                self.eigenvalues.round(dp).to_excel(writer, sheet_name='Item residual analysis',
                                                    startrow=row, startcol=0)
                row += (self.eigenvalues.shape[0] + 2)

                self.variance_explained.round(dp).to_excel(writer, sheet_name='Item residual analysis',
                                                           startrow=row, startcol=0)
                row += (self.variance_explained.shape[0] + 2)

                self.loadings.round(dp).to_excel(writer, sheet_name='Item residual analysis',
                                                 startrow=row, startcol=0)

                writer.save()

            else:
                if filename[-4:] != '.csv':
                    filename += '.csv'

                with open(filename, 'a') as f:
                    self.eigenvectors.round(dp).to_csv(f)
                    f.write("\n")
                    self.eigenvalues.round(dp).to_csv(f)
                    f.write("\n")
                    self.variance_explained.round(dp).to_csv(f)
                    f.write("\n")
                    self.loadings.round(dp).to_csv(f)

        else:
            if format == 'xlsx':

                if filename[-5:] != '.xlsx':
                    filename += '.xlsx'

                writer = pd.ExcelWriter(filename, engine='xlsxwriter')

                self.eigenvectors.round(dp).to_excel(writer, sheet_name='Eigenvectors')
                self.eigenvalues.round(dp).to_excel(writer, sheet_name='Eigenvalues')
                self.variance_explained.round(dp).to_excel(writer, sheet_name='Variance explained')
                self.loadings.round(dp).to_excel(writer, sheet_name='Principal Component loadings')

                writer.save()

            else:
                if filename[-4:] == '.csv':
                    filename = filename[:-4]

                self.eigenvectors.round(dp).to_csv(f'{filename}_eigenvectors.csv')
                self.eigenvalues.round(dp).to_csv(f'{filename}_eigenvalues.csv')
                self.variance_explained.round(dp).to_csv(f'{filename}_variance_explained.csv')
                self.loadings.round(dp).to_csv(f'{filename}_principal_component_loadings.csv')

    def class_intervals(self,
                        abilities,
                        items=None,
                        no_of_classes=5):

        class_groups = [f'class_{class_no + 1}' for class_no in range(no_of_classes)]

        df = self.dataframe.copy()

        if items is None:
            items = list(self.dataframe.columns)

        df = df[items].dropna()
        abils = abilities.loc[df.index]

        quantiles = (abils.quantile([(i + 1) / no_of_classes
                                     for i in range(no_of_classes - 1)]))

        mask_dict = {}
        mask_dict['class_1'] = (abils < quantiles.values[0])
        mask_dict[f'class_{no_of_classes}'] = (abils >= quantiles.values[no_of_classes - 2])
        for class_no in range(no_of_classes - 2):
            mask_dict[f'class_{class_no + 2}'] = ((abils >= quantiles.values[class_no]) &
                                                  (abils < quantiles.values[class_no + 1]))

        class_sizes = {class_group: sum(mask_dict[class_group])
                       for class_group in class_groups}
        class_sizes = pd.Series(class_sizes)

        response_classes = {class_group: df.index[mask_dict[class_group]]
                            for class_group in class_groups}

        mean_abilities = {class_group: abils[mask_dict[class_group]].mean()
                          for class_group in class_groups}
        mean_abilities = pd.Series(mean_abilities)

        obs = {class_group: df[mask_dict[class_group]].mean().sum()
               for class_group in class_groups}

        for class_group in class_groups:
            obs[class_group] = pd.Series(obs[class_group])

        obs = pd.concat(obs, keys=obs.keys())

        class_abilities = {class_group: abils[mask_dict[class_group]]
                           for class_group in class_groups}
        class_abilities = pd.concat(class_abilities, keys=class_abilities.keys())

        return class_sizes, response_classes, class_abilities, mean_abilities, obs

    def class_intervals_cats(self,
                             abilities,
                             item,
                             no_of_classes=5):

        (class_sizes,
         response_classes,
         class_abilities,
         mean_abilities,
         obs_means) = self.class_intervals(abilities, items=[item], no_of_classes=no_of_classes)

        class_groups = [f'class_{class_no + 1}' for class_no in range(no_of_classes)]

        obs_props = {class_group: np.array([1 - obs_means[class_group][0], obs_means[class_group][0]])
                     for class_group in class_groups}

        obs_props = pd.DataFrame(obs_props).to_numpy().T

        return class_sizes, response_classes, class_abilities, mean_abilities, obs_props

    '''
    Plots
    '''

    def plot_data(self,
                  x_data,
                  y_data,
                  x_min=-5,
                  x_max=5,
                  y_max=0,
                  items=None,
                  obs=False,
                  x_obs_data=np.array([]),
                  y_obs_data=np.array([]),
                  thresh_line=False,
                  score_lines_item=[None, []],
                  score_lines_test=None,
                  point_info_lines_item=[None, []],
                  point_info_lines_test=None,
                  point_csem_lines=None,
                  score_labels=False,
                  point_info_labels=False,
                  warm=True,
                  cat_highlight=None,
                  graph_title='',
                  y_label='',
                  plot_style='colorblind',
                  black=False,
                  figsize=(8, 6),
                  font='Times',
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

        if tex:
            plt.rcParams["text.latex.preamble"].join([r"\usepackage{dashbox}", r"\setmainfont{xcolor}",])
        else:
            plt.rcParams["text.usetex"] = False

        plt.style.use('seaborn-v0_8-' + plot_style)

        graph, ax = plt.subplots(figsize=figsize)

        no_of_plots = y_data.shape[1]

        if black:
            for i in range(no_of_plots):
                ax.plot(x_data, y_data[:, i], '', label=i+1, color='black')

        else:
            for i in range(no_of_plots):
                ax.plot(x_data, y_data[:, i], '', label=i+1)

        if obs:
            no_of_obs_plots = y_obs_data.shape[1]
            for j in range (no_of_obs_plots):
                ax.plot(x_obs_data, y_obs_data[:, j], 'o')

        if items is not None:
            difficulties = self.diffs.loc[items]

        else:
            difficulties = self.diffs

        if thresh_line:
            plt.axvline(x=self.diffs.loc[items], color='darkred', linestyle='--')

        if score_lines_item[1] is not None:

            if (all(x > 0 for x in score_lines_item[1]) &
                all(x < 1 for x in score_lines_item[1])):

                abils_set = [np.log(score) - np.log(1 - score) + self.diffs[items]
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

            if (all(x > 0 for x in score_lines_test) &
                all(x < len(difficulties) for x in score_lines_test)):

                abils_set = [self.score_abil(score, items=difficulties.keys(), warm_corr=warm)
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

            info_set = [self.variance(ability, self.diffs[item])
                        for ability in point_info_lines_item[1]]

            for abil, info in zip(point_info_lines_item[1], info_set):
                plt.vlines(x=abil, ymin=-100, ymax=info, color='black', linestyles='dashed')
                if point_info_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=info, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if point_info_labels:
                    plt.text(x_min + (x_max - x_min) / 100, info + y_max / 50, str(round(info, 3)))

        if point_info_lines_test is not None:

            info_set = [sum(self.variance(ability, difficulty)
                            for difficulty in difficulties)
                        for ability in point_info_lines_test]

            for abil, info in zip(point_info_lines_test, info_set):
                plt.vlines(x=abil, ymin=-100, ymax=info, color='black', linestyles='dashed')
                if point_info_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=info, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if point_info_labels:
                    plt.text(x_min + (x_max - x_min) / 100, info + y_max / 50, str(round(info, 3)))

        if point_csem_lines is not None:

            info_set = [sum(self.variance(ability, difficulty)
                            for difficulty in difficulties)
                        for ability in point_csem_lines]
            info_set = np.array(info_set)
            csem_set = 1 / np.sqrt(info_set)

            for abil, csem in zip(point_csem_lines, csem_set):
                plt.vlines(x=abil, ymin=-100, ymax=csem, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=csem, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(x_min + (x_max - x_min) / 100, csem + y_max / 50, str(round(csem, 3)))

        if cat_highlight == 0:
            plt.axvspan(-100, self.diffs[items], facecolor='blue', alpha=0.2)

        elif cat_highlight == 1:
            plt.axvspan(self.diffs[items], 100, facecolor='blue', alpha=0.2)

        if y_max <= 0:
            y_max = 1.01

        plt.xlim(x_min, x_max)
        plt.ylim(0, y_max)

        plt.xlabel('Ability', fontname=font, fontsize=axis_font_size, fontweight='bold')
        plt.ylabel(y_label, fontname=font, fontsize=axis_font_size, fontweight='bold')
        plt.title(graph_title, fontname=font, fontsize=title_font_size, fontweight='bold')

        plt.grid(True)

        plt.tick_params(axis="x", labelsize=labelsize)
        plt.tick_params(axis="y", labelsize=labelsize)

        if filename is not None:
            plt.savefig(f'{filename}.{file_format}', dpi=plot_density)

        plt.close()

        return graph

    def icc(self,
            item,
            obs=False,
            xmin=-5,
            xmax=5,
            no_of_classes=5,
            title=None,
            thresh_line=False,
            score_lines=None,
            score_labels=False,
            cat_highlight=None,
            plot_style='dark-palette',
            black=False,
            font='Times',
            title_font_size=15,
            axis_font_size=12,
            labelsize=12,
            filename=None,
            file_format='png',
            dpi=300):

        '''
        Plots Item Characteristic Curves for SLM, with optional overplotting
        of observed data, threshold lines and expected score threshold lines.
        '''

        abilities = np.arange(-20, 20, 0.1)
        y = [self.exp_score(ability, self.diffs[item]) for ability in abilities]
        y = np.array(y).reshape([len(abilities), 1])

        if obs:
            if hasattr(self, 'person_abiliites') == False:
                self.person_abils(warm_corr=False)

            _, _, _, mean_abilities, obs_means = self.class_intervals(self.person_abilities, items=[item],
                                                                      no_of_classes=no_of_classes)

            xobsdata = mean_abilities
            yobsdata = obs_means
            yobsdata = np.array(yobsdata).reshape(no_of_classes, 1)

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        if title is not None:
            graphtitle = title
                
        else:
            graphtitle = ''

        ylabel = 'Expected score'

        plot = self.plot_data(x_data=abilities, y_data=y, x_obs_data=xobsdata, y_obs_data=yobsdata, x_min=xmin,
                              x_max=xmax, y_max=self.max_score, items=item, y_label=ylabel, graph_title=graphtitle,
                              obs=obs, thresh_line=thresh_line, score_lines_item=[item, score_lines],
                              score_labels=score_labels, cat_highlight=cat_highlight, plot_style=plot_style,
                              black=black, font=font, title_font_size=title_font_size, axis_font_size=axis_font_size,
                              labelsize=labelsize, filename=filename, plot_density=dpi, file_format=file_format)

        return plot

    def crcs(self,
             item,
             obs=None,
             xmin=-5,
             xmax=5,
             no_of_classes=5,
             title=None,
             thresh_line=False,
             cat_highlight=None,
             plot_style='colorblind',
             black=False,
             font='Times',
             title_font_size=15,
             axis_font_size=12,
             labelsize=12,
             filename=None,
             file_format='png',
             dpi=300):

        '''
        Plots Category Response Curves for SLM, with optional overplotting
        of observed data and threshold lines.
        '''

        abilities = np.arange(-20, 20, 0.1)
        y = np.array([[self.cat_prob(ability, self.diffs[item], category)
                       for category in [0, 1]]
                      for ability in abilities])

        if obs:
            if hasattr(self, 'person_abiliites') == False:
                self.person_abils(warm_corr=False)

            _, _, _, mean_abilities, obs_props = self.class_intervals_cats(self.person_abilities, item,
                                                                           no_of_classes=no_of_classes)

            xobsdata = mean_abilities
            yobsdata = obs_props

            if obs != 'all':
                if not all(cat in [0, 1] for cat in obs):
                    print("Invalid 'obs'. Valid values are 'None', 'all' and list of categories.")
                    return

                else:
                    yobsdata = yobsdata[:, obs]

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Probability'

        plot = self.plot_data(x_data=abilities,  y_data=y, x_min=xmin, x_max=xmax, y_max=1, x_obs_data=xobsdata,
                              y_obs_data=yobsdata, items=item, graph_title=graphtitle, y_label=ylabel,
                              obs=obs, thresh_line=thresh_line, cat_highlight=cat_highlight, plot_style=plot_style,
                              black=black, font=font, title_font_size=title_font_size, axis_font_size=axis_font_size,
                              labelsize=labelsize, filename=filename, plot_density=dpi, file_format=file_format)

        return plot

    def iic(self,
            item,
            xmin=-5,
            xmax=5,
            ymax=None,
            title=None,
            thresh_line=False,
            point_info_lines=None,
            point_info_labels=False,
            cat_highlight=None,
            plot_style='colorblind',
            black=False,
            font='Times',
            title_font_size=15,
            axis_font_size=12,
            labelsize=12,
            filename=None,
            file_format='png',
            dpi=300):

        '''
        Plots Item Information Curves.
        '''

        abilities = np.arange(-20, 20, 0.1)

        y = [self.variance(ability, self.diffs[item])
             for ability in abilities]
        y = np.array(y).reshape(len(abilities), 1)

        if ymax is None:
            ymax = max(y) * 1.1

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Fisher information'

        plot = self.plot_data(x_data=abilities, y_data=y, x_min=xmin, x_max=xmax, y_max=ymax,
                              items=item, graph_title=graphtitle, y_label=ylabel, thresh_line=thresh_line,
                              point_info_lines_item=[item, point_info_lines], point_info_labels=point_info_labels,
                              cat_highlight=cat_highlight, plot_style=plot_style, black=black, font=font,
                              title_font_size=title_font_size, axis_font_size=axis_font_size, labelsize=labelsize,
                              filename=filename, plot_density=dpi, file_format=file_format)

        return plot

    def tcc(self,
            items=None,
            obs=False,
            xmin=-5,
            xmax=5,
            no_of_classes=5,
            title=None,
            score_lines=None,
            score_labels=False,
            plot_style='colorblind',
            black=False,
            font='Times',
            title_font_size=15,
            axis_font_size=12,
            labelsize=12,
            filename=None,
            file_format='png',
            dpi=300):

        '''
        Plots Test Characteristic Curve for SLM.
        '''

        abilities = np.arange(-20, 20, 0.1)

        if items is None:
            items = list(self.dataframe.columns)
            difficulties = self.diffs

        else:
            difficulties = self.diffs.loc[items]

        y = [sum(self.exp_score(ability, difficulties[item])
                 for item in items)
             for ability in abilities]
        y = np.array(y).reshape(len(abilities), 1)

        if obs:
            if hasattr(self, 'person_abiliites') == False:
                self.person_abils(warm_corr=False)

            _, _, _, mean_abilities, obs_means = self.class_intervals(self.person_abilities, items=items,
                                                                      no_of_classes=no_of_classes)

            xobsdata = mean_abilities
            yobsdata = obs_means
            yobsdata = np.array(yobsdata).reshape(no_of_classes, 1)

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Expected score'

        plot = self.plot_data(x_data=abilities, y_data=y, items=items, x_obs_data=xobsdata, y_obs_data=yobsdata,
                              x_min=xmin, x_max=xmax, y_max=len(items), score_lines_test=score_lines,
                              graph_title=graphtitle, y_label=ylabel, obs=obs, score_labels=score_labels,
                              plot_style=plot_style, black=black, font=font, title_font_size=title_font_size,
                              axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                              plot_density=dpi, file_format=file_format)

        return plot

    def test_info(self,
                  items=None,
                  point_info_lines=None,
                  point_info_labels=False,
                  xmin=-5,
                  xmax=5,
                  ymax=None,
                  title=None,
                  plot_style='colorblind',
                  black=False,
                  font='Times',
                  title_font_size=15,
                  axis_font_size=12,
                  labelsize=12,
                  filename=None,
                  file_format='png',
                  dpi=300):

        '''
        Plots Test Information Curve for SLM.
        '''

        if items is None:
            difficulties = self.diffs
            items = self.dataframe.columns

        else:
            difficulties = self.diffs.loc[items]

        abilities = np.arange(-20, 20, 0.1)
        y = [sum(self.variance(ability, difficulties[item]) for item in items)
             for ability in abilities]
        y = np.array(y).reshape(len(abilities), 1)

        if ymax is None:
            ymax = max(y) * 1.1

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Fisher information'

        plot = self.plot_data(x_data=abilities, y_data=y, items=items, x_min=xmin, x_max=xmax, y_max=ymax,
                              graph_title=graphtitle, point_info_lines_test=point_info_lines,
                              point_info_labels=point_info_labels, y_label=ylabel, plot_style=plot_style,
                              black=black, font=font, title_font_size=title_font_size, axis_font_size=axis_font_size,
                              labelsize=labelsize, filename=filename, plot_density=dpi, file_format=file_format)

        return plot

    def test_csem(self,
                  items=None,
                  point_csem_lines=None,
                  point_csem_labels=False,
                  xmin=-5,
                  xmax=5,
                  ymax=5,
                  title=None,
                  plot_style='colorblind',
                  black=False,
                  font='Times',
                  title_font_size=15,
                  axis_font_size=12,
                  labelsize=12,
                  filename=None,
                  file_format='png',
                  dpi=300):

        '''
        Plots Test Conditional Standard Error of Measurement Curve for SLM.
        '''

        if items is None:
            difficulties = self.diffs
            items = self.dataframe.columns

        else:
            difficulties = self.diffs.loc[items]

        abilities = np.arange(-20, 20, 0.1)
        y = np.array([sum(self.variance(ability, difficulties[item])
                          for item in items)
                      for ability in abilities])
        y = 1 / np.sqrt(y)
        y = y.reshape(len(abilities), 1)

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Conditional SEM'

        plot = self.plot_data(x_data=abilities, y_data=y, items=items, x_min=xmin, x_max=xmax, y_max=ymax,
                              graph_title=graphtitle, point_csem_lines=point_csem_lines, score_labels=point_csem_labels,
                              y_label=ylabel, plot_style=plot_style, black=black, font=font,
                              title_font_size=title_font_size, axis_font_size=axis_font_size, labelsize=labelsize,
                              filename=filename, plot_density=dpi, file_format=file_format)

        return plot

    def std_residuals_plot(self,
                           items=None,
                           bin_width=0.5,
                           x_min=-5,
                           x_max=5,
                           normal=False,
                           title=None,
                           plot_style='colorblind',
                           black=False,
                           font='Times',
                           title_font_size=15,
                           axis_font_size=12,
                           labelsize=12,
                           filename=None,
                           file_format='png',
                           plot_density=300):

        '''
        Plots histogram of standardised residuals for SLM, with optional overplotting of Standard Normal Distribution.
        '''

        if items is None:
            items = self.dataframe.columns

        std_residual_df = self.std_residual_df[items]
        std_residual_list = std_residual_df.unstack().dropna()

        plot = self.std_residuals_hist(std_residual_list, bin_width=bin_width, x_min=x_min, x_max=x_max, normal=normal,
                                       title=title, plot_style=plot_style, font=font, title_font_size=title_font_size,
                                       axis_font_size=axis_font_size, labelsize=labelsize, black=black,
                                       filename=filename, file_format=file_format, plot_density=plot_density)

        return plot

class PCM(Rasch):

    def __init__(self,
                 dataframe,
                 max_score_vector=None,
                 extreme_persons=True,
                 no_of_classes=5):

        if max_score_vector is None:
            self.max_score_vector = {item: int(max_score)
                                     for item, max_score in zip(dataframe.columns, np.array(dataframe.max()))}
            self.max_score_vector = pd.Series(self.max_score_vector)
        else:
            self.max_score_vector = {item: int(max_score)
                                     for item, max_score in zip(dataframe.columns, max_score_vector)}
            self.max_score_vector = pd.Series(self.max_score_vector)

        if extreme_persons:
            self.invalid_responses = dataframe[dataframe.isna().all(axis=1)]
            self.dataframe = dataframe[~dataframe.isna().all(axis=1)]

        else:
            scores = dataframe.sum(axis=1)
            max_scores = (dataframe == dataframe).mul(self.max_score_vector, axis=1).sum(axis=1)

            self.invalid_responses = dataframe[(scores == 0) | (scores == max_scores)]
            self.dataframe = dataframe[(scores > 0) & (scores < max_scores)]

        self.no_of_items = self.dataframe.shape[1]
        self.no_of_persons = self.dataframe.shape[0]
        self.persons = self.dataframe.index
        self.no_of_classes = no_of_classes

    '''
    Partial Credit Model (Masters 1982) formulation of the polytomous Rasch model,
    with associated methods.
    '''

    def cat_prob_centred(self,
                         ability,
                         difficulty,
                         category,
                         thresholds):

        '''
        Category probability function which calculates the probability of obtaining
        a score in a category given the person ability, item difficulty and set of
        Rasch-Andrich thresholds.
        '''

        max_score = len(thresholds) - 1

        cat_prob_nums = [exp(-sum(thresholds[:cat + 1]) +
                             cat * (ability - difficulty))
                         for cat in range(max_score + 1)]

        return cat_prob_nums[category] / sum(cat_prob_nums)

    def cat_prob_uncentred(self,
                           ability,
                           category,
                           thresholds):

        '''
        Category probability function which calculates the probability of obtaining a score
        in a category given the person ability and set of uncentred thresholds.
        '''

        max_score = len(thresholds)

        cat_prob_nums = [exp(cat * ability - sum(thresholds[:cat]))
                         for cat in range(max_score + 1)]

        return cat_prob_nums[category] / sum(cat_prob_nums)

    def exp_score_uncentred(self,
                            ability,
                            thresholds):

        '''
        Calculates the expected score on an intem given person ability and
        set of uncentred thresholds.
        '''

        max_score = len(thresholds)

        cat_prob_nums = [exp(category * ability - sum(thresholds[:category]))
                         for category in range(max_score + 1)]

        exp_score = (sum(category * prob for category, prob in enumerate(cat_prob_nums)) /
                     sum(cat_prob_nums))

        return exp_score

    def exp_score_centred(self,
                          ability,
                          difficulty,
                          thresholds):

        '''
        Calculates the expected score on an intem given person ability, item
        difficulty and set of Rasch-Andrich thresholds.
        '''

        max_score = len(thresholds) - 1

        cat_nums = [exp(category * (ability - difficulty) - sum(thresholds[:category + 1]))
                    for category in range(max_score + 1)]

        num = sum(category * cat_nums[category] for category in range(max_score + 1))

        den = sum(cat_nums)

        exp_score = num / den

        return exp_score

    def variance_uncentred(self,
                           ability,
                           thresholds):

        '''
        Item information function which calculates the item (Fisher) information
        from an item given the person ability and set of uncentred thresholds.
        This is also the variance of the expected score function
        for the person ability.
        '''

        max_score = len(thresholds)

        cat_prob_nums = [exp(category * ability - sum(thresholds[:category]))
                         for category in range(max_score + 1)]

        expected = self.exp_score_uncentred(ability, thresholds)

        variance = (sum(((category - expected) ** 2) * cat_prob
                        for category, cat_prob in enumerate(cat_prob_nums)) /
                    sum(cat_prob_nums))

        return variance

    def variance_centred(self,
                         ability,
                         difficulty,
                         thresholds):

        '''
        Item information function which calculates the item (Fisher) information
        from an item given the person ability, item difficulty and set of
        Rasch-Andrich thresholds.
        This is also the variance of the expected score function
        for the person ability.
        '''

        max_score = len(thresholds) - 1

        expected = self.exp_score_centred(ability,
                                          difficulty,
                                          thresholds)

        variance = sum(((category - expected) ** 2) *
                        self.cat_prob_centred(ability,
                                              difficulty,
                                              category,
                                              thresholds)
                       for category in range(max_score + 1))

        return variance

    def kurtosis_uncentred(self,
                           ability,
                           thresholds):

        '''
        Calculates an item's kurtosis given person ability and item difficulty.
        '''

        max_score = len(thresholds)

        cat_prob_nums = [exp(category * ability -
                             sum(thresholds[:category]))
                         for category in range(max_score + 1)]

        expected = self.exp_score_uncentred(ability,
                                            thresholds)

        kurtosis = (sum(((category - expected) ** 4) * cat_prob
                        for category, cat_prob in enumerate(cat_prob_nums)) /
                    sum(cat_prob_nums))

        return kurtosis

    def kurtosis_centred(self,
                         ability,
                         difficulty,
                         thresholds):

        '''
        Calculates an item's kurtosis given person ability and item difficulty.
        '''

        max_score = len(thresholds) - 1

        expected = self.exp_score_centred(ability,
                                          difficulty,
                                          thresholds)

        cat_prob_nums = [exp(-sum(thresholds[:category + 1]) +
                             category * (ability - difficulty))
                         for category in range(max_score + 1)]

        cat_probs = np.array(cat_prob_nums) / sum(cat_prob_nums)

        kurtosis = sum(((category - expected) ** 4) * cat_prob
                       for category, cat_prob in enumerate(cat_probs))

        return kurtosis

    def _matrix_element(self,
                        item_1,
                        item_2):

        '''
        ** Private method **
        Create mini-matrix of conditional category relative frequencies for all
        combinaitons of thresholds across a pair of items. This mini-matrix is
        a building block for the full matrix.
        '''

        df_array = np.array(self.dataframe)

        mat_block = [[np.count_nonzero((df_array[:, item_1] == i + 1) &
                                       (df_array[:, item_2] == j))
                      for j in range(self.max_score_vector[item_2])]
                     for i in range(self.max_score_vector[item_1])]

        mat_block = np.array(mat_block)

        return mat_block

    def _ccrf_block(self,
                    item):

        '''
        ** Private method **
        Create a block of mini-matrices for an item featuring combinations
        with all other items. Append these into a block formed of several rows.
        '''

        row_block = np.concatenate([self._matrix_element(item, item_2)
                                    for item_2 in range(self.no_of_items)],
                                   axis = 1)

        return row_block

    def calibrate(self,
                  constant=0.1,
                  method='cos',
                  matrix_power=3,
                  log_lik_tol=0.000001):

        '''
        PAIR item difficulty estimation with _matrix_element() & _ccrf_block().
        '''

        self.null_persons =  self.dataframe.index[self.dataframe.isnull().all(1)]
        self.dataframe = self.dataframe.drop(self.null_persons)
        self.no_of_persons = self.dataframe.shape[0]

        matrix = np.concatenate([self._ccrf_block(item)
                                 for item in range(self.no_of_items)],
                                axis = 0)

        matrix = np.array(matrix).astype(np.float64)

        constant_matrix = (matrix + matrix.T > 0).astype(np.float64)
        constant_matrix *= constant
        matrix += constant_matrix
        matrix += (np.identity(self.max_score_vector.sum()) * constant)

        mat = np.linalg.matrix_power(matrix, matrix_power)
        mat_pow = matrix_power

        while 0 in mat:

            mat = np.matmul(mat, matrix)
            mat_pow += 1

            if mat_pow == matrix_power + 5:
                break

        threshold_vector = self.priority_vector(mat, method=method, log_lik_tol=log_lik_tol, pcm=True)
        self.threshold_list = threshold_vector

        self.thresholds_uncentred = {}
        self.central_diffs = {}
        self.thresholds_centred = {}

        for i, item in enumerate(self.dataframe.columns):

            item_max = self.max_score_vector[i]

            start = sum(self.max_score_vector[:i])
            finish = start + item_max

            self.thresholds_uncentred[item] = threshold_vector[start:finish]
            self.central_diffs[item] = np.mean(self.thresholds_uncentred[item])

            thresholds_centred = np.zeros((len(self.thresholds_uncentred[item]) + 1))
            thresholds_centred[1:] = self.thresholds_uncentred[item] - self.central_diffs[item]
            self.thresholds_centred[item] = thresholds_centred

        self.central_diffs = pd.Series(self.central_diffs)

    def calibrate_anchor(self,
                         anchors,
                         sd_ratio_tol=1.1,
                         correlation_tol=0.95,
                         min_anchors=6,
                         constant=0.1,
                         method='cos',
                         matrix_power=3,
                         log_lik_tol=0.000001):

        if hasattr(self, 'central_diffs') == False:
            self.calibrate(constant=constant, method=method, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        x = anchors
        y = self.central_diffs.copy()[x.index]
        x = x.sort_index()
        y = y.sort_index()

        difference = x - y
        median_difference = np.median(difference)
        mad = np.median(abs(difference - median_difference))
        robust_z = 0.6745 * (difference - median_difference) / mad
        abs_z = abs(robust_z)

        drop_x = {}
        drop_y = {}

        keep_x = x.copy()
        keep_y = y.copy()

        for key in robust_z.keys():
            if abs(robust_z.loc[key]) > 2:
                drop_x[key] = x[key]
                drop_y[key] = y[key]

                keep_x = keep_x.drop(labels=key)
                keep_y = keep_y.drop(labels=key)
                abs_z = abs_z.drop(labels=key)

        correlation = np.corrcoef(keep_x, keep_y)[0, 1]
        sd_ratio = np.std(keep_x) / np.std(keep_y)
        fail = False

        while ((sd_ratio > sd_ratio_tol) or
               (sd_ratio < 1 / sd_ratio_tol) or
               (correlation < correlation_tol)):

            drop_item = abs_z.idxmax()

            drop_x[drop_item] = x[drop_item]
            drop_y[drop_item] = y[drop_item]

            keep_x = keep_x.drop(labels=drop_item)
            keep_y = keep_y.drop(labels=drop_item)
            abs_z = abs_z.drop(labels=drop_item)

            if len(abs_z) < min_anchors:
                fail = True
                break

            else:
                correlation = np.corrcoef(keep_x, keep_y)[0, 1]
                sd_ratio = np.std(keep_x) / np.std(keep_y)

        if fail:
            print('Anchoring failed: too few anchors. Please review data and parameters')

        else:
            drop_x = pd.Series(drop_x).sort_index()
            drop_y = pd.Series(drop_y).sort_index()

            self.anchor_trans_constant = keep_x.mean() - keep_y.mean()
            self.anchor_correlation = correlation
            self.anchor_sd_ratio = sd_ratio

            self.anchors_keep = list(keep_x.keys())
            self.anchors_drop = list(drop_x.keys())
            self.anchor_robust_z = robust_z

            fig, ax = plt.subplots(figsize=(6, 6))

            ax.scatter(keep_x, keep_y, s=50, alpha=0.75)
            ax.scatter(drop_x, drop_y, s=50, alpha=0.75)

            super_min = min(min(x), min(y))
            super_max = max(max(x), max(y))
            offset = (super_max - super_min) / 40

            b, a = np.polyfit(keep_x, keep_y, deg=1)
            reg_line_points = np.linspace(super_min, super_max, num=2)
            ax.plot(reg_line_points, a + b * reg_line_points, color='darkred')

            for i, txt in enumerate(keep_x.keys()):
                ax.annotate(txt, (keep_x[i] + offset, keep_y[i] - offset/2))

            for i, txt in enumerate(drop_x.keys()):
                ax.annotate(txt, (drop_x[i] + offset, drop_y[i] - offset/2))

            plt.xlabel('Anchor difficulty')
            plt.ylabel('Calibrated difficulty')
            plt.legend(['Used anchor item', 'Unused anchor item'])

            plt.savefig('anchor_selection.png', dpi=300)

            self.central_diffs_anchor = self.central_diffs.copy() + self.anchor_trans_constant
            for item in anchors.index:
                self.central_diffs_anchor[item] = anchors.loc[item]

            self.thresholds_uncentred_anchor = {item: self.thresholds_centred[item][1:] +
                                                      self.central_diffs_anchor[item]
                                                for item in self.dataframe.columns}
            for item in self.dataframe.columns:
                self.thresholds_uncentred_anchor[item] = pd.Series(self.thresholds_uncentred_anchor[item])
                self.thresholds_uncentred_anchor[item].index = self.thresholds_uncentred[item].index
            for item in self.dataframe.columns:
                if item in anchors.index:
                    self.thresholds_uncentred_anchor[item].iloc[0] = anchors[0]

    def std_errors(self,
                   interval=None,
                   constant=0.1,
                   method='cos',
                   matrix_power=3,
                   log_lik_tol=0.000001,
                   no_of_samples=100):

        '''
        Bootstraped standard error estimates for item difficulties.
        '''

        samples = [PCM(self.dataframe.sample(frac = 1, replace = True),
                       self.max_score_vector)
                   for sample in range(no_of_samples)]

        calibrations_thresholds = {}
        calibrations_central = {}

        for sample in samples:
            sample.calibrate(constant=constant, method=method, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        for item in self.dataframe.columns:
            calibrations_thresholds[item] = np.stack((samples[sample].thresholds_uncentred[item]
                                                      for sample in range(no_of_samples)))
            calibrations_central[item] = np.array([samples[sample].central_diffs[item]
                                             for sample in range(no_of_samples)])

        self.threshold_se = {item: np.std(calibrations_thresholds[item], axis=0)
                             for item in self.dataframe.columns}

        if interval is not None:
            self.threshold_low = {item: np.percentile(calibrations_thresholds[item], (1 - interval) * 50, axis=0)
                                  for item in self.dataframe.columns}

            self.threshold_high = {item: np.percentile(calibrations_thresholds[item], (1 + interval) * 50, axis=0)
                                   for item in self.dataframe.columns}

        else:
            self.threshold_low = None
            self.threshold_high = None

        self.central_se = pd.Series({item: np.std(calibrations_central[item], axis=0)
                                     for item in self.dataframe.columns})

        if interval is not None:
            self.central_low = pd.Series({item: np.percentile(calibrations_central[item], (1 - interval) * 50, axis=0)
                                          for item in self.dataframe.columns})

            self.central_high = pd.Series({item: np.percentile(calibrations_central[item], (1 + interval) * 50, axis=0)
                                           for item in self.dataframe.columns})

        else:
            self.threshold_low = None
            self.threshold_high = None

    def abil(self,
             person,
             thresholds=None,
             max_score_vector=None,
             warm_corr=True,
             tolerance=0.0000001,
             max_iters=100,
             ext_score_adjustment=0.5):

        '''
        Creates raw score to ability estimate look-ups for a set of items.
        Uses Newton-Raphson for ML with optional Warm (1989) bias correction.
        '''

        if thresholds is None:
            thresholds = self.thresholds_uncentred

        if max_score_vector is None:
            max_score_vector = self.max_score_vector

        person_data = self.dataframe.loc[person]
        person_filter = (person_data + 1) / (person_data + 1)
        score = np.nansum(person_data)

        ext_score = np.nansum(max_score_vector * person_filter)

        if score == 0:
            score = ext_score_adjustment

        elif score == ext_score:
            score -= ext_score_adjustment

        try:
            estimate = (log(score) -
                        log(ext_score - score) +
                        statistics.mean(self.threshold_list))
            change = 1
            iters = 0

            while (abs(change) > tolerance) & (iters <= max_iters):

                result = sum(self.exp_score_uncentred(estimate, item_thresholds)
                             for item, item_thresholds in thresholds.items()
                             if person_filter[item] == 1)

                info = sum(self.variance_uncentred(estimate, item_thresholds)
                           for item, item_thresholds in thresholds.items()
                           if person_filter[item] == 1)

                change = max(-1, min(1, (result - score) / info))
                estimate -= change
                iters += 1

            if warm_corr:
                estimate += self.warm(estimate, thresholds, person_filter)

            if iters >= max_iters:
                print('Maximum iterations reached before convergence.')

        except:
            estimate = np.nan

        return estimate

    def score_abil(self,
                   score,
                   items=None,
                   warm_corr=True,
                   tolerance=0.0000001,
                   max_iters=100,
                   ext_score_adjustment=0.5):

        '''
        Creates raw score to ability estimate look-ups for a set of items.
        Uses Newton-Raphson for ML with optional Warm (1989) bias correction.
        '''

        if items is None:
            items = self.dataframe.columns
            difficulties = self.central_diffs
            thresholds = self.thresholds_uncentred
            max_score_vector = self.max_score_vector

        else:
            difficulties = self.central_diffs.loc[items]
            thresholds = {item: self.thresholds_uncentred[item] for item in items}
            max_score_vector = self.max_score_vector.loc[items]

        person_filter_dict = {item: True for item in items}

        max_score_vector = np.array(max_score_vector)
        ext_score = np.nansum(max_score_vector)

        if score == 0:
            score = ext_score_adjustment

        elif score == ext_score:
            score -= ext_score_adjustment

        estimate = (log(score) -
                    log(ext_score - score) +
                    np.mean(difficulties))

        change = 1
        iters = 0

        while (abs(change) > tolerance) & (iters <= max_iters):

            result = sum(self.exp_score_uncentred(estimate, thresholds[item])
                         for item in thresholds.keys())

            info = sum(self.variance_uncentred(estimate, thresholds[item])
                       for item in thresholds.keys())

            change = max(-1, min(1, (result - score) / info))
            estimate -= change
            iters += 1

        if warm_corr:
            estimate += self.warm(estimate, thresholds, person_filter_dict)

        if iters >= max_iters:
            print('Maximum iterations reached before convergence.')

        return estimate

    def abil_lookup_table(self,
                          items=None,
                          warm_corr=True,
                          tolerance=0.0000001,
                          max_iters=100,
                          ext_score_adjustment=0.5):

        if items is None:
            items = self.dataframe.columns
            max_score_vector = self.max_score_vector

        else:
            max_score_vector = self.max_score_vector.loc[items]

        abil_table = {score: self.score_abil(score, items=items, warm_corr=warm_corr, tolerance=tolerance,
                                             max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)
                      for score in range(max_score_vector.sum() + 1)}
        abil_table = pd.Series(abil_table)

        self.abil_table = abil_table

    def warm(self,
             ability,
             thresholds_uncentred,
             person_filter):

        '''
        Warm's (1989) bias correction for ML abiity estimates
        '''

        exp_scores = [self.exp_score_uncentred(ability, thresholds)
                      for item, thresholds in thresholds_uncentred.items()
                      if person_filter[item] == 1]

        variances = [self.variance_uncentred(ability, thresholds)
                      for item, thresholds in thresholds_uncentred.items()
                      if person_filter[item] == 1]

        part_1 = sum(sum((category ** 3) * self.cat_prob_uncentred(ability, category, thresholds)
                         for category in range(self.max_score_vector[item] + 1))
                      for item, thresholds in thresholds_uncentred.items()
                      if person_filter[item] == 1)

        part_2 = 3 * sum((info + (exp_score ** 2)) * exp_score
                         for info, exp_score in zip(variances, exp_scores))

        part_3 = sum(2 * (exp_score ** 3) for exp_score in exp_scores)

        warm_correction = ((part_1 - part_2 + part_3) /
                           (2 * (sum(variances) ** 2)))

        return warm_correction

    def csem_uncentred(self,
                       person,
                       abilities=None,
                       thresholds=None):

        '''
        Calculates conditional standard error of measurement for an ability.
        '''

        if abilities is None:
            abilities = self.person_abilities

        if thresholds is None:
            thresholds = self.thresholds_uncentred

        person_data = self.dataframe.loc[person].to_numpy()
        person_filter = (person_data + 1) / (person_data + 1)
        person_filter_dict = {item: flag for item, flag in zip(self.dataframe.columns, person_filter)}

        total_info = sum(self.variance_uncentred(abilities[person], item_thresholds)
                         for item, item_thresholds in thresholds.items()
                         if person_filter_dict[item] == person_filter_dict[item])

        cond_sem = 1 / np.sqrt(total_info)

        return cond_sem

    def csem_centred(self,
                     person,
                     abilities=None,
                     thresholds=None):

        '''
        Calculates conditional standard error of measurement for an ability.
        '''

        if abilities is None:
            abilities = self.person_abilities

        if thresholds is None:
            thresholds = self.thresholds_centred

        person_data = self.dataframe.loc[person].to_numpy()
        person_filter = (person_data + 1) / (person_data + 1)
        person_filter_dict = {item: flag for item, flag in zip(self.dataframe.columns, person_filter)}

        total_info = sum(self.variance_centred(abilities[person], item_thresholds)
                         for item, item_thresholds in thresholds.items()
                         if person_filter_dict[item] == person_filter_dict[item])

        cond_sem = 1 / np.sqrt(total_info)

        return cond_sem

    def person_abils(self,
                     thresholds=None,
                     max_score_vector=None,
                     warm_corr=True,
                     tolerance=0.0000001,
                     max_iters=100,
                     ext_score_adjustment=0.5):

        '''
        Creates raw score to ability estimate look-up table. Newton-Raphson ML
        estimation, includes optional Warm (1989) bias correction.
        '''

        if thresholds is None:
            thresholds = self.thresholds_uncentred

        if max_score_vector is None:
            max_score_vector = self.max_score_vector

        estimates = {person: self.abil(person, thresholds=thresholds, max_score_vector=max_score_vector,
                                       warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                       ext_score_adjustment=ext_score_adjustment)
                     for person in self.dataframe.index}

        self.person_abilities = pd.Series(estimates)

    def category_counts_item(self,
                             item):

        if item in self.dataframe.columns:
            counts = self.dataframe.apply(pd.value_counts)[item][:self.max_score_vector[item] + 1].fillna(0).astype(int)

            return counts

        else:
            print('Invalid item name')

    def category_counts_df(self):

        max_score = max(self.max_score_vector)
        category_counts_df = pd.DataFrame(-1, index=self.dataframe.columns, columns=np.arange(max_score + 1))

        for item in self.dataframe.columns:
            for score, count in self.category_counts_item(item).items():
                category_counts_df.loc[item].iloc[int(score)] = count

            for score in range(self.max_score_vector[item] + 1):
                if (category_counts_df.loc[item].iloc[score] == -1):
                        category_counts_df.loc[item].iloc[score] = 0

        category_counts_df['Total'] = self.dataframe.count()
        category_counts_df['Missing'] = self.no_of_persons - category_counts_df['Total']

        category_counts_df = category_counts_df.astype(int)

        category_counts_df.loc['Total']= category_counts_df.sum()

        category_counts_df = category_counts_df.replace(-1, 'n/a')

        self.category_counts = category_counts_df

    def fit_statistics(self,
                       warm_corr=True,
                       tolerance=0.0000001,
                       max_iters=100,
                       ext_score_adjustment=0.5,
                       constant=0.1,
                       method='cos',
                       matrix_power=3,
                       log_lik_tol=0.000001,
                       no_of_samples=100,
                       interval=None):

        if hasattr(self, 'thresholds_uncentred') == False:
            self.calibrate(constant=constant, method=method)

        if hasattr(self, 'threshold_se') == False:
            self.std_errors(interval=interval, no_of_samples=no_of_samples,
                            constant=constant, method=method)

        if hasattr(self, 'person_abilities') == False:
            self.person_abils(warm_corr=warm_corr, tolerance=tolerance,
                              max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)

        '''
        Create matrices of expected scores, variances, kurtosis, residuals etc. to generate fit statistics
        '''

        df = self.dataframe.copy()

        scores = df.sum(axis=1)
        max_scores = (df == df).mul(self.max_score_vector, axis=1).sum(axis=1)

        df = df[(scores > 0) & (scores < max_scores)]
        abilities = self.person_abilities.loc[df.index]

        missing_mask = (df + 1) / (df + 1)
        item_count = (df == df).sum(axis=0)
        person_count = (df == df).sum(axis=1)

        max_max_score = self.max_score_vector.max()

        self.cat_prob_dict = {cat: {item: {person: self.cat_prob_uncentred(abil, cat, self.thresholds_uncentred[item])
                                           if self.max_score_vector[item] >= cat
                                           else 0
                                           for person, abil in abilities.items()}
                                    for item in self.dataframe.columns}
                              for cat in range(max_max_score + 1)}

        for cat in range(max_max_score + 1):
            self.cat_prob_dict[cat] = pd.DataFrame(self.cat_prob_dict[cat])

        self.exp_score_df = sum(cat * self.cat_prob_dict[cat] for cat in range(max_max_score + 1))
        self.exp_score_df *= missing_mask

        self.info_df = sum(((cat - self.exp_score_df) ** 2) * self.cat_prob_dict[cat]
                            for cat in range(max_max_score + 1))
        self.info_df *= missing_mask

        self.kurtosis_df = sum(self.cat_prob_dict[cat] * ((cat - self.exp_score_df) ** 4)
                               for cat in range(max_max_score + 1))
        self.kurtosis_df *= missing_mask

        self.residual_df = df - self.exp_score_df
        self.std_residual_df = self.residual_df / np.sqrt(self.info_df.astype(float))

        '''
        Item fit statistics
        '''

        self.item_outfit_ms = (self.std_residual_df ** 2).mean()
        self.item_infit_ms =  (self.residual_df ** 2).sum() / self.info_df.sum()

        item_outfit_q = ((self.kurtosis_df / (self.info_df ** 2)) / (item_count ** 2)).sum() - (1 / item_count)
        item_outfit_q = np.sqrt(item_outfit_q)
        self.item_outfit_zstd = (np.cbrt(self.item_outfit_ms) - 1) * (3 / item_outfit_q) + (item_outfit_q / 3)

        item_infit_q = (self.kurtosis_df - self.info_df ** 2).sum() / (self.info_df.sum() ** 2)
        item_infit_q = np.sqrt(item_infit_q)
        self.item_infit_zstd = (np.cbrt(self.item_infit_ms) - 1) * (3 / item_infit_q) + (item_infit_q / 3)

        self.response_counts = self.dataframe.count(axis=0)
        self.item_facilities = self.dataframe.mean(axis=0) / self.max_score_vector

        self.point_measure = [self.dataframe[item].corr(self.person_abilities)
                              for item in self.dataframe.columns]
        self.point_measure = pd.Series(self.point_measure)
        self.point_measure.index = self.dataframe.columns

        item_abil_dev = {item: self.person_abilities[self.dataframe[item] == self.dataframe[item]]
                         for item in self.dataframe.columns}
        for item in self.dataframe.columns:
            item_abil_dev[item] -= item_abil_dev[item].mean()

        exp_point_measure_df = self.exp_score_df - self.dataframe.mean()

        exp_point_measure_num = [(exp_point_measure_df[item] * item_abil_dev[item]).sum()
                                 for item in self.dataframe.columns]
        exp_point_measure_num = pd.Series(exp_point_measure_num)
        exp_point_measure_num.index = self.dataframe.columns

        exp_point_measure_den = exp_point_measure_df ** 2 + self.info_df
        exp_point_measure_den = exp_point_measure_den.sum()

        for item in self.dataframe.columns:
            exp_point_measure_den[item] *= (item_abil_dev[item] ** 2).sum()

        exp_point_measure_den = np.sqrt(exp_point_measure_den)

        self.exp_point_measure = exp_point_measure_num / exp_point_measure_den

        '''
        Threshold fit statistics
        '''

        dich_thresh = {}
        for item in self.dataframe.columns:
            dich_thresh[item] = {}

            for threshold in range(self.max_score_vector[item]):
                dich_thresh[item][threshold + 1] = self.dataframe[item].where(self.dataframe[item].isin([threshold,
                                                                                                         threshold + 1]),
                                                                              np.nan)
                dich_thresh[item][threshold + 1] -= threshold

        dich_thresh_exp = {item: {} for item in self.dataframe.columns}
        dich_thresh_var = {item: {} for item in self.dataframe.columns}
        dich_thresh_kur = {item: {} for item in self.dataframe.columns}
        dich_residuals = {item: {} for item in self.dataframe.columns}
        dich_std_residuals = {item: {} for item in self.dataframe.columns}

        dich_thresh_count = {item: {threshold + 1:
                                    (dich_thresh[item][threshold + 1] ==
                                     dich_thresh[item][threshold + 1]).sum().sum()
                                    for threshold in range(self.max_score_vector[item])}
                             for item in self.dataframe.columns}

        for item in self.dataframe.columns:
            for threshold in range(self.max_score_vector[item]):

                diff_list = [self.thresholds_uncentred[item][threshold]
                               for person in self.dataframe.index]
                diff_series = pd.Series(diff_list)
                diff_series.index = self.dataframe.index

                missing_mask = ((dich_thresh[item][threshold + 1] + 1) /
                                (dich_thresh[item][threshold + 1] + 1))

                dich_thresh_exp[item][threshold + 1] = 1 / (1 + np.exp(diff_series - self.person_abilities))
                dich_thresh_exp[item][threshold + 1] *= missing_mask

                dich_thresh_var[item][threshold + 1] = (dich_thresh_exp[item][threshold + 1] *
                                                        (1 - dich_thresh_exp[item][threshold + 1]))
                dich_thresh_var[item][threshold + 1] *= missing_mask

                dich_thresh_kur[item][threshold + 1] = (((-dich_thresh_exp[item][threshold + 1]) ** 4) *
                                                        (1 - dich_thresh_exp[item][threshold + 1]) +
                                                        ((1 - dich_thresh_exp[item][threshold + 1]) ** 4) *
                                                        dich_thresh_exp[item][threshold + 1])
                dich_thresh_kur[item][threshold + 1] *= missing_mask

                dich_residuals[item][threshold + 1] = (dich_thresh[item][threshold + 1] -
                                                       dich_thresh_exp[item][threshold + 1])
                dich_std_residuals[item][threshold + 1] = (dich_residuals[item][threshold + 1] /
                                                           np.sqrt(dich_thresh_var[item][threshold + 1]))

        self.threshold_outfit_ms = {item: {threshold + 1:
                                           (dich_std_residuals[item][threshold + 1] ** 2).sum() /
                                           dich_thresh[item][threshold + 1].count()
                                           for threshold in range(self.max_score_vector[item])}
                                    for item in self.dataframe.columns}

        for item in self.dataframe.columns:
            self.threshold_outfit_ms[item] = pd.Series(self.threshold_outfit_ms[item])
        self.threshold_outfit_ms = pd.concat(self.threshold_outfit_ms.values(), keys=self.dataframe.columns)

        self.threshold_infit_ms = {item: {threshold + 1:
                                          (dich_residuals[item][threshold + 1] ** 2).sum().sum() /
                                          dich_thresh_var[item][threshold + 1].sum().sum()
                                          for threshold in range(self.max_score_vector[item])}
                                   for item in self.dataframe.columns}

        for item in self.dataframe.columns:
            self.threshold_infit_ms[item] = pd.Series(self.threshold_infit_ms[item])
        self.threshold_infit_ms = pd.concat(self.threshold_infit_ms.values(), keys=self.dataframe.columns)

        threshold_outfit_q = {item: {threshold + 1:
                                     (((dich_thresh_kur[item][threshold + 1] /
                                        (dich_thresh_var[item][threshold + 1] ** 2)) /
                                       (dich_thresh_count[item][threshold + 1] ** 2)).sum().sum() -
                                      (1 / dich_thresh_count[item][threshold + 1]))
                                     for threshold in range(self.max_score_vector[item])}
                              for item in self.dataframe.columns}

        for item in self.dataframe.columns:
            threshold_outfit_q[item] = pd.Series(threshold_outfit_q[item])
        threshold_outfit_q = pd.concat(threshold_outfit_q.values(), keys=self.dataframe.columns)
        threshold_outfit_q = np.sqrt(threshold_outfit_q)

        self.threshold_outfit_zstd = ((np.cbrt(self.threshold_outfit_ms) - 1) *
                                      (3 / threshold_outfit_q) +
                                      (threshold_outfit_q / 3))

        threshold_infit_q = {item: {threshold + 1: ((dich_thresh_kur[item][threshold + 1] -
                                                      dich_thresh_var[item][threshold + 1] ** 2).sum().sum() /
                                                    (dich_thresh_var[item][threshold + 1].sum().sum() ** 2))
                                    for threshold in range(self.max_score_vector[item])}
                             for item in self.dataframe.columns}

        for item in self.dataframe.columns:
            threshold_infit_q[item] = pd.Series(threshold_infit_q[item])
        threshold_infit_q = pd.concat(threshold_infit_q.values(), keys=self.dataframe.columns)
        threshold_infit_q = np.sqrt(threshold_infit_q)

        self.threshold_infit_zstd = ((np.cbrt(self.threshold_infit_ms) - 1) *
                                     (3 / threshold_infit_q) +
                                     (threshold_infit_q / 3))

        abil_deviation = self.person_abilities.copy() - self.person_abilities.mean()

        point_measure_dict = {item: {threshold + 1:
                                     dich_thresh[item][threshold + 1].copy()
                                     for threshold in range(self.max_score_vector[item])}
                              for item in self.dataframe.columns}

        for item in self.dataframe.columns:
            for threshold in range(self.max_score_vector[item]):
                point_measure_dict[item][threshold + 1] -= point_measure_dict[item][threshold + 1].mean()

        point_measure_nums = {item: {threshold + 1:
                                     (point_measure_dict[item][threshold + 1] *
                                      abil_deviation).sum()
                                     for threshold in range(self.max_score_vector[item])}
                              for item in self.dataframe.columns}

        for item in self.dataframe.columns:
            point_measure_nums[item] = pd.Series(point_measure_nums[item])
        point_measure_nums = pd.concat(point_measure_nums.values(), keys = self.dataframe.columns)

        point_measure_dens = {item: {threshold + 1:
                                     (point_measure_dict[item][threshold + 1] ** 2).sum() *
                                     (abil_deviation ** 2).sum()
                                     for threshold in range(self.max_score_vector[item])}
                              for item in self.dataframe.columns}

        for item in self.dataframe.columns:
            point_measure_dens[item] = pd.Series(point_measure_dens[item])
        point_measure_dens = pd.concat(point_measure_dens.values(), keys = self.dataframe.columns)
        point_measure_dens = np.sqrt(point_measure_dens)

        self.threshold_point_measure = point_measure_nums / point_measure_dens

        threshold_exp_pm_dict = {item: {threshold + 1:
                                        dich_thresh_exp[item][threshold + 1] -
                                        (dich_thresh_exp[item][threshold + 1].sum() /
                                         dich_thresh_exp[item][threshold + 1].count())
                                        for threshold in range(self.max_score_vector[item])}
                              for item in self.dataframe.columns}

        threshold_exp_pm_num = {item: {threshold + 1:
                                       (threshold_exp_pm_dict[item][threshold + 1] *
                                        abil_deviation).sum()
                                for threshold in range(self.max_score_vector[item])}
                              for item in self.dataframe.columns}

        for item in self.dataframe.columns:
            threshold_exp_pm_num[item] = pd.Series(threshold_exp_pm_num[item])
        threshold_exp_pm_num = pd.concat(threshold_exp_pm_num.values(), keys=self.dataframe.columns)

        threshold_exp_pm_den = {item: {threshold + 1:
                                       ((threshold_exp_pm_dict[item][threshold + 1] ** 2) +
                                        dich_thresh_var[item][threshold + 1]).sum()
                                       for threshold in range(self.max_score_vector[item])}
                              for item in self.dataframe.columns}

        for item in self.dataframe.columns:
            threshold_exp_pm_den[item] = pd.Series(threshold_exp_pm_den[item])
        threshold_exp_pm_den = pd.concat(threshold_exp_pm_den.values(), keys=self.dataframe.columns)

        threshold_exp_pm_den *= (abil_deviation ** 2).sum()
        threshold_exp_pm_den = np.sqrt(threshold_exp_pm_den)

        self.threshold_exp_point_measure = threshold_exp_pm_num / threshold_exp_pm_den

        self.threshold_rmsr = {item: {threshold + 1:
                                      (dich_residuals[item][threshold + 1] ** 2).sum() /
                                      dich_residuals[item][threshold + 1].count()
                                      for threshold in range(self.max_score_vector[item])}
                              for item in self.dataframe.columns}

        for item in self.dataframe.columns:
            self.threshold_rmsr[item] = pd.Series(self.threshold_rmsr[item])
        self.threshold_rmsr = pd.concat(self.threshold_rmsr.values(), keys=self.dataframe.columns)

        self.threshold_rmsr = np.sqrt(self.threshold_rmsr)

        differences = {item: {threshold + 1: pd.DataFrame()
                              for threshold in range(self.max_score_vector[item])}
                       for item in self.dataframe.columns}

        for item in self.dataframe.columns:
            for threshold in range(self.max_score_vector[item]):
                differences[item][threshold + 1] = (self.person_abilities -
                                                    self.thresholds_uncentred[item][threshold])

        nums = {item: {threshold + 1:
                       (differences[item][threshold + 1] *
                        dich_residuals[item][threshold + 1]).sum()
                        for threshold in range(self.max_score_vector[item])}
                for item in self.dataframe.columns}

        for item in self.dataframe.columns:
            nums[item] = pd.Series(nums[item])
        nums = pd.concat(nums.values(), keys=self.dataframe.columns)

        dens = {item: {threshold + 1:
                       (dich_thresh_var[item][threshold + 1] *
                        (differences[item][threshold + 1] ** 2)).sum()
                       for threshold in range(self.max_score_vector[item])}
                for item in self.dataframe.columns}

        for item in self.dataframe.columns:
            dens[item] = pd.Series(dens[item])
        dens = pd.concat(dens.values(), keys=self.dataframe.columns)

        self.threshold_discrimination = 1 + nums / dens

        '''
        Person fit statistics
        '''

        self.csem_vector = 1 / np.sqrt(self.info_df.sum(axis=1))
        self.rsem_vector = np.sqrt((self.residual_df ** 2).sum(axis=1)) / self.info_df.sum(axis=1)

        self.person_outfit_ms = (self.std_residual_df ** 2).mean(axis=1)
        self.person_outfit_ms.name = 'Outfit MS'
        self.person_infit_ms = (self.residual_df ** 2).sum(axis=1) / self.info_df.sum(axis=1)
        self.person_infit_ms.name = 'Intfit MS'

        base_df = self.kurtosis_df / (self.info_df ** 2)
        for column in self.dataframe.columns:
            base_df[column] /= (person_count ** 2)
        person_outfit_q = base_df.sum(axis=1) -  (1 / person_count)
        person_outfit_q = np.sqrt(person_outfit_q)
        self.person_outfit_zstd = (np.cbrt(self.person_outfit_ms) - 1) * (3 / person_outfit_q) + (person_outfit_q / 3)
        self.person_outfit_zstd.name = 'Outfit Z'

        person_infit_q = (self.kurtosis_df - self.info_df ** 2).sum(axis=1) / (self.info_df.sum(axis=1) ** 2)
        person_infit_q = np.sqrt(person_infit_q)
        self.person_infit_zstd = (np.cbrt(self.person_infit_ms) - 1) * (3 / person_infit_q) + (person_infit_q / 3)
        self.person_infit_zstd.name = 'Infit Z'

        '''
        Test-level fit statistics
        '''

        self.threshold_list = itertools.chain.from_iterable(self.thresholds_uncentred.values())
        self.threshold_list = np.array(list(self.threshold_list))
        self.threshold_se_list = itertools.chain.from_iterable(self.threshold_se.values())
        self.threshold_se_list = np.array(list(self.threshold_se_list))

        self.isi_central = np.sqrt(self.central_diffs.var() / (self.central_se ** 2).mean() - 1)
        self.item_strata = (4 * self.isi_central + 1) / 3
        self.item_reliability = self.isi_central ** 2 / (1 + self.isi_central ** 2)

        self.isi_thresholds = np.sqrt(self.threshold_list.var() / (self.threshold_se_list ** 2).mean() - 1)
        self.threshold_strata = (4 * self.isi_thresholds + 1) / 3
        self.threshold_reliability = self.isi_thresholds ** 2 / (1 + self.isi_thresholds ** 2)

        self.psi = (np.sqrt(np.var(self.person_abilities) - (self.rsem_vector ** 2).mean()) /
                     np.sqrt((self.rsem_vector ** 2).mean()))
        self.person_strata = (4 * self.psi + 1) / 3
        self.person_reliability = (self.psi ** 2) / (1 + (self.psi ** 2))

    def res_corr_analysis(self,
                          warm_corr=True,
                          tolerance=0.0000001,
                          max_iters=100,
                          ext_score_adjustment=0.5,
                          constant=0.1,
                          method='cos',
                          matrix_power=3,
                          log_lik_tol=0.000001,
                          no_of_samples=100,
                          interval=None):

        '''
        Analysis of correlations of standardised residuals for violations of local item interdependence
        and unidimensionality
        '''

        if hasattr(self, 'std_residual_df') == False:
            self.fit_statistics(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                ext_score_adjustment=ext_score_adjustment, constant=constant, method=method,
                                matrix_power=matrix_power, log_lik_tol=log_lik_tol, no_of_samples=no_of_samples,
                                interval=interval)

        self.residual_correlations = self.std_residual_df.corr(numeric_only=False)

        pca = PCA()
        try:
            pca.fit(self.std_residual_df.corr())

            self.eigenvectors = pd.DataFrame(pca.components_)
            self.eigenvectors.columns = [f'Eigenvector {pc + 1}' for pc in range(self.no_of_items)]

            self.eigenvalues = pca.explained_variance_
            self.eigenvalues = pd.DataFrame(self.eigenvalues)
            self.eigenvalues.index = [f'PC {pc + 1}' for pc in range(self.no_of_items)]
            self.eigenvalues.columns = ['Eigenvalue']

            self.variance_explained = pd.DataFrame(pca.explained_variance_ratio_)
            self.variance_explained.index = [f'PC {pc + 1}' for pc in range(self.no_of_items)]
            self.variance_explained.columns = ['Variance explained']

            self.loadings = self.eigenvectors.T * np.sqrt(pca.explained_variance_)
            self.loadings = pd.DataFrame(self.loadings)
            self.loadings.columns = [f'PC {pc + 1}' for pc in range(self.no_of_items)]
            self.loadings.index = [item for item in self.dataframe.columns]

        except:
            self.pca_fail = True
            print('PCA of residuals failed')

            self.eigenvectors = None
            self.eigenvalues = None
            self.variance_explained = None
            self.loadings = None

    def item_stats_df(self,
                      full=False,
                      zstd=False,
                      point_measure_corr=False,
                      dp=3,
                      warm_corr=True,
                      tolerance=0.0000001,
                      max_iters=100,
                      ext_score_adjustment=0.5,
                      method='cos',
                      constant=0.1,
                      no_of_samples=100,
                      interval=None):

        if full:
            zstd = True
            point_measure_corr = True

            if interval is None:
                interval = 0.95
        
        if ((hasattr(self, 'threshold_low') == False) or
            (self.threshold_low is None) and (interval is not None)):
            self.std_errors(interval=interval, no_of_samples=no_of_samples, constant=constant, method=method)

        if hasattr(self, 'item_infit_ms') == False:
            self.fit_statistics(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                ext_score_adjustment=ext_score_adjustment, method=method,
                                constant=constant, no_of_samples=no_of_samples, interval=interval)

        self.item_stats = pd.DataFrame()

        self.item_stats['Estimate'] = self.central_diffs.round(dp)
        self.item_stats['SE'] = self.central_se.to_numpy().round(dp)

        if interval is not None:
            self.item_stats[f'{round((1 - interval) * 50, 1)}%'] = self.central_low.to_numpy().round(dp)
            self.item_stats[f'{round((1 + interval) * 50, 1)}%'] = self.central_high.to_numpy().round(dp)

        self.item_stats['Count'] = self.response_counts.to_numpy().astype(int)
        self.item_stats['Facility'] = self.item_facilities.to_numpy().round(dp)

        self.item_stats['Infit MS'] = self.item_infit_ms.to_numpy().round(dp)
        if zstd:
            self.item_stats['Infit Z'] = self.item_infit_zstd.to_numpy().round(dp)

        self.item_stats['Outfit MS'] = self.item_outfit_ms.to_numpy().round(dp)
        if zstd:
            self.item_stats['Outfit Z'] = self.item_outfit_zstd.to_numpy().round(dp)

        if point_measure_corr:
            self.item_stats['PM corr'] = self.point_measure.to_numpy().round(dp)
            self.item_stats['Exp PM corr'] = self.exp_point_measure.to_numpy().round(dp)

        self.item_stats.index = self.dataframe.columns

    def threshold_stats_df(self,
                           full=False,
                           disc=False,
                           point_measure_corr=False,
                           dp=3,
                           warm_corr=True,
                           tolerance=0.0000001,
                           max_iters=100,
                           ext_score_adjustment=0.5,
                           method='cos',
                           constant=0.1,
                           no_of_samples=100,
                           interval=None):

        if full:
            disc = True
            point_measure_corr = True

            if interval is None:
                interval = 0.95

        if hasattr(self, 'threshold_infit_ms') == False:
            self.fit_statistics(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                ext_score_adjustment=ext_score_adjustment, method=method,
                                constant=constant, no_of_samples=no_of_samples, interval=interval)

        estimate_array = np.array([threshold for item in self.dataframe.columns
        							for threshold in self.thresholds_uncentred[item]])
        se_array = np.array([se for item in self.dataframe.columns
        					 for se in self.threshold_se[item]])
        low_array = np.array([low for item in self.dataframe.columns
        					  for low in self.threshold_low[item]])
        high_array = np.array([high for item in self.dataframe.columns
        					   for high in self.threshold_high[item]])

        self.threshold_stats_uncentred = pd.DataFrame()

        self.threshold_stats_uncentred['Estimate'] = estimate_array.round(dp)
        self.threshold_stats_uncentred['SE'] = se_array.round(dp)
        if interval is not None:
            self.threshold_stats_uncentred[f'{round((1 - interval) * 50, 1)}%'] = low_array.round(dp)
            self.threshold_stats_uncentred[f'{round((1 + interval) * 50, 1)}%'] = high_array.round(dp)

        self.threshold_stats_uncentred['Infit MS'] = self.threshold_infit_ms.to_numpy().round(dp)
        self.threshold_stats_uncentred['Infit Z'] = self.threshold_infit_zstd.to_numpy().round(dp)
        self.threshold_stats_uncentred['Outfit MS'] = self.threshold_outfit_ms.to_numpy().round(dp)
        self.threshold_stats_uncentred['Outfit Z'] = self.threshold_outfit_zstd.to_numpy().round(dp)

        if disc:
            self.threshold_stats_uncentred['Discrim'] = self.threshold_discrimination.to_numpy().round(dp)

        if point_measure_corr:
            self.threshold_stats_uncentred['PM corr'] = self.threshold_point_measure.to_numpy().round(dp)
            self.threshold_stats_uncentred['Exp PM corr'] = self.threshold_exp_point_measure.to_numpy().round(dp)

        self.threshold_stats_uncentred.index = self.threshold_infit_ms.index

        self.threshold_stats_centred = self.threshold_stats_uncentred.copy()

        central_array = np.array([self.central_diffs[item]
                                  for item in self.dataframe.columns
                                  for threshold in range(self.max_score_vector[item])])

        self.threshold_stats_centred['Estimate'] -= central_array.round(dp)
        self.threshold_stats_centred[f'{round((1 - interval) * 50, 1)}%'] -= central_array.round(dp)
        self.threshold_stats_centred[f'{round((1 + interval) * 50, 1)}%'] -= central_array.round(dp)

    def person_stats_df(self,
                        full=False,
                        rsem=False,
                        dp=3,
                        warm_corr=True,
                        tolerance=0.0000001,
                        max_iters=100,
                        ext_score_adjustment=0.5,
                        method='cos',
                        constant=0.1):

        '''
        Produces a person stats dataframe with raw score, ability estimate,
        CSEM and RSEM for each person.
        '''

        if hasattr(self, 'person_infit_ms') == False:
            self.fit_statistics(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                ext_score_adjustment=ext_score_adjustment, method=method, constant=constant)

        if full:
            rsem = True

        person_stats_df = pd.DataFrame()
        person_stats_df.index = self.dataframe.index

        person_stats_df['Estimate'] = self.person_abilities.to_numpy().round(dp)

        person_stats_df['CSEM'] = self.csem_vector.round(dp)
        if rsem:
            person_stats_df['RSEM'] = self.rsem_vector.round(dp)

        person_stats_df['Score'] = self.dataframe.sum(axis=1).astype(int)

        max_score_matrix = (self.dataframe == self.dataframe).astype(int)
        max_score_matrix = max_score_matrix.mul(self.max_score_vector, axis=1)
        person_stats_df['Max score'] = max_score_matrix.sum(axis=1).astype(int)

        person_stats_df['p'] =  (person_stats_df['Score'] / person_stats_df['Max score']).round(dp)

        person_stats_df['Infit MS'] = [np.nan for person in self.dataframe.index]
        person_stats_df['Infit Z'] = [np.nan for person in self.dataframe.index]
        person_stats_df['Outfit MS'] = [np.nan for person in self.dataframe.index]
        person_stats_df['Outfit MS'] = [np.nan for person in self.dataframe.index]
        person_stats_df['Outfit Z'] = [np.nan for person in self.dataframe.index]

        person_stats_df['Infit MS'].update(self.person_infit_ms.round(dp))
        person_stats_df['Infit Z'].update(self.person_infit_zstd.round(dp))
        person_stats_df['Outfit MS'].update(self.person_outfit_ms.round(dp))
        person_stats_df['Outfit Z'].update(self.person_outfit_zstd.round(dp))

        self.person_stats = person_stats_df

    def test_stats_df(self,
                      dp=3,
                      warm_corr=True,
                      tolerance=0.0000001,
                      max_iters=100,
                      ext_score_adjustment=0.5,
                      method='cos',
                      constant=0.1):

        if hasattr(self, 'psi') == False:
            self.fit_statistics(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                ext_score_adjustment=ext_score_adjustment, method=method,
                                constant=constant)

        self.test_stats = pd.DataFrame()

        self.test_stats['Items'] = [self.central_diffs.mean(),
                                    self.central_diffs.std(),
                                    self.isi_central,
                                    self.item_strata,
                                    self.item_reliability]

        self.test_stats['Thresholds'] = [self.threshold_list.mean(),
                                         self.threshold_list.std(),
                                         self.isi_thresholds,
                                         self.threshold_strata,
                                         self.threshold_reliability]

        self.test_stats['Persons'] = [self.person_abilities.mean(),
                                      self.person_abilities.std(),
                                      self.psi,
                                      self.person_strata,
                                      self.person_reliability]

        self.test_stats.index = ['Mean', 'SD', 'Separation ratio', 'Strata', 'Reliability']
        self.test_stats = round(self.test_stats, dp)

    def save_stats(self,
                   filename,
                   format='csv',
                   dp=3,
                   warm_corr=True,
                   tolerance=0.0000001,
                   max_iters=100,
                   ext_score_adjustment=0.5,
                   method='cos',
                   constant=0.1,
                   no_of_samples=100,
                   interval=None):

        if hasattr(self, 'item_stats') == False:
            self.item_stats_df(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                               ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                               no_of_samples=no_of_samples, interval=interval)

        if hasattr(self, 'threshold_stats') == False:
            self.threshold_stats_df(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                    ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                    no_of_samples=no_of_samples, interval=interval)

        if hasattr(self, 'person_stats') == False:
            self.person_stats_df(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                 ext_score_adjustment=ext_score_adjustment, method=method, constant=constant)

        if hasattr(self, 'test_stats') == False:
            self.test_stats_df(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                               ext_score_adjustment=ext_score_adjustment, method=method, constant=constant)

        if format == 'xlsx':

            if filename[-5:] != '.xlsx':
                filename += '.xlsx'

            writer = pd.ExcelWriter(filename, engine='xlsxwriter')

            self.item_stats.to_excel(writer, sheet_name='Item statistics')
            self.threshold_stats_uncentred.to_excel(writer, sheet_name='Threshold statistics uncentred')
            self.threshold_stats_centred.to_excel(writer, sheet_name='Threshold statistics centred')
            self.person_stats.to_excel(writer, sheet_name='Person statistics')
            self.test_stats.to_excel(writer, sheet_name='Test statistics')

            writer.save()

        else:
            if filename[-4:] == '.csv':
                filename = filename[:-4]

            self.item_stats.to_csv(f'{filename}_item_stats.csv')
            self.threshold_stats_uncentred.to_csv(f'{filename}_threshold_stats_uncentred.csv')
            self.threshold_stats_centred.to_csv(f'{filename}_threshold_stats_centred.csv')
            self.person_stats.to_csv(f'{filename}_person_stats.csv')
            self.test_stats.to_csv(f'{filename}_test_stats.csv')

    def save_residuals(self,
                       filename,
                       format='csv',
                       single=True,
                       dp=3,
                       warm_corr=True,
                       tolerance=0.0000001,
                       max_iters=100,
                       ext_score_adjustment=0.5,
                       method='cos',
                       constant=0.1):

        if hasattr(self, 'eigenvectors') == False:
            self.fit_statistics(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                ext_score_adjustment=ext_score_adjustment, method=method, constant=constant)

        if single:
            if format == 'xlsxl':

                if filename[-5:] != '.xlsx':
                    filename += '.xlsx'

                writer = pd.ExcelWriter(filename, engine='xlsxwriter')
                row = 0

                self.eigenvectors.round(dp).to_excel(writer, sheet_name='Item residual analysis',
                                                     startrow=row, startcol=0)
                row += (self.eigenvectors.shape[0] + 2)

                self.eigenvalues.round(dp).to_excel(writer, sheet_name='Item residual analysis',
                                                    startrow=row, startcol=0)
                row += (self.eigenvalues.shape[0] + 2)

                self.variance_explained.round(dp).to_excel(writer, sheet_name='Item residual analysis',
                                                           startrow=row, startcol=0)
                row += (self.variance_explained.shape[0] + 2)

                self.loadings.round(dp).to_excel(writer, sheet_name='Item residual analysis',
                                                 startrow=row, startcol=0)

                writer.save()

            else:
                if filename[-4:] != '.csv':
                    filename += '.csv'

                with open(filename, 'a') as f:
                    self.eigenvectors.round(dp).to_csv(f)
                    f.write("\n")
                    self.eigenvalues.round(dp).to_csv(f)
                    f.write("\n")
                    self.variance_explained.round(dp).to_csv(f)
                    f.write("\n")
                    self.loadings.round(dp).to_csv(f)

        else:
            if format == 'xlsx':

                if filename[-5:] != '.xlsx':
                    filename += '.xlsx'

                writer = pd.ExcelWriter(filename, engine='xlsxwriter')

                self.eigenvectors.round(dp).to_excel(writer, sheet_name='Eigenvectors')
                self.eigenvalues.round(dp).to_excel(writer, sheet_name='Eigenvalues')
                self.variance_explained.round(dp).to_excel(writer, sheet_name='Variance explained')
                self.loadings.round(dp).to_excel(writer, sheet_name='Principal Component loadings')

                writer.save()

            else:
                if filename[-4:] == '.csv':
                    filename = filename[:-4]

                self.eigenvectors.round(dp).to_csv(f'{filename}_eigenvectors.csv')
                self.eigenvalues.round(dp).to_csv(f'{filename}_eigenvalues.csv')
                self.variance_explained.round(dp).to_csv(f'{filename}_variance_explained.csv')
                self.loadings.round(dp).to_csv(f'{filename}_principal_component_loadings.csv')

    def class_intervals(self,
                        items=None,
                        no_of_classes=5):

        class_groups = [f'class_{class_no + 1}' for class_no in range(no_of_classes)]

        df = self.dataframe

        if items is not None:
            df = df[items]
            abils = self.person_abilities[df == df]

        else:
            abils = self.person_abilities[~df.isnull().all(axis=1)]

        df = df.dropna(how='all')

        quantiles = (abils.quantile([(i + 1) / no_of_classes
                                     for i in range(no_of_classes - 1)]))

        mask_dict = {}
        mask_dict['class_1'] = (abils < quantiles.values[0])
        mask_dict[f'class_{no_of_classes}'] = (abils >= quantiles.values[no_of_classes - 2])
        for class_no in range(no_of_classes - 2):
            mask_dict[f'class_{class_no + 2}'] = ((abils >= quantiles.values[class_no]) &
                                                  (abils < quantiles.values[class_no + 1]))

        class_sizes = {class_group: sum(mask_dict[class_group])
                       for class_group in class_groups}
        class_sizes = pd.Series(class_sizes)

        response_classes = {class_group: df.index[mask_dict[class_group]]
                            for class_group in class_groups}

        mean_abilities = {class_group: abils[mask_dict[class_group]].mean()
                          for class_group in class_groups}
        mean_abilities = pd.Series(mean_abilities)

        obs = {class_group: df[mask_dict[class_group]].mean().sum()
               for class_group in class_groups}

        for class_group in class_groups:
            obs[class_group] = pd.Series(obs[class_group])

        obs = pd.concat(obs, keys=obs_means.keys())

        class_abilities = {class_group: abils[mask_dict[class_group]]
                           for class_group in class_groups}
        class_abilities = pd.concat(class_abilities, keys=class_abilities.keys())

        return class_sizes, response_classes, class_abilities, mean_abilities, obs

    def class_intervals_cats(self,
                        	 item,
                        	 no_of_classes=5):

        class_groups = [f'class_{class_no + 1}' for class_no in range(no_of_classes)]

        df = self.dataframe[item]
        abils = self.person_abilities[df == df]
        df = df.dropna(how='all')

        quantiles = (abils.quantile([(i + 1) / no_of_classes
                                     for i in range(no_of_classes - 1)]))

        mask_dict = {}
        mask_dict['class_1'] = (abils < quantiles.values[0])
        mask_dict[f'class_{no_of_classes}'] = (abils >= quantiles.values[no_of_classes - 2])
        for class_no in range(no_of_classes - 2):
            mask_dict[f'class_{class_no + 2}'] = ((abils >= quantiles.values[class_no]) &
                                                  (abils < quantiles.values[class_no + 1]))

        class_sizes = {class_group: sum(mask_dict[class_group]) for class_group in class_groups}
        class_sizes = pd.Series(class_sizes)

        response_classes = {class_group: df.index[mask_dict[class_group]]
                            for class_group in class_groups}

        mean_abilities = {class_group: abils[mask_dict[class_group]].mean()
                          for class_group in class_groups}
        mean_abilities = pd.Series(mean_abilities)

        obs_props = {class_group: np.array([(df[mask_dict[class_group]] == cat).sum()
                                            for cat in range(self.max_score_vector[item] + 1)])
                     for class_group in class_groups}

        for class_group in class_groups:
            obs_props[class_group] = obs_props[class_group] / obs_props[class_group].sum()

        obs_props = pd.DataFrame(obs_props).to_numpy().T

        class_abilities = {class_group: abils[mask_dict[class_group]]
                           for class_group in class_groups}
        class_abilities = pd.concat(class_abilities, keys=class_abilities.keys())

        return class_sizes, response_classes, class_abilities, mean_abilities, obs_props

    def class_intervals_thresholds(self,
                                   item,
                                   no_of_classes=5):

        if hasattr(self, 'person_abilities') == False:
            self.person_abils(warm_corr=False)

        class_groups = [f'class_{class_no + 1}' for class_no in range(no_of_classes)]

        df = self.dataframe[item]

        abils = self.person_abilities

        def mask_dictionary(abils):

            quantiles = (abils.quantile([(i + 1) / no_of_classes
                                         for i in range(no_of_classes - 1)]))

            mask_dict = {}

            mask_dict['class_1'] = (abils < quantiles.values[0])
            mask_dict[f'class_{no_of_classes}'] = (abils >= quantiles.values[no_of_classes - 2])
            for class_group in range(no_of_classes - 2):
                mask_dict[f'class_{class_group + 2}'] = ((abils >= quantiles.values[class_group]) &
                                                         (abils < quantiles.values[class_group + 1]))

            return mask_dict

        mean_abilities = []
        obs_props = []
        class_sizes = []
        response_classes = []

        for threshold in range(self.max_score_vector[item]):

            cond_df_mask = df.isin([threshold, threshold + 1])
            cond_df = df[cond_df_mask]
            adj_abils = abils[cond_df_mask]

            adj_abil_score_df = pd.DataFrame()
            adj_abil_score_df['abil'] = adj_abils
            adj_abil_score_df['score'] = cond_df
            adj_abil_score_df.index = cond_df.index

            masks = mask_dictionary(adj_abils)

            cond_classes = {class_group: adj_abil_score_df[masks[class_group]]
                            for class_group in class_groups}

            mean_abilities.append([cond_classes[class_group]['abil'].mean()
                                   for class_group in class_groups])
            obs_props.append([(cond_classes[class_group]['score'] - threshold).mean()
                              for class_group in class_groups])

            sizes = {class_group: sum(masks[class_group]) for class_group in class_groups}
            class_sizes.append(pd.Series(sizes))

            response_classes.append({class_group: cond_df.index[masks[class_group]]
                                     for class_group in class_groups})

        mean_abilities = np.array(mean_abilities).T
        obs_props = np.array(obs_props).T

        class_abilities = {class_group: adj_abils[masks[class_group]]
                           for class_group in class_groups}
        class_abilities = pd.concat(class_abilities, keys=class_abilities.keys())

        return class_sizes, response_classes, class_abilities, mean_abilities, obs_props

    '''
    Plots
    '''

    def plot_data(self,
                  x_data,
                  y_data,
                  x_min=-10,
                  x_max=10,
                  y_max=0,
                  item=None,
                  obs=None,
                  x_obs_data=np.array([]),
                  y_obs_data=np.array([]),
                  thresh_lines=False,
                  central_diff=False,
                  score_lines_item=[None, []],
                  score_lines_test=[],
                  point_info_lines_item=[None, []],
                  point_info_lines_test=[],
                  point_csem_lines=[],
                  score_labels=False,
                  warm=True,
                  cat_highlight=None,
                  graph_title='',
                  y_label='',
                  plot_style='colorblind',
                  black=False,
                  figsize=(8, 6),
                  font='Times',
                  title_font_size=15,
                  axis_font_size=12,
                  labelsize=12,
                  graph_name='plot',
                  tex=True,
                  plot_density=300,
                  save_title='',
                  file_format='png'):

        '''
        Basic plotting function to be called when plotting specific functions
        of person ability for RSM.
        '''

        if tex:
            plt.rcParams["text.latex.preamble"].join([r"\usepackage{dashbox}", r"\setmainfont{xcolor}",])
        else:
            plt.rcParams["text.usetex"] = False

        plt.style.use('seaborn-v0_8-' + plot_style)

        graph, ax = plt.subplots(figsize=figsize)

        no_of_plots = y_data.shape[1]

        if black:
            for i in range(no_of_plots):
                ax.plot(x_data, y_data[:, i], '', label=i+1, color='black')

        else:
            for i in range(no_of_plots):
                ax.plot(x_data, y_data[:, i], '', label=i+1)

        if obs is not None:
            try:
                no_of_observed_cats = y_obs_data.shape[1]
                if isinstance(x_obs_data, pd.Series):
                    for j in range (no_of_observed_cats):
                        ax.plot(x_obs_data, y_obs_data[:, j], 'o')

                else:
                    no_of_observed_cats = y_obs_data.shape[1]
                    for j in range (no_of_observed_cats):
                        ax.plot(x_obs_data[:, j], y_obs_data[:, j], 'o')

            except:
                pass

        if item is not None:
            thresholds = {item: self.thresholds_uncentred[item]}

        else:
            thresholds = self.thresholds_uncentred

        if thresh_lines:
            for threshold in self.thresholds_uncentred[item]:
                plt.axvline(x = threshold, color = 'black', linestyle='--')

        if item is not None:
            if central_diff:
                plt.axvline(x = thresholds[item].mean(), color = 'darkred', linestyle='--')

        if score_lines_item[1] != []:

            if (all(x > 0 for x in score_lines_item[1]) &
                    all(x < self.max_score_vector[item] for x in score_lines_item[1])):

                abils_set = [self.score_abil(score, items=thresholds.keys(), warm_corr=False)
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

        if score_lines_test != []:

            if (all(x > 0 for x in score_lines_test) &
                    all(x < sum(self.max_score_vector) for x in score_lines_test)):

                abils_set = [self.score_abil(score, items=thresholds.keys(), warm_corr=False)
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

        if point_info_lines_item[1] != []:

            item = point_info_lines_item[0]

            info_set = [self.variance_uncentred(ability, self.thresholds_uncentred[item])
            			for ability in point_info_lines_item[1]]

            for abil, info in zip(point_info_lines_item[1], info_set):
                plt.vlines(x=abil, ymin=-100, ymax=info, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=info, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(x_min + (x_max - x_min) / 100, info + y_max / 50, str(round(info, 3)))

        if point_info_lines_test != []:

            info_set = [sum(self.variance_uncentred(ability, self.thresholds_uncentred[item])
            				for item in self.dataframe.columns)
                        for ability in point_info_lines_test]

            for abil, info in zip(point_info_lines_test, info_set):
                plt.vlines(x=abil, ymin=-100, ymax=info, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=info, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(x_min + (x_max - x_min) / 100, info + y_max / 50, str(round(info, 3)))

        if point_csem_lines != []:

            info_set = [sum(self.variance_uncentred(ability, self.thresholds_uncentred[item])
            				for item in self.dataframe.columns)
                        for ability in point_csem_lines]
            info_set = np.array(info_set)
            csem_set = 1 / np.sqrt(info_set)

            for abil, csem in zip(point_csem_lines, csem_set):
                plt.vlines(x=abil, ymin=-100, ymax=csem, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=csem, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(x_min + (x_max - x_min) / 100, csem + y_max / 50, str(round(csem, 3)))

        if item is not None:
            if cat_highlight in range(self.max_score_vector[item] + 1):

                if cat_highlight == 0:
                    plt.axvspan(-100, self.thresholds_uncentred[item][0],
                                facecolor='blue', alpha=0.2)

                elif cat_highlight == self.max_score_vector[item]:
                    plt.axvspan(self.thresholds_uncentred[item][cat_highlight - 1], 100,
                                facecolor='blue', alpha=0.2)

                else:
                    if (self.thresholds_uncentred[item][cat_highlight] >
                        self.thresholds_uncentred[item][cat_highlight - 1]):
                        plt.axvspan(self.thresholds_uncentred[item][cat_highlight - 1],
                                    self.thresholds_uncentred[item][cat_highlight],
                                    facecolor='blue', alpha=0.2)

        if y_max <= 0:
            y_max = y_data.max() * 1.1

        plt.xlim(x_min, x_max)
        plt.ylim(0, y_max)

        plt.xlabel('Ability', fontname=font, fontsize=axis_font_size, fontweight='bold', wrap=True)
        plt.ylabel(y_label, fontname=font, fontsize=axis_font_size, fontweight='bold', wrap=True)
        plt.title(graph_title, fontname=font, fontsize=title_font_size, fontweight='bold', wrap=True)

        plt.grid(True)

        plt.tick_params(axis="x", labelsize=labelsize)
        plt.tick_params(axis="y", labelsize=labelsize)

        if save_title != '':
            save_title = save_title.translate(str.maketrans('', '', string.punctuation))
            save_title = save_title.translate({32: 95})
            plt.savefig(f'{save_title}.{file_format}', dpi=plot_density)
            
        plt.close()

        return graph;

    def icc(self,
            item,
            obs=False,
            xmin=-10,
            xmax=10,
            no_of_classes=5,
            title=True,
            thresh_lines=False,
            central_diff=False,
            score_lines=[],
            score_labels=False,
            cat_highlight=None,
            plot_style='colorblind',
            black=False,
            font='Times',
            title_font_size=15,
            axis_font_size=12,
            labelsize=12,
            save_title='',
            use_save_title=False,
            file_format='png',
            dpi=300):

        '''
        Plots Item Characteristic Curves for PCM, with optional overplotting
        of observed data, threshold lines and expected score threshold lines.
        '''

        abilities = np.arange(-20, 20, 0.1)
        y = [self.exp_score_uncentred(ability, self.thresholds_uncentred[item])
             for ability in abilities]
        y = np.array(y).reshape([len(abilities), 1])

        if obs:
            if hasattr(self, 'person_abiliites') == False:
                self.person_abils(warm_corr=False)

            _, _, _, mean_abilities, obs_means = self.class_intervals(item=item, no_of_classes=no_of_classes)

            xobsdata = pd.Series(mean_abilities)
            yobsdata = obs_means
            yobsdata = np.array(yobsdata).reshape((-1, 1))

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        ylabel = 'Expected score'

        if title:
            if use_save_title:
                if save_title != '':
                    graphtitle = f'ICC: {save_title}'

            else:
                graphtitle = f'ICC: {item}'
                
        else:
            graphtitle = ''

        plot = self.plot_data(x_data=abilities, y_data=y, x_obs_data=xobsdata, y_obs_data=yobsdata, x_min=xmin,
                              x_max=xmax, y_max=self.max_score_vector[item], item=item, graph_title=graphtitle,
                              y_label=ylabel, obs=obs, thresh_lines=thresh_lines, central_diff=central_diff,
                              score_lines_item=[item, score_lines], score_labels=score_labels, plot_style=plot_style,
                              cat_highlight=cat_highlight, black=black, font=font, title_font_size=title_font_size,
                              axis_font_size=axis_font_size, labelsize=labelsize, save_title=save_title,
                              plot_density=dpi, file_format=file_format)
        
        return plot

    def crcs(self,
             item,
             obs=None,
             xmin=-10,
             xmax=10,
             no_of_classes=5,
             title=True,
             thresh_lines=False,
             central_diff=False,
             cat_highlight=None,
             plot_style='colorblind',
             black=False,
             font='Times',
             title_font_size=15,
             axis_font_size=12,
             labelsize=12,
             save_title='',
             file_format='png',
             dpi=300):

        '''
        Plots Category Response Curves for PCM, with optional overplotting
        of observed data and threshold lines.
        '''

        abilities = np.arange(-20, 20, 0.1)

        y = np.array([[self.cat_prob_uncentred(ability, category, self.thresholds_uncentred[item])
                       for category in range(self.max_score_vector[item] + 1)]
                      for ability in abilities])

        if obs is not None:
            if hasattr(self, 'person_abiliites') == False:
                self.person_abils(warm_corr=False)

            _, _, _, mean_abilities, obs_props = self.class_intervals_cats(item=item, no_of_classes=no_of_classes)

            xobsdata = mean_abilities
            yobsdata = obs_props

            if obs != 'all':
                if not all(cat in np.arange(self.max_score_vector[item] + 1) for cat in obs):
                    print("Invalid 'obs'. Valid values are 'None', 'all' and list of categories.")
                    return

                else:
                    yobsdata = yobsdata[:, obs]

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        if title:
            graphtitle = f'Category response curves for item {item}'
        else:
            graphtitle = ''

        ylabel = 'Probability'

        plot = self.plot_data(x_data=abilities, y_data=y, x_min=xmin, x_max=xmax, y_max=1, x_obs_data=xobsdata,
                              y_obs_data=yobsdata, item=item, graph_title=graphtitle, y_label=ylabel, obs=obs,
                              thresh_lines=thresh_lines, central_diff=central_diff, cat_highlight=cat_highlight,
                              plot_style=plot_style, black=black, font=font, title_font_size=title_font_size,
                              axis_font_size=axis_font_size, labelsize=labelsize, save_title=save_title,
                              plot_density=dpi, file_format=file_format)

        return plot

    def threshold_ccs(self,
                      item,
                      obs=None,
                      xmin=-10,
                      xmax=10,
                      no_of_classes=5,
                      title=True,
                      thresh_lines=False,
                      central_diff=False,
                      cat_highlight=None,
                      plot_style='colorblind',
                      black=False,
                      font='Times',
                      title_font_size=15,
                      axis_font_size=12,
                      labelsize=12,
                      save_title='',
                      file_format='png',
                      dpi=300):

        '''
        Plots Threshold Characteristic Curves for RSM, with optional
        overplotting of observed data and threshold lines.
        '''

        abilities = np.arange(-20, 20, 0.1)
        y = np.array([[1 / (1 + np.exp(threshold - ability))
                       for threshold in self.thresholds_uncentred[item]]
                      for ability in abilities])

        if obs is not None:
            if hasattr(self, 'person_abiliites') == False:
                self.person_abils(warm_corr=False)

            _, _, _, mean_abilities, obs_props = self.class_intervals_thresholds(item, no_of_classes=no_of_classes)

            xobsdata = mean_abilities
            yobsdata = obs_props

            if obs != 'all':
                if not all(cat in np.arange(self.max_score_vector[item]) + 1 for cat in obs):
                    print("Invalid 'obs'. Valid values are 'None', 'all' and list of categories.")
                    return

                else:
                    obs = [ob - 1 for ob in obs]
                    xobsdata = xobsdata[:, obs]
                    yobsdata = yobsdata[:, obs]

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        if title:
            graphtitle = f'Threshold characteristic curves for item {item}'
        else:
            graphtitle = ''

        ylabel = 'Probability'

        plot = self.plot_data(x_data=abilities, y_data=y, y_max=1, x_min=xmin, x_max=xmax, item=item,
                              x_obs_data=xobsdata, y_obs_data=yobsdata, graph_title=graphtitle, y_label=ylabel,
                              obs=obs, thresh_lines=thresh_lines, central_diff=central_diff,
                              cat_highlight=cat_highlight, plot_style=plot_style, black=black, font=font,
                              title_font_size=title_font_size, axis_font_size=axis_font_size, labelsize=labelsize,
                              save_title=save_title, plot_density=dpi, file_format=file_format)

        return plot

    def iic(self,
            item,
            xmin=-10,
            xmax=10,
            ymax=None,
            thresh_lines=False,
            central_diff=False,
            point_info_lines=[],
            point_info_labels=False,
            cat_highlight=None,
            title=True,
            plot_style='colorblind',
            black=False,
            font='Times',
            title_font_size=15,
            axis_font_size=12,
            labelsize=12,
            save_title='',
            file_format='png',
            dpi=300):

        '''
        Plots Item Information Curves.
        '''

        abilities = np.arange(-20, 20, 0.1)
        y = [self.variance_uncentred(ability, self.thresholds_uncentred[item])
             for ability in abilities]
        y = np.array(y).reshape(len(abilities), 1)

        if ymax is None:
            ymax = max(y) * 1.1

        if title:
            graphtitle = f'Item information curve for item {item}'
        else:
            graphtitle = ''

        ylabel = 'Fisher information'

        plot = self.plot_data(x_data=abilities, y_data=y, x_min=xmin, x_max=xmax, y_max=ymax, thresh_lines=thresh_lines,
                              item=item, central_diff=central_diff, point_info_lines_item=[item, point_info_lines],
                              score_labels=point_info_labels, cat_highlight=cat_highlight, graph_title=graphtitle,
                              y_label=ylabel, plot_style=plot_style, black=black, font=font,
                              title_font_size=title_font_size, axis_font_size=axis_font_size, labelsize=labelsize,
                              save_title=save_title, file_format=file_format, plot_density=dpi)

        return plot

    def tcc(self,
            obs=False,
            xmin=-10,
            xmax=10,
            no_of_classes=5,
            title=True,
            score_lines=[],
            score_labels=False,
            warm=True,
            plot_style='colorblind',
            black=False,
            font='Times',
            title_font_size=15,
            axis_font_size=12,
            labelsize=12,
            save_title='',
            file_format='png',
            dpi=300):

        '''
        Plots Test Characteristic Curve for PCM.
        '''

        abilities = np.arange(-20, 20, 0.1)
        y = [sum(self.exp_score_uncentred(ability, self.thresholds_uncentred[item])
                 for item in self.dataframe.columns)
             for ability in abilities]
        y = np.array(y).reshape(len(abilities), 1)

        if obs:
            if hasattr(self, 'person_abiliites') == False:
                self.person_abils(warm_corr=False)

            _, _, _, mean_abilities, obs_means = self.class_intervals(no_of_classes=no_of_classes)

            xobsdata = mean_abilities
            yobsdata = obs_means
            yobsdata = np.array(yobsdata).reshape(no_of_classes, 1)

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        if title:
            graphtitle = 'Test characteristic curve'
        else:
            graphtitle = ''

        ylabel = 'Expected score'

        if obs:
            graphname = f'TCC_observed_categories'
        else:
            graphname = f'ICC'

        plot = self.plot_data(x_data=abilities, y_data=y, x_obs_data=xobsdata, y_obs_data=yobsdata, x_min=xmin,
                              x_max=xmax, y_max=sum(self.max_score_vector), score_lines_test=score_lines,
                              score_labels=score_labels, warm=warm, graph_title=graphtitle, y_label=ylabel, obs=obs,
                              plot_style=plot_style, black=black, font=font, title_font_size=title_font_size,
                              axis_font_size=axis_font_size, labelsize=labelsize, save_title=save_title,
                              file_format=file_format, plot_density=dpi)

        return plot

    def test_info(self,
                  point_info_lines=[],
                  point_info_labels=False,
                  xmin=-10,
                  xmax=10,
                  ymax=None,
                  title=True,
                  plot_style='colorblind',
                  black=False,
                  font='Times',
                  title_font_size=15,
                  axis_font_size=12,
                  labelsize=12,
                  save_title='',
                  file_format='png',
                  dpi=300):

        '''
        Plots Test Information Curve for PCM.
        '''

        abilities = np.arange(-20, 20, 0.1)
        y = [sum(self.variance_uncentred(ability, self.thresholds_uncentred[item])
                 for item in self.dataframe.columns)
             for ability in abilities]
        y = np.array(y).reshape(len(abilities), 1)

        if ymax is None:
            ymax = max(y) * 1.1

        if title:
            graphtitle = 'Test information curve'
        else:
            graphtitle = ''

        ylabel = 'Fisher information'

        plot = self.plot_data(x_data=abilities, y_data=y, x_min=xmin, x_max=xmax, y_max=ymax, graph_title=graphtitle,
                              point_info_lines_test=point_info_lines, score_labels=point_info_labels, y_label=ylabel,
                              plot_style=plot_style, black=black, font=font, title_font_size=title_font_size,
                              axis_font_size=axis_font_size, labelsize=labelsize, save_title=save_title,
                              file_format=file_format, plot_density=dpi)

        return plot

    def test_csem(self,
                  point_csem_lines=[],
                  point_csem_labels=False,
                  xmin=-10,
                  xmax=10,
                  ymax=5,
                  title=True,
                  plot_style='colorblind',
                  black=False,
                  font='Times',
                  title_font_size=15,
                  axis_font_size=12,
                  labelsize=12,
                  save_title='',
                  file_format='png',
                  dpi=300):

        '''
        Plots Test Conditional Standard Error of Measurement Curve for PCM.
        '''

        abilities = np.arange(-20, 20, 0.1)
        y = np.array([sum(self.variance_uncentred(ability, self.thresholds_uncentred[item])
                 for item in self.dataframe.columns)
             for ability in abilities])
        y = 1 / np.sqrt(y)
        y = y.reshape(len(abilities), 1)

        if title:
            graphtitle = 'Test conditional SEM curve'
        else:
            graphtitle = ''

        ylabel = 'Conditional SEM'

        plot = self.plot_data(x_data=abilities, y_data=y, x_min=xmin, x_max=xmax, y_max=ymax, graph_title=graphtitle,
                              point_csem_lines=point_csem_lines, score_labels=point_csem_labels, y_label=ylabel,
                              plot_style=plot_style, black=black, font=font, title_font_size=title_font_size,
                              axis_font_size=axis_font_size, labelsize=labelsize, save_title=save_title,
                              file_format=file_format, plot_density=dpi)

        return plot

    def std_residuals_plot(self,
                           bin_width=0.5,
                           x_min=-6,
                           x_max=6,
                           normal=False,
                           title=True,
                           plot_style='colorblind',
                           font='Times',
                           title_font_size=15,
                           axis_font_size=12,
                           labelsize=12,
                           save_title='',
                           file_format='png',
                           tex=True,
                           plot_density=300):

        '''
        Plots histogram of standardised residuals for SLM, with optional overplotting of Standard Normal Distribution.
        '''

        std_residual_list = self.std_residual_df.unstack().dropna()

        plot = self.std_residuals_hist(std_residual_list, bin_width=bin_width, x_min=x_min, x_max=x_max, normal=normal,
                                       title=title, plot_style=plot_style, font=font, title_font_size=title_font_size,
                                       axis_font_size=axis_font_size, labelsize=labelsize, save_title=save_title,
                                       file_format=file_format, tex=tex, plot_density=plot_density)

        return plot

class RSM(Rasch):

    def __init__(self,
                 dataframe,
                 max_score=None,
                 extreme_persons=True,
                 no_of_classes=5):

        if max_score is None:
            self.max_score = int(np.nanmax(dataframe))

        else:
            self.max_score = max_score

        if extreme_persons:
            self.invalid_responses = dataframe[dataframe.isna().all(axis=1)]
            self.dataframe = dataframe[~dataframe.isna().all(axis=1)]

        else:
            scores = dataframe.sum(axis=1)
            max_scores = ((dataframe == dataframe) * self.max_score).sum(axis=1)

            self.invalid_responses = dataframe[(scores == 0) | (scores == max_scores)]
            self.dataframe = dataframe[(scores > 0) & (scores < max_scores)]
        
        self.no_of_items = self.dataframe.shape[1]
        self.items = self.dataframe.columns
        self.no_of_persons = self.dataframe.shape[0]
        self.persons = self.dataframe.index
        self.no_of_classes = no_of_classes

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

        cat_prob_nums = [exp(category * (ability - difficulty) -
                         sum(thresholds[:category + 1]))
                         for category in range(self.max_score + 1)]

        return cat_prob_nums[category] / sum(cat_prob_nums)

    def exp_score(self,
                  ability,
                  difficulty,
                  thresholds):

        '''
        Calculates expected score for given person ability,
        item difficulty and set of Rasch-Andrich thresholds.
        '''

        cat_prob_nums = [exp(category * (ability - difficulty) -
                         sum(thresholds[:category + 1]))
                         for category in range(self.max_score + 1)]

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

        cat_prob_nums = [exp(category * (ability - difficulty) -
                             sum(thresholds[:category + 1]))
                         for category in range(self.max_score + 1)]

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

        cat_prob_nums = [exp(category * (ability - difficulty) -
                         sum(thresholds[:category + 1]))
                         for category in range(self.max_score + 1)]

        expected = self.exp_score(ability,
                                  difficulty,
                                  thresholds)

        kurtosis = (sum(((category - expected) ** 4) * cat_prob
                        for category, cat_prob in enumerate(cat_prob_nums)) /
                    sum(cat_prob_nums))

        return kurtosis

    def _threshold_distance(self,
                            threshold,
                            difficulties,
                            constant=0.1):

        '''
        ** Private method **
        Estimates the distance between adjacent Rasch-Andrich thresholds
        for CPAT threshold estimation.
        '''

        df_array = np.array(self.dataframe)

        estimator = 0
        weights_sum = 0

        for item_1 in range(self.no_of_items):

            for item_2 in range(self.no_of_items):

                num = np.count_nonzero((df_array[:, item_1] == threshold) &
                                       (df_array[:, item_2] == threshold))

                den = np.count_nonzero((df_array[:, item_1] == threshold - 1) &
                                       (df_array[:, item_2] == threshold + 1))

                if num + den == 0:
                    pass

                else:
                    num += constant
                    den += constant

                    weight = hmean([num, den])

                    estimator += weight * (log(num) - log(den) +
                                           difficulties[item_1] - difficulties[item_2])
                    weights_sum += weight

        try:
            estimator /= weights_sum

        except:
            estimator = np.nan

        return estimator

    def threshold_set(self,
                      difficulties,
                      constant=0.1):

        '''
        Calculates set of Rasch-Andrich threshold estimates
        for CPAT threshold estimation.
        '''

        thresh_distances = [self._threshold_distance(threshold + 1, difficulties, constant)
                            for threshold in range(self.max_score - 1)]

        thresholds = [sum(thresh_distances[:threshold])
                      for threshold in range(self.max_score)]

        thresholds = np.array(thresholds)

        np.add(thresholds, -np.mean(thresholds), out = thresholds, casting = 'unsafe')

        thresholds = np.insert(thresholds, 0, 0)

        return thresholds

    def calibrate(self,
                  constant=0.1,
                  method='cos',
                  matrix_power=3,
                  log_lik_tol=0.000001):

        '''
        Creates set of weighted CPAT threshold estimates plus
        PAIR item difficulty estimation (cosine similarity).
        '''

        self.null_persons =  self.dataframe.index[self.dataframe.isnull().all(1)]
        self.dataframe = self.dataframe.drop(self.null_persons)
        self.no_of_persons = self.dataframe.shape[0]

        df_array = self.dataframe.to_numpy()

        matrix = [[np.count_nonzero((df_array[:, item_1]) ==
                                    (df_array[:, item_2] + 1))
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
                break

        self.diffs = self.priority_vector(mat, method=method, log_lik_tol=log_lik_tol)

        thresh_distances = [self._threshold_distance(threshold + 1, self.diffs, constant)
                            for threshold in range(self.max_score - 1)]

        thresholds = [sum(thresh_distances[:threshold])
                      for threshold in range(self.max_score)]

        thresholds = np.array(thresholds)
        np.add(thresholds, -np.mean(thresholds), out = thresholds, casting = 'unsafe')
        thresholds = np.insert(thresholds, 0, 0)

        self.thresholds = thresholds

    def std_errors(self,
                   interval=None,
                   no_of_samples=100,
                   constant=0.1,
                   method='cos',
                   matrix_power=3,
                   log_lik_tol=0.000001):

        '''
        Bootstraped standard error estimates for item and threshold estimates.
        '''

        samples = [RSM(self.dataframe.sample(frac=1, replace=True))
                   for sample in range(no_of_samples)]

        for sample in samples:
            sample.calibrate(constant=constant, method=method, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        item_ests = np.array([sample.diffs.values for sample in samples])
        threshold_ests = np.array([sample.thresholds for sample in samples])

        item_se = {item: se for item, se in zip(self.dataframe.columns,
                                                np.nanstd(item_ests, axis=0))}
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

        item_bootstrap = pd.DataFrame(item_ests)
        item_bootstrap.columns = self.dataframe.columns
        item_bootstrap.index = [f'Sample {i + 1}' for i in range (no_of_samples)]

        threshold_bootstrap = pd.DataFrame(threshold_ests)
        threshold_bootstrap.columns = [cat + 1 for cat in range(self.max_score + 1)]
        threshold_bootstrap.index = [f'Sample {i + 1}' for i in range (no_of_samples)]

        cat_width_bootstrap = pd.DataFrame(cat_widths)
        cat_width_bootstrap.columns = [cat + 1 for cat in range(self.max_score - 1)]
        cat_width_bootstrap.index = [f'Sample {i + 1}' for i in range (no_of_samples)]

        self.item_bootstrap = item_bootstrap
        self.item_se = item_se
        self.item_low = item_low
        self.item_high = item_high
        self.threshold_bootstrap = threshold_bootstrap
        self.threshold_se = threshold_se
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.cat_width_bootstrap = cat_width_bootstrap
        self.cat_width_se = cat_width_se
        self.cat_width_low = cat_width_low
        self.cat_width_high = cat_width_high

    def abil(self,
             person,
             items=None,
             warm_corr=True,
             tolerance=0.0000001,
             max_iters=100,
             ext_score_adjustment=0.5):

        '''
        Creates a raw score to ability estimate look-up table for a set
        of items using ML estimation (Newton-Raphson procedure) with
        optional Warm (1989) bias correction.
        '''

        if items is None:
            items = self.items
            difficulties = self.diffs

        else:
            difficulties = self.diffs.loc[items]

        person_data = self.dataframe.loc[person].to_numpy()
        person_filter = (person_data + 1) / (person_data + 1)
        score = np.nansum(person_data)

        ext_score = np.nansum(person_filter) * self.max_score

        if score == 0:
            score = ext_score_adjustment

        elif score == ext_score:
            score -= ext_score_adjustment

        try:
            estimate = (log(score) -
                        log(ext_score - score) +
                        statistics.mean(difficulties))
            change = 1
            iters = 0

            while (abs(change) > tolerance) & (iters <= max_iters):

                person_exp_list = [self.exp_score(estimate, difficulty, self.thresholds)
                                   for flag, difficulty in zip(person_filter, difficulties)
                                   if flag == flag]
                result = sum(person_exp_list)

                person_info_list = [self.variance(estimate, difficulty, self.thresholds)
                                    for flag, difficulty in zip(person_filter, difficulties)
                                    if flag == flag]
                info = sum(person_info_list)

                change = max(-1, min(1, (result - score) / info))
                estimate -= change
                iters += 1

            if warm_corr:
                estimate += self.warm(estimate, person_filter, difficulties)

            if iters >= max_iters:
                print('Maximum iterations reached before convergence.')

        except:
            estimate =np.nan

        return estimate

    def person_abils(self,
                     items=None,
                     warm_corr=True,
                     tolerance=0.0000001,
                     max_iters=100,
                     ext_score_adjustment=0.5):

        '''
        Creates raw score to ability estimate look-up table. Newton-Raphson ML
        estimation, includes optional Warm (1989) bias correction.
        '''

        if items is None:
            items = self.items

        estimates = {person: self.abil(person, items=items, warm_corr=warm_corr, tolerance=tolerance,
                                       max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)
                     for person in self.dataframe.index}

        self.person_abilities = pd.Series(estimates)

    def score_abil(self,
                   score,
                   items=None,
                   warm_corr=True,
                   tolerance=0.0000001,
                   max_iters=100,
                   ext_score_adjustment=0.5):

        '''
        Creates a raw score to ability estimate look-up table for a set
        of items using ML estimation (Newton-Raphson procedure) with
        optional Warm (1989) bias correction.
        '''

        if items is None:
            items = self.items
            difficulties = self.diffs

        else:
            difficulties = self.diffs.loc[items]

        person_filter = np.array([True for item in items])
        ext_score = len(items) * self.max_score

        if score == 0:
            score = ext_score_adjustment

        elif score == ext_score:
            score -= ext_score_adjustment

        estimate = (log(score) - log(ext_score - score) +
                    statistics.mean(difficulties))
        change = 1
        iters = 0

        while (abs(change) > tolerance) & (iters <= max_iters):

            person_exp_list = [self.exp_score(estimate, difficulty, self.thresholds)
                               for flag, difficulty in zip(person_filter, difficulties)
                               if flag == flag]
            result = sum(person_exp_list)

            person_info_list = [self.variance(estimate, difficulty, self.thresholds)
                                for flag, difficulty in zip(person_filter, difficulties)
                                if flag == flag]
            info = sum(person_info_list)

            change = max(-1, min(1, (result - score) / info))
            estimate -= change
            iters += 1

        if warm_corr:
            estimate += self.warm(estimate, person_filter, difficulties)

        if iters >= max_iters:
            print('Maximum iterations reached before convergence.')

        return estimate

    def abil_lookup_table(self,
                          items=None,
                          ext_scores=True,
                          warm_corr=True,
                          tolerance=0.0000001,
                          max_iters=100,
                          ext_score_adjustment=0.5):

        if items is None:
            items = self.items
            difficulties = self.diffs

        else:
            difficulties = self.diffs.loc[items]

        if ext_scores:
            score_range = range(len(items) * self.max_score + 1)

        else:
            score_range = range(1, len(items) * self.max_score)

        ext_score = len(difficulties) * self.max_score

        abil_table = {score: self.score_abil(score, items=items, warm_corr=warm_corr, tolerance=tolerance,
                                             max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)
                      for score in score_range}
        abil_table = pd.Series(abil_table)

        self.abil_table = abil_table

    def warm(self,
             ability,
             person_filter,
             difficulties):

        '''
        Warm's (1989) bias correction for ML ability estimates
        '''

        exp_scores = [self.exp_score(ability, difficulty, self.thresholds)
                      for flag, difficulty in zip(person_filter, difficulties)
                      if flag == flag]

        variances = [self.variance(ability, difficulty, self.thresholds)
                     for flag, difficulty in zip(person_filter, difficulties)
                     if flag == flag]

        part_1 = sum(sum((category ** 3) * self.cat_prob(ability, difficulty, category, self.thresholds)
                         for category in range(self.max_score + 1))
                     for flag, difficulty in zip(person_filter, difficulties)
                     if flag == flag)

        part_2 = 3 * sum((variance + (exp_score ** 2)) * exp_score
                         for variance, exp_score in zip(variances, exp_scores))

        part_3 = sum(2 * (exp_score ** 3) for exp_score in exp_scores)

        warm_correction = ((part_1 - part_2 + part_3) /
                           (2 * (sum(variances) ** 2)))

        return warm_correction

    def csem(self,
             person,
             abilities=None,
             items=None):

        '''
        Calculates conditional standard error of measurement for an ability.
        '''

        if items is None:
            items = self.items
            difficulties = self.diffs

        else:
            difficulties = self.diffs.loc[items]

        if abilities is None:
            abilities = self.person_abilities

        person_data = self.dataframe.loc[person, items].to_numpy()
        person_filter = (person_data + 1) / (person_data + 1)

        total_info = sum(self.variance(abilities[person], difficulty, self.thresholds)
                         for flag, difficulty in zip(person_filter, difficulties)
                         if flag == flag)

        cond_sem = 1 / np.sqrt(total_info)

        return cond_sem

    def category_counts_item(self,
                             item):

        if item in self.dataframe.columns:
            return self.dataframe.apply(pd.value_counts)[item].fillna(0).astype(int)

        else:
            print('Invalid item name')

    def category_counts_df(self):

        category_counts_df = pd.DataFrame(0, index=self.dataframe.columns, columns=np.arange(self.max_score + 1))

        for item in self.dataframe.columns:
            for score, count in self.category_counts_item(item).items():
                category_counts_df.loc[item].iloc[int(score)] = count

        category_counts_df['Total'] = self.dataframe.count()
        category_counts_df['Missing'] = self.no_of_persons - category_counts_df['Total']

        category_counts_df = category_counts_df.astype(int)

        category_counts_df.loc['Total']= category_counts_df.sum()

        self.category_counts = category_counts_df

    def fit_statistics(self,
                       warm_corr=True,
                       tolerance=0.0000001,
                       max_iters=100,
                       ext_score_adjustment=0.5,
                       method='cos',
                       constant=0.1,
                       no_of_samples=100,
                       interval=None):

        if hasattr(self, 'thresholds') == False:
            self.calibrate(constant=constant, method=method)

        if hasattr(self, 'threshold_se') == False:
            self.std_errors(interval=interval, no_of_samples=no_of_samples,
                            constant=constant, method=method)

        if hasattr(self, 'person_abilities') == False:
            self.person_abils(warm_corr=warm_corr, tolerance=tolerance,
                              max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)

        '''
        Create matrices of expected scores, variances, kurtosis, residuals etc. to generate fit statistics
        '''

        missing_mask = (self.dataframe + 1) / (self.dataframe + 1)
        item_count = (self.dataframe == self.dataframe).sum(axis=0)
        person_count = (self.dataframe == self.dataframe).sum(axis=1)

        self.cat_prob_dict = {item: [[self.cat_prob(abil, self.diffs[item], cat, self.thresholds)
                                      for cat in range(self.max_score + 1)]
                                     for abil in self.person_abilities.values]
                              for item in self.dataframe.columns}
        for item in self.dataframe.columns:
            self.cat_prob_dict[item] = pd.DataFrame(self.cat_prob_dict[item])
            self.cat_prob_dict[item].index = self.dataframe.index
            self.cat_prob_dict[item].columns = np.arange(self.max_score + 1)

        self.exp_score_df = pd.DataFrame()
        for item in self.dataframe.columns:
            self.exp_score_df[item] = sum(cat * self.cat_prob_dict[item][cat]
                                          for cat in range(self.max_score + 1))
        self.exp_score_df *= missing_mask

        self.info_df = pd.DataFrame()
        for item in self.dataframe.columns:
            self.info_df[item] = sum(((cat - self.exp_score_df[item]) ** 2) * self.cat_prob_dict[item][cat]
                                     for cat in range(self.max_score + 1))
        self.info_df *= missing_mask

        self.kurtosis_df = pd.DataFrame()
        for item in self.dataframe.columns:
            self.kurtosis_df[item] = sum(self.cat_prob_dict[item][cat] * ((cat - self.exp_score_df[item]) ** 4)
                                         for cat in range(self.max_score + 1))
        self.kurtosis_df *= missing_mask

        self.residual_df = self.dataframe - self.exp_score_df
        self.std_residual_df = self.residual_df / np.sqrt(self.info_df.astype(float))

        scores = self.dataframe.sum(axis=1)
        max_scores = self.dataframe.count(axis=1) * self.max_score

        self.exp_score_df = self.exp_score_df[(scores > 0) & (scores < max_scores)]
        self.info_df = self.info_df[(scores > 0) & (scores < max_scores)]
        self.kurtosis_df = self.kurtosis_df[(scores > 0) & (scores < max_scores)]
        self.residual_df = self.residual_df[(scores > 0) & (scores < max_scores)]
        self.std_residual_df = self.std_residual_df[(scores > 0) & (scores < max_scores)]

        '''
        Item fit statistics
        '''

        self.item_outfit_ms = (self.std_residual_df ** 2).mean()
        self.item_infit_ms = (self.residual_df ** 2).sum() / self.info_df.sum()

        item_outfit_q = ((self.kurtosis_df / (self.info_df ** 2)) / (item_count ** 2)).sum() - (1 / item_count)
        item_outfit_q = np.sqrt(item_outfit_q)
        self.item_outfit_zstd = (np.cbrt(self.item_outfit_ms) - 1) * (3 / item_outfit_q) + (item_outfit_q / 3)

        item_infit_q = (self.kurtosis_df - self.info_df ** 2).sum() / (self.info_df.sum() ** 2)
        item_infit_q = np.sqrt(item_infit_q)
        self.item_infit_zstd = (np.cbrt(self.item_infit_ms) - 1) * (3 / item_infit_q) + (item_infit_q / 3)

        self.response_counts = self.dataframe.count(axis=0)
        self.item_facilities = self.dataframe.mean(axis=0) / self.max_score

        self.point_measure = [self.dataframe[item].astype(float).corr(self.person_abilities)
                              for item in self.dataframe.columns]
        self.point_measure = pd.Series(self.point_measure)
        self.point_measure.index = self.dataframe.columns

        item_abil_dev = {item: self.person_abilities[self.dataframe[item] == self.dataframe[item]]
                         for item in self.dataframe.columns}
        for item in self.dataframe.columns:
            item_abil_dev[item] -= item_abil_dev[item].mean()

        exp_point_measure_df = self.exp_score_df - self.dataframe.mean()

        exp_point_measure_num = [(exp_point_measure_df[item] * item_abil_dev[item]).sum()
                                 for item in self.dataframe.columns]
        exp_point_measure_num = pd.Series(exp_point_measure_num)
        exp_point_measure_num.index = self.dataframe.columns

        exp_point_measure_den = exp_point_measure_df ** 2 + self.info_df
        exp_point_measure_den = exp_point_measure_den.sum()

        for item in self.dataframe.columns:
            exp_point_measure_den[item] *= (item_abil_dev[item] ** 2).sum()

        exp_point_measure_den = np.sqrt(exp_point_measure_den.astype(float))

        self.exp_point_measure = exp_point_measure_num / exp_point_measure_den
        '''
        Threshold fit statistics
        '''

        abil_matrix = [[self.person_abilities[person] for item in self.dataframe.columns]
                       for person in self.dataframe.index]
        abil_df = pd.DataFrame(abil_matrix)
        abil_df.index = self.dataframe.index
        abil_df.columns = self.dataframe.columns

        dich_thresh = {}
        for threshold in range(self.max_score):
            dich_thresh[threshold + 1] = self.dataframe.where(self.dataframe.isin([threshold, threshold + 1]), np.nan)
            dich_thresh[threshold + 1] -= threshold

        dich_thresh_exp = {}
        dich_thresh_var = {}
        dich_thresh_kur = {}
        dich_residuals = {}
        dich_std_residuals = {}

        dich_thresh_count = {threshold + 1: (dich_thresh[threshold + 1] == dich_thresh[threshold + 1]).sum().sum()
                             for threshold in range(self.max_score)}

        for threshold in range(self.max_score):
            diff_matrix = [self.diffs + self.thresholds[threshold + 1] for person in self.dataframe.index]
            diff_df = pd.DataFrame(diff_matrix)
            diff_df.index = self.dataframe.index
            diff_df.columns = self.dataframe.columns

            missing_mask = (dich_thresh[threshold + 1] + 1) / (dich_thresh[threshold + 1] + 1)

            dich_thresh_exp[threshold + 1] = 1 / (1 + np.exp(diff_df - abil_df))
            dich_thresh_exp[threshold + 1] *= missing_mask

            dich_thresh_var[threshold + 1] = dich_thresh_exp[threshold + 1] * (1 - dich_thresh_exp[threshold + 1])
            dich_thresh_var[threshold + 1] *= missing_mask

            dich_thresh_kur[threshold + 1] = (
                        ((-dich_thresh_exp[threshold + 1]) ** 4) * (1 - dich_thresh_exp[threshold + 1]) +
                        ((1 - dich_thresh_exp[threshold + 1]) ** 4) * dich_thresh_exp[threshold + 1])
            dich_thresh_kur[threshold + 1] *= missing_mask

            dich_residuals[threshold + 1] = dich_thresh[threshold + 1] - dich_thresh_exp[threshold + 1]
            dich_std_residuals[threshold + 1] = (dich_residuals[threshold + 1] /
                                                 np.sqrt(dich_thresh_var[threshold + 1].astype(float)))

        self.threshold_outfit_ms = {threshold + 1: ((dich_std_residuals[threshold + 1] ** 2).sum().sum() /
                                                    dich_thresh[threshold + 1].count().sum())
                                    for threshold in range(self.max_score)}
        self.threshold_outfit_ms = pd.Series(self.threshold_outfit_ms)

        self.threshold_infit_ms = {threshold + 1: (dich_residuals[threshold + 1] ** 2).sum().sum() /
                                                  dich_thresh_var[threshold + 1].sum().sum()
                                   for threshold in range(self.max_score)}
        self.threshold_infit_ms = pd.Series(self.threshold_infit_ms)

        threshold_outfit_q = {threshold + 1: (((dich_thresh_kur[threshold + 1] / (dich_thresh_var[threshold + 1] ** 2)) /
                              (dich_thresh_count[threshold + 1] ** 2)).sum().sum() -
                              (1 / dich_thresh_count[threshold + 1]))
            for threshold in range(self.max_score)}
        threshold_outfit_q = pd.Series(threshold_outfit_q)
        threshold_outfit_q = np.sqrt(threshold_outfit_q.astype(float))

        self.threshold_outfit_zstd = (np.cbrt(self.threshold_outfit_ms.astype(float)) - 1) * (3 / threshold_outfit_q) + (
                    threshold_outfit_q / 3)

        threshold_infit_q = {threshold + 1: ((dich_thresh_kur[threshold + 1] -
                                              dich_thresh_var[threshold + 1] ** 2).sum().sum() /
                                             (dich_thresh_var[threshold + 1].sum().sum() ** 2))
                             for threshold in range(self.max_score)}
        threshold_infit_q = pd.Series(threshold_infit_q)
        threshold_infit_q = np.sqrt(threshold_infit_q.astype(float))

        self.threshold_infit_zstd = (np.cbrt(self.threshold_infit_ms.astype(float)) - 1) * (3 / threshold_infit_q) + (
                    threshold_infit_q / 3)

        dich_facilities = {threshold + 1: dich_thresh[threshold + 1].mean(axis=0)
                           for threshold in range(self.max_score)}

        abil_deviation = self.person_abilities.copy() - self.person_abilities.mean()

        point_measure_dict = dich_thresh.copy()

        for threshold in range(self.max_score):
            for item in self.dataframe.columns:
                point_measure_dict[threshold + 1][item] -= dich_facilities[threshold + 1][item]

        point_measure_nums = {threshold + 1: point_measure_dict[threshold + 1].mul(abil_deviation, axis=0).sum().sum()
                              for threshold in range(self.max_score)}
        point_measure_nums = pd.Series(point_measure_nums)

        point_measure_dens = {threshold + 1: np.sqrt((point_measure_dict[threshold + 1] ** 2).sum().sum() *
                                                     (abil_deviation ** 2).sum())
                              for threshold in range(self.max_score)}
        point_measure_dens = pd.Series(point_measure_dens)

        self.threshold_point_measure = point_measure_nums / point_measure_dens

        threshold_exp_pm_dict = {threshold + 1: dich_thresh_exp[threshold + 1] -
                                                (dich_thresh_exp[threshold + 1].sum().sum() /
                                                 dich_thresh_exp[threshold + 1].count().sum())
                                 for threshold in range(self.max_score)}

        threshold_exp_pm_num = {threshold + 1: threshold_exp_pm_dict[threshold + 1].mul(abil_deviation, axis=0).sum().sum()
                                for threshold in range(self.max_score)}
        threshold_exp_pm_num = pd.Series(threshold_exp_pm_num)

        threshold_exp_pm_den = {threshold + 1: ((threshold_exp_pm_dict[threshold + 1] ** 2) +
                                                dich_thresh_var[threshold + 1]).sum().sum()
                                for threshold in range(self.max_score)}
        threshold_exp_pm_den = pd.Series(threshold_exp_pm_den)
        threshold_exp_pm_den *= (abil_deviation ** 2).sum()
        threshold_exp_pm_den = np.sqrt(threshold_exp_pm_den)

        self.threshold_exp_point_measure = threshold_exp_pm_num / threshold_exp_pm_den

        self.threshold_rmsr = {threshold + 1: (np.sqrt((dich_residuals[threshold + 1] ** 2).sum().sum() /
                                                       dich_residuals[threshold + 1].count().sum()))
                               for threshold in range(self.max_score)}
        self.threshold_rmsr = pd.Series(self.threshold_rmsr)

        differences = {threshold + 1: pd.DataFrame() for threshold in range(self.max_score)}

        for threshold in range(self.max_score):
            for item in self.dataframe.columns:
                differences[threshold + 1][item] = (self.person_abilities -
                                                    self.diffs[item] -
                                                    self.thresholds[threshold + 1])

        nums = {threshold + 1: (differences[threshold + 1] * dich_residuals[threshold + 1]).sum().sum()
                for threshold in range(self.max_score)}
        nums = pd.Series(nums)

        dens = {threshold + 1: (dich_thresh_var[threshold + 1] * (differences[threshold + 1] ** 2)).sum().sum()
                for threshold in range(self.max_score)}
        dens = pd.Series(dens)

        self.threshold_discrimination = 1 + nums / dens
        '''
        Person fit statistics
        '''

        self.csem_vector = 1 / np.sqrt(self.info_df.sum(axis=1))
        self.rsem_vector = np.sqrt((self.residual_df ** 2).sum(axis=1)) / self.info_df.sum(axis=1)

        self.person_outfit_ms = (self.std_residual_df ** 2).mean(axis=1)
        self.person_outfit_ms.name = 'Outfit MS'
        self.person_infit_ms = (self.residual_df ** 2).sum(axis=1) / self.info_df.sum(axis=1)
        self.person_infit_ms.name = 'Infit MS'

        base_df = self.kurtosis_df / (self.info_df ** 2)
        for column in self.dataframe.columns:
            base_df[column] /= (person_count ** 2)
        person_outfit_q = base_df.sum(axis=1) -  (1 / person_count)
        person_outfit_q = np.sqrt(person_outfit_q.astype(float))
        self.person_outfit_zstd = (np.cbrt(self.person_outfit_ms.astype(float)) - 1) * (3 / person_outfit_q) + (person_outfit_q / 3)
        self.person_outfit_zstd.name = 'Outfit Z'

        person_infit_q = (self.kurtosis_df - self.info_df ** 2).sum(axis=1) / (self.info_df.sum(axis=1) ** 2)
        person_infit_q = np.sqrt(person_infit_q.astype(float))
        self.person_infit_zstd = (np.cbrt(self.person_infit_ms.astype(float)) - 1) * (3 / person_infit_q) + (person_infit_q / 3)
        self.person_infit_zstd.name = 'Infit Z'

        '''
        Test-level fit statistics
        '''

        self.isi = np.sqrt(self.diffs.var() / (self.item_se ** 2).mean() - 1)
        self.item_strata = (4 * self.isi + 1) / 3
        self.item_reliability = self.isi ** 2 / (1 + self.isi ** 2)

        self.psi = (np.sqrt(np.var(self.person_abilities) - (self.rsem_vector ** 2).mean()) /
                     np.sqrt((self.rsem_vector ** 2).mean()))
        self.person_strata = (4 * self.psi + 1) / 3
        self.person_reliability = (self.psi ** 2) / (1 + (self.psi ** 2))

    def res_corr_analysis(self,
                          warm_corr=True,
                          tolerance=0.0000001,
                          max_iters=100,
                          ext_score_adjustment=0.5,
                          constant=0.1,
                          method='cos',
                          matrix_power=3,
                          log_lik_tol=0.000001,
                          no_of_samples=100,
                          interval=None):

        '''
        Analysis of correlations of standardised residuals for violations of local item interdependence
        and unidimensionality
        '''

        if hasattr(self, 'std_residual_df') == False:
            self.fit_statistics(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                ext_score_adjustment=ext_score_adjustment, constant=constant, method=method,
                                matrix_power=matrix_power, log_lik_tol=log_lik_tol, no_of_samples=no_of_samples,
                                interval=interval)

        self.residual_correlations = self.std_residual_df.corr(numeric_only=False)

        pca = PCA()
        
        try:
            pca.fit(self.std_residual_df.corr())

            self.eigenvectors = pd.DataFrame(pca.components_)
            self.eigenvectors.columns = [f'Eigenvector {pc + 1}' for pc in range(self.no_of_items)]

            self.eigenvalues = pca.explained_variance_
            self.eigenvalues = pd.DataFrame(self.eigenvalues)
            self.eigenvalues.index = [f'PC {pc + 1}' for pc in range(self.no_of_items)]
            self.eigenvalues.columns = ['Eigenvalue']

            self.variance_explained = pd.DataFrame(pca.explained_variance_ratio_)
            self.variance_explained.index = [f'PC {pc + 1}' for pc in range(self.no_of_items)]
            self.variance_explained.columns = ['Variance explained']

            self.loadings = self.eigenvectors.T * np.sqrt(pca.explained_variance_)
            self.loadings = pd.DataFrame(self.loadings)
            self.loadings.columns = [f'PC {pc + 1}' for pc in range(self.no_of_items)]
            self.loadings.index = [item for item in self.dataframe.columns]

        except:
            self.pca_fail = True
            print('PCA of item residuals failed')

            self.eigenvectors = None
            self.eigenvalues = None
            self.variance_explained = None
            self.loadings = None

    def item_stats_df(self,
                      full=False,
                      point_measure_corr=False,
                      dp=3,
                      warm_corr=True,
                      tolerance=0.0000001,
                      max_iters=100,
                      ext_score_adjustment=0.5,
                      method='cos',
                      constant=0.1,
                      no_of_samples=100,
                      interval=None):

        if full:
            zstd = True
            point_measure_corr = True

            if interval is None:
                interval = 0.95

        if ((hasattr(self, 'threshold_low') == False) or
            (self.threshold_low is None) and (interval is not None)):
            self.std_errors(interval=interval, no_of_samples=no_of_samples, constant=constant, method=method)

        if hasattr(self, 'item_infit_ms') == False:
            self.fit_statistics(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                ext_score_adjustment=ext_score_adjustment, method=method,
                                constant=constant, no_of_samples=no_of_samples, interval=interval)

        self.item_stats = pd.DataFrame()

        self.item_stats['Estimate'] = self.diffs.astype(float).round(dp)
        self.item_stats['SE'] = self.item_se.to_numpy().round(dp)

        if interval is not None:
            self.item_stats[f'{round((1 - interval) * 50, 1)}%'] = self.item_low.to_numpy().round(dp)
            self.item_stats[f'{round((1 + interval) * 50, 1)}%'] = self.item_high.to_numpy().round(dp)

        self.item_stats['Count'] = self.response_counts.to_numpy().astype(int)
        self.item_stats['Facility'] = self.item_facilities.to_numpy().round(dp)

        self.item_stats['Infit MS'] = self.item_infit_ms.to_numpy().round(dp)
        if zstd:
            self.item_stats['Infit Z'] = self.item_infit_zstd.to_numpy().round(dp)

        self.item_stats['Outfit MS'] = self.item_outfit_ms.to_numpy().round(dp)
        if zstd:
            self.item_stats['Outfit Z'] = self.item_outfit_zstd.to_numpy().round(dp)

        if point_measure_corr:
            self.item_stats['PM corr'] = self.point_measure.astype(float).round(dp)
            self.item_stats['Exp PM corr'] = self.exp_point_measure.astype(float).round(dp)

        self.item_stats.index = self.dataframe.columns

    def threshold_stats_df(self,
                           full=False,
                           zstd=False,
                           disc=False,
                           point_measure_corr=False,
                           dp=3,
                           warm_corr=True,
                           tolerance=0.0000001,
                           max_iters=100,
                           ext_score_adjustment=0.5,
                           method='cos',
                           constant=0.1,
                           no_of_samples=100,
                           interval=None):

        if full:
            zstd = True
            disc = True
            point_measure_corr = True

            if interval is None:
                interval = 0.95

        if hasattr(self, 'threshold_infit_ms') == False:
            self.fit_statistics(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                ext_score_adjustment=ext_score_adjustment, method=method,
                                constant=constant, no_of_samples=no_of_samples, interval=interval)

        self.threshold_stats = pd.DataFrame()

        self.threshold_stats['Estimate'] = self.thresholds[1:].round(dp)
        self.threshold_stats['SE'] = self.threshold_se[1:].round(dp)
        if interval is not None:
            self.threshold_stats[f'{round((1 - interval) * 50, 1)}%'] = self.threshold_low[1:].round(dp)
            self.threshold_stats[f'{round((1 + interval) * 50, 1)}%'] = self.threshold_high[1:].round(dp)

        self.threshold_stats['Infit MS'] = self.threshold_infit_ms.to_numpy().round(dp)
        if zstd:
            self.threshold_stats['Infit Z'] = self.threshold_infit_zstd.to_numpy().round(dp)
        self.threshold_stats['Outfit MS'] = self.threshold_outfit_ms.to_numpy().round(dp)
        if zstd:
            self.threshold_stats['Outfit Z'] = self.threshold_outfit_zstd.to_numpy().round(dp)

        if disc:
            self.threshold_stats['Discrim'] = self.threshold_discrimination.to_numpy().round(dp)

        if point_measure_corr:
            self.threshold_stats['PM corr'] = self.threshold_point_measure.to_numpy().round(dp)
            self.threshold_stats['Exp PM corr'] = self.threshold_exp_point_measure.to_numpy().round(dp)

        self.threshold_stats.index = [f'Threshold {threshold + 1}' for threshold in range(self.max_score)]

    def person_stats_df(self,
                        full=False,
                        rsem=False,
                        dp=3,
                        warm_corr=True,
                        tolerance=0.0000001,
                        max_iters=100,
                        ext_score_adjustment=0.5,
                        method='cos',
                        constant=0.1):

        '''
        Produces a person stats dataframe with raw score, ability estimate,
        CSEM and RSEM for each person.
        '''

        if hasattr(self, 'person_infit_ms') == False:
            self.fit_statistics(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                ext_score_adjustment=ext_score_adjustment, method=method, constant=constant)

        if full:
            rsem = True

        person_stats_df = pd.DataFrame()
        person_stats_df.index = self.dataframe.index

        person_stats_df['Estimate'] = self.person_abilities.values.round(dp)

        person_stats_df['CSEM'] = self.csem_vector.round(dp)
        if rsem:
            person_stats_df['RSEM'] = self.rsem_vector.round(dp)

        person_stats_df['Score'] = self.dataframe.sum(axis=1).astype(int)
        person_stats_df['Max score'] = (self.dataframe.count(axis=1) * self.max_score).astype(int)
        person_stats_df['p'] = (self.dataframe.mean(axis=1) / self.max_score).round(dp)

        person_stats_df['Infit MS'] = [np.nan for person in self.dataframe.index]
        person_stats_df['Infit Z'] = [np.nan for person in self.dataframe.index]
        person_stats_df['Outfit MS'] = [np.nan for person in self.dataframe.index]
        person_stats_df['Outfit Z'] = [np.nan for person in self.dataframe.index]

        person_stats_df.update(self.person_infit_ms.round(dp))
        person_stats_df.update(self.person_infit_zstd.round(dp))
        person_stats_df.update(self.person_outfit_ms.round(dp))
        person_stats_df.update(self.person_outfit_zstd.round(dp))

        self.person_stats = person_stats_df

    def test_stats_df(self,
                      dp=3,
                      warm_corr=True,
                      tolerance=0.0000001,
                      max_iters=100,
                      ext_score_adjustment=0.5,
                      method='cos',
                      constant=0.1):

        if hasattr(self, 'psi') == False:
            self.fit_statistics(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                ext_score_adjustment=ext_score_adjustment, method=method, constant=constant)

        self.test_stats = pd.DataFrame()

        self.test_stats['Items'] = [self.diffs.mean(),
                                    self.diffs.std(),
                                    self.isi,
                                    self.item_strata,
                                    self.item_reliability]

        self.test_stats['Persons'] = [self.person_abilities.mean(),
                                      self.person_abilities.std(),
                                      self.psi,
                                      self.person_strata,
                                      self.person_reliability]

        self.test_stats.index = ['Mean', 'SD', 'Separation ratio', 'Strata', 'Reliability']
        self.test_stats = round(self.test_stats, dp)

    def save_stats(self,
                   filename,
                   format='csv',
                   dp=3,
                   warm_corr=True,
                   tolerance=0.0000001,
                   max_iters=100,
                   ext_score_adjustment=0.5,
                   method='cos',
                   constant=0.1,
                   no_of_samples=100,
                   interval=None):

        if hasattr(self, 'item_stats') == False:
            self.item_stats_df(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                               ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                               no_of_samples=no_of_samples, interval=interval)

        if hasattr(self, 'threshold_stats') == False:
            self.threshold_stats_df(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                    ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                    no_of_samples=no_of_samples, interval=interval)

        if hasattr(self, 'person_stats') == False:
            self.person_stats_df(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                 ext_score_adjustment=ext_score_adjustment, method=method, constant=constant)

        if hasattr(self, 'test_stats') == False:
            self.test_stats_df(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                               ext_score_adjustment=ext_score_adjustment, method=method, constant=constant)

        if format == 'xlsx':

            if filename[-5:] != '.xlsx':
                filename += '.xlsx'

            writer = pd.ExcelWriter(filename, engine='xlsxwriter')

            self.item_stats.to_excel(writer, sheet_name='Item statistics')
            self.threshold_stats.to_excel(writer, sheet_name='Threshold statistics')
            self.person_stats.to_excel(writer, sheet_name='Person statistics')
            self.test_stats.to_excel(writer, sheet_name='Test statistics')

            writer.save()

        else:
            if filename[-4:] == '.csv':
                filename = filename[:-4]

            self.item_stats.to_csv(f'{filename}_item_stats.csv')
            self.threshold_stats.to_csv(f'{filename}_threshold_stats.csv')
            self.person_stats.to_csv(f'{filename}_person_stats.csv')
            self.test_stats.to_csv(f'{filename}_test_stats.csv')

    def save_residuals(self,
                       filename,
                       format='csv',
                       single=True,
                       dp=3,
                       warm_corr=True,
                       tolerance=0.0000001,
                       max_iters=100,
                       ext_score_adjustment=0.5,
                       method='cos',
                       constant=0.1):

        if hasattr(self, 'eigenvectors') == False:
            self.fit_statistics(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                ext_score_adjustment=ext_score_adjustment, method=method, constant=constant)

        if single:
            if format == 'xlsx':

                if filename[-5:] != '.xlsx':
                    filename += '.xlsx'

                writer = pd.ExcelWriter(filename, engine='xlsxwriter')
                row = 0

                self.eigenvectors.round(dp).to_excel(writer, sheet_name='Item residual analysis',
                                                     startrow=row, startcol=0)
                row += (self.eigenvectors.shape[0] + 2)

                self.eigenvalues.round(dp).to_excel(writer, sheet_name='Item residual analysis',
                                                    startrow=row, startcol=0)
                row += (self.eigenvalues.shape[0] + 2)

                self.variance_explained.round(dp).to_excel(writer, sheet_name='Item residual analysis',
                                                           startrow=row, startcol=0)
                row += (self.variance_explained.shape[0] + 2)

                self.loadings.round(dp).to_excel(writer, sheet_name='Item residual analysis',
                                                 startrow=row, startcol=0)

                writer.save()

            else:
                if filename[-4:] != '.csv':
                    filename += '.csv'

                with open(filename, 'a') as f:
                    self.eigenvectors.round(dp).to_csv(f)
                    f.write("\n")
                    self.eigenvalues.round(dp).to_csv(f)
                    f.write("\n")
                    self.variance_explained.round(dp).to_csv(f)
                    f.write("\n")
                    self.loadings.round(dp).to_csv(f)

        else:
            if format == 'xlsx':

                if filename[-5:] != '.xlsx':
                    filename += '.xlsx'

                writer = pd.ExcelWriter(filename, engine='xlsxwriter')

                self.eigenvectors.round(dp).to_excel(writer, sheet_name='Eigenvectors')
                self.eigenvalues.round(dp).to_excel(writer, sheet_name='Eigenvalues')
                self.variance_explained.round(dp).to_excel(writer, sheet_name='Variance explained')
                self.loadings.round(dp).to_excel(writer, sheet_name='Principal Component loadings')

                writer.save()

            else:
                if filename[-4:] == '.csv':
                    filename = filename[:-4]

                self.eigenvectors.round(dp).to_csv(f'{filename}_eigenvectors.csv')
                self.eigenvalues.round(dp).to_csv(f'{filename}_eigenvalues.csv')
                self.variance_explained.round(dp).to_csv(f'{filename}_variance_explained.csv')
                self.loadings.round(dp).to_csv(f'{filename}_principal_component_loadings.csv')

    def class_intervals(self,
                        abilities,
                        items=None,
                        no_of_classes=5):

        class_groups = [f'class_{class_no + 1}' for class_no in range(no_of_classes)]

        df = self.dataframe.copy()

        if items is None:
            items = self.dataframe.columns.tolist()

        df = df[items].dropna()
        abils = abilities.loc[df.index]

        quantiles = (abils.quantile([(i + 1) / no_of_classes
                                     for i in range(no_of_classes - 1)]))

        mask_dict = {}
        mask_dict['class_1'] = (abils < quantiles.values[0])
        mask_dict[f'class_{no_of_classes}'] = (abils >= quantiles.values[no_of_classes - 2])
        for class_no in range(no_of_classes - 2):
            mask_dict[f'class_{class_no + 2}'] = ((abils >= quantiles.values[class_no]) &
                                                  (abils < quantiles.values[class_no + 1]))

        class_sizes = {class_group: sum(mask_dict[class_group])
                       for class_group in class_groups}
        class_sizes = pd.Series(class_sizes)

        response_classes = {class_group: df.index[mask_dict[class_group]]
                            for class_group in class_groups}

        mean_abilities = {class_group: abils[mask_dict[class_group]].mean()
                          for class_group in class_groups}
        mean_abilities = pd.Series(mean_abilities)

        obs = {class_group: df[mask_dict[class_group]].mean().sum()
               for class_group in class_groups}

        for class_group in class_groups:
            obs[class_group] = pd.Series(obs[class_group])

        obs = pd.concat(obs, keys=obs.keys())

        class_abilities = {class_group: abils[mask_dict[class_group]]
                           for class_group in class_groups}
        class_abilities = pd.concat(class_abilities, keys=class_abilities.keys())

        return class_sizes, response_classes, class_abilities, mean_abilities, obs

    def class_intervals_cats(self,
                             abilities,
                             item,
                             no_of_classes=5):

        class_groups = [f'class_{class_no + 1}' for class_no in range(no_of_classes)]

        df = self.dataframe.copy()
        df = df[item].dropna()

        abils = abilities.loc[df.index]

        quantiles = (abils.quantile([(i + 1) / no_of_classes
                                     for i in range(no_of_classes - 1)]))

        mask_dict = {}
        mask_dict['class_1'] = (abils < quantiles.values[0])
        mask_dict[f'class_{no_of_classes}'] = (abils >= quantiles.values[no_of_classes - 2])
        for class_no in range(no_of_classes - 2):
            mask_dict[f'class_{class_no + 2}'] = ((abils >= quantiles.values[class_no]) &
                                                  (abils < quantiles.values[class_no + 1]))

        class_sizes = {class_group: sum(mask_dict[class_group]) for class_group in class_groups}
        class_sizes = pd.Series(class_sizes)

        response_classes = {class_group: df.index[mask_dict[class_group]]
                            for class_group in class_groups}

        mean_abilities = {class_group: abils[mask_dict[class_group]].mean()
                          for class_group in class_groups}
        mean_abilities = pd.Series(mean_abilities)

        obs_props = {class_group: np.array([(df[mask_dict[class_group]] == cat).sum()
                                            for cat in range(self.max_score + 1)])
                     for class_group in class_groups}

        for class_group in class_groups:
            obs_props[class_group] = obs_props[class_group] / obs_props[class_group].sum()

        obs_props = pd.DataFrame(obs_props).to_numpy().T

        class_abilities = {class_group: abils[mask_dict[class_group]]
                           for class_group in class_groups}
        class_abilities = pd.concat(class_abilities, keys=class_abilities.keys())

        return class_sizes, response_classes, class_abilities, mean_abilities, obs_props

    def class_intervals_thresholds(self,
                                   item=None,
                                   no_of_classes=5):

        if hasattr(self, 'person_abilities') == False:
            self.person_abils(warm_corr=False)

        class_groups = [f'class_{class_no + 1}' for class_no in range(no_of_classes)]

        df = self.dataframe
        if item is not None:
            df = df[item]

        else:
            df_dict = {item: self.dataframe[item] for item in self.dataframe.columns}
            df = pd.concat(df_dict.values(), keys=df_dict.keys())

        if item is None:
            abils_dict = {item: self.person_abilities - self.diffs[item] for item in self.dataframe.columns}
            abils = pd.DataFrame(abils_dict.values(), keys=abils_dict.keys())
        else:
            abils = self.person_abilities

        def mask_dictionary(abils):

            quantiles = (abils.quantile([(i + 1) / no_of_classes
                                         for i in range(no_of_classes - 1)]))

            mask_dict = {}

            mask_dict['class_1'] = (abils < quantiles.values[0])
            mask_dict[f'class_{no_of_classes}'] = (abils >= quantiles.values[no_of_classes - 2])
            for class_group in range(no_of_classes - 2):
                mask_dict[f'class_{class_group + 2}'] = ((abils >= quantiles.values[class_group]) &
                                                         (abils < quantiles.values[class_group + 1]))

            return mask_dict

        mean_abilities = []
        obs_props = []
        class_sizes = []
        response_classes = []

        for threshold in range(self.max_score):

            cond_df_mask = df.isin([threshold, threshold + 1])
            cond_df = df[cond_df_mask]
            adj_abils = abils[cond_df_mask]

            adj_abil_score_df = pd.DataFrame()
            adj_abil_score_df['abil'] = adj_abils
            adj_abil_score_df['score'] = cond_df
            adj_abil_score_df.index = cond_df.index

            masks = mask_dictionary(adj_abils)

            cond_classes = {class_group: adj_abil_score_df[masks[class_group]]
                            for class_group in class_groups}

            mean_abilities.append([cond_classes[class_group]['abil'].mean()
                                   for class_group in class_groups])
            obs_props.append([(cond_classes[class_group]['score'] - threshold).mean()
                              for class_group in class_groups])

            sizes = {class_group: sum(masks[class_group]) for class_group in class_groups}
            class_sizes.append(pd.Series(sizes))

            response_classes.append({class_group: cond_df.index[masks[class_group]]
                                     for class_group in class_groups})

        mean_abilities = np.array(mean_abilities)
        obs_props = np.array(obs_props).T

        class_abilities = {class_group: adj_abils[masks[class_group]]
                           for class_group in class_groups}
        class_abilities = pd.concat(class_abilities, keys=class_abilities.keys())

        return class_sizes, response_classes, class_abilities, mean_abilities, obs_props

    '''
    *** PLOTS ***
    '''

    def plot_data(self,
                  x_data,
                  y_data,
                  items=None,
                  obs=None,
                  x_obs_data=np.array([]),
                  y_obs_data=np.array([]),
                  thresh_lines=False,
                  central_diff=False,
                  score_lines_item=[None, []],
                  score_lines_test=None,
                  point_info_lines_item=[None, []],
                  point_info_lines_test=None,
                  point_csem_lines=None,
                  score_labels=False,
                  x_min=-10,
                  x_max=10,
                  y_max=0,
                  warm=True,
                  cat_highlight=None,
                  graph_title='',
                  y_label='',
                  plot_style='colorblind',
                  black=False,
                  figsize=(8, 6),
                  font='Times',
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

        if items is None:
            items = self.items
            difficulties = self.diffs

        else:
            difficulties = self.diffs.loc[items]

        if tex:
            plt.rcParams["text.latex.preamble"].join([r"\usepackage{dashbox}", r"\setmainfont{xcolor}",])
        else:
            plt.rcParams["text.usetex"] = False

        plt.style.use('seaborn-v0_8-' + plot_style)

        graph, ax = plt.subplots(figsize=figsize)

        no_of_plots = y_data.shape[1]

        if black:
            for i in range(no_of_plots):
                ax.plot(x_data, y_data[:, i], '', label=i+1, color='black')

        else:
            for i in range(no_of_plots):
                ax.plot(x_data, y_data[:, i], '', label=i+1)

        if obs is not None:
            try:
                no_of_obs_plots = y_obs_data.shape[1]
                if isinstance(x_obs_data, pd.Series):
                    for j in range (no_of_obs_plots):
                        ax.plot(x_obs_data, y_obs_data[:, j], 'o')

                else:
                    no_of_obs_plots = y_obs_data.shape[1]
                    for j in range (no_of_obs_plots):
                        ax.plot(x_obs_data[:, j], y_obs_data[:, j], 'o')

            except:
                pass

        if thresh_lines:
            for threshold in range(self.max_score):
                if items is None:
                    plt.axvline(x=self.thresholds[threshold + 1],
                                color='black', linestyle='--')
                else:
                    plt.axvline(x = self.diffs[items] + self.thresholds[threshold + 1],
                                color = 'black', linestyle='--')

        if central_diff:
            plt.axvline(x = self.diffs[items], color = 'darkred', linestyle='--')

        if score_lines_item[1] != []:

            if (all(x > 0 for x in score_lines_item[1]) &
                all(x < self.max_score for x in score_lines_item[1])):

                abils_set = [self.score_abil(score, items=[score_lines_item[0]], warm_corr=False)
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

            if (all(x > 0 for x in score_lines_test) &
                all(x < self.max_score * len(difficulties) for x in score_lines_test)):

                abils_set = [self.score_abil(score, items=difficulties.keys(), warm_corr=warm)
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

        if point_info_lines_item[1] != []:

            info_set = [self.variance(ability, difficulties, self.thresholds)
            			for ability in point_info_lines_item[1]]

            for abil, info in zip(point_info_lines_item[1], info_set):
                plt.vlines(x=abil, ymin=-100, ymax=info, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=info, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(x_min + (x_max - x_min) / 100, info + y_max / 50, str(round(info, 3)))

        if point_info_lines_test is not None:

            info_set = [sum(self.variance(ability, difficulty, self.thresholds)
            				for difficulty in difficulties)
                        for ability in point_info_lines_test]

            for abil, info in zip(point_info_lines_test, info_set):
                plt.vlines(x=abil, ymin=-100, ymax=info, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=info, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(x_min + (x_max - x_min) / 100, info + y_max / 50, str(round(info, 3)))

        if point_csem_lines is not None:

            info_set = [sum(self.variance(ability, difficulty, self.thresholds)
            				for difficulty in difficulties)
                        for ability in point_csem_lines]
            info_set = np.array(info_set)
            csem_set = 1 / np.sqrt(info_set)

            for abil, csem in zip(point_csem_lines, csem_set):
                plt.vlines(x=abil, ymin=-100, ymax=csem, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=csem, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(x_min + (x_max - x_min) / 100, csem + y_max / 50, str(round(csem, 3)))

        if items is not None:
            if cat_highlight in range(self.max_score + 1):

                if cat_highlight == 0:
                    if items is None:
                        plt.axvspan(-100, self.thresholds[1],
                                    facecolor='blue', alpha=0.2)
                    else:
                        plt.axvspan(-100, self.diffs[items] + self.thresholds[1],
                                    facecolor='blue', alpha=0.2)

                elif cat_highlight == self.max_score:
                    if items is None:
                        plt.axvspan(self.thresholds[self.max_score], 100,
                                    facecolor='blue', alpha=0.2)
                    else:
                        plt.axvspan(self.diffs[items] + self.thresholds[self.max_score], 100,
                                    facecolor='blue', alpha=0.2)

                else:
                    if (self.thresholds[cat_highlight + 1] >
                        self.thresholds[cat_highlight]):
                        if items is None:
                            plt.axvspan(self.thresholds[cat_highlight], self.thresholds[cat_highlight + 1],
                                        facecolor='blue', alpha=0.2)
                        else:
                            plt.axvspan(self.diffs[items] + self.thresholds[cat_highlight],
                                        self.diffs[items] + self.thresholds[cat_highlight + 1],
                                        facecolor='blue', alpha=0.2)

        if y_max <= 0:
            y_max = y_data.max() * 1.1

        plt.xlim(x_min, x_max)
        plt.ylim(0, y_max)

        plt.xlabel('Ability', fontname=font, fontsize=axis_font_size, fontweight='bold')
        plt.ylabel(y_label, fontname=font, fontsize=axis_font_size, fontweight='bold')
        plt.title(graph_title, fontname=font, fontsize=title_font_size, fontweight='bold')

        plt.grid(True)

        plt.tick_params(axis="x", labelsize=labelsize)
        plt.tick_params(axis="y", labelsize=labelsize)

        if filename is not None:
            plt.savefig(f'{filename}.{file_format}', dpi=plot_density)

        plt.close()

        return graph

    def icc(self,
            item,
            obs=False,
            xmin=-10,
            xmax=10,
            no_of_classes=5,
            title=True,
            thresh_lines=False,
            central_diff=False,
            score_lines=None,
            score_labels=False,
            cat_highlight=None,
            plot_style='colorblind',
            black=False,
            font='Times',
            title_font_size=15,
            axis_font_size=12,
            labelsize=12,
            filename=None,
            use_save_title=False,
            file_format='png',
            dpi=300):

        '''
        Plots Item Characteristic Curves for RSM, with optional overplotting
        of observed data, threshold lines and expected score threshold lines.
        '''

        abilities = np.arange(-20, 20, 0.1)
        y = [self.exp_score(ability, self.diffs[item], self.thresholds)
             for ability in abilities]
        y = np.array(y).reshape([len(abilities), 1])

        if obs:
            if hasattr(self, 'person_abiliites') == False:
                self.person_abils(warm_corr=False)

            _, _, _, mean_abilities, obs_means = self.class_intervals(self.person_abilities, items=item,
                                                                      no_of_classes=no_of_classes)

            xobsdata = mean_abilities
            yobsdata = obs_means
            yobsdata = np.array(yobsdata).reshape((-1, 1))

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        if title:
            if use_save_title:
                if save_title != '':
                    graphtitle = f'ICC: {save_title}'

            else:
                graphtitle = f'ICC: {item}'

        else:
            graphtitle = ''

        ylabel = 'Expected score'

        plot = self.plot_data(x_data=abilities, y_data=y, x_obs_data=xobsdata, y_obs_data=yobsdata, x_min=xmin,
                              x_max=xmax, y_max=self.max_score, items=item, graph_title=graphtitle, y_label=ylabel,
                              obs=obs, thresh_lines=thresh_lines, central_diff=central_diff, plot_style=plot_style,
                              score_lines_item=[item, score_lines], score_labels=score_labels, black=black, font=font,
                              cat_highlight= cat_highlight, title_font_size=title_font_size, labelsize=labelsize,
                              axis_font_size=axis_font_size,  filename=filename, plot_density=dpi,
                              file_format=file_format)

        return plot

    def crcs(self,
             item,
             obs=None,
             xmin=-10,
             xmax=10,
             no_of_classes=5,
             title=True,
             thresh_lines=False,
             central_diff=False,
             cat_highlight=None,
             plot_style='colorblind',
             black=False,
             font='Times',
             title_font_size=15,
             axis_font_size=12,
             labelsize=12,
             filename='',
             file_format='png',
             dpi=300):

        '''
        Plots Category Response Curves for RSM, with optional overplotting
        of observed data and threshold lines.
        '''

        abilities = np.arange(-20, 20, 0.1)

        y = np.array([[self.cat_prob(ability, self.diffs[item], category, self.thresholds)
                       for category in range(self.max_score + 1)]
                      for ability in abilities])

        if obs is not None:
            if hasattr(self, 'person_abiliites') == False:
                self.person_abils(warm_corr=False)

            _, _, _, mean_abilities, obs_props = self.class_intervals_cats(self.person_abilities, item=item,
                                                                           no_of_classes=no_of_classes)

            xobsdata = mean_abilities
            yobsdata = obs_props

            if obs != 'all':
                if not all(cat in np.arange(self.max_score + 1) for cat in obs):
                    print("Invalid 'obs'. Valid values are 'None', 'all' and list of categories.")
                    return

                else:
                    yobsdata = yobsdata[:, obs]

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        if title:
            graphtitle = f'Category response curves for item {item}'
        else:
            graphtitle = ''

        ylabel = 'Probability'

        plot = self.plot_data(x_data=abilities, y_data=y, x_min=xmin, x_max=xmax, y_max=1, x_obs_data=xobsdata,
                              y_obs_data=yobsdata, items=item, graph_title=graphtitle, y_label=ylabel, obs=obs,
                              thresh_lines=thresh_lines, central_diff=central_diff, cat_highlight=cat_highlight,
                              plot_style=plot_style, black=black, font=font, title_font_size=title_font_size,
                              axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                              plot_density=dpi, file_format=file_format)

        return plot

    def threshold_ccs(self,
                      item=None,
                      obs=None,
                      xmin=-10,
                      xmax=10,
                      no_of_classes=5,
                      title=True,
                      thresh_lines=False,
                      central_diff=False,
                      cat_highlight=None,
                      plot_style='colorblind',
                      black=False,
                      font='Times',
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

        abilities = np.arange(-20, 20, 0.1)

        if item is None:
            abs_thresholds = self.thresholds[1:]
        else:
            abs_thresholds = self.thresholds[1:] + self.diffs[item]
        y = np.array([[1 / (1 + np.exp(threshold - ability))
                       for threshold in abs_thresholds]
                      for ability in abilities])

        if obs is not None:
            if hasattr(self, 'person_abiliites') == False:
                self.person_abils(warm_corr=False)

            _, _, _, mean_abilities, obs_props = self.class_intervals_thresholds(item=item, no_of_classes=no_of_classes)

            xobsdata = mean_abilities
            yobsdata = obs_props

            if obs != 'all':
                if not all(cat in np.arange(self.max_score) + 1 for cat in obs):
                    print("Invalid 'obs'. Valid values are 'None', 'all' and list of categories.")
                    return

                else:
                    obs = [ob - 1 for ob in obs]
                    xobsdata = xobsdata[:, obs]
                    yobsdata = yobsdata[:, obs]

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        if title:
            if item is None:
                graphtitle = f'Threshold characteristic curves'
            else:
                graphtitle = f'Threshold characteristic curves for item {item}'
        else:
            graphtitle = ''

        ylabel = 'Probability'

        plot = self.plot_data(x_data=abilities, y_data=y, y_max=1, x_min=xmin, x_max=xmax, items=item, obs=obs,
                              x_obs_data=xobsdata, y_obs_data=yobsdata, graph_title=graphtitle, y_label=ylabel,
                              thresh_lines=thresh_lines, central_diff=central_diff, cat_highlight=cat_highlight,
                              plot_style=plot_style, black=black, font=font, title_font_size=title_font_size,
                              axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                              file_format=file_format, plot_density=dpi)

        return plot

    def iic(self,
            item,
            xmin=-10,
            xmax=10,
            ymax=None,
            thresh_lines=False,
            central_diff=False,
            point_info_lines=None,
            point_info_labels=False,
            cat_highlight=None,
            title=True,
            plot_style='colorblind',
            black=False,
            font='Times',
            title_font_size=15,
            axis_font_size=12,
            labelsize=12,
            filename=None,
            file_format='png',
            dpi=300):

        '''
        Plots Item Information Curves.
        '''

        abilities = np.arange(-20, 20, 0.1)

        y = [self.variance(ability,
                           self.diffs[item],
                           self.thresholds)
             for ability in abilities]
        y = np.array(y).reshape(len(abilities), 1)

        if ymax is None:
            ymax = max(y) * 1.1

        if title:
            graphtitle = f'Item information curve for item {item}'
        else:
            graphtitle = ''

        ylabel = 'Fisher information'

        plot = self.plot_data(x_data=abilities, y_data=y, x_min=xmin, x_max=xmax, y_max=ymax, items=item,
                              thresh_lines=thresh_lines, point_info_lines_item=[item, point_info_lines],
                              score_labels=point_info_labels, cat_highlight=cat_highlight, central_diff=central_diff,
                              graph_title=graphtitle, y_label=ylabel, plot_style=plot_style, black=black, font=font,
                              title_font_size=title_font_size, axis_font_size=axis_font_size, labelsize=labelsize,
                              filename=filename, plot_density=dpi, file_format=file_format)

        return plot

    def tcc(self,
            items=None,
            obs=False,
            xmin=-10,
            xmax=10,
            no_of_classes=5,
            title=True,
            score_lines=None,
            score_labels=False,
            plot_style='colorblind',
            black=False,
            font='Times',
            title_font_size=15,
            axis_font_size=12,
            labelsize=12,
            filename=None,
            file_format='png',
            dpi=300):

        '''
        Plots Test Characteristic Curve for RSM.
        '''

        abilities = np.arange(-20, 20, 0.1)

        if items is None:
            items = list(self.dataframe.columns)
            difficulties = self.diffs

        else:
            difficulties = self.diffs.loc[items]

        y = [sum(self.exp_score(ability, difficulties[item], self.thresholds)
                 for item in items)
             for ability in abilities]
        y = np.array(y).reshape(len(abilities), 1)

        if obs:
            if hasattr(self, 'person_abiliites') == False:
                self.person_abils(warm_corr=False)

            _, _, _, mean_abilities, obs_means = self.class_intervals(self.person_abilities, items=items,
                                                                      no_of_classes=no_of_classes)

            xobsdata = mean_abilities
            yobsdata = obs_means
            yobsdata = np.array(yobsdata).reshape(no_of_classes, 1)

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Expected score'

        plot = self.plot_data(x_data=abilities, y_data=y, items=items, x_obs_data=xobsdata, y_obs_data=yobsdata,
                              x_min=xmin, x_max=xmax, y_max=self.max_score * len(difficulties),
                              score_lines_test=score_lines, score_labels=score_labels, graph_title=graphtitle,
                              y_label=ylabel, obs=obs, plot_style=plot_style, black=black, font=font,
                              title_font_size=title_font_size, axis_font_size=axis_font_size, labelsize=labelsize,
                              filename=filename, plot_density=dpi, file_format=file_format)

        return plot

    def test_info(self,
                  items=None,
                  point_info_lines=None,
                  point_info_labels=False,
                  xmin=-10,
                  xmax=10,
                  ymax=None,
                  title=True,
                  plot_style='colorblind',
                  black=False,
                  font='Times',
                  title_font_size=15,
                  axis_font_size=12,
                  labelsize=12,
                  filename=None,
                  file_format='png',
                  dpi=300):

        '''
        Plots Test Information Curve for RSM.
        '''

        if items is None:
            difficulties = self.diffs
            items = self.dataframe.columns

        else:
            difficulties = self.diffs.loc[items]

        abilities = np.arange(-20, 20, 0.1)
        y = np.array([sum(self.variance(ability, difficulty, self.thresholds)
                          for difficulty in difficulties)
                      for ability in abilities])
        y = y.reshape(len(abilities), 1)

        if ymax is None:
            ymax = max(y) * 1.1

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Fisher information'

        plot = self.plot_data(x_data=abilities, y_data=y, items=items, x_min=xmin, x_max=xmax, y_max=ymax,
                              graph_title=graphtitle, point_info_lines_test=point_info_lines,
                              score_labels=point_info_labels, y_label=ylabel, plot_style=plot_style, black=black,
                              font=font, title_font_size=title_font_size, axis_font_size=axis_font_size,
                              labelsize=labelsize, filename=filename, plot_density=dpi, file_format=file_format)

        return plot

    def test_csem(self,
                  items=None,
                  point_csem_lines=None,
                  point_csem_labels=False,
                  xmin=-10,
                  xmax=10,
                  ymax=5,
                  title=True,
                  plot_style='colorblind',
                  black=False,
                  font='Times',
                  title_font_size=15,
                  axis_font_size=12,
                  labelsize=12,
                  filename=None,
                  file_format='png',
                  dpi=300):

        '''
        Plots Test Conditional Standard Error of Measurement Curve for RSM.
        '''

        if items is None:
            difficulties = self.diffs
            items = self.dataframe.columns

        else:
            difficulties = self.diffs.loc[items]

        abilities = np.arange(-20, 20, 0.1)
        y = np.array([sum(self.variance(ability, difficulty, self.thresholds)
                          for difficulty in difficulties)
                      for ability in abilities])
        y = 1 / np.sqrt(y)
        y = y.reshape(len(abilities), 1)

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Conditional SEM'

        plot = self.plot_data(x_data=abilities, y_data=y, items=items, x_min=xmin, x_max=xmax, y_max=ymax,
                              graph_title=graphtitle, point_csem_lines=point_csem_lines, score_labels=point_csem_labels,
                              y_label=ylabel, plot_style=plot_style, black=black, font=font,
                              title_font_size=title_font_size, axis_font_size=axis_font_size, labelsize=labelsize,
                              filename=filename, plot_density=dpi, file_format=file_format)

        return plot

    def std_residuals_plot(self,
                           items=None,
                           bin_width=0.5,
                           x_min=-6,
                           x_max=6,
                           normal=False,
                           title=True,
                           plot_style='colorblind',
                           font='Times',
                           title_font_size=15,
                           axis_font_size=12,
                           labelsize=12,
                           filename=None,
                           file_format='png',
                           plot_density=300):

        '''
        Plots histogram of standardised residuals for SLM, with optional overplotting of Standard Normal Distribution.
        '''

        if items is None:
            items = self.dataframe.columns

        std_residual_df = self.std_residual_df[items]
        std_residual_list = std_residual_df.unstack().dropna()

        plot = self.std_residuals_hist(std_residual_list, bin_width=bin_width, x_min=x_min, x_max=x_max, normal=normal,
                                       title=title, plot_style=plot_style, font=font, title_font_size=title_font_size,
                                       axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                                       file_format=file_format, plot_density=plot_density)

        return plot

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
            max_scores = ((unstacked_df == unstacked_df) * self.max_score).sum(axis=1)
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

        if isinstance(new, str) == False:
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

        if all(isinstance(name, str) for name in new_names) == False:
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

        cat_prob_nums = [exp(cat * (ability - difficulties[item] - severities[rater]) -
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

        cat_prob_nums = [exp(cat * (ability - difficulties[item] - severities[rater][item]) -
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

        cat_prob_nums = [exp(cat * (ability - difficulties[item]) -
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

        cat_prob_nums = [exp(cat * (ability - difficulties[item]) -
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
                                           difficulties[item_1] - difficulties[item_2])
                    weights_sum += weight

        try:
            estimator /= weights_sum

        except:
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

        self.null_persons = []

        for person in self.persons:
            if self.dataframe.xs(person, level=1, drop_level=False).isnull().values.all().all():
                self.null_persons.append(person)

        for person in self.null_persons:
            self.dataframe = self.dataframe.drop(person, level=1)
            self.persons = self.persons.drop(person)

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

        self.null_persons = []

        for person in self.persons:
            if self.dataframe.xs(person, level=1, drop_level=False).isnull().values.all().all():
                self.null_persons.append(person)

        for person in self.null_persons:
            self.dataframe = self.dataframe.drop(person, level=1)
            self.persons = self.persons.drop(person)

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

        self.null_persons = []

        for person in self.persons:
            if self.dataframe.xs(person, level=1, drop_level=False).isnull().values.all().all():
                self.null_persons.append(person)

        for person in self.null_persons:
            self.dataframe = self.dataframe.drop(person, level=1)
            self.persons = self.persons.drop(person)

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

        self.null_persons = []

        for person in self.persons:
            if self.dataframe.xs(person, level=1, drop_level=False).isnull().values.all().all():
                self.null_persons.append(person)

        for person in self.null_persons:
            self.dataframe = self.dataframe.drop(person, level=1)
            self.persons = self.persons.drop(person)

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

        if anchor_raters != []:
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

        if anchor_raters != []:
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

        if anchor_raters != []:
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

        rater_se_marginal_items = {rater: {item: pd.DataFrame(marginal_rater_ests_items[rater]).std(axis=1)[i]
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
                                                                                                        axis=1)[i]
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
                                                                                                         axis=1)[i]
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

        if anchor_raters != []:
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
            self.anchor_diffs_items[item] += severity_adjustments[i]

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
            severities_thresholds_df.loc[rater].iloc[1:] -= severity_adjustments
        self.anchor_severities_thresholds = {rater: severities_thresholds_df.loc[rater] for rater in self.raters}

        self.anchor_thresholds_thresholds[1:] -= self.anchor_thresholds_thresholds[1:].mean()

        self.anchor_raters_thresholds = anchor_raters

    def calibrate_matrix_anchor(self,
                                anchor_raters,
                                calibrate=False,
                                constant=0.1,
                                method='cos',
                                matrix_power=3,
                                log_lik_tol=0.000001):

        if calibrate:
            self.calibrate_matrix(constant=constant, method=method,
                                  matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        self.anchor_diffs_matrix = self.diffs.copy()
        self.anchor_thresholds_matrix = self.thresholds.copy()

        severities_matrix_df = {rater: pd.DataFrame(self.severities_matrix.copy()[rater]).T
                                for rater in self.raters}
        severities_matrix_df = pd.concat(severities_matrix_df.values(), keys=severities_matrix_df.keys())

        severities_matrix_array = severities_matrix_df.values.reshape(self.no_of_raters, self.no_of_items, -1)

        anchor_severities_df = {rater: pd.DataFrame(self.severities_matrix.copy()[rater]).T
                                for rater in anchor_raters}
        anchor_severities_df = pd.concat(anchor_severities_df.values(), keys=anchor_severities_df.keys())

        anchor_severities_array = anchor_severities_df.values.reshape(len(anchor_raters),  self.no_of_items, -1)

        severity_adjustments = anchor_severities_array.mean(axis=0)
        diff_adjustments = severity_adjustments[:, 1:].mean(axis=1)
        threshold_adjustments = severity_adjustments[:, 1:].mean(axis=0)

        for i, item in enumerate(self.dataframe.columns):
            self.anchor_diffs_matrix[item] += diff_adjustments[i]

        self.anchor_thresholds_matrix[1:] += threshold_adjustments

        for rater in range(self.no_of_raters):
            severities_matrix_array[rater, :, :] -= severity_adjustments

        self.anchor_severities_matrix = {rater: {item: severities_matrix_array[i, j, :]
                                                 for j, item in enumerate(self.dataframe.columns)}
                                         for i, rater in enumerate(self.raters)}

        diff_centraliser = self.anchor_diffs_matrix.mean()
        for item in self.dataframe.columns:
            self.anchor_diffs_matrix[item] -= diff_centraliser

        self.anchor_thresholds_matrix[1:] -= self.anchor_thresholds_matrix[1:].mean()

        sev_dict = {rater: pd.DataFrame(self.anchor_severities_matrix[rater])
            for rater in self.raters}

        for rater in self.raters:
            sev_dict[rater] = sev_dict[rater].iloc[1:]

        sev_df = pd.concat(sev_dict.values(), keys=sev_dict.keys())

        marginal_items = {rater: sev_df.xs(rater).mean(axis=0) for rater in self.raters}

        marginal_thresholds = {rater: sev_df.xs(rater).mean(axis=1) for rater in self.raters}
        for rater in self.raters:
            marginal_thresholds[rater] = pd.concat([pd.Series([0]), marginal_thresholds[rater]])

        self.anchor_marginal_severities_items = marginal_items
        self.anchor_marginal_severities_thresholds = marginal_thresholds
        for rater in self.raters:
            adjustment = self.anchor_marginal_severities_thresholds[rater][1:].mean()
            self.anchor_marginal_severities_thresholds[rater][1:] -= adjustment

        self.anchor_raters_matrix = anchor_raters

    def abil_global(self,
                    person,
                    anchor=False,
                    items=None,
                    raters=None,
                    warm_corr=True,
                    tolerance=0.0000001,
                    max_iters=100,
                    ext_score_adjustment=0.5):

        if items is None:
            items = self.dataframe.columns.tolist()

        if raters is None:
            raters = self.raters.tolist()

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

        if isinstance(raters, list):
            person_data = [self.dataframe[items].xs(rater).loc[person].to_numpy()
                           for rater in raters]
            person_data = np.array(person_data)

        else:
            person_data = self.dataframe[items].xs(raters).loc[person].to_numpy()

        person_filter = (person_data + 1) / (person_data + 1)
        score = np.nansum(person_data)

        ext_score = np.nansum(person_filter) * self.max_score

        if score == 0:
            score = ext_score_adjustment

        elif score == ext_score:
            score -= ext_score_adjustment

        try:
            estimate = log(score) - log(ext_score - score) + np.mean(difficulties)

            change = 1
            iters = 0

            while (abs(change) > tolerance) & (iters <= max_iters):

                person_exp_matrix = [[self.exp_score_global(estimate, item, difficulties,
                                                            rater, severities, thresholds)
                                      for item in items]
                                     for rater in raters]
                person_exp_matrix = np.array(person_exp_matrix)
                person_exp_matrix *= person_filter
                result = np.nansum(person_exp_matrix)

                person_info_matrix = [[self.variance_global(estimate, item, difficulties,
                                                            rater, severities, thresholds)
                                      for item in items]
                                     for rater in raters]
                person_info_matrix = np.array(person_info_matrix)
                person_info_matrix *= person_filter
                info = np.nansum(person_info_matrix)

                change = max(-1, min(1, (result - score) / info))
                estimate -= change
                iters += 1

            if warm_corr:
                estimate += self.warm_global(estimate, person_filter, difficulties, thresholds, severities)

            if iters >= max_iters:
                print('Maximum iterations reached before convergence.')

        except:
            estimate = np.nan

        return estimate

    def person_abils_global(self,
                            anchor=False,
                            items=None,
                            raters=None,
                            warm_corr=True,
                            tolerance=0.0000001,
                            max_iters=100,
                            ext_score_adjustment=0.5):

        '''
        Creates raw score to ability estimate look-up table. Newton-Raphson ML
        estimation, includes optional Warm (1989) bias correction.
        '''

        if items is None:
            items = self.dataframe.columns.tolist()

        if raters is None:
            raters = self.raters.tolist()

        if anchor:
            if hasattr(self, 'anchor_diffs_global') == False:
                print('Anchor calibration required')
                return

        estimates = [self.abil_global(person, anchor=anchor, items=items, raters=raters, warm_corr=warm_corr,
                                      tolerance=tolerance, max_iters=max_iters,
                                      ext_score_adjustment=ext_score_adjustment)
                     for person in self.persons]
        estimates = {person: estimate for person, estimate in zip(self.persons, estimates)}

        if anchor:
            self.anchor_abils_global = pd.Series(estimates)

        else:
            self.abils_global = pd.Series(estimates)

    def score_abil_global(self,
                          score,
                          anchor=False,
                          items=None,
                          raters=None,
                          warm_corr=True,
                          tolerance=0.0000001,
                          max_iters=100,
                          ext_score_adjustment=0.5):

        if items is None:
            items = self.dataframe.columns.tolist()

        if raters == 'all':
            raters = self.raters.tolist()

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

        if isinstance(raters, str):
            raters = [raters]

        if isinstance(items, str):
            items = [items]

        if raters is None:
            person_filter = np.array([1 for item in items])

        else:
            person_filter = np.array([[1 for item in items]
                                      for rater in raters])

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
                dummy_sevs = pd.Series({'dummy_rater': 0})

                exp_list = [self.exp_score_global(estimate, item, difficulties, 'dummy_rater',
                                                  dummy_sevs, thresholds)
                            for item in items]

                info_list = [self.variance_global(estimate, item, difficulties, 'dummy_rater',
                                                  dummy_sevs, thresholds)
                             for item in items]

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
            sevs = severities[raters]
            estimate += self.warm_global(estimate, person_filter, difficulties, thresholds, sevs)

        if iters >= max_iters:
            print('Maximum iterations reached before convergence.')

        return estimate

    def abil_lookup_table_global(self,
                                 anchor=False,
                                 items=None,
                                 raters=None,
                                 warm_corr=True,
                                 tolerance=0.0000001,
                                 max_iters=100,
                                 ext_score_adjustment=0.5):

        if items is None:
            if raters is None:
                person_filter = np.array([1 for item in self.dataframe.columns])

            else:
                person_filter = np.array([[1 for item in self.dataframe.columns]
                                          for rater in raters])

        else:
            if raters is None:
                person_filter = np.array([1 for item in items])

            else:
                person_filter = np.array([[1 for item in items]
                                          for rater in raters])

        ext_score = person_filter.sum() * self.max_score

        abil_table = {score: self.score_abil_global(score, anchor=anchor, items=items, raters=raters,
                                                    warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                                    ext_score_adjustment=ext_score_adjustment)
                      for score in range(ext_score + 1)}

        self.abil_table_global = pd.Series(abil_table)

    def warm_global(self,
                    estimate,
                    person_filter,
                    difficulties,
                    thresholds,
                    severities):

        '''
        Warm's (1989) bias correction for ML abiity estimates
        '''

        exp_matrix = [[self.exp_score_global(estimate, item, difficulties, rater, severities, thresholds)
                       for item in difficulties.keys()]
                      for rater in severities.keys()]
        exp_matrix = np.array(exp_matrix)
        exp_matrix *= person_filter

        info_matrix = [[self.variance_global(estimate, item, difficulties, rater, severities, thresholds)
                        for item in difficulties.keys()]
                       for rater in severities.keys()]
        info_matrix = np.array(info_matrix)
        info_matrix *= person_filter

        cat_prob_dict = {category + 1: [[self.cat_prob_global(estimate, item, difficulties, rater, severities,
                                                              category + 1, thresholds)
                                         for item in difficulties.keys()]
                                        for rater in severities.keys()]
                         for category in range(self.max_score)}

        for category in range(self.max_score):
            cat_prob_dict[category + 1] = np.array(cat_prob_dict[category + 1])
            cat_prob_dict[category + 1] *= person_filter

        part_1 = sum(((category + 1) ** 3) * np.nansum(cat_prob_dict[category + 1])
                     for category in range(self.max_score))

        part_2 = 3 * np.nansum((info_matrix + (exp_matrix ** 2)) * exp_matrix)

        part_3 = np.nansum(2 * (exp_matrix ** 3))

        warm_correction = 0.5 * (part_1 - part_2 + part_3) / (np.nansum(info_matrix) ** 2)

        return warm_correction

    def csem_global(self,
                    person,
                    anchor=False,
                    items=None,
                    raters=None,
                    warm_corr=True,
                    tolerance=0.0000001,
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

        ability = self.abil_global(person, anchor=anchor, items=items, raters=raters, warm_corr=warm_corr,
                                   tolerance=tolerance, max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)

        info_list = [self.variance_global(ability, item, difficulties, rater, severities, thresholds)
                     for item in items for rater in raters
                     if self.dataframe.loc[(rater, person), item] == self.dataframe.loc[(rater, person), item]]

        total_info = sum(info_list)
        csem = 1 / np.sqrt(total_info)

        return csem

    def abil_items(self,
                   person,
                   anchor=False,
                   items=None,
                   raters=None,
                   warm_corr=True,
                   tolerance=0.0000001,
                   max_iters=100,
                   ext_score_adjustment=0.5):

        if items is None:
            items = self.dataframe.columns.tolist()

        if raters is None:
            raters = self.raters.tolist()

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

        if isinstance(raters, list):
            person_data = [self.dataframe[items].xs(rater).loc[person].to_numpy()
                           for rater in raters]
            person_data = np.array(person_data)

        else:
            person_data = self.dataframe[items].xs(raters).loc[person].to_numpy()

        person_filter = (person_data + 1) / (person_data + 1)
        score = np.nansum(person_data)

        ext_score = np.nansum(person_filter) * self.max_score

        if score == 0:
            score = ext_score_adjustment

        elif score == ext_score:
            score -= ext_score_adjustment
        try:
            estimate = log(score) - log(ext_score - score) + np.mean(difficulties)

            change = 1
            iters = 0

            while (abs(change) > tolerance) & (iters <= max_iters):

                person_exp_matrix = [[self.exp_score_items(estimate, item, difficulties,
                                                           rater, severities, thresholds)
                                      for item in items]
                                     for rater in raters]
                person_exp_matrix = np.array(person_exp_matrix)
                person_exp_matrix *= person_filter
                result = np.nansum(person_exp_matrix)

                person_info_matrix = [[self.variance_items(estimate, item, difficulties,
                                                           rater, severities, thresholds)
                                       for item in items]
                                      for rater in raters]
                person_info_matrix = np.array(person_info_matrix)
                person_info_matrix *= person_filter
                info = np.nansum(person_info_matrix)

                change = max(-1, min(1, (result - score) / info))
                estimate -= change
                iters += 1

            if warm_corr:
                estimate += self.warm_items(estimate, person_filter, difficulties, thresholds, severities)

            if iters >= max_iters:
                print('Maximum iterations reached before convergence.')

        except:
            estimate = np.nan

        return estimate

    def person_abils_items(self,
                           anchor=False,
                           items=None,
                           raters=None,
                           warm_corr=True,
                           tolerance=0.0000001,
                           max_iters=100,
                           ext_score_adjustment=0.5):

        '''
        Creates raw score to ability estimate look-up table. Newton-Raphson ML
        estimation, includes optional Warm (1989) bias correction.
        '''

        if items is None:
            items = self.dataframe.columns.tolist()

        if raters is None:
            raters = self.raters.tolist()

        if anchor:
            if hasattr(self, 'anchor_diffs_global') == False:
                print('Anchor calibration required')
                return

        estimates = [self.abil_items(person, anchor=anchor, items=items, raters=raters, warm_corr=warm_corr,
                                     tolerance=tolerance, max_iters=max_iters,
                                     ext_score_adjustment=ext_score_adjustment)
                     for person in self.persons]

        estimates = {person: estimate for person, estimate in zip(self.persons, estimates)}

        if anchor:
            self.anchor_abils_items = pd.Series(estimates)

        else:
            self.abils_items = pd.Series(estimates)

    def score_abil_items(self,
                         score,
                         anchor=False,
                         items=None,
                         raters=None,
                         warm_corr=True,
                         tolerance=0.0000001,
                         max_iters=100,
                         ext_score_adjustment=0.5):

        if items is None:
            items = self.dataframe.columns

        if raters == 'all':
            raters = self.raters.tolist()

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

        if isinstance(raters, str):
            raters = [raters]

        if isinstance(items, str):
            items = [items]

        if raters is None:
            person_filter = np.array([1 for item in items])

        else:
            person_filter = np.array([[1 for item in items]
                                      for rater in raters])

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
                dummy_sevs = {'dummy_rater': {item: 0 for item in self.dataframe.columns}}

                exp_list = [self.exp_score_items(estimate, item, difficulties, 'dummy_rater',
                                                 dummy_sevs, thresholds)
                            for item in items]

                info_list = [self.variance_items(estimate, item, difficulties, 'dummy_rater',
                                                 dummy_sevs, thresholds)
                             for item in items]

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
            if raters is None:
                dummy_sevs = {'dummy_rater': {item: 0 for item in self.dataframe.columns}}
                estimate += self.warm_items(estimate, person_filter, difficulties, thresholds, dummy_sevs)

            else:
                sevs = dict((rater, severities[rater]) for rater in raters)
                estimate += self.warm_items(estimate, person_filter, difficulties, thresholds, sevs)

        if iters >= max_iters:
            print('Maximum iterations reached before convergence.')

        return estimate

    def abil_lookup_table_items(self,
                                anchor=False,
                                items=None,
                                raters=None,
                                warm_corr=True,
                                tolerance=0.0000001,
                                max_iters=100,
                                ext_score_adjustment=0.5):

        if items is None:
            if raters is None:
                person_filter = np.array([1 for item in self.dataframe.columns])

            else:
                person_filter = np.array([[1 for item in self.dataframe.columns]
                                          for rater in raters])

        else:
            if raters is None:
                person_filter = np.array([1 for item in items])

            else:
                person_filter = np.array([[1 for item in items]
                                          for rater in raters])

        ext_score = person_filter.sum() * self.max_score

        abil_table = {score: self.score_abil_items(score, anchor=anchor, items=items, raters=raters,
                                                   warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                                   ext_score_adjustment=ext_score_adjustment)
                      for score in range(ext_score + 1)}

        self.abil_table_items = pd.Series(abil_table)

    def warm_items(self,
                   estimate,
                   person_filter,
                   difficulties,
                   thresholds,
                   severities):

        '''
        Warm's (1989) bias correction for ML abiity estimates
        '''

        exp_matrix = [[self.exp_score_items(estimate, item, difficulties, rater, severities, thresholds)
                       for item in difficulties.keys()]
                      for rater in severities.keys()]
        exp_matrix = np.array(exp_matrix)
        exp_matrix *= person_filter

        info_matrix = [[self.variance_items(estimate, item, difficulties, rater, severities, thresholds)
                        for item in difficulties.keys()]
                       for rater in severities.keys()]
        info_matrix = np.array(info_matrix)
        info_matrix *= person_filter

        cat_prob_dict = {category + 1: [[self.cat_prob_items(estimate, item, difficulties, rater,
                                                             severities, category + 1, thresholds)
                                         for item in difficulties.keys()]
                                        for rater in severities.keys()]
                         for category in range(self.max_score)}

        for category in range(self.max_score):
            cat_prob_dict[category + 1] = np.array(cat_prob_dict[category + 1])
            cat_prob_dict[category + 1] *= person_filter

        part_1 = sum(((category + 1) ** 3) * np.nansum(cat_prob_dict[category + 1])
                     for category in range(self.max_score))

        part_2 = 3 * np.nansum((info_matrix + (exp_matrix ** 2)) * exp_matrix)

        part_3 = np.nansum(2 * (exp_matrix ** 3))

        warm_correction = 0.5 * (part_1 - part_2 + part_3) / (np.nansum(info_matrix) ** 2)

        return warm_correction

    def csem_items(self,
                   person,
                   anchor=False,
                   items=None,
                   raters=None,
                   warm_corr=True,
                   tolerance=0.0000001,
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

        ability = self.abil_items(person, anchor=anchor, items=items, raters=raters, warm_corr=warm_corr,
                                   tolerance=tolerance, max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)

        info_list = [self.variance_items(ability, item, difficulties, rater, severities, thresholds)
                     for item in items for rater in raters
                     if self.dataframe.loc[(rater, person), item] == self.dataframe.loc[(rater, person), item]]

        total_info = sum(info_list)
        csem = 1 / np.sqrt(total_info)

        return csem

    def abil_thresholds(self,
                        person,
                        anchor=False,
                        items=None,
                        raters=None,
                        warm_corr=True,
                        tolerance=0.0000001,
                        max_iters=100,
                        ext_score_adjustment=0.5):

        if items is None:
            items = self.dataframe.columns

        if raters is None:
            raters = self.raters

        if anchor:
            if hasattr(self, 'anchor_diffs_global'):
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
        severities = severities[raters]

        person_data = self.dataframe[items].loc[raters].xs(person, level=1, drop_level=False).to_numpy()
        person_filter = (person_data + 1) / (person_data + 1)
        score = np.nansum(person_data)

        ext_score = np.nansum(person_filter) * self.max_score

        if score == 0:
            score = ext_score_adjustment

        elif score == ext_score:
            score -= ext_score_adjustment

        try:
            estimate = log(score) - log(ext_score - score) + np.mean(difficulties)

            change = 1
            iters = 0

            while (abs(change) > tolerance) & (iters <= max_iters):

                person_exp_matrix = [[self.exp_score_thresholds(estimate, item, difficulties, rater, severities,
                                                                thresholds)
                                      for item in difficulties.keys()]
                                     for rater in raters]
                person_exp_matrix = np.array(person_exp_matrix)
                person_exp_matrix *= person_filter
                result = np.nansum(person_exp_matrix)

                person_info_matrix = [[self.variance_thresholds(estimate, item, difficulties, rater, severities,
                                                                thresholds)
                                       for item in difficulties.keys()]
                                      for rater in raters]
                person_info_matrix = np.array(person_info_matrix)
                person_info_matrix *= person_filter
                info = np.nansum(person_info_matrix)

                change = max(-1, min(1, (result - score) / info))
                estimate -= change
                iters += 1

            if warm_corr:
                estimate += self.warm_thresholds(estimate, person_filter, difficulties, thresholds, severities)

            if iters >= max_iters:
                print('Maximum iterations reached before convergence.')

        except:
            estimate = np.nan

        return estimate

    def person_abils_thresholds(self,
                                anchor=False,
                                difficulties=None,
                                thresholds=None,
                                raters=None,
                                severities=None,
                                warm_corr=True,
                                tolerance=0.0000001,
                                max_iters=100,
                                ext_score_adjustment=0.5):

        '''
        Creates raw score to ability estimate look-up table. Newton-Raphson ML
        estimation, includes optional Warm (1989) bias correction.
        '''

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
        severities = severities[raters]

        estimates = [self.abil_thresholds(person, items=items, raters=raters, warm_corr=warm_corr, tolerance=tolerance,
                                          max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)
                     for person in self.persons]

        estimates = {person: estimate for person, estimate in zip(self.persons, estimates)}

        if anchor:
            self.anchor_abils_thresholds = pd.Series(estimates)

        else:
            self.abils_thresholds = pd.Series(estimates)

    def score_abil_thresholds(self,
                              score,
                              anchor=False,
                              items=None,
                              raters=None,
                              warm_corr=True,
                              tolerance=0.0000001,
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

            exp_list = [self.exp_score_thresholds(estimate, item, difficulties, rater, severities, thresholds)
                        for item in difficulties.keys()
                        for rater in raters]
            exp_list = np.array(exp_list)
            result = exp_list.sum()

            info_list = [self.variance_thresholds(estimate, item, difficulties, rater, severities, thresholds)
                         for item in difficulties.keys()
                         for rater in raters]
            info_list = np.array(info_list)
            info = info_list.sum()

            change = max(-1, min(1, (result - score) / info))
            estimate -= change
            iters += 1

        if warm_corr:
            sevs = dict((rater, severities[rater]) for rater in raters)
            estimate += self.warm_thresholds(estimate, person_filter, difficulties, thresholds, sevs)

        if iters >= max_iters:
            print('Maximum iterations reached before convergence.')

        return estimate

    def abil_lookup_table_thresholds(self,
                                     anchor=False,
                                     raters=[],
                                     difficulties=None,
                                     thresholds=None,
                                     severities=None,
                                     warm_corr=True,
                                     tolerance=0.0000001,
                                     max_iters=100,
                                     ext_score_adjustment=0.5):

        if anchor:
            if hasattr(self, 'anchor_diffs_thresholds') == False:
                print('Anchor calibration required')
                return None

        if (raters == []) | (raters == 'none'):
            raters = ['dummy_rater']
            severities = None

        if raters == 'all':
            raters = list(self.raters)

            if anchor:
                severities = self.anchor_severities_thresholds
            else:
                severities = self.severities_thresholds

        if difficulties is None:
            difficulties = self.diffs

        if thresholds is None:
            thresholds = self.thresholds

        if severities is None:
            if raters == ['dummy_rater']:
                severities = {'dummy_rater': np.zeros(self.max_score + 1)}
            else:

                if anchor:
                    severities = self.anchor_severities_thresholds
                else:
                    severities = self.severities_thresholds

        ext_score = len(difficulties) * self.max_score

        abil_table = {score: self.score_abil_thresholds(score, anchor=anchor, raters=raters, difficulties=difficulties,
                                                        thresholds=thresholds, severities=severities, warm_corr=warm_corr,
                                                        tolerance=tolerance, max_iters=max_iters,
                                                        ext_score_adjustment=ext_score_adjustment)
                      for score in range(ext_score + 1)}
        abil_table = pd.Series(abil_table)

        self.abil_table_thresholds = abil_table

    def warm_thresholds(self,
                        estimate,
                        person_filter,
                        difficulties,
                        thresholds,
                        severities):

        '''
        Warm's (1989) bias correction for ML abiity estimates
        '''

        exp_matrix = [[self.exp_score_thresholds(estimate, item, difficulties, rater, severities, thresholds)
                       for item in difficulties.keys()]
                      for rater in severities.keys()]
        exp_matrix = np.array(exp_matrix)
        exp_matrix *= person_filter

        info_matrix = [[self.variance_thresholds(estimate, item, difficulties, rater, severities, thresholds)
                        for item in difficulties.keys()]
                       for rater in severities.keys()]
        info_matrix = np.array(info_matrix)
        info_matrix *= person_filter

        cat_prob_dict = {category + 1: [[self.cat_prob_thresholds(estimate, item, difficulties, rater, severities,
                                                                  category + 1, thresholds)
                                         for item in difficulties.keys()]
                                        for rater in severities.keys()]
                         for category in range(self.max_score)}

        for category in range(self.max_score):
            cat_prob_dict[category + 1] = np.array(cat_prob_dict[category + 1])
            cat_prob_dict[category + 1] *= person_filter

        part_1 = sum(((category + 1) ** 3) * np.nansum(cat_prob_dict[category + 1])
                     for category in range(self.max_score))

        part_2 = 3 * np.nansum((info_matrix + (exp_matrix ** 2)) * exp_matrix)

        part_3 = np.nansum(2 * (exp_matrix ** 3))

        warm_correction = 0.5 * (part_1 - part_2 + part_3) / (np.nansum(info_matrix) ** 2)

        return warm_correction

    def csem_thresholds(self,
                        person,
                        anchor=False,
                        items=None,
                        raters=None,
                        warm_corr=True,
                        tolerance=0.0000001,
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
        severities = severities[raters]

        ability = self.abil_thresholds(person, anchor=anchor, items=items, raters=raters, warm_corr=warm_corr,
                                   tolerance=tolerance, max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)

        info_list = [self.variance_thresholds(ability, item, difficulties, rater, severities, thresholds)
                     for item in items for rater in raters
                     if self.dataframe.loc[(rater, person), item] == self.dataframe.loc[(rater, person), item]]

        total_info = sum(info_list)
        csem = 1 / np.sqrt(total_info)

        return csem

    def abil_matrix(self,
                    person,
                    anchor=False,
                    items=None,
                    raters=None,
                    warm_corr=True,
                    tolerance=0.0000001,
                    max_iters=100,
                    ext_score_adjustment=0.5):

        if items is None:
            items = self.dataframe.columns

        if raters is None:
            raters = self.raters

        if anchor:
            if hasattr(self, 'anchor_diffs_global'):
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
        severities = severities[raters]

        person_data = self.dataframe[items].loc[raters].xs(person, level=1, drop_level=False).to_numpy()
        person_filter = (person_data + 1) / (person_data + 1)
        score = np.nansum(person_data)

        ext_score = np.nansum(person_filter) * self.max_score

        if score == 0:
            score = ext_score_adjustment

        elif score == ext_score:
            score -= ext_score_adjustment

        try:
            estimate = log(score) - log(ext_score - score) + np.mean(difficulties)

            change = 1
            iters = 0

            while (abs(change) > tolerance) & (iters <= max_iters):

                person_exp_matrix = [[self.exp_score_matrix(estimate, item, difficulties, rater, severities, thresholds)
                                      for item in difficulties.keys()]
                                     for rater in raters]
                person_exp_matrix = np.array(person_exp_matrix)
                person_exp_matrix *= person_filter
                result = np.nansum(person_exp_matrix)

                person_info_matrix = [[self.variance_matrix(estimate, item, difficulties, rater, severities, thresholds)
                                       for item in difficulties.keys()]
                                      for rater in raters]
                person_info_matrix = np.array(person_info_matrix)
                person_info_matrix *= person_filter
                info = np.nansum(person_info_matrix)

                change = max(-1, min(1, (result - score) / info))
                estimate -= change
                iters += 1

            if warm_corr:
                estimate += self.warm_matrix(estimate, person_filter, difficulties, thresholds, severities)

            if iters >= max_iters:
                print('Maximum iterations reached before convergence.')

        except:
            estimate = np.nan

        return estimate

    def person_abils_matrix(self,
                            anchor=False,
                            items=None,
                            raters=None,
                            warm_corr=True,
                            tolerance=0.0000001,
                            max_iters=100,
                            ext_score_adjustment=0.5):

        '''
        Creates raw score to ability estimate look-up table. Newton-Raphson ML
        estimation, includes optional Warm (1989) bias correction.
        '''

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
            severities = severities[raters]

        estimates = [self.abil_matrix(person, items=items, raters=raters, warm_corr=warm_corr, tolerance=tolerance,
                                      max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)
                     for person in self.persons]

        estimates = {person: estimate for person, estimate in zip(self.persons, estimates)}

        if anchor:
            self.anchor_abils_matrix = pd.Series(estimates)

        else:
            self.abils_matrix = pd.Series(estimates)

    def score_abil_matrix(self,
                          score,
                          anchor=False,
                          items=None,
                          raters=None,
                          warm_corr=True,
                          tolerance=0.0000001,
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

            exp_list = [self.exp_score_matrix(estimate, item, difficulties, rater, severities, thresholds)
                        for item in difficulties.keys()
                        for rater in raters]
            exp_list = np.array(exp_list)
            result = exp_list.sum()

            info_list = [self.variance_matrix(estimate, item, difficulties, rater, severities, thresholds)
                         for item in difficulties.keys()
                         for rater in raters]
            info_list = np.array(info_list)
            info = info_list.sum()

            change = max(-1, min(1, (result - score) / info))
            estimate -= change
            iters += 1

        if warm_corr:
            sevs = dict((rater, severities[rater]) for rater in raters)
            estimate += self.warm_matrix(estimate, person_filter, difficulties, thresholds, sevs)

        if iters >= max_iters:
            print('Maximum iterations reached before convergence.')

        return estimate

    def abil_lookup_table_matrix(self,
                                 anchor=False,
                                 raters=[],
                                 difficulties=None,
                                 thresholds=None,
                                 severities=None,
                                 warm_corr=True,
                                 tolerance=0.0000001,
                                 max_iters=100,
                                 ext_score_adjustment=0.5):

        if anchor:
            if hasattr(self, 'anchor_diffs_matrix') == False:
                print('Anchor calibration required')
                return None

        if (raters == []) | (raters == 'none'):
            raters = ['dummy_rater']
            severities = None

        if raters == 'all':
            raters = list(self.raters)

            if anchor:
                severities = self.anchor_severities_matrix
            else:
                severities = self.severities_matrix

        if difficulties is None:
            difficulties = self.diffs

        if thresholds is None:
            thresholds = self.thresholds

        if severities is None:
            if raters == ['dummy_rater']:
                severities = {'dummy_rater': {item: np.zeros(self.max_score + 1)
                							  for item in self.dataframe.columns}}
            else:

                if anchor:
                    severities = self.anchor_severities_matrix
                else:
                    severities = self.severities_matrix

        ext_score = len(difficulties) * self.max_score

        abil_table = {score: self.score_abil_matrix(score, anchor=anchor, raters=raters, difficulties=difficulties,
                                                    thresholds=thresholds, severities=severities, warm_corr=warm_corr,
                                                    tolerance=tolerance, max_iters=max_iters,
                                                    ext_score_adjustment=ext_score_adjustment)
                      for score in range(ext_score + 1)}
        abil_table = pd.Series(abil_table)

        self.abil_table_matrix = abil_table

    def warm_matrix(self,
                    estimate,
                    person_filter,
                    difficulties,
                    thresholds,
                    severities):

        '''
        Warm's (1989) bias correction for ML abiity estimates
        '''

        exp_matrix = [[self.exp_score_matrix(estimate, item, difficulties, rater, severities, thresholds)
                       for item in difficulties.keys()]
                      for rater in severities.keys()]
        exp_matrix = np.array(exp_matrix)
        exp_matrix *= person_filter

        info_matrix = [[self.variance_matrix(estimate, item, difficulties, rater, severities, thresholds)
                        for item in difficulties.keys()]
                       for rater in severities.keys()]
        info_matrix = np.array(info_matrix)
        info_matrix *= person_filter

        cat_prob_dict = {category + 1: [[self.cat_prob_matrix(estimate, item, difficulties, rater,
                                                              severities, category + 1, thresholds)
                                         for item in difficulties.keys()]
                                        for rater in severities.keys()]
                         for category in range(self.max_score)}

        for category in range(self.max_score):
            cat_prob_dict[category + 1] = np.array(cat_prob_dict[category + 1])
            cat_prob_dict[category + 1] *= person_filter

        part_1 = sum(((category + 1) ** 3) * np.nansum(cat_prob_dict[category + 1])
                     for category in range(self.max_score))

        part_2 = 3 * np.nansum((info_matrix + (exp_matrix ** 2)) * exp_matrix)

        part_3 = np.nansum(2 * (exp_matrix ** 3))

        warm_correction = 0.5 * (part_1 - part_2 + part_3) / (np.nansum(info_matrix) ** 2)

        return warm_correction

    def csem_matrix(self,
                    person,
                    anchor=False,
                    items=None,
                    raters=None,
                    warm_corr=True,
                    tolerance=0.0000001,
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
        severities = severities[raters]

        ability = self.abil_matrix(person, anchor=anchor, items=items, raters=raters, warm_corr=warm_corr,
                                   tolerance=tolerance, max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)

        info_list = [self.variance_matrix(ability, item, difficulties, rater, severities, thresholds)
                     for item in items for rater in raters
                     if self.dataframe.loc[(rater, person), item] == self.dataframe.loc[(rater, person), item]]

        total_info = sum(info_list)
        csem = 1 / np.sqrt(total_info)

        return csem

    def category_counts_item(self,
                             item,
                             rater=None):

        if item in self.dataframe.columns:

            if rater is None:
                return self.dataframe.apply(pd.value_counts)[item][:self.max_score + 1].fillna(0).astype(int)

            else:
                if rater in self.raters:
                    return self.dataframe.xs(rater).apply(pd.value_counts)[item][:self.max_score + 1].fillna(0).astype(int)

                else:
                    print('Invalid rater name')

        else:
            print('Invalid item name')

    def category_counts_df(self):

        category_counts_df = pd.DataFrame(0, index=self.dataframe.columns, columns=np.arange(self.max_score + 1))

        for item in self.dataframe.columns:
            for score, count in self.category_counts_item(item).items():
                category_counts_df.loc[item].iloc[int(score)] = count

        category_counts_df['Total'] = self.dataframe.count()
        category_counts_df['Missing'] = self.dataframe.shape[0] - category_counts_df['Total']

        category_counts_df = category_counts_df.astype(int)
        category_counts_df.loc['Total'] = category_counts_df.sum()

        self.category_counts = category_counts_df

        self.category_counts_raters = {}

        for rater in self.raters:

            category_counts_df = pd.DataFrame(0, index=self.dataframe.columns, columns=np.arange(self.max_score + 1))

            for item in self.dataframe.columns:
                for score, count in self.category_counts_item(item, rater).items():
                    category_counts_df.loc[item].iloc[int(score)] = count

            category_counts_df['Total'] = self.dataframe.xs(rater).count()
            category_counts_df['Missing'] = len(self.dataframe.xs(rater).index) - category_counts_df['Total']

            category_counts_df = category_counts_df.astype(int)

            category_counts_df.loc['Total']= category_counts_df.sum()

            self.category_counts_raters[rater] = category_counts_df

        self.category_counts_raters = pd.concat(self.category_counts_raters.values(),
                                                keys=self.category_counts_raters.keys())

    def item_stats_df_global(self,
                             anchor_raters=None,
                             full=False,
                             zstd=False,
                             point_measure_corr=False,
                             dp=3,
                             warm_corr=True,
                             tolerance=0.0000001,
                             max_iters=100,
                             ext_score_adjustment=0.5,
                             method='cos',
                             constant=0.1,
                             no_of_samples=100,
                             interval=None):

        if full:
            zstd = True
            point_measure_corr = True

            if interval is None:
                interval = 0.95

        if anchor_raters is not None:
            if (hasattr(self, 'anchor_severites_global') == False) or (self.anchor_raters_global != anchor_raters):
                self.calibrate_global_anchor(anchor_raters, constant=constant, method=method)
                self.std_errors_global(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method)

            elif (self.anchor_item_low_global is None) and (interval is not None):
                self.std_errors_global(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method)
        else:
            if hasattr(self, 'anchor_item_se_global') == False:
                self.std_errors_global(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method)

            elif (self.item_low is None) and (interval is not None):
                self.std_errors_global(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method)

        if hasattr(self, 'item_outfit_ms_global') == False:
            self.item_fit_statistics_global(anchor_raters=anchor_raters, warm_corr=warm_corr, tolerance=tolerance,
                                            max_iters=max_iters, ext_score_adjustment=ext_score_adjustment,
                                            method=method, constant=constant, no_of_samples=no_of_samples,
                                            interval=interval)

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

        self.item_stats_global['Estimate'] = difficulties.round(dp)                                                                             .round(dp)
        self.item_stats_global['SE'] = std_errors.to_numpy().round(dp)

        if interval is not None:
            self.item_stats_global[f'{round((1 - interval) * 50, 1)}%'] = low.to_numpy().round(dp)
            self.item_stats_global[f'{round((1 + interval) * 50, 1)}%'] = high.to_numpy().round(dp)

        self.item_stats_global['Count'] = self.response_counts.to_numpy().astype(int)
        self.item_stats_global['Facility'] = self.item_facilities.to_numpy().round(dp)

        self.item_stats_global['Infit MS'] = self.item_infit_ms_global.to_numpy().round(dp)
        if zstd:
            self.item_stats_global['Infit Z'] = self.item_infit_zstd_global.to_numpy().round(dp)

        self.item_stats_global['Outfit MS'] = self.item_outfit_ms_global.to_numpy().round(dp)
        if zstd:
            self.item_stats_global['Outfit Z'] = self.item_outfit_zstd_global.to_numpy().round(dp)

        if point_measure_corr:
            self.item_stats_global['PM corr'] = self.point_measure_global.to_numpy().round(dp)
            self.item_stats_global['Exp PM corr'] = self.exp_point_measure_global.to_numpy().round(dp)

        self.item_stats_global.index = self.dataframe.columns

    def threshold_stats_df_global(self,
                                  anchor_raters=None,
                                  full=False,
                                  zstd=False,
                                  disc=False,
                                  point_measure_corr=False,
                                  dp=3,
                                  warm_corr=True,
                                  tolerance=0.0000001,
                                  max_iters=100,
                                  ext_score_adjustment=0.5,
                                  method='cos',
                                  constant=0.1,
                                  no_of_samples=100,
                                  interval=None):

        if full:
            zstd = True
            disc = True
            point_measure_corr = True

            if interval is None:
                interval = 0.95

        if anchor_raters is not None:
            if (hasattr(self, 'anchor_severites_global') == False) or (self.anchor_raters_global != anchor_raters):
                self.calibrate_global_anchor(anchor_raters, constant=constant, method=method)
                self.std_errors_global(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method)

            elif (self.anchor_item_low_global is None) and (interval is not None):
                self.std_errors_global(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method)
        else:
            if hasattr(self, 'anchor_item_se_global') == False:
                self.std_errors_global(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method)

            elif (self.item_low is None) and (interval is not None):
                self.std_errors_global(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method)

        if (hasattr(self, 'threshold_outfit_ms_global') == False) or (self.anchor_raters_global != anchor_raters):
            self.threshold_fit_statistics_global(anchor_raters=anchor_raters, warm_corr=warm_corr, tolerance=tolerance,
                                                 max_iters=max_iters, ext_score_adjustment=ext_score_adjustment,
                                                 method=method, constant=constant)

        if anchor_raters is not None:
            thresholds = self.anchor_thresholds_global
        else:
            thresholds = self.thresholds

        self.threshold_stats_global = pd.DataFrame()

        self.threshold_stats_global['Estimate'] = thresholds[1:].round(dp)
        self.threshold_stats_global['SE'] = self.threshold_se_global[1:].round(dp)

        if interval is not None:
            if anchor_raters is not None:
                self.threshold_stats_global[f'{round((1 - interval) * 50, 1)}%'] = self.anchor_threshold_low_global[1:].round(dp)
                self.threshold_stats_global[f'{round((1 + interval) * 50, 1)}%'] = self.anchor_threshold_high_global[1:].round(dp)

            else:
                self.threshold_stats_global[f'{round((1 - interval) * 50, 1)}%'] = self.threshold_low_global[1:].round(dp)
                self.threshold_stats_global[f'{round((1 + interval) * 50, 1)}%'] = self.threshold_high_global[1:].round(dp)

        self.threshold_stats_global['Infit MS'] = self.threshold_infit_ms_global.to_numpy().round(dp)
        if zstd:
            self.threshold_stats_global['Infit Z'] = self.threshold_infit_zstd_global.to_numpy().round(dp)
        self.threshold_stats_global['Outfit MS'] = self.threshold_outfit_ms_global.to_numpy().round(dp)
        if zstd:
            self.threshold_stats_global['Outfit Z'] = self.threshold_outfit_zstd_global.to_numpy().round(dp)

        if disc:
            self.threshold_stats_global['Discrim'] = self.threshold_discrimination_global.to_numpy().round(dp)

        if point_measure_corr:
            self.threshold_stats_global['PM corr'] = self.threshold_point_measure_global.to_numpy().round(dp)
            self.threshold_stats_global['Exp PM corr'] = self.threshold_exp_point_measure_global.to_numpy().round(dp)

        self.threshold_stats_global.index = [f'Threshold {threshold + 1}' for threshold in range(self.max_score)]

    def rater_stats_df_global(self,
                              anchor_raters=None,
                              full=False,
                              zstd=False,
                              dp=3,
                              warm_corr=True,
                              tolerance=0.0000001,
                              max_iters=100,
                              ext_score_adjustment=0.5,
                              method='cos',
                              constant=0.1,
                              no_of_samples=100,
                              interval=None):

        if full:
            zstd = True

            if interval is None:
                interval = 0.95

        if anchor_raters is not None:
            if (hasattr(self, 'anchor_severites_global') == False) or (self.anchor_raters_global != anchor_raters):
                self.calibrate_global_anchor(anchor_raters, constant=constant, method=method)
                self.std_errors_global(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method)

        else:
            if hasattr(self, 'anchor_item_se_global') == False:
                self.std_errors_global(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method)

        if (hasattr(self, 'rater_outfit_ms_global') == False) or (self.anchor_raters_global != anchor_raters):
            self.rater_fit_statistics_global(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                             ext_score_adjustment=ext_score_adjustment, method=method,
                                             constant=constant, no_of_samples=no_of_samples, interval=interval)

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

        self.rater_stats_global['Estimate'] = severities.to_numpy().round(dp)
        self.rater_stats_global['SE'] = se.to_numpy().round(dp)

        if interval is not None:
            self.rater_stats_global[f'{round((1 - interval) * 50, 1)}%'] = low.to_numpy().round(dp)
            self.rater_stats_global[f'{round((1 + interval) * 50, 1)}%'] = high.to_numpy().round(dp)

        self.rater_stats_global['Count'] = np.array([self.dataframe.xs(rater).count().sum()
                                                     for rater in self.raters]).astype(int)

        self.rater_stats_global['Infit MS'] = self.rater_infit_ms_global.to_numpy().round(dp)
        if zstd:
            self.rater_stats_global['Infit Z'] = self.rater_infit_zstd_global.to_numpy().round(dp)
        self.rater_stats_global['Outfit MS'] = self.rater_outfit_ms_global.to_numpy().round(dp)
        if zstd:
            self.rater_stats_global['Outfit Z'] = self.rater_outfit_zstd_global.to_numpy().round(dp)

        self.rater_stats_global.index = self.raters

    def person_stats_df_global(self,
                               anchor_raters=None,
                               full=False,
                               rsem=False,
                               zstd=False,
                               dp=3,
                               warm_corr=True,
                               tolerance=0.0000001,
                               max_iters=100,
                               ext_score_adjustment=0.5,
                               method='cos',
                               constant=0.1):

        '''
        Produces a person stats dataframe with raw score, ability estimate,
        CSEM and RSEM for each person.
        '''

        if anchor_raters is not None:
            if (hasattr(self, 'anchor_severites_global') == False) or (self.anchor_raters_global != anchor_raters):
                self.calibrate_global_anchor(anchor_raters, constant=constant, method=method)
                self.std_errors_global(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method)

        if hasattr(self, 'person_outfit_ms_global') == False:
            self.person_fit_statistics_global(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                              ext_score_adjustment=ext_score_adjustment, method=method,
                                              constant=constant)

        if full:
            rsem = True
            zstd = True

        person_stats_df = pd.DataFrame()
        person_stats_df.index = self.dataframe.index.get_level_values(1).unique()

        if anchor_raters is None:
            person_stats_df['Estimate'] = self.abils_global.to_numpy().round(dp)

        else:
            person_stats_df['Estimate'] = self.anchor_abils_global.to_numpy().round(dp)

        person_stats_df['CSEM'] = self.csem_vector_global.round(dp)
        if rsem:
            person_stats_df['RSEM'] = self.rsem_vector_global.round(dp)

        person_stats_df['Score'] = [np.nan for person in self.persons]
        person_stats_df['Score'].update(self.dataframe.unstack(level=0).sum(axis=1))
        person_stats_df['Score'] = person_stats_df['Score'].astype(int)

        person_stats_df['Max score'] = [np.nan for person in self.persons]
        person_stats_df['Max score'].update(self.dataframe.unstack(level=0).count(axis=1) * self.max_score)
        person_stats_df['Max score'] = person_stats_df['Max score'].astype(int)

        person_stats_df['p'] = [np.nan for person in self.persons]
        person_stats_df['p'].update(self.dataframe.unstack(level=0).mean(axis=1) / self.max_score)
        person_stats_df['p'] = person_stats_df['p'].round(dp)

        person_stats_df['Infit MS'] = [np.nan for person in self.persons]
        person_stats_df['Infit MS'].update(self.person_infit_ms_global.round(dp))

        if zstd:
            person_stats_df['Infit Z'] = [np.nan for person in self.persons]
            person_stats_df['Infit Z'].update(self.person_infit_zstd_global.round(dp))

        person_stats_df['Outfit MS'] = [np.nan for person in self.persons]
        person_stats_df['Outfit MS'].update(self.person_outfit_ms_global.round(dp))

        if zstd:
            person_stats_df['Outfit Z'] = [np.nan for person in self.persons]
            person_stats_df['Outfit Z'].update(self.person_outfit_zstd_global.round(dp))

        self.person_stats_global = person_stats_df

    def test_stats_df_global(self,
                             dp=3,
                             warm_corr=True,
                             tolerance=0.0000001,
                             max_iters=100,
                             ext_score_adjustment=0.5,
                             method='cos',
                             constant=0.1):

        if hasattr(self, 'psi_global') == False:
            self.test_fit_statistics_global(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                            ext_score_adjustment=ext_score_adjustment, method=method,
                                            constant=constant)

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
                          tolerance=0.0000001,
                          max_iters=100,
                          ext_score_adjustment=0.5,
                          method='cos',
                          constant=0.1,
                          no_of_samples=100,
                          interval=None):

        if hasattr(self, 'item_stats_global') == False:
            self.item_stats_df_global(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                      ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                      no_of_samples=no_of_samples, interval=interval)

        if hasattr(self, 'threshold_stats_global') == False:
            self.threshold_stats_df_global(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                           ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                           no_of_samples=no_of_samples, interval=interval)

        if hasattr(self, 'rater_stats_global') == False:
            self.rater_stats_df_global(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                       ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                       no_of_samples=no_of_samples, interval=interval)

        if hasattr(self, 'person_stats_global') == False:
            self.person_stats_df_global(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                        ext_score_adjustment=ext_score_adjustment, method=method, constant=constant)

        if hasattr(self, 'test_stats_global') == False:
            self.test_stats_df_global(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                      ext_score_adjustment=ext_score_adjustment, method=method, constant=constant)

        if format == 'xlsx':

            if filename[-5:] != '.xlsx':
                filename += '.xlsx'

            writer = pd.ExcelWriter(filename, engine='xlsxwriter')

            self.item_stats_global.to_excel(writer, sheet_name='Item statistics')
            self.threshold_stats_global.to_excel(writer, sheet_name='Threshold statistics')
            self.rater_stats_global.to_excel(writer, sheet_name='Rater statistics')
            self.person_stats_global.to_excel(writer, sheet_name='Person statistics')
            self.test_stats_global.to_excel(writer, sheet_name='Test statistics')

            writer.save()

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
                            tolerance=0.0000001,
                            max_iters=100,
                            ext_score_adjustment=0.5,
                            method='cos',
                            constant=0.1,
                            no_of_samples=100,
                            interval=None):

        if full:
            zstd=True
            point_measure_corr = True

            if interval is None:
                interval = 0.95

        if anchor_raters is not None:
            if (hasattr(self, 'anchor_severites_items') == False) or (self.anchor_raters_items != anchor_raters):
                self.calibrate_items_anchor(anchor_raters, constant=constant, method=method)
                self.std_errors_items(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                      constant=constant, method=method)

            elif (self.anchor_item_low_items is None) and (interval is not None):
                self.std_errors_items(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method)

        else:
            if hasattr(self, 'anchor_item_se_items') == False:
                self.std_errors_items(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                      constant=constant, method=method)

            elif (self.item_low is None) and (interval is not None):
                self.std_errors_items(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                      constant=constant, method=method)

        if hasattr(self, 'item_outfit_ms_items') == False:
            self.item_fit_statistics_items(anchor_raters=anchor_raters, warm_corr=warm_corr, tolerance=tolerance,
                                           max_iters=max_iters, ext_score_adjustment=ext_score_adjustment, method=method,
                                           constant=constant, no_of_samples=no_of_samples, interval=interval)

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

        self.item_stats_items['Estimate'] = difficulties.to_numpy().round(dp)
        self.item_stats_items['SE'] = std_errors.to_numpy().round(dp)
        if interval is not None:
            self.item_stats_items[f'{round((1 - interval) * 50, 1)}%'] = low.to_numpy().round(dp)
            self.item_stats_items[f'{round((1 + interval) * 50, 1)}%'] = high.to_numpy().round(dp)

        self.item_stats_items['Count'] = self.response_counts.to_numpy().astype(int)
        self.item_stats_items['Facility'] = self.item_facilities.to_numpy().round(dp)

        self.item_stats_items['Infit MS'] = self.item_infit_ms_items.to_numpy().round(dp)
        if zstd:
            self.item_stats_items['Infit Z'] = self.item_infit_zstd_items.to_numpy().round(dp)
        self.item_stats_items['Outfit MS'] = self.item_outfit_ms_items.to_numpy().round(dp)
        if zstd:
            self.item_stats_items['Outfit Z'] = self.item_outfit_zstd_items.to_numpy().round(dp)

        if point_measure_corr:
            self.item_stats_items['PM corr'] = self.point_measure_items.to_numpy().round(dp)
            self.item_stats_items['Exp PM corr'] = self.exp_point_measure_items.to_numpy().round(dp)

        self.item_stats_items.index = self.dataframe.columns

    def threshold_stats_df_items(self,
                                 anchor_raters=None,
                                 full=False,
                                 zstd=False,
                                 disc=False,
                                 point_measure_corr=False,
                                 dp=3,
                                 warm_corr=True,
                                 tolerance=0.0000001,
                                 max_iters=100,
                                 ext_score_adjustment=0.5,
                                 method='cos',
                                 constant=0.1,
                                 no_of_samples=100,
                                 interval=None):

        if full:
            zstd = True
            disc = True
            point_measure_corr = True

            if interval is None:
                interval = 0.95

        if anchor_raters is not None:
            if (hasattr(self, 'anchor_severites_items') == False) or (self.anchor_raters_items != anchor_raters):
                self.calibrate_items_anchor(anchor_raters, constant=constant, method=method)
                self.std_errors_items(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                      constant=constant, method=method)

            elif (self.anchor_item_low_items is None) and (interval is not None):
                self.std_errors_items(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method)

        else:
            if hasattr(self, 'anchor_item_se_items') == False:
                self.std_errors_items(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                      constant=constant, method=method)

            elif (self.item_low is None) and (interval is not None):
                self.std_errors_items(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                      constant=constant, method=method)

        if hasattr(self, 'threshold_outfit_ms_items') == False:
            self.threshold_fit_statistics_items(anchor_raters=anchor_raters, warm_corr=warm_corr, tolerance=tolerance,
                                                max_iters=max_iters, ext_score_adjustment=ext_score_adjustment,
                                                method=method, constant=constant)

        if anchor_raters is not None:
            thresholds = self.anchor_thresholds_items
        else:
            thresholds = self.thresholds

        self.threshold_stats_items = pd.DataFrame()

        self.threshold_stats_items['Estimate'] = thresholds[1:].round(dp)
        self.threshold_stats_items['SE'] = self.threshold_se_items[1:].round(dp)

        if interval is not None:
            if anchor_raters != []:
                self.threshold_stats_items[f'{round((1 - interval) * 50, 1)}%'] = self.anchor_threshold_low_items[1:].round(dp)
                self.threshold_stats_items[f'{round((1 + interval) * 50, 1)}%'] = self.anchor_threshold_high_items[1:].round(dp)

            else:
                self.threshold_stats_items[f'{round((1 - interval) * 50, 1)}%'] = self.threshold_low_items[1:].round(dp)
                self.threshold_stats_items[f'{round((1 + interval) * 50, 1)}%'] = self.threshold_high_items[1:].round(dp)

        self.threshold_stats_items['Infit MS'] = self.threshold_infit_ms_items.to_numpy().round(dp)
        if zstd:
            self.threshold_stats_items['Infit Z'] = self.threshold_infit_zstd_items.to_numpy().round(dp)
        self.threshold_stats_items['Outfit MS'] = self.threshold_outfit_ms_items.to_numpy().round(dp)
        if zstd:
            self.threshold_stats_items['Outfit Z'] = self.threshold_outfit_zstd_items.to_numpy().round(dp)

        if disc:
            self.threshold_stats_items['Discrim'] = self.threshold_discrimination_items.to_numpy().round(dp)

        if point_measure_corr:
            self.threshold_stats_items['PM corr'] = self.threshold_point_measure_items.to_numpy().round(dp)
            self.threshold_stats_items['Exp PM corr'] = self.threshold_exp_point_measure_items.to_numpy().round(dp)

        self.threshold_stats_items.index = [f'Threshold {threshold + 1}' for threshold in range(self.max_score)]

    def rater_stats_df_items(self,
                             anchor_raters=None,
                             full=False,
                             zstd=False,
                             dp=3,
                             warm_corr=True,
                             tolerance=0.0000001,
                             max_iters=100,
                             ext_score_adjustment=0.5,
                             method='cos',
                             constant=0.1,
                             no_of_samples=100,
                             interval=None):

        if full:
            zstd = True

            if interval is None:
                interval = 0.95

        if anchor_raters is not None:
            if (hasattr(self, 'anchor_severites_items') == False) or (self.anchor_raters_items != anchor_raters):
                self.calibrate_items_anchor(anchor_raters, constant=constant, method=method)
                self.std_errors_items(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                      constant=constant, method=method)

        else:
            if hasattr(self, 'anchor_item_se_items') == False:
                self.std_errors_items(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                      constant=constant, method=method)

        if hasattr(self, 'rater_outfit_ms_items') == False:
            self.rater_fit_statistics_items(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                            ext_score_adjustment=ext_score_adjustment, method=method,
                                            constant=constant, no_of_samples=no_of_samples, interval=interval)

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

            item_stats['Estimate'] = np.array([severities[rater][item] for rater in self.raters]).round(dp)
            item_stats['SE'] = np.array([se[rater][item] for rater in self.raters]).round(dp)

            if interval is not None:
                item_stats[ f'{round((1 - interval) * 50, 1)}%'] = np.array([low[rater][item]
                                                                             for rater in self.raters]).round(dp)
                item_stats[f'{round((1 + interval) * 50, 1)}%'] = np.array([high[rater][item]
                                                                            for rater in self.raters]).round(dp)

            item_stats.index = self.raters
            self.rater_stats_items[item] = item_stats.T

        ov_stats_df = pd.DataFrame()

        ov_stats_df['Count'] = np.array([self.dataframe.xs(rater).count().sum() for rater in self.raters]).astype(int)
        ov_stats_df['Infit MS'] = self.rater_infit_ms_items.to_numpy().round(dp)
        if zstd:
            ov_stats_df['Infit Z'] = self.rater_infit_zstd_items.to_numpy().round(dp)
        ov_stats_df['Outfit MS'] = self.rater_outfit_ms_items.to_numpy().round(dp)
        if zstd:
            ov_stats_df['Outfit Z'] = self.rater_outfit_zstd_items.to_numpy().round(dp)

        ov_stats_df.index = self.raters
        self.rater_stats_items['Overall statistics'] = ov_stats_df.T

        self.rater_stats_items = pd.concat(self.rater_stats_items.values(), keys=self.rater_stats_items.keys()).T

    def person_stats_df_items(self,
                              anchor_raters=None,
                              full=False,
                              rsem=False,
                              zstd=False,
                              dp=3,
                              warm_corr=True,
                              tolerance=0.0000001,
                              max_iters=100,
                              ext_score_adjustment=0.5,
                              method='cos',
                              constant=0.1):

        '''
        Produces a person stats dataframe with raw score, ability estimate,
        CSEM and RSEM for each person.
        '''

        if hasattr(self, 'person_outfit_ms_items') == False:
            self.person_fit_statistics_items(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                             ext_score_adjustment=ext_score_adjustment, method=method,
                                             constant=constant)

        if full:
            rsem = True
            zstd = True

        person_stats_df = pd.DataFrame()
        person_stats_df.index = self.dataframe.index.get_level_values(1).unique()

        if anchor_raters is None:
            person_stats_df['Estimate'] = self.abils_items.to_numpy().round(dp)

        else:
            person_stats_df['Estimate'] = self.anchor_abils_items.to_numpy().round(dp)

        person_stats_df['CSEM'] = self.csem_vector_items.round(dp)
        if rsem:
            person_stats_df['RSEM'] = self.rsem_vector_items.round(dp)

        person_stats_df['Score'] = [np.nan for person in self.persons]
        person_stats_df['Score'].update(self.dataframe.unstack(level=0).sum(axis=1))
        person_stats_df['Score'] = person_stats_df['Score'].astype(int)

        person_stats_df['Max score'] = [np.nan for person in self.persons]
        person_stats_df['Max score'].update(self.dataframe.unstack(level=0).count(axis=1) * self.max_score)
        person_stats_df['Max score'] = person_stats_df['Max score'].astype(int)

        person_stats_df['p'] = [np.nan for person in self.persons]
        person_stats_df['p'].update(self.dataframe.unstack(level=0).mean(axis=1) / self.max_score)
        person_stats_df['p'] = person_stats_df['p'].round(dp)

        person_stats_df['Infit MS'] = [np.nan for person in self.persons]
        person_stats_df['Infit MS'].update(self.person_infit_ms_items.round(dp))

        if zstd:
            person_stats_df['Infit Z'] = [np.nan for person in self.persons]
            person_stats_df['Infit Z'].update(self.person_infit_zstd_items.round(dp))

        person_stats_df['Outfit MS'] = [np.nan for person in self.persons]
        person_stats_df['Outfit MS'].update(self.person_outfit_ms_items.round(dp))

        if zstd:
            person_stats_df['Outfit Z'] = [np.nan for person in self.persons]
            person_stats_df['Outfit Z'].update(self.person_outfit_zstd_items.round(dp))

        self.person_stats_items = person_stats_df

    def test_stats_df_items(self,
                            dp=3,
                            warm_corr=True,
                            tolerance=0.0000001,
                            max_iters=100,
                            ext_score_adjustment=0.5,
                            method='cos',
                            constant=0.1):

        if hasattr(self, 'psi_items') == False:
            self.test_fit_statistics_items(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                           ext_score_adjustment=ext_score_adjustment, method=method, constant=constant)

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
                         tolerance=0.0000001,
                         max_iters=100,
                         ext_score_adjustment=0.5,
                         method='cos',
                         constant=0.1,
                         no_of_samples=100,
                         interval=None):

        if hasattr(self, 'item_stats_items') == False:
            self.item_stats_df_items(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                     ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                     no_of_samples=no_of_samples, interval=interval)

        if hasattr(self, 'threshold_stats_items') == False:
            self.threshold_stats_df_items(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                          ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                          no_of_samples=no_of_samples, interval=interval)

        if hasattr(self, 'rater_stats_items') == False:
            self.rater_stats_df_items(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                      ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                      no_of_samples=no_of_samples, interval=interval)

        if hasattr(self, 'person_stats_items') == False:
            self.person_stats_df_items(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                       ext_score_adjustment=ext_score_adjustment, method=method, constant=constant)

        if hasattr(self, 'test_stats_items') == False:
            self.test_stats_df_items(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                     ext_score_adjustment=ext_score_adjustment, method=method, constant=constant)

        if format == 'xlsx':

            if filename[-5:] != '.xlsx':
                filename += '.xlsx'

            writer = pd.ExcelWriter(filename, engine='xlsxwriter')

            self.item_stats_items.to_excel(writer, sheet_name='Item statistics')
            self.threshold_stats_items.to_excel(writer, sheet_name='Threshold statistics')
            self.rater_stats_items.to_excel(writer, sheet_name='Rater statistics')
            self.person_stats_items.to_excel(writer, sheet_name='Person statistics')
            self.test_stats_items.to_excel(writer, sheet_name='Test statistics')

            writer.save()

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
                                 tolerance=0.0000001,
                                 max_iters=100,
                                 ext_score_adjustment=0.5,
                                 method='cos',
                                 constant=0.1,
                                 no_of_samples=100,
                                 interval=None):

        if full:
            zstd = True
            point_measure_corr = True

            if interval is None:
                interval = 0.95

        if anchor_raters is not None:
            if (hasattr(self, 'anchor_severites_thresholds') == False) or (self.anchor_raters_thresholds != anchor_raters):
                self.calibrate_thresholds_anchor(anchor_raters, constant=constant, method=method)
                self.std_errors_thresholds(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                           constant=constant, method=method)

            elif (self.anchor_item_low_thresholds is None) and (interval is not None):
                self.std_errors_thresholds(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                           constant=constant, method=method)

        else:
            if hasattr(self, 'anchor_item_se_thresholds') == False:
                self.std_errors_thresholds(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                           constant=constant, method=method)

            elif (self.item_low is None) and (interval is not None):
                self.std_errors_thresholds(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                           constant=constant, method=method)

        if hasattr(self, 'item_outfit_ms_thresholds') == False:
            self.item_fit_statistics_thresholds(anchor_raters=anchor_raters, warm_corr=warm_corr, tolerance=tolerance,
                                                max_iters=max_iters, ext_score_adjustment=ext_score_adjustment,
                                                method=method, constant=constant, no_of_samples=no_of_samples,
                                                interval=interval)

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

        self.item_stats_thresholds['Estimate'] = difficulties.to_numpy().round(dp)
        self.item_stats_thresholds['SE'] = std_errors.to_numpy().round(dp)
        if interval is not None:
            self.item_stats_thresholds[f'{round((1 - interval) * 50, 1)}%'] = low.to_numpy().round(dp)
            self.item_stats_thresholds[f'{round((1 + interval) * 50, 1)}%'] = high.to_numpy().round(dp)

        self.item_stats_thresholds['Count'] = self.response_counts.to_numpy().astype(int)
        self.item_stats_thresholds['Facility'] = self.item_facilities.to_numpy().round(dp)

        self.item_stats_thresholds['Infit MS'] = self.item_infit_ms_thresholds.to_numpy().round(dp)
        if zstd:
            self.item_stats_thresholds['Infit Z'] = self.item_infit_zstd_thresholds.to_numpy().round(dp)
        self.item_stats_thresholds['Outfit MS'] = self.item_outfit_ms_thresholds.to_numpy().round(dp)
        if zstd:
            self.item_stats_thresholds['Outfit Z'] = self.item_outfit_zstd_thresholds.to_numpy().round(dp)

        if point_measure_corr:
            self.item_stats_thresholds['PM corr'] = self.point_measure_thresholds.to_numpy().round(dp)
            self.item_stats_thresholds['Exp PM corr'] = self.exp_point_measure_thresholds.to_numpy().round(dp)

        self.item_stats_thresholds.index = self.dataframe.columns

    def threshold_stats_df_thresholds(self,
                                      anchor_raters=None,
                                      full=False,
                                      zstd=False,
                                      disc=False,
                                      point_measure_corr=False,
                                      dp=3,
                                      warm_corr=True,
                                      tolerance=0.0000001,
                                      max_iters=100,
                                      ext_score_adjustment=0.5,
                                      method='cos',
                                      constant=0.1,
                                      no_of_samples=100,
                                      interval=None):

        if full:
            zstd = True
            disc = True
            point_measure_corr = True

            if interval is None:
                interval = 0.95

        if anchor_raters is not None:
            if (hasattr(self, 'anchor_severites_thresholds') == False) or (self.anchor_raters_thresholds != anchor_raters):
                self.calibrate_thresholds_anchor(anchor_raters, constant=constant, method=method)
                self.std_errors_thresholds(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                           constant=constant, method=method)

            elif (self.anchor_item_low_thresholds is None) and (interval is not None):
                self.std_errors_thresholds(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                           constant=constant, method=method)

        else:
            if hasattr(self, 'anchor_item_se_thresholds') == False:
                self.std_errors_thresholds(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                           constant=constant, method=method)

            elif (self.item_low is None) and (interval is not None):
                self.std_errors_thresholds(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                           constant=constant, method=method)

        if hasattr(self, 'threshold_outfit_ms_thresholds') == False:
            self.threshold_fit_statistics_thresholds(anchor_raters=anchor_raters, warm_corr=warm_corr,
                                                     tolerance=tolerance, max_iters=max_iters,
                                                     ext_score_adjustment=ext_score_adjustment, method=method,
                                                     constant=constant)

        if anchor_raters is not None:
            thresholds = self.anchor_thresholds_thresholds
        else:
            thresholds = self.thresholds

        self.threshold_stats_thresholds = pd.DataFrame()

        self.threshold_stats_thresholds['Estimate'] = thresholds[1:].round(dp)
        self.threshold_stats_thresholds['SE'] = self.threshold_se_thresholds[1:].round(dp)

        if interval is not None:
            if anchor_raters != []:
                self.threshold_stats_thresholds[f'{round((1 - interval) * 50, 1)}%'] = self.anchor_threshold_low_thresholds[1:].round(dp)
                self.threshold_stats_thresholds[f'{round((1 + interval) * 50, 1)}%'] = self.anchor_threshold_high_thresholds[1:].round(dp)

            else:
                self.threshold_stats_thresholds[f'{round((1 - interval) * 50, 1)}%'] = self.threshold_low_thresholds[1:].round(dp)
                self.threshold_stats_thresholds[f'{round((1 + interval) * 50, 1)}%'] = self.threshold_high_thresholds[1:].round(dp)

        self.threshold_stats_thresholds['Infit MS'] = self.threshold_infit_ms_thresholds.to_numpy().round(dp)
        if zstd:
            self.threshold_stats_thresholds['Infit Z'] = self.threshold_infit_zstd_thresholds.to_numpy().round(dp)
        self.threshold_stats_thresholds['Outfit MS'] = self.threshold_outfit_ms_thresholds.to_numpy().round(dp)
        if zstd:
            self.threshold_stats_thresholds['Outfit Z'] = self.threshold_outfit_zstd_thresholds.to_numpy().round(dp)

        if disc:
            self.threshold_stats_thresholds['Discrim'] = self.threshold_discrimination_thresholds.to_numpy().round(dp)

        if point_measure_corr:
            self.threshold_stats_thresholds['PM corr'] = self.threshold_point_measure_thresholds.to_numpy().round(dp)
            self.threshold_stats_thresholds['Exp PM corr'] = self.threshold_exp_point_measure_thresholds.to_numpy().round(dp)

        self.threshold_stats_thresholds.index = [f'Threshold {threshold + 1}' for threshold in range(self.max_score)]

    def rater_stats_df_thresholds(self,
                                  anchor_raters=None,
                                  full=False,
                                  zstd=True,
                                  dp=3,
                                  warm_corr=True,
                                  tolerance=0.0000001,
                                  max_iters=100,
                                  ext_score_adjustment=0.5,
                                  method='cos',
                                  constant=0.1,
                                  no_of_samples=100,
                                  interval=None):

        if full:
            zstd=True

            if interval is None:
                interval = 0.95

        if anchor_raters is not None:
            if (hasattr(self, 'anchor_severites_thresholds') == False) or (self.anchor_raters_thresholds != anchor_raters):
                self.calibrate_thresholds_anchor(anchor_raters, constant=constant, method=method)
                self.std_errors_thresholds(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                           constant=constant, method=method)

        else:
            if hasattr(self, 'anchor_item_se_thresholds') == False:
                self.std_errors_thresholds(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                           constant=constant, method=method)

        if hasattr(self, 'rater_outfit_ms_thresholds') == False:
            self.rater_fit_statistics_thresholds(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                                 ext_score_adjustment=ext_score_adjustment, method=method,
                                                 constant=constant, no_of_samples=no_of_samples, interval=interval)

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

            item_stats = pd.DataFrame()

            item_stats['Estimate'] = np.array([severities[rater][threshold + 1] for rater in self.raters]).round(dp)
            item_stats['SE'] = np.array([se[rater][threshold + 1] for rater in self.raters]).round(dp)

            if interval is not None:
                item_stats[ f'{round((1 - interval) * 50, 1)}%'] = np.array([low[rater][threshold + 1]
                                                                             for rater in self.raters]).round(dp)
                item_stats[f'{round((1 + interval) * 50, 1)}%'] = np.array([high[rater][threshold + 1]
                                                                            for rater in self.raters]).round(dp)

            item_stats.index = self.raters
            self.rater_stats_thresholds[f'Threshold {threshold + 1}'] = item_stats.T

        ov_stats_df = pd.DataFrame()

        ov_stats_df['Count'] = np.array([self.dataframe.xs(rater).count().sum() for rater in self.raters]).astype(int)
        ov_stats_df['Infit MS'] = self.rater_infit_ms_thresholds.to_numpy().round(dp)
        if zstd:
            ov_stats_df['Infit Z'] = self.rater_infit_zstd_thresholds.to_numpy().round(dp)
        ov_stats_df['Outfit MS'] = self.rater_outfit_ms_thresholds.to_numpy().round(dp)
        if zstd:
            ov_stats_df['Outfit Z'] = self.rater_outfit_zstd_thresholds.to_numpy().round(dp)

        ov_stats_df.index = self.raters
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
                                   tolerance=0.0000001,
                                   max_iters=100,
                                   ext_score_adjustment=0.5,
                                   method='cos',
                                   constant=0.1):

        '''
        Produces a person stats dataframe with raw score, ability estimate,
        CSEM and RSEM for each person.
        '''

        if hasattr(self, 'person_outfit_ms_thresholds') == False:
            self.person_fit_statistics_thresholds(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                                  ext_score_adjustment=ext_score_adjustment, method=method,
                                                  constant=constant)

        if full:
            rsem = True
            zstd = True

        person_stats_df = pd.DataFrame()
        person_stats_df.index = self.dataframe.index.get_level_values(1).unique()

        if anchor_raters is None:
            person_stats_df['Estimate'] = self.abils_thresholds.to_numpy().round(dp)

        else:
            person_stats_df['Estimate'] = self.anchor_abils_thresholds.to_numpy().round(dp)

        person_stats_df['CSEM'] = self.csem_vector_thresholds.round(dp)
        if rsem:
            person_stats_df['RSEM'] = self.rsem_vector_thresholds.round(dp)

        person_stats_df['Score'] = [np.nan for person in self.persons]
        person_stats_df['Score'].update(self.dataframe.unstack(level=0).sum(axis=1))
        person_stats_df['Score'] = person_stats_df['Score'].astype(int)

        person_stats_df['Max score'] = [np.nan for person in self.persons]
        person_stats_df['Max score'].update(self.dataframe.unstack(level=0).count(axis=1) * self.max_score)
        person_stats_df['Max score'] = person_stats_df['Max score'].astype(int)

        person_stats_df['p'] = [np.nan for person in self.persons]
        person_stats_df['p'].update(self.dataframe.unstack(level=0).mean(axis=1) / self.max_score)
        person_stats_df['p'] = person_stats_df['p'].round(dp)

        person_stats_df['Infit MS'] = [np.nan for person in self.persons]
        person_stats_df['Infit MS'].update(self.person_infit_ms_thresholds.round(dp))

        if zstd:
            person_stats_df['Infit Z'] = [np.nan for person in self.persons]
            person_stats_df['Infit Z'].update(self.person_infit_zstd_thresholds.round(dp))

        person_stats_df['Outfit MS'] = [np.nan for person in self.persons]
        person_stats_df['Outfit MS'].update(self.person_outfit_ms_thresholds.round(dp))

        if zstd:
            person_stats_df['Outfit Z'] = [np.nan for person in self.persons]
            person_stats_df['Outfit Z'].update(self.person_outfit_zstd_thresholds.round(dp))

        self.person_stats_thresholds = person_stats_df

    def test_stats_df_thresholds(self,
                                 dp=3,
                                 warm_corr=True,
                                 tolerance=0.0000001,
                                 max_iters=100,
                                 ext_score_adjustment=0.5,
                                 method='cos',
                                 constant=0.1):

        if hasattr(self, 'psi_thresholds') == False:
            self.test_fit_statistics_thresholds(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                                ext_score_adjustment=ext_score_adjustment, method=method,
                                                constant=constant)

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
                              tolerance=0.0000001,
                              max_iters=100,
                              ext_score_adjustment=0.5,
                              method='cos',
                              constant=0.1,
                              no_of_samples=100,
                              interval=None):

        if hasattr(self, 'item_stats_thresholds') == False:
            self.item_stats_df_thresholds(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                          ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                          no_of_samples=no_of_samples, interval=interval)

        if hasattr(self, 'threshold_stats_thresholds') == False:
            self.threshold_stats_df_thresholds(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                               ext_score_adjustment=ext_score_adjustment, method=method,
                                               constant=constant, no_of_samples=no_of_samples, interval=interval)

        if hasattr(self, 'rater_stats_thresholds') == False:
            self.rater_stats_df_thresholds(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                           ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                           no_of_samples=no_of_samples, interval=interval)

        if hasattr(self, 'person_stats_thresholds') == False:
            self.person_stats_df_thresholds(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                            ext_score_adjustment=ext_score_adjustment, method=method, constant=constant)

        if hasattr(self, 'test_stats_thresholds') == False:
            self.test_stats_df_thresholds(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                          ext_score_adjustment=ext_score_adjustment, method=method, constant=constant)

        if format == 'xlsx':

            if filename[-5:] != '.xlsx':
                filename += '.xlsx'

            writer = pd.ExcelWriter(filename, engine='xlsxwriter')

            self.item_stats_thresholds.to_excel(writer, sheet_name='Item statistics')
            self.threshold_stats_thresholds.to_excel(writer, sheet_name='Threshold statistics')
            self.rater_stats_thresholds.to_excel(writer, sheet_name='Rater statistics')
            self.person_stats_thresholds.to_excel(writer, sheet_name='Person statistics')
            self.test_stats_thresholds.to_excel(writer, sheet_name='Test statistics')

            writer.save()

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
                             tolerance=0.0000001,
                             max_iters=100,
                             ext_score_adjustment=0.5,
                             method='cos',
                             constant=0.1,
                             no_of_samples=100,
                             interval=None):

        if full:
            zstd = True
            point_measure_corr = True

            if interval is None:
                interval = 0.95

        if anchor_raters is not None:
            if (hasattr(self, 'anchor_severites_matrix') == False) or (self.anchor_raters_matrix != anchor_raters):
                self.calibrate_matrix_anchor(anchor_raters, constant=constant, method=method)
                self.std_errors_matrix(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method)

            elif (self.anchor_item_low_matrix is None) and (interval is not None):
                self.std_errors_global(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method)

        else:
            if hasattr(self, 'anchor_item_se_matrix') == False:
                self.std_errors_matrix(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method)

            elif (self.item_low is None) and (interval is not None):
                self.std_errors_matrix(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method)

        if hasattr(self, 'item_outfit_ms_matrix') == False:
            self.item_fit_statistics_matrix(anchor_raters=anchor_raters, warm_corr=warm_corr, tolerance=tolerance,
                                            max_iters=max_iters, ext_score_adjustment=ext_score_adjustment, method=method,
                                            constant=constant, no_of_samples=no_of_samples, interval=interval)

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

        self.item_stats_matrix['Estimate'] = difficulties.to_numpy().round(dp)
        self.item_stats_matrix['SE'] = std_errors.to_numpy().round(dp)

        if interval is not None:
            self.item_stats_matrix[f'{round((1 - interval) * 50, 1)}%'] = low.to_numpy().round(dp)
            self.item_stats_matrix[f'{round((1 + interval) * 50, 1)}%'] = high.to_numpy().round(dp)

        self.item_stats_matrix['Count'] = self.response_counts.to_numpy().astype(int)
        self.item_stats_matrix['Facility'] = self.item_facilities.to_numpy().round(dp)

        self.item_stats_matrix['Infit MS'] = self.item_infit_ms_matrix.to_numpy().round(dp)
        if zstd:
            self.item_stats_matrix['Infit Z'] = self.item_infit_zstd_matrix.to_numpy().round(dp)
        self.item_stats_matrix['Outfit MS'] = self.item_outfit_ms_matrix.to_numpy().round(dp)
        if zstd:
            self.item_stats_matrix['Outfit Z'] = self.item_outfit_zstd_matrix.to_numpy().round(dp)

        if point_measure_corr:
            self.item_stats_matrix['PM corr'] = self.point_measure_matrix.to_numpy().round(dp)
            self.item_stats_matrix['Exp PM corr'] = self.exp_point_measure_matrix.to_numpy().round(dp)

        self.item_stats_matrix.index = self.dataframe.columns

    def threshold_stats_df_matrix(self,
                                  anchor_raters=None,
                                  full=False,
                                  zstd=False,
                                  disc=False,
                                  point_measure_corr=False,
                                  dp=3,
                                  warm_corr=True,
                                  tolerance=0.0000001,
                                  max_iters=100,
                                  ext_score_adjustment=0.5,
                                  method='cos',
                                  constant=0.1,
                                  no_of_samples=100,
                                  interval=None):

        if full:
            zstd = True
            disc = True
            point_measure_corr = True

            if interval is None:
                interval = 0.95

        if anchor_raters is not None:
            if (hasattr(self, 'anchor_severites_matrix') == False) or (self.anchor_raters_matrix != anchor_raters):
                self.calibrate_matrix_anchor(anchor_raters, constant=constant, method=method)
                self.std_errors_matrix(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method)

            elif (self.anchor_item_low_matrix is None) and (interval is not None):
                self.std_errors_global(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method)

        else:
            if hasattr(self, 'anchor_item_se_matrix') == False:
                self.std_errors_matrix(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method)

            elif (self.item_low is None) and (interval is not None):
                self.std_errors_matrix(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method)

        if hasattr(self, 'threshold_outfit_ms_matrix') == False:
            self.threshold_fit_statistics_matrix(anchor_raters=anchor_raters, warm_corr=warm_corr, tolerance=tolerance,
                                                 max_iters=max_iters, ext_score_adjustment=ext_score_adjustment,
                                                 method=method, constant=constant)

        if anchor_raters is not None:
            thresholds = self.anchor_thresholds_matrix
        else:
            thresholds = self.thresholds

        self.threshold_stats_matrix = pd.DataFrame()

        self.threshold_stats_matrix['Estimate'] = thresholds[1:].round(dp)
        self.threshold_stats_matrix['SE'] = self.threshold_se_matrix[1:].round(dp)

        if interval is not None:
            if anchor_raters != []:
                self.threshold_stats_matrix[f'{round((1 - interval) * 50, 1)}%'] = self.anchor_threshold_low_matrix[1:].round(dp)
                self.threshold_stats_matrix[f'{round((1 + interval) * 50, 1)}%'] = self.anchor_threshold_high_matrix[1:].round(dp)

            else:
                self.threshold_stats_matrix[f'{round((1 - interval) * 50, 1)}%'] = self.threshold_low_matrix[1:].round(dp)
                self.threshold_stats_matrix[f'{round((1 + interval) * 50, 1)}%'] = self.threshold_high_matrix[1:].round(dp)

        self.threshold_stats_matrix['Infit MS'] = self.threshold_infit_ms_matrix.to_numpy().round(dp)
        if zstd:
            self.threshold_stats_matrix['Infit Z'] = self.threshold_infit_zstd_matrix.to_numpy().round(dp)
        self.threshold_stats_matrix['Outfit MS'] = self.threshold_outfit_ms_matrix.to_numpy().round(dp)
        if zstd:
            self.threshold_stats_matrix['Outfit Z'] = self.threshold_outfit_zstd_matrix.to_numpy().round(dp)

        if disc:
            self.threshold_stats_matrix['Discrim'] = self.threshold_discrimination_matrix.to_numpy().round(dp)

        if point_measure_corr:
            self.threshold_stats_matrix['PM corr'] = self.threshold_point_measure_matrix.to_numpy().round(dp)
            self.threshold_stats_matrix['Exp PM corr'] = self.threshold_exp_point_measure_matrix.to_numpy().round(dp)

        self.threshold_stats_matrix.index = [f'Threshold {threshold + 1}' for threshold in range(self.max_score)]

    def rater_stats_df_matrix(self,
                              anchor_raters=None,
                              full=False,
                              zstd=False,
                              marginal=True,
                              dp=3,
                              warm_corr=True,
                              tolerance=0.0000001,
                              max_iters=100,
                              ext_score_adjustment=0.5,
                              method='cos',
                              constant=0.1,
                              no_of_samples=100,
                              interval=None):

        if full:
            zstd = True

            if interval is None:
                interval = 0.95

        if anchor_raters is not None:
            if (hasattr(self, 'anchor_severites_matrix') == False) or (self.anchor_raters_matrix != anchor_raters):
                self.calibrate_matrix_anchor(anchor_raters, constant=constant, method=method)
                self.std_errors_matrix(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method)

        else:
            if hasattr(self, 'anchor_item_se_matrix') == False:
                self.std_errors_matrix(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method)

        if hasattr(self, 'rater_outfit_ms_matrix') == False:
            self.rater_fit_statistics_matrix(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                             ext_score_adjustment=ext_score_adjustment, method=method,
                                             constant=constant, no_of_samples=no_of_samples, interval=interval)

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
                                                   for rater in self.raters]).round(dp)

                if anchor_raters != []:
                    item_stats['SE'] = np.array([self.anchor_rater_se_marginal_items[rater][item]
                                                for rater in self.raters]).round(dp)
                else:
                    item_stats['SE'] = np.array([self.rater_se_marginal_items[rater][item]
                                                for rater in self.raters]).round(dp)

                if interval is not None:
                    if anchor_raters != []:
                        item_stats[f'{round((1 - interval) * 50, 1)}%'] = np.array([self.anchor_rater_low_marginal_items[rater][item]
                                                                                    for rater in self.raters]).round(dp)
                        item_stats[f'{round((1 + interval) * 50, 1)}%'] = np.array([self.anchor_rater_high_marginal_items[rater][item]
                                                                                for rater in self.raters]).round(dp)

                    else:
                        item_stats[f'{round((1 - interval) * 50, 1)}%'] = np.array([self.rater_low_marginal_items[rater][item]
                                                                                    for rater in self.raters]).round(dp)
                        item_stats[f'{round((1 + interval) * 50, 1)}%'] = np.array([self.rater_high_marginal_items[rater][item]
                                                                                for rater in self.raters]).round(dp)

                item_stats.index = self.raters
                self.rater_stats_matrix[item] = item_stats.T

            for threshold in range(self.max_score):

                item_stats = pd.DataFrame()

                item_stats['Estimate'] = np.array([marginal_thresholds[rater][threshold + 1]
                                                   for rater in self.raters]).round(dp)

                if anchor_raters != []:
                    item_stats['SE'] = np.array([self.anchor_rater_se_marginal_thresholds[rater][threshold + 1]
                                                 for rater in self.raters]).round(dp)
                else:
                    item_stats['SE'] = np.array([self.rater_se_marginal_thresholds[rater][threshold + 1]
                                                 for rater in self.raters]).round(dp)

                if interval is not None:
                    if anchor_raters != []:
                        item_stats[ f'{round((1 - interval) * 50, 1)}%'] = np.array([self.anchor_rater_low_marginal_thresholds[rater][threshold + 1]
                                                                                     for rater in self.raters]).round(dp)
                        item_stats[f'{round((1 + interval) * 50, 1)}%'] = np.array([self.anchor_rater_high_marginal_thresholds[rater][threshold + 1]
                                                                                    for rater in self.raters]).round(dp)

                    else:
                        item_stats[ f'{round((1 - interval) * 50, 1)}%'] = np.array([self.rater_low_marginal_thresholds[rater][threshold + 1]
                                                                                     for rater in self.raters]).round(dp)
                        item_stats[f'{round((1 + interval) * 50, 1)}%'] = np.array([self.rater_high_marginal_thresholds[rater][threshold + 1]
                                                                                    for rater in self.raters]).round(dp)

                item_stats.index = self.raters
                self.rater_stats_matrix[f'Threshold {threshold + 1}'] = item_stats.T

            ov_stats_df = pd.DataFrame()

            ov_stats_df['Count'] = np.array([self.dataframe.xs(rater).count().sum() for rater in self.raters]).astype(int)
            ov_stats_df['Infit MS'] = self.rater_infit_ms_matrix.to_numpy().round(dp)
            if zstd:
                ov_stats_df['Infit Z'] = self.rater_infit_zstd_matrix.to_numpy().round(dp)
            ov_stats_df['Outfit MS'] = self.rater_outfit_ms_matrix.to_numpy().round(dp)
            if zstd:
                ov_stats_df['Outfit Z'] = self.rater_outfit_zstd_matrix.to_numpy().round(dp)

            ov_stats_df.index = self.raters
            self.rater_stats_matrix['Overall statistics'] = ov_stats_df.T

            self.rater_stats_matrix = pd.concat(self.rater_stats_matrix.values(), keys=self.rater_stats_matrix.keys()).T

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

            ov_stats_df = pd.DataFrame()

            ov_stats_df['Count'] = np.array([self.dataframe.xs(rater).count().sum() for rater in self.raters]).astype(int)
            ov_stats_df['Infit MS'] = self.rater_infit_ms_matrix.to_numpy().round(dp)
            if zstd:
                ov_stats_df['Infit Z'] = self.rater_infit_zstd_matrix.to_numpy().round(dp)
            ov_stats_df['Outfit MS'] = self.rater_outfit_ms_matrix.to_numpy().round(dp)
            if zstd:
                ov_stats_df['Outfit Z'] = self.rater_outfit_zstd_matrix.to_numpy().round(dp)

            ov_stats_df.index = self.raters
            self.rater_stats_matrix['Overall statistics'] = ov_stats_df.T

            self.rater_stats_matrix = pd.concat(self.rater_stats_matrix.values(), keys=self.rater_stats_matrix.keys()).T

    def person_stats_df_matrix(self,
                               anchor_raters=None,
                               full=False,
                               rsem=False,
                               zstd=False,
                               dp=3,
                               warm_corr=True,
                               tolerance=0.0000001,
                               max_iters=100,
                               ext_score_adjustment=0.5,
                               method='cos',
                               constant=0.1):

        '''
        Produces a person stats dataframe with raw score, ability estimate,
        CSEM and RSEM for each person.
        '''

        if hasattr(self, 'person_outfit_ms_matrix') == False:
            self.person_fit_statistics_matrix(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                              ext_score_adjustment=ext_score_adjustment, method=method,
                                              constant=constant)

        if full:
            rsem = True
            zstd = True

        person_stats_df = pd.DataFrame()
        person_stats_df.index = self.dataframe.index.get_level_values(1).unique()

        if anchor_raters is None:
            person_stats_df['Estimate'] = self.abils_matrix.to_numpy().round(dp)

        else:
            person_stats_df['Estimate'] = self.anchor_abils_matrix.to_numpy().round(dp)

        person_stats_df['CSEM'] = self.csem_vector_matrix.round(dp)
        if rsem:
            person_stats_df['RSEM'] = self.rsem_vector_matrix.round(dp)

        person_stats_df['Score'] = [np.nan for person in self.persons]
        person_stats_df['Score'].update(self.dataframe.unstack(level=0).sum(axis=1))
        person_stats_df['Score'] = person_stats_df['Score'].astype(int)

        person_stats_df['Max score'] = [np.nan for person in self.persons]
        person_stats_df['Max score'].update(self.dataframe.unstack(level=0).count(axis=1) * self.max_score)
        person_stats_df['Max score'] = person_stats_df['Max score'].astype(int)

        person_stats_df['p'] = [np.nan for person in self.persons]
        person_stats_df['p'].update(self.dataframe.unstack(level=0).mean(axis=1) / self.max_score)
        person_stats_df['p'] = person_stats_df['p'].round(dp)

        person_stats_df['Infit MS'] = [np.nan for person in self.persons]
        person_stats_df['Infit MS'].update(self.person_infit_ms_matrix.round(dp))

        if zstd:
            person_stats_df['Infit Z'] = [np.nan for person in self.persons]
            person_stats_df['Infit Z'].update(self.person_infit_zstd_matrix.round(dp))

        person_stats_df['Outfit MS'] = [np.nan for person in self.persons]
        person_stats_df['Outfit MS'].update(self.person_outfit_ms_matrix.round(dp))

        if zstd:
            person_stats_df['Outfit Z'] = [np.nan for person in self.persons]
            person_stats_df['Outfit Z'].update(self.person_outfit_zstd_matrix.round(dp))

        self.person_stats_matrix = person_stats_df

    def test_stats_df_matrix(self,
                             dp=3,
                             warm_corr=True,
                             tolerance=0.0000001,
                             max_iters=100,
                             ext_score_adjustment=0.5,
                             method='cos',
                             constant=0.1):

        '''
        Produces a test statistics dataframe with raw score, ability estimate,
        CSEM and RSEM for each person.
        '''

        if hasattr(self, 'psi_matrix') == False:
            self.test_fit_statistics_matrix(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                            ext_score_adjustment=ext_score_adjustment, method=method,
                                            constant=constant)

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
                         tolerance=0.0000001,
                         max_iters=100,
                         ext_score_adjustment=0.5,
                         method='cos',
                         constant=0.1,
                         no_of_samples=100,
                         interval=None):

        if hasattr(self, 'item_stats_matrix') == False:
            self.item_stats_df_matrix(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                      ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                      no_of_samples=no_of_samples, interval=interval)

        if hasattr(self, 'threshold_stats_matrix') == False:
            self.threshold_stats_df_matrix(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                           ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                           no_of_samples=no_of_samples, interval=interval)

        if hasattr(self, 'rater_stats_matrix') == False:
            self.rater_stats_df_matrix(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                       ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                       no_of_samples=no_of_samples, interval=interval)

        if hasattr(self, 'person_stats_matrix') == False:
            self.person_stats_df_matrix(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                        ext_score_adjustment=ext_score_adjustment, method=method, constant=constant)

        if hasattr(self, 'test_stats_matrix') == False:
            self.test_stats_df_matrix(dp=dp, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                      ext_score_adjustment=ext_score_adjustment, method=method, constant=constant)

        if format == 'xlsx':

            if filename[-5:] != '.xlsx':
                filename += '.xlsx'

            writer = pd.ExcelWriter(filename, engine='xlsxwriter')

            self.item_stats_matrix.to_excel(writer, sheet_name='Item statistics')
            self.threshold_stats_matrix.to_excel(writer, sheet_name='Threshold statistics')
            self.rater_stats_matrix.to_excel(writer, sheet_name='Rater statistics')
            self.person_stats_matrix.to_excel(writer, sheet_name='Person statistics')
            self.test_stats_matrix.to_excel(writer, sheet_name='Test statistics')

            writer.save()

        else:
            if filename[-4:] == '.csv':
                filename = filename[:-4]

            self.item_stats_matrix.to_csv(f'{filename}_item_stats.csv')
            self.threshold_stats_matrix.to_csv(f'{filename}_threshold_stats.csv')
            self.rater_stats_matrix.to_csv(f'{filename}_rater_stats.csv')
            self.person_stats_matrix.to_csv(f'{filename}_person_stats.csv')
            self.test_stats_matrix.to_csv(f'{filename}_test_stats.csv')

    def category_probability_dict_global(self,
                                         anchor_raters=[],
                                         warm_corr=True,
                                         tolerance=0.0000001,
                                         max_iters=100,
                                         ext_score_adjustment=0.5,
                                         method='cos',
                                         constant=0.1,
                                         no_of_samples=100,
                                         interval=None):

        if hasattr(self, 'thresholds_global') == False:
            self.calibrate_global(constant, method)

        if hasattr(self, 'threshold_se_global') == False:
            self.std_errors_global(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                   constant=constant, method=method)

        if hasattr(self, 'abils_global') == False:
            self.person_abils_global(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                     ext_score_adjustment=ext_score_adjustment)

        if anchor_raters != []:
            difficulties = self.anchor_diffs_global
            thresholds = self.anchor_thresholds_global
            severities = self.anchor_severities_global

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_global

        cat_prob_dict = {cat: {rater: {item: {person: self.cat_prob_global(abil, item, difficulties, rater, severities,
                                                                           cat, thresholds)
                                              for person, abil in self.abils_global.items()}
                                       for item in self.dataframe.columns}
                               for rater in self.raters}
                         for cat in range(self.max_score + 1)}

        for cat in range(self.max_score + 1):
            for rater in self.raters:
                cat_prob_dict[cat][rater] = pd.DataFrame(cat_prob_dict[cat][rater])

        for cat in range(self.max_score + 1):
            cat_prob_dict[cat] = pd.concat(cat_prob_dict[cat].values(),
                                           keys=cat_prob_dict[cat].keys())

        self.cat_prob_dict_global = cat_prob_dict

    def category_probability_dict_items(self,
                                        anchor_raters=[],
                                        warm_corr=True,
                                        tolerance=0.0000001,
                                        max_iters=100,
                                        ext_score_adjustment=0.5,
                                        method='cos',
                                        constant=0.1,
                                        no_of_samples=100,
                                        interval=None):

        if hasattr(self, 'thresholds_items') == False:
            self.calibrate_items(constant, method)

        if anchor_raters != []:
            if hasattr(self, 'anchor_thresholds_items') == False:
                self.calibrate_items_anchor(anchor_raters, constant=constant, method=method)

        if hasattr(self, 'threshold_se_items') == False:
            self.std_errors_items(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                  constant=constant, method=method)

        if hasattr(self, 'abils_items') == False:

            if anchor_raters == []:
                anchor = False
            else:
                anchor = True

            self.person_abils_items(anchor=anchor, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                    ext_score_adjustment=ext_score_adjustment)

        if anchor_raters != []:
            difficulties = self.anchor_diffs_items
            thresholds = self.anchor_thresholds_items
            severities = self.anchor_severities_items

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_items

        cat_prob_dict = {cat: {rater: {item: {person: self.cat_prob_items(abil, item, difficulties, rater,
                                                             severities, cat, thresholds)
                                              for person, abil in self.abils_items.items()}
                                       for item in self.dataframe.columns}
                               for rater in self.raters}
                         for cat in range(self.max_score + 1)}

        for cat in range(self.max_score + 1):
            for rater in self.raters:
                cat_prob_dict[cat][rater] = pd.DataFrame(cat_prob_dict[cat][rater])

        for cat in range(self.max_score + 1):
            cat_prob_dict[cat] = pd.concat(cat_prob_dict[cat].values(),
                                           keys=cat_prob_dict[cat].keys())

        self.cat_prob_dict_items = cat_prob_dict

    def category_probability_dict_thresholds(self,
                                             anchor_raters=[],
                                             warm_corr=True,
                                             tolerance=0.0000001,
                                             max_iters=100,
                                             ext_score_adjustment=0.5,
                                             method='cos',
                                             constant=0.1,
                                             no_of_samples=100,
                                             interval=None):

        if hasattr(self, 'thresholds_thresholds') == False:
            self.calibrate_thresholds(constant, method)

        if anchor_raters != []:
            if hasattr(self, 'anchor_thresholds_thresholds') == False:
                self.calibrate_thresholds_anchor(anchor_raters, constant=constant, method=method)

        if hasattr(self, 'threshold_se_thresholds') == False:
            self.std_errors_thresholds(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                       constant=constant, method=method)

        if hasattr(self, 'abils_thresholds') == False:

            if anchor_raters == []:
                anchor = False
            else:
                anchor = True

            self.person_abils_thresholds(anchor=anchor, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                         ext_score_adjustment=ext_score_adjustment)

        if anchor_raters != []:
            difficulties = self.anchor_diffs_thresholds
            thresholds = self.anchor_thresholds_thresholds
            severities = self.anchor_severities_thresholds

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_thresholds

        cat_prob_dict = {cat: {rater: {item: {person: self.cat_prob_thresholds(abil, item, difficulties, rater,
                                                                               severities, cat, thresholds)
                                              for person, abil in self.abils_thresholds.items()}
                                       for item in self.dataframe.columns}
                               for rater in self.raters}
                         for cat in range(self.max_score + 1)}

        for cat in range(self.max_score + 1):
            for rater in self.raters:
                cat_prob_dict[cat][rater] = pd.DataFrame(cat_prob_dict[cat][rater])

        for cat in range(self.max_score + 1):
            cat_prob_dict[cat] = pd.concat(cat_prob_dict[cat].values(),
                                           keys=cat_prob_dict[cat].keys())

        self.cat_prob_dict_thresholds = cat_prob_dict

    def category_probability_dict_matrix(self,
                                         anchor_raters=[],
                                         warm_corr=True,
                                         tolerance=0.0000001,
                                         max_iters=100,
                                         ext_score_adjustment=0.5,
                                         method='cos',
                                         constant=0.1,
                                         no_of_samples=100,
                                         interval=None):

        if hasattr(self, 'thresholds_matrix') == False:
            self.calibrate_matrix(constant, method)

        if anchor_raters != []:
            if hasattr(self, 'anchor_thresholds_matrix') == False:
                self.calibrate_matrix_anchor(anchor_raters, constant=constant, method=method)

        if hasattr(self, 'threshold_se_matrix') == False:
            self.std_errors_matrix(anchor_raters=anchor_raters, interval=interval, no_of_samples=no_of_samples,
                                   constant=constant, method=method)

        if hasattr(self, 'abils_matrix') == False:

            if anchor_raters == []:
                anchor = False
            else:
                anchor = True

            self.person_abils_matrix(anchor=anchor, warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                     ext_score_adjustment=ext_score_adjustment)

        '''
        Create matrices of expected scores, variances, kurtosis, residuals etc. to generate fit statistics
        '''

        if anchor_raters != []:
            difficulties = self.anchor_diffs_matrix
            thresholds = self.anchor_thresholds_matrix
            severities = self.anchor_severities_matrix

        else:
            difficulties = self.diffs
            thresholds = self.thresholds
            severities = self.severities_matrix

        cat_prob_dict = {cat: {rater: {item: {person: self.cat_prob_matrix(abil, item, difficulties, rater,
                                                                           severities, cat, thresholds)
                                              for person, abil in self.abils_matrix.items()}
                                       for item in self.dataframe.columns}
                               for rater in self.raters}
                         for cat in range(self.max_score + 1)}

        for cat in range(self.max_score + 1):
            for rater in self.raters:
                cat_prob_dict[cat][rater] = pd.DataFrame(cat_prob_dict[cat][rater])

        for cat in range(self.max_score + 1):
            cat_prob_dict[cat] = pd.concat(cat_prob_dict[cat].values(),
                                           keys=cat_prob_dict[cat].keys())

        self.cat_prob_dict_matrix = cat_prob_dict

    def fit_matrices(self,
                     cat_prob_dict):

        '''
        Create matrices of expected scores, variances, kurtosis,
        residuals etc. to generate fit statistics
        '''

        missing_mask = (self.dataframe + 1) / (self.dataframe + 1)

        exp_score_df = sum([cat * df for cat, df in cat_prob_dict.items()])
        exp_score_df *= missing_mask

        info_df = sum([df * (cat - exp_score_df) ** 2 for cat, df in cat_prob_dict.items()])
        info_df *= missing_mask

        kurtosis_df = sum([df * (cat - exp_score_df) ** 4 for cat, df in cat_prob_dict.items()])
        kurtosis_df *= missing_mask

        residual_df = self.dataframe - exp_score_df
        std_residual_df = residual_df / np.sqrt(info_df)

        scores = self.dataframe.sum(axis=1)
        max_scores = self.dataframe.count(axis=1) * self.max_score

        extreme_mask = (scores > 0) & (scores < max_scores)
        extreme_mask = pd.Series(extreme_mask)
        extreme_mask.index = self.dataframe.index

        exp_score_df = exp_score_df.loc[extreme_mask]
        info_df = info_df.loc[extreme_mask]
        kurtosis_df = kurtosis_df.loc[extreme_mask]
        residual_df = residual_df.loc[extreme_mask]
        std_residual_df = std_residual_df.loc[extreme_mask]

        return exp_score_df, info_df, kurtosis_df, residual_df, std_residual_df

    def fit_matrices_global(self,
                            anchor_raters=[],
                            warm_corr=True,
                            tolerance=0.0000001,
                            max_iters=100,
                            ext_score_adjustment=0.5,
                            method='cos',
                            constant=0.1,
                            no_of_samples=100,
                            interval=None):

        if hasattr(self, 'cat_prob_dict_global') == False:
            self.category_probability_dict_global(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                                  ext_score_adjustment=ext_score_adjustment, method=method,
                                                  constant=constant, no_of_samples=no_of_samples, interval=interval)

        (self.exp_score_df_global,
         self.info_df_global,
         self.kurtosis_df_global,
         self.residual_df_global,
         self.std_residual_df_global) = self.fit_matrices(self.cat_prob_dict_global)

    def fit_matrices_items(self,
                           anchor_raters=[],
                           warm_corr=True,
                           tolerance=0.0000001,
                           max_iters=100,
                           ext_score_adjustment=0.5,
                           method='cos',
                           constant=0.1,
                           no_of_samples=100,
                           interval=None):

        if hasattr(self, 'cat_prob_dict_items') == False:
            self.category_probability_dict_items(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                                 ext_score_adjustment=ext_score_adjustment, method=method,
                                                 constant=constant, no_of_samples=no_of_samples, interval=interval)

        (self.exp_score_df_items,
         self.info_df_items,
         self.kurtosis_df_items,
         self.residual_df_items,
         self.std_residual_df_items) = self.fit_matrices(self.cat_prob_dict_items)

    def fit_matrices_thresholds(self,
                                anchor_raters=[],
                           	    warm_corr=True,
                           		tolerance=0.0000001,
                           		max_iters=100,
                           		ext_score_adjustment=0.5,
                           		method='cos',
                           		constant=0.1,
                           		no_of_samples=100,
                           		interval=None):

        if hasattr(self, 'cat_prob_dict_thresholds') == False:
            self.category_probability_dict_thresholds(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                                 	  ext_score_adjustment=ext_score_adjustment, method=method,
                                                 	  constant=constant, no_of_samples=no_of_samples, interval=interval)

        (self.exp_score_df_thresholds,
         self.info_df_thresholds,
         self.kurtosis_df_thresholds,
         self.residual_df_thresholds,
         self.std_residual_df_thresholds) = self.fit_matrices(self.cat_prob_dict_thresholds)

    def fit_matrices_matrix(self,
                            anchor_raters=[],
                            warm_corr=True,
                            tolerance=0.0000001,
                            max_iters=100,
                            ext_score_adjustment=0.5,
                            method='cos',
                            constant=0.1,
                            no_of_samples=100,
                            interval=None):

        if hasattr(self, 'cat_prob_dict_matrix') == False:
            self.category_probability_dict_matrix(anchor_raters=[], warm_corr=warm_corr, tolerance=tolerance,
                                                  max_iters=max_iters, ext_score_adjustment=ext_score_adjustment,
                                                  method=method, constant=constant, no_of_samples=no_of_samples,
                                                  interval=interval)

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
                            abilities,
                            warm_corr=True,
                            tolerance=0.0000001,
                            max_iters=100,
                            ext_score_adjustment=0.5,
                            method='cos',
                            constant=0.1,
                            no_of_samples=100,
                            interval=None):

        '''
        Item fit statistics
        '''

        scores = self.dataframe.sum(axis=1)
        max_scores = self.dataframe.count(axis=1) * self.max_score
        item_count = self.dataframe[(scores > 0) & (scores < max_scores)].count(axis=0)
        self.response_counts = self.dataframe.count(axis=0)

        item_outfit_ms = (std_residual_df ** 2).mean()
        item_outfit_zstd = (np.cbrt(item_outfit_ms) - 1 + (2 / (9 * item_count))) / np.sqrt(2 / (9 * item_count))

        item_infit_ms = (residual_df ** 2).sum() / info_df.sum()
        item_infit_zstd = (np.cbrt(item_infit_ms) - 1 + (2 / (9 * item_count))) / np.sqrt(2 / (9 * item_count))

        self.item_facilities = self.dataframe.mean(axis=0) / self.max_score

        abils_by_rater = {rater: abilities for rater in self.raters}
        abils_by_rater = pd.concat(abils_by_rater.values(), keys=abils_by_rater.keys())
        abils_by_rater.index.names = self.dataframe.index.names
        abil_deviation = abils_by_rater.copy() - abils_by_rater.mean()
        abil_deviation = abil_deviation.loc[exp_score_df.index]

        point_measure_df = self.dataframe.copy()
        for item in self.dataframe.columns:
            point_measure_df[item] -= self.item_facilities[item]

        point_measure_num = point_measure_df.mul(abil_deviation, axis=0).sum()
        point_measure_den = np.sqrt((point_measure_df ** 2).sum() * (abil_deviation ** 2).sum())

        item_point_measure = point_measure_num / point_measure_den

        exp_point_measure_df = exp_score_df - self.dataframe.loc[exp_score_df.index].mean()
        exp_point_measure_num = exp_point_measure_df.mul(abil_deviation, axis=0).sum()

        exp_point_measure_den = ((exp_point_measure_df ** 2) + info_df).sum()
        exp_point_measure_den *= (abil_deviation ** 2).sum()
        exp_point_measure_den = np.sqrt(exp_point_measure_den)

        item_exp_point_measure = exp_point_measure_num / exp_point_measure_den

        return (item_outfit_ms,
                item_outfit_zstd,
                item_infit_ms,
                item_infit_zstd,
                item_point_measure,
                item_exp_point_measure)

    def item_fit_statistics_global(self,
                                   anchor_raters=[],
                                   warm_corr=True,
                                   tolerance=0.0000001,
                                   max_iters=100,
                                   ext_score_adjustment=0.5,
                                   method='cos',
                                   constant=0.1,
                                   no_of_samples=100,
                                   interval=None):

        if hasattr(self, 'exp_score_df_global') == False:
            self.fit_matrices_global(anchor_raters=anchor_raters, warm_corr=warm_corr, tolerance=tolerance,
                                     max_iters=max_iters, ext_score_adjustment=ext_score_adjustment, method=method,
                                     constant=constant, no_of_samples=no_of_samples, interval=interval)

        (self.item_outfit_ms_global,
         self.item_outfit_zstd_global,
         self.item_infit_ms_global,
         self.item_infit_zstd_global,
         self.point_measure_global,
         self.exp_point_measure_global) = self.item_fit_statistics(self.exp_score_df_global, self.info_df_global,
                                                                   self.kurtosis_df_global, self.residual_df_global,
                                                                   self.std_residual_df_global, self.abils_global,
                                                                   warm_corr=warm_corr, tolerance=tolerance,
                                                                   max_iters=max_iters,
                                                                   ext_score_adjustment=ext_score_adjustment,
                                                                   method=method, constant=constant,
                                                                   no_of_samples=no_of_samples,
                                                                   interval=interval)

    def item_fit_statistics_items(self,
                                  anchor_raters=[],
                                  warm_corr=True,
                                  tolerance=0.0000001,
                                  max_iters=100,
                                  ext_score_adjustment=0.5,
                                  method='cos',
                                  constant=0.1,
                                  no_of_samples=100,
                                  interval=None):

        if hasattr(self, 'exp_score_df_items') == False:
            self.fit_matrices_items(anchor_raters=anchor_raters, warm_corr=warm_corr, tolerance=tolerance,
                                    max_iters=max_iters, ext_score_adjustment=ext_score_adjustment, method=method,
                                    constant=constant, no_of_samples=no_of_samples, interval=interval)

            (self.item_outfit_ms_items,
             self.item_outfit_zstd_items,
             self.item_infit_ms_items,
             self.item_infit_zstd_items,
             self.point_measure_items,
             self.exp_point_measure_items) = self.item_fit_statistics(self.exp_score_df_items, self.info_df_items,
                                                                      self.kurtosis_df_items, self.residual_df_items,
                                                                      self.std_residual_df_items, self.abils_items,
                                                                      warm_corr=warm_corr, tolerance=tolerance,
                                                                      max_iters=max_iters,
                                                                      ext_score_adjustment=ext_score_adjustment,
                                                                      method=method, constant=constant,
                                                                      no_of_samples=no_of_samples,
                                                                      interval=interval)

    def item_fit_statistics_thresholds(self,
                                       anchor_raters=[],
                                       warm_corr=True,
                                       tolerance=0.0000001,
                                       max_iters=100,
                                       ext_score_adjustment=0.5,
                                       method='cos',
                                       constant=0.1,
                                       no_of_samples=100,
                                       interval=None):

        if hasattr(self, 'exp_score_df_thresholds') == False:
            self.fit_matrices_thresholds(anchor_raters=anchor_raters, warm_corr=warm_corr, tolerance=tolerance,
                                         max_iters=max_iters, ext_score_adjustment=ext_score_adjustment, method=method,
                                         constant=constant, no_of_samples=no_of_samples, interval=interval)

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
                                                                       self.abils_thresholds, warm_corr=warm_corr,
                                                                       tolerance=tolerance, max_iters=max_iters,
                                                                       ext_score_adjustment=ext_score_adjustment,
                                                                       method=method, constant=constant,
                                                                       no_of_samples=no_of_samples,
                                                                       interval=interval)

    def item_fit_statistics_matrix(self,
                                   anchor_raters=[],
                                   warm_corr=True,
                                   tolerance=0.0000001,
                                   max_iters=100,
                                   ext_score_adjustment=0.5,
                                   method='cos',
                                   constant=0.1,
                                   no_of_samples=100,
                                   interval=None):

        if hasattr(self, 'exp_score_df_matrix') == False:
            self.fit_matrices_matrix(anchor_raters=anchor_raters, warm_corr=warm_corr, tolerance=tolerance,
                                     max_iters=max_iters, ext_score_adjustment=ext_score_adjustment, method=method,
                                     constant=constant, no_of_samples=no_of_samples, interval=interval)

        (self.item_outfit_ms_matrix,
         self.item_outfit_zstd_matrix,
         self.item_infit_ms_matrix,
         self.item_infit_zstd_matrix,
         self.point_measure_matrix,
         self.exp_point_measure_matrix) = self.item_fit_statistics(self.exp_score_df_matrix, self.info_df_matrix,
                                                                   self.kurtosis_df_matrix, self.residual_df_matrix,
                                                                   self.std_residual_df_matrix, self.abils_matrix,
                                                                   warm_corr=warm_corr, tolerance=tolerance,
                                                                   max_iters=max_iters,
                                                                   ext_score_adjustment=ext_score_adjustment,
                                                                   method=method, constant=constant,
                                                                   no_of_samples=no_of_samples, interval=interval)

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

            item_loadings = item_eigenvectors.T * np.sqrt(pca.explained_variance_)
            item_loadings = pd.DataFrame(item_loadings)
            item_loadings.columns = [f'PC {pc + 1}' for pc in range(self.no_of_items)]
            item_loadings.index = [item for item in self.dataframe.columns]

        except:
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
                                      tolerance=0.0000001,
                                      max_iters=100,
                                      ext_score_adjustment=0.5,
                                      constant=0.1,
                                      method='cos',
                                      matrix_power=3,
                                      log_lik_tol=0.000001):

        if hasattr(self, 'std_residual_df_global') == False:
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
                                     tolerance=0.0000001,
                                     max_iters=100,
                                     ext_score_adjustment=0.5,
                                     constant=0.1,
                                     method='cos',
                                     matrix_power=3,
                                     log_lik_tol=0.000001):

        if hasattr(self, 'std_residual_df_items') == False:
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
                                          tolerance=0.0000001,
                                          max_iters=100,
                                          ext_score_adjustment=0.5,
                                          constant=0.1,
                                          method='cos',
                                          matrix_power=3,
                                          log_lik_tol=0.000001):

        if hasattr(self, 'std_residual_df_thresholds') == False:
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
                                      tolerance=0.0000001,
                                      max_iters=100,
                                      ext_score_adjustment=0.5,
                                      constant=0.1,
                                      method='cos',
                                      matrix_power=3,
                                      log_lik_tol=0.000001):

        if hasattr(self, 'std_residual_df_matrix') == False:
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
            missing_mask = (dich_thresh[threshold + 1] + 1) / (dich_thresh[threshold + 1] + 1)
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
                                                 np.sqrt(dich_thresh_var[threshold + 1]))

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
        threshold_outfit_q = np.sqrt(threshold_outfit_q)

        threshold_outfit_zstd = ((np.cbrt(threshold_outfit_ms) - 1) *
                                 (3 / threshold_outfit_q) + (threshold_outfit_q / 3))

        threshold_infit_q = {threshold + 1: ((dich_thresh_kur[threshold + 1] -
                                              dich_thresh_var[threshold + 1] ** 2).sum().sum() /
                                             (dich_thresh_var[threshold + 1].sum().sum() ** 2))
                             for threshold in range(self.max_score)}
        threshold_infit_q = pd.Series(threshold_infit_q)
        threshold_infit_q = np.sqrt(threshold_infit_q)

        threshold_infit_zstd = ((np.cbrt(threshold_infit_ms) - 1) *
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

        point_measure_dens = {threshold + 1: np.sqrt((point_measure_dict[threshold + 1] ** 2).sum().sum() *
                                                     (abil_deviation ** 2).sum())
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
        threshold_exp_pm_den = np.sqrt(threshold_exp_pm_den)

        threshold_exp_point_measure = threshold_exp_pm_num / threshold_exp_pm_den

        threshold_rmsr = {threshold + 1: (np.sqrt((dich_residuals[threshold + 1] ** 2).sum().sum() /
                                                  dich_residuals[threshold + 1].count().sum()))
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
                                        anchor_raters=[],
                                        warm_corr=True,
                                        tolerance=0.0000001,
                                        max_iters=100,
                                        ext_score_adjustment=0.5,
                                        method='cos',
                                        constant=0.1):

        if hasattr(self, 'abils_global') == False:
            self.person_abils_global(anchor=False, warm_corr=warm_corr, tolerance=tolerance,
                                     max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)

        if anchor_raters != []:
            if hasattr(self, 'anchor_thresholds_global') == False:
                self.calibrate_global_anchor(anchor_raters, constant=constant, method=method)

                self.person_abils_global(anchor=False, warm_corr=warm_corr, tolerance=tolerance,
                                         max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)

        if anchor_raters != []:
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
                                       anchor_raters=[],
                                       warm_corr=True,
                                       tolerance=0.0000001,
                                       max_iters=100,
                                       ext_score_adjustment=0.5,
                                       method='cos',
                                       constant=0.1):

        if hasattr(self, 'abils_items') == False:
            self.person_abils_items(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                    ext_score_adjustment=ext_score_adjustment)

        if anchor_raters != []:
            if hasattr(self, 'anchor_thresholds_items') == False:
                self.calibrateitems_anchor(anchor_raters, constant=constant, method=method)

                self.person_abils_items(anchor_raters=anchor_raters, warm_corr=warm_corr, tolerance=tolerance,
                                        max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)

        if anchor_raters != []:
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
                                            anchor_raters=[],
                                            warm_corr=True,
                                            tolerance=0.0000001,
                                            max_iters=100,
                                            ext_score_adjustment=0.5,
                                            method='cos',
                                            constant=0.1):

        if hasattr(self, 'abils_thresholds') == False:
            self.person_abils_thresholds(anchor_raters=anchor_raters, warm_corr=warm_corr, tolerance=tolerance,
                                         max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)

        if anchor_raters != []:
            if hasattr(self, 'anchor_thresholds_thresholds') == False:
                self.calibrate_thresholds_anchor(anchor_raters, constant=constant, method=method)

                self.person_abils_thresholds(anchor_raters=anchor_raters, warm_corr=warm_corr, tolerance=tolerance,
                                             max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)

        if anchor_raters != []:
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
                                        anchor_raters=[],
                                        warm_corr=True,
                                        tolerance=0.0000001,
                                        max_iters=100,
                                        ext_score_adjustment=0.5,
                                        method='cos',
                                        constant=0.1):

        if hasattr(self, 'abils_matrix') == False:
            self.person_abils_matrix(anchor_raters=anchor_raters, warm_corr=warm_corr, tolerance=tolerance,
                                     max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)

        if anchor_raters != []:
            if hasattr(self, 'anchor_thresholds_matrix') == False:
                self.calibrate_matrix_anchor(anchor_raters, constant=constant, method=method)

                self.person_abils_matrix(anchor_raters=anchor_raters, warm_corr=warm_corr, tolerance=tolerance,
                                         max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)

        if anchor_raters != []:
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
        rater_outfit_q = np.sqrt(rater_outfit_q)
        rater_outfit_zstd = ((np.cbrt(rater_outfit_ms) - 1) *
                             (3 / rater_outfit_q)) + (rater_outfit_q / 3)

        rater_infit_q = ((self.rater_pivot(kurtosis_df) - self.rater_pivot(info_df) ** 2).sum() /
                         (self.rater_pivot(info_df).sum() ** 2))
        rater_infit_q = np.sqrt(rater_infit_q)
        rater_infit_zstd = ((np.cbrt(rater_infit_ms) - 1) *
                            (3 / rater_infit_q)) + (rater_infit_q / 3)

        return (rater_outfit_ms, rater_outfit_zstd, rater_infit_ms, rater_infit_zstd)

    def rater_fit_statistics_global(self,
                                    warm_corr=True,
                                    tolerance=0.0000001,
                                    max_iters=100,
                                    ext_score_adjustment=0.5,
                                    method='cos',
                                    constant=0.1,
                                    no_of_samples=100,
                                    interval=None):

        if hasattr(self, 'exp_score_df_global') == False:
            self.fit_matrices_global(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                     ext_score_adjustment=ext_score_adjustment, method=method,
                                     constant=constant, no_of_samples=no_of_samples, interval=interval)

        (self.rater_outfit_ms_global,
         self.rater_outfit_zstd_global,
         self.rater_infit_ms_global,
         self.rater_infit_zstd_global) = self.rater_fit_statistics(self.info_df_global, self.kurtosis_df_global,
                                                                   self.residual_df_global, self.std_residual_df_global)

    def rater_fit_statistics_items(self,
                                   warm_corr=True,
                                   tolerance=0.0000001,
                                   max_iters=100,
                                   ext_score_adjustment=0.5,
                                   method='cos',
                                   constant=0.1,
                                   no_of_samples=100,
                                   interval=None):

        if hasattr(self, 'exp_score_df_items') == False:
            self.fit_matrices_items(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                    ext_score_adjustment=ext_score_adjustment, method=method,
                                    constant=constant, no_of_samples=no_of_samples, interval=interval)

        (self.rater_outfit_ms_items,
         self.rater_outfit_zstd_items,
         self.rater_infit_ms_items,
         self.rater_infit_zstd_items) = self.rater_fit_statistics(self.info_df_items, self.kurtosis_df_items,
                                                                  self.residual_df_items, self.std_residual_df_items)

    def rater_fit_statistics_thresholds(self,
                                        warm_corr=True,
                                        tolerance=0.0000001,
                                        max_iters=100,
                                        ext_score_adjustment=0.5,
                                        method='cos',
                                        constant=0.1,
                                        no_of_samples=100,
                                        interval=None):

        if hasattr(self, 'exp_score_df_thresholds') == False:
            self.fit_matrices_thresholds(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                         ext_score_adjustment=ext_score_adjustment, method=method,
                                         constant=constant, no_of_samples=no_of_samples, interval=interval)

        (self.rater_outfit_ms_thresholds,
         self.rater_outfit_zstd_thresholds,
         self.rater_infit_ms_thresholds,
         self.rater_infit_zstd_thresholds) = self.rater_fit_statistics(self.info_df_thresholds,
                                                                       self.kurtosis_df_thresholds,
                                                                       self.residual_df_thresholds,
                                                                      self.std_residual_df_thresholds)

    def rater_fit_statistics_matrix(self,
                                    warm_corr=True,
                                    tolerance=0.0000001,
                                    max_iters=100,
                                    ext_score_adjustment=0.5,
                                    method='cos',
                                    constant=0.1,
                                    no_of_samples=100,
                                    interval=None):

        if hasattr(self, 'exp_score_df_matrix') == False:
            self.fit_matrices_matrix(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                     ext_score_adjustment=ext_score_adjustment, method=method,
                                     constant=constant, no_of_samples=no_of_samples, interval=interval)

        (self.rater_outfit_ms_matrix,
         self.rater_outfit_zstd_matrix,
         self.rater_infit_ms_matrix,
         self.rater_infit_zstd_matrix) = self.rater_fit_statistics(self.info_df_matrix, self.kurtosis_df_matrix,
                                                                   self.residual_df_matrix, self.std_residual_df_matrix)

    def rater_res_corr_analysis(self,
                                std_residual_df):

        rater_std_residual_df = self.rater_pivot(std_residual_df)
        rater_residual_correlations = rater_std_residual_df.corr(numeric_only=False)

        pca = PCA()

        try:
            pca.fit(rater_residual_correlations)

            rater_eigenvectors = pd.DataFrame(pca.components_)
            rater_eigenvectors.columns = [f'Eigenvector {pc + 1}' for pc in range(self.no_of_raters)]

            rater_eigenvalues = pca.explained_variance_
            rater_eigenvalues = pd.DataFrame(rater_eigenvalues)
            rater_eigenvalues.index = [f'PC {pc + 1}' for pc in range(self.no_of_raters)]
            rater_eigenvalues.columns = ['Eigenvalue']

            rater_variance_explained = pd.DataFrame(pca.explained_variance_ratio_)
            rater_variance_explained.index = [f'PC {pc + 1}' for pc in range(self.no_of_raters)]
            rater_variance_explained.columns = ['Variance explained']

            rater_loadings = rater_eigenvectors.T * np.sqrt(pca.explained_variance_)
            rater_loadings = pd.DataFrame(rater_loadings)
            rater_loadings.columns = [f'PC {pc + 1}' for pc in range(self.no_of_raters)]
            rater_loadings.index = [rater for rater in self.raters]

        except:
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
         self.rater_loadings_global) = self.rater_res_corr_analysis(self.std_residual_df_global)

    def rater_res_corr_analysis_items(self):

        (self.rater_residual_correlations_items,
         self.rater_eigenvectors_items,
         self.rater_eigenvalues_items,
         self.rater_variance_explained_items,
         self.rater_loadings_items) = self.rater_res_corr_analysis(self.std_residual_df_items)

    def rater_res_corr_analysis_thresholds(self):

        (self.rater_residual_correlations_thresholds,
         self.rater_eigenvectors_thresholds,
         self.rater_eigenvalues_thresholds,
         self.rater_variance_explained_thresholds,
         self.rater_loadings_thresholds) = self.rater_res_corr_analysis(self.std_residual_df_thresholds)

    def rater_res_corr_analysis_matrix(self):

        (self.rater_residual_correlations_matrix,
         self.rater_eigenvectors_matrix,
         self.rater_eigenvalues_matrix,
         self.rater_variance_explained_matrix,
         self.rater_loadings_matrix) = self.rater_res_corr_analysis(self.std_residual_df_matrix)

    def person_fit_statistics(self,
                              info_df,
                              kurtosis_df,
                              residual_df,
                              std_residual_df,
                              abilities,
                              warm_corr=True,
                              tolerance=0.0000001,
                              max_iters=100,
                              ext_score_adjustment=0.5,
                              method='cos',
                              constant=0.1,
                              no_of_samples=100,
                              interval=None):

        if hasattr(self, 'exp_score_df_global') == False:
            self.fit_matrices_global(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                     ext_score_adjustment=ext_score_adjustment, method=method, constant=constant,
                                     no_of_samples=no_of_samples, interval=interval)

        '''
        Person fit statistics
        '''

        csems = 1 / np.sqrt(info_df.unstack(level=0).sum(axis=1))

        rsems = (np.sqrt((residual_df.unstack(level=0) ** 2).sum(axis=1)) /
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
        person_outfit_q = np.sqrt(person_outfit_q.astype(float))
        person_outfit_zstd = ((np.cbrt(person_outfit_ms) - 1) * (3 / person_outfit_q)) + (person_outfit_q / 3)
        person_outfit_zstd = person_outfit_zstd[:self.no_of_persons]

        person_infit_q = ((kurtosis_df.unstack(level=0) - info_df.unstack(level=0) ** 2).sum(axis=1) /
                          (info_df.unstack(level=0).sum(axis=1) ** 2))
        person_infit_q = np.sqrt(person_infit_q)
        person_infit_zstd = ((np.cbrt(person_infit_ms) - 1) * (3 / person_infit_q)) + (person_infit_q / 3)

        return csems, rsems, person_outfit_ms, person_outfit_zstd, person_infit_ms, person_infit_zstd


    def person_fit_statistics_global(self,
                                     warm_corr=True,
                                     tolerance=0.0000001,
                                     max_iters=100,
                                     ext_score_adjustment=0.5,
                                     method='cos',
                                     constant=0.1):

        if hasattr(self, 'exp_score_df_global') == False:
            self.fit_matrices_global(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                     ext_score_adjustment=ext_score_adjustment, method=method, constant=constant)

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
                                                                     method=method, constant=constant)


    def person_fit_statistics_items(self,
                                    warm_corr=True,
                                    tolerance=0.0000001,
                                    max_iters=100,
                                    ext_score_adjustment=0.5,
                                    method='cos',
                                    constant=0.1):

        if hasattr(self, 'exp_score_df_items') == False:
            self.fit_matrices_items(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                    ext_score_adjustment=ext_score_adjustment, method=method, constant=constant)

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
                                                                    method=method, constant=constant)


    def person_fit_statistics_thresholds(self,
                                         warm_corr=True,
                                         tolerance=0.0000001,
                                         max_iters=100,
                                         ext_score_adjustment=0.5,
                                         method='cos',
                                         constant=0.1):

        if hasattr(self, 'exp_score_df_thresholds') == False:
            self.fit_matrices_thresholds(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                         ext_score_adjustment=ext_score_adjustment, method=method, constant=constant)

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
                                                                         method=method, constant=constant)

    def person_fit_statistics_matrix(self,
                                     warm_corr=True,
                                     tolerance=0.0000001,
                                     max_iters=100,
                                     ext_score_adjustment=0.5,
                                     method='cos',
                                     constant=0.1):

        if hasattr(self, 'exp_score_df_matrix') == False:
            self.fit_matrices_matrix(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                     ext_score_adjustment=ext_score_adjustment, method=method, constant=constant)

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

        isi = np.sqrt(self.diffs.var() / (self.item_se ** 2).mean() - 1)
        item_strata = (4 * isi + 1) / 3
        item_reliability = (isi ** 2) / (1 + isi ** 2)

        psi = np.sqrt(np.var(abilities) -  (rsems ** 2).mean()) / np.sqrt((rsems ** 2).mean())
        person_strata = (4 * psi + 1) / 3
        person_reliability = (psi ** 2) / (1 + (psi ** 2))

        return isi, item_strata, item_reliability, psi, person_strata, person_reliability


    def test_fit_statistics_global(self,
                                   warm_corr=True,
                                   tolerance=0.0000001,
                                   max_iters=100,
                                   ext_score_adjustment=0.5,
                                   method='cos',
                                   constant=0.1):

        if hasattr(self, 'self.csem_vector_global') == False:
            self.person_fit_statistics_global(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                              ext_score_adjustment=ext_score_adjustment, method=method, constant=constant)

        (self.isi_global,
         self.item_strata_global,
         self.item_reliability_global,
         self.psi_global,
         self.person_strata_global,
         self.person_reliability_global) = self.test_fit_statistics(self.abils_global, self.rsem_vector_global)


    def test_fit_statistics_items(self,
                                  warm_corr=True,
                                  tolerance=0.0000001,
                                  max_iters=100,
                                  ext_score_adjustment=0.5,
                                  method='cos',
                                  constant=0.1):

        if hasattr(self, 'self.csem_vector_items') == False:
            self.person_fit_statistics_items(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                             ext_score_adjustment=ext_score_adjustment, method=method, constant=constant)

        (self.isi_items,
         self.item_strata_items,
         self.item_reliability_items,
         self.psi_items,
         self.person_strata_items,
         self.person_reliability_items) = self.test_fit_statistics(self.abils_items, self.rsem_vector_items)


    def test_fit_statistics_thresholds(self,
                                       warm_corr=True,
                                       tolerance=0.0000001,
                                       max_iters=100,
                                       ext_score_adjustment=0.5,
                                       method='cos',
                                       constant=0.1):

        if hasattr(self, 'self.csem_vector_thresholds') == False:
            self.person_fit_statistics_thresholds(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                                  ext_score_adjustment=ext_score_adjustment, method=method, constant=constant)

        (self.isi_thresholds,
         self.item_strata_thresholds,
         self.item_reliability_thresholds,
         self.psi_thresholds,
         self.person_strata_thresholds,
         self.person_reliability_thresholds) = self.test_fit_statistics(self.abils_thresholds, self.rsem_vector_thresholds)


    def test_fit_statistics_matrix(self,
                                   warm_corr=True,
                                   tolerance=0.0000001,
                                   max_iters=100,
                                   ext_score_adjustment=0.5,
                                   method='cos',
                                   constant=0.1):

        if hasattr(self, 'self.csem_vector_matrix') == False:
            self.person_fit_statistics_matrix(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                              ext_score_adjustment=ext_score_adjustment, method=method, constant=constant)

        (self.isi_matrix,
         self.item_strata_matrix,
         self.item_reliability_matrix,
         self.psi_matrix,
         self.person_strata_matrix,
         self.person_reliability_matrix) = self.test_fit_statistics(self.abils_matrix, self.rsem_vector_matrix)

    def fit_statistics_global(self,
                              warm_corr=True,
                              tolerance=0.0000001,
                              max_iters=100,
                              ext_score_adjustment=0.5,
                              method='cos',
                              constant=0.1,
                              no_of_samples=100,
                              interval=None):

        '''
        All fit statistics
        '''

        if hasattr(self, 'thresholds_global') == False:
            self.calibrate_global(constant, method)

        if hasattr(self, 'threshold_se_global') == False:
            self.std_errors_global(interval, no_of_samples, constant, method)

        if hasattr(self, 'abils_global') == False:
            self.person_abils_global(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                     ext_score_adjustment=ext_score_adjustment)

        self.category_probability_dict_global(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                              ext_score_adjustment=ext_score_adjustment, method=method,
                                              constant=constant, no_of_samples=no_of_samples, interval=interval)

        self.fit_matrices_global()
        self.item_fit_statistics_global()
        self.threshold_fit_statistics_global()
        self.rater_fit_statistics_global()
        self.person_fit_statistics_global()
        self.test_fit_statistics_global()


    def fit_statistics_items(self,
                             warm_corr=True,
                             tolerance=0.0000001,
                             max_iters=100,
                             ext_score_adjustment=0.5,
                             method='cos',
                             constant=0.1,
                             no_of_samples=100,
                             interval=None):
        '''
        All fit statistics
        '''

        if hasattr(self, 'thresholds_items') == False:
            self.calibrate_items(constant, method)

        if hasattr(self, 'threshold_se_items') == False:
            self.std_errors_items(interval, no_of_samples, constant, method)

        if hasattr(self, 'abils_items') == False:
            self.person_abils_items(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                    ext_score_adjustment=ext_score_adjustment)

        self.category_probability_dict_items(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                             ext_score_adjustment=ext_score_adjustment, method=method,
                                             constant=constant, no_of_samples=no_of_samples, interval=interval)

        self.fit_matrices_items()
        self.item_fit_statistics_items()
        self.threshold_fit_statistics_items()
        self.rater_fit_statistics_items()
        self.person_fit_statistics_items()
        self.test_fit_statistics_items()


    def fit_statistics_thresholds(self,
                                  warm_corr=True,
                                  tolerance=0.0000001,
                                  max_iters=100,
                                  ext_score_adjustment=0.5,
                                  method='cos',
                                  constant=0.1,
                                  no_of_samples=100,
                                  interval=None):
        '''
        All fit statistics
        '''

        if hasattr(self, 'thresholds_thresholds') == False:
            self.calibrate_thresholds(constant, method)

        if hasattr(self, 'threshold_se_thresholds') == False:
            self.std_errors_thresholds(interval, no_of_samples, constant, method)

        if hasattr(self, 'abils_thresholds') == False:
            self.person_abils_thresholds(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                         ext_score_adjustment=ext_score_adjustment)

        self.category_probability_dict_thresholds(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                                  ext_score_adjustment=ext_score_adjustment, method=method,
                                                  constant=constant, no_of_samples=no_of_samples, interval=interval)

        self.fit_matrices_thresholds()
        self.item_fit_statistics_thresholds()
        self.threshold_fit_statistics_thresholds()
        self.rater_fit_statistics_thresholds()
        self.person_fit_statistics_thresholds()
        self.test_fit_statistics_thresholds()


    def fit_statistics_matrix(self,
                              warm_corr=True,
                              tolerance=0.0000001,
                              max_iters=100,
                              ext_score_adjustment=0.5,
                              method='cos',
                              constant=0.1,
                              no_of_samples=100,
                              interval=None):
        '''
        All fit statistics
        '''

        if hasattr(self, 'thresholds_matrix') == False:
            self.calibrate_matrix(constant, method)

        if hasattr(self, 'threshold_se_matrix') == False:
            self.std_errors_matrix(interval, no_of_samples, constant, method)

        if hasattr(self, 'abils_matrix') == False:
            self.person_abils_matrix(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                     ext_score_adjustment=ext_score_adjustment)

        self.category_probability_dict_matrix(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                              ext_score_adjustment=ext_score_adjustment, method=method,
                                              constant=constant, no_of_samples=no_of_samples, interval=interval)

        self.fit_matrices_matrix()
        self.item_fit_statistics_matrix()
        self.threshold_fit_statistics_matrix()
        self.rater_fit_statistics_matrix()
        self.person_fit_statistics_matrix()
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
                       tolerance=0.0000001,
                       max_iters=100,
                       ext_score_adjustment=0.5,
                       method='cos',
                       constant=0.1):

        if hasattr(self, eigenvector_string) == False:
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

                    writer.save()

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

                    writer.save()

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
                                    tolerance=0.0000001,
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
                                   tolerance=0.0000001,
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
                                     	tolerance=0.0000001,
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
                                    tolerance=0.0000001,
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
                                     tolerance=0.0000001,
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
                                    tolerance=0.0000001,
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
                                         tolerance=0.0000001,
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
                                     tolerance=0.0000001,
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
                        no_of_classes=5):

        if (items == 'all') | (items is None):
            items = self.dataframe.columns.tolist()

        if raters == 'all':
            raters = self.raters.tolist()

        if raters == 'none':
            raters = None

        if raters == 'zero':
            raters = None

        class_groups = [f'class_{class_no + 1}' for class_no in range(no_of_classes)]

        df = self.dataframe.copy()

        abil_index = self.dataframe[items].unstack(level=0).dropna(how='any').index
        abils = abilities.loc[abil_index]

        if isinstance(raters, list):
            df = {rater: df.xs(rater) for rater in raters}
            df = pd.concat(df.values(), keys=df.keys())

        elif isinstance(raters, str):
            df = {raters: df.xs(raters)}
            df = pd.concat(df.values(), keys=df.keys())

        elif raters is None:
            df = df

        df = df[items]

        if isinstance(items, list):
            df = df.loc[pd.IndexSlice[:, abil_index], :]

        if isinstance(items, str):
            df = df.loc[pd.IndexSlice[:, abil_index]]

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

        if (isinstance(raters, list)):
            for class_no in range(no_of_classes):
                class_name = f'class_{class_no + 1}'
                df_mask_dict[class_name] = {rater: mask_dict[class_name] for rater in raters}
                df_mask_dict[class_name] = pd.concat(df_mask_dict[class_name].values(),
                                                     keys=df_mask_dict[class_name].keys())

        if (isinstance(raters, str)):
            for class_no in range(no_of_classes):
                class_name = f'class_{class_no + 1}'
                df_mask_dict[class_name] = {raters: mask_dict[class_name]}
                df_mask_dict[class_name] = pd.concat(df_mask_dict[class_name].values(),
                                                     keys=df_mask_dict[class_name].keys())

        mean_abilities = {class_group: abils[mask_dict[class_group]].mean()
                          for class_group in class_groups}
        mean_abilities = pd.Series(mean_abilities)

        if isinstance(items, str):
            if raters is None:
                obs = {class_group: sum(df.loc[df_mask_dict[class_group]].mean()
                                        for rater in self.raters.tolist()) / self.no_of_raters
                       for class_group in class_groups}

            elif isinstance(raters, list):
                obs = {class_group: sum(df.loc[df_mask_dict[class_group]].mean()
                                        for rater in raters)
                       for class_group in class_groups}

            else:
                obs = {class_group: df.loc[df_mask_dict[class_group]].mean()
                       for class_group in class_groups}

        if isinstance(items, list):
            if raters is None:
                obs = {class_group: sum(df.loc[df_mask_dict[class_group]].xs(rater).mean().sum()
                                        for rater in self.raters.tolist()) / self.no_of_raters
                       for class_group in class_groups}

            elif isinstance(raters, list):
                obs = {class_group: sum(df.loc[df_mask_dict[class_group]].xs(rater).mean().sum()
                                        for rater in raters)
                       for class_group in class_groups}

            else:
                obs = {class_group: df.loc[df_mask_dict[class_group]].mean().sum()
                       for class_group in class_groups}

        for class_group in class_groups:
            obs[class_group] = pd.Series(obs[class_group])

        obs = pd.concat(obs, keys=obs.keys())

        return mean_abilities, obs

    def class_intervals_cats(self,
                             abilities,
                             difficulties,
                             severities,
                             item=None,
                             rater=None,
                             anchor=False,
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
                df_mask = (df + 1) / (df + 1)
                abil_df = abil_df * df_mask

                mask_scores = df.unstack().unstack()
                mask_abils = abil_df.unstack().unstack()

            else:
                df = df.xs(rater)
                df_mask = (df + 1) / (df + 1)
                abil_df = abil_df.xs(rater) * df_mask

                mask_scores = df.unstack()
                mask_abils = abil_df.unstack()

        else:
            if rater is None:
                df = df[item].unstack(level=0)
                df_mask = (df + 1) / (df + 1)
                abil_df = abil_df[item].unstack(level=0) * df_mask

                mask_scores = df.unstack()
                mask_abils = abil_df.unstack()

            else:
                df = df[item].xs(rater)
                df_mask = (df + 1) / (df + 1)
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

    def class_intervals_thresholds(self,
                                   abilities,
                                   item=None,
                                   rater=None,
                                   anchor=False,
                                   no_of_classes=5):

        if rater == 'none':
            rater = None

        if rater == 'zero':
            rater = None
            
        if anchor:
            difficulties = self.anchor_diffs_global
            severities = self.anchor_severities_global
        
        else:
            difficulties = self.diffs
            severities = self.severities_global

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

            cond_df_mask = (cond_df + 1) / (cond_df + 1)

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

            mean_abilities.append([obs_data_df.loc[mask_dict[class_group]]['ability'].mean()
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
                         score_lines_item=[None, None],
                         score_lines_test=None,
                         point_info_lines_item=[None, None],
                         point_info_lines_test=None,
                         point_csem_lines=None,
                         score_labels=False,
                         x_min=-10,
                         x_max=10,
                         y_max=0,
                         warm=True,
                         cat_highlight=None,
                         graph_title='',
                         y_label='',
                         plot_style='colorblind',
                         black=False,
                         figsize=(8, 6),
                         font='Times',
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

        if raters == 'all':
            raters = self.raters

        if raters == 'none':
            raters = None

        if items == 'all':
            items = self.dataframe.columns

        if items == 'none':
            items = None

        if tex:
            plt.rcParams["text.latex.preamble"].join([r"\usepackage{dashbox}", r"\setmainfont{xcolor}",])
        else:
            plt.rcParams["text.usetex"] = False

        plt.style.use('seaborn-v0_8-' + plot_style)

        graph, ax = plt.subplots(figsize=figsize)

        no_of_plots = y_data.shape[1]

        if black:
            for i in range(no_of_plots):
                ax.plot(x_data, y_data[:, i], '', label=i+1, color='black')

        else:
            for i in range(no_of_plots):
                ax.plot(x_data, y_data[:, i], '', label=i+1)

        if obs is not None:
            try:
                if isinstance(y_obs_data, pd.Series):
                    ax.plot(x_obs_data, y_obs_data, 'o')

                else:
                    no_of_observed_cats = y_obs_data.shape[1]
                    for j in range (no_of_observed_cats):
                        ax.plot(x_obs_data, y_obs_data[:, j], 'o')

            except:
                pass

        if thresh_obs is not None:
            if thresh_obs == 'all':
                thresh_obs = np.arange(self.max_score + 1)
            try:
                for ob in thresh_obs:
                        ax.plot(x_obs_data[ob - 1, :], y_obs_data[ob - 1, :], 'o')

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
                        plt.axvline(x=difficulties[items] + thresholds[threshold + 1],
                                    color='black', linestyle='--')

                else:
                    for threshold in range(self.max_score):
                        plt.axvline(x=difficulties[items] +thresholds[threshold + 1] + severities[raters],
                                    color='black', linestyle='--')

        if central_diff:
            if items is None:
                if raters is None:
                    plt.axvline(x=0, color='darkred', linestyle='--')

                else:
                    plt.axvline(x=severities[raters], color='darkred', linestyle='--')

            else:
                if raters is None:
                    plt.axvline(x = difficulties[items], color='darkred', linestyle='--')

                else:
                    plt.axvline(x = difficulties[items] + severities[raters], color='darkred', linestyle='--')

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
                info_set = [self.variance_global(ability, item, difficulties, 'dummy_rater',
                                                 pd.Series({'dummy_rater': 0}), thresholds)
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
                    info_set = [sum(self.variance_global(ability, item, difficulties, 'dummy_rater',
                                                         pd.Series({'dummy_rater': 0}), thresholds)
                                    for item in self.dataframe.columns)
                                for ability in point_info_lines_test]

                else:
                    info_set = [sum(self.variance_global(ability, item, difficulties, rater, severities, thresholds)
                                    for item in self.dataframe.columns for rater in raters)
                                for ability in point_info_lines_test]
                    
            else:
                if raters is None:
                    info_set = [sum(self.variance_global(ability, item, difficulties, 'dummy_rater',
                                                         pd.Series({'dummy_rater': 0}), thresholds)
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
                    info_set = [sum(self.variance_global(ability, item, difficulties, 'dummy_rater',
                                                         pd.Series({'dummy_rater': 0}), thresholds)
                                    for item in self.dataframe.columns)
                                for ability in point_csem_lines]

                else:
                    info_set = [sum(self.variance_global(ability, item, difficulties, rater, severities, thresholds)
                                    for item in self.dataframe.columns for rater in raters)
                                for ability in point_csem_lines]
                    
            else:
                if raters is None:
                    info_set = [sum(self.variance_global(ability, item, difficulties, 'dummy_rater',
                                                         pd.Series({'dummy_rater': 0}), thresholds)
                                    for item in items)
                                for ability in point_csem_lines]

                else:
                    info_set = [sum(self.variance_global(ability, item, difficulties, rater, severities, thresholds)
                                    for item in items for rater in raters)
                                for ability in point_csem_lines]
            
            info_set = np.array(info_set)
            csem_set = 1 / np.sqrt(info_set)

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
                    plt.axvspan(-100, thresholds[1] + severities,
                                facecolor='blue', alpha=0.2)
                else:
                    plt.axvspan(-100, difficulties + thresholds[1] + severities,
                                facecolor='blue', alpha=0.2)

            elif cat_highlight == self.max_score:
                if items is None:
                    plt.axvspan(thresholds[self.max_score] + severities,
                                100, facecolor='blue', alpha=0.2)
                else:
                    plt.axvspan(difficulties + thresholds[self.max_score] + severities,
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

        plt.xlabel('Ability', fontname=font, fontsize=axis_font_size, fontweight='bold')
        plt.ylabel(y_label, fontname=font, fontsize=axis_font_size, fontweight='bold')
        plt.title(graph_title, fontname=font, fontsize=title_font_size, fontweight='bold')

        plt.grid(True)

        plt.tick_params(axis="x", labelsize=labelsize)
        plt.tick_params(axis="y", labelsize=labelsize)

        if filename is not None:
            plt.savefig(f'{filename}.{file_format}', dpi=plot_density)

        plt.close()


        return graph

    def plot_data_items(self,
                        x_data,
                        y_data,
                        anchor=False,
                        rater=None,
                        x_min=-10,
                        x_max=10,
                        y_max=0,
                        item=None,
                        obs=None,
                        thresh_obs=None,
                        x_obs_data=np.array([]),
                        y_obs_data=np.array([]),
                        thresh_lines=False,
                        central_diff=False,
                        score_lines_item=[None, None],
                        score_lines_test=None,
                        point_info_lines_item=[None, None],
                        point_info_lines_test=None,
                        point_csem_lines=None,
                        score_labels=False,
                        cat_highlight=None,
                        graph_title='',
                        y_label='',
                        plot_style='colorblind',
                        black=False,
                        figsize=(8, 6),
                        warm=True,
                        font='Times',
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

        if tex:
            plt.rcParams["text.latex.preamble"].join([r"\usepackage{dashbox}", r"\setmainfont{xcolor}",])
        else:
            plt.rcParams["text.usetex"] = False

        plt.style.use('seaborn-v0_8-' + plot_style)

        graph, ax = plt.subplots(figsize=figsize)

        no_of_plots = y_data.shape[1]

        if black:
            for i in range(no_of_plots):
                ax.plot(x_data, y_data[:, i], '', label=i+1, color='black')

        else:
            for i in range(no_of_plots):
                ax.plot(x_data, y_data[:, i], '', label=i+1)

        if obs is not None:
            try:
                no_of_observed_cats = y_obs_data.shape[1]
                for j in range (no_of_observed_cats):
                    ax.plot(x_obs_data, y_obs_data[:, j], 'o')

            except:
                pass

        if thresh_obs is not None:
            try:
                no_of_observed_cats = y_obs_data.shape[1]
                for j in range (no_of_observed_cats):
                    ax.plot(x_obs_data[j, :], y_obs_data[j, :], 'o')

            except:
                pass

        if anchor:
            if item is not None:
                difficulties = {item: self.anchor_diffs_items[item]}
                difficulties = pd.Series(difficulties)
            thresholds = self.anchor_thresholds_items

            if rater is None:
                severities = {'dummy_rater': {item: 0 for item in self.dataframe.columns}}
                rater = 'dummy_rater'

            else:
                severities = self.anchor_severities_items

        else:
            if item is not None:
                difficulties = {item: self.diffs[item]}
                difficulties = pd.Series(difficulties)
            thresholds = self.thresholds

            if rater is None:
                severities = {'dummy_rater': {item: 0 for item in self.dataframe.columns}}
                rater = 'dummy_rater'

            else:
                severities = self.severities_items

        if thresh_lines:
            for threshold in range(self.max_score):
                if item is None:
                    plt.axvline(x=thresholds[threshold + 1] +
                                np.mean([severities[rater][item] for item in severities[rater].keys()]),
                                color='black', linestyle='--')
                else:
                    plt.axvline(x = difficulties[item] + thresholds[threshold + 1] +
                                severities[rater][item],
                                color='black', linestyle='--')

        if central_diff:
            if item is None:
                plt.axvline(x = np.mean([severities[rater][item] for item in self.dataframe.columns]),
                            color='darkred', linestyle='--')

            else:
                plt.axvline(x = difficulties[item] + severities[rater][item],
                            color='darkred', linestyle='--')

        if score_lines_item[1] is not None:

            if (all(x > 0 for x in score_lines_item[1]) &
                all(x < self.max_score for x in score_lines_item[1])):

                abils_set = [self.score_abil_items(score, anchor=anchor, rater=rater, difficulties=difficulties,
                                                   thresholds=thresholds, severities=severities, warm_corr=False)
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

            if (all(x > 0 for x in score_lines_test) &
                all(x < self.max_score * self.no_of_items for x in score_lines_test)):

                if anchor:
                    score_line_diffs = self.anchor_diffs_items

                else:
                    score_line_diffs = self.diffs

                abils_set = [self.score_abil_items(score, anchor=anchor, rater=rater, difficulties=score_line_diffs,
                                                   thresholds=thresholds, severities=severities, warm_corr=False)
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

            info_set = [self.variance_items(ability, item, difficulties, rater, severities, thresholds)
                        for ability in point_info_lines_item[1]]

            for abil, info in zip(point_info_lines_item[1], info_set):
                plt.vlines(x=abil, ymin=-100, ymax=info, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=info, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(x_min + (x_max - x_min) / 100, info + y_max / 50, str(round(info, 3)))

        if point_info_lines_test is not None:

            if anchor:
                point_info_diffs = self.anchor_diffs_items

            else:
                point_info_diffs = self.diffs

            info_set = [sum(self.variance_items(ability, item, point_info_diffs, rater, severities, thresholds)
                            for item in point_info_diffs.keys())
                        for ability in point_info_lines_test]

            for abil, info in zip(point_info_lines_test, info_set):
                plt.vlines(x=abil, ymin=-100, ymax=info, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=info, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(x_min + (x_max - x_min) / 100, info + y_max / 50, str(round(info, 3)))

        if point_csem_lines is not None:

            if anchor:
                point_csem_diffs = self.anchor_diffs_items

            else:
                point_csem_diffs = self.diffs

            info_set = [sum(self.variance_items(ability, item, point_csem_diffs, rater, severities, thresholds)
                            for item in point_csem_diffs.keys())
                        for ability in point_csem_lines]
            info_set = np.array(info_set)
            csem_set = 1 / np.sqrt(info_set)

            for abil, csem in zip(point_csem_lines, csem_set):
                plt.vlines(x=abil, ymin=-100, ymax=csem, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=csem, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(x_min + (x_max - x_min) / 100, csem + y_max / 50, str(round(csem, 3)))

        if cat_highlight in range(self.max_score + 1):

            mean_severity = np.mean([severities[rater][item] for item in self.dataframe.columns])

            if cat_highlight == 0:
                if item is None:
                    plt.axvspan(-100, thresholds[1] + mean_severity,
                                facecolor='blue', alpha=0.2)
                else:
                    plt.axvspan(-100, difficulties[item] + thresholds[1] + severities[rater][item],
                                facecolor='blue', alpha=0.2)

            elif cat_highlight == self.max_score:
                if item is None:
                    plt.axvspan(thresholds[self.max_score] + mean_severity, 100,
                                facecolor='blue', alpha=0.2)
                else:
                    plt.axvspan(difficulties[item] + thresholds[self.max_score] + severities[rater][item], 100,
                                facecolor='blue', alpha=0.2)

            else:
                if (thresholds[cat_highlight + 1] >
                    thresholds[cat_highlight]):
                    if item is None:
                        plt.axvspan(thresholds[cat_highlight] + mean_severity,
                                    thresholds[cat_highlight + 1] + mean_severity,
                                    facecolor='blue', alpha=0.2)
                    else:
                        plt.axvspan(difficulties[item] + thresholds[cat_highlight] + severities[rater][item],
                                    difficulties[item] + thresholds[cat_highlight + 1] + severities[rater][item],
                                    facecolor='blue', alpha=0.2)

        if y_max <= 0:
            y_max = y_data.max() * 1.1

        plt.xlim(x_min, x_max)
        plt.ylim(0, y_max)

        plt.xlabel('Ability', fontname=font, fontsize=axis_font_size, fontweight='bold')
        plt.ylabel(y_label, fontname=font, fontsize=axis_font_size, fontweight='bold')
        plt.title(graph_title, fontname=font, fontsize=title_font_size, fontweight='bold')

        plt.grid(True)

        plt.tick_params(axis="x", labelsize=labelsize)
        plt.tick_params(axis="y", labelsize=labelsize)

        if filename is not None:
            plt.savefig(f'{filename}.{file_format}', dpi=plot_density)

        plt.close()

        return graph

    def plot_data_thresholds(self,
                             x_data,
                             y_data,
                             anchor=False,
                             rater=None,
                             x_min=-10,
                             x_max=10,
                             y_max=0,
                             item=None,
                             obs=None,
                             thresh_obs=None,
                             x_obs_data=np.array([]),
                             y_obs_data=np.array([]),
                             thresh_lines=False,
                             central_diff=False,
                             score_lines_item=[None, None],
                             score_lines_test=None,
                             point_info_lines_item=[None, None],
                             point_info_lines_test=None,
                             point_csem_lines=None,
                             score_labels=False,
                             cat_highlight=None,
                             graph_title='',
                             y_label='',
                             plot_style='colorblind',
                             black=False,
                             figsize=(8, 6),
                             warm=True,
                             font='Times',
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

        if tex:
            plt.rcParams["text.latex.preamble"].join([r"\usepackage{dashbox}", r"\setmainfont{xcolor}",])
        else:
            plt.rcParams["text.usetex"] = False

        plt.style.use('seaborn-v0_8-' + plot_style)

        graph, ax = plt.subplots(figsize=figsize)

        no_of_plots = y_data.shape[1]

        if black:
            for i in range(no_of_plots):
                ax.plot(x_data, y_data[:, i], '', label=i+1, color='black')

        else:
            for i in range(no_of_plots):
                ax.plot(x_data, y_data[:, i], '', label=i+1)

        if obs is not None:
            try:
                no_of_observed_cats = y_obs_data.shape[1]
                for j in range (no_of_observed_cats):
                    ax.plot(x_obs_data, y_obs_data[:, j], 'o')

            except:
                pass

        if thresh_obs is not None:
            try:
                no_of_observed_cats = y_obs_data.shape[1]
                for j in range (no_of_observed_cats):
                    ax.plot(x_obs_data[j, :], y_obs_data[j, :], 'o')

            except:
                pass

        if anchor:
            if item is not None:
                difficulties = {item: self.anchor_diffs_thresholds[item]}
                difficulties = pd.Series(difficulties)
            thresholds = self.anchor_thresholds_thresholds

            if rater is None:
                severities = {'dummy_rater': np.zeros(self.max_score + 1)}
                rater = 'dummy_rater'

            else:
                severities = self.anchor_severities_thresholds

        else:
            if item is not None:
                difficulties = {item: self.diffs[item]}
                difficulties = pd.Series(difficulties)
            thresholds = self.thresholds

            if rater is None:
                severities = {'dummy_rater': np.zeros(self.max_score + 1)}
                rater = 'dummy_rater'

            else:
                severities = self.severities_thresholds

        if thresh_lines:
            for threshold in range(self.max_score):
                if item is None:
                    plt.axvline(x=thresholds[threshold + 1] + severities[rater][threshold + 1],
                                color='black', linestyle='--')
                else:
                    plt.axvline(x = difficulties[item] + thresholds[threshold + 1] +
                                severities[rater][threshold + 1],
                                color='black', linestyle='--')

        if central_diff:
            if item is None:
                plt.axvline(x = np.mean(severities[rater][1:]),
                            color='darkred', linestyle='--')

            else:
                plt.axvline(x = difficulties[item] + np.mean(severities[rater][1:]),
                            color='darkred', linestyle='--')

        if score_lines_item[1] is not None:

            if (all(x > 0 for x in score_lines_item[1]) &
                all(x < self.max_score for x in score_lines_item[1])):

                abils_set = [self.score_abil_thresholds(score, anchor=anchor, rater=rater, difficulties=difficulties,
                                                        thresholds=thresholds, severities=severities, warm_corr=False)
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

            if (all(x > 0 for x in score_lines_test) &
                all(x < self.max_score * self.no_of_items for x in score_lines_test)):

                if anchor:
                    score_line_diffs = self.anchor_diffs_thresholds

                else:
                    score_line_diffs = self.diffs

                abils_set = [self.score_abil_thresholds(score, anchor=anchor, rater=rater, difficulties=score_line_diffs,
                                                        thresholds=thresholds, severities=severities, warm_corr=False)
                             for score in score_lines_test]

                for thresh, abil in zip(score_lines_item[1], abils_set):
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

            info_set = [self.variance_thresholds(ability, item, difficulties, rater, severities, thresholds)
                        for ability in point_info_lines_item[1]]

            for abil, info in zip(point_info_lines_item[1], info_set):
                plt.vlines(x=abil, ymin=-100, ymax=info, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=info, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(x_min + (x_max - x_min) / 100, info + y_max / 50, str(round(info, 3)))

        if point_info_lines_test is not None:

            if anchor:
                point_info_diffs = self.anchor_diffs_thresholds

            else:
                point_info_diffs = self.diffs

            info_set = [sum(self.variance_thresholds(ability, item, point_info_diffs, rater, severities, thresholds)
                            for item in point_info_diffs.keys())
                        for ability in point_info_lines_test]

            for abil, info in zip(point_info_lines_test, info_set):
                plt.vlines(x=abil, ymin=-100, ymax=info, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=info, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(x_min + (x_max - x_min) / 100, info + y_max / 50, str(round(info, 3)))

        if point_csem_lines is not None:

            if anchor:
                point_csem_diffs = self.anchor_diffs_thresholds

            else:
                point_csem_diffs = self.diffs

            info_set = [sum(self.variance_thresholds(ability, item, point_csem_diffs, rater, severities, thresholds)
                            for item in point_csem_diffs.keys())
                        for ability in point_csem_lines]
            info_set = np.array(info_set)
            csem_set = 1 / np.sqrt(info_set)

            for abil, csem in zip(point_csem_lines, csem_set):
                plt.vlines(x=abil, ymin=-100, ymax=csem, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=csem, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(x_min + (x_max - x_min) / 100, csem + y_max / 50, str(round(csem, 3)))

        if cat_highlight in range(self.max_score + 1):

            if cat_highlight == 0:
                if item is None:
                    plt.axvspan(-100, thresholds[1] + severities[rater][1],
                                facecolor='blue', alpha=0.2)
                else:
                    plt.axvspan(-100, difficulties[item] + thresholds[1] + severities[rater][1],
                                facecolor='blue', alpha=0.2)

            elif cat_highlight == self.max_score:
                if item is None:
                    plt.axvspan(thresholds[self.max_score] + severities[rater][self.max_score], 100,
                                facecolor='blue', alpha=0.2)
                else:
                    plt.axvspan(difficulties[item] + thresholds[self.max_score] + severities[rater][self.max_score], 100,
                                facecolor='blue', alpha=0.2)

            else:
                if (thresholds[cat_highlight + 1] >
                    thresholds[cat_highlight]):
                    if item is None:
                        plt.axvspan(thresholds[cat_highlight] + severities[rater][cat_highlight],
                                    thresholds[cat_highlight + 1] + severities[rater][cat_highlight + 1],
                                    facecolor='blue', alpha=0.2)
                    else:
                        plt.axvspan(difficulties[item] + thresholds[cat_highlight] + severities[rater][cat_highlight],
                                    difficulties[item] + thresholds[cat_highlight + 1] + severities[rater][cat_highlight + 1],
                                    facecolor='blue', alpha=0.2)

        if y_max <= 0:
            y_max = y_data.max() * 1.1

        plt.xlim(x_min, x_max)
        plt.ylim(0, y_max)

        plt.xlabel('Ability', fontname=font, fontsize=axis_font_size, fontweight='bold')
        plt.ylabel(y_label, fontname=font, fontsize=axis_font_size, fontweight='bold')
        plt.title(graph_title, fontname=font, fontsize=title_font_size, fontweight='bold')

        plt.grid(True)

        plt.tick_params(axis="x", labelsize=labelsize)
        plt.tick_params(axis="y", labelsize=labelsize)

        if filename is not None:
            plt.savefig(f'{filename}.{file_format}', dpi=plot_density)

        plt.close()

        return graph

    def plot_data_matrix(self,
                         x_data,
                         y_data,
                         anchor=False,
                         rater=None,
                         x_min=-10,
                         x_max=10,
                         y_max=0,
                         item=None,
                         obs=None,
                         warm=True,
                         thresh_obs=None,
                         x_obs_data=np.array([]),
                         y_obs_data=np.array([]),
                         thresh_lines=False,
                         central_diff=False,
                         score_lines_item=[None, None],
                         score_lines_test=None,
                         point_info_lines_item=[None, None],
                         point_info_lines_test=None,
                         point_csem_lines=None,
                         score_labels=False,
                         cat_highlight=None,
                         graph_title='',
                         y_label='',
                         plot_style='colorblind',
                         black=False,
                         figsize=(8, 6),
                         font='Times',
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

        if tex:
            plt.rcParams["text.latex.preamble"].join([r"\usepackage{dashbox}", r"\setmainfont{xcolor}",])
        else:
            plt.rcParams["text.usetex"] = False

        plt.style.use('seaborn-v0_8-' + plot_style)

        graph, ax = plt.subplots(figsize=figsize)

        no_of_plots = y_data.shape[1]

        if black:
            for i in range(no_of_plots):
                ax.plot(x_data, y_data[:, i], '', label=i+1, color='black')

        else:
            for i in range(no_of_plots):
                ax.plot(x_data, y_data[:, i], '', label=i+1)

        if obs is not None:
            try:
                no_of_observed_cats = y_obs_data.shape[1]
                for j in range (no_of_observed_cats):
                    ax.plot(x_obs_data, y_obs_data[:, j], 'o')

            except:
                pass

        if thresh_obs is not None:
            try:
                no_of_observed_cats = y_obs_data.shape[1]
                for j in range (no_of_observed_cats):
                    ax.plot(x_obs_data[j, :], y_obs_data[j, :], 'o')

            except:
                pass

        if anchor:
            if item is not None:
                difficulties = {item: self.anchor_diffs_matrix[item]}
                difficulties = pd.Series(difficulties)
            thresholds = self.anchor_thresholds_matrix

            if rater is None:
                severities = {'dummy_rater': {item: np.zeros(self.max_score + 1)
                                              for item in self.dataframe.columns}}
                rater = 'dummy_rater'

            else:
                severities = self.anchor_severities_matrix

        else:
            if item is not None:
                difficulties = {item: self.diffs[item]}
                difficulties = pd.Series(difficulties)
            thresholds = self.thresholds

            if rater is None:
                severities = {'dummy_rater': {item: np.zeros(self.max_score + 1)
                                              for item in self.dataframe.columns}}
                rater = 'dummy_rater'

            else:
                severities = self.severities_matrix

        if thresh_lines:
            for threshold in range(self.max_score):
                if item is None:
                    plt.axvline(x=thresholds[threshold + 1] + np.mean([severities[rater][item][threshold + 1]
                                                                       for item in severities[rater].keys()]),
                                color='black', linestyle='--')
                else:
                    plt.axvline(x = difficulties[item] + thresholds[threshold + 1] +
                                severities[rater][item][threshold + 1],
                                color='black', linestyle='--')

        if central_diff:

            severities_mean_list = {item: np.mean(severities[rater][item][1:]) for item in self.dataframe.columns}
            severities_mean_list = pd.Series(severities_mean_list)
            severities_mean = np.mean(severities_mean_list)

            if item is None:
                plt.axvline(x = severitiies_mean, color='darkred', linestyle='--')

            else:
                plt.axvline(x = difficulties[item] + severities_mean_list[item], color='darkred', linestyle='--')

        if score_lines_item[1] is not None:

            if (all(x > 0 for x in score_lines_item[1]) &
                all(x < self.max_score for x in score_lines_item[1])):

                abils_set = [self.score_abil_matrix(score, anchor=anchor, items=item, raters=rater, warm_corr=False)
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

            if (all(x > 0 for x in score_lines_test) &
                all(x < self.max_score * self.no_of_items for x in score_lines_test)):

                if anchor:
                    score_line_diffs = self.anchor_diffs_matrix

                else:
                    score_line_diffs = self.diffs

                abils_set = [self.score_abil_matrix(score, anchor=anchor, raters=rater, difficulties=score_line_diffs,
                                                    thresholds=thresholds, severities=severities, warm_corr=False)
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

            if anchor:
                point_info_diffs = self.anchor_diffs_matrix

            else:
                point_info_diffs = self.diffs

            item = point_info_lines_item[0]

            info_set = [self.variance_matrix(ability, item, point_info_diffs, rater, severities, thresholds)
                        for ability in point_info_lines_item[1]]

            for abil, info in zip(point_info_lines_item[1], info_set):
                plt.vlines(x=abil, ymin=-100, ymax=info, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=info, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(x_min + (x_max - x_min) / 100, info + y_max / 50, str(round(info, 3)))

        if point_info_lines_test is not None:

            if anchor:
                point_info_diffs = self.anchor_diffs_matrix

            else:
                point_info_diffs = self.diffs

            info_set = [sum(self.variance_matrix(ability, item, point_info_diffs, rater, severities, thresholds)
                            for item in point_info_diffs.keys())
                        for ability in point_info_lines_test]

            for abil, info in zip(point_info_lines_test, info_set):
                plt.vlines(x=abil, ymin=-100, ymax=info, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=info, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(x_min + (x_max - x_min) / 100, info + y_max / 50, str(round(info, 3)))

        if point_csem_lines is not None:

            if anchor:
                point_csem_diffs = self.anchor_diffs_matrix

            else:
                point_csem_diffs = self.diffs

            info_set = [sum(self.variance_matrix(ability, item, point_csem_diffs, rater, severities, thresholds)
                            for item in point_csem_diffs.keys())
                        for ability in point_csem_lines]
            info_set = np.array(info_set)
            csem_set = 1 / np.sqrt(info_set)

            for abil, csem in zip(point_csem_lines, csem_set):
                plt.vlines(x=abil, ymin=-100, ymax=csem, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=csem, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if score_labels:
                    plt.text(x_min + (x_max - x_min) / 100, csem + y_max / 50, str(round(csem, 3)))

        if cat_highlight in range(self.max_score + 1):

            if anchor:
                difficulties = self.anchor_diffs_matrix
                thresholds = self.anchor_thresholds_matrix

            else:
                difficulties = self.diffs
                thresholds = self.thresholds

            if cat_highlight == 0:
                if item is None:
                    plt.axvspan(-100, thresholds[1], facecolor='blue', alpha=0.2)
                else:
                    plt.axvspan(-100, difficulties[item] + thresholds[1], facecolor='blue', alpha=0.2)

            elif cat_highlight == self.max_score:
                if item is None:
                    plt.axvspan(thresholds[self.max_score], 100, facecolor='blue', alpha=0.2)
                else:
                    plt.axvspan(difficulties[item] + thresholds[self.max_score], 100, facecolor='blue', alpha=0.2)

            else:
                if (thresholds[cat_highlight + 1] >
                    thresholds[cat_highlight]):
                    if item is None:
                        plt.axvspan(thresholds[cat_highlight],
                                    thresholds[cat_highlight + 1], facecolor='blue', alpha=0.2)
                    else:
                        plt.axvspan(difficulties[item] + thresholds[cat_highlight],
                                    difficulties[item] + thresholds[cat_highlight + 1], facecolor='blue', alpha=0.2)

        if y_max <= 0:
            y_max = y_data.max() * 1.1

        plt.xlim(x_min, x_max)
        plt.ylim(0, y_max)

        plt.xlabel('Ability', fontname=font, fontsize=axis_font_size, fontweight='bold')
        plt.ylabel(y_label, fontname=font, fontsize=axis_font_size, fontweight='bold')
        plt.title(graph_title, fontname=font, fontsize=title_font_size, fontweight='bold')

        plt.grid(True)
        plt.tick_params(axis="x", labelsize=labelsize)
        plt.tick_params(axis="y", labelsize=labelsize)

        if filename is not None:
            plt.savefig(f'{filename}.{file_format}', dpi=plot_density)

        plt.close()

        return graph

    def icc_global(self,
                   item,
                   anchor=False,
                   rater=None,
                   obs=None,
                   warm=True,
                   xmin=-10,
                   xmax=10,
                   no_of_classes=5,
                   title=None,
                   thresh_lines=False,
                   score_lines=None,
                   score_labels=False,
                   central_diff=False,
                   cat_highlight=None,
                   plot_style='colorblind',
                   black=False,
                   font='Times',
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

        if anchor:
            if hasattr(self, 'anchor_thresholds_global') == False:
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

        if obs:
            if anchor:
                if hasattr(self, 'anchor_abils_global') == False:
                    self.person_abils_global(anchor=True)
                abilities = self.anchor_abils_global

            else:
                if hasattr(self, 'abils_global') == False:
                    self.person_abils_global()
                abilities = self.abils_global

            xobsdata, yobsdata = self.class_intervals(abilities, items=item, raters=rater, no_of_classes=no_of_classes)

            yobsdata = yobsdata.values.reshape((-1, 1))

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)

        if rater is None:
            y = [self.exp_score_global(ability, item, difficulties, 'dummy_rater',
                                       pd.Series({'dummy_rater': 0}), thresholds)
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
                                     plot_style=plot_style, black=black, font=font, title_font_size=title_font_size,
                                     axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                                     plot_density=dpi, file_format=file_format)

        return plot

    def icc_items(self,
                  item,
                  anchor=False,
                  rater=None,
                  obs=None,
                  warm=True,
                  xmin=-10,
                  xmax=10,
                  no_of_classes=5,
                  title=True,
                  thresh_lines=False,
                  score_lines=None,
                  score_labels=False,
                  central_diff=False,
                  cat_highlight=None,
                  plot_style='colorblind',
                  black=False,
                  font='Times',
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

        if anchor:
            if hasattr(self, 'anchor_thresholds_items') == False:
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

        if obs:
            if anchor:
                if hasattr(self, 'anchor_abils_items') == False:
                    self.person_abils_items(anchor=True)
                abilities = self.anchor_abils_items

            else:
                if hasattr(self, 'abils_items') == False:
                    self.person_abils_items()
                abilities = self.abils_items

            xobsdata, yobsdata = self.class_intervals(abilities, items=item, raters=rater, no_of_classes=no_of_classes)

            yobsdata = yobsdata.values.reshape((-1, 1))

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)

        if rater is None:
            dummy_sevs = {'dummy_rater': {item: 0 for item in self.dataframe.columns}}

            y = [self.exp_score_items(ability, item, difficulties, 'dummy_rater',
                                      dummy_sevs, thresholds)
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

        plot = self.plot_data_global(x_data=abilities, y_data=y, anchor=anchor, raters=rater, x_min=xmin, x_max=xmax,
                                     y_max=self.max_score, items=item, obs=obs,  warm=warm, x_obs_data=xobsdata,
                                     y_obs_data=yobsdata, thresh_lines=thresh_lines, graph_title=graphtitle,
                                     score_lines_item=[item, score_lines], score_labels=score_labels,
                                     central_diff=central_diff, cat_highlight=cat_highlight, y_label=ylabel,
                                     plot_style=plot_style, black=black, font=font, title_font_size=title_font_size,
                                     axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                                     plot_density=dpi, file_format=file_format)

        return plot

    def icc_thresholds(self,
                       item,
                       anchor=False,
                       rater=None,
                       obs=None,
                       warm=True,
                       xmin=-10,
                       xmax=10,
                       no_of_classes=5,
                       title=True,
                       thresh_lines=False,
                       score_lines=None,
                       score_labels=False,
                       central_diff=False,
                       cat_highlight=None,
                       plot_style='colorblind',
                       black=False,
                       font='Times',
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

        if anchor:
            if hasattr(self, 'anchor_thresholds_thresholds') == False:
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

        if rater is None:
            severities = {'dummy_rater': np.zeros(self.max_score + 1)}
            rater = 'dummy_rater'

        if obs:
            if rater is not None:
                abilities = {person: self.abil_thresholds(person, raters=[rater])
                             for person in self.persons}
                abilities = pd.Series(abilities)

            else:
                if anchor:
                    if hasattr(self, 'anchor_abils_thresholds') == False:
                        self.person_abils_thresholds(anchor=True)
                    abilities = self.anchor_abils_thresholds

                else:
                    if hasattr(self, 'abils_thresholds') == False:
                        self.person_abils_thresholds()
                    abilities = self.abils_thresholds

            xobsdata, yobsdata = self.class_intervals(abilities, [item], no_of_classes=no_of_classes)

            xobsdata -= np.mean([severities[rater][1:].mean() for rater in self.raters])
            if rater is not None:
                xobsdata += severities[rater][1:].mean()

            yobsdata = np.array(yobsdata).reshape((-1, 1))

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)

        y = [self.exp_score_thresholds(ability, item, difficulties, rater, severities, thresholds)
             for ability in abilities]
        y = np.array(y).reshape([len(abilities), 1])

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Expected score'

        plot = self.plot_data_global(x_data=abilities, y_data=y, anchor=anchor, raters=rater, x_min=xmin, x_max=xmax,
                                     y_max=self.max_score, items=item, obs=obs,  warm=warm, x_obs_data=xobsdata,
                                     y_obs_data=yobsdata, thresh_lines=thresh_lines, graph_title = graphtitle,
                                     score_lines_item=[item, score_lines], score_labels=score_labels,
                                     central_diff=central_diff, cat_highlight= cat_highlight, y_label=ylabel,
                                     plot_style=plot_style, black=black, font=font, title_font_size=title_font_size,
                                     axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                                     plot_density=dpi, file_format=file_format)

        return plot

    def icc_matrix(self,
                   item,
                   anchor=False,
                   rater=None,
                   obs=None,
                   warm=True,
                   xmin=-10,
                   xmax=10,
                   no_of_classes=5,
                   title=True,
                   thresh_lines=False,
                   score_lines=None,
                   score_labels=False,
                   central_diff=False,
                   cat_highlight=None,
                   plot_style='colorblind',
                   black=False,
                   font='Times',
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

        if anchor:
            if hasattr(self, 'anchor_thresholds_matrix') == False:
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

        if rater is None:
            severities = {'dummy_rater': {item: np.zeros(self.max_score + 1)}}
            rater = 'dummy_rater'

        if obs:
            if rater is not None:
                abilities = {person: self.abil_matrix(person, raters=[rater])
                             for person in self.persons}
                abilities = pd.Series(abilities)

            else:
                if anchor:
                    if hasattr(self, 'anchor_abils_matrix') == False:
                        self.person_abils_matrix(anchor=True)
                    abilities = self.anchor_abils_matrix

                else:
                    if hasattr(self, 'abils_matrix') == False:
                        self.person_abils_matrix()
                    abilities = self.abils_matrix

            xobsdata, yobsdata = self.class_intervals(abilities, [item], no_of_classes=no_of_classes)

            xobsdata -= np.mean([severities[rater][item][1:].mean() for rater in self.raters])
            if rater is not None:
                xobsdata += severities[rater][item][1:].mean()

            yobsdata = np.array(yobsdata).reshape((-1, 1))

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)

        y = [self.exp_score_matrix(ability, item, difficulties, rater, severities, thresholds)
             for ability in abilities]
        y = np.array(y).reshape([len(abilities), 1])

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Expected score'

        plot = self.plot_data_global(x_data=abilities, y_data=y, anchor=anchor, raters=rater, x_min=xmin, x_max=xmax,
                                     y_max=self.max_score, items=item, obs=obs,  warm=warm, x_obs_data=xobsdata,
                                     y_obs_data=yobsdata, thresh_lines=thresh_lines, graph_title = graphtitle,
                                     score_lines_item=[item, score_lines], score_labels=score_labels,
                                     central_diff=central_diff, cat_highlight= cat_highlight, y_label=ylabel,
                                     plot_style=plot_style, black=black, font=font, title_font_size=title_font_size,
                                     axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                                     plot_density=dpi, file_format=file_format)

        return plot

    def crcs_global(self,
                    item=None,
                    anchor=False,
                    rater=None,
                    obs=None,
                    xmin=-10,
                    xmax=10,
                    no_of_classes=5,
                    title=None,
                    thresh_lines=False,
                    central_diff=False,
                    cat_highlight=None,
                    plot_style='colorblind',
                    black=False,
                    font='Times',
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

        if anchor:
            if hasattr(self, 'anchor_thresholds_global') == False:
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
            
        if obs == 'none':
            obs = None
            
        if obs == 'all':
            obs = [cat for cat in range(self.max_score + 1)]

        if obs is not None:
            if anchor:
                if hasattr(self, 'anchor_abils_global') == False:
                    self.person_abils_global(anchor=True)
                abilities = self.anchor_abils_global

            else:
                if hasattr(self, 'abils_global') == False:
                    self.person_abils_global()
                abilities = self.abils_global

            xobsdata, yobsdata = self.class_intervals_cats(abilities, difficulties, severities, item=item, rater=rater,
                                                           no_of_classes=no_of_classes)

            xobsdata -= severities.mean()
            if rater is not None:
                xobsdata += severities[rater]

            if obs != 'all':
                yobsdata = yobsdata[obs].T

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)

        if rater is None:
            if item is None:
                y = np.array([[self.cat_prob_global(ability, 'dummy_item', pd.Series({'dummy_item': 0}), 'dummy_rater',
                                                    pd.Series({'dummy_rater': 0}), category, thresholds)
                               for category in range(self.max_score + 1)]
                              for ability in abilities])


            else:
                y = np.array([[self.cat_prob_global(ability, item, difficulties, 'dummy_rater',
                                                    pd.Series({'dummy_rater': 0}), category, thresholds)
                               for category in range(self.max_score + 1)]
                              for ability in abilities])

        else:
            if item is None:
                y = np.array([[self.cat_prob_global(ability, 'dummy_item', pd.Series({'dummy_item': 0}), rater,
                                                    severities, category, thresholds)
                               for category in range(self.max_score + 1)]
                              for ability in abilities])


            else:
                y = np.array([[self.cat_prob_global(ability, item, difficulties, rater, severities, category, thresholds)
                               for category in range(self.max_score + 1)]
                              for ability in abilities])

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Probability'

        plot = self.plot_data_global(x_data=abilities, y_data=y, anchor=anchor, items=item, raters=rater, x_min=xmin,
                                     x_max=xmax, y_max=1, x_obs_data=xobsdata, y_obs_data=yobsdata, y_label=ylabel,
                                     graph_title=graphtitle, obs=obs, thresh_lines=thresh_lines, plot_style=plot_style,
                                     central_diff=central_diff, cat_highlight=cat_highlight, black=black, font=font,
                                     title_font_size=title_font_size, labelsize=labelsize,
                                     axis_font_size=axis_font_size, filename=filename, plot_density=dpi,
                                     file_format=file_format)

        return plot

    def crcs_items(self,
                   item=None,
                   anchor=False,
                   rater=None,
                   obs=None,
                   xmin=-10,
                   xmax=10,
                   no_of_classes=5,
                   title=None,
                   thresh_lines=False,
                   central_diff=False,
                   cat_highlight=None,
                   plot_style='colorblind',
                   black=False,
                   font='Times',
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

        if anchor:
            if hasattr(self, 'anchor_thresholds_items') == False:
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

        if rater is None:
            severities = {'dummy_rater': {item: 0}}
            rater = 'dummy_rater'

        if obs is not None:
            if rater is not None:
                abilities = {person: self.abil_items(person, raters=[rater])
                             for person in self.persons}
                abilities = pd.Series(abilities)

            else:
                if anchor:
                    if hasattr(self, 'anchor_abils_items') == False:
                        self.person_abils_items(anchor=True)
                    abilities = self.anchor_abils_items

                else:
                    if hasattr(self, 'abils_items') == False:
                        self.person_abils_items()
                    abilities = self.abils_items

            xobsdata, yobsdata = self.class_intervals_cats(abilities, [item], no_of_classes=no_of_classes)

            xobsdata -= np.mean([severities[rater][item] for rater in self.raters])
            if rater is not None:
                xobsdata += severities[rater][item]

            if obs != 'all':
                yobsdata = yobsdata[obs].T

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)
        y = np.array([[self.cat_prob_items(ability, item, difficulties, rater, severities, category, thresholds)
                       for category in range(self.max_score + 1)]
                      for ability in abilities])

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Probability'

        plot = self.plot_data_items(x_data=abilities, y_data=y, anchor=anchor, items=item, raters=rater, x_min=xmin,
                                    x_max=xmax, y_max=1, x_obs_data=xobsdata, y_obs_data=yobsdata,y_label=ylabel,
                                    graph_title=graphtitle, obs=obs, thresh_lines=thresh_lines, plot_style=plot_style,
                                    central_diff=central_diff, cat_highlight=cat_highlight, black=black, font=font,
                                    title_font_size=title_font_size, labelsize=labelsize, axis_font_size=axis_font_size,
                                    filename=filename, plot_density=dpi, file_format=file_format)

        return plot

    def crcs_thresholds(self,
                        item=None,
                        anchor=False,
                        rater=None,
                        obs=None,
                        xmin=-10,
                        xmax=10,
                        no_of_classes=5,
                        title=None,
                        thresh_lines=False,
                        central_diff=False,
                        cat_highlight=None,
                        plot_style='colorblind',
                        black=False,
                        font='Times',
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

        if anchor:
            if hasattr(self, 'anchor_thresholds_thresholds') == False:
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

        if rater is None:
            severities = {'dummy_rater': np.zeros(self.max_score + 1)}
            rater = 'dummy_rater'

        if obs is not None:
            if rater is not None:
                abilities = {person: self.abil_thresholds(person, raters=[rater])
                             for person in self.persons}
                abilities = pd.Series(abilities)

            else:
                if anchor:
                    if hasattr(self, 'anchor_abils_thresholds') == False:
                        self.person_abils_thresholds(anchor=True)
                    abilities = self.anchor_abils_thresholds

                else:
                    if hasattr(self, 'abils_thresholds') == False:
                        self.person_abils_thresholds()
                    abilities = self.abils_thresholds

            xobsdata, yobsdata = self.class_intervals_cats(abilities, [item], no_of_classes=no_of_classes)

            xobsdata -= np.mean([severities[rater][1:].mean() for rater in self.raters])
            if rater is not None:
                xobsdata += severities[rater][1:].mean()

            if obs != 'all':
                yobsdata = yobsdata[obs].T

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)

        y = np.array([[self.cat_prob_thresholds(ability, item, difficulties, rater, severities, category, thresholds)
                       for category in range(self.max_score + 1)]
                      for ability in abilities])

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Probability'

        plot = self.plot_data_thresholds(x_data=abilities, y_data=y, anchor=anchor, items=item, raters=rater,
                                         x_min=xmin, x_max=xmax, y_max=1, x_obs_data=xobsdata, y_obs_data=yobsdata,
                                         y_label=ylabel, graph_title=graphtitle, obs=obs, thresh_lines=thresh_lines,
                                         plot_style=plot_style, central_diff=central_diff, cat_highlight=cat_highlight,
                                         black=black, font=font, title_font_size=title_font_size, labelsize=labelsize,
                                         axis_font_size=axis_font_size, filename=filename, plot_density=dpi,
                                         file_format=file_format)

        return plot

    def crcs_matrix(self,
                    item=None,
                    anchor=False,
                    rater=None,
                    obs=None,
                    xmin=-10,
                    xmax=10,
                    no_of_classes=5,
                    title=None,
                    thresh_lines=False,
                    central_diff=False,
                    cat_highlight=None,
                    plot_style='colorblind',
                    black=False,
                    font='Times',
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

        if anchor:
            if hasattr(self, 'anchor_thresholds_matrix') == False:
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

        if rater is None:
            severities = {'dummy_rater': {item: np.zeros(self.max_score + 1)}}
            rater = 'dummy_rater'

        if obs is not None:
            if rater is not None:
                abilities = {person: self.abil_matrix(person, raters=[rater])
                             for person in self.persons}
                abilities = pd.Series(abilities)

            else:
                if anchor:
                    if hasattr(self, 'anchor_abils_matrix') == False:
                        self.person_abils_matrix(anchor=True)
                    abilities = self.anchor_abils_matrix

                else:
                    if hasattr(self, 'abils_matrix') == False:
                        self.person_abils_matrix()
                    abilities = self.abils_matrix

            xobsdata, yobsdata = self.class_intervals_cats(abilities, [item], no_of_classes=no_of_classes)

            xobsdata -= np.mean([severities[rater][item][1:].mean() for rater in self.raters])
            if rater is not None:
                xobsdata += severities[rater][item][1:].mean()

            if obs != 'all':
                yobsdata = yobsdata[obs].T

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)
        y = np.array([[self.cat_prob_matrix(ability, item, difficulties, rater, severities, category, thresholds)
                       for category in range(self.max_score + 1)]
                      for ability in abilities])

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Probability'

        plot = self.plot_data_matrix(x_data=abilities, y_data=y, anchor=anchor, items=item, raters=rater, x_min=xmin,
                                     x_max=xmax, y_max=1, x_obs_data=xobsdata, y_obs_data=yobsdata, y_label=ylabel,
                                     graph_title=graphtitle, obs=obs, thresh_lines=thresh_lines, plot_style=plot_style,
                                     central_diff=central_diff, cat_highlight=cat_highlight, black=black, font=font,
                                     title_font_size=title_font_size, labelsize=labelsize,
                                     axis_font_size=axis_font_size, filename=filename, plot_density=dpi,
                                     file_format=file_format)

        return plot

    def threshold_ccs_global(self,
                             item=None,
                             anchor=False,
                             rater=None,
                             obs=None,
                             warm=True,
                             xmin=-10,
                             xmax=10,
                             no_of_classes=5,
                             title=None,
                             thresh_lines=False,
                             central_diff=False,
                             cat_highlight=None,
                             plot_style='colorblind',
                             black=False,
                             font='Times',
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

        if anchor:
            if hasattr(self, 'anchor_thresholds_global') == False:
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

        if item is None:
            items = self.dataframe.columns

        else: items = [item]

        if obs is not None:

            if anchor:
                if hasattr(self, 'anchor_abils_global') == False:
                    self.person_abils_global(anchor=True)
                abilities = self.anchor_abils_global

            else:
                if hasattr(self, 'abils_global') == False:
                    self.person_abils_global()
                abilities = self.abils_global

            xobsdata, yobsdata = self.class_intervals_thresholds(abilities, item=item, rater=rater, anchor=anchor,
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
                            xmin=-10,
                            xmax=10,
                            no_of_classes=5,
                            title=None,
                            thresh_lines=False,
                            central_diff=False,
                            cat_highlight=None,
                            plot_style='colorblind',
                            black=False,
                            font='Times',
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

        if anchor:
            if hasattr(self, 'anchor_thresholds_items') == False:
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

        if item is None:
            items = self.dataframe.columns

        else: items = [item]

        if obs is not None:
            if rater is not None:
                abilities = {person: self.abil_items(person, raters=[rater])
                             for person in self.persons}
                abilities = pd.Series(abilities)

            else:
                if anchor:
                    if hasattr(self, 'anchor_abils_items') == False:
                        self.person_abils_items(anchor=True)
                    abilities = self.anchor_abils_items

                else:
                    if hasattr(self, 'abils_items') == False:
                        self.person_abils_items()
                    abilities = self.abils_items

            xobsdata, yobsdata = self.class_intervals_thresholds(abilities, items, difficulties=difficulties,
                                                                          no_of_classes=no_of_classes)

            if item is not None:
                xobsdata += difficulties[item]

            yobsdata = yobsdata.T

            if obs != 'all':
                try:
                    obs = [ob - 1 for ob in obs]
                except:
                    print("Invalid 'obs'. Valid values are 'None', 'all' and list of categories.")
                    return

                xobsdata = xobsdata[obs]
                yobsdata = yobsdata[obs]

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)

        if item is None:
            if rater is None:
                adj_thresholds = thresholds[1:]

            else:
                adj_thresholds = thresholds[1:] + np.mean([severities[rater][item] for item in self.dataframe.columns])

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

        plot = self.plot_data_global(x_data=abilities, y_data=y, anchor=anchor, raters=rater, y_max=1, x_min=xmin,
                                     x_max=xmax, items=item, warm=warm, x_obs_data=xobsdata, y_obs_data=yobsdata,
                                     graph_title=graphtitle, y_label=ylabel, thresh_obs=obs, thresh_lines=thresh_lines,
                                     central_diff=central_diff, cat_highlight=cat_highlight, plot_style=plot_style,
                                     black=black, font=font, title_font_size=title_font_size, labelsize=labelsize,
                                     axis_font_size=axis_font_size, filename=filename, file_format=file_format,
                                     plot_density=dpi)

        return plot

    def threshold_ccs_thresholds(self,
                                 item=None,
                                 anchor=False,
                                 rater=None,
                                 obs=None,
                                 warm=True,
                                 xmin=-10,
                                 xmax=10,
                                 no_of_classes=5,
                                 title=None,
                                 thresh_lines=False,
                                 central_diff=False,
                                 cat_highlight=None,
                                 plot_style='colorblind',
                                 black=False,
                                 font='Times',
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

        if anchor:
            if hasattr(self, 'anchor_thresholds_thresholds') == False:
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

        if item is None:
            items = self.dataframe.columns

        else: items = [item]

        if obs is not None:
            if rater is not None:
                abilities = {person: self.abil_thresholds(person, raters=[rater])
                             for person in self.persons}
                abilities = pd.Series(abilities)

            else:
                if anchor:
                    if hasattr(self, 'anchor_abils_thresholds') == False:
                        self.person_abils_thresholds(anchor=True)
                    abilities = self.anchor_abils_thresholds

                else:
                    if hasattr(self, 'abils_thresholds') == False:
                        self.person_abils_thresholds()
                    abilities = self.abils_thresholds

            xobsdata, yobsdata = self.class_intervals_thresholds(abilities, items, difficulties=difficulties,
                                                                          no_of_classes=no_of_classes)

            yobsdata = yobsdata.T

            if obs != 'all':
                try:
                    obs = [ob - 1 for ob in obs]
                except:
                    print("Invalid 'obs'. Valid values are 'None', 'all' and list of categories.")
                    return

                xobsdata = xobsdata[obs]
                yobsdata = yobsdata[obs]

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

        plot = self.plot_data_global(x_data=abilities, y_data=y, anchor=anchor, raters=rater, y_max=1, x_min=xmin,
                                     x_max=xmax, items=item, warm=warm, x_obs_data=xobsdata, y_obs_data=yobsdata,
                                     graph_title=graphtitle, y_label=ylabel, thresh_obs=obs, thresh_lines=thresh_lines,
                                     central_diff=central_diff, cat_highlight=cat_highlight, plot_style=plot_style,
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
                             xmin=-10,
                             xmax=10,
                             no_of_classes=5,
                             title=None,
                             thresh_lines=False,
                             central_diff=False,
                             cat_highlight=None,
                             plot_style='colorblind',
                             black=False,
                             font='Times',
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

        if anchor:
            if hasattr(self, 'anchor_thresholds_matrix') == False:
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

        if item is None:
            items = self.dataframe.columns

        else: items = [item]

        if obs is not None:
            if rater is not None:
                abilities = {person: self.abil_matrix(person, raters=[rater])
                             for person in self.persons}
                abilities = pd.Series(abilities)

            else:
                if anchor:
                    if hasattr(self, 'anchor_abils_matrix') == False:
                        self.person_abils_matrix(anchor=True)
                    abilities = self.anchor_abils_matrix

                else:
                    if hasattr(self, 'abils_matrix') == False:
                        self.person_abils_matrix()
                    abilities = self.abils_matrix

            xobsdata, yobsdata = self.class_intervals_thresholds(abilities, items, difficulties=difficulties,
                                                                          no_of_classes=no_of_classes)

            yobsdata = yobsdata.T

            if obs != 'all':
                try:
                    obs = [ob - 1 for ob in obs]
                except:
                    print("Invalid 'obs'. Valid values are 'None', 'all' and list of categories.")
                    return

                xobsdata = xobsdata[obs]
                yobsdata = yobsdata[obs]

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

        plot = self.plot_data_global(x_data=abilities, y_data=y, anchor=anchor, raters=rater, y_max=1, x_min=xmin,
                                     x_max=xmax, items=item, warm=warm, x_obs_data=xobsdata, y_obs_data=yobsdata,
                                     graph_title=graphtitle, y_label=ylabel, thresh_obs=obs, thresh_lines=thresh_lines,
                                     central_diff=central_diff, cat_highlight=cat_highlight, plot_style=plot_style,
                                     black=black, font=font, title_font_size=title_font_size, labelsize=labelsize,
                                     axis_font_size=axis_font_size, filename=filename, file_format=file_format,
                                     plot_density=dpi)

        return plot

    def iic_global(self,
                   item,
                   anchor=False,
                   rater=None,
                   xmin=-10,
                   xmax=10,
                   central_diff=False,
                   thresh_lines=False,
                   point_info_lines=None,
                   point_info_labels=False,
                   cat_highlight=None,
                   title=None,
                   plot_style='colorblind',
                   black=False,
                   font='Times',
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
            if hasattr(self, 'anchor_thresholds_global') == False:
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

        if rater is None:
            severities = {'dummy_rater': 0}
            severities = pd.Series(severities)
            rater = 'dummy_rater'

        abilities = np.arange(-20, 20, 0.1)
        y = [self.variance_global(ability, item, difficulties, rater, severities, thresholds)
             for ability in abilities]
        y = np.array(y).reshape(len(abilities), 1)

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Fisher information'

        plot = self.plot_data_global(x_data=abilities, y_data=y, anchor=anchor, raters=rater, x_min=xmin, x_max=xmax,
                                     y_max=max(y) * 1.1, items=item, thresh_lines=thresh_lines,  plot_style=plot_style,
                                     point_info_lines_item=[item, point_info_lines], score_labels=point_info_labels,
                                     cat_highlight=cat_highlight, central_diff=central_diff, graph_title=graphtitle,
                                     y_label=ylabel, black=black, font=font, title_font_size=title_font_size,
                                     axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                                     plot_density=dpi, file_format=file_format)

        return plot

    def iic_items(self,
                   item,
                   anchor=False,
                   rater=None,
                   xmin=-10,
                   xmax=10,
                   central_diff=False,
                   thresh_lines=False,
                   point_info_lines=None,
                   point_info_labels=False,
                   cat_highlight=None,
                   title=None,
                   plot_style='colorblind',
                   black=False,
                   font='Times',
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
            if hasattr(self, 'anchor_thresholds_items') == False:
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

        if rater is None:
            severities = {'dummy_rater': {item: 0}}
            rater = 'dummy_rater'

        abilities = np.arange(-20, 20, 0.1)
        y = [self.variance_items(ability, item, difficulties, rater, severities, thresholds)
             for ability in abilities]
        y = np.array(y).reshape(len(abilities), 1)

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Fisher information'

        plot = self.plot_data_global(x_data=abilities, y_data=y, anchor=anchor, raters=rater, x_min=xmin, x_max=xmax,
                                     y_max=max(y) * 1.1, items=item, thresh_lines=thresh_lines,  plot_style=plot_style,
                                     point_info_lines_item=[item, point_info_lines], score_labels=point_info_labels,
                                     cat_highlight=cat_highlight, central_diff=central_diff, graph_title=graphtitle,
                                     y_label=ylabel, black=black, font=font, title_font_size=title_font_size,
                                     axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                                     plot_density=dpi, file_format=file_format)

        return plot

    def iic_thresholds(self,
                       item,
                       anchor=False,
                       rater=None,
                       xmin=-10,
                       xmax=10,
                       central_diff=False,
                       thresh_lines=False,
                       point_info_lines=None,
                       point_info_labels=False,
                       cat_highlight=None,
                       title=None,
                       plot_style='colorblind',
                       black=False,
                       font='Times',
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
            if hasattr(self, 'anchor_thresholds_thresholds') == False:
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

        if rater is None:
            severities = {'dummy_rater': np.zeros(self.max_score + 1)}
            rater = 'dummy_rater'

        abilities = np.arange(-20, 20, 0.1)
        y = [self.variance_thresholds(ability, item, difficulties, rater, severities, thresholds)
             for ability in abilities]
        y = np.array(y).reshape(len(abilities), 1)

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Fisher information'

        plot = self.plot_data_global(x_data=abilities, y_data=y, anchor=anchor, raters=rater, x_min=xmin, x_max=xmax,
                                     y_max=max(y) * 1.1, items=item, thresh_lines=thresh_lines,  plot_style=plot_style,
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
                   xmin=-10,
                   xmax=10,
                   central_diff=False,
                   thresh_lines=False,
                   point_info_lines=None,
                   point_info_labels=False,
                   cat_highlight=None,
                   title=None,
                   plot_style='colorblind',
                   black=False,
                   font='Times',
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
            if hasattr(self, 'anchor_thresholds_matrix') == False:
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

        if rater is None:
            severities = {'dummy_rater': {item: np.zeros(self.max_score + 1)
                                          for item in self.dataframe.columns}}
            rater = 'dummy_rater'

        abilities = np.arange(-20, 20, 0.1)
        y = [self.variance_matrix(ability, item, difficulties, rater, severities, thresholds)
             for ability in abilities]
        y = np.array(y).reshape(len(abilities), 1)

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Fisher information'

        plot = self.plot_data_global(x_data=abilities, y_data=y, anchor=anchor, raters=rater, x_min=xmin, x_max=xmax,
                                     y_max=max(y) * 1.1, items=item, thresh_lines=thresh_lines,  plot_style=plot_style,
                                     point_info_lines_item=[item, point_info_lines], score_labels=point_info_labels,
                                     cat_highlight=cat_highlight, central_diff=central_diff, graph_title=graphtitle,
                                     y_label=ylabel, black=black, font=font, title_font_size=title_font_size,
                                     axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                                     plot_density=dpi, file_format=file_format)

        return plot

    def tcc_global(self,
                   anchor=False,
                   items='all',
                   raters='zero',
                   obs=False,
                   xmin=-10,
                   xmax=10,
                   no_of_classes=5,
                   title=None,
                   score_lines=None,
                   score_labels=False,
                   plot_style='colorblind',
                   black=False,
                   font='Times',
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

        if items == 'all':
            items = self.dataframe.columns.tolist()

        if raters == 'all':
            raters = self.raters.tolist()

        if raters == 'none':
            raters = None

        if raters == 'zero':
            raters = None

        if anchor:
            if hasattr(self, 'anchor_thresholds_global') == False:
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

        if obs:
            if anchor:
                if hasattr(self, 'anchor_abils_global') == False:
                    self.person_abils_global(anchor=True)
                abilities = self.anchor_abils_global

            else:
                if hasattr(self, 'abils_global') == False:
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
                y = [sum(self.exp_score_global(ability, item, difficulties, 'dummy_rater',
                                               pd.Series({'dummy_rater': 0}), thresholds)
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
                y = [sum(self.exp_score_global(ability, item, difficulties, 'dummy_rater',
                                               pd.Series({'dummy_rater': 0}), thresholds)
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
                y = [self.exp_score_global(ability, items, difficulties, 'dummy_rater',
                                           pd.Series({'dummy_rater': 0}), thresholds)
                     for ability in abilities]

            elif isinstance(raters, list):
                y = [sum(self.exp_score_global(ability, items, difficulties, rater, severities, thresholds)
                         for rater in raters)
                     for ability in abilities]

            else:
                y = [self.exp_score_global(ability, items, difficulties, raters, severities, thresholds)
                     for ability in abilities]

        y = np.array(y).reshape(len(abilities), 1)

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

        y_max = self.max_score * no_of_items * no_of_raters

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Expected score'

        plot = self.plot_data_global(x_data=abilities, y_data=y, anchor=anchor, raters=raters, x_obs_data=xobsdata,
                                     y_obs_data=yobsdata, x_min=xmin, x_max=xmax, y_max=y_max, items=items,
                                     score_lines_test=score_lines, score_labels=score_labels, graph_title=graphtitle,
                                     y_label=ylabel, obs=obs, plot_style=plot_style, black=black, font=font,
                                     title_font_size=title_font_size, axis_font_size=axis_font_size, labelsize=labelsize,
                                     filename=filename, plot_density=dpi, file_format=file_format)

        return plot

    def tcc_items(self,
                  anchor=False,
                  rater=None,
                  obs=False,
                  xmin=-10,
                  xmax=10,
                  no_of_classes=5,
                  title=None,
                  score_lines=None,
                  score_labels=False,
                  plot_style='colorblind',
                  black=False,
                  font='Times',
                  title_font_size=15,
                  axis_font_size=12,
                  labelsize=12,
                  filename=None,
                  file_format='png',
                  dpi=300):

        '''
        Plots Test Characteristic Curve.
        '''

        if anchor:
            if hasattr(self, 'anchor_thresholds_items') == False:
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

        if rater is None:
            severities = {'dummy_rater': {item: 0 for item in self.dataframe.columns}}
            rater = 'dummy_rater'

        if obs:
            if anchor:
                if hasattr(self, 'anchor_abils_items') == False:
                    self.person_abils_items(anchor=True)
                abilities = self.anchor_abils_items

            else:
                if hasattr(self, 'abils_items') == False:
                    self.person_abils_items()
                abilities = self.abils_items

            xobsdata, yobsdata = self.class_intervals(abilities, self.dataframe.columns,
                                                               no_of_classes=no_of_classes)

            xobsdata -= np.mean([np.mean([severities[rater][item] for rater in self.raters])
                                 for item in self.dataframe.columns])
            if rater is not None:
                xobsdata += np.mean([severities[rater][item] for item in self.dataframe.columns])

            yobsdata = np.array(yobsdata).reshape((-1, 1))

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)
        y = [sum(self.exp_score_items(ability, item, difficulties, rater, severities, thresholds)
                 for item in self.dataframe.columns)
             for ability in abilities]
        y = np.array(y).reshape(len(abilities), 1)

        y_max = self.max_score * len(items) * len(raters)

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Expected score'

        plot = self.plot_data_global(x_data=abilities, y_data=y, anchor=anchor, raters=rater, x_obs_data=xobsdata,
                                     items=items, y_obs_data=yobsdata, x_min=xmin, x_max=xmax, y_max=y_max,
                                     score_lines_test=score_lines, score_labels=score_labels, graph_title=graphtitle,
                                     y_label=ylabel, obs=obs, plot_style=plot_style, black=black, font=font,
                                     title_font_size=title_font_size, axis_font_size=axis_font_size, labelsize=labelsize,
                                     filename=filename, plot_density=dpi, file_format=file_format)

        return plot

    def tcc_thresholds(self,
                       anchor=False,
                       rater=None,
                       obs=False,
                       xmin=-10,
                       xmax=10,
                       no_of_classes=5,
                       title=None,
                       score_lines=None,
                       score_labels=False,
                       plot_style='colorblind',
                       black=False,
                       font='Times',
                       title_font_size=15,
                       axis_font_size=12,
                       labelsize=12,
                       filename=None,
                       file_format='png',
                       dpi=300):

        '''
        Plots Test Characteristic Curve for RSM.
        '''

        if anchor:
            if hasattr(self, 'anchor_thresholds_thresholds') == False:
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

        if rater is None:
            severities = {'dummy_rater': np.zeros(self.max_score + 1)}
            rater = 'dummy_rater'

        if obs:
            if anchor:
                if hasattr(self, 'anchor_abils_thresholds') == False:
                    self.person_abils_thresholds(anchor=True)
                abilities = self.anchor_abils_thresholds

            else:
                if hasattr(self, 'abils_thresholds') == False:
                    self.person_abils_thresholds()
                abilities = self.abils_thresholds

            xobsdata, yobsdata = self.class_intervals(abilities, self.dataframe.columns,
                                                               no_of_classes=no_of_classes)

            xobsdata -= np.mean([severities[rater][1:].mean() for rater in self.raters])
            if rater is not None:
                xobsdata += severities[rater][1:].mean()

            yobsdata = np.array(yobsdata).reshape((-1, 1))

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)
        y = [sum(self.exp_score_thresholds(ability, item, difficulties, rater, severities, thresholds)
                 for item in difficulties.keys())
             for ability in abilities]
        y = np.array(y).reshape(len(abilities), 1)

        y_max = self.max_score * len(items) * len(raters)

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Expected score'

        plot = self.plot_data_global(x_data=abilities, y_data=y, anchor=anchor, raters=rater, x_obs_data=xobsdata,
                                     y_obs_data=yobsdata, x_min=xmin, x_max=xmax, y_max=y_max, items=items,
                                     score_lines_test=score_lines, score_labels=score_labels, graph_title=graphtitle,
                                     y_label=ylabel, obs=obs, plot_style=plot_style, black=black, font=font,
                                     title_font_size=title_font_size, axis_font_size=axis_font_size, labelsize=labelsize,
                                     filename=filename, plot_density=dpi, file_format=file_format)

        return plot

    def tcc_matrix(self,
                   anchor=False,
                   rater=None,
                   obs=False,
                   xmin=-10,
                   xmax=10,
                   no_of_classes=5,
                   title=None,
                   score_lines=None,
                   score_labels=False,
                   plot_style='colorblind',
                   black=False,
                   font='Times',
                   title_font_size=15,
                   axis_font_size=12,
                   labelsize=12,
                   filename=None,
                   file_format='png',
                   dpi=300):

        '''
        Plots Test Characteristic Curve.
        '''

        if anchor:
            if hasattr(self, 'anchor_thresholds_matrix') == False:
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

        if rater is None:
            severities = {'dummy_rater': {item: np.zeros(self.max_score + 1)
                                            for item in self.dataframe.columns}}
            rater = 'dummy_rater'

        if obs:
            if anchor:
                if hasattr(self, 'anchor_abils_matrix') == False:
                    self.person_abils_matrix(anchor=True)
                abilities = self.anchor_abils_matrix

            else:
                if hasattr(self, 'abils_matrix') == False:
                    self.person_abils_matrix()
                abilities = self.abils_matrix

            xobsdata, yobsdata = self.class_intervals(abilities, self.dataframe.columns,
                                                               no_of_classes=no_of_classes)

            xobsdata -= np.mean([np.mean([severities[rater][item][1:].mean() for rater in self.raters])
                                 for item in self.dataframe.columns])
            if rater is not None:
                xobsdata += np.mean([severities[rater][item][1:].mean() for item in self.dataframe.columns])

            yobsdata = np.array(yobsdata).reshape((-1, 1))

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)
        y = [sum(self.exp_score_matrix(ability, item, difficulties, rater, severities, thresholds)
                 for item in difficulties.keys())
             for ability in abilities]
        y = np.array(y).reshape(len(abilities), 1)

        y_max = self.max_score * len(items) * len(raters)

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Expected score'

        plot = self.plot_data_global(x_data=abilities, y_data=y, anchor=anchor, raters=rater, x_obs_data=xobsdata,
                                     y_obs_data=yobsdata, x_min=xmin, x_max=xmax, y_max=y_max, items=items,
                                     score_lines_test=score_lines, score_labels=score_labels, graph_title=graphtitle,
                                     y_label=ylabel, obs=obs, plot_style=plot_style, black=black, font=font,
                                     title_font_size=title_font_size, axis_font_size=axis_font_size, labelsize=labelsize,
                                     filename=filename, plot_density=dpi, file_format=file_format)

        return plot

    def test_info_global(self,
                         anchor=False,
                         items=None,
                         raters=None,
                         point_info_lines=None,
                         point_info_labels=False,
                         xmin=-10,
                         xmax=10,
                         ymax=None,
                         title=None,
                         plot_style='colorblind',
                         black=False,
                         font='Times',
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
            if hasattr(self, 'anchor_thresholds_global') == False:
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

        if raters == 'all':
            raters = self.raters

        if raters == 'none':
            raters = None

        if isinstance(raters, str):
            raters = [raters]

        if items == 'all':
            items = self.dataframe.columns

        if items == 'none':
            items = None

        if isinstance(items, str):
            items = [items]

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
                                     plot_style=plot_style, black=black, font=font, title_font_size=title_font_size,
                                     axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                                     plot_density=dpi, file_format=file_format)

        return plot

    def test_info_items(self,
                        anchor=False,
                        items=None,
                        raters=None,
                        point_info_lines=None,
                        point_info_labels=False,
                        xmin=-10,
                        xmax=10,
                        ymax=None,
                        title=None,
                        plot_style='colorblind',
                        black=False,
                        font='Times',
                        title_font_size=15,
                        axis_font_size=12,
                        labelsize=12,
                        filename=None,
                        file_format='png',
                        dpi=300):

        '''
        Plots Test Information Curve for RSM.
        '''

        if anchor:
            if hasattr(self, 'anchor_thresholds_items') == False:
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

        abilities = np.arange(-20, 20, 0.1)

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

        plot = self.plot_data_global(x_data=abilities, y_data=y, anchor=anchor, items=items, raters=raters, x_min=xmin,
                                     x_max=xmax, y_max=ymax, point_info_lines_test=point_info_lines,
                                     score_labels=point_info_labels, graph_title=graphtitle, y_label=ylabel,
                                     plot_style=plot_style, black=black, font=font, title_font_size=title_font_size,
                                     axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                                     plot_density=dpi, file_format=file_format)

        return plot

    def test_info_thresholds(self,
                             anchor=False,
                             items=None,
                             raters=None,
                             point_info_lines=None,
                             point_info_labels=False,
                             xmin=-10,
                             xmax=10,
                             ymax=None,
                             title=None,
                             plot_style='colorblind',
                             black=False,
                             font='Times',
                             title_font_size=15,
                             axis_font_size=12,
                             labelsize=12,
                             filename=None,
                             file_format='png',
                             dpi=300):

        '''
        Plots Test Information Curve for RSM.
        '''

        if anchor:
            if hasattr(self, 'anchor_thresholds_thresholds') == False:
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

        abilities = np.arange(-20, 20, 0.1)

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
                                         plot_style=plot_style, black=black, font=font, title_font_size=title_font_size,
                                         axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                                         plot_density=dpi, file_format=file_format)

        return plot

    def test_info_matrix(self,
                         anchor=False,
                         items=None,
                         raters=None,
                         point_info_lines=None,
                         point_info_labels=False,
                         xmin=-10,
                         xmax=10,
                         ymax=None,
                         title=None,
                         plot_style='colorblind',
                         black=False,
                         font='Times',
                         title_font_size=15,
                         axis_font_size=12,
                         labelsize=12,
                         filename=None,
                         file_format='png',
                         dpi=300):

        '''
        Plots Test Information Curve for RSM.
        '''

        if anchor:
            if hasattr(self, 'anchor_thresholds_matrix') == False:
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

        abilities = np.arange(-20, 20, 0.1)

        y = [sum(self.variance_matrix(ability, item, difficulties, rater, severities, thresholds)
                 for item in items for rater in raters)
             for ability in abilities]
        y = np.array(y)
        y = y.reshape(len(abilities), 1)

        if ymax is None:
            ymax = max(y) * 1.1

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        plot = self.plot_data_global(x_data=abilities, y_data=y, anchor=anchor, items=items, raters=raters, x_min=xmin,
                                     x_max=xmax, y_max=ymax, point_info_lines_test=point_info_lines,
                                     score_labels=point_info_labels, graph_title=graphtitle, y_label=ylabel,
                                     plot_style=plot_style, black=black, font=font, title_font_size=title_font_size,
                                     axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                                     plot_density=dpi, file_format=file_format)

        return plot

    def test_csem_global(self,
                         anchor=False,
                         items=None,
                         raters=None,
                         point_csem_lines=None,
                         point_csem_labels=False,
                         xmin=-10,
                         xmax=10,
                         ymax=5,
                         title=None,
                         plot_style='colorblind',
                         black=False,
                         font='Times',
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
            if hasattr(self, 'anchor_thresholds_global') == False:
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

        if raters == 'all':
            raters = self.raters

        if raters == 'none':
            raters = None

        if isinstance(raters, str):
            raters = [raters]

        if items == 'all':
            items = self.dataframe.columns

        if items == 'none':
            items = None

        if isinstance(items, str):
            items = [items]

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
        y = 1 / np.sqrt(y)
        y = y.reshape(len(abilities), 1)

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Conditional SEM'

        plot = self.plot_data_global(x_data=abilities, y_data=y, anchor=anchor, items=items, raters=raters,
                                     x_min=xmin, x_max=xmax, y_max=ymax, point_csem_lines=point_csem_lines,
                                     score_labels=point_csem_labels, graph_title=graphtitle, y_label=ylabel,
                                     plot_style=plot_style, black=black, font=font, title_font_size=title_font_size,
                                     axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                                     plot_density=dpi, file_format=file_format)

        return plot

    def test_csem_items(self,
                        anchor=False,
                        items=None,
                        raters=None,
                        point_csem_lines=None,
                        point_csem_labels=False,
                        xmin=-10,
                        xmax=10,
                        ymax=5,
                        title=None,
                        plot_style='colorblind',
                        black=False,
                        font='Times',
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
            if hasattr(self, 'anchor_thresholds_items') == False:
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

        if rater is None:
            severities = {'dummy_rater': {item: 0 for item in self.dataframe.columns}}
            rater = 'dummy_rater'

        abilities = np.arange(-20, 20, 0.1)
        y = np.array([sum(self.variance_items(ability, item, difficulties, rater, severities, thresholds)
                          for item in items for rater in raters)
                      for ability in abilities])
        y = 1 / np.sqrt(y)
        y = y.reshape(len(abilities), 1)

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Conditional SEM'

        plot = self.plot_data_items(x_data=abilities, y_data=y, anchor=anchor, items=items, raters=raters,
                                    x_min=xmin, x_max=xmax, y_max=ymax, point_csem_lines=point_csem_lines,
                                    score_labels=point_csem_labels, graph_title=graphtitle, y_label=ylabel,
                                    plot_style=plot_style, black=black, font=font, title_font_size=title_font_size,
                                    axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                                    plot_density=dpi, file_format=file_format)

        return plot

    def test_csem_thresholds(self,
                             anchor=False,
                             items=None,
                             raters=None,
                             point_csem_lines=None,
                             point_csem_labels=False,
                             xmin=-10,
                             xmax=10,
                             ymax=5,
                             title=None,
                             plot_style='colorblind',
                             black=False,
                             font='Times',
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
            if hasattr(self, 'anchor_thresholds_thresholds') == False:
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

        if rater is None:
            severities = {'dummy_rater': np.zeros(self.max_score + 1)}
            rater = 'dummy_rater'

        abilities = np.arange(-20, 20, 0.1)
        y = np.array([sum(self.variance_thresholds(ability, item, difficulties, rater, severities, thresholds)
                          for item in items for rater in raters)
                      for ability in abilities])
        y = 1 / np.sqrt(y)
        y = y.reshape(len(abilities), 1)

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Conditional SEM'

        plot = self.plot_data_thresholds(x_data=abilities, y_data=y, anchor=anchor, items=items, raters=raters,
                                         x_min=xmin, x_max=xmax, y_max=ymax, point_csem_lines=point_csem_lines,
                                         score_labels=point_csem_labels, graph_title=graphtitle, y_label=ylabel,
                                         plot_style=plot_style, black=black, font=font, title_font_size=title_font_size,
                                         axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                                         plot_density=dpi, file_format=file_format)

        return plot

    def test_csem_matrix(self,
                         anchor=False,
                         items=None,
                         raters=None,
                         point_csem_lines=None,
                         point_csem_labels=False,
                         xmin=-10,
                         xmax=10,
                         ymax=5,
                         title=None,
                         plot_style='colorblind',
                         black=False,
                         font='Times',
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
            if hasattr(self, 'anchor_thresholds_matrix') == False:
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

        if rater is None:
            severities = {'dummy_rater': {item: np.zeros(self.max_score + 1)
                                            for item in self.dataframe.columns}}
            rater = 'dummy_rater'

        abilities = np.arange(-20, 20, 0.1)
        y = np.array([sum(self.variance_matrix(ability, item, difficulties, rater, severities, thresholds)
                          for item in items for rater in raters)
                      for ability in abilities])
        y = 1 / np.sqrt(y)
        y = y.reshape(len(abilities), 1)

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Conditional SEM'

        plot = self.plot_data_matrix(x_data=abilities, y_data=y, anchor=anchor, items=items, raters=raters,
                                     x_min=xmin, x_max=xmax, y_max=ymax, point_csem_lines=point_csem_lines,
                                     score_labels=point_csem_labels, graph_title=graphtitle, y_label=ylabel,
                                     plot_style=plot_style, black=black, font=font, title_font_size=title_font_size,
                                     axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                                     plot_density=dpi, file_format=file_format)

        return plot

    def std_residuals_plot_global(self,
                                  items=None,
                                  raters=None,
                                  bin_width=0.5,
                                  x_min=-6,
                                  x_max=6,
                                  normal=False,
                                  title=None,
                                  plot_style='colorblind',
                                  font='Times',
                                  title_font_size=15,
                                  axis_font_size=12,
                                  labelsize=12,
                                  filename=None,
                                  file_format='png',
                                  plot_density=300):

        '''
        Plots histogram of standardised residuals for SLM, with optional overplotting of Standard Normal Distribution.
        '''

        if items is None:
            if raters is None:
                std_residual_list = self.std_residual_df_global.unstack().unstack()

            else:
                std_residual_list = self.std_residual_df_global.loc[raters].unstack().unstack()

        else:
            if raters is None:
                std_residual_list = self.std_residual_df_global[items].unstack().unstack()

            else:
                std_residual_list = self.std_residual_df_global[items].loc[raters].unstack().unstack()

        plot = self.std_residuals_hist(std_residual_list, bin_width=bin_width, x_min=x_min, x_max=x_max, normal=normal,
                                       title=title, plot_style=plot_style, font=font, title_font_size=title_font_size,
                                       axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                                       file_format=file_format, plot_density=plot_density)

        return plot

    def std_residuals_plot_items(self,
                                 rater=None,
                                 bin_width=0.5,
                                 x_min=-6,
                                 x_max=6,
                                 normal=False,
                                 title=None,
                                 plot_style='colorblind',
                                 font='Times',
                                 title_font_size=15,
                                 axis_font_size=12,
                                 labelsize=12,
                                 filename=None,
                                 file_format='png',
                                 plot_density=300):

        '''
        Plots histogram of standardised residuals for SLM, with optional overplotting of Standard Normal Distribution.
        '''

        if rater is None:
            std_residual_list = self.std_residual_df_items.unstack().unstack()

        else:
            std_residual_list = self.std_residual_df_items.xs(rater).unstack()

        plot = self.std_residuals_hist(std_residual_list, bin_width=bin_width, x_min=x_min, x_max=x_max, normal=normal,
                                       title=title, plot_style=plot_style, font=font, title_font_size=title_font_size,
                                       axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                                       file_format=file_format, plot_density=plot_density)

        return plot

    def std_residuals_plot_thresholds(self,
                                      rater=None,
                                      bin_width=0.5,
                                      x_min=-6,
                                      x_max=6,
                                      normal=False,
                                      title=None,
                                      plot_style='colorblind',
                                      font='Times',
                                      title_font_size=15,
                                      axis_font_size=12,
                                      labelsize=12,
                                      filename=None,
                                      file_format='png',
                                      plot_density=300):

        if rater is None:
            std_residual_list = self.std_residual_df_thresholds.unstack().unstack()

        else:
            std_residual_list = self.std_residual_df_thresholds.xs(rater).unstack()

        plot = self.std_residuals_hist(std_residual_list, bin_width=bin_width, x_min=x_min, x_max=x_max, normal=normal,
                                       title=title, plot_style=plot_style, font=font, title_font_size=title_font_size,
                                       axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                                       file_format=file_format, plot_density=plot_density)

        return plot

    def std_residuals_plot_matrix(self,
                                  rater=None,
                                  bin_width=0.5,
                                  x_min=-6,
                                  x_max=6,
                                  normal=False,
                                  title=None,
                                  plot_style='colorblind',
                                  font='Times',
                                  title_font_size=15,
                                  axis_font_size=12,
                                  labelsize=12,
                                  filename=None,
                                  file_format='png',
                                  plot_density=300):

        if rater is None:
            std_residual_list = self.std_residual_df_matrix.unstack().unstack()

        else:
            std_residual_list = self.std_residual_df_matrix.xs(rater).unstack()

        plot = self.std_residuals_hist(std_residual_list, bin_width=bin_width, x_min=x_min, x_max=x_max, normal=normal,
                                       title=title, plot_style=plot_style, font=font, title_font_size=title_font_size,
                                       axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                                       file_format=file_format, plot_density=plot_density)

        return plot

'''
*** SIMULATIONS ***

Module to generate simulated data according to the SLM, PCM, RSM or MFRM
variants of the Rasch model family.
'''

class Rasch_Sim:

    def __init__():

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

        if isinstance(new, str) == False:
            print('Item names must be strings')

        else:
            self.scores.rename(columns={old: new}, inplace=True)

    def rename_items_all(self,
                         new_names):

        list_length = len(new_names)

        if len(new_names) != len(set(new_names)):
            print('List of new item names contains duplicates. Please ensure all items have unique names.')

        elif list_length != self.no_of_items:
            print(f'Incorrect number of item names. {list_length} in list, {self.no_of_items} items in data.')

        if all(isinstance(name, str) for name in new_names) == False:
            print('Item names must be strings')

        else:
            self.scores.rename(columns={old: new for old, new in zip(self.scores.columns, new_names)}, inplace=True)

    def rename_person(self,
                      old,
                      new):

        if old == new:
            print('New person name is the same as old person name.')

        elif new in self.scores.index:
            print('New person name is a duplicate of an existing person name.')

        if old not in self.scores.index:
            print(f'Old person name "{old}" not found in data. Please check.')

        if isinstance(new, str) == False:
            print('Item names must be strings')

        else:
            self.scores.rename(index={old: new}, inplace=True)

    def rename_persons_all(self,
                           new_names):

        list_length = len(new_names)

        if len(new_names) != len(set(new_names)):
            print('List of new person names contains duplicates. Please ensure all persons have unique names.')

        elif list_length != self.no_of_persons:
            print(f'Incorrect number of person names. {list_length} in list, {self.no_of_persons} persons in data.')

        if all(isinstance(name, str) for name in new_names) == False:
            print('Person names must be strings')

        else:
            self.scores.rename(index={old: new for old, new in zip(self.scores.index, new_names)}, inplace=True)

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
                 manual_diffs=None):
        
        self.no_of_items = int(no_of_items)
        self.no_of_persons = int(no_of_persons)
        self.item_range = item_range
        self.person_sd = person_sd
        self.offset = offset
        self.missing = missing
        self.abilities = manual_abilities
        self.diffs = manual_diffs
        self.dataframe = pd.DataFrame([1])
        self.slm = SLM(self.dataframe)
        
        '''
        Generates person and item parameters.
        '''
        
        if self.abilities is None:
            self.abilities = np.random.normal(0, self.person_sd, self.no_of_persons)
            self.abilities -= np.mean(self.abilities)
            self.abilities += self.offset
            self.abilities = {f'Person_{person + 1}': ability for person, ability in enumerate(self.abilities)}
            self.abilities = pd.Series(self.abilities)
            
        else:
            assert len(self.abilities) == self.no_of_persons, 'Length of manual abilities must match number of persons.'
            self.abilities = np.array(self.abilities)
            self.abilities = {f'Person_{person + 1}': ability for person, ability in enumerate(self.abilities)}
            self.abilities = pd.Series(self.abilities)
        
        if self.diffs is None:
            self.diffs = np.random.uniform(0, 1, self.no_of_items)
            self.diffs *= (self.item_range / (np.max(self.diffs) - np.min(self.diffs)))
            self.diffs -= np.mean(self.diffs)
            self.diffs = {f'Item_{item + 1}': diff for item, diff in enumerate(self.diffs)}
            self.diffs = pd.Series(self.diffs)

        else:
            assert len(self.diffs) == self.no_of_items, 'Length of manual difficulties must match number of items.'
            self.diffs = np.array(self.diffs)
            self.diffs = {f'Item_{item + 1}': diff for item, diff in enumerate(self.diffs)}
            self.diffs = pd.Series(self.diffs)
        
        '''
        Calculates probability of a correct response for each person on each item
        '''
        
        self.probs = [[self.slm.exp_score(ability, difficulty)
                       for difficulty in self.diffs]
                      for ability in self.abilities]
        self.probs = np.array(self.probs)
        
        '''
        Calculates scores and removes required amount of missing data
        '''

        scoring_randoms = self.randoms()
        self.scores = (scoring_randoms <= self.probs).astype(np.float64)
        
        missing_randoms = self.randoms()
        self.scores[missing_randoms < self.missing] = np.nan
        
        self.scores = pd.DataFrame(self.scores)

        for person in range(self.no_of_persons):
            self.scores.rename(index={person: f'Person_{person + 1}'},
                               inplace=True)
        for item in range(self.no_of_items):
            self.scores.rename(columns={item: f'Item_{item + 1}'},
                               inplace=True)

class PCM_Sim(Rasch_Sim):
    
    '''
    Generates simulated data accoding to the Partial Credit Model (PCM).
    '''

    def __init__(self,
                 no_of_items,
                 no_of_persons,
                 max_score_vector,
                 item_range=3,
                 category_mean=1,
                 person_sd=1.5,
                 max_disorder=0,
                 offset=0,
                 missing=0,
                 manual_abilities=None,
                 manual_diffs=None,
                 manual_thresholds=None):

        self.no_of_items = int(no_of_items)
        self.no_of_persons = int(no_of_persons)
        self.item_range = item_range
        self.max_score_vector = max_score_vector
        self.category_mean = category_mean
        self.person_sd = person_sd
        self.max_disorder = max_disorder
        self.offset = offset
        self.missing = missing
        self.abilities = manual_abilities
        self.diffs = manual_diffs
        self.dataframe = pd.DataFrame([1])
        self.pcm = PCM(self.dataframe, self.max_score_vector)
        
        '''
        Generates person, item and threshold parameters.
        '''

        assert len(self.max_score_vector) == self.no_of_items, 'Length of max score vector must match number of items.'
        
        if self.abilities is None:
            self.abilities = np.random.normal(0, self.person_sd, self.no_of_persons)
            self.abilities -= np.mean(self.abilities)
            self.abilities += self.offset
            self.abilities = {f'Person_{person + 1}': ability for person, ability in enumerate(self.abilities)}
            self.abilities = pd.Series(self.abilities)
            
        else:
            assert len(self.abilities) == self.no_of_persons, 'Length of manual abilities must match number of persons.'
            self.abilities = np.array(self.abilities)
            self.abilities = {f'Person_{person + 1}': ability for person, ability in enumerate(self.abilities)}
            self.abilities = pd.Series(self.abilities)
        
        if self.diffs is None:
            self.diffs = np.random.uniform(0, 1, self.no_of_items)
            self.diffs *= (self.item_range / (np.max(self.diffs) - np.min(self.diffs)))
            self.diffs -= np.mean(self.diffs)
            self.diffs = {f'Item_{item + 1}': diff for item, diff in enumerate(self.diffs)}
            self.diffs = pd.Series(self.diffs)

        else:
            assert len(self.diffs) == self.no_of_items, 'Length of manual difficulties must match number of items.'
            self.diffs = np.array(self.diffs)
            self.diffs = {f'Item_{item + 1}': diff for item, diff in enumerate(self.diffs)}
            self.diffs = pd.Series(self.diffs)

        if manual_thresholds is None:
            
            category_widths = {f'Item_{item + 1}':
                               np.random.uniform(self.max_disorder,
                                                 2 * self.category_mean - self.max_disorder,
                                                 value - 1)
                               for item, value in enumerate(self.max_score_vector)}
            
            self.thresholds_centred = {f'Item_{item + 1}':
                                       np.array([np.sum(category_widths[f'Item_{item + 1}'][:category])
                                                 for category in range(value)])
                                       for item, value in enumerate(self.max_score_vector)}
    
            for item in range(self.no_of_items):
    
                self.thresholds_centred[f'Item_{item + 1}'] -= np.mean(self.thresholds_centred[f'Item_{item + 1}'])
                self.thresholds_centred[f'Item_{item + 1}'] = np.insert(self.thresholds_centred[f'Item_{item + 1}'], 0, 0)

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

            self.thresholds_centred = {f'Item_{item + 1}': np.array(thresholds)
                                       for item, thresholds in enumerate(manual_thresholds)}

        self.thresholds_uncentred = {f'Item_{item + 1}':
                                     self.thresholds_centred[f'Item_{item + 1}'][1:] +
                                     self.diffs[item]
                                     for item in range(self.no_of_items)}

        threshold_list = itertools.chain.from_iterable(self.thresholds_uncentred.values())
        threshold_mean = statistics.mean(threshold_list)
        
        for item in range(self.no_of_items):
            self.diffs[item] -= threshold_mean
            self.thresholds_uncentred[f'Item_{item + 1}'] -= threshold_mean
            
        '''
        Calculates probability of a response in each category for each person on each item
        '''

        self.cat_probs = {f'Item_{item + 1}':
                          np.array([[self.pcm.cat_prob_uncentred(abil,
                                                                 cat,
                                                                 self.thresholds_uncentred[f'Item_{item + 1}'])
                                     for cat in range(self.max_score_vector[item] + 1)]
                                    for abil in self.abilities])
                          for item in range(self.no_of_items)}
        
        '''
        Calculates scores and removes required amount of missing data
        '''

        self.scores = np.zeros((self.no_of_persons, self.no_of_items))

        scoring_randoms = self.randoms()
        
        cat_scores = {}

        for item in range(self.no_of_items):

            cat_scores[f'Item_{item + 1}'] = np.zeros((self.no_of_persons,
                                                       self.max_score_vector[item]))

            for cat in range(self.max_score_vector[item]):

                cat_scores[f'Item_{item + 1}'][:, cat] = (scoring_randoms[:, item] >
                                                          np.sum(self.cat_probs[f'Item_{item + 1}'][:, :cat + 1],
                                                                 axis = 1)).astype(np.float64)

            self.scores[:, item] = np.sum(cat_scores[f'Item_{item + 1}'], axis = 1)

        missing_randoms = self.randoms()

        self.scores[(missing_randoms < self.missing)] = np.nan

        self.scores = pd.DataFrame(self.scores)

        for person in range(self.no_of_persons):
            self.scores.rename(index={person: f'Person_{person + 1}'},
                               inplace=True)
        for item in range(self.no_of_items):
            self.scores.rename(columns={item: f'Item_{item + 1}'},
                               inplace=True)
        
class RSM_Sim(Rasch_Sim):
    
    '''
    Generates simulated data accoding to the Rating Scale Model (RSM).
    '''

    def __init__(self,
                 no_of_items,
                 no_of_persons,
                 max_score,
                 item_range=3,
                 category_mean=1,
                 person_sd=1.5,
                 max_disorder=0,
                 offset=0,
                 missing=0 ,
                 manual_abilities=None,
                 manual_diffs=None,
                 manual_thresholds=None):

        self.no_of_items = int(no_of_items)
        self.no_of_persons = int(no_of_persons)
        self.item_range = item_range
        self.max_score = max_score
        self.category_mean = category_mean
        self.person_sd = person_sd
        self.max_disorder = max_disorder
        self.offset = offset
        self.missing = missing
        self.abilities = manual_abilities
        self.diffs = manual_diffs
        self.thresholds = manual_thresholds
        self.dataframe = pd.DataFrame([self.max_score])
        self.rsm = RSM(self.dataframe, self.max_score)

        '''
        Generates person, item and threshold parameters.
        '''
        
        if self.abilities is None:
            self.abilities = np.random.normal(0, self.person_sd, self.no_of_persons)
            self.abilities -= np.mean(self.abilities)
            self.abilities += self.offset
            self.abilities = {f'Person_{person + 1}': ability for person, ability in enumerate(self.abilities)}
            self.abilities = pd.Series(self.abilities)
            
        else:
            assert len(self.abilities) == self.no_of_persons, 'Length of manual abilities must match number of persons.'
            self.abilities = np.array(self.abilities)
            self.abilities = {f'Person_{person + 1}': ability for person, ability in enumerate(self.abilities)}
            self.abilities = pd.Series(self.abilities)
            
        
        if self.diffs is None:
            self.diffs = np.random.uniform(0, 1, self.no_of_items)
            self.diffs *= (self.item_range / (np.max(self.diffs) - np.min(self.diffs)))
            self.diffs -= np.mean(self.diffs)
            self.diffs = {f'Item_{item + 1}': diff for item, diff in enumerate(self.diffs)}
            self.diffs = pd.Series(self.diffs)

        else:
            assert len(self.diffs) == self.no_of_items, 'Length of manual difficulties must match number of items.'
            self.diffs = np.array(self.diffs)
            self.diffs = {f'Item_{item + 1}': diff for item, diff in enumerate(self.diffs)}
            self.diffs = pd.Series(self.diffs)

        if self.thresholds is None:
            category_widths = np.random.uniform(self.max_disorder,
                                                2 * self.category_mean - self.max_disorder,
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

        self.cat_probs = [[[self.rsm.cat_prob(ability,
                                              difficulty,
                                              category,
                                              self.thresholds)
                            for category in range(self.max_score + 1)]
                           for ability in self.abilities]
                          for difficulty in self.diffs]

        self.cat_probs = np.array(self.cat_probs)

        '''
        Calculated scores and removes required amount of missing data
        '''
        
        scoring_randoms = self.randoms().T
        
        cat_scores = np.zeros((self.no_of_items,
                               self.no_of_persons,
                               self.max_score + 1))
        
        for cat in range(self.max_score):
            
            cat_scores[:, :, cat] = (scoring_randoms >
                                     np.sum(self.cat_probs[:, :, :cat + 1],
                                            axis = 2)).astype(np.float64)
        
        self.scores = np.sum(cat_scores, axis = 2)
        
        missing_randoms = self.randoms().T
        
        self.scores[missing_randoms < self.missing] = np.nan
        
        self.scores = pd.DataFrame(self.scores.T)

        for person in range(self.no_of_persons):
            self.scores.rename(index={person: f'Person_{person + 1}'},
                               inplace=True)
        for item in range(self.no_of_items):
            self.scores.rename(columns={item: f'Item_{item + 1}'},
                               inplace=True)

class MFRM_Sim(Rasch_Sim):

    def __init__():

        pass

    def rename_rater(self,
                     old,
                     new):

        if old == new:
            print('New item name is the same as old item name.')

        elif new in self.raters:
            print('New item name is a duplicate of an existing item name')

        if old not in self.raters:
            print(f'Old item name "{old}" not found in data. Please check')

        if isinstance(new, str) == False:
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

        if all(isinstance(name, str) for name in new_names) == False:
            print('Rater names must be strings')

        else:
            df_dict = {new: self.scores.xs(old) for old, new in zip(self.raters, new_names)}
            self.scores = pd.concat(df_dict.values(), keys = df_dict.keys())
            self.raters = self.scores.index.get_level_values(0).unique()

    def rename_person(self,
                      old,
                      new):

        if old == new:
            print('New person name is the same as old person name.')

        elif new in self.scores.index.get_level_values(1):
            print('New person name is a duplicate of an existing person name')

        if old not in self.scores.index.get_level_values(1):
            print(f'Old person name "{old}" not found in data. Please check')

        if isinstance(new, str) == False:
            print('Person names must be strings')

        else:
            self.scores.rename(index={old: new},
                               inplace=True)

    def rename_persons_all(self,
                           new_names):

        list_length = len(new_names)
        old_names = self.scores.index.get_level_values(1)

        if len(new_names) != len(set(new_names)):
            print('List of new rater names contains duplicates. Please ensure all raters have unique names')

        elif list_length != self.no_of_persons:
            print(f'Incorrect number of rater names. {list_length} in list, {self.no_of_persons} raters in data.')

        if all(isinstance(name, str) for name in new_names) == False:
            print('Person names must be strings')

        else:
            self.scores.rename(index={old: new for old, new in zip(old_names, new_names)},
                               inplace=True)

class MFRM_Sim_Global(MFRM_Sim):
    
    '''
    Generates simulated data accoding to the RSM formulation
    of the basic form of the Many-Facet Rasch Model (MFRM).
    '''

    def __init__(self,
                 no_of_items,
                 no_of_persons,
                 no_of_raters,
                 max_score,
                 item_range=2,
                 rater_range=2,
                 category_mean=1,
                 person_sd=1.5,
                 max_disorder=0,
                 offset=0,
                 missing=0,
                 manual_abilities=None,
                 manual_diffs=None,
                 manual_thresholds=None,
                 manual_severities=None):

        self.no_of_items = int(no_of_items)
        self.no_of_persons = int(no_of_persons)
        self.no_of_raters = int(no_of_raters)
        self.item_range = item_range
        self.rater_range = rater_range
        self.max_score = max_score
        self.category_mean = category_mean
        self.person_sd = person_sd
        self.max_disorder = max_disorder
        self.offset = offset
        self.missing = missing
        self.abilities = manual_abilities
        self.diffs = manual_diffs
        self.thresholds = manual_thresholds
        self.severities = manual_severities
        self.dataframe = self.produce_df([[0], [1]], [[0]])
        self.dataframe.loc[0].iloc[0, 0] = self.max_score
        self.mfrm = MFRM(self.dataframe, self.max_score)

        '''
        Generates person, item and threshold parameters.
        '''

        if self.abilities is None:
            self.abilities = np.random.normal(0, self.person_sd, self.no_of_persons)
            self.abilities -= np.mean(self.abilities)
            self.abilities += self.offset
            self.abilities = {f'Person_{person + 1}': ability for person, ability in enumerate(self.abilities)}
            self.abilities = pd.Series(self.abilities)

        else:
            assert len(self.abilities) == self.no_of_persons, 'Length of manual abilities must match number of persons.'
            self.abilities = np.array(self.abilities)
            self.abilities = {f'Person_{person + 1}': ability for person, ability in enumerate(self.abilities)}
            self.abilities = pd.Series(self.abilities)

        if self.diffs is None:
            self.diffs = np.random.uniform(0, 1, self.no_of_items)
            self.diffs *= (self.item_range / (np.max(self.diffs) - np.min(self.diffs)))
            self.diffs -= np.mean(self.diffs)
            self.diffs = {f'Item_{item + 1}': diff for item, diff in enumerate(self.diffs)}
            self.diffs = pd.Series(self.diffs)

        else:
            assert len(self.diffs) == self.no_of_items, 'Length of manual difficulties must match number of items.'
            self.diffs = np.array(self.diffs)
            self.diffs = {f'Item_{item + 1}': diff for item, diff in enumerate(self.diffs)}
            self.diffs = pd.Series(self.diffs)

        if self.thresholds is None:
            category_widths = np.random.uniform(self.max_disorder,
                                                2 * self.category_mean - self.max_disorder,
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
        
        if self.severities is None:
            self.severities = truncnorm.rvs(-1.96, 1.96, size = self.no_of_raters)
            self.severities *= (self.rater_range /
                                (np.max(self.severities) - np.min(self.severities)))
            self.severities -= np.mean(self.severities)
            self.severities = {f'Rater_{rater + 1}': severity for rater, severity in enumerate(self.severities)}
            self.severities = pd.Series(self.severities)
            
        else:
            assert len(self.severities) == self.no_of_raters, 'Length of manual severities must match number of raters.'
            self.severities = np.array(self.severities)
            self.severities = {f'Rater_{rater + 1}': severity for rater, severity in enumerate(self.severities)}
            self.severities = pd.Series(self.severities)

        self.items = [f'Item_{item + 1}' for item in range(self.no_of_items)]
        self.raters = [f'Rater_{rater + 1}' for rater in range(self.no_of_raters)]
        self.persons = [f'Person_{person + 1}' for person in range(self.no_of_persons)]
        
        '''
        Calculates probability of a response in each category
        for each person on each item
        '''

        self.cat_probs = [[[[self.mfrm.cat_prob_global(ability, item, self.diffs, rater, self.severities, category,
                                                       self.thresholds)
                             for category in range(self.max_score + 1)]
                            for rater in self.raters]
                           for item in self.items]
                          for ability in self.abilities]

        self.cat_probs = np.array(self.cat_probs)
        
        '''
        Calculated scores and removes required amount of missing data
        '''
        
        scoring_randoms = np.array([self.randoms()
                                    for rater in range(self.no_of_raters)])
        scoring_randoms = np.swapaxes(scoring_randoms, 0, 2)
        scoring_randoms = np.swapaxes(scoring_randoms, 0, 1)
            
        cat_scores = np.zeros((self.no_of_persons,
                               self.no_of_items,
                               self.no_of_raters,
                               self.max_score + 1))
        
        for rater in range(self.no_of_raters):
            for cat in range(self.max_score):
            
                cat_scores[:, :, rater, cat] = (scoring_randoms[:, :, rater] >
                                                np.sum(self.cat_probs[:, :, rater, :cat + 1],
                                                       axis = 2)).astype(np.float64)
        
        self.scores = np.sum(cat_scores, axis = 3)
        
        missing_randoms = self.randoms()
        
        self.scores[missing_randoms < self.missing] = np.nan
        
        cols = [f'Item_{item + 1}' for item in range(self.no_of_items)]
        
        final = {f'Rater_{rater + 1}': pd.DataFrame(self.scores[:, :, rater],
                                                    columns = cols)
                 for rater in range(self.no_of_raters)}
        
        self.scores = pd.concat(final.values(),
                          keys = final.keys())

        for person in range(self.no_of_persons):
            self.scores.rename(index={person: f'Person_{person + 1}'},
                               inplace=True)

        self.items = self.scores.columns
        self.raters = self.scores.index.get_level_values(0).unique()
        self.persons = self.scores.index.get_level_values(1).unique()

class MFRM_Sim_Items(MFRM_Sim):
    
    '''
    Generates simulated data accoding to the RSM formulation
    of the extended vector-by-item form of the Many-Facet Rasch Model (MFRM).
    '''

    def __init__(self,
                 no_of_items,
                 no_of_persons,
                 no_of_raters,
                 max_score,
                 item_range=2,
                 rater_range=2,
                 category_mean=1,
                 person_sd=1.5,
                 max_disorder = 0,
                 offset = 0,
                 missing = 0,
                 manual_abilities=None,
                 manual_diffs=None,
                 manual_thresholds=None,
                 manual_severities=None):

        self.no_of_items = int(no_of_items)
        self.no_of_persons = int(no_of_persons)
        self.no_of_raters = int(no_of_raters)
        self.item_range = item_range
        self.rater_range = rater_range
        self.max_score = max_score
        self.category_mean = category_mean
        self.person_sd = person_sd
        self.max_disorder = max_disorder
        self.offset = offset
        self.missing = missing
        self.abilities = manual_abilities
        self.diffs = manual_diffs
        self.thresholds = manual_thresholds
        self.severities = manual_severities
        self.dataframe = self.produce_df([[0], [1]], [[0]])
        self.dataframe.loc[0].iloc[0, 0] = self.max_score
        self.mfrm = MFRM(self.dataframe, self.max_score)

        '''
        Generates person, item and threshold parameters.
        '''

        if self.abilities is None:
            self.abilities = np.random.normal(0, self.person_sd, self.no_of_persons)
            self.abilities -= np.mean(self.abilities)
            self.abilities += self.offset
            self.abilities = {f'Person_{person + 1}': ability for person, ability in enumerate(self.abilities)}
            self.abilities = pd.Series(self.abilities)

        else:
            assert len(self.abilities) == self.no_of_persons, 'Length of manual abilities must match number of persons.'
            self.abilities = np.array(self.abilities)
            self.abilities = {f'Person_{person + 1}': ability for person, ability in enumerate(self.abilities)}
            self.abilities = pd.Series(self.abilities)

        if self.diffs is None:
            self.diffs = np.random.uniform(0, 1, self.no_of_items)
            self.diffs *= (self.item_range / (np.max(self.diffs) - np.min(self.diffs)))
            self.diffs -= np.mean(self.diffs)
            self.diffs = {f'Item_{item + 1}': diff for item, diff in enumerate(self.diffs)}
            self.diffs = pd.Series(self.diffs)

        else:
            assert len(self.diffs) == self.no_of_items, 'Length of manual difficulties must match number of items.'
            self.diffs = np.array(self.diffs)
            self.diffs = {f'Item_{item + 1}': diff for item, diff in enumerate(self.diffs)}
            self.diffs = pd.Series(self.diffs)

        if self.thresholds is None:
            category_widths = np.random.uniform(self.max_disorder,
                                                2 * self.category_mean - self.max_disorder,
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
        
        if self.severities is None:
            severities = [truncnorm.rvs(-1.96, 1.96, size=self.no_of_items)
                               for rater in range(self.no_of_raters)]
            severities = np.array(severities)
            severities *= (self.rater_range / (np.max(severities) - np.min(severities)))
            severities -= np.mean(severities)

            self.severities = {f'Rater_{rater + 1}': {f'Item_{item + 1}': severities[rater, item]
                                                      for item in range(self.no_of_items)}
                               for rater in range(self.no_of_raters)}
            
        else:
            assert len(manual_severities) == self.no_of_raters, 'Length of manual severities must match number of raters.'
            for rater in range(self.no_of_raters):
                assert len(manual_severities[rater]) == self.no_of_items, ('Length of all sets manual severities ' +
                                                                           'must match number of items.')
            self.severities = np.array(self.severities)

            self.severities = {f'Rater_{rater + 1}': {f'Item_{item + 1}': severities[rater, item]
                                                      for item in range(self.no_of_items)}
                               for rater in range(self.no_of_raters)}

        self.items = [f'Item_{item + 1}' for item in range(self.no_of_items)]
        self.raters = [f'Rater_{rater + 1}' for rater in range(self.no_of_raters)]
        self.persons = [f'Person_{person + 1}' for person in range(self.no_of_persons)]
        
        '''
        Calculates probability of a response in each category
        for each person on each item
        '''

        items =[f'Item_{item + 1}' for item in range(self.no_of_items)]

        cat_probs = [[[[self.mfrm.cat_prob_items(ability,
                                                 item,
                                                 self.diffs,
                                                 rater,
                                                 self.severities,
                                                 category,
                                                 self.thresholds)
                        for category in range(self.max_score + 1)]
                       for rater in self.raters]
                      for item in items]
                     for ability in self.abilities]

        self.cat_probs = np.array(cat_probs)
        
        '''
        Calculates scores and removes required amount of missing data
        '''
        
        scoring_randoms = np.array([self.randoms()
                                    for rater in range(self.no_of_raters)])
        scoring_randoms = np.swapaxes(scoring_randoms, 0, 2)
        scoring_randoms = np.swapaxes(scoring_randoms, 0, 1)
            
        cat_scores = np.zeros((self.no_of_persons,
                               self.no_of_items,
                               self.no_of_raters,
                               self.max_score + 1))
        
        for rater in range(self.no_of_raters):
            for cat in range(self.max_score):
            
                cat_scores[:, :, rater, cat] = (scoring_randoms[:, :, rater] >
                                                np.sum(self.cat_probs[:, :, rater, :cat + 1],
                                                       axis = 2)).astype(np.float64)

        self.scores = np.sum(cat_scores, axis = 3)
        
        missing_randoms = self.randoms()
        
        self.scores[missing_randoms < self.missing] = np.nan
        
        cols = [f'Item_{item + 1}' for item in range(self.no_of_items)]
        
        final = {rater: pd.DataFrame(self.scores[:, :, i],
                                     columns=cols, index=self.persons)
                 for i, rater in enumerate(self.raters)}
        
        self.scores = pd.concat(final.values(), keys = final.keys())
    
class MFRM_Sim_Thresholds(MFRM_Sim):
    
    '''
    Generates simulated data accoding to the RSM formulation of the extended
    vector-by-threshold form of the Many-Facet Rasch Model (MFRM).
    '''

    def __init__(self,
                 no_of_items,
                 no_of_persons,
                 no_of_raters,
                 max_score,
                 item_range=2,
                 rater_range=2,
                 category_mean=1,
                 person_sd=1.5,
                 max_disorder=0,
                 offset=0,
                 missing=0,
                 manual_abilities=None,
                 manual_diffs=None,
                 manual_thresholds=None,
                 manual_severities=None):

        self.no_of_items = int(no_of_items)
        self.no_of_persons = int(no_of_persons)
        self.no_of_raters = int(no_of_raters)
        self.item_range = item_range
        self.rater_range = rater_range
        self.max_score = max_score
        self.category_mean = category_mean
        self.person_sd = person_sd
        self.max_disorder = max_disorder
        self.offset = offset
        self.missing = missing
        self.abilities = manual_abilities
        self.diffs = manual_diffs
        self.thresholds = manual_thresholds
        self.severities = manual_severities
        self.dataframe = self.produce_df([[0], [1]], [[0]])
        self.dataframe.loc[0].iloc[0, 0] = 0
        self.mfrm = MFRM(self.dataframe, self.max_score)

        '''
        Generates person, item and threshold parameters.
        '''

        if self.abilities is None:
            self.abilities = np.random.normal(0, self.person_sd, self.no_of_persons)
            self.abilities -= np.mean(self.abilities)
            self.abilities += self.offset
            self.abilities = {f'Person_{person + 1}': ability for person, ability in enumerate(self.abilities)}
            self.abilities = pd.Series(self.abilities)

        else:
            assert len(self.abilities) == self.no_of_persons, 'Length of manual abilities must match number of persons.'
            self.abilities = np.array(self.abilities)
            self.abilities = {f'Person_{person + 1}': ability for person, ability in enumerate(self.abilities)}
            self.abilities = pd.Series(self.abilities)

        if self.diffs is None:
            self.diffs = np.random.uniform(0, 1, self.no_of_items)
            self.diffs *= (self.item_range /
                           (np.max(self.diffs) - np.min(self.diffs)))
            self.diffs -= np.mean(self.diffs)
            self.diffs = {f'Item_{item + 1}': diff for item, diff in enumerate(self.diffs)}
            self.diffs = pd.Series(self.diffs)

        else:
            assert len(self.diffs) == self.no_of_items, 'Length of manual difficulties must match number of items.'
            self.diffs = np.array(self.diffs)
            self.diffs = {f'Item_{item + 1}': diff for item, diff in enumerate(self.diffs)}
            self.diffs = pd.Series(self.diffs)

        if self.thresholds is None:
            category_widths = np.random.uniform(self.max_disorder,
                                                2 * self.category_mean - self.max_disorder,
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
        
        if self.severities is None:
            severities = [truncnorm.rvs(-1.96, 1.96, size = self.max_score)
                               for rater in range(self.no_of_raters)]
            severities = np.array(severities)
            severities *= (self.rater_range / (np.max(severities) - np.min(severities)))
            severities -= np.mean(severities)
            severities = np.insert(severities, 0, 0, axis = 1)

            self.severities = {f'Rater_{rater + 1}': severities[rater, :]
                               for rater in range(self.no_of_raters)}
            
        else:
            assert len(manual_severities) == self.no_of_raters, 'Length of manual severities must match number of raters.'
            for rater in range(self.no_of_raters):
                assert len(manual_severities[rater]) == self.max_score + 1, ('Length of all sets of manual severities' +
                                                                             ' must be max score plus 1.')
                assert manual_severities[rater][0] == 0, ('First threshold in every set of manual severities must ' +
                                                          'have value zero.')
                assert sum(manual_severities[rater]) == 0 , ('Every set of manual thresholds must sum to zero.')
            self.severities = np.array(self.severities)

            self.severities = {f'Rater_{rater + 1}': np.array(severities[rater][:])
                               for rater in range(self.no_of_raters)}

        self.items = [f'Item_{item + 1}' for item in range(self.no_of_items)]
        self.raters = [f'Rater_{rater + 1}' for rater in range(self.no_of_raters)]
        self.persons = [f'Person_{item + 1}' for person in range(self.no_of_persons)]
        
        '''
        Calculates probability of a response in each category
        for each person on each item
        '''

        cat_probs = [[[[self.mfrm.cat_prob_thresholds(ability, item, difficulties, rater, self.severities, category,
                                                      self.thresholds)
                        for category in range(self.max_score + 1)]
                       for rater in self.raters]
                      for item in self.dataframe.columns]
                     for ability in self.abilities]

        self.cat_probs = np.array(cat_probs)
        
        '''
        Calculates scores and removes required amount of missing data
        '''
        
        scoring_randoms = np.array([self.randoms()
                                    for rater in range(self.no_of_raters)])
        scoring_randoms = np.swapaxes(scoring_randoms, 0, 2)
        scoring_randoms = np.swapaxes(scoring_randoms, 0, 1)
            
        cat_scores = np.zeros((self.no_of_persons,
                               self.no_of_items,
                               self.no_of_raters,
                               self.max_score + 1))
        
        for rater in range(self.no_of_raters):
            for cat in range(self.max_score):
            
                cat_scores[:, :, rater, cat] = (scoring_randoms[:, :, rater] >
                                                np.sum(self.cat_probs[:, :, rater, :cat + 1],
                                                       axis = 2)).astype(np.float64)
        
        self.scores = np.sum(cat_scores, axis = 3)
        
        missing_randoms = self.randoms()
        
        self.scores[missing_randoms < self.missing] = np.nan
        
        cols = [f'Item_{item + 1}' for item in range(self.no_of_items)]
        
        final = {rater: pd.DataFrame(self.scores[:, :, i],
                                     columns=cols, index=self.persons)
                 for i, rater in enumerate(self.raters)}
        
        self.scores = pd.concat(final.values(), keys = final.keys())

class MFRM_Sim_Matrix(MFRM_Sim):
    
    '''
    Generates simulated data accoding to the RSM formulation
    of the extended full-matrix form of the Many-Facet Rasch Model (MFRM).
    '''

    def __init__(self,
                 no_of_items,
                 no_of_persons,
                 no_of_raters,
                 max_score,
                 item_range=2,
                 rater_range=2,
                 category_mean=1,
                 person_sd=1.5,
                 max_disorder = 0,
                 offset = 0,
                 missing = 0,
                 manual_abilities=None,
                 manual_diffs=None,
                 manual_thresholds=None,
                 manual_severities=None):

        self.no_of_items = int(no_of_items)
        self.no_of_persons = int(no_of_persons)
        self.no_of_raters = int(no_of_raters)
        self.item_range = item_range
        self.rater_range = rater_range
        self.max_score = max_score
        self.category_mean = category_mean
        self.person_sd = person_sd
        self.max_disorder = max_disorder
        self.offset = offset
        self.missing = missing
        self.abilities = manual_abilities
        self.diffs = manual_diffs
        self.thresholds = manual_thresholds
        self.severities = manual_severities
        self.dataframe = self.produce_df([[0], [1]], [[0]])
        self.dataframe.loc[0].iloc[0, 0] = 0
        self.mfrm = MFRM(self.dataframe, self.max_score)

        '''
        Generates person, item and threshold parameters.
        '''

        if self.abilities is None:
            self.abilities = np.random.normal(0, self.person_sd, self.no_of_persons)
            self.abilities -= np.mean(self.abilities)
            self.abilities += self.offset
            self.abilities = {f'Person_{person + 1}': ability for person, ability in enumerate(self.abilities)}
            self.abilities = pd.Series(self.abilities)

        else:
            assert len(self.abilities) == self.no_of_persons, 'Length of manual abilities must match number of persons.'
            self.abilities = np.array(self.abilities)
            self.abilities = {f'Person_{person + 1}': ability for person, ability in enumerate(self.abilities)}
            self.abilities = pd.Series(self.abilities)

        if self.diffs is None:
            self.diffs = np.random.uniform(0, 1, self.no_of_items)
            self.diffs *= (self.item_range /
                           (np.max(self.diffs) - np.min(self.diffs)))
            self.diffs -= np.mean(self.diffs)
            self.diffs = {f'Item_{item + 1}': diff for item, diff in enumerate(self.diffs)}
            self.diffs = pd.Series(self.diffs)

        else:
            assert len(self.diffs) == self.no_of_items, 'Length of manual difficulties must match number of items.'
            self.diffs = np.array(self.diffs)
            self.diffs = {f'Item_{item + 1}': diff for item, diff in enumerate(self.diffs)}
            self.diffs = pd.Series(self.diffs)

        if self.thresholds is None:
            category_widths = np.random.uniform(self.max_disorder,
                                                2 * self.category_mean - self.max_disorder,
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
        
        if self.severities is None:
            severities = [[truncnorm.rvs(-1.96, 1.96, size = self.max_score)
                                for item in range(self.no_of_items)]
                               for rater in range(self.no_of_raters)]
            severities = np.array(severities)
            severities *= (self.rater_range / (np.max(severities) - np.min(severities)))
            severities -= np.mean(severities)
            severities = np.insert(severities, 0, 0, axis = 2)

            self.severities = {f'Rater_{rater + 1}': {f'Item_{item + 1}': severities[rater, item, :]
                                                      for item in range(self.no_of_items)}
                               for rater in range(self.no_of_raters)}
            
        else:
            assert len(manual_severities) == self.no_of_raters, 'Length of manual severities must match number of raters.'
            for rater in range(self.no_of_raters):
                assert len(manual_severities[rater]) == self.no_of_items, ('Length of all sets of manual threshold ' +
                                                                           'severity sets must match number of items.')
                for item in range(self.no_of_items):
                    assert len(manual_severities[rater][item]) == self.max_score + 1, ('Number of manual threshold ' +
                        'severities in each set must be max score plus 1.')
                    assert manual_severities[rater][item][0] == 0, ('First threshold in each set of manual ' +
                                                                    'threshold severities must have value zero.')
                    assert sum(manual_severities[rater][item]) == 0 , ('Each set of manual threshold severities ' +
                                                                       'must sum to zero.')
            self.severities = np.array(self.severities)

            self.severities = {f'Rater_{rater + 1}': {f'Item_{item + 1}': severities[rater, item, :]
                                                      for item in range(self.no_of_items)}
                               for rater in range(self.no_of_raters)}

        self.items = [f'Item_{item + 1}' for item in range(self.no_of_items)]
        self.raters = [f'Rater_{rater + 1}' for rater in range(self.no_of_raters)]
        self.persons = [f'Person_{item + 1}' for person in range(self.no_of_persons)]

        '''
        Calculates probability of a response in each category
        for each person on each item
        '''

        cat_probs = [[[[self.mfrm.cat_prob_matrix(ability,
                                                  item,
                                                  self.diffs,
                                                  rater,
                                                  self.severities,
                                                  category,
                                                  self.thresholds)
                        for category in range(self.max_score + 1)]
                       for rater in self.raters]
                      for item in self.diffs.keys()]
                     for ability in self.abilities]

        self.cat_probs = np.array(cat_probs)
        
        '''
        Calculates scores and removes required amount of missing data
        '''
        
        scoring_randoms = np.array([self.randoms()
                                    for rater in range(self.no_of_raters)])
        scoring_randoms = np.swapaxes(scoring_randoms, 0, 2)
        scoring_randoms = np.swapaxes(scoring_randoms, 0, 1)
            
        cat_scores = np.zeros((self.no_of_persons,
                               self.no_of_items,
                               self.no_of_raters,
                               self.max_score + 1))
        
        for rater in range(self.no_of_raters):
            
            for cat in range(self.max_score):
            
                cat_scores[:, :, rater, cat] = (scoring_randoms[:, :, rater] >
                                                np.sum(self.cat_probs[:, :, rater, :cat + 1],
                                                       axis = 2)).astype(np.float64)
        
        self.scores = np.sum(cat_scores, axis = 3)
        
        missing_randoms = self.randoms()
        
        self.scores[missing_randoms < self.missing] = np.nan
        
        cols = [f'Item_{item + 1}' for item in range(self.no_of_items)]
        
        final = {rater: pd.DataFrame(self.scores[:, :, i],
                                     columns=cols, index=self.persons)
                 for i, rater in enumerate(self.raters)}

        self.scores = pd.concat(final.values(), keys = final.keys())