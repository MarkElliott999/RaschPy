from math import log, sqrt
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
from matplotlib import colors as colors
from matplotlib import cm as cmx
import seaborn as sns

from raschpy.base import Rasch

class SLM(Rasch):

    def __init__(self,
                 dataframe,
                 extreme_persons=True,
                 no_of_classes=5,
                 validate=True):

        super().__init__()

        # Handle extreme/invalid persons
        if extreme_persons:
            self.invalid_responses = dataframe[dataframe.isna().all(axis=1)]
            self.dataframe = dataframe[~dataframe.isna().all(axis=1)]
        else:
            row_sums = dataframe.sum(axis=1)
            row_counts = dataframe.count(axis=1)

            zero_scores = dataframe[row_sums == 0]
            all_correct = dataframe[row_sums == row_counts]

            self.invalid_responses = pd.concat([zero_scores, all_correct], axis=0)
            self.dataframe = dataframe[~dataframe.index.isin(self.invalid_responses.index)]

        # Set foundational metadata attributes
        self.no_of_items = self.dataframe.shape[1]
        self.no_of_persons = self.dataframe.shape[0]

        self.items = self.dataframe.columns
        self.persons = self.dataframe.index
        self.no_of_classes = no_of_classes
        self.max_score = 1

        # RUN AUTOMATIC CONNECTION CHECK VALIDATION
        if validate:
            self.connectivity_status = self.check_data_connectivity()

            # THROW SYSTEM WARNING WITH MATHEMATICAL DETAILS IF DISCONNECTED
            if not self.connectivity_status["connected"]:
                warnings.warn(
                    f"\n"
                    f"⚠️  CRITICAL DATA INTEGRITY WARNING: The response data is disconnected into "
                    f"{self.connectivity_status['components_count']} separate sub-networks.\n"
                    f"Difficulty estimates will be problematic because there are no "
                    f"empirical comparisons spanning across these isolated groups, the item parameter "
                    f"estimates for each independent subset will separately sum to zero. This means items "
                    f"belonging to different subsets cannot be compared or calibrated onto a single scale.",
                    category=UserWarning,
                    stacklevel=2
                )

    def exp_score(self, ability, difficulty):
        '''
        Expected score function (also probability of correct response).
        Fully vectorised: Accepts scalars, 1D vectors, or 2D matrices.
        '''
        return 1.0 / (1.0 + np.exp(difficulty - ability))

    def cat_prob(self, ability, difficulty, category):
        '''
        Category probability function which calculates the probability
        of scoring 0 or 1 from person ability and item difficulty.
        Vectorised to avoid list construction and explicit sums.
        '''
        p = self.exp_score(ability, difficulty)
        # For category 1: returns p. For category 0: returns (1 - p).
        return p if category == 1 else (1.0 - p)

    def variance(self, ability, difficulty):
        '''
        Calculates Fisher information function from person ability
        and item difficulty. Also the variance and differential
        of the expected score function.
        '''
        expected = self.exp_score(ability, difficulty)
        return expected * (1.0 - expected)

    def kurtosis(self, ability, difficulty):
        '''
        Calculates kurtosis given person ability and item difficulty.
        Vectorised to eliminate the loop structure over categories.
        '''
        expected = self.exp_score(ability, difficulty)

        # Category 0 term: ((0 - expected)**4) * (1 - expected)
        term_1 = (expected ** 4) * (1.0 - expected)

        # Category 1 term: ((1 - expected)**4) * expected
        term_2 = ((1.0 - expected) ** 4) * expected

        return term_1 + term_2

    def calibrate(self,
                  constant=0.1,
                  method='cos',
                  matrix_power=3,
                  log_lik_tol=0.000001):
        '''
        Produces central item difficulty estimates (or difficulties for SLM)
        using Choppin's matrix power property to resolve structural zeroes.
        '''

        if len(self.dataframe.columns) == 1:
            warnings.warn("Only one item detected. This model with a single item reduces to RSM "
                          "with raters as items. Consider reconfiguring and using RSM instead.",
                          UserWarning, stacklevel=2)

        if constant == 0:
            all_max_items = [item for item in self.dataframe.columns
                             if self.dataframe[item].dropna().eq(self.dataframe[item].max()).all()]

            if all_max_items:
                warnings.warn(f"Items with all-maximum scores detected with constant=0: "
                              f"{list(all_max_items)}. Item estimation will fail. "
                              f"Either drop these items or use a non-zero constant.",
                              UserWarning, stacklevel=2)

        # 1. Clean up entirely empty rows (persons with zero data)
        self.null_persons = self.dataframe.index[self.dataframe.isnull().all(axis=1)]
        self.dataframe = self.dataframe.drop(self.null_persons)

        # Extract structural array and dimensions locally as guaranteed integers
        df_array = np.array(self.dataframe, dtype=np.float64)

        # 2. VECTORISED PAIRWISE COMPARISON MATRIX (With proper float casting)
        # CRITICAL FIX: Casting boolean arrays to float64 forces numerical summation
        is_one = ((df_array == 1) & (~np.isnan(df_array))).astype(np.float64)
        is_zero = ((df_array == 0) & (~np.isnan(df_array))).astype(np.float64)

        # Fast BLAS/LAPACK matrix dot product for true joint frequencies
        matrix = np.dot(is_one.T, is_zero)

        # 3. Compute matrix powers (Keep the diagonal as 0 so Choppin's math stays pure)
        mat = np.linalg.matrix_power(matrix, matrix_power)
        mat_pow = matrix_power

        # 4. CHOPPIN ZERO CHECK (Ignore the main diagonal)
        off_diagonal_mask = ~np.eye(self.no_of_items, dtype=bool)

        # The loop now only checks for structural zeroes where item_1 != item_2
        while np.any(mat[off_diagonal_mask] == 0):
            mat = np.dot(mat, matrix)
            mat_pow += 1

            # Breakout safeguard if the item network is fundamentally disconnected
            if mat_pow >= matrix_power + 5:
                # If graph is disconnected, apply the constant strictly to remaining off-diagonal zeroes
                mat[off_diagonal_mask & (mat == 0)] = constant
                break

        # 5. Extract Priority Vector using the corrected numerical matrix
        self.diffs = self.priority_vector(mat, method=method, log_lik_tol=log_lik_tol)

    def std_errors(self,
                   interval=None,
                   no_of_samples=500,
                   constant=0.1,
                   method='cos',
                   matrix_power=3,
                   log_lik_tol=0.000001):
        
        '''
        Bootstraped standard error estimates for item difficulties.
        '''

        samples = [SLM(self.dataframe.sample(frac=1, replace=True), validate=False)
                   for sample in range(no_of_samples)]

        for sample in samples:
            sample.calibrate(constant=constant, method=method, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        self.bootstrap_sample_diffs = {f'Sample_{i + 1}': sample.diffs for i, sample in enumerate(samples)}

        item_ests = np.array([sample.diffs.loc[self.items].values for sample in samples])

        self.item_se = {item: se for item, se in zip(self.items,
                                                     np.nanstd(item_ests, axis=0))}
        self.item_se = pd.Series(self.item_se)

        if interval is not None:
            self.item_low = {item: low for item, low in zip(self.items,
                                                            np.nanpercentile(item_ests, (1 - interval) * 50, axis=0))}
            self.item_low = pd.Series(self.item_low)

            self.item_high = {item: high for item, high in zip(self.items,
                                                               np.nanpercentile(item_ests, (1 + interval) * 50, axis=0))}
            self.item_high = pd.Series(self.item_high)

        else:
            self.item_low = None
            self.item_high = None

        self.item_bootstrap = pd.DataFrame(item_ests)
        self.item_bootstrap.columns = self.dataframe.columns
        self.item_bootstrap.index = [f'Sample {i + 1}' for i in range (no_of_samples)]

    def abil(self,
             persons,
             items=None,
             warm_corr=True,
             tolerance=0.00001,
             max_iters=100,
             ext_score_adjustment=0.5):

        '''
        Creates raw score to ability estimate look-up. Uses
        Newton-Raphson for ML with optional Warm (1989) bias correction.
        '''

        if isinstance(persons, str):
            if persons == 'all':
                persons = self.persons

            else:
                persons = [persons]

        if isinstance(items, str):
            if items == 'all':
                items = self.items

            else:
                items = [items]

        if items is None:
            items = self.items
            difficulties = self.diffs
            person_data = self.dataframe.loc[persons]

        else:
            difficulties = self.diffs.loc[items]
            person_data = self.dataframe.loc[persons, items]

        person_filter = (person_data + 1) / (person_data + 1)
        scores = person_data.sum(axis=1).astype(float)
        ext_scores = person_filter.sum(axis=1)

        scores[scores == 0] += ext_score_adjustment
        scores[scores == ext_scores] -= ext_score_adjustment

        diff_df = pd.DataFrame(np.tile(difficulties.values[None, :], (len(persons), 1)),
                               index=persons, columns=difficulties.index)
        diff_df *= person_filter

        try:
            estimates = np.log(scores) - np.log(ext_scores - scores) + diff_df.mean(axis=1)
            changes = pd.Series({person: 1 for person in persons})
            iters = 0

            while (abs(changes).max() > tolerance) & (iters <= max_iters):
                exp_score_df = pd.DataFrame(1 / (1 + np.exp(difficulties.values[None, :] - estimates.values[:, None])),
                                            index=persons, columns=difficulties.index)

                info_df = exp_score_df * (1 - exp_score_df)

                exp_score_df *= person_filter
                info_df *= person_filter

                result_list = exp_score_df.sum(axis=1)
                info_list = info_df.sum(axis=1)

                changes = (result_list - scores) / info_list
                changes = changes.clip(-1, 1)
                estimates -= changes
                iters += 1

            if warm_corr:
                estimates += self.warm(estimates, difficulties, person_filter)

            if iters >= max_iters:
                print('Maximum iterations reached before convergence.')

        except Exception:
            estimates = np.nan

        return estimates

    def person_abils(self,
                     items=None,
                     warm_corr=True,
                     tolerance=0.00001,
                     max_iters=100,
                     ext_score_adjustment=0.5):

        '''
        Creates raw score to ability estimate look-up table. Newton-Raphson ML
        estimation, includes optional Warm (1989) bias correction.
        '''

        self.person_abilities = self.abil(self.persons, items=items, warm_corr=warm_corr, tolerance=tolerance,
                                          max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)

    def score_abil(self,
                   score,
                   items=None,
                   warm_corr=True,
                   tolerance=0.00001,
                   max_iters=100,
                   ext_score_adjustment=0.5):

        '''
        Creates raw score to ability estimate look-up. Uses
        Newton-Raphson for ML with optional Warm (1989) bias correction.
        '''

        if items is None:
            items = self.items

        if isinstance(items, str):
            if items == 'all':
                items = self.items

        difficulties = self.diffs.loc[items]

        person_filter = np.ones(len(items))
        max_score = len(difficulties)

        if score == 0:
            score = ext_score_adjustment

        elif score == max_score:
            score -= ext_score_adjustment

        estimate = log(score) - log(max_score - score) + difficulties.mean()
        change = 1
        iters = 0

        diffs_arr = difficulties.values

        while (abs(change) > tolerance) & (iters <= max_iters):

            p = 1.0 / (1.0 + np.exp(diffs_arr - estimate))
            result = p.sum()
            info = (p * (1.0 - p)).sum()

            change = max(-1, min(1, (result - score) / info))
            estimate -= change
            iters += 1

        if warm_corr:
            estimate += self.warm(estimate, difficulties, person_filter)

        if iters >= max_iters:
            print('Maximum iterations reached before convergence.')

        return estimate

    def abil_lookup_table(self,
                          items=None,
                          ext_scores=True,
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

        no_of_items = len(items)
        difficulties = self.diffs.loc[items]
            
        if ext_scores:
            scores = np.arange(no_of_items + 1)

            used_scores = scores.astype(float)
            used_scores[0] += ext_score_adjustment
            used_scores[-1] -= ext_score_adjustment
            
        else:
            scores = np.arange(1, no_of_items)
            used_scores = scores.astype(float)

        estimates = {score: np.log(used_score) - np.log(no_of_items - used_score) + difficulties.mean()
                     for score, used_score in zip(scores, used_scores)}
        estimates = pd.Series(estimates, index=scores)

        changes = pd.Series(1, index=scores)
        iters = 0

        while (abs(changes).max() > tolerance) & (iters <= max_iters):
            exp_score_df = pd.DataFrame(1 / (1 + np.exp(difficulties.values[None, :] - estimates.values[:, None])),
                                        index=scores, columns=difficulties.index)

            info_df = exp_score_df * (1 - exp_score_df)

            result_list = exp_score_df.sum(axis=1)
            info_list = info_df.sum(axis=1)

            changes = (result_list - used_scores) / info_list
            changes = changes.clip(-1, 1)
            estimates -= changes
            iters += 1

        if warm_corr:
            person_filter = pd.DataFrame(1, columns=items, index=scores)
            estimates += self.warm(estimates, difficulties, person_filter)

        self.abil_table = estimates

    def warm(self,
             abilities,
             difficulties,
             person_filter):

        '''
        Warm's (1989) bias correction for ML ability estimates.
        '''

        if np.isscalar(abilities):
            p = 1.0 / (1.0 + np.exp(difficulties.values - abilities))
            info = p * (1.0 - p)
            i = (info * person_filter).sum()
            j = (info * (1 - 2 * p) * person_filter).sum()
            return j / (2 * i ** 2)

        exp_score_df = pd.DataFrame(
            1 / (1 + np.exp(difficulties.values[None, :] - abilities.values[:, None])),
            index=abilities.index, columns=difficulties.index)

        info_df = exp_score_df * (1 - exp_score_df)

        j_df = info_df * (1 - 2 * exp_score_df)

        exp_score_df *= person_filter
        info_df *= person_filter
        j_df *= person_filter

        i = info_df.sum(axis=1)
        j = j_df.sum(axis=1)

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

        diffs_arr = difficulties.values
        mask = person_filter.values
        p = 1.0 / (1.0 + np.exp(diffs_arr - abilities[person]))
        total_info = np.nansum(p * (1.0 - p) * mask)

        return 1 / sqrt(total_info)

    def category_counts_item(self,
                             item):

        if item in self.dataframe.columns:
            counts = self.dataframe[item].value_counts().fillna(0).astype(int)
            counts.sort_index(inplace=True)

            return counts

        else:
            print('Invalid item name')

    def category_counts_df(self):
        cat_counts_dict = {item: {int(score): count if count == count else 0
                                  for score, count in self.category_counts_item(item).items()}
                           for item in self.dataframe.columns}
        category_counts_df = pd.DataFrame(cat_counts_dict).T

        category_counts_df['Responses'] = self.dataframe.count()
        category_counts_df['Missing'] = self.no_of_persons - category_counts_df['Responses']

        category_counts_df = category_counts_df.astype(int)

        category_counts_df.loc['Total']= category_counts_df.sum()

        self.category_counts = category_counts_df

    def fit_statistics(self,
                       warm_corr=True,
                       se=True,
                       test_stats=True,
                       trim_cat_prob_dict=False,
                       tolerance=0.00001,
                       max_iters=100,
                       ext_score_adjustment=0.5,
                       constant=0.1,
                       method='cos',
                       matrix_power=3,
                       log_lik_tol=0.000001,
                       no_of_samples=500,
                       interval=None):

        if hasattr(self, 'diffs') == False:
            self.calibrate(constant=constant, method=method, matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if se:
            if hasattr(self, 'item_se') == False:
                self.std_errors(interval=interval, no_of_samples=no_of_samples, constant=constant, method=method,
                                matrix_power=matrix_power, log_lik_tol=log_lik_tol)

        if hasattr(self, 'person_abilities') == False:
            self.person_abils(warm_corr=warm_corr, tolerance=tolerance,
                              max_iters=max_iters, ext_score_adjustment=ext_score_adjustment)

        if se == False:
            test_stats = False

        '''
        Create matrices of expected scores, variances, kurtosis, residuals etc. to generate fit statistics
        '''

        item_count = (self.dataframe == self.dataframe).sum(axis=0)
        person_count = (self.dataframe == self.dataframe).sum(axis=1)

        df = self.dataframe.copy()
        scores = df.sum(axis=1)
        max_scores = (df == df).sum(axis=1)

        df = df[(scores > 0) & (scores < max_scores)]
        missing_mask = (df + 1) / (df + 1)

        p = pd.DataFrame(1 / (1 + np.exp(self.diffs.values[None, :] - self.person_abilities.values[:, None])),
                         index=self.persons, columns=self.items)
        self.cat_prob_dict = {1: p, 0: 1 - p}

        if trim_cat_prob_dict:
            for cat in [0, 1]:
                self.cat_prob_dict[cat] = self.cat_prob_dict[cat].loc[df.index]

        self.exp_score_df = self.cat_prob_dict[1].copy()
        self.exp_score_df *= missing_mask

        self.info_df = self.cat_prob_dict[1] * self.cat_prob_dict[0]
        self.info_df *= missing_mask

        p1 = self.cat_prob_dict[1]
        p0 = self.cat_prob_dict[0]
        self.kurtosis_df = p0 * (p1 ** 4) + p1 * (p0 ** 4)
        self.kurtosis_df *= missing_mask

        self.residual_df = self.dataframe - self.exp_score_df
        self.std_residual_df = self.residual_df / (self.info_df ** 0.5)

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

        item_outfit_q = ((self.kurtosis_df / (self.info_df ** 2)) /
                         (item_count.loc[self.kurtosis_df.columns] ** 2)).sum() - (1 / item_count.loc[self.kurtosis_df.columns])
        item_outfit_q = item_outfit_q ** 0.5
        self.item_outfit_zstd = ((self.item_outfit_ms ** (1/3)) - 1) * (3 / item_outfit_q) + (item_outfit_q / 3)

        item_infit_q = (self.kurtosis_df - self.info_df ** 2).sum() / (self.info_df.sum() ** 2)
        item_infit_q = item_infit_q ** 0.5
        self.item_infit_zstd = ((self.item_infit_ms ** (1/3)) - 1) * (3 / item_infit_q) + (item_infit_q / 3)

        self.response_counts = self.dataframe.count(axis=0)
        self.item_facilities = self.dataframe.mean(axis=0)

        (self.point_measure,
         self.exp_point_measure) = self.pt_meas(self.person_abilities, self.exp_score_df, self.info_df)

        '''
        Person fit statistics
        '''

        self.csem_vector = 1 / (self.info_df.sum(axis=1) ** 0.5)
        self.rsem_vector = ((self.residual_df ** 2).sum(axis=1) ** 0.5) / self.info_df.sum(axis=1)


        self.person_outfit_ms = (self.std_residual_df ** 2).mean(axis=1)
        self.person_outfit_ms.name = 'Outfit MS'
        self.person_infit_ms = (self.residual_df ** 2).sum(axis=1) / self.info_df.sum(axis=1)
        self.person_infit_ms.name = 'Infit MS'

        base_df = self.kurtosis_df / (self.info_df ** 2)
        base_df = base_df.div(person_count.loc[base_df.index] ** 2, axis=0)

        person_outfit_q = base_df.sum(axis=1) - (1 / person_count.loc[base_df.index])
        person_outfit_q = person_outfit_q ** 0.5
        self.person_outfit_zstd = ((self.person_outfit_ms ** (1/3)) - 1) * (3 / person_outfit_q) + (person_outfit_q / 3)
        self.person_outfit_zstd.name = 'Outfit Z'

        person_infit_q = (self.kurtosis_df - self.info_df ** 2).sum(axis=1) / (self.info_df.sum(axis=1) ** 2)
        person_infit_q = person_infit_q ** 0.5
        self.person_infit_zstd = ((self.person_infit_ms ** (1/3)) - 1) * (3 / person_infit_q) + (person_infit_q / 3)
        self.person_infit_zstd.name = 'Infit Z'

        differences = pd.DataFrame(
            self.person_abilities.values[:, None] - self.diffs.values[None, :],
            index=self.persons, columns=self.items)
        differences = differences.loc[self.residual_df.index]
        num = (differences * self.residual_df).sum(axis=0)
        den = (self.info_df * (differences ** 2)).sum(axis=0)
        self.discrimination = 1 + num / den

        '''
        Test-level fit statistics
        '''

        if test_stats:
            item_ests = self.item_bootstrap.values
            isi_samples = (np.var(item_ests, axis=1, ddof=1) /
                           np.var(item_ests - item_ests.mean(axis=1, keepdims=True), axis=0).mean())
            self.isi = np.sqrt(np.mean(isi_samples) - 1)

            self.item_strata = (4 * self.isi + 1) / 3
            self.item_reliability = self.isi ** 2 / (1 + self.isi ** 2)

            psi_var = max(0, np.var(self.person_abilities) - (self.rsem_vector ** 2).mean())
            self.psi = (psi_var ** 0.5) / ((self.rsem_vector ** 2).mean() ** 0.5)

            self.person_strata = (4 * self.psi + 1) / 3
            self.person_reliability = (self.psi ** 2) / (1 + (self.psi ** 2))

        self.item_residual_corr = self.std_residual_df.corrwith(self.diffs, axis=1)
        self.person_residual_corr = self.std_residual_df.corrwith(self.person_abilities.loc[self.std_residual_df.index],
                                                                  axis=0)

    def res_corr_analysis(self,
                          warm_corr=True,
                          tolerance=0.00001,
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

            self.loadings = self.eigenvectors.T * (pca.explained_variance_ ** 0.5)
            self.loadings = pd.DataFrame(self.loadings)
            self.loadings.columns = [f'PC {pc + 1}' for pc in range(self.no_of_items)]
            self.loadings.index = [item for item in self.dataframe.columns]

        except Exception:
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
                      tolerance=0.00001,
                      max_iters=100,
                      ext_score_adjustment=0.5,
                      method='cos',
                      constant=0.1,
                      no_of_samples=500,
                      interval=None):

        if full:
            zstd = True
            disc=True
            point_measure_corr = True

            if interval is None:
                interval = 0.95

        if hasattr(self, 'item_se') == False or (interval is not None and hasattr(self, 'item_low') == False):
            self.std_errors(interval=interval, no_of_samples=no_of_samples, constant=constant, method=method)

        if hasattr(self, 'item_infit_ms') == False:
            self.fit_statistics(warm_corr=warm_corr, tolerance=tolerance, max_iters=max_iters,
                                ext_score_adjustment=ext_score_adjustment, method=method,
                                constant=constant, no_of_samples=no_of_samples, interval=interval)

        self.item_stats = pd.DataFrame()

        self.item_stats['Estimate'] = self.diffs.astype(float).round(dp)
        self.item_stats['SE'] = self.item_se.astype(float).round(dp)

        if interval is not None:
            self.item_stats[f'{round((1 - interval) * 50, 1)}%'] = self.item_low.astype(float).round(dp)
            self.item_stats[f'{round((1 + interval) * 50, 1)}%'] = self.item_high.astype(float).round(dp)

        self.item_stats['Count'] = self.response_counts.astype(int)
        self.item_stats['Facility'] = self.item_facilities.astype(float).round(dp)

        self.item_stats['Infit MS'] = self.item_infit_ms.astype(float).round(dp)
        if zstd:
            self.item_stats['Infit Z'] = self.item_infit_zstd.astype(float).round(dp)

        self.item_stats['Outfit MS'] = self.item_outfit_ms.astype(float).round(dp)
        if zstd:
            self.item_stats['Outfit Z'] = self.item_outfit_zstd.astype(float).round(dp)

        if disc:
            self.item_stats['Discrim'] = self.discrimination.astype(float).round(dp)

        if point_measure_corr:
            self.item_stats['PM corr'] = self.point_measure.astype(float).round(dp)
            self.item_stats['Exp PM corr'] = self.exp_point_measure.astype(float).round(dp)

        self.item_stats.index = self.dataframe.columns

    def person_stats_df(self,
                        full=False,
                        rsem=False,
                        dp=3,
                        warm_corr=True,
                        tolerance=0.00001,
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

        person_stats_df['Infit MS'] = np.nan
        person_stats_df['Infit Z'] = np.nan
        person_stats_df['Outfit MS'] = np.nan
        person_stats_df['Outfit Z'] = np.nan

        person_stats_df.update({'Infit MS': self.person_infit_ms.round(dp)})
        person_stats_df.update({'Infit Z': self.person_infit_zstd.round(dp)})
        person_stats_df.update({'Outfit MS': self.person_outfit_ms.round(dp)})
        person_stats_df.update({'Outfit Z': self.person_outfit_zstd.round(dp)})

        self.person_stats = person_stats_df
        
    def test_stats_df(self,
                      dp=3,
                      warm_corr=True,
                      tolerance=0.00001,
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
                   tolerance=0.00001,
                   max_iters=100,
                   ext_score_adjustment=0.5,
                   method='cos',
                   constant=0.1,
                   no_of_samples=500,
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

            writer = pd.ExcelWriter(filename, engine='openpyxl')

            self.item_stats.to_excel(writer, sheet_name='Item statistics')
            self.person_stats.to_excel(writer, sheet_name='Person statistics')
            self.test_stats.to_excel(writer, sheet_name='Test statistics')

            writer.close()

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
                       tolerance=0.00001,
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

                writer = pd.ExcelWriter(filename, engine='openpyxl')
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

                writer.close()

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

                writer = pd.ExcelWriter(filename, engine='openpyxl')

                self.eigenvectors.round(dp).to_excel(writer, sheet_name='Eigenvectors')
                self.eigenvalues.round(dp).to_excel(writer, sheet_name='Eigenvalues')
                self.variance_explained.round(dp).to_excel(writer, sheet_name='Variance explained')
                self.loadings.round(dp).to_excel(writer, sheet_name='Principal Component loadings')

                writer.close()

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

        df = self.dataframe.copy()

        if items is None:
            items = list(self.dataframe.columns)

        df = df[items].dropna()
        abils = self.person_abilities.loc[df.index]

        quantiles = (abils.quantile([(i + 1) / no_of_classes
                                     for i in range(no_of_classes - 1)]))

        mask_dict = {}
        mask_dict['class_1'] = (abils < quantiles.values[0])
        mask_dict[f'class_{no_of_classes}'] = (abils >= quantiles.values[no_of_classes - 2])
        for class_no in range(no_of_classes - 2):
            mask_dict[f'class_{class_no + 2}'] = ((abils >= quantiles.values[class_no]) &
                                                  (abils < quantiles.values[class_no + 1]))

        mean_abilities = {class_group: abils[mask_dict[class_group]].mean()
                          for class_group in class_groups}
        mean_abilities = pd.Series(mean_abilities)

        obs = {class_group: df[mask_dict[class_group]].mean().sum()
               for class_group in class_groups}

        for class_group in class_groups:
            obs[class_group] = pd.Series(obs[class_group])

        obs = pd.concat(obs, keys=obs.keys())

        return mean_abilities, obs

    def class_intervals_cats(self,
                             item,
                             no_of_classes=5):

        class_groups = [f'class_{class_no + 1}' for class_no in range(no_of_classes)]

        mean_abilities, obs_means = self.class_intervals(items=[item], no_of_classes=no_of_classes)

        obs_props = {class_group: np.array([1 - obs_means[class_group][0], obs_means[class_group][0]])
                     for class_group in class_groups}

        obs_props = pd.DataFrame(obs_props).to_numpy().T

        return mean_abilities, obs_props

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
                  score_lines_item=[None, None],
                  score_lines_test=None,
                  point_info_lines_item=[None, None],
                  point_info_lines_test=None,
                  point_csem_lines=None,
                  score_labels=False,
                  point_info_labels=False,
                  warm=True,
                  cat_highlight=None,
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

        if tex:
            plt.rcParams["text.latex.preamble"].join([r"\usepackage{dashbox}", r"\setmainfont{xcolor}",])
        else:
            plt.rcParams["text.usetex"] = False

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

        if obs:
            no_of_obs_plots = y_obs_data.shape[1]
            for j in range (no_of_obs_plots):
                if 'multi' not in palette:
                    colorVal = scalarMap.to_rgba(j)
                else:
                    colorVal = color_map[j]

                ax.plot(x_obs_data, y_obs_data[:, j], 'o', color=colorVal)

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

            if items is None:
                no_of_items = self.no_of_items

            else:
                if isinstance(items, list):
                    no_of_items = len(items)

                else:
                    no_of_items = 1

            if (all(x > 0 for x in score_lines_test) & all(x < no_of_items for x in score_lines_test)):

                if items is None:
                    abils_set = [self.score_abil(score, items=self.dataframe.columns, warm_corr=False)
                                 for score in score_lines_test]

                else:
                    abils_set = [self.score_abil(score, items=items, warm_corr=False)
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

            diffs_arr = difficulties.values
            p = self.exp_score(np.array(point_info_lines_test)[:, None], diffs_arr[None, :])
            info_set = (p * (1 - p)).sum(axis=1)

            for abil, info in zip(point_info_lines_test, info_set):
                plt.vlines(x=abil, ymin=-100, ymax=info, color='black', linestyles='dashed')
                if point_info_labels:
                    plt.text(abil + (x_max - x_min) / 100, y_max / 50, str(round(abil, 2)))
                plt.hlines(y=info, xmin=-100, xmax=abil, color='black', linestyles='dashed')
                if point_info_labels:
                    plt.text(x_min + (x_max - x_min) / 100, info + y_max / 50, str(round(info, 3)))

        if point_csem_lines is not None:

            diffs_arr = difficulties.values
            p = self.exp_score(np.array(point_csem_lines)[:, None], diffs_arr[None, :])
            info_set = (p * (1 - p)).sum(axis=1)

            info_set = np.array(info_set)
            csem_set = 1 / (info_set ** 0.5)

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
            no_of_classes=5,
            title=None,
            thresh_line=False,
            score_lines=None,
            score_labels=False,
            cat_highlight=None,
            xmin=-5,
            xmax=5,
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
        Plots Item Characteristic Curves for SLM, with optional overplotting
        of observed data, threshold lines and expected score threshold lines.
        '''

        if obs:
            if hasattr(self, 'person_abilities') == False:
                self.person_abils(warm_corr=False)

            xobsdata, yobsdata = self.class_intervals(items=[item], no_of_classes=no_of_classes)

            yobsdata = np.array(yobsdata).reshape(no_of_classes, 1)

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)

        y = self.exp_score(abilities, self.diffs[item]).reshape(-1, 1)
        y = np.array(y).reshape([len(abilities), 1])

        if title is not None:
            graphtitle = title
                
        else:
            graphtitle = ''

        ylabel = 'Expected score'

        plot = self.plot_data(x_data=abilities, y_data=y, x_obs_data=xobsdata, y_obs_data=yobsdata, x_min=xmin,
                              x_max=xmax, y_max=self.max_score, items=item, y_label=ylabel, graph_title=graphtitle,
                              obs=obs, thresh_line=thresh_line, score_lines_item=[item, score_lines],
                              score_labels=score_labels, cat_highlight=cat_highlight, plot_style=plot_style,
                              palette=palette, black=black, font=font, title_font_size=title_font_size,
                              axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                              plot_density=dpi, file_format=file_format)

        return plot

    def crcs(self,
             item=None,
             obs=None,
             no_of_classes=5,
             title=None,
             thresh_line=False,
             cat_highlight=None,
             xmin=-5,
             xmax=5,
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
        Plots Category Response Curves for SLM, with optional overplotting
        of observed data and threshold lines.
        '''

        if item == 'none':
            item = None

        if obs:
            if hasattr(self, 'person_abilities') == False:
                self.person_abils(warm_corr=False)

            xobsdata, yobsdata = self.class_intervals_cats(item, no_of_classes=no_of_classes)

            if obs != 'all':
                if not all(cat in [0, 1] for cat in obs):
                    print("Invalid 'obs'. Valid values are 'None', 'all' and list of categories.")
                    return

                else:
                    yobsdata = yobsdata[:, obs]

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)

        p = self.exp_score(abilities, self.diffs[item] if item else 0)
        y = np.column_stack([1 - p, p])

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Probability'

        plot = self.plot_data(x_data=abilities,  y_data=y, x_min=xmin, x_max=xmax, y_max=1, x_obs_data=xobsdata,
                              y_obs_data=yobsdata, items=item, graph_title=graphtitle, y_label=ylabel,
                              obs=obs, thresh_line=thresh_line, cat_highlight=cat_highlight, plot_style=plot_style,
                              palette=palette, black=black, font=font, title_font_size=title_font_size,
                              axis_font_size=axis_font_size, labelsize=labelsize, filename=filename,
                              plot_density=dpi, file_format=file_format)

        return plot

    def iic(self,
            item,
            thresh_line=False,
            point_info_lines=None,
            point_info_labels=False,
            cat_highlight=None,
            xmin=-5,
            xmax=5,
            ymax=None,
            plot_style='white',
            palette='dark blue',
            black=False,
            title=None,
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

        abilities = np.arange(-20, 20, 0.1)

        p = self.exp_score(abilities, self.diffs[item])
        y = (p * (1 - p)).reshape(-1, 1)

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
                              cat_highlight=cat_highlight, plot_style=plot_style, palette=palette, black=black,
                              font=font, title_font_size=title_font_size, axis_font_size=axis_font_size,
                              labelsize=labelsize, filename=filename, plot_density=dpi, file_format=file_format)

        return plot

    def tcc(self,
            items=None,
            obs=False,
            no_of_classes=5,
            title=None,
            score_lines=None,
            score_labels=False,
            xmin=-5,
            xmax=5,
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
        Plots Test Characteristic Curve for SLM.
        '''

        if isinstance(items, str):
            if items == 'all':
                items = None

            elif items == 'none':
                items = None

            else:
                items = [items]

        if obs:
            if hasattr(self, 'person_abilities') == False:
                self.person_abils(warm_corr=False)

            xobsdata, yobsdata = self.class_intervals(items=items, no_of_classes=no_of_classes)

            yobsdata = np.array(yobsdata).reshape(no_of_classes, 1)

        else:
            xobsdata = np.array(np.nan)
            yobsdata = np.array(np.nan)

        abilities = np.arange(-20, 20, 0.1)

        if items is None:
            diffs_arr = self.diffs.values
            p = 1.0 / (1.0 + np.exp(diffs_arr[None, :] - abilities[:, None]))
            y = p.sum(axis=1, keepdims=True)

        else:
            diffs_arr = self.diffs[items].values
            p = 1.0 / (1.0 + np.exp(diffs_arr[None, :] - abilities[:, None]))
            y = p.sum(axis=1, keepdims=True)

        y = np.array(y).reshape(len(abilities), 1)

        if items is None:
            y_max = self.no_of_items

        else:
            y_max = len(items)

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Expected score'

        plot = self.plot_data(x_data=abilities, y_data=y, items=items, x_obs_data=xobsdata, y_obs_data=yobsdata,
                              x_min=xmin, x_max=xmax, y_max=y_max, score_lines_test=score_lines,
                              graph_title=graphtitle, y_label=ylabel, obs=obs, score_labels=score_labels,
                              plot_style=plot_style, palette=palette, black=black, font=font,
                              title_font_size=title_font_size, axis_font_size=axis_font_size, labelsize=labelsize,
                              filename=filename, plot_density=dpi, file_format=file_format)

        return plot

    def test_info(self,
                  items=None,
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
        Plots Test Information Curve for SLM.
        '''

        if isinstance(items, str):
            if items == 'all':
                items = None

            elif items == 'none':
                items = None

            else:
                items = [items]

        abilities = np.arange(-20, 20, 0.1)

        if items is None:
            diffs_arr = self.diffs.values
            p = 1.0 / (1.0 + np.exp(diffs_arr[None, :] - abilities[:, None]))
            y = (p * (1.0 - p)).sum(axis=1, keepdims=True)

        else:
            diffs_arr = self.diffs[items].values
            p = 1.0 / (1.0 + np.exp(diffs_arr[None, :] - abilities[:, None]))
            y = (p * (1.0 - p)).sum(axis=1, keepdims=True)

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
                              palette=palette, black=black, font=font, title_font_size=title_font_size,
                              axis_font_size=axis_font_size, labelsize=labelsize, filename=filename, plot_density=dpi,
                              file_format=file_format)

        return plot

    def test_csem(self,
                  items=None,
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
        Plots Test Conditional Standard Error of Measurement Curve for SLM.
        '''

        if isinstance(items, str):
            if items == 'all':
                items = None

            elif items == 'none':
                items = None

            else:
                items = [items]

        abilities = np.arange(-20, 20, 0.1)

        if items is None:
            diffs_arr = self.diffs.values
            p = 1.0 / (1.0 + np.exp(diffs_arr[None, :] - abilities[:, None]))
            y = (p * (1.0 - p)).sum(axis=1, keepdims=True)

        else:
            diffs_arr = self.diffs[items].values
            p = 1.0 / (1.0 + np.exp(diffs_arr[None, :] - abilities[:, None]))
            y = (p * (1.0 - p)).sum(axis=1, keepdims=True)

        y = 1 / (y ** 0.5)
        y = y.reshape(len(abilities), 1)

        if title is not None:
            graphtitle = title

        else:
            graphtitle = ''

        ylabel = 'Conditional SEM'

        plot = self.plot_data(x_data=abilities, y_data=y, items=items, x_min=xmin, x_max=xmax, y_max=ymax,
                              graph_title=graphtitle, point_csem_lines=point_csem_lines, score_labels=point_csem_labels,
                              y_label=ylabel, plot_style=plot_style, palette=palette, black=black, font=font,
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
                           plot_style='white',
                           black=False,
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

        if isinstance(items, str):
            if items == 'all':
                items = None

            elif items == 'none':
                items = None

            else:
                items = [items]

        if items is None:
            std_residual_df = self.std_residual_df

        else:
            std_residual_df = self.std_residual_df[items]

        std_residual_list = std_residual_df.unstack().dropna()

        plot = self.std_residuals_hist(std_residual_list, bin_width=bin_width, x_min=x_min, x_max=x_max, normal=normal,
                                       title=title, plot_style=plot_style, font=font, title_font_size=title_font_size,
                                       axis_font_size=axis_font_size, labelsize=labelsize, black=black,
                                       filename=filename, file_format=file_format, plot_density=plot_density)

        return plot